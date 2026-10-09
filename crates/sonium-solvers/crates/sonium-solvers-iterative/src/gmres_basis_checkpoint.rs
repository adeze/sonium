//! Checked GMRES Krylov-basis capture and continuation.
//!
//! The ordinary [`super::gmres`] checkpoint stores only a solution iterate.
//! This module stores the active Arnoldi basis and Hessenberg/Givens state so
//! a solve can continue from the exact inner iteration without rebuilding the
//! preceding Krylov vectors.  The replay path supports both fixed restart and
//! bounded adaptive restart policies while retaining the caller-supplied
//! checkpoint budget.

use crate::cgs_orth::KrylovOpCounts;
use crate::gmres::{GmresConfig, GmresSolution};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Result of a bounded basis-capture run.
#[derive(Debug)]
pub enum GmresBasisRun<T: ComplexField> {
    /// The requested checkpoint was reached before convergence.
    Checkpoint(GmresBasisCheckpoint<T>),
    /// The system converged before the checkpoint budget was reached.
    Solution(GmresSolution<T>),
}

/// Normalize both outcomes of a checked basis-capture run through the
/// shared solver result contract.  A checkpoint is an intentional
/// non-converged boundary, while a completed run delegates all metadata
/// validation to the ordinary GMRES result.
impl<T: ComplexField> SolverResultContract for GmresBasisRun<T> {
    fn solver_status(&self) -> SolverStatus {
        match self {
            Self::Checkpoint(_) => SolverStatus::MaxIterationsReached,
            Self::Solution(solution) => solution.solver_status(),
        }
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        match self {
            Self::Checkpoint(_) => AccuracyStatus::NotAttained,
            Self::Solution(solution) => solution.accuracy_status(),
        }
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        match self {
            Self::Checkpoint(checkpoint) => checkpoint.validate_state(),
            Self::Solution(solution) => solution.validate_contract(),
        }
    }
}

/// Active GMRES Arnoldi state, optionally carrying bounded adaptive-restart
/// metadata.
#[derive(Debug, Clone)]
pub struct GmresBasisCheckpoint<T: ComplexField> {
    dimension: usize,
    restart: usize,
    tolerance: T::Real,
    rhs_norm: T::Real,
    min_restart: Option<usize>,
    max_restart: Option<usize>,
    cycle_initial_residual: T::Real,
    solution: Vec<T>,
    basis: Vec<Vec<T>>,
    hessenberg: Vec<Vec<T>>,
    cosines: Vec<T>,
    sines: Vec<T>,
    transformed_rhs: Vec<T>,
    inner_iterations: usize,
    iterations: usize,
    restarts: usize,
    op_counts: KrylovOpCounts,
}

/// Version of the portable active-basis checkpoint format.
pub const GMRES_BASIS_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const BASIS_CHECKPOINT_MAGIC: &[u8; 8] = b"SGBASIS1";
const BASIS_CHECKPOINT_FLAG_ADAPTIVE_RESTART: u8 = 1;

impl<T: ComplexField> GmresBasisCheckpoint<T> {
    /// Capture a bounded fixed-restart GMRES run.
    pub fn capture<A, P>(
        operator: &A,
        preconditioner: &P,
        rhs: &[T],
        config: &GmresConfig<T::Real>,
        checkpoint_after: usize,
    ) -> Result<GmresBasisRun<T>, SolverError>
    where
        A: LinearOperator<T>,
        P: Preconditioner<T>,
    {
        validate_request(operator, rhs, config, checkpoint_after)?;
        let mut state = Self::initial(operator, preconditioner, rhs, config)?;
        if state.is_converged(config.tolerance) {
            return Ok(GmresBasisRun::Solution(state.finish(
                operator,
                rhs,
                config.tolerance,
            )?));
        }
        state.advance_until(operator, preconditioner, rhs, config, checkpoint_after)
    }

    /// Resume the captured Arnoldi state until convergence or `max_iterations`.
    pub fn resume<A, P>(
        &self,
        operator: &A,
        preconditioner: &P,
        rhs: &[T],
        config: &GmresConfig<T::Real>,
    ) -> Result<GmresSolution<T>, SolverError>
    where
        A: LinearOperator<T>,
        P: Preconditioner<T>,
    {
        validate_request(operator, rhs, config, self.iterations.saturating_add(1))?;
        let adaptive = self.min_restart.is_some() || self.max_restart.is_some();
        let restart_compatible = if adaptive {
            config.min_restart == self.min_restart
                && config.max_restart == self.max_restart
                && self
                    .min_restart
                    .is_some_and(|min_restart| config.restart >= min_restart)
                && self
                    .max_restart
                    .is_some_and(|max_restart| config.restart <= max_restart)
        } else {
            config.restart == self.restart
        };
        if !restart_compatible || config.tolerance != self.tolerance || rhs.len() != self.dimension
        {
            return Err(basis_error(
                "configuration or RHS is incompatible with checkpoint",
            ));
        }
        let mut state = self.clone();
        state
            .advance_until(operator, preconditioner, rhs, config, usize::MAX)
            .and_then(|run| match run {
                GmresBasisRun::Solution(solution) => Ok(solution),
                GmresBasisRun::Checkpoint(_) => {
                    Err(basis_error("basis replay stopped without a final solution"))
                }
            })
    }

    /// Number of rows in the replayed system.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of Arnoldi steps already executed.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Current Arnoldi restart length. Adaptive checkpoints may change this
    /// value between cycles while remaining within configured bounds.
    pub const fn restart_parameter(&self) -> usize {
        self.restart
    }

    /// Number of basis vectors currently retained.
    pub const fn basis_vectors(&self) -> usize {
        self.basis.len()
    }

    /// Approximate memory occupied by the replay state.
    pub fn storage_bytes(&self) -> usize {
        let scalar = std::mem::size_of::<T>();
        self.solution.len().saturating_mul(scalar)
            + self
                .basis
                .iter()
                .map(|v| v.len().saturating_mul(scalar))
                .sum::<usize>()
            + self
                .hessenberg
                .iter()
                .map(|v| v.len().saturating_mul(scalar))
                .sum::<usize>()
            + (self.cosines.len() + self.sines.len() + self.transformed_rhs.len())
                .saturating_mul(scalar)
    }

    /// Serialize the active basis using a versioned little-endian format.
    /// Numeric values are canonical `f64` real/imaginary pairs; restore
    /// requires the same scalar family and rejects malformed shapes.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        self.validate_state()?;
        let mut output = Vec::new();
        output.extend_from_slice(BASIS_CHECKPOINT_MAGIC);
        output.extend_from_slice(&GMRES_BASIS_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(scalar_is_complex::<T>());
        let adaptive = self.min_restart.is_some() || self.max_restart.is_some();
        output.push(if adaptive {
            BASIS_CHECKPOINT_FLAG_ADAPTIVE_RESTART
        } else {
            0
        });
        push_u64(&mut output, self.dimension)?;
        push_u64(&mut output, self.restart)?;
        push_u64(&mut output, self.inner_iterations)?;
        push_u64(&mut output, self.iterations)?;
        push_u64(&mut output, self.restarts)?;
        push_f64(&mut output, self.tolerance)?;
        push_f64(&mut output, self.rhs_norm)?;
        push_u64(&mut output, self.basis.len())?;
        push_u64(&mut output, self.hessenberg.len())?;
        push_u64(&mut output, self.hessenberg.first().map_or(0, Vec::len))?;
        push_u64(&mut output, self.cosines.len())?;
        push_u64(&mut output, self.sines.len())?;
        push_u64(&mut output, self.transformed_rhs.len())?;
        for value in [
            self.op_counts.spmv_count,
            self.op_counts.preconditioner_count,
            self.op_counts.dot_reductions,
            self.op_counts.norm_count,
            self.op_counts.axpy_passes,
            self.op_counts.synchronization_points,
            self.op_counts.exact_residual_checks,
        ] {
            push_u64(&mut output, value)?;
        }
        output.extend_from_slice(&self.op_counts.basis_bytes_touched.to_le_bytes());
        if adaptive {
            push_u64(
                &mut output,
                self.min_restart
                    .ok_or_else(|| basis_error("adaptive checkpoint is missing min_restart"))?,
            )?;
            push_u64(
                &mut output,
                self.max_restart
                    .ok_or_else(|| basis_error("adaptive checkpoint is missing max_restart"))?,
            )?;
            push_f64(&mut output, self.cycle_initial_residual)?;
        }
        for value in &self.solution {
            push_scalar(&mut output, *value)?;
        }
        for vector in &self.basis {
            for value in vector {
                push_scalar(&mut output, *value)?;
            }
        }
        for row in &self.hessenberg {
            for value in row {
                push_scalar(&mut output, *value)?;
            }
        }
        for value in self
            .cosines
            .iter()
            .chain(&self.sines)
            .chain(&self.transformed_rhs)
        {
            push_scalar(&mut output, *value)?;
        }
        Ok(output)
    }

    /// Restore an active basis checkpoint produced by [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = CheckpointReader { bytes, offset: 0 };
        if reader.take(8)? != BASIS_CHECKPOINT_MAGIC {
            return Err(basis_error("invalid GMRES basis checkpoint magic"));
        }
        if reader.read_u16()? != GMRES_BASIS_CHECKPOINT_FORMAT_VERSION {
            return Err(basis_error("unsupported GMRES basis checkpoint version"));
        }
        if reader.read_u8()? != std::mem::size_of::<T>() as u8
            || reader.read_u8()? != std::mem::size_of::<T::Real>() as u8
            || reader.read_u8()? != scalar_is_complex::<T>()
        {
            return Err(basis_error("basis checkpoint scalar family does not match"));
        }
        let flags = reader.read_u8()?;
        if flags & !BASIS_CHECKPOINT_FLAG_ADAPTIVE_RESTART != 0 {
            return Err(basis_error("unsupported GMRES basis checkpoint flags"));
        }
        let dimension = reader.read_usize()?;
        let restart = reader.read_usize()?;
        let inner_iterations = reader.read_usize()?;
        let iterations = reader.read_usize()?;
        let restarts = reader.read_usize()?;
        let tolerance = reader.read_f64()?;
        let rhs_norm = reader.read_f64()?;
        let basis_count = reader.read_usize()?;
        let h_rows = reader.read_usize()?;
        let h_cols = reader.read_usize()?;
        let cosine_count = reader.read_usize()?;
        let sine_count = reader.read_usize()?;
        let transformed_count = reader.read_usize()?;
        if restart == 0
            || inner_iterations > restart
            || basis_count != inner_iterations.saturating_add(1)
            || h_rows != restart.saturating_add(1)
            || h_cols != restart
            || cosine_count != inner_iterations
            || sine_count != inner_iterations
            || transformed_count != restart.saturating_add(1)
        {
            return Err(basis_error(
                "basis checkpoint shape metadata is inconsistent",
            ));
        }
        let op_counts = KrylovOpCounts {
            spmv_count: reader.read_usize()?,
            preconditioner_count: reader.read_usize()?,
            dot_reductions: reader.read_usize()?,
            norm_count: reader.read_usize()?,
            axpy_passes: reader.read_usize()?,
            synchronization_points: reader.read_usize()?,
            exact_residual_checks: reader.read_usize()?,
            basis_bytes_touched: reader.read_u64()?,
        };
        let (min_restart, max_restart, cycle_initial_residual) =
            if flags & BASIS_CHECKPOINT_FLAG_ADAPTIVE_RESTART != 0 {
                let min_restart = usize::try_from(reader.read_u64()?)
                    .map_err(|_| basis_error("adaptive min_restart does not fit usize"))?;
                let max_restart = usize::try_from(reader.read_u64()?)
                    .map_err(|_| basis_error("adaptive max_restart does not fit usize"))?;
                let cycle_initial_residual = T::Real::from_f64(reader.read_f64()?)
                    .ok_or_else(|| basis_error("adaptive cycle residual cannot be restored"))?;
                (Some(min_restart), Some(max_restart), cycle_initial_residual)
            } else {
                (None, None, T::Real::one())
            };
        let real_tolerance = T::Real::from_f64(tolerance)
            .ok_or_else(|| basis_error("basis tolerance cannot be restored"))?;
        let real_rhs_norm = T::Real::from_f64(rhs_norm)
            .ok_or_else(|| basis_error("basis RHS norm cannot be restored"))?;
        let scalar_count = dimension
            .checked_add(
                basis_count
                    .checked_mul(dimension)
                    .ok_or_else(|| basis_error("basis scalar count overflow"))?,
            )
            .and_then(|value| value.checked_add(h_rows.checked_mul(h_cols)?))
            .and_then(|value| value.checked_add(cosine_count))
            .and_then(|value| value.checked_add(sine_count))
            .and_then(|value| value.checked_add(transformed_count))
            .ok_or_else(|| basis_error("basis scalar count overflow"))?;
        if scalar_count > reader.remaining() / 16 {
            return Err(basis_error("truncated GMRES basis checkpoint payload"));
        }
        let solution = reader.read_scalars(dimension)?;
        let mut basis = Vec::with_capacity(basis_count);
        for _ in 0..basis_count {
            basis.push(reader.read_scalars(dimension)?);
        }
        let mut hessenberg = Vec::with_capacity(h_rows);
        for _ in 0..h_rows {
            hessenberg.push(reader.read_scalars(h_cols)?);
        }
        let cosines = reader.read_scalars(cosine_count)?;
        let sines = reader.read_scalars(sine_count)?;
        let transformed_rhs = reader.read_scalars(transformed_count)?;
        if reader.offset != bytes.len() {
            return Err(basis_error("trailing bytes in GMRES basis checkpoint"));
        }
        let cycle_initial_residual = if min_restart.is_none() {
            transformed_rhs
                .first()
                .map(|value: &T| value.norm())
                .unwrap_or(cycle_initial_residual)
        } else {
            cycle_initial_residual
        };
        let checkpoint = Self {
            dimension,
            restart,
            tolerance: real_tolerance,
            rhs_norm: real_rhs_norm,
            min_restart,
            max_restart,
            cycle_initial_residual,
            solution,
            basis,
            hessenberg,
            cosines,
            sines,
            transformed_rhs,
            inner_iterations,
            iterations,
            restarts,
            op_counts,
        };
        checkpoint.validate_state()?;
        Ok(checkpoint)
    }

    fn validate_state(&self) -> Result<(), SolverError> {
        if self.dimension != self.solution.len()
            || self.restart == 0
            || self.inner_iterations == 0
            || self.inner_iterations > self.restart
            || self.basis.len() != self.inner_iterations + 1
            || self.hessenberg.len() != self.restart + 1
            || self.hessenberg.iter().any(|row| row.len() != self.restart)
            || self.cosines.len() != self.inner_iterations
            || self.sines.len() != self.inner_iterations
            || self.transformed_rhs.len() != self.restart + 1
        {
            return Err(basis_error("basis checkpoint state shape is invalid"));
        }
        if !self.tolerance.is_finite()
            || self.tolerance <= T::Real::zero()
            || !self.rhs_norm.is_finite()
            || self.rhs_norm <= T::Real::zero()
            || !self.cycle_initial_residual.is_finite()
            || self.cycle_initial_residual <= T::Real::zero()
        {
            return Err(basis_error(
                "basis checkpoint metadata is non-finite or non-positive",
            ));
        }
        match (self.min_restart, self.max_restart) {
            (None, None) => {}
            (Some(min_restart), Some(max_restart))
                if min_restart > 0
                    && min_restart <= max_restart
                    && self.restart >= min_restart
                    && self.restart <= max_restart => {}
            _ => return Err(basis_error("adaptive restart bounds are invalid")),
        }
        let finite = |value: &T| value.norm().is_finite();
        if self.solution.iter().any(|value| !finite(value))
            || self.basis.iter().flatten().any(|value| !finite(value))
            || self.hessenberg.iter().flatten().any(|value| !finite(value))
            || self.cosines.iter().any(|value| !finite(value))
            || self.sines.iter().any(|value| !finite(value))
            || self.transformed_rhs.iter().any(|value| !finite(value))
        {
            return Err(basis_error("basis checkpoint contains non-finite values"));
        }
        Ok(())
    }

    fn initial<A, P>(
        operator: &A,
        preconditioner: &P,
        rhs: &[T],
        config: &GmresConfig<T::Real>,
    ) -> Result<Self, SolverError>
    where
        A: LinearOperator<T>,
        P: Preconditioner<T>,
    {
        let n = rhs.len();
        let rhs_norm = vector_norm(rhs);
        let rhs_norm = if rhs_norm > T::Real::zero() {
            rhs_norm
        } else {
            T::Real::one()
        };
        let solution = vec![T::zero(); n];
        let mut op_buf = vec![T::zero(); n];
        let mut residual = rhs.to_vec();
        let mut preconditioned = vec![T::zero(); n];
        apply_operator(operator, &solution, &mut op_buf)?;
        residual.iter_mut().zip(&op_buf).for_each(|(r, a)| *r -= *a);
        apply_preconditioner(preconditioner, &residual, &mut preconditioned)?;
        let beta = vector_norm(&preconditioned);
        if !beta.is_finite() {
            return Err(basis_error("initial preconditioned residual is non-finite"));
        }
        let mut basis = Vec::with_capacity(config.restart + 1);
        if beta > T::Real::zero() {
            let scale = T::from_real(T::Real::one() / beta);
            basis.push(preconditioned.iter().map(|v| *v * scale).collect());
        } else {
            basis.push(vec![T::zero(); n]);
        }
        let mut transformed_rhs = vec![T::zero(); config.restart + 1];
        transformed_rhs[0] = T::from_real(beta);
        Ok(Self {
            dimension: n,
            restart: config.restart,
            tolerance: config.tolerance,
            rhs_norm,
            min_restart: config.min_restart,
            max_restart: config.max_restart,
            cycle_initial_residual: beta,
            solution,
            basis,
            hessenberg: vec![vec![T::zero(); config.restart]; config.restart + 1],
            cosines: Vec::with_capacity(config.restart),
            sines: Vec::with_capacity(config.restart),
            transformed_rhs,
            inner_iterations: 0,
            iterations: 0,
            restarts: 0,
            op_counts: KrylovOpCounts {
                spmv_count: 1,
                preconditioner_count: 1,
                norm_count: 1,
                ..KrylovOpCounts::default()
            },
        })
    }

    fn advance_until<A, P>(
        &mut self,
        operator: &A,
        preconditioner: &P,
        rhs: &[T],
        config: &GmresConfig<T::Real>,
        checkpoint_after: usize,
    ) -> Result<GmresBasisRun<T>, SolverError>
    where
        A: LinearOperator<T>,
        P: Preconditioner<T>,
    {
        while self.iterations < config.max_iterations {
            if self.inner_iterations == self.restart {
                self.restart_cycle(operator, preconditioner, rhs)?;
                if self.is_converged(config.tolerance) {
                    return Ok(GmresBasisRun::Solution(self.finish(
                        operator,
                        rhs,
                        config.tolerance,
                    )?));
                }
            }
            let j = self.inner_iterations;
            let mut op_buf = vec![T::zero(); self.dimension];
            let mut w = vec![T::zero(); self.dimension];
            apply_operator(operator, &self.basis[j], &mut op_buf)?;
            self.op_counts.spmv_count += 1;
            apply_preconditioner(preconditioner, &op_buf, &mut w)?;
            self.op_counts.preconditioner_count += 1;
            let before = vector_norm(&w);
            self.op_counts.norm_count += 1;
            for i in 0..=j {
                let coefficient = inner_product(&self.basis[i], &w);
                self.hessenberg[i][j] += coefficient;
                for (value, basis_value) in w.iter_mut().zip(&self.basis[i]) {
                    *value -= coefficient * *basis_value;
                }
            }
            let mut after = vector_norm(&w);
            self.op_counts.norm_count += 1;
            if after < T::Real::from_f64(0.707).unwrap_or_else(T::Real::epsilon) * before {
                for i in 0..=j {
                    let coefficient = inner_product(&self.basis[i], &w);
                    self.hessenberg[i][j] += coefficient;
                    for (value, basis_value) in w.iter_mut().zip(&self.basis[i]) {
                        *value -= coefficient * *basis_value;
                    }
                }
                after = vector_norm(&w);
                self.op_counts.norm_count += 1;
            }
            self.hessenberg[j + 1][j] = T::from_real(after);
            if after > T::Real::zero() {
                let scale = T::from_real(T::Real::one() / after);
                self.basis.push(w.iter().map(|v| *v * scale).collect());
            } else {
                self.basis.push(vec![T::zero(); self.dimension]);
            }
            for i in 0..j {
                let temp = self.cosines[i].conj() * self.hessenberg[i][j]
                    + self.sines[i].conj() * self.hessenberg[i + 1][j];
                self.hessenberg[i + 1][j] = -self.sines[i] * self.hessenberg[i][j]
                    + self.cosines[i] * self.hessenberg[i + 1][j];
                self.hessenberg[i][j] = temp;
            }
            let (c, s, rotated) = givens(self.hessenberg[j][j], self.hessenberg[j + 1][j]);
            self.cosines.push(c);
            self.sines.push(s);
            self.hessenberg[j][j] = rotated;
            self.hessenberg[j + 1][j] = T::zero();
            let old_g = self.transformed_rhs[j];
            self.transformed_rhs[j] = c * old_g + s * self.transformed_rhs[j + 1];
            self.transformed_rhs[j + 1] = -s.conj() * old_g + c * self.transformed_rhs[j + 1];
            self.inner_iterations += 1;
            self.iterations += 1;
            let projected = self.transformed_rhs[self.inner_iterations].norm() / self.rhs_norm;
            if projected.is_finite() && projected <= config.tolerance {
                let result = self.finish(operator, rhs, config.tolerance)?;
                if result.converged {
                    return Ok(GmresBasisRun::Solution(result));
                }
                self.restart_cycle(operator, preconditioner, rhs)?;
            }
            if self.iterations >= checkpoint_after {
                return Ok(GmresBasisRun::Checkpoint(self.clone()));
            }
        }
        Ok(GmresBasisRun::Solution(self.finish(
            operator,
            rhs,
            config.tolerance,
        )?))
    }

    fn adapt_restart_parameter(&mut self) {
        let (Some(min_restart), Some(max_restart)) = (self.min_restart, self.max_restart) else {
            return;
        };
        if self.inner_iterations != self.restart || !self.cycle_initial_residual.is_finite() {
            return;
        }
        let residual_after_cycle = self.transformed_rhs[self.inner_iterations].norm();
        let ratio = residual_after_cycle / self.cycle_initial_residual;
        if !ratio.is_finite() {
            return;
        }
        let new_restart = if ratio > T::Real::from_f64(0.9).unwrap_or_else(T::Real::epsilon)
            && self.restart < max_restart
        {
            self.restart.saturating_mul(2).min(max_restart)
        } else if ratio < T::Real::from_f64(0.1).unwrap_or_else(T::Real::epsilon)
            && self.restart > min_restart
        {
            (self.restart / 2).max(min_restart)
        } else {
            self.restart
        };
        if new_restart != self.restart {
            self.restart = new_restart;
            self.hessenberg = vec![vec![T::zero(); new_restart]; new_restart + 1];
            self.cosines = Vec::with_capacity(new_restart);
            self.sines = Vec::with_capacity(new_restart);
            self.transformed_rhs = vec![T::zero(); new_restart + 1];
        }
    }

    fn restart_cycle<A, P>(
        &mut self,
        operator: &A,
        preconditioner: &P,
        rhs: &[T],
    ) -> Result<(), SolverError>
    where
        A: LinearOperator<T>,
        P: Preconditioner<T>,
    {
        self.update_solution();
        self.adapt_restart_parameter();
        self.restarts += 1;
        self.inner_iterations = 0;
        self.basis.clear();
        self.hessenberg
            .iter_mut()
            .for_each(|row| row.iter_mut().for_each(|v| *v = T::zero()));
        self.cosines.clear();
        self.sines.clear();
        self.transformed_rhs.fill(T::zero());
        let mut op_buf = vec![T::zero(); self.dimension];
        let mut residual = rhs.to_vec();
        let mut preconditioned = vec![T::zero(); self.dimension];
        apply_operator(operator, &self.solution, &mut op_buf)?;
        self.op_counts.spmv_count += 1;
        residual.iter_mut().zip(&op_buf).for_each(|(r, a)| *r -= *a);
        apply_preconditioner(preconditioner, &residual, &mut preconditioned)?;
        self.op_counts.preconditioner_count += 1;
        let beta = vector_norm(&preconditioned);
        self.op_counts.norm_count += 1;
        if !beta.is_finite() {
            return Err(basis_error("restart residual is non-finite"));
        }
        self.transformed_rhs[0] = T::from_real(beta);
        self.cycle_initial_residual = beta;
        if beta > T::Real::zero() {
            let scale = T::from_real(T::Real::one() / beta);
            self.basis
                .push(preconditioned.iter().map(|v| *v * scale).collect());
        } else {
            self.basis.push(vec![T::zero(); self.dimension]);
        }
        Ok(())
    }

    fn update_solution(&mut self) {
        let k = self.inner_iterations;
        if k == 0 {
            return;
        }
        let mut y = vec![T::zero(); k];
        for i in (0..k).rev() {
            let mut value = self.transformed_rhs[i];
            for (j, coefficient) in y.iter().enumerate().skip(i + 1) {
                value -= self.hessenberg[i][j] * *coefficient;
            }
            let diagonal = self.hessenberg[i][i];
            if diagonal.norm() > T::Real::zero() {
                y[i] = value / diagonal;
            }
        }
        for (j, coefficient) in y.into_iter().enumerate() {
            for (x, basis_value) in self.solution.iter_mut().zip(&self.basis[j]) {
                *x += coefficient * *basis_value;
            }
        }
    }

    fn is_converged(&self, tolerance: T::Real) -> bool {
        self.transformed_rhs
            .first()
            .is_some_and(|value| value.norm() / self.rhs_norm <= tolerance)
    }

    fn finish<A>(
        &mut self,
        operator: &A,
        rhs: &[T],
        tolerance: T::Real,
    ) -> Result<GmresSolution<T>, SolverError>
    where
        A: LinearOperator<T>,
    {
        if self.inner_iterations > 0 {
            self.update_solution();
            self.inner_iterations = 0;
        }
        let mut applied = vec![T::zero(); self.dimension];
        apply_operator(operator, &self.solution, &mut applied)?;
        self.op_counts.spmv_count += 1;
        let residual = rhs
            .iter()
            .zip(&applied)
            .map(|(b, a)| (*b - *a).norm_sqr())
            .fold(T::Real::zero(), |sum, value| sum + value)
            .sqrt()
            / self.rhs_norm;
        self.op_counts.norm_count += 1;
        let converged = residual.is_finite() && residual <= tolerance;
        Ok(GmresSolution {
            x: self.solution.clone(),
            iterations: self.iterations,
            restarts: self.restarts,
            residual,
            estimated_residual: residual,
            backward_error: residual,
            attainable_accuracy: if converged {
                AccuracyStatus::Attained
            } else {
                AccuracyStatus::NotAttained
            },
            converged,
            status: if converged {
                SolverStatus::Converged
            } else {
                SolverStatus::MaxIterationsReached
            },
            op_counts: self.op_counts,
        })
    }
}

fn validate_request<T, A>(
    operator: &A,
    rhs: &[T],
    config: &GmresConfig<T::Real>,
    checkpoint_after: usize,
) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if operator.num_rows() != operator.num_cols() || rhs.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: rhs.len(),
        });
    }
    if rhs.iter().any(|value| !value.norm().is_finite()) {
        return Err(basis_error("right-hand side contains NaN or infinity"));
    }
    if config.max_iterations == 0 || config.restart == 0 || checkpoint_after == 0 {
        return Err(basis_error(
            "basis replay requires positive iteration, restart, and checkpoint budgets",
        ));
    }
    if checkpoint_after > config.max_iterations {
        return Err(basis_error(
            "checkpoint budget exceeds GMRES max_iterations",
        ));
    }
    if !config.tolerance.is_finite() || config.tolerance <= T::Real::zero() {
        return Err(basis_error("GMRES tolerance must be finite and positive"));
    }
    match (config.min_restart, config.max_restart) {
        (None, None) => {}
        (Some(min_restart), Some(max_restart))
            if min_restart > 0
                && min_restart <= max_restart
                && config.restart >= min_restart
                && config.restart <= max_restart => {}
        _ => {
            return Err(basis_error(
                "adaptive restart requires valid min/max bounds containing restart",
            ));
        }
    }
    Ok(())
}

fn apply_operator<T, A>(operator: &A, input: &[T], output: &mut [T]) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if input.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: input.len(),
        });
    }
    if output.len() != operator.num_rows() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_rows(),
            got: output.len(),
        });
    }
    if input.iter().any(|value| !value.norm().is_finite()) {
        return Err(callback_error(
            "GMRES basis replay operator",
            "operator received NaN or infinity input",
        ));
    }
    catch_unwind(AssertUnwindSafe(|| operator.apply_into(input, output))).map_err(|payload| {
        callback_error(
            "GMRES basis replay operator",
            format!("operator callback panicked: {}", panic_reason(payload)),
        )
    })?;
    if output.iter().any(|value| !value.norm().is_finite()) {
        return Err(callback_error(
            "GMRES basis replay operator",
            "operator generated NaN or infinity",
        ));
    }
    Ok(())
}

fn apply_preconditioner<T, P>(
    preconditioner: &P,
    input: &[T],
    output: &mut [T],
) -> Result<(), SolverError>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    if input.len() != output.len() {
        return Err(SolverError::DimensionMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    if input.iter().any(|value| !value.norm().is_finite()) {
        return Err(callback_error(
            "GMRES basis replay preconditioner",
            "preconditioner received NaN or infinity input",
        ));
    }
    catch_unwind(AssertUnwindSafe(|| {
        preconditioner.apply_into(input, output)
    }))
    .map_err(|payload| {
        callback_error(
            "GMRES basis replay preconditioner",
            format!(
                "preconditioner callback panicked: {}",
                panic_reason(payload)
            ),
        )
    })?;
    if output.iter().any(|value| !value.norm().is_finite()) {
        return Err(callback_error(
            "GMRES basis replay preconditioner",
            "preconditioner generated NaN or infinity",
        ));
    }
    Ok(())
}

fn callback_error(operation: &'static str, reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation,
        reason: reason.into(),
    }
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn givens<T: ComplexField>(a: T, b: T) -> (T, T, T) {
    let abs_a = a.norm();
    let abs_b = b.norm();
    let norm = (abs_a * abs_a + abs_b * abs_b).sqrt();
    if norm <= T::Real::zero() {
        return (T::from_real(T::Real::one()), T::zero(), T::zero());
    }
    if abs_a <= T::Real::zero() {
        let s = b.conj() / T::from_real(abs_b);
        return (T::zero(), s, T::from_real(abs_b));
    }
    let c = T::from_real(abs_a / norm);
    let s = a / T::from_real(abs_a) * b.conj() / T::from_real(norm);
    (c, s, a / T::from_real(abs_a) * T::from_real(norm))
}

fn basis_error(reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "GMRES basis replay",
        reason: reason.into(),
    }
}

fn scalar_is_complex<T: ComplexField>() -> u8 {
    u8::from(T::from_re_im(T::Real::zero(), T::Real::one()).norm() > T::Real::zero())
}

fn push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value = u64::try_from(value).map_err(|_| basis_error("basis size does not fit u64"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_f64<T: Float + ToPrimitive>(output: &mut Vec<u8>, value: T) -> Result<(), SolverError> {
    let value = value
        .to_f64()
        .ok_or_else(|| basis_error("basis metadata cannot be represented as f64"))?;
    if !value.is_finite() {
        return Err(basis_error("basis metadata is non-finite"));
    }
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_scalar<T: ComplexField>(output: &mut Vec<u8>, value: T) -> Result<(), SolverError> {
    let re = value
        .re()
        .to_f64()
        .ok_or_else(|| basis_error("basis real component cannot be represented as f64"))?;
    let im = value
        .im()
        .to_f64()
        .ok_or_else(|| basis_error("basis imaginary component cannot be represented as f64"))?;
    if !re.is_finite() || !im.is_finite() {
        return Err(basis_error("basis checkpoint contains non-finite scalar"));
    }
    output.extend_from_slice(&re.to_le_bytes());
    output.extend_from_slice(&im.to_le_bytes());
    Ok(())
}

struct CheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> CheckpointReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| basis_error("basis checkpoint offset overflow"))?;
        if end > self.bytes.len() {
            return Err(basis_error("truncated GMRES basis checkpoint"));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(value))
    }

    fn read_u64(&mut self) -> Result<u64, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(value))
    }

    fn read_usize(&mut self) -> Result<usize, SolverError> {
        usize::try_from(self.read_u64()?).map_err(|_| basis_error("basis size does not fit usize"))
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        Ok(f64::from_le_bytes(value))
    }

    fn read_scalars<T: ComplexField>(&mut self, count: usize) -> Result<Vec<T>, SolverError> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            let re = T::Real::from_f64(self.read_f64()?)
                .ok_or_else(|| basis_error("basis real component cannot be restored"))?;
            let im = T::Real::from_f64(self.read_f64()?)
                .ok_or_else(|| basis_error("basis imaginary component cannot be restored"))?;
            let value = T::from_re_im(re, im);
            if !value.norm().is_finite() {
                return Err(basis_error("basis checkpoint restores non-finite scalar"));
            }
            values.push(value);
        }
        Ok(values)
    }
}
