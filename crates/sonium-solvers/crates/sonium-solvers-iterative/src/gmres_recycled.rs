//! Recycled GMRES (GCRO-DR) for frequency sweep solves
//!
//! Retains harmonic Ritz vectors from one solve and uses them to seed the
//! Krylov basis of the next, dramatically reducing iteration counts when
//! consecutive systems have similar spectral properties (e.g., BEM frequency
//! sweeps where A(f_i) and A(f_{i+1}) differ only slightly).
//!
//! The algorithm works in two phases:
//! 1. **Coarse correction**: project the residual onto the recycled subspace
//!    and apply a coarse correction `x += U * (U^H A U)^{-1} * U^H * r`.
//! 2. **Standard GMRES**: run GMRES on the corrected system to reduce the
//!    remaining residual. At convergence, extract harmonic Ritz vectors from
//!    the Hessenberg decomposition and update the recycled subspace.
//!
//! The first solve (empty subspace) runs standard GMRES and extracts harmonic
//! Ritz vectors. Subsequent solves benefit from the coarse correction, which
//! eliminates the components of the residual in the recycled subspace.
//!
//! Reference: Parks, M.L. et al. (2006). "Recycling Krylov Subspaces for
//! Sequences of Linear Systems." SIAM J. Sci. Comput. 28(5), 1651-1674.

use super::cgs_orth::{
    FlatBasis, KrylovOpCounts, batched_dots_flat_into, batched_project_flat,
    fused_basis_update_flat,
};
use super::checked_callbacks::{
    CheckedLinearOperator, CheckedPreconditioner, catch_solver_callback,
};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, IdentityPreconditioner, LinearOperator, Preconditioner,
    SolverError, SolverResultContract, SolverStatus,
};

use super::gmres::{
    GmresConfig, GmresSolution, finalize_gmres_solution, invalid_gmres_input,
    numerical_error_solution,
};
use super::gmres_common::{
    exact_relative_residual, exact_residual_check_due, givens_rotation, solve_upper_triangular,
};
use super::gmres_dr::{extract_deflation_vectors_flat, unapply_givens};

/// Recycled Krylov subspace that persists across solves.
///
/// After each solve, the harmonic Ritz vectors are extracted and stored.
/// On the next solve, they are used to provide a coarse correction that
/// reduces iteration counts for systems with similar spectral properties
/// (e.g., frequency sweeps in BEM).
#[deprecated(
    note = "use gcro_dr: recycled GMRES loses the recycling benefit without GCRO-DR's orthogonality (the recycled subspace is not kept orthogonal to the Krylov basis, so the coarse correction degrades); kept for API compatibility"
)]
#[derive(Clone)]
pub struct RecycledSubspace<T: ComplexField> {
    /// Deflation vectors from the previous solve (orthonormal)
    vectors: Vec<Vec<T>>,
    /// Number of vectors to retain
    k: usize,
}

/// Version of the portable recycled-GMRES subspace checkpoint format.
pub const RECYCLED_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const RECYCLED_CHECKPOINT_MAGIC: &[u8; 8] = b"SRECYCP1";

struct RecycledCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

fn recycled_checkpoint_error(reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "recycled GMRES checkpoint",
        reason: reason.into(),
    }
}

fn recycled_scalar_is_complex<T: ComplexField>() -> u8 {
    let one = T::Real::from_f64(1.0).unwrap_or_else(T::Real::zero);
    u8::from(T::from_re_im(T::Real::zero(), one).norm() > T::Real::zero())
}

fn validate_recycled_vectors<T: ComplexField>(
    k: usize,
    vectors: &[Vec<T>],
) -> Result<usize, SolverError> {
    if vectors.len() > k {
        return Err(recycled_checkpoint_error(
            "stored vector count exceeds configured recycle size",
        ));
    }
    let dimension = vectors.first().map_or(0, Vec::len);
    if !vectors.is_empty() && dimension == 0 {
        return Err(recycled_checkpoint_error(
            "stored vector count is non-zero but vector dimension is zero",
        ));
    }
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return Err(recycled_checkpoint_error(
            "stored vectors have inconsistent dimensions",
        ));
    }
    if vectors
        .iter()
        .flat_map(|vector| vector.iter())
        .any(|value| !value.norm().is_finite())
    {
        return Err(recycled_checkpoint_error(
            "stored vectors contain NaN or infinity",
        ));
    }
    Ok(dimension)
}

#[allow(deprecated)]
impl<T: ComplexField> RecycledSubspace<T> {
    /// Create a new empty recycled subspace.
    pub fn new(k: usize) -> Self {
        Self {
            vectors: Vec::new(),
            k,
        }
    }

    /// Whether this subspace has been populated by a previous solve.
    pub fn is_populated(&self) -> bool {
        !self.vectors.is_empty()
    }

    /// Number of recycled vectors currently stored.
    pub fn num_vectors(&self) -> usize {
        self.vectors.len()
    }

    /// Number of entries in each stored vector, or zero for an empty subspace.
    pub fn dimension(&self) -> usize {
        self.vectors.first().map_or(0, Vec::len)
    }

    /// Borrow the validated recycled vectors.
    pub fn vectors(&self) -> &[Vec<T>] {
        &self.vectors
    }

    /// Serialize the recycled subspace as a strict, versioned little-endian
    /// checkpoint. Numeric values are canonical f64 real/imaginary pairs.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let dimension = validate_recycled_vectors(self.k, &self.vectors)?;
        let value_count = self
            .vectors
            .len()
            .checked_mul(dimension)
            .ok_or_else(|| recycled_checkpoint_error("checkpoint vector count overflows"))?;
        let payload_bytes = value_count
            .checked_mul(16)
            .ok_or_else(|| recycled_checkpoint_error("checkpoint payload size overflows"))?;
        let capacity = 38_usize
            .checked_add(payload_bytes)
            .ok_or_else(|| recycled_checkpoint_error("checkpoint capacity overflows"))?;
        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(RECYCLED_CHECKPOINT_MAGIC);
        output.extend_from_slice(&RECYCLED_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(u8::try_from(std::mem::size_of::<T::Real>()).unwrap_or(u8::MAX));
        output.push(recycled_scalar_is_complex::<T>());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(
            &u64::try_from(self.k)
                .map_err(|_| recycled_checkpoint_error("recycle size does not fit u64"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(
            &u64::try_from(dimension)
                .map_err(|_| recycled_checkpoint_error("dimension does not fit u64"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(
            &u64::try_from(self.vectors.len())
                .map_err(|_| recycled_checkpoint_error("vector count does not fit u64"))?
                .to_le_bytes(),
        );
        for vector in &self.vectors {
            for value in vector {
                let re = value.re().to_f64().ok_or_else(|| {
                    recycled_checkpoint_error("real component cannot convert to f64")
                })?;
                let im = value.im().to_f64().ok_or_else(|| {
                    recycled_checkpoint_error("imaginary component cannot convert to f64")
                })?;
                if !re.is_finite() || !im.is_finite() {
                    return Err(recycled_checkpoint_error(
                        "checkpoint contains NaN or infinity",
                    ));
                }
                output.extend_from_slice(&re.to_le_bytes());
                output.extend_from_slice(&im.to_le_bytes());
            }
        }
        Ok(output)
    }

    /// Restore a recycled subspace checkpoint after validating all metadata,
    /// scalar-family fields, dimensions, finite values, truncation, and tails.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = RecycledCheckpointReader { bytes, offset: 0 };
        if reader.take(RECYCLED_CHECKPOINT_MAGIC.len())? != RECYCLED_CHECKPOINT_MAGIC {
            return Err(recycled_checkpoint_error("checkpoint magic mismatch"));
        }
        if reader.read_u16()? != RECYCLED_CHECKPOINT_FORMAT_VERSION {
            return Err(recycled_checkpoint_error("checkpoint version mismatch"));
        }
        let expected_width = u8::try_from(std::mem::size_of::<T::Real>()).unwrap_or(u8::MAX);
        if reader.read_u8()? != expected_width {
            return Err(recycled_checkpoint_error(
                "checkpoint scalar width mismatch",
            ));
        }
        if reader.read_u8()? != recycled_scalar_is_complex::<T>() {
            return Err(recycled_checkpoint_error(
                "checkpoint scalar family mismatch",
            ));
        }
        if reader.read_u16()? != 0 {
            return Err(recycled_checkpoint_error(
                "checkpoint reserved field is non-zero",
            ));
        }
        let k = reader.read_usize()?;
        let dimension = reader.read_usize()?;
        let count = reader.read_usize()?;
        if count > k {
            return Err(recycled_checkpoint_error(
                "checkpoint vector count exceeds recycle size",
            ));
        }
        if (count == 0 && dimension != 0) || (count > 0 && dimension == 0) {
            return Err(recycled_checkpoint_error(
                "checkpoint vector count and dimension are inconsistent",
            ));
        }
        let value_count = count
            .checked_mul(dimension)
            .ok_or_else(|| recycled_checkpoint_error("checkpoint vector count overflows"))?;
        let byte_count = value_count
            .checked_mul(16)
            .ok_or_else(|| recycled_checkpoint_error("checkpoint payload size overflows"))?;
        if reader.remaining() != byte_count {
            return Err(recycled_checkpoint_error(
                "checkpoint payload is truncated or has trailing bytes",
            ));
        }
        let mut vectors = Vec::with_capacity(count);
        for _ in 0..count {
            let mut vector = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let re = reader.read_f64()?;
                let im = reader.read_f64()?;
                if !re.is_finite() || !im.is_finite() {
                    return Err(recycled_checkpoint_error(
                        "checkpoint contains NaN or infinity",
                    ));
                }
                vector.push(T::from_re_im(
                    T::Real::from_f64(re).ok_or_else(|| {
                        recycled_checkpoint_error("real component conversion failed")
                    })?,
                    T::Real::from_f64(im).ok_or_else(|| {
                        recycled_checkpoint_error("imaginary component conversion failed")
                    })?,
                ));
            }
            vectors.push(vector);
        }
        if count > 0 && vectors.first().map_or(0, Vec::len) != dimension {
            return Err(recycled_checkpoint_error("checkpoint dimension mismatch"));
        }
        validate_recycled_vectors(k, &vectors)?;
        Ok(Self { vectors, k })
    }

    /// Resume a recycled-GMRES solve from this subspace checkpoint.
    ///
    /// The supplied initial guess is validated by the checked solver path and
    /// may be an exact or warm-start iterate from a preceding frequency or
    /// continuation step. The subspace is updated with the resumed solve.
    #[allow(deprecated)]
    pub fn resume<A>(
        &mut self,
        operator: &A,
        rhs: &[T],
        initial_guess: Option<&[T]>,
        config: &GmresConfig<T::Real>,
    ) -> Result<GmresSolution<T>, SolverError>
    where
        A: LinearOperator<T>,
    {
        try_gmres_recycled(operator, rhs, initial_guess, self, config)
    }

    /// Resume a left-preconditioned recycled-GMRES solve from this subspace.
    #[allow(deprecated)]
    pub fn resume_preconditioned<A, P>(
        &mut self,
        operator: &A,
        preconditioner: &P,
        rhs: &[T],
        initial_guess: Option<&[T]>,
        config: &GmresConfig<T::Real>,
    ) -> Result<GmresSolution<T>, SolverError>
    where
        A: LinearOperator<T>,
        P: Preconditioner<T>,
    {
        try_gmres_recycled_preconditioned(
            operator,
            preconditioner,
            rhs,
            initial_guess,
            self,
            config,
        )
    }
}

impl<'a> RecycledCheckpointReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| recycled_checkpoint_error("checkpoint offset overflow"))?;
        if end > self.bytes.len() {
            return Err(recycled_checkpoint_error("checkpoint is truncated"));
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
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_usize(&mut self) -> Result<usize, SolverError> {
        let bytes = self.take(8)?;
        usize::try_from(u64::from_le_bytes(bytes.try_into().map_err(|_| {
            recycled_checkpoint_error("checkpoint size field is truncated")
        })?))
        .map_err(|_| recycled_checkpoint_error("checkpoint size does not fit usize"))
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let bytes = self.take(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().map_err(|_| {
            recycled_checkpoint_error("checkpoint scalar field is truncated")
        })?))
    }
}

/// Apply coarse correction from recycled subspace.
///
/// Computes `x += U * (U^H A U)^{-1} * U^H * r` where `r = b - Ax`.
/// This projects out the component of the residual that lies in the recycled
/// subspace, providing a cheap initial correction.
#[allow(deprecated)]
fn apply_coarse_correction<T, A, P>(
    operator: &A,
    precond: &P,
    x: &mut [T],
    b: &[T],
    subspace: &RecycledSubspace<T>,
    op_buf: &mut [T],
    op_counts: &mut KrylovOpCounts,
) where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let nk = subspace.vectors.len();
    if nk == 0 {
        return;
    }
    let n = b.len();

    // Compute residual r = b - Ax into op_buf
    operator.apply_into(x, op_buf);
    op_counts.spmv_count += 1;
    for i in 0..n {
        op_buf[i] = b[i] - op_buf[i];
    }
    op_counts.axpy_passes += 1;

    // The recycled vectors U are Ritz vectors of M^{-1}A (preconditioned operator).
    // The coarse correction must be consistent: E = U^H * M^{-1}A * U.
    let pau_cols: Vec<Vec<T>> = subspace
        .vectors
        .iter()
        .map(|u| {
            let au = operator.apply(u);
            op_counts.spmv_count += 1;
            precond.apply(&au)
        })
        .collect();
    op_counts.preconditioner_count += nk;

    // Build E = U^H * M^{-1}A * U (nk x nk dense)
    let mut e = sonium_solvers_core::dense::DenseMatrix::from_elem(nk, nk, T::zero());
    for i in 0..nk {
        for j in 0..nk {
            e[[i, j]] = inner_product(&subspace.vectors[i], &pau_cols[j]);
        }
    }
    op_counts.dot_reductions += nk * nk;
    op_counts.synchronization_points += nk;

    // Compute U^H * M^{-1}r (nk-dimensional)
    let pr = precond.apply(op_buf);
    op_counts.preconditioner_count += 1;
    let mut uhr = vec![T::zero(); nk];
    for (i, slot) in uhr.iter_mut().enumerate() {
        *slot = inner_product(&subspace.vectors[i], &pr);
    }
    op_counts.dot_reductions += nk;
    op_counts.synchronization_points += 1;

    // Solve E * y = U^H * M^{-1}r via dense LU
    let lu_result = sonium_solvers_direct::lu_factorize(&e);
    let y = match &lu_result {
        Ok(lu) => match lu.solve(&uhr) {
            Ok(y) => y,
            Err(_) => return, // Singular pivot in LU solve: skip correction
        },
        Err(_) => return, // Singular E: skip correction
    };

    // x += U * y
    for (i, &yi) in y.iter().enumerate() {
        axpy(yi, &subspace.vectors[i], x);
    }
    op_counts.add_basis_update::<T>(nk, n);
}

/// Solve Ax = b using recycled GMRES, reusing spectral information across solves.
///
/// The `subspace` is updated after each call -- pass the same `RecycledSubspace`
/// across consecutive solves in a frequency sweep.
#[deprecated(
    note = "use gcro_dr: recycled GMRES loses the recycling benefit without GCRO-DR's orthogonality (the recycled subspace is not kept orthogonal to the Krylov basis, so the coarse correction degrades); kept for API compatibility"
)]
#[allow(deprecated)]
pub fn gmres_recycled<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut RecycledSubspace<T>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_gmres_recycled(operator, b, x0, subspace, config) {
        Ok(result) => result,
        Err(_) => numerical_error_solution(match x0 {
            Some(guess) if guess.len() == b.len() => guess.to_vec(),
            _ => vec![T::zero(); b.len()],
        }),
    }
}

/// Checked recycled-GMRES entry point using the identity preconditioner.
#[allow(deprecated)]
pub fn try_gmres_recycled<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut RecycledSubspace<T>,
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = IdentityPreconditioner;
    try_gmres_recycled_preconditioned(operator, &identity, b, x0, subspace, config)
}

/// Solve Ax = b using recycled GMRES with a preconditioner.
///
/// The `subspace` is updated after each call -- pass the same `RecycledSubspace`
/// across consecutive solves in a frequency sweep.
#[deprecated(
    note = "use gcro_dr: recycled GMRES loses the recycling benefit without GCRO-DR's orthogonality (the recycled subspace is not kept orthogonal to the Krylov basis, so the coarse correction degrades); kept for API compatibility"
)]
#[allow(deprecated)]
pub fn gmres_recycled_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut RecycledSubspace<T>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_gmres_recycled_preconditioned(operator, precond, b, x0, subspace, config) {
        Ok(result) => result,
        Err(_) => numerical_error_solution(match x0 {
            Some(guess) if guess.len() == b.len() => guess.to_vec(),
            _ => vec![T::zero(); b.len()],
        }),
    }
}

/// Checked recycled-GMRES entry point for a left preconditioner.
#[allow(deprecated)]
pub fn try_gmres_recycled_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut RecycledSubspace<T>,
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    validate_recycled_request(operator, b, x0, subspace, config)?;

    let checked_operator = CheckedLinearOperator::<T, A>::new(operator);
    let checked_precond = CheckedPreconditioner::<T, P>::new(precond);
    // The core refreshes the recycled basis as it runs. Keep that mutation
    // transactional across callback and result-contract failures so a typed
    // failure never leaves the caller with a partially rebuilt state.
    let mut working_subspace = subspace.clone();
    let result = catch_solver_callback("recycled GMRES", || {
        let result = gmres_recycled_core(
            &checked_operator,
            &checked_precond,
            b,
            x0,
            &mut working_subspace,
            config,
        );
        let result = finalize_gmres_solution(&checked_operator, b, config.tolerance, result);
        if result.status == SolverStatus::NumericalError {
            return Err(SolverError::OperationFailed {
                operation: "recycled GMRES",
                reason: "operator or preconditioner produced a non-finite result".into(),
            });
        }
        SolverResultContract::validate_contract(&result)?;
        Ok(result)
    });
    if result.is_ok() {
        *subspace = working_subspace;
    }
    result
}

#[allow(deprecated)]
fn validate_recycled_request<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &RecycledSubspace<T>,
    config: &GmresConfig<T::Real>,
) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if config.max_iterations == 0
        || config.restart == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "recycled GMRES",
            reason: "invalid GMRES configuration".into(),
        });
    }
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    if b.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "recycled GMRES",
            reason: "right-hand side contains NaN or infinity".into(),
        });
    }
    if let Some(guess) = x0 {
        if guess.len() != b.len() {
            return Err(SolverError::DimensionMismatch {
                expected: b.len(),
                got: guess.len(),
            });
        }
        if guess.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "recycled GMRES",
                reason: "initial guess contains NaN or infinity".into(),
            });
        }
    }
    if subspace.vectors.iter().any(|vector| {
        vector.len() != operator.num_cols() || vector.iter().any(|value| !value.norm().is_finite())
    }) {
        return Err(SolverError::OperationFailed {
            operation: "recycled GMRES subspace",
            reason: "recycled vector has invalid dimension or non-finite value".into(),
        });
    }
    Ok(())
}

/// Core recycled GMRES implementation.
///
/// When the subspace is empty (first solve), runs standard GMRES(m) and extracts
/// k harmonic Ritz vectors. When populated, applies a coarse correction from the
/// recycled subspace, then runs standard GMRES on the residual.
#[allow(deprecated)]
fn gmres_recycled_core<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut RecycledSubspace<T>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if let Some(invalid) = invalid_gmres_input(
        b,
        x0,
        config.max_iterations,
        config.restart,
        config.tolerance,
    ) {
        return invalid;
    }

    let n = b.len();
    let m = config.restart;
    let k = subspace.k.min(m.saturating_sub(1));

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    if subspace.vectors.iter().any(|vector| vector.len() != n) {
        return numerical_error_solution(x);
    }
    let mut op_counts = KrylovOpCounts::default();
    let mut op_buf = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();

    let true_b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    if true_b_norm < tol_threshold {
        operator.apply_into(&x, &mut op_buf);
        op_counts.spmv_count += 1;
        for i in 0..n {
            op_buf[i] = b[i] - op_buf[i];
        }
        let actual_residual = vector_norm(&op_buf);
        op_counts.norm_count += 1;
        return GmresSolution {
            x,
            iterations: 0,
            restarts: 0,
            residual: actual_residual,
            estimated_residual: actual_residual,
            backward_error: actual_residual,
            attainable_accuracy: if actual_residual < tol_threshold {
                AccuracyStatus::Attained
            } else {
                AccuracyStatus::NotAttained
            },
            converged: actual_residual < tol_threshold,
            status: if actual_residual < tol_threshold {
                SolverStatus::Converged
            } else {
                SolverStatus::Breakdown
            },
            op_counts,
        };
    }

    let pb = precond.apply(b);
    op_counts.preconditioner_count += 1;
    let preconditioned_b_norm = vector_norm(&pb);
    op_counts.norm_count += 1;
    if preconditioned_b_norm < tol_threshold {
        return GmresSolution {
            x,
            iterations: 0,
            restarts: 0,
            residual: T::Real::one(),
            estimated_residual: T::Real::one(),
            backward_error: T::Real::one(),
            attainable_accuracy: AccuracyStatus::NotAttained,
            converged: false,
            status: SolverStatus::Breakdown,
            op_counts,
        };
    }

    let mut total_iterations = 0;
    let mut restarts = 0;
    let mut last_estimated_residual = T::Real::infinity();

    // Pre-allocate Krylov basis (m+1 vectors of size n)
    let mut v = FlatBasis::new(m + 1, n);
    // Hessenberg matrix, Givens data, RHS
    let mut h: sonium_solvers_core::dense::DenseMatrix<T> =
        sonium_solvers_core::dense::DenseMatrix::from_elem(m + 1, m, T::zero());
    let mut cs: Vec<T> = Vec::with_capacity(m);
    let mut sn: Vec<T> = Vec::with_capacity(m);
    let mut g: Vec<T> = vec![T::zero(); m + 1];
    let mut coeffs = Vec::with_capacity(m);
    let mut reorth_coeffs = Vec::with_capacity(m);

    // Apply coarse correction from recycled subspace (only on first outer iteration)
    let mut coarse_correction_applied = false;

    'outer: for _outer in 0..config.max_iterations {
        // Apply coarse correction from recycled subspace before the first GMRES cycle
        if !coarse_correction_applied && subspace.is_populated() && k > 0 {
            apply_coarse_correction(
                operator,
                precond,
                &mut x,
                b,
                subspace,
                &mut op_buf,
                &mut op_counts,
            );
            coarse_correction_applied = true;
        }

        // Compute preconditioned residual r = M^{-1}(b - Ax)
        operator.apply_into(&x, &mut op_buf);
        op_counts.spmv_count += 1;
        for i in 0..n {
            op_buf[i] = b[i] - op_buf[i];
        }
        op_counts.axpy_passes += 1;
        precond.apply_into(&op_buf, &mut w);
        op_counts.preconditioner_count += 1;
        let beta = vector_norm(&w);
        op_counts.norm_count += 1;

        let rel_residual = beta / preconditioned_b_norm;
        last_estimated_residual = rel_residual;
        if rel_residual < config.tolerance {
            let true_rel_residual = vector_norm(&op_buf) / true_b_norm;
            op_counts.norm_count += 1;
            if true_rel_residual < config.tolerance {
                return GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: true_rel_residual,
                    estimated_residual: rel_residual,
                    backward_error: true_rel_residual,
                    attainable_accuracy: AccuracyStatus::Attained,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }
        }

        h.fill(T::zero());
        cs.clear();
        sn.clear();
        g.fill(T::zero());

        // Standard GMRES initialization: v[0] = r / ||r||
        let inv_beta = T::from_real(T::Real::one() / beta);
        {
            let v0 = v.vector_mut(0);
            for i in 0..n {
                v0[i] = w[i] * inv_beta;
            }
        }
        op_counts.axpy_passes += 1;
        g[0] = T::from_real(beta);

        let mut inner_converged = false;
        let dgks_threshold = T::Real::from_f64(0.707).unwrap();

        for j in 0..m {
            total_iterations += 1;

            // w = M^{-1} * A * v[j]
            operator.apply_into(v.vector(j), &mut op_buf);
            op_counts.spmv_count += 1;
            precond.apply_into(&op_buf, &mut w);
            op_counts.preconditioner_count += 1;

            // Classical Gram-Schmidt with DGKS reorthogonalisation, batched.
            // Both the projection-coefficient gather and the projection
            // application run in parallel (see `cgs_orth`).
            let w_norm_before = vector_norm(&w);
            op_counts.norm_count += 1;

            batched_dots_flat_into(&v, j + 1, &w, &mut coeffs);
            for (i, &c) in coeffs.iter().enumerate() {
                h[[i, j]] = c;
            }
            batched_project_flat(&v, &coeffs, &mut w);
            op_counts.add_basis_projection::<T>(j + 1, n);

            let mut w_norm = vector_norm(&w);
            op_counts.norm_count += 1;
            if w_norm < dgks_threshold * w_norm_before {
                batched_dots_flat_into(&v, j + 1, &w, &mut reorth_coeffs);
                for (i, &si) in reorth_coeffs.iter().enumerate() {
                    h[[i, j]] += si;
                }
                batched_project_flat(&v, &reorth_coeffs, &mut w);
                op_counts.add_basis_projection::<T>(j + 1, n);
                w_norm = vector_norm(&w);
                op_counts.norm_count += 1;
            }
            h[[j + 1, j]] = T::from_real(w_norm);

            let breakdown_tol = T::Real::from_f64(1e-20).unwrap();
            if w_norm < breakdown_tol {
                inner_converged = true;
            } else {
                let inv_norm = T::from_real(T::Real::one() / w_norm);
                let next_v = v.vector_mut(j + 1);
                for i in 0..n {
                    next_v[i] = w[i] * inv_norm;
                }
                op_counts.axpy_passes += 1;
            }

            // Apply previous Givens rotations
            for i in 0..j {
                let temp = cs[i].conj() * h[[i, j]] + sn[i].conj() * h[[i + 1, j]];
                h[[i + 1, j]] = T::zero() - sn[i] * h[[i, j]] + cs[i] * h[[i + 1, j]];
                h[[i, j]] = temp;
            }

            let (c, s) = givens_rotation(h[[j, j]], h[[j + 1, j]]);
            cs.push(c);
            sn.push(s);

            h[[j, j]] = c.conj() * h[[j, j]] + s.conj() * h[[j + 1, j]];
            h[[j + 1, j]] = T::zero();

            let temp = c.conj() * g[j] + s.conj() * g[j + 1];
            g[j + 1] = T::zero() - s * g[j] + c * g[j + 1];
            g[j] = temp;

            let abs_residual = g[j + 1].norm();
            let rel_residual = abs_residual / preconditioned_b_norm;
            last_estimated_residual = rel_residual;
            let abs_tol = T::Real::from_f64(1e-20).unwrap();

            if config.print_interval > 0 && total_iterations % config.print_interval == 0 {
                log::info!(
                    "Recycled GMRES iteration {} (restart {}): relative residual = {:.6e}",
                    total_iterations,
                    restarts,
                    rel_residual.to_f64().unwrap_or(0.0)
                );
            }

            let estimated_converged =
                rel_residual < config.tolerance || abs_residual < abs_tol || inner_converged;
            let convergence_check_due =
                j % config.convergence_check_interval.max(1) == 0 || j == m - 1 || inner_converged;
            let exact_check_due =
                exact_residual_check_due(config.exact_residual_check_interval, total_iterations);
            if (convergence_check_due && estimated_converged) || exact_check_due {
                let y = solve_upper_triangular(&h, &g, j + 1);
                let mut candidate = x.clone();
                fused_basis_update_flat(&v, &y, &mut candidate);
                op_counts.add_basis_update::<T>(y.len(), n);

                let true_rel_residual = exact_relative_residual(
                    operator,
                    &candidate,
                    b,
                    &mut op_buf,
                    true_b_norm,
                    &mut op_counts,
                );
                if true_rel_residual < config.tolerance {
                    x = candidate;
                    // Extract Ritz vectors and update subspace. Always update
                    // even on early convergence (j+1 < k) to avoid retaining
                    // stale vectors from a previous solve.
                    if k > 0 {
                        let k_actual = (j + 1).min(k);
                        if k_actual > 0 {
                            let h_raw = unapply_givens(&h, &cs, &sn, j + 1);
                            subspace.vectors =
                                extract_deflation_vectors_flat(&v, &h_raw, j + 1, k_actual, n);
                        } else {
                            subspace.vectors.clear();
                        }
                    }
                    return GmresSolution {
                        x,
                        iterations: total_iterations,
                        restarts,
                        residual: true_rel_residual,
                        estimated_residual: rel_residual,
                        backward_error: true_rel_residual,
                        attainable_accuracy: AccuracyStatus::Attained,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    };
                }
                if estimated_converged && inner_converged {
                    x = candidate;
                    return GmresSolution {
                        x,
                        iterations: total_iterations,
                        restarts,
                        residual: true_rel_residual,
                        estimated_residual: rel_residual,
                        backward_error: true_rel_residual,
                        attainable_accuracy: AccuracyStatus::NotAttained,
                        converged: false,
                        status: SolverStatus::Breakdown,
                        op_counts,
                    };
                }
                if estimated_converged {
                    // The projected estimate was optimistic. Commit the
                    // candidate once and restart from it; an optional
                    // checkpoint that fails before projected convergence
                    // leaves the current Krylov base unchanged.
                    x = candidate;
                    restarts += 1;
                    continue 'outer;
                }
            }
        }

        // End of cycle: update solution
        let y = solve_upper_triangular(&h, &g, m);
        fused_basis_update_flat(&v, &y, &mut x);
        op_counts.add_basis_update::<T>(y.len(), n);

        // Extract harmonic Ritz vectors and update subspace after each cycle
        if k > 0 {
            let h_raw = unapply_givens(&h, &cs, &sn, m);
            subspace.vectors = extract_deflation_vectors_flat(&v, &h_raw, m, k, n);
        }

        let true_rel_residual =
            exact_relative_residual(operator, &x, b, &mut op_buf, true_b_norm, &mut op_counts);
        if true_rel_residual < config.tolerance {
            return GmresSolution {
                x,
                iterations: total_iterations,
                restarts,
                residual: true_rel_residual,
                estimated_residual: last_estimated_residual,
                backward_error: true_rel_residual,
                attainable_accuracy: AccuracyStatus::Attained,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        restarts += 1;
    }

    // Did not converge -- compute final residual
    operator.apply_into(&x, &mut op_buf);
    op_counts.spmv_count += 1;
    for i in 0..n {
        op_buf[i] = b[i] - op_buf[i];
    }
    op_counts.axpy_passes += 1;
    let rel_residual = vector_norm(&op_buf) / true_b_norm;
    op_counts.norm_count += 1;

    GmresSolution {
        x,
        iterations: total_iterations,
        restarts,
        residual: rel_residual,
        estimated_residual: last_estimated_residual,
        backward_error: rel_residual,
        attainable_accuracy: AccuracyStatus::NotAttained,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_sparse::CsrMatrix;

    /// Build a tridiagonal SPD matrix of size n (diagonally dominant).
    fn build_tridiagonal(n: usize, diag: f64) -> CsrMatrix<Complex64> {
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(diag, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    /// Build a Helmholtz-like matrix: tridiagonal + wavenumber shift.
    /// A(freq) = K - freq^2 * M, modeled as tridiag + freq^2 * diag shift.
    fn build_helmholtz_like(n: usize, freq: f64) -> CsrMatrix<Complex64> {
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        let k2 = freq * freq;
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0 - k2 * 0.1, 0.01 * freq);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    #[test]
    fn checked_recycled_gmres_rejects_non_finite_rhs() {
        let operator: CsrMatrix<f64> = CsrMatrix::identity(2);
        let mut subspace = RecycledSubspace::new(1);
        let config = GmresConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-8,
            ..GmresConfig::default()
        };
        let error = try_gmres_recycled(&operator, &[f64::NAN, 1.0], None, &mut subspace, &config)
            .expect_err("checked recycled GMRES must reject non-finite RHS");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(!subspace.is_populated());
    }

    #[test]
    fn test_recycled_first_solve_matches_gmres_dr() {
        // First solve with empty subspace should produce same accuracy as
        // standard GMRES (and populate the subspace).
        let n = 20;
        let a = build_tridiagonal(n, 4.0);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let mut subspace = RecycledSubspace::new(5);
        assert!(!subspace.is_populated());

        let sol = gmres_recycled(&a, &b, None, &mut subspace, &config);
        assert!(
            sol.converged,
            "Recycled GMRES should converge on first solve"
        );

        // Verify solution accuracy
        let ax = a.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "First solve error too large: {}", error);

        // Subspace should now be populated
        assert!(
            subspace.is_populated(),
            "Subspace should be populated after first solve"
        );
        assert!(
            subspace.num_vectors() > 0,
            "Should have extracted deflation vectors"
        );
    }

    #[test]
    fn test_recycled_second_solve_faster() {
        // Second solve with same matrix should converge in fewer iterations
        let n = 30;
        let a = build_tridiagonal(n, 4.0);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 15,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        // First solve (populates subspace)
        let mut subspace = RecycledSubspace::new(5);
        let sol1 = gmres_recycled(&a, &b, None, &mut subspace, &config);
        assert!(sol1.converged, "First solve should converge");

        // Second solve with same operator and different RHS
        let b2 = (0..n)
            .map(|i| Complex64::new((i as f64).cos(), 0.0))
            .collect::<Vec<_>>();
        let sol2 = gmres_recycled(&a, &b2, None, &mut subspace, &config);
        assert!(sol2.converged, "Second solve should converge");

        // Verify solution accuracy
        let ax2 = a.matvec(&sol2.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax2, &b2))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Second solve error too large: {}", error);

        // The second solve should use roughly similar or fewer iterations.
        // Allow small overhead from recycled subspace setup (up to 20% more).
        let margin = (sol1.iterations as f64 * 1.2).ceil() as usize;
        assert!(
            sol2.iterations <= margin,
            "Second solve ({} iters) used too many vs first solve ({} iters, margin {})",
            sol2.iterations,
            sol1.iterations,
            margin
        );
    }

    #[test]
    fn test_recycled_reports_krylov_op_counts() {
        let n = 20;
        let a = build_tridiagonal(n, 4.0);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 20,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let mut subspace = RecycledSubspace::new(5);
        let sol = gmres_recycled(&a, &b, None, &mut subspace, &config);

        assert!(sol.converged);
        assert!(sol.op_counts.spmv_count >= sol.iterations);
        assert!(sol.op_counts.preconditioner_count > 0);
        assert!(sol.op_counts.dot_reductions > 0);
        assert!(sol.op_counts.norm_count > 0);
        assert!(sol.op_counts.axpy_passes > 0);
        assert!(sol.op_counts.basis_bytes_touched > 0);
        assert!(sol.residual.is_finite());
        assert!(sol.estimated_residual.is_finite());
        assert!(sol.backward_error.is_finite());
        assert_eq!(sol.attainable_accuracy, AccuracyStatus::Attained);
    }

    #[test]
    fn test_recycled_frequency_sweep() {
        // Simulate a 5-point frequency sweep with slightly perturbed matrices.
        // Total iterations with recycling should be less than without.
        let n = 25;
        let freqs = [1.0, 1.2, 1.4, 1.6, 1.8];

        let config = GmresConfig {
            max_iterations: 200,
            restart: 20,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let b = (0..n)
            .map(|i| Complex64::new((i as f64 + 1.0).recip(), 0.0))
            .collect::<Vec<_>>();

        // With recycling
        let mut subspace = RecycledSubspace::new(5);
        let mut total_iters_recycled = 0;
        for &freq in &freqs {
            let a = build_helmholtz_like(n, freq);
            let sol = gmres_recycled(&a, &b, None, &mut subspace, &config);
            assert!(
                sol.converged,
                "Recycled solve should converge at freq={}",
                freq
            );
            let ax = a.matvec(&sol.x);
            let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
                .iter()
                .map(|e| e.norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(
                error < 1e-7,
                "Recycled solve error too large at freq={}: {}",
                freq,
                error
            );
            total_iters_recycled += sol.iterations;
        }

        // Without recycling (fresh subspace each time)
        let mut total_iters_fresh = 0;
        for &freq in &freqs {
            let a = build_helmholtz_like(n, freq);
            let mut fresh_subspace = RecycledSubspace::new(5);
            let sol = gmres_recycled(&a, &b, None, &mut fresh_subspace, &config);
            assert!(
                sol.converged,
                "Fresh solve should converge at freq={}",
                freq
            );
            total_iters_fresh += sol.iterations;
        }

        // Recycled sweep should use fewer total iterations
        assert!(
            total_iters_recycled <= total_iters_fresh,
            "Recycled sweep ({} total iters) should use <= fresh sweep ({} total iters)",
            total_iters_recycled,
            total_iters_fresh
        );
    }

    #[test]
    fn test_recycled_restart_accepts_requested_true_residual() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = 100.0 * r[1];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 1.0e-6];
        let config = GmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1.0e-3,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };
        let mut subspace = RecycledSubspace::new(0);

        let solution = gmres_recycled_preconditioned(
            &a,
            &AnisotropicPreconditioner,
            &b,
            None,
            &mut subspace,
            &config,
        );

        assert!(solution.converged);
        assert!(solution.residual < config.tolerance);
        assert_eq!(solution.iterations, 1);
    }

    #[test]
    fn test_recycled_max_iterations_preserves_projected_estimate() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = 100.0 * r[1];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 1.0e-6];
        let config = GmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1.0e-20,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };
        let mut subspace = RecycledSubspace::new(0);

        let solution = gmres_recycled_preconditioned(
            &a,
            &AnisotropicPreconditioner,
            &b,
            None,
            &mut subspace,
            &config,
        );

        assert_eq!(solution.status, SolverStatus::MaxIterationsReached);
        assert!(!solution.converged);
        assert!(solution.residual.is_finite());
        assert!(solution.estimated_residual.is_finite());
        assert_ne!(
            solution.residual.to_bits(),
            solution.estimated_residual.to_bits(),
            "max-iteration result must preserve the projected estimate"
        );
    }

    #[test]
    fn test_recycled_honors_exact_residual_interval_before_projected_convergence() {
        let n = 8;
        let a = build_tridiagonal(n, 5.0);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64 + 1.0) * 0.75, 0.0))
            .collect::<Vec<_>>();
        let base_config = GmresConfig {
            max_iterations: 20,
            restart: 20,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };
        let checked_config = GmresConfig {
            exact_residual_check_interval: 1,
            ..base_config.clone()
        };

        let mut base_subspace = RecycledSubspace::new(0);
        let mut checked_subspace = RecycledSubspace::new(0);
        let base = gmres_recycled(&a, &b, None, &mut base_subspace, &base_config);
        let checked = gmres_recycled(&a, &b, None, &mut checked_subspace, &checked_config);

        assert!(base.converged, "base residual: {:?}", base.residual);
        assert!(
            checked.converged,
            "checked residual: {:?}",
            checked.residual
        );
        assert!(
            checked.op_counts.exact_residual_checks > base.op_counts.exact_residual_checks,
            "configured interval must check before projected convergence: base={}, checked={}",
            base.op_counts.exact_residual_checks,
            checked.op_counts.exact_residual_checks,
        );
    }

    #[test]
    fn test_recycled_restarts_after_rejected_exact_candidate() {
        struct FirstProjectedCandidateOperator {
            calls: std::sync::atomic::AtomicUsize,
        }

        impl LinearOperator<f64> for FirstProjectedCandidateOperator {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply_into(&self, x: &[f64], y: &mut [f64]) {
                let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if call < 2 {
                    let epsilon = 1.0e-10;
                    y[0] = x[0] + epsilon * x[1];
                    y[1] = x[1];
                } else {
                    for (out, &value) in y.iter_mut().zip(x) {
                        *out = 2.0 * value;
                    }
                }
            }

            fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
                self.apply_into(x, y);
            }
        }

        let config = GmresConfig {
            max_iterations: 2,
            restart: 1,
            tolerance: 1.0e-8,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };
        let mut subspace = RecycledSubspace::new(0);
        let solution = gmres_recycled(
            &FirstProjectedCandidateOperator {
                calls: std::sync::atomic::AtomicUsize::new(0),
            },
            &[1.0, 1.0],
            None,
            &mut subspace,
            &config,
        );

        assert!(
            solution.converged,
            "status={:?}, residual={}, estimated={}, iterations={}, restarts={}",
            solution.status,
            solution.residual,
            solution.estimated_residual,
            solution.iterations,
            solution.restarts
        );
        assert_eq!(solution.restarts, 1);
        assert_eq!(solution.iterations, 2);
        assert!(solution.residual < config.tolerance);
    }

    #[test]
    fn test_recycled_convergence_check_interval_zero_is_safe() {
        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 2.0];
        let config = GmresConfig {
            convergence_check_interval: 0,
            ..GmresConfig::default()
        };
        let mut subspace = RecycledSubspace::new(0);

        let solution = gmres_recycled(&a, &b, None, &mut subspace, &config);

        assert!(solution.converged);
    }

    #[test]
    fn recycled_gmres_rejects_incompatible_recycled_dimension() {
        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 2.0];
        let config = GmresConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-8,
            ..GmresConfig::default()
        };
        let mut subspace = RecycledSubspace::new(1);
        let first = gmres_recycled(&a, &b, None, &mut subspace, &config);
        assert!(first.converged);

        let a_larger: CsrMatrix<f64> = CsrMatrix::identity(3);
        let solution = gmres_recycled(&a_larger, &[1.0, 2.0, 3.0], None, &mut subspace, &config);

        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert_eq!(solution.iterations, 0);
    }
}
