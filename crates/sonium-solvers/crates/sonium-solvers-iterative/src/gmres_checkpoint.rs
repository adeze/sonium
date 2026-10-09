//! Checked GMRES continuation and reusable solution checkpoints.
//!
//! A checkpoint stores the latest solution iterate, not the Krylov basis.  It
//! is therefore cheap to carry between frequency/design solves and remains
//! valid when only the right-hand side changes.  The checked resume path
//! revalidates the operator, RHS, configuration, and checkpoint dimensions
//! before entering the compatibility GMRES implementation.

use crate::gmres::{GmresConfig, GmresSolution, gmres_with_guess};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::traits::{
    ComplexField, LinearOperator, SolverError, SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// A reusable GMRES solution iterate for continuation across related solves.
#[derive(Debug, Clone)]
pub struct GmresCheckpoint<T: ComplexField> {
    solution: Vec<T>,
    dimension: usize,
    iterations: usize,
    restarts: usize,
    residual: T::Real,
}

/// Version of the portable GMRES solution-iterate checkpoint format.
pub const GMRES_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const CHECKPOINT_MAGIC: &[u8; 8] = b"SGMRESCP";

impl<T: ComplexField> GmresCheckpoint<T> {
    /// Capture a finite solution iterate from a previous GMRES result.
    pub fn from_solution(solution: &GmresSolution<T>) -> Result<Self, SolverError> {
        if let Some(index) = solution
            .x
            .iter()
            .position(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "GMRES checkpoint",
                reason: format!("solution contains NaN or infinity at index {index}"),
            });
        }
        if !solution.residual.is_finite() || solution.residual < T::Real::zero() {
            return Err(checkpoint_error(
                "solution residual must be finite and non-negative",
            ));
        }

        Ok(Self {
            dimension: solution.x.len(),
            solution: solution.x.clone(),
            iterations: solution.iterations,
            restarts: solution.restarts,
            residual: solution.residual,
        })
    }

    /// Capture a finite caller-provided initial iterate.
    pub fn from_iterate(iterate: &[T]) -> Result<Self, SolverError> {
        if let Some(index) = iterate.iter().position(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "GMRES checkpoint",
                reason: format!("iterate contains NaN or infinity at index {index}"),
            });
        }

        Ok(Self {
            solution: iterate.to_vec(),
            dimension: iterate.len(),
            iterations: 0,
            restarts: 0,
            residual: T::Real::zero(),
        })
    }

    /// Number of entries expected in a compatible operator and RHS.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow the stored iterate without exposing mutable checkpoint state.
    pub fn solution(&self) -> &[T] {
        &self.solution
    }

    /// Number of bytes occupied by the stored iterate.
    pub fn storage_bytes(&self) -> usize {
        self.solution.len() * std::mem::size_of::<T>()
    }

    /// Diagnostics captured when the checkpoint was created.
    pub const fn source_iterations(&self) -> usize {
        self.iterations
    }

    /// Diagnostics captured when the checkpoint was created.
    pub const fn source_restarts(&self) -> usize {
        self.restarts
    }

    /// Final residual captured when the checkpoint was created.
    pub fn source_residual(&self) -> T::Real {
        self.residual
    }

    /// Serialize this solution-iterate checkpoint in a versioned,
    /// little-endian format.
    ///
    /// Numeric values are stored as canonical `f64` real/imaginary pairs;
    /// restore requires an identical scalar family (including scalar size
    /// and real/complex representation).
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let validated = Self::from_parts(
            &self.solution,
            self.iterations,
            self.restarts,
            self.residual,
        )?;
        let mut output = Vec::new();
        output.extend_from_slice(CHECKPOINT_MAGIC);
        output.extend_from_slice(&GMRES_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(scalar_is_complex::<T>());
        output.push(0);
        push_u64(&mut output, validated.dimension)?;
        push_u64(&mut output, validated.iterations)?;
        push_u64(&mut output, validated.restarts)?;
        let residual = validated
            .residual
            .to_f64()
            .ok_or_else(|| checkpoint_error("residual cannot be represented as f64"))?;
        if !residual.is_finite() || residual < 0.0 {
            return Err(checkpoint_error(
                "checkpoint residual must be finite and non-negative",
            ));
        }
        output.extend_from_slice(&residual.to_le_bytes());
        push_u64(&mut output, validated.solution.len())?;
        for value in &validated.solution {
            let re = value
                .re()
                .to_f64()
                .ok_or_else(|| checkpoint_error("real component cannot be represented as f64"))?;
            let im = value.im().to_f64().ok_or_else(|| {
                checkpoint_error("imaginary component cannot be represented as f64")
            })?;
            if !re.is_finite() || !im.is_finite() {
                return Err(checkpoint_error("checkpoint contains non-finite value"));
            }
            output.extend_from_slice(&re.to_le_bytes());
            output.extend_from_slice(&im.to_le_bytes());
        }
        Ok(output)
    }

    /// Restore a checkpoint produced by [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = CheckpointReader { bytes, offset: 0 };
        if reader.take(8)? != CHECKPOINT_MAGIC {
            return Err(checkpoint_error("invalid GMRES checkpoint magic"));
        }
        if reader.read_u16()? != GMRES_CHECKPOINT_FORMAT_VERSION {
            return Err(checkpoint_error("unsupported GMRES checkpoint version"));
        }
        if reader.read_u8()? != std::mem::size_of::<T>() as u8
            || reader.read_u8()? != std::mem::size_of::<T::Real>() as u8
            || reader.read_u8()? != scalar_is_complex::<T>()
        {
            return Err(checkpoint_error("checkpoint scalar family does not match"));
        }
        if reader.read_u8()? != 0 {
            return Err(checkpoint_error("unsupported GMRES checkpoint flags"));
        }
        let dimension = reader.read_usize()?;
        let iterations = reader.read_usize()?;
        let restarts = reader.read_usize()?;
        let residual = reader.read_f64()?;
        if !residual.is_finite() || residual < 0.0 {
            return Err(checkpoint_error(
                "checkpoint residual must be finite and non-negative",
            ));
        }
        let solution_len = reader.read_usize()?;
        if solution_len != dimension {
            return Err(checkpoint_error(
                "checkpoint solution dimension does not match header",
            ));
        }
        let remaining_bytes = bytes.len().saturating_sub(reader.offset);
        if solution_len > remaining_bytes / 16 {
            return Err(checkpoint_error("truncated GMRES checkpoint solution"));
        }
        let mut solution = Vec::with_capacity(solution_len);
        for _ in 0..solution_len {
            let re = T::Real::from_f64(reader.read_f64()?)
                .ok_or_else(|| checkpoint_error("real component cannot be restored"))?;
            let im = T::Real::from_f64(reader.read_f64()?)
                .ok_or_else(|| checkpoint_error("imaginary component cannot be restored"))?;
            let value = T::from_re_im(re, im);
            if !value.norm().is_finite() {
                return Err(checkpoint_error("checkpoint restores non-finite value"));
            }
            solution.push(value);
        }
        if reader.offset != bytes.len() {
            return Err(checkpoint_error("trailing bytes in GMRES checkpoint"));
        }
        Self::from_parts(
            &solution,
            iterations,
            restarts,
            T::Real::from_f64(residual)
                .ok_or_else(|| checkpoint_error("residual cannot be restored"))?,
        )
    }

    /// Resume a checked GMRES solve using this iterate as the initial guess.
    pub fn resume<A>(
        &self,
        operator: &A,
        rhs: &[T],
        config: &GmresConfig<T::Real>,
    ) -> Result<GmresSolution<T>, SolverError>
    where
        A: LinearOperator<T>,
    {
        try_gmres_with_guess(operator, rhs, Some(&self.solution), config)
    }
}

impl<T: ComplexField> GmresCheckpoint<T> {
    fn from_parts(
        solution: &[T],
        iterations: usize,
        restarts: usize,
        residual: T::Real,
    ) -> Result<Self, SolverError> {
        if !residual.is_finite() || residual < T::Real::zero() {
            return Err(checkpoint_error(
                "solution residual must be finite and non-negative",
            ));
        }
        if let Some(index) = solution.iter().position(|value| !value.norm().is_finite()) {
            return Err(checkpoint_error(format!(
                "solution contains NaN or infinity at index {index}"
            )));
        }
        Ok(Self {
            solution: solution.to_vec(),
            dimension: solution.len(),
            iterations,
            restarts,
            residual,
        })
    }
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
            .ok_or_else(|| checkpoint_error("checkpoint offset overflow"))?;
        if end > self.bytes.len() {
            return Err(checkpoint_error("truncated GMRES checkpoint"));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(value))
    }

    fn read_usize(&mut self) -> Result<usize, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        usize::try_from(u64::from_le_bytes(value))
            .map_err(|_| checkpoint_error("checkpoint size does not fit usize"))
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        Ok(f64::from_le_bytes(value))
    }
}

fn scalar_is_complex<T: ComplexField>() -> u8 {
    u8::from(T::from_re_im(T::Real::zero(), T::Real::one()).norm() > T::Real::zero())
}

fn push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value =
        u64::try_from(value).map_err(|_| checkpoint_error("checkpoint size does not fit u64"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn checkpoint_error(reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "GMRES checkpoint",
        reason: reason.into(),
    }
}

/// Checked GMRES entry point with an optional initial iterate.
///
/// This mirrors [`crate::gmres::try_gmres`] while making the initial-guess
/// path available to checkpoint users.  Invalid checkpoint data is rejected
/// before any operator callback is invoked, and panics from legacy operators
/// are converted into the stable `SolverError` taxonomy.
pub fn try_gmres_with_guess<T, A>(
    operator: &A,
    rhs: &[T],
    initial_guess: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    let dimension = operator.num_cols();
    if rhs.len() != dimension {
        return Err(SolverError::DimensionMismatch {
            expected: dimension,
            got: rhs.len(),
        });
    }
    if let Some(guess) = initial_guess {
        if guess.len() != dimension {
            return Err(SolverError::DimensionMismatch {
                expected: dimension,
                got: guess.len(),
            });
        }
        if let Some(index) = guess.iter().position(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "GMRES",
                reason: format!("initial guess contains NaN or infinity at index {index}"),
            });
        }
    }
    if let Some(index) = rhs.iter().position(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "GMRES",
            reason: format!("right-hand side contains NaN or infinity at index {index}"),
        });
    }
    if config.max_iterations == 0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "max_iterations must be greater than zero".to_string(),
        });
    }
    if config.restart == 0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "restart must be greater than zero".to_string(),
        });
    }
    if !config.tolerance.is_finite() || config.tolerance < T::Real::zero() {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "tolerance must be finite and non-negative".to_string(),
        });
    }
    if config.min_restart == Some(0) || config.max_restart == Some(0) {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "adaptive restart bounds must be greater than zero".to_string(),
        });
    }
    if let (Some(min_restart), Some(max_restart)) = (config.min_restart, config.max_restart)
        && min_restart > max_restart
    {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "minimum restart exceeds maximum restart".to_string(),
        });
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        gmres_with_guess(operator, rhs, initial_guess, config)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "GMRES",
        reason: panic_reason(payload),
    })?;

    if result.status == SolverStatus::NumericalError {
        return Err(SolverError::OperationFailed {
            operation: "GMRES",
            reason: "GMRES produced a numerical-error result".to_string(),
        });
    }
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "operator panicked".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_sparse::CsrMatrix;

    fn config() -> GmresConfig<f64> {
        GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            ..GmresConfig::default()
        }
    }

    #[test]
    fn checkpoint_resume_matches_fresh_solve_for_second_rhs() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (2, 2, 2.0),
            ],
        );
        let first_rhs = [5.0, 4.0, 2.0];
        let second_rhs = [1.0, 7.0, 6.0];
        let first = crate::gmres::try_gmres(&matrix, &first_rhs, &config()).unwrap();
        let checkpoint = GmresCheckpoint::from_solution(&first).unwrap();
        assert_eq!(checkpoint.dimension(), 3);
        assert!(checkpoint.storage_bytes() >= 3 * std::mem::size_of::<f64>());

        let resumed = checkpoint.resume(&matrix, &second_rhs, &config()).unwrap();
        let fresh = crate::gmres::try_gmres(&matrix, &second_rhs, &config()).unwrap();
        for (actual, expected) in resumed.x.iter().zip(fresh.x.iter()) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        assert!(resumed.residual < 1.0e-10);
    }

    #[test]
    fn checkpoint_rejects_incompatible_operator_dimension_before_apply() {
        let checkpoint = GmresCheckpoint::from_iterate(&[0.0_f64, 0.0]).unwrap();
        let matrix = CsrMatrix::<f64>::identity(3);
        let error = checkpoint
            .resume(&matrix, &[1.0, 2.0, 3.0], &config())
            .expect_err("checkpoint dimension must be rejected");
        assert!(matches!(
            error,
            SolverError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn checkpoint_rejects_non_finite_iterates() {
        let error = GmresCheckpoint::from_iterate(&[0.0_f64, f64::NAN])
            .expect_err("non-finite checkpoint must be rejected");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "GMRES checkpoint",
                ..
            }
        ));
    }

    #[test]
    fn checked_guess_rejects_non_finite_guess_and_bad_configuration() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let error =
            try_gmres_with_guess(&matrix, &[1.0, 2.0], Some(&[0.0, f64::INFINITY]), &config())
                .expect_err("non-finite guess must be rejected");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "GMRES",
                ..
            }
        ));

        let mut invalid = config();
        invalid.restart = 0;
        let error = try_gmres_with_guess(&matrix, &[1.0, 2.0], None, &invalid)
            .expect_err("zero restart must be rejected");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "GMRES configuration",
                ..
            }
        ));
    }
}
