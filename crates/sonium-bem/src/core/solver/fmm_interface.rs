//! FMM-solver interface
//!
//! This module provides a unified interface for solving BEM systems using
//! either direct methods (TBEM) or iterative methods with FMM acceleration.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::{
    BiCgstabConfig, BiCgstabSolution, CgsConfig, CgsSolution, GmresConfig, GmresSolution,
};
use sonium_solvers::iterative::{bicgstab, gmres, try_gmres};
use sonium_solvers::preconditioners::{HiluPreconditioner, IluPreconditioner};
use sonium_solvers::traits::{
    AccuracyStatus, AdjointCapability, LinearOperator, Preconditioner, SolverError, SolverStatus,
};

use crate::core::assembly::mlfmm::{MlfmmNearFieldError, MlfmmSystem};
use crate::core::assembly::slfmm::{SlfmmNearFieldError, SlfmmSystem};
use sonium_solvers::CsrMatrix;

// ============================================================================
// Linear Operators
// ============================================================================

/// Dense matrix linear operator
pub struct DenseOperator {
    matrix: sonium_solvers::dense::DenseMatrix<Complex64>,
}

/// Errors returned by checked direct dense-operator actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DenseOperatorError {
    /// A vector input does not match the action domain.
    #[error("dense operator {operation} input length {actual} does not match expected {expected}")]
    InputLength {
        /// Action name.
        operation: &'static str,
        /// Required input length.
        expected: usize,
        /// Supplied input length.
        actual: usize,
    },
    /// A vector output buffer does not match the action range.
    #[error("dense operator {operation} output length {actual} does not match expected {expected}")]
    OutputLength {
        /// Action name.
        operation: &'static str,
        /// Required output length.
        expected: usize,
        /// Supplied output length.
        actual: usize,
    },
    /// A block input has the wrong number of rows.
    #[error(
        "dense operator {operation} block input rows {actual} do not match expected {expected}"
    )]
    BlockInputRows {
        /// Action name.
        operation: &'static str,
        /// Required input row count.
        expected: usize,
        /// Supplied input row count.
        actual: usize,
    },
    /// A block output has the wrong shape.
    #[error(
        "dense operator {operation} block output shape {actual_rows}x{actual_cols} does not match expected {expected_rows}x{expected_cols}"
    )]
    BlockOutputShape {
        /// Action name.
        operation: &'static str,
        /// Required output row count.
        expected_rows: usize,
        /// Required output column count.
        expected_cols: usize,
        /// Supplied output row count.
        actual_rows: usize,
        /// Supplied output column count.
        actual_cols: usize,
    },
    /// A matrix coefficient is non-finite.
    #[error("dense operator coefficient at flat index {index} is non-finite")]
    NonFiniteCoefficient {
        /// Flat row-major coefficient index.
        index: usize,
    },
    /// An input value is non-finite.
    #[error("dense operator {operation} input at index {index} is non-finite")]
    NonFiniteInput {
        /// Action name.
        operation: &'static str,
        /// Flat vector or block index.
        index: usize,
    },
    /// An output value is non-finite.
    #[error("dense operator {operation} output at index {index} is non-finite")]
    NonFiniteOutput {
        /// Action name.
        operation: &'static str,
        /// Flat vector or block index.
        index: usize,
    },
}

impl DenseOperator {
    /// Create a new dense operator from a matrix
    pub fn new(matrix: sonium_solvers::dense::DenseMatrix<Complex64>) -> Self {
        Self { matrix }
    }

    fn validate_matrix(&self) -> Result<(), DenseOperatorError> {
        if let Some(index) = self
            .matrix
            .as_slice()
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(DenseOperatorError::NonFiniteCoefficient { index });
        }
        Ok(())
    }

    fn validate_vector_input(
        &self,
        input: &[Complex64],
        expected: usize,
        operation: &'static str,
    ) -> Result<(), DenseOperatorError> {
        if input.len() != expected {
            return Err(DenseOperatorError::InputLength {
                operation,
                expected,
                actual: input.len(),
            });
        }
        if let Some(index) = input
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(DenseOperatorError::NonFiniteInput { operation, index });
        }
        self.validate_matrix()
    }

    fn validate_vector_output(
        output: &[Complex64],
        expected: usize,
        operation: &'static str,
    ) -> Result<(), DenseOperatorError> {
        if output.len() != expected {
            return Err(DenseOperatorError::OutputLength {
                operation,
                expected,
                actual: output.len(),
            });
        }
        if let Some(index) = output
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(DenseOperatorError::NonFiniteOutput { operation, index });
        }
        Ok(())
    }

    fn validate_block_input(
        &self,
        input: &DenseMatrix<Complex64>,
        expected_rows: usize,
        operation: &'static str,
    ) -> Result<(), DenseOperatorError> {
        if input.nrows() != expected_rows {
            return Err(DenseOperatorError::BlockInputRows {
                operation,
                expected: expected_rows,
                actual: input.nrows(),
            });
        }
        if let Some(index) = input
            .as_slice()
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(DenseOperatorError::NonFiniteInput { operation, index });
        }
        self.validate_matrix()
    }

    fn validate_block_output(
        output: &DenseMatrix<Complex64>,
        expected_rows: usize,
        expected_cols: usize,
        operation: &'static str,
    ) -> Result<(), DenseOperatorError> {
        if output.nrows() != expected_rows || output.ncols() != expected_cols {
            return Err(DenseOperatorError::BlockOutputShape {
                operation,
                expected_rows,
                expected_cols,
                actual_rows: output.nrows(),
                actual_cols: output.ncols(),
            });
        }
        if let Some(index) = output
            .as_slice()
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(DenseOperatorError::NonFiniteOutput { operation, index });
        }
        Ok(())
    }

    /// Checked direct dense forward action `A*x`.
    pub fn apply_checked(&self, x: &[Complex64]) -> Result<Vec<Complex64>, DenseOperatorError> {
        self.validate_vector_input(x, self.matrix.ncols(), "forward")?;
        let mut y = vec![Complex64::new(0.0, 0.0); self.matrix.nrows()];
        self.matrix.matvec_into(x, &mut y);
        Self::validate_vector_output(&y, self.matrix.nrows(), "forward")?;
        Ok(y)
    }

    /// Checked allocation-free direct dense forward action `A*x`.
    pub fn apply_into_checked(
        &self,
        x: &[Complex64],
        y: &mut [Complex64],
    ) -> Result<(), DenseOperatorError> {
        self.validate_vector_input(x, self.matrix.ncols(), "forward")?;
        if y.len() != self.matrix.nrows() {
            return Err(DenseOperatorError::OutputLength {
                operation: "forward",
                expected: self.matrix.nrows(),
                actual: y.len(),
            });
        }
        self.matrix.matvec_into(x, y);
        Self::validate_vector_output(y, self.matrix.nrows(), "forward")
    }

    /// Checked direct dense algebraic-transpose action `Aᵀ*x`.
    pub fn apply_transpose_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, DenseOperatorError> {
        self.validate_vector_input(x, self.matrix.nrows(), "transpose")?;
        let mut y = vec![Complex64::new(0.0, 0.0); self.matrix.ncols()];
        self.matrix.matvec_transpose_into(x, &mut y);
        Self::validate_vector_output(&y, self.matrix.ncols(), "transpose")?;
        Ok(y)
    }

    /// Checked allocation-free direct dense algebraic-transpose action.
    pub fn apply_transpose_into_checked(
        &self,
        x: &[Complex64],
        y: &mut [Complex64],
    ) -> Result<(), DenseOperatorError> {
        self.validate_vector_input(x, self.matrix.nrows(), "transpose")?;
        if y.len() != self.matrix.ncols() {
            return Err(DenseOperatorError::OutputLength {
                operation: "transpose",
                expected: self.matrix.ncols(),
                actual: y.len(),
            });
        }
        self.matrix.matvec_transpose_into(x, y);
        Self::validate_vector_output(y, self.matrix.ncols(), "transpose")
    }

    /// Checked direct dense Hermitian-transpose action `Aᴴ*x`.
    pub fn apply_hermitian_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, DenseOperatorError> {
        self.validate_vector_input(x, self.matrix.nrows(), "Hermitian")?;
        let mut y = vec![Complex64::new(0.0, 0.0); self.matrix.ncols()];
        self.matrix.matvec_hermitian_into(x, &mut y);
        Self::validate_vector_output(&y, self.matrix.ncols(), "Hermitian")?;
        Ok(y)
    }

    /// Checked allocation-free direct dense Hermitian-transpose action.
    pub fn apply_hermitian_into_checked(
        &self,
        x: &[Complex64],
        y: &mut [Complex64],
    ) -> Result<(), DenseOperatorError> {
        self.validate_vector_input(x, self.matrix.nrows(), "Hermitian")?;
        if y.len() != self.matrix.ncols() {
            return Err(DenseOperatorError::OutputLength {
                operation: "Hermitian",
                expected: self.matrix.ncols(),
                actual: y.len(),
            });
        }
        self.matrix.matvec_hermitian_into(x, y);
        Self::validate_vector_output(y, self.matrix.ncols(), "Hermitian")
    }

    /// Checked direct dense forward block action.
    pub fn apply_block_checked(
        &self,
        x: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, DenseOperatorError> {
        self.validate_block_input(x, self.matrix.ncols(), "forward block")?;
        let y = self.matrix.matmul(x);
        Self::validate_block_output(&y, self.matrix.nrows(), x.ncols(), "forward block")?;
        Ok(y)
    }

    /// Checked direct dense transpose block action.
    pub fn apply_transpose_block_checked(
        &self,
        x: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, DenseOperatorError> {
        self.validate_block_input(x, self.matrix.nrows(), "transpose block")?;
        let mut y = DenseMatrix::zeros(self.matrix.ncols(), x.ncols());
        for column in 0..x.ncols() {
            let rhs = (0..x.nrows())
                .map(|row| x[[row, column]])
                .collect::<Vec<_>>();
            let result = self.apply_transpose_checked(&rhs)?;
            for (row, value) in result.into_iter().enumerate() {
                y[[row, column]] = value;
            }
        }
        Self::validate_block_output(&y, self.matrix.ncols(), x.ncols(), "transpose block")?;
        Ok(y)
    }

    /// Checked direct dense Hermitian block action.
    pub fn apply_hermitian_block_checked(
        &self,
        x: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, DenseOperatorError> {
        self.validate_block_input(x, self.matrix.nrows(), "Hermitian block")?;
        let mut y = DenseMatrix::zeros(self.matrix.ncols(), x.ncols());
        for column in 0..x.ncols() {
            let rhs = (0..x.nrows())
                .map(|row| x[[row, column]])
                .collect::<Vec<_>>();
            let result = self.apply_hermitian_checked(&rhs)?;
            for (row, value) in result.into_iter().enumerate() {
                y[[row, column]] = value;
            }
        }
        Self::validate_block_output(&y, self.matrix.ncols(), x.ncols(), "Hermitian block")?;
        Ok(y)
    }
}

impl LinearOperator<Complex64> for DenseOperator {
    fn num_rows(&self) -> usize {
        self.matrix.nrows()
    }

    fn num_cols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.matrix.matvec(x)
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_into(x, y);
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.matrix.ncols()];
        self.matrix.matvec_transpose_into(x, &mut y);
        y
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_transpose_into(x, y);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

/// SLFMM linear operator
pub struct SlfmmOperator {
    system: SlfmmSystem,
}

impl SlfmmOperator {
    /// Create a new SLFMM operator
    pub fn new(system: SlfmmSystem) -> Self {
        Self { system }
    }

    /// Get a reference to the underlying system
    pub fn system(&self) -> &SlfmmSystem {
        &self.system
    }

    /// Get the RHS vector
    pub fn rhs(&self) -> &[Complex64] {
        &self.system.rhs
    }
}

impl LinearOperator<Complex64> for SlfmmOperator {
    fn num_rows(&self) -> usize {
        self.system.num_dofs
    }

    fn num_cols(&self) -> usize {
        self.system.num_dofs
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.system.num_dofs];
        self.system.matvec_into(x, &mut y);
        y
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.system.matvec_into(x, y);
    }

    fn apply_block(
        &self,
        x: &sonium_solvers::dense::DenseMatrix<Complex64>,
        y: &mut sonium_solvers::dense::DenseMatrix<Complex64>,
    ) {
        self.system.matvec_block_into(x, y);
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.system.num_dofs];
        self.system.matvec_transpose_into(x, &mut y);
        y
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.system.matvec_transpose_into(x, y);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

/// MLFMM linear operator
pub struct MlfmmOperator {
    system: MlfmmSystem,
}

impl MlfmmOperator {
    /// Create a new MLFMM operator
    pub fn new(system: MlfmmSystem) -> Self {
        Self { system }
    }

    /// Get a reference to the underlying system
    pub fn system(&self) -> &MlfmmSystem {
        &self.system
    }

    /// Get the RHS vector
    pub fn rhs(&self) -> &[Complex64] {
        &self.system.rhs
    }
}

impl LinearOperator<Complex64> for MlfmmOperator {
    fn num_rows(&self) -> usize {
        self.system.num_dofs
    }

    fn num_cols(&self) -> usize {
        self.system.num_dofs
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.system.num_dofs];
        self.system.matvec_into(x, &mut y);
        y
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.system.matvec_into(x, y);
    }

    fn apply_block(
        &self,
        x: &sonium_solvers::dense::DenseMatrix<Complex64>,
        y: &mut sonium_solvers::dense::DenseMatrix<Complex64>,
    ) {
        self.system.matvec_block_into(x, y);
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.system.num_dofs];
        self.system.matvec_transpose_into(x, &mut y);
        y
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.system.matvec_transpose_into(x, y);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

/// Linear-operator view of an exact Hermitian-transposed action.
struct HermitianOperator<O> {
    inner: O,
}

impl<O> LinearOperator<Complex64> for HermitianOperator<O>
where
    O: LinearOperator<Complex64>,
{
    fn num_rows(&self) -> usize {
        self.inner.num_cols()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_rows()
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let applied = self.inner.apply_hermitian(x);
        y.copy_from_slice(&applied);
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let conjugated_input: Vec<_> = x.iter().map(|value| value.conj()).collect();
        let applied = self.inner.apply(&conjugated_input);
        for (output, value) in y.iter_mut().zip(applied) {
            *output = value.conj();
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

/// Type alias for CSR operator — `CsrMatrix<Complex64>` from sonium-solvers
/// already implements `LinearOperator<Complex64>`, so no wrapper is needed.
pub type CsrOperator = CsrMatrix<Complex64>;

// ============================================================================
// Preconditioners
// ============================================================================

/// Diagonal preconditioner
pub struct DiagonalPreconditioner {
    inv_diag: Vec<Complex64>,
}

impl DiagonalPreconditioner {
    /// Create from a CSR matrix
    pub fn from_csr(matrix: &CsrMatrix<Complex64>) -> Self {
        let diag = matrix.diagonal();
        let inv_diag: Vec<Complex64> = diag
            .into_iter()
            .map(|d| {
                if d.norm() > 1e-15 {
                    Complex64::new(1.0, 0.0) / d
                } else {
                    Complex64::new(1.0, 0.0)
                }
            })
            .collect();
        Self { inv_diag }
    }

    /// Create from a diagonal vector
    pub fn from_diagonal(diag: Vec<Complex64>) -> Self {
        let inv_diag: Vec<Complex64> = diag
            .into_iter()
            .map(|d| {
                if d.norm() > 1e-15 {
                    Complex64::new(1.0, 0.0) / d
                } else {
                    Complex64::new(1.0, 0.0)
                }
            })
            .collect();
        Self { inv_diag }
    }
}

impl Preconditioner<Complex64> for DiagonalPreconditioner {
    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        x.iter()
            .zip(self.inv_diag.iter())
            .map(|(&xi, &di)| xi * di)
            .collect()
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        <Self as Preconditioner<Complex64>>::apply(self, x)
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

impl LinearOperator<Complex64> for DiagonalPreconditioner {
    fn num_rows(&self) -> usize {
        self.inv_diag.len()
    }

    fn num_cols(&self) -> usize {
        self.inv_diag.len()
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        <Self as Preconditioner<Complex64>>::apply(self, x)
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        y.copy_from_slice(&<Self as Preconditioner<Complex64>>::apply(self, x));
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        <Self as Preconditioner<Complex64>>::apply(self, x)
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        y.copy_from_slice(&<Self as LinearOperator<Complex64>>::apply_transpose(
            self, x,
        ));
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

// ============================================================================
// Solver Wrappers
// ============================================================================

/// Left-preconditioned view used by the legacy CGS compatibility wrapper.
///
/// The transformed system is `M⁻¹ A x = M⁻¹ b`; the iterate `x` therefore
/// remains the solution of the original system. Keeping this adapter local
/// gives checked callers a real ILU path without changing the solver crate's
/// historical CGS API.
struct LeftPreconditionedOperator<'a, O, P> {
    operator: &'a O,
    preconditioner: &'a P,
}

impl<O, P> LinearOperator<Complex64> for LeftPreconditionedOperator<'_, O, P>
where
    O: LinearOperator<Complex64>,
    P: Preconditioner<Complex64>,
{
    fn num_rows(&self) -> usize {
        self.operator.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.operator.num_cols()
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        let applied = self.operator.apply(x);
        self.preconditioner.apply(&applied)
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let applied = self.operator.apply(x);
        self.preconditioner.apply_into(&applied, y);
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let preconditioned = self.preconditioner.apply_transpose(x);
        self.operator.apply_transpose(&preconditioned)
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let preconditioned = self.preconditioner.apply_transpose(x);
        self.operator.apply_transpose_into(&preconditioned, y);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        combine_adjoint_capabilities(
            self.operator.transpose_capability(),
            self.preconditioner.transpose_capability(),
        )
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        combine_adjoint_capabilities(
            self.operator.hermitian_capability(),
            self.preconditioner.hermitian_capability(),
        )
    }
}

fn combine_adjoint_capabilities(
    left: AdjointCapability,
    right: AdjointCapability,
) -> AdjointCapability {
    if left == AdjointCapability::Unsupported || right == AdjointCapability::Unsupported {
        AdjointCapability::Unsupported
    } else if left == AdjointCapability::Unknown || right == AdjointCapability::Unknown {
        AdjointCapability::Unknown
    } else {
        AdjointCapability::Exact
    }
}

/// Recompute CGS metrics against the original (unpreconditioned) operator.
///
/// CGS sees `M⁻¹ A` and `M⁻¹ b`, so its recurrence estimate is not a physical
/// residual for the BEM system. The public result must instead be certified
/// against `A x = b` before exposing convergence to callers.
fn certify_cgs_solution<O: LinearOperator<Complex64>>(
    operator: &O,
    rhs: &[Complex64],
    config: &CgsConfig<f64>,
    solution: &mut CgsSolution<Complex64>,
) -> Result<(), SolverError> {
    let applied = operator.try_apply(&solution.x)?;
    if applied.len() != rhs.len() {
        return Err(SolverError::DimensionMismatch {
            expected: rhs.len(),
            got: applied.len(),
        });
    }
    let residual_norm = applied
        .iter()
        .zip(rhs)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let action_norm = applied
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    if !residual_norm.is_finite() || !action_norm.is_finite() || !rhs_norm.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "BEM preconditioned CGS",
            reason: "original-operator residual is non-finite".to_string(),
        });
    }

    solution.residual = residual_norm / rhs_norm.max(f64::MIN_POSITIVE);
    solution.backward_error = residual_norm / (action_norm + rhs_norm).max(f64::MIN_POSITIVE);
    if solution.residual <= config.tolerance {
        solution.converged = true;
        solution.status = SolverStatus::Converged;
        solution.attainable_accuracy = AccuracyStatus::Attained;
    } else {
        solution.converged = false;
        if solution.status == SolverStatus::Converged {
            solution.status = SolverStatus::Stagnated;
        }
        solution.attainable_accuracy = AccuracyStatus::NotAttained;
    }
    Ok(())
}

/// Recompute GMRES metrics against the original operator after a
/// preconditioned solve. The iterative estimate alone is not sufficient for a
/// compressed/FMM BEM acceptance gate.
fn solver_panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "solver callback panicked".to_string()
    }
}

fn certify_gmres_solution<O: LinearOperator<Complex64>>(
    operator: &O,
    rhs: &[Complex64],
    config: &GmresConfig<f64>,
    solution: &mut GmresSolution<Complex64>,
) -> Result<(), SolverError> {
    let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        operator.try_apply(&solution.x)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "BEM GMRES",
        reason: solver_panic_reason(payload),
    })??;
    if applied.len() != rhs.len() {
        return Err(SolverError::DimensionMismatch {
            expected: rhs.len(),
            got: applied.len(),
        });
    }
    let residual_norm = applied
        .iter()
        .zip(rhs)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let action_norm = applied
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    if !residual_norm.is_finite() || !action_norm.is_finite() || !rhs_norm.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "BEM preconditioned GMRES",
            reason: "original-operator residual is non-finite".to_string(),
        });
    }

    solution.residual = residual_norm / rhs_norm.max(f64::MIN_POSITIVE);
    solution.backward_error = residual_norm / (action_norm + rhs_norm).max(f64::MIN_POSITIVE);
    if solution.residual <= config.tolerance {
        solution.converged = true;
        solution.status = SolverStatus::Converged;
        solution.attainable_accuracy = AccuracyStatus::Attained;
    } else {
        solution.converged = false;
        if solution.status == SolverStatus::Converged {
            solution.status = SolverStatus::Stagnated;
        }
        solution.attainable_accuracy = AccuracyStatus::NotAttained;
    }
    Ok(())
}

fn try_ilu_from_checked_mlfmm_near_field<F>(
    extract_near_field: F,
    operation: &'static str,
) -> Result<IluPreconditioner<Complex64>, SolverError>
where
    F: FnOnce() -> Result<sonium_solvers::dense::DenseMatrix<Complex64>, MlfmmNearFieldError>,
{
    let preconditioner = extract_near_field()
        .map_err(|error| SolverError::OperationFailed {
            operation,
            reason: error.to_string(),
        })
        .and_then(|nearfield_matrix| {
            let csr = CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
            IluPreconditioner::try_from_csr(&csr).map_err(|error| SolverError::OperationFailed {
                operation,
                reason: error.to_string(),
            })
        })?;
    Ok(preconditioner)
}

fn try_ilu_from_checked_slfmm_near_field<F>(
    extract_near_field: F,
    operation: &'static str,
) -> Result<IluPreconditioner<Complex64>, SolverError>
where
    F: FnOnce() -> Result<sonium_solvers::dense::DenseMatrix<Complex64>, SlfmmNearFieldError>,
{
    let nearfield_matrix = extract_near_field().map_err(|error| SolverError::OperationFailed {
        operation,
        reason: error.to_string(),
    })?;
    let csr = CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
    IluPreconditioner::try_from_csr(&csr).map_err(|error| SolverError::OperationFailed {
        operation,
        reason: error.to_string(),
    })
}

/// Run a checked left-preconditioned GMRES solve and certify the returned
/// vector against the original, unpreconditioned operator.
///
/// The iterative crate exposes a total `gmres_preconditioned` compatibility
/// wrapper.  BEM's checked FMM entry points use this adapter so malformed
/// dimensions/configuration, callback panics, numerical failure statuses, and
/// residuals that only satisfy the preconditioned system are all surfaced as
/// typed errors.
fn try_gmres_preconditioned_checked<O, P>(
    operator: &O,
    preconditioner: &P,
    rhs: &[Complex64],
    config: &GmresConfig<f64>,
    operation: &'static str,
) -> Result<GmresSolution<Complex64>, SolverError>
where
    O: LinearOperator<Complex64>,
    P: Preconditioner<Complex64>,
{
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    if rhs.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: rhs.len(),
        });
    }
    if rhs
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if config.max_iterations == 0 || config.restart == 0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "max_iterations and restart must be positive".to_string(),
        });
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "tolerance must be finite and positive".to_string(),
        });
    }
    if config.min_restart == Some(0)
        || config.max_restart == Some(0)
        || matches!((config.min_restart, config.max_restart), (Some(min), Some(max)) if min > max)
    {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "adaptive restart bounds must be positive and ordered".to_string(),
        });
    }

    let mut solution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sonium_solvers::iterative::gmres_preconditioned(operator, preconditioner, rhs, config)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation,
        reason: solver_panic_reason(payload),
    })?;
    if solution.status == SolverStatus::NumericalError {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "GMRES returned numerical failure".to_string(),
        });
    }
    certify_gmres_solution(operator, rhs, config, &mut solution)?;
    Ok(solution)
}

/// Checked ILU-preconditioned CGS solve for a dense BEM matrix.
///
/// Unlike the historical infallible wrapper, this path never falls back to an
/// unpreconditioned solve. Invalid matrices, ILU setup failures, callback
/// failures, and failed residual certification are returned as typed errors.
pub fn try_solve_with_ilu(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &CgsConfig<f64>,
) -> Result<CgsSolution<Complex64>, SolverError> {
    let operator = DenseOperator::new(matrix.clone());
    try_solve_with_ilu_operator(&operator, matrix, b, config)
}

/// Checked ILU-preconditioned CGS solve for an arbitrary BEM operator.
pub fn try_solve_with_ilu_operator<O: LinearOperator<Complex64>>(
    operator: &O,
    nearfield_matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &CgsConfig<f64>,
) -> Result<CgsSolution<Complex64>, SolverError> {
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
    if nearfield_matrix.nrows() != operator.num_rows()
        || nearfield_matrix.ncols() != operator.num_cols()
    {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_rows(),
            got: nearfield_matrix.nrows(),
        });
    }
    if !b
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "BEM preconditioned CGS",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }

    let csr = CsrMatrix::from_dense(nearfield_matrix, 1.0e-15);
    let preconditioner = IluPreconditioner::try_from_csr(&csr)?;
    let transformed_rhs = preconditioner.try_apply(b)?;
    let transformed_operator = LeftPreconditionedOperator {
        operator,
        preconditioner: &preconditioner,
    };
    let mut solution =
        sonium_solvers::iterative::try_cgs(&transformed_operator, &transformed_rhs, config)?;
    certify_cgs_solution(operator, b, config, &mut solution)?;
    Ok(solution)
}

/// Checked ILU-preconditioned GMRES solve for a dense BEM matrix.
pub fn try_gmres_solve_with_ilu(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let operator = DenseOperator::new(matrix.clone());
    try_gmres_solve_with_ilu_operator(&operator, matrix, b, config)
}

/// Checked ILU-preconditioned GMRES solve for an arbitrary BEM operator.
pub fn try_gmres_solve_with_ilu_operator<O: LinearOperator<Complex64>>(
    operator: &O,
    nearfield_matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
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
    if nearfield_matrix.nrows() != operator.num_rows()
        || nearfield_matrix.ncols() != operator.num_cols()
    {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_rows(),
            got: nearfield_matrix.nrows(),
        });
    }
    if !b
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "BEM preconditioned GMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }

    let csr = CsrMatrix::from_dense(nearfield_matrix, 1.0e-15);
    let preconditioner = IluPreconditioner::try_from_csr(&csr)?;
    let mut solution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sonium_solvers::iterative::gmres_preconditioned(operator, &preconditioner, b, config)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "BEM preconditioned GMRES",
        reason: if let Some(message) = payload.downcast_ref::<&'static str>() {
            (*message).to_string()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "GMRES callback panicked".to_string()
        },
    })?;
    if solution.status == SolverStatus::NumericalError {
        return Err(SolverError::OperationFailed {
            operation: "BEM preconditioned GMRES",
            reason: "GMRES returned a numerical failure".to_string(),
        });
    }
    certify_gmres_solution(operator, b, config, &mut solution)?;
    Ok(solution)
}

/// Solves a linear system using the Conjugate Gradient Squared (CGS) method.
///
/// CGS remains available here solely for source compatibility. New BEM call
/// sites should select GMRES or BiCGSTAB explicitly.
#[allow(deprecated)]
pub fn solve_cgs<O: LinearOperator<Complex64>>(
    operator: &O,
    b: &[Complex64],
    config: &CgsConfig<f64>,
) -> CgsSolution<Complex64> {
    sonium_solvers::iterative::cgs(operator, b, config)
}

/// Solves a linear system using the Biconjugate Gradient Stabilized (BiCGSTAB) method.
pub fn solve_bicgstab<O: LinearOperator<Complex64>>(
    operator: &O,
    b: &[Complex64],
    config: &BiCgstabConfig<f64>,
) -> BiCgstabSolution<Complex64> {
    bicgstab(operator, b, config)
}

/// Solves a linear system using the Generalized Minimum Residual (GMRES) method.
pub fn solve_gmres<O: LinearOperator<Complex64>>(
    operator: &O,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    gmres(operator, b, config)
}

/// Solves a linear system with an ILU preconditioner.
///
/// The checked implementation is available as [`try_solve_with_ilu`]. This
/// historical wrapper keeps its infallible signature for compatibility.
#[doc(hidden)]
#[allow(deprecated)]
pub fn solve_with_ilu(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &CgsConfig<f64>,
) -> CgsSolution<Complex64> {
    match try_solve_with_ilu(matrix, b, config) {
        Ok(solution) => solution,
        Err(error) => {
            log::error!("ILU-preconditioned CGS failed; using compatibility CGS result: {error}");
            let op = DenseOperator::new(matrix.clone());
            sonium_solvers::iterative::cgs(&op, b, config)
        }
    }
}

/// Solves a linear system with a given operator and an ILU preconditioner.
#[doc(hidden)]
#[allow(deprecated)]
pub fn solve_with_ilu_operator<O: LinearOperator<Complex64>>(
    operator: &O,
    nearfield_matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &CgsConfig<f64>,
) -> CgsSolution<Complex64> {
    match try_solve_with_ilu_operator(operator, nearfield_matrix, b, config) {
        Ok(solution) => solution,
        Err(error) => {
            log::error!("ILU-preconditioned CGS failed; using compatibility CGS result: {error}");
            sonium_solvers::iterative::cgs(operator, b, config)
        }
    }
}

/// Solve a TBEM system (dense matrix) using CGS with an ILU preconditioner.
///
/// This function acts as a wrapper around `solve_with_ilu` specifically for
/// dense TBEM matrices.
///
/// # Arguments
/// * `matrix` - The dense TBEM matrix to solve.
/// * `b` - The right-hand side vector.
/// * `config` - Configuration for the CGS solver.
///
/// # Returns
/// The CGS solution including the solution vector, iterations, and residual.
#[doc(hidden)]
pub fn solve_tbem_with_ilu(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &CgsConfig<f64>,
) -> CgsSolution<Complex64> {
    solve_with_ilu(matrix, b, config)
}

/// Solves a linear system using GMRES with an ILU preconditioner derived from a dense matrix.
pub fn gmres_solve_with_ilu(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    match try_gmres_solve_with_ilu(matrix, b, config) {
        Ok(solution) => solution,
        Err(error) => {
            log::error!(
                "ILU-preconditioned GMRES failed; using compatibility GMRES result: {error}"
            );
            let op = DenseOperator::new(matrix.clone());
            sonium_solvers::iterative::gmres(&op, b, config)
        }
    }
}

/// Solves a linear system using GMRES with an ILU preconditioner derived from a nearfield matrix for a given operator.
pub fn gmres_solve_with_ilu_operator<O: LinearOperator<Complex64>>(
    operator: &O,
    nearfield_matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    match try_gmres_solve_with_ilu_operator(operator, nearfield_matrix, b, config) {
        Ok(solution) => solution,
        Err(error) => {
            log::error!(
                "ILU-preconditioned GMRES failed; using compatibility GMRES result: {error}"
            );
            sonium_solvers::iterative::gmres(operator, b, config)
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed-precision iterative refinement (Carson-Higham 2018) for BEM
//
// Outer Krylov runs in Complex64 against the full FMM/dense operator
// `O`; inner Krylov runs in Complex32 against the near-field CSR with
// a Chebyshev preconditioner. On a 2-D Helmholtz benchmark this is
// 1.9–2.2× faster than the legacy `gmres_solve_with_ilu_operator` at
// FEM scales (see `crates/sonium-solvers/examples/metal_ir_complex_bench.rs`);
// the win grows with kappa.
//
// The bridge `HostLinearOperator` round-trips between `Array1<Complex64>`
// and `HostBuffer<Complex64>` per matvec — one full alloc + memcpy per
// outer SpMV. For a 10M-node FMM that's ~160 MB/op, non-trivial. The
// follow-up is a `LinearOperator::apply_into_slice` that elides the
// `Array1` round-trip; that's a future patch.
// ---------------------------------------------------------------------------

/// Status returned by [`gmres_ir_solve_with_chebyshev_operator`].
#[derive(Debug, Clone)]
pub struct BemIrSolution {
    /// Solution vector (Complex64, recovered to outer precision).
    pub solution: Vec<Complex64>,
    /// Outer FGMRES Arnoldi iterations.
    pub outer_iterations: usize,
    /// Total Complex32 inner Krylov iterations across all outer steps.
    pub total_inner_iterations: usize,
    /// Final relative residual `‖b − A x‖ / ‖b‖` at outer (Complex64)
    /// precision. Note this is the IR loop's bookkeeping; the true
    /// physical residual is recomputed at every restart boundary.
    pub final_residual: f64,
    /// Whether the outer FGMRES residual reached `outer_tolerance`.
    pub converged: bool,
}

fn failed_bem_ir_solution(size: usize) -> BemIrSolution {
    BemIrSolution {
        solution: vec![Complex64::new(0.0, 0.0); size],
        outer_iterations: 0,
        total_inner_iterations: 0,
        final_residual: f64::INFINITY,
        converged: false,
    }
}

/// Configuration for the mixed-precision IR BEM solve.
#[derive(Debug, Clone)]
pub struct BemIrConfig {
    /// Max outer Complex64 FGMRES iterations.
    pub max_outer_iterations: usize,
    /// Outer relative-residual tolerance.
    pub outer_tolerance: f64,
    /// Max Complex32 inner Krylov iterations per outer step.
    pub max_inner_iterations: usize,
    /// Inner Complex32 relative-residual tolerance. Loose by design;
    /// outer refinement recovers full Complex64 accuracy.
    pub inner_tolerance: f32,
    /// Chebyshev polynomial degree for the Complex32 inner precond.
    /// 5 is the typical sweet spot — gave 1.4× over PolyJac K=2 in the
    /// 2-D Helmholtz benchmark.
    pub chebyshev_degree: usize,
    /// Eigenvalue ratio `λ_max / λ_min` for the Chebyshev interval.
    /// Default 30.0 follows PETSc / HYPRE.
    pub chebyshev_eigenvalue_ratio: f32,
}

impl Default for BemIrConfig {
    fn default() -> Self {
        Self {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-10,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-3,
            chebyshev_degree: 5,
            chebyshev_eigenvalue_ratio: 30.0,
        }
    }
}

/// Solve `A x = b` using Carson-Higham 2018 FGMRES-IR with the BEM
/// near-field block as the inner-precision preconditioner basis.
///
/// - `operator` — the full BEM matvec (dense or FMM-compressed); runs
///   at Complex64 via the host adapter.
/// - `nearfield_matrix` — the dense near-field block; cast to Complex32
///   and used both as the inner Krylov system matrix and as the basis
///   for the Chebyshev preconditioner.
/// - `b` — Complex64 right-hand side.
/// - `config` — IR knobs (iter caps, tolerances, Chebyshev degree).
///
/// Returns full Complex64 accuracy provided `κ(A) · ε_{Complex32} · ε_{Complex64} < 1`,
/// i.e. up to κ ≈ 10¹⁴.
///
/// Invalid dimensions, non-finite inputs, invalid iteration/tolerance
/// settings, allocation failures, and inner-solver breakdowns are reported as
/// a non-converged [`BemIrSolution`] with an infinite residual. This legacy
/// status-returning API therefore remains total; callers that need a typed
/// error should gate on `converged` and `final_residual` before using the
/// solution vector.
pub fn gmres_ir_solve_with_chebyshev_operator<O>(
    operator: &O,
    nearfield_matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &BemIrConfig,
) -> BemIrSolution
where
    O: LinearOperator<Complex64>,
{
    use num_complex::Complex32;
    use sonium_compute::Backend;
    use sonium_compute::ChebyshevBackend;
    use sonium_compute::cpu::CpuBackend;
    use sonium_compute::gmres::BackendCsr;
    use sonium_compute::gmres_ir::{GmresIrConfig, gmres_ir_backend_preconditioned};
    use sonium_solvers::iterative::host_operator::HostLinearOperator;

    let backend = CpuBackend::new();
    let n = b.len();
    if n == 0
        || operator.num_rows() != n
        || operator.num_cols() != n
        || nearfield_matrix.nrows() != n
        || nearfield_matrix.ncols() != n
        || b.iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || config.max_outer_iterations == 0
        || config.max_inner_iterations == 0
        || config.chebyshev_degree == 0
        || !config.outer_tolerance.is_finite()
        || config.outer_tolerance <= 0.0
        || !config.inner_tolerance.is_finite()
        || config.inner_tolerance <= 0.0
        || !config.chebyshev_eigenvalue_ratio.is_finite()
        || config.chebyshev_eigenvalue_ratio <= 1.0
    {
        return failed_bem_ir_solution(n);
    }

    // Convert near-field dense → CSR (drop near-zero entries) at Complex64.
    let nearfield_csr = CsrMatrix::from_dense(nearfield_matrix, 1.0e-15);
    let rp: Vec<i32> = nearfield_csr.row_ptrs.iter().map(|&v| v as i32).collect();
    let ci: Vec<i32> = nearfield_csr
        .col_indices
        .iter()
        .map(|&v| v as i32)
        .collect();
    let va_c64: Vec<Complex64> = nearfield_csr.values.clone();

    // Inner-precision (Complex32) near-field copy on the device.
    let a_inner = match BackendCsr::<CpuBackend, Complex32>::try_from_host_cast::<Complex64>(
        &backend, n, n, &rp, &ci, &va_c64,
    ) {
        Ok(upload) => upload,
        Err(_) => return failed_bem_ir_solution(n),
    };

    // Chebyshev K=5 on the Complex32 near-field — strong, parallel-
    // friendly, no triangular solve.
    let va_c32: Vec<Complex32> = va_c64
        .iter()
        .map(|&v| <Complex64 as sonium_compute::LowerPrecision>::cast_to_lower(v))
        .collect();
    let cheb = match ChebyshevBackend::<CpuBackend, Complex32>::try_from_csr_host(
        &backend,
        &a_inner,
        &rp,
        &ci,
        &va_c32,
        config.chebyshev_degree,
        config.chebyshev_eigenvalue_ratio,
    ) {
        Ok(preconditioner) => preconditioner,
        Err(_) => return failed_bem_ir_solution(n),
    };

    // Outer operator: wrap the host LinearOperator via the bridge.
    let a_outer = HostLinearOperator::new(operator);

    // Upload b and allocate solution buffer.
    let mut b_buf = match backend.raw_alloc::<Complex64>(n) {
        Ok(buffer) => buffer,
        Err(_) => return failed_bem_ir_solution(n),
    };
    backend.copy_h2d(b, &mut b_buf);
    let mut out_buf = match backend.raw_alloc::<Complex64>(n) {
        Ok(buffer) => buffer,
        Err(_) => return failed_bem_ir_solution(n),
    };

    let ir_cfg: GmresIrConfig<Complex64> = GmresIrConfig {
        max_outer_iterations: config.max_outer_iterations,
        outer_tolerance: config.outer_tolerance,
        max_inner_iterations: config.max_inner_iterations,
        inner_tolerance: config.inner_tolerance,
        ..Default::default()
    };

    let status = match gmres_ir_backend_preconditioned::<CpuBackend, _, _, _, Complex64>(
        &backend,
        &a_outer,
        &a_inner,
        &cheb,
        &b_buf,
        None,
        &mut out_buf,
        &ir_cfg,
    ) {
        Ok(status) => status,
        Err(_) => return failed_bem_ir_solution(n),
    };

    let mut solution_host = vec![Complex64::new(0.0, 0.0); n];
    backend.copy_d2h(&out_buf, &mut solution_host);
    BemIrSolution {
        solution: solution_host,
        outer_iterations: status.outer_iterations,
        total_inner_iterations: status.total_inner_iterations,
        final_residual: status.final_outer_residual,
        converged: status.converged,
    }
}

/// Solves a TBEM system (dense matrix) using GMRES with an ILU preconditioner.
///
/// This function acts as a wrapper around `gmres_solve_with_ilu` specifically for
/// dense TBEM matrices.
pub fn gmres_solve_tbem_with_ilu(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    gmres_solve_with_ilu(matrix, b, config)
}

/// Solves a linear system using GMRES with a batched FMM operator (unpreconditioned).
pub fn gmres_solve_fmm_batched(
    fmm_system: &crate::core::assembly::slfmm::SlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    let op = SlfmmOperator::new(fmm_system.clone());
    gmres(&op, b, config)
}

/// Checked unpreconditioned SLFMM GMRES solve with residual certification
/// against the original matrix-free operator.
pub fn try_gmres_solve_fmm_batched(
    fmm_system: &crate::core::assembly::slfmm::SlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = SlfmmOperator::new(fmm_system.clone());
    let mut solution = try_gmres(&op, b, config)?;
    certify_gmres_solution(&op, b, config, &mut solution)?;
    Ok(solution)
}

/// Solves a linear system using GMRES with a batched FMM operator and an ILU preconditioner.
pub fn gmres_solve_fmm_batched_with_ilu(
    fmm_system: &crate::core::assembly::slfmm::SlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    let op = SlfmmOperator::new(fmm_system.clone());
    let nearfield_matrix = fmm_system.extract_near_field_matrix();
    let csr = CsrMatrix::from_dense(&nearfield_matrix, 1e-15);
    let precond = IluPreconditioner::from_csr(&csr);
    sonium_solvers::iterative::gmres_preconditioned(&op, &precond, b, config)
}

/// Checked ILU-preconditioned SLFMM GMRES solve.  The returned residual is
/// recomputed against the full SLFMM operator, never only the near-field ILU
/// system.
pub fn try_gmres_solve_fmm_batched_with_ilu(
    fmm_system: &crate::core::assembly::slfmm::SlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = SlfmmOperator::new(fmm_system.clone());
    let preconditioner = try_ilu_from_checked_slfmm_near_field(
        || fmm_system.try_extract_near_field_matrix(),
        "BEM checked SLFMM GMRES",
    )?;
    try_gmres_preconditioned_checked(&op, &preconditioner, b, config, "BEM checked SLFMM GMRES")
}

/// Checked hierarchical-ILU-preconditioned SLFMM GMRES solve.
///
/// The H-ILU is assembled from the exact SLFMM near-field matrix. Far-field
/// interactions remain matrix-free in the SLFMM operator, while the returned
/// solution is certified against that full operator.
pub fn try_gmres_solve_fmm_batched_with_hilu(
    fmm_system: &crate::core::assembly::slfmm::SlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = SlfmmOperator::new(fmm_system.clone());
    let nearfield_matrix = fmm_system
        .try_extract_near_field_matrix()
        .map_err(|error| SolverError::OperationFailed {
            operation: "BEM SLFMM near-field extraction",
            reason: error.to_string(),
        })?;
    let csr = CsrMatrix::from_dense(&nearfield_matrix, 1e-15);
    let preconditioner = HiluPreconditioner::try_from_csr(&csr)?;
    try_gmres_preconditioned_checked(
        &op,
        &preconditioner,
        b,
        config,
        "BEM hierarchical SLFMM GMRES",
    )
}

/// Checked Hermitian-adjoint SLFMM GMRES solve with near-field ILU.
///
/// The operator action is `Aᴴ`; the near-field ILU is wrapped with the same
/// Hermitian orientation, and the returned residual is recomputed against
/// the full matrix-free SLFMM adjoint action.
pub fn try_gmres_solve_fmm_batched_adjoint_with_ilu(
    fmm_system: &crate::core::assembly::slfmm::SlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = HermitianOperator {
        inner: SlfmmOperator::new(fmm_system.clone()),
    };
    let preconditioner = try_ilu_from_checked_slfmm_near_field(
        || fmm_system.try_extract_near_field_matrix(),
        "BEM SLFMM adjoint near-field ILU",
    )?;
    let adjoint_preconditioner = HermitianPreconditioner {
        inner: preconditioner,
    };
    try_gmres_preconditioned_checked(
        &op,
        &adjoint_preconditioner,
        b,
        config,
        "BEM SLFMM adjoint GMRES",
    )
}

/// Checked Hermitian-adjoint SLFMM GMRES solve with hierarchical ILU.
///
/// H-ILU is assembled from the validated SLFMM near-field matrix while the
/// far-field action remains matrix-free. The result is certified against the
/// full Hermitian-adjoint operator before it is returned.
pub fn try_gmres_solve_fmm_batched_adjoint_with_hilu(
    fmm_system: &crate::core::assembly::slfmm::SlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = HermitianOperator {
        inner: SlfmmOperator::new(fmm_system.clone()),
    };
    let nearfield_matrix = fmm_system
        .try_extract_near_field_matrix()
        .map_err(|error| SolverError::OperationFailed {
            operation: "BEM SLFMM adjoint near-field extraction",
            reason: error.to_string(),
        })?;
    let csr = CsrMatrix::from_dense(&nearfield_matrix, 1e-15);
    let preconditioner = HiluPreconditioner::try_from_csr(&csr)?;
    let adjoint_preconditioner = HermitianPreconditioner {
        inner: preconditioner,
    };
    try_gmres_preconditioned_checked(
        &op,
        &adjoint_preconditioner,
        b,
        config,
        "BEM hierarchical SLFMM adjoint GMRES",
    )
}

/// Solves a linear system using a batched MLFMM operator and an ILU
/// preconditioner assembled from its exact leaf near field.
///
/// Far interactions remain matrix-free; only the near-field blocks are
/// materialized for the local preconditioner.  This is the MLFMM analogue of
/// [`gmres_solve_fmm_batched_with_ilu`] and deliberately does not advertise a
/// distributed or hierarchical preconditioner.
pub fn gmres_solve_mlfmm_batched_with_ilu(
    fmm_system: &crate::core::assembly::mlfmm::MlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    let op = MlfmmOperator::new(fmm_system.clone());
    let nearfield_matrix = fmm_system.extract_near_field_matrix();
    let csr = CsrMatrix::from_dense(&nearfield_matrix, 1e-15);
    let precond = IluPreconditioner::from_csr(&csr);
    sonium_solvers::iterative::gmres_preconditioned(&op, &precond, b, config)
}

/// Checked ILU-preconditioned MLFMM GMRES solve with full-operator residual
/// certification.
pub fn try_gmres_solve_mlfmm_batched_with_ilu(
    fmm_system: &crate::core::assembly::mlfmm::MlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = MlfmmOperator::new(fmm_system.clone());
    let preconditioner = try_ilu_from_checked_mlfmm_near_field(
        || fmm_system.try_extract_near_field_matrix(),
        "BEM checked MLFMM GMRES",
    )?;
    try_gmres_preconditioned_checked(&op, &preconditioner, b, config, "BEM checked MLFMM GMRES")
}

/// Checked hierarchical-ILU-preconditioned MLFMM GMRES solve.
///
/// The H-ILU is assembled from exact leaf near-field blocks; the hierarchical
/// MLFMM far field remains matrix-free and the residual is certified against
/// the complete MLFMM operator.
pub fn try_gmres_solve_mlfmm_batched_with_hilu(
    fmm_system: &crate::core::assembly::mlfmm::MlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = MlfmmOperator::new(fmm_system.clone());
    let nearfield_matrix = fmm_system
        .try_extract_near_field_matrix()
        .map_err(|error| SolverError::OperationFailed {
            operation: "BEM MLFMM near-field extraction",
            reason: error.to_string(),
        })?;
    let csr = CsrMatrix::from_dense(&nearfield_matrix, 1e-15);
    let preconditioner = HiluPreconditioner::try_from_csr(&csr)?;
    try_gmres_preconditioned_checked(
        &op,
        &preconditioner,
        b,
        config,
        "BEM hierarchical MLFMM GMRES",
    )
}

/// Solves the Hermitian-transposed MLFMM system with the exact leaf ILU
/// adjoint.  The operator and preconditioner are both transposed; no
/// approximate transpose or silently substituted forward solve is allowed.
pub fn gmres_solve_mlfmm_batched_adjoint_with_ilu(
    fmm_system: &crate::core::assembly::mlfmm::MlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    let op = HermitianOperator {
        inner: MlfmmOperator::new(fmm_system.clone()),
    };
    let nearfield_matrix = fmm_system.extract_near_field_matrix();
    let csr = CsrMatrix::from_dense(&nearfield_matrix, 1e-15);
    let precond = IluPreconditioner::from_csr(&csr);
    let adjoint_preconditioner = HermitianPreconditioner { inner: precond };
    sonium_solvers::iterative::gmres_preconditioned(&op, &adjoint_preconditioner, b, config)
}

/// Checked Hermitian-adjoint MLFMM GMRES solve with an adjoint ILU
/// preconditioner and residual certification against `Aᴴ`.
pub fn try_gmres_solve_mlfmm_batched_adjoint_with_ilu(
    fmm_system: &crate::core::assembly::mlfmm::MlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = HermitianOperator {
        inner: MlfmmOperator::new(fmm_system.clone()),
    };
    let preconditioner = try_ilu_from_checked_mlfmm_near_field(
        || fmm_system.try_extract_near_field_matrix(),
        "BEM checked MLFMM adjoint GMRES",
    )?;
    let adjoint_preconditioner = HermitianPreconditioner {
        inner: preconditioner,
    };
    try_gmres_preconditioned_checked(
        &op,
        &adjoint_preconditioner,
        b,
        config,
        "BEM checked MLFMM adjoint GMRES",
    )
}

/// Checked Hermitian-adjoint MLFMM GMRES solve with a hierarchical ILU
/// preconditioner built from the exact near-field blocks.
///
/// The preconditioner is explicitly wrapped in the Hermitian orientation;
/// the far-field MLFMM action remains matrix-free and the checked GMRES
/// wrapper certifies the returned iterate against the complete `Aᴴ` action.
pub fn try_gmres_solve_mlfmm_batched_adjoint_with_hilu(
    fmm_system: &crate::core::assembly::mlfmm::MlfmmSystem,
    b: &[Complex64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<Complex64>, SolverError> {
    let op = HermitianOperator {
        inner: MlfmmOperator::new(fmm_system.clone()),
    };
    let nearfield_matrix = fmm_system
        .try_extract_near_field_matrix()
        .map_err(|error| SolverError::OperationFailed {
            operation: "BEM MLFMM adjoint near-field extraction",
            reason: error.to_string(),
        })?;
    let csr = CsrMatrix::from_dense(&nearfield_matrix, 1e-15);
    let preconditioner = HiluPreconditioner::try_from_csr(&csr)?;
    let adjoint_preconditioner = HermitianPreconditioner {
        inner: preconditioner,
    };
    try_gmres_preconditioned_checked(
        &op,
        &adjoint_preconditioner,
        b,
        config,
        "BEM hierarchical MLFMM adjoint GMRES",
    )
}

struct HermitianPreconditioner<P> {
    inner: P,
}

impl<P> Preconditioner<Complex64> for HermitianPreconditioner<P>
where
    P: Preconditioner<Complex64>,
{
    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.inner.apply_hermitian(x)
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.inner.apply_hermitian_into(x, y);
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.inner.apply_into(x, y);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

// ============================================================================
// Adaptive Mesh Utilities
// ============================================================================

/// Calculate recommended mesh resolution for a given frequency
pub fn recommended_mesh_resolution(
    frequency: f64,
    speed_of_sound: f64,
    elements_per_wavelength: usize,
) -> f64 {
    let wavelength = speed_of_sound / frequency;
    elements_per_wavelength as f64 / wavelength
}

/// Calculate mesh resolution for a frequency range
pub fn mesh_resolution_for_frequency_range(
    _min_freq: f64,
    max_freq: f64,
    speed_of_sound: f64,
    elements_per_wavelength: usize,
) -> f64 {
    recommended_mesh_resolution(max_freq, speed_of_sound, elements_per_wavelength)
}

/// Estimate element count for a rectangular room
pub fn estimate_element_count(room_dimensions: (f64, f64, f64), mesh_resolution: f64) -> usize {
    let (w, d, h) = room_dimensions;
    let surface_area = 2.0 * (w * d + w * h + d * h);
    let element_size = 1.0 / mesh_resolution;
    let element_area = element_size * element_size;
    (surface_area / element_area).ceil() as usize
}

/// Adaptive mesh configuration
pub struct AdaptiveMeshConfig {
    /// Base resolution (elements per meter)
    pub base_resolution: f64,
    /// Refinement factor near sources (1.0 = no refinement)
    pub source_refinement: f64,
    /// Radius around sources where refinement is applied (meters)
    pub source_refinement_radius: f64,
}

impl AdaptiveMeshConfig {
    /// Create configuration for a frequency range
    pub fn for_frequency_range(min_freq: f64, max_freq: f64) -> Self {
        let speed_of_sound = 343.0;
        let base = mesh_resolution_for_frequency_range(min_freq, max_freq, speed_of_sound, 6);
        Self {
            base_resolution: base,
            source_refinement: 1.5,
            source_refinement_radius: 0.5,
        }
    }

    /// Create configuration from a fixed resolution
    pub fn from_resolution(resolution: f64) -> Self {
        Self {
            base_resolution: resolution,
            source_refinement: 1.0,
            source_refinement_radius: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::assembly::mlfmm::NearFieldBlock;
    use crate::core::assembly::slfmm::NearFieldBlock as SlfmmNearFieldBlock;
    use crate::core::types::{Cluster, ClusterLevel};

    fn c(value: f64) -> Complex64 {
        Complex64::new(value, 0.0)
    }

    fn solve_2x2(a: &DenseMatrix<Complex64>, b: &[Complex64]) -> Vec<Complex64> {
        let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
        assert!(det.norm() > 1e-14);
        vec![
            (a[[1, 1]] * b[0] - a[[0, 1]] * b[1]) / det,
            (-a[[1, 0]] * b[0] + a[[0, 0]] * b[1]) / det,
        ]
    }

    fn dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
    }

    fn dense_from_operator(operator: &MlfmmOperator) -> DenseMatrix<Complex64> {
        let n = operator.num_cols();
        let mut matrix = DenseMatrix::<Complex64>::zeros(operator.num_rows(), n);
        for col in 0..n {
            let mut basis = vec![Complex64::new(0.0, 0.0); n];
            basis[col] = c(1.0);
            let image = operator.apply(&basis);
            for row in 0..operator.num_rows() {
                matrix[[row, col]] = image[row];
            }
        }
        matrix
    }

    fn two_dof_nearfield_slfmm_operator() -> SlfmmOperator {
        let mut system = SlfmmSystem::new(2, 1, 1, 1);
        system.cluster_dof_indices = vec![vec![0, 1]];
        system.near_matrix.push(SlfmmNearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(
                2,
                2,
                vec![c(2.0), c(0.3), c(-0.4), c(1.5)],
            ),
        });
        SlfmmOperator::new(system)
    }

    #[test]
    fn checked_slfmm_wrapper_certifies_original_operator() {
        let operator = two_dof_nearfield_slfmm_operator();
        let rhs = vec![c(1.0), c(-0.5)];
        let config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };

        let solution = try_gmres_solve_fmm_batched(operator.system(), &rhs, &config)
            .expect("checked SLFMM GMRES solution");
        assert!(solution.converged, "SLFMM residual={}", solution.residual);

        let residual = operator
            .apply(&solution.x)
            .iter()
            .zip(rhs.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1.0e-11, "SLFMM operator residual={residual}");
    }

    #[test]
    fn checked_slfmm_ilu_wrapper_rejects_malformed_cluster_layout() {
        let mut system = SlfmmSystem::new(2, 1, 1, 1);
        system.near_matrix.push(SlfmmNearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0)]),
        });
        let config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };

        let result = try_gmres_solve_fmm_batched_with_ilu(&system, &[c(1.0), c(-0.5)], &config);
        assert!(matches!(result, Err(SolverError::OperationFailed { .. })));
    }

    #[test]
    fn checked_adjoint_capability_is_preserved_through_bem_left_preconditioning() {
        let matrix = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(2.0, 0.5),
                Complex64::new(-0.25, 1.0),
                Complex64::new(0.75, -0.5),
                Complex64::new(1.5, -0.25),
            ],
        );
        let operator = DenseOperator::new(matrix);
        assert_eq!(operator.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(operator.hermitian_capability(), AdjointCapability::Exact);

        let diagonal = DiagonalPreconditioner::from_diagonal(vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(1.5, -0.25),
        ]);
        assert_eq!(
            <DiagonalPreconditioner as Preconditioner<Complex64>>::transpose_capability(&diagonal),
            AdjointCapability::Exact
        );
        assert_eq!(
            <DiagonalPreconditioner as Preconditioner<Complex64>>::hermitian_capability(&diagonal),
            AdjointCapability::Exact
        );

        let left = LeftPreconditionedOperator {
            operator: &operator,
            preconditioner: &diagonal,
        };
        assert_eq!(left.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(left.hermitian_capability(), AdjointCapability::Exact);
        let rhs = vec![Complex64::new(0.5, -0.75), Complex64::new(-1.0, 0.25)];
        assert_eq!(left.try_apply_transpose(&rhs).unwrap().len(), 2);
        assert_eq!(left.try_apply_hermitian(&rhs).unwrap().len(), 2);
    }

    #[test]
    fn slfmm_nearfield_ilu_adjoint_solves_against_full_matrix_free_operator() {
        let operator = two_dof_nearfield_slfmm_operator();
        let rhs = vec![c(0.75), c(-1.25)];
        let config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };

        let solution =
            try_gmres_solve_fmm_batched_adjoint_with_ilu(operator.system(), &rhs, &config)
                .expect("SLFMM adjoint near-field ILU solution");
        assert!(
            solution.converged,
            "SLFMM adjoint ILU residual={}",
            solution.residual
        );
        let residual = operator
            .apply_hermitian(&solution.x)
            .iter()
            .zip(rhs.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(residual <= 1.0e-11, "SLFMM adjoint ILU residual={residual}");
    }

    #[test]
    fn slfmm_hilu_adjoint_solves_and_rejects_nonfinite_rhs() {
        let operator = two_dof_nearfield_slfmm_operator();
        let rhs = vec![c(0.75), c(-1.25)];
        let config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };

        let solution =
            try_gmres_solve_fmm_batched_adjoint_with_hilu(operator.system(), &rhs, &config)
                .expect("SLFMM adjoint hierarchical ILU solution");
        assert!(
            solution.converged,
            "SLFMM adjoint H-ILU residual={}",
            solution.residual
        );
        let residual = operator
            .apply_hermitian(&solution.x)
            .iter()
            .zip(rhs.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            residual <= 1.0e-11,
            "SLFMM adjoint H-ILU residual={residual}"
        );

        let rejected = try_gmres_solve_fmm_batched_adjoint_with_hilu(
            operator.system(),
            &[Complex64::new(f64::NAN, 0.0), c(1.0)],
            &config,
        )
        .expect_err("SLFMM adjoint H-ILU must reject non-finite RHS");
        assert!(matches!(rejected, SolverError::OperationFailed { .. }));
    }

    fn two_dof_nearfield_mlfmm_operator() -> MlfmmOperator {
        let mut level = ClusterLevel::new(2);
        level.clusters.push(Cluster::new([0.0, 0.0, 0.0]));
        level.clusters.push(Cluster::new([1.0, 0.0, 0.0]));

        let mut system = MlfmmSystem::new(2, 1, vec![level], vec![1]);
        system.leaf_dof_indices = vec![vec![0], vec![1]];
        for (source_cluster, field_cluster, value) in [
            (0, 0, c(2.0)),
            (0, 1, c(0.3)),
            (1, 0, c(-0.4)),
            (1, 1, c(1.5)),
        ] {
            system.near_matrix.push(NearFieldBlock {
                source_cluster,
                field_cluster,
                coefficients: sonium_solvers::dense::DenseMatrix::from_row_major_vec(
                    1,
                    1,
                    vec![value],
                ),
            });
        }

        MlfmmOperator::new(system)
    }

    #[test]
    fn mlfmm_nearfield_ilu_solves_against_the_full_matrix_free_operator() {
        let operator = two_dof_nearfield_mlfmm_operator();
        let rhs = vec![c(1.0), c(-0.5)];
        let config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };

        let solution = try_gmres_solve_mlfmm_batched_with_ilu(operator.system(), &rhs, &config)
            .expect("checked MLFMM near-field ILU solution");
        assert!(
            solution.converged,
            "MLFMM near-field ILU residual={}",
            solution.residual
        );
        assert!(solution.residual < 1.0e-12);

        let residual = operator
            .apply(&solution.x)
            .iter()
            .zip(rhs.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1.0e-11, "full MLFMM residual={residual}");

        let near = operator.system().extract_near_field_matrix();
        assert_eq!(near.nrows(), 2);
        assert_eq!(near.ncols(), 2);
        assert_eq!(near[(0, 1)], c(0.3));
        assert_eq!(near[(1, 0)], c(-0.4));
    }

    #[test]
    fn mlfmm_nearfield_ilu_adjoint_solves_hermitian_system() {
        let operator = two_dof_nearfield_mlfmm_operator();
        let rhs = vec![c(0.75), c(-1.25)];
        let config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };

        let solution =
            try_gmres_solve_mlfmm_batched_adjoint_with_ilu(operator.system(), &rhs, &config)
                .expect("checked MLFMM adjoint near-field ILU solution");
        assert!(
            solution.converged,
            "MLFMM adjoint ILU residual={}",
            solution.residual
        );
        let residual = operator
            .apply_hermitian(&solution.x)
            .iter()
            .zip(rhs.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1.0e-11, "Hermitian MLFMM residual={residual}");

        let probe = vec![c(-0.2), c(0.9)];
        let lhs = dot(&operator.apply(&probe), &rhs);
        let rhs_inner = dot(&probe, &operator.apply_hermitian(&rhs));
        assert!((lhs - rhs_inner).norm() < 1.0e-12);
    }

    #[test]
    fn mlfmm_nearfield_hilu_adjoint_solves_and_rejects_nonfinite_rhs() {
        let operator = two_dof_nearfield_mlfmm_operator();
        let rhs = vec![c(0.75), c(-1.25)];
        let config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };

        let solution =
            try_gmres_solve_mlfmm_batched_adjoint_with_hilu(operator.system(), &rhs, &config)
                .expect("checked MLFMM adjoint H-ILU solution");
        assert!(
            solution.converged,
            "MLFMM adjoint H-ILU residual={}",
            solution.residual
        );
        let residual = operator
            .apply_hermitian(&solution.x)
            .iter()
            .zip(rhs.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            residual < 1.0e-11,
            "Hermitian MLFMM H-ILU residual={residual}"
        );

        let rejected = try_gmres_solve_mlfmm_batched_adjoint_with_hilu(
            operator.system(),
            &[Complex64::new(f64::NAN, 0.0), c(1.0)],
            &config,
        )
        .expect_err("MLFMM adjoint H-ILU must reject non-finite RHS");
        assert!(matches!(rejected, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_mlfmm_wrapper_rejects_nonfinite_rhs_and_invalid_configuration() {
        let operator = two_dof_nearfield_mlfmm_operator();
        let mut config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };
        let nonfinite = try_gmres_solve_mlfmm_batched_with_ilu(
            operator.system(),
            &[c(f64::NAN), c(1.0)],
            &config,
        )
        .expect_err("non-finite MLFMM RHS must fail closed");
        assert!(matches!(nonfinite, SolverError::OperationFailed { .. }));

        config.max_iterations = 0;
        let invalid = try_gmres_solve_mlfmm_batched_adjoint_with_ilu(
            operator.system(),
            &[c(1.0), c(-1.0)],
            &config,
        )
        .expect_err("invalid checked MLFMM configuration must fail closed");
        assert!(matches!(invalid, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn gmres_ir_rejects_dimension_and_configuration_errors_without_panicking() {
        let operator = DenseOperator::new(DenseMatrix::eye(2));
        let nearfield = DenseMatrix::eye(2);
        let mut config = BemIrConfig::default();

        let mismatch =
            gmres_ir_solve_with_chebyshev_operator(&operator, &nearfield, &[c(1.0)], &config);
        assert!(!mismatch.converged);
        assert!(mismatch.final_residual.is_infinite());
        assert_eq!(mismatch.solution.len(), 1);

        config.max_outer_iterations = 0;
        let invalid_config = gmres_ir_solve_with_chebyshev_operator(
            &operator,
            &nearfield,
            &[c(1.0), c(0.0)],
            &config,
        );
        assert!(!invalid_config.converged);
        assert!(invalid_config.final_residual.is_infinite());
    }

    #[test]
    fn gmres_ir_rejects_nonfinite_rhs_and_invalid_inner_parameters() {
        let operator = DenseOperator::new(DenseMatrix::eye(2));
        let nearfield = DenseMatrix::eye(2);
        let valid_rhs = [c(1.0), c(0.0)];

        let nonfinite_rhs = [Complex64::new(f64::NAN, 0.0), c(0.0)];
        let rejected = gmres_ir_solve_with_chebyshev_operator(
            &operator,
            &nearfield,
            &nonfinite_rhs,
            &BemIrConfig::default(),
        );
        assert!(!rejected.converged);
        assert!(rejected.final_residual.is_infinite());

        let mut invalid_configs = Vec::new();
        let mut config = BemIrConfig::default();
        config.max_inner_iterations = 0;
        invalid_configs.push(config);
        let mut config = BemIrConfig::default();
        config.chebyshev_degree = 0;
        invalid_configs.push(config);
        let mut config = BemIrConfig::default();
        config.outer_tolerance = f64::NAN;
        invalid_configs.push(config);
        let mut config = BemIrConfig::default();
        config.inner_tolerance = f32::NAN;
        invalid_configs.push(config);
        let mut config = BemIrConfig::default();
        config.chebyshev_eigenvalue_ratio = 1.0;
        invalid_configs.push(config);

        for config in invalid_configs {
            let rejected =
                gmres_ir_solve_with_chebyshev_operator(&operator, &nearfield, &valid_rhs, &config);
            assert!(!rejected.converged);
            assert!(rejected.final_residual.is_infinite());
            assert_eq!(rejected.solution.len(), valid_rhs.len());
        }
    }

    #[test]
    fn gmres_ir_reports_inner_solver_failure_without_panicking() {
        let operator = DenseOperator::new(DenseMatrix::eye(2));
        let zero_nearfield = DenseMatrix::zeros(2, 2);
        let result = gmres_ir_solve_with_chebyshev_operator(
            &operator,
            &zero_nearfield,
            &[c(1.0), c(0.0)],
            &BemIrConfig::default(),
        );

        assert!(!result.converged);
        assert_eq!(result.solution.len(), 2);
    }

    #[test]
    fn test_mlfmm_transpose_supports_shape_lagrangian_adjoint() {
        let operator = two_dof_nearfield_mlfmm_operator();
        let a = dense_from_operator(&operator);

        let b = vec![c(1.0), c(-0.25)];
        let target = vec![c(0.2), c(-0.1)];
        let state = solve_2x2(&a, &b);

        let d_objective_du: Vec<Complex64> = state
            .iter()
            .zip(target.iter())
            .map(|(s, t)| s - t)
            .collect();
        let adjoint_rhs: Vec<Complex64> = d_objective_du.iter().map(|v| -v).collect();
        let a_t = a.transposed();
        let adjoint = solve_2x2(&a_t, &adjoint_rhs);

        let adjoint_action: Vec<Complex64> = operator.apply_transpose(&adjoint);
        let lagrangian_stationarity: Vec<Complex64> = d_objective_du
            .iter()
            .zip(adjoint_action.iter())
            .map(|(d, a)| d + a)
            .collect();
        let max_stationarity = lagrangian_stationarity
            .iter()
            .map(|entry| entry.norm())
            .fold(0.0, f64::max);
        assert!(max_stationarity < 1e-12);

        let d_a_d_shape =
            DenseMatrix::from_row_major_vec(2, 2, vec![c(0.1), c(-0.2), c(0.05), c(0.3)]);
        let d_a_state = d_a_d_shape.matvec(&state);
        let neg_solve = solve_2x2(&a, &d_a_state);
        let direct_state_sensitivity: Vec<Complex64> = neg_solve.iter().map(|v| -v).collect();
        let direct_gradient = dot(&d_objective_du, &direct_state_sensitivity);
        let adjoint_gradient = dot(&adjoint, &d_a_state);

        assert!(
            (direct_gradient - adjoint_gradient).norm() < 1e-12,
            "shape Lagrangian gradient mismatch: direct={direct_gradient}, adjoint={adjoint_gradient}"
        );
    }

    /// End-to-end smoke test for the new mixed-precision IR BEM solver.
    /// Drives a complex Helmholtz-like dense matrix through
    /// [`gmres_ir_solve_with_chebyshev_operator`] and verifies the
    /// physical residual `‖b − A x‖ / ‖b‖` is below 1e-9.
    ///
    /// The operator is the full dense matrix; the "near-field" is the
    /// same matrix (worst-case for the precond — gives the IR no help
    /// from compression). For real BEM the near-field is much sparser.
    #[test]
    fn gmres_ir_chebyshev_converges_on_complex_helmholtz_dense() {
        // Build a complex tridiagonal Helmholtz: A[i,i] = 2 + iα,
        // A[i,i±1] = -1. Diagonally dominant, well-conditioned.
        let n = 32;
        let alpha = 0.05;
        let mut matrix = sonium_solvers::dense::DenseMatrix::<Complex64>::zeros(n, n);
        for i in 0..n {
            matrix[[i, i]] = Complex64::new(2.0, alpha);
            if i > 0 {
                matrix[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i + 1 < n {
                matrix[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }

        let b: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = (i as f64 + 1.0) / (n as f64);
                Complex64::new((6.0 * t).sin(), (4.0 * t).cos())
            })
            .collect();

        let operator = DenseOperator::new(matrix.clone());
        let cfg = BemIrConfig::default();
        let solution = gmres_ir_solve_with_chebyshev_operator(&operator, &matrix, &b, &cfg);

        assert!(
            solution.converged,
            "BEM IR must converge; final={} after {} outer / {} inner",
            solution.final_residual, solution.outer_iterations, solution.total_inner_iterations,
        );

        // Physical residual cross-check (independent of IR bookkeeping).
        let ax = matrix.matvec(&solution.solution);
        let r_norm: f64 = b
            .iter()
            .zip(ax.iter())
            .map(|(a, x)| (a - x).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let rel = r_norm / b_norm;
        assert!(
            rel < 1.0e-9,
            "BEM IR physical residual {rel:.3e} exceeds 1e-9"
        );
    }

    #[test]
    fn checked_ilu_cgs_certifies_against_original_operator() {
        let matrix = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(4.0, 0.2),
                Complex64::new(1.0, -0.1),
                Complex64::new(-0.5, 0.3),
                Complex64::new(3.0, -0.4),
            ],
        );
        let rhs = vec![Complex64::new(2.0, -1.0), Complex64::new(-0.5, 1.5)];
        let config = CgsConfig {
            max_iterations: 30,
            tolerance: 1.0e-12,
            print_interval: 0,
        };

        let solution = try_solve_with_ilu(&matrix, &rhs, &config)
            .expect("valid near-field ILU should produce a checked result");
        assert!(solution.converged, "true residual={}", solution.residual);
        assert!(solution.residual < 1.0e-12);
        assert!(solution.backward_error.is_finite());

        let applied = matrix.matvec(&solution.x);
        let residual = applied
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1.0e-11, "original residual={residual}");

        let gmres_config = GmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };
        let gmres = try_gmres_solve_with_ilu(&matrix, &rhs, &gmres_config)
            .expect("valid near-field ILU should produce a checked GMRES result");
        assert!(gmres.converged, "GMRES true residual={}", gmres.residual);
        assert!(gmres.residual < 1.0e-12);
        assert!(gmres.backward_error.is_finite());
    }
}
