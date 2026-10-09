//! Quadratic eigenproblems for damped structural dynamics.
//!
//! Solves `(lambda^2 M + lambda C + K) x = 0` for real mass, damping, and
//! stiffness matrices through first-companion linearization, reusing the
//! complex shift-invert Arnoldi driver. Non-symmetric damping (gyroscopic or
//! circulatory terms) is supported: no symmetry is assumed beyond real
//! inputs.
//!
//! Scope: small-to-medium dense-spectrum problems where a sparse LU of the
//! shifted `2n x 2n` companion pencil is affordable. Infinite eigenvalues
//! from a singular mass matrix are not extracted.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, SolverError, SolverResultContract, SolverStatus},
};
use sonium_solvers_sparse::{CsrMatrix, TripletMatrix};

use crate::complex_arnoldi::{ComplexArnoldiConfig, complex_shift_invert_arnoldi_generalized};
use crate::result_contract::complex_result_shape_valid;

/// Configuration for damped quadratic eigenpairs near a spectral target.
#[derive(Debug, Clone)]
pub struct QuadraticEigenConfig {
    /// Complex shift used by the shift-invert transformation.
    pub target: Complex64,
    /// Number of eigenpairs requested (at most `2n`).
    pub num_eigenvalues: usize,
    /// Maximum Arnoldi basis dimension over the `2n` companion system.
    pub krylov_dimension: usize,
    /// Relative true-residual target on the companion linearization.
    pub tolerance: f64,
    /// Maximum shifted-QR iterations for the projected Hessenberg matrix.
    pub max_qr_iterations: usize,
    /// Relative Arnoldi breakdown threshold.
    pub breakdown_tolerance: f64,
}

impl Default for QuadraticEigenConfig {
    fn default() -> Self {
        Self {
            target: Complex64::new(0.0, 0.0),
            num_eigenvalues: 6,
            krylov_dimension: 40,
            tolerance: 1.0e-8,
            max_qr_iterations: 4_000,
            breakdown_tolerance: 1.0e-14,
        }
    }
}

/// Damped eigenpairs recovered from the companion linearization.
#[derive(Debug, Clone)]
pub struct QuadraticEigenResult {
    /// Eigenvalues ordered by distance to the requested target.
    pub eigenvalues: Vec<Complex64>,
    /// Euclidean-normalized `n`-vectors recovered from the companion basis.
    pub eigenvectors: DenseMatrix<Complex64>,
    /// Independently recomputed relative QEP residuals
    /// `||(l^2 M + l C + K) x|| / ((|l|^2 ||M|| + |l| ||C|| + ||K||) ||x||)`.
    pub residual_norms: Vec<f64>,
    /// Number of Arnoldi vectors used on the companion system.
    pub iterations: usize,
    /// Whether every requested pair met the residual tolerance.
    pub converged: bool,
}

impl QuadraticEigenResult {
    /// Return the canonical status for the recovered pairs.
    pub fn solver_status(&self) -> SolverStatus {
        let shape_valid = complex_result_shape_valid(
            &self.eigenvalues,
            &self.eigenvectors,
            &self.residual_norms,
            false,
        );
        if !shape_valid {
            SolverStatus::NumericalError
        } else if self.converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        }
    }

    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            SolverStatus::MaxIterationsReached => AccuracyStatus::NotAttained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverResultContract for QuadraticEigenResult {
    fn solver_status(&self) -> SolverStatus {
        QuadraticEigenResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        QuadraticEigenResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        let status = QuadraticEigenResult::solver_status(self);
        let accuracy = QuadraticEigenResult::accuracy_status(self);
        if status == SolverStatus::NumericalError {
            return Err(SolverError::OperationFailed {
                operation: "quadratic eigen result contract",
                reason: "result shape or diagnostic is non-finite".into(),
            });
        }
        if (status == SolverStatus::Converged) != (accuracy == AccuracyStatus::Attained) {
            return Err(SolverError::OperationFailed {
                operation: "quadratic eigen result contract",
                reason: "status and accuracy disagree".into(),
            });
        }
        Ok(())
    }
}

fn validate_qep_input(matrix: &CsrMatrix<f64>, name: &'static str) -> Result<usize, SolverError> {
    let operation = "quadratic eigenproblem inputs";
    if matrix.num_rows != matrix.num_cols {
        return Err(SolverError::DimensionMismatch {
            expected: matrix.num_rows,
            got: matrix.num_cols,
        });
    }
    let dimension = matrix.num_rows;
    if dimension == 0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!("{name} must be non-empty"),
        });
    }
    if matrix.row_ptrs.len() != dimension + 1
        || matrix.values.len() != matrix.col_indices.len()
        || matrix.row_ptrs[dimension] != matrix.values.len()
    {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!("{name} CSR storage is inconsistent"),
        });
    }
    for row in 0..dimension {
        if matrix.row_ptrs[row] > matrix.row_ptrs[row + 1] {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!("{name} has a decreasing row pointer"),
            });
        }
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            if matrix.col_indices[index] >= dimension {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!("{name} has an out-of-range column index"),
                });
            }
            if !matrix.values[index].is_finite() {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!("{name} contains non-finite entries"),
                });
            }
        }
    }
    Ok(dimension)
}

fn csr_frobenius_norm(matrix: &CsrMatrix<f64>) -> f64 {
    matrix
        .values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn apply_real_csr(matrix: &CsrMatrix<f64>, vector: &[Complex64]) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
    for (row, slot) in output.iter_mut().enumerate() {
        let mut accumulator = Complex64::new(0.0, 0.0);
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let value = Complex64::new(matrix.values[index], 0.0);
            accumulator += value * vector[matrix.col_indices[index]];
        }
        *slot = accumulator;
    }
    output
}

/// Compute damped eigenpairs of `(lambda^2 M + lambda C + K) x = 0`.
///
/// The first-companion linearization `A z = lambda B z` with
/// `A = [0 I; -K -C]` and `B = [I 0; 0 M]` is solved by complex
/// shift-invert Arnoldi; `n`-vector modes are recovered from the top half
/// of each companion vector and rechecked with independent QEP residuals.
///
/// # Errors
///
/// Returns [`SolverError`] when the inputs are not matching square CSR
/// matrices with finite entries, when the configuration is invalid, or
/// when the companion Arnoldi driver fails.
pub fn quadratic_eigen_damped(
    mass: &CsrMatrix<f64>,
    damping: &CsrMatrix<f64>,
    stiffness: &CsrMatrix<f64>,
    config: &QuadraticEigenConfig,
) -> Result<QuadraticEigenResult, SolverError> {
    let operation = "quadratic eigenproblem";
    let mass_dim = validate_qep_input(mass, "mass")?;
    let damping_dim = validate_qep_input(damping, "damping")?;
    let stiffness_dim = validate_qep_input(stiffness, "stiffness")?;
    if mass_dim != damping_dim || mass_dim != stiffness_dim {
        return Err(SolverError::DimensionMismatch {
            expected: mass_dim,
            got: damping_dim.max(stiffness_dim),
        });
    }
    let dimension = mass_dim;
    let companion_dim = 2 * dimension;
    if config.num_eigenvalues == 0 || config.num_eigenvalues > companion_dim {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "num_eigenvalues must lie in 1..={companion_dim} for a {dimension}x{dimension} triple"
            ),
        });
    }
    if config.krylov_dimension == 0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "krylov_dimension must be positive".to_string(),
        });
    }
    for (name, value) in [
        ("tolerance", config.tolerance),
        ("breakdown_tolerance", config.breakdown_tolerance),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!("{name} must be finite and positive"),
            });
        }
    }

    let mut companion_a = TripletMatrix::<Complex64>::new(companion_dim, companion_dim);
    let mut companion_b = TripletMatrix::<Complex64>::new(companion_dim, companion_dim);
    let one = Complex64::new(1.0, 0.0);
    for index in 0..dimension {
        companion_a.push(index, dimension + index, one);
        companion_b.push(index, index, one);
    }
    for row in 0..dimension {
        for index in stiffness.row_ptrs[row]..stiffness.row_ptrs[row + 1] {
            let column = stiffness.col_indices[index];
            companion_a.push(
                dimension + row,
                column,
                Complex64::new(-stiffness.values[index], 0.0),
            );
        }
        for index in damping.row_ptrs[row]..damping.row_ptrs[row + 1] {
            let column = damping.col_indices[index];
            companion_a.push(
                dimension + row,
                dimension + column,
                Complex64::new(-damping.values[index], 0.0),
            );
        }
        for index in mass.row_ptrs[row]..mass.row_ptrs[row + 1] {
            let column = mass.col_indices[index];
            companion_b.push(
                dimension + row,
                dimension + column,
                Complex64::new(mass.values[index], 0.0),
            );
        }
    }
    let matrix_a = companion_a.to_csr();
    let matrix_b = companion_b.to_csr();

    let arnoldi = ComplexArnoldiConfig {
        target: config.target,
        num_eigenvalues: config.num_eigenvalues,
        krylov_dimension: config.krylov_dimension,
        tolerance: config.tolerance,
        max_qr_iterations: config.max_qr_iterations,
        breakdown_tolerance: config.breakdown_tolerance,
    };
    let companion = complex_shift_invert_arnoldi_generalized(&matrix_a, &matrix_b, &arnoldi)?;

    let mass_norm = csr_frobenius_norm(mass);
    let damping_norm = csr_frobenius_norm(damping);
    let stiffness_norm = csr_frobenius_norm(stiffness);
    let mut eigenvalues = Vec::with_capacity(companion.eigenvalues.len());
    let mut recovered: Vec<Vec<Complex64>> = Vec::with_capacity(companion.eigenvalues.len());
    let mut residual_norms = Vec::with_capacity(companion.eigenvalues.len());
    for (pair, &lambda) in companion.eigenvalues.iter().enumerate() {
        if pair >= companion.eigenvectors.ncols() {
            continue;
        }
        let companion_vector = companion.eigenvectors.column_to_vec(pair);
        if companion_vector.len() != companion_dim {
            continue;
        }
        let top = &companion_vector[..dimension];
        let top_norm_squared: f64 = top.iter().map(Complex64::norm_sqr).sum();
        let top_norm = top_norm_squared.sqrt();
        if !top_norm.is_finite() || top_norm < 1.0e-12 {
            continue;
        }
        let mode: Vec<Complex64> = top.iter().map(|value| value / top_norm).collect();
        let mass_term = apply_real_csr(mass, &mode);
        let damping_term = apply_real_csr(damping, &mode);
        let stiffness_term = apply_real_csr(stiffness, &mode);
        let lambda_squared = lambda * lambda;
        let mut residual_squared = 0.0;
        for row in 0..dimension {
            let residual =
                lambda_squared * mass_term[row] + lambda * damping_term[row] + stiffness_term[row];
            residual_squared += residual.norm_sqr();
        }
        let scale =
            lambda_squared.norm() * mass_norm + lambda.norm() * damping_norm + stiffness_norm;
        let residual = if scale > 0.0 {
            residual_squared.sqrt() / scale
        } else {
            residual_squared.sqrt()
        };
        if !residual.is_finite() {
            continue;
        }
        eigenvalues.push(lambda);
        recovered.push(mode);
        residual_norms.push(residual);
    }

    let count = eigenvalues.len();
    let eigenvectors =
        DenseMatrix::<Complex64>::from_fn(dimension, count, |row, column| recovered[column][row]);
    let converged = companion.converged && count == config.num_eigenvalues;
    Ok(QuadraticEigenResult {
        eigenvalues,
        eigenvectors,
        residual_norms,
        iterations: companion.iterations,
        converged,
    })
}
