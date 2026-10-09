//! Independent CPU reference for complex generalized eigenvalues.
//!
//! The pencil is a non-Hermitian diagonal-plus-upper-triangular matrix pair.
//! Its generalized eigenvalues are known from the diagonal ratios, while the
//! off-diagonal entries exercise the nonsymmetric Arnoldi path.  Residuals
//! are recomputed here from the returned eigenvectors rather than copied from
//! the solver metadata.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::{
    ComplexArnoldiConfig, CsrMatrix, SolverResultContract, SolverStatus,
    complex_shift_invert_arnoldi_generalized,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.eigen-complex-generalized-reference.v1";
const DIMENSION: usize = 4;
const REQUESTED_EIGENPAIRS: usize = 2;
const MAX_RELATIVE_EIGENVALUE_ERROR: f64 = 1.0e-7;
const MAX_RELATIVE_RESIDUAL: f64 = 1.0e-7;

#[derive(Debug, Clone, Copy)]
struct Pencil {
    eigenvalues: [Complex64; DIMENSION],
    b_diagonal: [f64; DIMENSION],
}

impl Pencil {
    fn independent() -> Self {
        Self {
            eigenvalues: [
                Complex64::new(-2.0, 0.5),
                Complex64::new(0.2, 1.5),
                Complex64::new(1.0, -0.3),
                Complex64::new(4.0, 2.0),
            ],
            b_diagonal: [1.0, 2.0, 3.0, 4.0],
        }
    }

    fn matrices(self) -> (CsrMatrix<Complex64>, CsrMatrix<Complex64>) {
        let mut a = Vec::with_capacity(DIMENSION * 2);
        let mut b = Vec::with_capacity(DIMENSION);
        for row in 0..DIMENSION {
            let diagonal = Complex64::new(self.b_diagonal[row], 0.0) * self.eigenvalues[row];
            a.push((row, row, diagonal));
            b.push((row, row, Complex64::new(self.b_diagonal[row], 0.0)));
            if row + 1 < DIMENSION {
                a.push((row, row + 1, Complex64::new(0.4 + 0.1 * row as f64, -0.2)));
            }
        }
        (
            CsrMatrix::from_triplets(DIMENSION, DIMENSION, a),
            CsrMatrix::from_triplets(DIMENSION, DIMENSION, b),
        )
    }

    fn apply_a(self, vector: &[Complex64]) -> Vec<Complex64> {
        let mut result = vec![Complex64::new(0.0, 0.0); DIMENSION];
        for row in 0..DIMENSION {
            result[row] += self.b_diagonal[row] * self.eigenvalues[row] * vector[row];
            if row + 1 < DIMENSION {
                result[row] += Complex64::new(0.4 + 0.1 * row as f64, -0.2) * vector[row + 1];
            }
        }
        result
    }

    fn apply_b(self, vector: &[Complex64]) -> Vec<Complex64> {
        vector
            .iter()
            .enumerate()
            .map(|(row, value)| self.b_diagonal[row] * *value)
            .collect()
    }
}

struct Metrics {
    max_relative_eigenvalue_error: f64,
    max_relative_residual: f64,
    iterations: usize,
    eigenpairs: usize,
    dimension_rejection: f64,
}

fn relative_residual(pencil: Pencil, eigenvalue: Complex64, vector: &[Complex64]) -> f64 {
    let ax = pencil.apply_a(vector);
    let bx = pencil.apply_b(vector);
    let residual_squared = ax
        .iter()
        .zip(&bx)
        .map(|(a, b)| (*a - eigenvalue * *b).norm_sqr())
        .sum::<f64>();
    let scale_squared = ax
        .iter()
        .zip(&bx)
        .map(|(a, b)| a.norm_sqr() + eigenvalue.norm_sqr() * b.norm_sqr())
        .sum::<f64>();
    residual_squared.sqrt() / scale_squared.sqrt().max(1.0e-30)
}

fn run() -> Result<Metrics, String> {
    let pencil = Pencil::independent();
    let (a, b) = pencil.matrices();
    let target = Complex64::new(0.75, -0.05);
    let mut expected = pencil.eigenvalues.to_vec();
    expected.sort_by(|left, right| {
        (left - target)
            .norm_sqr()
            .total_cmp(&(right - target).norm_sqr())
    });
    expected.truncate(REQUESTED_EIGENPAIRS);

    let config = ComplexArnoldiConfig {
        target,
        num_eigenvalues: REQUESTED_EIGENPAIRS,
        krylov_dimension: DIMENSION,
        tolerance: 1.0e-10,
        max_qr_iterations: 1_000,
        breakdown_tolerance: 1.0e-14,
    };
    let result = complex_shift_invert_arnoldi_generalized(&a, &b, &config)
        .map_err(|error| format!("complex generalized Arnoldi failed: {error}"))?;
    SolverResultContract::validate_contract(&result)
        .map_err(|error| format!("complex eigen result contract failed: {error}"))?;
    if !result.converged || result.eigenvalues.len() != REQUESTED_EIGENPAIRS {
        return Err(format!(
            "complex generalized Arnoldi did not converge: pairs={}, converged={}",
            result.eigenvalues.len(),
            result.converged
        ));
    }

    let mut max_relative_eigenvalue_error = 0.0_f64;
    let mut max_relative_residual = 0.0_f64;
    for (column, (actual, expected)) in result.eigenvalues.iter().zip(expected.iter()).enumerate() {
        let eigenvalue_error = (*actual - *expected).norm() / expected.norm().max(1.0);
        if !eigenvalue_error.is_finite() {
            return Err("non-finite complex eigenvalue error".into());
        }
        max_relative_eigenvalue_error = max_relative_eigenvalue_error.max(eigenvalue_error);

        let vector: Vec<_> = (0..DIMENSION)
            .map(|row| result.eigenvectors[(row, column)])
            .collect();
        let residual = relative_residual(pencil, *actual, &vector);
        if !residual.is_finite() {
            return Err("non-finite independently recomputed eigen-residual".into());
        }
        max_relative_residual = max_relative_residual.max(residual);
    }

    if max_relative_eigenvalue_error > MAX_RELATIVE_EIGENVALUE_ERROR {
        return Err(format!(
            "complex eigenvalue error {:.3e} exceeds {:.3e}",
            max_relative_eigenvalue_error, MAX_RELATIVE_EIGENVALUE_ERROR
        ));
    }
    if max_relative_residual > MAX_RELATIVE_RESIDUAL {
        return Err(format!(
            "complex eigen residual {:.3e} exceeds {:.3e}",
            max_relative_residual, MAX_RELATIVE_RESIDUAL
        ));
    }

    let dimension_error = complex_shift_invert_arnoldi_generalized(
        &a,
        &b,
        &ComplexArnoldiConfig {
            target,
            num_eigenvalues: DIMENSION + 1,
            krylov_dimension: DIMENSION + 1,
            tolerance: 1.0e-10,
            max_qr_iterations: 100,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect_err("complex Arnoldi must reject requests beyond operator dimension");
    if dimension_error.status() != SolverStatus::NumericalError {
        return Err(format!(
            "complex Arnoldi dimension rejection had status {:?}",
            dimension_error.status()
        ));
    }

    Ok(Metrics {
        max_relative_eigenvalue_error,
        max_relative_residual,
        iterations: result.iterations,
        eigenpairs: result.eigenvalues.len(),
        dimension_rejection: 1.0,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("known diagonal ratios of an independent complex generalized pencil")
                .reference_checksum("analytic/deterministic-fixture:complex-generalized-eigen-v1")
                .formulation("non-Hermitian diagonal-plus-upper-triangular generalized pencil")
                .dimension("4x4 complex pencil; two eigenpairs nearest a complex target")
                .metric(
                    "max_relative_eigenvalue_error",
                    metrics.max_relative_eigenvalue_error,
                )
                .maximum(
                    "max_relative_eigenvalue_error",
                    MAX_RELATIVE_EIGENVALUE_ERROR,
                )
                .metric("max_relative_residual", metrics.max_relative_residual)
                .maximum("max_relative_residual", MAX_RELATIVE_RESIDUAL)
                .metric("eigenpairs", metrics.eigenpairs as f64)
                .minimum("eigenpairs", REQUESTED_EIGENPAIRS as f64)
                .metric("dimension_rejection", metrics.dimension_rejection)
                .minimum("dimension_rejection", 1.0)
                .metric("iterations", metrics.iterations as f64)
                .seed("solvers:eigen-complex-generalized:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("complex eigen reference QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("complex eigen failure QA result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
