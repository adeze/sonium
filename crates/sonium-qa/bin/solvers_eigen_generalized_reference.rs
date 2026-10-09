//! Independent CPU reference for the real generalized eigenvalue path.

use sonium_qa::QaResult;
use sonium_solvers::{CsrMatrix, ShiftInvertConfig, shift_invert_lanczos_generalized};
use std::time::Instant;

const CASE_ID: &str = "solvers.eigen-generalized-reference.v1";
const SIZE: usize = 6;
const REQUESTED_EIGENPAIRS: usize = 3;
const MAX_RELATIVE_EIGENVALUE_ERROR: f64 = 1.0e-9;
const MAX_RELATIVE_RESIDUAL: f64 = 1.0e-9;

#[derive(Clone, Copy)]
struct Pencil {
    eigenvalues: [f64; SIZE],
    mass: [f64; SIZE],
}

impl Pencil {
    fn independent() -> Self {
        Self {
            eigenvalues: [1.0, 3.0, 5.0, 8.0, 13.0, 21.0],
            mass: [1.0, 2.0, 1.5, 0.5, 3.0, 2.5],
        }
    }

    fn matrices(self) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
        let a = self
            .eigenvalues
            .into_iter()
            .zip(self.mass)
            .enumerate()
            .map(|(index, (eigenvalue, mass))| (index, index, eigenvalue * mass))
            .collect();
        let b = self
            .mass
            .into_iter()
            .enumerate()
            .map(|(index, mass)| (index, index, mass))
            .collect();
        (
            CsrMatrix::from_triplets(SIZE, SIZE, a),
            CsrMatrix::from_triplets(SIZE, SIZE, b),
        )
    }

    fn relative_residual(self, eigenvalue: f64, vector: &[f64]) -> f64 {
        let mut residual_squared = 0.0;
        let mut scale_squared = 0.0;
        for index in 0..SIZE {
            let ax = self.eigenvalues[index] * self.mass[index] * vector[index];
            let bx = self.mass[index] * vector[index];
            residual_squared += (ax - eigenvalue * bx).powi(2);
            scale_squared += ax.powi(2) + (eigenvalue * bx).powi(2);
        }
        residual_squared.sqrt() / scale_squared.sqrt().max(1.0e-30)
    }
}

struct Metrics {
    max_relative_eigenvalue_error: f64,
    max_relative_residual: f64,
    iterations: usize,
    eigenpairs: usize,
}

fn run() -> Result<Metrics, String> {
    let pencil = Pencil::independent();
    let (a, b) = pencil.matrices();
    let config = ShiftInvertConfig {
        target: 7.0,
        num_eigenvalues: REQUESTED_EIGENPAIRS,
        krylov_dimension: SIZE,
        tolerance: 1.0e-10,
        breakdown_tolerance: 1.0e-14,
    };
    let result = shift_invert_lanczos_generalized(&a, &b, &config)
        .map_err(|error| format!("real generalized Lanczos failed: {error}"))?;
    let expected = [8.0, 5.0, 3.0];
    if !result.converged || result.eigenvalues.len() != REQUESTED_EIGENPAIRS {
        return Err(format!(
            "real generalized Lanczos did not converge: pairs={}, converged={}",
            result.eigenvalues.len(),
            result.converged
        ));
    }

    let mut max_relative_eigenvalue_error = 0.0_f64;
    let mut max_relative_residual = 0.0_f64;
    for (column, (actual, expected)) in result.eigenvalues.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs() / expected.abs().max(1.0);
        if !error.is_finite() {
            return Err("non-finite generalized eigenvalue error".into());
        }
        max_relative_eigenvalue_error = max_relative_eigenvalue_error.max(error);
        let vector: Vec<_> = (0..SIZE)
            .map(|row| result.eigenvectors[(row, column)])
            .collect();
        let residual = pencil.relative_residual(*actual, &vector);
        if !residual.is_finite() {
            return Err("non-finite independently recomputed generalized residual".into());
        }
        max_relative_residual = max_relative_residual.max(residual);
    }

    if max_relative_eigenvalue_error > MAX_RELATIVE_EIGENVALUE_ERROR {
        return Err(format!(
            "generalized eigenvalue error {:.3e} exceeds {:.3e}",
            max_relative_eigenvalue_error, MAX_RELATIVE_EIGENVALUE_ERROR
        ));
    }
    if max_relative_residual > MAX_RELATIVE_RESIDUAL {
        return Err(format!(
            "generalized eigen residual {:.3e} exceeds {:.3e}",
            max_relative_residual, MAX_RELATIVE_RESIDUAL
        ));
    }

    Ok(Metrics {
        max_relative_eigenvalue_error,
        max_relative_residual,
        iterations: result.iterations,
        eigenpairs: result.eigenvalues.len(),
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("known diagonal ratios of an independent SPD generalized pencil")
                .reference_checksum("analytic/deterministic-fixture:real-generalized-eigen-v1")
                .formulation("diagonal SPD generalized pencil with non-unit positive mass")
                .dimension("6x6 real generalized pencil; three eigenpairs nearest target 7")
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
                .metric("iterations", metrics.iterations as f64)
                .seed("solvers:eigen-generalized:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("generalized eigen reference QA result must serialize")
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
                    .expect("generalized eigen failure QA result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
