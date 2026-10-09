//! Independent CPU spectral reference for the checked LOBPCG path.
//!
//! The tridiagonal Dirichlet Laplacian has a closed-form spectrum.  The
//! matrix is assembled independently of the eigensolver and the reported
//! eigenvalues are checked against that spectrum as well as the returned
//! residual metadata.

use sonium_qa::QaResult;
use sonium_solvers::{CsrBuilder, LobpcgConfig, SolverResultContract, lobpcg};
use std::time::Instant;

const CASE_ID: &str = "solvers.eigen-laplacian-reference.v1";
const MAX_RELATIVE_EIGENVALUE_ERROR: f64 = 1.0e-6;
const MAX_RESIDUAL_NORM: f64 = 1.0e-8;

struct Metrics {
    max_relative_eigenvalue_error: f64,
    max_residual_norm: f64,
    iterations: usize,
    eigenpairs: usize,
}

fn dirichlet_laplacian(size: usize) -> Result<sonium_solvers::CsrMatrix<f64>, String> {
    if size < 3 {
        return Err("Dirichlet reference requires at least three interior points".into());
    }

    let h = 1.0 / (size as f64 + 1.0);
    let scale = 1.0 / (h * h);
    let mut builder = CsrBuilder::new(size, size);
    for row in 0..size {
        let mut entries = Vec::with_capacity(3);
        if row > 0 {
            entries.push((row - 1, -scale));
        }
        entries.push((row, 2.0 * scale));
        if row + 1 < size {
            entries.push((row + 1, -scale));
        }
        builder.add_row_entries(entries.into_iter());
    }
    Ok(builder.finish())
}

fn exact_eigenvalues(size: usize, count: usize) -> Vec<f64> {
    let h = 1.0 / (size as f64 + 1.0);
    (1..=count)
        .map(|mode| {
            let angle = (mode as f64 * std::f64::consts::PI) / (2.0 * (size as f64 + 1.0));
            4.0 * angle.sin().powi(2) / (h * h)
        })
        .collect()
}

fn run() -> Result<Metrics, String> {
    let size = 16;
    let requested = 3;
    let matrix = dirichlet_laplacian(size)?;
    let result = lobpcg(
        &matrix,
        &LobpcgConfig {
            num_eigenvalues: requested,
            max_iterations: 300,
            tolerance: 1.0e-10,
            block_padding: 0,
        },
    )
    .map_err(|error| format!("LOBPCG failed on Dirichlet Laplacian: {error}"))?;
    SolverResultContract::validate_contract(&result)
        .map_err(|error| format!("LOBPCG result contract failed: {error}"))?;

    if !result.converged {
        return Err(format!(
            "LOBPCG did not converge after {} iterations",
            result.iterations
        ));
    }
    if result.eigenvalues.len() != requested || result.residual_norms.len() != requested {
        return Err(format!(
            "expected {requested} eigenpairs, got {} values and {} residuals",
            result.eigenvalues.len(),
            result.residual_norms.len()
        ));
    }

    let reference = exact_eigenvalues(size, requested);
    let mut max_relative_eigenvalue_error = 0.0_f64;
    for (actual, expected) in result.eigenvalues.iter().zip(reference.iter()) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("non-finite eigenvalue in spectral reference".into());
        }
        max_relative_eigenvalue_error =
            max_relative_eigenvalue_error.max((actual - expected).abs() / expected.abs().max(1.0));
    }

    let max_residual_norm =
        result
            .residual_norms
            .iter()
            .copied()
            .try_fold(0.0_f64, |maximum, residual| {
                if residual.is_finite() {
                    Ok(maximum.max(residual))
                } else {
                    Err("non-finite LOBPCG residual metadata".to_string())
                }
            })?;

    if max_relative_eigenvalue_error > MAX_RELATIVE_EIGENVALUE_ERROR {
        return Err(format!(
            "eigenvalue error {:.3e} exceeds {:.3e}",
            max_relative_eigenvalue_error, MAX_RELATIVE_EIGENVALUE_ERROR
        ));
    }
    if max_residual_norm > MAX_RESIDUAL_NORM {
        return Err(format!(
            "residual norm {:.3e} exceeds {:.3e}",
            max_residual_norm, MAX_RESIDUAL_NORM
        ));
    }

    Ok(Metrics {
        max_relative_eigenvalue_error,
        max_residual_norm,
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
                .reference("closed-form eigenvalues 4 h^-2 sin^2(k pi / (2(n+1)))")
                .reference_checksum("analytic/deterministic-fixture:dirichlet-laplacian-v1")
                .formulation("independent symmetric tridiagonal finite-difference assembly")
                .dimension("16 interior points; three lowest eigenpairs")
                .metric(
                    "max_relative_eigenvalue_error",
                    metrics.max_relative_eigenvalue_error,
                )
                .maximum(
                    "max_relative_eigenvalue_error",
                    MAX_RELATIVE_EIGENVALUE_ERROR,
                )
                .metric("max_residual_norm", metrics.max_residual_norm)
                .maximum("max_residual_norm", MAX_RESIDUAL_NORM)
                .metric("eigenpairs", metrics.eigenpairs as f64)
                .minimum("eigenpairs", 3.0)
                .metric("iterations", metrics.iterations as f64)
                .seed("solvers:eigen-laplacian:dirichlet:cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("eigen reference QA result must serialize")
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
                    .expect("eigen reference QA failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
