//! Release-facing deterministic PDE validation workloads.

use sonium_qa::QaResult;
use sonium_solvers::{GmresConfig, convection_diffusion_1d_fixture, gmres, poisson_1d_fixture};
use std::time::Instant;

const CASE_ID: &str = "solvers.validation-corpus-workloads.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:validation-corpus-workloads-v1";

fn relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = expected
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt();
    numerator / denominator.max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let expected: Vec<f64> = (0..16).map(|index| 1.0 + index as f64 / 16.0).collect();
    let config = GmresConfig {
        max_iterations: 200,
        restart: 16,
        tolerance: 1.0e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        ..Default::default()
    };

    let poisson = poisson_1d_fixture(16).map_err(|error| format!("Poisson fixture: {error}"))?;
    if poisson.values.len() != 46 {
        return Err(format!(
            "Poisson metadata mismatch: nnz={}",
            poisson.values.len()
        ));
    }
    let poisson_rhs = poisson.matvec(&expected);
    let poisson_solution = gmres(&poisson, &poisson_rhs, &config);
    if !poisson_solution.converged {
        return Err(format!(
            "Poisson GMRES did not converge: residual={}",
            poisson_solution.residual
        ));
    }

    let convection = convection_diffusion_1d_fixture(16)
        .map_err(|error| format!("convection-diffusion fixture: {error}"))?;
    if convection.values.len() != 46 {
        return Err(format!(
            "convection-diffusion metadata mismatch: nnz={}",
            convection.values.len()
        ));
    }
    let convection_rhs = convection.matvec(&expected);
    let convection_solution = gmres(&convection, &convection_rhs, &config);
    if !convection_solution.converged {
        return Err(format!(
            "convection-diffusion GMRES did not converge: residual={}",
            convection_solution.residual
        ));
    }

    Ok(vec![
        (
            "poisson_solution_error",
            relative_error(&poisson_solution.x, &expected),
        ),
        ("poisson_true_residual", poisson_solution.residual),
        (
            "convection_diffusion_solution_error",
            relative_error(&convection_solution.x, &expected),
        ),
        (
            "convection_diffusion_true_residual",
            convection_solution.residual,
        ),
        ("fixture_dimension", poisson.num_rows as f64),
        ("fixture_nnz", poisson.values.len() as f64),
    ])
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(metrics) => {
            let mut record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("analytic known-solution vectors with independent true residuals")
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .dimension("16x16 Poisson and upwind convection-diffusion CSR stencils")
                .formulation("GMRES on checked deterministic PDE fixtures")
                .tolerance("solution and true residual errors <= 1e-10; 46 nnz")
                .seed("solvers-validation-corpus-workloads:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                record = record.metric(name, value);
            }
            record
                .maximum("poisson_solution_error", 1.0e-10)
                .maximum("poisson_true_residual", 1.0e-10)
                .maximum("convection_diffusion_solution_error", 1.0e-10)
                .maximum("convection_diffusion_true_residual", 1.0e-10)
                .minimum("fixture_dimension", 16.0)
                .minimum("fixture_nnz", 46.0)
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .input_checksum(CHECKSUM)
            .reference_checksum(CHECKSUM)
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
