//! Reference-only QA for the steady manufactured transport family.
//!
//! This checks the exact advection-diffusion-reaction identity. It is a
//! reusable field/source reference, not a claim that a FEM transport solve has
//! converged independently.

use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedTransport1d;
use std::f64::consts::PI;

const CASE_ID: &str = "wave.transport-manufactured-1d.v1";
const EXACT_TOLERANCE: f64 = 5.0e-13;
const FINITE_DIFFERENCE_TOLERANCE: f64 = 8.0e-5;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_exact_pde_residual", metrics.max_exact_pde_residual)
                .metric(
                    "max_centered_pde_relative_residual",
                    metrics.max_centered_pde_relative_residual,
                )
                .metric("max_dirichlet_endpoint", metrics.max_dirichlet_endpoint)
                .metric("sample_count", metrics.sample_count as f64)
                .maximum("max_exact_pde_residual", EXACT_TOLERANCE)
                .maximum(
                    "max_centered_pde_relative_residual",
                    FINITE_DIFFERENCE_TOLERANCE,
                )
                .maximum("max_dirichlet_endpoint", EXACT_TOLERANCE)
                .seed("manufactured-transport-1d:adr-uniform-257:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("manufactured transport QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("manufactured transport QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    max_exact_pde_residual: f64,
    max_centered_pde_relative_residual: f64,
    max_dirichlet_endpoint: f64,
    sample_count: usize,
}

fn run() -> Result<Metrics, String> {
    let reference = ManufacturedTransport1d::try_new(0.7, -1.1, 0.4, 1.3, 2.0, -0.2)
        .map_err(|error| error.to_string())?;
    let sample = reference
        .try_sample(-1.0, 1.0, 257)
        .map_err(|error| error.to_string())?;
    sample.validate().map_err(|error| error.to_string())?;
    let max_exact_pde_residual = sample
        .values
        .iter()
        .zip(&sample.spatial_gradients)
        .zip(&sample.laplacians)
        .zip(&sample.sources)
        .map(|(((value, gradient), laplacian), source)| {
            (-reference.diffusivity_m2_per_s() * laplacian
                + reference.velocity_m_per_s() * gradient
                + reference.reaction_rate_per_s() * value
                - source)
                .abs()
        })
        .fold(0.0_f64, f64::max);

    let dx_m = sample.positions_m[1] - sample.positions_m[0];
    let max_centered_pde_relative_residual = sample
        .values
        .windows(3)
        .zip(sample.sources.iter().skip(1))
        .map(|(value, source)| {
            let gradient = (value[2] - value[0]) / (2.0 * dx_m);
            let laplacian = (value[2] - 2.0 * value[1] + value[0]) / dx_m.powi(2);
            let centered_source = -reference.diffusivity_m2_per_s() * laplacian
                + reference.velocity_m_per_s() * gradient
                + reference.reaction_rate_per_s() * value[1];
            (centered_source - source).abs() / source.abs().max(1.0)
        })
        .fold(0.0_f64, f64::max);

    let length_m = 2.0;
    let constrained = ManufacturedTransport1d::try_new(0.5, 1.2, 0.8, PI / length_m, 1.5, 0.0)
        .map_err(|error| error.to_string())?;
    let constrained_sample = constrained
        .try_sample(0.0, length_m, 65)
        .map_err(|error| error.to_string())?;
    let max_dirichlet_endpoint = constrained_sample.values[0]
        .abs()
        .max(constrained_sample.values[constrained_sample.values.len() - 1].abs());

    for (name, value, tolerance) in [
        ("exact PDE", max_exact_pde_residual, EXACT_TOLERANCE),
        (
            "centered PDE",
            max_centered_pde_relative_residual,
            FINITE_DIFFERENCE_TOLERANCE,
        ),
        (
            "Dirichlet endpoint",
            max_dirichlet_endpoint,
            EXACT_TOLERANCE,
        ),
    ] {
        if !value.is_finite() || value > tolerance {
            return Err(format!("{name} failed: {value:.3e} > {tolerance:.3e}"));
        }
    }

    Ok(Metrics {
        max_exact_pde_residual,
        max_centered_pde_relative_residual,
        max_dirichlet_endpoint,
        sample_count: sample.positions_m.len() + constrained_sample.positions_m.len(),
    })
}
