//! Reference-only QA for the scalar manufactured diffusion family.
//!
//! This gate checks the published state/derivative/source identity and is
//! deliberately not labelled as a FEM or BEM validation. FEM consumers should
//! use the same source field for an independent discretization-convergence
//! case.

use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedDiffusion1d;
use std::f64::consts::PI;

const CASE_ID: &str = "wave.diffusion-manufactured.v1";
const PDE_RESIDUAL_TOLERANCE: f64 = 5.0e-14;
const ENDPOINT_TOLERANCE: f64 = 5.0e-14;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_pde_residual", metrics.max_pde_residual)
                .metric("max_unforced_source", metrics.max_unforced_source)
                .metric("max_dirichlet_endpoint", metrics.max_dirichlet_endpoint)
                .metric("sample_count", metrics.sample_count as f64)
                .maximum("max_pde_residual", PDE_RESIDUAL_TOLERANCE)
                .maximum("max_unforced_source", PDE_RESIDUAL_TOLERANCE)
                .maximum("max_dirichlet_endpoint", ENDPOINT_TOLERANCE)
                .seed("manufactured-diffusion-1d:uniform-65:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("manufactured diffusion QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("manufactured diffusion QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    max_pde_residual: f64,
    max_unforced_source: f64,
    max_dirichlet_endpoint: f64,
    sample_count: usize,
}

fn run() -> Result<Metrics, String> {
    let reference = ManufacturedDiffusion1d::try_new(0.7, 1.3, 0.4, 2.0, -0.2)
        .map_err(|error| error.to_string())?;
    let sample = reference
        .try_sample(-1.0, 1.0, 65, 0.8)
        .map_err(|error| error.to_string())?;
    sample.validate().map_err(|error| error.to_string())?;

    let max_pde_residual = sample
        .time_derivatives
        .iter()
        .zip(&sample.laplacians)
        .zip(&sample.sources)
        .map(|((time_derivative, laplacian), source)| {
            (time_derivative - reference.diffusivity_m2_per_s() * laplacian - source).abs()
        })
        .fold(0.0_f64, f64::max);

    let length = 2.0;
    let wavenumber = PI / length;
    let unforced =
        ManufacturedDiffusion1d::try_new(0.5, wavenumber, 0.5 * wavenumber.powi(2), 1.5, 0.0)
            .map_err(|error| error.to_string())?;
    let unforced_sample = unforced
        .try_sample(0.0, length, 65, 0.3)
        .map_err(|error| error.to_string())?;
    let max_unforced_source = unforced_sample
        .sources
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let max_dirichlet_endpoint = unforced_sample.values[0]
        .abs()
        .max(unforced_sample.values[unforced_sample.values.len() - 1].abs());

    for (name, value, tolerance) in [
        ("PDE residual", max_pde_residual, PDE_RESIDUAL_TOLERANCE),
        (
            "unforced source",
            max_unforced_source,
            PDE_RESIDUAL_TOLERANCE,
        ),
        (
            "Dirichlet endpoint",
            max_dirichlet_endpoint,
            ENDPOINT_TOLERANCE,
        ),
    ] {
        if !value.is_finite() || value > tolerance {
            return Err(format!("{name} failed: {value:.3e} > {tolerance:.3e}"));
        }
    }

    Ok(Metrics {
        max_pde_residual,
        max_unforced_source,
        max_dirichlet_endpoint,
        sample_count: sample.positions_m.len() + unforced_sample.positions_m.len(),
    })
}
