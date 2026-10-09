//! Reference-only QA for the complex manufactured Helmholtz family.
//!
//! This validates the published exact state/operator/source identity. It is
//! deliberately not a FEM or BEM validation: numerical consumers must compare
//! their independently solved fields against this reference in a separate
//! convergence gate.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedHelmholtz1d;
use std::f64::consts::PI;

const CASE_ID: &str = "wave.helmholtz-manufactured.v1";
const TOLERANCE: f64 = 5.0e-13;

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
                .maximum("max_pde_residual", TOLERANCE)
                .maximum("max_unforced_source", TOLERANCE)
                .maximum("max_dirichlet_endpoint", TOLERANCE)
                .seed("manufactured-helmholtz-1d:complex-uniform-65:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("manufactured Helmholtz QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("manufactured Helmholtz QA failure result must serialize");
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
    let reference = ManufacturedHelmholtz1d::try_new(3.1, 1.3, Complex64::new(2.0, -0.7), -0.2)
        .map_err(|error| error.to_string())?;
    let sample = reference
        .try_sample(-1.0, 1.0, 65)
        .map_err(|error| error.to_string())?;
    sample.validate().map_err(|error| error.to_string())?;
    let max_pde_residual = sample
        .laplacians_pa_per_m2
        .iter()
        .zip(&sample.pressures_pa)
        .zip(&sample.sources_pa_per_m2)
        .map(|((laplacian, pressure), source)| {
            (laplacian + pressure * reference.equation_wavenumber_per_m().powi(2) - source).norm()
        })
        .fold(0.0_f64, f64::max);

    let length = 2.0;
    let wavenumber = PI / length;
    let unforced =
        ManufacturedHelmholtz1d::try_new(wavenumber, wavenumber, Complex64::new(1.5, -0.4), 0.0)
            .map_err(|error| error.to_string())?;
    let unforced_sample = unforced
        .try_sample(0.0, length, 65)
        .map_err(|error| error.to_string())?;
    let max_unforced_source = unforced_sample
        .sources_pa_per_m2
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    let max_dirichlet_endpoint = unforced_sample.pressures_pa[0]
        .norm()
        .max(unforced_sample.pressures_pa[unforced_sample.pressures_pa.len() - 1].norm());

    for (name, value) in [
        ("PDE residual", max_pde_residual),
        ("unforced source", max_unforced_source),
        ("Dirichlet endpoint", max_dirichlet_endpoint),
    ] {
        if !value.is_finite() || value > TOLERANCE {
            return Err(format!("{name} failed: {value:.3e} > {TOLERANCE:.3e}"));
        }
    }
    Ok(Metrics {
        max_pde_residual,
        max_unforced_source,
        max_dirichlet_endpoint,
        sample_count: sample.positions_m.len() + unforced_sample.positions_m.len(),
    })
}
