//! Reference-only QA for the static manufactured axial-elasticity family.
//!
//! This validates the exact displacement/strain/stress/force/load identities.
//! It deliberately does not claim a FEM solve: a discretization must compare
//! its independently computed field against this reference in a separate gate.

use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedElasticity1d;
use std::f64::consts::PI;

const CASE_ID: &str = "wave.elasticity-manufactured-1d.v1";
const EXACT_TOLERANCE: f64 = 5.0e-13;
const FINITE_DIFFERENCE_TOLERANCE: f64 = 6.0e-5;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric(
                    "max_force_balance_relative_residual",
                    metrics.max_force_balance_relative_residual,
                )
                .metric(
                    "max_exact_equilibrium_relative_residual",
                    metrics.max_exact_equilibrium_relative_residual,
                )
                .metric(
                    "max_centered_equilibrium_relative_residual",
                    metrics.max_centered_equilibrium_relative_residual,
                )
                .metric("max_dirichlet_endpoint_m", metrics.max_dirichlet_endpoint_m)
                .metric("sample_count", metrics.sample_count as f64)
                .maximum("max_force_balance_relative_residual", EXACT_TOLERANCE)
                .maximum("max_exact_equilibrium_relative_residual", EXACT_TOLERANCE)
                .maximum(
                    "max_centered_equilibrium_relative_residual",
                    FINITE_DIFFERENCE_TOLERANCE,
                )
                .maximum("max_dirichlet_endpoint_m", EXACT_TOLERANCE)
                .seed("manufactured-elasticity-1d:axial-bar-uniform-257:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("manufactured elasticity QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("manufactured elasticity QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    max_force_balance_relative_residual: f64,
    max_exact_equilibrium_relative_residual: f64,
    max_centered_equilibrium_relative_residual: f64,
    max_dirichlet_endpoint_m: f64,
    sample_count: usize,
}

fn run() -> Result<Metrics, String> {
    let reference = ManufacturedElasticity1d::try_new(70.0e9, 2.5e-4, 2.3, 1.2e-3, -0.15)
        .map_err(|error| error.to_string())?;
    let sample = reference
        .try_sample(-1.0, 1.0, 257)
        .map_err(|error| error.to_string())?;
    sample.validate().map_err(|error| error.to_string())?;

    let stiffness_n = reference.youngs_modulus_pa() * reference.area_m2();
    let max_force_balance_relative_residual = sample
        .axial_stresses_pa
        .iter()
        .zip(&sample.axial_forces_n)
        .map(|(stress, force)| (stress * reference.area_m2() - force).abs() / force.abs().max(1.0))
        .fold(0.0_f64, f64::max);
    let max_exact_equilibrium_relative_residual = sample
        .displacements_m
        .iter()
        .zip(&sample.distributed_loads_n_per_m)
        .map(|(displacement, load)| {
            (stiffness_n * reference.spatial_wavenumber_per_m().powi(2) * displacement - load).abs()
                / load.abs().max(1.0)
        })
        .fold(0.0_f64, f64::max);

    let dx_m = sample.positions_m[1] - sample.positions_m[0];
    let max_centered_equilibrium_relative_residual = sample
        .axial_forces_n
        .windows(3)
        .zip(sample.distributed_loads_n_per_m.iter().skip(1))
        .map(|(force, load)| {
            let centered_load = -(force[2] - force[0]) / (2.0 * dx_m);
            (centered_load - load).abs() / load.abs().max(1.0)
        })
        .fold(0.0_f64, f64::max);

    let length_m = 2.0;
    let constrained =
        ManufacturedElasticity1d::try_new(210.0e9, 1.0e-4, PI / length_m, 2.0e-3, 0.0)
            .map_err(|error| error.to_string())?;
    let constrained_sample = constrained
        .try_sample(0.0, length_m, 65)
        .map_err(|error| error.to_string())?;
    let max_dirichlet_endpoint_m = constrained_sample.displacements_m[0].abs().max(
        constrained_sample.displacements_m[constrained_sample.displacements_m.len() - 1].abs(),
    );

    for (name, value, tolerance) in [
        (
            "force balance",
            max_force_balance_relative_residual,
            EXACT_TOLERANCE,
        ),
        (
            "exact equilibrium",
            max_exact_equilibrium_relative_residual,
            EXACT_TOLERANCE,
        ),
        (
            "centered equilibrium",
            max_centered_equilibrium_relative_residual,
            FINITE_DIFFERENCE_TOLERANCE,
        ),
        (
            "Dirichlet endpoint",
            max_dirichlet_endpoint_m,
            EXACT_TOLERANCE,
        ),
    ] {
        if !value.is_finite() || value > tolerance {
            return Err(format!("{name} failed: {value:.3e} > {tolerance:.3e}"));
        }
    }

    Ok(Metrics {
        max_force_balance_relative_residual,
        max_exact_equilibrium_relative_residual,
        max_centered_equilibrium_relative_residual,
        max_dirichlet_endpoint_m,
        sample_count: sample.positions_m.len() + constrained_sample.positions_m.len(),
    })
}
