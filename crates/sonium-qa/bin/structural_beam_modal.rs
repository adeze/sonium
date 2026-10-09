//! Production structural modal orthogonality and cantilever-frequency QA.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{Beam2DStudy, run_beam2d_modal};

const CASE_ID: &str = "structural.beam-modal-orthogonality.v1";
const ORTHOGONALITY_TOLERANCE: f64 = 1.0e-8;
const FREQUENCY_RELATIVE_TOLERANCE: f64 = 5.0e-2;

struct Metrics {
    first_frequency_hz: f64,
    analytical_first_frequency_hz: f64,
    first_frequency_relative_error: f64,
    max_residual_norm: f64,
    mass_normalization_error: f64,
    orthogonality_error: f64,
    solver_iterations: usize,
}

fn run() -> Result<Metrics, String> {
    let mut study = Beam2DStudy::introductory();
    study.geometry.elements = 8;
    study.modal_modes = 3;
    let result = run_beam2d_modal(&study).map_err(|error| error.to_string())?;
    if !result.converged || result.frequencies_hz.len() != study.modal_modes {
        return Err(
            "production beam modal solve did not converge or returned the wrong mode count".into(),
        );
    }
    if result
        .frequencies_hz
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || result
            .residual_norms
            .iter()
            .any(|residual| !residual.is_finite())
        || !result.mass_normalization_error.is_finite()
        || !result.orthogonality_error.is_finite()
    {
        return Err(
            "production beam modal result contained a non-finite or non-positive value".into(),
        );
    }

    let beta_1 = 1.875_104_068_711_961_f64;
    let analytical_first_frequency_hz = beta_1.powi(2)
        * (study.material.youngs_modulus_pa * study.geometry.second_moment_m4()
            / (study.material.density_kg_m3
                * study.geometry.area_m2()
                * study.geometry.length_m.powi(4)))
        .sqrt()
        / std::f64::consts::TAU;
    let first_frequency_hz = result.frequencies_hz[0];
    let first_frequency_relative_error =
        (first_frequency_hz - analytical_first_frequency_hz).abs() / analytical_first_frequency_hz;
    let max_residual_norm = result.residual_norms.iter().copied().fold(0.0, f64::max);
    if first_frequency_relative_error > FREQUENCY_RELATIVE_TOLERANCE
        || result.mass_normalization_error > ORTHOGONALITY_TOLERANCE
        || result.orthogonality_error > ORTHOGONALITY_TOLERANCE
    {
        return Err(format!(
            "modal budgets exceeded: frequency={first_frequency_relative_error:.3e}, mass={:.3e}, orthogonality={:.3e}",
            result.mass_normalization_error, result.orthogonality_error
        ));
    }

    Ok(Metrics {
        first_frequency_hz,
        analytical_first_frequency_hz,
        first_frequency_relative_error,
        max_residual_norm,
        mass_normalization_error: result.mass_normalization_error,
        orthogonality_error: result.orthogonality_error,
        solver_iterations: result.solver_iterations,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("first_frequency_hz", metrics.first_frequency_hz)
                .metric(
                    "analytical_first_frequency_hz",
                    metrics.analytical_first_frequency_hz,
                )
                .metric(
                    "first_frequency_relative_error",
                    metrics.first_frequency_relative_error,
                )
                .metric("max_residual_norm", metrics.max_residual_norm)
                .metric("mass_normalization_error", metrics.mass_normalization_error)
                .metric("orthogonality_error", metrics.orthogonality_error)
                .metric("solver_iterations", metrics.solver_iterations as f64)
                .maximum(
                    "first_frequency_relative_error",
                    FREQUENCY_RELATIVE_TOLERANCE,
                )
                .maximum("mass_normalization_error", ORTHOGONALITY_TOLERANCE)
                .maximum("orthogonality_error", ORTHOGONALITY_TOLERANCE)
                .seed("beam2d-introductory:elements=8:modes=3:mass-normalized:cpu");
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL BEAM MODAL FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
