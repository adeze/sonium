//! Production structural-mechanics equilibrium and energy QA.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{Beam2DStudy, run_beam2d_static};

const CASE_ID: &str = "structural.beam-static-balance.v1";
const RESIDUAL_TOLERANCE: f64 = 2.0e-8;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("force_balance_residual_n", metrics.force_balance_residual)
                .metric(
                    "free_equilibrium_residual_n",
                    metrics.free_equilibrium_residual,
                )
                .metric(
                    "energy_balance_relative_error",
                    metrics.energy_balance_relative_error,
                )
                .metric(
                    "analytical_relative_error",
                    metrics.analytical_relative_error,
                )
                .metric("tip_deflection_m", metrics.tip_deflection)
                .maximum("force_balance_residual_n", RESIDUAL_TOLERANCE)
                .maximum("free_equilibrium_residual_n", RESIDUAL_TOLERANCE)
                .maximum("energy_balance_relative_error", RESIDUAL_TOLERANCE)
                .maximum("analytical_relative_error", 1.0e-8)
                .seed("beam2d-introductory:elements=8:cpu");
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
            eprintln!("STRUCTURAL BEAM BALANCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    force_balance_residual: f64,
    free_equilibrium_residual: f64,
    energy_balance_relative_error: f64,
    analytical_relative_error: f64,
    tip_deflection: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = Beam2DStudy::introductory();
    study.geometry.elements = 8;
    let result = run_beam2d_static(&study).map_err(|error| error.to_string())?;
    if !result.converged {
        return Err("production beam solve did not converge".into());
    }

    let values = [
        result.force_balance_residual_n,
        result.free_equilibrium_residual_n,
        result.energy_balance_relative_error,
        result.analytical_relative_error,
        result.tip_deflection_m,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("production beam result contained a non-finite metric".into());
    }
    if result.force_balance_residual_n > RESIDUAL_TOLERANCE
        || result.free_equilibrium_residual_n > RESIDUAL_TOLERANCE
        || result.energy_balance_relative_error > RESIDUAL_TOLERANCE
        || result.analytical_relative_error > 1.0e-8
    {
        return Err(format!(
            "beam balance exceeded budgets: force={:.3e}, free={:.3e}, energy={:.3e}, analytical={:.3e}",
            result.force_balance_residual_n,
            result.free_equilibrium_residual_n,
            result.energy_balance_relative_error,
            result.analytical_relative_error,
        ));
    }

    Ok(Metrics {
        force_balance_residual: result.force_balance_residual_n,
        free_equilibrium_residual: result.free_equilibrium_residual_n,
        energy_balance_relative_error: result.energy_balance_relative_error,
        analytical_relative_error: result.analytical_relative_error,
        tip_deflection: result.tip_deflection_m,
    })
}
