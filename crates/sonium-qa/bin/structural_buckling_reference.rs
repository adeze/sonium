//! Independent CPU release wrapper for the structural linear Euler buckling slice.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{BucklingEndCondition, BucklingStudy, run_linear_buckling};
use std::time::Instant;

const CASE_ID: &str = "structural.linear-euler-buckling-reference.v1";
const RELATIVE_TOLERANCE: f64 = 5.0e-2;
const RESIDUAL_TOLERANCE: f64 = 1.0e-8;

struct Metrics {
    cases: f64,
    max_relative_error: f64,
    max_residual_norm: f64,
    max_boundary_error: f64,
}

fn independent_euler_load(study: &BucklingStudy) -> f64 {
    let second_moment_m4 = study.geometry.width_m * study.geometry.height_m.powi(3) / 12.0;
    let effective_length_factor = match study.end_condition {
        BucklingEndCondition::PinnedPinned => 1.0,
        BucklingEndCondition::Cantilever => 2.0,
    };
    std::f64::consts::PI.powi(2) * study.youngs_modulus_pa * second_moment_m4
        / (effective_length_factor * study.geometry.length_m).powi(2)
}

fn run() -> Result<Metrics, String> {
    let mut max_relative_error: f64 = 0.0;
    let mut max_residual_norm: f64 = 0.0;
    let mut max_boundary_error: f64 = 0.0;

    for end_condition in [
        BucklingEndCondition::PinnedPinned,
        BucklingEndCondition::Cantilever,
    ] {
        let mut study = BucklingStudy::introductory();
        study.geometry.elements = 8;
        study.end_condition = end_condition;

        let reference = independent_euler_load(&study);
        let result = run_linear_buckling(&study).map_err(|error| error.to_string())?;
        if !result.converged {
            return Err(format!("{end_condition:?}: solver did not converge"));
        }
        if !result.eigenvalue_n.is_finite() || !result.residual_norm.is_finite() {
            return Err(format!("{end_condition:?}: non-finite eigen result"));
        }

        let relative_error = (result.eigenvalue_n - reference).abs() / reference;
        let first_boundary_error = result
            .mode_shape
            .first()
            .ok_or_else(|| format!("{end_condition:?}: empty mode shape"))?[0]
            .abs();
        let boundary_error = if end_condition == BucklingEndCondition::PinnedPinned {
            first_boundary_error.max(
                result
                    .mode_shape
                    .last()
                    .ok_or_else(|| format!("{end_condition:?}: empty mode shape"))?[0]
                    .abs(),
            )
        } else {
            first_boundary_error
        };
        if relative_error > RELATIVE_TOLERANCE {
            return Err(format!(
                "{end_condition:?}: eigenvalue relative error {relative_error} exceeds {RELATIVE_TOLERANCE}"
            ));
        }
        if result.residual_norm > RESIDUAL_TOLERANCE {
            return Err(format!(
                "{end_condition:?}: residual {} exceeds {RESIDUAL_TOLERANCE}",
                result.residual_norm
            ));
        }
        if boundary_error > 1.0e-10 {
            return Err(format!(
                "{end_condition:?}: boundary mode displacement {boundary_error}"
            ));
        }

        max_relative_error = max_relative_error.max(relative_error);
        max_residual_norm = max_residual_norm.max(result.residual_norm);
        max_boundary_error = max_boundary_error.max(boundary_error);
    }

    Ok(Metrics {
        cases: 2.0,
        max_relative_error,
        max_residual_norm,
        max_boundary_error,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("cases", metrics.cases)
                .metric("max_relative_error", metrics.max_relative_error)
                .metric("max_residual_norm", metrics.max_residual_norm)
                .metric("max_boundary_error", metrics.max_boundary_error)
                .minimum("cases", 2.0)
                .maximum("max_relative_error", RELATIVE_TOLERANCE)
                .maximum("max_residual_norm", RESIDUAL_TOLERANCE)
        .maximum("max_boundary_error", 1.0e-10)
        .reference("independent Euler load pi^2*EI/(K*L)^2 plus residual and constrained-mode checks")
        .reference_checksum(
            "analytic/deterministic-fixture:structural-linear-euler-buckling-v1",
        )
        .dimension("2-D beam, 8 elements, pinned-pinned and cantilever end conditions")
        .input_checksum("structural-beam:elements=8:pinned,cantilever:cpu")
        .formulation("assembled Euler-Bernoulli geometric-stiffness eigenproblem")
        .tolerance("relative Euler-load error <= 5e-2; residual <= 1e-8; boundary error <= 1e-10")
                .seed("structural-linear-euler-buckling-reference-v1:elements=8:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL BUCKLING REFERENCE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
