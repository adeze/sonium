//! Bounded CPU contract for fail-closed high-level BEM field evaluation.

use sonium_bem::core::postprocess::pressure::FieldEvaluationError;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.checked-field-evaluation.v1";

fn run() -> Result<(usize, bool, bool, bool, bool), String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let solution = solver.solve(&problem).map_err(|error| error.to_string())?;
    let points = [[0.0, 0.0, 0.24], [0.12, 0.04, 0.22], [-0.16, 0.05, 0.28]];
    let fields = solution
        .evaluate_pressure_field_checked(&points)
        .map_err(|error| error.to_string())?;
    if fields.len() != points.len()
        || fields
            .iter()
            .any(|field| !field.p_total.re.is_finite() || !field.p_total.im.is_finite())
    {
        return Err("checked field returned non-finite or incomplete values".to_string());
    }
    let rhs = solution.surface_pressure.clone();
    let loads = solver
        .solve_driver_loads(&problem, std::slice::from_ref(&rhs))
        .map_err(|error| error.to_string())?;
    let adjoints = solver
        .solve_adjoint_many(&problem, std::slice::from_ref(&rhs))
        .map_err(|error| error.to_string())?;
    if loads.len() != 1 || adjoints.len() != 1 {
        return Err("checked load/adjoint solve count mismatch".to_string());
    }
    let load_fields = loads[0]
        .evaluate_scattered_pressure_field_checked(&points)
        .map_err(|error| error.to_string())?;
    let adjoint_fields = adjoints[0]
        .evaluate_scattered_pressure_field_checked(&points)
        .map_err(|error| error.to_string())?;
    let load_field_finite = load_fields.len() == points.len()
        && load_fields
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite());
    let adjoint_field_finite = adjoint_fields.len() == points.len()
        && adjoint_fields
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite());

    let invalid_point_rejected = matches!(
        solution.evaluate_pressure_field_checked(&[[f64::NAN, 0.0, 0.24]]),
        Err(FieldEvaluationError::NonFiniteEvaluationPoint { point: 0 })
    );
    let mut malformed = solution.clone();
    malformed.surface_pressure.pop();
    let malformed_buffer_rejected = matches!(
        malformed.evaluate_pressure_field_checked(&[[0.0, 0.0, 0.24]]),
        Err(FieldEvaluationError::SurfacePressureCount { .. })
    );
    Ok((
        fields.len(),
        invalid_point_rejected,
        malformed_buffer_rejected,
        load_field_finite,
        adjoint_field_finite,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((
            point_count,
            invalid_point_rejected,
            malformed_buffer_rejected,
            load_field_finite,
            adjoint_field_finite,
        )) if invalid_point_rejected
            && malformed_buffer_rejected
            && load_field_finite
            && adjoint_field_finite =>
        {
            QaResult::passed(CASE_ID)
                .metric("point_count", point_count as f64)
                .metric("invalid_point_rejected", 1.0)
                .metric("malformed_buffer_rejected", 1.0)
                .metric("load_field_finite", 1.0)
                .metric("adjoint_field_finite", 1.0)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .input_checksum("analytic/deterministic-fixture:bem-checked-field-evaluation-v1")
                    .reference_checksum(
                        "analytic/deterministic-fixture:bem-checked-field-evaluation-v1",
                    )
                    .dimension("3-D constant-panel dense checked field evaluation")
                    .tolerance("finite fields; non-finite points and malformed buffers reject")
                    .seed("bem-checked-field-evaluation:cpu:v1")
                    .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
            }
        Ok((
            point_count,
            invalid_point_rejected,
            malformed_buffer_rejected,
            load_field_finite,
            adjoint_field_finite,
        )) => {
            QaResult::failed(
                CASE_ID,
                format!(
                    "checked-field contract failed: points={point_count}, invalid_point={invalid_point_rejected}, malformed_buffer={malformed_buffer_rejected}, load_field={load_field_finite}, adjoint_field={adjoint_field_finite}"
                ),
            )
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
