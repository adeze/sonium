//! Generated-residual validation for public Newton solver entry points.

use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, InexactNewtonConfig, LineSearch, NewtonConfig, NewtonError, SolverFailure,
    SolverStatus, newton_raphson, newton_raphson_continuation,
};
use std::cell::Cell;
use std::time::Instant;

const CASE_ID: &str = "solvers.nonlinear-generated-residual.v1";

fn run() -> Result<(), String> {
    let calls = Cell::new(0usize);
    let error = newton_raphson(
        &[0.0_f64],
        |_x| {
            let call = calls.get();
            calls.set(call + 1);
            if call == 0 { vec![1.0] } else { vec![f64::NAN] }
        },
        |_x, rhs| Ok(rhs.to_vec()),
        &NewtonConfig {
            max_iterations: 1,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            ..NewtonConfig::default()
        },
    )
    .expect_err("Newton must reject a non-finite final residual");
    assert_numerical_error(&error, "Newton final residual")?;

    let calls = Cell::new(0usize);
    let error = newton_raphson(
        &[0.0_f64],
        |_x| {
            let call = calls.get();
            calls.set(call + 1);
            match call {
                0 => vec![1.0],
                1 => Vec::new(),
                _ => vec![0.25],
            }
        },
        |_x, rhs| Ok(rhs.to_vec()),
        &NewtonConfig {
            max_iterations: 2,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            ..NewtonConfig::default()
        },
    )
    .expect_err("Newton must reject a malformed no-line-search trial residual");
    assert_numerical_error(&error, "Newton trial residual")?;

    let calls = Cell::new(0usize);
    let error = newton_raphson_continuation(
        &[0.0_f64],
        |_x| {
            let call = calls.get();
            calls.set(call + 1);
            match call {
                0 => vec![1.0],
                1 => vec![0.5],
                _ => vec![f64::NAN],
            }
        },
        |_x, v| v.to_vec(),
        &InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 1,
                absolute_tolerance: 0.0,
                relative_tolerance: 0.0,
                line_search: LineSearch::None,
                ..NewtonConfig::default()
            },
            ..InexactNewtonConfig::default()
        },
    )
    .expect_err("continuation must reject a non-finite final residual");
    assert_numerical_error(&error, "continuation final residual")?;

    let config = InexactNewtonConfig {
        newton: NewtonConfig {
            max_iterations: 1,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            line_search: LineSearch::None,
            ..NewtonConfig::default()
        },
        ..InexactNewtonConfig::default()
    };
    let error =
        newton_raphson_continuation(&[0.0_f64], |_x| vec![1.0], |_x, _v| Vec::new(), &config)
            .expect_err("malformed Jacobian matvec output must fail closed");
    assert_numerical_error(&error, "malformed Jacobian matvec")?;

    let error =
        newton_raphson_continuation(&[0.0_f64], |_x| vec![1.0], |_x, _v| vec![f64::NAN], &config)
            .expect_err("non-finite Jacobian matvec output must fail closed");
    assert_numerical_error(&error, "non-finite Jacobian matvec")?;

    Ok(())
}

fn assert_numerical_error(error: &NewtonError, label: &str) -> Result<(), String> {
    if !matches!(error, NewtonError::NumericalError { .. }) {
        return Err(format!(
            "{label} returned {error:?} instead of NumericalError"
        ));
    }
    if error.status() != SolverStatus::NumericalError
        || error.accuracy_status() != AccuracyStatus::NotChecked
    {
        return Err(format!("{label} reported a non-canonical failure contract"));
    }
    Ok(())
}

fn main() {
    let started = Instant::now();
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(())) => QaResult::passed(CASE_ID)
            .metric("generated_residual_checks", 5.0)
            .metric("typed_numerical_failures", 5.0)
            .metric("trial_shape_rejections", 1.0)
            .metric("jacobian_output_rejections", 2.0)
            .input_checksum(
                "analytic/deterministic-fixture:solvers-nonlinear-generated-residual-v1",
            )
            .reference("public Newton and matrix-free continuation final-residual contract")
            .reference_checksum("analytic")
            .dimension("f64 scalar residual; one Newton and one Newton-Krylov continuation path")
            .formulation("non-finite generated residuals fail closed as typed NumericalError")
            .tolerance("5/5 generated-output cases report NumericalError and NotChecked accuracy, including two Jacobian matvec failures")
            .seed("solvers-nonlinear-generated-residual:cpu:v1")
            .timing_ms(
                "generated_residual_contract",
                started.elapsed().as_millis() as u64,
            ),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message),
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "nonlinear generated-residual contract panicked".to_owned()
            };
            QaResult::failed(CASE_ID, message)
        }
    }
    .requested_backend("cpu")
    .actual_backend("cpu")
    .elapsed_ms(started.elapsed().as_millis() as u64);

    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("nonlinear generated-residual result must serialize")
    );
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
