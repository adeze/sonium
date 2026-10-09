//! Independent Taylor/replay contract evidence for differentiable solvers.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::{
    SolverError, check_taylor_remainder, eigen_value_contract, verify_forward_replay,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.differentiable-taylor-replay.v1";

fn run() -> Result<(f64, f64, f64, f64), String> {
    let real = check_taylor_remainder(
        &[1.0_f64, -0.5],
        &[0.75_f64, 0.25],
        &[0.2, 0.1, 0.05],
        |point| Ok(vec![point[0] * point[0] + 2.0 * point[1]]),
        |base, direction| Ok(vec![2.0 * base[0] * direction[0] + 2.0 * direction[1]]),
    )
    .map_err(|error| format!("real Taylor contract failed: {error}"))?;
    let complex = check_taylor_remainder(
        &[Complex64::new(1.0, -0.25)],
        &[Complex64::new(0.5, 0.75)],
        &[0.2, 0.1, 0.05],
        |point| Ok(vec![point[0] * point[0]]),
        |base, direction| Ok(vec![Complex64::new(2.0, 0.0) * base[0] * direction[0]]),
    )
    .map_err(|error| format!("complex Taylor contract failed: {error}"))?;
    real.require_order(1.9)
        .map_err(|error| format!("real Taylor order failed: {error}"))?;
    complex
        .require_order(1.9)
        .map_err(|error| format!("complex Taylor order failed: {error}"))?;
    let exact_affine = check_taylor_remainder(
        &[1.0_f64, -0.5],
        &[1.0_f64, 0.0],
        &[0.5, 0.25, 0.125],
        |point| Ok(vec![point[0]]),
        |_base, direction| Ok(vec![direction[0]]),
    )
    .map_err(|error| format!("exact affine Taylor contract failed: {error}"))?;
    exact_affine
        .require_order(10.0)
        .map_err(|error| format!("exact affine Taylor order failed: {error}"))?;
    if !exact_affine
        .observed_orders
        .iter()
        .all(|order| order.is_infinite() && *order > 0.0)
    {
        return Err("exact affine Taylor order was not positive infinity".into());
    }

    let deterministic = eigen_value_contract(
        |input: &[Complex64]| Ok(input[0] * input[0]),
        |forward: &Complex64, _seed: &[Complex64]| Ok(*forward),
    );
    let replay = verify_forward_replay(
        &deterministic,
        &[Complex64::new(0.5, -0.25)],
        |first, second| first == second,
    )
    .map_err(|error| format!("deterministic replay failed: {error}"))?;
    if !replay.deterministic || replay.evaluations != 2 {
        return Err("replay report did not record two deterministic evaluations".into());
    }

    let calls = std::cell::Cell::new(0_u32);
    let nondeterministic = eigen_value_contract(
        move |_input: &[f64]| {
            calls.set(calls.get() + 1);
            Ok(f64::from(calls.get()))
        },
        |forward: &f64, _seed: &[f64]| Ok(*forward),
    );
    let error = verify_forward_replay(&nondeterministic, &[0.0], |first, second| first == second)
        .expect_err("nondeterministic replay must fail");
    if !matches!(
        error,
        SolverError::OperationFailed {
            operation: "differentiable replay",
            ..
        }
    ) {
        return Err(format!("unexpected replay error: {error}"));
    }

    Ok((
        real.minimum_observed_order().unwrap_or(0.0),
        complex.minimum_observed_order().unwrap_or(0.0),
        f64::from(replay.evaluations as u32),
        exact_affine
            .observed_orders
            .iter()
            .filter(|order| order.is_infinite() && **order > 0.0)
            .count() as f64,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((real_order, complex_order, replay_evaluations, exact_affine_infinite_pairs)) => {
            let record = QaResult::passed(CASE_ID)
                .metric("real_minimum_observed_order", real_order)
                .metric("complex_minimum_observed_order", complex_order)
                .metric("replay_evaluations", replay_evaluations)
                .metric(
                    "exact_affine_positive_infinite_pairs",
                    exact_affine_infinite_pairs,
                )
                .minimum("real_minimum_observed_order", 1.9)
                .minimum("complex_minimum_observed_order", 1.9)
                .minimum("replay_evaluations", 2.0)
                .minimum("exact_affine_positive_infinite_pairs", 2.0)
                .reference("analytic quadratic and exact-affine Taylor remainder and deterministic replay")
                .input_checksum("analytic/deterministic-fixture:differentiable-taylor-replay-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:differentiable-taylor-replay-v1",
                )
                .dimension("real 2-vector and Complex64 scalar callbacks")
                .tolerance(
                    "real/complex observed order >= 1.9; exact-affine order = +infinity; replay evaluations = 2",
                )
                .seed("solvers-differentiable-taylor-replay:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            emit(record, true);
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .reference(
                    "analytic quadratic and exact-affine Taylor remainder and deterministic replay",
                )
                .input_checksum("analytic/deterministic-fixture:differentiable-taylor-replay-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:differentiable-taylor-replay-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            eprintln!("FAILED: {error}");
            emit(record, false);
        }
    }
}

fn emit(record: QaResult, success: bool) {
    match record.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("QA_RESULT serialization failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
    if success {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
