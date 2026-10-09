//! Generated callback-output boundary for the public optimization line searches.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{
    BacktrackingConfig, HagerZhangConfig, StrongWolfeConfig, backtracking, hager_zhang_line_search,
    strong_wolfe,
};
use sonium_solvers::{AccuracyStatus, SolverStatus};
use std::cell::Cell;
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-line-search-generated-output.v1";
const CHECKS: f64 = 3.0;

fn run() -> Result<(), String> {
    let objective_calls = Cell::new(0usize);
    let objective = |_: &[f64]| {
        let call = objective_calls.get();
        objective_calls.set(call + 1);
        if call == 0 { f64::NAN } else { 0.0 }
    };
    let result = backtracking(
        &objective,
        &[1.0],
        &[-1.0],
        1.0,
        &[1.0],
        &BacktrackingConfig::default(),
    );
    assert_numerical(&result, "backtracking objective")?;
    if objective_calls.get() != 1 {
        return Err(format!(
            "backtracking evaluated {} callbacks after a non-finite objective",
            objective_calls.get()
        ));
    }

    let objective_calls = Cell::new(0usize);
    let gradient_calls = Cell::new(0usize);
    let objective = |_: &[f64]| {
        let call = objective_calls.get();
        objective_calls.set(call + 1);
        if call == 0 { f64::INFINITY } else { 0.0 }
    };
    let gradient = |_: &[f64]| {
        gradient_calls.set(gradient_calls.get() + 1);
        vec![1.0]
    };
    let result = strong_wolfe(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0],
        1.0,
        &[1.0],
        &StrongWolfeConfig::default(),
    );
    assert_numerical(&result, "strong-Wolfe objective")?;
    if objective_calls.get() != 1 || gradient_calls.get() != 0 {
        return Err(format!(
            "strong-Wolfe callback counts were objective={} gradient={}",
            objective_calls.get(),
            gradient_calls.get()
        ));
    }

    let gradient_calls = Cell::new(0usize);
    let objective = |_: &[f64]| 0.0;
    let gradient = |_: &[f64]| {
        let call = gradient_calls.get();
        gradient_calls.set(call + 1);
        if call == 0 {
            vec![f64::NAN]
        } else {
            vec![-1.0]
        }
    };
    let result = hager_zhang_line_search(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0],
        1.0,
        &[1.0],
        &HagerZhangConfig::default(),
    );
    assert_numerical(&result, "Hager-Zhang gradient")?;
    if gradient_calls.get() != 1 {
        return Err(format!(
            "Hager-Zhang evaluated {} gradients after a non-finite output",
            gradient_calls.get()
        ));
    }

    Ok(())
}

fn assert_numerical<T: sonium_solvers::RealScalar>(
    result: &sonium_solvers::optimization::LineSearchResult<T>,
    label: &str,
) -> Result<(), String> {
    if result.status() != SolverStatus::NumericalError
        || result.accuracy_status() != AccuracyStatus::NotChecked
    {
        return Err(format!(
            "{label} returned status {:?} and accuracy {:?}",
            result.status(),
            result.accuracy_status()
        ));
    }
    Ok(())
}

fn main() {
    let started = Instant::now();
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(())) => QaResult::passed(CASE_ID)
            .metric("generated_output_checks", CHECKS)
            .metric("backtracking_checks", 1.0)
            .metric("strong_wolfe_checks", 1.0)
            .metric("hager_zhang_checks", 1.0)
            .input_checksum("analytic/deterministic-fixture:solvers-optimization-line-search-generated-output-v1")
            .reference("canonical SolverStatus::NumericalError and AccuracyStatus::NotChecked")
            .reference_checksum("analytic")
            .dimension("f64; one-dimensional trial callbacks; three public line-search families")
            .formulation("generated non-finite objective/gradient outputs fail before a later trial can be accepted")
            .tolerance("3/3 generated-output cases report NumericalError with immediate callback termination")
            .seed("solvers-optimization-line-search-generated-output:cpu:v1")
            .timing_ms("line_search_generated_output", started.elapsed().as_millis() as u64),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message),
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "optimization line-search generated-output QA panicked".to_owned()
            };
            QaResult::failed(CASE_ID, message)
        }
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);

    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA result serializes")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
