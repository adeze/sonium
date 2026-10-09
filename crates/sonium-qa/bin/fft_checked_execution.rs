//! Release-facing checked native FFT buffer and scratch admission gate.

use num_complex::Complex64;
use sonium_fft::{FftDirection, FftPlanError, FftPlanner};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fft.checked-native-execution.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let planner = FftPlanner::<f64>::new();
    let plan = planner.plan(16, FftDirection::Forward);

    let mut malformed = vec![Complex64::new(0.0, 0.0); 17];
    let shape_rejections = match plan.try_process(&mut malformed) {
        Err(FftPlanError::InvalidBufferLength {
            plan_length: 16,
            buffer_length: 17,
        }) => 1.0,
        other => return Err(format!("unexpected shape result: {other:?}")),
    };

    let scratch_rejections = if plan.scratch_len() == 0 {
        1.0
    } else {
        let mut buffer = vec![Complex64::new(0.0, 0.0); 16];
        let mut scratch = Vec::new();
        match plan.try_process_with_scratch(&mut buffer, &mut scratch) {
            Err(FftPlanError::ScratchTooShort {
                required,
                actual: 0,
            }) if required == plan.scratch_len() => 1.0,
            other => return Err(format!("unexpected scratch result: {other:?}")),
        }
    };

    let mut buffer = vec![Complex64::new(0.0, 0.0); 16];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
    plan.try_process_with_scratch(&mut buffer, &mut scratch)
        .map_err(|error| format!("valid checked execution failed: {error}"))?;
    if buffer
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("valid checked execution produced non-finite output".into());
    }

    Ok((shape_rejections, scratch_rejections, buffer.len() as f64))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((shape_rejections, scratch_rejections, valid_samples)) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("shape_rejections", shape_rejections)
                .metric("scratch_rejections", scratch_rejections)
                .metric("valid_samples", valid_samples)
                .minimum("shape_rejections", 1.0)
                .minimum("scratch_rejections", 1.0)
                .minimum("valid_samples", 16.0)
                .input_checksum("analytic/deterministic-fixture:fft-checked-execution-v1")
                .reference("checked native FftPlan buffer and scratch admission")
                .reference_checksum("analytic/deterministic-fixture:fft-checked-execution-v1")
                .dimension("1D complex CPU FFT, length 16")
                .formulation("native FftPlan::try_process and try_process_with_scratch")
                .tolerance("typed malformed-shape and short-scratch rejection; finite valid output")
                .seed("fft-checked-execution:length-16:v1")
                .elapsed_ms(elapsed_ms);
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
            eprintln!("FFT CHECKED EXECUTION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
