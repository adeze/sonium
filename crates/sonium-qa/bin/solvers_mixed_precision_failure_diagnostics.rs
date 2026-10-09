//! Checked CPU evidence for mixed-precision precision-boundary diagnostics.

use sonium_qa::QaResult;
use sonium_solvers::Preconditioner;
use sonium_solvers::mixed_precision::MixedPrecisionPreconditioner;
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-failure-diagnostics.v1";

fn run() -> Result<f64, String> {
    let low = DiagonalPreconditioner::<f32>::from_inverse_diagonal(vec![1.0]);
    let preconditioner = MixedPrecisionPreconditioner::<f64, f32>::new(Box::new(low));
    let mut output = vec![0.0_f64];

    preconditioner.apply_into(&[f64::NAN], &mut output);
    preconditioner.apply_into(&[f64::INFINITY], &mut output);

    let failures = preconditioner.low_precision_failure_count() as f64;
    if failures != 2.0 {
        return Err(format!(
            "expected two precision-boundary failures, got {failures}"
        ));
    }
    Ok(failures)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(failures) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("low_precision_failure_count", failures)
                .minimum("low_precision_failure_count", 2.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", record.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("MIXED-PRECISION DIAGNOSTICS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
