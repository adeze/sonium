//! Independent CPU evidence for recycled-GMRES checkpoint continuation.

use sonium_qa::QaResult;
use sonium_solvers::{GmresConfig, RecycledSubspace, try_gmres_recycled};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.recycled-resume.v1";
const TOLERANCE: f64 = 1.0e-10;

fn operator() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.5),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 5.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 5.5),
        ],
    )
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 80,
        restart: 6,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    matrix
        .matvec(solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(String, f64)>, String> {
    let matrix = operator();
    let expected = [1.0, -0.5, 0.75, 2.0];
    let rhs = matrix.matvec(&expected);
    let mut source = RecycledSubspace::new(2);
    let first = try_gmres_recycled(&matrix, &rhs, None, &mut source, &config())
        .map_err(|error| format!("initial recycled solve failed: {error}"))?;
    if !first.converged {
        return Err("initial recycled solve did not converge".into());
    }
    let bytes = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("checkpoint serialization failed: {error}"))?;
    let mut restored = RecycledSubspace::<f64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("checkpoint restore failed: {error}"))?;
    let resumed = restored
        .resume(&matrix, &rhs, Some(&expected), &config())
        .map_err(|error| format!("checkpoint continuation failed: {error}"))?;
    if !resumed.converged || resumed.iterations != 0 {
        return Err(format!(
            "exact continuation was not accepted: converged={} iterations={}",
            resumed.converged, resumed.iterations
        ));
    }
    let malformed_rejected = restored
        .resume(&matrix, &rhs, Some(&[0.0, 0.0, 0.0]), &config())
        .is_err();
    if !malformed_rejected {
        return Err("malformed initial-guess dimensions were accepted".into());
    }
    Ok(vec![
        (
            "initial_solution_error".into(),
            max_error(&first.x, &expected),
        ),
        (
            "initial_independent_residual".into(),
            residual(&matrix, &first.x, &rhs),
        ),
        (
            "resumed_solution_error".into(),
            max_error(&resumed.x, &expected),
        ),
        (
            "resumed_independent_residual".into(),
            residual(&matrix, &resumed.x, &rhs),
        ),
        ("resumed_iterations".into(), resumed.iterations as f64),
        ("continuation_contract".into(), 1.0),
        (
            "malformed_guess_rejection".into(),
            if malformed_rejected { 1.0 } else { 0.0 },
        ),
        ("checkpoint_bytes".into(), bytes.len() as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:solvers-recycled-resume-v1")
                .reference_checksum("analytic/deterministic-fixture:solvers-recycled-resume-v1");
            for (name, value) in metrics {
                let bounded = name.contains("error") || name.contains("residual");
                result = result.metric(name.clone(), value);
                if bounded {
                    result = result.maximum(name, TOLERANCE);
                }
            }
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .input_checksum("analytic/deterministic-fixture:solvers-recycled-resume-v1")
                .reference_checksum("analytic/deterministic-fixture:solvers-recycled-resume-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
