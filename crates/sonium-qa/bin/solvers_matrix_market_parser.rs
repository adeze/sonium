//! Independent CPU evidence for checked Matrix Market parsing.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, GmresConfig, MatrixMarketError, SolverErrorKind, SolverFailure, SolverStatus,
    gmres, parse_complex_matrix_market, parse_matrix_market,
};
use std::io::Cursor;
use std::time::Instant;

const CASE_ID: &str = "solvers.matrix-market-parser.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:matrix-market-parser-v1";

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let symmetric = b"%%MatrixMarket matrix coordinate real symmetric\n% 3x3 SPD fixture\n3 3 5\n1 1 4\n1 2 -1\n2 2 4\n2 3 -1\n3 3 3\n";
    let parsed = parse_matrix_market(Cursor::new(symmetric))
        .map_err(|error| format!("symmetric parse failed: {error}"))?;
    if parsed.declared_entries != 5 || parsed.stored_nonzeros != 7 {
        return Err(format!(
            "symmetric metadata mismatch: declared={} stored={}",
            parsed.declared_entries, parsed.stored_nonzeros
        ));
    }
    let expected = [1.0, -2.0, 0.5];
    let rhs = parsed.matrix.matvec(&expected);
    let solution = gmres(
        &parsed.matrix,
        &rhs,
        &GmresConfig {
            max_iterations: 30,
            restart: 3,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            ..Default::default()
        },
    );
    if !solution.converged {
        return Err(format!(
            "parsed SPD solve did not converge: {}",
            solution.residual
        ));
    }
    let solution_error = solution
        .x
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    let pattern = b"%%MatrixMarket matrix coordinate pattern general\n3 3 3\n1 2\n2 3\n3 1\n";
    let pattern = parse_matrix_market(Cursor::new(pattern))
        .map_err(|error| format!("pattern parse failed: {error}"))?;
    if pattern.stored_nonzeros != 3 || pattern.matrix.matvec(&[2.0, 3.0, 5.0]) != [3.0, 5.0, 2.0] {
        return Err("pattern fixture action mismatch".into());
    }

    let malformed = b"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n";
    let malformed_rejected = f64::from(parse_matrix_market(Cursor::new(malformed)).is_err());
    if malformed_rejected != 1.0 {
        return Err("malformed Matrix Market packet was accepted".into());
    }

    let hermitian =
        b"%%MatrixMarket matrix coordinate complex hermitian\n2 2 3\n1 1 2 0\n1 2 1 2\n2 2 3 0\n";
    let complex = parse_complex_matrix_market(Cursor::new(hermitian))
        .map_err(|error| format!("complex Hermitian parse failed: {error}"))?;
    let complex_expected = vec![Complex64::new(0.0, 1.0), Complex64::new(1.0, 1.0)];
    let complex_actual = complex
        .matrix
        .matvec(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]);
    let complex_action_error = complex_actual
        .iter()
        .zip(&complex_expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    if complex.stored_nonzeros != 4 || complex_action_error > 1.0e-12 {
        return Err(format!(
            "complex Hermitian mismatch: stored={} action_error={complex_action_error:e}",
            complex.stored_nonzeros
        ));
    }
    let complex_malformed = b"%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 NaN 0\n";
    let complex_malformed_rejected =
        f64::from(parse_complex_matrix_market(Cursor::new(complex_malformed)).is_err());
    if complex_malformed_rejected != 1.0 {
        return Err("malformed complex Matrix Market input was accepted".into());
    }
    let shared_failure_contract_error = [
        (
            MatrixMarketError::Parse("malformed banner".into()),
            SolverErrorKind::OperationFailed,
        ),
        (
            MatrixMarketError::Io(std::io::Error::other("fixture")),
            SolverErrorKind::OperationFailed,
        ),
    ]
    .into_iter()
    .filter(|(error, kind)| {
        error.error_kind() == *kind
            && error.status() == SolverStatus::NumericalError
            && error.accuracy_status() == AccuracyStatus::NotChecked
    })
    .count() as f64;
    Ok(vec![
        ("symmetric_declared_entries", parsed.declared_entries as f64),
        ("symmetric_stored_nonzeros", parsed.stored_nonzeros as f64),
        ("symmetric_solution_error", solution_error),
        ("symmetric_true_residual", solution.residual),
        ("pattern_stored_nonzeros", pattern.stored_nonzeros as f64),
        ("malformed_rejected", malformed_rejected),
        ("complex_stored_nonzeros", complex.stored_nonzeros as f64),
        ("complex_action_error", complex_action_error),
        ("complex_malformed_rejected", complex_malformed_rejected),
        (
            "shared_failure_contract_error",
            (2.0 - shared_failure_contract_error).abs(),
        ),
    ])
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(metrics) => {
            let mut record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("standard and complex Matrix Market expansion, GMRES residual")
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .dimension("3x3 real and 2x2 complex coordinate fixtures")
                .formulation("checked real and complex Matrix Market parser to canonical CSR")
                .tolerance(
                    "solution/residual/action errors <= 1e-10; malformed packets rejected; shared failure-contract error <= 1e-12",
                )
                .seed("solvers-matrix-market-parser:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                record = record.metric(name, value);
            }
            record
                .minimum("symmetric_declared_entries", 5.0)
                .minimum("symmetric_stored_nonzeros", 7.0)
                .maximum("symmetric_solution_error", 1.0e-10)
                .maximum("symmetric_true_residual", 1.0e-10)
                .minimum("pattern_stored_nonzeros", 3.0)
                .minimum("malformed_rejected", 1.0)
                .minimum("complex_stored_nonzeros", 4.0)
                .maximum("complex_action_error", 1.0e-10)
                .minimum("complex_malformed_rejected", 1.0)
                .maximum("shared_failure_contract_error", 1.0e-12)
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .input_checksum(CHECKSUM)
            .reference_checksum(CHECKSUM)
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
