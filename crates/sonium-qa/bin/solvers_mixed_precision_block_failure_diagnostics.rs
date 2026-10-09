//! Checked mixed-precision block-FGMRES failure provenance.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::mixed_precision::try_mixed_precision_block_solve;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::{LinearOperator, Preconditioner};
use sonium_solvers::{BlockFgmresConfig, SolverError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-block-failure-diagnostics.v1";
const SEED: &str = "solvers-mixed-precision-block-failure-diagnostics:fixture-v1";

struct NonFinitePreconditioner;

impl Preconditioner<f32> for NonFinitePreconditioner {
    fn apply(&self, r: &[f32]) -> Vec<f32> {
        vec![f32::NAN; r.len()]
    }
}

struct CountingOperator {
    inner: CsrMatrix<f64>,
    calls: AtomicUsize,
}

impl LinearOperator<f64> for CountingOperator {
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_cols()
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_into(x, y);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_transpose_into(x, y);
    }

    fn apply_block(&self, x: &DenseMatrix<f64>, y: &mut DenseMatrix<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_block(x, y);
    }
}

fn run() -> Result<(f64, f64), String> {
    let operator = CountingOperator {
        inner: CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0_f64, -1.0, -1.0, 4.0],
        ),
        calls: AtomicUsize::new(0),
    };
    let rhs = DenseMatrix::from_row_major_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]);
    let config = BlockFgmresConfig {
        max_iterations: 8,
        restart: 2,
        tolerance: 1.0e-10,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };
    let error = try_mixed_precision_block_solve(
        &operator,
        &rhs,
        Box::new(NonFinitePreconditioner),
        &config,
    )
    .expect_err("non-finite low-precision block application must fail closed");
    match error {
        SolverError::OperationFailed { operation, reason }
            if operation == "mixed-precision block FGMRES preconditioner"
                && reason.contains("non-finite intermediate value")
                && reason.contains("1 failures") => {}
        other => return Err(format!("unexpected mixed-precision block error: {other:?}")),
    }
    Ok((1.0, operator.calls.load(Ordering::Relaxed) as f64))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((failure_count, operator_calls)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed(SEED)
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-failure-diagnostics-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-failure-diagnostics-v1",
                )
                .reference("independent two-RHS low-precision non-finite preconditioner injection")
                .metric("low_precision_failure_count", failure_count)
                .minimum("low_precision_failure_count", 1.0)
                .metric("operator_calls_before_failure", operator_calls)
                .maximum("operator_calls_before_failure", 2.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed(SEED)
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-failure-diagnostics-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-failure-diagnostics-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
