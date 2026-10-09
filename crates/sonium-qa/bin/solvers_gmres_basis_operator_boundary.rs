//! Independent CPU QA for checked active-basis GMRES callback boundaries.

use sonium_qa::QaResult;
use sonium_solvers::{
    CsrMatrix, GmresBasisCheckpoint, GmresBasisRun, GmresConfig, IdentityPreconditioner,
    LinearOperator, SolverError,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

const CASE_ID: &str = "solvers.gmres-basis-operator-boundary.v1";
const EXPECTED_CALLS: usize = 4;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, 5.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 5.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 5.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 5.0),
        ],
    )
}

struct LateNanOperator {
    matrix: CsrMatrix<f64>,
    calls: Arc<AtomicUsize>,
}

impl LinearOperator<f64> for LateNanOperator {
    fn num_rows(&self) -> usize {
        self.matrix.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.matrix.num_cols()
    }

    fn apply_into(&self, input: &[f64], output: &mut [f64]) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == EXPECTED_CALLS {
            output[0] = f64::NAN;
        } else {
            self.matrix.apply_into(input, output);
        }
    }

    fn apply_transpose_into(&self, input: &[f64], output: &mut [f64]) {
        self.matrix.apply_transpose_into(input, output);
    }
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let calls = Arc::new(AtomicUsize::new(0));
    let operator = LateNanOperator {
        matrix: matrix(),
        calls: Arc::clone(&calls),
    };
    let preconditioner = IdentityPreconditioner;
    let rhs = [1.0, 2.0, -1.0, 3.0];
    let config = GmresConfig {
        max_iterations: 2,
        restart: 4,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    };
    let captured = GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config, 1)
        .map_err(|error| format!("basis capture failed: {error:?}"))?;
    let checkpoint = match captured {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => return Err("fixture converged before checkpoint".into()),
    };
    let error = checkpoint
        .resume(&operator, &preconditioner, &rhs, &config)
        .expect_err("late operator output must fail the checked boundary");
    let typed_boundary = matches!(
        error,
        SolverError::OperationFailed {
            operation: "GMRES basis replay operator",
            ref reason,
        } if reason.contains("NaN or infinity")
    );
    let call_count = calls.load(Ordering::SeqCst);
    if !typed_boundary || call_count != EXPECTED_CALLS {
        return Err(format!(
            "unexpected callback boundary: typed={typed_boundary}, calls={call_count}, error={error:?}"
        ));
    }
    Ok(vec![
        ("basis_operator_boundary_typed_error", 1.0),
        ("basis_operator_boundary_calls", call_count as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic 4x4 SPD GMRES basis replay with injected final NaN")
                .input_checksum("analytic/deterministic-fixture:gmres-basis-operator-boundary-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:gmres-basis-operator-boundary-v1",
                )
                .dimension("4x4 CSR; checkpoint after 1 step; final callback is the 4th apply")
                .tolerance("typed callback failure and exact final-residual call count")
                .seed("solvers-gmres-basis-operator-boundary:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("analytic 4x4 SPD GMRES basis replay with injected final NaN")
                .input_checksum("analytic/deterministic-fixture:gmres-basis-operator-boundary-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:gmres-basis-operator-boundary-v1",
                )
                .tolerance("typed callback failure and exact final-residual call count")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
