//! Independent CPU evidence for transactional GCRO-DR recycled state.

use sonium_qa::QaResult;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::Preconditioner;
use sonium_solvers::{
    GcroDrCheckpoint, GcroDrConfig, GcroDrSubspace, SolverError, try_gcro_dr,
    try_gcro_dr_preconditioned,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const CASE_ID: &str = "solvers.gcro-dr-transactional.v1";

struct FailingPreconditioner {
    calls: AtomicUsize,
    fail_on: usize,
}

impl Preconditioner<f64> for FailingPreconditioner {
    fn apply_into(&self, input: &[f64], output: &mut [f64]) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        assert_ne!(call, self.fail_on, "injected GCRO-DR callback failure");
        output.copy_from_slice(input);
    }
}

fn config() -> GcroDrConfig<f64> {
    GcroDrConfig {
        max_iterations: 4,
        restart: 2,
        tolerance: 1.0e-10,
        print_interval: 0,
    }
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let operator: CsrMatrix<f64> = CsrMatrix::identity(2);
    let mut committed = GcroDrSubspace::new(1);
    let solution = try_gcro_dr(&operator, &[1.0, 2.0], None, &mut committed, &config())
        .map_err(|error| format!("seed GCRO-DR solve failed: {error:?}"))?;
    if !solution.converged || committed.num_vectors() != 1 {
        return Err(format!(
            "successful checked solve did not commit recycled state: status={:?}, vectors={}",
            solution.status,
            committed.num_vectors()
        ));
    }
    let before = GcroDrCheckpoint::from_subspace(&committed)
        .map_err(|error| format!("pre-failure checkpoint capture failed: {error:?}"))?
        .to_checkpoint_bytes()
        .map_err(|error| format!("pre-failure checkpoint serialization failed: {error:?}"))?;

    // A zero operator makes the existing recycled vector rank-deficient in
    // C/R construction, forcing the working state to clear before the later
    // injected preconditioner callback failure.
    let zero = CsrMatrix::from_triplets(2, 2, Vec::new());
    let preconditioner = FailingPreconditioner {
        calls: AtomicUsize::new(0),
        fail_on: 3,
    };
    let mut after_failure = committed.clone();
    let failure = try_gcro_dr_preconditioned(
        &zero,
        &preconditioner,
        &[1.0, 2.0],
        None,
        &mut after_failure,
        &config(),
    )
    .expect_err("injected callback failure must be typed");
    if !matches!(failure, SolverError::OperationFailed { .. }) {
        return Err(format!("unexpected transactional failure: {failure:?}"));
    }
    let after = GcroDrCheckpoint::from_subspace(&after_failure)
        .map_err(|error| format!("post-failure checkpoint capture failed: {error:?}"))?
        .to_checkpoint_bytes()
        .map_err(|error| format!("post-failure checkpoint serialization failed: {error:?}"))?;
    let state_unchanged = (before == after) as u8 as f64;
    if state_unchanged < 1.0 {
        return Err("caller recycled state changed after typed callback failure".into());
    }

    Ok(vec![
        ("committed_vectors", committed.num_vectors() as f64),
        ("failure_typed", 1.0),
        ("state_unchanged", state_unchanged),
        ("seed_iterations", solution.iterations as f64),
        (
            "failure_callback_count",
            preconditioner.calls.load(Ordering::SeqCst) as f64,
        ),
    ])
}

fn main() {
    let started = Instant::now();
    let previous_panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = run();
    std::panic::set_hook(previous_panic_hook);
    match outcome {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent identity/zero-operator transactional GCRO-DR fixture")
                .input_checksum("analytic/deterministic-fixture:gcro-dr-transactional-v1")
                .reference_checksum("analytic/deterministic-fixture:gcro-dr-transactional-v1")
                .dimension("2x2 CSR; checked GCRO-DR target=1, restart=2")
                .tolerance("typed callback failure preserves serialized recycled state exactly")
                .seed("solvers-gcro-dr-transactional:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = match name {
                    "failure_typed" | "state_unchanged" => {
                        result.minimum(name, 1.0).metric(name, value)
                    }
                    _ => result.metric(name, value),
                };
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent identity/zero-operator transactional GCRO-DR fixture")
                .input_checksum("analytic/deterministic-fixture:gcro-dr-transactional-v1")
                .reference_checksum("analytic/deterministic-fixture:gcro-dr-transactional-v1")
                .tolerance("typed callback failure preserves serialized recycled state exactly")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
