//! Runtime and coarse-solve diagnostics for composed preconditioners.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{
    AdditiveSchwarzPreconditioner, BlockJacobiPreconditioner, IlutConfig, SchwarzCoarseSetup,
};
use sonium_solvers::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-runtime-diagnostics.v1";
const SEED: &str = "solvers-preconditioner-runtime-diagnostics:fixture-v1";

fn run() -> Result<QaResult, String> {
    let matrix = CsrMatrix::<f64>::identity(4);
    let rhs = [1.0, -2.0, 3.0, -4.0];

    let block = BlockJacobiPreconditioner::from_csr(&matrix, 2, IlutConfig::default());
    block
        .try_apply(&rhs)
        .map_err(|error| format!("Block-Jacobi forward apply failed: {error}"))?;
    let _ = block.apply_transpose(&rhs);
    let _ = block.apply_hermitian(&rhs);
    if block.try_apply(&[f64::NAN; 4]).is_ok() {
        return Err("Block-Jacobi accepted a non-finite checked input".into());
    }
    let block_runtime = block.runtime_diagnostics();
    if block_runtime.apply_calls != 1
        || block_runtime.transpose_calls != 1
        || block_runtime.hermitian_calls != 1
        || block_runtime.failed_applications != 1
        || block_runtime.non_finite_inputs != 1
    {
        return Err(format!(
            "unexpected Block-Jacobi runtime diagnostics: {block_runtime:?}"
        ));
    }

    let partitions = vec![vec![0, 1], vec![2, 3]];
    let schwarz = AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
        .with_nicolaides_coarse_space(&matrix);
    if schwarz.coarse_setup() != SchwarzCoarseSetup::Attached || schwarz.coarse_dimension() != 2 {
        return Err(format!(
            "unexpected Schwarz coarse setup: {:?}, dimension {}",
            schwarz.coarse_setup(),
            schwarz.coarse_dimension()
        ));
    }
    schwarz
        .try_apply(&rhs)
        .map_err(|error| format!("Schwarz forward apply failed: {error}"))?;
    let _ = schwarz.apply_transpose(&rhs);
    let _ = schwarz.apply_hermitian(&rhs);
    if schwarz.try_apply(&[f64::INFINITY; 4]).is_ok() {
        return Err("Schwarz accepted a non-finite checked input".into());
    }
    let schwarz_runtime = schwarz.runtime_diagnostics();
    if schwarz_runtime.apply_calls != 1
        || schwarz_runtime.transpose_calls != 1
        || schwarz_runtime.hermitian_calls != 1
        || schwarz_runtime.coarse_correction_calls != 3
        || schwarz_runtime.coarse_solve_failures != 0
        || schwarz_runtime.failed_applications != 1
        || schwarz_runtime.non_finite_inputs != 1
        || schwarz_runtime.coarse_setup != SchwarzCoarseSetup::Attached
        || schwarz_runtime.coarse_dimension != 2
    {
        return Err(format!(
            "unexpected Schwarz runtime diagnostics: {schwarz_runtime:?}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .seed(SEED)
        .input_checksum("analytic/deterministic-fixture:preconditioner-runtime-diagnostics-v1")
        .reference_checksum("analytic/deterministic-fixture:preconditioner-runtime-diagnostics-v1")
        .reference("checked forward/adjoint and two-level coarse runtime diagnostics")
        .metric("block_apply_calls", block_runtime.apply_calls as f64)
        .metric(
            "block_transpose_calls",
            block_runtime.transpose_calls as f64,
        )
        .metric(
            "block_hermitian_calls",
            block_runtime.hermitian_calls as f64,
        )
        .metric(
            "block_failed_applications",
            block_runtime.failed_applications as f64,
        )
        .metric("schwarz_apply_calls", schwarz_runtime.apply_calls as f64)
        .metric(
            "schwarz_transpose_calls",
            schwarz_runtime.transpose_calls as f64,
        )
        .metric(
            "schwarz_hermitian_calls",
            schwarz_runtime.hermitian_calls as f64,
        )
        .metric(
            "schwarz_coarse_correction_calls",
            schwarz_runtime.coarse_correction_calls as f64,
        )
        .metric(
            "schwarz_coarse_solve_failures",
            schwarz_runtime.coarse_solve_failures as f64,
        )
        .minimum("block_apply_calls", 1.0)
        .minimum("block_transpose_calls", 1.0)
        .minimum("block_hermitian_calls", 1.0)
        .minimum("schwarz_coarse_correction_calls", 3.0))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(result) => {
            let result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .seed(SEED)
                .input_checksum(
                    "analytic/deterministic-fixture:preconditioner-runtime-diagnostics-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:preconditioner-runtime-diagnostics-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
