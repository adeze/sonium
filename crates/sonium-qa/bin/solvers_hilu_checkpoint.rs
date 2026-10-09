//! Independent CPU evidence for hierarchical-ILU checkpoint persistence.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{HiluConfig, HiluPreconditioner};
use sonium_solvers::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.hilu-checkpoint.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:hilu-checkpoint-v1";
const TOLERANCE: f64 = 1.0e-11;

fn matrix() -> CsrMatrix<f64> {
    let n = 20;
    let mut row_ptrs = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        if row > 0 {
            col_indices.push(row - 1);
            values.push(0.2);
        }
        col_indices.push(row);
        values.push(4.0);
        if row + 1 < n {
            col_indices.push(row + 1);
            values.push(0.2);
        }
        row_ptrs.push(values.len());
    }
    CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
}

fn max_error(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64, f64), String> {
    let source = HiluPreconditioner::try_from_csr_coupled_with_config(
        &matrix(),
        &HiluConfig {
            min_block_size: 4,
            max_levels: Some(3),
        },
        0.0,
    )
    .map_err(|error| format!("HILU setup failed: {error:?}"))?;
    let checkpoint = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("HILU checkpoint serialization failed: {error:?}"))?;
    let restored = HiluPreconditioner::<f64>::from_checkpoint_bytes(&checkpoint)
        .map_err(|error| format!("HILU checkpoint restore failed: {error:?}"))?;
    let rhs: Vec<_> = (0..20).map(|index| 1.0 + index as f64 * 0.03).collect();
    let forward_error = max_error(&source.apply(&rhs), &restored.apply(&rhs));
    let transpose_error = max_error(
        &source.apply_transpose(&rhs),
        &restored.apply_transpose(&rhs),
    );
    let hermitian_error = max_error(
        &source.apply_hermitian(&rhs),
        &restored.apply_hermitian(&rhs),
    );

    let mut malformed_rejections = 0.0;
    let mut bad_magic = checkpoint.clone();
    bad_magic[0] ^= 1;
    malformed_rejections +=
        f64::from(HiluPreconditioner::<f64>::from_checkpoint_bytes(&bad_magic).is_err());
    malformed_rejections += f64::from(
        HiluPreconditioner::<f64>::from_checkpoint_bytes(&checkpoint[..checkpoint.len() - 1])
            .is_err(),
    );
    let mut trailing = checkpoint.clone();
    trailing.push(0);
    malformed_rejections +=
        f64::from(HiluPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err());
    malformed_rejections += f64::from(
        HiluPreconditioner::<num_complex::Complex64>::from_checkpoint_bytes(&checkpoint).is_err(),
    );
    let mut bad_dimension = checkpoint;
    bad_dimension[14..22].copy_from_slice(&1_u64.to_le_bytes());
    malformed_rejections +=
        f64::from(HiluPreconditioner::<f64>::from_checkpoint_bytes(&bad_dimension).is_err());

    Ok((
        forward_error,
        transpose_error,
        hermitian_error,
        malformed_rejections,
        restored.num_blocks() as f64,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            forward_error,
            transpose_error,
            hermitian_error,
            malformed_rejections,
            block_count,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension("20x20 tridiagonal CSR ILUM-style HILU")
                .tolerance("all action errors <= 1e-11; five malformed packets rejected")
                .metric("forward_error", forward_error)
                .metric("transpose_error", transpose_error)
                .metric("hermitian_error", hermitian_error)
                .metric("malformed_rejections", malformed_rejections)
                .metric("restored_block_count", block_count)
                .maximum("forward_error", TOLERANCE)
                .maximum("transpose_error", TOLERANCE)
                .maximum("hermitian_error", TOLERANCE)
                .minimum("malformed_rejections", 5.0)
                .minimum("restored_block_count", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
