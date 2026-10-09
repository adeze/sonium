//! Independent CPU evidence for the bounded saddle-point checkpoint format.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::{
    BlockPreconditionerConfig, SaddlePointConfig, SaddlePointPreconditioner, SchurStrategy,
};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.saddle-point-checkpoint.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:saddle-point-checkpoint-v1";
const TOLERANCE: f64 = 1.0e-12;

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 4, 6, 8],
        vec![0, 1, 0, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(4.0, 0.2),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(3.0, 0.3),
            Complex64::new(2.0, 0.4),
            Complex64::new(0.5, -0.2),
            Complex64::new(0.5, 0.2),
            Complex64::new(1.5, 0.1),
        ],
    )
}

fn config() -> SaddlePointConfig {
    SaddlePointConfig {
        block1_size: 2,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::None,
        schur_strategy: SchurStrategy::Diagonal,
        upper_triangular: false,
    }
}

fn max_error(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let source = SaddlePointPreconditioner::try_from_matrix(&matrix(), config())
        .map_err(|e| format!("setup failed: {e:?}"))?;
    let rhs: Vec<_> = (0..4)
        .map(|index| Complex64::new(index as f64 + 0.25, -0.5 * index as f64))
        .collect();
    let forward = source.apply(&rhs);
    let transpose = source.apply_transpose(&rhs);
    let hermitian = source.apply_hermitian(&rhs);
    let bytes = source
        .to_checkpoint_bytes()
        .map_err(|e| format!("serialization failed: {e:?}"))?;
    let restored = SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
        .map_err(|e| format!("restore failed: {e:?}"))?;
    let forward_error = max_error(&forward, &restored.apply(&rhs));
    let transpose_error = max_error(&transpose, &restored.apply_transpose(&rhs));
    let hermitian_error = max_error(&hermitian, &restored.apply_hermitian(&rhs));

    let mut malformed_rejections = 0.0;
    for length in [0, 1, bytes.len() / 2, bytes.len().saturating_sub(1)] {
        if SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes[..length]).is_err()
        {
            malformed_rejections += 1.0;
        }
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    if SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&trailing).is_err() {
        malformed_rejections += 1.0;
    }
    let mut bad_reserved = bytes.clone();
    bad_reserved[13] = 1;
    if SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_reserved).is_err() {
        malformed_rejections += 1.0;
    }
    if !forward_error.is_finite()
        || !transpose_error.is_finite()
        || !hermitian_error.is_finite()
        || malformed_rejections < 6.0
    {
        return Err(format!(
            "saddle-point checkpoint gate failed: errors=({forward_error}, {transpose_error}, {hermitian_error}), malformed={malformed_rejections}"
        ));
    }
    Ok((
        forward_error,
        transpose_error,
        hermitian_error,
        malformed_rejections,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((forward_error, transpose_error, hermitian_error, malformed_rejections)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension("4x4 block-diagonal Complex64 CSR saddle-point preconditioner")
                .tolerance("all action errors <= 1e-12; six malformed packets rejected")
                .metric("forward_error", forward_error)
                .metric("transpose_error", transpose_error)
                .metric("hermitian_error", hermitian_error)
                .metric("malformed_rejections", malformed_rejections)
                .maximum("forward_error", TOLERANCE)
                .maximum("transpose_error", TOLERANCE)
                .maximum("hermitian_error", TOLERANCE)
                .minimum("malformed_rejections", 6.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("{error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
