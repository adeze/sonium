//! CPU contract evidence for strict saddle-point preconditioner adjoint metadata.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::{
    BlockPreconditionerConfig, CsrMatrix, SaddlePointConfig, SaddlePointPreconditioner,
    SchurStrategy,
};
use sonium_solvers_core::traits::{AdjointCapability, Preconditioner};
use std::time::Instant;

const CASE_ID: &str = "solvers.saddle-point-adjoint.v1";

fn diagonal_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::try_from_raw_parts(
        4,
        4,
        vec![0, 1, 2, 3, 4],
        vec![0, 1, 2, 3],
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -0.5),
            Complex64::new(4.0, 0.25),
            Complex64::new(5.0, -0.75),
        ],
    )
    .expect("canonical block diagonal matrix")
}

fn coupled_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, Complex64::new(2.0, 0.5)),
            (0, 1, Complex64::new(0.2, -0.1)),
            (0, 2, Complex64::new(0.7, 0.3)),
            (0, 3, Complex64::new(-0.4, 0.2)),
            (1, 0, Complex64::new(-0.1, 0.4)),
            (1, 1, Complex64::new(3.0, -0.25)),
            (1, 2, Complex64::new(0.5, -0.2)),
            (1, 3, Complex64::new(0.6, 0.1)),
            (2, 0, Complex64::new(0.3, -0.2)),
            (2, 1, Complex64::new(-0.8, 0.1)),
            (2, 2, Complex64::new(1.0, 0.2)),
            (3, 0, Complex64::new(0.2, 0.5)),
            (3, 1, Complex64::new(0.4, -0.3)),
            (3, 3, Complex64::new(1.5, -0.1)),
        ],
    )
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = diagonal_matrix();
    let diagonal_config = SaddlePointConfig {
        block1_size: 2,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::Diagonal,
        schur_strategy: SchurStrategy::Diagonal,
        upper_triangular: false,
    };
    let diagonal = SaddlePointPreconditioner::from_matrix(&matrix, diagonal_config);
    if diagonal.transpose_capability() != AdjointCapability::Exact
        || diagonal.hermitian_capability() != AdjointCapability::Exact
    {
        return Err("block-diagonal saddle preconditioner did not advertise exact adjoints".into());
    }

    let input = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(0.75, 1.5),
        Complex64::new(-1.25, -0.5),
    ];
    let transpose = diagonal
        .try_apply_transpose(&input)
        .map_err(|error| error.to_string())?;
    let hermitian = diagonal
        .try_apply_hermitian(&input)
        .map_err(|error| error.to_string())?;
    let values = [
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -0.5),
        Complex64::new(4.0, 0.25),
        Complex64::new(5.0, -0.75),
    ];
    let transpose_error = transpose
        .iter()
        .zip(&input)
        .zip(values)
        .map(|((actual, expected), diagonal)| (*actual - *expected / diagonal).norm())
        .fold(0.0, f64::max);
    let hermitian_error = hermitian
        .iter()
        .zip(&input)
        .zip(values)
        .map(|((actual, expected), diagonal)| (*actual - *expected / diagonal.conj()).norm())
        .fold(0.0, f64::max);

    let triangular_config = SaddlePointConfig {
        block1_size: 2,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::Diagonal,
        schur_strategy: SchurStrategy::External,
        upper_triangular: true,
    };
    let schur = CsrMatrix::try_from_raw_parts(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    )
    .expect("canonical Schur approximation");
    let triangular_matrix = coupled_matrix();
    let triangular = SaddlePointPreconditioner::from_matrix_with_schur(
        &triangular_matrix,
        &schur,
        triangular_config,
    );
    let probe = vec![
        Complex64::new(-0.3, 0.8),
        Complex64::new(1.2, -0.4),
        Complex64::new(0.6, 0.1),
        Complex64::new(-0.7, -1.1),
    ];
    let triangular_forward = triangular.apply(&input);
    let triangular_transpose = triangular
        .try_apply_transpose(&probe)
        .map_err(|error| error.to_string())?;
    let triangular_hermitian = triangular
        .try_apply_hermitian(&probe)
        .map_err(|error| error.to_string())?;
    let triangular_transpose_lhs: Complex64 = triangular_forward
        .iter()
        .zip(&probe)
        .map(|(a, b)| *a * *b)
        .sum();
    let triangular_transpose_rhs: Complex64 = input
        .iter()
        .zip(&triangular_transpose)
        .map(|(a, b)| *a * *b)
        .sum();
    let triangular_transpose_error = (triangular_transpose_lhs - triangular_transpose_rhs).norm();
    let triangular_hermitian_lhs: Complex64 = triangular_forward
        .iter()
        .zip(&probe)
        .map(|(a, b)| a.conj() * *b)
        .sum();
    let triangular_hermitian_rhs: Complex64 = input
        .iter()
        .zip(&triangular_hermitian)
        .map(|(a, b)| a.conj() * *b)
        .sum();
    let triangular_hermitian_error = (triangular_hermitian_lhs - triangular_hermitian_rhs).norm();
    let triangular_exact = (triangular.transpose_capability() == AdjointCapability::Exact
        && triangular.hermitian_capability() == AdjointCapability::Exact
        && triangular_transpose_error <= 1.0e-12
        && triangular_hermitian_error <= 1.0e-12) as u8;

    Ok(vec![
        ("transpose_error", transpose_error),
        ("hermitian_error", hermitian_error),
        ("triangular_transpose_error", triangular_transpose_error),
        ("triangular_hermitian_error", triangular_hermitian_error),
        ("triangular_exact_contract", f64::from(triangular_exact)),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            if metrics[0].1 > 1.0e-12
                || metrics[1].1 > 1.0e-12
                || metrics[2].1 > 1.0e-12
                || metrics[3].1 > 1.0e-12
                || metrics[4].1 != 1.0
            {
                eprintln!("SADDLE-POINT ADJOINT FAILED: {metrics:?}");
                println!("FAILED");
                std::process::exit(1);
            }
            let mut result = QaResult::passed(CASE_ID);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let mut result = QaResult::failed(CASE_ID, error);
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
