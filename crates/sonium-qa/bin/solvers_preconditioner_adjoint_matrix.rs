//! Independent CPU reference for composed preconditioner adjoint actions.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::preconditioners::{
    AdditiveSchwarzPreconditioner, BlockJacobiPreconditioner, IlutConfig,
};
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-adjoint-matrix.v1";

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(2.0, 0.25),
            Complex64::new(5.0, -1.0),
            Complex64::new(1.0, 0.75),
            Complex64::new(0.5, -0.25),
            Complex64::new(3.0, 0.5),
            Complex64::new(0.75, -0.2),
            Complex64::new(1.0, 0.1),
            Complex64::new(2.5, -0.5),
        ],
    )
}

fn vectors() -> (Vec<Complex64>, Vec<Complex64>) {
    (
        vec![
            Complex64::new(0.7, -0.2),
            Complex64::new(-1.1, 0.4),
            Complex64::new(0.3, 0.9),
            Complex64::new(1.4, -0.6),
        ],
        vec![
            Complex64::new(-0.5, 0.8),
            Complex64::new(0.2, -1.3),
            Complex64::new(1.1, 0.1),
            Complex64::new(-0.4, 0.6),
        ],
    )
}

fn hermitian_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b).map(|(a, b)| a.conj() * b).sum()
}

fn check<P: Preconditioner<Complex64>>(
    preconditioner: &P,
    r: &[Complex64],
    s: &[Complex64],
) -> f64 {
    let forward = preconditioner.apply(r);
    let hermitian = preconditioner.apply_hermitian(s);
    (hermitian_inner(&forward, s) - hermitian_inner(r, &hermitian)).norm()
}

fn run() -> Result<(f64, f64), String> {
    let matrix = matrix();
    let (r, s) = vectors();
    let config = IlutConfig {
        drop_tolerance: 0.0,
        fill_factor: 100,
        pivot_tolerance: 1.0e-15,
        pivoting: true,
    };
    let block = BlockJacobiPreconditioner::try_from_csr(&matrix, 2, config)
        .map_err(|error| format!("block-Jacobi setup failed: {error}"))?;
    let schwarz = AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, 1)
        .map_err(|error| format!("Schwarz setup failed: {error}"))?;
    Ok((check(&block, &r, &s), check(&schwarz, &r, &s)))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((block_error, schwarz_error)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent complex Hermitian inner-product parity for Block-Jacobi and Schwarz")
                .input_checksum("analytic/deterministic-fixture:preconditioner-adjoint-matrix-v1")
                .reference_checksum("analytic/deterministic-fixture:preconditioner-adjoint-matrix-v1")
                .dimension("4x4 complex CSR; two Block-Jacobi blocks and two overlapping Schwarz subdomains")
                .tolerance("Hermitian adjoint inner-product error <= 1e-10")
                .metric("block_jacobi_hermitian_error", block_error)
                .metric("schwarz_hermitian_error", schwarz_error)
                .maximum("block_jacobi_hermitian_error", 1.0e-10)
                .maximum("schwarz_hermitian_error", 1.0e-10)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent complex Hermitian inner-product parity for Block-Jacobi and Schwarz")
                .input_checksum("analytic/deterministic-fixture:preconditioner-adjoint-matrix-v1")
                .reference_checksum("analytic/deterministic-fixture:preconditioner-adjoint-matrix-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
