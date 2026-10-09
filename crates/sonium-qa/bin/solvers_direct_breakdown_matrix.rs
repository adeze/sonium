//! Independent direct-factor breakdown and failure-exit packet.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, SparseDirectConfig,
    SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "solvers.direct-breakdown-matrix.v1";
const CHECKSUM: &str = "deterministic-fixture:direct-breakdown-matrix-v1";

fn checked_failure(label: &str, operation: impl FnOnce() -> bool) -> Result<f64, String> {
    let result = catch_unwind(AssertUnwindSafe(operation))
        .map_err(|_| format!("{label}: direct breakdown panicked"))?;
    if result {
        Ok(1.0)
    } else {
        Err(format!("{label}: invalid factorization succeeded"))
    }
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let singular_dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    let dense_lu = checked_failure("dense singular LU", || {
        match factorize_dense(
            &singular_dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::General,
                method: DenseDirectMethod::Lu,
                ..Default::default()
            },
        ) {
            Err(_) => true,
            Ok(factor) => factor.solve(&[1.0, 2.0]).is_err(),
        }
    })?;

    let non_spd_dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
    let dense_cholesky = checked_failure("dense non-SPD Cholesky", || {
        factorize_dense(
            &non_spd_dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::SymmetricPositiveDefinite,
                method: DenseDirectMethod::Cholesky,
                ..Default::default()
            },
        )
        .is_err()
    })?;
    let rank_deficient_qr =
        DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let dense_qr = checked_failure("dense rank-deficient QR", || {
        factorize_dense(
            &rank_deficient_qr,
            &DenseDirectConfig {
                matrix_class: MatrixClass::RectangularFullRank,
                method: DenseDirectMethod::Qr,
                ..Default::default()
            },
        )
        .is_err()
    })?;

    let singular_sparse = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 4.0)],
    );
    let sparse_lu = checked_failure("sparse singular LU", || {
        match factorize_sparse(
            &singular_sparse,
            &SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method: SparseDirectMethod::SimplicialLu,
                ..Default::default()
            },
            None,
        ) {
            Err(_) => true,
            Ok(factor) => factor.solve(&[1.0, 2.0]).is_err(),
        }
    })?;

    let non_spd_sparse = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 1.0)],
    );
    let sparse_cholesky = checked_failure("sparse non-SPD Cholesky", || {
        factorize_sparse(
            &non_spd_sparse,
            &SparseDirectConfig {
                matrix_class: MatrixClass::SymmetricPositiveDefinite,
                method: SparseDirectMethod::SimplicialCholesky,
                ..Default::default()
            },
            None,
        )
        .is_err()
    })?;

    Ok(vec![
        ("dense_singular_lu_rejected", dense_lu),
        ("dense_non_spd_cholesky_rejected", dense_cholesky),
        ("dense_rank_deficient_qr_rejected", dense_qr),
        ("sparse_singular_lu_rejected", sparse_lu),
        ("sparse_non_spd_cholesky_rejected", sparse_cholesky),
        ("panic_free", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = metrics.into_iter().fold(
                QaResult::passed(CASE_ID)
                    .reference("independent typed direct-factor failure matrix")
                    .reference_checksum(CHECKSUM)
                    .input_checksum(CHECKSUM)
                    .formulation(
                        "dense/sparse LU, rank-deficient QR, and non-SPD Cholesky failures",
                    )
                    .dimension("2x2 real dense/CSR plus 3x2 rank-deficient QR breakdown fixtures")
                    .tolerance("all five invalid factors reject at factor or solve without panic")
                    .elapsed_ms(started.elapsed().as_millis() as u64),
                |result, (name, value)| result.metric(name, value),
            );
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
