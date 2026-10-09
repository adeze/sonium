use sonium_solvers_core::traits::SolverError;
use sonium_solvers_sparse::{BsrMatrix, CsrMatrix};

#[test]
fn bsr_checked_conversion_rejects_overflow_generated_by_duplicate_packing() {
    let csr = CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![0, 0], vec![f64::MAX, f64::MAX]);

    let error = BsrMatrix::<f64, 2>::try_from_csr(&csr).unwrap_err();
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "BSR construction",
            ref reason,
        } if reason.contains("generated block values")
    ));
}
