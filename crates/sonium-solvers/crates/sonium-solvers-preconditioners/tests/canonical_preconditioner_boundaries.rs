use sonium_solvers_preconditioners::{
    DiagonalPreconditioner, HiluPreconditioner, IluColoringPreconditioner,
    IluFixedPointPreconditioner, IluPreconditioner, IlutPreconditioner, ParIlutPreconditioner,
};
use sonium_solvers_sparse::CsrMatrix;

fn noncanonical_matrix() -> CsrMatrix<f64> {
    CsrMatrix::try_from_raw_parts(2, 2, vec![0, 2, 3], vec![1, 0, 1], vec![2.0, 3.0, 4.0])
        .expect("raw CSR structure is valid before the canonical policy")
}

#[test]
fn canonical_validation_rejects_unsorted_rows_but_spmv_structure_accepts_them() {
    let matrix = noncanonical_matrix();
    assert!(matrix.validate_structure().is_ok());
    assert!(matrix.validate_canonical_structure().is_err());
}

#[test]
fn checked_factorization_preconditioners_reject_noncanonical_rows() {
    let matrix = noncanonical_matrix();
    assert!(DiagonalPreconditioner::try_from_csr(&matrix).is_err());
    assert!(IluPreconditioner::try_from_csr(&matrix).is_err());
    assert!(IlutPreconditioner::try_from_csr(&matrix).is_err());
    assert!(HiluPreconditioner::try_from_csr(&matrix).is_err());
    assert!(IluColoringPreconditioner::try_from_csr(&matrix).is_err());
    assert!(IluFixedPointPreconditioner::try_from_csr(&matrix, 2).is_err());
    assert!(ParIlutPreconditioner::try_from_csr(&matrix).is_err());
}

#[test]
fn checked_ilu_setup_propagates_canonical_failure() {
    let matrix = noncanonical_matrix();
    assert!(IluPreconditioner::try_from_csr(&matrix).is_err());
}
