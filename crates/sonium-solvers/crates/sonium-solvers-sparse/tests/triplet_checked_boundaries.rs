use sonium_solvers_core::traits::SolverError;
use sonium_solvers_sparse::TripletMatrix;

#[test]
fn checked_triplet_conversion_rejects_public_buffer_mismatch() {
    let malformed = TripletMatrix {
        rows: vec![0, 1],
        cols: vec![0],
        values: vec![1.0_f64, 2.0],
        num_rows: 2,
        num_cols: 2,
    };
    assert!(matches!(
        malformed.try_to_csr(),
        Err(SolverError::DimensionMismatch {
            expected: 2,
            got: 1,
        })
    ));
}

#[test]
fn checked_triplet_conversion_rejects_indices_and_generated_non_finite_values() {
    let out_of_range = TripletMatrix {
        rows: vec![2],
        cols: vec![0],
        values: vec![1.0_f64],
        num_rows: 2,
        num_cols: 1,
    };
    assert!(matches!(
        out_of_range.try_to_csr(),
        Err(SolverError::OperationFailed {
            operation: "CSR triplet construction",
            ..
        })
    ));

    let overflowing = TripletMatrix {
        rows: vec![0, 0],
        cols: vec![0, 0],
        values: vec![f64::MAX, f64::MAX],
        num_rows: 1,
        num_cols: 1,
    };
    assert!(matches!(
        overflowing.try_to_csr(),
        Err(SolverError::OperationFailed {
            operation: "CSR triplet construction",
            ref reason,
        }) if reason.contains("duplicate coalescing generated non-finite")
    ));
}

#[test]
fn compatibility_triplet_conversion_remains_finite_for_valid_input() {
    let mut triplets = TripletMatrix::new(2, 2);
    triplets.push(0, 0, 1.0_f64);
    triplets.push(0, 0, 2.0);
    triplets.push(1, 1, 3.0);
    let csr = triplets.to_csr();
    assert_eq!(csr.values, vec![3.0, 3.0]);
    assert!(csr.validate_canonical_structure().is_ok());
}
