use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{LinearOperator, SolverError};
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sparse::spmv_plan::{
    CompactCsrMatrix, RowBucketedCsrMatrix, SellCSigmaMatrix, SpmvPlan, SpmvPlanConfig,
};

fn assert_csr_validation_error(error: SolverError) {
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR validation",
            ..
        }
    ));
}

#[test]
fn checked_spmv_plan_rejects_public_csr_mutation() {
    let mut matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]);
    matrix.row_ptrs[2] = 1;

    assert_csr_validation_error(
        SpmvPlan::try_from_csr(&matrix, SpmvPlanConfig::default()).unwrap_err(),
    );
    assert_csr_validation_error(matrix.validate_structure().unwrap_err());
}

#[test]
fn checked_spmv_plan_rejects_out_of_range_column_after_construction() {
    let mut matrix = CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![0], vec![1.0_f64]);
    matrix.col_indices[0] = 2;

    assert_csr_validation_error(matrix.try_spmv_plan(SpmvPlanConfig::default()).unwrap_err());
}

#[test]
fn checked_sparse_format_constructors_reject_public_csr_mutation() {
    let mut matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]);
    matrix.row_ptrs[2] = 1;

    assert_csr_validation_error(CompactCsrMatrix::try_from_csr(&matrix).unwrap_err());
    assert_csr_validation_error(RowBucketedCsrMatrix::try_from_csr(&matrix).unwrap_err());
    assert_csr_validation_error(SellCSigmaMatrix::try_from_csr(&matrix, 2, 8).unwrap_err());
}

#[test]
fn checked_compact_transpose_constructors_reject_out_of_range_columns() {
    let mut matrix = CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![0], vec![1.0_f64]);
    matrix.col_indices[0] = 2;

    assert_csr_validation_error(CompactCsrMatrix::try_transpose_from_csr(&matrix).unwrap_err());
    assert_csr_validation_error(CompactCsrMatrix::try_hermitian_from_csr(&matrix).unwrap_err());
}

#[test]
fn checked_spmv_plan_rejects_zero_numa_partition_configuration() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0_f64]);
    let config = SpmvPlanConfig {
        preference: sonium_solvers_sparse::spmv_plan::SpmvPlanPreference::NumaCsr,
        numa_partitions: Some(0),
        ..SpmvPlanConfig::default()
    };

    let error = SpmvPlan::try_from_csr(&matrix, config).unwrap_err();
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "SpMV plan configuration",
            ..
        }
    ));
}

#[test]
fn checked_direct_csr_spmv_rejects_public_mutation_before_traversal() {
    let mut matrix = CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![0], vec![1.0_f64]);
    matrix.col_indices[0] = 2;

    assert_csr_validation_error(matrix.try_matvec(&[1.0, 1.0]).unwrap_err());
    assert_csr_validation_error(matrix.try_matvec_transpose(&[1.0]).unwrap_err());
    assert_csr_validation_error(matrix.try_matvec_hermitian(&[1.0]).unwrap_err());
    assert_csr_validation_error(matrix.try_apply(&[1.0, 1.0]).unwrap_err());
    assert_csr_validation_error(matrix.try_apply_transpose(&[1.0]).unwrap_err());
    assert_csr_validation_error(matrix.try_apply_hermitian(&[1.0]).unwrap_err());

    let x = DenseMatrix::zeros(2, 1);
    let mut y = DenseMatrix::zeros(1, 1);
    assert_csr_validation_error(matrix.try_apply_block(&x, &mut y).unwrap_err());

    let x = DenseMatrix::zeros(1, 1);
    let mut y = DenseMatrix::zeros(2, 1);
    assert_csr_validation_error(matrix.try_apply_transpose_block(&x, &mut y).unwrap_err());
    assert_csr_validation_error(matrix.try_apply_hermitian_block(&x, &mut y).unwrap_err());
}

#[test]
fn checked_csr_block_actions_reject_nonfinite_input_and_preserve_output() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![2.0_f64]);
    let x = DenseMatrix::from_row_major_vec(1, 1, vec![f64::NAN]);

    let mut y = DenseMatrix::from_row_major_vec(1, 1, vec![7.0]);
    let original = y.as_slice().to_vec();
    let error = matrix
        .try_spmm_into(&x, &mut y)
        .expect_err("non-finite block input must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR block SpMV",
            ref reason,
        } if reason.contains("input")
    ));
    assert_eq!(y.as_slice(), original.as_slice());

    let error = matrix
        .try_apply_block(&x, &mut y)
        .expect_err("checked block input must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR block SpMV",
            ref reason,
        } if reason.contains("input")
    ));
    assert_eq!(y.as_slice(), original.as_slice());
}

#[test]
fn checked_csr_block_actions_stage_generated_nonfinite_values() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let x = DenseMatrix::from_row_major_vec(1, 1, vec![f64::MAX]);

    let mut forward = DenseMatrix::from_row_major_vec(1, 1, vec![11.0]);
    let forward_original = forward.as_slice().to_vec();
    let error = matrix
        .try_apply_block(&x, &mut forward)
        .expect_err("overflowing forward block must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR block SpMV",
            ref reason,
        } if reason.contains("computed output")
    ));
    assert_eq!(forward.as_slice(), forward_original.as_slice());

    let mut transpose = DenseMatrix::from_row_major_vec(1, 1, vec![13.0]);
    let transpose_original = transpose.as_slice().to_vec();
    let error = matrix
        .try_apply_transpose_block(&x, &mut transpose)
        .expect_err("overflowing transpose block must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR block transpose SpMV",
            ref reason,
        } if reason.contains("computed output")
    ));
    assert_eq!(transpose.as_slice(), transpose_original.as_slice());

    let mut hermitian = DenseMatrix::from_row_major_vec(1, 1, vec![17.0]);
    let hermitian_original = hermitian.as_slice().to_vec();
    let error = matrix
        .try_apply_hermitian_block(&x, &mut hermitian)
        .expect_err("overflowing Hermitian block must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR block Hermitian SpMV",
            ref reason,
        } if reason.contains("computed output")
    ));
    assert_eq!(hermitian.as_slice(), hermitian_original.as_slice());
}

#[test]
fn checked_csr_single_vector_actions_stage_generated_nonfinite_values() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);

    let mut forward = vec![11.0];
    let original = forward.clone();
    let error = matrix
        .try_matvec_into_buf(&[2.0], &mut forward)
        .expect_err("overflowing forward SpMV must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR SpMV",
            ref reason,
        } if reason.contains("computed output")
    ));
    assert_eq!(forward, original);

    let mut accumulated = vec![13.0];
    let original = accumulated.clone();
    let error = matrix
        .try_matvec_add(&[2.0], &mut accumulated)
        .expect_err("overflowing accumulated SpMV must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR SpMV add",
            ref reason,
        } if reason.contains("computed output")
    ));
    assert_eq!(accumulated, original);

    let mut transpose = vec![17.0];
    let original = transpose.clone();
    let error = matrix
        .try_matvec_transpose_into_buf(&[2.0], &mut transpose)
        .expect_err("overflowing transpose SpMV must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR transpose SpMV",
            ref reason,
        } if reason.contains("computed output")
    ));
    assert_eq!(transpose, original);

    let mut hermitian = vec![19.0];
    let original = hermitian.clone();
    let error = matrix
        .try_matvec_hermitian_into_buf(&[2.0], &mut hermitian)
        .expect_err("overflowing Hermitian SpMV must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR Hermitian SpMV",
            ref reason,
        } if reason.contains("computed output")
    ));
    assert_eq!(hermitian, original);
}

#[test]
fn checked_compact_single_vector_actions_stage_generated_nonfinite_values() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let compact = CompactCsrMatrix::try_from_csr(&matrix).expect("valid compact CSR");

    let mut forward = vec![23.0];
    let original = forward.clone();
    let error = compact
        .try_matvec_into_buf(&[2.0], &mut forward)
        .expect_err("overflowing compact forward SpMV must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "compact CSR SpMV",
            ref reason,
        } if reason.contains("output")
    ));
    assert_eq!(forward, original);

    let mut transpose = vec![29.0];
    let original = transpose.clone();
    let error = compact
        .try_matvec_transpose_into_buf(&[2.0], &mut transpose)
        .expect_err("overflowing compact transpose SpMV must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "compact CSR transpose SpMV",
            ref reason,
        } if reason.contains("output")
    ));
    assert_eq!(transpose, original);

    let mut hermitian = vec![31.0];
    let original = hermitian.clone();
    let error = compact
        .try_matvec_hermitian_into_buf(&[2.0], &mut hermitian)
        .expect_err("overflowing compact Hermitian SpMV must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "compact CSR Hermitian SpMV",
            ref reason,
        } if reason.contains("output")
    ));
    assert_eq!(hermitian, original);
}

#[test]
fn checked_triplet_construction_rejects_generated_non_finite_duplicate() {
    let finite = CsrMatrix::try_from_triplets(1, 1, vec![(0, 0, 1.25_f64), (0, 0, 2.75_f64)])
        .expect("finite duplicate coalescing should remain valid");
    assert_eq!(finite.get(0, 0), 4.0);

    let overflowing = CsrMatrix::try_from_triplets(1, 1, vec![(0, 0, f64::MAX), (0, 0, f64::MAX)])
        .expect_err("duplicate overflow must be rejected at construction");
    assert!(matches!(
        overflowing,
        SolverError::OperationFailed {
            operation: "CSR triplet construction",
            ref reason,
        } if reason.contains("duplicate coalescing generated non-finite")
    ));
}

#[test]
fn checked_matmul_rejects_generated_non_finite_product() {
    let left = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![2.0_f64]);
    let right = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![3.0_f64]);
    let product = left
        .try_matmul(&right)
        .expect("finite CSR product should succeed");
    assert_eq!(product.get(0, 0), 6.0);

    let overflowing = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let err = overflowing
        .try_matmul(&right)
        .expect_err("overflow-generated product must be rejected");
    assert!(matches!(
        err,
        SolverError::OperationFailed {
            operation: "CSR matmul",
            ref reason,
        } if reason.contains("non-finite")
    ));

    let err = left
        .try_matmul_with_drop(&right, -1.0)
        .expect_err("negative drop tolerance must be rejected");
    assert!(matches!(
        err,
        SolverError::OperationFailed {
            operation: "CSR matmul",
            ..
        }
    ));
}

#[test]
fn checked_csr_mutations_preserve_finite_results() {
    let mut matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]);
    matrix
        .try_scale(2.0)
        .expect("finite CSR scaling should succeed");
    assert_eq!(matrix.values, vec![4.0, 6.0]);

    matrix
        .try_add_diagonal(0.5)
        .expect("finite CSR diagonal update should succeed");
    assert_eq!(matrix.values, vec![4.5, 6.5]);
}

#[test]
fn checked_csr_mutations_reject_generated_non_finite_without_mutating() {
    let mut scale = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let original = scale.values.clone();
    let error = scale
        .try_scale(2.0)
        .expect_err("finite scaling overflow must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR scale",
            ref reason,
        } if reason.contains("non-finite")
    ));
    assert_eq!(scale.values, original);

    let error = scale
        .try_scale(f64::NAN)
        .expect_err("non-finite scaling scalar must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR scale",
            ref reason,
        } if reason.contains("scalar")
    ));
    assert_eq!(scale.values, original);

    let mut diagonal = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let original = diagonal.values.clone();
    let error = diagonal
        .try_add_diagonal(f64::MAX)
        .expect_err("finite diagonal overflow must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR add_diagonal",
            ref reason,
        } if reason.contains("non-finite")
    ));
    assert_eq!(diagonal.values, original);

    let error = diagonal
        .try_add_diagonal(f64::INFINITY)
        .expect_err("non-finite diagonal scalar must be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "CSR add_diagonal",
            ref reason,
        } if reason.contains("scalar")
    ));
    assert_eq!(diagonal.values, original);
}
