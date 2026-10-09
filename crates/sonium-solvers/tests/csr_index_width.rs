use sonium_solvers::SparseLuFactorization;
use sonium_solvers::sparse::CsrMatrix;

fn rhs_for_known_solution(matrix: &CsrMatrix<f64>) -> (Vec<f64>, Vec<f64>) {
    let expected = vec![0.5, -1.25, 2.0];
    (matrix.matvec(&expected), expected)
}

#[test]
fn checked_u32_and_u64_csr_indices_match_usize_sparse_lu() {
    let row_ptrs = [0, 2, 4, 5];
    let col_indices = [0, 1, 0, 1, 2];
    let values = vec![4.0, -1.0, -1.0, 4.0, 3.0];

    let u32_matrix = CsrMatrix::try_from_raw_parts_indexed(
        3,
        3,
        row_ptrs.map(|index| index as u32),
        col_indices.map(|index| index as u32),
        values.clone(),
    )
    .unwrap();
    let u64_matrix = CsrMatrix::try_from_raw_parts_indexed(
        3,
        3,
        row_ptrs.map(|index| index as u64),
        col_indices.map(|index| index as u64),
        values.clone(),
    )
    .unwrap();
    let usize_matrix =
        CsrMatrix::try_from_raw_parts(3, 3, row_ptrs.to_vec(), col_indices.to_vec(), values)
            .unwrap();

    assert_eq!(u32_matrix.row_ptrs, usize_matrix.row_ptrs);
    assert_eq!(u32_matrix.col_indices, usize_matrix.col_indices);
    assert_eq!(u64_matrix.row_ptrs, usize_matrix.row_ptrs);
    assert_eq!(u64_matrix.col_indices, usize_matrix.col_indices);

    let mixed_matrix = CsrMatrix::try_from_raw_parts_indexed(
        3,
        3,
        [0_u32, 2, 4, 5],
        [0_u64, 1, 0, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, 3.0],
    )
    .unwrap();
    assert_eq!(mixed_matrix.row_ptrs, usize_matrix.row_ptrs);
    assert_eq!(mixed_matrix.col_indices, usize_matrix.col_indices);

    let (rhs, expected) = rhs_for_known_solution(&usize_matrix);
    for matrix in [&u32_matrix, &u64_matrix] {
        let factor = SparseLuFactorization::factorize(matrix).unwrap();
        let solution = factor.solve(&rhs).unwrap();
        assert!(
            solution
                .iter()
                .zip(&expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12)
        );
        let residual = matrix
            .matvec(&solution)
            .iter()
            .zip(&rhs)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(residual < 1.0e-12, "residual={residual:e}");
    }
}

#[test]
fn checked_indexed_csr_rejects_invalid_shape_and_indices() {
    let values = vec![1.0, 2.0];
    assert!(
        CsrMatrix::<f64>::try_from_raw_parts_indexed(
            2,
            2,
            [0_u32, 1, 1],
            [0_u32, 1],
            values.clone(),
        )
        .is_err()
    );
    assert!(
        CsrMatrix::<f64>::try_from_raw_parts_indexed(1, 2, [0_u64, 1], [2_u64], vec![1.0],)
            .is_err()
    );
}
