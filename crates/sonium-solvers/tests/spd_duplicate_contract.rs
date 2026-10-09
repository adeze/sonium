use sonium_solvers::{
    AssumptionProvenance, CsrMatrix, LinearSolveConfig, LinearSolverType, LinearSystemAssumption,
    solve_real, solve_real_with_assumption,
};

#[test]
fn cg_amg_accepts_spd_matrix_with_duplicate_off_diagonal_entries() {
    // CSR SpMV sums duplicate columns. This is the same SPD matrix as
    // [[2, 1], [1, 2]], with the first diagonal and off-diagonal split.
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 4, 6],
        vec![0, 0, 1, 1, 0, 1],
        vec![1.0_f64, 1.0, 0.5, 0.5, 1.0, 2.0],
    );
    let rhs = [3.0, 3.0];
    let config = LinearSolveConfig {
        solver: LinearSolverType::CgAmg,
        max_iterations: 20,
        tolerance: 1.0e-12,
        ..LinearSolveConfig::default()
    };

    let result = solve_real(&matrix, &rhs, &config)
        .expect("duplicate CSR entries still represent an SPD system");
    assert!(result.converged);
    assert!(
        result
            .x
            .iter()
            .zip([1.0, 1.0])
            .all(|(got, expected)| (got - expected).abs() < 1.0e-10)
    );
}

#[test]
fn cg_amg_rejects_nonpositive_coalesced_duplicate_diagonal() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 3, 4],
        vec![0, 0, 1, 1],
        vec![2.0_f64, -3.0, 0.0, 1.0],
    );
    let config = LinearSolveConfig {
        solver: LinearSolverType::CgAmg,
        ..LinearSolveConfig::default()
    };

    let error = solve_real(&matrix, &[1.0, 1.0], &config)
        .expect_err("non-positive summed diagonal must be rejected");
    assert!(matches!(
        error,
        sonium_solvers::LinearSolveError::InvalidConfig(_)
    ));
}

#[test]
fn certified_large_spd_accepts_duplicate_off_diagonal_rows() {
    const SIZE: usize = 1_025;
    let mut row_ptrs = Vec::with_capacity(SIZE + 1);
    let mut col_indices = Vec::with_capacity(3 * SIZE);
    let mut values = Vec::with_capacity(3 * SIZE);
    row_ptrs.push(0);
    for row in 0..SIZE {
        if row > 0 {
            col_indices.push(row - 1);
            values.push(0.5);
        }
        col_indices.push(row);
        values.push(3.0);
        if row + 1 < SIZE {
            col_indices.push(row + 1);
            values.push(if row == 0 { 0.25 } else { 0.5 });
            if row == 0 {
                col_indices.push(row + 1);
                values.push(0.25);
            }
        }
        row_ptrs.push(col_indices.len());
    }
    let matrix = CsrMatrix::from_raw_parts(SIZE, SIZE, row_ptrs, col_indices, values);
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 1,
        tolerance: 1.0e-2,
        ..LinearSolveConfig::default()
    };

    let result = solve_real_with_assumption(
        &matrix,
        &vec![1.0; SIZE],
        &config,
        LinearSystemAssumption::PositiveDefiniteCertified,
    )
    .expect("large certified SPD validation should coalesce duplicate rows");
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::ScalableCertificate
    );
}
