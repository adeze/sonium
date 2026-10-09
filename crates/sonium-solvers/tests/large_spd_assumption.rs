use num_complex::Complex64;
use sonium_solvers::{
    AssumptionProvenance, CsrMatrix, LinearSolveConfig, LinearSolveError, LinearSolverType,
    LinearSystemAssumption, solve_real, solve_real_with_assumption,
    solve_real_with_guess_and_assumption, solve_with_assumption,
};

fn large_identity(size: usize) -> CsrMatrix<f64> {
    let row_ptrs = (0..=size).collect::<Vec<_>>();
    let col_indices = (0..size).collect::<Vec<_>>();
    let values = vec![1.0; size];
    CsrMatrix::from_raw_parts(size, size, row_ptrs, col_indices, values)
}

#[test]
fn large_explicit_cg_requires_a_positive_definite_assumption() {
    let matrix = large_identity(1_025);
    let rhs = vec![1.0; matrix.num_rows];
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 4,
        tolerance: 1e-12,
        ..Default::default()
    };

    let error = solve_real(&matrix, &rhs, &config)
        .expect_err("large CG must not infer positive definiteness");
    assert!(matches!(error, LinearSolveError::InvalidConfig(message) if
        message.contains("PositiveDefinite")));
}

#[test]
fn large_explicit_cg_accepts_checked_positive_definite_assumption() {
    let matrix = large_identity(1_025);
    let rhs = vec![1.0; matrix.num_rows];
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 4,
        tolerance: 1e-12,
        ..Default::default()
    };

    let result = solve_real_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefinite,
    )
    .expect("explicitly assumed SPD identity must solve");
    assert!(result.converged);
    assert_eq!(result.assumption, LinearSystemAssumption::PositiveDefinite);
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::ScalableCertificate
    );
    assert!(result.residual <= config.tolerance);
    assert!(result.x.iter().all(|&value| (value - 1.0).abs() <= 1e-12));

    let generic_result = solve_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefinite,
    )
    .expect("generic explicitly assumed SPD identity must solve");
    assert!(generic_result.converged);

    let warm_result = solve_real_with_guess_and_assumption(
        &matrix,
        &rhs,
        &vec![0.0; matrix.num_cols],
        &config,
        LinearSystemAssumption::PositiveDefinite,
    )
    .expect("explicitly assumed SPD identity with a guess must solve");
    assert!(warm_result.converged);
    assert!(warm_result.residual <= config.tolerance);
}

fn large_spd_without_scalable_certificate(size: usize) -> CsrMatrix<f64> {
    let mut row_ptrs = Vec::with_capacity(size + 1);
    let mut col_indices = Vec::with_capacity(3 * size - 2);
    let mut values = Vec::with_capacity(3 * size - 2);
    row_ptrs.push(0);

    for row in 0..size {
        if row > 0 {
            col_indices.push(row - 1);
            values.push(-1.0);
        }
        col_indices.push(row);
        values.push(2.0);
        if row + 1 < size {
            col_indices.push(row + 1);
            values.push(-1.0);
        }
        row_ptrs.push(col_indices.len());
    }

    CsrMatrix::from_raw_parts(size, size, row_ptrs, col_indices, values)
}

fn large_nonsymmetric_strictly_dominant(size: usize) -> CsrMatrix<f64> {
    let mut row_ptrs = Vec::with_capacity(size + 1);
    let mut col_indices = Vec::with_capacity(3 * size - 2);
    let mut values = Vec::with_capacity(3 * size - 2);
    row_ptrs.push(0);

    for row in 0..size {
        if row > 0 {
            col_indices.push(row - 1);
            values.push(0.25);
        }
        col_indices.push(row);
        values.push(3.0);
        if row + 1 < size {
            col_indices.push(row + 1);
            values.push(if row.is_multiple_of(2) { 1.0 } else { 0.5 });
        }
        row_ptrs.push(col_indices.len());
    }

    CsrMatrix::from_raw_parts(size, size, row_ptrs, col_indices, values)
}

fn large_scaled_spd_with_scalable_certificate(size: usize) -> CsrMatrix<f64> {
    let mut row_ptrs = Vec::with_capacity(size + 1);
    let mut col_indices = Vec::with_capacity(3 * size - 2);
    let mut values = Vec::with_capacity(3 * size - 2);
    row_ptrs.push(0);
    for row in 0..size {
        let row_scale = if row.is_multiple_of(2) { 0.5 } else { 2.0 };
        if row > 0 {
            let column_scale = if (row - 1).is_multiple_of(2) {
                0.5
            } else {
                2.0
            };
            col_indices.push(row - 1);
            values.push(1.0 / (row_scale * column_scale));
        }
        col_indices.push(row);
        values.push(3.0 / (row_scale * row_scale));
        if row + 1 < size {
            let column_scale = if (row + 1).is_multiple_of(2) {
                0.5
            } else {
                2.0
            };
            col_indices.push(row + 1);
            values.push(1.0 / (row_scale * column_scale));
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::from_raw_parts(size, size, row_ptrs, col_indices, values)
}

#[test]
fn large_certified_cg_accepts_jacobi_scaled_spd_certificate() {
    let matrix = large_scaled_spd_with_scalable_certificate(1_025);
    let expected = vec![1.0; matrix.num_cols];
    let rhs = matrix.matvec(&expected);
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 128,
        tolerance: 1e-10,
        ..Default::default()
    };
    let result = solve_real_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefiniteCertified,
    )
    .expect("Jacobi-scaled SPD certificate should admit certified CG");
    assert!(result.converged, "residual={}", result.residual);
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::ScalableCertificate
    );
    let residual = matrix
        .matvec(&result.x)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(residual <= 1e-7, "true residual={residual}");
}

fn large_scaled_hpd_with_complex_off_diagonal(size: usize) -> CsrMatrix<Complex64> {
    let mut row_ptrs = Vec::with_capacity(size + 1);
    let mut col_indices = Vec::with_capacity(3 * size - 2);
    let mut values = Vec::with_capacity(3 * size - 2);
    row_ptrs.push(0);
    for row in 0..size {
        let row_scale = if row.is_multiple_of(2) { 0.5 } else { 2.0 };
        if row > 0 {
            let column_scale = if (row - 1).is_multiple_of(2) {
                0.5
            } else {
                2.0
            };
            col_indices.push(row - 1);
            values.push(Complex64::new(1.0, -0.5) / (row_scale * column_scale));
        }
        col_indices.push(row);
        values.push(Complex64::new(3.0 / (row_scale * row_scale), 0.0));
        if row + 1 < size {
            let column_scale = if (row + 1).is_multiple_of(2) {
                0.5
            } else {
                2.0
            };
            col_indices.push(row + 1);
            values.push(Complex64::new(1.0, 0.5) / (row_scale * column_scale));
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::from_raw_parts(size, size, row_ptrs, col_indices, values)
}

#[test]
fn large_certified_cg_accepts_complex_hpd_certificate() {
    let matrix = large_scaled_hpd_with_complex_off_diagonal(1_025);
    let expected = vec![Complex64::new(1.0, -0.5); matrix.num_cols];
    let rhs = matrix.matvec(&expected);
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 256,
        tolerance: 1e-10,
        ..Default::default()
    };

    let result = solve_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefiniteCertified,
    )
    .expect("complex HPD certificate should admit CG");

    assert!(result.converged, "residual={}", result.residual);
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::ScalableCertificate
    );
    let residual = matrix
        .matvec(&result.x)
        .iter()
        .zip(&rhs)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    let error = result
        .x
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    assert!(residual <= 1e-7, "independent residual={residual}");
    assert!(error <= 1e-7, "solution error={error}");
}

#[test]
fn large_certified_cg_rejects_missing_scalable_certificate() {
    let matrix = large_spd_without_scalable_certificate(1_025);
    let rhs = vec![1.0; matrix.num_rows];
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 4,
        tolerance: 1e-12,
        ..Default::default()
    };

    let error = solve_real_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefiniteCertified,
    )
    .expect_err("certified large CG must reject an uncertified SPD declaration");
    assert!(matches!(error, LinearSolveError::InvalidConfig(message) if
        message.contains("scalable positive-definite certificate")));
}

#[test]
fn large_certified_cg_rejects_nonsymmetric_diagonal_dominance() {
    let matrix = large_nonsymmetric_strictly_dominant(1_025);
    let rhs = vec![1.0; matrix.num_rows];
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 4,
        tolerance: 1e-12,
        ..Default::default()
    };

    let error = solve_real_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefiniteCertified,
    )
    .expect_err("certified CG must reject nonsymmetric diagonal dominance");
    assert!(matches!(error, LinearSolveError::InvalidConfig(_)));
}

#[test]
fn small_certified_cg_uses_exact_validation_provenance() {
    let matrix = CsrMatrix::identity(2);
    let rhs = vec![1.0, 2.0];
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 4,
        tolerance: 1e-12,
        ..Default::default()
    };

    let result = solve_real_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefiniteCertified,
    )
    .expect("small certified SPD system should use exact validation");
    assert_eq!(
        result.assumption,
        LinearSystemAssumption::PositiveDefiniteCertified
    );
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::IndependentlyValidated
    );
}
