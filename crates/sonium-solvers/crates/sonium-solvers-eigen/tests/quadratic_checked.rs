//! Checked quadratic-eigenproblem tests for damped structural spectra.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use sonium_solvers_eigen::{QuadraticEigenConfig, quadratic_eigen_damped};
use sonium_solvers_sparse::{CsrMatrix, TripletMatrix};

fn csr_from_dense(dense: &[Vec<f64>]) -> CsrMatrix<f64> {
    let rows = dense.len();
    let cols = dense[0].len();
    let mut triplets = TripletMatrix::<f64>::new(rows, cols);
    for (row, values) in dense.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            if value != 0.0 {
                triplets.push(row, column, value);
            }
        }
    }
    triplets.to_csr()
}

#[test]
fn decoupled_diagonal_matches_scalar_quadratic_roots() {
    let mass = csr_from_dense(&[
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let damping = csr_from_dense(&[
        vec![0.1, 0.0, 0.0],
        vec![0.0, 0.4, 0.0],
        vec![0.0, 0.0, 0.0],
    ]);
    let stiffness = csr_from_dense(&[
        vec![4.0, 0.0, 0.0],
        vec![0.0, 9.0, 0.0],
        vec![0.0, 0.0, 16.0],
    ]);
    let config = QuadraticEigenConfig {
        target: Complex64::new(0.0, 0.0),
        num_eigenvalues: 6,
        krylov_dimension: 6,
        tolerance: 1.0e-10,
        max_qr_iterations: 4_000,
        breakdown_tolerance: 1.0e-14,
    };
    let result = quadratic_eigen_damped(&mass, &damping, &stiffness, &config).expect("valid QEP");
    assert!(result.converged);
    assert_eq!(result.eigenvalues.len(), 6);
    assert!(
        result
            .residual_norms
            .iter()
            .all(|residual| *residual < 1.0e-8)
    );

    let mut expected = Vec::new();
    for (m, (c, k)) in [1.0, 2.0, 1.0]
        .iter()
        .zip([0.1, 0.4, 0.0].iter().zip([4.0, 9.0, 16.0].iter()))
    {
        let discriminant = Complex64::new(c * c - 4.0 * m * k, 0.0).sqrt();
        expected.push((Complex64::new(-c, 0.0) + discriminant) / (2.0 * m));
        expected.push((Complex64::new(-c, 0.0) - discriminant) / (2.0 * m));
    }
    // Greedy nearest-neighbor matching: positional pairing after a sort is
    // fragile when solver noise reorders nearly equal real parts.
    let mut remaining = expected.clone();
    for got in &result.eigenvalues {
        let (slot, _) = remaining
            .iter()
            .enumerate()
            .map(|(index, want)| (index, (got - want).norm()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"))
            .expect("non-empty");
        let want = remaining.remove(slot);
        assert!((got - want).norm() < 1.0e-6, "got {got}, want {want}");
    }
}

#[test]
fn nonsymmetric_damping_passes_determinant_and_pairing_checks() {
    let mass = csr_from_dense(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    let damping = csr_from_dense(&[vec![0.2, 0.5], vec![-0.3, 0.1]]);
    let stiffness = csr_from_dense(&[vec![5.0, -1.0], vec![-1.0, 3.0]]);
    let config = QuadraticEigenConfig {
        target: Complex64::new(0.0, 3.0),
        num_eigenvalues: 4,
        krylov_dimension: 4,
        tolerance: 1.0e-10,
        max_qr_iterations: 4_000,
        breakdown_tolerance: 1.0e-14,
    };
    let result = quadratic_eigen_damped(&mass, &damping, &stiffness, &config).expect("valid QEP");
    assert!(result.converged);
    assert_eq!(result.eigenvalues.len(), 4);
    assert!(
        result
            .residual_norms
            .iter()
            .all(|residual| *residual < 1.0e-8)
    );

    let mass_norm = 2.0f64.sqrt();
    let damping_norm = (0.04 + 0.25 + 0.09 + 0.01f64).sqrt();
    let stiffness_norm = (25.0 + 1.0 + 1.0 + 9.0f64).sqrt();
    for &lambda in &result.eigenvalues {
        assert!(lambda.re < 0.0, "stable system must decay: {lambda}");
        let l2 = lambda * lambda;
        // det(l^2 M + l C + K) for the 2x2 case, evaluated directly.
        let d00 = l2 + lambda * 0.2 + 5.0;
        let d01 = lambda * 0.5 - 1.0;
        let d10 = lambda * (-0.3) - 1.0;
        let d11 = l2 + lambda * 0.1 + 3.0;
        let determinant = d00 * d11 - d01 * d10;
        let scale = l2.norm() * mass_norm + lambda.norm() * damping_norm + stiffness_norm;
        assert!(
            determinant.norm() < 1.0e-8 * scale * scale,
            "lambda = {lambda}, det = {determinant}"
        );
        // Real inputs: non-real eigenvalues arrive in conjugate pairs.
        if lambda.im.abs() > 1.0e-9 {
            let partner = result
                .eigenvalues
                .iter()
                .any(|mu| (mu.re - lambda.re).hypot(mu.im + lambda.im) < 1.0e-6);
            assert!(partner, "missing conjugate partner for {lambda}");
        }
    }
}

#[test]
fn invalid_quadratic_inputs_are_rejected() {
    let two = csr_from_dense(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    let three = csr_from_dense(&[
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let config = QuadraticEigenConfig::default();
    assert!(quadratic_eigen_damped(&two, &two, &three, &config).is_err());

    let non_square = CsrMatrix {
        num_rows: 2,
        num_cols: 3,
        values: Vec::new(),
        col_indices: Vec::new(),
        row_ptrs: vec![0, 0, 0],
    };
    assert!(quadratic_eigen_damped(&non_square, &two, &two, &config).is_err());

    // Built manually: TripletMatrix::to_csr rejects NaN before we get here.
    let non_finite = CsrMatrix {
        num_rows: 2,
        num_cols: 2,
        values: vec![f64::NAN, 1.0],
        col_indices: vec![0, 1],
        row_ptrs: vec![0, 1, 2],
    };
    assert!(quadratic_eigen_damped(&non_finite, &two, &two, &config).is_err());

    for count in [0, 5] {
        let bad_count = QuadraticEigenConfig {
            num_eigenvalues: count,
            ..QuadraticEigenConfig::default()
        };
        assert!(quadratic_eigen_damped(&two, &two, &two, &bad_count).is_err());
    }

    let bad_krylov = QuadraticEigenConfig {
        krylov_dimension: 0,
        ..QuadraticEigenConfig::default()
    };
    assert!(quadratic_eigen_damped(&two, &two, &two, &bad_krylov).is_err());
}
