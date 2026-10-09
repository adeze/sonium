//! Repeated-mode comparison through invariant subspaces, not vector pairing.

// Rust guideline compliant 2026-02-21

use sonium_solvers_eigen::{LobpcgConfig, lobpcg_generalized, subspace_distance};
use sonium_solvers_sparse::TripletMatrix;

#[test]
fn repeated_eigenvalue_compares_subspaces_not_vectors() {
    // Diagonal spectrum with a triple eigenvalue at 1: any orthonormal basis
    // of span{e1, e2, e3} is correct, so only a subspace metric is meaningful.
    let spectrum = [1.0, 1.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
    let dimension = spectrum.len();
    let mut stiffness = TripletMatrix::<f64>::new(dimension, dimension);
    let mut mass = TripletMatrix::<f64>::new(dimension, dimension);
    for (index, &value) in spectrum.iter().enumerate() {
        stiffness.push(index, index, value);
        mass.push(index, index, 1.0);
    }
    let stiffness = stiffness.to_csr();
    let mass = mass.to_csr();
    let config = LobpcgConfig {
        num_eigenvalues: 3,
        max_iterations: 500,
        tolerance: 1.0e-10,
        block_padding: 2,
    };
    let result = lobpcg_generalized(&stiffness, &mass, &config).expect("LOBPCG must converge");
    assert!(result.converged);
    assert_eq!(result.eigenvalues.len(), 3);
    for &value in &result.eigenvalues {
        assert!((value - 1.0).abs() < 1.0e-8, "eigenvalue = {value}");
    }
    assert!(
        result
            .residual_norms
            .iter()
            .all(|residual| *residual < 1.0e-8)
    );

    let computed: Vec<Vec<f64>> = (0..3)
        .map(|j| result.eigenvectors.column_to_vec(j))
        .collect();
    let mut exact = vec![vec![0.0; dimension]; 3];
    for (basis, slot) in exact.iter_mut().enumerate() {
        slot[basis] = 1.0;
    }
    let distance = subspace_distance(&computed, &exact).expect("valid subspaces");
    assert!(distance < 1.0e-6, "subspace distance = {distance}");
}
