use sonium_solvers_eigen::{LobpcgConfig, lobpcg};
use sonium_solvers_sparse::CsrMatrix;

/// LOBPCG on a spectrum with a triple cluster must return the complete
/// cluster as a multiset with an invariant subspace, not one eigenvector
/// compared at a time. Eigenvector-by-eigenvector comparison is misleading
/// for degenerate modes: any orthonormal basis of the cluster eigenspace is
/// correct, so this test certifies the eigenvalue multiset, per-pair
/// residuals recomputed from the original matrix, and subspace containment
/// without pairing vectors.
#[test]
fn lobpcg_recovers_complete_degenerate_cluster_as_invariant_subspace() {
    let spectrum = [1.0_f64, 1.0, 1.0, 2.0, 3.0, 5.0];
    let n = spectrum.len();
    let matrix = CsrMatrix::from_triplets(
        n,
        n,
        spectrum
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, index, value))
            .collect(),
    );
    let config = LobpcgConfig {
        num_eigenvalues: 3,
        max_iterations: 200,
        tolerance: 1e-10,
        ..Default::default()
    };

    let result = lobpcg(&matrix, &config).expect("clustered diagonal LOBPCG must succeed");
    assert!(result.converged, "triple cluster must converge");
    assert_eq!(result.eigenvalues.len(), 3);
    assert_eq!(result.residual_norms.len(), 3);

    // Completeness as a multiset: the three smallest eigenvalues are all 1.0.
    let mut values = result.eigenvalues.clone();
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite eigenvalues"));
    for (index, &value) in values.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < 1e-8,
            "eigenvalue {index} = {value} is outside the triple cluster at 1.0",
        );
    }

    // Per-pair residuals recomputed from the original matrix.
    let vectors = result.eigenvectors.as_slice();
    for k in 0..3 {
        let mut residual_sq = 0.0;
        for i in 0..n {
            let mut ax = 0.0;
            for j in 0..n {
                let a_ij = if i == j { spectrum[i] } else { 0.0 };
                ax += a_ij * vectors[j * 3 + k];
            }
            let difference = ax - result.eigenvalues[k] * vectors[i * 3 + k];
            residual_sq += difference * difference;
        }
        let recomputed = residual_sq.sqrt();
        assert!(
            recomputed < 1e-8,
            "recomputed residual {recomputed} of pair {k} is too large",
        );
        assert!(
            (result.residual_norms[k] - recomputed).abs() < 1e-8,
            "reported residual {} of pair {k} disagrees with recomputed {recomputed}",
            result.residual_norms[k],
        );
    }

    // Invariant subspace: the cluster eigenspace is span(e0, e1, e2), so every
    // computed vector must have negligible components outside rows 0..3.
    let mut outside_sq = 0.0;
    for k in 0..3 {
        for i in 3..n {
            let component = vectors[i * 3 + k];
            outside_sq += component * component;
        }
    }
    assert!(
        outside_sq.sqrt() < 1e-8,
        "cluster vectors leak outside the degenerate eigenspace: {outside_sq}",
    );
}
