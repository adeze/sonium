use sonium_solvers_eigen::{LobpcgConfig, lobpcg};
use sonium_solvers_sparse::CsrMatrix;

fn diagonal(values: &[f64]) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        values.len(),
        values.len(),
        values
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, index, value))
            .collect(),
    )
}

/// Mode tracking through an eigenvalue crossing. For `A(t) = diag(1-t, 1+t,
/// 3)` the two lowest branches cross at `t = 0`: which physical mode is
/// smallest swaps sides. An eigensolver that pairs vectors by continuity, or
/// that collapses near-degenerate modes onto one branch, reports a wrong set
/// on one side. The oracle compares sorted eigenvalue sets (pairing by value,
/// robust to the swap) on both sides of the crossing and requires both
/// branches present with small recomputed residuals.
#[test]
fn lobpcg_tracks_both_branches_through_eigenvalue_crossing() {
    for t in [-0.05_f64, 0.05] {
        let spectrum = [1.0 - t, 1.0 + t, 3.0];
        let matrix = diagonal(&spectrum);
        let config = LobpcgConfig {
            num_eigenvalues: 2,
            max_iterations: 200,
            tolerance: 1e-10,
            ..Default::default()
        };

        let result = lobpcg(&matrix, &config).expect("crossing LOBPCG must succeed");
        assert!(result.converged, "crossing solve at t = {t} must converge");

        let mut values = result.eigenvalues.clone();
        values.sort_by(|a, b| a.partial_cmp(b).expect("finite eigenvalues"));
        let mut expected = [1.0 - t.abs(), 1.0 + t.abs()];
        expected.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        for (k, (&got, &want)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "t = {t}: branch {k} = {got}, expected {want} (crossing branch lost)",
            );
        }

        let vectors = result.eigenvectors.as_slice();
        for k in 0..2 {
            let mut residual_sq = 0.0;
            for i in 0..3 {
                let mut ax = 0.0;
                for j in 0..3 {
                    let a_ij = if i == j { spectrum[i] } else { 0.0 };
                    ax += a_ij * vectors[j * 2 + k];
                }
                let difference = ax - result.eigenvalues[k] * vectors[i * 2 + k];
                residual_sq += difference * difference;
            }
            assert!(
                residual_sq.sqrt() < 1e-8,
                "t = {t}: recomputed residual of pair {k} too large",
            );
        }
    }
}
