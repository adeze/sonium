use sonium_solvers_core::traits::LinearOperator;
use sonium_solvers_iterative::{GmresConfig, gmres};
use sonium_solvers_sparse::CsrMatrix;

/// Restarted GMRES on an ill-conditioned system must report the true
/// original-system residual, not a projected estimate that diverged across
/// restarts. The Hilbert matrix (cond(H_8) ~ 1.5e10) with a tiny restart
/// forces the restart path while making fast convergence impossible, so the
/// test holds regardless of whether the tolerance is attained.
#[test]
fn restarted_gmres_reports_true_residual_on_hilbert_system() {
    let n = 8usize;
    let mut triplets = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            triplets.push((i, j, 1.0_f64 / ((i + j + 1) as f64)));
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, triplets);
    let rhs = vec![1.0_f64; n];
    let config = GmresConfig {
        max_iterations: 10,
        restart: 2,
        tolerance: 1e-12,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        ..Default::default()
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(
        solution.restarts > 0,
        "tiny restart must exercise the restart path"
    );

    let ax = matrix.apply(&solution.x);
    let residual_norm = ax
        .iter()
        .zip(rhs.iter())
        .map(|(&actual, &expected)| {
            let difference = expected - actual;
            difference * difference
        })
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    let expected_relative_residual = residual_norm / rhs_norm;

    assert!(
        (solution.residual - expected_relative_residual).abs() < 1e-12,
        "reported residual {} must match recomputed true residual {expected_relative_residual}",
        solution.residual,
    );
    assert_eq!(
        solution.converged,
        expected_relative_residual <= config.tolerance
    );
}
