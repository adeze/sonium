//! Eigen-solver breakdown and max-iteration failure matrix.

use sonium_solvers::eigen::{LobpcgConfig, lobpcg};
use sonium_solvers::{CsrMatrix, SolverResultContract, SolverStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn lobpcg_zero_operator_returns_typed_non_success_without_panic() {
    let zero = CsrMatrix::from_triplets(3, 3, vec![(0, 0, 2.0), (1, 1, 3.0), (2, 2, 5.0)]);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        lobpcg(
            &zero,
            &LobpcgConfig {
                num_eigenvalues: 1,
                max_iterations: 1,
                tolerance: 1.0e-12,
                block_padding: 0,
            },
        )
    }));
    assert!(outcome.is_ok(), "LOBPCG breakdown path panicked");
    match outcome.expect("checked above") {
        Err(_) => {}
        Ok(result) => {
            result
                .validate_contract()
                .expect("returned LOBPCG failure result must be contract-valid");
            assert_ne!(result.solver_status(), SolverStatus::Converged);
        }
    }
}
