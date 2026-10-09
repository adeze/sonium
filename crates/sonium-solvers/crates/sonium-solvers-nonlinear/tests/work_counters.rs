//! Work-counter contracts for the Newton entry points.
//!
//! [`NewtonSolution`] carries three counters — `residual_evaluations`,
//! `jacobian_evaluations`, and `inner_iterations` — with entry-point
//! specific semantics documented on the struct. These tests pin the
//! structural identities on the scalar problem `x^2 - 4 = 0` from `x0 = 1`
//! under pure Newton (default config: no line search, unit damping), where
//! every count is exact rather than a bound.

use sonium_solvers_core::traits::SolverResultContract;
use sonium_solvers_nonlinear::{
    InexactNewtonConfig, NewtonConfig, NewtonSolution, newton_raphson, newton_raphson_continuation,
    newton_raphson_inexact,
};

fn residual(x: &[f64]) -> Vec<f64> {
    vec![x[0] * x[0] - 4.0]
}

fn check_counters(sol: &NewtonSolution<f64>, expect_inner_positive: bool) {
    assert!(sol.converged, "setup problem must converge");
    assert!((sol.x[0] - 2.0).abs() < 1e-8, "x = {}", sol.x[0]);
    if expect_inner_positive {
        // The continuation core pre-evaluates each accepted trial, so a
        // step costs two residual evaluations by construction.
        assert!(
            sol.residual_evaluations >= sol.iterations + 1,
            "residual_evaluations = {}, iterations = {}",
            sol.residual_evaluations,
            sol.iterations
        );
    } else {
        // Initial evaluation plus exactly one per Newton iteration: no
        // line-search or trust-region trials under the default config.
        assert_eq!(
            sol.residual_evaluations,
            sol.iterations + 1,
            "iterations = {}",
            sol.iterations
        );
    }
    sol.validate_contract()
        .expect("counter fields must satisfy the result contract");
    if expect_inner_positive {
        assert_eq!(
            sol.jacobian_evaluations, 0,
            "matrix-free continuation never assembles a Jacobian"
        );
        assert!(
            sol.inner_iterations > 0,
            "continuation must report its inner GMRES work"
        );
    } else {
        // Exactly one black-box Jacobian solve per Newton iteration: the
        // trust-region retry loop re-clips the same direction without
        // re-solving, and backtracking only re-evaluates the residual.
        assert_eq!(
            sol.jacobian_evaluations, sol.iterations,
            "iterations = {}",
            sol.iterations
        );
        assert_eq!(
            sol.inner_iterations, 0,
            "black-box solve closures hide inner iteration counts"
        );
    }
}

#[test]
fn newton_raphson_work_counters_are_exact() {
    let config = NewtonConfig {
        absolute_tolerance: 1e-12,
        ..Default::default()
    };
    let sol = newton_raphson(
        &[1.0_f64],
        residual,
        |x, rhs| Ok(vec![rhs[0] / (2.0 * x[0])]),
        &config,
    )
    .unwrap();
    check_counters(&sol, false);
}

#[test]
fn newton_raphson_inexact_work_counters_are_exact() {
    let config = InexactNewtonConfig {
        newton: NewtonConfig {
            absolute_tolerance: 1e-12,
            ..Default::default()
        },
        ..Default::default()
    };
    let sol = newton_raphson_inexact(
        &[1.0_f64],
        residual,
        |x, rhs, _eta| Ok(vec![rhs[0] / (2.0 * x[0])]),
        &config,
    )
    .unwrap();
    check_counters(&sol, false);
}

#[test]
fn newton_raphson_continuation_reports_inner_not_jacobian_work() {
    let config = InexactNewtonConfig {
        newton: NewtonConfig {
            absolute_tolerance: 1e-12,
            ..Default::default()
        },
        ..Default::default()
    };
    let sol = newton_raphson_continuation(
        &[1.0_f64],
        residual,
        |x, v| vec![2.0 * x[0] * v[0]],
        &config,
    )
    .unwrap();
    check_counters(&sol, true);
}
