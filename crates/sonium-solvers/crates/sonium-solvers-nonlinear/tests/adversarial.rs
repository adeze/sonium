//! Adversarial nonlinear-solver coverage: singular Jacobians, stiff value
//! scaling, complex line search, non-contractive Anderson maps, and the
//! `depth = 0` damped-fixed-point mode.

use num_complex::Complex64;
use sonium_solvers_nonlinear::{
    AndersonConfig, AndersonError, LineSearch, NewtonConfig, NewtonError, anderson_acceleration,
    newton_raphson,
};

fn newton_config(line_search: LineSearch) -> NewtonConfig {
    NewtonConfig {
        max_iterations: 100,
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-8,
        damping: 1.0,
        line_search,
        verbosity: 0,
    }
}

#[test]
fn newton_converges_linearly_on_singular_at_root_jacobian() {
    // F(x) = x^2: J = 2x is singular at the root. Newton halves the error
    // per step (linear, not quadratic) but must still converge — never stall
    // or error on the exact singularity, which is only reached in the limit.
    let residual = |x: &[f64]| vec![x[0] * x[0]];
    let solve =
        |x: &[f64], rhs: &[f64]| -> Result<Vec<f64>, String> { Ok(vec![rhs[0] / (2.0 * x[0])]) };
    let solution = newton_raphson(&[0.5], residual, solve, &newton_config(LineSearch::None))
        .expect("singular-at-root scalar Newton must converge");
    assert!(solution.converged);
    // Relative tolerance 1e-8 against |F(x0)| = 0.25 binds first: |F| < 2.5e-9
    // stops the linear-convergence halving at x ~ 5e-5 (never quadratic
    // near the singularity, but convergent).
    assert!(solution.x[0].abs() < 1.0e-4);
}

#[test]
fn newton_maps_singular_initial_jacobian_to_typed_error() {
    // Starting exactly at the singularity: the J-solve callback reports
    // failure and Newton must surface LinearSolveFailed — never panic.
    let residual = |x: &[f64]| vec![x[0] * x[0] - 2.0];
    let solve = |x: &[f64], _rhs: &[f64]| -> Result<Vec<f64>, String> {
        if x[0] == 0.0 {
            return Err("Jacobian exactly singular at x = 0".to_string());
        }
        Ok(vec![_rhs[0] / (2.0 * x[0])])
    };
    let error = newton_raphson(&[0.0], residual, solve, &newton_config(LineSearch::None))
        .expect_err("singular initial Jacobian must fail loudly");
    assert!(
        matches!(error, NewtonError::LinearSolveFailed { .. }),
        "unexpected error: {error:?}"
    );
}

#[test]
fn newton_is_robust_to_stiff_value_scaling() {
    // Twelve decades of value scaling with an exact J-solve: the residual
    // norms span 1e-6..1e6, but one full Newton step still lands on the root.
    let residual = |x: &[f64]| vec![1.0e6 * x[0] - 2.0, 1.0e-6 * x[1] + 3.0];
    let solve = |_: &[f64], rhs: &[f64]| -> Result<Vec<f64>, String> {
        Ok(vec![rhs[0] / 1.0e6, rhs[1] / 1.0e-6])
    };
    let solution = newton_raphson(
        &[0.0, 0.0],
        residual,
        solve,
        &newton_config(LineSearch::None),
    )
    .expect("scaled linear problem must solve");
    assert!(solution.converged);
    assert!((solution.x[0] - 2.0e-6).abs() < 1.0e-12);
    assert!((solution.x[1] + 3.0e6).abs() < 1.0);
}

#[test]
fn newton_solves_complex_problem_with_backtracking() {
    // F(z) = z^2 + 1, roots at ±i. Complex Armijo line search from a
    // non-real start must converge to one of them.
    let residual = |x: &[Complex64]| vec![x[0] * x[0] + Complex64::new(1.0, 0.0)];
    let solve = |x: &[Complex64], rhs: &[Complex64]| -> Result<Vec<Complex64>, String> {
        Ok(vec![rhs[0] / (Complex64::new(2.0, 0.0) * x[0])])
    };
    let config = newton_config(LineSearch::Backtracking {
        c1: 1.0e-4,
        max_halvings: 20,
    });
    let solution = newton_raphson(&[Complex64::new(0.5, 0.5)], residual, solve, &config)
        .expect("complex Newton with line search must converge");
    assert!(solution.converged);
    let root = solution.x[0];
    let residual_norm = (root * root + Complex64::new(1.0, 0.0)).norm();
    assert!(residual_norm < 1.0e-9, "residual {residual_norm:.3e}");
    assert!(
        (root.im.abs() - 1.0).abs() < 1.0e-8 && root.re.abs() < 1.0e-8,
        "expected ±i, got {root:?}"
    );
}

fn anderson_config(depth: usize, safeguarding: bool) -> AndersonConfig {
    AndersonConfig {
        depth,
        mixing: 1.0,
        max_iterations: 200,
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-8,
        safeguarding,
        verbosity: 0,
    }
}

#[test]
fn anderson_depth_zero_is_damped_fixed_point() {
    // depth = 0 disables the affine combination: plain Picard iteration on
    // cos, converging to the Dottie number.
    let solution = anderson_acceleration(
        &[0.5],
        |x: &[f64]| vec![x[0].cos()],
        &anderson_config(0, false),
    )
    .expect("Picard on a contraction must converge");
    assert!(solution.converged);
    assert!(
        (solution.x[0] - 0.739_085_133_215_160_7).abs() < 1.0e-6,
        "expected the Dottie number, got {:?}",
        solution.x
    );
}

#[test]
fn anderson_terminates_loudly_on_expansive_map() {
    // g(x) = x^2 + 0.5 has no real fixed point (x^2 - x + 0.5 = 0 has
    // negative discriminant) and is nonlinear, so Anderson cannot solve it
    // exactly the way it does linear maps: iterates escape to infinity.
    // Both safeguarding modes must terminate with a typed error — never hang
    // or overflow into a panic.
    for safeguarding in [false, true] {
        let error = anderson_acceleration(
            &[0.5],
            |x: &[f64]| vec![x[0] * x[0] + 0.5],
            &anderson_config(3, safeguarding),
        )
        .expect_err("expansive map must not converge");
        assert!(
            matches!(
                error,
                AndersonError::NotConverged { .. } | AndersonError::Diverged { .. }
            ),
            "safeguarding={safeguarding}: unexpected error {error:?}"
        );
    }
}
