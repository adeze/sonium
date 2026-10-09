//! Deterministic active/inactive KKT property matrix for constrained optimizers.

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_optimization::{
    AdvancedSqpConfig, Bound, Constraint, ConstraintKind, LbfgsbConfig, OptimResult, SlsqpConfig,
    SqpConfig, lbfgsb, slsqp, sqp, sqp_advanced,
};

#[derive(Clone, Copy)]
enum ConstraintCase {
    Equality,
    ActiveInequality,
    InactiveInequality,
}

#[derive(Clone, Copy, Debug)]
enum MultidimensionalConstraintCase {
    Equality,
    ActiveInequality,
    InactiveInequality,
}

fn constraint(case: ConstraintCase) -> Constraint<f64> {
    let (kind, slope, offset) = match case {
        ConstraintCase::Equality => (ConstraintKind::Equality, 1.0, -1.0),
        ConstraintCase::ActiveInequality => (ConstraintKind::Inequality, -1.0, 1.0),
        ConstraintCase::InactiveInequality => (ConstraintKind::Inequality, -1.0, 1.0),
    };
    Constraint {
        kind,
        dim: 1,
        fun: Box::new(move |x: &[f64]| vec![slope * x[0] + offset]),
        jac: Some(Box::new(move |_| {
            DenseMatrix::from_row_major_vec(1, 1, vec![slope])
        })),
    }
}

fn objective(target: f64, x: &[f64]) -> f64 {
    (x[0] - target).powi(2)
}

fn gradient(target: f64, x: &[f64]) -> Vec<f64> {
    vec![2.0 * (x[0] - target)]
}

fn assert_independent_kkt(
    result: &OptimResult<f64>,
    case: ConstraintCase,
    target: f64,
    tolerance: f64,
) {
    assert!(result.converged, "optimizer did not converge: {result:?}");
    let report = result
        .kkt
        .as_ref()
        .expect("successful result must carry KKT report");
    let x = result.x[0];
    let g = 2.0 * (x - target);
    let (kind, value, jacobian) = match case {
        ConstraintCase::Equality => (ConstraintKind::Equality, x - 1.0, 1.0),
        ConstraintCase::ActiveInequality | ConstraintCase::InactiveInequality => {
            (ConstraintKind::Inequality, 1.0 - x, -1.0)
        }
    };
    let multiplier = report.multipliers.constraints[0];
    let primal = match kind {
        ConstraintKind::Equality => value.abs(),
        ConstraintKind::Inequality => (-value).max(0.0),
    };
    let dual = match kind {
        ConstraintKind::Equality => 0.0,
        ConstraintKind::Inequality => (-multiplier).max(0.0),
    };
    let stationarity = (g - multiplier * jacobian).abs();
    let complementarity = (multiplier * value).abs();
    assert!(
        primal <= tolerance,
        "independent primal residual {primal:e}"
    );
    assert!(dual <= tolerance, "independent dual residual {dual:e}");
    assert!(
        stationarity <= tolerance,
        "independent stationarity {stationarity:e}"
    );
    assert!(
        complementarity <= tolerance,
        "independent complementarity {complementarity:e}"
    );
    assert!(report.norm.is_finite() && report.norm <= tolerance);

    match case {
        ConstraintCase::Equality => assert!(multiplier.abs() > 1.0e-3),
        ConstraintCase::ActiveInequality => assert!(multiplier > 1.0e-3),
        ConstraintCase::InactiveInequality => assert!(multiplier.abs() <= tolerance),
    }
}

fn run_sqp_case(case: ConstraintCase, target: f64) {
    let objective_fn = |x: &[f64]| objective(target, x);
    let gradient_fn = |x: &[f64]| gradient(target, x);
    let config = SqpConfig {
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        constraint_tolerance: 1.0e-9,
        ..SqpConfig::default()
    };
    let result = sqp(
        &objective_fn,
        &gradient_fn,
        &[0.0],
        &[constraint(case)],
        &config,
    )
    .expect("SQP case");
    assert_independent_kkt(&result, case, target, 2.0e-5);
}

fn run_advanced_sqp_case(case: ConstraintCase, target: f64) {
    let objective_fn = |x: &[f64]| objective(target, x);
    let gradient_fn = |x: &[f64]| gradient(target, x);
    let hessian_fn = |_: &[f64], _: &[f64]| DenseMatrix::from_row_major_vec(1, 1, vec![2.0]);
    let config = AdvancedSqpConfig {
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        constraint_tolerance: 1.0e-9,
        ..AdvancedSqpConfig::default()
    };
    let result = sqp_advanced(
        &objective_fn,
        &gradient_fn,
        &hessian_fn,
        &[0.0],
        &[constraint(case)],
        &config,
    )
    .expect("advanced SQP case");
    assert_independent_kkt(&result, case, target, 2.0e-5);
}

fn run_slsqp_case(case: ConstraintCase, target: f64) {
    let objective_fn = |x: &[f64]| objective(target, x);
    let gradient_fn = |x: &[f64]| gradient(target, x);
    let config = SlsqpConfig {
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        constraint_tolerance: 1.0e-9,
        ..SlsqpConfig::default()
    };
    let result = slsqp(
        &objective_fn,
        &gradient_fn,
        &[0.0],
        &[constraint(case)],
        &config,
    )
    .expect("SLSQP case");
    assert_independent_kkt(&result, case, target, 2.0e-5);
}

fn multidimensional_constraint(case: MultidimensionalConstraintCase) -> Constraint<f64> {
    let (kind, coefficients, offset) = match case {
        MultidimensionalConstraintCase::Equality => (ConstraintKind::Equality, [1.0, 2.0], -1.0),
        MultidimensionalConstraintCase::ActiveInequality => {
            (ConstraintKind::Inequality, [1.0, 0.0], -3.0)
        }
        MultidimensionalConstraintCase::InactiveInequality => {
            (ConstraintKind::Inequality, [1.0, 0.0], 1.0)
        }
    };
    Constraint {
        kind,
        dim: 1,
        fun: Box::new(move |x: &[f64]| {
            vec![coefficients[0] * x[0] + coefficients[1] * x[1] + offset]
        }),
        jac: Some(Box::new(move |_| {
            DenseMatrix::from_row_major_vec(1, 2, coefficients.to_vec())
        })),
    }
}

fn multidimensional_objective(x: &[f64]) -> f64 {
    (x[0] - 2.0).powi(2) + 4.0 * (x[1] + 1.0).powi(2)
}

fn multidimensional_gradient(x: &[f64]) -> Vec<f64> {
    vec![2.0 * (x[0] - 2.0), 8.0 * (x[1] + 1.0)]
}

fn assert_multidimensional_kkt(
    result: &OptimResult<f64>,
    case: MultidimensionalConstraintCase,
    tolerance: f64,
) {
    assert!(
        result.converged,
        "multidimensional optimizer did not converge: {result:?}"
    );
    let report = result
        .kkt
        .as_ref()
        .expect("successful multidimensional result must carry KKT report");
    let x = &result.x;
    let gradient = multidimensional_gradient(x);
    let (value, jacobian) = match case {
        MultidimensionalConstraintCase::Equality => (x[0] + 2.0 * x[1] - 1.0, [1.0, 2.0]),
        MultidimensionalConstraintCase::ActiveInequality => (x[0] - 3.0, [1.0, 0.0]),
        MultidimensionalConstraintCase::InactiveInequality => (x[0] + 1.0, [1.0, 0.0]),
    };
    let multiplier = report.multipliers.constraints[0];
    let primal = match case {
        MultidimensionalConstraintCase::Equality => value.abs(),
        MultidimensionalConstraintCase::ActiveInequality
        | MultidimensionalConstraintCase::InactiveInequality => (-value).max(0.0),
    };
    let dual = match case {
        MultidimensionalConstraintCase::Equality => 0.0,
        MultidimensionalConstraintCase::ActiveInequality
        | MultidimensionalConstraintCase::InactiveInequality => (-multiplier).max(0.0),
    };
    let stationarity = gradient
        .iter()
        .zip(jacobian)
        .map(|(&g, j)| (g - multiplier * j).abs())
        .fold(0.0, f64::max);
    let complementarity = (multiplier * value).abs();
    let independent = primal.max(dual).max(stationarity).max(complementarity);
    assert!(
        independent <= tolerance,
        "independent KKT residual {independent:e}"
    );
    assert!(report.norm.is_finite() && report.norm <= tolerance);
    match case {
        MultidimensionalConstraintCase::Equality => assert!(multiplier.abs() > 1.0e-3),
        MultidimensionalConstraintCase::ActiveInequality => assert!(
            multiplier > 1.0e-3,
            "active inequality multiplier {multiplier:e} for {case:?}, x={x:?}"
        ),
        MultidimensionalConstraintCase::InactiveInequality => {
            assert!(multiplier.abs() <= tolerance)
        }
    }
}

fn run_multidimensional_sqp_case(case: MultidimensionalConstraintCase) {
    let objective_fn = multidimensional_objective;
    let gradient_fn = multidimensional_gradient;
    let config = SqpConfig {
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        constraint_tolerance: 1.0e-9,
        ..SqpConfig::default()
    };
    let result = sqp(
        &objective_fn,
        &gradient_fn,
        &[0.0, 0.0],
        &[multidimensional_constraint(case)],
        &config,
    )
    .expect("multidimensional SQP case");
    assert_multidimensional_kkt(&result, case, 2.0e-5);
}

fn run_multidimensional_advanced_sqp_case(case: MultidimensionalConstraintCase) {
    let hessian_fn =
        |_: &[f64], _: &[f64]| DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 0.0, 0.0, 8.0]);
    let config = AdvancedSqpConfig {
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        constraint_tolerance: 1.0e-9,
        ..AdvancedSqpConfig::default()
    };
    let result = sqp_advanced(
        &multidimensional_objective,
        &multidimensional_gradient,
        &hessian_fn,
        &[0.0, 0.0],
        &[multidimensional_constraint(case)],
        &config,
    )
    .expect("multidimensional advanced SQP case");
    assert_multidimensional_kkt(&result, case, 2.0e-5);
}

fn run_multidimensional_slsqp_case(case: MultidimensionalConstraintCase) {
    let config = SlsqpConfig {
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        constraint_tolerance: 1.0e-9,
        ..SlsqpConfig::default()
    };
    let result = slsqp(
        &multidimensional_objective,
        &multidimensional_gradient,
        &[0.0, 0.0],
        &[multidimensional_constraint(case)],
        &config,
    )
    .expect("multidimensional SLSQP case");
    assert_multidimensional_kkt(&result, case, 2.0e-5);
}

#[test]
fn native_multiplier_kkt_matrix_covers_active_and_inactive_constraints() {
    let cases = [
        (ConstraintCase::Equality, 2.0),
        (ConstraintCase::ActiveInequality, 2.0),
        (ConstraintCase::InactiveInequality, 0.25),
    ];
    for (case, target) in cases {
        run_sqp_case(case, target);
        run_advanced_sqp_case(case, target);
        run_slsqp_case(case, target);
    }
}

#[test]
fn native_bound_multiplier_kkt_matrix_covers_active_bound() {
    let objective_fn = |x: &[f64]| objective(2.0, x);
    let gradient_fn = |x: &[f64]| gradient(2.0, x);
    let config = LbfgsbConfig {
        bounds: vec![Bound::upper(1.0)],
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        ..LbfgsbConfig::default()
    };
    let result = lbfgsb(&objective_fn, &gradient_fn, &[0.0], &config).expect("L-BFGS-B case");
    assert!(result.converged, "L-BFGS-B did not converge: {result:?}");
    let report = result
        .kkt
        .as_ref()
        .expect("bound result must carry KKT report");
    assert_eq!(report.diagnostics.active_upper_bounds, vec![0]);
    assert!(report.multipliers.upper_bounds[0] > 1.0e-3);
    assert!(report.norm.is_finite() && report.norm <= 2.0e-5);
}

fn next_unit(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 11) as f64) / ((1_u64 << 53) as f64)
}

#[test]
fn seeded_random_scaled_bound_kkt_matrix() {
    let mut seed = 0x8f3a_2d19_7c55_1041_u64;
    for case in 0..8 {
        let scale0 = 0.125 + 15.875 * next_unit(&mut seed);
        let scale1 = 0.125 + 15.875 * next_unit(&mut seed);
        let target0 = 0.75 + 2.0 * next_unit(&mut seed);
        let target1 = -0.75 - 2.0 * next_unit(&mut seed);
        let objective =
            |x: &[f64]| scale0 * (x[0] - target0).powi(2) + scale1 * (x[1] - target1).powi(2);
        let gradient = |x: &[f64]| {
            vec![
                2.0 * scale0 * (x[0] - target0),
                2.0 * scale1 * (x[1] - target1),
            ]
        };
        let config = LbfgsbConfig {
            bounds: vec![Bound::upper(0.5), Bound::lower(-0.25)],
            max_iterations: 250,
            grad_tolerance: 1.0e-9,
            ..LbfgsbConfig::default()
        };
        let result = lbfgsb(&objective, &gradient, &[0.0, 0.0], &config)
            .unwrap_or_else(|error| panic!("seeded case {case} failed: {error}"));
        assert!(
            result.converged,
            "seeded case {case} did not converge: {result:?}"
        );
        assert!((result.x[0] - 0.5).abs() <= 3.0e-5);
        assert!((result.x[1] + 0.25).abs() <= 3.0e-5);
        let report = result
            .kkt
            .as_ref()
            .unwrap_or_else(|| panic!("seeded case {case} missing KKT report"));
        assert_eq!(report.diagnostics.active_upper_bounds, vec![0]);
        assert_eq!(report.diagnostics.active_lower_bounds, vec![1]);
        assert!(report.norm.is_finite() && report.norm <= 3.0e-5);
        let stationarity = (gradient(&result.x)[0] + report.multipliers.upper_bounds[0])
            .abs()
            .max((gradient(&result.x)[1] - report.multipliers.lower_bounds[1]).abs());
        assert!(stationarity.is_finite() && stationarity <= 3.0e-5);
    }
}

#[test]
fn native_multiplier_kkt_matrix_covers_scaled_multidimensional_cases() {
    let cases = [
        MultidimensionalConstraintCase::Equality,
        MultidimensionalConstraintCase::ActiveInequality,
        MultidimensionalConstraintCase::InactiveInequality,
    ];
    for case in cases {
        run_multidimensional_sqp_case(case);
        run_multidimensional_advanced_sqp_case(case);
        run_multidimensional_slsqp_case(case);
    }
}

#[test]
fn seeded_scaled_lbfgsb_active_bounds_keep_kkt_gate() {
    // These fixtures span three conditionings while retaining an analytic
    // active-bound solution.  The objective and gradient are independent
    // oracles; only the returned KKT report is used for the gate.
    let cases = [
        (0.25, 4.0, [1.5, -1.5]),
        (1.0, 1.0, [1.25, -1.0]),
        (4.0, 0.25, [2.0, -2.0]),
        (16.0, 0.0625, [1.75, -1.25]),
    ];

    for (scale0, scale1, target) in cases {
        let objective = move |x: &[f64]| {
            scale0 * (x[0] - target[0]).powi(2) + scale1 * (x[1] - target[1]).powi(2)
        };
        let gradient = move |x: &[f64]| {
            vec![
                2.0 * scale0 * (x[0] - target[0]),
                2.0 * scale1 * (x[1] - target[1]),
            ]
        };
        let config = LbfgsbConfig {
            bounds: vec![Bound::upper(0.5), Bound::lower(-0.25)],
            max_iterations: 200,
            grad_tolerance: 1.0e-9,
            ..LbfgsbConfig::default()
        };
        let result = lbfgsb(&objective, &gradient, &[0.0, 0.0], &config)
            .expect("seeded/scaled L-BFGS-B case");
        assert!(result.converged, "L-BFGS-B did not converge: {result:?}");
        assert_eq!(result.x.len(), 2);
        assert!((result.x[0] - 0.5).abs() <= 2.0e-5);
        assert!((result.x[1] + 0.25).abs() <= 2.0e-5);
        let report = result
            .kkt
            .as_ref()
            .expect("successful bound solve must carry KKT report");
        assert_eq!(report.diagnostics.active_upper_bounds, vec![0]);
        assert_eq!(report.diagnostics.active_lower_bounds, vec![1]);
        assert!(report.norm.is_finite() && report.norm <= 2.0e-5);
        assert!(report.multipliers.upper_bounds[0] > 1.0e-3);
        assert!(report.multipliers.lower_bounds[1] > 1.0e-3);
    }
}

#[test]
fn native_bound_multiplier_kkt_matrix_covers_scaled_multidimensional_bound() {
    let config = LbfgsbConfig {
        bounds: vec![Bound::upper(1.0), Bound::unbounded()],
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        ..LbfgsbConfig::default()
    };
    let result = lbfgsb(
        &multidimensional_objective,
        &multidimensional_gradient,
        &[0.0, 0.0],
        &config,
    )
    .expect("multidimensional L-BFGS-B bound case");
    assert!(
        result.converged,
        "multidimensional L-BFGS-B did not converge: {result:?}"
    );
    let report = result
        .kkt
        .as_ref()
        .expect("bound result must carry KKT report");
    assert_eq!(report.diagnostics.active_upper_bounds, vec![0]);
    assert!(report.multipliers.upper_bounds[0] > 1.0e-3);
    assert!(report.norm.is_finite() && report.norm <= 2.0e-5);
}
