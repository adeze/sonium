//! Integration tests for optimization solvers
//!
//! Tests BFGS, L-BFGS, Nelder-Mead, SQP on standard test problems.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::{
    BfgsConfig, Constraint, ConstraintKind, LbfgsConfig, NelderMeadConfig, SqpConfig, bfgs, lbfgs,
    nelder_mead, sqp,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-contracts.v1";
const TEST_COUNT: f64 = 8.0;

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

fn sphere_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| 2.0 * xi).collect()
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    sum
}

fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    for i in 1..n - 1 {
        g[i] = -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) + 400.0 * (x[i] - x[i - 1] * x[i - 1])
            - 2.0 * (1.0 - x[i]);
    }
    g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
    g
}

fn test_bfgs_sphere() {
    let x0 = vec![3.0_f64, 4.0, 5.0];
    let config = BfgsConfig {
        max_iterations: 100,
        grad_tolerance: 1e-8,
        fun_tolerance: 1e-12,
        step_tolerance: 1e-12,
        line_search: sonium_solvers::LineSearchMethod::StrongWolfe,
    };

    let result = bfgs(&sphere, &sphere_grad, &x0, &config).expect("BFGS should succeed");

    assert!(result.converged, "BFGS should converge on sphere");
    assert!(result.fun < 1e-10, "Minimum should be near zero");

    let dist_from_origin = result
        .x
        .iter()
        .map(|&xi: &f64| xi.abs())
        .fold(0.0f64, f64::max);
    assert!(dist_from_origin < 1e-4, "Solution should be near origin");
}

fn test_lbfgs_sphere() {
    let x0 = vec![3.0_f64, 4.0, 5.0, 6.0];
    let config = LbfgsConfig {
        max_iterations: 100,
        grad_tolerance: 1e-8,
        fun_tolerance: 1e-12,
        step_tolerance: 1e-12,
        memory: 10,
        line_search: sonium_solvers::LineSearchMethod::StrongWolfe,
    };

    let result = lbfgs(&sphere, &sphere_grad, &x0, &config).expect("L-BFGS should succeed");

    assert!(result.converged, "L-BFGS should converge on sphere");
    assert!(result.fun < 1e-10, "Minimum should be near zero");
}

fn test_bfgs_rosenbrock() {
    let x0 = vec![-1.2_f64, 1.0];
    let config = BfgsConfig {
        max_iterations: 1000,
        grad_tolerance: 1e-6,
        fun_tolerance: 1e-12,
        step_tolerance: 1e-12,
        line_search: sonium_solvers::LineSearchMethod::StrongWolfe,
    };

    let result = bfgs(&rosenbrock, &rosenbrock_grad, &x0, &config).expect("BFGS should succeed");

    assert!(result.converged, "BFGS should converge on Rosenbrock");
    assert!(
        result.fun < 1e-8,
        "BFGS Rosenbrock objective is too large: {}",
        result.fun
    );

    let dist_to_opt = ((result.x[0] - 1.0_f64).powi(2) + (result.x[1] - 1.0_f64).powi(2)).sqrt();
    assert!(dist_to_opt < 0.1, "Solution should be near (1, 1)");
}

fn test_lbfgs_rosenbrock() {
    let x0 = vec![-1.2_f64, 1.0];
    let config = LbfgsConfig {
        max_iterations: 1000,
        grad_tolerance: 1e-6,
        fun_tolerance: 1e-12,
        step_tolerance: 1e-12,
        memory: 10,
        line_search: sonium_solvers::LineSearchMethod::StrongWolfe,
    };

    let result = lbfgs(&rosenbrock, &rosenbrock_grad, &x0, &config).expect("L-BFGS should succeed");

    assert!(result.converged, "L-BFGS should converge on Rosenbrock");
    assert!(
        result.fun < 1e-8,
        "L-BFGS Rosenbrock objective is too large: {}",
        result.fun
    );
    let dist_to_opt = ((result.x[0] - 1.0_f64).powi(2) + (result.x[1] - 1.0_f64).powi(2)).sqrt();
    assert!(dist_to_opt < 1e-3, "L-BFGS solution should be near (1, 1)");
}

fn test_nelder_mead_sphere() {
    let x0 = vec![3.0_f64, 4.0];
    let config = NelderMeadConfig {
        max_iterations: 1000,
        max_fun_evals: 10000,
        fun_tolerance: 1e-8,
        x_tolerance: 1e-8,
        adaptive: true,
        initial_simplex_size: None,
    };

    let result = nelder_mead(&sphere, &x0, &config).expect("Nelder-Mead should succeed");

    assert!(result.converged, "Nelder-Mead should converge on sphere");
    assert!(result.fun < 1e-6, "Minimum should be near zero");
}

fn test_sqp_equality_constraint() {
    let x0 = vec![1.0_f64, 1.0];

    let constraint = Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![x[0].powi(2) + x[1].powi(2) - 1.0]),
        jac: Some(Box::new(|x: &[f64]| {
            DenseMatrix::from_row_major_vec(1, 2, vec![2.0 * x[0], 2.0 * x[1]])
        })),
    };

    let config = SqpConfig {
        max_iterations: 100,
        grad_tolerance: 1e-6,
        constraint_tolerance: 1e-6,
        second_order_correction: true,
        use_multiplier_penalty: true,
        penalty_margin: 0.1,
        bounds: vec![],
        ..Default::default()
    };

    let result =
        sqp(&sphere, &sphere_grad, &x0, &[constraint], &config).expect("SQP should succeed");

    assert!(
        result.converged,
        "SQP should converge on the equality-constrained sphere"
    );
    let constraint_residual = (result.x[0].powi(2) + result.x[1].powi(2) - 1.0).abs();
    assert!(
        constraint_residual < config.constraint_tolerance,
        "SQP equality residual {constraint_residual:.3e} exceeds {:.3e}",
        config.constraint_tolerance
    );
}

fn test_bfgs_large_dimension() {
    let dim = 100;
    let x0 = vec![1.0_f64; dim];

    let config = BfgsConfig {
        max_iterations: 200,
        grad_tolerance: 1e-6,
        fun_tolerance: 1e-12,
        step_tolerance: 1e-12,
        line_search: sonium_solvers::LineSearchMethod::StrongWolfe,
    };

    let result = bfgs(&sphere, &sphere_grad, &x0, &config).expect("BFGS should succeed");

    assert!(result.converged, "BFGS should converge on high-dim sphere");
    assert!(result.fun < 1e-6, "Minimum should be near zero");
}

fn test_lbfgs_vs_bfgs_consistency() {
    let x0 = vec![2.0_f64, 3.0];
    let config_bfgs = BfgsConfig {
        max_iterations: 100,
        grad_tolerance: 1e-8,
        fun_tolerance: 1e-12,
        step_tolerance: 1e-12,
        line_search: sonium_solvers::LineSearchMethod::StrongWolfe,
    };
    let config_lbfgs = LbfgsConfig {
        max_iterations: 100,
        grad_tolerance: 1e-8,
        fun_tolerance: 1e-12,
        step_tolerance: 1e-12,
        memory: 10,
        line_search: sonium_solvers::LineSearchMethod::StrongWolfe,
    };

    let bfgs_result = bfgs(&sphere, &sphere_grad, &x0, &config_bfgs).expect("BFGS should succeed");
    let lbfgs_result =
        lbfgs(&sphere, &sphere_grad, &x0, &config_lbfgs).expect("L-BFGS should succeed");

    let diff: f64 = bfgs_result
        .x
        .iter()
        .zip(lbfgs_result.x.iter())
        .map(|(a, b): (&f64, &f64)| (a - b).abs())
        .sum();
    assert!(diff < 1e-4, "BFGS and L-BFGS should give similar solutions");
}

fn main() {
    let started = Instant::now();
    let result = std::panic::catch_unwind(|| {
        test_bfgs_sphere();
        test_lbfgs_sphere();
        test_bfgs_rosenbrock();
        test_lbfgs_rosenbrock();
        test_nelder_mead_sphere();
        test_sqp_equality_constraint();
        test_bfgs_large_dimension();
        test_lbfgs_vs_bfgs_consistency();
    });
    match result {
        Ok(()) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("passed_test_count", TEST_COUNT)
                .minimum("passed_test_count", TEST_COUNT)
                .input_checksum("analytic/deterministic-fixture:solvers-optimization-v1")
                .reference("analytic sphere/Rosenbrock minima and unit-circle equality")
                .reference_checksum("analytic")
                .dimension("f64; 2-100 variables")
                .formulation("deterministic unconstrained and equality-constrained optimization fixtures")
                .tolerance("8/8 contracts pass: objective, distance-to-optimum, feasibility, and cross-solver consistency")
                .seed("solvers-optimization:8-contracts:v1")
                .timing_ms("optimization_suite", elapsed_ms)
                .elapsed_ms(elapsed_ms);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("solver optimization result must serialize")
            );
            println!("SUCCESS");
        }
        Err(_) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, "one or more optimization contracts panicked")
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .elapsed_ms(elapsed_ms)
                    .to_json_line()
                    .expect("solver optimization failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
