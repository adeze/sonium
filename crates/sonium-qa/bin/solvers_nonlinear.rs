//! Integration tests for nonlinear solvers
//!
//! Tests Newton-Raphson method.

use sonium_solvers::{NewtonConfig, newton_raphson};

fn solve_2x2_system() {
    let x0 = vec![2.0_f64, 1.0];

    let config = NewtonConfig {
        max_iterations: 50,
        absolute_tolerance: 1e-10,
        relative_tolerance: 1e-8,
        damping: 1.0,
        line_search: sonium_solvers::LineSearch::None,
        verbosity: 0,
    };

    let result = newton_raphson(
        &x0,
        |x: &[f64]| -> Vec<f64> { vec![x[0].powi(2) + x[1].powi(2) - 4.0, x[0] - x[1] - 1.0] },
        |x: &[f64], rhs: &[f64]| -> Result<Vec<f64>, String> {
            // Jacobian:
            //   [2*x[0]  2*x[1]]
            //   [    1       -1]
            let j00 = 2.0 * x[0];
            let j01 = 2.0 * x[1];
            let j10 = 1.0_f64;
            let j11 = -1.0_f64;

            let det = j00 * j11 - j01 * j10;
            if det.abs() < 1e-14 {
                return Err("Singular Jacobian".to_string());
            }

            let d0 = (j11 * rhs[0] - j01 * rhs[1]) / det;
            let d1 = (-j10 * rhs[0] + j00 * rhs[1]) / det;
            Ok(vec![d0, d1])
        },
        &config,
    );

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.converged, "Newton-Raphson should converge");
}

fn test_newton_raphson_basic() {
    solve_2x2_system();
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_newton_raphson_basic();
    });
    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
