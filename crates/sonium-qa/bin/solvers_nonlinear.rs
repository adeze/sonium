//! Newton accuracy against independently recomputed residuals and both exact roots.
//!
//! Each returned solution must satisfy the circle/line equations and reach
//! its independently known root, in addition to the solver's convergence flag.

// Rust guideline compliant 2026-02-21
use sonium_qa::QaResult;
use sonium_solvers::{NewtonConfig, newton_raphson};

// Absolute tolerances for this dimensionless O(1) system.
const RESIDUAL_TOLERANCE: f64 = 1e-9;
const ROOT_TOLERANCE: f64 = 1e-9;
const CASE_ID: &str = "solvers.nonlinear-independent-accuracy.v1";

fn validate_solution(x: &[f64], expected: [f64; 2]) -> Result<(f64, f64), String> {
    if x.len() != 2 || x.iter().any(|value| !value.is_finite()) {
        return Err("Newton returned an invalid solution vector".into());
    }
    let circle_residual = x[0].mul_add(x[0], x[1] * x[1]) - 4.0;
    let line_residual = x[0] - x[1] - 1.0;
    let residual = circle_residual.hypot(line_residual);
    let root_error = (x[0] - expected[0]).hypot(x[1] - expected[1]);
    if !residual.is_finite() || residual > RESIDUAL_TOLERANCE {
        return Err(format!(
            "independent final residual {residual} exceeds {RESIDUAL_TOLERANCE}"
        ));
    }
    if !root_error.is_finite() || root_error > ROOT_TOLERANCE {
        return Err(format!(
            "known-root error {root_error} exceeds {ROOT_TOLERANCE}"
        ));
    }
    Ok((residual, root_error))
}

fn solve_2x2_system(x0: [f64; 2], branch: f64) -> Result<(f64, f64), String> {
    let config = NewtonConfig {
        max_iterations: 50,
        // Request tighter accuracy than the independent 1e-9 gate so the
        // relative stopping rule cannot accept a larger final residual.
        absolute_tolerance: 1e-12,
        relative_tolerance: 1e-12,
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
    )
    .map_err(|error| format!("Newton solve: {error:?}"))?;

    if !result.converged {
        return Err(format!("Newton did not converge from {x0:?}"));
    }
    // Substitution x=y+1 gives 2y^2+2y-3=0 independently of Newton.
    let expected = [
        (1.0 + branch * 7.0_f64.sqrt()) / 2.0,
        (-1.0 + branch * 7.0_f64.sqrt()) / 2.0,
    ];
    validate_solution(&result.x, expected)
}

fn run() -> Result<QaResult, String> {
    let mut max_residual = 0.0_f64;
    let mut max_root_error = 0.0_f64;
    for (initial, branch) in [([2.0, 1.0], 1.0), ([-1.0, -2.0], -1.0)] {
        let (residual, root_error) = solve_2x2_system(initial, branch)?;
        println!(
            "branch {branch:+}: independent residual={residual:.3e}, root error={root_error:.3e}"
        );
        max_residual = max_residual.max(residual);
        max_root_error = max_root_error.max(root_error);
    }
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu").actual_backend("cpu")
        .seed("circle-radius2:line-x-y1:initial-positive-and-negative")
        .reference("exact roots ((1 +/- sqrt(7))/2,(-1 +/- sqrt(7))/2)")
        .reference_checksum("analytic:circle-line-two-roots:v1")
        .formulation("native Newton; independent returned-coordinate equation residual and branch-root error")
        .metric("maximum_true_residual", max_residual).maximum("maximum_true_residual", RESIDUAL_TOLERANCE)
        .metric("maximum_root_error", max_root_error).maximum("maximum_root_error", ROOT_TOLERANCE)
        .metric("executed_solves", 2.0).minimum("executed_solves", 2.0))
}

fn main() {
    match run().and_then(|record| record.to_json_line()) {
        Ok(line) => {
            println!("QA_RESULT: {line}");
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(line) = QaResult::failed(CASE_ID, &error).to_json_line() {
                println!("QA_RESULT: {line}");
            }
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positive_root() -> [f64; 2] {
        [(1.0 + 7.0_f64.sqrt()) / 2.0, (-1.0 + 7.0_f64.sqrt()) / 2.0]
    }

    #[test]
    fn false_convergence_at_initial_state_is_rejected() {
        assert!(validate_solution(&[2.0, 1.0], positive_root()).is_err());
    }

    #[test]
    fn zero_residual_at_wrong_root_is_rejected() {
        let wrong = [(1.0 - 7.0_f64.sqrt()) / 2.0, (-1.0 - 7.0_f64.sqrt()) / 2.0];
        assert!(validate_solution(&wrong, positive_root()).is_err());
    }

    #[test]
    fn nonfinite_solution_is_rejected() {
        assert!(validate_solution(&[f64::NAN, 1.0], positive_root()).is_err());
        assert!(validate_solution(&[f64::INFINITY, 1.0], positive_root()).is_err());
    }
}
