use crate::results::{RunResult, ValidationFailure, ValidationSummary};
use std::collections::HashMap;

/// Grouping key for matching sonium and petsc runs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RunKey {
    solver: String,
    preconditioner: String,
    matrix: String,
    scalar: String,
    n: usize,
    rhs_count: usize,
}

impl RunKey {
    fn from_result(r: &RunResult) -> Self {
        Self {
            solver: r.solver.clone(),
            preconditioner: r.preconditioner.clone(),
            matrix: r.matrix.clone(),
            scalar: r.scalar.clone(),
            n: r.n,
            rhs_count: r.rhs_count.max(1),
        }
    }
}

/// Validate benchmark results by comparing sonium runs against petsc runs.
///
/// For each (solver, preconditioner, matrix, scalar, n) group with both engines present:
/// 1. Convergence mismatch -> failure
/// 2. Iteration ratio > 3x -> warning (generous threshold)
/// 3. Residual ratio > 100x -> failure
pub fn validate(runs: &[RunResult]) -> ValidationSummary {
    // Group runs by key and engine
    let mut sonium_runs: HashMap<RunKey, &RunResult> = HashMap::new();
    let mut petsc_runs: HashMap<RunKey, &RunResult> = HashMap::new();

    for run in runs {
        let key = RunKey::from_result(run);
        match run.engine.as_str() {
            "sonium" => {
                sonium_runs.insert(key, run);
            }
            "petsc" => {
                petsc_runs.insert(key, run);
            }
            engine => {
                log::warn!("Unknown engine in run results: {engine}");
            }
        }
    }

    let mut total_comparisons = 0;
    let mut passed = 0;
    let mut failed = 0;
    let mut warnings = 0;
    let mut failures = Vec::new();

    // Compare each group that has both sonium and petsc results
    for (key, sonium) in &sonium_runs {
        let petsc = match petsc_runs.get(key) {
            Some(p) => p,
            None => continue, // No petsc run for this config; skip
        };

        total_comparisons += 1;

        // Check 1: convergence mismatch
        if sonium.converged != petsc.converged {
            failed += 1;
            failures.push(ValidationFailure {
                solver: key.solver.clone(),
                preconditioner: key.preconditioner.clone(),
                matrix: key.matrix.clone(),
                scalar: key.scalar.clone(),
                n: key.n,
                reason: format!(
                    "Convergence mismatch: sonium={}, petsc={}",
                    sonium.converged, petsc.converged
                ),
                sonium_residual: sonium.residual,
                petsc_residual: petsc.residual,
                sonium_iterations: sonium.iterations,
                petsc_iterations: petsc.iterations,
            });
            continue;
        }

        // Only compare further if both converged
        if !sonium.converged && !petsc.converged {
            // Both failed to converge; not really a pass, but not a comparison failure either
            passed += 1;
            continue;
        }

        let mut this_passed = true;

        // Check 2: iteration ratio > 3x (warning only, be generous)
        if petsc.iterations > 0 && sonium.iterations > 0 {
            let ratio = sonium.iterations as f64 / petsc.iterations as f64;
            if !(1.0 / 3.0..=3.0).contains(&ratio) {
                warnings += 1;
                log::warn!(
                    "Iteration ratio warning: {}/{}/{}/{} n={}: sonium={} petsc={} ratio={:.2}",
                    key.solver,
                    key.preconditioner,
                    key.matrix,
                    key.scalar,
                    key.n,
                    sonium.iterations,
                    petsc.iterations,
                    ratio
                );
            }
        }

        // Check 3: residual ratio > 100x
        if petsc.residual > 0.0 && sonium.residual > 0.0 {
            let ratio = sonium.residual / petsc.residual;
            if ratio > 100.0 {
                failed += 1;
                this_passed = false;
                failures.push(ValidationFailure {
                    solver: key.solver.clone(),
                    preconditioner: key.preconditioner.clone(),
                    matrix: key.matrix.clone(),
                    scalar: key.scalar.clone(),
                    n: key.n,
                    reason: format!(
                        "Residual ratio too large: sonium/petsc = {:.2e}/{:.2e} = {:.1}x",
                        sonium.residual, petsc.residual, ratio
                    ),
                    sonium_residual: sonium.residual,
                    petsc_residual: petsc.residual,
                    sonium_iterations: sonium.iterations,
                    petsc_iterations: petsc.iterations,
                });
            }
        }

        if this_passed {
            passed += 1;
        }
    }

    ValidationSummary {
        total_comparisons,
        passed,
        failed,
        warnings,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::RunResult;

    fn make_run(engine: &str, converged: bool, iterations: usize, residual: f64) -> RunResult {
        RunResult {
            engine: engine.to_string(),
            solver: "gmres".to_string(),
            preconditioner: "none".to_string(),
            matrix: "poisson_2d".to_string(),
            scalar: "f64".to_string(),
            n: 1000,
            nnz: 5000,
            rhs_count: 1,
            setup_ms: 1.0,
            solve_ms: 10.0,
            total_ms: 11.0,
            iterations,
            residual,
            converged,
            bytes_per_spmv: None,
            bytes_per_nnz: None,
            estimated_spmv_gb_s: None,
            setup_fraction: None,
            time_limited_by: None,
            mean_solve_ms_per_rhs: None,
            mean_iterations_per_rhs: None,
            setup_reuse_scope: None,
            setup_reuse_break_even_rhs: None,
            amortized_total_ms_per_rhs_4: None,
            amortized_total_ms_per_rhs_16: None,
            krylov_spmv_count: None,
            krylov_preconditioner_count: None,
            krylov_dot_reductions: None,
            krylov_norm_count: None,
            krylov_axpy_passes: None,
            krylov_basis_bytes_touched: None,
            krylov_synchronization_points: None,
            krylov_exact_residual_checks: None,
        }
    }

    #[test]
    fn test_validate_both_converged_pass() {
        let runs = vec![
            make_run("sonium", true, 50, 1e-10),
            make_run("petsc", true, 45, 1e-10),
        ];
        let summary = validate(&runs);
        assert_eq!(summary.total_comparisons, 1);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 0);
    }

    #[test]
    fn test_validate_convergence_mismatch() {
        let runs = vec![
            make_run("sonium", false, 1000, 1e-2),
            make_run("petsc", true, 50, 1e-10),
        ];
        let summary = validate(&runs);
        assert_eq!(summary.total_comparisons, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.failures.len(), 1);
        assert!(summary.failures[0].reason.contains("Convergence mismatch"));
    }

    #[test]
    fn test_validate_residual_too_large() {
        let runs = vec![
            make_run("sonium", true, 50, 1e-5), // 100000x worse
            make_run("petsc", true, 45, 1e-10),
        ];
        let summary = validate(&runs);
        assert_eq!(summary.failed, 1);
        assert!(summary.failures[0].reason.contains("Residual ratio"));
    }

    #[test]
    fn test_validate_no_petsc_runs() {
        let runs = vec![make_run("sonium", true, 50, 1e-10)];
        let summary = validate(&runs);
        assert_eq!(summary.total_comparisons, 0);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.failed, 0);
    }

    #[test]
    fn test_validate_both_diverged() {
        let runs = vec![
            make_run("sonium", false, 1000, 1.0),
            make_run("petsc", false, 1000, 1.0),
        ];
        let summary = validate(&runs);
        assert_eq!(summary.total_comparisons, 1);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 0);
    }
}
