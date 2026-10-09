use std::collections::HashMap;
use std::path::Path;

use crate::results::{BenchmarkReport, RunResult};

#[derive(Debug, Clone, Copy)]
pub struct RegressionGateConfig {
    pub max_total_slowdown: f64,
    pub max_solve_slowdown: f64,
    pub max_iteration_regression: f64,
}

impl Default for RegressionGateConfig {
    fn default() -> Self {
        Self {
            max_total_slowdown: 0.10,
            max_solve_slowdown: 0.10,
            max_iteration_regression: 0.10,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RegressionSummary {
    pub total_comparisons: usize,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
    pub warning_messages: Vec<String>,
    pub failures: Vec<RegressionFailure>,
}

#[derive(Debug, Clone)]
pub struct RegressionFailure {
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RunKey {
    engine: String,
    solver: String,
    preconditioner: String,
    matrix: String,
    scalar: String,
    n: usize,
    rhs_count: usize,
}

impl RunKey {
    fn from_result(run: &RunResult) -> Self {
        Self {
            engine: run.engine.clone(),
            solver: run.solver.clone(),
            preconditioner: run.preconditioner.clone(),
            matrix: run.matrix.clone(),
            scalar: run.scalar.clone(),
            n: run.n,
            rhs_count: run.rhs_count.max(1),
        }
    }

    fn label(&self) -> String {
        format!(
            "{}/{}/{}/{}/{} n={} rhs={}",
            self.engine,
            self.solver,
            self.preconditioner,
            self.matrix,
            self.scalar,
            self.n,
            self.rhs_count
        )
    }
}

pub fn check_report_against_path(
    current: &BenchmarkReport,
    baseline_path: &Path,
    config: RegressionGateConfig,
) -> Result<RegressionSummary, String> {
    let baseline_json = std::fs::read_to_string(baseline_path)
        .map_err(|err| format!("read {}: {err}", baseline_path.display()))?;
    let baseline = serde_json::from_str::<BenchmarkReport>(&baseline_json)
        .map_err(|err| format!("parse {}: {err}", baseline_path.display()))?;
    Ok(check_report(current, &baseline, config))
}

pub fn check_report(
    current: &BenchmarkReport,
    baseline: &BenchmarkReport,
    config: RegressionGateConfig,
) -> RegressionSummary {
    let mut summary = RegressionSummary::default();
    let baseline_runs: HashMap<RunKey, &RunResult> = baseline
        .runs
        .iter()
        .map(|run| (RunKey::from_result(run), run))
        .collect();

    if current.platform.os != baseline.platform.os
        || current.platform.arch != baseline.platform.arch
        || current.platform.cpu_model != baseline.platform.cpu_model
    {
        summary.warnings += 1;
        summary.warning_messages.push(format!(
            "platform differs from baseline: current={} {}/{} baseline={} {}/{}",
            current.platform.cpu_model,
            current.platform.os,
            current.platform.arch,
            baseline.platform.cpu_model,
            baseline.platform.os,
            baseline.platform.arch
        ));
    }

    for current_run in &current.runs {
        let key = RunKey::from_result(current_run);
        let Some(baseline_run) = baseline_runs.get(&key) else {
            summary.warnings += 1;
            summary
                .warning_messages
                .push(format!("no baseline run for {}", key.label()));
            continue;
        };

        summary.total_comparisons += 1;
        let before = summary.failures.len();

        if baseline_run.converged && !current_run.converged {
            summary.failures.push(RegressionFailure {
                reason: format!(
                    "{} stopped converging; baseline residual={:.2e}, current residual={:.2e}",
                    key.label(),
                    baseline_run.residual,
                    current_run.residual
                ),
            });
        }

        check_slowdown(
            &mut summary,
            &key,
            "total_ms",
            current_run.total_ms,
            baseline_run.total_ms,
            config.max_total_slowdown,
        );
        check_slowdown(
            &mut summary,
            &key,
            "solve_ms",
            current_run.solve_ms,
            baseline_run.solve_ms,
            config.max_solve_slowdown,
        );
        check_iteration_regression(
            &mut summary,
            &key,
            current_run.iterations,
            baseline_run.iterations,
            config.max_iteration_regression,
        );

        if summary.failures.len() == before {
            summary.passed += 1;
        }
    }

    summary.failed = summary.failures.len();
    summary
}

fn check_slowdown(
    summary: &mut RegressionSummary,
    key: &RunKey,
    metric: &str,
    current: f64,
    baseline: f64,
    max_slowdown: f64,
) {
    if baseline <= 0.0 || current <= 0.0 {
        return;
    }
    let limit = baseline * (1.0 + max_slowdown);
    if current > limit {
        summary.failures.push(RegressionFailure {
            reason: format!(
                "{} {metric} regression: current={current:.3} baseline={baseline:.3} limit={limit:.3}",
                key.label()
            ),
        });
    }
}

fn check_iteration_regression(
    summary: &mut RegressionSummary,
    key: &RunKey,
    current: usize,
    baseline: usize,
    max_regression: f64,
) {
    if baseline == 0 {
        return;
    }
    let limit = ((baseline as f64) * (1.0 + max_regression)).ceil() as usize;
    if current > limit {
        summary.failures.push(RegressionFailure {
            reason: format!(
                "{} iteration regression: current={current} baseline={baseline} limit={limit}",
                key.label()
            ),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::{BenchmarkReport, PlatformInfo, ValidationSummary};

    fn report(run: RunResult) -> BenchmarkReport {
        BenchmarkReport {
            schema_version: 1,
            timestamp: "2026-05-18T00:00:00Z".to_string(),
            platform: PlatformInfo {
                name: "test".to_string(),
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                cpu_model: "test cpu".to_string(),
                total_cores: 8,
            },
            sonium_version: "0.0.0".to_string(),
            git_sha: None,
            petsc_available: false,
            runs: vec![run],
            validation: ValidationSummary {
                total_comparisons: 0,
                passed: 0,
                failed: 0,
                warnings: 0,
                failures: Vec::new(),
            },
        }
    }

    fn run(total_ms: f64, solve_ms: f64, iterations: usize, converged: bool) -> RunResult {
        RunResult {
            engine: "sonium".to_string(),
            solver: "gmres".to_string(),
            preconditioner: "amg".to_string(),
            matrix: "poisson_2d".to_string(),
            scalar: "f64".to_string(),
            n: 1000,
            nnz: 5000,
            rhs_count: 1,
            setup_ms: total_ms - solve_ms,
            solve_ms,
            total_ms,
            iterations,
            residual: if converged { 1e-9 } else { 1e-2 },
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
    fn passes_within_regression_tolerance() {
        let baseline = report(run(100.0, 80.0, 40, true));
        let current = report(run(108.0, 86.0, 44, true));

        let summary = check_report(&current, &baseline, RegressionGateConfig::default());

        assert_eq!(summary.total_comparisons, 1);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 0);
    }

    #[test]
    fn fails_on_slowdown_and_iteration_regression() {
        let baseline = report(run(100.0, 80.0, 40, true));
        let current = report(run(125.0, 100.0, 50, true));

        let summary = check_report(&current, &baseline, RegressionGateConfig::default());

        assert_eq!(summary.total_comparisons, 1);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.failed, 3);
        assert!(
            summary
                .failures
                .iter()
                .any(|failure| failure.reason.contains("total_ms regression"))
        );
        assert!(
            summary
                .failures
                .iter()
                .any(|failure| failure.reason.contains("solve_ms regression"))
        );
        assert!(
            summary
                .failures
                .iter()
                .any(|failure| failure.reason.contains("iteration regression"))
        );
    }

    #[test]
    fn fails_when_current_stops_converging() {
        let baseline = report(run(100.0, 80.0, 40, true));
        let current = report(run(90.0, 70.0, 30, false));

        let summary = check_report(&current, &baseline, RegressionGateConfig::default());

        assert_eq!(summary.failed, 1);
        assert!(summary.failures[0].reason.contains("stopped converging"));
    }
}
