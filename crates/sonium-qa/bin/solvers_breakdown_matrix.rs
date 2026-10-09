//! Deterministic checked breakdown and non-convergence matrix for iterative solvers.

use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, BiCgstabConfig, CaCgConfig, CgConfig, CgsConfig, FgmresConfig, GmresConfig,
    GmresSolution, IdentityPreconditioner, IdrsConfig, IdrsStabConfig, LinearOperator,
    MinresConfig, SolverResultContract, SolverStatus, TfqmrConfig, try_bicgstab,
    try_bicgstab_pipelined, try_ca_cg, try_cg, try_cgs, try_fgmres, try_gmres, try_gmres_pipelined,
    try_idrs, try_idrs_stab, try_minres, try_tfqmr,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-breakdown-matrix.v1";

struct ZeroOperator;

impl LinearOperator<f64> for ZeroOperator {
    fn num_rows(&self) -> usize {
        5
    }

    fn num_cols(&self) -> usize {
        5
    }

    fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
    }

    fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
    }
}

/// A deterministic diagonal operator with five distinct eigenvalues. One
/// Krylov step is intentionally insufficient for the chosen right-hand side.
struct LimitedOperator;

impl LinearOperator<f64> for LimitedOperator {
    fn num_rows(&self) -> usize {
        5
    }

    fn num_cols(&self) -> usize {
        5
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        y[0] = 2.0 * x[0];
        y[1] = 3.0 * x[1];
        y[2] = 5.0 * x[2];
        y[3] = 7.0 * x[3];
        y[4] = 11.0 * x[4];
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        self.apply_into(x, y);
    }
}

fn record_non_converged<T>(
    label: &str,
    result: Result<T, sonium_solvers::SolverError>,
) -> Result<f64, String>
where
    T: SolverResultContract,
{
    match result {
        Err(error) => {
            if error.status() == SolverStatus::Converged
                || error.accuracy_status() == AccuracyStatus::Attained
            {
                return Err(format!("{label}: checked error claimed convergence"));
            }
        }
        Ok(solution) => {
            solution
                .validate_contract()
                .map_err(|error| format!("{label}: invalid checked result: {error}"))?;
            if solution.solver_status() == SolverStatus::Converged
                || solution.accuracy_status() == AccuracyStatus::Attained
            {
                return Err(format!(
                    "{label}: checked result claimed convergence: status={:?}, accuracy={:?}",
                    solution.solver_status(),
                    solution.accuracy_status()
                ));
            }
        }
    }
    Ok(1.0)
}

fn try_gmres_pipelined_identity(
    operator: &impl LinearOperator<f64>,
    b: &[f64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<f64>, sonium_solvers::SolverError> {
    let preconditioner = IdentityPreconditioner;
    let mut config = config.clone();
    config.restart = 1;
    config.max_iterations = 1;
    try_gmres_pipelined(operator, &preconditioner, b, None, &config)
}

fn try_gmres_one_step(
    operator: &impl LinearOperator<f64>,
    b: &[f64],
    config: &GmresConfig<f64>,
) -> Result<GmresSolution<f64>, sonium_solvers::SolverError> {
    let mut config = config.clone();
    config.restart = 1;
    config.max_iterations = 1;
    try_gmres(operator, b, &config)
}

fn try_fgmres_one_step(
    operator: &impl LinearOperator<f64>,
    b: &[f64],
    config: &FgmresConfig<f64>,
) -> Result<sonium_solvers::FgmresSolution<f64>, sonium_solvers::SolverError> {
    let mut config = config.clone();
    config.restart = 1;
    config.max_iterations = 1;
    try_fgmres(operator, b, &config)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let rhs = [1.0, -2.0, 3.0, -4.0, 5.0];
    let zero = ZeroOperator;
    let limited = LimitedOperator;
    let mut metrics = Vec::new();

    macro_rules! cover {
        ($name:literal, $config:expr, $solver:path) => {{
            let config = $config;
            metrics.push((
                concat!($name, "_zero_operator_breakdown"),
                record_non_converged(
                    concat!($name, " zero operator"),
                    $solver(&zero, &rhs, &config),
                )?,
            ));
            let mut limited_config = config.clone();
            limited_config.max_iterations = 1;
            metrics.push((
                concat!($name, "_one_step_nonconvergence"),
                record_non_converged(
                    concat!($name, " one-step distinct-spectrum"),
                    $solver(&limited, &rhs, &limited_config),
                )?,
            ));
        }};
    }

    cover!("gmres", GmresConfig::default(), try_gmres_one_step);
    cover!(
        "gmres_pipelined",
        GmresConfig::default(),
        try_gmres_pipelined_identity
    );
    cover!("fgmres", FgmresConfig::default(), try_fgmres_one_step);
    cover!("bicgstab", BiCgstabConfig::default(), try_bicgstab);
    cover!(
        "bicgstab_pipelined",
        BiCgstabConfig::default(),
        try_bicgstab_pipelined
    );
    cover!("cg", CgConfig::default(), try_cg);
    cover!("cgs", CgsConfig::default(), try_cgs);
    cover!("idrs", IdrsConfig::default(), try_idrs);
    cover!("idrs_stab", IdrsStabConfig::default(), try_idrs_stab);
    cover!("minres", MinresConfig::default(), try_minres);
    cover!("tfqmr", TfqmrConfig::default(), try_tfqmr);
    cover!("ca_cg", CaCgConfig::default(), try_ca_cg);

    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .reference("independent checked iterative breakdown/non-convergence matrix")
                .reference_checksum("analytic/deterministic-fixture:iterative-breakdown-matrix-v1")
                .input_checksum("analytic/deterministic-fixture:iterative-breakdown-matrix-v1")
                .formulation("public checked iterative entry points")
                .dimension("5x5 real zero and distinct-spectrum operator fixtures")
                .tolerance("all 24 fixtures reject or return finite non-converged checked status")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            let result = metrics
                .into_iter()
                .fold(result, |result, (name, value)| result.metric(name, value));
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
