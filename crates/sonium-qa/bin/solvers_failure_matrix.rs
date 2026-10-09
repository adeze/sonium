//! Deterministic checked-failure matrix for the public iterative solver family.

#![allow(deprecated)]

use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, BiCgstabConfig, CaCgConfig, CgConfig, CgsConfig, FgmresConfig, GmresConfig,
    GmresSolution, IdentityPreconditioner, IdrsConfig, IdrsStabConfig, LinearOperator,
    MinresConfig, SolverResultContract, SolverStatus, TfqmrConfig, try_bicgstab,
    try_bicgstab_pipelined, try_ca_cg, try_cg, try_cgs, try_fgmres, try_gmres, try_gmres_pipelined,
    try_idrs, try_idrs_stab, try_minres, try_tfqmr,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-failure-matrix.v1";

struct ShapeMismatch;

impl LinearOperator<f64> for ShapeMismatch {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        3
    }

    fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
    }

    fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
    }
}

struct PanicOperator;

impl LinearOperator<f64> for PanicOperator {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
        panic!("iterative failure-matrix callback panic");
    }

    fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
        panic!("iterative failure-matrix transpose callback panic");
    }
}

struct IdentityOperator;
struct NonFiniteOutputOperator;

impl LinearOperator<f64> for IdentityOperator {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
}

impl LinearOperator<f64> for NonFiniteOutputOperator {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(f64::NAN);
    }

    fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(f64::NAN);
    }
}

fn record_failure<T>(
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
                return Err(format!("{label}: checked result claimed convergence"));
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
    try_gmres_pipelined(operator, &preconditioner, b, None, config)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let rhs = [1.0, -2.0];
    let non_finite_rhs = [f64::NAN, -2.0];
    let shape_mismatch = ShapeMismatch;
    let panic_operator = PanicOperator;
    let identity = IdentityOperator;
    let non_finite_output = NonFiniteOutputOperator;
    let mut metrics = Vec::new();

    macro_rules! cover {
        ($name:literal, $config:expr, $solver:path) => {{
            let mut config = $config;
            metrics.push((
                concat!($name, "_dimension"),
                record_failure(
                    concat!($name, " dimension"),
                    $solver(&shape_mismatch, &rhs, &config),
                )?,
            ));
            metrics.push((
                concat!($name, "_non_finite_rhs"),
                record_failure(
                    concat!($name, " non-finite RHS"),
                    $solver(&identity, &non_finite_rhs, &config),
                )?,
            ));
            metrics.push((
                concat!($name, "_callback_panic"),
                record_failure(
                    concat!($name, " callback panic"),
                    $solver(&panic_operator, &rhs, &config),
                )?,
            ));
            metrics.push((
                concat!($name, "_non_finite_output"),
                record_failure(
                    concat!($name, " non-finite operator output"),
                    $solver(&non_finite_output, &rhs, &config),
                )?,
            ));
            config.max_iterations = 0;
            metrics.push((
                concat!($name, "_zero_iteration"),
                record_failure(
                    concat!($name, " zero iteration"),
                    $solver(&identity, &rhs, &config),
                )?,
            ));
        }};
    }

    cover!("gmres", GmresConfig::default(), try_gmres);
    cover!(
        "gmres_pipelined",
        GmresConfig::default(),
        try_gmres_pipelined_identity
    );
    cover!("fgmres", FgmresConfig::default(), try_fgmres);
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
                .reference("independent checked iterative failure/status matrix including generated operator-output rejection")
                .reference_checksum("analytic/deterministic-fixture:iterative-failure-matrix-v1")
                .input_checksum("analytic/deterministic-fixture:iterative-failure-matrix-v1")
                .formulation("public checked iterative entry points")
                .dimension("2x2 real operator fixtures: shape mismatch, non-finite RHS, callback panic, non-finite output, zero iteration")
                .tolerance("all 60 failure fixtures reject or return a non-converged checked status")
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
