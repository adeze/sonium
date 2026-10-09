//! Independent checked failure-exit evidence for public iterative solvers.

use sonium_qa::QaResult;
use sonium_solvers::iterative::{
    BiCgstabConfig, CaCgConfig, CgConfig, CgsConfig, FgmresConfig, GmresConfig, IdrsConfig,
    IdrsStabConfig, MinresConfig, TfqmrConfig, try_bicgstab, try_ca_cg, try_cg, try_cgs,
    try_fgmres, try_gmres, try_idrs, try_idrs_stab, try_minres, try_tfqmr,
};
use sonium_solvers_core::traits::{LinearOperator, SolverError, SolverStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-breakdown-matrix.v2";
const CHECKSUM: &str = "deterministic-fixture:iterative-breakdown-matrix-v2";

struct ZeroOperator;

impl LinearOperator<f64> for ZeroOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = 0.0;
    }

    fn apply_transpose_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = 0.0;
    }
}

fn checked_status(
    name: &str,
    solve: impl FnOnce() -> Result<SolverStatus, SolverError>,
) -> Result<SolverStatus, String> {
    let outcome = catch_unwind(AssertUnwindSafe(solve))
        .map_err(|_| format!("{name}: checked breakdown path panicked"))?;
    outcome.map_err(|error| format!("{name}: checked solve rejected failure path: {error}"))
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let rhs = [1.0];
    let cases: [(&str, Box<dyn FnOnce() -> Result<SolverStatus, SolverError>>); 10] = [
        (
            "gmres",
            Box::new(move || {
                try_gmres(
                    &ZeroOperator,
                    &rhs,
                    &GmresConfig {
                        max_iterations: 4,
                        ..GmresConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "fgmres",
            Box::new(move || {
                try_fgmres(
                    &ZeroOperator,
                    &rhs,
                    &FgmresConfig {
                        max_iterations: 4,
                        ..FgmresConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "bicgstab",
            Box::new(move || {
                try_bicgstab(
                    &ZeroOperator,
                    &rhs,
                    &BiCgstabConfig {
                        max_iterations: 4,
                        ..BiCgstabConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "cg",
            Box::new(move || {
                try_cg(
                    &ZeroOperator,
                    &rhs,
                    &CgConfig {
                        max_iterations: 4,
                        ..CgConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "minres",
            Box::new(move || {
                try_minres(
                    &ZeroOperator,
                    &rhs,
                    &MinresConfig {
                        max_iterations: 4,
                        ..MinresConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "tfqmr",
            Box::new(move || {
                try_tfqmr(
                    &ZeroOperator,
                    &rhs,
                    &TfqmrConfig {
                        max_iterations: 4,
                        ..TfqmrConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "ca_cg",
            Box::new(move || {
                try_ca_cg(
                    &ZeroOperator,
                    &rhs,
                    &CaCgConfig {
                        max_iterations: 4,
                        ..CaCgConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "idrs",
            Box::new(move || {
                try_idrs(
                    &ZeroOperator,
                    &rhs,
                    &IdrsConfig {
                        max_iterations: 4,
                        ..IdrsConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "idrs_stab",
            Box::new(move || {
                try_idrs_stab(
                    &ZeroOperator,
                    &rhs,
                    &IdrsStabConfig {
                        max_iterations: 4,
                        ..IdrsStabConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
        (
            "cgs",
            Box::new(move || {
                try_cgs(
                    &ZeroOperator,
                    &rhs,
                    &CgsConfig {
                        max_iterations: 4,
                        ..CgsConfig::default()
                    },
                )
                .map(|solution| solution.status)
            }),
        ),
    ];

    let mut breakdowns = 0.0;
    let mut nonconverged = 0.0;
    let mut metrics = Vec::with_capacity(cases.len() + 3);
    for (name, solve) in cases {
        let status = checked_status(name, solve)?;
        if !matches!(
            status,
            SolverStatus::Breakdown | SolverStatus::MaxIterationsReached | SolverStatus::Stagnated
        ) {
            return Err(format!(
                "{name}: unexpected zero-operator status {status:?}"
            ));
        }
        if status == SolverStatus::Breakdown {
            breakdowns += 1.0;
        }
        nonconverged += 1.0;
        let metric_name: &'static str =
            Box::leak(format!("{name}_failure_status").into_boxed_str());
        metrics.push((metric_name, 1.0));
    }
    metrics.push(("breakdown_statuses", breakdowns));
    metrics.push(("typed_nonconverged_statuses", nonconverged));
    metrics.push(("panic_free", 1.0));
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent zero-operator failure-exit oracle across ten checked iterative methods")
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .formulation("typed breakdown, stagnation, or max-iteration status without panic")
                .dimension("1x1 zero operator with nonzero RHS; GMRES, FGMRES, BiCGSTAB, CG, MINRES, TFQMR, CA-CG, IDR(s), IDR(s)stab, CGS")
                .tolerance("all 10 methods return typed non-converged status; no panic; breakdown statuses are reported separately")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .minimum("typed_nonconverged_statuses", 10.0)
                .minimum("panic_free", 1.0);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
