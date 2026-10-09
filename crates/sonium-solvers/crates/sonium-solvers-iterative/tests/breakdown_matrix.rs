//! Deterministic checked breakdown matrix for the public iterative methods.

use sonium_solvers_core::traits::{LinearOperator, SolverError, SolverStatus};
use sonium_solvers_iterative::{
    BiCgstabConfig, CaCgConfig, CgConfig, CgsConfig, FgmresConfig, GmresConfig, IdrsConfig,
    IdrsStabConfig, MinresConfig, TfqmrConfig, try_bicgstab, try_ca_cg, try_cg, try_cgs,
    try_fgmres, try_gmres, try_idrs, try_idrs_stab, try_minres, try_tfqmr,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

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

fn assert_breakdown(name: &str, solve: impl FnOnce() -> Result<SolverStatus, SolverError>) {
    let outcome = catch_unwind(AssertUnwindSafe(solve));
    assert!(outcome.is_ok(), "{name}: breakdown path panicked");
    let status = outcome
        .expect("panic was checked")
        .unwrap_or_else(|error| panic!("{name}: checked solve rejected breakdown: {error}"));
    assert!(
        matches!(
            status,
            SolverStatus::Breakdown | SolverStatus::MaxIterationsReached | SolverStatus::Stagnated
        ),
        "{name}: zero operator must return typed breakdown/non-convergence status, got {status:?}"
    );
}

#[test]
fn checked_iterative_breakdown_matrix_is_panic_free_and_typed() {
    let operator = ZeroOperator;
    let rhs = [1.0];

    assert_breakdown("GMRES", || {
        try_gmres(
            &operator,
            &rhs,
            &GmresConfig {
                max_iterations: 4,
                ..GmresConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("FGMRES", || {
        try_fgmres(
            &operator,
            &rhs,
            &FgmresConfig {
                max_iterations: 4,
                ..FgmresConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("BiCGSTAB", || {
        try_bicgstab(
            &operator,
            &rhs,
            &BiCgstabConfig {
                max_iterations: 4,
                ..BiCgstabConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("CG", || {
        try_cg(
            &operator,
            &rhs,
            &CgConfig {
                max_iterations: 4,
                ..CgConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("MINRES", || {
        try_minres(
            &operator,
            &rhs,
            &MinresConfig {
                max_iterations: 4,
                ..MinresConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("TFQMR", || {
        try_tfqmr(
            &operator,
            &rhs,
            &TfqmrConfig {
                max_iterations: 4,
                ..TfqmrConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("CA-CG", || {
        try_ca_cg(
            &operator,
            &rhs,
            &CaCgConfig {
                max_iterations: 4,
                ..CaCgConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("IDR(s)", || {
        try_idrs(
            &operator,
            &rhs,
            &IdrsConfig {
                max_iterations: 4,
                ..IdrsConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("IDR(s)stab", || {
        try_idrs_stab(
            &operator,
            &rhs,
            &IdrsStabConfig {
                max_iterations: 4,
                ..IdrsStabConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
    assert_breakdown("CGS", || {
        try_cgs(
            &operator,
            &rhs,
            &CgsConfig {
                max_iterations: 4,
                ..CgsConfig::default()
            },
        )
        .map(|solution| solution.status)
    });
}
