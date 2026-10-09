//! Deterministic SLFMM Hermitian-adjoint ILU/H-ILU solve and residual gate.

use num_complex::Complex64;
use sonium_bem::core::assembly::slfmm::{NearFieldBlock, SlfmmSystem};
use sonium_bem::core::solver::{
    GmresConfig, LinearOperator, try_gmres_solve_fmm_batched_adjoint_with_hilu,
    try_gmres_solve_fmm_batched_adjoint_with_ilu,
};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.slfmm-adjoint-preconditioners.v1";

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn fixture() -> SlfmmSystem {
    let mut system = SlfmmSystem::new(2, 1, 1, 1);
    system.cluster_dof_indices = vec![vec![0, 1]];
    system.near_matrix.push(NearFieldBlock {
        source_cluster: 0,
        field_cluster: 0,
        coefficients: DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![c(2.0, 0.1), c(0.3, -0.2), c(-0.4, 0.25), c(1.5, -0.15)],
        ),
    });
    system
}

fn residual(system: &SlfmmSystem, solution: &[Complex64], rhs: &[Complex64]) -> f64 {
    system
        .apply_hermitian(solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn run() -> Result<(f64, f64, f64, usize), String> {
    let system = fixture();
    let rhs = vec![c(0.75, -0.2), c(-1.25, 0.4)];
    let config = GmresConfig {
        max_iterations: 20,
        restart: 4,
        tolerance: 1.0e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };

    let ilu = try_gmres_solve_fmm_batched_adjoint_with_ilu(&system, &rhs, &config)
        .map_err(|error| format!("SLFMM adjoint ILU solve failed: {error}"))?;
    let hilu = try_gmres_solve_fmm_batched_adjoint_with_hilu(&system, &rhs, &config)
        .map_err(|error| format!("SLFMM adjoint H-ILU solve failed: {error}"))?;
    if !ilu.converged || !hilu.converged {
        return Err(format!(
            "SLFMM adjoint preconditioners did not converge (ILU {}, H-ILU {})",
            ilu.residual, hilu.residual
        ));
    }

    let ilu_residual = residual(&system, &ilu.x, &rhs);
    let hilu_residual = residual(&system, &hilu.x, &rhs);
    let solution_delta = ilu
        .x
        .iter()
        .zip(hilu.x.iter())
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0_f64, f64::max);
    if !ilu_residual.is_finite()
        || !hilu_residual.is_finite()
        || !solution_delta.is_finite()
        || ilu_residual > 1.0e-11
        || hilu_residual > 1.0e-11
        || solution_delta > 1.0e-11
    {
        return Err(format!(
            "SLFMM adjoint residual/parity gate failed: ILU={ilu_residual:e}, H-ILU={hilu_residual:e}, delta={solution_delta:e}"
        ));
    }

    let rejected = usize::from(
        try_gmres_solve_fmm_batched_adjoint_with_hilu(
            &system,
            &[Complex64::new(f64::NAN, 0.0), c(1.0, 0.0)],
            &config,
        )
        .is_err(),
    );
    if rejected != 1 {
        return Err("SLFMM adjoint H-ILU accepted a non-finite RHS".to_string());
    }

    Ok((ilu_residual, hilu_residual, solution_delta, rejected))
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok((ilu_residual, hilu_residual, solution_delta, rejected))) => {
            QaResult::passed(CASE_ID)
                .metric("ilu_full_adjoint_residual", ilu_residual)
                .metric("hilu_full_adjoint_residual", hilu_residual)
                .metric("solution_delta", solution_delta)
                .metric("nonfinite_rhs_rejections", rejected as f64)
                .maximum("ilu_full_adjoint_residual", 1.0e-11)
                .maximum("hilu_full_adjoint_residual", 1.0e-11)
                .maximum("solution_delta", 1.0e-11)
                .minimum("nonfinite_rhs_rejections", 1.0)
                .input_checksum(
                    "analytic/deterministic-fixture:bem-slfmm-adjoint-preconditioners-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-slfmm-adjoint-preconditioners-v1",
                )
                .dimension("2-DOF complex SLFMM near-field")
                .reference("independent full matrix-free SLFMM Hermitian-adjoint residual")
                .seed("bem-slfmm-adjoint-preconditioners:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
        }
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(_) => QaResult::failed(
            CASE_ID,
            "panic while running SLFMM adjoint preconditioner QA",
        ),
    };
    let passed = matches!(&result.status, QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
