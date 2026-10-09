//! High-level SLFMM/MLFMM H-ILU dispatch and residual gate.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::BemPreconditioner;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.high-level-hilu-dispatch.v1";
const RESIDUAL_LIMIT: f64 = 1.0e-4;

#[derive(Debug)]
struct Metrics {
    maximum_forward_residual: f64,
    maximum_adjoint_residual: f64,
    non_fmm_rejections: usize,
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let centers: Vec<[f64; 3]> = problem
        .mesh
        .elements
        .iter()
        .map(|element| element.center)
        .collect();
    let normals: Vec<[f64; 3]> = problem
        .mesh
        .elements
        .iter()
        .map(|element| element.normal)
        .collect();
    let forward_rhs = problem.incident_field.compute_rhs(
        &centers,
        &normals,
        &problem.physics,
        problem.use_burton_miller,
    );

    let mut maximum_forward_residual = 0.0_f64;
    let mut maximum_adjoint_residual = 0.0_f64;

    for assembly in [AssemblyMethod::Slfmm, AssemblyMethod::Mlfmm] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly)
            .with_solver_method(SolverMethod::Gmres)
            .with_preconditioner(BemPreconditioner::HierarchicalIlu)
            .with_tolerance(1.0e-7)
            .with_max_iterations(120)
            .with_compressed_retries(0)
            .with_compressed_exact_residual_certification(64, RESIDUAL_LIMIT);

        let forward = solver
            .solve_rhs_many(&problem, std::slice::from_ref(&forward_rhs))
            .map_err(|error| format!("{assembly:?} high-level H-ILU forward solve: {error}"))?;
        let forward = forward.first().ok_or_else(|| {
            format!("{assembly:?} high-level H-ILU forward solve returned no result")
        })?;
        let forward_residual = forward
            .exact_reference_residual()
            .ok_or_else(|| format!("{assembly:?} high-level H-ILU forward certificate missing"))?;
        if !forward_residual.is_finite() || forward_residual > RESIDUAL_LIMIT {
            return Err(format!(
                "{assembly:?} high-level H-ILU forward residual {forward_residual:e} exceeds {RESIDUAL_LIMIT:e}"
            ));
        }
        maximum_forward_residual = maximum_forward_residual.max(forward_residual);

        let adjoint_rhs = vec![Complex64::new(0.25, -0.1); forward.num_dofs()];
        let adjoint = solver
            .solve_adjoint_many(&problem, std::slice::from_ref(&adjoint_rhs))
            .map_err(|error| format!("{assembly:?} high-level H-ILU adjoint solve: {error}"))?;
        let adjoint = adjoint.first().ok_or_else(|| {
            format!("{assembly:?} high-level H-ILU adjoint solve returned no result")
        })?;
        let adjoint_residual = adjoint
            .exact_reference_residual()
            .ok_or_else(|| format!("{assembly:?} high-level H-ILU adjoint certificate missing"))?;
        if !adjoint_residual.is_finite() || adjoint_residual > RESIDUAL_LIMIT {
            return Err(format!(
                "{assembly:?} high-level H-ILU adjoint residual {adjoint_residual:e} exceeds {RESIDUAL_LIMIT:e}"
            ));
        }
        maximum_adjoint_residual = maximum_adjoint_residual.max(adjoint_residual);
    }

    // H-ILU is only meaningful for matrix-free FMM operators.  The public
    // high-level path must reject a non-FMM configuration instead of silently
    // degrading to an identity preconditioner.
    let non_fmm_solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::HierarchicalIlu);
    let non_fmm_rejections = usize::from(
        non_fmm_solver
            .solve_rhs_many(&problem, std::slice::from_ref(&forward_rhs))
            .is_err(),
    );
    if non_fmm_rejections != 1 {
        return Err("high-level H-ILU accepted a non-FMM assembly".to_string());
    }

    Ok(Metrics {
        maximum_forward_residual,
        maximum_adjoint_residual,
        non_fmm_rejections,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("maximum_forward_residual", metrics.maximum_forward_residual)
            .metric("maximum_adjoint_residual", metrics.maximum_adjoint_residual)
            .metric("non_fmm_rejections", metrics.non_fmm_rejections as f64)
            .maximum("maximum_forward_residual", RESIDUAL_LIMIT)
            .maximum("maximum_adjoint_residual", RESIDUAL_LIMIT)
            .minimum("non_fmm_rejections", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-high-level-hilu-dispatch-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-high-level-hilu-dispatch-v1")
            .dimension("3-D 48-DOF constant-panel SLFMM/MLFMM")
            .reference("independent exact full-operator forward and Hermitian-adjoint residual certificates")
            .seed("bem-high-level-hilu-dispatch:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(_) => QaResult::failed(CASE_ID, "high-level H-ILU dispatch QA panicked"),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
