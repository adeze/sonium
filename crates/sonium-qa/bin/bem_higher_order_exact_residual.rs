//! Bounded CPU QA for higher-order compressed exact-reference certificates.
//!
//! This deterministic Quad4 fixture exercises discontinuous linear/quadratic,
//! continuous quadratic, and constant-panel impedance H-matrix actions, plus
//! bounded higher-order MLFMM and directional-MLFMM forward/adjoint paths. This
//! fixture is deliberately small and is not intended as a simulator-scale
//! benchmark.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::{
    AssemblyMethod, BemPreconditioner, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::{BasisContinuity, BasisOrder, PhysicsParams};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-exact-residual.v1";

fn run_fmm_case(
    problem: &BemProblem,
    assembly: AssemblyMethod,
    continuity: BasisContinuity,
    dof_count: usize,
) -> Result<(f64, f64), String> {
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_basis_continuity(continuity)
        .with_assembly_method(assembly)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300)
        .with_compressed_exact_residual_certification(64, 1.0e-4);
    let rhs = vec![Complex64::new(0.4, -0.2); dof_count];

    let forward = solver
        .solve_rhs_many(problem, std::slice::from_ref(&rhs))
        .map_err(|error| format!("{assembly:?} forward solve: {error}"))?;
    let forward_residual = forward
        .first()
        .and_then(|solution| solution.exact_reference_residual())
        .ok_or_else(|| format!("{assembly:?} forward exact certificate missing"))?;
    if !forward_residual.is_finite() || forward_residual > 1.0e-4 {
        return Err(format!(
            "{assembly:?} forward exact residual {forward_residual:e} exceeds 1e-4"
        ));
    }

    let adjoint = solver
        .solve_adjoint_many(problem, std::slice::from_ref(&rhs))
        .map_err(|error| format!("{assembly:?} adjoint solve: {error}"))?;
    let adjoint_residual = adjoint
        .first()
        .and_then(|solution| solution.exact_reference_residual())
        .ok_or_else(|| format!("{assembly:?} adjoint exact certificate missing"))?;
    if !adjoint_residual.is_finite() || adjoint_residual > 1.0e-4 {
        return Err(format!(
            "{assembly:?} adjoint exact residual {adjoint_residual:e} exceeds 1e-4"
        ));
    }
    Ok((forward_residual, adjoint_residual))
}

fn run() -> Result<(f64, f64, usize, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    let mut maximum_forward = 0.0_f64;
    let mut maximum_adjoint = 0.0_f64;
    let mut case_count = 0_usize;
    let mut maximum_slfmm_forward = 0.0_f64;
    let mut maximum_slfmm_adjoint = 0.0_f64;
    let mut maximum_slfmm_continuous_forward = 0.0_f64;
    let mut maximum_slfmm_continuous_adjoint = 0.0_f64;
    let mut maximum_mlfmm_forward = 0.0_f64;
    let mut maximum_mlfmm_adjoint = 0.0_f64;
    let mut maximum_directional_forward = 0.0_f64;
    let mut maximum_directional_adjoint = 0.0_f64;

    for (basis, dof_count, continuous) in [
        (BasisOrder::Linear, 24, false),
        (BasisOrder::Quadratic, 54, false),
        (BasisOrder::Quadratic, 26, true),
    ] {
        let solver = BemSolver::new().with_basis_order(basis);
        let solver = if continuous {
            solver.with_basis_continuity(BasisContinuity::Continuous)
        } else {
            solver
        };
        let solver = solver
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres)
            .with_preconditioner(BemPreconditioner::Jacobi)
            .with_hmatrix_accuracy(1.0e-9, 64)
            .with_compressed_retries(0)
            .with_tolerance(1.0e-8)
            .with_max_iterations(300)
            .with_compressed_exact_residual_certification(64, 1.0e-4);
        let rhs = vec![Complex64::new(0.4, -0.2); dof_count];

        let forward = solver
            .solve_rhs_many(&problem, std::slice::from_ref(&rhs))
            .map_err(|error| format!("{basis:?} forward solve: {error}"))?;
        let forward_residual = forward
            .first()
            .and_then(|solution| solution.exact_reference_residual())
            .ok_or_else(|| format!("{basis:?} forward exact certificate missing"))?;
        if !forward_residual.is_finite() || forward_residual > 1.0e-4 {
            return Err(format!(
                "{basis:?} forward exact residual {forward_residual:e} exceeds 1e-4"
            ));
        }
        maximum_forward = maximum_forward.max(forward_residual);

        let adjoint = solver
            .solve_adjoint_many(&problem, std::slice::from_ref(&rhs))
            .map_err(|error| format!("{basis:?} adjoint solve: {error}"))?;
        let adjoint_residual = adjoint
            .first()
            .and_then(|solution| solution.exact_reference_residual())
            .ok_or_else(|| format!("{basis:?} adjoint exact certificate missing"))?;
        if !adjoint_residual.is_finite() || adjoint_residual > 1.0e-4 {
            return Err(format!(
                "{basis:?} adjoint exact residual {adjoint_residual:e} exceeds 1e-4"
            ));
        }
        maximum_adjoint = maximum_adjoint.max(adjoint_residual);
        case_count += 1;
    }

    let mut impedance_problem = problem.clone();
    impedance_problem.bc_type = BoundaryConditionType::Impedance;
    let impedance_dof_count = impedance_problem.mesh.elements.len();
    let impedance_solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300)
        .with_compressed_exact_residual_certification(64, 1.0e-4);
    let impedance_rhs = vec![Complex64::new(0.4, -0.2); impedance_dof_count];
    let impedance_forward = impedance_solver
        .solve_rhs_many(&impedance_problem, std::slice::from_ref(&impedance_rhs))
        .map_err(|error| format!("impedance forward solve: {error}"))?;
    let impedance_forward_residual = impedance_forward
        .first()
        .and_then(|solution| solution.exact_reference_residual())
        .ok_or_else(|| "impedance forward exact certificate missing".to_string())?;
    if !impedance_forward_residual.is_finite() || impedance_forward_residual > 1.0e-4 {
        return Err(format!(
            "impedance forward exact residual {impedance_forward_residual:e} exceeds 1e-4"
        ));
    }
    maximum_forward = maximum_forward.max(impedance_forward_residual);
    let impedance_adjoint = impedance_solver
        .solve_adjoint_many(&impedance_problem, std::slice::from_ref(&impedance_rhs))
        .map_err(|error| format!("impedance adjoint solve: {error}"))?;
    let impedance_adjoint_residual = impedance_adjoint
        .first()
        .and_then(|solution| solution.exact_reference_residual())
        .ok_or_else(|| "impedance adjoint exact certificate missing".to_string())?;
    if !impedance_adjoint_residual.is_finite() || impedance_adjoint_residual > 1.0e-4 {
        return Err(format!(
            "impedance adjoint exact residual {impedance_adjoint_residual:e} exceeds 1e-4"
        ));
    }
    maximum_adjoint = maximum_adjoint.max(impedance_adjoint_residual);
    case_count += 1;

    let (slfmm_forward, slfmm_adjoint) = run_fmm_case(
        &problem,
        AssemblyMethod::Slfmm,
        BasisContinuity::Discontinuous,
        54,
    )?;
    maximum_slfmm_forward = maximum_slfmm_forward.max(slfmm_forward);
    maximum_slfmm_adjoint = maximum_slfmm_adjoint.max(slfmm_adjoint);
    case_count += 1;

    let (slfmm_continuous_forward, slfmm_continuous_adjoint) = run_fmm_case(
        &problem,
        AssemblyMethod::Slfmm,
        BasisContinuity::Continuous,
        26,
    )?;
    maximum_slfmm_continuous_forward =
        maximum_slfmm_continuous_forward.max(slfmm_continuous_forward);
    maximum_slfmm_continuous_adjoint =
        maximum_slfmm_continuous_adjoint.max(slfmm_continuous_adjoint);
    case_count += 1;

    let (mlfmm_forward, mlfmm_adjoint) = run_fmm_case(
        &problem,
        AssemblyMethod::Mlfmm,
        BasisContinuity::Discontinuous,
        54,
    )?;
    maximum_mlfmm_forward = maximum_mlfmm_forward.max(mlfmm_forward);
    maximum_mlfmm_adjoint = maximum_mlfmm_adjoint.max(mlfmm_adjoint);
    case_count += 1;

    let (directional_forward, directional_adjoint) = run_fmm_case(
        &problem,
        AssemblyMethod::DirectionalMlfmm,
        BasisContinuity::Continuous,
        26,
    )?;
    maximum_directional_forward = maximum_directional_forward.max(directional_forward);
    maximum_directional_adjoint = maximum_directional_adjoint.max(directional_adjoint);
    case_count += 1;

    Ok((
        maximum_forward,
        maximum_adjoint,
        case_count,
        maximum_slfmm_forward,
        maximum_slfmm_adjoint,
        maximum_slfmm_continuous_forward,
        maximum_slfmm_continuous_adjoint,
        maximum_mlfmm_forward,
        maximum_mlfmm_adjoint,
        maximum_directional_forward,
        maximum_directional_adjoint,
    ))
}

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(run));
    let result = match outcome {
        Ok(Ok((
            maximum_forward,
            maximum_adjoint,
            case_count,
            maximum_slfmm_forward,
            maximum_slfmm_adjoint,
            maximum_slfmm_continuous_forward,
            maximum_slfmm_continuous_adjoint,
            maximum_mlfmm_forward,
            maximum_mlfmm_adjoint,
            maximum_directional_forward,
            maximum_directional_adjoint,
        ))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("case_count", case_count as f64)
            .metric("maximum_forward_exact_reference_residual", maximum_forward)
            .metric("maximum_adjoint_exact_reference_residual", maximum_adjoint)
            .metric(
                "maximum_slfmm_forward_exact_reference_residual",
                maximum_slfmm_forward,
            )
            .metric(
                "maximum_slfmm_adjoint_exact_reference_residual",
                maximum_slfmm_adjoint,
            )
            .metric(
                "maximum_slfmm_continuous_forward_exact_reference_residual",
                maximum_slfmm_continuous_forward,
            )
            .metric(
                "maximum_slfmm_continuous_adjoint_exact_reference_residual",
                maximum_slfmm_continuous_adjoint,
            )
            .metric(
                "maximum_mlfmm_forward_exact_reference_residual",
                maximum_mlfmm_forward,
            )
            .metric(
                "maximum_mlfmm_adjoint_exact_reference_residual",
                maximum_mlfmm_adjoint,
            )
            .metric(
                "maximum_directional_mlfmm_forward_exact_reference_residual",
                maximum_directional_forward,
            )
            .metric(
                "maximum_directional_mlfmm_adjoint_exact_reference_residual",
                maximum_directional_adjoint,
            )
            .minimum("case_count", 8.0)
            .maximum("maximum_forward_exact_reference_residual", 1.0e-4)
            .maximum("maximum_adjoint_exact_reference_residual", 1.0e-4)
            .maximum("maximum_slfmm_forward_exact_reference_residual", 1.0e-4)
            .maximum("maximum_slfmm_adjoint_exact_reference_residual", 1.0e-4)
            .maximum(
                "maximum_slfmm_continuous_forward_exact_reference_residual",
                1.0e-4,
            )
            .maximum(
                "maximum_slfmm_continuous_adjoint_exact_reference_residual",
                1.0e-4,
            )
            .maximum("maximum_mlfmm_forward_exact_reference_residual", 1.0e-4)
            .maximum("maximum_mlfmm_adjoint_exact_reference_residual", 1.0e-4)
            .maximum(
                "maximum_directional_mlfmm_forward_exact_reference_residual",
                1.0e-4,
            )
            .maximum(
                "maximum_directional_mlfmm_adjoint_exact_reference_residual",
                1.0e-4,
            )
            .input_checksum("analytic/deterministic-fixture:bem-higher-order-exact-residual-v1")
            .reference("independent matrix-free basis-weighted TBEM forward/Hermitian action")
            .reference_checksum("analytic/deterministic-fixture:bem-higher-order-exact-residual-v1")
            .dimension(
            "3-D closed cubed-sphere Quad4; H-matrix linear/quadratic/continuous plus impedance and higher-order SLFMM/MLFMM/directional-MLFMM",
            )
            .formulation("production BemSolver compressed exact-reference residual admission")
            .tolerance("seven bounded cases; each forward and adjoint exact residual <= 1e-4")
            .seed("bem-higher-order-exact-residual:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(_) => QaResult::failed(CASE_ID, "higher-order exact-residual QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA result serializes")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
