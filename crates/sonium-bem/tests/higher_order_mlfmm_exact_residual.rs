//! Independent exact-TBEM residual certificates for higher-order FMM solves.

use sonium_bem::core::bem_solver::BemPreconditioner;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::{BemReusePolicy, generate_cubed_sphere_quad_mesh};
use sonium_bem::core::types::{BasisContinuity, BasisOrder, PhysicsParams};
use sonium_bem::core::{
    AssemblyMethod, BemOperatorCache, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};

fn problem() -> BemProblem {
    BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    }
}

fn assert_exact_certificates(
    assembly: AssemblyMethod,
    continuity: BasisContinuity,
    expected_dofs: usize,
) {
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_basis_continuity(continuity)
        .with_assembly_method(assembly)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300)
        .with_compressed_retries(0)
        .with_compressed_exact_residual_certification(64, 1.0e-4);
    // Keep the field reference independent from the compressed action: a
    // dense TBEM solve with the same higher-order layout provides the
    // reference coefficients, while the compressed solve remains the
    // production path under test.
    let dense_solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_basis_continuity(continuity)
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let reference_problem = problem();
    let dense_forward = dense_solver
        .solve(&reference_problem)
        .expect("higher-order dense forward field reference");
    let rhs = dense_forward.coefficients().to_vec();
    assert_eq!(rhs.len(), expected_dofs);
    let field_points = [[0.0, 0.0, 0.24], [0.08, 0.03, 0.21], [0.14, -0.02, 0.17]];
    let dense_forward_field = dense_forward
        .evaluate_pressure_field_checked(&field_points)
        .expect("finite dense higher-order forward field")
        .into_iter()
        .map(|field| field.p_total)
        .collect::<Vec<_>>();

    let forward = solver
        .solve_rhs_many(&problem(), std::slice::from_ref(&rhs))
        .expect("higher-order FMM forward solve");
    assert_eq!(forward[0].num_dofs(), expected_dofs);
    let forward_residual = forward[0]
        .exact_reference_residual()
        .expect("higher-order FMM forward exact residual certificate");
    assert!(
        forward_residual.is_finite() && forward_residual <= 1.0e-4,
        "{assembly:?} forward exact residual {forward_residual:e}"
    );
    let compressed_forward = solver
        .solve(&problem())
        .unwrap_or_else(|error| panic!("{assembly:?} compressed forward field solve: {error}"));
    let compressed_forward_field = compressed_forward
        .evaluate_pressure_field_checked(&field_points)
        .unwrap_or_else(|error| panic!("{assembly:?} compressed forward field: {error}"))
        .into_iter()
        .map(|field| field.p_total)
        .collect::<Vec<_>>();
    let numerator = compressed_forward_field
        .iter()
        .zip(&dense_forward_field)
        .map(|(value, expected)| (*value - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = dense_forward_field
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-12);
    assert!(
        numerator / denominator <= 0.3,
        "{assembly:?} higher-order forward field parity error {:.3e} exceeds 30%",
        numerator / denominator
    );

    let adjoint = solver
        .solve_adjoint_many(&problem(), std::slice::from_ref(&rhs))
        .expect("higher-order FMM adjoint solve");
    assert_eq!(adjoint[0].num_dofs(), expected_dofs);
    let adjoint_residual = adjoint[0]
        .exact_reference_residual()
        .expect("higher-order FMM adjoint exact residual certificate");
    assert!(
        adjoint_residual.is_finite() && adjoint_residual <= 1.0e-4,
        "{assembly:?} adjoint exact residual {adjoint_residual:e}"
    );
    let compressed_adjoint_field = adjoint[0]
        .evaluate_scattered_pressure_field_dual_basis_checked(&field_points)
        .unwrap_or_else(|error| panic!("{assembly:?} dual-basis adjoint field: {error}"));
    // Adjoint coefficients are dual trace DOFs; the explicit dual-basis map
    // converts them to primal traces before field evaluation.
    assert!(
        compressed_adjoint_field
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn higher_order_mlfmm_exact_certificate_covers_discontinuous_forward_and_adjoint() {
    assert_exact_certificates(AssemblyMethod::Mlfmm, BasisContinuity::Discontinuous, 54);
}

#[test]
fn higher_order_slfmm_exact_certificate_covers_discontinuous_forward_and_adjoint() {
    assert_exact_certificates(AssemblyMethod::Slfmm, BasisContinuity::Discontinuous, 54);
}

#[test]
fn higher_order_mlfmm_exact_certificate_covers_continuous_forward_and_adjoint() {
    assert_exact_certificates(AssemblyMethod::Mlfmm, BasisContinuity::Continuous, 26);
}

#[test]
fn higher_order_slfmm_exact_certificate_covers_continuous_forward_and_adjoint() {
    assert_exact_certificates(AssemblyMethod::Slfmm, BasisContinuity::Continuous, 26);
}

#[test]
fn higher_order_directional_mlfmm_exact_certificate_covers_continuous_forward_and_adjoint() {
    assert_exact_certificates(
        AssemblyMethod::DirectionalMlfmm,
        BasisContinuity::Continuous,
        26,
    );
}

#[test]
fn high_level_mlfmm_hierarchical_ilu_dispatches_checked_forward_batch() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    for assembly in [AssemblyMethod::Slfmm, AssemblyMethod::Mlfmm] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly)
            .with_solver_method(SolverMethod::Gmres)
            .with_preconditioner(BemPreconditioner::HierarchicalIlu)
            .with_max_iterations(120)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);

        let solutions = solver.solve(&problem).unwrap_or_else(|error| {
            panic!("high-level {assembly:?} hierarchical ILU solve: {error}")
        });
        assert!(
            solutions
                .surface_pressure
                .iter()
                .all(|value| value.is_finite())
        );

        let adjoints = solver
            .solve_adjoint_many(&problem, &[solutions.surface_pressure.clone()])
            .unwrap_or_else(|error| {
                panic!("high-level {assembly:?} hierarchical ILU adjoint solve: {error}")
            });
        assert!(
            adjoints[0]
                .coefficients()
                .iter()
                .all(|value| value.is_finite())
        );

        let mut cache = BemOperatorCache::new();
        let cached = solver
            .solve_with_cache(&problem, &mut cache, BemReusePolicy::default())
            .unwrap_or_else(|error| {
                panic!("high-level {assembly:?} hierarchical ILU cached solve: {error}")
            });
        assert!(
            cached
                .surface_pressure
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
