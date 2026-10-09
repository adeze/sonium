//! Independent release evidence for checked adjacent-order TBEM quadrature.

use sonium_bem::core::assembly::{
    TbemQuadraturePolicy, build_tbem_system_checked, validate_tbem_quadrature,
};
use sonium_bem::core::mesh::generate_sphere_mesh;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, PhysicsParams, SolverMethod};

pub struct TbemQuadraturePolicyMetrics {
    pub active_elements: usize,
    pub checked_dofs: usize,
    pub permissive_passed: bool,
    pub strict_rejected: bool,
    pub solver_permissive_passed: bool,
    pub solver_strict_rejected: bool,
    pub compressed_strict_rejected: bool,
}

pub fn run_tbem_quadrature_policy_reference() -> TbemQuadraturePolicyMetrics {
    let mesh = generate_sphere_mesh(0.1, 4, 8);
    let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
    let default_policy = TbemQuadraturePolicy {
        absolute_tolerance: 1.0e6,
        relative_tolerance: 1.0e6,
        ..TbemQuadraturePolicy::default()
    };
    validate_tbem_quadrature(&mesh.elements, &mesh.nodes, &physics, &default_policy)
        .expect("default adjacent-order TBEM policy should pass the bounded sphere");
    let checked = build_tbem_system_checked(&mesh.elements, &mesh.nodes, &physics, default_policy)
        .expect("checked TBEM assembly should preserve the production system");
    assert!(checked.num_dofs > 0);

    let strict_policy = TbemQuadraturePolicy {
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
        ..default_policy
    };
    let strict_rejected =
        validate_tbem_quadrature(&mesh.elements, &mesh.nodes, &physics, &strict_policy).is_err();
    assert!(
        strict_rejected,
        "an unrealistically strict pairwise budget must fail closed"
    );

    let solver_problem = BemProblem::rigid_sphere_scattering_custom(0.1, 800.0, 343.0, 1.21, 4, 8);
    let solver_solution = BemSolver::new()
        .with_quadrature_policy(default_policy)
        .solve(&solver_problem)
        .expect("solver-level quadrature policy should admit dense assembly");
    assert!(
        solver_solution
            .surface_pressure
            .iter()
            .all(|value| value.is_finite())
    );
    let solver_strict_rejected = BemSolver::new()
        .with_quadrature_policy(strict_policy)
        .solve(&solver_problem)
        .is_err();
    assert!(solver_strict_rejected);
    let compressed_strict_rejected = [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
    ]
    .into_iter()
    .all(|assembly_method| {
        BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_quadrature_policy(strict_policy)
            .solve(&solver_problem)
            .is_err()
    });
    assert!(compressed_strict_rejected);

    TbemQuadraturePolicyMetrics {
        active_elements: mesh.elements.len(),
        checked_dofs: checked.num_dofs,
        permissive_passed: true,
        strict_rejected,
        solver_permissive_passed: true,
        solver_strict_rejected,
        compressed_strict_rejected,
    }
}

#[test]
fn checked_tbem_quadrature_policy_is_a_real_admission_gate() {
    let metrics = run_tbem_quadrature_policy_reference();
    assert!(metrics.permissive_passed);
    assert!(metrics.strict_rejected);
    assert!(metrics.solver_permissive_passed);
    assert!(metrics.solver_strict_rejected);
    assert!(metrics.compressed_strict_rejected);
    assert_eq!(metrics.active_elements, 48);
    assert_eq!(metrics.checked_dofs, metrics.active_elements);
    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.tbem-quadrature-policy-reference.v1\",\"status\":\"passed\",\"active_elements\":{},\"checked_dofs\":{},\"permissive_passed\":{},\"strict_rejected\":{}}}",
        metrics.active_elements,
        metrics.checked_dofs,
        metrics.permissive_passed,
        metrics.strict_rejected,
    );
}
