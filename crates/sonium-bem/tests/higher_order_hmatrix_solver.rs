//! Public solver-dispatch coverage for discontinuous higher-order Quad4 H-matrices.

use sonium_bem::core::bem_solver::{
    AssemblyMethod, BemPreconditioner, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::{generate_cubed_sphere_quad_mesh, generate_icosphere_mesh};
use sonium_bem::core::types::{BasisOrder, PhysicsParams};

fn quad4_problem() -> BemProblem {
    BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    }
}

fn tri3_problem() -> BemProblem {
    BemProblem {
        mesh: generate_icosphere_mesh(0.1, 0),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    }
}

#[test]
fn quad4_discontinuous_linear_and_quadratic_hmatrix_solver_paths_are_finite() {
    let problem = quad4_problem();
    problem.validate().expect("valid Quad4 sphere fixture");

    for basis in [BasisOrder::Linear, BasisOrder::Quadratic] {
        let solver = BemSolver::new()
            .with_basis_order(basis)
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres)
            .with_preconditioner(BemPreconditioner::Jacobi)
            .with_hmatrix_accuracy(1.0e-9, 64)
            .with_compressed_retries(0)
            .with_tolerance(1.0e-8)
            .with_max_iterations(300);
        let solution = solver
            .solve(&problem)
            .unwrap_or_else(|error| panic!("Quad4 {basis:?} H-matrix solve failed: {error}"));
        assert!(solution.num_dofs() > 0);
        assert!(
            solution
                .surface_pressure
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
        assert!(field.re.is_finite() && field.im.is_finite());
    }
}

#[test]
fn quad4_discontinuous_quadratic_mlfmm_solver_path_is_finite() {
    let problem = quad4_problem();
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_assembly_method(AssemblyMethod::Mlfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .expect("discontinuous quadratic Quad4 MLFMM solve");
    assert_eq!(solution.num_dofs(), 54);
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    assert!(field.re.is_finite() && field.im.is_finite());
}

#[test]
fn continuous_quadratic_quad4_directional_mlfmm_solver_path_is_finite() {
    let problem = quad4_problem();
    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::DirectionalMlfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .expect("continuous quadratic Quad4 directional MLFMM solve");
    assert_eq!(solution.num_dofs(), 26);
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    assert!(field.re.is_finite() && field.im.is_finite());
}

#[test]
fn continuous_quadratic_quad4_slfmm_solver_path_is_finite() {
    let problem = quad4_problem();
    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Slfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .expect("continuous quadratic Quad4 SLFMM solve");
    assert_eq!(solution.num_dofs(), 26);
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    assert!(field.re.is_finite() && field.im.is_finite());
}

#[test]
fn continuous_quadratic_quad4_mlfmm_solver_path_is_finite() {
    let problem = quad4_problem();
    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Mlfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .expect("continuous quadratic Quad4 MLFMM solve");
    assert_eq!(solution.num_dofs(), 26);
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    assert!(field.re.is_finite() && field.im.is_finite());
}

#[test]
fn continuous_quad4_hmatrix_solver_path_is_finite() {
    let problem = quad4_problem();
    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .expect("continuous Quad4 H-matrix solve");
    // Six closed-cube Quad4 panels have eight shared vertices, twelve shared
    // edge midpoints, and six panel-interior quadratic DOFs.
    assert_eq!(solution.num_dofs(), 26);
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}
#[test]
fn continuous_linear_quad4_hmatrix_solver_path_is_finite() {
    let problem = quad4_problem();
    let solver = BemSolver::new()
        .with_continuous_linear()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .expect("continuous linear Quad4 H-matrix solve");
    assert_eq!(solution.num_dofs(), 8);
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    assert!(field.re.is_finite() && field.im.is_finite());
}

#[test]
fn continuous_linear_quad4_tbem_solver_path_is_finite() {
    let problem = quad4_problem();
    let solver = BemSolver::new()
        .with_continuous_linear()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let solution = solver
        .solve(&problem)
        .expect("continuous linear Quad4 TBEM solve");
    assert_eq!(solution.num_dofs(), 8);
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn continuous_linear_tri3_hmatrix_solver_path_is_finite() {
    let problem = tri3_problem();
    let solution = BemSolver::new()
        .with_continuous_linear()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300)
        .solve(&problem)
        .expect("continuous linear Tri3 H-matrix solve");
    assert_eq!(solution.num_dofs(), 12);
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}
