//! Dense-TBEM parity for the continuous-linear Quad4 H-matrix path.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::{
    AssemblyMethod, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::PhysicsParams;

const MAX_RELATIVE_FIELD_ERROR: f64 = 1.0e-4;

fn problem() -> BemProblem {
    BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    }
}

fn finite(values: &[Complex64]) -> bool {
    values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
}

#[test]
fn continuous_linear_quad4_hmatrix_matches_dense_tbem_field() {
    let problem = problem();
    let hmatrix = BemSolver::new()
        .with_continuous_linear()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(sonium_bem::core::bem_solver::BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300)
        .solve(&problem)
        .expect("continuous-linear Quad4 H-matrix solve");
    let dense = BemSolver::new()
        .with_continuous_linear()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .solve(&problem)
        .expect("continuous-linear Quad4 dense TBEM solve");

    assert_eq!(hmatrix.num_dofs(), 8);
    assert_eq!(dense.num_dofs(), 8);
    assert!(finite(&hmatrix.surface_pressure));
    assert!(finite(&dense.surface_pressure));

    let points = [
        [0.0, 0.0, 0.2],
        [0.13, -0.07, 0.23],
        [-0.11, 0.09, 0.19],
        [0.04, 0.12, -0.18],
    ];
    let max_relative_error = points
        .iter()
        .map(|point| {
            let compressed = hmatrix.evaluate_pressure(point);
            let reference = dense.evaluate_pressure(point);
            assert!(compressed.re.is_finite() && compressed.im.is_finite());
            assert!(reference.re.is_finite() && reference.im.is_finite());
            (compressed - reference).norm() / reference.norm().max(1.0e-12)
        })
        .fold(0.0, f64::max);

    assert!(
        max_relative_error <= MAX_RELATIVE_FIELD_ERROR,
        "continuous-linear Quad4 H-matrix/dense-TBEM field error {max_relative_error:e} exceeds {MAX_RELATIVE_FIELD_ERROR:e}"
    );
}
