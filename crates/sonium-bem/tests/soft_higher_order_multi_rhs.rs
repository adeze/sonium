//! Regression coverage for batched soft-boundary higher-order solves.
//!
//! Soft (Dirichlet) problems use the indirect single-layer formulation for
//! every basis order.  Keep the batch path equivalent to the scalar path for
//! a small linear sphere fixture so a dispatch regression cannot silently
//! route higher-order batches through the regular rigid/impedance assembly.

use num_complex::Complex64;
use sonium_bem::core::{
    BasisOrder, BemProblem, BemSolver, BoundaryConditionType, IncidentField, SolverMethod,
};

#[test]
fn soft_linear_multi_rhs_matches_independent_solves() {
    let mut problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    problem.bc_type = BoundaryConditionType::Soft;

    let incident_fields = vec![
        IncidentField::plane_wave_z(),
        IncidentField::plane_wave_neg_z(),
    ];
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Linear)
        .with_solver_method(SolverMethod::Direct);

    let batched = solver
        .solve_many(&problem, &incident_fields)
        .expect("soft linear multi-RHS solve");
    assert_eq!(batched.len(), incident_fields.len());

    for (incident, batch_solution) in incident_fields.iter().zip(&batched) {
        let mut scalar_problem = problem.clone();
        scalar_problem.incident_field = incident.clone();
        let scalar = solver
            .solve(&scalar_problem)
            .expect("soft linear scalar solve");

        assert_eq!(batch_solution.num_dofs(), scalar.num_dofs());
        let error = batch_solution
            .coefficients()
            .iter()
            .zip(scalar.coefficients())
            .map(|(batch, scalar)| (*batch - *scalar).norm())
            .fold(0.0_f64, f64::max);
        // The blocked multi-RHS triangular solve intentionally uses a
        // different summation order than the single-RHS dot path, so compare
        // relative to the solution scale instead of an absolute budget.
        let scale = scalar
            .coefficients()
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        assert!(
            error / scale < 1.0e-8,
            "batched/scalar coefficient error: {error:e} (scale {scale:e})"
        );

        assert!(
            batch_solution
                .coefficients()
                .iter()
                .all(|value: &Complex64| value.re.is_finite() && value.im.is_finite())
        );
    }

    let num_dofs = batched[0].num_dofs();
    let rhs_columns = vec![
        (0..num_dofs)
            .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1))
            .collect::<Vec<_>>(),
        (0..num_dofs)
            .map(|index| Complex64::new(-0.3, 0.4 + index as f64 / 23.0))
            .collect::<Vec<_>>(),
    ];
    let adjoint_batch = solver
        .solve_adjoint_many(&problem, &rhs_columns)
        .expect("soft linear adjoint multi-RHS solve");
    assert_eq!(adjoint_batch.len(), rhs_columns.len());
    for (rhs, batch_solution) in rhs_columns.iter().zip(adjoint_batch) {
        let scalar = solver
            .solve_adjoint_many(&problem, std::slice::from_ref(rhs))
            .expect("soft linear adjoint scalar solve")
            .pop()
            .expect("soft linear adjoint scalar solution");
        let error = batch_solution
            .coefficients()
            .iter()
            .zip(scalar.coefficients())
            .map(|(batch, scalar)| (*batch - *scalar).norm())
            .fold(0.0_f64, f64::max);
        // Same scale-aware comparison as the forward path: residuals are
        // the correctness oracle, and both paths must agree to rounding.
        let scale = scalar
            .coefficients()
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        assert!(
            error / scale < 1.0e-8,
            "adjoint batched/scalar error: {error:e} (scale {scale:e})"
        );
        assert!(batch_solution.relative_residual() <= 1.0e-10);
        assert!(scalar.relative_residual() <= 1.0e-10);
    }
}
