use num_complex::Complex64;
use sonium_bem::core::postprocess::pressure::{
    FieldEvaluationError, compute_scattered_field_checked, compute_scattered_field_gradient_checked,
};
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolver, BoundaryCondition, Element, ElementProperty,
    ElementType, PhysicsParams, SolverMethod,
};

fn fixture_problem() -> BemProblem {
    BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
}

fn solved_fixture() -> sonium_bem::core::BemSolution {
    let problem = fixture_problem();
    BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .solve(&problem)
        .expect("bounded dense rigid-sphere solve")
}

#[test]
fn checked_field_evaluation_matches_infallible_finite_results() {
    let solution = solved_fixture();
    let points = [[0.0, 0.0, 0.24], [0.12, 0.04, 0.22], [-0.16, 0.05, 0.28]];
    let checked = solution
        .evaluate_pressure_field_checked(&points)
        .expect("finite field evaluation");
    let infallible = solution.evaluate_pressure_field(&points);
    assert_eq!(checked.len(), points.len());
    for (left, right) in checked.iter().zip(infallible.iter()) {
        assert!((left.p_total - right.p_total).norm() <= 1.0e-12);
        assert!(left.p_total.re.is_finite() && left.p_total.im.is_finite());
        assert!(left.p_scattered.re.is_finite() && left.p_scattered.im.is_finite());
    }
}

#[test]
fn checked_field_evaluation_rejects_nonfinite_points_geometry_and_samples() {
    let solution = solved_fixture();
    let invalid_point = [[f64::NAN, 0.0, 0.24]];
    assert!(matches!(
        solution.evaluate_pressure_field_checked(&invalid_point),
        Err(FieldEvaluationError::NonFiniteEvaluationPoint { point: 0 })
    ));

    let mut invalid_geometry = solution.clone();
    invalid_geometry.nodes[0][0] = f64::NAN;
    assert!(matches!(
        invalid_geometry.evaluate_pressure_field_checked(&[[0.0, 0.0, 0.24]]),
        Err(FieldEvaluationError::NonFiniteNode { .. })
    ));

    let mut invalid_sample = solution.clone();
    invalid_sample.surface_pressure[0] = Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        invalid_sample.evaluate_pressure_field_checked(&[[0.0, 0.0, 0.24]]),
        Err(FieldEvaluationError::NonFiniteSurfaceSample { .. })
    ));
}

#[test]
fn checked_field_evaluation_rejects_malformed_basis_buffer() {
    let mut solution = solved_fixture();
    solution.surface_pressure.pop();
    let error = solution
        .evaluate_pressure_field_checked(&[[0.0, 0.0, 0.24]])
        .expect_err("malformed surface pressure must fail closed");
    assert!(matches!(
        error,
        FieldEvaluationError::SurfacePressureCount { .. }
    ));
}

#[test]
fn checked_pressure_gradient_matches_central_difference() {
    let solution = solved_fixture();
    let point = [0.12, 0.04, 0.22];
    let analytic = solution
        .evaluate_pressure_gradient_checked(&point)
        .expect("analytic pressure gradient");
    let step = 2.0e-5;
    for axis in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[axis] += step;
        minus[axis] -= step;
        let finite_difference =
            (solution.evaluate_pressure(&plus) - solution.evaluate_pressure(&minus)) / (2.0 * step);
        let scale = analytic[axis]
            .norm()
            .max(finite_difference.norm())
            .max(1.0e-10);
        assert!(
            (analytic[axis] - finite_difference).norm() / scale < 2.0e-3,
            "axis {axis}: analytic={:?}, finite_difference={:?}",
            analytic[axis],
            finite_difference
        );
    }
}

#[test]
fn checked_pressure_gradient_rejects_nonfinite_points() {
    let solution = solved_fixture();
    let error = solution
        .evaluate_pressure_gradient_field_checked(&[[f64::NAN, 0.0, 0.24]])
        .expect_err("non-finite gradient point must fail closed");
    assert!(matches!(
        error,
        FieldEvaluationError::NonFiniteEvaluationPoint { point: 0 }
    ));
}

#[test]
fn checked_linear_and_quadratic_gradients_match_field_differences() {
    let nodes = vec![[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.1, 0.7, 0.0]];
    let physics = PhysicsParams::try_new(800.0, 343.0, 1.21, false).expect("valid physics");
    let point = [0.2, 0.15, 0.9];

    for (basis_order, dof_addresses, surface_pressure) in [
        (
            sonium_bem::BasisOrder::Linear,
            vec![0, 1, 2],
            vec![
                Complex64::new(1.0, 0.1),
                Complex64::new(0.4, -0.2),
                Complex64::new(-0.3, 0.7),
            ],
        ),
        (
            sonium_bem::BasisOrder::Quadratic,
            vec![0, 1, 2, 3, 4, 5],
            vec![
                Complex64::new(1.0, 0.1),
                Complex64::new(0.4, -0.2),
                Complex64::new(-0.3, 0.7),
                Complex64::new(0.2, -0.4),
                Complex64::new(-0.5, 0.3),
                Complex64::new(0.6, 0.2),
            ],
        ),
    ] {
        let elements = vec![Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: Vec::new(),
            center: [0.3, 0.23333333333333334, 0.0],
            area: 0.28,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses,
        }];
        let analytic = compute_scattered_field_gradient_checked(
            &[point],
            &elements,
            &nodes,
            &surface_pressure,
            None,
            &physics,
            basis_order,
        )
        .expect("analytic basis gradient")[0];
        let step = 2.0e-5;
        for axis in 0..3 {
            let mut plus = point;
            let mut minus = point;
            plus[axis] += step;
            minus[axis] -= step;
            let plus = compute_scattered_field_checked(
                &[plus],
                &elements,
                &nodes,
                &surface_pressure,
                None,
                &physics,
                basis_order,
            )
            .expect("plus field")[0];
            let minus = compute_scattered_field_checked(
                &[minus],
                &elements,
                &nodes,
                &surface_pressure,
                None,
                &physics,
                basis_order,
            )
            .expect("minus field")[0];
            let finite_difference = (plus - minus) / (2.0 * step);
            let scale = analytic[axis]
                .norm()
                .max(finite_difference.norm())
                .max(1.0e-10);
            assert!(
                (analytic[axis] - finite_difference).norm() / scale < 3.0e-3,
                "{basis_order:?} axis {axis}: analytic={:?}, finite_difference={:?}",
                analytic[axis],
                finite_difference
            );
        }
    }
}
#[test]
fn checked_load_and_adjoint_fields_reject_invalid_points() {
    let problem = fixture_problem();
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let forward = solver.solve(&problem).expect("bounded dense solve");
    let rhs = forward.surface_pressure.clone();
    let loads = solver
        .solve_driver_loads(&problem, std::slice::from_ref(&rhs))
        .expect("driver/load solve");
    let adjoints = solver
        .solve_adjoint_many(&problem, std::slice::from_ref(&rhs))
        .expect("adjoint solve");
    let points = [[0.0, 0.0, 0.24], [0.12, 0.04, 0.22]];
    assert_eq!(loads.len(), 1);
    assert_eq!(adjoints.len(), 1);
    assert_eq!(
        loads[0]
            .evaluate_scattered_pressure_field_checked(&points)
            .expect("checked load field")
            .len(),
        2
    );
    assert_eq!(
        adjoints[0]
            .evaluate_scattered_pressure_field_checked(&points)
            .expect("checked adjoint field")
            .len(),
        2
    );
    assert!(matches!(
        loads[0].evaluate_scattered_pressure_field_checked(&[[f64::NAN, 0.0, 0.24]]),
        Err(FieldEvaluationError::NonFiniteEvaluationPoint { point: 0 })
    ));
    assert!(matches!(
        adjoints[0].evaluate_scattered_pressure_field_checked(&[[f64::NAN, 0.0, 0.24]]),
        Err(FieldEvaluationError::NonFiniteEvaluationPoint { point: 0 })
    ));
}
