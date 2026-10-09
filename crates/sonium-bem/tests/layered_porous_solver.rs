use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolver, BoundaryCondition, BoundaryConditionType, PhysicsParams,
    SolverMethod,
};
use sonium_bem::{AmbientMedium, LayeredPorousSurface, PorousLayer};

fn test_surface() -> LayeredPorousSurface {
    LayeredPorousSurface::try_new(vec![
        PorousLayer::miki(0.035, 12_000.0).with_air_gap(0.01),
        PorousLayer::delany_bazley(0.025, 18_000.0),
    ])
    .expect("valid porous stack")
}

#[test]
fn solver_prepares_frequency_dependent_layered_porous_admittance() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 800.0, 343.0, 1.21, 3, 4)
        .with_boundary_condition(BoundaryConditionType::Impedance);
    let surface = test_surface();
    let expected = surface
        .try_admittance(
            problem.physics.frequency,
            AmbientMedium::new(problem.physics.density, problem.physics.speed_of_sound),
        )
        .expect("finite expected admittance");
    let solution = BemSolver::new()
        .with_solver_method(SolverMethod::Direct)
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_layered_porous_surface(surface)
        .solve(&problem)
        .expect("layered porous solve");

    assert!(!solution.elements.is_empty());
    for element in &solution.elements {
        match &element.boundary_condition {
            BoundaryCondition::VelocityWithAdmittance {
                velocity,
                admittance,
            } => {
                assert_eq!(velocity, &[Complex64::new(0.0, 0.0)]);
                assert!((*admittance - expected).norm() < 1.0e-12);
            }
            other => panic!("unexpected prepared boundary condition: {other:?}"),
        }
    }
    assert!(
        solution
            .coefficients()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn frequency_sweep_refreshes_layered_porous_admittance_per_frequency() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 500.0, 343.0, 1.21, 3, 4)
        .with_boundary_condition(BoundaryConditionType::Impedance);
    let surface = test_surface();
    let solver = BemSolver::new()
        .with_tolerance(1.0e-10)
        .with_layered_porous_surface(surface.clone());

    let sweep = solver
        .solve_frequency_sweep(&problem, &[500.0, 1_000.0])
        .expect("layered porous frequency sweep");
    let mut second_problem = problem.clone();
    second_problem.physics = PhysicsParams::try_new(1_000.0, 343.0, 1.21, false)
        .expect("valid second-frequency physics");
    let reference = solver
        .solve(&second_problem)
        .expect("single-frequency reference");
    let expected_admittance = surface
        .try_admittance(1_000.0, AmbientMedium::new(1.21, 343.0))
        .expect("finite second-frequency admittance");
    assert!(sweep[1].elements.iter().all(|element| {
        matches!(
            element.boundary_condition,
            BoundaryCondition::VelocityWithAdmittance { admittance, .. }
                if (admittance - expected_admittance).norm() < 1.0e-12
        )
    }));
    let max_error = sweep[1]
        .coefficients()
        .iter()
        .zip(reference.coefficients())
        .map(|(swept, expected)| (*swept - *expected).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 1.0e-6,
        "stale porous sweep admittance: {max_error:e}"
    );
}

#[test]
fn layered_porous_surface_rejects_non_impedance_problems() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 800.0, 343.0, 1.21, 3, 4);
    let error = BemSolver::new()
        .with_layered_porous_surface(test_surface())
        .solve(&problem)
        .expect_err("porous model must not silently change a rigid boundary");
    assert!(error.to_string().contains("requires an impedance boundary"));
}
