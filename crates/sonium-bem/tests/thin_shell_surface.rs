use num_complex::Complex64;
use sonium_bem::core::mesh::BemReusePolicy;
use sonium_bem::core::{
    BemOperatorCache, BemProblem, BemSolver, BoundaryCondition, BoundaryConditionType,
    PhysicsParams,
};
use sonium_bem::materials::{AmbientMedium, LayeredPorousSurface, PorousLayer, ThinShellSurface};

fn shell() -> ThinShellSurface {
    ThinShellSurface::try_new(2.5, 1.8e7, 42.0).expect("valid shell")
}

#[test]
fn thin_shell_impedance_matches_mass_spring_damper_model() {
    let model = shell();
    let frequency = 800.0;
    let omega = 2.0 * std::f64::consts::PI * frequency;
    let expected = Complex64::new(42.0, 1.8e7 / omega - omega * 2.5);
    let actual = model
        .try_surface_impedance(frequency)
        .expect("finite impedance");
    assert!((actual - expected).norm() < 1.0e-12);
}

#[test]
fn thin_shell_admittance_and_frequency_derivative_match_centered_difference() {
    let model = shell();
    let frequency = 800.0;
    let step = 0.25;
    let analytic = model
        .try_admittance_frequency_derivative(frequency)
        .expect("finite derivative");
    let plus = model.try_admittance(frequency + step).expect("finite plus");
    let minus = model
        .try_admittance(frequency - step)
        .expect("finite minus");
    let centered = (plus - minus) / (2.0 * step);
    let scale = analytic.norm().max(centered.norm()).max(1.0);
    assert!((analytic - centered).norm() / scale < 2.0e-6);
}

#[test]
fn thin_shell_absorption_is_passive_and_finite() {
    let model = shell();
    let absorption = model
        .try_absorption_coefficient(800.0, AmbientMedium::default(), 1.0e-12)
        .expect("passive shell");
    assert!((0.0..=1.0).contains(&absorption));
}

#[test]
fn thin_shell_rejects_invalid_parameters_and_resonant_zero_impedance() {
    assert!(ThinShellSurface::try_new(0.0, 1.0, 1.0).is_err());
    assert!(ThinShellSurface::try_new(1.0, -1.0, 1.0).is_err());
    assert!(ThinShellSurface::try_new(1.0, 1.0, -1.0).is_err());
    assert!(shell().try_surface_impedance(f64::NAN).is_err());

    let undamped = ThinShellSurface::try_new(1.0, 4.0 * std::f64::consts::PI.powi(2), 0.0)
        .expect("valid undamped shell");
    assert!(
        undamped
            .try_admittance(1.0)
            .expect_err("resonance must not produce infinity")
            .to_string()
            .contains("non-finite")
    );
}

#[test]
fn solver_wires_thin_shell_admittance_into_impedance_elements() {
    let frequency = 500.0;
    let medium = AmbientMedium::default();
    let shell = shell();
    let expected = shell
        .try_admittance(frequency)
        .expect("finite shell admittance");
    let problem = BemProblem::rigid_sphere_scattering_custom(
        0.1,
        frequency,
        medium.speed_of_sound,
        medium.density,
        3,
        4,
    )
    .with_boundary_condition(BoundaryConditionType::Impedance);
    let solution = BemSolver::new()
        .with_thin_shell_surface(shell)
        .solve(&problem)
        .expect("thin-shell impedance solve");
    assert!(solution.elements.iter().all(|element| {
        matches!(
            element.boundary_condition,
            BoundaryCondition::VelocityWithAdmittance { admittance, .. }
                if (admittance - expected).norm() < 1.0e-12
        )
    }));
}

#[test]
fn frequency_sweep_refreshes_thin_shell_admittance_per_frequency() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 500.0, 343.0, 1.21, 3, 4)
        .with_boundary_condition(BoundaryConditionType::Impedance);
    let shell = shell();
    let solver = BemSolver::new()
        .with_tolerance(1.0e-10)
        .with_thin_shell_surface(shell);

    let sweep = solver
        .solve_frequency_sweep(&problem, &[500.0, 1_000.0])
        .expect("thin-shell frequency sweep");
    let mut second_problem = problem.clone();
    second_problem.physics = PhysicsParams::try_new(1_000.0, 343.0, 1.21, false)
        .expect("valid second-frequency physics");
    let reference = solver
        .solve(&second_problem)
        .expect("single-frequency reference");
    let expected_admittance = shell
        .try_admittance(1_000.0)
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
        "frequency sweep used stale shell admittance: error={max_error:e}"
    );
}

#[test]
fn solver_rejects_thin_shell_on_non_impedance_or_with_porous_surface() {
    let shell = shell();
    let rigid_problem = BemProblem::rigid_sphere_scattering_custom(0.1, 500.0, 343.0, 1.21, 3, 4);
    assert!(
        BemSolver::new()
            .with_thin_shell_surface(shell)
            .solve(&rigid_problem)
            .is_err()
    );

    let impedance_problem = rigid_problem.with_boundary_condition(BoundaryConditionType::Impedance);
    let porous = LayeredPorousSurface::try_new(vec![PorousLayer::miki(0.03, 12_000.0)])
        .expect("valid porous surface");
    assert!(
        BemSolver::new()
            .with_layered_porous_surface(porous)
            .with_thin_shell_surface(shell)
            .solve(&impedance_problem)
            .is_err()
    );
}

#[test]
fn changing_thin_shell_configuration_invalidates_persistent_operator_cache() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 500.0, 343.0, 1.21, 3, 4)
        .with_boundary_condition(BoundaryConditionType::Impedance);
    let first_shell = shell();
    let second_shell = ThinShellSurface::try_new(3.0, 1.8e7, 42.0).expect("valid shell");
    let first_solver = BemSolver::new().with_thin_shell_surface(first_shell);
    let second_solver = BemSolver::new().with_thin_shell_surface(second_shell);
    let mut cache = BemOperatorCache::default();
    first_solver
        .solve_with_cache(&problem, &mut cache, BemReusePolicy::default())
        .expect("initial cached shell solve");
    second_solver
        .solve_with_cache(&problem, &mut cache, BemReusePolicy::default())
        .expect("changed shell must rebuild");
    assert_eq!(cache.rebuilds(), 2);
    assert_eq!(cache.reuse_hits(), 0);
}
