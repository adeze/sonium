//! Accuracy evidence for implicit integrators: Newmark refinement order,
//! generalized-alpha annihilation at `rho_inf = 0`, multi-step
//! adjoint-vs-finite-difference gradients (damped and undamped), and the
//! zero-step contract.

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_time_integration::{
    ExplicitConfig, GeneralizedAlphaConfig, NewmarkConfig, NewmarkInitial,
    generalized_alpha_adjoint, generalized_alpha_integrate, newmark_adjoint, newmark_integrate,
    try_explicit_integrate, try_newmark_integrate,
};

fn sdof(k: f64) -> (Option<CsrMatrix<f64>>, CsrMatrix<f64>) {
    let stiffness = CsrMatrix::from_dense(&DenseMatrix::from_row_major_vec(1, 1, vec![k]), 0.0);
    (None, stiffness)
}

fn initial_state(u0: f64, v0: f64, k: f64) -> NewmarkInitial<f64> {
    NewmarkInitial {
        displacement: vec![u0],
        velocity: vec![v0],
        acceleration: vec![-k * u0],
    }
}

#[test]
fn newmark_average_acceleration_is_second_order() {
    // u'' + 4u = 0, u(0) = 1: halving dt must quarter the endpoint error.
    let k = 4.0;
    let omega = 2.0;
    let t_final = 1.25 * 2.0 * std::f64::consts::PI / omega;
    let u_exact = (omega * t_final).cos();
    let (_, stiffness) = sdof(k);
    let mut errors = Vec::new();
    for &num_steps in &[40_usize, 80, 160] {
        let dt = t_final / num_steps as f64;
        let config = NewmarkConfig::average_acceleration(dt, num_steps);
        let eff = 1.0 + 0.25 * dt * dt * k;
        let result = newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &stiffness,
            |_t| vec![0.0],
            &initial_state(1.0, 0.0, k),
            |rhs: &[f64]| vec![rhs[0] / eff],
            &config,
        );
        errors.push((result.displacement[0] - u_exact).abs());
    }
    for window in errors.windows(2) {
        let ratio = window[0] / window[1];
        assert!(
            (3.5..=4.5).contains(&ratio),
            "expected 2nd order (ratio ~4), got {ratio:.2} ({errors:?})"
        );
    }
}

#[test]
fn generalized_alpha_rho_zero_annihilates_high_frequency() {
    // omega*dt = 4 >> 1 with rho_inf = 0: maximal high-frequency
    // dissipation must kill the response within a few steps.
    let k = 4.0;
    let dt = 2.0;
    let num_steps = 20;
    let config = GeneralizedAlphaConfig::from_spectral_radius(0.0, dt, num_steps);
    let mass = CsrMatrix::from_dense(&DenseMatrix::from_row_major_vec(1, 1, vec![1.0]), 0.0);
    let (_, stiffness) = sdof(k);
    let eff = (1.0 - config.alpha_m) + (1.0 - config.alpha_f) * config.beta * dt * dt * k;
    let result = generalized_alpha_integrate(
        &mass,
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_t| vec![0.0],
        &initial_state(1.0, 0.0, k),
        |rhs: &[f64]| vec![rhs[0] / eff],
        &config,
    );
    assert!(
        result.displacement[0].abs() < 0.05,
        "rho_inf = 0 must annihilate omega*dt = 4 response, got {}",
        result.displacement[0]
    );
}

fn adjoint_matches_finite_difference(damping: Option<f64>) {
    // Forced SDOF u'' + c u' + u = t, J = u(T): dJ/du0 from the stored
    // adjoint must match central finite differences.
    let k = 1.0;
    let dt = 0.05;
    let num_steps = 40;
    let config = NewmarkConfig::average_acceleration(dt, num_steps).with_history();
    let damping_matrix;
    let damping_ref = if let Some(c) = damping {
        damping_matrix =
            CsrMatrix::from_dense(&DenseMatrix::from_row_major_vec(1, 1, vec![c]), 0.0);
        Some(&damping_matrix)
    } else {
        None
    };
    let (_, stiffness) = sdof(k);
    let gamma = 0.5;
    let beta = 0.25;
    let eff = 1.0 + gamma * dt * damping.unwrap_or(0.0) + beta * dt * dt * k;
    let solve = |rhs: &[f64]| vec![rhs[0] / eff];
    // The adjoint differentiates w.r.t. the independent inputs
    // (u0, v0, a0), so the FD probe holds a0 fixed at its base value.
    let base_a0 = -k * 0.3;
    let objective = |u0: f64| {
        let initial = NewmarkInitial {
            displacement: vec![u0],
            velocity: vec![0.0],
            acceleration: vec![base_a0],
        };
        newmark_integrate(
            damping_ref,
            &stiffness,
            |t| vec![t],
            &initial,
            solve,
            &config,
        )
        .displacement[0]
    };
    let forward = {
        let initial = NewmarkInitial {
            displacement: vec![0.3],
            velocity: vec![0.0],
            acceleration: vec![base_a0],
        };
        newmark_integrate(
            damping_ref,
            &stiffness,
            |t| vec![t],
            &initial,
            solve,
            &config,
        )
    };
    let adjoint = newmark_adjoint(
        damping_ref,
        &stiffness,
        &forward,
        &[1.0],
        &[0.0],
        &[0.0],
        |rhs: &[f64]| Ok(vec![rhs[0] / eff]),
        &config,
    )
    .expect("multi-step adjoint must succeed");
    let h = 1.0e-6;
    let fd = (objective(0.3 + h) - objective(0.3 - h)) / (2.0 * h);
    let got = adjoint.initial_displacement_gradient[0];
    assert!(
        (got - fd).abs() < 1.0e-5 * fd.abs().max(1.0),
        "adjoint {got:.6} vs FD {fd:.6} (damping {damping:?})"
    );
}

#[test]
fn newmark_adjoint_matches_fd_undamped() {
    adjoint_matches_finite_difference(None);
}

#[test]
fn newmark_adjoint_matches_fd_damped() {
    adjoint_matches_finite_difference(Some(0.2));
}

#[test]
fn generalized_alpha_adjoint_matches_fd() {
    // Same FD protocol for the generalized-alpha reverse pass (rho_inf =
    // 0.9, undamped): the direct-seed fix must hold for this recurrence too.
    let k = 1.0;
    let dt = 0.05;
    let num_steps = 40;
    let config = GeneralizedAlphaConfig::from_spectral_radius(0.9, dt, num_steps).with_history();
    let mass = CsrMatrix::from_dense(&DenseMatrix::from_row_major_vec(1, 1, vec![1.0]), 0.0);
    let (_, stiffness) = sdof(k);
    let eff = (1.0 - config.alpha_m) + (1.0 - config.alpha_f) * config.beta * dt * dt * k;
    let solve = |rhs: &[f64]| vec![rhs[0] / eff];
    let base_a0 = -k * 0.3;
    let objective = |u0: f64| {
        let initial = NewmarkInitial {
            displacement: vec![u0],
            velocity: vec![0.0],
            acceleration: vec![base_a0],
        };
        generalized_alpha_integrate(
            &mass,
            None::<&CsrMatrix<f64>>,
            &stiffness,
            |t| vec![t],
            &initial,
            solve,
            &config,
        )
        .displacement[0]
    };
    let initial = NewmarkInitial {
        displacement: vec![0.3],
        velocity: vec![0.0],
        acceleration: vec![base_a0],
    };
    let forward = generalized_alpha_integrate(
        &mass,
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |t| vec![t],
        &initial,
        solve,
        &config,
    );
    let adjoint = generalized_alpha_adjoint(
        &mass,
        None::<&CsrMatrix<f64>>,
        &stiffness,
        &forward,
        &[1.0],
        &[0.0],
        &[0.0],
        |rhs: &[f64]| Ok(vec![rhs[0] / eff]),
        &config,
    )
    .expect("multi-step GA adjoint must succeed");
    let h = 1.0e-6;
    let fd = (objective(0.3 + h) - objective(0.3 - h)) / (2.0 * h);
    let got = adjoint.initial_displacement_gradient[0];
    assert!(
        (got - fd).abs() < 1.0e-5 * fd.abs().max(1.0),
        "GA adjoint {got:.6} vs FD {fd:.6}"
    );
}

#[test]
fn zero_steps_returns_initial_state() {
    // num_steps = 0 is a valid configuration: the trajectory is a single
    // point — the initial state at t = 0.
    let k = 4.0;
    let (_, stiffness) = sdof(k);
    let initial = initial_state(1.0, 0.5, k);
    let config = NewmarkConfig::average_acceleration(0.1, 0).with_history();
    let result = try_newmark_integrate(
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_t| vec![0.0],
        &initial,
        |rhs: &[f64]| rhs.to_vec(),
        &config,
    )
    .expect("zero steps must be accepted");
    assert_eq!(result.steps_completed, 0);
    assert_eq!(result.final_time, 0.0);
    assert_eq!(result.displacement, initial.displacement);
    assert_eq!(result.velocity, initial.velocity);
    assert_eq!(
        result.displacement_history.expect("history").len(),
        1,
        "history holds the single initial point"
    );

    let explicit = ExplicitConfig {
        dt: 0.1,
        num_steps: 0,
        store_history: true,
    };
    let result = try_explicit_integrate(
        |_: &[f64], _: &[f64], _: f64| vec![0.0],
        |rhs: &[f64]| rhs.to_vec(),
        &initial,
        &explicit,
    )
    .expect("zero explicit steps must be accepted");
    assert_eq!(result.steps_completed, 0);
    assert_eq!(result.displacement, initial.displacement);
}
