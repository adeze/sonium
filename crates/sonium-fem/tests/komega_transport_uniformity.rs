//! k-omega transport operator verification: uniformity preservation,
//! positivity, determinism, and input admission.
//!
//! This pins the reusable segregated-transport operator. Closure *validation*
//! against turbulence benchmarks (channel, flat plate) rides the coupled
//! production CFD path owned by the pcfan track, not this operator test.

// Rust guideline compliant 2026-02-21

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::physics::fluid::turbulence::{
    KOmegaTransportConfig, KOmegaTransportState, advance_komega_bdf_step,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn check_uniform(name: &str, field: &[f64]) {
    let mean = field.iter().sum::<f64>() / field.len() as f64;
    assert!(mean.is_finite() && mean > 0.0, "{name}: mean = {mean}");
    let max_deviation = field
        .iter()
        .map(|value| (value - mean).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_deviation / mean < 1.0e-9,
        "{name}: max deviation = {max_deviation}, mean = {mean}"
    );
    assert!(field.iter().all(|value| value.is_finite() && *value > 0.0));
}

#[test]
fn uniform_state_stays_uniform_with_frozen_coefficients() {
    let mesh = unit_square_triangles(4);
    let space = H1Space::new(PolynomialDegree::P1);
    let dofs = space.num_global_dofs(&mesh);
    let config = KOmegaTransportConfig {
        dt: 1.0e-3,
        bdf_order: 1,
        ..KOmegaTransportConfig::default()
    };
    let state = KOmegaTransportState::from_constant(dofs, 1.0e-3, 10.0, 0.0, 0)
        .expect("valid constant state");
    let step = advance_komega_bdf_step(&mesh, PolynomialDegree::P1, &config, &state)
        .expect("transport advance");
    check_uniform("k", &step.state.k);
    check_uniform("omega", &step.state.omega);
    assert!(step.report.k_residual_l2.is_finite());
    assert!(step.report.omega_residual_l2.is_finite());
    assert_eq!(step.report.bdf_order, 1);
    assert_eq!(step.state.step, 1);

    // Determinism: identical inputs reproduce identical fields bit-for-bit.
    let repeat = KOmegaTransportState::from_constant(dofs, 1.0e-3, 10.0, 0.0, 0)
        .expect("valid constant state");
    let second = advance_komega_bdf_step(&mesh, PolynomialDegree::P1, &config, &repeat)
        .expect("transport advance");
    assert_eq!(step.state.k, second.state.k);
    assert_eq!(step.state.omega, second.state.omega);
}

#[test]
fn invalid_transport_inputs_are_rejected() {
    assert!(KOmegaTransportState::from_constant(0, 1.0e-3, 10.0, 0.0, 0).is_err());
    assert!(KOmegaTransportState::from_constant(4, 0.0, 10.0, 0.0, 0).is_err());
    assert!(KOmegaTransportState::from_constant(4, 1.0e-3, -1.0, 0.0, 0).is_err());
    assert!(KOmegaTransportState::from_constant(4, f64::NAN, 10.0, 0.0, 0).is_err());

    let mesh = unit_square_triangles(2);
    let space = H1Space::new(PolynomialDegree::P1);
    let dofs = space.num_global_dofs(&mesh);
    let config = KOmegaTransportConfig::default();
    let mut state = KOmegaTransportState::from_constant(dofs, 1.0e-3, 10.0, 0.0, 0)
        .expect("valid constant state");
    state.k.pop();
    assert!(advance_komega_bdf_step(&mesh, PolynomialDegree::P1, &config, &state).is_err());
}
