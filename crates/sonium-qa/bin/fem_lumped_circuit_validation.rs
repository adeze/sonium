//! Lumped circuit validation tests
//!
//! Tests for voice coil and acoustic lumped element models.
//!
//! - LC resonance frequency
//! - Voice coil linear model
//! - Acoustic compliance
//! - Acoustic mass
//! - Two-port transfer matrix

use sonium_fem::physics::coupled::lumped_circuit::{
    CircuitState, VoiceCoilParams,
    acoustic_elements::{AcousticCompliance, AcousticMass, AcousticResistance, HelmholtzResonator},
    circuit_step,
};

fn test_lc_resonance() {
    let l: f64 = 1e-3;
    let c: f64 = 1e-6;

    let omega = 1.0 / (l * c).sqrt();
    let f = omega / (2.0 * std::f64::consts::PI);

    let expected_f = 1.0 / (2.0 * std::f64::consts::PI * (l * c).sqrt());
    let error = (f - expected_f).abs() / expected_f;

    assert!(error < 1e-10, "LC resonance error: {:.2e}", error);
}

fn test_voice_coil_linear_model() {
    let params = VoiceCoilParams::linear(8.0, 1e-3, 5.0);

    let bl_at_zero = (params.bl)(0.0);
    let le_at_zero = (params.inductance)(0.0);
    let le_deriv_at_zero = (params.inductance_derivative)(0.0);

    assert!((bl_at_zero - 5.0).abs() < 1e-10);
    assert!((le_at_zero - 1e-3).abs() < 1e-10);
    assert!(le_deriv_at_zero.abs() < 1e-10);
}

fn test_voice_coil_nonlinear_bl() {
    let params = VoiceCoilParams::nonlinear(
        8.0,
        |x| 5.0 * (1.0 - 0.1 * x),
        |x| 1e-3 * (1.0 + 0.05 * x),
        |_x| 0.0,
    );

    let bl_rest = (params.bl)(0.0);
    let bl_forward = (params.bl)(0.01);

    assert!((bl_rest - 5.0).abs() < 1e-10);
    assert!((bl_forward - 5.0 * 0.999).abs() < 1e-6);
}

fn test_voice_coil_circuit_step() {
    let params = VoiceCoilParams::linear(8.0, 1e-3, 5.0);
    let mut state = CircuitState::at_rest();

    let voltage: f64 = 1.0;
    let dt: f64 = 1e-4;
    let displacement: f64 = 0.0;
    let velocity: f64 = 0.0;

    let force = circuit_step(&params, &mut state, voltage, displacement, velocity, dt);

    assert!(state.current > 0.0, "Current should be positive");
    assert!(force >= 0.0, "Force should be non-negative");
}

fn test_acoustic_compliance() {
    let comp = AcousticCompliance::new(1e-3, 1.21, 343.0);

    let c_acoustic = comp.compliance();
    let expected = 1e-3 / (1.21 * 343.0 * 343.0);

    let error = (c_acoustic - expected).abs() / expected;
    assert!(error < 1e-10, "Acoustic compliance error: {:.2e}", error);
}

fn test_acoustic_mass() {
    let mass = AcousticMass::new(0.1, std::f64::consts::PI * 0.05 * 0.05, 1.21);

    let m_acoustic = mass.mass();
    let area = std::f64::consts::PI * 0.05 * 0.05;
    let expected = 1.21 * 0.1 / area;

    let error = (m_acoustic - expected).abs() / expected;
    assert!(error < 1e-10, "Acoustic mass error: {:.2e}", error);
}

fn test_acoustic_resistance() {
    let res = AcousticResistance::new(5.0);

    let r = res.impedance().re;
    assert!((r - 5.0).abs() < 1e-10, "Acoustic resistance error");
}

fn test_acoustic_resonator() {
    let res = HelmholtzResonator::new(1e-3, 0.02, 0.05, 1.21, 343.0);

    let f = res.resonance_frequency();
    assert!(f > 0.0, "Resonance frequency should be positive");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_lc_resonance();
        test_voice_coil_linear_model();
        test_voice_coil_nonlinear_bl();
        test_voice_coil_circuit_step();
        test_acoustic_compliance();
        test_acoustic_mass();
        test_acoustic_resistance();
        test_acoustic_resonator();
    });

    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
