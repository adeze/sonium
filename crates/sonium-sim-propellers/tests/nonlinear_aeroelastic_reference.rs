use sonium_sim_propellers::{
    NonlinearAeroelasticModalError, NonlinearAeroelasticModalOptions,
    evaluate_nonlinear_aeroelastic_modal,
};

fn options() -> NonlinearAeroelasticModalOptions {
    NonlinearAeroelasticModalOptions {
        effective_mass_kg: 0.2,
        natural_frequency_hz: 100.0,
        damping_ratio: 0.02,
        cubic_stiffness_n_m3: 0.0,
        forcing_amplitude_n: 1.0,
        forcing_frequency_hz: 80.0,
        modal_stress_per_m_pa: 1_000.0,
        maximum_amplitude_m: 0.1,
        maximum_stress_pa: 500.0,
        bracket_samples: 64,
        max_iterations: 100,
        residual_tolerance_n: 1.0e-10,
    }
}

#[test]
fn zero_cubic_stiffness_matches_linear_frequency_response() {
    let options = options();
    let result = evaluate_nonlinear_aeroelastic_modal(options).expect("linear root should exist");
    let natural_omega = 2.0 * std::f64::consts::PI * options.natural_frequency_hz;
    let forcing_omega = 2.0 * std::f64::consts::PI * options.forcing_frequency_hz;
    let stiffness = options.effective_mass_kg * natural_omega.powi(2);
    let damping =
        2.0 * options.damping_ratio * options.effective_mass_kg * natural_omega * forcing_omega;
    let expected = options.forcing_amplitude_n
        / ((stiffness - options.effective_mass_kg * forcing_omega.powi(2)).powi(2)
            + damping.powi(2))
        .sqrt();

    assert!((result.amplitude_m - expected).abs() < 1.0e-10);
    assert!((result.nonlinear_to_linear_amplitude_ratio - 1.0).abs() < 1.0e-10);
    assert!((result.nonlinear_residual_n) <= options.residual_tolerance_n);
    assert!(result.feasible);
    assert!(result.is_feasible(&options) == result.feasible);
}

#[test]
fn hardening_at_resonance_reduces_amplitude_against_linear_response() {
    let options = NonlinearAeroelasticModalOptions {
        forcing_frequency_hz: 100.0,
        cubic_stiffness_n_m3: 50_000.0,
        maximum_stress_pa: 10_000.0,
        ..options()
    };
    let result =
        evaluate_nonlinear_aeroelastic_modal(options).expect("hardening root should exist");

    assert!(result.amplitude_m < result.linear_amplitude_m);
    assert!(result.nonlinear_to_linear_amplitude_ratio < 1.0);
    assert!(result.phase_rad > 0.0);
    assert!(result.is_finite());
}

#[test]
fn softening_no_root_within_declared_amplitude_is_reported() {
    let result = evaluate_nonlinear_aeroelastic_modal(NonlinearAeroelasticModalOptions {
        cubic_stiffness_n_m3: -1.0e9,
        maximum_amplitude_m: 1.0e-6,
        ..options()
    });

    assert!(matches!(
        result,
        Err(NonlinearAeroelasticModalError::NoRootWithinAmplitude { .. })
    ));
}

#[test]
fn amplitude_and_stress_gates_are_reported_without_hiding_finite_response() {
    let options = NonlinearAeroelasticModalOptions {
        maximum_stress_pa: 1.0e-3,
        ..options()
    };
    let result = evaluate_nonlinear_aeroelastic_modal(options).expect("response root should exist");

    assert!(result.is_finite());
    assert!(!result.feasible);
    assert!(!result.is_feasible(&options));
    assert!(result.alternating_stress_pa > options.maximum_stress_pa);
}

#[test]
fn malformed_options_fail_closed_before_root_search() {
    let invalid_damping = evaluate_nonlinear_aeroelastic_modal(NonlinearAeroelasticModalOptions {
        damping_ratio: 0.0,
        ..options()
    });
    assert!(matches!(
        invalid_damping,
        Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "damping_ratio",
            ..
        })
    ));

    let invalid_cubic = evaluate_nonlinear_aeroelastic_modal(NonlinearAeroelasticModalOptions {
        cubic_stiffness_n_m3: f64::INFINITY,
        ..options()
    });
    assert!(matches!(
        invalid_cubic,
        Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "cubic_stiffness_n_m3",
            ..
        })
    ));
}
