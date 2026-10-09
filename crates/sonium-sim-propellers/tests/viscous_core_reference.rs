use sonium_sim_propellers::{
    ViscousCoreError, ViscousCoreOptions, evaluate_viscous_core_evolution,
};

fn options() -> ViscousCoreOptions {
    ViscousCoreOptions {
        initial_core_radius_m: 0.01,
        kinematic_viscosity_m2_s: 0.5,
        time_step_s: 0.01,
        steps: 4,
        maximum_steps: 16,
        probe_radius_m: 0.02,
        maximum_core_radius_m: 1.0,
    }
}

#[test]
fn viscous_core_matches_diffusion_and_lamb_oseen_velocity() {
    let result = evaluate_viscous_core_evolution(2.0, options()).expect("finite viscous core");
    let final_age_s = 0.04;
    let expected_radius = (0.01_f64.powi(2) + 4.0 * 0.5 * final_age_s).sqrt();
    let ratio = 0.02_f64.powi(2) / expected_radius.powi(2);
    let expected_velocity = 2.0 / (2.0 * std::f64::consts::PI * 0.02) * -(-ratio).exp_m1();

    assert!(result.is_finite());
    assert!((result.final_core_radius_m - expected_radius).abs() < 1.0e-14);
    assert!((result.final_tangential_velocity_m_s - expected_velocity).abs() < 1.0e-14);
    assert!(result.steps.windows(2).all(|pair| {
        pair[1].core_radius_m > pair[0].core_radius_m
            && pair[1].tangential_velocity_m_s < pair[0].tangential_velocity_m_s
    }));
}

#[test]
fn zero_viscosity_preserves_core_and_signed_circulation() {
    let result = evaluate_viscous_core_evolution(
        -3.0,
        ViscousCoreOptions {
            kinematic_viscosity_m2_s: 0.0,
            ..options()
        },
    )
    .expect("zero-viscosity core");

    assert!(result.is_finite());
    assert!(
        result
            .steps
            .iter()
            .all(|step| step.core_radius_m == options().initial_core_radius_m)
    );
    assert!(
        result
            .steps
            .iter()
            .all(|step| step.tangential_velocity_m_s < 0.0)
    );
    assert!(
        result
            .steps
            .iter()
            .all(|step| step.circulation_m2_s == -3.0)
    );
}

#[test]
fn viscous_core_evolution_is_deterministic() {
    let first = evaluate_viscous_core_evolution(1.25, options()).expect("first core evolution");
    let second = evaluate_viscous_core_evolution(1.25, options()).expect("second core evolution");

    assert_eq!(first, second);
}

#[test]
fn viscous_core_reference_rejects_invalid_and_budgeted_inputs() {
    assert!(matches!(
        evaluate_viscous_core_evolution(
            1.0,
            ViscousCoreOptions {
                steps: 0,
                ..options()
            }
        ),
        Err(ViscousCoreError::InvalidOptions { field: "steps", .. })
    ));
    assert!(matches!(
        evaluate_viscous_core_evolution(
            1.0,
            ViscousCoreOptions {
                steps: 5,
                maximum_steps: 4,
                ..options()
            }
        ),
        Err(ViscousCoreError::InvalidOptions { field: "steps", .. })
    ));
    assert!(matches!(
        evaluate_viscous_core_evolution(
            1.0,
            ViscousCoreOptions {
                maximum_core_radius_m: 0.02,
                ..options()
            }
        ),
        Err(ViscousCoreError::CoreRadiusExceeded { step: 0, .. })
    ));
    assert!(matches!(
        evaluate_viscous_core_evolution(f64::NAN, options()),
        Err(ViscousCoreError::InvalidCirculation)
    ));
}
