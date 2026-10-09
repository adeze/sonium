use sonium_sim_propellers::{
    DynamicStallError, DynamicStallInput, DynamicStallOptions, evaluate_dynamic_stall_reference,
};

fn options() -> DynamicStallOptions {
    DynamicStallOptions {
        separation_onset_angle_deg: 10.0,
        separation_transition_width_deg: 2.0,
        separated_lift_fraction: 0.25,
        lift_time_constant_s: 0.01,
        separation_time_constant_s: 0.05,
        recovery_time_constant_s: 0.15,
        maximum_step_s: 0.1,
        ..DynamicStallOptions::default()
    }
}

#[test]
fn dynamic_stall_has_delayed_separation_and_lift_recovery() {
    let result = evaluate_dynamic_stall_reference(
        &[
            DynamicStallInput {
                time_s: 0.0,
                angle_of_attack_deg: 0.0,
            },
            DynamicStallInput {
                time_s: 0.01,
                angle_of_attack_deg: 20.0,
            },
            DynamicStallInput {
                time_s: 0.02,
                angle_of_attack_deg: 20.0,
            },
            DynamicStallInput {
                time_s: 0.08,
                angle_of_attack_deg: 0.0,
            },
        ],
        options(),
    )
    .expect("valid dynamic stall history");

    assert!(result.is_finite());
    assert!(result.samples[1].separation_target > result.samples[1].separation_state);
    assert!(result.samples[1].dynamic_lift_coefficient > 0.0);
    assert!(result.samples[3].separation_state > 0.0);
    assert!(result.samples[3].dynamic_lift_coefficient > 0.0);
    assert!(result.samples[3].separation_state < result.samples[1].separation_target);
    assert!(result.samples[3].separation_state < result.samples[2].separation_state);
}

#[test]
fn dynamic_stall_matches_exact_first_order_lift_relaxation_below_stall() {
    let configured = options();
    let result = evaluate_dynamic_stall_reference(
        &[
            DynamicStallInput {
                time_s: 0.0,
                angle_of_attack_deg: 0.0,
            },
            DynamicStallInput {
                time_s: 0.02,
                angle_of_attack_deg: 5.0,
            },
        ],
        configured,
    )
    .expect("valid attached-flow history");
    let attached = configured.lift_slope_per_rad * 5.0_f64.to_radians();
    let expected = attached * (1.0 - (-0.02 / configured.lift_time_constant_s).exp());

    assert_eq!(result.samples[1].separation_state, 0.0);
    assert!((result.samples[1].dynamic_lift_coefficient - expected).abs() < 1.0e-12);
}

#[test]
fn dynamic_stall_is_deterministic_and_signed_for_negative_angle() {
    let history = [
        DynamicStallInput {
            time_s: 0.0,
            angle_of_attack_deg: 0.0,
        },
        DynamicStallInput {
            time_s: 0.02,
            angle_of_attack_deg: -18.0,
        },
        DynamicStallInput {
            time_s: 0.04,
            angle_of_attack_deg: -18.0,
        },
    ];
    let first = evaluate_dynamic_stall_reference(&history, options()).expect("first replay");
    let second = evaluate_dynamic_stall_reference(&history, options()).expect("second replay");

    assert_eq!(first, second);
    assert!(first.samples[1].dynamic_lift_coefficient < 0.0);
    assert!(first.maximum_separation_state > 0.0);
}

#[test]
fn dynamic_stall_reference_rejects_invalid_history_options_and_steps() {
    assert!(matches!(
        evaluate_dynamic_stall_reference(&[], options()),
        Err(DynamicStallError::EmptyHistory)
    ));
    assert!(matches!(
        evaluate_dynamic_stall_reference(
            &[
                DynamicStallInput {
                    time_s: 0.0,
                    angle_of_attack_deg: 0.0,
                },
                DynamicStallInput {
                    time_s: 0.2,
                    angle_of_attack_deg: 5.0,
                },
            ],
            options(),
        ),
        Err(DynamicStallError::StepTooLarge { index: 1, .. })
    ));
    assert!(matches!(
        evaluate_dynamic_stall_reference(
            &[DynamicStallInput {
                time_s: 0.0,
                angle_of_attack_deg: f64::NAN,
            }],
            options(),
        ),
        Err(DynamicStallError::InvalidInput {
            field: "angle_of_attack_deg",
            ..
        })
    ));
    assert!(matches!(
        evaluate_dynamic_stall_reference(
            &[DynamicStallInput {
                time_s: 0.0,
                angle_of_attack_deg: 0.0,
            }],
            DynamicStallOptions {
                separated_lift_fraction: 2.0,
                ..options()
            },
        ),
        Err(DynamicStallError::InvalidOptions {
            field: "separated_lift_fraction",
            ..
        })
    ));
}
