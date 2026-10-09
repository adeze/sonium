use sonium_sim_propellers::{
    MultiRotorReferenceError, MultiRotorReferenceOptions, MultiRotorReferenceRotor,
    evaluate_multi_rotor_reference,
};

fn rotor(center_m: [f64; 3], axis: [f64; 3]) -> MultiRotorReferenceRotor {
    MultiRotorReferenceRotor {
        center_m,
        axis,
        radius_m: 0.5,
        thrust_n: 10.0,
    }
}

#[test]
fn single_rotor_matches_hover_momentum_reference() {
    let result = evaluate_multi_rotor_reference(
        &[rotor([0.0, 0.0, 0.0], [0.0, 0.0, 2.0])],
        MultiRotorReferenceOptions::default(),
    )
    .expect("single rotor should be admitted");
    let expected = (10.0 / (2.0 * 1.225 * std::f64::consts::PI * 0.5_f64.powi(2))).sqrt();

    assert!((result.rotors[0].isolated_induced_velocity_m_s - expected).abs() < 1.0e-12);
    assert_eq!(result.rotors[0].external_induced_velocity_m_s, 0.0);
    assert_eq!(result.pairwise_induced_velocity_m_s, vec![vec![0.0]]);
    assert!(result.is_finite());
}

#[test]
fn downstream_rotor_retains_signed_pairwise_wake_evidence() {
    let result = evaluate_multi_rotor_reference(
        &[
            rotor([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            rotor([0.0, 0.0, 1.5], [0.0, 0.0, 1.0]),
        ],
        MultiRotorReferenceOptions::default(),
    )
    .expect("aligned rotors should be admitted");

    assert_eq!(result.pair_evaluation_count, 2);
    assert!(result.pairwise_induced_velocity_m_s[1][0] > 0.0);
    assert_eq!(result.pairwise_induced_velocity_m_s[0][1], 0.0);
    assert_eq!(
        result.rotors[1].external_induced_velocity_m_s,
        result.pairwise_induced_velocity_m_s[1][0]
    );
    assert!(
        result.rotors[1].effective_axial_velocity_m_s
            > result.rotors[1].isolated_induced_velocity_m_s
    );
    assert!(result.is_finite());
}

#[test]
fn transverse_receiving_axis_projects_wake_to_zero() {
    let result = evaluate_multi_rotor_reference(
        &[
            rotor([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            rotor([0.0, 0.0, 1.5], [1.0, 0.0, 0.0]),
        ],
        MultiRotorReferenceOptions::default(),
    )
    .expect("transverse rotors should be admitted");

    assert!(result.pairwise_induced_velocity_m_s[1][0].abs() < 1.0e-12);
    assert!(result.rotors[1].external_induced_velocity_m_s.abs() < 1.0e-12);
}

#[test]
fn invalid_inputs_and_pair_budget_fail_closed() {
    let invalid_axis = evaluate_multi_rotor_reference(
        &[rotor([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])],
        MultiRotorReferenceOptions::default(),
    );
    assert!(matches!(
        invalid_axis,
        Err(MultiRotorReferenceError::InvalidRotor { field: "axis", .. })
    ));

    let budget = evaluate_multi_rotor_reference(
        &[
            rotor([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            rotor([2.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ],
        MultiRotorReferenceOptions {
            max_pair_evaluations: 1,
            ..MultiRotorReferenceOptions::default()
        },
    );
    assert!(matches!(
        budget,
        Err(MultiRotorReferenceError::PairEvaluationBudgetExceeded {
            required: 2,
            maximum: 1,
        })
    ));
}
