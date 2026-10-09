use sonium_sim_propellers::{
    FatigueCycle, FatigueError, FatigueOptions, evaluate_s_n_fatigue_reference,
};

fn options() -> FatigueOptions {
    FatigueOptions {
        ultimate_strength_pa: 200.0,
        reference_stress_pa: 100.0,
        reference_cycles: 1_000.0,
        sn_slope: 1.0,
        endurance_limit_pa: 10.0,
        safety_factor: 1.0,
    }
}

#[test]
fn fatigue_reference_applies_goodman_and_miner_damage() {
    let result = evaluate_s_n_fatigue_reference(
        &[
            FatigueCycle {
                mean_stress_pa: 0.0,
                alternating_stress_pa: 100.0,
                cycles: 100.0,
            },
            FatigueCycle {
                mean_stress_pa: 50.0,
                alternating_stress_pa: 50.0,
                cycles: 100.0,
            },
        ],
        options(),
    )
    .expect("valid fatigue spectrum");

    assert!(result.is_finite());
    assert!(result.is_feasible());
    assert!(
        (result.cycles[1].corrected_alternating_stress_pa - 66.666_666_666_666_67).abs() < 1.0e-12
    );
    assert!((result.total_damage_fraction - (0.1 + 1.0 / 15.0)).abs() < 1.0e-12);
    assert!(result.predicted_life_blocks.is_some());
}

#[test]
fn endurance_limited_cycles_contribute_zero_damage() {
    let result = evaluate_s_n_fatigue_reference(
        &[FatigueCycle {
            mean_stress_pa: -100.0,
            alternating_stress_pa: 1.0,
            cycles: 1.0e12,
        }],
        options(),
    )
    .expect("endurance-limited spectrum");

    assert!(result.is_finite());
    assert!(result.is_feasible());
    assert_eq!(result.total_damage_fraction, 0.0);
    assert_eq!(result.predicted_life_blocks, None);
    assert_eq!(result.cycles[0].allowable_cycles, None);
}

#[test]
fn damage_scales_with_repeated_spectrum_blocks() {
    let result = evaluate_s_n_fatigue_reference(
        &[FatigueCycle {
            mean_stress_pa: 0.0,
            alternating_stress_pa: 100.0,
            cycles: 2_000.0,
        }],
        options(),
    )
    .expect("valid damaging spectrum");

    assert_eq!(result.total_damage_fraction, 2.0);
    assert_eq!(result.predicted_life_blocks, Some(0.5));
    assert!(!result.is_feasible());
}

#[test]
fn fatigue_reference_rejects_invalid_options_cycles_and_goodman_limit() {
    let mut invalid_options = options();
    invalid_options.sn_slope = 0.0;
    assert!(matches!(
        evaluate_s_n_fatigue_reference(
            &[FatigueCycle {
                mean_stress_pa: 0.0,
                alternating_stress_pa: 1.0,
                cycles: 1.0,
            }],
            invalid_options,
        ),
        Err(FatigueError::InvalidOptions {
            field: "sn_slope",
            ..
        })
    ));

    assert!(matches!(
        evaluate_s_n_fatigue_reference(
            &[FatigueCycle {
                mean_stress_pa: 0.0,
                alternating_stress_pa: -1.0,
                cycles: 1.0,
            }],
            options(),
        ),
        Err(FatigueError::InvalidCycle {
            field: "alternating_stress_pa",
            ..
        })
    ));

    assert!(matches!(
        evaluate_s_n_fatigue_reference(
            &[FatigueCycle {
                mean_stress_pa: 200.0,
                alternating_stress_pa: 1.0,
                cycles: 1.0,
            }],
            options(),
        ),
        Err(FatigueError::GoodmanSingularity { index: 0 })
    ));
}
