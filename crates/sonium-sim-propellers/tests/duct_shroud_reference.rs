use sonium_sim_propellers::{
    DuctShroudError, DuctShroudOptions, DuctShroudStation, evaluate_duct_shroud_reference,
};

fn stations() -> [DuctShroudStation; 3] {
    [
        DuctShroudStation {
            axial_position_m: 0.0,
            flow_area_m2: 0.08,
            hydraulic_diameter_m: 0.30,
        },
        DuctShroudStation {
            axial_position_m: 0.5,
            flow_area_m2: 0.04,
            hydraulic_diameter_m: 0.22,
        },
        DuctShroudStation {
            axial_position_m: 1.0,
            flow_area_m2: 0.06,
            hydraulic_diameter_m: 0.26,
        },
    ]
}

fn options() -> DuctShroudOptions {
    DuctShroudOptions {
        mass_flow_kg_s: 0.8,
        minor_loss_coefficient: 0.02,
        maximum_velocity_m_s: 200.0,
        maximum_mach: 0.8,
        ..DuctShroudOptions::default()
    }
}

#[test]
fn contraction_and_recovery_preserve_continuity_with_loss_accounting() {
    let result = evaluate_duct_shroud_reference(&stations(), options()).expect("valid duct flow");
    assert!(result.is_finite());
    assert!(result.stations[1].velocity_m_s > result.stations[0].velocity_m_s);
    assert!(result.stations[2].velocity_m_s < result.stations[1].velocity_m_s);
    assert!(result.stations[1].static_pressure_pa < result.stations[0].static_pressure_pa);
    assert!(result.stations[2].static_pressure_pa > result.stations[1].static_pressure_pa);
    assert!(result.total_wall_loss_pa > 0.0);
    assert!(result.total_minor_loss_pa > 0.0);
    assert!(
        result
            .stations
            .iter()
            .all(|station| station.continuity_residual_kg_s.abs() < 1.0e-12)
    );
}

#[test]
fn lossless_diffuser_recovers_static_pressure() {
    let result = evaluate_duct_shroud_reference(
        &[
            DuctShroudStation {
                axial_position_m: 0.0,
                flow_area_m2: 0.04,
                hydraulic_diameter_m: 0.25,
            },
            DuctShroudStation {
                axial_position_m: 0.1,
                flow_area_m2: 0.08,
                hydraulic_diameter_m: 0.32,
            },
        ],
        DuctShroudOptions {
            mass_flow_kg_s: 0.4,
            minor_loss_coefficient: 0.0,
            ..options()
        },
    )
    .expect("valid lossless diffuser");
    assert!(result.stations[1].static_pressure_pa > result.stations[0].static_pressure_pa);
    assert!(result.stations[1].dynamic_pressure_pa < result.stations[0].dynamic_pressure_pa);
    assert!(result.stations[1].reynolds_number > result.stations[0].reynolds_number * 0.5);
}

#[test]
fn duct_shroud_reference_is_deterministic() {
    let first = evaluate_duct_shroud_reference(&stations(), options()).expect("first evaluation");
    let second = evaluate_duct_shroud_reference(&stations(), options()).expect("second evaluation");
    assert_eq!(first, second);
}

#[test]
fn duct_shroud_reference_rejects_invalid_geometry_and_budgets() {
    assert!(matches!(
        evaluate_duct_shroud_reference(&[], options()),
        Err(DuctShroudError::EmptyStations)
    ));
    assert!(matches!(
        evaluate_duct_shroud_reference(
            &stations(),
            DuctShroudOptions {
                maximum_stations: 2,
                ..options()
            }
        ),
        Err(DuctShroudError::TooManyStations { .. })
    ));
    assert!(matches!(
        evaluate_duct_shroud_reference(
            &[
                DuctShroudStation {
                    axial_position_m: 0.0,
                    flow_area_m2: 0.04,
                    hydraulic_diameter_m: 0.2,
                },
                DuctShroudStation {
                    axial_position_m: 0.0,
                    flow_area_m2: 0.04,
                    hydraulic_diameter_m: 0.2,
                },
            ],
            options()
        ),
        Err(DuctShroudError::NonMonotonicAxial { index: 1 })
    ));
    assert!(matches!(
        evaluate_duct_shroud_reference(
            &stations(),
            DuctShroudOptions {
                maximum_velocity_m_s: 1.0,
                ..options()
            }
        ),
        Err(DuctShroudError::VelocityExceeded { index: 0, .. })
    ));
    assert!(matches!(
        evaluate_duct_shroud_reference(
            &stations(),
            DuctShroudOptions {
                maximum_mach: 0.001,
                ..options()
            }
        ),
        Err(DuctShroudError::MachExceeded { index: 0, .. })
    ));
}
