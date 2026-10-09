use std::f64::consts::PI;

use sonium_sim_propellers::{
    FwhError, FwhOptions, FwhSurfaceElement, Point3, evaluate_fwh_reference,
};

fn element() -> FwhSurfaceElement {
    FwhSurfaceElement {
        position: Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        normal: Point3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        area_m2: 0.2,
        pressure_real_pa: 1.0,
        pressure_imaginary_pa: 0.0,
        normal_velocity_real_m_s: 0.1,
        normal_velocity_imaginary_m_s: 0.0,
    }
}

fn options() -> FwhOptions {
    FwhOptions {
        observer_position: Point3 {
            x: 0.0,
            y: 0.0,
            z: 10.0,
        },
        minimum_observer_distance_m: 0.5,
        ..FwhOptions::default()
    }
}

#[test]
fn monopole_and_dipole_terms_match_compact_free_field_identities() {
    let result = evaluate_fwh_reference(&[element()], options()).expect("valid FW-H reference");
    let omega = 2.0 * PI * options().frequency_hz;
    let expected_monopole = omega * options().density_kg_m3 * 0.2 * 0.1 / (4.0 * PI * 10.0);
    let expected_dipole = omega * 0.2 * 1.0 / (4.0 * PI * options().sound_speed_m_s * 10.0);
    assert!((result.monopole_pressure_pa.magnitude_pa() - expected_monopole).abs() < 1.0e-12);
    assert!((result.dipole_pressure_pa.magnitude_pa() - expected_dipole).abs() < 1.0e-12);
    assert!(result.total_pressure_pa.magnitude_pa().is_finite());
    assert!(result.is_finite());
}

#[test]
fn multiple_elements_are_coherent_and_deterministic() {
    let mut second = element();
    second.position = Point3 {
        x: 0.5,
        y: 0.0,
        z: 0.0,
    };
    second.pressure_imaginary_pa = 0.25;
    let first = evaluate_fwh_reference(&[element(), second], options()).expect("first reference");
    let repeat = evaluate_fwh_reference(&[element(), second], options()).expect("repeat reference");
    assert_eq!(first, repeat);
    assert_eq!(first.total_surface_area_m2, 0.4);
    assert!(first.maximum_observer_distance_m > first.minimum_observer_distance_m);
}

#[test]
fn fwh_reference_rejects_invalid_inputs_and_budgets() {
    assert!(matches!(
        evaluate_fwh_reference(&[], options()),
        Err(FwhError::EmptySurface)
    ));
    assert!(matches!(
        evaluate_fwh_reference(
            &[element()],
            FwhOptions {
                maximum_elements: 0,
                ..options()
            }
        ),
        Err(FwhError::InvalidOptions {
            field: "maximum_elements",
            ..
        })
    ));
    assert!(matches!(
        evaluate_fwh_reference(
            &[element()],
            FwhOptions {
                minimum_observer_distance_m: 20.0,
                ..options()
            }
        ),
        Err(FwhError::ObserverTooClose { index: 0, .. })
    ));
    assert!(matches!(
        evaluate_fwh_reference(
            &[FwhSurfaceElement {
                normal: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..element()
            }],
            options()
        ),
        Err(FwhError::InvalidElement {
            index: 0,
            field: "normal",
            ..
        })
    ));
}

#[test]
fn fwh_reference_preserves_signed_dipole_orientation() {
    let mut reversed = element();
    reversed.normal.z = -1.0;
    reversed.normal_velocity_real_m_s = 0.0;
    let result = evaluate_fwh_reference(&[reversed], options()).expect("reversed dipole");
    assert!(
        result.dipole_pressure_pa.real_pa < 0.0 || result.dipole_pressure_pa.imaginary_pa < 0.0
    );
}
