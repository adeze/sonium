//! Compressibility and compact-source admission checks.
//!
//! Covers the audit's explicit compressibility/noncompact-source requirement:
//! subsonic-tip admission, Helmholtz compactness reporting with optional
//! fail-closed rejection, and malformed-input handling.

use sonium_sim_propellers::{
    CompressibilityError, CompressibilityOptions, evaluate_compressibility_reference,
};

fn base_options() -> CompressibilityOptions {
    CompressibilityOptions {
        rpm: 1_800.0,
        tip_radius_m: 0.15,
        sound_speed_m_s: 343.0,
        frequency_hz: 100.0,
        source_extent_m: 0.3,
        maximum_tip_mach: 1.0,
        maximum_compact_helmholtz: 1.0,
        require_compact: false,
    }
}

#[test]
fn compact_subsonic_condition_passes_with_expected_diagnostics() {
    let result = evaluate_compressibility_reference(base_options())
        .expect("compact subsonic condition is admitted");
    assert!(result.is_finite());
    assert!(result.is_compact);
    let expected_tip_speed = 1_800.0 * 2.0 * std::f64::consts::PI / 60.0 * 0.15;
    assert!((result.tip_speed_m_s - expected_tip_speed).abs() < 1.0e-12);
    assert!((result.tip_mach - expected_tip_speed / 343.0).abs() < 1.0e-12);
    let expected_helmholtz = 2.0 * std::f64::consts::PI * 100.0 / 343.0 * 0.3;
    assert!((result.helmholtz_number - expected_helmholtz).abs() < 1.0e-12);
    assert!(expected_helmholtz < 1.0);
}

#[test]
fn noncompact_source_is_reported_without_require_compact() {
    let options = CompressibilityOptions {
        frequency_hz: 2_000.0,
        ..base_options()
    };
    let result =
        evaluate_compressibility_reference(options).expect("screening mode reports non-compact");
    assert!(result.is_finite());
    assert!(!result.is_compact);
    assert!(result.helmholtz_number > 1.0);
}

#[test]
fn noncompact_source_fails_closed_when_required() {
    let options = CompressibilityOptions {
        frequency_hz: 2_000.0,
        require_compact: true,
        ..base_options()
    };
    match evaluate_compressibility_reference(options) {
        Err(CompressibilityError::NonCompactSource { helmholtz, maximum }) => {
            assert!(helmholtz > maximum);
            assert_eq!(maximum, 1.0);
        }
        other => panic!("expected NonCompactSource, got {other:?}"),
    }
}

#[test]
fn supersonic_tip_fails_closed() {
    let options = CompressibilityOptions {
        rpm: 30_000.0,
        tip_radius_m: 0.3,
        ..base_options()
    };
    match evaluate_compressibility_reference(options) {
        Err(CompressibilityError::SupersonicTip { mach, maximum }) => {
            assert!(mach >= maximum);
            assert!(mach > 1.0);
        }
        other => panic!("expected SupersonicTip, got {other:?}"),
    }
}

#[test]
fn malformed_options_fail_closed() {
    for (field, options) in [
        (
            "rpm",
            CompressibilityOptions {
                rpm: f64::NAN,
                ..base_options()
            },
        ),
        (
            "tip_radius_m",
            CompressibilityOptions {
                tip_radius_m: 0.0,
                ..base_options()
            },
        ),
        (
            "sound_speed_m_s",
            CompressibilityOptions {
                sound_speed_m_s: -10.0,
                ..base_options()
            },
        ),
        (
            "frequency_hz",
            CompressibilityOptions {
                frequency_hz: f64::INFINITY,
                ..base_options()
            },
        ),
        (
            "source_extent_m",
            CompressibilityOptions {
                source_extent_m: -0.3,
                ..base_options()
            },
        ),
    ] {
        match evaluate_compressibility_reference(options) {
            Err(CompressibilityError::InvalidOptions { field: actual, .. }) => {
                assert_eq!(actual, field)
            }
            other => panic!("expected InvalidOptions({field}), got {other:?}"),
        }
    }
}
