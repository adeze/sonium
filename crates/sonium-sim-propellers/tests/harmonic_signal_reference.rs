use sonium_sim_propellers::{
    HarmonicSignalComponent, HarmonicSignalError, HarmonicSignalOptions,
    evaluate_harmonic_signal_reference,
};

fn options() -> HarmonicSignalOptions {
    HarmonicSignalOptions {
        fundamental_frequency_hz: 600.0,
        cycles: 8,
        samples_per_cycle: 256,
        reference_pressure_pa: 20.0e-6,
        band_low_hz: 1_000.0,
        band_high_hz: 5_000.0,
        closure_tolerance_pa2: 1.0e-12,
    }
}

#[test]
fn one_harmonic_matches_peak_to_rms_and_spl_reference() {
    let component = HarmonicSignalComponent {
        harmonic: 2,
        peak_pressure_pa: 0.2,
        phase_rad: 0.37,
        directivity_gain: 1.0,
    };
    let result = evaluate_harmonic_signal_reference(&[component], options())
        .expect("single harmonic should be valid");
    let expected_rms = 0.2 / 2.0_f64.sqrt();

    assert!((result.total_rms_pressure_pa - expected_rms).abs() < 1.0e-12);
    assert_eq!(result.band_component_count, 1);
    assert!((result.band_rms_pressure_pa.expect("band RMS") - expected_rms).abs() < 1.0e-12);
    assert!(result.total_spl_db.is_some());
    assert!(result.band_spl_db.is_some());
    assert!(result.closure_passed);
    assert!(result.total_closure_error_pa2.abs() < 1.0e-12);
    assert!(result.is_finite());
}

#[test]
fn orthogonal_harmonics_have_additive_mean_square_and_band_selection() {
    let result = evaluate_harmonic_signal_reference(
        &[
            HarmonicSignalComponent {
                harmonic: 1,
                peak_pressure_pa: 0.1,
                phase_rad: 0.1,
                directivity_gain: 1.0,
            },
            HarmonicSignalComponent {
                harmonic: 3,
                peak_pressure_pa: 0.2,
                phase_rad: 1.2,
                directivity_gain: -0.5,
            },
        ],
        options(),
    )
    .expect("ordered harmonics should be valid");
    let expected_total_mean_square = (0.1_f64.powi(2) + (0.2_f64 * 0.5).powi(2)) / 2.0;
    let expected_band_mean_square = (0.2_f64 * 0.5).powi(2) / 2.0;

    assert!((result.analytical_total_mean_square_pa2 - expected_total_mean_square).abs() < 1.0e-12);
    assert!((result.analytical_band_mean_square_pa2 - expected_band_mean_square).abs() < 1.0e-12);
    assert!((result.sampled_total_mean_square_pa2 - expected_total_mean_square).abs() < 1.0e-12);
    assert!((result.sampled_band_mean_square_pa2 - expected_band_mean_square).abs() < 1.0e-12);
    assert_eq!(result.band_component_count, 1);
    assert!(result.closure_passed);
}

#[test]
fn zero_directivity_retains_zero_signal_without_fake_spl() {
    let result = evaluate_harmonic_signal_reference(
        &[HarmonicSignalComponent {
            harmonic: 2,
            peak_pressure_pa: 0.2,
            phase_rad: 0.0,
            directivity_gain: 0.0,
        }],
        options(),
    )
    .expect("zero directivity should be valid");

    assert_eq!(result.total_rms_pressure_pa, 0.0);
    assert_eq!(result.band_rms_pressure_pa, Some(0.0));
    assert_eq!(result.total_spl_db, None);
    assert_eq!(result.band_spl_db, None);
    assert!(result.closure_passed);
}

#[test]
fn aliasing_and_ordering_fail_closed() {
    let aliasing = evaluate_harmonic_signal_reference(
        &[HarmonicSignalComponent {
            harmonic: 8,
            peak_pressure_pa: 0.1,
            phase_rad: 0.0,
            directivity_gain: 1.0,
        }],
        HarmonicSignalOptions {
            samples_per_cycle: 16,
            ..options()
        },
    );
    assert!(matches!(
        aliasing,
        Err(HarmonicSignalError::AliasingRisk { .. })
    ));

    let unordered = evaluate_harmonic_signal_reference(
        &[
            HarmonicSignalComponent {
                harmonic: 2,
                peak_pressure_pa: 0.1,
                phase_rad: 0.0,
                directivity_gain: 1.0,
            },
            HarmonicSignalComponent {
                harmonic: 2,
                peak_pressure_pa: 0.1,
                phase_rad: 0.0,
                directivity_gain: 1.0,
            },
        ],
        options(),
    );
    assert!(matches!(
        unordered,
        Err(HarmonicSignalError::InvalidComponent {
            field: "harmonic",
            ..
        })
    ));
}

#[test]
fn malformed_options_and_components_fail_closed() {
    let invalid_band = evaluate_harmonic_signal_reference(
        &[HarmonicSignalComponent {
            harmonic: 1,
            peak_pressure_pa: 0.1,
            phase_rad: 0.0,
            directivity_gain: 1.0,
        }],
        HarmonicSignalOptions {
            band_high_hz: 100.0,
            band_low_hz: 100.0,
            ..options()
        },
    );
    assert!(matches!(
        invalid_band,
        Err(HarmonicSignalError::InvalidOptions {
            field: "band_low_hz/band_high_hz",
            ..
        })
    ));

    let invalid_component = evaluate_harmonic_signal_reference(
        &[HarmonicSignalComponent {
            harmonic: 1,
            peak_pressure_pa: f64::NAN,
            phase_rad: 0.0,
            directivity_gain: 1.0,
        }],
        options(),
    );
    assert!(matches!(
        invalid_component,
        Err(HarmonicSignalError::InvalidComponent {
            field: "peak_pressure_pa",
            ..
        })
    ));
}
