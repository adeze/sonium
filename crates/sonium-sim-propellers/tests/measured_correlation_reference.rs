use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    EvaluationMode, MeasuredHarmonicCorrelationError, MeasuredHarmonicCorrelationOptions,
    MeasuredHarmonicPoint, PropellerDesign, correlate_measured_harmonics,
};

fn spectrum() -> sonium_sim_propellers::PropellerSpectrum {
    PropellerDesign::conventional(PropellerSource::drone_5inch(18_000.0))
        .spectrum(3, EvaluationMode::FastSurrogate)
}

fn measurements_with_pressures(
    spectrum: &sonium_sim_propellers::PropellerSpectrum,
    pressures: &[f64],
) -> Vec<MeasuredHarmonicPoint> {
    spectrum
        .harmonics
        .iter()
        .zip(pressures)
        .map(|(prediction, pressure_pa)| MeasuredHarmonicPoint {
            harmonic: prediction.harmonic,
            frequency_hz: prediction.frequency_hz,
            pressure_pa: *pressure_pa,
        })
        .collect()
}

#[test]
fn exact_reference_measurements_have_zero_error_and_pass() {
    let predicted = spectrum();
    let pressures = predicted
        .harmonics
        .iter()
        .map(|harmonic| harmonic.pressure_pa)
        .collect::<Vec<_>>();

    let result = correlate_measured_harmonics(
        &predicted,
        &measurements_with_pressures(&predicted, &pressures),
        MeasuredHarmonicCorrelationOptions::default(),
    )
    .expect("exact spectrum measurements");

    assert!(result.is_finite());
    assert!(result.passed);
    assert_eq!(result.points.len(), 3);
    assert_eq!(result.rmse_pa, 0.0);
    assert_eq!(result.max_abs_error_pa, 0.0);
    assert_eq!(result.mean_error_pa, 0.0);
    assert_eq!(result.relative_rmse, Some(0.0));
}

#[test]
fn correlation_reports_signed_bias_rmse_and_admission_limits() {
    let predicted = spectrum();
    let scale_factors = [1.1, 0.8, 1.4];
    let pressures = predicted
        .harmonics
        .iter()
        .enumerate()
        .map(|(index, harmonic)| harmonic.pressure_pa * scale_factors[index])
        .collect::<Vec<_>>();
    let expected_errors = predicted
        .harmonics
        .iter()
        .enumerate()
        .map(|(index, harmonic)| harmonic.pressure_pa * (1.0 - scale_factors[index]))
        .collect::<Vec<_>>();
    let expected_rmse = (expected_errors
        .iter()
        .map(|error| error * error)
        .sum::<f64>()
        / expected_errors.len() as f64)
        .sqrt();
    let expected_max_abs = expected_errors
        .iter()
        .map(|error| error.abs())
        .fold(0.0, f64::max);
    let expected_mean = expected_errors.iter().sum::<f64>() / expected_errors.len() as f64;
    let result = correlate_measured_harmonics(
        &predicted,
        &measurements_with_pressures(&predicted, &pressures),
        MeasuredHarmonicCorrelationOptions {
            max_rmse_pa: expected_rmse + 1.0e-12,
            max_abs_error_pa: expected_max_abs + 1.0e-12,
            ..MeasuredHarmonicCorrelationOptions::default()
        },
    )
    .expect("finite measured error metrics");

    assert!(result.passed);
    assert!((result.rmse_pa - expected_rmse).abs() < 1.0e-12);
    assert!((result.max_abs_error_pa - expected_max_abs).abs() < 1.0e-12);
    assert!((result.mean_error_pa - expected_mean).abs() < 1.0e-12);
    assert!(result.points[0].relative_error.is_some());

    let rejected = correlate_measured_harmonics(
        &predicted,
        &measurements_with_pressures(&predicted, &pressures),
        MeasuredHarmonicCorrelationOptions {
            max_abs_error_pa: expected_max_abs * 0.5,
            ..MeasuredHarmonicCorrelationOptions::default()
        },
    )
    .expect("metrics are finite even when admission fails");
    assert!(!rejected.passed);
}

#[test]
fn partial_matching_is_explicit_and_frequency_tolerance_is_checked() {
    let predicted = spectrum();
    let first = predicted.harmonics[0].clone();
    let partial = [MeasuredHarmonicPoint {
        harmonic: first.harmonic,
        frequency_hz: first.frequency_hz,
        pressure_pa: first.pressure_pa,
    }];
    let result = correlate_measured_harmonics(
        &predicted,
        &partial,
        MeasuredHarmonicCorrelationOptions {
            require_complete_match: false,
            ..MeasuredHarmonicCorrelationOptions::default()
        },
    )
    .expect("explicit partial matching");
    assert_eq!(result.points.len(), 1);

    let missing = correlate_measured_harmonics(
        &predicted,
        &partial,
        MeasuredHarmonicCorrelationOptions::default(),
    )
    .expect_err("complete matching must reject missing harmonics");
    assert!(matches!(
        missing,
        MeasuredHarmonicCorrelationError::MissingMeasurement { .. }
    ));

    let frequency_mismatch = correlate_measured_harmonics(
        &predicted,
        &[MeasuredHarmonicPoint {
            frequency_hz: first.frequency_hz + 0.1,
            ..partial[0]
        }],
        MeasuredHarmonicCorrelationOptions {
            require_complete_match: false,
            frequency_tolerance_hz: 1.0e-3,
            ..MeasuredHarmonicCorrelationOptions::default()
        },
    )
    .expect_err("frequency mismatch must fail closed");
    assert!(matches!(
        frequency_mismatch,
        MeasuredHarmonicCorrelationError::FrequencyMismatch { harmonic: 1, .. }
    ));
}

#[test]
fn malformed_measurements_and_duplicates_fail_closed() {
    let predicted = spectrum();
    let first = predicted.harmonics[0].clone();
    let duplicate = vec![
        MeasuredHarmonicPoint {
            harmonic: first.harmonic,
            frequency_hz: first.frequency_hz,
            pressure_pa: first.pressure_pa,
        },
        MeasuredHarmonicPoint {
            harmonic: first.harmonic,
            frequency_hz: first.frequency_hz,
            pressure_pa: first.pressure_pa,
        },
    ];
    assert!(matches!(
        correlate_measured_harmonics(
            &predicted,
            &duplicate,
            MeasuredHarmonicCorrelationOptions {
                require_complete_match: false,
                ..MeasuredHarmonicCorrelationOptions::default()
            }
        ),
        Err(MeasuredHarmonicCorrelationError::DuplicateHarmonic {
            source: "measurements",
            ..
        })
    ));
    assert!(matches!(
        correlate_measured_harmonics(
            &predicted,
            &[MeasuredHarmonicPoint {
                harmonic: 1,
                frequency_hz: first.frequency_hz,
                pressure_pa: f64::NAN,
            }],
            MeasuredHarmonicCorrelationOptions {
                require_complete_match: false,
                ..MeasuredHarmonicCorrelationOptions::default()
            }
        ),
        Err(MeasuredHarmonicCorrelationError::InvalidMeasuredPoint {
            field: "pressure_pa",
            ..
        })
    ));
    assert!(matches!(
        correlate_measured_harmonics(
            &predicted,
            &[],
            MeasuredHarmonicCorrelationOptions::default()
        ),
        Err(MeasuredHarmonicCorrelationError::EmptyMeasurements)
    ));
}
