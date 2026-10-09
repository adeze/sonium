use sonium_sim_propellers::{
    MeasuredCalibrationError, MeasuredCalibrationOptions, MeasuredCalibrationSample,
    calibrate_measured_response,
};

fn samples() -> Vec<MeasuredCalibrationSample> {
    (0..5)
        .map(|index| {
            let predicted_value = index as f64 + 1.0;
            MeasuredCalibrationSample {
                operating_point: 0.1 * (index as f64 + 1.0),
                predicted_value,
                measured_value: 1.25 * predicted_value - 0.2,
            }
        })
        .collect()
}

#[test]
fn public_calibration_recovers_caller_supplied_gain_and_offset() {
    let result = calibrate_measured_response(&samples(), MeasuredCalibrationOptions::default())
        .expect("affine calibration should converge");
    assert!(result.is_finite());
    assert!((result.gain - 1.25).abs() < 1.0e-12);
    assert!((result.offset + 0.2).abs() < 1.0e-12);
    assert!(result.passed);
}

#[test]
fn public_calibration_preserves_operating_point_and_residual_ledger() {
    let mut input = samples();
    input[3].measured_value += 0.15;
    let result = calibrate_measured_response(&input, MeasuredCalibrationOptions::default())
        .expect("finite calibration should return diagnostics");
    assert_eq!(result.points.len(), input.len());
    assert_eq!(result.points[3].operating_point, input[3].operating_point);
    assert!(result.maximum_absolute_error > 0.0);
    assert!(
        result
            .points
            .iter()
            .all(|point| { (point.absolute_error - point.error.abs()).abs() < 1.0e-12 })
    );
}

#[test]
fn public_calibration_can_reject_residuals_without_hiding_fit() {
    let mut input = samples();
    input[0].measured_value += 0.5;
    let result = calibrate_measured_response(
        &input,
        MeasuredCalibrationOptions {
            maximum_rmse: 1.0e-4,
            maximum_absolute_error: 1.0e-4,
            ..MeasuredCalibrationOptions::default()
        },
    )
    .expect("admission failure should remain inspectable");
    assert!(result.is_finite());
    assert!(!result.passed);
    assert!(result.rmse > 1.0e-4);
}

#[test]
fn public_calibration_rejects_invalid_conditioning_and_options() {
    let degenerate = [
        MeasuredCalibrationSample {
            operating_point: 0.0,
            predicted_value: 2.0,
            measured_value: 2.0,
        },
        MeasuredCalibrationSample {
            operating_point: 1.0,
            predicted_value: 2.0,
            measured_value: 3.0,
        },
    ];
    assert!(matches!(
        calibrate_measured_response(&degenerate, MeasuredCalibrationOptions::default()),
        Err(MeasuredCalibrationError::DegeneratePredictor { .. })
    ));
    assert!(matches!(
        calibrate_measured_response(
            &samples(),
            MeasuredCalibrationOptions {
                maximum_rmse: f64::NAN,
                ..MeasuredCalibrationOptions::default()
            }
        ),
        Err(MeasuredCalibrationError::InvalidOptions {
            field: "maximum_rmse",
            ..
        })
    ));
}

#[test]
fn public_calibration_is_deterministic() {
    let first = calibrate_measured_response(&samples(), MeasuredCalibrationOptions::default())
        .expect("first calibration");
    let second = calibrate_measured_response(&samples(), MeasuredCalibrationOptions::default())
        .expect("second calibration");
    assert_eq!(first, second);
}
