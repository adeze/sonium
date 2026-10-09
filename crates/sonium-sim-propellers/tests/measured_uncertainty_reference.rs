use sonium_sim_propellers::{
    MeasuredUncertaintyError, MeasuredUncertaintyOptions, MeasuredUncertaintySample,
    evaluate_measured_uncertainty,
};

fn exact_samples(predicted_standard_deviation: f64) -> Vec<MeasuredUncertaintySample> {
    vec![
        MeasuredUncertaintySample {
            operating_point: 0.1,
            predicted_value: 1.0,
            measured_value: 2.0,
            predicted_standard_deviation,
            measured_standard_deviation: 0.2,
        },
        MeasuredUncertaintySample {
            operating_point: 0.2,
            predicted_value: 2.0,
            measured_value: 4.0,
            predicted_standard_deviation,
            measured_standard_deviation: 0.2,
        },
        MeasuredUncertaintySample {
            operating_point: 0.3,
            predicted_value: 3.0,
            measured_value: 6.0,
            predicted_standard_deviation,
            measured_standard_deviation: 0.2,
        },
    ]
}

#[test]
fn exact_affine_fit_propagates_independent_uncertainty() {
    let result =
        evaluate_measured_uncertainty(&exact_samples(0.1), MeasuredUncertaintyOptions::default())
            .expect("exact uncertainty envelope");
    let expected_standard_deviation = (0.2_f64.powi(2) + 0.2_f64.powi(2)).sqrt();

    assert_eq!(result.calibration.gain, 2.0);
    assert_eq!(result.calibration.offset, 0.0);
    assert_eq!(result.maximum_z_score, 0.0);
    assert_eq!(result.coverage_fraction, 1.0);
    assert_eq!(result.reduced_chi_squared, 0.0);
    assert!(result.passed);
    assert!(
        (result.points[0].combined_standard_deviation - expected_standard_deviation).abs()
            < 1.0e-14
    );
    assert!(result.points.iter().all(|point| point.within_limit));
    assert!(result.is_finite());
}

#[test]
fn affine_gain_scales_prediction_uncertainty() {
    let samples = exact_samples(0.2)
        .into_iter()
        .map(|mut sample| {
            sample.measured_value *= 1.5;
            sample.measured_standard_deviation = 0.1;
            sample
        })
        .collect::<Vec<_>>();
    let result = evaluate_measured_uncertainty(&samples, MeasuredUncertaintyOptions::default())
        .expect("scaled uncertainty envelope");
    let expected_standard_deviation = (0.6_f64.powi(2) + 0.1_f64.powi(2)).sqrt();

    assert!((result.calibration.gain - 3.0).abs() < 1.0e-14);
    assert!(
        (result.points[1].combined_standard_deviation - expected_standard_deviation).abs()
            < 1.0e-14
    );
    assert_eq!(result.reduced_chi_squared, 0.0);
}

#[test]
fn residuals_produce_standardized_errors_and_fail_strict_gate() {
    let mut samples = exact_samples(0.01);
    samples[1].measured_value = 4.4;
    let result = evaluate_measured_uncertainty(
        &samples,
        MeasuredUncertaintyOptions {
            maximum_z_score: 0.5,
            minimum_coverage_fraction: 1.0,
            maximum_reduced_chi_squared: 0.5,
            ..MeasuredUncertaintyOptions::default()
        },
    )
    .expect("residual uncertainty envelope");

    assert!(result.maximum_z_score > 0.5);
    assert!(result.reduced_chi_squared > 0.5);
    assert!(!result.passed);
    assert!(result.points.iter().any(|point| point.z_score > 0.5));
}

#[test]
fn coverage_gate_is_distinct_from_maximum_z_score_gate() {
    let mut samples = exact_samples(0.01);
    for sample in &mut samples {
        sample.measured_standard_deviation = 0.001;
    }
    samples[0].measured_value = 2.1;
    let result = evaluate_measured_uncertainty(
        &samples,
        MeasuredUncertaintyOptions {
            maximum_z_score: 0.5,
            minimum_coverage_fraction: 1.0,
            maximum_reduced_chi_squared: 1.0e6,
            ..MeasuredUncertaintyOptions::default()
        },
    )
    .expect("coverage uncertainty envelope");

    assert!(result.maximum_z_score > 0.5);
    assert!(result.coverage_fraction < 1.0);
    assert!(!result.passed);
}

#[test]
fn uncertainty_evaluation_is_deterministic() {
    let samples = exact_samples(0.15);
    let options = MeasuredUncertaintyOptions::default();
    let first = evaluate_measured_uncertainty(&samples, options).expect("first evaluation");
    let second = evaluate_measured_uncertainty(&samples, options).expect("second evaluation");

    assert_eq!(first, second);
}

#[test]
fn invalid_samples_options_and_budgets_fail_closed() {
    let mut invalid = exact_samples(0.1);
    invalid[0].predicted_standard_deviation = 0.0;
    assert!(matches!(
        evaluate_measured_uncertainty(&invalid, MeasuredUncertaintyOptions::default()),
        Err(MeasuredUncertaintyError::InvalidSample {
            field: "predicted_standard_deviation",
            ..
        })
    ));

    let mut options = MeasuredUncertaintyOptions::default();
    options.maximum_z_score = 0.0;
    assert!(matches!(
        evaluate_measured_uncertainty(&exact_samples(0.1), options),
        Err(MeasuredUncertaintyError::InvalidOptions {
            field: "maximum_z_score",
            ..
        })
    ));

    let mut budget = MeasuredUncertaintyOptions::default();
    budget.calibration.max_samples = 2;
    assert!(matches!(
        evaluate_measured_uncertainty(&exact_samples(0.1), budget),
        Err(MeasuredUncertaintyError::TooManySamples { .. })
    ));

    assert!(matches!(
        evaluate_measured_uncertainty(&[], MeasuredUncertaintyOptions::default()),
        Err(MeasuredUncertaintyError::EmptySamples)
    ));
}

#[test]
fn nested_calibration_failure_is_preserved() {
    let mut samples = exact_samples(0.1);
    samples[2].predicted_value = samples[1].predicted_value;
    let mut options = MeasuredUncertaintyOptions::default();
    options.calibration.minimum_predictor_range = 10.0;

    assert!(matches!(
        evaluate_measured_uncertainty(&samples, options),
        Err(MeasuredUncertaintyError::Calibration(_))
    ));
}
