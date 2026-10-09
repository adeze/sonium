use sonium_sim_structural_mechanics::{
    DampingConvention, MODAL_HARMONIC_RESPONSE_SCHEMA_VERSION, ModalHarmonicResponseResult,
    ModalHarmonicResponseStudy, ModalNormalization, Solid3DModalResult, StructuralModelAssumption,
    run_modal_harmonic_response,
};

fn modal_result() -> Solid3DModalResult {
    Solid3DModalResult {
        eigenvalues: vec![3947.8417604357433, 35530.57584392169],
        frequencies_hz: vec![10.0, 30.0],
        residual_norms: vec![1.0e-10, 2.0e-10],
        solver_iterations: 12,
        converged: true,
        normalization: ModalNormalization::MassNormalized,
        mass_normalization_error: 0.0,
        orthogonality_error: 1.0e-14,
        damping_ratio: 0.02,
        damping_convention: DampingConvention::MassProportionalAtFirstMode,
        model_assumption: StructuralModelAssumption::Solid3D,
        modes: vec![
            vec![[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            vec![[-0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
        ],
    }
}

fn study() -> ModalHarmonicResponseStudy {
    let mut study = ModalHarmonicResponseStudy::introductory();
    study.frequencies_hz = vec![5.0, 10.0, 15.0, 30.0, 60.0];
    study.modal_force_phasors = vec![[1.0, 0.0], [0.5, 0.0]];
    study.damping_ratios = vec![0.02, 0.03];
    study.allowable_displacement_m = 0.1;
    study
}

#[test]
fn modal_harmonic_response_peaks_near_resonance_and_reports_lag() {
    let result = run_modal_harmonic_response(&modal_result(), &study()).unwrap();

    assert_eq!(
        result.schema_version,
        MODAL_HARMONIC_RESPONSE_SCHEMA_VERSION
    );
    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::ModalHarmonicResponse
    );
    assert_eq!(result.normalization, ModalNormalization::MassNormalized);
    assert!(result.converged);
    assert!(result.feasible);
    assert_eq!(result.points.len(), 5);
    assert_eq!(result.peak_frequency_hz, 10.0);
    assert!(result.points[1].displacement_abs_m > result.points[0].displacement_abs_m);
    assert!(result.points[1].phase_lag_rad > 1.4);
    assert!(result.points.iter().all(|point| {
        point.displacement_abs_m.is_finite()
            && point.phase_lag_rad.is_finite()
            && point.modal_displacement_phasors_m.len() == 2
    }));
}

#[test]
fn modal_harmonic_response_scales_with_modal_force() {
    let baseline = run_modal_harmonic_response(&modal_result(), &study()).unwrap();
    let mut doubled = study();
    for force in &mut doubled.modal_force_phasors {
        force[0] *= 2.0;
        force[1] *= 2.0;
    }
    let scaled = run_modal_harmonic_response(&modal_result(), &doubled).unwrap();

    assert!(
        (scaled.peak_displacement_abs_m - 2.0 * baseline.peak_displacement_abs_m).abs() < 1.0e-12
    );
    for (base, scaled) in baseline.points.iter().zip(&scaled.points) {
        assert!((scaled.displacement_abs_m - 2.0 * base.displacement_abs_m).abs() < 1.0e-12);
    }
}

#[test]
fn modal_harmonic_response_damping_reduces_resonant_amplitude() {
    let lightly_damped = run_modal_harmonic_response(&modal_result(), &study()).unwrap();
    let mut heavily_damped = study();
    heavily_damped.damping_ratios = vec![0.1, 0.1];
    let heavily_damped = run_modal_harmonic_response(&modal_result(), &heavily_damped).unwrap();

    assert!(
        lightly_damped.points[1].displacement_abs_m > heavily_damped.points[1].displacement_abs_m
    );
    assert!(lightly_damped.points[1].phase_lag_rad.is_finite());
    assert!(heavily_damped.points[1].phase_lag_rad.is_finite());
}

#[test]
fn modal_harmonic_response_rejects_invalid_modal_and_control_inputs() {
    let mut mismatched = study();
    mismatched.modal_force_phasors.pop();
    assert!(run_modal_harmonic_response(&modal_result(), &mismatched).is_err());

    let mut nonconverged = modal_result();
    nonconverged.converged = false;
    assert!(run_modal_harmonic_response(&nonconverged, &study()).is_err());

    let mut non_mass_normalized = modal_result();
    non_mass_normalized.normalization = ModalNormalization::Euclidean;
    assert!(run_modal_harmonic_response(&non_mass_normalized, &study()).is_err());

    let mut invalid_observation = study();
    invalid_observation.observation_component = 3;
    assert!(run_modal_harmonic_response(&modal_result(), &invalid_observation).is_err());

    let mut nonmonotone = study();
    nonmonotone.frequencies_hz[1] = nonmonotone.frequencies_hz[0];
    assert!(run_modal_harmonic_response(&modal_result(), &nonmonotone).is_err());

    let mut unresolved = study();
    unresolved.damping_ratios[0] = 0.0;
    unresolved.frequencies_hz = vec![10.0];
    assert!(run_modal_harmonic_response(&modal_result(), &unresolved).is_err());
}

#[test]
fn modal_harmonic_response_rejects_wrong_model_and_observation_shape() {
    let mut wrong_model = modal_result();
    wrong_model.model_assumption = StructuralModelAssumption::Beam;
    assert!(run_modal_harmonic_response(&wrong_model, &study()).is_err());

    let mut short_mode = modal_result();
    short_mode.modes[0].clear();
    assert!(run_modal_harmonic_response(&short_mode, &study()).is_err());

    let mut invalid_shape = modal_result();
    invalid_shape.modes[1][0][2] = f64::NAN;
    assert!(run_modal_harmonic_response(&invalid_shape, &study()).is_err());
}

#[test]
fn modal_harmonic_response_round_trips_through_serde() {
    let result = run_modal_harmonic_response(&modal_result(), &study()).unwrap();
    let encoded = serde_json::to_string(&result).unwrap();
    let decoded: ModalHarmonicResponseResult = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study, result.study);
    assert_eq!(decoded.modal_frequencies_hz, result.modal_frequencies_hz);
    assert_eq!(decoded.points.len(), result.points.len());
    assert_eq!(decoded.normalization, result.normalization);
    assert_eq!(decoded.model_assumption, result.model_assumption);
    assert_eq!(decoded.execution, result.execution);
    assert!((decoded.peak_displacement_abs_m - result.peak_displacement_abs_m).abs() < 1.0e-15);
    assert_eq!(decoded.peak_frequency_hz, result.peak_frequency_hz);
    assert_eq!(decoded.converged, result.converged);
    assert_eq!(decoded.feasible, result.feasible);
    for (decoded_point, result_point) in decoded.points.iter().zip(&result.points) {
        assert_eq!(decoded_point.frequency_hz, result_point.frequency_hz);
        assert!(
            (decoded_point.displacement_abs_m - result_point.displacement_abs_m).abs() < 1.0e-15
        );
        assert!((decoded_point.phase_lag_rad - result_point.phase_lag_rad).abs() < 1.0e-15);
        for (decoded_value, result_value) in decoded_point
            .displacement_phasor_m
            .iter()
            .zip(result_point.displacement_phasor_m)
        {
            assert!((*decoded_value - result_value).abs() < 1.0e-15);
        }
    }
}
