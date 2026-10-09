use sonium_sim_structural_mechanics::{
    DampingConvention, MODAL_RANDOM_VIBRATION_SCHEMA_VERSION, ModalNormalization,
    ModalRandomVibrationResult, ModalRandomVibrationStudy, Solid3DModalResult,
    StructuralModelAssumption, run_modal_random_vibration,
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
        modes: vec![Vec::new(), Vec::new()],
    }
}

fn study() -> ModalRandomVibrationStudy {
    let mut study = ModalRandomVibrationStudy::introductory();
    study.mode_participation.truncate(2);
    study.damping_ratios.truncate(2);
    study
}

#[test]
fn modal_random_vibration_consumes_converged_mass_normalized_modes() {
    let result = run_modal_random_vibration(&modal_result(), &study())
        .expect("modal PSD fixture should evaluate");

    assert_eq!(result.schema_version, MODAL_RANDOM_VIBRATION_SCHEMA_VERSION);
    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::ModalRandomVibration
    );
    assert_eq!(result.normalization, ModalNormalization::MassNormalized);
    assert_eq!(result.modes.len(), 2);
    assert!(result.total_displacement_variance_m2.is_finite());
    assert!(result.rms_displacement_m.is_finite());
    assert!(result.peak_displacement_m.is_finite());
    assert!(result.feasible);
    assert!(result.modes.iter().all(|mode| {
        mode.displacement_psd_m2_per_hz.len() == study().frequencies_hz.len()
            && mode.maximum_dynamic_amplification_s2.is_finite()
    }));
}

#[test]
fn modal_random_vibration_scales_variance_with_psd_and_rms_with_sqrt_psd() {
    let baseline = run_modal_random_vibration(&modal_result(), &study()).unwrap();
    let mut scaled_study = study();
    for value in &mut scaled_study.acceleration_psd_m2_per_s3 {
        *value *= 4.0;
    }
    let scaled = run_modal_random_vibration(&modal_result(), &scaled_study).unwrap();

    assert!(
        (scaled.total_displacement_variance_m2 - 4.0 * baseline.total_displacement_variance_m2)
            .abs()
            <= 1.0e-12 * (1.0 + scaled.total_displacement_variance_m2.abs())
    );
    assert!(
        (scaled.rms_displacement_m - 2.0 * baseline.rms_displacement_m).abs()
            <= 1.0e-12 * (1.0 + scaled.rms_displacement_m.abs())
    );
}

#[test]
fn modal_random_vibration_rejects_invalid_modal_and_spectral_inputs() {
    let mut mismatched = study();
    mismatched.mode_participation.pop();
    assert!(run_modal_random_vibration(&modal_result(), &mismatched).is_err());

    let mut nonconverged = modal_result();
    nonconverged.converged = false;
    assert!(run_modal_random_vibration(&nonconverged, &study()).is_err());

    let mut nonmonotone = study();
    nonmonotone.frequencies_hz[1] = nonmonotone.frequencies_hz[0];
    assert!(run_modal_random_vibration(&modal_result(), &nonmonotone).is_err());

    let mut unresolved_resonance = study();
    unresolved_resonance.damping_ratios[0] = 0.0;
    assert!(run_modal_random_vibration(&modal_result(), &unresolved_resonance).is_err());
}

#[test]
fn modal_random_vibration_result_round_trips() {
    let result = run_modal_random_vibration(&modal_result(), &study()).unwrap();
    let encoded = serde_json::to_string(&result).expect("result should serialize");
    let decoded: ModalRandomVibrationResult =
        serde_json::from_str(&encoded).expect("result should deserialize");

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.modal_frequencies_hz, result.modal_frequencies_hz);
    assert_eq!(decoded.modes.len(), result.modes.len());
    assert_eq!(decoded.feasible, result.feasible);
    assert_eq!(decoded.normalization, result.normalization);
    assert_eq!(decoded.execution, result.execution);
    assert!((decoded.rms_displacement_m - result.rms_displacement_m).abs() < 1.0e-12);
}
