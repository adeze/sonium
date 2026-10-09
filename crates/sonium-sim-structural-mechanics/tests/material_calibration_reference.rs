use sonium_sim_structural_mechanics::{
    ElastoplasticBarStudy, MATERIAL_CALIBRATION_SCHEMA_VERSION, MaterialCalibrationModelAssumption,
    MaterialCalibrationStudy, calibrate_material, run_elastoplastic_bar,
};

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "{left} differs from {right}"
    );
}

#[test]
fn material_calibration_recovers_bilinear_fixture_and_accepts_fit() {
    let result = calibrate_material(&MaterialCalibrationStudy::introductory())
        .expect("bilinear material fixture should calibrate");

    assert_eq!(result.schema_version, MATERIAL_CALIBRATION_SCHEMA_VERSION);
    assert_eq!(
        result.model_assumption,
        MaterialCalibrationModelAssumption::MonotoneBilinearIsotropicHardening
    );
    assert_eq!(result.material.source_id, result.study.source_id);
    assert_close(result.material.youngs_modulus_pa, 200.0e9, 1.0e6);
    assert_close(result.material.isotropic_hardening_pa, 4.0e9, 20.0e6);
    assert!(result.material.yield_stress_pa > 200.0e6);
    assert!(result.fit_rmse_pa <= result.study.max_fit_rmse_pa);
    assert!(result.fit_max_relative_error <= result.study.max_fit_relative_error);
    assert!(result.fit_accepted);

    let bar = ElastoplasticBarStudy {
        length_m: 1.0,
        area_m2: 1.0e-4,
        elements: 4,
        youngs_modulus_pa: result.material.youngs_modulus_pa,
        yield_stress_pa: result.material.yield_stress_pa,
        isotropic_hardening_pa: result.material.isotropic_hardening_pa,
        max_force_n: 30_000.0,
        load_steps: 12,
        max_newton_iterations: 20,
        residual_tolerance_n: 1.0e-8,
    };
    let bar_result = run_elastoplastic_bar(&bar).expect("calibrated material should drive bar");
    assert!(
        bar_result
            .steps
            .iter()
            .any(|step| step.yielded_elements > 0)
    );
    assert!(bar_result.total_plastic_dissipation_j > 0.0);
}

#[test]
fn material_calibration_result_round_trips() {
    let result = calibrate_material(&MaterialCalibrationStudy::introductory())
        .expect("bilinear material fixture should calibrate");
    let encoded = serde_json::to_string(&result).expect("serialize calibration result");
    let decoded: sonium_sim_structural_mechanics::MaterialCalibrationResult =
        serde_json::from_str(&encoded).expect("deserialize calibration result");

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study.source_id, result.study.source_id);
    assert_eq!(
        decoded.study.observations.len(),
        result.study.observations.len()
    );
    assert_eq!(decoded.material.source_id, result.material.source_id);
    assert_eq!(decoded.model_assumption, result.model_assumption);
    assert_eq!(decoded.execution, result.execution);
    assert_close(
        decoded.material.youngs_modulus_pa,
        result.material.youngs_modulus_pa,
        1.0e-6,
    );
    assert_close(decoded.fit_rmse_pa, result.fit_rmse_pa, 1.0e-6);
}

#[test]
fn material_calibration_rejects_nonmonotone_or_unfittable_data() {
    let mut nonmonotone = MaterialCalibrationStudy::introductory();
    nonmonotone.observations[4].stress_pa = nonmonotone.observations[3].stress_pa - 1.0;
    assert!(calibrate_material(&nonmonotone).is_err());

    let mut no_offset_crossing = MaterialCalibrationStudy::introductory();
    no_offset_crossing.offset_strain = 1.0;
    assert!(calibrate_material(&no_offset_crossing).is_err());
}

#[test]
fn material_calibration_rejects_invalid_controls() {
    let mut invalid_source = MaterialCalibrationStudy::introductory();
    invalid_source.source_id.clear();
    assert!(calibrate_material(&invalid_source).is_err());

    let mut invalid_elastic_count = MaterialCalibrationStudy::introductory();
    invalid_elastic_count.elastic_sample_count = invalid_elastic_count.observations.len();
    assert!(calibrate_material(&invalid_elastic_count).is_err());
}
