use sonium_sim_structural_mechanics::{
    HYPERELASTIC_3D_FORCE_CONTACT_SCHEMA_VERSION, Hyperelastic3DForceContactResult,
    Hyperelastic3DForceContactStudy, run_hyperelastic_3d_force_contact,
};

#[test]
fn force_contact_reference_reaches_contact_with_monotone_load() {
    let result =
        run_hyperelastic_3d_force_contact(&Hyperelastic3DForceContactStudy::introductory())
            .expect("force-controlled 3-D contact reference should solve");

    assert_eq!(
        result.schema_version,
        HYPERELASTIC_3D_FORCE_CONTACT_SCHEMA_VERSION
    );
    assert_eq!(result.nodes, 36);
    assert_eq!(result.elements, 48);
    assert_eq!(result.dofs, result.nodes * 3);
    assert_eq!(result.bottom_external_nodes, 6);
    assert_eq!(result.top_external_nodes, 6);
    assert!(result.bottom_contact_faces > 0);
    assert!(result.top_contact_faces > 0);
    assert_eq!(result.steps.len(), result.study.load_steps);
    assert!(result.steps.iter().all(|step| step.converged));
    assert!(result.steps.iter().all(|step| step.residual.is_finite()));
    assert!(
        result
            .steps
            .windows(2)
            .all(|pair| pair[1].applied_force_n > pair[0].applied_force_n)
    );
    assert!(
        result
            .steps
            .windows(2)
            .all(|pair| pair[1].average_top_displacement_m < pair[0].average_top_displacement_m)
    );
    assert!(result.contact_engaged_step.is_some());
    assert!(result.max_penetration_m.is_finite());
    assert!(result.max_penetration_m > 0.0);
    assert!(result.final_average_top_displacement_m < 0.0);
    assert!(result.final_residual.is_finite());
    assert!(result.converged);
}

#[test]
fn force_contact_result_round_trips() {
    let result =
        run_hyperelastic_3d_force_contact(&Hyperelastic3DForceContactStudy::introductory())
            .expect("force-controlled contact reference should serialize");
    let encoded = serde_json::to_string(&result).expect("result should serialize");
    let decoded: Hyperelastic3DForceContactResult =
        serde_json::from_str(&encoded).expect("result should deserialize");

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study, result.study);
    assert_eq!(decoded.nodes, result.nodes);
    assert_eq!(decoded.elements, result.elements);
    assert_eq!(decoded.dofs, result.dofs);
    assert_eq!(decoded.steps.len(), result.steps.len());
    for (decoded_step, result_step) in decoded.steps.iter().zip(&result.steps) {
        assert_eq!(decoded_step.load_step, result_step.load_step);
        assert_eq!(decoded_step.active_pairs, result_step.active_pairs);
        assert_eq!(decoded_step.converged, result_step.converged);
        assert!((decoded_step.applied_force_n - result_step.applied_force_n).abs() < 1.0e-12);
        assert!(
            (decoded_step.average_top_displacement_m - result_step.average_top_displacement_m)
                .abs()
                < 1.0e-15
        );
        assert!((decoded_step.max_penetration_m - result_step.max_penetration_m).abs() < 1.0e-15);
    }
    assert_eq!(decoded.contact_engaged_step, result.contact_engaged_step);
    assert_eq!(decoded.execution, result.execution);
}

#[test]
fn invalid_force_contact_controls_are_rejected() {
    let mut invalid = Hyperelastic3DForceContactStudy::introductory();
    invalid.total_force_n = f64::NAN;
    let error =
        run_hyperelastic_3d_force_contact(&invalid).expect_err("NaN force must fail closed");
    assert!(error.to_string().contains("total_force_n"));

    let mut invalid = Hyperelastic3DForceContactStudy::introductory();
    invalid.max_uzawa_iterations = 0;
    let error = run_hyperelastic_3d_force_contact(&invalid)
        .expect_err("zero Uzawa iterations must fail closed");
    assert!(error.to_string().contains("iterations"));
}
