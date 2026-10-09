use sonium_sim_structural_mechanics::{
    HYPERELASTIC_3D_CONTACT_SCHEMA_VERSION, Hyperelastic3DContactStudy, run_hyperelastic_3d_contact,
};

#[test]
fn hyperelastic_3d_contact_reference_crosses_contact_and_preserves_gates() {
    let result = run_hyperelastic_3d_contact(&Hyperelastic3DContactStudy::introductory())
        .expect("small 3-D hyperelastic contact reference should solve");

    assert_eq!(
        result.schema_version,
        HYPERELASTIC_3D_CONTACT_SCHEMA_VERSION
    );
    assert!(result.nodes > 0);
    assert!(result.elements > 0);
    assert_eq!(result.nodes, 36);
    assert_eq!(result.elements, 48);
    assert_eq!(result.dofs, result.nodes * 3);
    assert!(result.bottom_contact_faces > 0);
    assert!(result.top_contact_faces > 0);
    assert_eq!(result.steps.len(), result.study.load_steps);
    assert!(result.steps.iter().all(|step| step.converged));
    assert!(result.contact_engaged_step.is_some());
    assert!(result.max_penetration_m.is_finite());
    assert!(result.max_penetration_m > 0.0);
    assert!(result.final_residual.is_finite());
    assert!(result.converged);
}

#[test]
fn hyperelastic_3d_contact_result_round_trips_and_keeps_step_diagnostics() {
    let result = run_hyperelastic_3d_contact(&Hyperelastic3DContactStudy::introductory())
        .expect("small 3-D hyperelastic contact reference should solve");
    let encoded = serde_json::to_string(&result).expect("contact result should serialize");
    let decoded: sonium_sim_structural_mechanics::Hyperelastic3DContactResult =
        serde_json::from_str(&encoded).expect("contact result should deserialize");

    assert_eq!(decoded, result);
    assert!(decoded.steps.iter().any(|step| step.active_pairs > 0));
}

#[test]
fn hyperelastic_3d_contact_rejects_invalid_controls_before_mesh_work() {
    let mut study = Hyperelastic3DContactStudy::introductory();
    study.nz = 0;
    assert!(matches!(
        run_hyperelastic_3d_contact(&study),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_contact.mesh_resolution",
                ..
            }
        )
    ));

    let mut study = Hyperelastic3DContactStudy::introductory();
    study.prescribed_compression_m = 2.0 * study.height_m;
    assert!(matches!(
        run_hyperelastic_3d_contact(&study),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_contact.prescribed_compression_m",
                ..
            }
        )
    ));
}
