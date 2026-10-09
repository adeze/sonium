use sonium_sim_structural_mechanics::{
    HYPERELASTIC_3D_FORCE_SCHEMA_VERSION, Hyperelastic3DForceResult, Hyperelastic3DForceStudy,
    run_hyperelastic_3d_force,
};

#[test]
fn force_control_reference_has_monotone_finite_response() {
    let result = run_hyperelastic_3d_force(&Hyperelastic3DForceStudy::introductory())
        .expect("force-controlled 3-D hyperelastic reference should solve");

    assert_eq!(result.schema_version, HYPERELASTIC_3D_FORCE_SCHEMA_VERSION);
    assert_eq!(result.nodes, 18);
    assert_eq!(result.elements, 24);
    assert_eq!(result.dofs, result.nodes * 3);
    assert_eq!(result.bottom_nodes, 6);
    assert_eq!(result.top_nodes, 6);
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
    assert!(result.final_average_top_displacement_m < 0.0);
    assert!(result.final_residual.is_finite());
    assert!(result.converged);
}

#[test]
fn force_control_result_round_trips() {
    let result = run_hyperelastic_3d_force(&Hyperelastic3DForceStudy::introductory())
        .expect("force-controlled reference should serialize");
    let encoded = serde_json::to_string(&result).expect("serialize force result");
    let decoded: Hyperelastic3DForceResult =
        serde_json::from_str(&encoded).expect("deserialize force result");
    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study, result.study);
    assert_eq!(decoded.nodes, result.nodes);
    assert_eq!(decoded.elements, result.elements);
    assert_eq!(decoded.dofs, result.dofs);
    assert_eq!(decoded.steps.len(), result.steps.len());
    assert_eq!(decoded.execution, result.execution);
    assert!((decoded.final_force_n - result.final_force_n).abs() < 1.0e-12);
    assert!(
        (decoded.final_average_top_displacement_m - result.final_average_top_displacement_m).abs()
            < 1.0e-15
    );
    assert!((decoded.final_residual - result.final_residual).abs() < 1.0e-15);
}

#[test]
fn invalid_force_controls_are_rejected() {
    let mut invalid = Hyperelastic3DForceStudy::introductory();
    invalid.total_force_n = f64::NAN;
    let error = run_hyperelastic_3d_force(&invalid).expect_err("NaN load must fail closed");
    assert!(error.to_string().contains("total_force_n"));

    let mut invalid = Hyperelastic3DForceStudy::introductory();
    invalid.nz = 0;
    let error = run_hyperelastic_3d_force(&invalid).expect_err("zero mesh must fail closed");
    assert!(error.to_string().contains("mesh_resolution"));
}
