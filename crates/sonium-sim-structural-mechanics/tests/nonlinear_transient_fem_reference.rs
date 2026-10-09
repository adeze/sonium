use sonium_sim_structural_mechanics::{
    NONLINEAR_TRANSIENT_FEM_3D_SCHEMA_VERSION, NonlinearTransientFem3DStudy,
    StructuralMechanicsError, run_nonlinear_transient_fem_3d,
};

#[test]
fn nonlinear_transient_fem_runs_actual_newmark_hyperelastic_steps() {
    let result = run_nonlinear_transient_fem_3d(&NonlinearTransientFem3DStudy::introductory())
        .expect("bounded nonlinear transient FEM fixture should converge");

    assert_eq!(
        result.schema_version,
        NONLINEAR_TRANSIENT_FEM_3D_SCHEMA_VERSION
    );
    assert!(result.converged);
    assert_eq!(result.steps.len(), 6);
    assert!(result.mesh_nodes > 0);
    assert!(result.elements > 0);
    assert_eq!(result.dofs, result.mesh_nodes * 3);
    assert!(result.bottom_nodes > 0);
    assert!(result.top_nodes > 0);
    assert!(result.final_displacement_m.is_finite());
    assert!(result.final_velocity_m_per_s.is_finite());
    assert!(result.final_acceleration_m_per_s2.is_finite());
    assert!(result.max_dynamic_equilibrium_residual_n.is_finite());
    assert!(result.max_total_energy_j.is_finite());
    assert!(result.steps.iter().all(|step| {
        step.converged
            && step.dynamic_equilibrium_residual_n.is_finite()
            && step.max_displacement_m.is_finite()
            && step.max_velocity_m_per_s.is_finite()
            && step.max_acceleration_m_per_s2.is_finite()
            && step.kinetic_energy_j.is_finite()
            && step.strain_energy_j.is_finite()
            && step.total_energy_j.is_finite()
            && step.damping_power_w.is_finite()
    }));
    assert!(result.steps[1].max_displacement_m > 0.0);
    assert_eq!(
        result.execution.capability_id,
        "sonium-structural.cpu-reference.f64.nonlinear-transient-fem-3d.v1"
    );
}

#[test]
fn nonlinear_transient_fem_accepts_reversal_and_records_energy_work() {
    let mut study = NonlinearTransientFem3DStudy::introductory();
    study.load_factors = vec![0.0, 0.5, 1.0, -0.25, 0.0];
    let result = run_nonlinear_transient_fem_3d(&study)
        .expect("bounded force reversal should remain a valid transient history");

    assert!(result.converged);
    assert_eq!(result.steps.len(), study.load_factors.len());
    assert!(result.total_input_work_j.is_finite());
    assert!(result.total_damping_dissipation_j.is_finite());
    assert!(result.total_damping_dissipation_j >= 0.0);
    assert_eq!(result.steps[3].load_factor, -0.25);
    assert_eq!(result.steps[4].applied_force_n, 0.0);
}

#[test]
fn nonlinear_transient_fem_rejects_invalid_controls() {
    let mut study = NonlinearTransientFem3DStudy::introductory();
    study.load_factors[0] = 0.1;
    assert!(matches!(
        run_nonlinear_transient_fem_3d(&study),
        Err(StructuralMechanicsError::InvalidConfiguration { .. })
    ));

    let mut study = NonlinearTransientFem3DStudy::introductory();
    study.time_step_s = 0.0;
    assert!(run_nonlinear_transient_fem_3d(&study).is_err());

    let mut study = NonlinearTransientFem3DStudy::introductory();
    study.load_factors.push(f64::NAN);
    assert!(run_nonlinear_transient_fem_3d(&study).is_err());

    let mut study = NonlinearTransientFem3DStudy::introductory();
    study.damping_ratio = 1.1;
    assert!(run_nonlinear_transient_fem_3d(&study).is_err());
}

#[test]
fn nonlinear_transient_fem_result_round_trips_through_serde() {
    let result = run_nonlinear_transient_fem_3d(&NonlinearTransientFem3DStudy::introductory())
        .expect("fixture should converge before serde round-trip");
    let encoded = serde_json::to_string(&result).expect("serialize nonlinear transient FEM");
    let decoded: sonium_sim_structural_mechanics::NonlinearTransientFem3DResult =
        serde_json::from_str(&encoded).expect("deserialize nonlinear transient FEM");
    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study, result.study);
    assert_eq!(decoded.mesh_nodes, result.mesh_nodes);
    assert_eq!(decoded.elements, result.elements);
    assert_eq!(decoded.dofs, result.dofs);
    assert_eq!(decoded.steps.len(), result.steps.len());
    assert_eq!(decoded.execution, result.execution);
    assert!((decoded.final_displacement_m - result.final_displacement_m).abs() < 1.0e-18);
    assert!((decoded.final_velocity_m_per_s - result.final_velocity_m_per_s).abs() < 1.0e-15);
    assert!((decoded.max_total_energy_j - result.max_total_energy_j).abs() < 1.0e-15);
    assert_eq!(decoded.converged, result.converged);
}
