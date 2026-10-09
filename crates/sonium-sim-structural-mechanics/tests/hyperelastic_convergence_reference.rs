use sonium_sim_structural_mechanics::{
    HYPERELASTIC_3D_CONVERGENCE_SCHEMA_VERSION, Hyperelastic3DMeshConvergenceReport,
    Hyperelastic3DStudy, run_hyperelastic_3d_mesh_convergence,
};

fn study() -> Hyperelastic3DStudy {
    let mut study = Hyperelastic3DStudy::introductory();
    study.nx = 1;
    study.ny = 1;
    study.nz = 1;
    study.load_steps = 2;
    study.prescribed_extension_m = 0.02;
    study
}

#[test]
fn hyperelastic_mesh_convergence_runs_actual_refined_newton_paths() {
    let report = run_hyperelastic_3d_mesh_convergence(&study(), 1.0e-6, 1.0e-9, 1.0e-8)
        .expect("hyperelastic mesh convergence");

    assert_eq!(
        report.schema_version,
        HYPERELASTIC_3D_CONVERGENCE_SCHEMA_VERSION
    );
    assert_eq!(report.coarse_resolution, [1, 1, 1]);
    assert_eq!(report.refined_resolution, [2, 2, 2]);
    assert!(report.topology_refined);
    assert!(report.coarse.converged);
    assert!(report.refined.converged);
    assert!(report.coarse.newton_iterations > 0);
    assert!(report.refined.newton_iterations > 0);
    assert!(report.relative_strain_energy_change.is_finite());
    assert!(report.relative_displacement_change.is_finite());
    assert!(report.refined_affine_displacement_error_m.is_finite());
    assert!(report.refined_deformation_gradient_error.is_finite());
    assert!(report.refined_constitutive_gradient_error.is_finite());
    assert!(report.converged);
}

#[test]
fn hyperelastic_mesh_convergence_retains_failed_gate_diagnostics() {
    let report = run_hyperelastic_3d_mesh_convergence(&study(), 1.0e-14, 1.0e-14, 1.0e-14)
        .expect("failed convergence should be reported");

    assert!(!report.converged);
    assert!(report.relative_strain_energy_change > 1.0e-14);
}

#[test]
fn hyperelastic_mesh_convergence_rejects_invalid_controls_and_overflow() {
    assert!(run_hyperelastic_3d_mesh_convergence(&study(), 0.0, 1.0, 1.0).is_err());
    assert!(run_hyperelastic_3d_mesh_convergence(&study(), 1.0, 0.0, 1.0).is_err());
    assert!(run_hyperelastic_3d_mesh_convergence(&study(), 1.0, 1.0, 0.0).is_err());

    let mut overflow = study();
    overflow.nx = usize::MAX;
    assert!(run_hyperelastic_3d_mesh_convergence(&overflow, 1.0, 1.0, 1.0).is_err());
}

#[test]
fn hyperelastic_mesh_convergence_round_trips_through_serde() {
    let report = run_hyperelastic_3d_mesh_convergence(&study(), 1.0e-6, 1.0e-9, 1.0e-8).unwrap();
    let encoded = serde_json::to_string(&report).unwrap();
    let decoded: Hyperelastic3DMeshConvergenceReport = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.schema_version, report.schema_version);
    assert_eq!(decoded.coarse_resolution, report.coarse_resolution);
    assert_eq!(decoded.refined_resolution, report.refined_resolution);
    assert_eq!(decoded.coarse.mesh_nodes, report.coarse.mesh_nodes);
    assert_eq!(decoded.refined.mesh_nodes, report.refined.mesh_nodes);
    assert_eq!(decoded.topology_refined, report.topology_refined);
    assert_eq!(decoded.converged, report.converged);
    assert!(
        (decoded.relative_strain_energy_change - report.relative_strain_energy_change).abs()
            < 1.0e-15
    );
    assert!(
        (decoded.relative_displacement_change - report.relative_displacement_change).abs()
            < 1.0e-15
    );
}
