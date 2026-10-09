use sonium_sim_structural_mechanics::{
    SOLID3D_STATIC_CONVERGENCE_SCHEMA_VERSION, Solid3DStudy, run_solid3d_static_mesh_convergence,
};

#[test]
fn production_solid3d_mesh_convergence_reports_refinement_and_gates() {
    let mut study = Solid3DStudy::introductory();
    study.geometry.nx = 1;
    study.geometry.ny = 1;
    study.geometry.nz = 1;
    let report = run_solid3d_static_mesh_convergence(&study, 0.5)
        .expect("small production solid should converge");

    assert_eq!(
        report.schema_version,
        SOLID3D_STATIC_CONVERGENCE_SCHEMA_VERSION
    );
    assert_eq!(report.coarse_resolution, [1, 1, 1]);
    assert_eq!(report.refined_resolution, [2, 2, 2]);
    assert!(report.topology_refined);
    assert!(report.converged);
    assert!(report.displacement_relative_change.is_finite());
    assert!(report.energy_relative_change.is_finite());
    assert!(report.centroid_stress_relative_change.is_finite());
}

#[test]
fn production_solid3d_mesh_convergence_rejects_invalid_controls_before_solving() {
    let study = Solid3DStudy::introductory();
    assert!(run_solid3d_static_mesh_convergence(&study, 0.0).is_err());
    assert!(run_solid3d_static_mesh_convergence(&study, f64::NAN).is_err());

    let mut overflowing = study;
    overflowing.geometry.nx = usize::MAX;
    let error = run_solid3d_static_mesh_convergence(&overflowing, 0.5)
        .expect_err("overflow must fail before attempting a huge mesh");
    assert!(error.to_string().contains("nx"));
}

#[test]
fn production_solid3d_mesh_convergence_result_round_trips() {
    let mut study = Solid3DStudy::introductory();
    study.geometry.nx = 1;
    study.geometry.ny = 1;
    study.geometry.nz = 1;
    let report = run_solid3d_static_mesh_convergence(&study, 0.5).unwrap();
    let encoded = serde_json::to_string(&report).expect("report should serialize");
    let decoded: sonium_sim_structural_mechanics::Solid3DStaticMeshConvergenceReport =
        serde_json::from_str(&encoded).expect("report should deserialize");

    assert_eq!(decoded.schema_version, report.schema_version);
    assert_eq!(decoded.coarse_resolution, report.coarse_resolution);
    assert_eq!(decoded.refined_resolution, report.refined_resolution);
    assert_eq!(decoded.topology_refined, report.topology_refined);
    assert_eq!(decoded.converged, report.converged);
    assert!((decoded.energy_relative_change - report.energy_relative_change).abs() < 1.0e-12);
}
