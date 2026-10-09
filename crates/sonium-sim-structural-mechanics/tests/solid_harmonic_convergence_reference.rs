use sonium_sim_structural_mechanics::{
    SOLID3D_HARMONIC_CONVERGENCE_SCHEMA_VERSION, Solid3DHarmonicMeshConvergenceReport,
    Solid3DHarmonicResponseStudy, run_solid3d_harmonic_mesh_convergence,
};

fn study() -> Solid3DHarmonicResponseStudy {
    let mut study = Solid3DHarmonicResponseStudy::introductory();
    study.base.geometry.nx = 1;
    study.base.geometry.ny = 1;
    study.base.geometry.nz = 1;
    study.frequencies_hz = vec![1.0, 10.0];
    study.observation_node = 5;
    study
}

#[test]
fn direct_solid3d_harmonic_convergence_tracks_physical_observation_and_refinement() {
    let report = run_solid3d_harmonic_mesh_convergence(&study(), 1.0, std::f64::consts::PI)
        .expect("direct harmonic mesh convergence");

    assert_eq!(
        report.schema_version,
        SOLID3D_HARMONIC_CONVERGENCE_SCHEMA_VERSION
    );
    assert_eq!(report.coarse_resolution, [1, 1, 1]);
    assert_eq!(report.refined_resolution, [2, 2, 2]);
    assert!(report.topology_refined);
    assert_eq!(report.coarse.points.len(), 2);
    assert_eq!(report.refined.points.len(), 2);
    assert_eq!(report.coarse_observation_node, 5);
    assert!(report.refined_observation_node < report.refined.nodes);
    assert_eq!(report.displacement_relative_changes.len(), 2);
    assert_eq!(report.phase_changes_rad.len(), 2);
    assert!(report.observation_coordinate_m[0].is_finite());
    assert!(report.max_displacement_relative_change.is_finite());
    assert!(report.max_phase_change_rad.is_finite());
    assert!(report.peak_displacement_relative_change.is_finite());
    assert!(report.peak_frequency_relative_change.is_finite());
    assert!(report.converged);
    assert_eq!(
        report.refined.study.observation_node,
        report.refined_observation_node
    );
}

#[test]
fn direct_solid3d_harmonic_convergence_can_fail_tight_response_gate() {
    let report = run_solid3d_harmonic_mesh_convergence(&study(), 1.0e-14, 1.0e-14)
        .expect("report should retain failed convergence diagnostics");

    assert!(!report.converged);
    assert!(report.max_displacement_relative_change > 1.0e-14);
}

#[test]
fn direct_solid3d_harmonic_convergence_rejects_invalid_controls_and_overflow() {
    assert!(run_solid3d_harmonic_mesh_convergence(&study(), 0.0, 0.1).is_err());
    assert!(run_solid3d_harmonic_mesh_convergence(&study(), 0.1, 0.0).is_err());
    assert!(
        run_solid3d_harmonic_mesh_convergence(&study(), 0.1, std::f64::consts::PI + 1.0e-6)
            .is_err()
    );

    let mut overflow = study();
    overflow.base.geometry.nx = usize::MAX;
    assert!(run_solid3d_harmonic_mesh_convergence(&overflow, 0.1, 0.1).is_err());
}

#[test]
fn direct_solid3d_harmonic_convergence_round_trips_through_serde() {
    let report =
        run_solid3d_harmonic_mesh_convergence(&study(), 1.0, std::f64::consts::PI).unwrap();
    let encoded = serde_json::to_string(&report).unwrap();
    let decoded: Solid3DHarmonicMeshConvergenceReport = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.schema_version, report.schema_version);
    assert_eq!(decoded.coarse_resolution, report.coarse_resolution);
    assert_eq!(decoded.refined_resolution, report.refined_resolution);
    assert_eq!(
        decoded.coarse_observation_node,
        report.coarse_observation_node
    );
    assert_eq!(
        decoded.refined_observation_node,
        report.refined_observation_node
    );
    assert_eq!(decoded.coarse.nodes, report.coarse.nodes);
    assert_eq!(decoded.refined.nodes, report.refined.nodes);
    assert_eq!(decoded.coarse.points.len(), report.coarse.points.len());
    assert_eq!(decoded.refined.points.len(), report.refined.points.len());
    assert_eq!(decoded.topology_refined, report.topology_refined);
    assert_eq!(decoded.converged, report.converged);
    for (decoded_point, report_point) in decoded
        .coarse
        .points
        .iter()
        .zip(&report.coarse.points)
        .chain(decoded.refined.points.iter().zip(&report.refined.points))
    {
        assert!((decoded_point.displacement_abs_m - report_point.displacement_abs_m).abs() < 1e-15);
        assert!((decoded_point.phase_lag_rad - report_point.phase_lag_rad).abs() < 1e-15);
        assert!(
            (decoded_point.relative_solver_residual - report_point.relative_solver_residual).abs()
                < 1e-15
        );
    }
    assert!(
        (decoded.max_displacement_relative_change - report.max_displacement_relative_change).abs()
            < 1e-15
    );
    assert!((decoded.max_phase_change_rad - report.max_phase_change_rad).abs() < 1e-15);
}
