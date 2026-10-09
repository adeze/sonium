//! Bounded production-FEM source-power calibration evidence.
//!
//! The fixture deliberately uses two small full-domain CPU meshes and two
//! frequencies. It exercises the physical-power ledger at the runner boundary
//! before handing observations to the source calibration contract; it is not a
//! production-scale room simulation or a QA-binary invocation.

use sonium_sim_room::fem_runner::{
    FemMeshConfig, FemPhysicalEvidenceStatus, run_fem_full_domain_with_mesh_report,
};
use sonium_sim_room::source_calibration::{
    SourcePowerCalibrationObservation, calibrate_source_power,
};
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

fn fixture() -> RoomSimulation {
    let mut simulation = RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(0.9, 0.8, 0.7)),
        vec![Source::omnidirectional(Point3D::new(0.22, 0.31, 0.28), 1.0)],
        vec![Point3D::new(0.68, 0.53, 0.42)],
        80.0,
        100.0,
        2,
    );
    let wall = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: wall.clone(),
        ceiling: wall.clone(),
        walls: wall,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation
}

fn observations(
    mesh_id: &str,
    diagnostics: &sonium_sim_room::fem_runner::FemRunDiagnostics,
) -> Vec<SourcePowerCalibrationObservation> {
    assert_eq!(
        diagnostics.physical_evidence_status,
        FemPhysicalEvidenceStatus::Complete
    );
    diagnostics.require_converged().unwrap();
    diagnostics
        .frequency_stats
        .iter()
        .map(|stats| {
            let evidence = stats.physical_power.as_ref().unwrap();
            assert!(evidence.source_power_w.is_finite());
            assert!(evidence.source_power_w > 0.0);
            SourcePowerCalibrationObservation::from_physical_power(
                stats.frequency_hz,
                mesh_id,
                1.0,
                evidence,
            )
        })
        .collect()
}

#[test]
fn production_fem_physical_power_feeds_mesh_frequency_calibration() {
    let simulation = fixture();
    let coarse_config = FemMeshConfig {
        fine_radius: 0.25,
        modal_damping_ratio: 0.02,
        ..FemMeshConfig::default()
    };
    let fine_config = FemMeshConfig {
        fine_radius: 0.2,
        modal_damping_ratio: 0.02,
        ..FemMeshConfig::default()
    };

    let (coarse_output, coarse_diagnostics) =
        run_fem_full_domain_with_mesh_report(&simulation, &coarse_config).unwrap();
    let (fine_output, fine_diagnostics) =
        run_fem_full_domain_with_mesh_report(&simulation, &fine_config).unwrap();
    coarse_output.validate_layout().unwrap();
    fine_output.validate_layout().unwrap();

    let mut calibration_observations = observations("coarse", &coarse_diagnostics);
    calibration_observations.extend(observations("fine", &fine_diagnostics));
    assert_eq!(calibration_observations.len(), 4);

    let target_power_w = calibration_observations
        .iter()
        .map(|observation| observation.source_power_w)
        .sum::<f64>()
        / calibration_observations.len() as f64;
    let report = calibrate_source_power(&calibration_observations, target_power_w).unwrap();
    report
        .require_input_power_budget(report.input_max_abs_relative_power_error)
        .unwrap();
    assert_eq!(report.observation_count, 4);
    assert!(report.input_relative_power_rms.is_finite());
    assert!(report.input_max_abs_relative_power_error.is_finite());
    assert!(report.calibrated_relative_power_rms < 1.0e-12);
    assert!(report.calibrated_max_abs_relative_power_error < 1.0e-12);
    assert!(
        report
            .calibrated_source_amplitudes
            .iter()
            .all(|amplitude| amplitude.is_finite() && *amplitude > 0.0)
    );
}
