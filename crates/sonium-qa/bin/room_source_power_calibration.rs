//! Compile-only contract for FEM source-power calibration.
//!
//! This source exercises the public power-normalization API using accepted
//! physical-power evidence. It intentionally performs no FEM solve, QA gate,
//! or simulator workload when compiled.

use sonium_sim_room::energy_validation::FemPhysicalPowerEvidence;
use sonium_sim_room::source_calibration::{
    SourcePowerCalibrationObservation, calibrate_source_power,
};

fn main() {
    let evidence = FemPhysicalPowerEvidence {
        source_power_w: 0.5,
        volume_damping_power_w: 0.2,
        boundary_power_w: 0.3,
        balance_residual_w: 0.0,
    };
    let observations = [
        SourcePowerCalibrationObservation::from_physical_power(125.0, "coarse", 1.0, &evidence),
        SourcePowerCalibrationObservation {
            frequency_hz: 250.0,
            mesh_id: "fine".into(),
            source_amplitude: 0.5,
            source_power_w: 0.125,
        },
    ];
    let report = calibrate_source_power(&observations, 0.25)
        .expect("compile-only source-power calibration fixture must be valid");
    assert_eq!(report.observation_count, 2);
    assert!(
        report
            .calibrated_source_amplitudes
            .iter()
            .all(|value| value.is_finite())
    );
    println!("SUCCESS");
}
