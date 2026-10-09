//! Release-facing RoomSim fast-room modal truncation convergence gate.
//!
//! This source is compile-checked as part of QA promotion. Running the QA
//! binary performs one bounded reduced-order modal ladder; it does not claim
//! FEM, BEM, measured-room, or browser parity.

use sonium_qa::QaResult;
use sonium_sim_room::fast_room::run_modal_convergence;
use sonium_xem::{
    Directivity, DirectivityGrid, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source,
    SurfaceConfig,
};

const CASE_ID: &str = "roomsim.modal-convergence.v1";

fn build_simulation() -> RoomSimulation {
    let source = Source::new(
        Point3D::new(0.37, 0.43, 0.51),
        Directivity::Grid(DirectivityGrid::omnidirectional()),
        1.0,
    )
    .with_name("main".into());
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(2.0, 2.0, 2.0)),
        vec![source],
        vec![Point3D::new(1.37, 1.43, 1.51)],
    );
    simulation.speed_of_sound = 343.0;
    simulation.boundaries.walls = SurfaceConfig::Rigid;
    simulation.boundaries.floor = SurfaceConfig::Rigid;
    simulation.boundaries.ceiling = SurfaceConfig::Rigid;
    simulation
}

fn run() -> Result<QaResult, String> {
    let report = run_modal_convergence(
        &build_simulation(),
        0,
        0,
        30.0,
        [120.0, 240.0, 480.0],
        0.15,
        1.0e-6,
    )
    .map_err(|error| format!("fast-room modal convergence failed: {error}"))?;
    report
        .require_converged()
        .map_err(|error| format!("fast-room modal convergence budget failed: {error}"))?;

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("coarse_cutoff_hz", report.cutoff_frequencies_hz[0])
        .metric("medium_cutoff_hz", report.cutoff_frequencies_hz[1])
        .metric("fine_cutoff_hz", report.cutoff_frequencies_hz[2])
        .metric("coarse_mode_count", report.mode_counts[0] as f64)
        .metric("medium_mode_count", report.mode_counts[1] as f64)
        .metric("fine_mode_count", report.mode_counts[2] as f64)
        .metric("coarse_to_medium_absolute_error", report.absolute_errors[0])
        .metric("medium_to_fine_absolute_error", report.absolute_errors[1])
        .metric("coarse_to_medium_relative_error", report.relative_errors[0])
        .metric("medium_to_fine_relative_error", report.relative_errors[1])
        .minimum("coarse_mode_count", 1.0)
        .minimum("medium_mode_count", report.mode_counts[0] as f64)
        .minimum("fine_mode_count", report.mode_counts[1] as f64)
        .maximum("coarse_to_medium_relative_error", 0.15)
        .maximum("medium_to_fine_relative_error", 0.15)
        .input_checksum("analytic/deterministic-fixture:roomsim-modal-convergence-v1")
        .reference_checksum("production/fast-hybrid-modal-basis:three-cutoff-ladder-v1")
        .dimension("2 x 2 x 2 m rigid shoebox; one source/listener; 30 Hz")
        .formulation(
            "production fast-hybrid modal evaluator with three increasing frequency cutoffs",
        )
        .tolerance("adjacent complex-pressure changes <= 15% relative plus 1e-6 Pa absolute")
        .seed("roomsim-modal-convergence:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim modal-convergence result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("RoomSim modal-convergence failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
