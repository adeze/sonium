//! Release-backed RoomSim fast-room modal reference.
//!
//! This packet exercises the production fast-hybrid modal path in a rigid
//! shoebox whose first frequencies are independently known in closed form.
//! It intentionally covers only the low-frequency zero-mode/reference slice;
//! it is not a claim of browser, FEM, BEM, or full spatial-field parity.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_sim_room::fast_room::{
    FastRoomConfig, rigid_shoebox_mode_frequency_reference,
    rigid_shoebox_zero_mode_pressure_reference, run_fast_hybrid_with_config,
};
use sonium_xem::{
    Directivity, DirectivityGrid, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source,
    SurfaceConfig,
};

const CASE_ID: &str = "roomsim.fast-modal-reference.v1";

fn simulation() -> RoomSimulation {
    let source = Source::new(
        Point3D::new(0.031, 0.037, 0.043),
        Directivity::Grid(DirectivityGrid::omnidirectional()),
        1.0,
    )
    .with_name("main".into());
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(0.1, 0.1, 0.1)),
        vec![source],
        vec![Point3D::new(0.067, 0.052, 0.061)],
    );
    simulation.speed_of_sound = 343.0;
    simulation.frequencies = vec![40.0, 80.0];
    simulation.boundaries.walls = SurfaceConfig::Rigid;
    simulation.boundaries.floor = SurfaceConfig::Rigid;
    simulation.boundaries.ceiling = SurfaceConfig::Rigid;
    simulation
}

fn run() -> Result<(f64, f64, f64), String> {
    let simulation = simulation();
    let dimensions = [0.1, 0.1, 0.1];
    let frequencies = [40.0, 80.0];
    let mode_frequency_error = [
        ([0, 0, 0], 0.0),
        ([1, 0, 0], 1715.0),
        ([0, 1, 0], 1715.0),
        ([0, 0, 1], 1715.0),
    ]
    .into_iter()
    .map(|(indices, expected)| {
        let actual = rigid_shoebox_mode_frequency_reference(dimensions, indices, 343.0)
            .map_err(|error| error.to_string())?;
        Ok::<f64, String>((actual - expected).abs())
    })
    .collect::<Result<Vec<_>, _>>()?
    .into_iter()
    .fold(0.0, f64::max);

    let output = run_fast_hybrid_with_config(&simulation, &FastRoomConfig::default())
        .map_err(|error| error.to_string())?;
    let mut zero_mode_error: f64 = 0.0;
    let mut finite_samples = 0.0;
    for (index, frequency) in frequencies.into_iter().enumerate() {
        let reference = rigid_shoebox_zero_mode_pressure_reference(dimensions, frequency, 343.0)
            .map_err(|error| error.to_string())?;
        let source_amplitude =
            simulation.sources[0].amplitude_towards(&simulation.listening_positions[0], frequency);
        let expected = reference * Complex64::new(source_amplitude, 0.0);
        let actual = output.pressures[0][0][index];
        if !(actual.re.is_finite() && actual.im.is_finite()) {
            return Err(format!(
                "non-finite modal pressure at {frequency} Hz: {actual:?}"
            ));
        }
        finite_samples += 1.0;
        zero_mode_error = zero_mode_error.max((actual - expected).norm());
    }

    let rejection_checks = [
        rigid_shoebox_mode_frequency_reference([0.0, 0.1, 0.1], [1, 0, 0], 343.0).is_err(),
        rigid_shoebox_zero_mode_pressure_reference(dimensions, f64::NAN, 343.0).is_err(),
        rigid_shoebox_zero_mode_pressure_reference([f64::NAN, 0.1, 0.1], 40.0, 343.0).is_err(),
    ]
    .into_iter()
    .filter(|rejected| *rejected)
    .count() as f64;

    Ok((
        mode_frequency_error,
        zero_mode_error,
        finite_samples + rejection_checks,
    ))
}

fn main() {
    match run() {
        Ok((mode_frequency_error, zero_mode_error, checks)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("mode_frequency_error_hz", mode_frequency_error)
                .metric("zero_mode_pressure_error", zero_mode_error)
                .metric("boundary_checks", checks)
                .maximum("mode_frequency_error_hz", 1.0e-12)
                .maximum("zero_mode_pressure_error", 1.0e-12)
                .minimum("boundary_checks", 5.0)
                .input_checksum("analytic/deterministic-fixture:roomsim-fast-modal-v1")
                .reference("independent rigid-shoebox Neumann mode and zero-mode pressure formulas")
                .reference_checksum("analytic/deterministic-fixture:roomsim-fast-modal-v1")
                .dimension("0.1 m rigid shoebox; two low-frequency samples; one source/listener")
                .formulation("production fast-hybrid modal path against independent rigid-shoebox mode and zero-mode formulas")
                .tolerance("mode and zero-mode errors <= 1e-12; all outputs finite; malformed inputs reject")
                .seed("roomsim-fast-modal-reference:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim result must serialize")
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
                    .expect("RoomSim failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
