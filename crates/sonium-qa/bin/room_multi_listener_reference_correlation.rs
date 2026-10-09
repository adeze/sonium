//! Release-backed RoomSim multi-listener free-field correlation reference.

use sonium_qa::QaResult;
use sonium_sim_room::SimulationOutput;
use sonium_sim_room::fem_runner::{FemMeshConfig, FemSourceModel, run_fem_with_mesh};
use sonium_sim_room::source_calibration::free_field_peak_pressure_for_amplitude;
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

const CASE_ID: &str = "roomsim.multi-listener-reference-correlation.v1";

fn build_simulation() -> (RoomSimulation, Point3D, Vec<Point3D>) {
    let source = Point3D::new(0.4, 0.35, 0.35);
    let listeners = vec![Point3D::new(1.2, 1.1, 1.0), Point3D::new(1.0, 0.55, 1.05)];
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        vec![Source::omnidirectional(source, 1.0)],
        listeners.clone(),
    );
    simulation.boundaries = BoundaryConfig {
        floor: SurfaceConfig::Admittance {
            real: 1.0,
            imag: 0.0,
        },
        ceiling: SurfaceConfig::Admittance {
            real: 1.0,
            imag: 0.0,
        },
        walls: SurfaceConfig::Admittance {
            real: 1.0,
            imag: 0.0,
        },
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation.frequencies = vec![200.0, 400.0, 800.0];
    (simulation, source, listeners)
}

fn run() -> Result<(f64, f64), String> {
    let (simulation, source, listeners) = build_simulation();
    let output: SimulationOutput = run_fem_with_mesh(
        &simulation,
        &FemMeshConfig {
            fine_radius: 0.4,
            modal_damping_ratio: 0.0,
            source_model: FemSourceModel::PointLoad,
            ..FemMeshConfig::default()
        },
    )
    .map_err(|error| format!("RoomSim FEM reference run failed: {error}"))?;
    output.validate_layout()?;
    if output
        .provenance
        .iter()
        .any(|provenance| *provenance != sonium_sim_room::FrequencyProvenance::Solved)
    {
        return Err("RoomSim reference run returned non-solved provenance".into());
    }

    let mut maximum_relative_error = 0.0_f64;
    let mut sample_count = 0usize;
    for (listener_idx, listener) in listeners.iter().enumerate() {
        let distance = ((listener.x - source.x).powi(2)
            + (listener.y - source.y).powi(2)
            + (listener.z - source.z).powi(2))
        .sqrt();
        for (frequency_idx, &frequency) in simulation.frequencies.iter().enumerate() {
            let reference = free_field_peak_pressure_for_amplitude(
                frequency,
                simulation.speed_of_sound,
                sonium_xem::constants::AIR_DENSITY_20C,
                1.0,
                distance,
            )?;
            let numerical = output.pressures[0][listener_idx][frequency_idx].norm();
            let ratio = numerical / reference;
            if !ratio.is_finite() || !(0.6..=1.7).contains(&ratio) {
                return Err(format!(
                    "listener {listener_idx} frequency {frequency} Hz ratio {ratio} outside [0.6, 1.7]"
                ));
            }
            maximum_relative_error = maximum_relative_error.max((ratio - 1.0).abs());
            sample_count += 1;
        }
    }
    Ok((maximum_relative_error, sample_count as f64))
}

fn main() {
    match run() {
        Ok((maximum_relative_error, sample_count)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("maximum_free_field_ratio_error", maximum_relative_error)
                .metric("solved_sample_count", sample_count)
                .maximum("maximum_free_field_ratio_error", 0.7)
                .minimum("solved_sample_count", 6.0)
                .input_checksum("analytic/deterministic-fixture:roomsim-multi-listener-correlation-v1")
                .reference("independent monopole Green-function magnitude omega*rho*Q/(4*pi*r) at two listeners")
                .reference_checksum("analytic/deterministic-fixture:roomsim-multi-listener-correlation-v1")
                .dimension("1.6 x 1.5 x 1.4 m room; one source; two listeners; 200/400/800 Hz")
                .formulation("production RoomSim FEM point-load path with matched admittance boundaries")
                .tolerance("each numerical/reference magnitude ratio in [0.6, 1.7]; all six samples solved")
                .seed("roomsim-multi-listener-reference-correlation:cpu:v1");
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
                    .expect("RoomSim failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
