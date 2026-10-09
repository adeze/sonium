//! Release-backed RoomSim physical FEM nodal-field spatial reference.

use sonium_qa::QaResult;
use sonium_sim_room::{
    fem_runner::{self, FemMeshConfig, FemSourceModel},
    source_calibration::SOURCE_VOLUME_VELOCITY_REFERENCE,
};
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
    constants::AIR_DENSITY_20C,
};

const CASE_ID: &str = "roomsim.fem-spatial-field-reference.v1";

fn run() -> Result<QaResult, String> {
    let source_position = Point3D::new(0.4, 0.35, 0.35);
    let probes = vec![
        Point3D::new(1.2, 1.1, 1.0),
        Point3D::new(1.0, 0.95, 0.9),
        Point3D::new(1.25, 0.6, 0.85),
    ];
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        vec![Source::omnidirectional(source_position, 1.0)],
        probes.clone(),
    );
    let matched_wall = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: matched_wall.clone(),
        ceiling: matched_wall.clone(),
        walls: matched_wall,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation.frequencies = vec![400.0];

    let (_, field, diagnostics) = fem_runner::run_fem_with_mesh_field_report(
        &simulation,
        &FemMeshConfig {
            fine_radius: 0.4,
            modal_damping_ratio: 0.0,
            source_model: FemSourceModel::PointLoad,
            ..FemMeshConfig::default()
        },
    )
    .map_err(|error| error.to_string())?;
    field.validate()?;
    if !diagnostics.max_residual.is_finite() {
        return Err("FEM field residual is non-finite".into());
    }

    let sampled = field.sample_p1(0, 0, &probes)?;
    let mut maximum_ratio_error = 0.0_f64;
    for (probe_index, (actual, probe)) in sampled.iter().zip(&probes).enumerate() {
        let distance = source_position.distance_to(probe);
        let expected = 2.0
            * std::f64::consts::PI
            * simulation.frequencies[0]
            * AIR_DENSITY_20C
            * SOURCE_VOLUME_VELOCITY_REFERENCE
            / (4.0 * std::f64::consts::PI * distance);
        let ratio = actual.norm() / expected;
        if !ratio.is_finite() || !(0.6..=1.7).contains(&ratio) {
            return Err(format!(
                "FEM spatial field ratio out of bounds at probe {probe_index}: {ratio:.6}"
            ));
        }
        maximum_ratio_error = maximum_ratio_error.max((ratio - 1.0).abs());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("maximum_ratio_error", maximum_ratio_error)
        .metric("probe_count", sampled.len() as f64)
        .metric("max_residual", diagnostics.max_residual)
        .minimum("probe_count", 3.0)
        .maximum("maximum_ratio_error", 0.7)
        .input_checksum("analytic/deterministic-fixture:roomsim-fem-spatial-field-v1")
        .reference("independent outgoing monopole Green-function magnitude at off-node probes")
        .reference_checksum("analytic/deterministic-fixture:roomsim-fem-spatial-field-v1")
        .dimension(
            "1.6 x 1.5 x 1.4 m matched-boundary room; one source; three off-node probes; 400 Hz",
        )
        .formulation("full physical-domain CPU FEM P1 nodal field with P1 probe interpolation")
        .tolerance("three finite probes with FEM/reference magnitude ratios in [0.6, 1.7]")
        .seed("roomsim-fem-spatial-field:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim FEM spatial-field result must serialize")
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
                    .expect("RoomSim FEM spatial-field failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
