//! Independent gate for physical-domain FEM nodal-field extraction and export.

use sonium_qa::QaResult;
use sonium_sim_room::{gallery, scenarios};

const CASE_ID: &str = "roomsim.fem-nodal-gallery.v1";

fn run() -> Result<QaResult, String> {
    let scenario = scenarios::scenario_by_name("small_stereo_2_0")
        .ok_or_else(|| "small_stereo_2_0 scenario is unavailable".to_string())?;
    let mut simulation = scenario.simulation.clone();
    // The gallery includes a line plot, whose renderer requires at least two
    // samples. Two solved frequencies keep this bounded while exercising the
    // same nodal-field export path.
    simulation.frequencies.truncate(2);

    let (output, field, diagnostics) = sonium_sim_room::fem_runner::run_fem_with_mesh_field_report(
        &simulation,
        &sonium_sim_room::fem_runner::FemMeshConfig::default(),
    )
    .map_err(|error| error.to_string())?;
    output.validate_layout()?;
    field.validate()?;
    if field.mesh.nodes.is_empty() || field.pressures[0][0].is_empty() {
        return Err("FEM nodal field is empty".into());
    }

    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    gallery::write_gallery_with_fem_field(
        &output,
        directory.path(),
        &scenario.name,
        "fem",
        &simulation,
        &field,
        &gallery::GalleryOptions::default(),
    )?;

    let sidecar: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            directory
                .path()
                .join("fem_pressure_slice_xy_provenance.json"),
        )
        .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if sidecar["provenance"] != "solved"
        || sidecar["measured"] != false
        || sidecar["fem_nodal_field"] != true
    {
        return Err("FEM nodal gallery sidecar does not preserve solved-field provenance".into());
    }
    let csv = std::fs::read_to_string(directory.path().join("fem_pressure_slice_xy.csv"))
        .map_err(|error| error.to_string())?;
    if csv.lines().count() != 32 * 32 + 1 || !csv.contains(",solved\n") {
        return Err("FEM nodal gallery slice has an unexpected shape or provenance".into());
    }
    sonium_xem::io::vtu::validate_vtu_file(directory.path().join("fem_pressure_nodal_field.vtu"))
        .map_err(|error| error.to_string())?;

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("mesh_node_count", field.mesh.nodes.len() as f64)
        .metric("field_frequency_count", field.frequencies.len() as f64)
        .metric("solver_max_residual", diagnostics.max_residual)
        .minimum("mesh_node_count", 4.0)
        .reference("physical-domain FEM nodal solution, P1 slice interpolation, and VTU export")
        .reference_checksum("analytic/deterministic-fixture:roomsim-fem-nodal-gallery-v1")
        .dimension("small_stereo_2_0, two solved FEM frequencies, full physical domain")
        .formulation("CPU FEM nodal field capture; measured=false; CSV/VTU are authoritative")
        .tolerance("finite accepted nodal field, 32x32 solved P1 slice, valid VTU")
        .seed("roomsim-fem-nodal-gallery:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim FEM nodal result must serialize")
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
                    .expect("RoomSim FEM nodal failure result must serialize")
            );
            eprintln!("FAILED");
            std::process::exit(1);
        }
    }
}
