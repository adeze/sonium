//! Release artifact contract for the XEM VTU/PVD interchange path.
//!
//! The writer is exercised through the public XEM API, while the validation
//! below deliberately uses only stable XML markers and independent byte-level
//! checks.  The release runner performs the authoritative SHA-256 hashing of
//! the declared files in its bundle.

use sonium_qa::QaResult;
use sonium_xem::io::{PvdCollection, VtuWriter};
use sonium_xem::{Point3D, RoomMesh, SurfaceElement};
use std::fs;
use std::path::{Path, PathBuf};

const CASE_ID: &str = "xem.artifact-bundle.v1";
const VTU_ARTIFACT: &str = "xem/frame_000.vtu";
const PVD_ARTIFACT: &str = "xem/series.pvd";

fn fixture() -> RoomMesh {
    RoomMesh {
        nodes: vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ],
        elements: vec![
            SurfaceElement::triangle(0, 1, 2),
            SurfaceElement::triangle(0, 2, 3),
        ],
    }
}

fn output_dir() -> PathBuf {
    std::env::var_os("SONIUM_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/sonium-qa-artifacts"))
}

fn check_artifact(path: &Path, required_markers: &[&str]) -> Result<usize, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let text = String::from_utf8(bytes.clone())
        .map_err(|error| format!("{} is not UTF-8 XML: {error}", path.display()))?;
    for marker in required_markers {
        if !text.contains(marker) {
            return Err(format!("{} is missing marker {marker:?}", path.display()));
        }
    }
    Ok(bytes.len())
}

fn run() -> Result<QaResult, String> {
    let root = output_dir();
    let xem_dir = root.join("xem");
    fs::create_dir_all(&xem_dir).map_err(|error| format!("create artifact directory: {error}"))?;

    let mesh = fixture();
    mesh.validate()
        .map_err(|error| format!("fixture validation: {error}"))?;
    let frame = xem_dir.join("frame_000.vtu");
    let mut writer = VtuWriter::new(&mesh);
    writer
        .try_add_point_scalar("temperature", &[1.0, 2.0, 3.0, 4.0])
        .map_err(|error| format!("point field: {error}"))?;
    writer
        .try_add_cell_scalar("region", &[7.0, 8.0])
        .map_err(|error| format!("cell field: {error}"))?;
    writer
        .write(&frame)
        .map_err(|error| format!("write VTU: {error}"))?;

    let pvd = xem_dir.join("series.pvd");
    let mut collection = PvdCollection::new(&pvd);
    collection
        .try_push(0.0, "frame_000.vtu")
        .map_err(|error| format!("PVD entry: {error}"))?;
    collection
        .finish()
        .map_err(|error| format!("write PVD: {error}"))?;

    let frame_bytes = check_artifact(
        &frame,
        &[
            "<VTKFile",
            "UnstructuredGrid",
            "NumberOfPoints=\"4\"",
            "NumberOfCells=\"2\"",
            "temperature",
            "region",
        ],
    )?;
    let pvd_bytes = check_artifact(&pvd, &["<VTKFile", "Collection", "frame_000.vtu"])?;

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("node_count", mesh.num_nodes() as f64)
        .metric("element_count", mesh.num_elements() as f64)
        .metric("vtu_bytes", frame_bytes as f64)
        .metric("pvd_bytes", pvd_bytes as f64)
        .metric("artifact_count", 2.0)
        .minimum("node_count", 4.0)
        .minimum("element_count", 2.0)
        .minimum("artifact_count", 2.0)
        .artifact(VTU_ARTIFACT)
        .artifact(PVD_ARTIFACT)
        .input_checksum("analytic/deterministic-fixture:xem-artifact-bundle-v1")
        .reference("independent XML marker and topology checks")
        .reference_checksum("analytic/deterministic-fixture")
        .dimension("4-node, 2-triangle surface mesh; VTU + PVD")
        .formulation("public XEM VtuWriter and PvdCollection artifact path")
        .tolerance("exact node/cell counts and required external-format markers")
        .seed("xem-artifact-bundle:two-files:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("XEM artifact result must serialize")
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
                    .expect("XEM artifact failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
