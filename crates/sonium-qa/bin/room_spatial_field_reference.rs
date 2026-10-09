//! Release-backed RoomSim free-field spatial-field reference.

use sonium_qa::QaResult;
use sonium_sim_room::spatial_field::{SpatialFieldProvenance, SpatialFieldReference};
use sonium_xem::Point3D;

const CASE_ID: &str = "roomsim.spatial-field-reference.v1";

fn run() -> Result<QaResult, String> {
    let reference = SpatialFieldReference::monopole(
        Point3D::new(0.0, 0.0, 0.0),
        vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)],
        vec![200.0, 400.0],
        343.0,
        5.88e-4,
    )?;
    reference.validate_layout()?;

    let provenance_ok =
        reference.provenance == SpatialFieldProvenance::AnalyticMonopoleGreenFunction;
    let radial_ratio = reference.pressures[0][0].norm() / reference.pressures[1][0].norm();
    let radial_error = (radial_ratio - 2.0).abs();
    let frequency_ratio = reference.pressures[0][1].norm() / reference.pressures[0][0].norm();
    let frequency_error = (frequency_ratio - 2.0).abs();
    if !provenance_ok {
        return Err("spatial-field reference lost analytic provenance".to_string());
    }
    if radial_error > 1.0e-12 || frequency_error > 1.0e-12 {
        return Err(format!(
            "monopole ratios differ from independent 1/r and frequency references: radial={radial_error:e}, frequency={frequency_error:e}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", 4.0)
        .metric("radial_ratio_error", radial_error)
        .metric("frequency_ratio_error", frequency_error)
        .minimum("checks_passed", 4.0)
        .maximum("radial_ratio_error", 1.0e-12)
        .maximum("frequency_ratio_error", 1.0e-12)
        .input_checksum("analytic/deterministic-fixture:roomsim-spatial-field-v1")
        .reference("independent outgoing monopole Green-function 1/r and frequency scaling")
        .reference_checksum("analytic/deterministic-fixture:roomsim-spatial-field-v1")
        .dimension("two probes at 1 m and 2 m; two frequencies; 343 m/s air")
        .formulation("production RoomSim analytic spatial-field reference with explicit provenance")
        .tolerance("finite layout; analytic provenance; radial and frequency ratio errors <= 1e-12")
        .seed("roomsim-spatial-field-reference:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim spatial-field result must serialize")
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
                    .expect("RoomSim spatial-field failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
