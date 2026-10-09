//! Release-backed XEM duplicate-key rejection reference.

use sonium_qa::QaResult;
use sonium_xem::{RoomConfig, io};

const CASE_ID: &str = "xem.duplicate-key-rejection.v1";

fn run() -> Result<QaResult, String> {
    let nested_room = r#"{
        "schema_version": 1,
        "room": {"width": 1, "width": 2},
        "sources": [],
        "listening_positions": [],
        "frequencies": {"min_freq": 100, "max_freq": 200, "num_points": 2}
    }"#;
    let room_error = io::read_external_room_document_checked(nested_room)
        .expect_err("nested duplicate room keys must be rejected");
    let room_rejected = room_error.to_string().contains("duplicate JSON object key");

    let envelope = r#"{"id":"first","id":"second"}"#;
    let envelope_error = io::read_external_multiphysics_document(envelope)
        .expect_err("duplicate envelope keys must be rejected");
    let envelope_rejected = envelope_error
        .to_string()
        .contains("duplicate JSON object key");

    let typed_room = r#"{
        "schema_version": 0,
        "room": {},
        "room": {},
        "sources": [],
        "listening_positions": [],
        "frequencies": {"min_freq": 100, "max_freq": 200, "num_points": 2}
    }"#;
    let typed_checked_rejected = RoomConfig::from_json_checked(typed_room).is_err();
    let typed_migrated_rejected = RoomConfig::from_json_migrated(typed_room).is_err();
    let checks = [
        room_rejected,
        envelope_rejected,
        typed_checked_rejected,
        typed_migrated_rejected,
    ];
    let checks_passed = checks.iter().filter(|check| **check).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "duplicate-key rejection checks: {checks_passed}/{}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .minimum("checks_passed", checks.len() as f64)
        .input_checksum("analytic/deterministic-fixture:xem-duplicate-key-rejection-v1")
        .reference(
            "independent duplicate-key rejection at external and typed XEM reader boundaries",
        )
        .reference_checksum("analytic/deterministic-fixture:xem-duplicate-key-rejection-v1")
        .dimension("nested room, multiphysics envelope, strict typed reader, migrated typed reader")
        .formulation("serde visitor rejects repeated object keys before last-value-wins collapse")
        .tolerance("four duplicate-key rejection checks pass")
        .seed("xem-duplicate-key-rejection:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("XEM duplicate-key result must serialize")
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
                    .expect("XEM duplicate-key failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
