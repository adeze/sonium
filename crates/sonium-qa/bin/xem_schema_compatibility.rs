//! Release-facing XEM schema compatibility and migration reference.
//!
//! This packet exercises the published room-configuration boundary rather
//! than only the in-crate unit tests: a legacy v0 fixture must be rejected by
//! the strict reader, accepted by the explicit migration path, validated
//! against the current schema, and survive a strict serde round trip. Future
//! schema versions must fail closed.

use serde_json::Value;
use sonium_qa::QaResult;
use sonium_xem::{CURRENT_SCHEMA_VERSION, RoomConfig, io, schema};
use std::time::Instant;

const CASE_ID: &str = "xem.schema-compatibility.v1";
const LEGACY_FIXTURE: &str = include_str!("../../sonium-xem/fixtures/room-config-v0.json");

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let legacy_rejected = RoomConfig::from_json_checked(LEGACY_FIXTURE).is_err();
    if !legacy_rejected {
        return Err("strict reader accepted legacy v0 fixture".into());
    }

    let migrated = RoomConfig::from_json_migrated(LEGACY_FIXTURE)
        .map_err(|error| format!("v0 migration failed: {error}"))?;
    if migrated.schema_version.0 != CURRENT_SCHEMA_VERSION {
        return Err(format!(
            "migration produced schema version {}, expected {CURRENT_SCHEMA_VERSION}",
            migrated.schema_version.0
        ));
    }

    let migrated_value = serde_json::to_value(&migrated)
        .map_err(|error| format!("migrated config serialization failed: {error}"))?;
    io::validate_schema_document(&schema::room_config_json_schema(), &migrated_value)
        .map_err(|error| format!("migrated config failed current schema validation: {error}"))?;
    let migrated_json = serde_json::to_string(&migrated)
        .map_err(|error| format!("migrated config JSON serialization failed: {error}"))?;
    let round_tripped = RoomConfig::from_json_checked(&migrated_json)
        .map_err(|error| format!("migrated strict round trip failed: {error}"))?;
    if round_tripped.schema_version != migrated.schema_version
        || round_tripped.sources.len() != migrated.sources.len()
        || round_tripped.listening_positions.len() != migrated.listening_positions.len()
    {
        return Err("migrated configuration changed structural identity on round trip".into());
    }

    let mut future_value: Value = serde_json::from_str(LEGACY_FIXTURE)
        .map_err(|error| format!("legacy fixture JSON parse failed: {error}"))?;
    future_value["schema_version"] = Value::from(CURRENT_SCHEMA_VERSION + 1);
    let future_json = serde_json::to_string(&future_value)
        .map_err(|error| format!("future-version fixture serialization failed: {error}"))?;
    if RoomConfig::from_json_checked(&future_json).is_ok()
        || RoomConfig::from_json_migrated(&future_json).is_ok()
    {
        return Err("future schema version was not rejected by both readers".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("legacy_strict_rejection", f64::from(legacy_rejected))
        .metric("migration_success", 1.0)
        .metric("schema_validation_success", 1.0)
        .metric("strict_round_trip_success", 1.0)
        .metric("future_version_rejection", 1.0)
        .metric("migrated_source_count", migrated.sources.len() as f64)
        .metric(
            "migrated_listener_count",
            migrated.listening_positions.len() as f64,
        )
        .minimum("legacy_strict_rejection", 1.0)
        .minimum("migration_success", 1.0)
        .minimum("schema_validation_success", 1.0)
        .minimum("strict_round_trip_success", 1.0)
        .minimum("future_version_rejection", 1.0)
        .minimum("migrated_source_count", 1.0)
        .minimum("migrated_listener_count", 1.0)
        .reference("independent legacy v0 migration and current-schema compatibility checks")
        .reference_checksum("analytic/deterministic-fixture:xem-schema-compatibility-v1")
        .dimension("room-config v0 fixture, current v1 schema, and future-version rejection")
        .formulation("strict v0 rejection, explicit v0-to-v1 migration, JSON Schema validation, serde round trip, future-version rejection")
        .tolerance("five compatibility gates and non-empty migrated source/listener sets pass")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("XEM schema compatibility result must serialize")
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
                    .expect("XEM schema compatibility failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
