//! External reference health and native parity gate for shell-clamped-plate.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` extended writer, then runs the
//! starter and engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and compares the
//! center deflection, thickness, and hoop moment against independent references.
//! Native shell mesh/timestep studies and full-system force balance must also pass.
//!
//! Missing references emit a structured optional skip, or fail when
//! SONIUM_QA_REQUIRE_OPENCOURANT=1. External probe health is recorded separately; release
//! success also requires a matching native solve and full named phenomenon.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    assert_probe_close, assert_probes_payload_eq, engine_exe, extended_probes_json,
    run_extended_case, shell_clamped_plate_case, write_extended_engine, write_extended_starter,
};
use std::path::PathBuf;

#[path = "support/plate_native.rs"]
#[expect(
    dead_code,
    reason = "LU and scaled-coordinate variants are standalone diagnostics, not production gate choices"
)]
mod plate_native;

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/shell_clamped_plate")
}

const CASE_ID: &str = "fem.opencourant.shell-clamped-plate.native-parity.v1";

fn main() {
    let case = shell_clamped_plate_case();
    case.validate().expect("shell_clamped_plate must validate");
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        let dir = decks_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("shell_clamped_plate_0000.rad"),
            write_extended_starter(&case),
        )
        .unwrap();
        std::fs::write(
            dir.join("shell_clamped_plate_0001.rad"),
            write_extended_engine(&case),
        )
        .unwrap();
        std::fs::write(
            dir.join("probes.json"),
            extended_probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for shell_clamped_plate");
        return;
    }
    let dir = decks_dir();
    let starter_path = dir.join("shell_clamped_plate_0000.rad");
    let engine_path = dir.join("shell_clamped_plate_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in trio must match the writer.
    let starter = write_extended_starter(&case);
    let engine = write_extended_engine(&case);
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    let checked_probes = std::fs::read_to_string(&probes_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/shell_clamped_plate/shell_clamped_plate_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/shell_clamped_plate/shell_clamped_plate_0001.rad drifted from the rad writer"
    );
    assert_probes_payload_eq(
        &checked_probes,
        &extended_probes_json(&case, "none (prove-out deferred)"),
        "decks/shell_clamped_plate/probes.json drifted from the rad writer",
    );

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!(
        "sonium_rad_shell_clamped_plate_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("shell_clamped_plate_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("shell_clamped_plate_0001.rad"), &engine).unwrap();
    let output = run_extended_case(&work, &exe, &case).unwrap_or_else(|error| {
        sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::execution_failure_result(
            CASE_ID,
            &error.to_string(),
        ));
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    let mut external = Vec::new();
    for probe in &case.base.probes {
        match output.history.final_value(&probe.channel) {
            Ok(actual) => {
                external.push(actual);
                println!(
                    "probe {}: actual {actual:.6e}, expected {:.6e}",
                    probe.label, probe.expected
                );
                if let Err(error) = assert_probe_close(
                    actual,
                    probe.expected,
                    probe.abs_tol,
                    probe.rel_tol,
                    &probe.label,
                ) {
                    eprintln!("FAIL: {error}");
                    failed = true;
                }
            }
            Err(error) => {
                eprintln!("FAIL: probe {} unreadable: {error}", probe.label);
                failed = true;
            }
        }
    }
    std::fs::remove_dir_all(&work).ok();
    let result = plate_native::parity_result(&case, CASE_ID, &output.engine, &external, !failed)
        .unwrap_or_else(|error| {
            sonium_qa::QaResult::failed(
                CASE_ID,
                format!("native full plate validation failed: {error}"),
            )
            .requested_backend("cpu+opencourant")
            .actual_backend("cpu+opencourant")
            .metric("external_reference_executed", 1.0)
            .metric("native_parity_executed", 0.0)
        });
    sonium_qa::rad::gate::publish_and_exit(result);
}
