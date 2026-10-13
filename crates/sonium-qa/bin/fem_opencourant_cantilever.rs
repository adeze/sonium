//! External reference health and native parity gate for cantilever.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs the starter and
//! engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set. The matching native
//! three-dimensional transient uses the original nodal load history, consistent mass,
//! and quadratic tetrahedra. Three spatial levels, a finest-grid timestep comparison,
//! and every-step dynamic residuals establish accuracy before tip-displacement parity.
//!
//! Missing references emit a structured optional skip, or fail when
//! SONIUM_QA_REQUIRE_OPENCOURANT=1. External probe health is recorded separately; release
//! success also requires a matching native solve and full named phenomenon.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    assert_probe_close, cantilever_case, engine_exe, probes_json, run_case, write_engine,
    write_starter,
};
use std::path::PathBuf;

#[path = "support/cantilever_native.rs"]
mod cantilever_native;

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/cantilever")
}

const CASE_ID: &str = "fem.opencourant.cantilever.native-parity.v1";

fn main() {
    let case = cantilever_case();
    let dir = decks_dir();
    let starter_path = dir.join("cantilever_0000.rad");
    let engine_path = dir.join("cantilever_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in pair must match the writer.
    let starter = write_starter(&case);
    let engine = write_engine(&case);
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1")
        || std::env::var("SONIUM_RAD_WRITE_DECKS").is_ok()
    {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&starter_path, &starter).unwrap();
        std::fs::write(&engine_path, &engine).unwrap();
        std::fs::write(
            &probes_path,
            probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for cantilever");
        return;
    }
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/cantilever/cantilever_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/cantilever/cantilever_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/cantilever/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_cantilever_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("cantilever_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("cantilever_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::execution_failure_result(
            CASE_ID,
            &error.to_string(),
        ));
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    for probe in &case.probes {
        match output.history.final_value(&probe.channel) {
            Ok(actual) => {
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
    let result = output
        .history
        .final_value(&case.probes[0].channel)
        .map_err(|error| error.to_string())
        .and_then(|tip| {
            cantilever_native::parity_result(&case, CASE_ID, &output.engine, tip, !failed)
                .map_err(|error| error.to_string())
        })
        .unwrap_or_else(|error| {
            sonium_qa::QaResult::failed(
                CASE_ID,
                format!("native transient validation failed: {error}"),
            )
            .requested_backend("cpu+opencourant")
            .actual_backend("cpu+opencourant")
            .metric("external_reference_executed", 1.0)
            .metric("native_parity_executed", 0.0)
        });
    sonium_qa::rad::gate::publish_and_exit(result);
}
