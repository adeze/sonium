//! QA: FEM cantilever vs Radioss/OpenRadioss explicit quasi-statics.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs
//! the starter and engine when `RADISS_EXE` or `OPENCOURANT_EXE` is set and
//! compares the tip displacement against Euler-Bernoulli. Without an engine
//! it prints the documented skip message and exits 0.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    assert_probe_close, cantilever_case, engine_exe, probes_json, run_case, write_engine,
    write_starter,
};
use std::path::PathBuf;

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/cantilever")
}

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
        println!("skipped: set RADISS_EXE or OPENCOURANT_EXE");
        return;
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_cantilever_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("cantilever_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("cantilever_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        eprintln!("rad cantilever run failed: {error}");
        std::process::exit(1);
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
    if failed {
        std::process::exit(1);
    }
    println!("PASS: qa-fem-radioss-cantilever");
}
