//! QA: Scordelis-Lo roof shells vs Radioss/OpenRadioss quasi-statics.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` extended
//! writer, then runs the starter and engine when `RADISS_EXE` or
//! `OPENCOURANT_EXE` is set and compares the free-edge midspan
//! deflection against a REGRESSION baseline (-0.2574), not the MacNeal
//! target (-0.3024): the engine's quads lock 15% stiff on this curved
//! membrane benchmark (see `shell_scordelis_lo_case`). Without an engine
//! it prints the documented skip message and exits 0.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    assert_probe_close, assert_probes_payload_eq, engine_exe, extended_probes_json,
    run_extended_case, shell_scordelis_lo_case, write_extended_engine, write_extended_starter,
};
use std::path::PathBuf;

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/shell_scordelis_lo")
}

fn main() {
    let case = shell_scordelis_lo_case();
    case.validate().expect("shell_scordelis_lo must validate");
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        let dir = decks_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("shell_scordelis_lo_0000.rad"),
            write_extended_starter(&case),
        )
        .unwrap();
        std::fs::write(
            dir.join("shell_scordelis_lo_0001.rad"),
            write_extended_engine(&case),
        )
        .unwrap();
        std::fs::write(
            dir.join("probes.json"),
            extended_probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for shell_scordelis_lo");
        return;
    }
    let dir = decks_dir();
    let starter_path = dir.join("shell_scordelis_lo_0000.rad");
    let engine_path = dir.join("shell_scordelis_lo_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in trio must match the writer.
    let starter = write_extended_starter(&case);
    let engine = write_extended_engine(&case);
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    let checked_probes = std::fs::read_to_string(&probes_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/shell_scordelis_lo/shell_scordelis_lo_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/shell_scordelis_lo/shell_scordelis_lo_0001.rad drifted from the rad writer"
    );
    assert_probes_payload_eq(
        &checked_probes,
        &extended_probes_json(&case, "none (prove-out deferred)"),
        "decks/shell_scordelis_lo/probes.json drifted from the rad writer",
    );

    let Some(exe) = engine_exe() else {
        println!("skipped: set RADISS_EXE or OPENCOURANT_EXE");
        return;
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!(
        "sonium_rad_shell_scordelis_lo_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("shell_scordelis_lo_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("shell_scordelis_lo_0001.rad"), &engine).unwrap();
    let output = run_extended_case(&work, &exe, &case).unwrap_or_else(|error| {
        eprintln!("rad shell_scordelis_lo run failed: {error}");
        std::process::exit(1);
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    for (index, probe) in case.base.probes.iter().enumerate() {
        // The edge DY averages the hold (t >= 5.5); the slow ramp
        // leaves only a small corner transient around the static center.
        let actual = if index == 0 {
            output.history.channel(&probe.channel).and_then(|values| {
                let mut sum = 0.0;
                let mut n = 0;
                for (time, value) in output.history.times.iter().zip(values.iter()) {
                    if *time >= 5.5 {
                        sum += value;
                        n += 1;
                    }
                }
                if n == 0 {
                    return Err(sonium_qa::rad::RadError::Parse(format!(
                        "channel {} has no hold samples",
                        probe.channel
                    )));
                }
                Ok(sum / n as f64)
            })
        } else {
            output.history.final_value(&probe.channel)
        };
        match actual {
            Ok(value) => {
                println!(
                    "probe {}: actual {value:.6e}, expected {:.6e}",
                    probe.label, probe.expected
                );
                if let Err(error) = assert_probe_close(
                    value,
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
    println!("PASS: qa-fem-radioss-shell-scordelis-lo");
}
