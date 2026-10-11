//! QA: Hertz sphere vs Radioss/OpenRadioss rigid-wall contact.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs
//! the starter and engine when `RADISS_EXE` or `OPENCOURANT_EXE` is set and
//! compares the pole displacement against the wall travel plus two custom
//! force checks (wall-force balance, contact-stiffness window). The Hertz
//! force magnitude itself is not asserted: the `sqrt(R delta)` patch is
//! unresolvable on 10 mm bricks (see `hertz_sphere_case`). Without an engine
//! it prints the documented skip message and exits 0.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    assert_probe_close, engine_exe, hertz_sphere_case, probes_json, run_case, write_engine,
    write_starter,
};
use std::path::PathBuf;

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/hertz_sphere")
}

fn main() {
    let case = hertz_sphere_case();
    let dir = decks_dir();
    let starter_path = dir.join("hertz_sphere_0000.rad");
    let engine_path = dir.join("hertz_sphere_0001.rad");
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
        println!("regenerated decks for hertz_sphere");
        return;
    }
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/hertz_sphere/hertz_sphere_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/hertz_sphere/hertz_sphere_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/hertz_sphere/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        println!("skipped: set RADISS_EXE or OPENCOURANT_EXE");
        return;
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_hertz_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("hertz_sphere_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("hertz_sphere_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        eprintln!("rad hertz_sphere run failed: {error}");
        std::process::exit(1);
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    for probe in &case.probes {
        match output.history.final_value(&probe.channel) {
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
    // Quasi-static squeeze: the wall forces must balance (Newton's 3rd law).
    // th_to_csv already reports forces, not impulses.
    match (
        output.history.final_value("FNZ/RWALL/1"),
        output.history.final_value("FNZ/RWALL/2"),
    ) {
        (Ok(bottom), Ok(top)) => {
            let scale = bottom.abs().max(top.abs());
            println!("probe wall-force balance: bottom {bottom:.6e}, top {top:.6e}");
            if (bottom + top).abs() > 0.01 * scale {
                eprintln!("FAIL: wall forces do not balance: {bottom:.6e} + {top:.6e}");
                failed = true;
            }
            // Contact alive and mesh-plausible: single-node stiffness of a
            // 10 mm steel brick is order E * h ~ 1e9 N/m; the window spans
            // a decade each way. Travel is minus the DZ probe expectation.
            let travel = -case.probes[0].expected;
            let stiffness = scale / travel;
            println!("probe contact stiffness: {stiffness:.6e} N/m");
            if !(1.0e8..=3.0e9).contains(&stiffness) {
                eprintln!("FAIL: contact stiffness {stiffness:.6e} outside [1e8, 3e9]");
                failed = true;
            }
        }
        _ => {
            eprintln!("FAIL: wall force channels FNZ/RWALL/1..2 unreadable");
            failed = true;
        }
    }
    std::fs::remove_dir_all(&work).ok();
    if failed {
        std::process::exit(1);
    }
    println!("PASS: qa-fem-radioss-hertz-sphere");
}
