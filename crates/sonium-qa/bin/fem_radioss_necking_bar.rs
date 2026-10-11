//! QA: necking bar vs Radioss/OpenRadioss (elastic prove-out leg).
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs
//! the starter and engine when `RADISS_EXE` or `OPENCOURANT_EXE` is set and
//! compares the tip elongation against the elastic bar closed form (FL/EA)
//! at a sub-yield load. Without an engine it prints the documented skip
//! message and exits 0.
//!
//! The full necking bar has no closed form: this deck plus the Wolfram
//! incremental reference and the Consider/Bridgman checks in
//! `qa-fem-necking-bar` are the validation legs. Considere-strain and
//! post-necking probes need the plastic LAW cards tracked for Worker 1.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    RadBoundaryCondition, RadCase, RadConcentratedLoad, RadElasticMaterial, RadFunction,
    RadNodeSet, RadProbe, assert_probe_close, cantilever_brick_mesh, engine_exe, probes_json,
    run_case, write_engine, write_starter,
};
use std::path::PathBuf;

const NAME: &str = "necking_bar";

/// Builds the necking bar case on the real `sonium_qa::rad` API.
fn necking_bar_case() -> RadCase {
    let length = 0.1;
    let section = 0.01;
    let youngs = 200.0e9;
    // 20 kN holds the bar at 200 MPa: elastic, sub-yield, comparable with
    // the future plastic leg before diffuse necking.
    let force = 20.0e3;
    let (mesh, root_set, tip_set, tip_node) =
        cantilever_brick_mesh(10, 2, 2, length, section, section);
    let expected = force * length / (youngs * section * section);
    let tip_scale = force / tip_set.len() as f64;
    RadCase {
        name: NAME.to_string(),
        title: "sonium_qa rad necking bar elastic tension".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: youngs,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "grip".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "pull".to_string(),
                nodes: tip_set,
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "clamp grip".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            name: "ramp 0 to 1 over stop time".to_string(),
            points: vec![(0.0, 0.0), (0.9, 0.9), (1.0, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "tensile pull".to_string(),
            set: 1,
            direction: 1,
            function: 0,
            force_scale_n: tip_scale,
        }],
        walls: Vec::new(),
        th_nodes: vec![tip_node],
        probes: vec![RadProbe {
            label: "tip DX".to_string(),
            channel: format!("DX/{tip_node}"),
            expected,
            abs_tol: 1e-9,
            rel_tol: 0.05,
        }],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    }
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/necking_bar")
}

fn main() {
    let case = necking_bar_case();
    let dir = decks_dir();
    let starter_path = dir.join("necking_bar_0000.rad");
    let engine_path = dir.join("necking_bar_0001.rad");
    let probes_path = dir.join("probes.json");

    let starter = write_starter(&case);
    let engine = write_engine(&case);
    if std::env::var("SONIUM_RAD_WRITE_DECKS").is_ok() {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&starter_path, &starter).unwrap();
        std::fs::write(&engine_path, &engine).unwrap();
        std::fs::write(
            &probes_path,
            probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("wrote decks/necking_bar artifacts");
        return;
    }
    // No hand-built decks: the checked-in pair must match the writer.
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/necking_bar/necking_bar_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/necking_bar/necking_bar_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/necking_bar/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        println!("skipped: set RADISS_EXE or OPENCOURANT_EXE");
        return;
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_necking_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("necking_bar_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("necking_bar_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        eprintln!("rad necking_bar run failed: {error}");
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
    println!("PASS: qa-fem-radioss-necking-bar");
}
