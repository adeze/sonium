//! Wolfram golden replay: restart serialization bitrot guard.
//!
//! Replays `fixtures/solids_restart_bitrot_golden.json` (derived by
//! `fixtures/solids_restart_bitrot.wls` at 50-digit precision) through the
//! production `sonium_compute::explicit_restart` API: the fixed golden
//! state is rebuilt from the golden inputs, serialized to a restart file,
//! read back, and its vector sums are recomputed and checked against the
//! 50-digit references. Regenerate with:
//! `wolframscript -script fixtures/solids_restart_bitrot.wls > fixtures/solids_restart_bitrot_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_compute::explicit_restart::{read_restart, write_restart};
use sonium_compute::explicit_state::{EXPLICIT_STATE_SCHEMA, ExplicitState};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const GOLDEN: &str = include_str!("../fixtures/solids_restart_bitrot_golden.json");

fn golden_vector(doc: &Value, name: &str) -> Vec<f64> {
    doc["inputs"][name]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks input {name}"))
        .iter()
        .map(|value| {
            value
                .as_f64()
                .unwrap_or_else(|| panic!("golden input {name} not numeric"))
        })
        .collect()
}

fn golden_value(doc: &Value, name: &str) -> f64 {
    doc["values"][name]["values"][0]
        .as_str()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .parse()
        .unwrap()
}

fn check(doc: &Value, name: &str, actual: f64) {
    let expected = golden_value(doc, name);
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}: replay {actual:.6e} vs golden {expected:.6e}"
    );
}

/// Builds a unique scratch path under `TMPDIR` (no `tempfile` in fem).
fn scratch_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "solids_restart_bitrot_{}_{}_{tag}.json",
        std::process::id(),
        nanos
    ))
}

#[test]
fn restart_round_trip_matches_wolfram_sums() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    assert_eq!(doc["meta"]["case"], "restart_bitrot");
    assert_eq!(
        doc["inputs"]["restart_schema"].as_u64().unwrap(),
        u64::from(EXPLICIT_STATE_SCHEMA),
        "golden predates the running restart schema; regenerate it"
    );

    let mut state = ExplicitState::new(doc["inputs"]["node_count"].as_u64().unwrap() as usize);
    state.positions = golden_vector(&doc, "positions");
    state.velocities = golden_vector(&doc, "velocities");
    state.masses = golden_vector(&doc, "masses");
    state.forces = golden_vector(&doc, "forces");
    state.time = doc["inputs"]["time_s"].as_f64().unwrap();
    state.step = doc["inputs"]["step"].as_u64().unwrap();
    state.dt = doc["inputs"]["dt_s"].as_f64().unwrap();
    assert!(state.is_consistent());

    let first = scratch_path("first");
    let second = scratch_path("second");
    write_restart(&first, &state).unwrap();
    write_restart(&second, &state).unwrap();
    // Deterministic bytes: the same state must produce identical files.
    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap()
    );
    let back = read_restart(&first).unwrap();
    std::fs::remove_file(&first).unwrap();
    std::fs::remove_file(&second).unwrap();

    // Bit-exact round trip: any flipped bit fails here.
    assert_eq!(back, state);

    // Arithmetic pins: recomputed sums must match Wolfram's 50-digit
    // references, catching serializer drift the equality above cannot
    // see (e.g. a format that round-trips but mis-encodes values).
    check(&doc, "sum_positions", back.positions.iter().sum());
    check(&doc, "sum_velocities", back.velocities.iter().sum());
    check(&doc, "sum_masses", back.masses.iter().sum());
    check(&doc, "sum_forces", back.forces.iter().sum());
    check(
        &doc,
        "sumsq_positions",
        back.positions.iter().map(|value| value * value).sum(),
    );
    check(
        &doc,
        "sumsq_velocities",
        back.velocities.iter().map(|value| value * value).sum(),
    );
    check(
        &doc,
        "sumsq_masses",
        back.masses.iter().map(|value| value * value).sum(),
    );
    check(
        &doc,
        "sumsq_forces",
        back.forces.iter().map(|value| value * value).sum(),
    );
}
