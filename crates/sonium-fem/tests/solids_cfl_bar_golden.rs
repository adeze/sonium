//! Wolfram golden replay: 1-D bar CFL limit.
//!
//! Replays `fixtures/solids_cfl_bar_golden.json` (derived by
//! `fixtures/solids_cfl_bar.wls` at 50-digit precision) through the production
//! `sonium_compute::explicit_dt` kernels (`sound_speed_bar`, `critical_dt`,
//! `critical_dt_parallel`, `courant_number`). Regenerate with:
//! `wolframscript -script fixtures/solids_cfl_bar.wls > fixtures/solids_cfl_bar_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_compute::explicit_dt::{
    courant_number, critical_dt, critical_dt_parallel, sound_speed_bar,
};

const GOLDEN: &str = include_str!("../fixtures/solids_cfl_bar_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .unwrap_or_else(|| panic!("golden lacks {name} entry"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn input_array(doc: &Value, name: &str) -> Vec<f64> {
    doc["inputs"][name]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks input {name}"))
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()
        .unwrap()
}

fn check(doc: &Value, name: &str, index: usize, actual: f64) {
    let expected = golden_values(doc, name)[index];
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}[{index}]: production {actual:.17e} vs golden {expected:.17e}"
    );
}

#[test]
fn cfl_bar_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let young = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let rho = doc["inputs"]["density_kg_m3"].as_f64().unwrap();
    let lengths = input_array(&doc, "lengths_m");
    let stepped = input_array(&doc, "stepped_lengths_m");

    let sound = sound_speed_bar(young, rho);
    check(&doc, "sound_speed", 0, sound);

    let speeds = vec![sound; lengths.len()];
    for (index, &le) in lengths.iter().enumerate() {
        // Single-element critical step per length.
        check(&doc, "dt_crit", index, critical_dt(&[le], &[sound]));
        // The Courant number is exactly 1 at the critical step.
        let dt = critical_dt(&[le], &[sound]);
        assert!((courant_number(le, sound, dt) - 1.0).abs() <= 1.0e-15);
    }
    // The uniform mesh reduces to its finest member.
    let mesh_dt = critical_dt(&lengths, &speeds);
    check(&doc, "dt_crit", 0, mesh_dt);

    // Stepped mesh: the min rule picks the finest element, in both
    // serial and parallel reductions.
    let stepped_speeds = vec![sound; stepped.len()];
    check(
        &doc,
        "dt_crit_stepped",
        0,
        critical_dt(&stepped, &stepped_speeds),
    );
    check(
        &doc,
        "dt_crit_stepped",
        0,
        critical_dt_parallel(&stepped, &stepped_speeds),
    );
}
