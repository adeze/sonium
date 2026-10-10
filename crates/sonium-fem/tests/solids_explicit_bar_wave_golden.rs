//! Wolfram golden replay: explicit bar wave speed and d'Alembert pulse.
//!
//! Replays `fixtures/solids_explicit_bar_wave_golden.json` (derived by
//! `fixtures/solids_explicit_bar_wave.wls` at 50-digit precision) through
//! the production explicit-dynamics API. Regenerate with:
//! `wolframscript -script fixtures/solids_explicit_bar_wave.wls > fixtures/solids_explicit_bar_wave_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::solver::explicit_dynamics::{
    bar_wave_speed, dalembert_gaussian, gaussian_pulse_strain_energy,
};

const GOLDEN: &str = include_str!("../fixtures/solids_explicit_bar_wave_golden.json");

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
        "{name}: production {actual:.6e} vs golden {expected:.6e}"
    );
}

#[test]
fn explicit_bar_wave_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let c = bar_wave_speed(200.0e9, 7800.0);
    check(&doc, "wave_speed", c);
    check(&doc, "u_center", dalembert_gaussian(10.0, 0.0008, c, 10.0, 0.4, 1.0e-3));
    check(
        &doc,
        "u_right",
        dalembert_gaussian(10.0 + c * 0.0008, 0.0008, c, 10.0, 0.4, 1.0e-3),
    );
    check(
        &doc,
        "u_left",
        dalembert_gaussian(10.0 - c * 0.0008, 0.0008, c, 10.0, 0.4, 1.0e-3),
    );
    check(
        &doc,
        "pulse_strain_energy",
        gaussian_pulse_strain_energy(200.0e9, 0.01, 0.4, 1.0e-3),
    );
}
