//! Wolfram golden replay: necking Considere strains and Bridgman factors.
//!
//! Replays `fixtures/solids_necking_consider_golden.json` (derived by
//! `fixtures/solids_necking_consider.wls` at 50-digit precision) through
//! the production ductile-damage reference API. Regenerate with:
//! `wolframscript -script fixtures/solids_necking_consider.wls > fixtures/solids_necking_consider_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::fracture::ductile::{bridgman_correction, hollomon_necking_strain};

const GOLDEN: &str = include_str!("../fixtures/solids_necking_consider_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|v| v.as_str().unwrap().parse().unwrap())
        .collect()
}

fn check_all(doc: &Value, name: &str, actual: &[f64]) {
    let expected = golden_values(doc, name);
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= abs + rel * e.abs(),
            "{name}[{i}]: production {a:.6e} vs golden {e:.6e}"
        );
    }
}

#[test]
fn necking_consider_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let strains: Vec<f64> = [0.1, 0.2, 0.3].iter().map(|n| hollomon_necking_strain(*n)).collect();
    check_all(&doc, "uniform_strain", &strains);
    let bridgman: Vec<f64> = [0.5, 1.0, 2.0].iter().map(|x| bridgman_correction(*x)).collect();
    check_all(&doc, "bridgman", &bridgman);
}
