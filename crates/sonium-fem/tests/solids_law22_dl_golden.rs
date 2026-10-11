//! Wolfram golden replay: LAW22 damage factor.
//!
//! Replays `fixtures/law22_dl_golden.json` (derived by
//! `fixtures/law22_dl.wls` at 50-digit precision) through the production
//! LAW22 API. Regenerate with:
//! `wolframscript -script fixtures/law22_dl.wls > fixtures/law22_dl_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::Law22;

const GOLDEN: &str = include_str!("../fixtures/law22_dl_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("golden lacks {name}"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn check_all(doc: &Value, name: &str, actual: &[f64]) {
    let expected = golden_values(doc, name);
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= abs + rel * want.abs(),
            "{name}[{i}]: production {got:.6e} vs golden {want:.6e}"
        );
    }
}

#[test]
fn law22_damage_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let alu = Law22::card_aluminum();
    check_all(&doc, "damage_0_05", &[alu.damage_factor(0.05)]);
    check_all(&doc, "damage_0_11", &[alu.damage_factor(0.11)]);
    check_all(&doc, "damage_0_12", &[alu.damage_factor(0.12)]);
    check_all(&doc, "damage_0_15", &[alu.damage_factor(0.15)]);
    check_all(&doc, "damage_0_20", &[alu.damage_factor(0.20)]);
}
