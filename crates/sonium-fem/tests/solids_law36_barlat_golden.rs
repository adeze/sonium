//! Wolfram golden replay: LAW36 tabulated yield.
//!
//! Replays `fixtures/law36_barlat_golden.json` (derived by
//! `fixtures/law36_barlat.wls` at 50-digit precision) through the
//! production LAW36 API. Regenerate with:
//! `wolframscript -script fixtures/law36_barlat.wls > fixtures/law36_barlat_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::Law36;

const GOLDEN: &str = include_str!("../fixtures/law36_barlat_golden.json");

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
fn law36_tabulated_yield_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let alu = Law36::card_aluminum();
    check_all(&doc, "yield_0", &[alu.nominal_yield(0.0, 0.0)]);
    check_all(&doc, "yield_0_04", &[alu.nominal_yield(0.04, 0.0)]);
    check_all(&doc, "slope_0_04", &[alu.nominal_slope(0.04, 0.0)]);
    check_all(&doc, "yield_0_075", &[alu.nominal_yield(0.075, 0.0)]);
    check_all(&doc, "yield_0_10", &[alu.nominal_yield(0.10, 0.0)]);
}
