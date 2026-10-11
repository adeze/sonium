//! Wolfram golden replay: LAW44 Cowper-Symonds flow.
//!
//! Replays `fixtures/law44_cs_golden.json` (derived by
//! `fixtures/law44_cs.wls` at 50-digit precision) through the production
//! LAW44 API. Regenerate with:
//! `wolframscript -script fixtures/law44_cs.wls > fixtures/law44_cs_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::Law44;

const GOLDEN: &str = include_str!("../fixtures/law44_cs_golden.json");

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
fn law44_cowper_symonds_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let metal = Law44::card_metal();
    check_all(&doc, "flow_quasi", &[metal.flow_stress(0.01, 0.0)]);
    check_all(&doc, "flow_mid", &[metal.flow_stress(0.04, 100.0)]);
    check_all(&doc, "flow_fast", &[metal.flow_stress(0.01, 1000.0)]);
    check_all(&doc, "factor_100", &[metal.rate_factor(100.0)]);
    check_all(&doc, "factor_1000", &[metal.rate_factor(1000.0)]);
}
