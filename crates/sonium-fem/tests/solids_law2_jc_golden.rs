//! Wolfram golden replay: LAW2 Johnson-Cook flow stress.
//!
//! Replays `fixtures/law2_jc_golden.json` (derived by
//! `fixtures/law2_jc.wls` at 50-digit precision) through the production
//! LAW2 API. Regenerate with:
//! `wolframscript -script fixtures/law2_jc.wls > fixtures/law2_jc_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::Law2;

const GOLDEN: &str = include_str!("../fixtures/law2_jc_golden.json");

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
fn law2_johnson_cook_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let steel = Law2::c4340_steel();
    check_all(
        &doc,
        "flow_quasistatic",
        &[steel.flow_stress(0.01, 1.0, 298.0)],
    );
    check_all(&doc, "flow_fast", &[steel.flow_stress(0.01, 1000.0, 298.0)]);
    check_all(&doc, "flow_hot", &[steel.flow_stress(0.01, 1.0, 900.0)]);
    check_all(&doc, "thermal_hot", &[steel.thermal_factor(900.0)]);
    check_all(
        &doc,
        "homologous_hot",
        &[steel.homologous_temperature(900.0)],
    );
}
