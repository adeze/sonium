//! Wolfram golden replay: LAW27 brittle damage ramp.
//!
//! Replays `fixtures/law27_brittle_golden.json` (derived by
//! `fixtures/law27_brittle.wls` at 50-digit precision) through the
//! production LAW27 API. Regenerate with:
//! `wolframscript -script fixtures/law27_brittle.wls > fixtures/law27_brittle_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::Law27;

const GOLDEN: &str = include_str!("../fixtures/law27_brittle_golden.json");

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
fn law27_brittle_ramp_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let law = Law27::steel_brittle();
    check_all(&doc, "damage_0_05", &[law.dir1.damage_at(0.05)]);
    check_all(&doc, "damage_0_08", &[law.dir1.damage_at(0.08)]);
    check_all(&doc, "damage_0_115", &[law.dir1.damage_at(0.115)]);
    let knee = law.dir1.eps_t + law.dir1.dmax * (law.dir1.eps_m - law.dir1.eps_t);
    check_all(&doc, "damage_knee", &[law.dir1.damage_at(knee)]);
    check_all(&doc, "damage_0_15", &[law.dir1.damage_at(0.15)]);
    check_all(&doc, "damage_0_20", &[law.dir1.damage_at(0.20)]);
}
