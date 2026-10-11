//! Wolfram golden replay: LAW62 Ogden stresses.
//!
//! Replays `fixtures/law62_vep_golden.json` (derived by
//! `fixtures/law62_vep.wls` at 50-digit precision) through the production
//! LAW62 API. Regenerate with:
//! `wolframscript -script fixtures/law62_vep.wls > fixtures/law62_vep_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::Law62;

const GOLDEN: &str = include_str!("../fixtures/law62_vep_golden.json");

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
fn law62_ogden_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let rubber = Law62::rubber_prony();
    let lateral = 1.0 / 1.2f64.sqrt();
    let principal = rubber.ogden_principal(&[1.2, lateral, lateral]);
    check_all(&doc, "ogden_principal", &principal);
    check_all(&doc, "ground_shear", &[rubber.ground_shear()]);
    check_all(&doc, "gamma_infinity", &[rubber.gamma_infinity()]);
}
