//! Wolfram golden replay: Johnson-Cook crash response.
//!
//! Replays `fixtures/solids_crash_johnson_cook_golden.json` (derived by
//! `fixtures/solids_crash_johnson_cook.wls` at 50-digit precision) through
//! the production crash-material API. Regenerate with:
//! `wolframscript -script fixtures/solids_crash_johnson_cook.wls > fixtures/solids_crash_johnson_cook_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::fracture::ductile::JohnsonCookDamage;
use sonium_fem::physics::solid::plasticity::crash::JohnsonCookMaterial;

const GOLDEN: &str = include_str!("../fixtures/solids_crash_johnson_cook_golden.json");

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
fn johnson_cook_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let steel = JohnsonCookMaterial::c4340_steel();
    check(&doc, "flow_quasistatic", steel.flow_stress(0.01, 1.0, 293.0));
    check(&doc, "flow_fast", steel.flow_stress(0.01, 1000.0, 293.0));
    check(&doc, "flow_hot", steel.flow_stress(0.01, 1.0, 900.0));
    let damage = JohnsonCookDamage::c4340_steel();
    check(
        &doc,
        "fracture_strain_uniaxial",
        damage.fracture_strain(1.0 / 3.0, 1.0, 0.0),
    );
}
