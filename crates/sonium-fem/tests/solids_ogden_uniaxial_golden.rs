//! Wolfram golden replay: Ogden uniaxial Cauchy stress.
//!
//! Replays `fixtures/solids_ogden_uniaxial_golden.json` (derived by
//! `fixtures/solids_ogden_uniaxial.wls` at 50-digit precision) through the
//! production Ogden closed form. Regenerate with:
//! `wolframscript -script fixtures/solids_ogden_uniaxial.wls > fixtures/solids_ogden_uniaxial_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::nonlinear_solid::{OgdenMaterial, ogden_uniaxial_cauchy};

const GOLDEN: &str = include_str!("../fixtures/solids_ogden_uniaxial_golden.json");

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
fn ogden_uniaxial_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let treloar = OgdenMaterial::treloar();
    check(
        &doc,
        "cauchy_tension",
        ogden_uniaxial_cauchy(&treloar.mu, &treloar.alpha, 1.5),
    );
    check(
        &doc,
        "cauchy_compression",
        ogden_uniaxial_cauchy(&treloar.mu, &treloar.alpha, 0.8),
    );
}
