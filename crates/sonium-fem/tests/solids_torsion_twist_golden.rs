//! Wolfram golden replay: pure-torsion twist.
//!
//! Replays `fixtures/solids_torsion_twist_golden.json` (derived by
//! `fixtures/solids_torsion_twist.wls` at 50-digit precision) through the
//! production `sonium_wave` torsion closed forms. Regenerate with:
//! `wolframscript -script fixtures/solids_torsion_twist.wls > fixtures/solids_torsion_twist_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_wave::analytical::manufactured_elasticity::beams;

const GOLDEN: &str = include_str!("../fixtures/solids_torsion_twist_golden.json");

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
fn torsion_twist_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let t = doc["inputs"]["torque_n_m"].as_f64().unwrap();
    let l = doc["inputs"]["length_m"].as_f64().unwrap();
    let r = doc["inputs"]["radius_m"].as_f64().unwrap();
    let g = doc["inputs"]["shear_modulus_pa"].as_f64().unwrap();
    let j = beams::torsional_constant_solid_shaft(r);
    check(&doc, "torsional_constant", j);
    check(&doc, "twist", beams::torsion_circular_twist(t, l, g, j));
    check(
        &doc,
        "max_shear_stress",
        beams::torsion_circular_max_shear_stress(t, r, j),
    );
}
