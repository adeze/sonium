//! Wolfram golden replay: Timoshenko cantilever tip.
//!
//! Replays `fixtures/solids_timoshenko_cantilever_tip_golden.json` (derived
//! by `fixtures/solids_timoshenko_cantilever_tip.wls` at 50-digit precision)
//! through the production `sonium_wave` beam closed forms. Regenerate with:
//! `wolframscript -script fixtures/solids_timoshenko_cantilever_tip.wls > fixtures/solids_timoshenko_cantilever_tip_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_wave::analytical::manufactured_elasticity::beams;

const GOLDEN: &str = include_str!("../fixtures/solids_timoshenko_cantilever_tip_golden.json");

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
fn timoshenko_cantilever_tip_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let f = doc["inputs"]["load_n"].as_f64().unwrap();
    let l = doc["inputs"]["length_m"].as_f64().unwrap();
    let e = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let i = doc["inputs"]["second_moment_m4"].as_f64().unwrap();
    let g = doc["inputs"]["shear_modulus_pa"].as_f64().unwrap();
    let a = doc["inputs"]["area_m2"].as_f64().unwrap();
    let kappa = doc["inputs"]["shear_factor"].as_f64().unwrap();
    let bending = beams::euler_bernoulli_cantilever_tip_point(f, l, e, i);
    let tip = beams::timoshenko_cantilever_tip_point(f, l, e, i, g, a, kappa);
    check(&doc, "tip_deflection", tip);
    check(&doc, "bending_part", bending);
    check(&doc, "shear_part", tip - bending);
    check(
        &doc,
        "tip_rotation",
        beams::timoshenko_cantilever_rotation_point(f, l, e, i),
    );
}
