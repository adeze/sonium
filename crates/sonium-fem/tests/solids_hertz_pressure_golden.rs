//! Wolfram golden replay: Hertz pressure profile.
//!
//! Replays `fixtures/solids_hertz_pressure_golden.json` (derived by
//! `fixtures/solids_hertz_pressure.wls` at 50-digit precision) through the
//! production `sonium_wave` Hertz closed forms. Regenerate with:
//! `wolframscript -script fixtures/solids_hertz_pressure.wls > fixtures/solids_hertz_pressure_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_wave::analytical::manufactured_elasticity::contact;

const GOLDEN: &str = include_str!("../fixtures/solids_hertz_pressure_golden.json");

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
fn hertz_pressure_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let f = doc["inputs"]["force_n"].as_f64().unwrap();
    let r = doc["inputs"]["radius_m"].as_f64().unwrap();
    let e = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    let estar = contact::hertz_effective_modulus(e, nu, e, nu);
    check(&doc, "effective_modulus", estar);
    let a = contact::hertz_sphere_contact_radius(f, r, estar);
    check(&doc, "contact_radius", a);
    let p0 = contact::hertz_sphere_peak_pressure(f, a);
    check(&doc, "peak_pressure", p0);
    check(
        &doc,
        "pressure_half_radius",
        contact::hertz_sphere_pressure(a / 2.0, p0, a),
    );
    check(&doc, "approach", contact::hertz_sphere_approach(a, r));
}
