//! Wolfram golden replay: Hertz impact approach, force, and time.
//!
//! Replays `fixtures/solids_hertz_impact_golden.json` (derived by
//! `fixtures/solids_hertz_impact.wls` at 50-digit precision) through the
//! production explicit-contact impact API. Regenerate with:
//! `wolframscript -script fixtures/solids_hertz_impact.wls > fixtures/solids_hertz_impact_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::contact::explicit::{
    hertz_impact_contact_time, hertz_impact_max_approach, hertz_impact_peak_force,
};

const GOLDEN: &str = include_str!("../fixtures/solids_hertz_impact_golden.json");

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
fn hertz_impact_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    check(&doc, "max_approach", hertz_impact_max_approach(2.0, 3.0, 0.05, 110.0e9));
    check(&doc, "peak_force", hertz_impact_peak_force(2.0, 3.0, 0.05, 110.0e9));
    check(&doc, "contact_time", hertz_impact_contact_time(2.0, 3.0, 0.05, 110.0e9));
}
