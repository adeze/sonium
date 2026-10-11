//! Wolfram golden replay: Bao-Xue-Wierzbicki locus.
//!
//! Replays `fixtures/solids_wierzbicki_locus_golden.json` (derived by
//! `fixtures/solids_wierzbicki_locus.wls` at 50-digit precision) through
//! the production `/FAIL/WIERZBICKI` API. Regenerate with:
//! `wolframscript -script fixtures/solids_wierzbicki_locus.wls > fixtures/solids_wierzbicki_locus_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::fracture::wierzbicki::{Wierzbicki, xi_shell};

const GOLDEN: &str = include_str!("../fixtures/solids_wierzbicki_locus_golden.json");

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
fn wierzbicki_locus_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let locus = Wierzbicki::deck_shell();
    let uni = 1.0 / 3.0;
    check(&doc, "emax_uniaxial", locus.emax(uni));
    check(&doc, "emin_uniaxial", locus.emin(uni));
    check(&doc, "xi_uniaxial", xi_shell(uni));
    check(
        &doc,
        "ef_uniaxial",
        locus.fracture_strain(uni, xi_shell(uni)),
    );
    check(&doc, "ef_shear", locus.fracture_strain(0.0, 0.0));
    check(&doc, "xi_mid", xi_shell(0.5));
    check(&doc, "ef_mid", locus.fracture_strain(0.5, xi_shell(0.5)));
    check(&doc, "xi_biaxial", xi_shell(2.0 / 3.0));
}
