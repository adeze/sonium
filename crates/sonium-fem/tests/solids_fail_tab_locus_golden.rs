//! Wolfram golden replay: tabulated failure locus.
//!
//! Replays `fixtures/solids_fail_tab_locus_golden.json` (derived by
//! `fixtures/solids_fail_tab_locus.wls` at 50-digit precision) through
//! the production `/FAIL/TAB1` API. Regenerate with:
//! `wolframscript -script fixtures/solids_fail_tab_locus.wls > fixtures/solids_fail_tab_locus_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::fracture::fail_tab::FailTab;
use sonium_fem::physics::solid::plasticity::funct::Funct1D;

const GOLDEN: &str = include_str!("../fixtures/solids_fail_tab_locus_golden.json");

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

/// Builds the golden grid: base ordinates with a half-scale slow row.
fn golden_table() -> FailTab {
    let triax = vec![-1.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0];
    let base = vec![3.0, 5.0 / 6.0, 0.3, 0.36, 0.05];
    let fast = Funct1D::new(triax.clone(), base.clone()).unwrap();
    let slow = Funct1D::new(triax, base).unwrap().scaled_ordinates(0.5);
    let mut tab = FailTab::new(vec![1.0e-5, 0.5], vec![0.0], vec![slow, fast]).unwrap();
    tab.dcrit = 1.1;
    tab.dp = 2.0;
    tab.exponent = 2.0;
    tab
}

#[test]
fn fail_tab_locus_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let tab = golden_table();
    check(&doc, "ef_node", tab.table_strain(1.0 / 3.0, 0.5, 0.0));
    check(&doc, "ef_mid_triax", tab.table_strain(1.0 / 6.0, 0.5, 0.0));
    check(
        &doc,
        "ef_blend_rate",
        tab.table_strain(1.0 / 3.0, 0.25, 0.0),
    );
    check(
        &doc,
        "ef_extrapolate",
        tab.table_strain(1.0 / 3.0, 1.0, 0.0),
    );
    check(
        &doc,
        "damage_959",
        tab.increment_at_rate(0.01, 0.0, 1.0 / 3.0, 0.0, 0.5, None, None),
    );
    check(&doc, "rupture_flag", f64::from(tab.ruptured(1.15)));
}
