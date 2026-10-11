//! Wolfram golden replay: erosion energy accounting.
//!
//! Replays `fixtures/solids_erosion_energy_golden.json` (derived by
//! `fixtures/solids_erosion_energy.wls` at 50-digit precision) through
//! the production deletion-policy and ledger API. Regenerate with:
//! `wolframscript -script fixtures/solids_erosion_energy.wls > fixtures/solids_erosion_energy_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::fracture::deletion::{DeletionPolicy, ErosionLedger};

const GOLDEN: &str = include_str!("../fixtures/solids_erosion_energy_golden.json");

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
fn erosion_energy_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let mut ledger = ErosionLedger::new(3);
    ledger.record(0, 2.0e-6, 1.5e8, 1.6e8, false);
    ledger.record(1, 1.0e-6, 3.0e8, 3.2e8, true);
    ledger.record(2, 5.0e-7, 2.0e8, 2.1e8, false);
    let totals = ledger.totals();
    check(&doc, "active_plastic", totals.active_plastic);
    check(&doc, "eroded_plastic", totals.eroded_plastic);
    check(&doc, "active_internal", totals.active_internal);
    check(&doc, "eroded_internal", totals.eroded_internal);
    let policy = DeletionPolicy::shell_layers();
    let thick = [0.2, 0.2, 0.2, 0.2, 0.2];
    check(
        &doc,
        "verdict_two_layers",
        f64::from(policy.shell_deleted(&[true, true, false, false, false], Some(&thick))),
    );
    check(
        &doc,
        "verdict_three_layers",
        f64::from(policy.shell_deleted(&[true, true, true, false, false], Some(&thick))),
    );
    check(
        &doc,
        "verdict_split_layers",
        f64::from(policy.shell_deleted(&[true, true, false, true, true], Some(&thick))),
    );
}
