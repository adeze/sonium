//! Wolfram golden replay: central-difference amplification factor.
//!
//! Replays `fixtures/solids_cd_amplification_golden.json` (derived by
//! `fixtures/solids_cd_amplification.wls` at 50-digit precision) through the
//! production `sonium_compute::explicit_dt::cd_amplification` kernel.
//! Regenerate with:
//! `wolframscript -script fixtures/solids_cd_amplification.wls > fixtures/solids_cd_amplification_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_compute::explicit_dt::cd_amplification;

const GOLDEN: &str = include_str!("../fixtures/solids_cd_amplification_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .unwrap_or_else(|| panic!("golden lacks {name} entry"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn check(doc: &Value, name: &str, index: usize, actual: f64) {
    let expected = golden_values(doc, name)[index];
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}[{index}]: production {actual:.17e} vs golden {expected:.17e}"
    );
}

#[test]
fn cd_amplification_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let omegas = doc["inputs"]["omega_dt"]
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()
        .unwrap();
    assert_eq!(omegas.len(), 5);
    for (index, &omega) in omegas.iter().enumerate() {
        let (abs, arg) = cd_amplification(omega);
        check(&doc, "abs_A", index, abs);
        check(&doc, "arg_A", index, arg);
        check(&doc, "period_ratio", index, omega / arg);
    }
}
