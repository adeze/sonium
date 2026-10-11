//! Wolfram golden replay: damped central-difference stability chart.
//!
//! Replays `fixtures/solids_cd_stability_chart_golden.json` (derived by
//! `fixtures/solids_cd_stability_chart.wls` at 50-digit precision) through the
//! production `sonium_solvers::time_integration::damped_stability_margin`
//! helper: `|A_max|` and the margin on the 5x3 `(omega_dt, xi)` grid plus the
//! boundary row. Regenerate with:
//! `wolframscript -script fixtures/solids_cd_stability_chart.wls > fixtures/solids_cd_stability_chart_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_solvers::time_integration::damped_stability_margin;

const GOLDEN: &str = include_str!("../fixtures/solids_cd_stability_chart_golden.json");

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
        (actual - expected).abs() <= abs + rel * expected.abs().max(1e-300),
        "{name}[{index}]: production {actual:.17e} vs golden {expected:.17e}"
    );
}

#[test]
fn cd_stability_chart_matches_wolfram() {
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
    let xis = doc["inputs"]["xi"]
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()
        .unwrap();
    assert_eq!(omegas.len(), 5);
    assert_eq!(xis.len(), 3);

    // Grid replay, row-major over {omega, xi} (index = i_omega*3 + i_xi).
    for (i_omega, &omega) in omegas.iter().enumerate() {
        for (i_xi, &xi) in xis.iter().enumerate() {
            let index = i_omega * xis.len() + i_xi;
            let margin = damped_stability_margin(omega, xi);
            check(&doc, "margin", index, margin);
            check(&doc, "abs_Amax", index, 1.0 - margin);
        }
    }

    // Boundary row: the exact crossing sits on |A_max| = 1.
    let crits = golden_values(&doc, "boundary_omega_crit");
    assert_eq!(crits.len(), 3);
    for (i_xi, &xi) in xis.iter().enumerate() {
        let margin = damped_stability_margin(crits[i_xi], xi);
        check(&doc, "boundary_abs_Amax", i_xi, 1.0 - margin);
        assert!(
            margin.abs() < 1e-9,
            "boundary margin at xi={xi} should vanish, got {margin:.3e}"
        );
    }
}
