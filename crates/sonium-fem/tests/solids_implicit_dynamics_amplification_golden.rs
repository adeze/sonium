//! Wolfram golden replay: implicit dynamics amplification matrices.
//!
//! Replays `fixtures/solids_implicit_dynamics_amplification_golden.json`
//! (emitted by `fixtures/solids_implicit_dynamics_amplification.wls`)
//! through the production Newmark/HHT coefficient path. Regenerate with:
//! `wolframscript -script fixtures/solids_implicit_dynamics_amplification.wls > fixtures/solids_implicit_dynamics_amplification_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::solver::implicit_dynamics::{amplification_matrix_sdof, spectral_radius_3x3};

const GOLDEN: &str = include_str!("../fixtures/solids_implicit_dynamics_amplification_golden.json");

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
        (actual - expected).abs() <= abs + rel * expected.abs().max(1e-300),
        "{name}: production {actual:.6e} vs golden {expected:.6e}"
    );
}

#[test]
fn amplification_matrices_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let input = |name: &str| doc["inputs"][name].as_f64().unwrap();
    let (m, k, dt) = (input("mass"), input("stiffness"), input("dt"));
    let newmark = amplification_matrix_sdof(
        m,
        input("newmark_damping"),
        k,
        dt,
        input("newmark_beta"),
        input("newmark_gamma"),
        0.0,
    );
    let hht = amplification_matrix_sdof(
        m,
        input("hht_damping"),
        k,
        dt,
        input("hht_beta"),
        input("hht_gamma"),
        input("hht_alpha"),
    );
    for (label, matrix) in [("newmark", newmark), ("hht", hht)] {
        for i in 0..3 {
            for j in 0..3 {
                check(&doc, &format!("{label}_a{}{}", i + 1, j + 1), matrix[i][j]);
            }
        }
    }
    check(&doc, "newmark_rho", spectral_radius_3x3(newmark));
    check(&doc, "hht_rho", spectral_radius_3x3(hht));
}
