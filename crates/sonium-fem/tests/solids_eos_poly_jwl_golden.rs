//! Wolfram golden replay: polynomial and JWL equations of state.
//!
//! Replays `fixtures/eos_poly_jwl_golden.json` (derived by
//! `fixtures/eos_poly_jwl.wls` at 50-digit precision) through the
//! production EOS API. Regenerate with:
//! `wolframscript -script fixtures/eos_poly_jwl.wls > fixtures/eos_poly_jwl_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::eos::{JwlEos, PolynomialEos};

const GOLDEN: &str = include_str!("../fixtures/eos_poly_jwl_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("golden lacks {name}"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn check_all(doc: &Value, name: &str, actual: &[f64]) {
    let expected = golden_values(doc, name);
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= abs + rel * want.abs(),
            "{name}[{i}]: production {got:.6e} vs golden {want:.6e}"
        );
    }
}

#[test]
fn eos_polynomial_and_jwl_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let steel = PolynomialEos::steel();
    check_all(&doc, "poly_compression", &[steel.pressure(0.1, 0.0)]);
    check_all(&doc, "poly_expansion", &[steel.pressure(-0.05, 1.0e9)]);
    check_all(&doc, "poly_hot", &[steel.pressure(0.2, 5.0e9)]);
    let tnt = JwlEos::tnt();
    check_all(
        &doc,
        "jwl_unit_volume",
        &[tnt.pressure_jwl(1.0, tnt.e0, 0.0)],
    );
    check_all(&doc, "jwl_expanded", &[tnt.pressure_jwl(2.0, tnt.e0, 0.0)]);
    check_all(&doc, "burn_half_volume", &[tnt.burn_fraction_volume(0.5)]);
}
