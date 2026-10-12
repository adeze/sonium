//! Wolfram golden replay: Flanagan-Belytschko Q4 hourglass energy.
//!
//! Replays `fixtures/solids_q4_hourglass_golden.json` (derived by
//! `fixtures/solids_q4_hourglass.wls` in exact arithmetic with 50-digit
//! exports) through the production `sonium_compute::explicit_hourglass`
//! Q4 Gamma and force kernels. Regenerate with:
//! `wolframscript -script fixtures/solids_q4_hourglass.wls > fixtures/solids_q4_hourglass_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_compute::explicit_hourglass::{q4_gamma, q4_hourglass_forces};

const GOLDEN: &str = include_str!("../fixtures/solids_q4_hourglass_golden.json");

fn golden_string(doc: &Value, group: &str, name: &str, index: usize) -> f64 {
    doc[group][name][index]
        .as_str()
        .unwrap_or_else(|| panic!("golden lacks {group}.{name}[{index}]"))
        .parse()
        .unwrap()
}

fn golden_value(doc: &Value, name: &str, index: usize) -> f64 {
    doc["values"][name]["values"][index]
        .as_str()
        .unwrap_or_else(|| panic!("golden lacks values.{name}[{index}]"))
        .parse()
        .unwrap()
}

#[test]
fn q4_hourglass_energy_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    assert_eq!(doc["meta"]["case"], "q4_hourglass");

    // Unit-square coords from the golden (exact 0/1 inputs).
    let coords: Vec<f64> = doc["inputs"]["square_coords"]
        .as_array()
        .expect("golden lacks inputs.square_coords")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(coords.len(), 8);
    let kappa = doc["inputs"]["kappa"].as_f64().unwrap();

    // Gamma vector must match the closed-form mode bit-for-bit.
    let mut gamma = vec![0.0; 4];
    q4_gamma(&coords, &mut gamma);
    for (index, got) in gamma.iter().enumerate() {
        let expected = golden_value(&doc, "gamma", index);
        assert!(
            *got == expected,
            "gamma[{index}]: production {got} vs golden {expected}"
        );
    }

    // Orthogonality residuals Gamma.{1,x,y} are exactly 0.
    let ones = [1.0; 4];
    let mut xc = [0.0; 4];
    let mut yc = [0.0; 4];
    for node in 0..4 {
        xc[node] = coords[2 * node];
        yc[node] = coords[2 * node + 1];
    }
    let fields: [&[f64; 4]; 3] = [&ones, &xc, &yc];
    for (f, field) in fields.iter().enumerate() {
        let mut dot = 0.0;
        for node in 0..4 {
            dot += gamma[node] * field[node];
        }
        let expected = golden_value(&doc, "orthogonality_residuals", f);
        assert!(
            dot == expected,
            "residual field {f}: production {dot} vs golden {expected}"
        );
    }

    // Dissipation P = v.f on the 3 sample velocities.
    let abs: f64 = doc["tolerances"]["dissipation_abs"].as_f64().unwrap();
    let rel: f64 = doc["tolerances"]["dissipation_rel"].as_f64().unwrap();
    for (v, name) in ["vel_rigid", "vel_hg", "vel_mixed"].iter().enumerate() {
        let mut rates = vec![0.0; 8];
        for (index, rate) in rates.iter_mut().enumerate() {
            *rate = golden_string(&doc, "inputs", name, index);
        }
        let mut forces = vec![0.0; 8];
        q4_hourglass_forces(&gamma, &rates, kappa, &mut forces);
        let power: f64 = rates.iter().zip(forces.iter()).map(|(v, f)| v * f).sum();
        let expected = golden_value(&doc, "dissipation", v);
        assert!(power >= 0.0, "{name}: dissipation must be non-negative");
        assert!(
            (power - expected).abs() <= abs + rel * expected.abs(),
            "{name}: production {power:.6e} vs golden {expected:.6e}"
        );
    }
}
