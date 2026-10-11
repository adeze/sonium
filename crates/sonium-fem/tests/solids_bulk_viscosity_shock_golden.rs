//! Wolfram golden replay: bulk-viscosity shock identities.
//!
//! Replays `fixtures/solids_bulk_viscosity_shock_golden.json` (derived by
//! `fixtures/solids_bulk_viscosity_shock.wls` in exact arithmetic with
//! 50-digit exports) through the production
//! `sonium_compute::explicit_hourglass::bulk_viscosity_q` kernel. Regenerate
//! with: `wolframscript -script fixtures/solids_bulk_viscosity_shock.wls > fixtures/solids_bulk_viscosity_shock_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_compute::explicit_hourglass::bulk_viscosity_q;
use std::f64::consts::PI;

const GOLDEN: &str = include_str!("../fixtures/solids_bulk_viscosity_shock_golden.json");

fn golden_value(doc: &Value, name: &str, index: usize) -> f64 {
    doc["values"][name]["values"][index]
        .as_str()
        .unwrap_or_else(|| panic!("golden lacks values.{name}[{index}]"))
        .parse()
        .unwrap()
}

fn check(doc: &Value, name: &str, index: usize, actual: f64) {
    let expected = golden_value(doc, name, index);
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}[{index}]: production {actual:.6e} vs golden {expected:.6e}"
    );
}

#[test]
fn bulk_viscosity_shock_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    assert_eq!(doc["meta"]["case"], "bulk_viscosity_shock");

    let rho = doc["inputs"]["density"].as_f64().unwrap();
    let c = doc["inputs"]["sound_speed"].as_f64().unwrap();
    let s = doc["inputs"]["hugoniot_slope"].as_f64().unwrap();
    let up = doc["inputs"]["piston_velocity"].as_f64().unwrap();
    let len = doc["inputs"]["element_length"].as_f64().unwrap();
    let c0 = doc["inputs"]["c0_quadratic"].as_f64().unwrap();
    let c1 = doc["inputs"]["c1_linear"].as_f64().unwrap();

    // Rankine-Hugoniot jump for the linear Us-Up Hugoniot.
    let us = c + s * up;
    check(&doc, "shock_speed", 0, us);
    check(&doc, "pressure_jump", 0, rho * us * up);

    // Viscosity samples through the production kernel.
    let rates = doc["inputs"]["sample_rates"].as_array().unwrap();
    assert_eq!(rates.len(), 5);
    for (index, rate) in rates.iter().enumerate() {
        let d = rate.as_f64().unwrap();
        check(
            &doc,
            "q_samples",
            index,
            bulk_viscosity_q(rho, c, len, c0, c1, d),
        );
    }

    // Shock-center self-consistency: q(-zc) equals the Hugoniot gap for
    // every c0, and the width law L = Pi*l*Sqrt[c0/s] holds.
    let c0_set = doc["inputs"]["c0_set"].as_array().unwrap();
    assert_eq!(c0_set.len(), 3);
    for (index, c0v) in c0_set.iter().enumerate() {
        let c0v = c0v.as_f64().unwrap();
        let zc = golden_value(&doc, "center_strain_rates", index);
        assert!(zc > 0.0, "center strain rate must be positive");
        check(
            &doc,
            "center_q",
            0,
            bulk_viscosity_q(rho, c, len, c0v, 0.0, -zc),
        );
        check(&doc, "shock_widths", index, PI * len * (c0v / s).sqrt());
    }

    // Shock ends carry no viscous pressure for any coefficients, so the
    // Rankine-Hugoniot jump is preserved independent of c0/c1.
    let end_c0 = doc["inputs"]["end_c0"].as_array().unwrap();
    let end_c1 = doc["inputs"]["end_c1"].as_array().unwrap();
    assert_eq!(end_c0.len(), 2);
    assert_eq!(end_c1.len(), 2);
    for index in 0..2 {
        let q = bulk_viscosity_q(
            rho,
            c,
            len,
            end_c0[index].as_f64().unwrap(),
            end_c1[index].as_f64().unwrap(),
            0.0,
        );
        check(&doc, "shock_end_q", index, q);
    }
}
