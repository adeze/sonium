//! Wolfram golden replay: modal NVH closed forms.
//!
//! Replays `fixtures/solids_nvh_modal_golden.json` (emitted by
//! `fixtures/solids_nvh_modal.wls`) through the production vibration
//! post-processing API. Regenerate with:
//! `wolframscript -script fixtures/solids_nvh_modal.wls > fixtures/solids_nvh_modal_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::solver::vibration_fatigue::{
    miles_rms, modal_receptance, sdof_white_noise_variance,
};

const GOLDEN: &str = include_str!("../fixtures/solids_nvh_modal_golden.json");

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
fn nvh_closed_forms_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let input = |name: &str| doc["inputs"][name].as_f64().unwrap();
    let white_var = sdof_white_noise_variance(
        input("white_s0"),
        input("white_zeta"),
        input("white_omega_n"),
    );
    check(&doc, "white_variance", white_var);
    check(&doc, "white_rms", white_var.sqrt());
    let miles_var = miles_rms(
        input("miles_s0"),
        input("miles_zeta"),
        input("miles_fn_hz"),
    );
    check(&doc, "miles_rms", miles_var);
    check(&doc, "miles_variance", miles_var * miles_var);
    let wn = input("receptance_omega_n");
    let zeta = input("receptance_zeta");
    let h = modal_receptance(&[0.0, wn, 2.0 * wn], &[wn], &[(1.0, 1.0)], &[zeta]).unwrap();
    check(&doc, "receptance_dc_re", h[0].re);
    check(&doc, "receptance_dc_im", h[0].im);
    check(&doc, "receptance_res_re", h[1].re);
    check(&doc, "receptance_res_im", h[1].im);
    check(&doc, "receptance_2x_re", h[2].re);
    check(&doc, "receptance_2x_im", h[2].im);
}
