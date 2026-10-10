//! Wolfram-derived golden for hard-wall annular-duct radial roots.
//!
//! `fixtures/annular_duct.wls` (Wolfram 15.0.0) brackets sign changes
//! of J_m'(s x)Y_m'(x) - J_m'(x)Y_m'(s x) and refines them with
//! FindRoot at 55-digit working precision for hub ratio 0.4. This test
//! replays the checked-in `annular_duct_golden.json` through the
//! production `annular_duct_radial_root` path — no live engine needed
//! per commit. Regenerate with:
//! `wolframscript -script fixtures/annular_duct.wls > fixtures/annular_duct_golden.json`.

use serde::Deserialize;
use sonium_wave::analytical::annular_duct_radial_root;
use std::collections::HashMap;

const GOLDEN: &str = include_str!("../fixtures/annular_duct_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    hub_tip_ratio: f64,
    roots: HashMap<String, f64>,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    root_abs: f64,
}

#[test]
fn annular_duct_roots_match_wolfram() {
    let golden: Golden = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "wave.annular-duct.v1");
    for (key, azimuthal, radial_mode) in [
        ("m0_rank1", 0, 1),
        ("m0_rank2", 0, 2),
        ("m1_rank0", 1, 0),
        ("m1_rank1", 1, 1),
        ("m3_rank0", 3, 0),
    ] {
        let expected = golden.roots[key];
        let actual =
            annular_duct_radial_root(azimuthal, radial_mode, golden.hub_tip_ratio).unwrap();
        assert!(
            (actual - expected).abs() < golden.tolerances.root_abs,
            "{key}: got {actual:.16}, want {expected:.16}"
        );
    }
}
