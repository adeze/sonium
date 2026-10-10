//! Wolfram golden replay: fatigue S-N closed forms.
//!
//! Replays `fixtures/solids_fatigue_sn_golden.json` (emitted by
//! `fixtures/solids_fatigue_sn.wls`) through the production fatigue
//! API. Regenerate with:
//! `wolframscript -script fixtures/solids_fatigue_sn.wls > fixtures/solids_fatigue_sn_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::solver::vibration_fatigue::{
    BasquinCurve, ForceLifeCurve, RainflowCycle, miner_damage, rainflow_counts,
    spotweld_force_damage,
};

const GOLDEN: &str = include_str!("../fixtures/solids_fatigue_sn_golden.json");

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
fn fatigue_closed_forms_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let input = |name: &str| doc["inputs"][name].as_f64().unwrap();
    let curve = BasquinCurve {
        fatigue_strength: input("basquin_sf"),
        exponent: input("basquin_b"),
        endurance_limit: None,
        ultimate_tensile: Some(input("goodman_uts")),
    };
    curve.validate().unwrap();
    check(&doc, "basquin_n500", curve.cycles_to_failure(500.0, 0.0));
    check(&doc, "basquin_n300", curve.cycles_to_failure(300.0, 0.0));
    check(
        &doc,
        "goodman_n",
        curve.cycles_to_failure(300.0, input("goodman_mean")),
    );
    let blocks = vec![
        RainflowCycle { range: 600.0, mean: 0.0, count: 100.0 },
        RainflowCycle { range: 300.0, mean: 0.0, count: 1000.0 },
    ];
    check(&doc, "miner_two_block", miner_damage(&blocks, &curve));
    let cycles = rainflow_counts(&[0.0, 1.0, -1.0, 1.0, -1.0, 0.0]);
    let full: Vec<_> = cycles.iter().filter(|c| c.count == 1.0).collect();
    assert_eq!(full.len(), 1);
    check(&doc, "rainflow_full_range", full[0].range);
    check(&doc, "rainflow_full_mean", full[0].mean);
    let full_total: f64 = full.iter().map(|c| c.count).sum();
    check(&doc, "rainflow_full_count", full_total);
    let half_total: f64 = cycles.iter().filter(|c| c.count == 0.5).map(|c| c.count).sum();
    check(&doc, "rainflow_half_total", half_total);
    let fn_curve = ForceLifeCurve {
        force_intercept: input("fn_f0"),
        exponent: input("fn_b"),
    };
    fn_curve.validate().unwrap();
    check(&doc, "spotweld_life_250", fn_curve.cycles_to_failure(250.0));
    let history: Vec<f64> = (0..10).map(|i| if i % 2 == 0 { 0.0 } else { 500.0 }).collect();
    check(&doc, "spotweld_damage", spotweld_force_damage(&history, &fn_curve));
    let total: f64 = rainflow_counts(&history).iter().map(|c| c.count).sum();
    check(&doc, "spotweld_cycle_total", total);
}
