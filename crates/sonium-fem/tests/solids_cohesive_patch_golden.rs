//! Wolfram golden replay: cohesive patch openings and dissipation.
//!
//! Replays `fixtures/solids_cohesive_patch_golden.json` (derived by
//! `fixtures/solids_cohesive_patch.wls` at 50-digit precision) through the
//! production cohesive-zone API. Regenerate with:
//! `wolframscript -script fixtures/solids_cohesive_patch.wls > fixtures/solids_cohesive_patch_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::fracture::cohesive::{
    BilinearCohesiveLaw, CohesiveState, cohesive_patch_response, patch_dissipated_energy,
};

const GOLDEN: &str = include_str!("../fixtures/solids_cohesive_patch_golden.json");

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
fn cohesive_patch_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let law = BilinearCohesiveLaw::epoxy();
    check(&doc, "onset_opening", law.onset_opening());
    check(&doc, "failure_opening", law.failure_opening());
    let mut peak_state = CohesiveState::new();
    check(
        &doc,
        "peak_traction",
        law.traction(law.onset_opening(), &mut peak_state),
    );
    // Midpoint dissipation through the production patch driver. The
    // tracked dissipated energy excludes the stored elastic triangle,
    // unlike the raw trapezoid of the loading curve.
    let dmax = 0.5 * (law.onset_opening() + law.failure_opening());
    let mut state = CohesiveState::new();
    let _curve = cohesive_patch_response(2000, dmax, |d| law.traction(d, &mut state));
    check(&doc, "dissipated_at_midpoint", state.dissipated);
    // Full patch dissipates the fracture energy.
    let mut full_state = CohesiveState::new();
    let full = cohesive_patch_response(2000, law.failure_opening(), |d| {
        law.traction(d, &mut full_state)
    });
    check(
        &doc,
        "fracture_energy_check",
        patch_dissipated_energy(&full),
    );
}
