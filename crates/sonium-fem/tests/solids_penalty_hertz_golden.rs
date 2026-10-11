//! Wolfram golden replay: penalized Hertz normal contact.
//!
//! Replays `fixtures/solids_penalty_hertz_golden.json` (derived by
//! `fixtures/solids_penalty_hertz.wls` at 50-digit precision) through the
//! production penalty-contact API. Regenerate with:
//! `wolframscript -script fixtures/solids_penalty_hertz.wls > fixtures/solids_penalty_hertz_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::contact::explicit::{hertz_effective_modulus, penalized_hertz};

const GOLDEN: &str = include_str!("../fixtures/solids_penalty_hertz_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .unwrap_or_else(|| panic!("golden lacks {name}"))
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
        "{name}[{index}]: production {actual:.6e} vs golden {expected:.6e}"
    );
}

#[test]
fn penalty_hertz_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let f = doc["inputs"]["force_n"].as_f64().unwrap();
    let r = doc["inputs"]["radius_m"].as_f64().unwrap();
    let e = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    // Rigid plane is the infinite-modulus limit of the two-body formula.
    let estar = hertz_effective_modulus(e, nu, f64::INFINITY, 0.0);
    check(&doc, "effective_modulus", 0, estar);
    let penalties = doc["inputs"]["penalty_stiffness_n_per_m"]
        .as_array()
        .unwrap();
    assert_eq!(penalties.len(), 3);
    for (index, penalty) in penalties.iter().enumerate() {
        let kp = penalty.as_f64().unwrap();
        let response = penalized_hertz(f, r, estar, kp);
        if index == 0 {
            check(&doc, "contact_radius", 0, response.contact_radius);
            check(&doc, "hertz_approach", 0, response.hertz_approach);
            check(&doc, "hertz_stiffness", 0, response.hertz_stiffness);
        }
        check(&doc, "approach_total", index, response.total_approach);
        check(
            &doc,
            "effective_stiffness",
            index,
            response.effective_stiffness,
        );
    }
}
