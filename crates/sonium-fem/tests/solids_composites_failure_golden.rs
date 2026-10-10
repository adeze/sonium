//! Wolfram golden replay: Hashin and Puck criterion indices.
//!
//! Replays `fixtures/solids_composites_failure_golden.json` (emitted by
//! `fixtures/solids_composites_failure.wls`, which implements both criteria
//! independently with a 0.05-degree Puck reference scan) through the
//! production composites failure API. Regenerate with:
//! `wolframscript -script fixtures/solids_composites_failure.wls > fixtures/solids_composites_failure_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::composites::{
    FailureCriterion, PlyStrengths, PuckParams, assess_ply_failure,
};

const GOLDEN: &str = include_str!("../fixtures/solids_composites_failure_golden.json");

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

fn check(doc: &Value, name: &str, actual: &[f64]) {
    let expected = golden_values(doc, name);
    assert_eq!(actual.len(), expected.len(), "{name}: arity mismatch");
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= abs + rel * e.abs().max(1e-30),
            "{name}[{i}]: production {a:.6e} vs golden {e:.6e}"
        );
    }
}

fn strengths(doc: &Value) -> PlyStrengths {
    PlyStrengths {
        xt: doc["inputs"]["xt_pa"].as_f64().unwrap(),
        xc: doc["inputs"]["xc_pa"].as_f64().unwrap(),
        yt: doc["inputs"]["yt_pa"].as_f64().unwrap(),
        yc: doc["inputs"]["yc_pa"].as_f64().unwrap(),
        s12: doc["inputs"]["s12_pa"].as_f64().unwrap(),
        s23: doc["inputs"]["s23_pa"].as_f64().unwrap(),
    }
}

fn puck_params(doc: &Value) -> PuckParams {
    PuckParams {
        p_t_par: doc["inputs"]["puck_p_t_par"].as_f64().unwrap(),
        p_c_par: doc["inputs"]["puck_p_c_par"].as_f64().unwrap(),
        p_t_perp: doc["inputs"]["puck_p_t_perp"].as_f64().unwrap(),
        p_c_perp: doc["inputs"]["puck_p_c_perp"].as_f64().unwrap(),
    }
}

const STATES: [(&str, [f64; 3]); 8] = [
    ("fiber_tension", [1.0e9, 0.0, 0.0]),
    ("fiber_compression", [-1.0e9, 0.0, 0.0]),
    ("matrix_tension", [0.0, 3.0e7, 0.0]),
    ("matrix_compression", [0.0, -1.2e8, 0.0]),
    ("pure_shear", [0.0, 0.0, 5.0e7]),
    ("biaxial_tension_shear", [8.0e8, 2.0e7, 3.0e7]),
    ("transverse_compression_shear", [0.0, -1.0e8, 4.0e7]),
    ("fiber_compression_shear", [-8.0e8, -5.0e7, 3.0e7]),
];

#[test]
fn composites_failure_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let strengths = strengths(&doc);
    let puck = puck_params(&doc);
    for (name, stress) in STATES {
        let hashin = assess_ply_failure(stress, &strengths, FailureCriterion::Hashin);
        let detail = match hashin.detail {
            sonium_fem::physics::solid::composites::FailureDetail::Hashin(e) => e,
            _ => panic!("Hashin detail mismatch"),
        };
        check(
            &doc,
            &format!("{name}_hashin"),
            &[
                detail.fiber_tension,
                detail.fiber_compression,
                detail.matrix_tension,
                detail.matrix_compression,
            ],
        );
        let pucked = assess_ply_failure(stress, &strengths, FailureCriterion::Puck(puck));
        let detail = match pucked.detail {
            sonium_fem::physics::solid::composites::FailureDetail::Puck(e) => e,
            _ => panic!("Puck detail mismatch"),
        };
        check(&doc, &format!("{name}_puck_fiber"), &[detail.fiber]);
        check(&doc, &format!("{name}_puck_iff"), &[detail.iff]);
        check(
            &doc,
            &format!("{name}_puck_angle"),
            &[detail.fracture_angle_deg],
        );
    }
}
