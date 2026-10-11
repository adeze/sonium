//! Wolfram golden replay: RBODY rigid kinematics identities.
//!
//! Replays `fixtures/solids_rbody_kinematics_golden.json` (derived by
//! `fixtures/solids_rbody_kinematics.wls` at 50-digit precision) through the
//! production finite-rotation rigid-body API: pair distances stay constant,
//! `det(R) = 1`, the skew-part recovery reproduces the spin, and slave
//! projection matches the translated positions. Regenerate with:
//! `wolframscript -script fixtures/solids_rbody_kinematics.wls > fixtures/solids_rbody_kinematics_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_compute::explicit_state::ExplicitState;
use sonium_fem::physics::solid::joints::{
    FiniteRigidBody, angular_velocity_from_rate, rotation_rate,
};

const GOLDEN: &str = include_str!("../fixtures/solids_rbody_kinematics_golden.json");
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .unwrap_or_else(|| panic!("golden entry of {name} is not a string"))
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
    for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= abs + rel * want.abs(),
            "{name}[{index}]: production {got:.6e} vs golden {want:.6e}"
        );
    }
}

fn input_vector(doc: &Value, name: &str) -> [f64; 3] {
    let entries = doc["inputs"][name].as_array().unwrap();
    [0, 1, 2].map(|i| entries[i].as_f64().unwrap())
}

#[test]
fn rbody_kinematics_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["case"], "rbody_kinematics");

    // Golden self-check: rigid motion preserves every pair distance, so the
    // three golden distance vectors must agree exactly.
    let golden_t0 = golden_values(&doc, "pair_distances_t0");
    let golden_t05 = golden_values(&doc, "pair_distances_t05");
    let golden_t1 = golden_values(&doc, "pair_distances_t1");
    assert_eq!(golden_t0, golden_t05);
    assert_eq!(golden_t0, golden_t1);

    let nodes: Vec<[f64; 3]> = doc["inputs"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let row = row.as_array().unwrap();
            [0, 1, 2].map(|i| row[i].as_f64().unwrap())
        })
        .collect();
    assert_eq!(nodes.len(), 4);
    let axis = input_vector(&doc, "rotation_axis");
    let theta_rate = doc["inputs"]["theta_rate_rad_s"].as_f64().unwrap();
    let translation_rate = input_vector(&doc, "translation_rate_m_s");
    let times: Vec<f64> = doc["inputs"]["times_s"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_f64().unwrap())
        .collect();
    let suffixes = ["t0", "t05", "t1"];
    assert_eq!(times.len(), suffixes.len());
    let omega = [
        axis[0] * theta_rate,
        axis[1] * theta_rate,
        axis[2] * theta_rate,
    ];

    for (time, suffix) in times.iter().zip(suffixes) {
        let mut body =
            FiniteRigidBody::new(0, vec![1, 2, 3], nodes.clone()).with_angular_velocity(omega);
        body.set_rotation_from_axis_angle(axis, theta_rate * time);
        let mut state = ExplicitState::new(4);
        state.masses = vec![1.0; 4];
        state.positions[0..3].copy_from_slice(&[
            translation_rate[0] * time,
            translation_rate[1] * time,
            translation_rate[2] * time,
        ]);
        state.velocities[0..3].copy_from_slice(&translation_rate);
        body.apply_to_state(&mut state);

        let distances: Vec<f64> = PAIRS
            .iter()
            .map(|&(a, b)| {
                let dx = state.positions[3 * a] - state.positions[3 * b];
                let dy = state.positions[3 * a + 1] - state.positions[3 * b + 1];
                let dz = state.positions[3 * a + 2] - state.positions[3 * b + 2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        check_all(&doc, &format!("pair_distances_{suffix}"), &distances);
        check_all(
            &doc,
            &format!("det_rotation_{suffix}"),
            &[body.determinant()],
        );

        // Skew-part recovery through the production API.
        let rate = rotation_rate(body.rotation(), omega);
        let recovered = angular_velocity_from_rate(body.rotation(), &rate);
        check_all(&doc, &format!("omega_{suffix}"), &recovered);

        // Full slave projection including the rigid translation.
        if suffix == "t0" {
            for (node, reference) in nodes.iter().enumerate() {
                for (offset, &want) in reference.iter().enumerate() {
                    let got = state.positions[3 * node + offset];
                    assert!(
                        (got - want).abs() < 1.0e-14,
                        "t0 position drift at node {node}"
                    );
                }
            }
        } else {
            check_all(
                &doc,
                &format!("positions_{suffix}"),
                &state.positions.clone(),
            );
        }
    }
}
