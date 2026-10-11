//! Wolfram golden replay: mass-scaling frequency shift.
//!
//! Replays `fixtures/solids_mass_scaling_shift_golden.json` (derived by
//! `fixtures/solids_mass_scaling_shift.wls` at 50-digit precision) through the
//! production `sonium_compute::explicit_dt` kernels (`mass_scale_uniform`,
//! `critical_dt_modal`, `selective_added_mass`). Regenerate with:
//! `wolframscript -script fixtures/solids_mass_scaling_shift.wls > fixtures/solids_mass_scaling_shift_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_compute::explicit_dt::{critical_dt_modal, mass_scale_uniform, selective_added_mass};

const GOLDEN: &str = include_str!("../fixtures/solids_mass_scaling_shift_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .unwrap_or_else(|| panic!("golden lacks {name} entry"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn input_array(doc: &Value, name: &str) -> Vec<f64> {
    doc["inputs"][name]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks input {name}"))
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()
        .unwrap()
}

fn check(doc: &Value, name: &str, index: usize, actual: f64) {
    let expected = golden_values(doc, name)[index];
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}[{index}]: production {actual:.17e} vs golden {expected:.17e}"
    );
}

#[test]
fn mass_scaling_shift_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let mass = doc["inputs"]["mass_kg"].as_f64().unwrap();
    let stiff = doc["inputs"]["stiffness_n_m"].as_f64().unwrap();
    let scales = input_array(&doc, "scales");
    let vel = input_array(&doc, "rigid_velocity_m_s");
    assert_eq!(scales.len(), 3);
    assert_eq!(vel.len(), 2);

    // Closed form of the free-free 2-DOF chain, pinned by the golden.
    let omega_elastic = (2.0 * stiff / mass).sqrt();
    check(&doc, "omega_elastic_unscaled", 0, omega_elastic);
    let dt_unscaled = critical_dt_modal(omega_elastic);
    let ke_unscaled = 0.5 * mass * (vel[0] * vel[0] + vel[1] * vel[1]);
    check(&doc, "ke_rigid_unscaled", 0, ke_unscaled);

    for (index, &scale) in scales.iter().enumerate() {
        let mut masses = vec![mass, mass];
        mass_scale_uniform(&mut masses, scale);
        // Effective scale read back from the production-scaled masses.
        let effective = (masses[0] / mass).sqrt();

        // Rigid mode stays at zero frequency under uniform scaling.
        check(&doc, "omega_rigid", 0, 0.0 / effective);

        let omega_scaled = omega_elastic / effective;
        check(&doc, "omega_elastic_scaled", index, omega_scaled);
        let dt_scaled = critical_dt_modal(omega_scaled);
        check(&doc, "dt_crit_scaled", index, dt_scaled);

        // Selective hook reproduces the uniform added mass when the
        // target is the scaled step (unit sound speed folds dt into L_e).
        let added = selective_added_mass(dt_unscaled, 1.0, mass, dt_scaled);
        check(&doc, "added_mass_per_node", index, added);

        // Rigid-mode kinetic energy follows the s² scaling law exactly.
        let ke_scaled = 0.5 * masses[0] * vel[0] * vel[0] + 0.5 * masses[1] * vel[1] * vel[1];
        check(&doc, "ke_rigid_scaled", index, ke_scaled);
        check(
            &doc,
            "ke_scaling_error",
            index,
            ke_scaled - effective * effective * ke_unscaled,
        );
    }
}
