//! Wolfram golden replay: 1-D spring and rivet.
//!
//! Replays `fixtures/solids_spring_rivet_1d_golden.json` (derived by
//! `fixtures/solids_spring_rivet_1d.wls` at 50-digit precision) through
//! the production spring/rivet API. Regenerate with:
//! `wolframscript -script fixtures/solids_spring_rivet_1d.wls > fixtures/solids_spring_rivet_1d_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::beam::{
    Rivet, RivetState, Spring1D, Spring1DState, SpringRupture, spring_critical_step,
};
use sonium_fem::physics::solid::plasticity::funct::Funct1D;

const GOLDEN: &str = include_str!("../fixtures/solids_spring_rivet_1d_golden.json");

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

fn check_all(doc: &Value, name: &str, actual: &[f64]) {
    let expected = golden_values(doc, name);
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= abs + rel * want.abs(),
            "{name}[{i}]: production {got:.6e} vs golden {want:.6e}"
        );
    }
}

fn check(doc: &Value, name: &str, actual: f64) {
    check_all(doc, name, &[actual]);
}

#[test]
fn spring_rivet_1d_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");

    let linear = Spring1D::linear(2000.0, 5.0, 1.0);
    let state = Spring1DState::new();
    check(&doc, "linear_force", linear.axial_force(&state, 0.05, 2.0));

    let curve = Funct1D::new(vec![0.0, 0.1], vec![0.0, 50.0]).unwrap();
    let nonlinear = Spring1D::linear(1000.0, 0.0, 0.0).with_curve(curve);
    check(
        &doc,
        "nonlinear_force",
        nonlinear.axial_force(&state, 0.05, 0.0),
    );
    check(
        &doc,
        "nonlinear_tangent",
        nonlinear.axial_tangent(&state, 0.05),
    );
    check(
        &doc,
        "clamped_force",
        nonlinear.axial_force(&state, 1.0, 0.0),
    );

    let diagonal = Spring1D::linear(1000.0, 0.0, 0.0);
    let out = diagonal.nodal_forces(
        &state,
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        4.0,
    );
    check_all(&doc, "axis_force", &out.force2);

    let ramp =
        Spring1D::linear(1000.0, 0.0, 0.0).with_rupture(SpringRupture::new(1.0e9, 1.0e9, 1.0, 1.0));
    let mut ramp_state = Spring1DState::new();
    ramp.commit(&mut ramp_state, 0.0, 100.0, 0.0, 0.1);
    check(&doc, "ramp_energy", ramp_state.energy);

    check(
        &doc,
        "critical_step",
        spring_critical_step(2.0, 2.0, 1000.0),
    );

    let rivet = Rivet::new(1.0e6, 1.0e3, 5000.0, 2000.0, 50.0);
    let rivet_state = RivetState::new();
    let shear = rivet.forces(&rivet_state, [1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0; 6]);
    check(&doc, "rivet_shear", shear.utilization);
    let combined = rivet.forces(&rivet_state, [1.0e-3, 0.0, 0.0, 0.03, 0.0, 0.0], [0.0; 6]);
    check(&doc, "rivet_combined", combined.utilization);
    let axial = rivet.forces(&rivet_state, [0.0, 0.0, 6.0e-3, 0.0, 0.0, 0.0], [0.0; 6]);
    check(&doc, "rivet_axial", axial.utilization);
}
