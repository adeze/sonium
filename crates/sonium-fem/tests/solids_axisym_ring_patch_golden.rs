//! Wolfram golden replay: axisymmetric QUAD ring patch.
//!
//! Replays `fixtures/axisym_ring_patch_golden.json` (derived by
//! `fixtures/axisym_ring_patch.wls` at 50-digit precision) through
//! the production QUAD ring API. Regenerate with:
//! `wolframscript -script fixtures/axisym_ring_patch.wls > fixtures/axisym_ring_patch_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::axisym_explicit::{
    AxisymExplicitParams, QuadRingGeom, QuadRingState, quad_ring_forces,
};
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};

const GOLDEN: &str = include_str!("../fixtures/axisym_ring_patch_golden.json");

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
fn ring_patch_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let geom = QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]).unwrap();
    let b_r: Vec<f64> = geom.grad.iter().map(|g| g[0]).collect();
    let b_z: Vec<f64> = geom.grad.iter().map(|g| g[1]).collect();
    check_all(&doc, "b_r_row", &b_r);
    check_all(&doc, "b_z_row", &b_z);
    check_all(&doc, "gamma", &geom.gamma);
    check(&doc, "area", geom.area);
    check(&doc, "radius_bar", geom.radius_bar);
    check(&doc, "volume", geom.volume);
    check(&doc, "nodal_mass", geom.lumped_masses(7800.0)[0]);
    check(&doc, "char_length", geom.char_length);
    check(&doc, "critical_step", geom.critical_step(4500.0));

    // Uniaxial-strain elastic step at 1/s over dt = 1e-3.
    let material = J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect);
    let params = AxisymExplicitParams::new(7800.0, 4500.0);
    let mut state = QuadRingState::new(PlasticState::new_3d());
    let velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let out = quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
    assert!(!out.yielding);
    assert_eq!(state.point.equiv_plastic_strain, 0.0);
    check(&doc, "elastic_srr", state.point.stress[0]);
    check(&doc, "elastic_szz", state.point.stress[1]);
    check(&doc, "elastic_sth", state.point.stress[2]);
    let radial: f64 = (0..4).map(|n| out.forces[n * 2]).sum();
    check(&doc, "radial_sum", radial);
    check(&doc, "bulk_q", out.bulk_q);

    // Uniform engineering shear at 1/s over dt = 1e-3.
    let mut state = QuadRingState::new(PlasticState::new_3d());
    let shear = [0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0];
    let out = quad_ring_forces(&geom, &mut state, &material, &shear, 1.0e-3, &params).unwrap();
    assert!(!out.yielding);
    check(&doc, "shear_srz", state.point.stress[5]);
}
