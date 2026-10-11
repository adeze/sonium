//! Wolfram golden replay: axisymmetric Taylor-analog bar.
//!
//! Replays `fixtures/axisym_taylor_golden.json` (derived by
//! `fixtures/axisym_taylor.wls` at 50-digit precision) through the
//! production QUAD ring API: a single ring spanning a copper bar
//! cross-section with an edge on the axis, driven at impact rate.
//! Regenerate with:
//! `wolframscript -script fixtures/axisym_taylor.wls > fixtures/axisym_taylor_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::axisym_explicit::{
    AxisymExplicitParams, QuadRingGeom, QuadRingState, quad_ring_forces,
};
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};

const GOLDEN: &str = include_str!("../fixtures/axisym_taylor_golden.json");
const RADIUS: f64 = 0.003175;
const LENGTH: f64 = 0.0254;
const VELOCITY: f64 = 227.0;

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
fn taylor_bar_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let geom =
        QuadRingGeom::new([[0.0, 0.0], [RADIUS, 0.0], [RADIUS, LENGTH], [0.0, LENGTH]]).unwrap();
    let b_r: Vec<f64> = geom.grad.iter().map(|g| g[0]).collect();
    let b_z: Vec<f64> = geom.grad.iter().map(|g| g[1]).collect();
    check_all(&doc, "b_r_row", &b_r);
    check_all(&doc, "b_z_row", &b_z);
    check_all(&doc, "gamma", &geom.gamma);
    check(&doc, "area", geom.area);
    check(&doc, "radius_bar", geom.radius_bar);
    check(&doc, "volume", geom.volume);
    check(&doc, "nodal_mass", geom.lumped_masses(8930.0)[0]);
    check(&doc, "char_length", geom.char_length);
    check(&doc, "critical_step", geom.critical_step(4500.0));

    // Impact-rate driver: bottom face rises at 227 m/s, top fixed.
    let velocities = [0.0, VELOCITY, 0.0, VELOCITY, 0.0, 0.0, 0.0, 0.0];
    let params = AxisymExplicitParams::new(8930.0, 4500.0);

    // Elastic step over dt = 1 ns.
    let elastic = J2Material::new(117.0e9, 0.35, 1.0e12, HardeningLaw::Perfect);
    let mut state = QuadRingState::new(PlasticState::new_3d());
    let out = quad_ring_forces(&geom, &mut state, &elastic, &velocities, 1.0e-9, &params).unwrap();
    assert!(!out.yielding);
    check(&doc, "elastic_srr", state.point.stress[0]);
    check(&doc, "elastic_szz", state.point.stress[1]);
    check(&doc, "elastic_sth", state.point.stress[2]);
    check(&doc, "bulk_q", out.bulk_q);
    check(&doc, "bottom_el", out.forces[1] + out.forces[3]);

    // Plastic single step over dt = 1 us with linear hardening.
    let copper = J2Material::new(
        117.0e9,
        0.35,
        300.0e6,
        HardeningLaw::LinearIsotropic { modulus: 1.0e9 },
    );
    let mut state = QuadRingState::new(PlasticState::new_3d());
    let out = quad_ring_forces(&geom, &mut state, &copper, &velocities, 1.0e-6, &params).unwrap();
    assert!(out.yielding);
    check(&doc, "plastic_srr", state.point.stress[0]);
    check(&doc, "plastic_szz", state.point.stress[1]);
    check(&doc, "plastic_sth", state.point.stress[2]);
    check(&doc, "plastic_dep", out.dep);
    check(&doc, "bottom_pl", out.forces[1] + out.forces[3]);
}
