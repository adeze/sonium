//! Wolfram golden replay: H8 explicit patch.
//!
//! Replays `fixtures/solids_hex_explicit_patch_golden.json` (derived by
//! `fixtures/solids_hex_explicit_patch.wls` at 50-digit precision)
//! through the production H8 explicit API. Regenerate with:
//! `wolframscript -script fixtures/solids_hex_explicit_patch.wls > fixtures/solids_hex_explicit_patch_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::nonlinear_solid::{
    HexExplicitGeom, HexExplicitParams, HexExplicitState, HexMesh, hex_explicit_forces,
};
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};

const GOLDEN: &str = include_str!("../fixtures/solids_hex_explicit_patch_golden.json");

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
fn hex_explicit_patch_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let mesh = HexMesh::unit_cube(1.0);
    let geom = HexExplicitGeom::from_mesh(&mesh, 0).unwrap();
    let params = HexExplicitParams::new(7800.0, 4500.0);
    check(&doc, "volume", geom.volume);
    check_all(&doc, "grad_node0", &geom.grad_b[0]);
    check_all(&doc, "hg_mode0", &geom.gamma[0..8]);
    check(&doc, "char_length", geom.char_length);
    check(&doc, "nodal_mass", geom.lumped_masses(7800.0)[0]);
    check(&doc, "critical_step", geom.critical_step(4500.0));
    check(&doc, "hg_coeff", geom.hg_coefficient(&params));

    // Full patch: uniaxial stretch at 1/s over dt = 1e-3.
    let material = J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect);
    let mut state = HexExplicitState::new(PlasticState::new_3d());
    let mut velocities = [0.0; 24];
    for node in 0..8 {
        velocities[node * 3] = mesh.coords[node][0];
    }
    hex_explicit_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
    check(&doc, "uniaxial_sxx", state.centroid.stress[0]);
}
