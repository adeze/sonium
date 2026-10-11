//! Wolfram golden replay: tetra/wedge patch.
//!
//! Replays `fixtures/solids_tetra_penta_patch_golden.json` (derived by
//! `fixtures/solids_tetra_penta_patch.wls` at 50-digit precision)
//! through the production tetra/wedge API. Regenerate with:
//! `wolframscript -script fixtures/solids_tetra_penta_patch.wls > fixtures/solids_tetra_penta_patch_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::nonlinear_solid::{
    Penta6Geom, Tet4Geom, Tet4State, TetPentaParams, penta_quadrature, tet4_forces,
};
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};

const GOLDEN: &str = include_str!("../fixtures/solids_tetra_penta_patch_golden.json");

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
fn tetra_penta_patch_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let geom = Tet4Geom::new([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();
    check(&doc, "tet_volume", geom.volume);
    check_all(&doc, "tet_b_row_xx", &geom.b_matrix[0..12]);
    check_all(&doc, "tet_b_row_xy", &geom.b_matrix[36..48]);
    check(&doc, "tet_char_length", geom.char_length);
    check(&doc, "tet_nodal_mass", geom.lumped_masses(7800.0)[0]);

    // Full patch: uniaxial stretch at 1/s over dt = 1e-3.
    let material = J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect);
    let mut state = Tet4State::new(PlasticState::new_3d());
    let params = TetPentaParams::new(7800.0, 4500.0);
    let mut velocities = [0.0; 12];
    for node in 0..4 {
        velocities[node * 3] = geom.coords[node][0];
    }
    tet4_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
    check(&doc, "uniaxial_sxx", state.point.stress[0]);

    let wedge = Penta6Geom::new([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    .unwrap();
    check(&doc, "wedge_volume", wedge.volume);
    check(&doc, "wedge_char_length", wedge.char_length);
    check(&doc, "wedge_nodal_mass", wedge.lumped_masses(7800.0)[0]);
    let quad_sum: f64 = penta_quadrature().iter().map(|p| p.weight).sum();
    check(&doc, "wedge_quad_sum", quad_sum);
}
