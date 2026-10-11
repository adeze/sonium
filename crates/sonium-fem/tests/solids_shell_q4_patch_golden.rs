//! Wolfram golden replay: Q4 crash-shell patch.
//!
//! Replays `fixtures/solids_shell_q4_patch_golden.json` (derived by
//! `fixtures/solids_shell_q4_patch.wls` at 50-digit precision) through
//! the production crash-shell API. Regenerate with:
//! `wolframscript -script fixtures/solids_shell_q4_patch.wls > fixtures/solids_shell_q4_patch_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};
use sonium_fem::physics::solid::shell::crash::{
    DefShell, HourglassParams, LayerStates, NoThickShell, Q4CrashGeom, ShellPlasticityFlag,
    plane_stress_step, q4_crash_forces, thickness_points,
};

const GOLDEN: &str = include_str!("../fixtures/solids_shell_q4_patch_golden.json");

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
fn q4_patch_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let geom = Q4CrashGeom::from_corners(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        0.002,
    )
    .unwrap();
    check_all(&doc, "membrane_b_row0", &geom.membrane_b[0..8]);
    check_all(&doc, "membrane_b_row1", &geom.membrane_b[8..16]);
    check_all(&doc, "gamma", &geom.gamma);
    check(&doc, "area", geom.area);
    check(&doc, "nodal_mass", geom.lumped_masses(7800.0)[0]);
    check(&doc, "critical_step", geom.critical_step(4500.0));

    let material = J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect);
    let state = PlasticState::new_3d();
    let (update, dezz) = plane_stress_step(
        &material,
        &state,
        &[1.0e-3, 0.0, 0.0],
        &[0.0; 3],
        0.0,
        1.0e-3,
        ShellPlasticityFlag::RadialReturn,
    )
    .unwrap();
    check(&doc, "plane_sxx", update.stress[0]);
    check(&doc, "plane_syy", update.stress[1]);
    check(&doc, "plane_dezz", dezz);

    // Full patch through the force kernel: uniaxial stretch at 1/s.
    let mut layers = LayerStates::new(PlasticState::new_3d(), 5, geom.thickness).unwrap();
    let def = DefShell::explicit_defaults();
    let hg = HourglassParams::for_shell(
        def.formulation,
        7800.0,
        4500.0,
        200.0e9 / 2.6,
        geom.area,
        geom.thickness,
        geom.char_length,
        false,
    )
    .unwrap();
    let mut v = [0.0; 24];
    for node in 0..4 {
        v[node * 6] = geom.local[node].0;
    }
    let out = q4_crash_forces(
        &geom,
        &mut layers,
        &material,
        &def,
        &[0.0; 24],
        &v,
        1.0e-3,
        &hg,
        &NoThickShell,
    )
    .unwrap();
    check(&doc, "resultant_nxx", out.resultants.membrane[0]);

    let points = thickness_points(5, 0.002).unwrap();
    check(&doc, "thickness_z1", points[0].z);
}
