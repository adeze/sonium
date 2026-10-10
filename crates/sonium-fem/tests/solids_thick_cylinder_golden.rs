//! Wolfram golden replay: thick-cylinder stresses.
//!
//! Replays `fixtures/solids_thick_cylinder_golden.json` (derived by
//! `fixtures/solids_thick_cylinder.wls` at 50-digit precision) through the
//! production `sonium_wave` Lame closed forms. Regenerate with:
//! `wolframscript -script fixtures/solids_thick_cylinder.wls > fixtures/solids_thick_cylinder_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_wave::analytical::manufactured_elasticity::shells;

const GOLDEN: &str = include_str!("../fixtures/solids_thick_cylinder_golden.json");

fn golden_value(doc: &Value, name: &str) -> f64 {
    doc["values"][name]["values"][0]
        .as_str()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .parse()
        .unwrap()
}

fn check(doc: &Value, name: &str, actual: f64) {
    let expected = golden_value(doc, name);
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}: production {actual:.6e} vs golden {expected:.6e}"
    );
}

#[test]
fn thick_cylinder_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let a = doc["inputs"]["inner_radius_m"].as_f64().unwrap();
    let b = doc["inputs"]["outer_radius_m"].as_f64().unwrap();
    let p = doc["inputs"]["inner_pressure_pa"].as_f64().unwrap();
    let e = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    let (sr_a, st_a) = shells::thick_cylinder_internal_pressure(a, a, b, p);
    check(&doc, "sigma_r_bore", sr_a);
    check(&doc, "sigma_theta_bore", st_a);
    let mid = (a + b) / 2.0;
    let (sr_m, st_m) = shells::thick_cylinder_internal_pressure(mid, a, b, p);
    check(&doc, "sigma_r_mid", sr_m);
    check(&doc, "sigma_theta_mid", st_m);
    let (sr_b, st_b) = shells::thick_cylinder_internal_pressure(b, a, b, p);
    check(&doc, "sigma_r_outer", sr_b);
    check(&doc, "sigma_theta_outer", st_b);
    check(
        &doc,
        "axial_stress",
        shells::thick_cylinder_axial_stress(a, b, p, 0.0),
    );
    check(
        &doc,
        "displacement_bore_plane_strain",
        shells::thick_cylinder_radial_displacement_plane_strain(a, a, b, p, e, nu),
    );
    check(
        &doc,
        "displacement_bore_plane_stress",
        shells::thick_cylinder_radial_displacement(a, a, b, p, 0.0, e, nu),
    );
}
