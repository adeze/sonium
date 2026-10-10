//! Golden test: thick-cylinder autofrettage (solids-golden-v1).
//!
//! Replays `fixtures/solids_plasticity_thick_cylinder_autofrettage_golden.json`
//! (derived by `fixtures/solids_plasticity_thick_cylinder_autofrettage.wls`,
//! triple-checked against Rust and Python to 1.3e-14) through the analytic
//! autofrettage solution: first-yield and limit pressures, the
//! elastic-plastic interface radius at overload, and the residual field
//! after elastic unloading. Regenerate with:
//! `wolframscript -file fixtures/solids_plasticity_thick_cylinder_autofrettage.wls`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;

const GOLDEN: &str =
    include_str!("../fixtures/solids_plasticity_thick_cylinder_autofrettage_golden.json");

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
fn autofrettage_metadata_final() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(
        doc["meta"]["case"],
        "plasticity_thick_cylinder_autofrettage"
    );
    assert_eq!(doc["meta"]["units"], "SI");
    assert!(
        doc["meta"].get("provisional").is_none(),
        "analytic golden is final"
    );
}

#[test]
fn autofrettage_pressures_and_interface() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let a = doc["inputs"]["a_m"].as_f64().unwrap();
    let b = doc["inputs"]["b_m"].as_f64().unwrap();
    let sy = doc["inputs"]["sigma_y_pa"].as_f64().unwrap();
    let ps = doc["inputs"]["overload_p_star_pa"].as_f64().unwrap();
    let sqrt3 = 3.0_f64.sqrt();

    check(
        &doc,
        "yield_pressure_pa",
        sy * (1.0 - (a / b).powi(2)) / sqrt3,
    );
    let plim = (2.0 / sqrt3) * sy * (b / a).ln();
    check(&doc, "limit_pressure_pa", plim);
    let tresca = sy * (b / a).ln();
    check(&doc, "tresca_limit_crosscheck_pa", tresca);
    assert!(
        (plim / tresca - 2.0 / sqrt3).abs() < 1e-12,
        "von Mises factor"
    );

    let target = ps * sqrt3 / (2.0 * sy);
    let (mut lo, mut hi) = (a, b);
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        let g = (mid / a).ln() + (b * b - mid * mid) / (2.0 * b * b) - target;
        if g < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    check(&doc, "interface_radius_m", 0.5 * (lo + hi));
}

#[test]
fn autofrettage_residual_field() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let a = doc["inputs"]["a_m"].as_f64().unwrap();
    let b = doc["inputs"]["b_m"].as_f64().unwrap();
    let sy = doc["inputs"]["sigma_y_pa"].as_f64().unwrap();
    let ps = doc["inputs"]["overload_p_star_pa"].as_f64().unwrap();
    let sqrt3 = 3.0_f64.sqrt();
    let c = golden_value(&doc, "interface_radius_m");

    let lame = |p: f64, r: f64| {
        let amp = p * a * a / (b * b - a * a);
        let shape = b * b / (r * r);
        (amp * (1.0 - shape), amp * (1.0 + shape))
    };
    let loaded = |r: f64| {
        let flow = 2.0 * sy / sqrt3;
        if r <= c {
            let sr = -ps + flow * (r / a).ln();
            (sr, sr + flow)
        } else {
            let qc = ps - flow * (c / a).ln();
            let amp = qc * c * c / (b * b - c * c);
            let shape = b * b / (r * r);
            (amp * (1.0 - shape), amp * (1.0 + shape))
        }
    };
    let resid = |r: f64| {
        let (lr, lt) = loaded(r);
        let (ur, ut) = lame(ps, r);
        (lr - ur, lt - ut)
    };

    let (bore_sr, bore_st) = resid(a);
    check(&doc, "bore_radial_residual_pa", bore_sr);
    check(&doc, "bore_hoop_residual_pa", bore_st);
    assert!(bore_st < 0.0, "bore hoop must be compressive");

    let (outer_sr, outer_st) = resid(b);
    check(&doc, "outer_radial_residual_pa", outer_sr);
    check(&doc, "outer_hoop_residual_pa", outer_st);

    let mut max_q: f64 = 0.0;
    for i in 0..=2000 {
        let r = a + (b - a) * f64::from(i) / 2000.0;
        let (sr, st) = resid(r);
        max_q = max_q.max((sqrt3 / 2.0) * (st - sr).abs());
    }
    check(&doc, "max_residual_equiv_pa", max_q);
    assert!(max_q < sy, "no reverse yielding at this overload");
}
