//! Golden test: simple-shear-with-rotation objectivity (solids-golden-v1).
//!
//! THE TRAP: a Jaumann-rate implementation produces spurious oscillations of
//! the shear stress under simple shear. This test replays
//! `fixtures/solids_plasticity_shear_rotation_objectivity_golden.json`
//! through the finite-strain J2 model: a monotone shear curve saturating
//! toward the shear yield, the stress on the hardened yield surface at every
//! plastic step, and exact transport under superposed rigid rotation. Curve
//! values are PROVISIONAL regression locks; the trap properties are the
//! validated behaviors.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::{
    FiniteStrainJ2Material, FiniteStrainState, HardeningLaw, StressUpdateModel,
    update::von_mises_c1,
};

const GOLDEN: &str =
    include_str!("../fixtures/solids_plasticity_shear_rotation_objectivity_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("{name} must hold decimal strings"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn check(doc: &Value, name: &str, actual: f64, expected: f64) {
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}: production {actual:.6e} vs golden {expected:.6e}"
    );
}

fn run_shear(material: &FiniteStrainJ2Material, gamma_max: f64, n_steps: usize) -> Vec<[f64; 6]> {
    let mut state = FiniteStrainState::new();
    let dg = gamma_max / n_steps as f64;
    let mut history = Vec::with_capacity(n_steps + 1);
    history.push(state.stress);
    for _ in 0..n_steps {
        let f = [1.0, dg, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let update = material.update(&state, &f);
        material.commit(&mut state, &update);
        history.push(state.stress);
    }
    history
}

#[test]
fn shear_fixture_metadata_provisional() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["case"], "plasticity_shear_rotation_objectivity");
    assert_eq!(doc["meta"]["units"], "SI");
    assert_eq!(
        doc["meta"]["provisional"], true,
        "curve is a regression lock"
    );
}

#[test]
fn shear_curve_matches_golden() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let gold_sxy = golden_values(&doc, "sigma_xy_at_gamma_pa");
    let gold_q = golden_values(&doc, "q_at_gamma");
    let material = FiniteStrainJ2Material::mild_steel();
    let history = run_shear(&material, 1.0, 200);
    for (k, target) in [0.25_f64, 0.5, 0.75, 1.0].iter().enumerate() {
        let idx = (target * 200.0).round() as usize;
        check(&doc, "sigma_xy_at_gamma_pa", history[idx][3], gold_sxy[k]);
        check(&doc, "q_at_gamma", von_mises_c1(&history[idx]), gold_q[k]);
    }
}

#[test]
fn shear_trap_monotonic_no_oscillation() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let material = FiniteStrainJ2Material::mild_steel();
    let history = run_shear(&material, 1.0, 200);
    let peak = history.iter().map(|s| s[3]).fold(0.0, f64::max);
    let mut negative_variation = 0.0;
    for pair in history.windows(2) {
        let step = pair[1][3] - pair[0][3];
        if step < 0.0 {
            negative_variation -= step;
        }
    }
    // A Jaumann rate gives O(1) relative oscillation here; Green-Naghdi
    // gives roundoff at most.
    let gold = golden_values(&doc, "negative_variation_sigma_xy")[0];
    check(
        &doc,
        "negative_variation_sigma_xy",
        negative_variation,
        gold,
    );
    assert!(
        negative_variation < 1.0e-9 * peak,
        "shear stress oscillates: {negative_variation:.3e} vs peak {peak:.3e}"
    );
}

#[test]
fn shear_stress_stays_on_yield_surface() {
    let material = FiniteStrainJ2Material::mild_steel();
    let mut state = FiniteStrainState::new();
    for _ in 0..200 {
        let f = [1.0, 0.005, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let update = material.update(&state, &f);
        material.commit(&mut state, &update);
        if state.equiv_plastic_strain > 1.0e-6 {
            let q = von_mises_c1(&state.stress);
            let ys = material
                .base
                .current_yield_stress(state.equiv_plastic_strain);
            assert!(
                (q - ys).abs() / ys < 0.02,
                "off yield surface: q={q:.3e} vs {ys:.3e}"
            );
        }
    }
}

#[test]
fn shear_perfect_plasticity_saturates_at_shear_yield() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let gold = golden_values(&doc, "perfect_plasticity_shear_saturation_pa")[0];
    let material = FiniteStrainJ2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
    let history = run_shear(&material, 1.0, 200);
    let late = history[150..].iter().map(|s| s[3]).fold(0.0, f64::max);
    check(&doc, "perfect_plasticity_shear_saturation_pa", late, gold);
    let k = 250e6 / 3.0_f64.sqrt();
    assert!(
        (late - k).abs() / k < 0.05,
        "saturation not at shear yield k={k:.6e}"
    );
}

#[test]
fn shear_rotation_hold_freezes_plastic_state() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let material = FiniteStrainJ2Material::mild_steel();
    let mut state = FiniteStrainState::new();
    for _ in 0..100 {
        let f = [1.0, 0.005, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let update = material.update(&state, &f);
        material.commit(&mut state, &update);
    }
    assert!(state.yielding);
    let q_before = von_mises_c1(&state.stress);
    let ep_before = state.equiv_plastic_strain;
    let rot90 = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let update = material.update(&state, &rot90);
    assert!(!update.yielding, "rigid rotation must not yield");
    material.commit(&mut state, &update);
    let drift = (von_mises_c1(&state.stress) - q_before).abs() / q_before;
    let gold = golden_values(&doc, "rotation_hold_q_drift_relative")[0];
    check(&doc, "rotation_hold_q_drift_relative", drift, gold);
    assert_eq!(state.equiv_plastic_strain, ep_before);
}

#[test]
fn shear_frame_indifference_bounded() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let gold = golden_values(&doc, "frame_indifference_max_relative")[0];
    let material = FiniteStrainJ2Material::mild_steel();
    let plain = run_shear(&material, 0.1, 20);
    let theta = 37.0_f64.to_radians();
    let (s, c) = theta.sin_cos();
    let q = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0];
    let qt = [c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0];
    let mat3 = |a: &[f64; 9], b: &[f64; 9]| {
        std::array::from_fn(|k| {
            let (i, j) = (k / 3, k % 3);
            a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]
        })
    };
    let mut state = FiniteStrainState::new();
    let mut max_diff: f64 = 0.0;
    for r in plain.iter().skip(1) {
        let f = [1.0, 0.005, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let update = material.update(&state, &mat3(&mat3(&q, &f), &qt));
        material.commit(&mut state, &update);
        let t = [r[0], r[3], r[5], r[3], r[1], r[4], r[5], r[4], r[2]];
        let rot = mat3(&mat3(&q, &t), &qt);
        let expected = [rot[0], rot[4], rot[8], rot[1], rot[5], rot[2]];
        let scale = von_mises_c1(&expected).max(1.0);
        for (got, exp) in state.stress.iter().zip(expected.iter()) {
            max_diff = max_diff.max((got - exp).abs() / scale);
        }
    }
    // The 100-step binary value is a looser regression bound for this
    // 20-step spot check; the property bound is what must hold.
    check(&doc, "frame_indifference_max_relative", max_diff, gold);
}
