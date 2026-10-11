//! QA: crash-material single-element drivers vs response curves.
//!
//! Drives every crash model (Johnson-Cook, Cowper-Symonds, piecewise-linear,
//! crushable foam, GTN, Ogden, Yeoh) through single-element histories and
//! checks the response against closed forms. Prints PASS with the curve
//! endpoints; exits 1 on any mismatch.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::fracture::ductile::{
    GursonMaterial, GursonState, JohnsonCookDamage, stress_triaxiality_c1,
};
use sonium_fem::physics::solid::nonlinear_solid::{
    HyperelasticMaterial, OgdenMaterial, YeohMaterial, ogden_uniaxial_cauchy,
};
use sonium_fem::physics::solid::plasticity::PlasticState;
use sonium_fem::physics::solid::plasticity::StressUpdateModel;
use sonium_fem::physics::solid::plasticity::crash::{
    CowperSymondsJ2, CrushableFoamMaterial, CrushableFoamState, JohnsonCookMaterial,
    JohnsonCookState, PiecewiseLinearPlasticity, drive_uniaxial_stress, drive_uniaxial_stress_rate,
};

fn check(name: &str, actual: f64, expected: f64, rel: f64) {
    let ok = (actual - expected).abs() <= expected.abs() * rel;
    println!(
        "{name}: actual {actual:.6e}, expected {expected:.6e} -> {}",
        if ok { "OK" } else { "FAIL" }
    );
    assert!(ok, "{name} mismatch");
}

fn main() {
    // Johnson-Cook quasi-static uniaxial follows its closed form.
    let jc = JohnsonCookMaterial::c4340_steel();
    let mut jc_state = JohnsonCookState::new(293.0);
    let jc_curve = drive_uniaxial_stress(&jc, &mut jc_state, 0.05, 100);
    let jc_last = jc_curve.last().unwrap();
    check(
        "jc-quasistatic",
        jc_last.stress_xx,
        jc.flow_stress(jc_last.equiv_plastic_strain, 1.0, 293.0),
        0.01,
    );
    // Johnson-Cook fast uniaxial hardens by the rate factor.
    let mut jc_fast = JohnsonCookState::new(293.0);
    let fast_curve = drive_uniaxial_stress_rate(&jc, &mut jc_fast, 0.05, 100, Some(5.0e-6));
    let fast_last = fast_curve.last().unwrap();
    let rate = 0.05 / 100.0 / 5.0e-6;
    check(
        "jc-rate",
        fast_last.stress_xx,
        jc.flow_stress(fast_last.equiv_plastic_strain, rate, 293.0),
        0.03,
    );
    // Cowper-Symonds scales the base yield by the dynamic factor.
    let cs = CowperSymondsJ2::mild_steel();
    let mut cs_state = PlasticState::new_3d();
    let cs_curve = drive_uniaxial_stress_rate(&cs, &mut cs_state, 0.02, 40, Some(5.0e-6));
    let cs_last = cs_curve.last().unwrap();
    let cs_rate = 0.02 / 40.0 / 5.0e-6;
    let cs_base = cs.base.current_yield_stress(cs_last.equiv_plastic_strain);
    check(
        "cs-dynamic",
        cs_last.stress_xx,
        cs_base * cs.dynamic_factor(cs_rate),
        0.03,
    );
    // Piecewise-linear hits its hardening curve.
    let pwl = PiecewiseLinearPlasticity::mild_steel();
    let mut pwl_state = PlasticState::new_3d();
    let pwl_curve = drive_uniaxial_stress(&pwl, &mut pwl_state, 0.06, 120);
    let pwl_last = pwl_curve.last().unwrap();
    check(
        "pwl-curve",
        pwl_last.stress_xx,
        pwl.yield_stress(pwl_last.equiv_plastic_strain),
        0.01,
    );
    // Crushable foam hydrostatic crush tracks the crush curve.
    let foam = CrushableFoamMaterial::aluminum_foam();
    let mut foam_state = CrushableFoamState::new();
    for _ in 0..50 {
        let update = foam.update(&foam_state, &[-0.001, -0.001, -0.001, 0.0, 0.0, 0.0]);
        foam.commit(&mut foam_state, &update);
    }
    let p = (foam_state.stress[0] + foam_state.stress[1] + foam_state.stress[2]) / 3.0;
    check(
        "foam-hydrostatic",
        p,
        -foam.crush_strength(foam_state.compaction) * foam.normalizer() / foam.alpha,
        0.02,
    );
    // GTN update lands on the porous yield surface.
    let gurson = GursonMaterial::mild_steel();
    let gurson_state = GursonState::with_porosity(0.05);
    let g_update = gurson.update(&gurson_state, &[0.01, -0.003, -0.003, 0.0, 0.0, 0.0]);
    let gp = (g_update.stress[0] + g_update.stress[1] + g_update.stress[2]) / 3.0;
    let gq = {
        let s = &g_update.stress;
        let m = (s[0] + s[1] + s[2]) / 3.0;
        (1.5 * ((s[0] - m).powi(2) + (s[1] - m).powi(2) + (s[2] - m).powi(2))
            + 3.0 * (s[3].powi(2) + s[4].powi(2) + s[5].powi(2)))
        .sqrt()
    };
    let phi = gurson.yield_function(gp, gq, 250.0e6, 0.05);
    println!(
        "gtn-surface: residual {phi:.3e} -> {}",
        if phi.abs() < 1e-9 { "OK" } else { "FAIL" }
    );
    assert!(phi.abs() < 1e-9);
    // JC damage orders ductility by triaxiality.
    let damage = JohnsonCookDamage::c4340_steel();
    let shear = damage.fracture_strain(0.0, 1.0, 0.0);
    let uni = damage.fracture_strain(
        stress_triaxiality_c1(&[300.0e6, 0.0, 0.0, 0.0, 0.0, 0.0]),
        1.0,
        0.0,
    );
    println!("jc-damage: shear {shear:.4}, uniaxial {uni:.4}");
    assert!(shear > uni);
    // Ogden uniaxial energy and closed-form stress agree.
    let ogden = OgdenMaterial::treloar();
    let l = 1.5f64;
    let lt = 1.0 / l.sqrt();
    let f = [l, 0.0, 0.0, 0.0, lt, 0.0, 0.0, 0.0, lt];
    let mut w_expected = 0.0;
    for (mu, a) in ogden.mu.iter().zip(ogden.alpha.iter()) {
        w_expected += mu / a * (l.powf(*a) + 2.0 * lt.powf(*a) - 3.0);
    }
    check("ogden-energy", ogden.energy(&f), w_expected, 1e-9);
    let sigma = ogden_uniaxial_cauchy(&ogden.mu, &ogden.alpha, l);
    println!("ogden-cauchy: {sigma:.6e} at l={l}");
    assert!(sigma > 0.0);
    // Yeoh uniaxial energy follows its invariant form.
    let yeoh = YeohMaterial::filled_rubber();
    let d = l * l + 2.0 * lt * lt - 3.0;
    check(
        "yeoh-energy",
        yeoh.energy(&f),
        yeoh.c1 * d + yeoh.c2 * d * d + yeoh.c3 * d * d * d,
        1e-9,
    );
    println!("PASS: qa-fem-crash-materials");
}
