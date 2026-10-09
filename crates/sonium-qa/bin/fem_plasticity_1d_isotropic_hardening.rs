//! 1D isotropic-hardening validation
//!
//! Validates linear isotropic hardening against the 1D analytical
//! stress–strain curve.

use sonium_fem::physics::solid::plasticity::{
    HardeningLaw, J2Material, PlasticState, return_mapping_3d, update_plastic_state,
};
use sonium_qa::analytical::plasticity_1d_hardening_stress;

fn test_elastic_step() {
    let h = 1e9;
    let mat = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearIsotropic { modulus: h },
    );
    let state = PlasticState::new_3d();

    let eps = 0.001;
    let deps = [eps, -0.3 * eps, -0.3 * eps, 0.0, 0.0, 0.0];
    let update = return_mapping_3d(&mat, &state, &deps);

    assert!(!update.yielding);
    let expected = plasticity_1d_hardening_stress(eps, mat.youngs_modulus, mat.yield_stress, h);
    assert!((update.stress[0] - expected).abs() < 1e3);
}

fn test_hardening_increases_yield_surface() {
    let h = 1e9;
    let mat = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearIsotropic { modulus: h },
    );
    let mut state = PlasticState::new_3d();

    let _eps_y = mat.yield_stress / mat.youngs_modulus;
    // Two equal plastic steps
    for _ in 0..2 {
        let eps = 0.005;
        let deps = [eps, -0.3 * eps, -0.3 * eps, 0.0, 0.0, 0.0];
        let update = return_mapping_3d(&mat, &state, &deps);
        update_plastic_state(&mut state, &update, &mat);
    }

    // Von Mises stress should exceed the initial yield stress due to hardening
    let p = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
    let s = [
        state.stress[0] - p,
        state.stress[1] - p,
        state.stress[2] - p,
        state.stress[3],
        state.stress[4],
        state.stress[5],
    ];
    let j2 =
        0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) + s[3] * s[3] + s[4] * s[4] + s[5] * s[5];
    let q = (3.0 * j2).sqrt();

    let expected_yield = mat.yield_stress + h * state.equiv_plastic_strain;
    assert!(
        q > mat.yield_stress,
        "Von Mises {q} should exceed initial yield {}",
        mat.yield_stress
    );
    assert!(
        (q - expected_yield).abs() < 1e5,
        "Von Mises {q} should match updated yield {expected_yield}"
    );
}

fn test_higher_hardening_modulus_higher_stress() {
    let mat_soft = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearIsotropic { modulus: 0.5e9 },
    );
    let mat_hard = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearIsotropic { modulus: 2.0e9 },
    );

    let eps = 0.01;
    let deps = [eps, -0.3 * eps, -0.3 * eps, 0.0, 0.0, 0.0];

    let mut state_soft = PlasticState::new_3d();
    let update_soft = return_mapping_3d(&mat_soft, &state_soft, &deps);
    update_plastic_state(&mut state_soft, &update_soft, &mat_soft);

    let mut state_hard = PlasticState::new_3d();
    let update_hard = return_mapping_3d(&mat_hard, &state_hard, &deps);
    update_plastic_state(&mut state_hard, &update_hard, &mat_hard);

    assert!(
        state_hard.stress[0] > state_soft.stress[0],
        "Higher hardening modulus should give higher stress"
    );
}

fn test_accumulated_plastic_strain() {
    let h = 1e9;
    let mat = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearIsotropic { modulus: h },
    );
    let mut state = PlasticState::new_3d();

    let eps = 0.01;
    let deps = [eps, -0.3 * eps, -0.3 * eps, 0.0, 0.0, 0.0];
    let update = return_mapping_3d(&mat, &state, &deps);
    update_plastic_state(&mut state, &update, &mat);

    assert!(
        state.equiv_plastic_strain > 0.0,
        "Should accumulate plastic strain"
    );
    // Equivalent plastic strain ≈ axial plastic strain for 1D
    assert!(state.equiv_plastic_strain < eps);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elastic_step();
        test_hardening_increases_yield_surface();
        test_higher_hardening_modulus_higher_stress();
        test_accumulated_plastic_strain();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
