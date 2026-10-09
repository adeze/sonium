//! 1D perfect-plasticity validation
//!
//! Compares the 3D radial-return mapping against the analytical
//! 1D uniaxial perfect-plasticity solution.

use sonium_fem::physics::solid::plasticity::{
    HardeningLaw, J2Material, PlasticState, return_mapping_3d, update_plastic_state,
};
use sonium_qa::analytical::plasticity_1d_perfect_stress;

fn test_elastic_step() {
    let mat = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
    let state = PlasticState::new_3d();

    // Small strain: well below yield (~0.00125 strain for 250 MPa / 200 GPa)
    let eps = 0.001;
    let deps = [eps, -0.3 * eps, -0.3 * eps, 0.0, 0.0, 0.0];
    let update = return_mapping_3d(&mat, &state, &deps);

    assert!(!update.yielding, "Should be elastic");
    let sigma_xx = update.stress[0];
    let expected = plasticity_1d_perfect_stress(eps, mat.youngs_modulus, mat.yield_stress);
    assert!(
        (sigma_xx - expected).abs() < 1e3,
        "Elastic stress mismatch: {sigma_xx} vs {expected}"
    );
}

fn test_yield_point() {
    let mat = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
    let state = PlasticState::new_3d();

    // Strain exactly at yield: ε_y = σ_y / E = 250e6 / 200e9 = 0.00125
    let eps_y = mat.yield_stress / mat.youngs_modulus;
    let deps = [eps_y, -0.3 * eps_y, -0.3 * eps_y, 0.0, 0.0, 0.0];
    let update = return_mapping_3d(&mat, &state, &deps);

    assert!(!update.yielding, "Exactly at yield should still be elastic");
    assert!((update.stress[0] - mat.yield_stress).abs() < 1e3);
}

fn test_plastic_step_deviatoric_clamped() {
    let mat = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
    let mut state = PlasticState::new_3d();

    // First step: yield
    let eps_y = mat.yield_stress / mat.youngs_modulus;
    let deps1 = [eps_y, -0.3 * eps_y, -0.3 * eps_y, 0.0, 0.0, 0.0];
    let update1 = return_mapping_3d(&mat, &state, &deps1);
    update_plastic_state(&mut state, &update1, &mat);

    // Second step: well into plasticity
    let eps_extra = 0.005;
    let deps2 = [eps_extra, -0.3 * eps_extra, -0.3 * eps_extra, 0.0, 0.0, 0.0];
    let update2 = return_mapping_3d(&mat, &state, &deps2);

    assert!(update2.yielding, "Should be plastic");

    // For perfect plasticity, the von Mises equivalent stress must equal σ_y
    let p = (update2.stress[0] + update2.stress[1] + update2.stress[2]) / 3.0;
    let s = [
        update2.stress[0] - p,
        update2.stress[1] - p,
        update2.stress[2] - p,
        update2.stress[3],
        update2.stress[4],
        update2.stress[5],
    ];
    let j2 =
        0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) + s[3] * s[3] + s[4] * s[4] + s[5] * s[5];
    let q = (3.0 * j2).sqrt();

    assert!(
        (q - mat.yield_stress).abs() < 1e4,
        "Von Mises = {q}, expected {} for perfect plasticity",
        mat.yield_stress
    );
}

fn test_von_mises_at_yield() {
    let mat = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
    let mut state = PlasticState::new_3d();

    let eps = 0.01;
    let deps = [eps, -0.3 * eps, -0.3 * eps, 0.0, 0.0, 0.0];
    let update = return_mapping_3d(&mat, &state, &deps);
    update_plastic_state(&mut state, &update, &mat);

    // Von Mises stress should equal yield stress for perfect plasticity
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

    assert!(
        (q - mat.yield_stress).abs() < 1e4,
        "Von Mises = {q}, expected {}",
        mat.yield_stress
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elastic_step();
        test_yield_point();
        test_plastic_step_deviatoric_clamped();
        test_von_mises_at_yield();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
