//! Cyclic loading with kinematic hardening — Bauschinger effect
//!
//! Validates that reverse yielding occurs at a lower stress magnitude
//! due to the evolved back-stress.

use sonium_fem::physics::solid::plasticity::{
    HardeningLaw, J2Material, PlasticState, return_mapping_3d, update_plastic_state,
};

fn test_kinematic_hardening_reduces_reverse_yield() {
    let c = 5e9; // kinematic hardening modulus
    let mat = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearKinematic { modulus: c },
    );
    let mut state = PlasticState::new_3d();

    // Step 1: load in tension past yield
    let eps_tensile = 0.01;
    let deps1 = [
        eps_tensile,
        -0.3 * eps_tensile,
        -0.3 * eps_tensile,
        0.0,
        0.0,
        0.0,
    ];
    let update1 = return_mapping_3d(&mat, &state, &deps1);
    update_plastic_state(&mut state, &update1, &mat);

    let sigma_forward = state.stress[0];
    assert!(
        sigma_forward > mat.yield_stress,
        "Forward stress should exceed yield: {sigma_forward}"
    );

    // Step 2: unload to zero
    let deps2 = [
        -eps_tensile,
        0.3 * eps_tensile,
        0.3 * eps_tensile,
        0.0,
        0.0,
        0.0,
    ];
    let update2 = return_mapping_3d(&mat, &state, &deps2);
    update_plastic_state(&mut state, &update2, &mat);

    // After unloading, stress should be near zero but may have some residual
    // due to Bauschinger effect (yielding in reverse before reaching zero)
    let sigma_after_unload = state.stress[0];
    println!("After unload: sigma_xx = {:.3e}", sigma_after_unload);

    // Step 3: load in compression
    let eps_comp = -0.005;
    let deps3 = [eps_comp, -0.3 * eps_comp, -0.3 * eps_comp, 0.0, 0.0, 0.0];
    let update3 = return_mapping_3d(&mat, &state, &deps3);
    update_plastic_state(&mut state, &update3, &mat);

    let sigma_reverse = state.stress[0];
    println!("After reverse load: sigma_xx = {:.3e}", sigma_reverse);

    // The reverse yield stress magnitude should be smaller than forward
    // due to back-stress (Bauschinger effect)
    assert!(
        sigma_reverse.abs() < sigma_forward,
        "Bauschinger: reverse yield |{sigma_reverse}| should be < forward {sigma_forward}"
    );
}

fn test_back_stress_nonzero_after_yield() {
    let c = 5e9;
    let mat = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearKinematic { modulus: c },
    );
    let mut state = PlasticState::new_3d();

    let eps = 0.01;
    let deps = [eps, -0.3 * eps, -0.3 * eps, 0.0, 0.0, 0.0];
    let update = return_mapping_3d(&mat, &state, &deps);
    update_plastic_state(&mut state, &update, &mat);

    // Back-stress should have a positive xx component after tensile yielding
    let alpha_xx = state.back_stress[0];
    assert!(
        alpha_xx > 0.0,
        "Back-stress alpha_xx should be positive after tension"
    );
}

fn test_combined_hardening() {
    let mat = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::Combined {
            isotropic_modulus: 0.5e9,
            kinematic_modulus: 2.0e9,
        },
    );
    let mut state = PlasticState::new_3d();

    // Load past yield
    let eps = 0.01;
    let deps = [eps, -0.3 * eps, -0.3 * eps, 0.0, 0.0, 0.0];
    let update = return_mapping_3d(&mat, &state, &deps);
    update_plastic_state(&mut state, &update, &mat);

    assert!(state.yielding, "Should be yielding");
    assert!(
        state.equiv_plastic_strain > 0.0,
        "Should accumulate plastic strain"
    );
    assert!(
        state.back_stress[0] != 0.0,
        "Combined hardening should produce back-stress"
    );
}

fn test_cyclic_hysteresis_loop() {
    let c = 2e9;
    let mat = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearKinematic { modulus: c },
    );
    let mut state = PlasticState::new_3d();

    // Simple 2-cycle strain-controlled loading
    let strain_amplitude = 0.008;
    let n_cycles = 2;
    let steps_per_half = 4;
    let step_strain = strain_amplitude / steps_per_half as f64;

    let mut stresses = Vec::new();

    for _cycle in 0..n_cycles {
        // Tension half-cycle
        for _ in 0..steps_per_half {
            let deps = [
                step_strain,
                -0.3 * step_strain,
                -0.3 * step_strain,
                0.0,
                0.0,
                0.0,
            ];
            let update = return_mapping_3d(&mat, &state, &deps);
            update_plastic_state(&mut state, &update, &mat);
            stresses.push(state.stress[0]);
        }
        // Compression half-cycle
        for _ in 0..steps_per_half {
            let deps = [
                -step_strain,
                0.3 * step_strain,
                0.3 * step_strain,
                0.0,
                0.0,
                0.0,
            ];
            let update = return_mapping_3d(&mat, &state, &deps);
            update_plastic_state(&mut state, &update, &mat);
            stresses.push(state.stress[0]);
        }
    }

    // Max and min stresses should be different (hysteresis)
    let max_s = stresses.iter().copied().fold(f64::MIN, f64::max);
    let min_s = stresses.iter().copied().fold(f64::MAX, f64::min);
    assert!(max_s > 0.0, "Max stress should be positive");
    assert!(min_s < 0.0, "Min stress should be negative");
    assert!(
        max_s - min_s > 2.0 * mat.yield_stress,
        "Stress range should exceed 2*sigma_y"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_kinematic_hardening_reduces_reverse_yield();
        test_back_stress_nonzero_after_yield();
        test_combined_hardening();
        test_cyclic_hysteresis_loop();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
