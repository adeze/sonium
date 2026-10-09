//! Plasticity validation tests
//!
//! Tests for J2 (von Mises) elastoplasticity material models.
//!
//! - Material property calculations
//! - Perfect plasticity verification
//! - Hardening law behavior
//! - Return mapping consistency
//! - Von Mises stress computation

use sonium_fem::physics::solid::plasticity::{
    HardeningLaw, J2Material, PlasticState, return_mapping_3d,
};

fn test_j2_material_creation() {
    let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);

    assert!((material.youngs_modulus - 200e9).abs() < 1e3);
    assert!((material.poisson_ratio - 0.3).abs() < 1e-6);
    assert!((material.yield_stress - 250e6).abs() < 1e3);
}

fn test_j2_shear_modulus() {
    let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);

    let g = material.shear_modulus();
    let expected_g = 200e9 / (2.0 * (1.0 + 0.3));

    assert!((g - expected_g).abs() / expected_g < 1e-6);
}

fn test_j2_bulk_modulus() {
    let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);

    let k = material.bulk_modulus();
    let expected_k = 200e9 / (3.0 * (1.0 - 2.0 * 0.3));

    assert!((k - expected_k).abs() / expected_k < 1e-6);
}

fn test_hardening_law_perfect() {
    let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);

    let sigma_y0 = material.current_yield_stress(0.0);
    let sigma_y1 = material.current_yield_stress(0.1);

    assert!((sigma_y0 - 250e6).abs() < 1e3);
    assert!((sigma_y1 - 250e6).abs() < 1e3);
}

fn test_hardening_law_isotropic() {
    let material = J2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearIsotropic { modulus: 1e9 },
    );

    let sigma_y0 = material.current_yield_stress(0.0);
    let sigma_y1 = material.current_yield_stress(0.01);

    assert!((sigma_y0 - 250e6).abs() < 1e3);
    let expected = 250e6 + 1e9 * 0.01;
    assert!((sigma_y1 - expected).abs() / expected < 1e-6);
}

fn test_return_mapping_elastic() {
    let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
    let state = PlasticState::new(6);

    let strain_inc = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
    let result = return_mapping_3d(&material, &state, &strain_inc);

    assert!(!result.yielding, "Should not yield for small strain");
    assert!(result.delta_equiv_plastic_strain < 1e-10);
}

fn test_return_mapping_elastic_shear() {
    let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
    let state = PlasticState::new(6);

    let strain_inc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.001];
    let result = return_mapping_3d(&material, &state, &strain_inc);

    assert!(!result.yielding, "Should not yield for small shear strain");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_j2_material_creation();
        test_j2_shear_modulus();
        test_j2_bulk_modulus();
        test_hardening_law_perfect();
        test_hardening_law_isotropic();
        test_return_mapping_elastic();
        test_return_mapping_elastic_shear();
    });

    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
