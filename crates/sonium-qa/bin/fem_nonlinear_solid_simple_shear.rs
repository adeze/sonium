//! Neo-Hookean simple-shear validation
//!
//! Validates invariants I₁, I₂ and the shear stress P₁₂ for
//! a simple-shear deformation gradient.

use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticMaterial, NeoHookean};
use sonium_qa::analytical::{neohookean_simple_shear_invariants, neohookean_simple_shear_stress};

fn test_simple_shear_invariants() {
    let gamma = 0.2;
    let (i1, i2, j) = neohookean_simple_shear_invariants(gamma);

    // Analytical: I1 = I2 = 3 + γ², J = 1
    let expected_i1 = 3.0 + gamma * gamma;
    assert!((i1 - expected_i1).abs() < 1e-12, "I1 mismatch");
    assert!((i2 - expected_i1).abs() < 1e-12, "I2 mismatch");
    assert!((j - 1.0).abs() < 1e-12, "J should be 1 for isochoric shear");
}

fn test_simple_shear_stress() {
    let e_mod = 1.0;
    let nu = 0.3;
    let mat = NeoHookean::from_young(e_mod, nu, 3);
    let mu = e_mod / (2.0 * (1.0 + nu));

    for gamma in [0.01, 0.05, 0.10, 0.20, 0.50] {
        let f = [1.0, gamma, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p_fem = mat.stress(&f);
        // P_12 is at index 1 in row-major 3x3
        let p_ana = neohookean_simple_shear_stress(gamma, mu);

        let err = (p_fem[1] - p_ana).abs() / p_ana.abs().max(1.0);
        assert!(
            err < 1e-10,
            "γ={gamma}: FEM P12={:.6e}, analytical={:.6e}, rel_err={:.2e}",
            p_fem[1],
            p_ana,
            err
        );
    }
}

fn test_normal_stress_under_simple_shear() {
    // For Neo-Hookean simple shear, P_11 and P_22 are nonzero (Poynting effect)
    // P_11 = μ + (λ*ln(1) - μ)*(1) = μ - μ = 0? Wait...
    // F^{-T} = [[1,0,0],[-γ,1,0],[0,0,1]]
    // P_11 = μ*1 + (-μ)*1 = 0
    // P_22 = μ*1 + (-μ)*1 = 0
    // But for compressible with λ != 0: P_11 = μ + (λ*0 - μ)*1 = 0
    // So P_11 = P_22 = 0 for simple shear in Neo-Hookean
    let mat = NeoHookean::from_young(1.0, 0.3, 3);
    let gamma = 0.1;
    let f = [1.0, gamma, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let p = mat.stress(&f);
    assert!(p[0].abs() < 1e-12, "P_11 should be 0 for simple shear");
    assert!(p[4].abs() < 1e-12, "P_22 should be 0 for simple shear");
}

fn test_energy_density() {
    let mat = NeoHookean::from_young(1.0, 0.3, 3);
    let gamma = 0.1;
    let f = [1.0, gamma, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let w = mat.energy(&f);
    // W = (μ/2)(I1 - 3) = (μ/2)γ²
    let mu = 1.0 / (2.0 * 1.3);
    let w_ana: f64 = 0.5 * mu * gamma * gamma;
    let err = (w - w_ana).abs() / w_ana.max(1.0);
    assert!(err < 1e-10, "Energy mismatch: {w} vs {w_ana}");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_simple_shear_invariants();
        test_simple_shear_stress();
        test_normal_stress_under_simple_shear();
        test_energy_density();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
