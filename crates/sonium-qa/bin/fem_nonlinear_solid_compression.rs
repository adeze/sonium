//! Neo-Hookean uniaxial compression validation
//!
//! Validates stress-stretch relation for compressive deformations (λ < 1).

use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticMaterial, NeoHookean};
use sonium_qa::analytical::neohookean_uniaxial_stress;

fn test_compression_stress_negative() {
    let e_mod = 1e6;
    let nu = 0.3;
    let mat = NeoHookean::from_young(e_mod, nu, 3);

    for stretch in [0.99, 0.95, 0.90, 0.80] {
        let f = [stretch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p = mat.stress(&f);
        // Compression → P_11 < 0
        assert!(p[0] < 0.0, "Compression should give negative P_11");
    }
}

fn test_compression_vs_analytical() {
    let e_mod = 1e6;
    let nu = 0.3;
    let mat = NeoHookean::from_young(e_mod, nu, 3);
    let mu = e_mod / (2.0 * (1.0 + nu));
    let lambda = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    for stretch in [0.99, 0.95, 0.90, 0.80] {
        let f = [stretch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p_fem = mat.stress(&f);
        let p_ana = neohookean_uniaxial_stress(stretch, mu, lambda);

        let err = (p_fem[0] - p_ana).abs() / p_ana.abs().max(1.0);
        assert!(
            err < 1e-10,
            "λ={stretch}: FEM P11={:.6e}, analytical={:.6e}, rel_err={:.2e}",
            p_fem[0],
            p_ana,
            err
        );
    }
}

fn test_tangent_symmetry_under_compression() {
    let mat = NeoHookean::from_young(1e6, 0.3, 3);
    let stretch = 0.9;
    let f = [stretch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let a = mat.tangent(&f);
    let n = 9;
    for i in 0..n {
        for j in i + 1..n {
            let diff = (a[i * n + j] - a[j * n + i]).abs();
            let scale = a[i * n + i].abs().max(1.0);
            assert!(
                diff / scale < 1e-10,
                "Tangent not symmetric at ({i},{j}) under compression"
            );
        }
    }
}

fn test_bulk_compression_modulus() {
    // For hydrostatic compression F = diag(λ,λ,λ) with λ < 1
    let e_mod = 2.0;
    let nu = 0.3;
    let mat = NeoHookean::from_young(e_mod, nu, 3);
    let k = e_mod / (3.0 * (1.0 - 2.0 * nu));

    let lambda_comp = 0.99; // 1% volumetric compression
    let f = [
        lambda_comp,
        0.0,
        0.0,
        0.0,
        lambda_comp,
        0.0,
        0.0,
        0.0,
        lambda_comp,
    ];
    let p = mat.stress(&f);
    // Mean normal stress for small volumetric strain ≈ K · ε_vol
    let p_hydro = (p[0] + p[4] + p[8]) / 3.0;
    let strain_vol = 3.0 * (lambda_comp - 1.0);
    let p_expected = k * strain_vol; // strain_vol < 0 → p_expected < 0
    let err = (p_hydro - p_expected).abs() / p_expected.abs();
    println!(
        "Hydrostatic small-strain bulk-modulus error: {:.4}%",
        100.0 * err
    );
    // The finite-strain Neo-Hookean stress differs from its linearization by
    // 1.21% at this prescribed 1% stretch compression.
    assert!(
        err < 0.015,
        "Bulk modulus mismatch: {p_hydro} vs {p_expected}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_compression_stress_negative();
        test_compression_vs_analytical();
        test_tangent_symmetry_under_compression();
        test_bulk_compression_modulus();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
