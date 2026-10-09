//! Neo-Hookean uniaxial tension validation
//!
//! Compares the material stress output against the analytical
//! first Piola–Kirchhoff stress for prescribed stretches.

use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticMaterial, NeoHookean};
use sonium_qa::analytical::neohookean_uniaxial_stress;

fn test_stress_identity() {
    let mat = NeoHookean::from_young(1e6, 0.3, 3);
    let f = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let p = mat.stress(&f);
    for (i, &pi) in p.iter().enumerate() {
        assert!(pi.abs() < 1e-12, "P[{i}] = {pi}, expected 0 at identity");
    }
}

fn test_uniaxial_tension_stress() {
    let e_mod = 1e6;
    let nu = 0.3;
    let mat = NeoHookean::from_young(e_mod, nu, 3);
    let mu = e_mod / (2.0 * (1.0 + nu));
    let lambda = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    for stretch in [1.01, 1.05, 1.10, 1.20, 1.50] {
        let f = [stretch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p_fem = mat.stress(&f);
        let p_ana = neohookean_uniaxial_stress(stretch, mu, lambda);

        let err = (p_fem[0] - p_ana).abs() / p_ana.abs().max(1.0);
        assert!(
            err < 1e-10,
            "Stretch={stretch}: FEM P11={:.6e}, analytical={:.6e}, rel_err={:.2e}",
            p_fem[0],
            p_ana,
            err
        );
    }
}

fn test_lateral_stress_consistent() {
    // For F = diag(λ,1,1), P_22 = P_33 (by symmetry).
    let mat = NeoHookean::from_young(1e6, 0.3, 3);
    let stretch = 1.1;
    let f = [stretch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let p = mat.stress(&f);
    // P_22 is at index 4, P_33 at index 8
    assert!(
        (p[4] - p[8]).abs() < 1e-10,
        "Lateral stresses P_22 and P_33 should be equal"
    );
}

fn test_tangent_symmetry_at_stretched_state() {
    let mat = NeoHookean::from_young(1e6, 0.3, 3);
    let stretch = 1.1;
    let f = [stretch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let a = mat.tangent(&f);
    let n = 9;
    for i in 0..n {
        for j in i + 1..n {
            let diff = (a[i * n + j] - a[j * n + i]).abs();
            let scale = a[i * n + i].abs().max(1.0);
            assert!(diff / scale < 1e-10, "Tangent not symmetric at ({i},{j})");
        }
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_stress_identity();
        test_uniaxial_tension_stress();
        test_lateral_stress_consistent();
        test_tangent_symmetry_at_stretched_state();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
