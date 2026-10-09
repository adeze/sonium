//! Piezoelectric 1D axial-load validation
//!
//! Validates the stress–charge constitutive relations for a d₃₃
//! rod under blocked and free boundary conditions.

use sonium_fem::physics::coupled::piezoelectric::PiezoMaterial;
use sonium_qa::analytical::{
    piezoelectric_blocked_stress, piezoelectric_d33, piezoelectric_free_elongation,
    piezoelectric_permittivity_stress,
};

fn test_d33_constant() {
    let mat = PiezoMaterial::pzt_5a();
    let e33 = mat.piezo_stress[2][2];
    let c33 = mat.stiffness[2][2];

    let d33_fem = piezoelectric_d33(e33, c33);
    let d33_ana = e33 / c33;

    assert!(
        (d33_fem - d33_ana).abs() < 1e-20,
        "d33 mismatch: {d33_fem} vs {d33_ana}"
    );
}

fn test_free_elongation() {
    let mat = PiezoMaterial::pzt_5a();
    let voltage = 100.0; // 100 V
    let length = 0.05; // 50 mm
    let thickness = 0.001; // 1 mm electrode spacing

    let e33 = mat.piezo_stress[2][2];
    let c33 = mat.stiffness[2][2];
    let d33 = piezoelectric_d33(e33, c33);

    let u_fem = piezoelectric_free_elongation(d33, voltage, length, thickness);
    let u_ana = d33 * voltage * length / thickness;

    assert!(
        (u_fem - u_ana).abs() < 1e-20,
        "Free elongation mismatch: {u_fem} vs {u_ana}"
    );
}

fn test_blocked_stress() {
    let mat = PiezoMaterial::pzt_5a();
    let voltage = 100.0;
    let thickness = 0.001;

    let e33 = mat.piezo_stress[2][2];
    let t_fem = piezoelectric_blocked_stress(e33, voltage, thickness);
    let t_ana = -e33 * voltage / thickness;

    assert!(
        (t_fem - t_ana).abs() < 1e-20,
        "Blocked stress mismatch: {t_fem} vs {t_ana}"
    );
}

fn test_permittivity_at_constant_stress() {
    let mat = PiezoMaterial::pzt_5a();
    let eps_s = mat.permittivity[2][2];
    let e33 = mat.piezo_stress[2][2];
    let c33 = mat.stiffness[2][2];

    let eps_t_fem = piezoelectric_permittivity_stress(eps_s, e33, c33);
    let eps_t_ana = eps_s + e33 * e33 / c33;

    assert!(
        (eps_t_fem - eps_t_ana).abs() < 1e-20,
        "Permittivity mismatch: {eps_t_fem} vs {eps_t_ana}"
    );

    // εᵀ should be larger than εˢ
    assert!(
        eps_t_fem > eps_s,
        "Permittivity at constant stress should exceed constant-strain permittivity"
    );
}

fn test_coupling_coefficient() {
    let mat = PiezoMaterial::pzt_5a();
    let k33_sq = mat.coupling_k33_squared();

    // k33² = e33² / (c33 * εˢ₃₃)
    let e33 = mat.piezo_stress[2][2];
    let c33 = mat.stiffness[2][2];
    let eps33 = mat.permittivity[2][2];
    let expected = e33 * e33 / (c33 * eps33);

    assert!(
        (k33_sq - expected).abs() < 1e-10,
        "k33² mismatch: {k33_sq} vs {expected}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_d33_constant();
        test_free_elongation();
        test_blocked_stress();
        test_permittivity_at_constant_stress();
        test_coupling_coefficient();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
