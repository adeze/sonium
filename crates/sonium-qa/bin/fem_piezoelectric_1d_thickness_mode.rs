//! Piezoelectric 1D thickness-mode (bar resonance) validation
//!
//! Validates the bar resonance frequency against the analytical
//! longitudinal-wave formula.

use sonium_fem::physics::coupled::piezoelectric::PiezoMaterial;
use sonium_qa::analytical::piezoelectric_bar_resonance;

fn test_pzt5a_resonance() {
    let mat = PiezoMaterial::pzt_5a();
    let length = 0.01; // 10 mm bar

    let f_fem = mat.bar_resonance_frequency(length);
    let f_ana = piezoelectric_bar_resonance(length, mat.stiffness[2][2], mat.density);

    let err = (f_fem - f_ana).abs() / f_ana;
    assert!(
        err < 1e-10,
        "PZT-5A resonance mismatch: FEM={f_fem}, analytical={f_ana}"
    );
}

fn test_pzt5h_resonance() {
    let mat = PiezoMaterial::pzt_5h();
    let length = 0.005; // 5 mm bar

    let f_fem = mat.bar_resonance_frequency(length);
    let f_ana = piezoelectric_bar_resonance(length, mat.stiffness[2][2], mat.density);

    let err = (f_fem - f_ana).abs() / f_ana;
    assert!(err < 1e-10);
}

fn test_resonance_scales_with_length() {
    let mat = PiezoMaterial::pzt_5a();
    let f1 = mat.bar_resonance_frequency(0.01);
    let f2 = mat.bar_resonance_frequency(0.02);

    // f ∝ 1/L
    assert!(
        (f1 - 2.0 * f2).abs() < 1e-6,
        "Resonance should scale as 1/L"
    );
}

fn test_resonance_scales_with_stiffness() {
    let mat_5a = PiezoMaterial::pzt_5a();
    let mat_5h = PiezoMaterial::pzt_5h();
    let length = 0.01;

    let f_5a = mat_5a.bar_resonance_frequency(length);
    let f_5h = mat_5h.bar_resonance_frequency(length);

    // PZT-5H has higher c33 → higher frequency
    assert!(f_5h > f_5a, "PZT-5H should resonate higher than PZT-5A");
}

fn test_wave_speed_reasonable() {
    let mat = PiezoMaterial::pzt_5a();
    let c = mat.longitudinal_wave_speed();
    // PZT wave speed is typically 3500–4500 m/s
    assert!(
        c > 3000.0 && c < 5000.0,
        "Wave speed {c} out of expected range"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_pzt5a_resonance();
        test_pzt5h_resonance();
        test_resonance_scales_with_length();
        test_resonance_scales_with_stiffness();
        test_wave_speed_reasonable();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
