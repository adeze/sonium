//! Eddy-current thin-sheet shielding validation
//!
//! Validates the exponential attenuation / shielding effectiveness
//! of a conductive sheet against analytical formulas, then solves the
//! production complex H(curl) finite-sheet field and integrated current impedance.

#[path = "support/eddy_current_slab.rs"]
mod eddy_current_slab;

use sonium_fem::physics::electromagnetics::eddy_current::ConductorMaterial;
use sonium_qa::analytical::eddy_current_shielding_db;

fn test_copper_sheet_shielding() {
    let cu = ConductorMaterial::copper();
    let f = 1e6; // 1 MHz
    let delta = cu.skin_depth(f);
    let t = 0.5e-3; // 0.5 mm sheet

    let se_db = eddy_current_shielding_db(t, delta);
    // SE should be positive (attenuation)
    assert!(se_db > 0.0, "Shielding effectiveness must be positive");

    // At 1 MHz copper δ ≈ 66 µm, t = 500 µm → SE ≈ 8.7*500/66 ≈ 66 dB
    assert!(se_db > 50.0, "0.5 mm copper at 1 MHz should give >50 dB");
}

fn test_thicker_sheet_better_shielding() {
    let cu = ConductorMaterial::copper();
    let f = 100e3; // 100 kHz
    let delta = cu.skin_depth(f);

    let se_1mm = eddy_current_shielding_db(1.0e-3, delta);
    let se_2mm = eddy_current_shielding_db(2.0e-3, delta);

    assert!(
        se_2mm > se_1mm,
        "Thicker sheet should give better shielding"
    );
    let ratio = (se_2mm - se_1mm) / se_1mm;
    assert!(
        (ratio - 1.0).abs() < 0.01,
        "SE should scale linearly with thickness"
    );
}

fn test_lower_frequency_poorer_shielding() {
    let cu = ConductorMaterial::copper();
    let t = 1.0e-3;

    let delta_1k = cu.skin_depth(1e3);
    let delta_1m = cu.skin_depth(1e6);

    let se_1k = eddy_current_shielding_db(t, delta_1k);
    let se_1m = eddy_current_shielding_db(t, delta_1m);

    // Higher frequency → smaller δ → better shielding
    assert!(
        se_1m > se_1k,
        "Higher frequency should give better shielding"
    );
}

fn test_field_ratio_from_db() {
    let cu = ConductorMaterial::copper();
    let f = 1e6;
    let delta = cu.skin_depth(f);
    let t = 0.1e-3; // 0.1 mm

    let se_db = eddy_current_shielding_db(t, delta);
    let ratio_linear = 10.0f64.powf(-se_db / 20.0);
    let ratio_exp = (-t / delta).exp();

    // Both formulas should agree (SE_dB = 20*log10(1/ratio))
    let err = (ratio_linear - ratio_exp).abs() / ratio_exp;
    assert!(
        err < 1e-4,
        "Linear ratio mismatch: {ratio_linear} vs {ratio_exp}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_copper_sheet_shielding();
        test_thicker_sheet_better_shielding();
        test_lower_frequency_poorer_shielding();
        test_field_ratio_from_db();
    });
    match result {
        Ok(()) => match eddy_current_slab::numerical_gate(true) {
            Ok(record) => {
                match record.to_json_line() {
                    Ok(json) => println!("QA_RESULT: {json}"),
                    Err(error) => {
                        eprintln!("FAILED: {error}");
                        std::process::exit(1);
                    }
                }
                println!("SUCCESS");
            }
            Err(error) => {
                let id = "fem.eddy-current-thin-sheet.v1";
                if let Ok(json) = sonium_qa::QaResult::failed(id, error.clone()).to_json_line() {
                    println!("QA_RESULT: {json}");
                }
                eprintln!("FAILED: {error}");
                std::process::exit(1);
            }
        },
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
