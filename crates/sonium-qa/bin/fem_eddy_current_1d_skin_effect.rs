//! Eddy-current skin-effect validation
//!
//! Compares the analytical exponential amplitude decay against the
//! material skin-depth formula. Also solves the production complex H(curl)
//! conductor field and checks phase, skin depth, impedance, and refinement.

#[path = "support/eddy_current_slab.rs"]
mod eddy_current_slab;

use sonium_fem::physics::electromagnetics::eddy_current::ConductorMaterial;
use sonium_qa::analytical::{eddy_current_amplitude_decay, eddy_current_skin_depth};

fn test_copper_skin_depth() {
    let cu = ConductorMaterial::copper();
    let f = 1000.0; // 1 kHz
    let delta_mat = cu.skin_depth(f);
    let delta_ana = eddy_current_skin_depth(f, cu.permeability(), cu.conductivity);
    let err = (delta_mat - delta_ana).abs() / delta_ana;
    assert!(
        err < 1e-10,
        "Skin depth mismatch: {delta_mat} vs {delta_ana}"
    );
}

fn test_aluminum_skin_depth() {
    let al = ConductorMaterial::aluminum();
    let f = 10_000.0; // 10 kHz
    let delta_mat = al.skin_depth(f);
    let delta_ana = eddy_current_skin_depth(f, al.permeability(), al.conductivity);
    let err = (delta_mat - delta_ana).abs() / delta_ana;
    assert!(err < 1e-10);
}

fn test_exponential_decay() {
    let steel = ConductorMaterial::steel_pole_piece();
    let f = 1000.0;
    let delta = steel.skin_depth(f);

    // At x = 0: amplitude = 1.0
    let a0 = eddy_current_amplitude_decay(0.0, delta);
    assert!((a0 - 1.0).abs() < 1e-12);

    // At x = δ: amplitude = exp(-1) ≈ 0.3679
    let a1 = eddy_current_amplitude_decay(delta, delta);
    assert!(
        (a1 - std::f64::consts::E.recip()).abs() < 1e-12,
        "a1 = {a1}"
    );

    // At x = 5δ: amplitude ≈ 0.0067
    let a5 = eddy_current_amplitude_decay(5.0 * delta, delta);
    assert!(a5 < 0.01, "At 5δ amplitude should be <1%");
}

fn test_steel_vs_copper_at_same_frequency() {
    let f = 1000.0;
    let steel = ConductorMaterial::steel_pole_piece();
    let cu = ConductorMaterial::copper();

    // Steel has higher mu and lower sigma → much smaller skin depth
    let d_steel = steel.skin_depth(f);
    let d_cu = cu.skin_depth(f);
    assert!(
        d_steel < d_cu,
        "Steel skin depth should be < copper at 1kHz"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_copper_skin_depth();
        test_aluminum_skin_depth();
        test_exponential_decay();
        test_steel_vs_copper_at_same_frequency();
    });
    match result {
        Ok(()) => match eddy_current_slab::numerical_gate(false) {
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
                let id = "fem.eddy-current-1d-skin-effect.v1";
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
