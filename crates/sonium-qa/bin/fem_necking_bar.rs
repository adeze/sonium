//! Necking references and native finite-strain spatial bar evolution.
//!
//! Analytical helper assertions are reported separately from a native
//! finite-strain slender-bar evolution gate. The spatial gate resolves
//! diffuse necking onset and the reaction turning point; it makes no
//! post-localization fracture or three-dimensional Bridgman claim.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::fracture::ductile::{bridgman_correction, hollomon_necking_strain};

#[path = "support/reference_result.rs"]
mod reference_result;

#[path = "support/necking_bar.rs"]
mod necking_bar;

fn main() {
    let result = run();
    reference_result::emit("reference.fem-necking-bar.v1", result.is_ok());
    match result {
        Ok(()) => {
            match necking_bar::run().and_then(|r| r.to_json_line().map_err(|e| e.to_string())) {
                Ok(line) => {
                    println!("QA_RESULT: {line}");
                    println!("SUCCESS: qa-fem-necking-bar");
                }
                Err(error) => {
                    if let Ok(line) =
                        sonium_qa::QaResult::failed("fem.necking-bar-spatial.v1", &error)
                            .to_json_line()
                    {
                        println!("QA_RESULT: {line}");
                    }
                    eprintln!("FAILED: native bar: {error}");
                    std::process::exit(1);
                }
            }
        }
        Err(error) => {
            eprintln!("FAILED: qa-fem-necking-bar: {error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    println!("analytical subcase: Considere/Bridgman identities; numerical_solve_executed=0");
    // Leg 1a: Consider uniform strain equals the Hollomon exponent.
    for n in [0.1, 0.2, 0.3] {
        let uniform = hollomon_necking_strain(n);
        println!("considere: n={n} -> uniform strain {uniform:.6}");
        assert!((uniform - n).abs() < 1e-15);
    }
    // Leg 1b: Bridgman correction limits and closed form.
    assert!((bridgman_correction(1e-9) - 1.0).abs() < 1e-6);
    for x in [0.5f64, 1.0, 2.0] {
        let expected = 1.0 / ((1.0 + 2.0 / x) * (1.0 + x / 2.0).ln());
        let got = bridgman_correction(x);
        println!("bridgman: a/R={x} -> {got:.6}");
        assert!((got - expected).abs() < 1e-15);
    }
    assert!(bridgman_correction(2.0) < 0.8);
    println!("PASS: qa-fem-necking-bar");
    Ok(())
}
