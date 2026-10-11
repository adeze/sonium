//! QA: necking-bar benchmark references and validation legs.
//!
//! Documents the benchmark status: the full necking bar has NO closed form.
//! Validation rests on three legs: (1) the Consider diffuse-necking strain
//! and Bridgman correction checked here, (2) the Wolfram incremental
//! reference behind the necking C2 golden, and (3) the Radioss necking-bar
//! deck (`qa-fem-radioss-necking-bar`). Prints PASS; exits 1 on mismatch.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::fracture::ductile::{bridgman_correction, hollomon_necking_strain};

fn main() {
    println!("necking-bar: no closed form exists for the full post-necking response;");
    println!(
        "necking-bar: legs are (1) considere/bridgman, (2) wolfram incremental reference, (3) radioss deck."
    );
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
}
