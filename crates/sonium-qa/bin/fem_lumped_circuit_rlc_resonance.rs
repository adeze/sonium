//! Lumped circuit RLC resonance benchmark
//!
//! Validates the MNA frequency-domain solver against the analytical
//! resonance frequency of a series RLC circuit.
//!
//! Circuit: V_src → R → L → C → GND, output across C.
//! Analytical resonance: f_0 = 1 / (2π√(LC))
//! At resonance, |V_C| / |V_in| = Q = (1/R)√(L/C)

use num_complex::Complex64;
use sonium_fem::physics::coupled::lumped_circuit::network::CircuitNetwork;
use std::f64::consts::PI;

const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };

fn test_rlc_series_resonance() {
    println!("=== RLC Series Resonance Benchmark ===\n");

    let r = 10.0_f64;
    let l = 0.01_f64; // 10 mH
    let c = 1e-6_f64; // 1 µF
    let f0 = 1.0 / (2.0 * PI * (l * c).sqrt()); // ≈ 1591.5 Hz
    let q = (l / c).sqrt() / r; // = 10

    println!("R = {r} Ω, L = {l} H, C = {c} F");
    println!("Expected f0 = {f0:.2} Hz, Q = {q:.2}\n");

    let mut net = CircuitNetwork::new();
    let vs = net.add_voltage_source(1, 0);
    net.add_resistor(1, 2, r);
    net.add_inductor(2, 3, l);
    net.add_capacitor(3, 0, c);

    // Sweep around resonance
    let mut peak_gain = 0.0_f64;
    let mut peak_freq = 0.0_f64;
    let n_points = 200;
    for i in 0..n_points {
        let f = f0 * (0.5 + 1.5 * (i as f64 / n_points as f64));
        let omega = 2.0 * PI * f;
        let sol = net.solve(omega, &[(vs, ONE)]).unwrap();
        let v_c = sol.node_voltage(3).norm();
        if v_c > peak_gain {
            peak_gain = v_c;
            peak_freq = f;
        }
    }

    println!("Numerical peak: f = {peak_freq:.2} Hz, gain = {peak_gain:.2}");

    let f_err = (peak_freq - f0).abs() / f0;
    let q_err = (peak_gain - q).abs() / q;

    println!("Frequency error: {:.4}%", 100.0 * f_err);
    println!("Q-factor error:  {:.4}%", 100.0 * q_err);

    assert!(
        f_err < 0.01,
        "Resonance frequency error {f_err:.4} exceeds 1%"
    );
    assert!(q_err < 0.001, "Q-factor error {q_err:.4} exceeds 0.1%");

    println!("\n✓ PASSED: RLC resonance validated");
    println!("\n=== Benchmark Complete ===");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_rlc_series_resonance();
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
