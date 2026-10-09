//! Lumped circuit low-pass filter Bode benchmark
//!
//! Validates the MNA frequency-domain solver against the analytical
//! transfer function of an RC low-pass filter.
//!
//! Circuit: V_src → R → node_out → C → GND
//! Transfer function: H(jω) = 1 / (1 + jωRC)
//! Cutoff: f_c = 1 / (2πRC)
//! At f_c: |H| = 1/√2 ≈ -3.01 dB, phase = -45°

use num_complex::Complex64;
use sonium_fem::physics::coupled::lumped_circuit::network::CircuitNetwork;
use std::f64::consts::PI;

const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };

fn db(mag: f64) -> f64 {
    20.0 * mag.log10()
}

fn test_rc_lowpass_bode() {
    println!("=== RC Low-Pass Filter Bode Benchmark ===\n");

    let r = 1000.0_f64;
    let c = 1e-6_f64;
    let f_c = 1.0 / (2.0 * PI * r * c); // ≈ 159.15 Hz

    println!("R = {r} Ω, C = {c} F");
    println!("Expected f_c = {f_c:.2} Hz\n");

    let mut net = CircuitNetwork::new();
    let vs = net.add_voltage_source(1, 0);
    net.add_resistor(1, 2, r);
    net.add_capacitor(2, 0, c);

    // Check at cutoff frequency
    let omega_c = 2.0 * PI * f_c;
    let sol_c = net.solve(omega_c, &[(vs, ONE)]).unwrap();
    let v_out_c = sol_c.node_voltage(2);
    let mag_c = v_out_c.norm();
    let phase_c = v_out_c.arg().to_degrees();
    let db_c = db(mag_c);

    println!("At f_c:");
    println!("  Magnitude: {:.4} (expected 0.7071)", mag_c);
    println!("  dB:        {:.2} (expected -3.01)", db_c);
    println!("  Phase:     {:.1}° (expected -45.0°)", phase_c);

    assert!(
        (mag_c - 1.0_f64 / 2.0_f64.sqrt()).abs() < 0.01,
        "Magnitude at f_c deviates too much"
    );
    assert!(
        (db_c - (-3.0103)).abs() < 0.1,
        "dB at f_c deviates too much"
    );
    assert!(
        (phase_c - (-45.0)).abs() < 2.0,
        "Phase at f_c deviates too much"
    );

    // Check well below cutoff
    let sol_low = net.solve(2.0 * PI * 1.0, &[(vs, ONE)]).unwrap();
    let db_low = db(sol_low.node_voltage(2).norm());
    println!("\nAt 1 Hz: dB = {db_low:.2} (expected ~0)");
    assert!(db_low > -0.1, "Gain at 1 Hz should be ~0 dB");

    // Check well above cutoff: should roll off at -20 dB/decade
    let sol_10fc = net.solve(2.0 * PI * f_c * 10.0, &[(vs, ONE)]).unwrap();
    let db_10fc = db(sol_10fc.node_voltage(2).norm());
    println!("At 10×f_c: dB = {db_10fc:.2} (expected ~-20)");
    assert!(
        db_10fc < -18.0 && db_10fc > -22.0,
        "Roll-off should be ~-20 dB/decade"
    );

    // Check slope from f_c to 10×f_c
    let slope = db_10fc - db_c;
    println!("Slope f_c → 10×f_c: {:.1} dB/decade", slope);
    assert!(slope < -15.0, "Slope should be close to -20 dB/decade");

    println!("\n✓ PASSED: RC low-pass Bode response validated");
    println!("\n=== Benchmark Complete ===");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_rc_lowpass_bode();
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
