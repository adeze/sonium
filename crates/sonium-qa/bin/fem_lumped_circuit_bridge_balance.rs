//! Lumped circuit Wheatstone bridge balance benchmark
//!
//! Validates the MNA frequency-domain solver on a resistive Wheatstone
//! bridge. When the bridge is balanced (R1/R2 = R3/R4), the voltage
//! difference between the two midpoints is zero.
//!
//! Circuit topology (node 0 = GND, node 1 = source +):
//!        node 1 (V_src)
//!       /            \
//!     R1            R3
//!     /              \
//!   node 2          node 3
//!     |               |
//!     R2              R4
//!     |               |
//!   node 0 (GND)    node 0 (GND)
//!
//! V_out = V(2) - V(3)

use num_complex::Complex64;
use sonium_fem::physics::coupled::lumped_circuit::network::CircuitNetwork;

const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };

fn test_wheatstone_bridge_balanced() {
    println!("=== Wheatstone Bridge Balance Benchmark ===\n");

    let r1 = 1000.0_f64;
    let r2 = 1000.0_f64;
    let r3 = 1000.0_f64;
    let r4 = 1000.0_f64;

    let mut net = CircuitNetwork::new();
    let vs = net.add_voltage_source(1, 0);

    // Top left arm: R1 from source to midpoint 2
    net.add_resistor(1, 2, r1);
    // Bottom left arm: R2 from midpoint 2 to GND
    net.add_resistor(2, 0, r2);
    // Top right arm: R3 from source to midpoint 3
    net.add_resistor(1, 3, r3);
    // Bottom right arm: R4 from midpoint 3 to GND
    net.add_resistor(3, 0, r4);

    let sol = net.solve(0.0, &[(vs, ONE)]).unwrap();

    let v_mid_top = sol.node_voltage(2);
    let v_mid_bottom = sol.node_voltage(3);
    let v_diff = v_mid_top.re - v_mid_bottom.re;

    println!("Balanced bridge (R1=R2=R3=R4=1kΩ):");
    println!("  V_mid_top  = {:.6e} V (expected 0.5 V)", v_mid_top.re);
    println!("  V_mid_bot  = {:.6e} V (expected 0.5 V)", v_mid_bottom.re);
    println!("  V_diff     = {:.6e} V (expected 0.0 V)", v_diff);

    assert!(
        (v_mid_top.re - 0.5).abs() < 1e-10,
        "Balanced bridge midpoint top should be V_in/2 = 0.5 V"
    );
    assert!(
        (v_mid_bottom.re - 0.5).abs() < 1e-10,
        "Balanced bridge midpoint bottom should be V_in/2 = 0.5 V"
    );
    assert!(
        v_diff.abs() < 1e-10,
        "Balanced bridge differential voltage should be zero"
    );

    println!("\n✓ PASSED: Balanced Wheatstone bridge validated");
    println!("\n=== Benchmark Complete ===");
}

fn test_wheatstone_bridge_unbalanced() {
    println!("\n--- Unbalanced Bridge ---\n");

    let r1 = 1000.0_f64;
    let r2 = 1000.0_f64;
    let r3 = 1000.0_f64;
    let r4 = 1100.0_f64; // 10% imbalance

    let mut net = CircuitNetwork::new();
    let vs = net.add_voltage_source(1, 0);

    net.add_resistor(1, 2, r1);
    net.add_resistor(2, 0, r2);
    net.add_resistor(1, 3, r3);
    net.add_resistor(3, 0, r4);

    let sol = net.solve(0.0, &[(vs, ONE)]).unwrap();

    let v_mid_top = sol.node_voltage(2);
    let v_mid_bottom = sol.node_voltage(3);
    let v_diff = v_mid_top.re - v_mid_bottom.re;

    // Analytical: V_top = V_in * R2/(R1+R2) = 0.5
    // V_bot = V_in * R4/(R3+R4) = 1100/2100 ≈ 0.5238
    let expected_diff = r2 / (r1 + r2) - r4 / (r3 + r4);

    println!("Unbalanced bridge (R4 = 1.1kΩ):");
    println!("  V_mid_top  = {:.6e} V", v_mid_top.re);
    println!("  V_mid_bot  = {:.6e} V", v_mid_bottom.re);
    println!(
        "  V_diff     = {:.6e} V (expected {:.6e})",
        v_diff, expected_diff
    );

    assert!(
        (v_diff - expected_diff).abs() < 1e-10,
        "Bridge differential voltage mismatch"
    );

    println!("\n✓ PASSED: Unbalanced Wheatstone bridge validated");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_wheatstone_bridge_balanced();
        test_wheatstone_bridge_unbalanced();
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
