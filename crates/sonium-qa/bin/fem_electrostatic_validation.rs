//! Electrostatic validation tests
//!
//! Validates the analytical formulas for parallel-plate electrostatic transducers.
//! These are lumped parameter tests that verify the physics formulas directly.
//!
//! - Parallel plate capacitance: C = εA/d
//! - Bias pressure: p₀ = εV₀²/(2d₀²)
//! - Coupling factor: η = εV₀/d₀²
//! - Pull-in voltage: V_pi = √(8k_m d₀³/(27ε))
//! - Electrostatic negative spring: k_e = εV₀²/d₀³

use sonium_fem::physics::electromagnetics::electrostatic::{EPSILON_0, ParallelPlateGap};

fn test_parallel_plate_capacitance() {
    let gap = ParallelPlateGap::air_gap(100.0, 1e-3);
    let area = 1e-4; // 1 cm²

    let c_fem = gap.capacitance(area);
    let c_exact = EPSILON_0 * area / gap.gap_distance;

    let error = (c_fem - c_exact).abs() / c_exact;
    assert!(error < 1e-10, "Capacitance error: {:.2e}", error);
}

fn test_bias_pressure() {
    let gap = ParallelPlateGap::air_gap(200.0, 5e-4);

    let p_fem = gap.bias_pressure();
    let p_exact = EPSILON_0 * gap.bias_voltage.powi(2) / (2.0 * gap.gap_distance.powi(2));

    let error = (p_fem - p_exact).abs() / p_exact;
    assert!(error < 1e-10, "Bias pressure error: {:.2e}", error);
}

fn test_coupling_factor() {
    let gap = ParallelPlateGap::air_gap(150.0, 2e-4);

    let eta_fem = gap.coupling_factor();
    let eta_exact = EPSILON_0 * gap.bias_voltage / gap.gap_distance.powi(2);

    let error = (eta_fem - eta_exact).abs() / eta_exact;
    assert!(error < 1e-10, "Coupling factor error: {:.2e}", error);
}

fn test_electrostatic_stiffness() {
    let gap = ParallelPlateGap::air_gap(100.0, 1e-3);

    let k_e_fem = gap.electrostatic_spring_per_area();
    let k_e_exact = EPSILON_0 * gap.bias_voltage.powi(2) / gap.gap_distance.powi(3);

    let error = (k_e_fem - k_e_exact).abs() / k_e_exact;
    assert!(
        error < 1e-10,
        "Electrostatic stiffness error: {:.2e}",
        error
    );
}

fn test_pull_in_voltage() {
    let gap = ParallelPlateGap::air_gap(100.0, 1e-3);
    let k_mech = 1000.0;
    let area = 1e-4;

    let v_pi_fem = gap.pull_in_voltage(k_mech, area);
    let v_pi_exact = (8.0 * k_mech * gap.gap_distance.powi(3) / (27.0 * EPSILON_0 * area)).sqrt();

    let error = (v_pi_fem - v_pi_exact).abs() / v_pi_exact;
    assert!(error < 1e-10, "Pull-in voltage error: {:.2e}", error);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_parallel_plate_capacitance();
        test_bias_pressure();
        test_coupling_factor();
        test_electrostatic_stiffness();
        test_pull_in_voltage();
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
