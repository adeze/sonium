//! QA: implicit dynamics SDOF exact-response verification.
//!
//! Drives the production Newmark/HHT integrator on single-degree systems
//! against closed forms: undamped free vibration (period accuracy plus a
//! second-order refinement check), step-load response, Rayleigh-decay
//! envelope, and resonantly forced steady state.

// Rust guideline compliant 2026-02-21

use sonium_fem::solver::implicit_dynamics::{
    ImplicitDynamicsConfig, ImplicitDynamicsIntegrator, ImplicitDynamicsSystem,
};
use sonium_solvers::TripletMatrix;
use std::f64::consts::PI;

fn sdof(mass: f64, stiffness: f64) -> ImplicitDynamicsSystem {
    let mut m = TripletMatrix::new(1, 1);
    m.push(0, 0, mass);
    let mut k = TripletMatrix::new(1, 1);
    k.push(0, 0, stiffness);
    ImplicitDynamicsSystem {
        mass: m.to_csr(),
        stiffness: k.to_csr(),
        damping: None,
    }
}

fn integrate(
    system: ImplicitDynamicsSystem,
    config: ImplicitDynamicsConfig,
    dt: f64,
    u0: f64,
    v0: f64,
    force: &[f64],
) -> Vec<f64> {
    let mut integrator = ImplicitDynamicsIntegrator::new(system, config, dt).unwrap();
    integrator.initialize(&[u0], &[v0], &[force[0]]).unwrap();
    let mut out = Vec::with_capacity(force.len());
    for step in 1..force.len() {
        out.push(integrator.step(&[force[step]]).unwrap().displacement[0]);
    }
    out
}

fn test_free_vibration_order() {
    let omega = 2.0 * PI;
    let config = ImplicitDynamicsConfig::default();
    let mut errors = Vec::new();
    for &divisions in &[50usize, 100, 200] {
        let dt = 1.0 / divisions as f64;
        let steps = 5 * divisions;
        let force = vec![0.0; steps + 1];
        let trace = integrate(sdof(1.0, omega * omega), config, dt, 1.0, 0.0, &force);
        // Max error over the full trace vs cos(omega t): trapezoidal
        // amplitude is exact, so this reads the second-order phase lag.
        // (Sampling only at whole periods would show fourth order from
        // the cos^2 superconvergence at the return nodes.)
        let mut worst = 0.0_f64;
        for (i, value) in trace.iter().enumerate() {
            let exact = (omega * (i + 1) as f64 * dt).cos();
            worst = worst.max((value - exact).abs());
        }
        errors.push(worst);
        println!("free vibration dt={dt:.4e}: trace error {worst:.3e}");
    }
    assert!(errors[2] < 4e-3, "fine error {}", errors[2]);
    for pair in errors.windows(2) {
        let ratio = pair[0] / pair[1];
        assert!(
            (2.5..6.0).contains(&ratio),
            "second-order ratio out of range: {ratio:.3}"
        );
    }
}

fn test_step_load() {
    let omega = 10.0;
    let period = 2.0 * PI / omega;
    let dt = period / 200.0;
    let steps = 200;
    let force = vec![1.0; steps + 1];
    let trace = integrate(
        sdof(1.0, omega * omega),
        ImplicitDynamicsConfig::default(),
        dt,
        0.0,
        0.0,
        &force,
    );
    for &fraction in &[0.25, 0.5, 0.75, 1.0] {
        let index = (fraction * steps as f64) as usize - 1;
        let t = (index + 1) as f64 * dt;
        let exact = (1.0 - (omega * t).cos()) / (omega * omega);
        // Normalize by the static scale: the full-period exact value is 0.
        let error = (trace[index] - exact).abs() * omega * omega;
        println!(
            "step load t={t:.4e}: FEM={:.6e} exact={exact:.6e}",
            trace[index]
        );
        assert!(error < 0.02, "step error {error:.3e} at t={t}");
    }
}

fn test_rayleigh_decay() {
    let omega = 2.0 * PI;
    let zeta = 0.05;
    let config = ImplicitDynamicsConfig {
        rayleigh_mass: 2.0 * zeta * omega,
        ..ImplicitDynamicsConfig::default()
    };
    let dt = 0.005;
    let steps = 200;
    let force = vec![0.0; steps + 1];
    let trace = integrate(sdof(1.0, omega * omega), config, dt, 1.0, 0.0, &force);
    let expected = (-2.0 * PI * zeta).exp();
    let error = (trace[steps - 1] - expected).abs() / expected;
    println!(
        "rayleigh decay: FEM={:.6e} exact={expected:.6e}",
        trace[steps - 1]
    );
    assert!(error < 0.02, "decay error {error:.3e}");
}

fn test_hht_mild_damping() {
    let omega = 2.0 * PI;
    let config = ImplicitDynamicsConfig::hht(-0.05);
    let dt = 0.01;
    let steps = 100;
    let force = vec![0.0; steps + 1];
    let trace = integrate(sdof(1.0, omega * omega), config, dt, 1.0, 0.0, &force);
    let amplitude = trace[steps - 1];
    println!("HHT free vibration after one period: {amplitude:.6e}");
    assert!(
        amplitude < 1.0 && amplitude > 0.95,
        "HHT amplitude {amplitude} outside (0.95, 1)"
    );
}

fn test_resonant_steady_state() {
    let omega_n = 5.0;
    let zeta = 0.02;
    let c = 2.0 * zeta * omega_n;
    let mut system = sdof(1.0, omega_n * omega_n);
    let mut triplet = TripletMatrix::new(1, 1);
    triplet.push(0, 0, c);
    system.damping = Some(triplet.to_csr());
    let dt = 2.0 * PI / omega_n / 100.0;
    let periods = 60;
    let steps = periods * 100;
    let force: Vec<f64> = (0..=steps)
        .map(|i| (omega_n * i as f64 * dt).sin())
        .collect();
    let trace = integrate(
        system,
        ImplicitDynamicsConfig::default(),
        dt,
        0.0,
        0.0,
        &force,
    );
    // Measure the last-period amplitude; resonance gives F0/(c omega_n).
    let tail = &trace[steps - 100..steps - 1];
    let measured = tail.iter().map(|u| u.abs()).fold(0.0, f64::max);
    let expected = 1.0 / (c * omega_n);
    let error = (measured - expected).abs() / expected;
    println!("resonant steady state: FEM={measured:.6e} exact={expected:.6e}");
    assert!(error < 0.03, "steady-state error {error:.3e}");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_free_vibration_order();
        test_step_load();
        test_rayleigh_decay();
        test_hht_mild_damping();
        test_resonant_steady_state();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
