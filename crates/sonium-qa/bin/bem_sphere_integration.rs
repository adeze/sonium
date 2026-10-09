//! End-to-end rigid-sphere BEM validation against the complex Mie field.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::{BemProblem, BemSolver};
use sonium_qa::{relative_error, rigid_sphere_surface_relative_l2_error};
use std::f64::consts::PI;

const RADIUS: f64 = 0.1;
const C0: f64 = 343.0;
const RHO: f64 = 1.21;

fn evaluate_complex_field(frequency: f64, n_theta: usize, n_phi: usize) -> f64 {
    let wavenumber = 2.0 * PI * frequency / C0;
    let problem =
        BemProblem::rigid_sphere_scattering_custom(RADIUS, frequency, C0, RHO, n_theta, n_phi);
    let solution = BemSolver::new()
        .solve(&problem)
        .unwrap_or_else(|error| panic!("BEM solve failed: {error}"));
    let eval_radius = 2.0 * RADIUS;
    let theta: Vec<f64> = (0..=24).map(|index| PI * index as f64 / 24.0).collect();
    let points: Vec<[f64; 3]> = theta
        .iter()
        .map(|angle| [eval_radius * angle.sin(), 0.0, eval_radius * angle.cos()])
        .collect();
    let bem_pressure: Vec<Complex64> = solution
        .evaluate_pressure_field(&points)
        .iter()
        .map(|value| value.p_total)
        .collect();
    let analytical = sphere_scattering_3d(wavenumber, RADIUS, 40, vec![eval_radius], theta);
    relative_error(&bem_pressure, &analytical.pressure)
}

fn test_bem_vs_analytical_fields() {
    for (name, frequency, n_theta, n_phi, tolerance) in [
        ("Rayleigh", 100.0, 6, 12, 0.003),
        ("Mie", 546.0, 8, 16, 0.0075),
    ] {
        let error = evaluate_complex_field(frequency, n_theta, n_phi);
        println!("{name}: complex relative L2 error={:.3}%", error * 100.0);
        assert!(
            error < tolerance,
            "{name} complex field error {:.3}% exceeds {:.1}%",
            error * 100.0,
            tolerance * 100.0
        );
    }
}

fn test_bem_surface_pressure_distribution() {
    let frequency = 200.0;
    let wavenumber = 2.0 * PI * frequency / C0;
    let problem = BemProblem::rigid_sphere_scattering_custom(RADIUS, frequency, C0, RHO, 20, 40);
    let solution = BemSolver::new()
        .solve(&problem)
        .expect("BEM surface solve failed");
    let error = rigid_sphere_surface_relative_l2_error(
        &solution.surface_pressure,
        &solution.elements,
        wavenumber,
        RADIUS,
        40,
    );
    println!("Surface trace complex error={:.3}%", error * 100.0);
    assert!(
        error < 0.005,
        "piecewise-constant surface trace error must be below 0.5%, got {:.3}%",
        error * 100.0
    );
}

fn test_bem_mesh_convergence() {
    let frequency = 300.0;
    let mut errors = Vec::new();
    for (n_theta, n_phi) in [(4, 8), (6, 12), (8, 16), (12, 24)] {
        let error = evaluate_complex_field(frequency, n_theta, n_phi);
        println!("{n_theta:2}x{n_phi:2}: complex error={:.3}%", error * 100.0);
        errors.push(error);
    }
    assert!(
        errors.windows(2).all(|pair| pair[1] < pair[0]),
        "sphere field error must decrease under refinement: {errors:?}"
    );
    assert!(
        *errors.last().unwrap() < 0.003,
        "finest sphere field error must be below 0.3%: {errors:?}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_bem_vs_analytical_fields();
        test_bem_surface_pressure_distribution();
        test_bem_mesh_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
