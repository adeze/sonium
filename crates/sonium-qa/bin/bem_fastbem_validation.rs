//! Integration tests for BEM based on FastBEM verification cases
//!
//! Reference: https://fastbem.com/VerificationCases.html
//!
//! These tests validate BEM against analytical Mie solutions.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::{BemProblem, BemSolver};
use sonium_qa::relative_error;
use sonium_xem::io::binary_output_dir;
use std::f64::consts::PI;

fn out_dir() -> std::path::PathBuf {
    binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
}

const RADIUS: f64 = 0.1;
const C0: f64 = 343.0;
const RHO: f64 = 1.21;

fn test_fastbem_rigid_sphere_rayleigh() {
    let radius = RADIUS;
    let frequency = 100.0;
    let speed_of_sound = C0;
    let density = RHO;

    let k = 2.0 * PI * frequency / speed_of_sound;
    let ka = k * radius;

    println!("\n=== Rayleigh Regime ===");
    println!("ka = {:.4}", ka);

    let problem = BemProblem::rigid_sphere_scattering_custom(
        radius,
        frequency,
        speed_of_sound,
        density,
        6,
        12,
    );

    let solver = BemSolver::new();
    let solution = solver.solve(&problem).expect("BEM solve failed");

    let _ = solution.write_vtu(&problem.mesh, out_dir().join("rayleigh_sphere.vtu"));

    println!("BEM DOFs: {}", solution.num_dofs());

    let eval_radius = 2.0 * radius;
    let theta_points: Vec<f64> = (0..9).map(|i| PI * i as f64 / 8.0).collect();

    let eval_points: Vec<[f64; 3]> = theta_points
        .iter()
        .map(|&theta| {
            let x = eval_radius * theta.sin();
            let z = eval_radius * theta.cos();
            [x, 0.0, z]
        })
        .collect();

    let bem_field = solution.evaluate_pressure_field(&eval_points);

    let analytical = sphere_scattering_3d(k, radius, 20, vec![eval_radius], theta_points.clone());

    let bem_pressure: Vec<Complex64> = bem_field.iter().map(|value| value.p_total).collect();
    let error = relative_error(&bem_pressure, &analytical.pressure);
    println!("Complex relative L2 error: {:.3}%", error * 100.0);

    assert!(
        error < 0.01,
        "Rayleigh complex field error must be below 1%, got {:.3}%",
        error * 100.0
    );
}

fn test_fastbem_rigid_sphere_mie() {
    let radius = RADIUS;
    let frequency = 546.0;
    let speed_of_sound = C0;
    let density = RHO;

    let k = 2.0 * PI * frequency / speed_of_sound;
    let ka = k * radius;

    println!("\n=== Mie Regime ===");
    println!("ka = {:.4}", ka);

    let problem = BemProblem::rigid_sphere_scattering_custom(
        radius,
        frequency,
        speed_of_sound,
        density,
        12,
        24,
    );

    let solver = BemSolver::new();
    let solution = solver.solve(&problem).expect("BEM solve failed");

    let _ = solution.write_vtu(&problem.mesh, out_dir().join("mie_sphere.vtu"));

    println!("BEM DOFs: {}", solution.num_dofs());

    let eval_radius = 2.0 * radius;
    let theta_points: Vec<f64> = (0..9).map(|i| PI * i as f64 / 8.0).collect();

    let eval_points: Vec<[f64; 3]> = theta_points
        .iter()
        .map(|&theta| {
            let x = eval_radius * theta.sin();
            let z = eval_radius * theta.cos();
            [x, 0.0, z]
        })
        .collect();

    let bem_field = solution.evaluate_pressure_field(&eval_points);

    let analytical = sphere_scattering_3d(k, radius, 30, vec![eval_radius], theta_points.clone());

    let bem_pressure: Vec<Complex64> = bem_field.iter().map(|value| value.p_total).collect();
    let error = relative_error(&bem_pressure, &analytical.pressure);
    println!("Complex relative L2 error: {:.3}%", error * 100.0);

    assert!(
        error < 0.015,
        "Mie complex field error must be below 1.5%, got {:.3}%",
        error * 100.0
    );
}

fn test_fastbem_convergence_with_refinement() {
    let radius = RADIUS;
    let frequency = 500.0;
    let speed_of_sound = C0;
    let density = RHO;

    let k = 2.0 * PI * frequency / speed_of_sound;
    let ka = k * radius;

    println!("\n=== Convergence Study ===");
    println!("ka = {:.4}", ka);

    let resolutions = [(6, 12), (12, 24), (16, 32)];
    let mut errors = Vec::new();

    let eval_radius = 2.0 * radius;
    let theta_points: Vec<f64> = (0..=16).map(|index| PI * index as f64 / 16.0).collect();

    let analytical = sphere_scattering_3d(k, radius, 30, vec![eval_radius], theta_points.clone());

    for (n_theta, n_phi) in resolutions {
        let problem = BemProblem::rigid_sphere_scattering_custom(
            radius,
            frequency,
            speed_of_sound,
            density,
            n_theta,
            n_phi,
        );

        let solver = BemSolver::new();
        let solution = solver.solve(&problem).expect("BEM solve failed");

        let _ = solution.write_vtu(
            &problem.mesh,
            out_dir().join(format!("convergence_n{}_n{}.vtu", n_theta, n_phi)),
        );

        let eval_points: Vec<[f64; 3]> = theta_points
            .iter()
            .map(|&theta| {
                let x = eval_radius * theta.sin();
                let z = eval_radius * theta.cos();
                [x, 0.0, z]
            })
            .collect();

        let bem_field = solution.evaluate_pressure_field(&eval_points);

        let bem_pressure: Vec<Complex64> = bem_field.iter().map(|value| value.p_total).collect();
        let error = relative_error(&bem_pressure, &analytical.pressure);

        println!(
            "n={}x{}: complex relative L2 error = {:.3}%",
            n_theta,
            n_phi,
            error * 100.0
        );
        errors.push(error);
    }

    let improving = errors.windows(2).all(|pair| pair[1] < pair[0]);
    assert!(
        improving,
        "complex errors must improve with refinement: {:?}",
        errors
    );
    assert!(
        *errors.last().unwrap() < 0.01,
        "finest complex field error must be below 1%, got {:.3}%",
        errors.last().unwrap() * 100.0
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_fastbem_rigid_sphere_rayleigh();
        test_fastbem_rigid_sphere_mie();
        test_fastbem_convergence_with_refinement();
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
