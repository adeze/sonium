//! Integration tests for BEM solver
//!
//! Tests basic BEM functionality using analytical Mie series solutions.

use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::{BemProblem, BemSolver};
use sonium_qa::{relative_error, rigid_sphere_surface_relative_l2_error};
use std::f64::consts::PI;

const RADIUS: f64 = 0.1;
const C0: f64 = 343.0;
const RHO: f64 = 1.21;

fn test_bem_rigid_sphere_basic() {
    let problem = BemProblem::rigid_sphere_scattering(0.1, 100.0, 343.0, 1.21);

    let solver = BemSolver::new();
    let result = solver.solve(&problem);

    assert!(result.is_ok(), "BEM solver should succeed");

    let solution = result.unwrap();
    assert!(solution.num_dofs() > 0);
    assert!(solution.max_surface_pressure().is_finite());
    assert!(solution.max_surface_pressure() > 0.0);
}

fn test_bem_vs_analytical_low_frequency() {
    let radius = RADIUS;
    let frequency = 100.0;
    let speed_of_sound = C0;
    let density = RHO;

    let k = 2.0 * PI * frequency / speed_of_sound;
    let ka = k * radius;

    println!("\n=== Low Frequency (ka = {:.4}) ===", ka);

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

    let eval_radius = 2.0 * radius;
    let theta_points: Vec<f64> = (0..5).map(|i| PI * i as f64 / 4.0).collect();

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

    let bem_pressure: Vec<_> = bem_field.iter().map(|value| value.p_total).collect();
    let error = relative_error(&bem_pressure, &analytical.pressure);

    println!("Complex relative L2 error: {:.3}%", error * 100.0);
    assert!(
        error < 0.01,
        "low-frequency complex field error must be below 1%, got {:.3}%",
        error * 100.0
    );
}

fn test_bem_surface_pressure() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 200.0, 343.0, 1.21, 20, 40);

    let solver = BemSolver::new();
    let solution = solver.solve(&problem).expect("BEM solve failed");

    let n_elem = solution.surface_pressure.len();
    let k = 2.0 * PI * 200.0 / C0;
    let error = rigid_sphere_surface_relative_l2_error(
        &solution.surface_pressure,
        &solution.elements,
        k,
        RADIUS,
        30,
    );

    println!(
        "Elements: {n_elem}, complex surface error: {:.3}%",
        error * 100.0
    );

    assert!(
        error < 0.005,
        "surface trace error must be below 0.5%, got {:.3}%",
        error * 100.0
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_bem_rigid_sphere_basic();
        test_bem_vs_analytical_low_frequency();
        test_bem_surface_pressure();
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
