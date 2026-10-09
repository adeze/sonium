//! BEM mesh-convergence study against the analytical rigid-sphere Mie field.
//!
//! Every refinement must solve successfully. Accuracy is measured on aligned
//! complex total-pressure samples over the full polar arc, so neither phase
//! errors nor a single favourable observation direction can hide divergence.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::{BemProblem, BemSolver, SolverMethod};
use sonium_qa::relative_error;
use std::f64::consts::PI;

fn test_convergence_study() {
    println!("BEM Convergence Study: Rigid Sphere Scattering");
    println!("==============================================\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let meshes = [(4, 8), (6, 12), (8, 16), (12, 24), (16, 32)];

    for ka in [0.2, 0.5, 1.0, 2.0] {
        let frequency = ka * speed_of_sound / (2.0 * PI * radius);
        let wavenumber = ka / radius;
        let eval_radius = 2.0 * radius;
        let theta: Vec<f64> = (0..=24).map(|index| PI * index as f64 / 24.0).collect();
        let eval_points: Vec<[f64; 3]> = theta
            .iter()
            .map(|angle| [eval_radius * angle.sin(), 0.0, eval_radius * angle.cos()])
            .collect();
        let analytical = sphere_scattering_3d(wavenumber, radius, 50, vec![eval_radius], theta);

        println!("\n=== ka = {ka:.2} (frequency = {frequency:.1} Hz) ===");
        println!("Mesh       DOFs     complex relative L2 error");
        println!("---------------------------------------------");

        let mut errors = Vec::with_capacity(meshes.len());
        for (n_theta, n_phi) in meshes {
            let problem = BemProblem::rigid_sphere_scattering_custom(
                radius,
                frequency,
                speed_of_sound,
                density,
                n_theta,
                n_phi,
            );
            let solution = BemSolver::new()
                .with_solver_method(SolverMethod::BiCgStab)
                .with_tolerance(1e-8)
                .solve(&problem)
                .unwrap_or_else(|error| {
                    panic!("BEM solve failed for {n_theta}x{n_phi}, ka={ka}: {error}")
                });
            let bem_field = solution.evaluate_pressure_field(&eval_points);
            let bem_pressure: Vec<Complex64> =
                bem_field.iter().map(|value| value.p_total).collect();
            let error = relative_error(&bem_pressure, &analytical.pressure);
            assert!(
                error.is_finite(),
                "non-finite field error for {n_theta}x{n_phi}, ka={ka}"
            );
            errors.push(error);

            println!(
                "{n_theta:2}x{n_phi:2}       {:4}             {:7.3}%",
                solution.num_dofs(),
                error * 100.0
            );
        }

        let improving_steps = errors.windows(2).filter(|pair| pair[1] < pair[0]).count();
        assert!(
            errors.last().unwrap() < errors.first().unwrap(),
            "refinement did not reduce the complex field error at ka={ka}: {errors:?}"
        );
        assert!(
            improving_steps >= errors.len() - 2,
            "more than one refinement increased error at ka={ka}: {errors:?}"
        );
        assert!(
            *errors.last().unwrap() < 0.015,
            "finest-mesh complex error at ka={ka} must be below 1.5%, got {:.3}%",
            errors.last().unwrap() * 100.0
        );
    }
}

fn main() {
    let result = std::panic::catch_unwind(test_convergence_study);
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
