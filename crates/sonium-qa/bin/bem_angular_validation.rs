//! Angular rigid-sphere validation using every collocation direction.
//!
//! The previous check averaged numerical values in 10° bins and compared them
//! with Mie values at the bin edges. That aliases direction, discards phase,
//! and can silently skip empty bins. This QA compares the complex trace at each
//! element's own direction with triangle-area weights.

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::build_tbem_system_with_beta;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generators::generate_sphere_mesh;
use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
use sonium_bem::core::types::{BoundaryCondition, Mesh, PhysicsParams};
use sonium_qa::rigid_sphere_surface_relative_l2_error;
use std::f64::consts::PI;

fn solve_surface_error(
    mesh: &Mesh,
    physics: &PhysicsParams,
    beta: Complex64,
    wavenumber: f64,
    radius: f64,
) -> (f64, usize) {
    let mut elements = mesh.elements.clone();
    for (i, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![i];
    }

    let system = build_tbem_system_with_beta(&elements, &mesh.nodes, physics, beta);
    let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|element| element.normal).collect();
    let rhs =
        IncidentField::plane_wave_z().compute_rhs_with_beta(&centers, &normals, physics, beta);
    let solve = gmres_solve_tbem_with_ilu(
        &system.matrix,
        &rhs,
        &GmresConfig {
            max_iterations: 500,
            restart: 50,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(solve.converged, "GMRES+ILU failed");

    (
        rigid_sphere_surface_relative_l2_error(&solve.x, &elements, wavenumber, radius, 50),
        solve.iterations,
    )
}

fn test_angular_validation() {
    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    let ka = 1.0;
    let frequency = ka * speed_of_sound / (2.0 * PI * radius);
    let k = ka / radius;
    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let (beta, _) = physics.burton_miller_beta_adaptive(radius);

    println!("=== Angular Validation: aligned complex BEM vs Mie trace ===");
    let mut refinement_errors = Vec::new();
    for (n_theta, n_phi, name) in [(8, 16, "coarse"), (12, 24, "medium"), (16, 32, "fine")] {
        let mesh = generate_sphere_mesh(radius, n_theta, n_phi);
        let (error, iterations) = solve_surface_error(&mesh, &physics, beta, k, radius);
        println!(
            "{name:>6}: {:>4} elements, {:>3} iterations, weighted complex error={:.3}%",
            mesh.elements.len(),
            iterations,
            100.0 * error
        );
        refinement_errors.push(error);
    }

    assert!(
        refinement_errors.iter().all(|error| error.is_finite()),
        "angular surface errors must be finite"
    );
    assert!(
        refinement_errors.windows(2).all(|pair| pair[1] < pair[0]),
        "angular surface error must decrease under mesh refinement: {refinement_errors:?}"
    );
    assert!(
        refinement_errors.last().is_some_and(|error| *error < 0.01),
        "fine-mesh angular surface error must be below 1%"
    );

    let ka = 2.0;
    let frequency = ka * speed_of_sound / (2.0 * PI * radius);
    let k = ka / radius;
    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let mesh = generate_sphere_mesh(radius, 16, 32);
    let traditional_beta = physics.burton_miller_beta();
    let floored_beta = physics.burton_miller_beta_floored(70.0, 10.0);
    let (traditional_error, _) = solve_surface_error(&mesh, &physics, traditional_beta, k, radius);
    let (floored_error, _) = solve_surface_error(&mesh, &physics, floored_beta, k, radius);

    println!(
        "ka=2: traditional beta error={:.3}%, floored beta error={:.3}%",
        100.0 * traditional_error,
        100.0 * floored_error
    );
    for (name, error) in [
        ("traditional", traditional_error),
        ("floored", floored_error),
    ] {
        assert!(
            error < 0.02,
            "{name} beta area-weighted complex surface error is {:.3}%",
            100.0 * error
        );
    }
}

fn main() {
    let result = std::panic::catch_unwind(test_angular_validation);
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
