//! Rigid-sphere TBEM validation against the aligned complex Mie surface trace.

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::build_tbem_system_with_beta;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::io::write_surface_pressure_vtu;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
use sonium_bem::core::types::{BasisOrder, BoundaryCondition, PhysicsParams};
use sonium_qa::rigid_sphere_surface_relative_l2_error;
use sonium_xem::io::binary_output_dir;
use std::f64::consts::PI;

fn test_simple_sphere() {
    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let k = ka / radius;
    let frequency = k * speed_of_sound / (2.0 * PI);

    let mesh = generate_icosphere_mesh(radius, 2);
    let mut elements = mesh.elements.clone();
    for (i, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![i];
    }

    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let (beta, beta_scale) = physics.burton_miller_beta_adaptive(radius);
    let system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);
    let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|element| element.normal).collect();
    let rhs =
        IncidentField::plane_wave_z().compute_rhs_with_beta(&centers, &normals, &physics, beta);

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

    let ax = system.matrix.matvec(&solve.x);
    let residual_norm = ax
        .iter()
        .zip(&rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    let relative_residual = residual_norm / rhs_norm;
    let surface_error = rigid_sphere_surface_relative_l2_error(&solve.x, &elements, k, radius, 50);

    println!("=== Simple Sphere BEM Test ===");
    println!(
        "ka={ka:.2}, elements={}, beta scale={beta_scale:.2}, iterations={}",
        elements.len(),
        solve.iterations
    );
    println!("True relative residual: {relative_residual:.3e}");
    println!(
        "Area-weighted complex Mie surface error: {:.3}%",
        100.0 * surface_error
    );

    let out_dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    write_surface_pressure_vtu(
        &mesh,
        &solve.x,
        BasisOrder::Constant,
        out_dir.join("simple_sphere_ka1.vtu"),
    )
    .expect("write surface pressure VTU");

    assert!(
        relative_residual < 1e-8,
        "true relative residual is {relative_residual:.3e}"
    );
    assert!(
        surface_error < 0.01,
        "area-weighted complex surface error is {:.3}%",
        100.0 * surface_error
    );
}

fn main() {
    let result = std::panic::catch_unwind(test_simple_sphere);
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
