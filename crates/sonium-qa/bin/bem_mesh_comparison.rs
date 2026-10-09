//! Mesh Comparison Test - Compare Fibonacci-hull vs icosphere meshes
//!
//! Both generators are shape-regular but use different point distributions.
//! This checks that topology choice does not compromise BEM accuracy.
//!
//! Also compares standard β = i/k vs scaled β = 2i/k for conditioning improvement.

fn test_mesh_comparison() {
    use sonium_bem::core::mesh::generators::{generate_icosphere_mesh, generate_sphere_mesh};

    use std::f64::consts::PI;

    println!("=== Mesh Comparison: Fibonacci-hull vs Icosphere ===");
    println!("=== Comparing standard β vs scaled β (2×) ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    // Test at different ka values
    for ka_target in [0.2, 0.5, 1.0] {
        let frequency = ka_target * speed_of_sound / (2.0 * PI * radius);
        let k = 2.0 * PI * frequency / speed_of_sound;

        println!("\n{}", "=".repeat(60));
        println!(
            "=== ka = {:.2} (f = {:.1} Hz, k = {:.2}) ===",
            ka_target, frequency, k
        );
        println!("{}", "=".repeat(60));

        // Test quasi-uniform sphere with both beta values
        println!("\n--- Fibonacci-hull mesh (n_theta=8, n_phi=16) ---");
        let mesh_uv = generate_sphere_mesh(radius, 8, 16);
        println!("\n  Standard β = i/k:");
        test_mesh(
            &mesh_uv,
            frequency,
            speed_of_sound,
            density,
            k,
            radius,
            "Fibonacci-hull",
            None,
        );
        println!("\n  Scaled β = 2i/k:");
        test_mesh(
            &mesh_uv,
            frequency,
            speed_of_sound,
            density,
            k,
            radius,
            "Fibonacci-hull",
            Some(2.0),
        );

        // Test Icosphere with both beta values
        println!("\n--- Icosphere mesh (subdivisions=2, ~320 elements) ---");
        let mesh_ico = generate_icosphere_mesh(radius, 2);
        println!("\n  Standard β = i/k:");
        test_mesh(
            &mesh_ico,
            frequency,
            speed_of_sound,
            density,
            k,
            radius,
            "Icosphere",
            None,
        );
        println!("\n  Scaled β = 2i/k:");
        test_mesh(
            &mesh_ico,
            frequency,
            speed_of_sound,
            density,
            k,
            radius,
            "Icosphere",
            Some(2.0),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn test_mesh(
    mesh: &sonium_bem::core::types::Mesh,
    frequency: f64,
    speed_of_sound: f64,
    density: f64,
    k: f64,
    radius: f64,
    _name: &str,
    beta_scale: Option<f64>,
) {
    use num_complex::Complex64;

    use sonium_bem::core::assembly::tbem::{build_tbem_system, build_tbem_system_scaled};
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;

    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

    // Prepare elements
    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    // Build system with standard or scaled beta
    let system = match beta_scale {
        Some(scale) => build_tbem_system_scaled(&elements, &mesh.nodes, &physics, scale),
        None => build_tbem_system(&elements, &mesh.nodes, &physics),
    };

    // Compute RHS with corresponding beta
    let incident = IncidentField::plane_wave_z();
    let n = elements.len();
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

    // RHS must use the same beta as the matrix!
    let beta = match beta_scale {
        Some(scale) => physics.burton_miller_beta_scaled(scale),
        None => physics.burton_miller_beta(),
    };
    let total_rhs = incident.compute_rhs_with_beta(&centers, &normals, &physics, beta);

    // Solve
    let gmres_cfg = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };
    let gmres_result = gmres_solve_tbem_with_ilu(&system.matrix, &total_rhs, &gmres_cfg);
    assert!(gmres_result.converged, "GMRES+ILU failed");
    let solution_x = gmres_result.x;

    let overall_error_pct =
        100.0 * rigid_sphere_surface_relative_l2_error(&solution_x, &elements, k, radius, 50);
    println!("  {n} elements, area-weighted complex surface error = {overall_error_pct:.3}%");
    assert!(
        overall_error_pct < 2.0,
        "area-weighted complex surface error is {overall_error_pct:.3}%"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_mesh_comparison();
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
