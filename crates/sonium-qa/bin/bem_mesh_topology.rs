//! Mesh Topology Comparison: Fibonacci-hull sphere vs Icosphere
//!
//! Tests whether quasi-uniform Fibonacci topology avoids resonance issues
//! seen in icosphere at ka=0.7-0.8 and ka=2.0

fn test_mesh_topology() {
    use num_complex::Complex64;

    use sonium_bem::core::assembly::tbem::build_tbem_system_scaled;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::{generate_icosphere_mesh, generate_sphere_mesh};
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, Mesh, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;
    use std::f64::consts::PI;

    println!("=== Mesh Topology Comparison: UV vs Icosphere ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    // Focus on problematic ka values
    let ka_values: Vec<f64> = vec![0.5, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0];

    // Generate comparable meshes (~300-400 elements each)
    let ico_mesh = generate_icosphere_mesh(radius, 2); // 320 elements
    let uv_mesh = generate_sphere_mesh(radius, 12, 24); // ~576 elements (close enough)

    println!("Icosphere: {} elements", ico_mesh.elements.len());
    println!(
        "Fibonacci-hull sphere: {} elements\n",
        uv_mesh.elements.len()
    );

    println!(
        "{:>6} | {:>12} | {:>12} | {:>10}",
        "ka", "Icosphere", "UV Sphere", "Better"
    );
    println!("{}", "-".repeat(50));

    fn solve_for_mesh(mesh: &Mesh, ka: f64, radius: f64, speed_of_sound: f64, density: f64) -> f64 {
        let freq = ka * speed_of_sound / (2.0 * PI * radius);
        let k = ka / radius;
        // Prepare elements
        let mut elements = mesh.elements.clone();
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
        }

        // Build centers/normals
        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

        // Physics with adaptive beta
        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);
        let (beta, scale) = physics.burton_miller_beta_adaptive(radius);

        // Build and solve
        let system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, scale);
        let incident = IncidentField::plane_wave_z();
        let rhs = incident.compute_rhs_with_beta(&centers, &normals, &physics, beta);

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
        let gmres_result = gmres_solve_tbem_with_ilu(&system.matrix, &rhs, &gmres_cfg);
        assert!(gmres_result.converged, "GMRES+ILU failed");
        let solution_x = gmres_result.x;
        let num_terms = (ka as usize + 30).max(50);
        100.0 * rigid_sphere_surface_relative_l2_error(&solution_x, &elements, k, radius, num_terms)
    }

    for &ka in &ka_values {
        let ico_err = solve_for_mesh(&ico_mesh, ka, radius, speed_of_sound, density);
        let uv_err = solve_for_mesh(&uv_mesh, ka, radius, speed_of_sound, density);

        let better = if ico_err < uv_err { "Ico" } else { "UV" };
        let marker = if ico_err.min(uv_err) < 2.0 {
            "✓"
        } else {
            "✗"
        };

        println!(
            "{:>6.1} | {:>10.1}% | {:>10.1}% | {:>6} {}",
            ka, ico_err, uv_err, better, marker
        );

        let error = ico_err.min(uv_err);
        assert!(
            error < 2.0,
            "Best area-weighted complex surface error {error:.2}% exceeds 2%"
        );
    }

    println!("\n=== Conclusion ===");
    println!("If Fibonacci-hull sphere performs better at ka=0.7-0.8 and ka=2.0,");
    println!("the icosphere resonance is a mesh topology issue.");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_mesh_topology();
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
