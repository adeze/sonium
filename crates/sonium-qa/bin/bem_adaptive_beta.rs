//! Adaptive β Tuning for Burton-Miller BEM
//!
//! Automatically finds the optimal Burton-Miller parameter β for each frequency
//! to minimize error vs analytical (Mie) solution.
//!
//! ## Background
//!
//! The Burton-Miller formulation uses β to combine CBIE and HBIE:
//!   (c*I + K' + β*H) * p = p_inc + β * ∂p_inc/∂n
//!
//! Standard choice: β = i/k (scale = 1.0)
//! But optimal β varies with:
//! - Wavenumber k (frequency)
//! - Mesh resolution (elements per wavelength)
//! - Geometry
//!
//! ## Algorithm
//!
//! For each frequency:
//! 1. Try multiple β scales (0.5, 1.0, 2.0, 4.0, 8.0, ...)
//! 2. Solve BEM system with each β
//! 3. Compare with Mie analytical solution
//! 4. Select β that minimizes error

fn test_adaptive_beta() {
    use num_complex::Complex64;

    use sonium_bem::core::assembly::tbem::build_tbem_system_scaled;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;
    use std::f64::consts::PI;

    println!("=== Adaptive β Tuning for Burton-Miller BEM ===\n");

    // Physical parameters
    let radius = 0.1; // 10 cm sphere
    let speed_of_sound = 343.0;
    let density = 1.21;

    // Mesh
    let subdivisions = 2;
    let mesh = generate_icosphere_mesh(radius, subdivisions);
    let n = mesh.elements.len();

    println!("Physical setup:");
    println!("  Sphere radius: {} m", radius);
    println!("  Speed of sound: {} m/s", speed_of_sound);
    println!("  Mesh elements: {}", n);
    println!();

    // Prepare elements
    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    // Build centers/normals
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

    // β scales to try
    let beta_scales: Vec<f64> = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

    // Frequencies to test (wider range)
    let ka_values: Vec<f64> = vec![0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0];

    println!("=== Phase 1: Finding Optimal β for Each Frequency ===\n");
    println!(
        "{:>8} | {:>8} | {:>12} | {:>12} | {:>10}",
        "Freq Hz", "ka", "Best β scale", "Best Error%", "Metric"
    );
    println!("{}", "-".repeat(60));

    let mut optimal_betas: Vec<(f64, f64, f64)> = Vec::new(); // (ka, optimal_scale, error)

    for &ka in &ka_values {
        let freq = ka * speed_of_sound / (2.0 * PI * radius);
        let k = ka / radius;

        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);

        let num_terms = (ka as usize + 20).max(30);

        // Incident field
        let incident = IncidentField::plane_wave_z();

        // Try each β scale and find the best one
        let mut best_scale = 1.0;
        let mut best_error = f64::MAX;

        for &scale in &beta_scales {
            let beta = physics.burton_miller_beta_scaled(scale);

            // Build BEM system
            let system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, scale);

            let rhs = incident.compute_rhs_with_beta(&centers, &normals, &physics, beta);

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
            let gmres_result = gmres_solve_tbem_with_ilu(&system.matrix, &rhs, &gmres_cfg);
            assert!(
                gmres_result.converged,
                "GMRES+ILU failed at ka={}, scale={}",
                ka, scale
            );
            let solution_x = gmres_result.x;

            // Compute error
            let error_pct = 100.0
                * rigid_sphere_surface_relative_l2_error(
                    &solution_x,
                    &elements,
                    k,
                    radius,
                    num_terms,
                );

            if error_pct < best_error {
                best_error = error_pct;
                best_scale = scale;
            }
        }

        optimal_betas.push((ka, best_scale, best_error));

        println!(
            "{:>8.0} | {:>8.2} | {:>12.1} | {:>11.1}% | {:>10.4}",
            freq, ka, best_scale, best_error, "weighted L2"
        );
    }

    println!();
    println!("=== Phase 2: Results with Adaptive β ===\n");
    println!(
        "{:>8} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>12}",
        "Freq Hz", "ka", "β scale", "Metric", "Field", "Error %", "Status"
    );
    println!("{}", "-".repeat(85));

    for &(ka, optimal_scale, _) in &optimal_betas {
        let freq = ka * speed_of_sound / (2.0 * PI * radius);
        let k = ka / radius;

        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);
        let beta = physics.burton_miller_beta_scaled(optimal_scale);

        let num_terms = (ka as usize + 20).max(30);

        // Build and solve with optimal β
        let system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, optimal_scale);

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
        assert!(gmres_result.converged, "GMRES+ILU failed at ka={}", ka);
        let solution_x = gmres_result.x;

        let error_pct = 100.0
            * rigid_sphere_surface_relative_l2_error(&solution_x, &elements, k, radius, num_terms);

        let status = if error_pct < 3.0 { "✓" } else { "✗" };

        println!(
            "{:>8.0} | {:>8.2} | {:>8.1} | {:>10.4} | {:>10.4} | {:>9.1}% | {:>12}",
            freq, ka, optimal_scale, "weighted", "complex", error_pct, status
        );
    }

    println!();
    println!("=== Comparison: Fixed β=4 vs Adaptive β ===\n");
    println!(
        "{:>8} | {:>8} | {:>12} | {:>12} | {:>12}",
        "Freq Hz", "ka", "Fixed β=4", "Adaptive β", "Improvement"
    );
    println!("{}", "-".repeat(65));

    for &(ka, _optimal_scale, adaptive_error) in &optimal_betas {
        let freq = ka * speed_of_sound / (2.0 * PI * radius);
        let k = ka / radius;

        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);

        let num_terms = (ka as usize + 20).max(30);

        // Fixed β = 4
        let fixed_scale = 4.0;
        let beta_fixed = physics.burton_miller_beta_scaled(fixed_scale);
        let system_fixed = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, fixed_scale);

        let incident = IncidentField::plane_wave_z();
        let rhs_fixed = incident.compute_rhs_with_beta(&centers, &normals, &physics, beta_fixed);

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
        let gmres_result = gmres_solve_tbem_with_ilu(&system_fixed.matrix, &rhs_fixed, &gmres_cfg);
        assert!(
            gmres_result.converged,
            "GMRES+ILU failed for fixed beta at ka={}",
            ka
        );
        let solution_fixed = gmres_result.x;
        let fixed_error = 100.0
            * rigid_sphere_surface_relative_l2_error(
                &solution_fixed,
                &elements,
                k,
                radius,
                num_terms,
            );

        let improvement = if fixed_error > adaptive_error {
            format!("{:.1}x better", fixed_error / adaptive_error)
        } else {
            "same".to_string()
        };

        println!(
            "{:>8.0} | {:>8.2} | {:>11.1}% | {:>11.1}% | {:>12}",
            freq, ka, fixed_error, adaptive_error, improvement
        );
    }

    println!();
    println!("=== Optimal β Scale vs ka ===\n");
    println!("Empirical relationship for this mesh:");
    for &(ka, optimal_scale, error) in &optimal_betas {
        println!(
            "  ka={:.1}: β_scale={:.1} (error={:.1}%)",
            ka, optimal_scale, error
        );
    }

    // Every selected beta must certify the full aligned complex surface field.
    for &(ka, _scale, error) in &optimal_betas {
        assert!(
            error < 3.0,
            "Adaptive beta area-weighted complex error {:.1}% at ka={:.1} exceeds 3%",
            error,
            ka
        );
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_adaptive_beta();
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
