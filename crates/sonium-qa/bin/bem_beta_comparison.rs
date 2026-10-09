//! Beta Comparison - Find optimal β scaling for different ka
//!
//! Tests different β scalings to find what works best at different frequencies

fn test_beta_comparison() {
    use num_complex::Complex64;

    use sonium_bem::core::assembly::tbem::build_tbem_system_with_beta;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;
    use std::f64::consts::PI;

    println!("=== Beta Scaling Comparison ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    let mesh = generate_icosphere_mesh(radius, 2);
    let n = mesh.elements.len();

    println!("Testing with {} elements\n", n);

    let mut best_errors: Vec<f64> = Vec::new();
    for &ka_target in &[0.3, 0.5, 1.0, 2.0] {
        let frequency = ka_target * speed_of_sound / (2.0 * PI * radius);
        let k = 2.0 * PI * frequency / speed_of_sound;

        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        println!("=== ka = {:.1} ===", ka_target);
        println!(
            "{:>10} | {:>10} | {:>10} | {:>10}",
            "β scale", "β value", "BEM avg", "Error%"
        );
        println!("{}", "-".repeat(50));

        let mut min_error = f64::MAX;
        for &scale in &[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0] {
            let beta = physics.burton_miller_beta_scaled(scale);

            let mut elements = mesh.elements.clone();
            for (i, elem) in elements.iter_mut().enumerate() {
                elem.boundary_condition =
                    BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
                elem.dof_addresses = vec![i];
            }

            let system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);

            let incident = IncidentField::plane_wave_z();
            let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
            let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();
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

            let avg_p: f64 = solution_x.iter().map(|p| p.norm()).sum::<f64>() / n as f64;
            let error = 100.0
                * rigid_sphere_surface_relative_l2_error(&solution_x, &elements, k, radius, 50);
            min_error = min_error.min(error);

            let marker = if error < 2.0 { " *" } else { "" };
            println!(
                "{:>10.1} | {:>10.4}i | {:>10.4} | {:>10.1}{}",
                scale, beta.im, avg_p, error, marker
            );
        }
        best_errors.push(min_error);
        println!();
    }

    println!("Legend: * = area-weighted complex error < 2%");
    for (i, &err) in best_errors.iter().enumerate() {
        assert!(
            err < 2.0,
            "Best area-weighted complex error for ka[{}] is {:.1}%, expected < 2%",
            i,
            err
        );
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_beta_comparison();
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
