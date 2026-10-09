//! Mesh Convergence Study: Finding optimal mesh density per ka
//!
//! Tests different subdivision levels (2, 3, 4) across the ka range
//! to determine minimum mesh density for target accuracy.
//!
//! Key insight: Different ka values may require different mesh densities.
//! Low ka (long wavelength) → coarser mesh OK
//! High ka (short wavelength) → finer mesh needed

fn test_mesh_convergence() {
    use num_complex::Complex64;

    use sonium_bem::core::assembly::tbem::build_tbem_system_scaled;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;
    use std::f64::consts::PI;

    println!("=== Mesh Convergence Study ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    // Test ka values focusing on problematic ones. Default tier keeps the
    // sweep small enough to fit a 900s-per-binary qa-bem budget; the full
    // 9-value sweep (`MESH_HEAVY=1`) reproduces the original analysis.
    let ka_values: Vec<f64> = if std::env::var("MESH_HEAVY").as_deref() == Ok("1") {
        vec![0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]
    } else {
        vec![0.5, 1.0, 2.0]
    };

    // Test subdivisions 2, 3, and optionally 4
    let subdivisions = [2, 3];

    // Precompute meshes
    let meshes: Vec<_> = subdivisions
        .iter()
        .map(|&s| {
            let mesh = generate_icosphere_mesh(radius, s);
            let n = mesh.elements.len();
            let sphere_area = 4.0 * PI * radius * radius;
            let avg_element_size = (sphere_area / n as f64).sqrt();
            println!(
                "Subdivision {}: {} elements, avg size {:.4}m",
                s, n, avg_element_size
            );
            (s, mesh, avg_element_size)
        })
        .collect();

    println!();
    println!(
        "{:>6} | {:>8} | {:>12} | {:>12} | {:>10}",
        "ka", "Freq Hz", "Sub2 (320)", "Sub3 (1280)", "Best"
    );
    println!("{}", "-".repeat(65));

    for &ka in &ka_values {
        let freq = ka * speed_of_sound / (2.0 * PI * radius);
        let k = ka / radius;
        let wavelength = 2.0 * PI / k;

        let num_terms = (ka as usize + 30).max(50);

        let mut errors: Vec<(usize, f64, f64)> = Vec::new(); // (subdivision, error%, epw)

        for (subdivision, mesh, avg_element_size) in &meshes {
            let epw = wavelength / avg_element_size; // elements per wavelength

            // Prepare elements
            let mut elements = mesh.elements.clone();
            for (i, elem) in elements.iter_mut().enumerate() {
                elem.boundary_condition =
                    BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
                elem.dof_addresses = vec![i];
            }

            // Build centers/normals
            let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
            let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

            // Physics with adaptive beta
            let physics = PhysicsParams::new(freq, speed_of_sound, density, false);
            let (beta, scale) = physics.burton_miller_beta_adaptive(radius);

            // Build system
            let system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, scale);

            // Incident field
            let incident = IncidentField::plane_wave_z();
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
            assert!(gmres_result.converged, "GMRES+ILU failed");
            let error_pct = 100.0
                * rigid_sphere_surface_relative_l2_error(
                    &gmres_result.x,
                    &elements,
                    k,
                    radius,
                    num_terms,
                );

            errors.push((*subdivision, error_pct, epw));
        }

        // Find best
        let best = errors
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let sub2_err = errors
            .iter()
            .find(|e| e.0 == 2)
            .map(|e| e.1)
            .unwrap_or(f64::NAN);
        let sub3_err = errors
            .iter()
            .find(|e| e.0 == 3)
            .map(|e| e.1)
            .unwrap_or(f64::NAN);

        let sub2_str = format!("{:>5.1}%", sub2_err);
        let sub3_str = format!("{:>5.1}%", sub3_err);

        let best_str = format!("Sub{}", best.0);

        println!(
            "{:>6.1} | {:>8.0} | {:>12} | {:>12} | {:>10}",
            ka, freq, sub2_str, sub3_str, best_str
        );

        assert!(
            errors.iter().all(|(_, error, _)| error.is_finite()),
            "surface-field errors must be finite at ka={ka}"
        );
        assert!(
            sub3_err < sub2_err,
            "area-weighted complex error must decrease under refinement at ka={ka}: sub2={sub2_err:.3}%, sub3={sub3_err:.3}%"
        );
        assert!(
            sub3_err < 1.0,
            "finest area-weighted complex surface error at ka={ka} is {sub3_err:.3}%"
        );
    }

    println!();
    println!("=== Analysis ===");
    println!();
    println!("Elements per wavelength (EPW) at each subdivision:");
    for (subdivision, mesh, avg_element_size) in &meshes {
        println!(
            "  Subdivision {}: {} elements",
            subdivision,
            mesh.elements.len()
        );
        for &ka in &[0.5, 1.0, 2.0, 3.0] {
            let k = ka / radius;
            let wavelength = 2.0 * PI / k;
            let epw = wavelength / avg_element_size;
            println!("    ka={:.1}: {:.1} elements/wavelength", ka, epw);
        }
    }

    println!();
    println!("Recommendation:");
    println!("  NumCalc recommends 6-8 elements per wavelength.");
    println!("  Use subdivision 2 (320 elem) for ka < 1.0");
    println!("  Use subdivision 3 (1280 elem) for ka >= 1.0");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_mesh_convergence();
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
