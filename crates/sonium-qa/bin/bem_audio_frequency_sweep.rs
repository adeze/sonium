//! Audio Frequency Sweep: Pure-Rust BEM vs Analytical (Mie Theory)
//!
//! Tests the pure-Rust TBEM solver with **adaptive β tuning** across audio
//! frequency range for rigid sphere scattering, comparing with exact Mie series.
//!
//! ## Adaptive β Tuning
//!
//! The Burton-Miller parameter β is automatically optimized based on ka:
//!
//! | ka range   | β scale | Typical error |
//! |------------|---------|---------------|
//! | < 0.85     | 32.0    | 0.4-27%       |
//! | 0.85-0.92  | 8.0     | 0.2-7%        |
//! | 0.92-1.2   | 4.0     | 1.8-4%        |
//! | 1.2-1.8    | 8.0     | 3-10%         |
//! | > 1.8      | 16.0    | 3-10%         |
//!
//! This provides up to **59x better accuracy** than fixed β=4 at certain frequencies.
//!
//! ## Key Features
//!
//! - **Adaptive β**: `physics.burton_miller_beta_adaptive(radius)`
//! - **Direct TBEM assembly**: O(N²) complexity
//! - **Direct LU solver**: O(N³) complexity
//!
//! ## Accuracy vs NumCalc
//!
//! | Method | Low ka (<0.8) | ka≈0.9 | ka≈1.0 | High ka |
//! |--------|---------------|--------|--------|---------|
//! | Fixed β=4 | 50-200% | 12% | 2% | 50-200% |
//! | Adaptive β | 5-27% | 0.2% | 2% | 3-10% |
//! | NumCalc | <0.001% | <0.001% | <0.001% | <0.001% |
//!
//! NumCalc achieves higher accuracy via FMM + fine meshes.
//!
//! ## Key Parameters
//! - ka = wavenumber × radius (dimensionless frequency)
//! - For r=0.1m: ka=1.0 corresponds to ~546 Hz

fn test_audio_frequency_sweep() {
    use num_complex::Complex64;

    use sonium_bem::core::assembly::tbem::build_tbem_system_scaled;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::io::write_surface_pressure_vtu;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
    use sonium_bem::core::types::{BasisOrder, BoundaryCondition, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;
    use sonium_xem::io::{PvdCollection, binary_output_dir};
    use std::f64::consts::PI;

    let out_dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let mut pvd = PvdCollection::new(out_dir.join("audio_sweep.pvd"));

    println!("=== Audio Frequency Sweep: BEM vs Analytical (Mie Theory) ===\n");

    // Physical parameters
    let radius = 0.1; // 10 cm sphere
    let speed_of_sound = 343.0; // m/s (air at ~20°C)
    let density = 1.21; // kg/m³

    // Mesh: subdivision 2 gives 320 elements
    let subdivisions = 2;

    let mesh = generate_icosphere_mesh(radius, subdivisions);
    let n = mesh.elements.len();

    // Estimate average element size
    let sphere_area = 4.0 * PI * radius * radius;
    let avg_element_area = sphere_area / n as f64;
    let avg_element_size = avg_element_area.sqrt();

    println!("Physical setup:");
    println!("  Sphere radius: {} m", radius);
    println!("  Speed of sound: {} m/s", speed_of_sound);
    println!("  Mesh elements: {} (subdivision {})", n, subdivisions);
    println!("  Avg element size: {:.4} m", avg_element_size);
    println!();

    // Key frequency-ka relationships for r=0.1m:
    // ka = 2πf × r / c = f × 0.00183
    // f = ka × 546 Hz
    println!("Frequency-ka relationship (for r=0.1m):");
    println!(
        "  ka=0.5 → {:>6.0} Hz",
        0.5 * speed_of_sound / (2.0 * PI * radius)
    );
    println!(
        "  ka=1.0 → {:>6.0} Hz (optimal range)",
        1.0 * speed_of_sound / (2.0 * PI * radius)
    );
    println!(
        "  ka=2.0 → {:>6.0} Hz",
        2.0 * speed_of_sound / (2.0 * PI * radius)
    );
    println!();

    // Prepare elements once
    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    // Build centers/normals arrays
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

    // Test across ka range (focus on audio frequencies)
    // Using ka values directly to show the limitation
    let ka_values: Vec<f64> = vec![0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0];

    println!(
        "{:>8} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Freq Hz", "ka", "β scale", "Metric", "Field", "Error %", "Status"
    );
    println!("{}", "-".repeat(85));

    let mut max_error_pct = 0.0f64;
    for &ka in &ka_values {
        // Convert ka to frequency: f = ka × c / (2π × r)
        let freq = ka * speed_of_sound / (2.0 * PI * radius);
        let k = ka / radius;
        let wavelength = speed_of_sound / freq;
        let _epw = wavelength / avg_element_size; // Elements per wavelength

        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);

        // Use adaptive β scale based on ka (automatically optimized)
        let (beta, scale) = physics.burton_miller_beta_adaptive(radius);

        let num_terms = (ka as usize + 20).max(30);

        // Build BEM system
        let system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, scale);

        // Incident field (plane wave in +z direction)
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
        let solution_x = gmres_result.x;

        let vtu_name = format!("audio_sweep_ka{:.2}.vtu", ka);
        let _ = write_surface_pressure_vtu(
            &mesh,
            &solution_x,
            BasisOrder::Constant,
            out_dir.join(&vtu_name),
        );
        pvd.push(freq, vtu_name);

        let error_pct = 100.0
            * rigid_sphere_surface_relative_l2_error(&solution_x, &elements, k, radius, num_terms);
        max_error_pct = max_error_pct.max(error_pct);

        let status = if error_pct < 3.0 { "✓" } else { "✗" };

        println!(
            "{:>8.0} | {:>8.3} | {:>8.1} | {:>10.4} | {:>10.4} | {:>9.1}% | {}",
            freq, ka, scale, "weighted", "complex", error_pct, status
        );
    }

    println!();
    println!("=== Legend ===");
    println!("  ka: wavenumber × radius (dimensionless frequency)");
    println!("  β scale: Burton-Miller parameter scaling (β = scale × i/k)");
    println!("  ✓ = area-weighted complex error < 3%, ✗ = failure");
    println!();
    println!("=== Adaptive β Tuning (NOW ENABLED) ===");
    println!("  Using: physics.burton_miller_beta_adaptive(radius)");
    println!("  - all ka:       β_scale = 0.2 (measured stable optimum)");
    println!();
    let _ = pvd.finish();
    assert!(
        max_error_pct < 3.0,
        "Maximum area-weighted complex surface error {:.1}% exceeds 3%",
        max_error_pct
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_audio_frequency_sweep();
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
