//! Residual Check - Verify solver is working correctly
//!
//! Checks ||Ap - b|| to ensure the linear system is being solved correctly.
//! If residual is small but solution is wrong, the issue is in formulation or RHS.

fn test_residual_check() {
    use num_complex::Complex64;
    use sonium_bem::core::assembly::tbem::build_tbem_system_scaled;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;
    use std::f64::consts::PI;

    println!("=== Residual Check ===");
    println!("Verifying ||Ap - b|| to ensure solver works correctly\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka_target = 1.0;
    let frequency = ka_target * speed_of_sound / (2.0 * PI * radius);
    let k = 2.0 * PI * frequency / speed_of_sound;

    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let beta = physics.burton_miller_beta_scaled(2.0);

    let mesh = generate_icosphere_mesh(radius, 2);
    let n = mesh.elements.len();

    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    // Build system
    let system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, 2.0);

    // Compute RHS
    let incident = IncidentField::plane_wave_z();
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();
    let p_inc = incident.evaluate_pressure(&centers, &physics);
    let dpdn_inc = incident.evaluate_normal_derivative(&centers, &normals, &physics);
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

    // Compute residual Ap - b
    let mut residual: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        let mut ax_i = Complex64::new(0.0, 0.0);
        for j in 0..n {
            ax_i += system.matrix[[i, j]] * solution_x[j];
        }
        residual[i] = ax_i - rhs[i];
    }

    // Compute norms
    let residual_norm = residual.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let rhs_norm = rhs.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let relative_residual = residual_norm / rhs_norm;
    let solution_norm: f64 = solution_x.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
    let surface_error =
        rigid_sphere_surface_relative_l2_error(&solution_x, &elements, k, radius, 50);

    println!("ka = {:.2}, {} elements, β = {:.4}i", ka_target, n, beta.im);
    println!();
    println!("||b||₂ = {:.6}", rhs_norm);
    println!(
        "||x|| / n = {:.6} (average solution magnitude)",
        solution_norm
    );
    println!("||Ax - b||₂ = {:.10}", residual_norm);
    println!("Relative residual = {:.2e}", relative_residual);
    println!(
        "Area-weighted complex Mie surface error = {:.3}%",
        100.0 * surface_error
    );

    // Check some individual elements
    println!("\nSample residuals:");
    for i in [0, n / 4, n / 2, 3 * n / 4, n - 1] {
        println!(
            "  elem[{}]: residual = {:.2e} + {:.2e}i, |rhs| = {:.4}, |x| = {:.4}",
            i,
            residual[i].re,
            residual[i].im,
            rhs[i].norm(),
            solution_x[i].norm()
        );
    }

    // Verify the incident field RHS values
    println!("\n=== Incident Field Values ===");
    println!("Front element sample:");
    for (i, elem) in elements.iter().enumerate() {
        if elem.center[2] > 0.08 {
            println!(
                "  elem[{}]: z = {:.4}, |p_inc| = {:.4}, |dpdn| = {:.4}",
                i,
                elem.center[2],
                p_inc[i].norm(),
                dpdn_inc[i].norm()
            );
            println!(
                "           |rhs| = {:.4} = |-(p_inc + β*dpdn)|",
                rhs[i].norm()
            );
            break;
        }
    }

    println!("\nBack element sample:");
    for (i, elem) in elements.iter().enumerate() {
        if elem.center[2] < -0.08 {
            println!(
                "  elem[{}]: z = {:.4}, |p_inc| = {:.4}, |dpdn| = {:.4}",
                i,
                elem.center[2],
                p_inc[i].norm(),
                dpdn_inc[i].norm()
            );
            println!(
                "           |rhs| = {:.4} = |-(p_inc + β*dpdn)|",
                rhs[i].norm()
            );
            break;
        }
    }

    assert!(
        relative_residual < 1e-8,
        "true relative residual is {relative_residual:.3e}"
    );
    assert!(
        surface_error < 0.015,
        "area-weighted complex surface error is {:.3}%",
        100.0 * surface_error
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_residual_check();
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
