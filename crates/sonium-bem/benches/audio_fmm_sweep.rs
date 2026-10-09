//! Audio Frequency Sweep: CGS with ILU Preconditioning

fn main() {
    use sonium_bem::analytical::sphere_scattering_3d;
    use sonium_bem::core::assembly::tbem::build_tbem_system_scaled;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;

    use num_complex::Complex64;
    use sonium_bem::core::solver::direct::lu_solve;
    use sonium_bem::core::solver::{CgsConfig, solve_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use std::f64::consts::PI;

    println!("=== Audio Frequency Sweep: CGS + ILU Preconditioning ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    let subdivisions = 2;
    let mesh = generate_icosphere_mesh(radius, subdivisions);
    let n = mesh.elements.len();

    println!("Physical setup:");
    println!("  Sphere radius: {} m", radius);
    println!("  Speed of sound: {} m/s", speed_of_sound);
    println!("  Mesh elements: {} (subdivision {})", n, subdivisions);
    println!();

    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

    let ka_values: Vec<f64> = vec![0.5, 0.7, 1.0, 1.2, 1.5];

    println!(
        "{:>8} | {:>8} | {:>10} | {:>12} | {:>12} | {:>10} | {:>8}",
        "Freq Hz", "ka", "Mie avg", "Direct err%", "CGS avg", "CGS err%", "Iters"
    );
    println!("{}", "-".repeat(85));

    for &ka in &ka_values {
        let freq = ka * speed_of_sound / (2.0 * PI * radius);
        let k = ka / radius;
        let scale = 4.0;

        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);
        let beta = physics.burton_miller_beta_scaled(scale);

        let num_terms = (ka as usize + 20).max(30);
        let mie = sphere_scattering_3d(k, radius, num_terms, vec![radius], vec![0.0, PI / 2.0, PI]);
        if mie.pressure.iter().any(|p| !p.is_finite()) {
            println!(
                "{:>8.0} | {:>8.2} | {:>10} | {:>12} | {:>12} | {:>10} | {:>8}",
                freq, ka, "NaN", "-", "-", "-", "-"
            );
            continue;
        }
        let mie_avg =
            (mie.pressure[0].norm() + mie.pressure[1].norm() + mie.pressure[2].norm()) / 3.0;

        let incident = IncidentField::plane_wave_z();
        let p_inc = incident.evaluate_pressure(&centers, &physics);
        let dpdn_inc = incident.evaluate_normal_derivative(&centers, &normals, &physics);

        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            rhs[i] = p_inc[i] + beta * dpdn_inc[i];
        }

        let tbem_system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, scale);

        let tbem_solution_x = lu_solve(&tbem_system.matrix, &rhs).expect("Solver failed");
        let direct_avg: f64 = tbem_solution_x.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
        let direct_err = 100.0 * (direct_avg - mie_avg).abs() / mie_avg;

        let print_interval = if ka == ka_values[0] { 100 } else { 0 };
        let cgs_config = CgsConfig {
            max_iterations: 500,
            tolerance: 1e-6,
            print_interval,
        };

        let cgs_solution = solve_with_ilu(&tbem_system.matrix, &rhs, &cgs_config);

        let cgs_avg: f64 = cgs_solution.x.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
        let cgs_err = 100.0 * (cgs_avg - mie_avg).abs() / mie_avg;

        let status = if cgs_solution.converged { "OK" } else { "FAIL" };

        println!(
            "{:>8.0} | {:>8.3} | {:>10.4} | {:>11.1}% | {:>12.4} | {:>9.1}% | {:>6} {}",
            freq, ka, mie_avg, direct_err, cgs_avg, cgs_err, cgs_solution.iterations, status
        );
    }
}
