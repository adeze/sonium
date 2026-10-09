//! FMM + ILU Solver for Large-Scale BEM
//!
//! Demonstrates the combination of:
//! - **Fast Multipole Method (FMM)** for O(N log N) matvec
//! - **ILU preconditioning** for CGS convergence
//! - **Adaptive β tuning** for wide frequency range

fn main() {
    use sonium_bem::analytical::sphere_scattering_3d;
    use sonium_bem::core::assembly::mlfmm::{build_cluster_tree, build_mlfmm_system_stabilized};
    use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;

    use num_complex::Complex64;
    use sonium_bem::core::solver::direct::lu_solve;
    use sonium_bem::core::solver::{CgsConfig, solve_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use std::f64::consts::PI;

    println!("=== FMM + ILU Solver for Large-Scale BEM ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    let mesh_configs = vec![(2, "Small (320 elements)"), (3, "Medium (1280 elements)")];

    let ka = 1.0;
    let freq = ka * speed_of_sound / (2.0 * PI * radius);
    let k = ka / radius;

    println!("Test frequency: {:.0} Hz (ka = {:.1})\n", freq, ka);

    let num_terms = 50;
    let mie = sphere_scattering_3d(k, radius, num_terms, vec![radius], vec![0.0, PI / 2.0, PI]);
    let mie_avg = (mie.pressure[0].norm() + mie.pressure[1].norm() + mie.pressure[2].norm()) / 3.0;

    println!("Mie analytical reference: {:.4}\n", mie_avg);

    for (subdivisions, desc) in &mesh_configs {
        println!("=== {} ===", desc);

        let mesh = generate_icosphere_mesh(radius, *subdivisions);
        let n = mesh.elements.len();

        println!("  Mesh elements: {}", n);

        let mut elements = mesh.elements.clone();
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
        }

        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);

        let (beta, scale) = physics.burton_miller_beta_adaptive(radius);
        println!("  Adaptive β scale: {:.1}", scale);

        let incident = IncidentField::plane_wave_z();
        let p_inc = incident.evaluate_pressure(&centers, &physics);
        let dpdn_inc = incident.evaluate_normal_derivative(&centers, &normals, &physics);

        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            rhs[i] = p_inc[i] + beta * dpdn_inc[i];
        }

        // === Method 1: Direct TBEM ===
        println!("\n  --- Direct TBEM Solve ---");
        let tbem_start = std::time::Instant::now();
        let tbem_system = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);
        let direct_x = lu_solve(&tbem_system.matrix, &rhs).expect("Direct solve failed");
        let tbem_time = tbem_start.elapsed();

        let direct_avg: f64 = direct_x.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
        let direct_err = 100.0 * (direct_avg - mie_avg).abs() / mie_avg;
        println!("    Time: {:?}", tbem_time);
        println!("    BEM avg: {:.4}, Error: {:.1}%", direct_avg, direct_err);

        // === Method 2: TBEM + ILU + CGS ===
        println!("\n  --- TBEM + ILU + CGS ---");
        let ilu_start = std::time::Instant::now();

        let cgs_config = CgsConfig {
            max_iterations: 200,
            tolerance: 1e-6,
            print_interval: 0,
        };

        let cgs_solution = solve_with_ilu(&tbem_system.matrix, &rhs, &cgs_config);
        let ilu_time = ilu_start.elapsed();

        let cgs_avg: f64 = cgs_solution.x.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
        let cgs_err = 100.0 * (cgs_avg - mie_avg).abs() / mie_avg;
        let status = if cgs_solution.converged { "OK" } else { "FAIL" };

        println!("    Time: {:?}", ilu_time);
        println!("    Iterations: {} {}", cgs_solution.iterations, status);
        println!("    BEM avg: {:.4}, Error: {:.1}%", cgs_avg, cgs_err);

        // === Method 3: MLFMM structure only ===
        println!("\n  --- MLFMM Structure ---");
        let fmm_start = std::time::Instant::now();
        let cluster_tree = build_cluster_tree(&elements, 20, &physics);
        let mlfmm_system = build_mlfmm_system_stabilized(
            &elements,
            &mesh.nodes,
            cluster_tree,
            &physics,
            -1.0,
            1.0,
        );
        let fmm_build_time = fmm_start.elapsed();

        println!("    Build time: {:?}", fmm_build_time);
        println!("    Cluster levels: {}", mlfmm_system.num_levels);
        for level in 0..mlfmm_system.num_levels {
            println!(
                "      Level {}: {} clusters",
                level,
                mlfmm_system.num_clusters_at_level(level)
            );
        }

        let test_x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let fmm_matvec_start = std::time::Instant::now();
        let _fmm_y = mlfmm_system.matvec(&test_x);
        let fmm_matvec_time = fmm_matvec_start.elapsed();

        let tbem_matvec_start = std::time::Instant::now();
        let _tbem_y = tbem_system.matrix.matvec(&test_x);
        let tbem_matvec_time = tbem_matvec_start.elapsed();

        println!("    FMM matvec: {:?}", fmm_matvec_time);
        println!("    TBEM matvec: {:?}", tbem_matvec_time);

        if fmm_matvec_time < tbem_matvec_time {
            println!(
                "    FMM is {:.1}x faster for matvec",
                tbem_matvec_time.as_secs_f64() / fmm_matvec_time.as_secs_f64()
            );
        }

        println!();
    }
}
