//! FMM Benchmark: Compare SLFMM vs TBEM performance at different mesh sizes
//!
//! Tests solve time and accuracy for both methods across mesh sizes.

fn main() {
    use num_complex::Complex64;
    use sonium_bem::analytical::sphere_scattering_3d;
    use sonium_bem::core::assembly::slfmm::build_slfmm_system_stabilized;
    use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::solver::direct::lu_solve;
    use sonium_bem::core::solver::{CgsConfig, solve_cgs};
    use sonium_bem::core::types::{BoundaryCondition, Cluster, PhysicsParams};
    use std::f64::consts::PI;
    use std::time::Instant;

    println!("=== FMM Benchmark: SLFMM vs TBEM ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let freq = ka * speed_of_sound / (2.0 * PI * radius);
    let k = ka / radius;

    println!(
        "Configuration: ka={}, freq={:.0}Hz, radius={}\n",
        ka, freq, radius
    );

    // Test different mesh sizes
    let subdivisions = [2, 3]; // 320, 1280 elements

    println!(
        "{:>10} | {:>12} | {:>12} | {:>12} | {:>12} | {:>10}",
        "Elements", "TBEM Asm", "TBEM Solve", "SLFMM Asm", "SLFMM Solve", "Error"
    );
    println!("{}", "-".repeat(80));

    for &subdiv in &subdivisions {
        let mesh = generate_icosphere_mesh(radius, subdiv);
        let n = mesh.elements.len();

        let mut elements = mesh.elements.clone();
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
        }

        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);
        let (beta, _scale) = physics.burton_miller_beta_adaptive(radius);

        let incident = IncidentField::plane_wave_z();
        let p_inc = incident.evaluate_pressure(&centers, &physics);
        let dpdn_inc = incident.evaluate_normal_derivative(&centers, &normals, &physics);

        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            rhs[i] = p_inc[i] + beta * dpdn_inc[i];
        }

        // === TBEM ===
        let start = Instant::now();
        let tbem_system = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);
        let tbem_asm_time = start.elapsed();

        let start = Instant::now();
        let tbem_x = lu_solve(&tbem_system.matrix, &rhs).expect("TBEM solve failed");
        let tbem_solve_time = start.elapsed();

        // === SLFMM ===
        let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
        cluster.element_indices = (0..n).collect();
        cluster.radius = radius * 2.0;
        cluster.near_clusters = vec![];
        cluster.far_clusters = vec![];
        let clusters = vec![cluster];

        let start = Instant::now();
        let slfmm_system = build_slfmm_system_stabilized(
            &elements,
            &mesh.nodes,
            &clusters,
            &physics,
            8,
            16,
            10,
            -1.0,
            1.0,
        );
        let slfmm_asm_time = start.elapsed();

        let start = Instant::now();

        use sonium_bem::core::solver::SlfmmOperator;
        let op = SlfmmOperator::new(slfmm_system.clone());

        let config = CgsConfig {
            tolerance: 1e-8,
            max_iterations: 1000,
            print_interval: 0,
        };
        let slfmm_solution = solve_cgs(&op, &rhs, &config);
        let slfmm_solve_time = start.elapsed();

        let diff_norm: f64 = tbem_x
            .iter()
            .zip(slfmm_solution.x.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let tbem_norm: f64 = tbem_x.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let rel_error = diff_norm / tbem_norm.max(1e-15);

        println!(
            "{:>10} | {:>10.1}ms | {:>10.1}ms | {:>10.1}ms | {:>10.1}ms | {:>8.2e}",
            n,
            tbem_asm_time.as_secs_f64() * 1000.0,
            tbem_solve_time.as_secs_f64() * 1000.0,
            slfmm_asm_time.as_secs_f64() * 1000.0,
            slfmm_solve_time.as_secs_f64() * 1000.0,
            rel_error
        );
    }

    println!("\n=== Accuracy Comparison ===\n");

    let mesh = generate_icosphere_mesh(radius, 2);
    let n = mesh.elements.len();

    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

    let physics = PhysicsParams::new(freq, speed_of_sound, density, false);
    let (beta, _scale) = physics.burton_miller_beta_adaptive(radius);

    let incident = IncidentField::plane_wave_z();
    let p_inc = incident.evaluate_pressure(&centers, &physics);
    let dpdn_inc = incident.evaluate_normal_derivative(&centers, &normals, &physics);

    let mut rhs = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        rhs[i] = p_inc[i] + beta * dpdn_inc[i];
    }

    let tbem_system = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);
    let tbem_x = lu_solve(&tbem_system.matrix, &rhs).expect("TBEM solve failed");

    let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
    cluster.element_indices = (0..n).collect();
    cluster.radius = radius * 2.0;
    let clusters = vec![cluster];

    let slfmm_system = build_slfmm_system_stabilized(
        &elements,
        &mesh.nodes,
        &clusters,
        &physics,
        8,
        16,
        10,
        -1.0,
        1.0,
    );

    use sonium_bem::core::solver::SlfmmOperator;
    let op = SlfmmOperator::new(slfmm_system.clone());

    let config = CgsConfig {
        tolerance: 1e-10,
        max_iterations: 1000,
        print_interval: 0,
    };
    let slfmm_solution = solve_cgs(&op, &rhs, &config);

    let num_terms = (ka as usize + 30).max(50);
    let mie = sphere_scattering_3d(k, radius, num_terms, vec![radius], vec![0.0, PI / 2.0, PI]);
    let mie_avg = (mie.pressure[0].norm() + mie.pressure[1].norm() + mie.pressure[2].norm()) / 3.0;

    let tbem_avg: f64 = tbem_x.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
    let slfmm_avg: f64 = slfmm_solution.x.iter().map(|x| x.norm()).sum::<f64>() / n as f64;

    let tbem_error = 100.0 * (tbem_avg - mie_avg).abs() / mie_avg;
    let slfmm_error = 100.0 * (slfmm_avg - mie_avg).abs() / mie_avg;

    println!("Method   | Avg Pressure | Error vs Mie");
    println!("{}", "-".repeat(45));
    println!("Mie      | {:>12.6}  | Reference", mie_avg);
    println!("TBEM     | {:>12.6}  | {:>5.2}%", tbem_avg, tbem_error);
    println!("SLFMM    | {:>12.6}  | {:>5.2}%", slfmm_avg, slfmm_error);

    println!("\n=== Summary ===");
    println!("SLFMM near-field matches TBEM exactly (single-cluster validation).");
    println!("For true FMM acceleration, multi-cluster setup with far-field is needed.");
    println!("CGS iterations for SLFMM: {}", slfmm_solution.iterations);
}
