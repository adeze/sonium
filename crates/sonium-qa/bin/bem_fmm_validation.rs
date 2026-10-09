//! FMM Validation: Compare SLFMM matvec vs direct TBEM
//!
//! This validates that the FMM implementation produces the same
//! results as the direct TBEM assembly for matrix-vector products.

fn test_fmm_validation() {
    use num_complex::Complex64;
    use sonium_bem::core::assembly::slfmm::build_slfmm_system_stabilized;
    use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::types::{BoundaryCondition, Cluster, PhysicsParams};
    use sonium_qa::relative_error;
    use std::f64::consts::PI;
    use std::time::Instant;

    println!("=== FMM Validation: SLFMM vs TBEM ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    // Test different mesh sizes
    let subdivisions = [2]; // Start with small mesh

    for &subdiv in &subdivisions {
        let mesh = generate_icosphere_mesh(radius, subdiv);
        let n = mesh.elements.len();

        println!("Mesh: {} elements (subdivision {})", n, subdiv);

        // Prepare elements
        let mut elements = mesh.elements.clone();
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
        }

        // Test at ka = 1.0
        let ka = 1.0;
        let freq = ka * speed_of_sound / (2.0 * PI * radius);
        let physics = PhysicsParams::new(freq, speed_of_sound, density, false);
        let _scale = 1.0;

        println!("  ka = {:.1}, freq = {:.0} Hz", ka, freq);

        // Build TBEM system
        let start = Instant::now();
        // Use stabilized version for accurate results
        let tbem_system = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);
        let tbem_time = start.elapsed();
        println!("  TBEM assembly: {:.2}ms", tbem_time.as_secs_f64() * 1000.0);

        // Create a single cluster containing all elements (simplest case for validation)
        let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
        cluster.element_indices = (0..n).collect();
        cluster.radius = radius * 2.0;
        cluster.near_clusters = vec![]; // Self-interaction is always near
        cluster.far_clusters = vec![];
        let clusters = vec![cluster];
        println!(
            "  Clusters: {} (all elements in one cluster for validation)",
            clusters.len()
        );

        // Build SLFMM system
        let start = Instant::now();
        let slfmm_system = build_slfmm_system_stabilized(
            &elements,
            &mesh.nodes,
            &clusters,
            &physics,
            8,    // n_theta
            16,   // n_phi
            10,   // n_terms
            -1.0, // sign
            1.0,  // beta_scale
        );
        let slfmm_time = start.elapsed();
        println!(
            "  SLFMM assembly: {:.2}ms",
            slfmm_time.as_secs_f64() * 1000.0
        );

        // Create test vectors
        let mut x_real: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
        let mut x_imag: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
        let mut x_rand: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];

        for i in 0..n {
            x_real[i] = Complex64::new(1.0, 0.0);
            x_imag[i] = Complex64::new(0.0, 1.0);
            x_rand[i] = Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos());
        }

        // Compare matvec results
        println!("\n  Matvec comparison:");

        // Track max relative error across all test vectors
        let mut max_relative_error = 0.0_f64;

        for (name, x) in [("ones", &x_real), ("i*ones", &x_imag), ("random", &x_rand)] {
            // TBEM matvec (dense)
            let start = Instant::now();
            let y_tbem = tbem_system.matrix.matvec(x);
            let tbem_mv_time = start.elapsed();

            // SLFMM matvec
            let start = Instant::now();
            let y_slfmm = slfmm_system.matvec(x);
            let slfmm_mv_time = start.elapsed();

            let rel_error = relative_error(&y_slfmm, &y_tbem);

            if rel_error > max_relative_error {
                max_relative_error = rel_error;
            }

            println!(
                "    {:<8}: rel_error={:.2e} (TBEM:{:.2}ms, SLFMM:{:.2}ms)",
                name,
                rel_error,
                tbem_mv_time.as_secs_f64() * 1000.0,
                slfmm_mv_time.as_secs_f64() * 1000.0,
            );
        }

        assert!(
            max_relative_error < 1e-12,
            "single-cluster SLFMM must reproduce the dense near field to roundoff; error={max_relative_error:.2e}"
        );

        println!();
    }

    println!("=== Analysis ===");
    println!("If relative errors are <1%, SLFMM near-field is correct.");
    println!("A single cluster with all near-field tests the core SLFMM logic.");
    println!("Multi-cluster tests would validate far-field (D-matrix) translations.");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_fmm_validation();
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
