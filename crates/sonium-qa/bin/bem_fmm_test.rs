//! Fast Multipole Method Test
//!
//! Verifies that SLFMM and MLFMM matvec implementations produce correct results

fn test_fmm() {
    use num_complex::Complex64;
    use sonium_bem::core::assembly::mlfmm::{build_cluster_tree, build_mlfmm_system_stabilized};
    use sonium_bem::core::assembly::slfmm::build_slfmm_system_stabilized;
    use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::types::{BoundaryCondition, Cluster, PhysicsParams};
    use std::f64::consts::PI;

    println!("=== Fast Multipole Method Test ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka_target = 0.5;
    let frequency = ka_target * speed_of_sound / (2.0 * PI * radius);

    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let _beta_scale = 4.0;

    println!("ka = {:.2}, k = {:.2}", ka_target, physics.wave_number);
    println!();

    // Generate a small mesh
    let mesh = generate_icosphere_mesh(radius, 2);
    let n = mesh.elements.len();
    println!("Mesh: {} elements\n", n);

    // Set up elements with boundary conditions
    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    // === Test TBEM (reference) ===
    // Use stabilized version for accurate comparison
    let tbem_system = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);

    // Create a test vector
    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    // TBEM matvec: y = A * x
    let y_tbem: Vec<Complex64> = tbem_system.matrix.matvec(&x);

    println!("TBEM matvec result:");
    println!(
        "  ||y|| = {:.6}",
        y_tbem.iter().map(|v| v.norm()).sum::<f64>() / n as f64
    );

    // === Test SLFMM ===
    // Create a single cluster containing all elements
    let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
    cluster.element_indices = (0..n).collect();
    cluster.radius = radius;
    cluster.near_clusters = vec![]; // All elements are in the same cluster = all near-field
    cluster.far_clusters = vec![];

    let slfmm_system = build_slfmm_system_stabilized(
        &elements,
        &mesh.nodes,
        &[cluster],
        &physics,
        4,
        8,
        5,
        -1.0, // sign
        1.0,  // beta_scale
    );

    let y_slfmm = slfmm_system.matvec(&x);

    println!("\nSLFMM matvec result:");
    println!(
        "  ||y|| = {:.6}",
        y_slfmm.iter().map(|v| v.norm()).sum::<f64>() / n as f64
    );

    // === Test MLFMM ===
    let cluster_tree = build_cluster_tree(&elements, 20, &physics);
    let mlfmm_system = build_mlfmm_system_stabilized(
        &elements,
        &mesh.nodes,
        cluster_tree,
        &physics,
        -1.0, // sign
        1.0,  // beta_scale
    );

    println!("\nMLFMM cluster tree: {} levels", mlfmm_system.num_levels);
    for level in 0..mlfmm_system.num_levels {
        println!(
            "  Level {}: {} clusters",
            level,
            mlfmm_system.num_clusters_at_level(level)
        );
    }

    let y_mlfmm = mlfmm_system.matvec(&x);

    println!("\nMLFMM matvec result:");
    println!(
        "  ||y|| = {:.6}",
        y_mlfmm.iter().map(|v| v.norm()).sum::<f64>() / n as f64
    );

    // === Compare results ===
    let diff_slfmm: f64 = y_tbem
        .iter()
        .zip(y_slfmm.iter())
        .map(|(a, b)| (a - b).norm())
        .sum::<f64>()
        / n as f64;

    let diff_mlfmm: f64 = y_tbem
        .iter()
        .zip(y_mlfmm.iter())
        .map(|(a, b)| (a - b).norm())
        .sum::<f64>()
        / n as f64;

    println!("\n=== Comparison with TBEM ===");
    println!("SLFMM error: {:.2e}", diff_slfmm);
    println!("MLFMM error: {:.2e}", diff_mlfmm);

    // Compute relative differences
    let y_norm: f64 = y_tbem.iter().map(|v| v.norm()).sum::<f64>() / n as f64;
    let relative_diff_slfmm = diff_slfmm / y_norm.max(1e-15);
    let relative_diff_mlfmm = diff_mlfmm / y_norm.max(1e-15);

    println!("SLFMM relative error: {:.2e}", relative_diff_slfmm);
    println!("MLFMM relative error: {:.2e}", relative_diff_mlfmm);

    assert!(
        relative_diff_slfmm < 1e-12,
        "SLFMM vs TBEM relative difference {relative_diff_slfmm:.2e} exceeds roundoff"
    );
    assert!(
        relative_diff_mlfmm < 1e-12,
        "MLFMM vs TBEM relative difference {relative_diff_mlfmm:.2e} exceeds roundoff"
    );

    println!("\n=== Summary ===");
    println!("Both FMM implementations now have complete matvec operations:");
    println!("- SLFMM: Single-level with near/far decomposition");
    println!("- MLFMM: Multi-level with upward/downward passes");
    println!("\nFor production use, ensure clusters are properly partitioned");
    println!("into near and far lists based on separation criteria.");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_fmm();
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
