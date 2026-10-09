//! Slow BEM FMM validation sweeps.
//!
//! These checks used to live in `sonium-bem` unit tests. They compare
//! FMM matvecs against dense TBEM references and are intentionally outside
//! the default `cargo test -p sonium-bem` path.
//!
//! Run with:
//! ```sh
//! cargo run -p sonium-qa --release --bin qa-bem-fmm-slow-validation
//! ```

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{build_cluster_tree, build_mlfmm_system_stabilized};
use sonium_bem::core::assembly::slfmm::build_slfmm_system_stabilized;
use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::mesh::octree::Octree;
use sonium_bem::core::types::{BoundaryCondition, Cluster, ClusterLevel, Element, PhysicsParams};
use sonium_qa::{QaResult, relative_error};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

struct ValidationOutcome {
    passed: bool,
    elapsed_ms: u64,
}

fn deterministic_vector(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let t = (i + 1) as f64;
            Complex64::new((0.37 * t).sin(), (0.19 * t).cos())
        })
        .collect()
}

fn with_velocity_dofs(elements: &mut [Element]) {
    for (i, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![i];
    }
}

fn measure_single_pair_m2l_precision(
    radius: f64,
    k_target: f64,
    separation_radii: f64,
    expansion_terms: usize,
    phi_factor: usize,
) -> (f64, f64) {
    let speed_of_sound = 343.0;
    let density = 1.21;
    let frequency = k_target * speed_of_sound / (2.0 * PI);
    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

    let mesh_a = generate_icosphere_mesh(radius, 1);
    let mesh_b = generate_icosphere_mesh(radius, 1);
    let n_a = mesh_a.elements.len();
    let n_b = mesh_b.elements.len();
    let separation = separation_radii * radius;

    let total_nodes = mesh_a.nodes.len() + mesh_b.nodes.len();
    let mut combined_nodes: Vec<[f64; 3]> = vec![[0.0; 3]; total_nodes];
    for i in 0..mesh_a.nodes.len() {
        for d in 0..3 {
            combined_nodes[i][d] = mesh_a.nodes[i][d];
        }
    }
    for i in 0..mesh_b.nodes.len() {
        combined_nodes[mesh_a.nodes.len() + i][0] = mesh_b.nodes[i][0] + separation;
        combined_nodes[mesh_a.nodes.len() + i][1] = mesh_b.nodes[i][1];
        combined_nodes[mesh_a.nodes.len() + i][2] = mesh_b.nodes[i][2];
    }

    let mut elements = Vec::with_capacity(n_a + n_b);
    for (i, e) in mesh_a.elements.iter().enumerate() {
        let mut element = e.clone();
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![i];
        elements.push(element);
    }
    for (i, e) in mesh_b.elements.iter().enumerate() {
        let mut element = e.clone();
        for c in element.connectivity.iter_mut() {
            *c += mesh_a.nodes.len();
        }
        element.center[0] += separation;
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![n_a + i];
        elements.push(element);
    }
    let n = elements.len();

    let tbem = build_tbem_system_stabilized(&elements, &combined_nodes, &physics, -1.0, 1.0);

    let mut leaf_a = Cluster::new([0.0, 0.0, 0.0]);
    leaf_a.element_indices = (0..n_a).collect();
    leaf_a.num_elements = n_a;
    leaf_a.radius = radius * 1.05;
    leaf_a.level = 1;
    leaf_a.father = Some(0);
    leaf_a.near_clusters = vec![0];
    leaf_a.far_clusters = vec![1];

    let mut leaf_b = Cluster::new([separation, 0.0, 0.0]);
    leaf_b.element_indices = (n_a..n).collect();
    leaf_b.num_elements = n_b;
    leaf_b.radius = radius * 1.05;
    leaf_b.level = 1;
    leaf_b.father = Some(0);
    leaf_b.near_clusters = vec![1];
    leaf_b.far_clusters = vec![0];

    let mut leaf_level = ClusterLevel::new(2);
    leaf_level.clusters = vec![leaf_a, leaf_b];
    leaf_level.num_original = 2;
    leaf_level.expansion_terms = expansion_terms;
    leaf_level.theta_points = expansion_terms;
    leaf_level.phi_points = phi_factor * expansion_terms + 2;

    let mut root = Cluster::new([separation / 2.0, 0.0, 0.0]);
    root.element_indices = (0..n).collect();
    root.num_elements = n;
    root.radius = (separation / 2.0) + radius;
    root.level = 0;
    root.sons = vec![0, 1];

    let mut root_level = ClusterLevel::new(1);
    root_level.clusters = vec![root];
    root_level.num_original = 1;
    root_level.expansion_terms = expansion_terms;
    root_level.theta_points = expansion_terms;
    root_level.phi_points = phi_factor * expansion_terms + 2;

    let system = build_mlfmm_system_stabilized(
        &elements,
        &combined_nodes,
        vec![root_level, leaf_level],
        &physics,
        -1.0,
        1.0,
    );

    let x = deterministic_vector(n);
    let y_dense = tbem.matrix.matvec(&x);
    let y_mlfmm = system.matvec(&x);
    (relative_error(&y_mlfmm, &y_dense), k_target * separation)
}

fn validate_mlfmm_per_pair_precision_sweep() {
    let radius = 0.22;
    let separation_radii = 4.0;
    println!(
        "[MLFMM per-pair sweep] radius={radius}, sep={separation_radii}R, expansion=28, phi=58"
    );
    for k_target in [0.5_f64, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0] {
        let (rel_err, kr) =
            measure_single_pair_m2l_precision(radius, k_target, separation_radii, 28, 2);
        println!("  k={k_target:>5.1}, kr={kr:>6.2}, rel_err={rel_err:.3e}");
        assert!(
            rel_err < 1e-4,
            "MLFMM pair translation must match dense TBEM within 0.01% at k={k_target}, kr={kr}; error={rel_err:.3e}"
        );
    }
}

fn validate_mlfmm_dense_ka_sweep() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;

    let mesh = generate_icosphere_mesh(radius, 3);
    let mut elements = mesh.elements.clone();
    with_velocity_dofs(&mut elements);
    let n = elements.len();

    println!("[MLFMM dense ka sweep] n={n}");
    for ka in [1.0_f64, 2.0, 5.0, 10.0] {
        let frequency = ka * speed_of_sound / (2.0 * PI * radius);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        let tbem = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);
        let tree = build_cluster_tree(&elements, 16, &physics);
        let n_levels = tree.len();
        let expansion_terms = tree
            .first()
            .expect("MLFMM cluster-tree builder must produce a root level")
            .expansion_terms;
        let system =
            build_mlfmm_system_stabilized(&elements, &mesh.nodes, tree, &physics, -1.0, 1.0);

        let x = deterministic_vector(n);
        let y_dense = tbem.matrix.matvec(&x);
        let y_mlfmm = system.matvec(&x);
        let rel_err = relative_error(&y_mlfmm, &y_dense);

        println!(
            "  ka={ka:>5.1}  levels={n_levels}  expansion_terms={expansion_terms}  rel_err={rel_err:>12.3e}",
        );
        assert!(
            rel_err < 1e-2,
            "MLFMM matvec must match dense to <= 1% at ka={ka} (n={n}), got {rel_err:.3e}"
        );
    }
}

fn slfmm_case(
    radius: f64,
    frequency: f64,
    separation_ratio: f64,
) -> (
    Vec<Element>,
    Vec<[f64; 3]>,
    PhysicsParams,
    Vec<Cluster>,
    usize,
) {
    let mesh = generate_icosphere_mesh(radius, 2);
    let mut elements = mesh.elements.clone();
    with_velocity_dofs(&mut elements);

    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let mut octree = Octree::build(&centers, 16, 8);
    octree.compute_interaction_lists(separation_ratio);

    let leaf_indices = octree.leaves();
    let mut node_to_cluster = HashMap::new();
    for (cluster_idx, &node_idx) in leaf_indices.iter().enumerate() {
        node_to_cluster.insert(node_idx, cluster_idx);
    }

    let mut clusters = Vec::new();
    let mut total_far_pairs = 0;
    for &node_idx in &leaf_indices {
        let node = &octree.nodes[node_idx];
        let mut cluster = Cluster::new(node.center);
        cluster.element_indices = node.element_indices.clone();
        cluster.num_elements = node.element_indices.len();
        cluster.radius = node.radius();
        cluster.near_clusters = node
            .near_clusters
            .iter()
            .filter_map(|ni| node_to_cluster.get(ni).copied())
            .collect();
        cluster.far_clusters = node
            .far_clusters
            .iter()
            .filter_map(|ni| node_to_cluster.get(ni).copied())
            .collect();
        total_far_pairs += cluster.far_clusters.len();
        clusters.push(cluster);
    }

    assert!(
        total_far_pairs > 0,
        "SLFMM slow validation requires at least one far cluster pair"
    );

    (
        elements,
        mesh.nodes.clone(),
        PhysicsParams::new(frequency, 343.0, 1.21, false),
        clusters,
        total_far_pairs,
    )
}

fn validate_slfmm_multi_cluster_far_field_vs_tbem() {
    let (elements, nodes, physics, clusters, total_far_pairs) = slfmm_case(0.1, 500.0, 1.8);
    let n = elements.len();
    let tbem = build_tbem_system_stabilized(&elements, &nodes, &physics, -1.0, 1.0);
    let slfmm = build_slfmm_system_stabilized(
        &elements, &nodes, &clusters, &physics, 20, 40, 10, -1.0, 1.0,
    );

    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()))
        .collect();
    let y_tbem = tbem.matrix.matvec(&x);
    let y_slfmm = slfmm.matvec(&x);
    let rel_error = relative_error(&y_slfmm, &y_tbem);

    println!(
        "[SLFMM multi-cluster] n={n}, clusters={}, far_pairs={total_far_pairs}, rel_err={rel_error:.3e}",
        clusters.len(),
    );
    assert!(
        rel_error < 1e-4,
        "Multi-cluster SLFMM vs TBEM rel_err={rel_error:.3e} exceeds 0.01%"
    );
}

fn validate_slfmm_far_field_convergence() {
    let (elements, nodes, physics, clusters, _) = slfmm_case(0.1, 2000.0, 1.8);
    let n = elements.len();
    let tbem = build_tbem_system_stabilized(&elements, &nodes, &physics, -1.0, 1.0);
    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()))
        .collect();
    let y_tbem = tbem.matrix.matvec(&x);

    let slfmm_base =
        build_slfmm_system_stabilized(&elements, &nodes, &clusters, &physics, 6, 12, 3, -1.0, 1.0);
    let y_near = slfmm_base.near_field_matvec(&x);
    let y_far_ref: Vec<Complex64> = y_tbem
        .iter()
        .zip(y_near.iter())
        .map(|(a, b)| a - b)
        .collect();
    let far_ref_norm: f64 = y_far_ref.iter().map(|y| y.norm_sqr()).sum::<f64>().sqrt();

    let mut errors = Vec::new();
    let mut final_far: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    let expansion_configs = [(3, 6, 12), (6, 12, 24), (10, 20, 40), (16, 20, 64)];
    for (n_terms, n_theta, n_phi) in expansion_configs {
        let slfmm = build_slfmm_system_stabilized(
            &elements, &nodes, &clusters, &physics, n_theta, n_phi, n_terms, -1.0, 1.0,
        );
        let y_slfmm_total = slfmm.matvec(&x);
        let y_slfmm_near = slfmm.near_field_matvec(&x);
        let y_slfmm_far: Vec<Complex64> = y_slfmm_total
            .iter()
            .zip(y_slfmm_near.iter())
            .map(|(a, b)| a - b)
            .collect();
        let diff_norm: f64 = y_far_ref
            .iter()
            .zip(y_slfmm_far.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        errors.push(diff_norm / far_ref_norm.max(1e-15));
        final_far = y_slfmm_far;
    }

    let scaled_down: Vec<Complex64> = final_far.iter().map(|v| v / (4.0 * PI)).collect();
    let scaled_up: Vec<Complex64> = final_far.iter().map(|v| v * (4.0 * PI)).collect();
    let scaled_down_error = y_far_ref
        .iter()
        .zip(scaled_down.iter())
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / far_ref_norm.max(1e-15);
    let scaled_up_error = y_far_ref
        .iter()
        .zip(scaled_up.iter())
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / far_ref_norm.max(1e-15);

    println!("[SLFMM convergence] errors n_terms [3,6,10,16] = {errors:?}");
    assert!(
        errors.windows(2).all(|pair| pair[1] < pair[0]),
        "Far-field error must decrease with every expansion refinement: {errors:?}"
    );
    assert!(
        *errors.last().unwrap() < 1e-5,
        "16-term SLFMM far field must match dense TBEM within 0.001%: {errors:?}"
    );
    assert!(
        errors.last().unwrap() < &(0.5 * scaled_down_error.min(scaled_up_error)),
        "SLFMM far-field normalization regression: unscaled={:.4e}, scaled_down={scaled_down_error:.4e}, scaled_up={scaled_up_error:.4e}",
        errors.last().unwrap(),
    );
}

fn run_validation(name: &str, validation: fn()) -> ValidationOutcome {
    print!("{name} ... ");
    let start = Instant::now();
    let result = panic::catch_unwind(AssertUnwindSafe(validation));
    let elapsed_ms = start.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
    match result {
        Ok(()) => {
            println!("ok ({:.2}s)", elapsed_ms as f64 / 1_000.0);
            ValidationOutcome {
                passed: true,
                elapsed_ms,
            }
        }
        Err(_) => {
            println!("FAILED ({:.2}s)", elapsed_ms as f64 / 1_000.0);
            ValidationOutcome {
                passed: false,
                elapsed_ms,
            }
        }
    }
}

fn main() {
    println!("=== Slow BEM FMM Validation ===");
    println!("These dense-reference sweeps are intentionally outside default cargo test.\n");
    let suite_start = Instant::now();

    let validations: &[(&str, fn())] = &[
        (
            "mlfmm_per_pair_precision_sweep",
            validate_mlfmm_per_pair_precision_sweep,
        ),
        ("mlfmm_dense_ka_sweep", validate_mlfmm_dense_ka_sweep),
        (
            "slfmm_multi_cluster_far_field_vs_tbem",
            validate_slfmm_multi_cluster_far_field_vs_tbem,
        ),
        (
            "slfmm_far_field_convergence",
            validate_slfmm_far_field_convergence,
        ),
    ];

    let mut failed = false;
    let mut failed_case_count = 0_u64;
    let mut max_case_elapsed_ms = 0_u64;
    for (name, validation) in validations {
        let outcome = run_validation(name, *validation);
        if !outcome.passed {
            failed = true;
            failed_case_count += 1;
        }
        max_case_elapsed_ms = max_case_elapsed_ms.max(outcome.elapsed_ms);
    }

    let elapsed_ms = suite_start
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    let result = if failed {
        QaResult::failed(
            "bem.mlfmm-slow-validation.v1",
            "one or more dense-reference FMM sweeps failed",
        )
    } else {
        QaResult::passed("bem.mlfmm-slow-validation.v1")
    }
    .actual_backend("cpu")
    .seed("deterministic-trigonometric-v1")
    .metric("validation_case_count", validations.len() as f64)
    .metric("failed_case_count", failed_case_count as f64)
    .metric("max_case_elapsed_ms", max_case_elapsed_ms as f64)
    .maximum("failed_case_count", 0.0)
    .elapsed_ms(elapsed_ms);
    match result.to_json_line() {
        Ok(line) => println!("QA_RESULT: {line}"),
        Err(error) => {
            eprintln!("FAILED: could not emit QA result: {error}");
            std::process::exit(1);
        }
    }

    if failed {
        println!("\nFAILED");
        std::process::exit(1);
    }

    println!("\nSUCCESS");
}
