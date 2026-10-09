//! BEM Stage 2.1 node-level scaling harness.
//!
//! This binary measures the existing single-node parallel paths before any
//! MPI work starts. It runs the same BEM fixture under local Rayon pools with
//! different thread counts and reports speedup/parallel efficiency for:
//!
//! - H-matrix assembly and matvec,
//! - MLFMM assembly and matvec.
//!
//! It is intentionally a long-running QA/benchmark binary rather than a unit
//! test. Default settings stay modest; set `BEM_NODE_SCALING_HEAVY=1` for a
//! larger mesh, or `BEM_NODE_SCALING_THREADS=1,2,4,8` to pin the sweep.

use num_complex::Complex64;
use rayon::ThreadPoolBuilder;
use sonium_bem::core::assembly::hmatrix::build_hmatrix;
use sonium_bem::core::assembly::mlfmm::{build_cluster_tree, build_mlfmm_system_stabilized};
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::mesh::octree::Octree;
use sonium_bem::core::types::{
    BoundaryCondition, Cluster, Element, ElementProperty, PhysicsParams,
};
use sonium_solvers::traits::LinearOperator;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::{Duration, Instant};

const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const FIXED_KA: f64 = 2.0;
const HMATRIX_ELEMENTS_PER_LEAF: usize = 32;
const MLFMM_ELEMENTS_PER_LEAF: usize = 16;

fn freq_for_ka(ka: f64, radius: f64) -> f64 {
    ka * SPEED_OF_SOUND / (2.0 * PI * radius)
}

fn deterministic_complex_vector(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = (i + 1) as f64;
            Complex64::new((0.37 * x).sin(), (0.19 * x).cos())
        })
        .collect()
}

fn parse_thread_counts() -> Vec<usize> {
    if let Ok(value) = std::env::var("BEM_NODE_SCALING_THREADS") {
        let mut threads: Vec<usize> = value
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .collect();
        threads.sort_unstable();
        threads.dedup();
        if !threads.is_empty() {
            return threads;
        }
    }

    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut threads = vec![1, 2.min(available), 4.min(available), available];
    threads.sort_unstable();
    threads.dedup();
    threads
}

fn prepare_fixture(subdivisions: usize) -> (Vec<Element>, Vec<[f64; 3]>, PhysicsParams) {
    let radius = 1.0;
    let frequency = freq_for_ka(FIXED_KA, radius);
    let physics = PhysicsParams::new(frequency, SPEED_OF_SOUND, DENSITY, false);
    let mesh = generate_icosphere_mesh(radius, subdivisions);
    let mut elements = mesh.elements;
    for (i, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![i];
    }
    (elements, mesh.nodes, physics)
}

fn clusters_from_octree(elements: &[Element], octree: &Octree) -> Vec<Cluster> {
    let leaves = octree.leaves();
    let leaf_to_cluster: HashMap<usize, usize> = leaves
        .iter()
        .filter(|&&i| !octree.nodes[i].element_indices.is_empty())
        .enumerate()
        .map(|(cluster_idx, &leaf_idx)| (leaf_idx, cluster_idx))
        .collect();

    let mut clusters = Vec::with_capacity(leaf_to_cluster.len());
    for &leaf_idx in &leaves {
        let node = &octree.nodes[leaf_idx];
        if node.element_indices.is_empty() {
            continue;
        }

        let mut cluster = Cluster::new(node.center);
        cluster.element_indices = node.element_indices.clone();
        cluster.num_elements = node.element_indices.len();
        cluster.element_property = ElementProperty::Surface;
        cluster.radius = node.radius();
        cluster.level = node.level;
        cluster.num_dofs = node
            .element_indices
            .iter()
            .filter(|&&i| !elements[i].property.is_evaluation())
            .count();
        cluster.dofs_per_element = 1;
        clusters.push(cluster);
    }

    for &leaf_idx in &leaves {
        let Some(&cluster_idx) = leaf_to_cluster.get(&leaf_idx) else {
            continue;
        };
        let octree_node = &octree.nodes[leaf_idx];
        if let Some(cluster) = clusters.get_mut(cluster_idx) {
            cluster.near_clusters = octree_node
                .near_clusters
                .iter()
                .filter_map(|&n| leaf_to_cluster.get(&n).copied())
                .collect();
            cluster.far_clusters = octree_node
                .far_clusters
                .iter()
                .filter_map(|&f| leaf_to_cluster.get(&f).copied())
                .collect();
        }
    }

    clusters
}

fn build_hmatrix_clusters(elements: &[Element]) -> Vec<Cluster> {
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let mut octree = Octree::build(&centers, HMATRIX_ELEMENTS_PER_LEAF, 8);
    octree.compute_interaction_lists(2.0);
    clusters_from_octree(elements, &octree)
}

#[derive(Clone, Copy)]
struct Timing {
    threads: usize,
    hmatrix_assembly: Duration,
    hmatrix_matvec: Duration,
    mlfmm_assembly: Duration,
    mlfmm_matvec: Duration,
}

fn run_for_threads(
    threads: usize,
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    hmatrix_clusters: &[Cluster],
    x: &[Complex64],
) -> Timing {
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build local Rayon pool");

    pool.install(|| {
        let t0 = Instant::now();
        let hmat = build_hmatrix(elements, nodes, hmatrix_clusters, physics, 1e-4, 64);
        let hmatrix_assembly = t0.elapsed();

        let t1 = Instant::now();
        let y_h = hmat.apply(x);
        let hmatrix_matvec = t1.elapsed();
        assert!(
            y_h.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
            "FAILED: non-finite H-matrix matvec output"
        );

        let t2 = Instant::now();
        let tree = build_cluster_tree(elements, MLFMM_ELEMENTS_PER_LEAF, physics);
        let mlfmm = build_mlfmm_system_stabilized(elements, nodes, tree, physics, -1.0, 1.0);
        let mlfmm_assembly = t2.elapsed();

        let t3 = Instant::now();
        let y_m = mlfmm.matvec(x);
        let mlfmm_matvec = t3.elapsed();
        assert!(
            y_m.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
            "FAILED: non-finite MLFMM matvec output"
        );

        Timing {
            threads,
            hmatrix_assembly,
            hmatrix_matvec,
            mlfmm_assembly,
            mlfmm_matvec,
        }
    })
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn print_row(label: &str, baseline: f64, threads: usize, elapsed: f64) {
    let speedup = baseline / elapsed.max(1e-12);
    let efficiency = speedup / threads as f64;
    println!("{label:<18} {threads:>3}  {elapsed:>9.4}s  {speedup:>8.2}x  {efficiency:>8.2}");
}

fn main() {
    println!("=== BEM Stage 2.1 node-level scaling ===");
    println!("Fixed ka={FIXED_KA:.1}; local Rayon pools isolate each thread-count run.");

    let subdivisions = if std::env::var("BEM_NODE_SCALING_HEAVY").as_deref() == Ok("1") {
        4
    } else {
        3
    };
    let threads = parse_thread_counts();
    let (elements, nodes, physics) = prepare_fixture(subdivisions);
    let hmatrix_clusters = build_hmatrix_clusters(&elements);
    let x = deterministic_complex_vector(elements.len());

    println!(
        "Fixture: subdiv={subdivisions}, DOFs={}, H-clusters={}, thread sweep={threads:?}",
        elements.len(),
        hmatrix_clusters.len()
    );
    println!();

    let mut timings = Vec::new();
    for &n_threads in &threads {
        timings.push(run_for_threads(
            n_threads,
            &elements,
            &nodes,
            &physics,
            &hmatrix_clusters,
            &x,
        ));
    }

    let baseline = timings
        .iter()
        .find(|t| t.threads == 1)
        .or_else(|| timings.first())
        .expect("at least one timing");

    println!(
        "{:<18} {:>3}  {:>10}  {:>9}  {:>10}",
        "operation", "thr", "wall", "speedup", "eff"
    );
    println!("{}", "-".repeat(58));
    for timing in &timings {
        print_row(
            "Hmatrix assembly",
            seconds(baseline.hmatrix_assembly),
            timing.threads,
            seconds(timing.hmatrix_assembly),
        );
        print_row(
            "Hmatrix matvec",
            seconds(baseline.hmatrix_matvec),
            timing.threads,
            seconds(timing.hmatrix_matvec),
        );
        print_row(
            "MLFMM assembly",
            seconds(baseline.mlfmm_assembly),
            timing.threads,
            seconds(timing.mlfmm_assembly),
        );
        print_row(
            "MLFMM matvec",
            seconds(baseline.mlfmm_matvec),
            timing.threads,
            seconds(timing.mlfmm_matvec),
        );
        println!();
    }

    println!("SUCCESS");
}
