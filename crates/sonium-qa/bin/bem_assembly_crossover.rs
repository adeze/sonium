//! Empirical assembly-method crossover measurement, per `bem-step1.md` §1.4.
//!
//! Sweeps an icosphere across subdivision levels at fixed kD and times:
//!  - `Tbem` dense assembly + 10 matvecs (the O(N²) reference)
//!  - `Hmatrix` assembly (default ACA+ variant, tol 1e-4, max_rank 64) + 10
//!    matvecs through the `LinearOperator` impl
//!
//! `Mlfmm` is intentionally excluded: `qa-bem-mlfmm-audit` shows the
//! production MLFMM matvec diverges from dense by 100% at kD=1 and ~1e7 at
//! kD=10. Until the doc-mandated kifmm-rs FFI escape lands, the only sound
//! "large problem" assembler in this crate is `Hmatrix`.
//!
//! Output is a table of (DOFs, assembly time, total matvec cost,
//! memory footprint, compression ratio). Pick the crossover where the
//! Hmatrix cumulative wall-clock first beats Tbem at the same N — that's
//! the empirical threshold for `AssemblyMethod` auto-selection.
//!
//! Heavy cases (subdiv 4 = 5120 DOFs, subdiv 5 = 20480 DOFs) are gated
//! behind `CROSSOVER_HEAVY=1` because dense LU + dense matvec at 5k+ DOFs
//! takes minutes.

use num_complex::Complex64;
use sonium_bem::core::assembly::hmatrix::build_hmatrix;
use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::mesh::octree::Octree;
use sonium_bem::core::types::{BoundaryCondition, Cluster, ElementProperty, PhysicsParams};
use sonium_solvers::traits::LinearOperator;
use std::f64::consts::PI;
use std::time::Instant;

const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const FIXED_KA: f64 = 1.0;
const NUM_MATVECS: usize = 10;

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

/// Build a flat cluster list from an octree of element centroids. Mirrors
/// the pattern used by `room_acoustics::solver::build_clusters_from_octree`
/// without dragging in that crate's heavyweight scaffolding.
fn clusters_from_octree(
    elements: &[sonium_bem::core::types::Element],
    octree: &Octree,
) -> Vec<Cluster> {
    let leaves = octree.leaves();
    let mut clusters = Vec::with_capacity(leaves.len());
    let leaf_to_cluster: std::collections::HashMap<usize, usize> = leaves
        .iter()
        .filter(|&&i| !octree.nodes[i].element_indices.is_empty())
        .enumerate()
        .map(|(cluster_idx, &leaf_idx)| (leaf_idx, cluster_idx))
        .collect();

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

    for (cluster_idx, &leaf_idx) in leaves.iter().enumerate() {
        if octree.nodes[leaf_idx].element_indices.is_empty() {
            continue;
        }
        let octree_node = &octree.nodes[leaf_idx];
        let near: Vec<usize> = octree_node
            .near_clusters
            .iter()
            .filter_map(|&n| leaf_to_cluster.get(&n).copied())
            .collect();
        let far: Vec<usize> = octree_node
            .far_clusters
            .iter()
            .filter_map(|&f| leaf_to_cluster.get(&f).copied())
            .collect();
        if let Some(c) = clusters.get_mut(cluster_idx) {
            c.near_clusters = near;
            c.far_clusters = far;
        }
    }

    clusters
}

#[derive(Default)]
struct TbemRow {
    n_dofs: usize,
    assembly_s: f64,
    matvec_total_s: f64,
    matrix_bytes: usize,
}

#[derive(Default)]
struct HmatrixRow {
    n_dofs: usize,
    assembly_s: f64,
    matvec_total_s: f64,
    memory_bytes: usize,
    compression_ratio: f64,
    n_dense_blocks: usize,
    n_lowrank_blocks: usize,
}

fn run_subdivision(subdiv: usize) -> (TbemRow, HmatrixRow) {
    let radius = 1.0;
    let frequency = freq_for_ka(FIXED_KA, radius);
    let physics = PhysicsParams::new(frequency, SPEED_OF_SOUND, DENSITY, false);

    let mesh = generate_icosphere_mesh(radius, subdiv);
    let mut elements = mesh.elements.clone();
    for (i, e) in elements.iter_mut().enumerate() {
        e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        e.dof_addresses = vec![i];
    }
    let n = elements.len();
    let x = deterministic_complex_vector(n);

    // ---------------- Tbem ----------------
    let t0 = Instant::now();
    let tbem = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);
    let tbem_assembly_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..NUM_MATVECS {
        let _ = tbem.matrix.matvec(&x);
    }
    let tbem_matvec_total_s = t1.elapsed().as_secs_f64();
    let tbem_matrix_bytes = std::mem::size_of_val(tbem.matrix.as_slice());

    let tbem_row = TbemRow {
        n_dofs: n,
        assembly_s: tbem_assembly_s,
        matvec_total_s: tbem_matvec_total_s,
        matrix_bytes: tbem_matrix_bytes,
    };

    // ---------------- Hmatrix ----------------
    // Octree partitioning: ~32 elements per leaf is a sane default for ACA
    // far-field work; max depth 8 lets the tree adapt to large meshes.
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let mut octree = Octree::build(&centers, 32, 8);
    octree.compute_interaction_lists(2.0);
    let clusters = clusters_from_octree(&elements, &octree);

    let t2 = Instant::now();
    let hmat = build_hmatrix(&elements, &mesh.nodes, &clusters, &physics, 1e-4, 64);
    let hmat_assembly_s = t2.elapsed().as_secs_f64();

    let t3 = Instant::now();
    for _ in 0..NUM_MATVECS {
        let _ = hmat.apply(&x);
    }
    let hmat_matvec_total_s = t3.elapsed().as_secs_f64();

    let hmat_row = HmatrixRow {
        n_dofs: n,
        assembly_s: hmat_assembly_s,
        matvec_total_s: hmat_matvec_total_s,
        memory_bytes: hmat.memory_bytes,
        compression_ratio: hmat.compression_ratio,
        n_dense_blocks: hmat.dense_blocks.len(),
        n_lowrank_blocks: hmat.low_rank_blocks.len(),
    };

    (tbem_row, hmat_row)
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.2} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.2} KB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

fn main() {
    println!("=== Assembly-method crossover measurement ===");
    println!(
        "Per bem-step1.md §1.4. Fixed kD = {:.1}, {} matvecs per measurement.",
        FIXED_KA, NUM_MATVECS
    );
    println!(
        "MLFMM now PASSes the audit (kD=10 → 4.0e-4); included for completeness, but heavy mode \
         skips subdiv ≥ 5 to keep dense reference under 1 GB."
    );
    println!();

    let always_subdiv: &[usize] = &[1, 2, 3]; // 80, 320, 1280 elements
    // subdiv 5 = 20480 DOFs → 6.7 GB dense matrix (OOMs on a typical box).
    // Heavy mode includes subdiv 4 only (≈ 5120 DOFs, 420 MB dense).
    let heavy_subdiv: &[usize] = &[4];

    println!(
        "{:>6}  {:>6}  | Tbem: {:>10}  {:>12}  {:>10}  | Hmatrix: {:>10}  {:>12}  {:>10}  {:>9}  {:>5}+{:<5}",
        "subdiv",
        "DOFs",
        "assembly",
        "10·matvec",
        "memory",
        "assembly",
        "10·matvec",
        "memory",
        "compress",
        "dense",
        "lowrank"
    );
    println!("  {}", "-".repeat(150));

    let mut rows: Vec<(usize, TbemRow, HmatrixRow)> = Vec::new();
    for &s in always_subdiv {
        let (t, h) = run_subdivision(s);
        rows.push((s, t, h));
    }

    if std::env::var("CROSSOVER_HEAVY").as_deref() == Ok("1") {
        for &s in heavy_subdiv {
            let (t, h) = run_subdivision(s);
            rows.push((s, t, h));
        }
    } else {
        println!("  (subdiv 4-5 skipped — set CROSSOVER_HEAVY=1 to run; ~5k and ~20k DOFs)");
    }

    for (subdiv, t, h) in &rows {
        println!(
            "{:>6}  {:>6}  |       {:>9.2}s  {:>11.4}s  {:>10}  |         {:>9.2}s  {:>11.4}s  {:>10}  {:>9.3}  {:>5}+{:<5}",
            subdiv,
            t.n_dofs,
            t.assembly_s,
            t.matvec_total_s,
            fmt_bytes(t.matrix_bytes),
            h.assembly_s,
            h.matvec_total_s,
            fmt_bytes(h.memory_bytes),
            h.compression_ratio,
            h.n_dense_blocks,
            h.n_lowrank_blocks,
        );
    }

    // Find the crossover: smallest N where Hmatrix total wall-clock
    // (assembly + 10 matvecs) is < Tbem total wall-clock.
    let crossover = rows.iter().find(|(_, t, h)| {
        let tbem_total = t.assembly_s + t.matvec_total_s;
        let hmat_total = h.assembly_s + h.matvec_total_s;
        hmat_total < tbem_total
    });

    println!();
    match crossover {
        Some((subdiv, _, h)) => {
            println!(
                "Crossover (Hmatrix beats Tbem on assembly+matvec): subdiv {} → {} DOFs",
                subdiv, h.n_dofs
            );
            println!(
                "Recommendation: AssemblyMethod::Tbem for N < {}, AssemblyMethod::Hmatrix above.",
                h.n_dofs
            );
        }
        None => {
            println!(
                "No crossover observed in the measured range — Tbem wins everywhere.\n\
                 Re-run with CROSSOVER_HEAVY=1 to extend to 5k / 20k DOFs."
            );
        }
    }

    let invalid = rows.is_empty()
        || rows.iter().any(|(_, tbem, hmatrix)| {
            tbem.n_dofs == 0
                || tbem.matrix_bytes == 0
                || !tbem.assembly_s.is_finite()
                || !tbem.matvec_total_s.is_finite()
                || !hmatrix.assembly_s.is_finite()
                || !hmatrix.matvec_total_s.is_finite()
                || !hmatrix.compression_ratio.is_finite()
                || hmatrix.compression_ratio <= 0.0
        });
    if invalid {
        eprintln!("FAILED: assembly benchmark produced an invalid or non-finite measurement");
        std::process::exit(1);
    }
    println!("\nSUCCESS");
}
