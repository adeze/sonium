//! Experimental directional MLFMM audit.
//!
//! This compares the new cone-pruned M2L translator against the standard
//! MLFMM oracle on the same deterministic vector. It is intentionally an
//! algorithmic A/B test: dense TBEM remains covered by `qa-bem-mlfmm-audit`.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    DirectionalMlfmmConfig, build_cluster_tree,
    build_directional_mlfmm_system_stabilized_with_config, build_mlfmm_system_stabilized,
};
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
use std::f64::consts::PI;
use std::time::{Duration, Instant};

const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;

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

fn relative_l2(reference: &[Complex64], candidate: &[Complex64]) -> f64 {
    let diff_norm: f64 = reference
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let ref_norm: f64 = reference.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    diff_norm / ref_norm.max(1e-30)
}

fn timed_repeated_matvec(
    system: &sonium_bem::core::assembly::mlfmm::MlfmmSystem,
    x: &[Complex64],
    repeats: usize,
) -> (Vec<Complex64>, Duration) {
    let mut last = Vec::new();
    let start = Instant::now();
    for _ in 0..repeats {
        last = std::hint::black_box(system.matvec(std::hint::black_box(x)));
    }
    (last, start.elapsed())
}

fn main() {
    println!("=== Experimental directional MLFMM audit ===");
    println!(
        "Compares cone-pruned M2L against standard MLFMM; dense parity is covered by qa-bem-mlfmm-audit."
    );

    let radius = 1.0;
    let ka = 50.0;
    let subdivisions = 4;
    let elements_per_leaf = 32;
    let repeats = 5;
    let frequency = freq_for_ka(ka, radius);
    let physics = PhysicsParams::new(frequency, SPEED_OF_SOUND, DENSITY, false);
    let mesh = generate_icosphere_mesh(radius, subdivisions);
    let mut elements = mesh.elements.clone();
    for (i, e) in elements.iter_mut().enumerate() {
        e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        e.dof_addresses = vec![i];
    }

    let config = DirectionalMlfmmConfig {
        min_kr: 18.0,
        cone_safety: 7.0,
        min_half_angle: 0.35,
        max_half_angle: 1.70,
        min_active_points: 16,
        num_cones: 64,
        basis_half_angle: 0.50,
    };

    let cluster_tree = build_cluster_tree(&elements, elements_per_leaf, &physics);
    let cluster_tree_directional = cluster_tree.clone();

    let t0 = Instant::now();
    let full =
        build_mlfmm_system_stabilized(&elements, &mesh.nodes, cluster_tree, &physics, -1.0, 1.0);
    let full_assembly = t0.elapsed();

    let t1 = Instant::now();
    let directional = build_directional_mlfmm_system_stabilized_with_config(
        &elements,
        &mesh.nodes,
        cluster_tree_directional,
        &physics,
        -1.0,
        1.0,
        config,
    );
    let directional_assembly = t1.elapsed();

    let x = deterministic_complex_vector(elements.len());
    let (y_full, full_matvec) = timed_repeated_matvec(&full, &x, repeats);
    let (y_directional, directional_matvec) = timed_repeated_matvec(&directional, &x, repeats);
    let rel_err = relative_l2(&y_full, &y_directional);

    let full_stats = full.translation_stats();
    let dir_stats = directional.translation_stats();
    let matvec_speedup = full_matvec.as_secs_f64() / directional_matvec.as_secs_f64().max(1e-12);

    println!(
        "case: kD={ka:.1}, subdiv={subdivisions}, DOFs={}, levels={}",
        elements.len(),
        directional.num_levels
    );
    println!(
        "translation entries: {}, directional entries: {}, active points: {}/{} ({:.1}%)",
        dir_stats.entries,
        dir_stats.directional_entries,
        dir_stats.active_points,
        dir_stats.full_points,
        100.0 * dir_stats.active_fraction()
    );
    println!(
        "cone bases: {}, compact workspace points: {}",
        dir_stats.cone_count, dir_stats.cone_workspace_points
    );
    println!(
        "standard MLFMM points: {}/{} ({:.1}%)",
        full_stats.active_points,
        full_stats.full_points,
        100.0 * full_stats.active_fraction()
    );
    println!(
        "assembly: standard {:.2}s, directional {:.2}s",
        full_assembly.as_secs_f64(),
        directional_assembly.as_secs_f64()
    );
    println!(
        "matvec x{repeats}: standard {:.4}s, directional {:.4}s, speedup {:.2}x",
        full_matvec.as_secs_f64(),
        directional_matvec.as_secs_f64(),
        matvec_speedup
    );
    println!("relative drift vs standard MLFMM: {rel_err:.3e}");

    let finite = y_directional
        .iter()
        .all(|z| z.re.is_finite() && z.im.is_finite());
    if !finite {
        eprintln!("FAILED: directional MLFMM produced non-finite output");
        std::process::exit(1);
    }
    if dir_stats.directional_entries == 0 {
        eprintln!("FAILED: directional MLFMM did not prune any high-kr translation entries");
        std::process::exit(1);
    }
    if rel_err > 2.5e-1 {
        eprintln!(
            "FAILED: directional MLFMM drift {rel_err:.3e} exceeds experimental 2.5e-1 budget"
        );
        std::process::exit(1);
    }

    println!("SUCCESS");
}
