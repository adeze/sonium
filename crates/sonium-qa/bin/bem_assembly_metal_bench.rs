//! BEM CPU-parallel vs Metal-backed TBEM assembly QA.
//!
//! Builds icosphere meshes and compares:
//!   - `build_tbem_system_with_beta_parallel` — the existing rayon-multi-
//!     core CPU path,
//!   - `build_tbem_system_with_beta_metal` — the Phase 1.3 GPU path that
//!     pre-buckets pairs and dispatches the regular far-field tail to
//!     `sonium_bem_regular_pair_c32` on Metal.
//!
//! Default mode runs QA-sized parity rows. Set `BEM_CPU_GPU_HEAVY=1` to include
//! the larger subdivision-4 row; it holds both dense matrices in memory so it
//! belongs in BEM QA, not quick tests.
//!
//! Run with:
//! `cargo run -p sonium-qa --release --features metal --bin qa-bem-assembly-metal-bench`

#[cfg(feature = "metal")]
use num_complex::Complex64;
#[cfg(feature = "metal")]
use sonium_bem::core::assembly::tbem::{
    build_tbem_system_with_beta_metal, build_tbem_system_with_beta_parallel,
};
#[cfg(feature = "metal")]
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
#[cfg(feature = "metal")]
use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
#[cfg(feature = "metal")]
use sonium_compute::MetalBackend;
#[cfg(feature = "metal")]
use std::f64::consts::PI;
#[cfg(feature = "metal")]
use std::time::Instant;

#[cfg(feature = "metal")]
const REL_TOL: f64 = 1e-3;

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct Row {
    subdiv: usize,
    dofs: usize,
    cpu_s: f64,
    gpu_s: f64,
    speedup: f64,
    matrix_rel: f64,
    rhs_rel: f64,
}

#[cfg(feature = "metal")]
fn relative_l2<'a>(
    reference: impl Iterator<Item = &'a Complex64>,
    candidate: impl Iterator<Item = &'a Complex64>,
) -> f64 {
    let (diff, norm) = reference
        .zip(candidate)
        .fold((0.0, 0.0), |(diff, norm), (a, b)| {
            (diff + (a - b).norm_sqr(), norm + a.norm_sqr())
        });
    diff.sqrt() / norm.sqrt().max(1e-30)
}

#[cfg(feature = "metal")]
fn run_subdivision(subdiv: usize, backend: &MetalBackend) -> Row {
    let radius = 1.0_f64;
    // Pick a frequency so kD ≈ 1 at this radius. The benchmark cares about
    // pair count and quadrature work, not solver convergence — kD=1 is the
    // cheapest defensible setting.
    let frequency = 343.0 / (2.0 * PI * radius);
    let physics = PhysicsParams::new(frequency, 343.0, 1.21, false);
    let beta = physics.burton_miller_beta();

    let mesh = generate_icosphere_mesh(radius, subdiv);
    let mut elements = mesh.elements.clone();
    for (i, e) in elements.iter_mut().enumerate() {
        e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        e.dof_addresses = vec![i];
    }
    elements[0].boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(1.0, 0.0)]);
    let n_dofs = elements.len();

    let t_cpu = Instant::now();
    let cpu = build_tbem_system_with_beta_parallel(&elements, &mesh.nodes, &physics, beta);
    let cpu_s = t_cpu.elapsed().as_secs_f64();

    let t_gpu = Instant::now();
    let gpu = build_tbem_system_with_beta_metal(&elements, &mesh.nodes, &physics, beta, backend);
    let gpu_s = t_gpu.elapsed().as_secs_f64();

    assert_eq!(cpu.matrix.dim(), gpu.matrix.dim(), "matrix dims differ");
    assert_eq!(cpu.rhs.len(), gpu.rhs.len(), "rhs lengths differ");

    let matrix_rel = relative_l2(cpu.matrix.iter(), gpu.matrix.iter());
    let rhs_rel = relative_l2(cpu.rhs.iter(), gpu.rhs.iter());

    let speedup = if gpu_s > 0.0 {
        cpu_s / gpu_s
    } else {
        f64::INFINITY
    };
    Row {
        subdiv,
        dofs: n_dofs,
        cpu_s,
        gpu_s,
        speedup,
        matrix_rel,
        rhs_rel,
    }
}

#[cfg(feature = "metal")]
fn main() {
    println!("=== BEM CPU-parallel vs Metal TBEM assembly QA ===");
    println!("kD = 1, BoundaryCondition::Velocity, parity tolerance = {REL_TOL:.1e}\n");

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(err) => {
            println!("SKIPPED: Metal backend not available: {err}");
            println!("SUCCESS");
            return;
        }
    };

    let mut subdivisions = vec![2, 3]; // ~320, ~1280 DOFs
    if std::env::var("BEM_CPU_GPU_HEAVY").as_deref() == Ok("1") {
        subdivisions.push(4); // ~5120 DOFs, dense parity holds both matrices
    }

    println!(
        "{:>6}  {:>6}  | {:>10}  {:>10}  {:>8}  | {:>11}  {:>11}",
        "subdiv", "DOFs", "CPU wall", "Metal", "speedup", "matrix rel", "rhs rel"
    );
    println!("  {}", "-".repeat(88));

    let mut rows = Vec::new();
    for subdiv in subdivisions {
        let row = run_subdivision(subdiv, &backend);
        println!(
            "{:>6}  {:>6}  | {:>9.2}s  {:>9.2}s  {:>7.2}x  | {:>11.3e}  {:>11.3e}",
            row.subdiv, row.dofs, row.cpu_s, row.gpu_s, row.speedup, row.matrix_rel, row.rhs_rel
        );
        rows.push(row);
    }

    if std::env::var("BEM_CPU_GPU_HEAVY").as_deref() != Ok("1") {
        println!("  (subdiv 4 skipped — set BEM_CPU_GPU_HEAVY=1 for dense 5120-DOF parity)");
    }

    let max_speedup = rows
        .iter()
        .map(|r| r.speedup)
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "\nBest speedup observed: {:.2}x (recorded; not a hard acceptance gate)",
        max_speedup
    );

    if rows
        .iter()
        .any(|r| r.matrix_rel > REL_TOL || r.rhs_rel > REL_TOL)
    {
        eprintln!(
            "FAILED: Metal path differs from CPU by more than {REL_TOL:.1e} on at least one row"
        );
        std::process::exit(1);
    }
    if rows.iter().any(|r| r.gpu_s > 10.0 * r.cpu_s) {
        eprintln!("FAILED: Metal path is more than 10x slower than CPU on at least one row");
        std::process::exit(1);
    }

    println!("\nSUCCESS");
}

#[cfg(not(feature = "metal"))]
fn main() {
    println!(
        "qa-bem-assembly-metal-bench requires the `metal` feature; \
         re-run with `--features metal` (macOS only)."
    );
    println!("SUCCESS");
}
