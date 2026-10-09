//! Wall-time benchmark for Complex64 ↔ Complex32 GMRES-IR.
//!
//! Compares three configurations on a 2-D 5-point complex Helmholtz
//! stencil — representative of an FEM Helmholtz assembly with absorbing
//! BCs (the BEM near-field block has similar density). Five nnz/row
//! gives non-trivial SpMV arithmetic intensity, unlike the 1-D
//! tridiagonal where Metal dispatch overhead drowns the per-iter work.
//!
//! - **CPU `Complex64` direct GMRES** — what sonium-fem currently runs.
//! - **CPU `(Complex64, Complex32)` IR** — the new mixed-precision path
//!   on the existing in-house BLAS-1 implementation.
//! - **Metal `(Complex64, Complex32)` IR** — same algorithm with the
//!   inner Krylov on the native MSL Complex32 kernels added in step b
//!   of the IR roadmap; outer Complex64 ops run host-staged on Apple
//!   Silicon's unified memory.
//!
//! Each configuration uses an identical polynomial-Jacobi (K=2)
//! preconditioner so the comparison isolates precision/backend cost.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release --example metal_ir_complex_bench --features metal
//! # Override grid (each entry is the side length of the n×n 2-D grid):
//! GRID=64,128,256 cargo run --release --example metal_ir_complex_bench --features metal
//! ```

use std::time::{Duration, Instant};

use num_complex::{Complex32, Complex64};

use sonium_compute::gmres::{
    BackendCsr, BackendPreconditioner, GmresBackendConfig, gmres_backend_preconditioned,
};
use sonium_compute::gmres_ir::{GmresIrConfig, gmres_ir_backend_preconditioned};
use sonium_compute::{
    Backend, ChebyshevBackend, PolynomialJacobiBackend, cpu::CpuBackend, metal::MetalBackend,
};

const ALPHA: f64 = 0.05; // absorption coefficient (imaginary diagonal shift)
const KH_SQ: f64 = 1.0; // (k·h)² — well-conditioned regime; raise for indefinite

/// 2-D 5-point complex Helmholtz on an `n × n` grid (interior only,
/// homogeneous Dirichlet). Centre stencil = `4 − (kh)² + iα`, neighbour
/// stencil = `−1`. N = n² unknowns, ~5 nnz/row.
fn complex_helmholtz_2d(n: usize) -> (Vec<i32>, Vec<i32>, Vec<Complex64>) {
    let total = n * n;
    let centre = Complex64::new(4.0 - KH_SQ, ALPHA);
    let neigh = Complex64::new(-1.0, 0.0);

    let mut row_ptr = vec![0_i32; total + 1];
    let mut col_idx: Vec<i32> = Vec::with_capacity(5 * total);
    let mut values: Vec<Complex64> = Vec::with_capacity(5 * total);

    let idx = |i: usize, j: usize| (i * n + j) as i32;
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            // South
            if i > 0 {
                col_idx.push(idx(i - 1, j));
                values.push(neigh);
            }
            // West
            if j > 0 {
                col_idx.push(idx(i, j - 1));
                values.push(neigh);
            }
            // Centre
            col_idx.push(idx(i, j));
            values.push(centre);
            // East
            if j + 1 < n {
                col_idx.push(idx(i, j + 1));
                values.push(neigh);
            }
            // North
            if i + 1 < n {
                col_idx.push(idx(i + 1, j));
                values.push(neigh);
            }
            row_ptr[row + 1] = values.len() as i32;
        }
    }
    (row_ptr, col_idx, values)
}

fn make_rhs(total: usize) -> Vec<Complex64> {
    (0..total)
        .map(|i| {
            let t = (i as f64 + 1.0) / (total as f64);
            Complex64::new((6.0 * t).sin(), (4.0 * t).cos())
        })
        .collect()
}

#[derive(Debug, Clone)]
struct Row {
    label: &'static str,
    converged: bool,
    outer_iters: usize,
    inner_iters: Option<usize>,
    elapsed: Duration,
    residual: f64,
}

fn run_cpu_direct(grid: usize) -> Row {
    let total = grid * grid;
    let backend = CpuBackend::new();
    let (rp, ci, va) = complex_helmholtz_2d(grid);
    let a =
        BackendCsr::<CpuBackend, Complex64>::try_from_host(&backend, total, total, &rp, &ci, &va)
            .expect("bench fixture CSR upload must succeed");
    let pc = PolynomialJacobiBackend::<CpuBackend, Complex64>::from_csr_host(
        &backend,
        &a,
        &rp,
        &ci,
        &va,
        2,
        Complex64::new(1.0, 0.0),
    );

    let b_host = make_rhs(total);
    let mut b_dev = backend.raw_alloc::<Complex64>(total).unwrap();
    backend.copy_h2d(&b_host, &mut b_dev);
    let mut out = backend.raw_alloc::<Complex64>(total).unwrap();

    // Generous iter cap so direct GMRES has a fair chance against IR.
    let cfg = GmresBackendConfig::<Complex64> {
        max_iterations: 1000,
        tolerance: 1.0e-10,
    };
    let t0 = Instant::now();
    let s = gmres_backend_preconditioned::<CpuBackend, _, _, Complex64>(
        &backend, &a, &pc, &b_dev, None, &mut out, &cfg,
    )
    .expect("solve");
    let elapsed = t0.elapsed();

    Row {
        label: "CPU  Complex64 GMRES (direct)",
        converged: s.converged,
        outer_iters: s.iterations,
        inner_iters: None,
        elapsed,
        residual: s.final_residual,
    }
}

#[derive(Debug, Clone, Copy)]
enum InnerPrecond {
    PolyJacobi { degree: usize },
    Chebyshev { degree: usize, eig_ratio: f32 },
}

fn run_ir<B: Backend>(label: &'static str, backend: &B, grid: usize, precond: InnerPrecond) -> Row {
    let total = grid * grid;
    let (rp, ci, va_c64) = complex_helmholtz_2d(grid);
    let a_outer =
        BackendCsr::<B, Complex64>::try_from_host(backend, total, total, &rp, &ci, &va_c64)
            .expect("bench fixture CSR upload must succeed");
    let a_inner = BackendCsr::<B, Complex32>::try_from_host_cast::<Complex64>(
        backend, total, total, &rp, &ci, &va_c64,
    )
    .expect("bench fixture f32 CSR upload must succeed");

    let va_c32: Vec<Complex32> = va_c64
        .iter()
        .map(|&v| <Complex64 as sonium_compute::LowerPrecision>::cast_to_lower(v))
        .collect();

    let b_host = make_rhs(total);
    let mut b_dev = backend.raw_alloc::<Complex64>(total).unwrap();
    backend.copy_h2d(&b_host, &mut b_dev);
    let mut out = backend.raw_alloc::<Complex64>(total).unwrap();

    let cfg: GmresIrConfig<Complex64> = GmresIrConfig {
        max_outer_iterations: 30,
        outer_tolerance: 1.0e-10,
        max_inner_iterations: 200,
        inner_tolerance: 1.0e-3,
        ..Default::default()
    };

    // Build the chosen preconditioner once and dispatch via dyn-trait
    // — keeps the call site identical for both branches without
    // monomorphising `gmres_ir_backend_preconditioned` twice.
    let pj_holder;
    let cb_holder;
    let pc: &dyn BackendPreconditioner<B, Complex32> = match precond {
        InnerPrecond::PolyJacobi { degree } => {
            pj_holder = PolynomialJacobiBackend::<B, Complex32>::from_csr_host(
                backend,
                &a_inner,
                &rp,
                &ci,
                &va_c32,
                degree,
                Complex32::new(1.0, 0.0),
            );
            &pj_holder
        }
        InnerPrecond::Chebyshev { degree, eig_ratio } => {
            cb_holder = ChebyshevBackend::<B, Complex32>::from_csr_host(
                backend, &a_inner, &rp, &ci, &va_c32, degree, eig_ratio,
            );
            &cb_holder
        }
    };

    let t0 = Instant::now();
    let s = gmres_ir_backend_preconditioned::<B, _, _, _, Complex64>(
        backend, &a_outer, &a_inner, pc, &b_dev, None, &mut out, &cfg,
    )
    .expect("solve");
    let elapsed = t0.elapsed();

    Row {
        label,
        converged: s.converged,
        outer_iters: s.outer_iterations,
        inner_iters: Some(s.total_inner_iterations),
        elapsed,
        residual: s.final_outer_residual,
    }
}

fn print_row(r: &Row) {
    let inner_str = match r.inner_iters {
        Some(k) => format!("{k}"),
        None => "—".to_string(),
    };
    println!(
        "  {:<32}  conv={:5}  outer={:>3}  inner={:>5}  wall={:>9.2}ms  resid={:.2e}",
        r.label,
        r.converged,
        r.outer_iters,
        inner_str,
        r.elapsed.as_secs_f64() * 1000.0,
        r.residual,
    );
}

fn parse_grid() -> Vec<usize> {
    if let Ok(s) = std::env::var("GRID") {
        s.split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect::<Vec<usize>>()
    } else {
        vec![64, 128, 256] // 4096 / 16 384 / 65 536 unknowns
    }
}

fn main() {
    println!(
        "Complex64 ↔ Complex32 GMRES-IR benchmark — 2-D 5-point Helmholtz (kh)²={KH_SQ}, α={ALPHA}"
    );

    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().ok();
    if metal.is_none() {
        eprintln!("warning: no Metal device — skipping the Metal arm.");
    }

    let polyjac = InnerPrecond::PolyJacobi { degree: 2 };
    let cheb = InnerPrecond::Chebyshev {
        degree: 5,
        eig_ratio: 30.0,
    };

    for &grid in &parse_grid() {
        let total = grid * grid;
        println!("\n[ grid = {grid}×{grid}  →  N = {total} ]");

        let cpu_direct = run_cpu_direct(grid);
        print_row(&cpu_direct);

        let cpu_pj = run_ir::<CpuBackend>("CPU  IR + PolyJac K=2", &cpu, grid, polyjac);
        print_row(&cpu_pj);

        let cpu_cb = run_ir::<CpuBackend>("CPU  IR + Chebyshev K=5", &cpu, grid, cheb);
        print_row(&cpu_cb);

        let metal_pj = metal
            .as_ref()
            .map(|m| run_ir::<MetalBackend>("Metal IR + PolyJac K=2", m, grid, polyjac));
        if let Some(r) = &metal_pj {
            print_row(r);
        }
        let metal_cb = metal
            .as_ref()
            .map(|m| run_ir::<MetalBackend>("Metal IR + Chebyshev K=5", m, grid, cheb));
        if let Some(r) = &metal_cb {
            print_row(r);
        }

        // Speedup table relative to the CPU-direct baseline. The
        // Chebyshev-vs-PolyJac ratio isolates the preconditioner choice.
        let base = cpu_direct.elapsed.as_secs_f64();
        let pj = cpu_pj.elapsed.as_secs_f64();
        let cb = cpu_cb.elapsed.as_secs_f64();
        println!("  → CPU PolyJac vs direct : {:>5.2}×", base / pj);
        println!("  → CPU Cheb    vs direct : {:>5.2}×", base / cb);
        println!("  → CPU Cheb    vs PolyJac: {:>5.2}×", pj / cb);
        if let (Some(mp), Some(mc)) = (&metal_pj, &metal_cb) {
            let mp_t = mp.elapsed.as_secs_f64();
            let mc_t = mc.elapsed.as_secs_f64();
            println!("  → Metal PolyJac vs CPU PolyJac: {:>5.2}×", pj / mp_t);
            println!("  → Metal Cheb    vs CPU Cheb   : {:>5.2}×", cb / mc_t);
            println!("  → Metal Cheb    vs Metal PolyJac: {:>5.2}×", mp_t / mc_t);
        }
    }

    println!(
        "\nNote: wall time on the first run includes Metal pipeline-state JIT (~50 ms).\n\
         Re-run for steady-state numbers."
    );
}
