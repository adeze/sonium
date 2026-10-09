//! Mixed-precision GMRES-IR scaling benchmark.
//!
//! Compares two end-to-end solves on identical matrices:
//!
//!   * **plain f64**: `gmres_backend_preconditioned` driving f64 GMRES with
//!     an f64 Chebyshev preconditioner.
//!   * **mixed-precision IR**: `gmres_ir_backend_preconditioned` doing
//!     f64 outer FGMRES-IR refinement around an f32 inner solve whose
//!     preconditioner is also f32 Chebyshev (Carson-Higham 2018, Alg 5.3).
//!
//! Both runs use the *same* preconditioner family at the *same* degree,
//! so any difference in wall time is attributable to the precision of the
//! inner work (memory traffic halved, kernel throughput up to 2× on most
//! GPUs, cuBLAS kernels switch to higher-arithmetic-intensity SGEMM
//! variants). The outer f64 refinement keeps accuracy at the level of a
//! pure f64 solve up to κ(A) ≲ 10¹⁴ for the algorithm's regime.
//!
//! What this bench answers:
//!   1. Does mixed-precision IR converge to the same f64 tolerance the
//!      plain solve would have?
//!   2. How much wall-clock does it save *per solve* (not per apply)?
//!   3. Does the speedup grow with size (memory-bandwidth regime) or
//!      shrink (kernel-launch regime)?
//!
//! Build:
//! ```text
//! cargo build --release -p sonium-solvers --features "cli,cuda" \
//!     --bin gmres-ir-scaling
//! ```
//!
//! Run (Threadripper + NVIDIA GPU):
//! ```text
//! ./target/release/gmres-ir-scaling \
//!     --sizes 100000,1000000,5000000 \
//!     --degree 4 --tolerance 1e-8 --iters 3
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use sonium_compute::gmres::{
    BackendCsr, BackendOperator, BackendPreconditioner, GmresBackendConfig,
    gmres_backend_preconditioned,
};
use sonium_compute::gmres_ir::{GmresIrConfig, gmres_ir_backend_preconditioned};
use sonium_compute::{Backend, ChebyshevBackend, CpuBackend};

#[cfg(feature = "cuda")]
use sonium_compute::CudaBackend;

#[derive(Parser, Debug)]
#[command(
    name = "gmres-ir-scaling",
    about = "End-to-end GMRES-IR (f64 outer, f32 inner) vs plain f64 GMRES, \
             driven by Chebyshev preconditioning. Compares wall-clock and \
             iteration counts at a fixed outer tolerance."
)]
struct Args {
    /// Comma-separated target DOF counts. Same 3D 7-point Laplacian as
    /// `precond-cuda-scaling` so results compare directly.
    #[arg(long, default_value = "100000,500000,1000000", value_delimiter = ',')]
    sizes: Vec<usize>,

    /// Chebyshev preconditioner degree (same for f64 and f32 inner).
    #[arg(long, default_value_t = 4)]
    degree: usize,

    /// Eigenvalue ratio λ_max/λ_min for the Chebyshev polynomial. PETSc
    /// and HYPRE default to 30.
    #[arg(long, default_value_t = 30.0)]
    cheb_ratio: f64,

    /// Outer convergence tolerance (relative residual). The plain f64
    /// solver uses this directly; the IR solver uses it as the outer
    /// tolerance and a loose `--inner-tolerance` on the inner f32 GMRES.
    #[arg(long, default_value_t = 1.0e-8)]
    tolerance: f64,

    /// Inner f32 GMRES tolerance for the IR path. Loose on purpose —
    /// the outer f64 refinement cleans up.
    #[arg(long, default_value_t = 1.0e-3)]
    inner_tolerance: f32,

    /// Max outer iterations (IR) / max plain GMRES iterations.
    #[arg(long, default_value_t = 200)]
    max_iter: usize,

    /// Max inner GMRES iterations per outer IR step.
    #[arg(long, default_value_t = 50)]
    max_inner_iter: usize,

    /// Number of full-solve samples per cell. Median is reported.
    #[arg(long, default_value_t = 3)]
    iters: usize,

    /// Skip the CPU backend.
    #[arg(long)]
    skip_cpu: bool,

    /// Auto-skip the CPU backend for matrices above this DOF count. CPU
    /// f64 GMRES + degree-4 Chebyshev runs into hundreds of iterations at
    /// large N, which can take minutes per size and isn't useful when the
    /// point of the bench is the GPU. Set to 0 to disable the auto-skip.
    #[arg(long, default_value_t = 200_000)]
    cpu_size_limit: usize,

    /// Skip the CUDA backend.
    #[arg(long)]
    skip_cuda: bool,

    /// Optional CSV output path.
    #[arg(long)]
    csv: Option<PathBuf>,
}

/// Build a 3D 7-point Laplacian on an n×n×n grid (real f64).
fn build_laplacian_3d(n: usize) -> (usize, Vec<i32>, Vec<i32>, Vec<f64>) {
    let nn = n * n * n;
    let h2 = 1.0 / ((n + 1) as f64).powi(2);
    let diag = 6.0 / h2;
    let off = -1.0 / h2;

    let mut row_ptr: Vec<i32> = Vec::with_capacity(nn + 1);
    let mut col_idx: Vec<i32> = Vec::with_capacity(7 * nn);
    let mut values: Vec<f64> = Vec::with_capacity(7 * nn);
    row_ptr.push(0);
    let push = |c: i32, v: f64, ci: &mut Vec<i32>, vs: &mut Vec<f64>| {
        ci.push(c);
        vs.push(v);
    };

    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let row = (i + n * (j + n * k)) as i32;
                if k > 0 {
                    push(row - (n * n) as i32, off, &mut col_idx, &mut values);
                }
                if j > 0 {
                    push(row - n as i32, off, &mut col_idx, &mut values);
                }
                if i > 0 {
                    push(row - 1, off, &mut col_idx, &mut values);
                }
                push(row, diag, &mut col_idx, &mut values);
                if i + 1 < n {
                    push(row + 1, off, &mut col_idx, &mut values);
                }
                if j + 1 < n {
                    push(row + n as i32, off, &mut col_idx, &mut values);
                }
                if k + 1 < n {
                    push(row + (n * n) as i32, off, &mut col_idx, &mut values);
                }
                row_ptr.push(values.len() as i32);
            }
        }
    }
    (nn, row_ptr, col_idx, values)
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    backend: &'static str,
    mode: &'static str, // "f64" | "ir"
    n: usize,
    wall: Duration,
    outer_iters: usize,
    inner_iters: usize,
    residual: f64,
    converged: bool,
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s >= 1.0 {
        format!("{s:>7.2} s")
    } else {
        format!("{:>5.1} ms", s * 1000.0)
    }
}

fn run_size<B: Backend>(
    backend: &B,
    backend_label: &'static str,
    args: &Args,
    target: usize,
    rows: &mut Vec<Cell>,
) {
    let n_grid = (target as f64).cbrt().ceil() as usize;
    let (n, row_ptr_host, col_idx_host, values_host) = build_laplacian_3d(n_grid);
    let nnz = values_host.len();
    println!("\nMatrix: n={n_grid} cube → {n} DOFs, {nnz} nnz (target {target})");

    // Upload f64 system to backend.
    let a64 = BackendCsr::<B, f64>::try_from_host(
        backend,
        n,
        n,
        &row_ptr_host,
        &col_idx_host,
        &values_host,
    )
    .expect("scaling fixture CSR upload must succeed");
    // Cast f64 → f32 once for the inner IR matrix.
    let values_f32: Vec<f32> = values_host.iter().map(|&v| v as f32).collect();
    let a32 = BackendCsr::<B, f32>::try_from_host(
        backend,
        n,
        n,
        &row_ptr_host,
        &col_idx_host,
        &values_f32,
    )
    .expect("scaling fixture CSR upload must succeed");

    // RHS: well-conditioned, all-ones-ish so the solver actually has work
    // to do. Same vector for both f64 and f32 paths so iterations are
    // comparable.
    let rhs_host: Vec<f64> = (0..n).map(|i| (i % 17) as f64 * 0.1 + 1.0).collect();
    let mut rhs64 = backend.raw_alloc::<f64>(n).expect("rhs alloc");
    backend.copy_h2d(&rhs_host, &mut rhs64);

    // f64 Chebyshev preconditioner (host-built once, drives the device
    // ops thereafter).
    let p64 = ChebyshevBackend::<B, f64>::from_csr_host(
        backend,
        &a64,
        &row_ptr_host,
        &col_idx_host,
        &values_host,
        args.degree,
        args.cheb_ratio,
    );
    // f32 Chebyshev preconditioner for the IR inner solver.
    let p32 = ChebyshevBackend::<B, f32>::from_csr_host(
        backend,
        &a32,
        &row_ptr_host,
        &col_idx_host,
        &values_f32,
        args.degree,
        args.cheb_ratio as f32,
    );

    println!(
        "{:>4} | {:>11} | {:>5} | {:>5} | {:>9}",
        "mode", "wall", "outer", "inner", "residual"
    );
    println!(
        "{:->4}-+-{:->11}-+-{:->5}-+-{:->5}-+-{:->9}",
        "", "", "", "", ""
    );

    // ---- Plain f64 GMRES baseline ----
    let mut wall_samples: Vec<Duration> = Vec::with_capacity(args.iters);
    let mut last_iters = 0;
    let mut last_resid = 0.0_f64;
    let mut last_converged = false;
    let cfg_f64 = GmresBackendConfig::<f64> {
        max_iterations: args.max_iter,
        tolerance: args.tolerance,
    };
    for _ in 0..args.iters {
        let mut out = backend.raw_alloc::<f64>(n).expect("out alloc");
        backend
            .synchronize()
            .expect("backend synchronization failed");
        let t = Instant::now();
        let status = gmres_backend_preconditioned(
            backend,
            &a64 as &dyn BackendOperator<B, f64>,
            &p64 as &dyn BackendPreconditioner<B, f64>,
            &rhs64,
            None,
            &mut out,
            &cfg_f64,
        )
        .expect("plain f64 GMRES failed");
        backend
            .synchronize()
            .expect("backend synchronization failed");
        wall_samples.push(t.elapsed());
        last_iters = status.iterations;
        last_resid = status.final_residual;
        last_converged = status.converged;
    }
    let wall_f64 = median(&mut wall_samples);
    println!(
        "{:>4} | {:>11} | {:>5} | {:>5} | {:>9.2e}",
        "f64",
        fmt_dur(wall_f64),
        last_iters,
        0,
        last_resid,
    );
    rows.push(Cell {
        backend: backend_label,
        mode: "f64",
        n,
        wall: wall_f64,
        outer_iters: last_iters,
        inner_iters: 0,
        residual: last_resid,
        converged: last_converged,
    });

    // ---- f64-outer / f32-inner IR ----
    let cfg_ir = GmresIrConfig::<f64> {
        max_outer_iterations: args.max_iter,
        outer_tolerance: args.tolerance,
        max_inner_iterations: args.max_inner_iter,
        inner_tolerance: args.inner_tolerance,
        restart_size: None,
        adaptive_inner_tolerance: true,
        true_residual_check_period: None,
    };
    let mut wall_samples: Vec<Duration> = Vec::with_capacity(args.iters);
    let mut last_outer = 0;
    let mut last_inner = 0;
    let mut last_resid_ir = 0.0_f64;
    let mut last_converged_ir = false;
    for _ in 0..args.iters {
        let mut out = backend.raw_alloc::<f64>(n).expect("out alloc");
        backend
            .synchronize()
            .expect("backend synchronization failed");
        let t = Instant::now();
        let status = gmres_ir_backend_preconditioned::<B, _, _, _, f64>(
            backend, &a64, &a32, &p32, &rhs64, None, &mut out, &cfg_ir,
        )
        .expect("GMRES-IR failed");
        backend
            .synchronize()
            .expect("backend synchronization failed");
        wall_samples.push(t.elapsed());
        last_outer = status.outer_iterations;
        last_inner = status.total_inner_iterations;
        last_resid_ir = status.final_outer_residual;
        last_converged_ir = status.converged;
    }
    let wall_ir = median(&mut wall_samples);
    println!(
        "{:>4} | {:>11} | {:>5} | {:>5} | {:>9.2e}",
        "ir",
        fmt_dur(wall_ir),
        last_outer,
        last_inner,
        last_resid_ir,
    );
    rows.push(Cell {
        backend: backend_label,
        mode: "ir",
        n,
        wall: wall_ir,
        outer_iters: last_outer,
        inner_iters: last_inner,
        residual: last_resid_ir,
        converged: last_converged_ir,
    });

    let speedup = if wall_ir.as_secs_f64() > 0.0 {
        wall_f64.as_secs_f64() / wall_ir.as_secs_f64()
    } else {
        0.0
    };
    println!("       IR speedup vs f64: {speedup:.2}×");
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let mut sizes = args.sizes.clone();
    sizes.sort();
    sizes.dedup();

    let mut rows: Vec<Cell> = Vec::new();

    if !args.skip_cpu {
        let backend = CpuBackend::new();
        println!("\n##### backend: cpu #####");
        for &target in &sizes {
            if args.cpu_size_limit > 0 && target > args.cpu_size_limit {
                println!(
                    "\n[skipping CPU at target={target}: above --cpu-size-limit={}]",
                    args.cpu_size_limit
                );
                continue;
            }
            run_size(&backend, "cpu", &args, target, &mut rows);
        }
    }

    #[cfg(feature = "cuda")]
    if !args.skip_cuda {
        match CudaBackend::new(0) {
            Ok(backend) => {
                println!("\n##### backend: cuda #####");
                for &target in &sizes {
                    run_size(&backend, "cuda", &args, target, &mut rows);
                }
            }
            Err(e) => {
                eprintln!("warning: CudaBackend::new() failed ({e}); skipping CUDA path");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    if !args.skip_cuda {
        eprintln!(
            "note: built without --features cuda; CUDA path skipped. \
             Rebuild with `--features \"cli,cuda\"` to enable."
        );
    }

    // Cross-mode summary table.
    println!("\n##### IR speedup vs plain f64 (per backend / size) #####");
    println!(
        "{:>8} | {:>9} | {:>11} | {:>11} | {:>5} | {:>7}",
        "backend", "n", "f64 wall", "ir wall", "iters", "speedup"
    );
    println!(
        "{:->8}-+-{:->9}-+-{:->11}-+-{:->11}-+-{:->5}-+-{:->7}",
        "", "", "", "", "", ""
    );
    let backend_labels = ["cpu", "cuda"];
    for &bl in &backend_labels {
        for &target in &sizes {
            let nv = (target as f64).cbrt().ceil() as usize;
            let nv3 = nv * nv * nv;
            let f64_row = rows
                .iter()
                .find(|c| c.backend == bl && c.mode == "f64" && c.n == nv3);
            let ir_row = rows
                .iter()
                .find(|c| c.backend == bl && c.mode == "ir" && c.n == nv3);
            let (f64_wall, ir_wall, iters, speedup) = match (f64_row, ir_row) {
                (Some(f), Some(i)) => {
                    let s = if i.wall.as_secs_f64() > 0.0 {
                        f.wall.as_secs_f64() / i.wall.as_secs_f64()
                    } else {
                        0.0
                    };
                    (
                        fmt_dur(f.wall),
                        fmt_dur(i.wall),
                        format!("{}/{}", f.outer_iters, i.outer_iters),
                        format!("{s:.2}×"),
                    )
                }
                _ => continue,
            };
            println!(
                "{:>8} | {:>9} | {:>11} | {:>11} | {:>5} | {:>7}",
                bl, nv3, f64_wall, ir_wall, iters, speedup,
            );
        }
    }

    if let Some(path) = args.csv.as_ref() {
        let mut out = String::new();
        out.push_str("backend,mode,n,wall_seconds,outer_iters,inner_iters,residual,converged\n");
        for row in &rows {
            out.push_str(&format!(
                "{},{},{},{},{},{},{:.6e},{}\n",
                row.backend,
                row.mode,
                row.n,
                row.wall.as_secs_f64(),
                row.outer_iters,
                row.inner_iters,
                row.residual,
                row.converged,
            ));
        }
        std::fs::write(path, out)?;
        println!("\nCSV written to {}", path.display());
    }

    Ok(())
}
