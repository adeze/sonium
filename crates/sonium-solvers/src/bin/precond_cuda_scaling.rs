//! CUDA preconditioner scaling benchmark.
//!
//! Companion to `precond-scaling`. The CPU-only preconditioners
//! (`AmgPreconditioner`, `IlutPreconditioner`, `BlockJacobiPreconditioner`,
//! `AdditiveSchwarzPreconditioner`) don't have GPU implementations because
//! they're inherently fork/join algorithms (recursive multigrid V-cycles,
//! sequential triangular factorisations, overlap-domain ILU). The
//! backend-aware ones — `PointJacobiBackend`, `PolynomialJacobiBackend`,
//! `ChebyshevBackend` — work on any [`Backend`] including CUDA.
//!
//! This binary benchmarks **just the GPU-portable preconditioners**, on
//! both `CpuBackend` and `CudaBackend`, so you get an apples-to-apples
//! comparison: same algorithm, same matrix, two devices.
//!
//! Build:
//! ```text
//! cargo build --release -p sonium-solvers --features "cli,cuda" \
//!     --bin precond-cuda-scaling
//! ```
//!
//! Run:
//! ```text
//! ./target/release/precond-cuda-scaling \
//!     --sizes 100000,500000,1000000,5000000 \
//!     --precond point-jacobi,poly-jacobi,chebyshev \
//!     --degree 4 --iters 3 --applies-per-sample 10 \
//!     --csv cuda.csv
//! ```
//!
//! Output is a per-(size × precond × backend) table. The "throughput"
//! column reports the apply rate in GDOFs/sec — for memory-bandwidth-bound
//! preconditioners the GPU wins by the ratio of HBM/GDDR bandwidth to
//! DDR bandwidth (roughly 5–20×).
//!
//! The bench uses a 3D 7-point Laplacian (matches `precond-scaling`).
//! Real-valued f64 throughout — the CUDA backend's f64 path goes through
//! cuBLAS / cuSPARSE which is the mature route. f32 / mixed precision is a
//! separate axis you can layer on top later.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use sonium_compute::gmres::{BackendCsr, BackendPreconditioner};
use sonium_compute::{
    Backend, ChebyshevBackend, CpuBackend, PointJacobiBackend, PolynomialJacobiBackend,
};

#[cfg(feature = "cuda")]
use sonium_compute::CudaBackend;

#[derive(Parser, Debug)]
#[command(
    name = "precond-cuda-scaling",
    about = "Benchmark backend-aware preconditioners (PointJacobi, PolynomialJacobi, Chebyshev) \
             across CPU and CUDA backends at increasing matrix sizes."
)]
struct Args {
    /// Comma-separated target DOF counts. The bench builds the smallest
    /// 3D cubic 7-point Laplacian whose node count is at least each value.
    #[arg(long, default_value = "100000,500000,1000000", value_delimiter = ',')]
    sizes: Vec<usize>,

    /// Comma-separated preconditioner kinds: point-jacobi, poly-jacobi, chebyshev.
    #[arg(
        long,
        default_value = "point-jacobi,poly-jacobi,chebyshev",
        value_delimiter = ','
    )]
    precond: Vec<String>,

    /// Polynomial degree for `poly-jacobi` and `chebyshev` (ignored for
    /// `point-jacobi`).
    #[arg(long, default_value_t = 4)]
    degree: usize,

    /// Damping factor `omega` for poly-jacobi (ignored otherwise).
    #[arg(long, default_value_t = 0.7)]
    poly_omega: f64,

    /// Eigenvalue ratio for Chebyshev (λ_max / λ_min). PETSc/HYPRE default = 30.
    #[arg(long, default_value_t = 30.0)]
    cheb_ratio: f64,

    /// Number of (setup, apply×k) samples per cell. The median is reported.
    #[arg(long, default_value_t = 3)]
    iters: usize,

    /// Number of `apply()` calls per sample.
    #[arg(long, default_value_t = 10)]
    applies_per_sample: usize,

    /// Skip CPU backend (only run on CUDA).
    #[arg(long)]
    skip_cpu: bool,

    /// Skip CUDA backend (only run on CPU). Useful when validating the
    /// bench harness on a machine without an NVIDIA GPU.
    #[arg(long)]
    skip_cuda: bool,

    /// Optional CSV output path.
    #[arg(long)]
    csv: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrecondKind {
    PointJacobi,
    PolyJacobi,
    Chebyshev,
}

impl PrecondKind {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "point-jacobi" | "pointjacobi" | "jacobi" | "diagonal" => Some(Self::PointJacobi),
            "poly-jacobi" | "polynomial-jacobi" | "polyjacobi" | "neumann" => {
                Some(Self::PolyJacobi)
            }
            "chebyshev" | "cheby" => Some(Self::Chebyshev),
            _ => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::PointJacobi => "point-jacobi",
            Self::PolyJacobi => "poly-jacobi",
            Self::Chebyshev => "chebyshev",
        }
    }
}

/// Build a 3D 7-point Laplacian on an n×n×n grid (real f64). For a
/// requested target DOF count we choose n = ceil(target^(1/3)).
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
                // Strictly increasing column order to satisfy CSR
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
    precond: &'static str,
    n: usize,
    setup: Duration,
    apply: Duration,
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

fn fmt_throughput(n: usize, apply: Duration) -> String {
    let s = apply.as_secs_f64();
    if s <= 0.0 {
        return "       —".to_string();
    }
    let gdofs = (n as f64) / s / 1e9;
    format!("{gdofs:>5.2} GDOFs/s")
}

/// Benchmark one (precond × backend) cell. Returns (setup_time, mean_apply_time).
#[allow(clippy::too_many_arguments)]
fn bench_one<B: Backend>(
    backend: &B,
    a: &BackendCsr<B, f64>,
    row_ptr_host: &[i32],
    col_idx_host: &[i32],
    values_host: &[f64],
    rhs_dev: &B::Buffer<f64>,
    z_dev: &mut B::Buffer<f64>,
    kind: PrecondKind,
    args: &Args,
) -> (Duration, Duration) {
    let mut setup_samples = Vec::with_capacity(args.iters);
    let mut apply_samples = Vec::with_capacity(args.iters);

    // CUDA kernel launches are asynchronous — without a `synchronize()`
    // bracket we'd be measuring dispatch latency, not actual compute. We
    // sync once before timing setup (in case prior work is still in
    // flight), once after setup (so the timer captures the real H→D
    // upload + power-iteration), and once after the apply loop.
    for _ in 0..args.iters {
        backend
            .synchronize()
            .expect("backend synchronization failed");
        match kind {
            PrecondKind::PointJacobi => {
                let setup_t = Instant::now();
                let p = PointJacobiBackend::<B, f64>::from_csr_host(
                    backend,
                    a.m,
                    row_ptr_host,
                    col_idx_host,
                    values_host,
                );
                backend
                    .synchronize()
                    .expect("backend synchronization failed");
                setup_samples.push(setup_t.elapsed());
                let apply_t = Instant::now();
                for _ in 0..args.applies_per_sample {
                    p.apply(backend, rhs_dev, z_dev);
                }
                backend
                    .synchronize()
                    .expect("backend synchronization failed");
                apply_samples.push(apply_t.elapsed() / args.applies_per_sample as u32);
            }
            PrecondKind::PolyJacobi => {
                let setup_t = Instant::now();
                let p = PolynomialJacobiBackend::<B, f64>::from_csr_host(
                    backend,
                    a,
                    row_ptr_host,
                    col_idx_host,
                    values_host,
                    args.degree,
                    args.poly_omega,
                );
                backend
                    .synchronize()
                    .expect("backend synchronization failed");
                setup_samples.push(setup_t.elapsed());
                let apply_t = Instant::now();
                for _ in 0..args.applies_per_sample {
                    p.apply(backend, rhs_dev, z_dev);
                }
                backend
                    .synchronize()
                    .expect("backend synchronization failed");
                apply_samples.push(apply_t.elapsed() / args.applies_per_sample as u32);
            }
            PrecondKind::Chebyshev => {
                let setup_t = Instant::now();
                let p = ChebyshevBackend::<B, f64>::from_csr_host(
                    backend,
                    a,
                    row_ptr_host,
                    col_idx_host,
                    values_host,
                    args.degree,
                    args.cheb_ratio,
                );
                backend
                    .synchronize()
                    .expect("backend synchronization failed");
                setup_samples.push(setup_t.elapsed());
                let apply_t = Instant::now();
                for _ in 0..args.applies_per_sample {
                    p.apply(backend, rhs_dev, z_dev);
                }
                backend
                    .synchronize()
                    .expect("backend synchronization failed");
                apply_samples.push(apply_t.elapsed() / args.applies_per_sample as u32);
            }
        }
    }

    (median(&mut setup_samples), median(&mut apply_samples))
}

fn run_backend<B: Backend>(
    backend: &B,
    label: &'static str,
    args: &Args,
    sizes: &[usize],
    kinds: &[PrecondKind],
    rows: &mut Vec<Cell>,
) {
    println!("\n##### backend: {label} #####");
    for &target in sizes {
        let n_grid = (target as f64).cbrt().ceil() as usize;
        let (n, row_ptr_host, col_idx_host, values_host) = build_laplacian_3d(n_grid);
        let nnz = values_host.len();
        println!("\nMatrix: n={n_grid} cube → {n} DOFs, {nnz} nnz (target {target})");

        let a = BackendCsr::<B, f64>::try_from_host(
            backend,
            n,
            n,
            &row_ptr_host,
            &col_idx_host,
            &values_host,
        )
        .expect("scaling fixture CSR upload must succeed");

        // Synthetic RHS, uploaded once and reused.
        let rhs_host: Vec<f64> = (0..n).map(|i| (i % 17) as f64 * 0.1 + 1.0).collect();
        let mut rhs_dev = backend.raw_alloc::<f64>(n).expect("rhs alloc failed");
        backend.copy_h2d(&rhs_host, &mut rhs_dev);
        let mut z_dev = backend.raw_alloc::<f64>(n).expect("z alloc failed");
        // Drain any in-flight H→D before we start timing.
        backend
            .synchronize()
            .expect("backend synchronization failed");

        println!(
            "{:>14} | {:>11} | {:>11} | {:>14}",
            "precond", "setup", "apply", "throughput"
        );
        println!("{:->14}-+-{:->11}-+-{:->11}-+-{:->14}", "", "", "", "");
        for &kind in kinds {
            let (setup, apply) = bench_one(
                backend,
                &a,
                &row_ptr_host,
                &col_idx_host,
                &values_host,
                &rhs_dev,
                &mut z_dev,
                kind,
                args,
            );
            println!(
                "{:>14} | {:>11} | {:>11} | {:>14}",
                kind.label(),
                fmt_dur(setup),
                fmt_dur(apply),
                fmt_throughput(n, apply),
            );
            rows.push(Cell {
                backend: label,
                precond: kind.label(),
                n,
                setup,
                apply,
            });
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let kinds: Vec<PrecondKind> = args
        .precond
        .iter()
        .filter_map(|s| {
            let k = PrecondKind::parse(s);
            if k.is_none() {
                eprintln!("warning: unknown precond '{s}', ignored");
            }
            k
        })
        .collect();
    if kinds.is_empty() {
        anyhow::bail!("no valid preconditioners requested");
    }

    let mut sizes = args.sizes.clone();
    sizes.sort();
    sizes.dedup();

    let mut rows: Vec<Cell> = Vec::new();

    if !args.skip_cpu {
        let backend = CpuBackend::new();
        run_backend(&backend, "cpu", &args, &sizes, &kinds, &mut rows);
    }

    #[cfg(feature = "cuda")]
    if !args.skip_cuda {
        match CudaBackend::new(0) {
            Ok(backend) => {
                run_backend(&backend, "cuda", &args, &sizes, &kinds, &mut rows);
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

    // Cross-backend speedup summary.
    println!("\n##### cross-backend ratio (apply): cuda speedup vs cpu #####");
    println!(
        "{:>14} | {:>9} | {:>11} | {:>11} | {:>9}",
        "precond", "n", "cpu apply", "cuda apply", "speedup"
    );
    println!(
        "{:->14}-+-{:->9}-+-{:->11}-+-{:->11}-+-{:->9}",
        "", "", "", "", ""
    );
    for n in &sizes {
        let nv = (*n as f64).cbrt().ceil() as usize;
        let nv3 = nv * nv * nv;
        for kind in &kinds {
            let cpu = rows
                .iter()
                .find(|c| c.backend == "cpu" && c.precond == kind.label() && c.n == nv3)
                .map(|c| c.apply);
            let cuda = rows
                .iter()
                .find(|c| c.backend == "cuda" && c.precond == kind.label() && c.n == nv3)
                .map(|c| c.apply);
            let speedup = match (cpu, cuda) {
                (Some(c), Some(g)) if g.as_secs_f64() > 0.0 => {
                    format!("{:.2}×", c.as_secs_f64() / g.as_secs_f64())
                }
                _ => "—".to_string(),
            };
            println!(
                "{:>14} | {:>9} | {:>11} | {:>11} | {:>9}",
                kind.label(),
                nv3,
                cpu.map(fmt_dur).unwrap_or_else(|| "—".to_string()),
                cuda.map(fmt_dur).unwrap_or_else(|| "—".to_string()),
                speedup,
            );
        }
    }

    if let Some(path) = args.csv.as_ref() {
        let mut out = String::new();
        out.push_str("backend,precond,n,setup_seconds,apply_seconds\n");
        for row in &rows {
            out.push_str(&format!(
                "{},{},{},{},{}\n",
                row.backend,
                row.precond,
                row.n,
                row.setup.as_secs_f64(),
                row.apply.as_secs_f64(),
            ));
        }
        std::fs::write(path, out)?;
        println!("\nCSV written to {}", path.display());
    }

    Ok(())
}
