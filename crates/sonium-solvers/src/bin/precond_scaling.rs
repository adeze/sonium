//! Preconditioner thread-scaling benchmark
//!
//! Sweeps the rayon worker count from 1 up to the machine's physical core
//! count and reports setup-time and apply-time speedup curves for each
//! requested preconditioner. Designed to surface serial bottlenecks inside
//! preconditioner construction (interpolation, Galerkin RAP, triplet
//! sort/dedupe, ILU factorisation) and apply (smoother sweeps, triangular
//! solves) — that is, the things `cargo bench` doesn't easily isolate
//! because criterion can't change the global rayon pool size mid-run.
//!
//! Run:
//! ```text
//! cargo run --release -p sonium-solvers --features cli \
//!     --bin precond-scaling -- \
//!     --size 1000000 --threads 1,2,4,8,16,32,64 \
//!     --precond amg,block-jacobi,ilut --iters 3
//! ```
//!
//! Output is a table + CSV (when `--csv path.csv` is given). Each row pins
//! the rayon pool to the requested thread count via a fresh
//! `rayon::ThreadPool` and runs the preconditioner under `pool.install(...)`,
//! so the result reflects exactly that core count regardless of the
//! ambient `RAYON_NUM_THREADS` setting.
//!
//! The matrix used is a 3D 7-point Laplacian (mirrors the FEM Helmholtz
//! sparsity pattern at low frequency), generated at the requested target
//! DOF count. For a 1 M-DOF target the bench builds an n=100 cube
//! (n³ = 1_000_000 nodes, ~7 M nnz) which stresses both the BlockJacobi /
//! ILUT factorisation path and the AMG Galerkin matmul / from_triplets
//! pipeline on a Threadripper-class workload.
//!
//! Reads SONIUM_BW_THREADS the same way the room sweep does so the cap
//! lifts uniformly across the test points.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use num_complex::Complex64;
use rayon::ThreadPoolBuilder;

use sonium_solvers::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgPreconditioner, BlockJacobiPreconditioner,
    CsrMatrix, IluColoringPreconditioner, IluFixedPointPreconditioner, IlutConfig,
    IlutPreconditioner, Preconditioner,
};

#[derive(Parser, Debug)]
#[command(
    name = "precond-scaling",
    about = "Sweep rayon worker counts and measure preconditioner setup/apply scaling."
)]
struct Args {
    /// Target number of unknowns. The bench builds the smallest 3D cubic
    /// 7-point Laplacian whose node count is at least this value.
    #[arg(long, default_value_t = 500_000)]
    size: usize,

    /// Comma-separated rayon thread counts to test. Use a single value to
    /// pin a single configuration.
    #[arg(long, default_value = "1,2,4,8,16,32,64", value_delimiter = ',')]
    threads: Vec<usize>,

    /// Comma-separated preconditioner kinds: amg, block-jacobi,
    /// additive-schwarz, ilut, ilu-coloring, ilu-fixedpoint.
    #[arg(
        long,
        default_value = "amg,block-jacobi,additive-schwarz,ilut,ilu-coloring,ilu-fixedpoint",
        value_delimiter = ','
    )]
    precond: Vec<String>,

    /// Number of (setup, apply×k) samples per (thread × precond) cell. The
    /// median is reported.
    #[arg(long, default_value_t = 3)]
    iters: usize,

    /// Number of `apply()` calls per sample (amortises one-shot overhead;
    /// real GMRES does ~30–60 applies per restart).
    #[arg(long, default_value_t = 20)]
    applies_per_sample: usize,

    /// Optional CSV output path. If given, writes one row per
    /// (precond, threads) cell.
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Use a Helmholtz-style indefinite operator (3D 7-point with `−k² M`
    /// shift on the diagonal). Defaults to a Poisson 3D Laplacian.
    #[arg(long)]
    helmholtz: bool,

    /// Wavenumber `k` when `--helmholtz` is set. Ignored for Poisson.
    #[arg(long, default_value_t = 1.0)]
    k: f64,

    /// AMG configuration: default (RugeStuben), for-parallel (PMIS),
    /// for-fem (RS + SGS), for-aggregation (SA + Chebyshev).
    #[arg(long, default_value = "for-parallel")]
    amg_config: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrecondKind {
    Amg,
    BlockJacobi,
    AdditiveSchwarz,
    Ilut,
    IluColoring,
    IluFixedPoint,
}

impl PrecondKind {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "amg" => Some(Self::Amg),
            "block-jacobi" | "blockjacobi" | "bj" => Some(Self::BlockJacobi),
            "additive-schwarz" | "schwarz" | "as" => Some(Self::AdditiveSchwarz),
            "ilut" => Some(Self::Ilut),
            "ilu-coloring" | "ilucoloring" | "coloring" => Some(Self::IluColoring),
            "ilu-fixedpoint" | "ilu-fixed-point" | "ilufixedpoint" | "fixedpoint"
            | "fixed-point" => Some(Self::IluFixedPoint),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Amg => "amg",
            Self::BlockJacobi => "block-jacobi",
            Self::AdditiveSchwarz => "additive-schwarz",
            Self::Ilut => "ilut",
            Self::IluColoring => "ilu-coloring",
            Self::IluFixedPoint => "ilu-fixedpoint",
        }
    }
}

/// Build a 3D 7-point Laplacian on an n×n×n grid. SPD, real-valued. For a
/// requested target DOF count we choose n = ceil(target^(1/3)).
fn build_laplacian_3d(n: usize) -> CsrMatrix<Complex64> {
    let nn = n * n * n;
    let mut triplets: Vec<(usize, usize, Complex64)> = Vec::with_capacity(7 * nn);
    let h2 = 1.0 / ((n + 1) as f64).powi(2);
    let diag = Complex64::new(6.0 / h2, 0.0);
    let off = Complex64::new(-1.0 / h2, 0.0);

    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let row = i + n * (j + n * k);
                triplets.push((row, row, diag));
                if i > 0 {
                    triplets.push((row, row - 1, off));
                }
                if i + 1 < n {
                    triplets.push((row, row + 1, off));
                }
                if j > 0 {
                    triplets.push((row, row - n, off));
                }
                if j + 1 < n {
                    triplets.push((row, row + n, off));
                }
                if k > 0 {
                    triplets.push((row, row - n * n, off));
                }
                if k + 1 < n {
                    triplets.push((row, row + n * n, off));
                }
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

/// Apply a Helmholtz-style indefinite shift A := A − k² · I to a Laplacian.
/// Approximates the spectrum of the FEM shifted-Laplacian without needing
/// a real mesh.
fn add_helmholtz_shift(mut a: CsrMatrix<Complex64>, k: f64) -> CsrMatrix<Complex64> {
    let shift = Complex64::new(-k * k, 0.05 * k * k);
    let n = a.num_rows;
    for i in 0..n {
        let r = a.row_range(i);
        for idx in r {
            if a.col_indices[idx] == i {
                a.values[idx] += shift;
                break;
            }
        }
    }
    a
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn fmt_secs(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s >= 1.0 {
        format!("{s:>7.2} s")
    } else {
        format!("{:>5.1} ms", s * 1000.0)
    }
}

fn pick_amg_config(label: &str) -> AmgConfig {
    match label.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "default" => AmgConfig::default(),
        "for-parallel" | "parallel" => AmgConfig::for_parallel(),
        "for-fem" | "fem" => AmgConfig::for_fem(),
        "for-aggregation" | "aggregation" | "sa" => AmgConfig::for_aggregation(),
        "for-difficult" | "difficult" => AmgConfig::for_difficult_problems(),
        other => {
            eprintln!("warning: unknown amg config '{other}', using for-parallel");
            AmgConfig::for_parallel()
        }
    }
}

/// Run setup once + N applies. Returns (setup_time, mean_apply_time).
fn measure_one_sample(
    matrix: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    kind: PrecondKind,
    applies: usize,
    amg_config: &AmgConfig,
) -> (Duration, Duration) {
    match kind {
        PrecondKind::Amg => {
            let setup = Instant::now();
            let p = AmgPreconditioner::from_csr(matrix, amg_config.clone());
            let setup_dt = setup.elapsed();
            let apply_start = Instant::now();
            let mut sink = Complex64::new(0.0, 0.0);
            for _ in 0..applies {
                let z = p.apply(rhs);
                sink += z[0];
            }
            std::hint::black_box(sink);
            (setup_dt, apply_start.elapsed() / applies as u32)
        }
        PrecondKind::BlockJacobi => {
            let cfg = IlutConfig {
                drop_tolerance: 1e-4,
                fill_factor: 20,
                pivot_tolerance: 1e-10,
                pivoting: true,
            };
            let setup = Instant::now();
            let p = BlockJacobiPreconditioner::from_csr_auto(matrix, cfg);
            let setup_dt = setup.elapsed();
            let apply_start = Instant::now();
            let mut sink = Complex64::new(0.0, 0.0);
            for _ in 0..applies {
                let z = p.apply(rhs);
                sink += z[0];
            }
            std::hint::black_box(sink);
            (setup_dt, apply_start.elapsed() / applies as u32)
        }
        PrecondKind::AdditiveSchwarz => {
            let setup = Instant::now();
            let p = AdditiveSchwarzPreconditioner::from_csr_auto(matrix, 1);
            let setup_dt = setup.elapsed();
            let apply_start = Instant::now();
            let mut sink = Complex64::new(0.0, 0.0);
            for _ in 0..applies {
                let z = p.apply(rhs);
                sink += z[0];
            }
            std::hint::black_box(sink);
            (setup_dt, apply_start.elapsed() / applies as u32)
        }
        PrecondKind::Ilut => {
            let setup = Instant::now();
            let cfg = IlutConfig {
                drop_tolerance: 1e-4,
                fill_factor: 20,
                pivot_tolerance: 1e-10,
                pivoting: true,
            };
            let p = IlutPreconditioner::from_csr_with_config(matrix, &cfg);
            let setup_dt = setup.elapsed();
            let apply_start = Instant::now();
            let mut sink = Complex64::new(0.0, 0.0);
            for _ in 0..applies {
                let z = p.apply(rhs);
                sink += z[0];
            }
            std::hint::black_box(sink);
            (setup_dt, apply_start.elapsed() / applies as u32)
        }
        PrecondKind::IluColoring => {
            let setup = Instant::now();
            let p = IluColoringPreconditioner::from_csr(matrix);
            let setup_dt = setup.elapsed();
            let apply_start = Instant::now();
            let mut sink = Complex64::new(0.0, 0.0);
            for _ in 0..applies {
                let z = p.apply(rhs);
                sink += z[0];
            }
            std::hint::black_box(sink);
            (setup_dt, apply_start.elapsed() / applies as u32)
        }
        PrecondKind::IluFixedPoint => {
            // Fixed-point iteration count matches the FEM runner default (3
            // sweeps per apply — adequate for moderately ill-conditioned
            // operators, cheap enough that many threads matter).
            const FP_ITERS: usize = 3;
            let setup = Instant::now();
            let p = IluFixedPointPreconditioner::from_csr(matrix, FP_ITERS);
            let setup_dt = setup.elapsed();
            let apply_start = Instant::now();
            let mut sink = Complex64::new(0.0, 0.0);
            for _ in 0..applies {
                let z = p.apply(rhs);
                sink += z[0];
            }
            std::hint::black_box(sink);
            (setup_dt, apply_start.elapsed() / applies as u32)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    threads: usize,
    setup: Duration,
    apply: Duration,
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

    let n = (args.size as f64).cbrt().ceil() as usize;
    let actual = n * n * n;
    println!(
        "Building 3D 7-point Laplacian: n={n} ({actual} DOFs, target {}). \
         Helmholtz shift: {}",
        args.size,
        if args.helmholtz {
            format!("k={:.3}", args.k)
        } else {
            "off".to_string()
        }
    );
    let build_start = Instant::now();
    let mut matrix = build_laplacian_3d(n);
    if args.helmholtz {
        matrix = add_helmholtz_shift(matrix, args.k);
    }
    println!(
        "Matrix built in {} ({} nnz)",
        fmt_secs(build_start.elapsed()),
        matrix.nnz()
    );

    let rhs = Vec::from_iter(
        (0..matrix.num_rows).map(|i| Complex64::new((i % 17) as f64 * 0.1 + 1.0, 0.0)),
    );

    let mut threads_sorted: Vec<usize> = args.threads.iter().copied().filter(|&t| t > 0).collect();
    threads_sorted.sort();
    threads_sorted.dedup();

    let mut all_results: Vec<(PrecondKind, Vec<Cell>)> = Vec::with_capacity(kinds.len());

    for kind in &kinds {
        println!("\n=== {} ===", kind.label());
        println!(
            "{:>8} | {:>11} | {:>11} | {:>9} | {:>9}",
            "threads", "setup", "apply", "setup×", "apply×"
        );
        println!(
            "{:->8}-+-{:->11}-+-{:->11}-+-{:->9}-+-{:->9}",
            "", "", "", "", ""
        );

        let mut cells: Vec<Cell> = Vec::with_capacity(threads_sorted.len());
        let mut baseline: Option<Cell> = None;

        for &t in &threads_sorted {
            let pool = ThreadPoolBuilder::new()
                .num_threads(t)
                .stack_size(8 * 1024 * 1024)
                .thread_name(move |idx| format!("scaling-{t}-{idx}"))
                .build()?;

            let mut setups = Vec::with_capacity(args.iters);
            let mut applies = Vec::with_capacity(args.iters);
            let amg_cfg = pick_amg_config(&args.amg_config);
            for _ in 0..args.iters {
                let (s, a) = pool.install(|| {
                    measure_one_sample(&matrix, &rhs, *kind, args.applies_per_sample, &amg_cfg)
                });
                setups.push(s);
                applies.push(a);
            }
            let setup = median(&mut setups);
            let apply = median(&mut applies);
            let cell = Cell {
                threads: t,
                setup,
                apply,
            };
            if baseline.is_none() {
                baseline = Some(cell);
            }
            let b = baseline.unwrap();
            let setup_speedup = b.setup.as_secs_f64() / setup.as_secs_f64();
            let apply_speedup = b.apply.as_secs_f64() / apply.as_secs_f64();
            println!(
                "{:>8} | {:>11} | {:>11} | {:>8.2}× | {:>8.2}×",
                t,
                fmt_secs(setup),
                fmt_secs(apply),
                setup_speedup,
                apply_speedup,
            );
            cells.push(cell);
        }
        all_results.push((*kind, cells));
    }

    if let Some(path) = args.csv.as_ref() {
        let mut out = String::new();
        out.push_str("precond,threads,setup_seconds,apply_seconds,setup_speedup,apply_speedup\n");
        for (kind, cells) in &all_results {
            let base = cells.first().copied().unwrap_or(Cell {
                threads: 1,
                setup: Duration::from_secs(1),
                apply: Duration::from_secs(1),
            });
            for c in cells {
                let s_speedup = base.setup.as_secs_f64() / c.setup.as_secs_f64();
                let a_speedup = base.apply.as_secs_f64() / c.apply.as_secs_f64();
                out.push_str(&format!(
                    "{},{},{},{},{:.4},{:.4}\n",
                    kind.label(),
                    c.threads,
                    c.setup.as_secs_f64(),
                    c.apply.as_secs_f64(),
                    s_speedup,
                    a_speedup,
                ));
            }
        }
        std::fs::write(path, out)?;
        println!("\nCSV written to {}", path.display());
    }

    Ok(())
}
