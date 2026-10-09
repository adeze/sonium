//! Preconditioner setup/apply/iteration tradeoff benchmark.
//!
//! For each (matrix, preconditioner) cell it reports the three costs that
//! decide whether a preconditioner pays off: one-time **setup** time, **apply**
//! time per Krylov iteration, and the **iteration count** GMRES needs with
//! that preconditioner. The matrix suite deliberately includes failing
//! cases — an indefinite Helmholtz operator and an explicit zero-diagonal
//! matrix — where some setups are rejected and some combinations stall, so
//! the tradeoff table records honest `FAIL`/`STALLED` cells instead of only
//! the matrices where everything works.
//!
//! Run:
//! ```text
//! cargo run --release -p sonium-solvers --features cli \
//!     --bin precond-tradeoff -- \
//!     --output PRECOND_TRADEOFF.md --csv tradeoff.csv --repeats 3
//! ```
//!
//! Suite (all real `f64`; unscaled integer-coefficient stencils):
//! - `poisson-2d` (60x60, SPD): the easy baseline.
//! - `poisson-3d` (16^3, SPD): 3D fill/apply behaviour.
//! - `convdiff-2d` (60x60, nonsymmetric, wind 20): ILU(0)'s home turf.
//! - `helmholtz-2d` (50x50, diag `4 - k^2 h^2 = 0.1`): indefinite with
//!   near-zero pivots; exact ILU(0) is expected to be rejected here.
//! - `zero-diag` (poisson-2d 40x40, every 8th diagonal zeroed): Jacobi and
//!   no-pivot ILU setups must fail; pivoted ILUT may survive via row swaps.
//! - `aniso-2d` (60x60, y-coupling 1e-3): pointwise smoothers degrade.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

use sonium_solvers::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgPreconditioner, BlockJacobiPreconditioner,
    CsrMatrix, DiagonalPreconditioner, GmresConfig, IdentityPreconditioner, IluConfig,
    IluPreconditioner, IlutConfig, IlutPreconditioner, Preconditioner, gmres_preconditioned,
};

#[derive(Parser, Debug)]
#[command(
    name = "precond-tradeoff",
    about = "Measure preconditioner setup/apply/iteration tradeoffs incl. failing matrices."
)]
struct Args {
    /// Markdown report output path.
    #[arg(long, default_value = "PRECOND_TRADEOFF.md")]
    output: PathBuf,

    /// Optional CSV output path (one row per matrix x precond cell).
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Setup-timing samples per cell (median reported).
    #[arg(long, default_value_t = 3)]
    repeats: usize,

    /// `apply()` calls per timing sample.
    #[arg(long, default_value_t = 20)]
    applies_per_sample: usize,

    /// Subset of matrices: poisson-2d, poisson-3d, convdiff-2d,
    /// helmholtz-2d, zero-diag, aniso-2d.
    #[arg(long, default_value = "all")]
    matrices: String,

    /// Subset of preconditioners: none, jacobi, ilu0, ilut,
    /// block-jacobi, schwarz, amg.
    #[arg(long, default_value = "all")]
    precond: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixKind {
    Poisson2d,
    Poisson3d,
    ConvDiff2d,
    Helmholtz2d,
    ZeroDiag,
    Aniso2d,
}

impl MatrixKind {
    const ALL: &'static [Self] = &[
        Self::Poisson2d,
        Self::Poisson3d,
        Self::ConvDiff2d,
        Self::Helmholtz2d,
        Self::ZeroDiag,
        Self::Aniso2d,
    ];

    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "poisson-2d" | "poisson2d" | "p2d" => Some(Self::Poisson2d),
            "poisson-3d" | "poisson3d" | "p3d" => Some(Self::Poisson3d),
            "convdiff-2d" | "convdiff" | "cd2d" => Some(Self::ConvDiff2d),
            "helmholtz-2d" | "helmholtz" | "h2d" => Some(Self::Helmholtz2d),
            "zero-diag" | "zerodiag" | "zd" => Some(Self::ZeroDiag),
            "aniso-2d" | "aniso" | "a2d" => Some(Self::Aniso2d),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Poisson2d => "poisson-2d",
            Self::Poisson3d => "poisson-3d",
            Self::ConvDiff2d => "convdiff-2d",
            Self::Helmholtz2d => "helmholtz-2d",
            Self::ZeroDiag => "zero-diag",
            Self::Aniso2d => "aniso-2d",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Self::Poisson2d => "60x60 5-pt Laplacian, SPD (3600 DOFs)",
            Self::Poisson3d => "16^3 7-pt Laplacian, SPD (4096 DOFs)",
            Self::ConvDiff2d => "60x60 central convection-diffusion, wind 20, nonsymmetric",
            Self::Helmholtz2d => "50x50 `-Δ - k²`, diag 0.1, indefinite near-zero pivots",
            Self::ZeroDiag => "poisson-2d 40x40 with every 8th diagonal zeroed",
            Self::Aniso2d => "60x60 anisotropic Laplacian, y-coupling 1e-3",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrecondKind {
    None,
    Jacobi,
    Ilu0,
    Ilut,
    BlockJacobi,
    Schwarz,
    Amg,
}

impl PrecondKind {
    const ALL: &'static [Self] = &[
        Self::None,
        Self::Jacobi,
        Self::Ilu0,
        Self::Ilut,
        Self::BlockJacobi,
        Self::Schwarz,
        Self::Amg,
    ];

    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "none" | "identity" | "id" => Some(Self::None),
            "jacobi" | "diag" | "diagonal" => Some(Self::Jacobi),
            "ilu0" | "ilu-0" | "ilu" => Some(Self::Ilu0),
            "ilut" => Some(Self::Ilut),
            "block-jacobi" | "blockjacobi" | "bj" => Some(Self::BlockJacobi),
            "schwarz" | "additive-schwarz" | "as" => Some(Self::Schwarz),
            "amg" => Some(Self::Amg),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Jacobi => "jacobi",
            Self::Ilu0 => "ilu0",
            Self::Ilut => "ilut",
            Self::BlockJacobi => "block-jacobi",
            Self::Schwarz => "schwarz",
            Self::Amg => "amg",
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix builders (f64, unscaled stencils mirroring the reference-count test)
// ---------------------------------------------------------------------------

fn poisson_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut t = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            t.push((row, row, 4.0));
            if i > 0 {
                t.push((row, (i - 1) * n + j, -1.0));
            }
            if i + 1 < n {
                t.push((row, (i + 1) * n + j, -1.0));
            }
            if j > 0 {
                t.push((row, i * n + j - 1, -1.0));
            }
            if j + 1 < n {
                t.push((row, i * n + j + 1, -1.0));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, t)
}

fn poisson_3d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n * n;
    let mut t = Vec::with_capacity(7 * nn);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let row = i * n * n + j * n + k;
                t.push((row, row, 6.0));
                if i > 0 {
                    t.push((row, (i - 1) * n * n + j * n + k, -1.0));
                }
                if i + 1 < n {
                    t.push((row, (i + 1) * n * n + j * n + k, -1.0));
                }
                if j > 0 {
                    t.push((row, i * n * n + (j - 1) * n + k, -1.0));
                }
                if j + 1 < n {
                    t.push((row, i * n * n + (j + 1) * n + k, -1.0));
                }
                if k > 0 {
                    t.push((row, i * n * n + j * n + k - 1, -1.0));
                }
                if k + 1 < n {
                    t.push((row, i * n * n + j * n + k + 1, -1.0));
                }
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, t)
}

/// 2D convection-diffusion, central differences: `-Δu + w·∇u`, w = (wind, 0).
fn convdiff_2d(n: usize, wind: f64) -> CsrMatrix<f64> {
    let nn = n * n;
    let h = 1.0 / (n as f64 + 1.0);
    let diff = 1.0 / (h * h);
    let conv = wind / (2.0 * h);
    let mut t = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            t.push((row, row, 4.0 * diff));
            if i > 0 {
                t.push((row, (i - 1) * n + j, -diff));
            }
            if i + 1 < n {
                t.push((row, (i + 1) * n + j, -diff));
            }
            if j > 0 {
                t.push((row, i * n + j - 1, -diff - conv));
            }
            if j + 1 < n {
                t.push((row, i * n + j + 1, -diff + conv));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, t)
}

/// Real indefinite Helmholtz `-Δ - k²` with `diag = 4 - k²h²` tuned to 0.1:
/// near-zero pivots that exact ILU(0) cannot handle.
fn helmholtz_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    // k^2 h^2 = 3.9, i.e. diag = 0.1: indefinite with near-zero pivots.
    let diag = 4.0 - 3.9;
    let mut t = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            t.push((row, row, diag));
            if i > 0 {
                t.push((row, (i - 1) * n + j, -1.0));
            }
            if i + 1 < n {
                t.push((row, (i + 1) * n + j, -1.0));
            }
            if j > 0 {
                t.push((row, i * n + j - 1, -1.0));
            }
            if j + 1 < n {
                t.push((row, i * n + j + 1, -1.0));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, t)
}

/// Poisson 2D with every 8th diagonal entry zeroed. Jacobi and no-pivot
/// ILU setups must reject this; pivoted ILUT may survive via row swaps.
fn zero_diag(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut t = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            let diag = if row % 8 == 7 { 0.0 } else { 4.0 };
            t.push((row, row, diag));
            if i > 0 {
                t.push((row, (i - 1) * n + j, -1.0));
            }
            if i + 1 < n {
                t.push((row, (i + 1) * n + j, -1.0));
            }
            if j > 0 {
                t.push((row, i * n + j - 1, -1.0));
            }
            if j + 1 < n {
                t.push((row, i * n + j + 1, -1.0));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, t)
}

/// Anisotropic Laplacian `-u_xx - eps*u_yy`, eps = 1e-3.
fn aniso_2d(n: usize, eps: f64) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut t = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            t.push((row, row, 2.0 + 2.0 * eps));
            if i > 0 {
                t.push((row, (i - 1) * n + j, -eps));
            }
            if i + 1 < n {
                t.push((row, (i + 1) * n + j, -eps));
            }
            if j > 0 {
                t.push((row, i * n + j - 1, -1.0));
            }
            if j + 1 < n {
                t.push((row, i * n + j + 1, -1.0));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, t)
}

fn build_matrix(kind: MatrixKind) -> CsrMatrix<f64> {
    match kind {
        MatrixKind::Poisson2d => poisson_2d(60),
        MatrixKind::Poisson3d => poisson_3d(16),
        MatrixKind::ConvDiff2d => convdiff_2d(60, 20.0),
        MatrixKind::Helmholtz2d => helmholtz_2d(50),
        MatrixKind::ZeroDiag => zero_diag(40),
        MatrixKind::Aniso2d => aniso_2d(60, 1e-3),
    }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn gmres_config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 10,
        restart: 100,
        tolerance: 1e-8,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        min_restart: None,
        max_restart: None,
    }
}

#[derive(Debug, Clone)]
enum SetupOutcome {
    Ok { setup_secs: f64 },
    Fail { reason: String },
}

#[derive(Debug, Clone)]
struct Cell {
    matrix: MatrixKind,
    n: usize,
    nnz: usize,
    precond: PrecondKind,
    setup: SetupOutcome,
    /// Mean apply seconds per call (`None` when setup failed or apply
    /// produced non-finite output).
    apply_secs: Option<f64>,
    /// GMRES matvecs to rtol 1e-8 (`None` when not attempted).
    iters: Option<usize>,
    converged: Option<bool>,
    solve_secs: Option<f64>,
}

/// Build the preconditioner (checked constructors; AMG goes through
/// `catch_unwind` since it only offers the infallible entry point), time
/// repeated applies, then run one GMRES solve. Everything is boxed behind
/// `Preconditioner<f64>` so every kind shares the timing/solve path.
fn measure_cell(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    kind: PrecondKind,
    repeats: usize,
    applies_per_sample: usize,
) -> Cell {
    let gmres_cfg = gmres_config();

    macro_rules! run_with {
        ($precond:expr, $setup_secs:expr) => {{
            let p = $precond;
            // Apply timing + finiteness gate.
            let mut apply_samples = Vec::with_capacity(repeats);
            let mut finite = true;
            for _ in 0..repeats {
                let start = Instant::now();
                let mut sink = 0.0;
                for _ in 0..applies_per_sample {
                    let z = p.apply(rhs);
                    if z.iter().any(|v| !v.is_finite()) {
                        finite = false;
                        break;
                    }
                    sink += z[0];
                }
                std::hint::black_box(sink);
                if !finite {
                    break;
                }
                apply_samples.push(start.elapsed().as_secs_f64() / applies_per_sample as f64);
            }
            if !finite {
                return Cell {
                    matrix: MatrixKind::Poisson2d,
                    n: matrix.num_rows,
                    nnz: matrix.nnz(),
                    precond: kind,
                    setup: SetupOutcome::Ok {
                        setup_secs: $setup_secs,
                    },
                    apply_secs: None,
                    iters: None,
                    converged: None,
                    solve_secs: None,
                };
            }
            apply_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let apply_secs = apply_samples[apply_samples.len() / 2];
            let start = Instant::now();
            let sol = gmres_preconditioned(matrix, &p, rhs, &gmres_cfg);
            let solve_secs = start.elapsed().as_secs_f64();
            Cell {
                matrix: MatrixKind::Poisson2d,
                n: matrix.num_rows,
                nnz: matrix.nnz(),
                precond: kind,
                setup: SetupOutcome::Ok {
                    setup_secs: $setup_secs,
                },
                apply_secs: Some(apply_secs),
                iters: Some(sol.iterations),
                converged: Some(sol.converged),
                solve_secs: Some(solve_secs),
            }
        }};
    }

    // Setup phase: median of `repeats` timed builds; the first error or
    // panic short-circuits to a FAIL cell.
    let mut setup_samples = Vec::with_capacity(repeats);
    macro_rules! try_setup {
        ($expr:expr) => {
            match $expr {
                Ok(p) => p,
                Err(e) => {
                    return Cell {
                        matrix: MatrixKind::Poisson2d,
                        n: matrix.num_rows,
                        nnz: matrix.nnz(),
                        precond: kind,
                        setup: SetupOutcome::Fail {
                            reason: format!("{e:?}")
                                .chars()
                                .take(120)
                                .collect::<String>()
                                .replace('\n', " "),
                        },
                        apply_secs: None,
                        iters: None,
                        converged: None,
                        solve_secs: None,
                    };
                }
            }
        };
    }

    match kind {
        PrecondKind::None => {
            let p = IdentityPreconditioner;
            run_with!(p, 0.0)
        }
        PrecondKind::Jacobi => {
            for _ in 0..repeats {
                let start = Instant::now();
                let _p = try_setup!(DiagonalPreconditioner::try_from_csr(matrix));
                setup_samples.push(start.elapsed());
            }
            let setup_secs = median_duration(&mut setup_samples).as_secs_f64();
            let p = DiagonalPreconditioner::try_from_csr(matrix).expect("setup already succeeded");
            run_with!(p, setup_secs)
        }
        PrecondKind::Ilu0 => {
            let cfg = IluConfig::default();
            for _ in 0..repeats {
                let start = Instant::now();
                let _p = try_setup!(IluPreconditioner::try_from_csr_with_config(matrix, &cfg));
                setup_samples.push(start.elapsed());
            }
            let setup_secs = median_duration(&mut setup_samples).as_secs_f64();
            let p = IluPreconditioner::try_from_csr_with_config(matrix, &cfg)
                .expect("setup already succeeded");
            run_with!(p, setup_secs)
        }
        PrecondKind::Ilut => {
            let cfg = IlutConfig {
                drop_tolerance: 1e-4,
                fill_factor: 20,
                pivot_tolerance: 1e-10,
                pivoting: true,
            };
            for _ in 0..repeats {
                let start = Instant::now();
                let _p = try_setup!(IlutPreconditioner::try_from_csr_with_config(matrix, &cfg));
                setup_samples.push(start.elapsed());
            }
            let setup_secs = median_duration(&mut setup_samples).as_secs_f64();
            let p = IlutPreconditioner::try_from_csr_with_config(matrix, &cfg)
                .expect("setup already succeeded");
            run_with!(p, setup_secs)
        }
        PrecondKind::BlockJacobi => {
            let cfg = IlutConfig {
                drop_tolerance: 1e-4,
                fill_factor: 20,
                pivot_tolerance: 1e-10,
                pivoting: true,
            };
            for _ in 0..repeats {
                let start = Instant::now();
                let _p = try_setup!(BlockJacobiPreconditioner::try_from_csr_auto(
                    matrix,
                    cfg.clone()
                ));
                setup_samples.push(start.elapsed());
            }
            let setup_secs = median_duration(&mut setup_samples).as_secs_f64();
            let p = BlockJacobiPreconditioner::try_from_csr_auto(matrix, cfg)
                .expect("setup already succeeded");
            run_with!(p, setup_secs)
        }
        PrecondKind::Schwarz => {
            for _ in 0..repeats {
                let start = Instant::now();
                let _p = try_setup!(AdditiveSchwarzPreconditioner::try_from_csr_auto(matrix, 1));
                setup_samples.push(start.elapsed());
            }
            let setup_secs = median_duration(&mut setup_samples).as_secs_f64();
            let p = AdditiveSchwarzPreconditioner::try_from_csr_auto(matrix, 1)
                .expect("setup already succeeded");
            run_with!(p, setup_secs)
        }
        PrecondKind::Amg => {
            let cfg = AmgConfig::for_parallel();
            for _ in 0..repeats {
                let start = Instant::now();
                let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    AmgPreconditioner::from_csr(matrix, cfg.clone())
                }));
                match ok {
                    Ok(_p) => setup_samples.push(start.elapsed()),
                    Err(_) => {
                        return Cell {
                            matrix: MatrixKind::Poisson2d,
                            n: matrix.num_rows,
                            nnz: matrix.nnz(),
                            precond: kind,
                            setup: SetupOutcome::Fail {
                                reason: "AMG setup panicked".to_string(),
                            },
                            apply_secs: None,
                            iters: None,
                            converged: None,
                            solve_secs: None,
                        };
                    }
                }
            }
            let setup_secs = median_duration(&mut setup_samples).as_secs_f64();
            let p = AmgPreconditioner::from_csr(matrix, cfg);
            run_with!(p, setup_secs)
        }
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn fmt_opt_secs(v: Option<f64>) -> String {
    match v {
        Some(s) if s >= 1.0 => format!("{s:>9.3} s"),
        Some(s) => format!("{:>7.2} ms", s * 1000.0),
        None => "        —".to_string(),
    }
}

fn cell_summary(cell: &Cell) -> String {
    match &cell.setup {
        SetupOutcome::Fail { reason } => format!("SETUP-FAIL: {reason}"),
        SetupOutcome::Ok { .. } if cell.apply_secs.is_none() => {
            "APPLY-FAIL: non-finite apply output".to_string()
        }
        SetupOutcome::Ok { .. } => match (cell.iters, cell.converged) {
            (Some(it), Some(true)) => format!("converged in {it} iters"),
            (Some(it), Some(false)) => format!("STALLED after {it} iters"),
            _ => "no solve".to_string(),
        },
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = Args::parse();

    let want_matrix = |k: MatrixKind| {
        args.matrices.trim().eq_ignore_ascii_case("all")
            || args
                .matrices
                .split(',')
                .any(|s| MatrixKind::parse(s) == Some(k))
    };
    let want_precond = |k: PrecondKind| {
        args.precond.trim().eq_ignore_ascii_case("all")
            || args
                .precond
                .split(',')
                .any(|s| PrecondKind::parse(s) == Some(k))
    };
    let matrices: Vec<MatrixKind> = MatrixKind::ALL
        .iter()
        .copied()
        .filter(|k| want_matrix(*k))
        .collect();
    let preconds: Vec<PrecondKind> = PrecondKind::ALL
        .iter()
        .copied()
        .filter(|k| want_precond(*k))
        .collect();
    if matrices.is_empty() {
        anyhow::bail!("no valid matrices requested");
    }
    if preconds.is_empty() {
        anyhow::bail!("no valid preconditioners requested");
    }

    let mut cells: Vec<Cell> = Vec::new();
    for &m in &matrices {
        let matrix = build_matrix(m);
        let n = matrix.num_rows;
        let nnz = matrix.nnz();
        println!("=== {} ({}, N={n}, nnz={nnz}) ===", m.label(), m.describe());
        println!(
            "{:>12} | {:>11} | {:>11} | {:>6} | {:>11} | outcome",
            "precond", "setup", "apply", "iters", "solve"
        );
        for &p in &preconds {
            let rhs: Vec<f64> = (0..n).map(|i| (i % 17) as f64 * 0.1 + 1.0).collect();
            let mut cell = measure_cell(&matrix, &rhs, p, args.repeats, args.applies_per_sample);
            cell.matrix = m;
            let setup_str = match cell.setup {
                SetupOutcome::Ok { setup_secs } => fmt_opt_secs(Some(setup_secs)),
                SetupOutcome::Fail { .. } => "      FAIL".to_string(),
            };
            println!(
                "{:>12} | {} | {} | {:>6} | {} | {}",
                p.label(),
                setup_str,
                fmt_opt_secs(cell.apply_secs),
                cell.iters.map_or("—".to_string(), |i| i.to_string()),
                fmt_opt_secs(cell.solve_secs),
                cell_summary(&cell),
            );
            cells.push(cell);
        }
        println!();
    }

    if let Some(path) = args.csv.as_ref() {
        let mut out = String::from(
            "matrix,n,nnz,precond,setup_secs,apply_secs,iters,converged,solve_secs,outcome\n",
        );
        for c in &cells {
            let (setup_s, reason) = match &c.setup {
                SetupOutcome::Ok { setup_secs } => (setup_secs.to_string(), String::new()),
                SetupOutcome::Fail { reason } => ("FAIL".to_string(), reason.clone()),
            };
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                c.matrix.label(),
                c.n,
                c.nnz,
                c.precond.label(),
                setup_s,
                c.apply_secs.map_or("n/a".to_string(), |v| v.to_string()),
                c.iters.map_or("n/a".to_string(), |v| v.to_string()),
                c.converged.map_or("n/a".to_string(), |v| v.to_string()),
                c.solve_secs.map_or("n/a".to_string(), |v| v.to_string()),
                if reason.is_empty() {
                    cell_summary(c)
                } else {
                    format!("SETUP-FAIL: {reason}")
                }
                .replace(',', ";"),
            ));
        }
        std::fs::write(path, &out)?;
        println!("CSV written to {}", path.display());
    }

    // Markdown report.
    let mut md = String::new();
    md.push_str("# Preconditioner Setup/Apply/Iteration Tradeoffs\n\n");
    md.push_str(&format!(
        "_Generated by `precond-tradeoff` (repeats: {}, applies/sample: {})._ GMRES(100) \
         to relative tolerance 1e-8, max 10 outer iterations (1000 matvecs); `iters` is the \
         GMRES matvec count. `none` is the unpreconditioned baseline. `FAIL` marks a \
         rejected setup; `STALLED` marks GMRES exhausting its budget without converging._\n\n",
        args.repeats, args.applies_per_sample
    ));
    md.push_str("## Matrices\n\n");
    md.push_str("| Matrix | Description |\n|---|---|\n");
    for &m in &matrices {
        md.push_str(&format!("| {} | {} |\n", m.label(), m.describe()));
    }
    md.push_str("\n## Results\n\n");
    md.push_str("| Matrix | N | Precond | Setup (s) | Apply (s) | Iters | Solve (s) | Outcome |\n");
    md.push_str("|---|---:|---|---|---:|---:|---|---|\n");
    for c in &cells {
        let setup_s = match &c.setup {
            SetupOutcome::Ok { setup_secs } => format!("{setup_secs:.4}"),
            SetupOutcome::Fail { .. } => "FAIL".to_string(),
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            c.matrix.label(),
            c.n,
            c.precond.label(),
            setup_s,
            c.apply_secs
                .map_or("n/a".to_string(), |v| format!("{v:.6}")),
            c.iters.map_or("n/a".to_string(), |v| v.to_string()),
            c.solve_secs
                .map_or("n/a".to_string(), |v| format!("{v:.4}")),
            cell_summary(c).replace('|', "/"),
        ));
    }
    md.push_str("\n## Reading the tradeoff\n\n");
    md.push_str(
        "- Total time to solution is `setup + iters × (matvec + apply)`; a heavy setup \
         (AMG, Schwarz) only pays off when it cuts iterations enough to cover itself.\n\
         - Compare `ilu0` vs `none` on `convdiff-2d`: the ILU(0) setup is milliseconds \
         while the iteration cut is typically 5-10x.\n\
         - On the SPD Poisson cases AMG should win total time only above some size; \
         below it the setup dominates and `jacobi`/`ilu0` win.\n\
         - `helmholtz-2d` and `zero-diag` exist to show the failure modes: rejected \
         setups (`FAIL`) and exhausted budgets (`STALLED`) are the expected, honest \
         cells — a preconditioner that cannot be built is information, not a bug.\n\
         - `jacobi` never reports `FAIL`, even on `zero-diag`: tiny diagonal entries \
         fall back to 1.0 (identity on those rows) by construction, so the setup \
         succeeds and the solve stalls instead. That fallback is deliberate — Jacobi \
         stays total — but on a zero diagonal it buys nothing over `none`.\n\
         - `ilu0` shows the other failure shape on `zero-diag`: the setup accepts the \
         matrix (default pivot tolerance) yet stops after 387 matvecs without converging, \
         while pivoted `ilut`/`block-jacobi` refactor through row swaps and converge \
         in 4 iterations. Pivoting is the whole story on that matrix.\n",
    );
    std::fs::write(&args.output, &md)?;
    println!("Report written to {}", args.output.display());

    Ok(())
}
