//! CLI binary to generate room measurement data for RoomEQ testing
//!
//! Usage:
//!   cargo run --bin generate-roomeq-data --release -- --output-dir data_tests/roomeq/generated

// Rust guideline compliant 2026-02-21

use anyhow::Result;
use clap::Parser;
use num_complex::Complex64;
use rayon::prelude::*;
use serde::Serialize;
#[cfg(feature = "cuda")]
use sonium_bem::room_acoustics::build_room_tbem_system_with_prepared_geometry_cuda;
#[cfg(feature = "metal")]
use sonium_bem::room_acoustics::build_room_tbem_system_with_prepared_geometry_metal;
use sonium_bem::room_acoustics::{
    RoomSourceDirectivityMode, RoomTbemConfig, build_room_tbem_rhs_with_prepared_geometry,
    build_room_tbem_system_with_prepared_geometry,
    calculate_field_pressure_bem_with_field_evaluator, prepare_room_tbem_field_evaluator,
    prepare_room_tbem_geometry_with_config,
};
use sonium_mesher::RoomSurfaceMeshing;
use sonium_sim_room::measured_acceptance::{
    MeasuredAcceptanceBudget, MeasuredAcceptanceReport, MeasuredUncertaintyEnvelope,
    compare_solved_output_to_measured_with_uncertainty,
};
use sonium_sim_room::measured_validation::MeasuredComparisonReport;
use sonium_sim_room::{
    SimulationOutput, csv_export, external_reference, fast_room, fem_runner, gallery, hf_extension,
    measured_io, roomeq_config_gen, scenarios, wav_export,
};
use sonium_solvers::{
    dense::DenseMatrix,
    direct::{BlrConfig, MultifrontalBlrFactorization},
    sparse::CsrMatrix,
};
use sonium_xem::{Point3D, RoomGeometry, RoomSimulation, detect_symmetry, map_to_reduced_domain};
use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use sysinfo::System;

const DEFAULT_DENSE_MAX_FREQ: f64 = 250.0;
const DENSE_BLR_ELEMENTS_PER_WAVELENGTH: f64 = 8.0;
const MEASURED_COMPARISON_SCHEMA_V1: &str = "roomsim.measured-comparison.v1";

#[derive(Parser, Debug)]
#[command(
    name = "generate-roomeq-data",
    version,
    about = "Generate simulated room measurement data for RoomEQ testing.",
    long_about = "\
Generate simulated room measurement data for RoomEQ testing.

Runs a FEM acoustic solver by default over a set of built-in room scenarios
(stereo, 2.1, multi-sub, 5.0/5.1/5.1.4 surround). With `--dense`, low
frequencies are solved with dense BLR TBEM before the FEM sweep takes over.
Outputs are per-(source × listening position) CSV measurement files plus a
`config.json` consumable by the RoomEQ optimizer.

Memory: the FEM mesh size grows roughly with `(--max-freq)^3`, so by
default the binary estimates the peak memory of the worst-case scenario
and refuses to start if it would exceed the budget. CPU runs default to
80% of free system RAM at start; CUDA runs use the smaller of that host
budget and 80% of free VRAM on `--gpu-device`. Use `--memory-limit-gb`
to change the host/user side of the budget and `--auto-reduce-freq` to
silently lower `--max-freq` to fit.

Examples:
  # Run every scenario, default budget (80% of free RAM)
  generate-roomeq-data --output-dir out/

  # One scenario, 16 GiB cap, auto-fit max_freq if needed
  generate-roomeq-data --scenario medium_surround_5_1 \\
      --memory-limit-gb 16 --auto-reduce-freq

  # Coarsen far-field mesh 4x outside a 0.6 m bubble around POIs
  generate-roomeq-data --mesh-grading 4.0 --mesh-fine-radius 0.6

  # Compare parallel preconditioners
  generate-roomeq-data --scenario medium_multi_seat \\
      --preconditioner block-jacobi

  # Try recycled GMRES across adjacent frequencies
  generate-roomeq-data --scenario medium_multi_seat \\
      --recycle-subspace-size 8

  # Use dense BLR TBEM for low frequencies, then FEM above 300 Hz
  generate-roomeq-data --dense --dense-max-freq 300
"
)]
struct Args {
    /// Output directory for generated data.
    #[arg(
        short,
        long,
        default_value = "data_tests/roomeq/generated",
        value_name = "DIR"
    )]
    output_dir: PathBuf,

    /// Write deterministic, website-ready SVG/CSV gallery assets and
    /// provenance manifests. The numerical CSV/WAV exports remain the source
    /// of truth for every plotted value.
    #[arg(long, value_name = "DIR")]
    gallery: Option<PathBuf>,
    /// Side length of the square XY slice sampling grid used by gallery
    /// pressure slices. Sampling cost grows quadratically; 32 keeps smoke
    /// galleries fast while 128+ suits website figures.
    #[arg(long, value_name = "N", default_value_t = 32, requires = "gallery")]
    gallery_slice_grid: usize,
    /// Compare a selected gallery against a versioned solved-reference JSON
    /// envelope, such as an independently exported FreeFEM response.
    #[arg(
        long,
        value_name = "PATH",
        requires_all = ["gallery", "scenario"]
    )]
    gallery_reference: Option<PathBuf>,
    /// Execute an external reference driver and compare its response in the
    /// selected gallery. The driver receives absolute manifest/data/output
    /// paths through the `ROOMSIM_EXTERNAL_REFERENCE_*` environment variables.
    #[arg(
        long,
        value_name = "PROGRAM",
        requires_all = ["gallery", "scenario"],
        conflicts_with = "gallery_reference"
    )]
    gallery_reference_command: Option<OsString>,
    /// Backend label required in the external response envelope. Defaults to
    /// the command program spelling when omitted.
    #[arg(
        long = "gallery-reference-backend",
        value_name = "LABEL",
        requires = "gallery_reference_command"
    )]
    gallery_reference_backend: Option<String>,
    /// Repeat an argument passed verbatim to `--gallery-reference-command`.
    #[arg(
        long = "gallery-reference-arg",
        value_name = "ARG",
        requires = "gallery_reference_command"
    )]
    gallery_reference_args: Vec<OsString>,
    /// Frequency pairing tolerance for the gallery reference comparison in Hz.
    #[arg(
        long,
        default_value_t = 1.0e-6,
        value_name = "HZ",
        requires = "gallery_reference"
    )]
    gallery_reference_frequency_tolerance_hz: f64,
    /// Relative-pressure floor for gallery reference error normalization.
    #[arg(
        long,
        default_value_t = 20.0e-6,
        value_name = "PA",
        requires = "gallery_reference"
    )]
    gallery_reference_relative_floor_pa: f64,
    /// Compare the selected scenario against a versioned measured-response
    /// dataset and emit `measured_comparison.json` beside its exports.
    #[arg(long, value_name = "PATH", requires = "scenario")]
    measured_response: Option<PathBuf>,
    /// Host-pinned measured-data signing keys as KEY_ID=LOWERCASE_PUBLIC_KEY_HEX.
    /// Supplying one or more keys authenticates the selected measured dataset.
    #[arg(
        long = "measured-trusted-key",
        value_name = "KEY_ID=PUBLIC_KEY_HEX",
        requires = "measured_response"
    )]
    measured_trusted_keys: Vec<String>,
    /// Require a detached trusted-key signature for --measured-response.
    #[arg(long, requires = "measured_response")]
    measured_require_signature: bool,
    /// Listener names in model tensor order for measured-response comparison.
    /// Repeat once per listening position when `--measured-response` is used.
    #[arg(
        long = "measured-listener-name",
        value_name = "NAME",
        requires = "measured_response"
    )]
    measured_listener_names: Vec<String>,
    /// Maximum allowed measured/model peak-SPL error in dB.
    #[arg(
        long,
        default_value_t = 3.0,
        value_name = "DB",
        requires = "measured_response"
    )]
    measured_max_magnitude_error_db: f64,
    /// Maximum allowed measured/model wrapped phase error in radians.
    #[arg(
        long,
        default_value_t = std::f64::consts::PI,
        value_name = "RAD",
        requires = "measured_response"
    )]
    measured_max_phase_error_rad: f64,
    /// Frequency pairing tolerance for measured-response comparison in Hz.
    #[arg(
        long,
        default_value_t = 1.0e-6,
        value_name = "HZ",
        requires = "measured_response"
    )]
    measured_frequency_tolerance_hz: f64,
    /// Digest-bound per-sample magnitude/phase uncertainty envelope.
    #[arg(long, value_name = "PATH", requires = "measured_response")]
    measured_uncertainty: Option<PathBuf>,
    /// Maximum magnitude error divided by its supplied uncertainty.
    #[arg(
        long,
        default_value_t = 3.0,
        value_name = "RATIO",
        requires = "measured_uncertainty"
    )]
    measured_max_normalized_magnitude_error: f64,
    /// Maximum phase error divided by its supplied uncertainty.
    #[arg(
        long,
        default_value_t = 3.0,
        value_name = "RATIO",
        requires = "measured_uncertainty"
    )]
    measured_max_normalized_phase_error: f64,
    /// Minimum non-zero phase samples required by uncertainty acceptance.
    #[arg(
        long,
        default_value_t = 1,
        value_name = "COUNT",
        requires = "measured_uncertainty"
    )]
    measured_minimum_phase_samples: usize,

    /// Run only a specific scenario by name (omit to run all built-in scenarios).
    #[arg(long, value_name = "NAME")]
    scenario: Option<String>,

    /// Maximum simulated frequency in Hz. Higher values require a finer mesh
    /// and grow memory roughly as f^3.
    #[arg(long, default_value = "2000", value_name = "HZ")]
    max_freq: f64,

    /// Minimum simulated frequency in Hz.
    #[arg(long, default_value = "20", value_name = "HZ")]
    min_freq: f64,

    /// Number of frequency points, log-spaced from `--min-freq` to `--max-freq`.
    #[arg(long, default_value = "200", value_name = "N")]
    num_points: usize,

    /// Number of rayon worker threads. `0` = use all performance cores.
    /// Scenarios run sequentially so each one gets the entire pool for its
    /// nested FEM assembly and GMRES par_iters.
    #[arg(long, default_value = "0", value_name = "N")]
    max_concurrent: usize,

    /// Print a bounded CPU execution contract as JSON and exit.
    #[arg(long)]
    execution_contract: bool,

    /// Room solver: `fem` (accurate sparse Helmholtz solve) or `fast-hybrid`
    /// (rectangular-room modes plus coherent image sources).
    #[arg(long, default_value = "fem", value_name = "NAME")]
    solver: String,

    /// Far-field mesh coarsening multiplier. Far-field elements are this many
    /// times larger than the wavelength-resolved fine spacing near sources
    /// and microphones. `1.0` = uniform mesh (highest memory).
    #[arg(long, default_value = "1.0", value_name = "RATIO")]
    mesh_grading: f64,

    /// Radius (m) around each source / listening position within which the
    /// FEM mesh keeps the fine wavelength-resolved spacing.
    #[arg(long, default_value = "0.5", value_name = "METERS")]
    mesh_fine_radius: f64,

    /// FEM preconditioner: ilut, block-jacobi, additive-schwarz, oras, habc,
    /// sweeping-schwarz, ilu-coloring, ilu-fixedpoint, or amg.
    #[arg(
        long,
        default_value_t = fem_runner::FemPreconditionerKind::default(),
        value_name = "NAME",
        value_parser = parse_preconditioner,
    )]
    preconditioner: fem_runner::FemPreconditionerKind,

    /// Recycled GMRES subspace dimension across adjacent frequencies.
    /// `0` disables recycling and uses only the previous frequency solution
    /// as the initial guess. Values like 4-20 are reasonable experiments.
    #[arg(
        long,
        default_value_t = fem_runner::DEFAULT_RECYCLE_SUBSPACE_SIZE,
        value_name = "N"
    )]
    recycle_subspace_size: usize,

    /// Krylov-subspace recycler. `gcro-dr` (default) uses Parks-de Sturler
    /// augmented Arnoldi orthogonal to `M⁻¹A · U` (the structural win for
    /// frequency sweeps). `legacy-coarse` keeps the older one-shot
    /// `x += U(U^H A U)⁻¹ U^H r` correction path.
    #[arg(
        long,
        default_value_t = fem_runner::RecyclerKind::default(),
        value_name = "NAME",
        value_parser = parse_recycler,
    )]
    recycler: fem_runner::RecyclerKind,

    /// Rebuild the shifted-Laplacian preconditioner only every N-th frequency
    /// of the sweep. `1` rebuilds on every frequency — exact but slow.
    /// `N>1` reuses the same factorisation for N adjacent frequencies,
    /// amortising the ILUT/AMG setup cost; iteration count grows sub-linearly
    /// with N, and the wall-clock optimum is empirically 5–10. Default `5`.
    #[arg(long, default_value_t = 5, value_name = "N")]
    rebuild_precond_every: usize,

    /// Clear the Krylov recycler every M-th frequency so it rebuilds from a
    /// fresh state. `0` (default) never resets — the solver already clears
    /// the recycler automatically on a first-pass convergence failure, so a
    /// periodic forced reset usually does more harm than good (it can throw
    /// away a useful subspace right when the recycler is helping). Set M>0
    /// only if you observe sustained recycler drift in a dense-modal band.
    #[arg(long, default_value_t = 0, value_name = "M")]
    recycle_reset_every: usize,

    /// GMRES relative-residual convergence tolerance. `1e-6` (default) matches
    /// f64 assembly/quadrature error. Loosen to `1e-5` for ~15–20% fewer
    /// iterations with no audible impact; tightening below `1e-7` is wasted
    /// work because discretisation error dominates.
    #[arg(long, default_value_t = 1e-6, value_name = "TOL")]
    gmres_tolerance: f64,

    /// GMRES restart window `m`. Caps Arnoldi orthogonalisation cost at
    /// O(m·n) per cycle and Hessenberg storage at O(m²). Default `50` suits
    /// block-Jacobi / ILUT. AMG/FGMRES can use `m=30`; weak preconditioners
    /// benefit from `60–80`.
    #[arg(long, default_value_t = 50, value_name = "M")]
    gmres_restart: usize,

    /// Solve all source right-hand sides of a frequency as ONE block system
    /// (block GMRES / block FGMRES) instead of independent per-source solves.
    /// Off by default: with the full rayon pool the parallel per-source path
    /// usually wins on wall-clock; the block path helps when threads are
    /// scarce (e.g. under frequency-lane parallelism). Per-source GCRO-DR
    /// recycling is not used on the block path (warm-start X₀ instead);
    /// sources failing the true-residual gate fall back to the per-source
    /// retry ladder.
    #[arg(long)]
    block_source_solve: bool,
    /// Permit an intentional room-side crossover after a complex directivity
    /// grid. Defaults off because complex grids normally include crossover data.
    #[arg(long)]
    allow_chained_complex_crossover: bool,

    /// Minimum FEM modal damping floor as a fraction of real wavenumber.
    /// Default `0.0005` adds 0.05% volumetric modal damping on top of boundary
    /// loss and atmospheric absorption, retained only as a pole regularizer;
    /// set `0.0` to disable this floor.
    #[arg(
        long,
        default_value_t = fem_runner::DEFAULT_MODAL_DAMPING_RATIO,
        value_name = "RATIO"
    )]
    modal_damping_ratio: f64,

    /// FEM volume-source model. `point` (default) is the mesh-independent
    /// discrete delta: the integrated source strength equals the prescribed
    /// volume velocity on any mesh, so the broadband monopole is not low-pass
    /// filtered by an artificial source width. `gaussian` reproduces the
    /// historical fixed-0.1 m Gaussian for regression cross-checks.
    #[arg(
        long,
        default_value = "point",
        value_name = "NAME",
        value_parser = parse_source_model
    )]
    source_model: fem_runner::FemSourceModel,

    /// Shifted-Laplacian family: `legacy`, `k2`, or experimental `csl`.
    /// `k2` uses the production `(alpha=1.0, beta=0.5)` coercive preset. The
    /// indefinite CSL form is primarily for preconditioner research and can
    /// converge very slowly with ILUT.
    #[arg(long, default_value = "legacy", value_name = "NAME")]
    shifted_laplacian: fem_runner::ShiftedLaplacianShift,

    /// Memory budget in GiB for the estimated FEM peak. If the worst-case
    /// scenario would exceed this, you'll be prompted to lower `--max-freq`.
    /// `0` (default) selects 80% of the free system memory at program start.
    #[arg(long, default_value = "0", value_name = "GIB")]
    memory_limit_gb: f64,

    /// Automatically reduce `--max-freq` to fit `--memory-limit-gb` without
    /// prompting.
    #[arg(long)]
    auto_reduce_freq: bool,

    /// Use dense BLR TBEM for a single-scenario reference check.
    ///
    /// Dense BLR has no cross-frequency factorization reuse and is therefore
    /// limited to at most eight requested samples. Production dataset
    /// generation uses FEM.
    #[arg(long)]
    dense: bool,

    /// Highest frequency solved with dense BLR when `--dense` is enabled.
    /// If omitted, dense BLR covers frequencies up to 250 Hz.
    #[arg(long, value_name = "HZ", requires = "dense")]
    dense_max_freq: Option<f64>,

    /// Width in Hz of an exact-grid dense-BLR/FEM complex crossfade around
    /// `--dense-max-freq`. `0` keeps the strict non-overlapping splice.
    #[arg(long, default_value = "0", value_name = "HZ", requires = "dense")]
    dense_overlap_hz: f64,

    /// Number of dense-BLR frequencies solved concurrently. `1` (default)
    /// runs frequencies sequentially so each one gets the entire TBEM thread
    /// pool. Set to `N` on a large machine to run `N` frequencies in parallel;
    /// total live workers will be roughly `N × --dense-tbem-threads`.
    #[arg(long, default_value = "1", value_name = "N", requires = "dense")]
    dense_freq_concurrency: usize,

    /// Worker threads dedicated to each dense-BLR per-frequency solve (TBEM
    /// assembly, BLR factor, field-pressure evaluation). `0` (default) uses
    /// the global rayon pool — appropriate when `--dense-freq-concurrency=1`.
    /// When running frequencies concurrently, set this so that
    /// `dense_freq_concurrency × dense_tbem_threads` matches your core count.
    #[arg(long, default_value = "0", value_name = "N", requires = "dense")]
    dense_tbem_threads: usize,

    /// Backend used for the dense-BLR TBEM matrix assembly. `cpu` (default)
    /// uses the rayon-parallel stabilised assembler. `metal` and `cuda`
    /// route the far-field regular-pair tail through the GPU and keep self /
    /// near-field pairs on the CPU. Requires building with the matching
    /// feature. The downstream BLR factor and GMRES still run on the CPU.
    #[arg(long, default_value = "cpu", value_name = "NAME", requires = "dense")]
    dense_backend: String,

    /// Compute backend: `cpu` (default) drives the multi-threaded host FEM
    /// solver. `cuda` and `metal` route the per-frequency solve through the
    /// optional GPU paths using the `--gpu-precision` and `--gpu-precond`
    /// settings. Requires building with the matching feature.
    #[arg(long, default_value = "cpu", value_name = "NAME")]
    backend: String,

    /// GPU precision strategy (consulted by GPU FEM backends):
    /// `f32` (default) runs pure Complex32 GMRES with promoted residual
    /// correction on CUDA, which is the fastest convergent path on RTX-class GPUs.
    /// `f64-outer` runs the Carson-Higham IR path (Complex64 outer FGMRES +
    /// Complex32 inner) for a stricter f64 residual target. Metal FEM currently
    /// supports only `f64-outer`.
    ///
    /// On RTX-class GPUs f64 throughput is much lower than f32; use
    /// `f64-outer` explicitly when you need the stricter f64 residual target.
    #[arg(long, default_value = "f32", value_name = "MODE")]
    gpu_precision: String,

    /// CUDA device ordinal (only consulted when `--backend cuda`).
    #[arg(long, default_value = "0", value_name = "N")]
    gpu_device: usize,

    /// GPU preconditioner choice (consulted by GPU FEM backends):
    /// `point-jacobi` is fastest per apply (one diag-CSR SpMV) but stalls
    /// for indefinite Helmholtz around residual ~1e-2. `poly-jacobi-N`
    /// runs N Jacobi-Richardson iterations on
    /// the shifted-Laplacian P, approximating P⁻¹ much better; per apply
    /// it does N SpMVs of P. `chebyshev-N` applies a stronger GPU-resident
    /// Chebyshev smoother to P. `auto` selects CUDA AMG or Metal Chebyshev-8.
    /// Explicit `amg` is CUDA-only in the current prototype.
    #[arg(long, default_value = "auto", value_name = "SPEC")]
    gpu_precond: String,

    /// Enable verbose (debug-level) logging.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolverKind {
    Fem,
    FastHybrid,
}

impl SolverKind {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fem" => Ok(Self::Fem),
            "fast-hybrid" | "fast" | "hybrid" => Ok(Self::FastHybrid),
            other => Err(format!("unknown solver '{other}'. Valid: fem, fast-hybrid")),
        }
    }
}

impl BackendKind {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "metal" | "apple" => Ok(Self::Metal),
            "cuda" | "gpu" | "nvidia" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown backend '{other}'. Valid: cpu, metal, cuda"
            )),
        }
    }
}

fn parse_preconditioner(value: &str) -> Result<fem_runner::FemPreconditionerKind, String> {
    value.parse()
}

/// Resolve the `--max-concurrent` CLI value to a concrete thread count for
/// guard messages. `0` means "use the whole machine", matching the Rayon
/// pool installed later by `install_fem_rayon_pool`.
fn effective_thread_count(max_concurrent: usize) -> usize {
    if max_concurrent > 0 {
        max_concurrent
    } else {
        std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
    }
}

/// Whether a `--gpu-precond` spec selects plain point-Jacobi.
///
/// Mirrors the `point-jacobi | pointjacobi | diagonal | jacobi` aliases
/// accepted by `GpuPrecond::parse` without depending on the cuda/metal
/// feature gates, so the CLI can warn in every build.
fn is_point_jacobi_gpu_precond(spec: &str) -> bool {
    matches!(
        spec.trim().to_ascii_lowercase().replace('_', "-").as_str(),
        "point-jacobi" | "pointjacobi" | "diagonal" | "jacobi"
    )
}

/// Whether a CPU preconditioner applies mostly serial triangular solves.
///
/// The ILU family converges strongly but its apply phase does not scale with
/// cores, unlike `block-jacobi` / `additive-schwarz` / `oras` / `habc` /
/// `amg`, so combining it with a wide thread pool wastes the budget.
fn uses_serial_apply_preconditioner(kind: fem_runner::FemPreconditionerKind) -> bool {
    matches!(
        kind,
        fem_runner::FemPreconditionerKind::Ilut
            | fem_runner::FemPreconditionerKind::IluColoring
            | fem_runner::FemPreconditionerKind::IluFixedPoint
    )
}

fn parse_source_model(value: &str) -> Result<fem_runner::FemSourceModel, String> {
    match value {
        "point" => Ok(fem_runner::FemSourceModel::PointLoad),
        "gaussian" => Ok(fem_runner::FemSourceModel::Gaussian {
            sigma: fem_runner::DEFAULT_SOURCE_WIDTH,
        }),
        other => Err(format!(
            "unknown source model '{other}' (expected 'point' or 'gaussian')"
        )),
    }
}

#[cfg(feature = "cuda")]
fn run_fem_cuda(
    simulation: &RoomSimulation,
    mesh_config: &fem_runner::FemMeshConfig,
    gpu_precision: &str,
    gpu_precond: &str,
    gpu_device: usize,
) -> Result<(
    sonium_sim_room::SimulationOutput,
    fem_runner::FemRunDiagnostics,
)> {
    let precision =
        fem_runner::GpuPrecision::parse(gpu_precision).map_err(|e| anyhow::anyhow!("{e}"))?;
    let precond = fem_runner::GpuPrecond::parse(gpu_precond)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .resolve_for_cuda();
    fem_runner::run_fem_gpu_report(simulation, mesh_config, precision, precond, gpu_device)
}

#[cfg(not(feature = "cuda"))]
fn run_fem_cuda(
    _simulation: &RoomSimulation,
    _mesh_config: &fem_runner::FemMeshConfig,
    _gpu_precision: &str,
    _gpu_precond: &str,
    _gpu_device: usize,
) -> Result<(
    sonium_sim_room::SimulationOutput,
    fem_runner::FemRunDiagnostics,
)> {
    anyhow::bail!(
        "--backend cuda requires building with `--features cuda`. \
         Rebuild: `cargo build --release -p sonium-sim-room --features cuda --bin generate-roomeq-data`"
    )
}

#[cfg(feature = "metal")]
fn run_fem_metal_report(
    simulation: &RoomSimulation,
    mesh_config: &fem_runner::FemMeshConfig,
    gpu_precision: &str,
    gpu_precond: &str,
) -> Result<(
    sonium_sim_room::SimulationOutput,
    fem_runner::FemRunDiagnostics,
)> {
    let precision =
        fem_runner::GpuPrecision::parse(gpu_precision).map_err(|e| anyhow::anyhow!("{e}"))?;
    let precond = fem_runner::GpuPrecond::parse(gpu_precond).map_err(|e| anyhow::anyhow!("{e}"))?;
    fem_runner::run_fem_metal_report(simulation, mesh_config, precision, precond)
}

#[cfg(not(feature = "metal"))]
fn run_fem_metal_report(
    _simulation: &RoomSimulation,
    _mesh_config: &fem_runner::FemMeshConfig,
    _gpu_precision: &str,
    _gpu_precond: &str,
) -> Result<(
    sonium_sim_room::SimulationOutput,
    fem_runner::FemRunDiagnostics,
)> {
    anyhow::bail!(
        "--backend metal requires building with `--features metal`. \
Rebuild: `cargo build --release -p sonium-sim-room --features metal --bin generate-roomeq-data`"
    )
}

fn parse_recycler(value: &str) -> Result<fem_runner::RecyclerKind, String> {
    value.parse()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseBackend {
    Cpu,
    Metal,
    Cuda,
}

impl DenseBackend {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "metal" | "gpu" | "apple" => Ok(Self::Metal),
            "cuda" | "nvidia" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown dense backend '{other}'. Valid: cpu, metal, cuda"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DenseBlrOptions {
    max_freq: f64,
    overlap_hz: f64,
    freq_concurrency: usize,
    tbem_threads: usize,
    #[cfg(feature = "cuda")]
    gpu_device: usize,
    backend: DenseBackend,
}

fn dense_blr_options(args: &Args) -> Result<Option<DenseBlrOptions>> {
    if !args.dense {
        return Ok(None);
    }
    if args.scenario.is_none() {
        anyhow::bail!("--dense is a reference-check path and requires selecting one --scenario");
    }
    if args.num_points > 8 {
        anyhow::bail!(
            "--dense is limited to at most 8 reference samples; use FEM for production sweeps"
        );
    }

    let max_freq = args.dense_max_freq.unwrap_or(DEFAULT_DENSE_MAX_FREQ);
    if !max_freq.is_finite() || max_freq <= 0.0 {
        anyhow::bail!("--dense-max-freq must be a positive finite frequency in Hz");
    }
    if !args.dense_overlap_hz.is_finite() || args.dense_overlap_hz < 0.0 {
        anyhow::bail!("--dense-overlap-hz must be a finite non-negative frequency in Hz");
    }

    let freq_concurrency = args.dense_freq_concurrency.max(1);
    let backend = DenseBackend::parse(&args.dense_backend).map_err(|e| anyhow::anyhow!(e))?;
    if backend == DenseBackend::Metal && cfg!(not(feature = "metal")) {
        anyhow::bail!(
            "--dense-backend metal requires building with `--features metal`. \
             Rebuild: `cargo build --release -p sonium-sim-room --features metal \
             --bin generate-roomeq-data`"
        );
    }
    if backend == DenseBackend::Cuda && cfg!(not(feature = "cuda")) {
        anyhow::bail!(
            "--dense-backend cuda requires building with `--features cuda`. \
             Rebuild: `cargo build --release -p sonium-sim-room --features cuda \
             --bin generate-roomeq-data`"
        );
    }

    Ok(Some(DenseBlrOptions {
        max_freq,
        overlap_hz: args.dense_overlap_hz,
        freq_concurrency,
        tbem_threads: args.dense_tbem_threads,
        #[cfg(feature = "cuda")]
        gpu_device: args.gpu_device,
        backend,
    }))
}

fn dense_frequency_count(frequencies: &[f64], dense_max_freq: f64) -> usize {
    frequencies
        .iter()
        .position(|&freq| freq > dense_max_freq)
        .unwrap_or(frequencies.len())
}

/// Return the dense end and FEM start indices for the hybrid runner.
///
/// With a non-zero overlap, both slices contain the exact global-grid points
/// in `[dense_max_freq - overlap_hz, dense_max_freq + overlap_hz]`. Keeping
/// this operation index-based prevents the two solvers from receiving subtly
/// different frequency values that cannot be blended deterministically.
fn dense_frequency_bounds(
    frequencies: &[f64],
    dense_max_freq: f64,
    overlap_hz: f64,
) -> Result<(usize, usize), String> {
    if !dense_max_freq.is_finite() || dense_max_freq <= 0.0 {
        return Err("dense cutoff must be finite and positive".into());
    }
    if !overlap_hz.is_finite() || overlap_hz < 0.0 {
        return Err("dense overlap must be finite and non-negative".into());
    }
    let dense_end = if overlap_hz == 0.0 {
        dense_frequency_count(frequencies, dense_max_freq)
    } else {
        frequencies
            .iter()
            .position(|&frequency| frequency > dense_max_freq + overlap_hz)
            .unwrap_or(frequencies.len())
    };
    let fem_start = if overlap_hz == 0.0 {
        dense_end
    } else {
        frequencies
            .iter()
            .position(|&frequency| frequency >= dense_max_freq - overlap_hz)
            .unwrap_or(frequencies.len())
    };
    if overlap_hz > 0.0 && fem_start >= dense_end {
        return Err(
            "dense overlap does not contain an exact point on the simulation frequency grid".into(),
        );
    }
    Ok((dense_end, fem_start))
}

fn simulation_with_frequencies(base: &RoomSimulation, frequencies: Vec<f64>) -> RoomSimulation {
    let mut simulation = base.clone();
    simulation.frequencies = frequencies;
    simulation
}

fn dense_to_csr(matrix: &DenseMatrix<Complex64>) -> Result<CsrMatrix<Complex64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        anyhow::bail!("dense BLR room matrix must be square, got {rows}x{cols}");
    }
    Ok(CsrMatrix::from_dense(matrix, 0.0))
}

fn run_dense_blr(
    simulation: &RoomSimulation,
    options: DenseBlrOptions,
) -> Result<SimulationOutput> {
    let dense_max_freq = options.max_freq;
    if simulation.frequencies.is_empty() {
        return Ok(SimulationOutput::solved(
            Vec::new(),
            Vec::new(),
            simulation.sources.iter().map(|s| s.name.clone()).collect(),
        ));
    }

    let mesh_freq = simulation
        .frequencies
        .last()
        .copied()
        .unwrap_or(dense_max_freq)
        .min(dense_max_freq);
    let elements_per_meter = (DENSE_BLR_ELEMENTS_PER_WAVELENGTH * mesh_freq
        / simulation.speed_of_sound)
        .ceil()
        .max(1.0) as usize;
    let mesh = simulation.room.generate_mesh(elements_per_meter);
    let field_points = simulation.listening_positions.clone();

    let backend_label = match options.backend {
        DenseBackend::Cpu => "cpu",
        DenseBackend::Metal => "metal",
        DenseBackend::Cuda => "cuda",
    };
    log::info!(
        "[dense-blr] mesh: {} nodes, {} surface elements, {} elements/m for <= {:.0} Hz \
         (backend={}, freq_concurrency={}, tbem_threads={})",
        mesh.num_nodes(),
        mesh.num_elements(),
        elements_per_meter,
        dense_max_freq,
        backend_label,
        options.freq_concurrency,
        if options.tbem_threads == 0 {
            "ambient".to_string()
        } else {
            options.tbem_threads.to_string()
        },
    );

    // GPU backends initialise once and share by Arc across per-frequency
    // tasks. CPU backend leaves these `None`.
    #[cfg(feature = "metal")]
    let metal_backend: Option<Arc<sonium_compute::MetalBackend>> = match options.backend {
        DenseBackend::Cpu => None,
        DenseBackend::Metal => Some(Arc::new(
            sonium_compute::MetalBackend::new()
                .map_err(|e| anyhow::anyhow!("failed to initialise Metal backend: {e:?}"))?,
        )),
        DenseBackend::Cuda => None,
    };
    #[cfg(feature = "cuda")]
    let cuda_backend: Option<Arc<sonium_compute::CudaBackend>> = match options.backend {
        DenseBackend::Cpu | DenseBackend::Metal => None,
        DenseBackend::Cuda => {
            let backend = sonium_compute::CudaBackend::new(options.gpu_device)
                .map_err(|e| anyhow::anyhow!("failed to initialise CUDA backend: {e}"))?;
            if !backend.has_bem_regular_pair_c32() {
                anyhow::bail!(
                    "--dense-backend cuda requested, but the CUDA BEM regular-pair kernel \
                     is unavailable"
                );
            }
            Some(Arc::new(backend))
        }
    };

    let source_count = simulation.sources.len();
    let lp_count = simulation.listening_positions.len();
    let freq_count = simulation.frequencies.len();
    let blr_config = BlrConfig::default();

    // Two nested rayon pools. The outer pool gives `freq_concurrency` workers,
    // each running one frequency at a time. The inner pool is shared per
    // outer worker (via thread-local OnceCell) and sized to `tbem_threads`;
    // a `tbem_threads = 0` config falls back to the global pool. Peak
    // concurrency is `freq_concurrency × max(tbem_threads, ambient pool)`.
    let outer_pool: Option<Arc<rayon::ThreadPool>> = if options.freq_concurrency > 1 {
        Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(options.freq_concurrency)
                .stack_size(32 * 1024 * 1024)
                .thread_name(|i| format!("dense-blr-freq-{i}"))
                .build()
                .map_err(|e| anyhow::anyhow!("failed to build dense-BLR freq pool: {e}"))?,
        ))
    } else {
        None
    };

    type FreqOutput = (Vec<Vec<Complex64>>, f64, usize);

    let tbem_threads = options.tbem_threads;
    let assemble_one = |prepared: &sonium_bem::room_acoustics::RoomTbemPreparedGeometry,
                        sources: &[sonium_xem::Source]| match options.backend
    {
        DenseBackend::Cpu => build_room_tbem_system_with_prepared_geometry(prepared, sources),
        #[cfg(feature = "metal")]
        DenseBackend::Metal => build_room_tbem_system_with_prepared_geometry_metal(
            prepared,
            sources,
            metal_backend
                .as_ref()
                .expect("metal backend missing despite DenseBackend::Metal")
                .as_ref(),
        ),
        #[cfg(not(feature = "metal"))]
        DenseBackend::Metal => unreachable!(
            "dense_blr_options should have rejected --dense-backend metal without --features metal"
        ),
        #[cfg(feature = "cuda")]
        DenseBackend::Cuda => build_room_tbem_system_with_prepared_geometry_cuda(
            prepared,
            sources,
            cuda_backend
                .as_ref()
                .expect("cuda backend missing despite DenseBackend::Cuda")
                .as_ref(),
        ),
        #[cfg(not(feature = "cuda"))]
        DenseBackend::Cuda => unreachable!(
            "dense_blr_options should have rejected --dense-backend cuda without --features cuda"
        ),
    };
    let solve_one_frequency = |freq_idx: usize, &frequency: &f64| -> Result<FreqOutput> {
        let body = || -> Result<FreqOutput> {
            let freq_start = Instant::now();
            let k = simulation.wavenumber(frequency);
            let first_source = simulation
                .sources
                .first()
                .ok_or_else(|| anyhow::anyhow!("dense BLR requires at least one source"))?;

            log::debug!(
                "[dense-blr] freq {:.1} Hz ({}/{}) started",
                frequency,
                freq_idx + 1,
                freq_count,
            );

            let tbem_config = RoomTbemConfig {
                element_admittance: Some(fem_runner::try_tbem_element_admittance(
                    simulation, &mesh, frequency,
                )?),
                source_directivity: RoomSourceDirectivityMode::ReceiverOnly,
                ..RoomTbemConfig::default()
            };
            let prepared =
                prepare_room_tbem_geometry_with_config(&mesh, k, frequency, &tbem_config);
            let field_evaluator = prepare_room_tbem_field_evaluator(&prepared, &field_points);

            let assemble_start = Instant::now();
            let first_system = assemble_one(&prepared, std::slice::from_ref(first_source));
            let mut assemble_ms = assemble_start.elapsed().as_secs_f64() * 1e3;

            let factor_start = Instant::now();
            let csr = dense_to_csr(&first_system.matrix)?;
            let factor = MultifrontalBlrFactorization::factorize_with_config(&csr, &blr_config)
                .map_err(|e| {
                    anyhow::anyhow!("dense BLR factorization failed at {frequency:.1} Hz: {e}")
                })?;
            let factor_ms = factor_start.elapsed().as_secs_f64() * 1e3;

            let source_outputs: Vec<_> = simulation
                .sources
                .par_iter()
                .enumerate()
                .map(|(source_idx, source)| {
                    let rhs_storage;
                    let (rhs, rhs_ms) = if source_idx == 0 {
                        (first_system.rhs.as_slice(), 0.0)
                    } else {
                        let t = Instant::now();
                        rhs_storage = build_room_tbem_rhs_with_prepared_geometry(
                            &prepared,
                            std::slice::from_ref(source),
                        );
                        (rhs_storage.as_slice(), t.elapsed().as_secs_f64() * 1e3)
                    };

                    let t = Instant::now();
                    let surface_pressure = factor.solve(rhs);
                    let solve_ms = t.elapsed().as_secs_f64() * 1e3;

                    let t = Instant::now();
                    let field_pressure = calculate_field_pressure_bem_with_field_evaluator(
                        &field_evaluator,
                        &surface_pressure,
                        std::slice::from_ref(source),
                    );
                    let field_ms = t.elapsed().as_secs_f64() * 1e3;

                    (field_pressure, rhs_ms, solve_ms, field_ms)
                })
                .collect();
            let mut solve_ms = 0.0;
            let mut field_ms = 0.0;
            let mut per_source = vec![vec![Complex64::new(0.0, 0.0); lp_count]; source_count];
            for (source_idx, (field_pressure, rhs_ms, src_solve_ms, src_field_ms)) in
                source_outputs.into_iter().enumerate()
            {
                assemble_ms += rhs_ms;
                solve_ms += src_solve_ms;
                field_ms += src_field_ms;
                per_source[source_idx][..lp_count].copy_from_slice(&field_pressure[..lp_count]);
            }
            let total_ms = freq_start.elapsed().as_secs_f64() * 1e3;

            log::info!(
                "[dense-blr] freq {:.1} Hz ({}/{}) done in {} \
                     (ndof={}, assemble={:.0}ms, factor={:.0}ms, solve={:.0}ms, field={:.0}ms, \
                      {}src, avg_rank={:.1}, factor_mem={})",
                frequency,
                freq_idx + 1,
                freq_count,
                fmt_ms(total_ms),
                first_system.matrix.nrows(),
                assemble_ms,
                factor_ms,
                solve_ms,
                field_ms,
                source_count,
                factor.average_rank(),
                fmt_gib(factor.factor_memory_bytes() as u64),
            );

            Ok((per_source, total_ms, first_system.matrix.nrows()))
        };

        if tbem_threads == 0 {
            body()
        } else {
            with_tbem_pool(tbem_threads, body)
        }
    };

    let per_freq: Vec<Result<FreqOutput>> = match outer_pool.as_ref() {
        Some(pool) => pool.install(|| {
            simulation
                .frequencies
                .par_iter()
                .enumerate()
                .map(|(idx, f)| solve_one_frequency(idx, f))
                .collect()
        }),
        None => simulation
            .frequencies
            .iter()
            .enumerate()
            .map(|(idx, f)| solve_one_frequency(idx, f))
            .collect(),
    };

    let mut pressures =
        vec![vec![vec![Complex64::new(0.0, 0.0); freq_count]; lp_count]; source_count];
    let mut freq_times_ms: Vec<f64> = Vec::with_capacity(freq_count);
    let mut ndof_seen: usize = 0;
    for (freq_idx, freq_res) in per_freq.into_iter().enumerate() {
        let (freq_data, total_ms, ndof) = freq_res?;
        freq_times_ms.push(total_ms);
        ndof_seen = ndof_seen.max(ndof);
        for (src_idx, lp_vals) in freq_data.into_iter().enumerate() {
            for (lp_idx, val) in lp_vals.into_iter().enumerate() {
                pressures[src_idx][lp_idx][freq_idx] = val;
            }
        }
    }

    log_dense_blr_summary(&freq_times_ms, ndof_seen, source_count);

    Ok(SimulationOutput::solved(
        simulation.frequencies.clone(),
        pressures,
        simulation.sources.iter().map(|s| s.name.clone()).collect(),
    ))
}

fn log_dense_blr_summary(freq_times_ms: &[f64], ndof: usize, source_count: usize) {
    if freq_times_ms.is_empty() {
        return;
    }
    let mut sorted = freq_times_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let mean = sum / n as f64;
    let median = sorted[n / 2];
    let p95 = sorted[((n as f64 * 0.95) as usize).min(n - 1)];
    let min = sorted[0];
    let max = sorted[n - 1];
    log::info!(
        "[dense-blr] summary: {} freqs × {} sources, ndof={}, total={}, \
         per-freq min={} median={} mean={} p95={} max={}",
        n,
        source_count,
        ndof,
        fmt_ms(sum),
        fmt_ms(min),
        fmt_ms(median),
        fmt_ms(mean),
        fmt_ms(p95),
        fmt_ms(max),
    );
}

/// Run `body` inside a thread-local TBEM rayon pool sized to `tbem_threads`.
/// The pool is created lazily per outer-pool worker and reused for every
/// frequency that worker handles, avoiding per-frequency thread spawn cost.
fn with_tbem_pool<R: Send>(tbem_threads: usize, body: impl FnOnce() -> R + Send) -> R {
    use std::cell::OnceCell;
    thread_local! {
        static TBEM_POOL: OnceCell<(usize, Arc<rayon::ThreadPool>)> = const { OnceCell::new() };
    }

    TBEM_POOL.with(|cell| {
        let entry = cell.get_or_init(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(tbem_threads)
                .stack_size(32 * 1024 * 1024)
                .thread_name(|i| format!("dense-blr-tbem-{i}"))
                .build()
                .expect("failed to build dense-BLR TBEM pool");
            (tbem_threads, Arc::new(pool))
        });
        if entry.0 != tbem_threads {
            log::warn!(
                "dense-BLR TBEM pool already initialized with {} threads, ignoring request for {}",
                entry.0,
                tbem_threads
            );
        }
        entry.1.install(body)
    })
}

fn run_fem_solver(
    simulation: &RoomSimulation,
    fem_mesh: &fem_runner::FemMeshConfig,
    backend: BackendKind,
    gpu_precision: &str,
    gpu_precond: &str,
    gpu_device: usize,
    capture_field: bool,
) -> Result<FemSolverOutput> {
    if capture_field && backend == BackendKind::Cpu {
        let (output, field, diagnostics) =
            fem_runner::run_fem_with_mesh_field_report(simulation, fem_mesh)?;
        return Ok(FemSolverOutput {
            output,
            fem_field: Some(field),
            diagnostics: Some(diagnostics),
        });
    }

    let (output, diagnostics) = match backend {
        BackendKind::Cpu => {
            let (output, diagnostics) = fem_runner::run_fem_with_mesh_report(simulation, fem_mesh)?;
            (output, Some(diagnostics))
        }
        BackendKind::Metal => {
            let (output, diagnostics) =
                run_fem_metal_report(simulation, fem_mesh, gpu_precision, gpu_precond)?;
            (output, Some(diagnostics))
        }
        BackendKind::Cuda => {
            let (output, diagnostics) =
                run_fem_cuda(simulation, fem_mesh, gpu_precision, gpu_precond, gpu_device)?;
            (output, Some(diagnostics))
        }
    };
    Ok(FemSolverOutput {
        output,
        fem_field: None,
        diagnostics,
    })
}

struct FemSolverOutput {
    output: SimulationOutput,
    fem_field: Option<fem_runner::FemNodalField>,
    diagnostics: Option<fem_runner::FemRunDiagnostics>,
}

struct SolverRunConfig<'a> {
    solver: SolverKind,
    fem_mesh: &'a fem_runner::FemMeshConfig,
    backend: BackendKind,
    gpu_precision: &'a str,
    gpu_precond: &'a str,
    gpu_device: usize,
    dense_blr: Option<DenseBlrOptions>,
    measured_dataset: Option<&'a measured_io::MeasuredResponseDataset>,
    measured_signature_key_id: Option<&'a str>,
    measured_listener_names: &'a [String],
    measured_max_magnitude_error_db: f64,
    measured_max_phase_error_rad: f64,
    measured_frequency_tolerance_hz: f64,
    measured_uncertainty: Option<&'a MeasuredUncertaintyEnvelope>,
    measured_max_normalized_magnitude_error: f64,
    measured_max_normalized_phase_error: f64,
    measured_minimum_phase_samples: usize,
    gallery_reference: Option<&'a gallery::GalleryReference>,
    gallery_reference_frequency_tolerance_hz: f64,
    gallery_reference_relative_floor_pa: f64,
    gallery_slice_grid: usize,
}

#[derive(Debug, Serialize)]
struct MeasuredComparisonArtifact<'a> {
    schema_version: &'static str,
    scenario: &'a str,
    solver: &'a str,
    dataset: &'a measured_io::MeasuredDatasetMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    measured_signature_key_id: Option<&'a str>,
    measured_provenance_label: &'a str,
    source_names: &'a [String],
    listener_names: &'a [String],
    frequency_tolerance_hz: f64,
    max_magnitude_error_db: f64,
    max_phase_error_rad: f64,
    within_budget: bool,
    budget_error: Option<&'a str>,
    report: MeasuredComparisonReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    uncertainty: Option<MeasuredUncertaintyArtifact>,
}

#[derive(Debug, Serialize)]
struct MeasuredUncertaintyArtifact {
    max_normalized_magnitude_error: f64,
    max_normalized_phase_error: f64,
    minimum_phase_samples: usize,
    within_budget: bool,
    budget_error: Option<String>,
    report: MeasuredAcceptanceReport,
}

#[expect(
    clippy::too_many_arguments,
    reason = "datagen writer threading paths, scenario, datasets and comparison options; grouping would churn the generation call sites"
)]
fn write_measured_comparison(
    path: &std::path::Path,
    scenario_name: &str,
    solver_dir: &str,
    output: &SimulationOutput,
    dataset: &measured_io::MeasuredResponseDataset,
    measured_signature_key_id: Option<&str>,
    listener_names: &[String],
    frequency_tolerance_hz: f64,
    max_magnitude_error_db: f64,
    max_phase_error_rad: f64,
    report: MeasuredComparisonReport,
    budget_error: Option<&str>,
    uncertainty: Option<MeasuredUncertaintyArtifact>,
) -> Result<()> {
    let artifact = MeasuredComparisonArtifact {
        schema_version: MEASURED_COMPARISON_SCHEMA_V1,
        scenario: scenario_name,
        solver: solver_dir,
        dataset: &dataset.metadata,
        measured_signature_key_id,
        measured_provenance_label: &dataset.response.provenance_label,
        source_names: &output.source_names,
        listener_names,
        frequency_tolerance_hz,
        max_magnitude_error_db,
        max_phase_error_rad,
        within_budget: budget_error.is_none(),
        budget_error,
        report,
        uncertainty,
    };
    let document = serde_json::to_vec_pretty(&artifact)?;
    std::fs::write(path, document)?;
    Ok(())
}

fn load_or_execute_gallery_reference(
    simulation: &RoomSimulation,
    scenario_name: &str,
    output_dir: &Path,
    reference_path: Option<&Path>,
    command_program: Option<&OsStr>,
    backend_label: Option<&str>,
    command_args: &[OsString],
) -> Result<Option<gallery::GalleryReference>> {
    if reference_path.is_some() && command_program.is_some() {
        anyhow::bail!("gallery reference path and command are mutually exclusive");
    }

    if let Some(path) = reference_path {
        return gallery::load_reference_response_json(path)
            .map(Some)
            .map_err(anyhow::Error::msg);
    }

    let Some(program) = command_program else {
        return Ok(None);
    };

    std::fs::create_dir_all(output_dir)?;
    let backend_label = backend_label
        .map(str::to_owned)
        .unwrap_or_else(|| program.to_string_lossy().into_owned());
    let manifest = external_reference::ExternalReferenceManifest::from_simulation(
        scenario_name,
        backend_label,
        simulation,
    )
    .map_err(anyhow::Error::msg)?;
    let command = external_reference::ExternalReferenceCommand::new(program.to_os_string())
        .with_args(command_args.iter().cloned());
    let execution = external_reference::execute_external_reference(&manifest, output_dir, &command)
        .map_err(anyhow::Error::msg)?;
    gallery::load_reference_response_json(&execution.response_path)
        .map(Some)
        .map_err(anyhow::Error::msg)
}

fn run_solver(
    scenario: &scenarios::Scenario,
    output_base: &std::path::Path,
    config: &SolverRunConfig<'_>,
    gallery_base: Option<&std::path::Path>,
) -> Result<()> {
    let solver_dir = match config.solver {
        SolverKind::Fem => "fem",
        SolverKind::FastHybrid => "fast-hybrid",
    };
    let scenario_dir = output_base.join(solver_dir).join(&scenario.name);
    std::fs::create_dir_all(&scenario_dir)?;

    log::info!(
        "[{solver_dir}] Running scenario: {} ({}, backend={:?})",
        scenario.name,
        scenario.description,
        config.backend,
    );

    let capture_fem_field = config.solver == SolverKind::Fem
        && config.backend == BackendKind::Cpu
        && gallery_base.is_some();
    let (sim_output, fem_field, fem_diagnostics) = if config.solver == SolverKind::FastHybrid {
        (
            fast_room::run_fast_hybrid(&scenario.simulation)?,
            None,
            None,
        )
    } else if let Some(dense_blr) = config.dense_blr {
        let (dense_end, fem_start) = dense_frequency_bounds(
            &scenario.simulation.frequencies,
            dense_blr.max_freq,
            dense_blr.overlap_hz,
        )
        .map_err(anyhow::Error::msg)?;
        let dense_freqs = scenario.simulation.frequencies[..dense_end].to_vec();
        let fem_freqs = scenario.simulation.frequencies[fem_start..].to_vec();

        log::info!(
            "[mixed] dense BLR frequencies={}, FEM frequencies={}, dense_max_freq={:.0} Hz, overlap={:.0} Hz",
            dense_freqs.len(),
            fem_freqs.len(),
            dense_blr.max_freq,
            dense_blr.overlap_hz,
        );

        let dense_output = if dense_freqs.is_empty() {
            None
        } else {
            let dense_sim = simulation_with_frequencies(&scenario.simulation, dense_freqs);
            Some(run_dense_blr(&dense_sim, dense_blr)?)
        };

        let fem_output = if fem_freqs.is_empty() {
            None
        } else {
            let fem_sim = simulation_with_frequencies(&scenario.simulation, fem_freqs);
            Some(run_fem_solver(
                &fem_sim,
                config.fem_mesh,
                config.backend,
                config.gpu_precision,
                config.gpu_precond,
                config.gpu_device,
                capture_fem_field,
            )?)
        };

        match (dense_output, fem_output) {
            (Some(dense), Some(fem)) => (
                if dense_blr.overlap_hz > 0.0 {
                    SimulationOutput::merge_frequency_segments_with_overlap(dense, fem.output)
                } else {
                    SimulationOutput::merge_frequency_segments(dense, fem.output)
                }
                .map_err(anyhow::Error::msg)?,
                fem.fem_field,
                fem.diagnostics,
            ),
            (Some(dense), None) => (dense, None, None),
            (None, Some(fem)) => (fem.output, fem.fem_field, fem.diagnostics),
            (None, None) => (
                SimulationOutput::solved(
                    Vec::new(),
                    Vec::new(),
                    scenario
                        .simulation
                        .sources
                        .iter()
                        .map(|s| s.name.clone())
                        .collect(),
                ),
                None,
                None,
            ),
        }
    } else {
        let fem = run_fem_solver(
            &scenario.simulation,
            config.fem_mesh,
            config.backend,
            config.gpu_precision,
            config.gpu_precond,
            config.gpu_device,
            capture_fem_field,
        )?;
        (fem.output, fem.fem_field, fem.diagnostics)
    };

    if let Some(dataset) = config.measured_dataset {
        let report = measured_io::compare_solved_output_to_measured_dataset(
            &sim_output,
            dataset,
            config.measured_listener_names,
            config.measured_frequency_tolerance_hz,
        )
        .map_err(anyhow::Error::msg)?;
        let budget_error = report
            .require_within(
                config.measured_max_magnitude_error_db,
                config.measured_max_phase_error_rad,
            )
            .err();
        let uncertainty_artifact = if let Some(uncertainty) = config.measured_uncertainty {
            let uncertainty_report = compare_solved_output_to_measured_with_uncertainty(
                &sim_output,
                dataset,
                uncertainty,
                config.measured_listener_names,
                config.measured_frequency_tolerance_hz,
            )
            .map_err(anyhow::Error::msg)?;
            let uncertainty_budget = MeasuredAcceptanceBudget {
                max_abs_magnitude_error_db: config.measured_max_magnitude_error_db,
                max_abs_phase_error_rad: config.measured_max_phase_error_rad,
                max_normalized_magnitude_error: config.measured_max_normalized_magnitude_error,
                max_normalized_phase_error: config.measured_max_normalized_phase_error,
                minimum_phase_samples: config.measured_minimum_phase_samples,
            };
            let uncertainty_budget_error =
                uncertainty_report.require_within(uncertainty_budget).err();
            Some(MeasuredUncertaintyArtifact {
                max_normalized_magnitude_error: uncertainty_budget.max_normalized_magnitude_error,
                max_normalized_phase_error: uncertainty_budget.max_normalized_phase_error,
                minimum_phase_samples: uncertainty_budget.minimum_phase_samples,
                within_budget: uncertainty_budget_error.is_none(),
                budget_error: uncertainty_budget_error,
                report: uncertainty_report,
            })
        } else {
            None
        };
        let uncertainty_budget_error = uncertainty_artifact
            .as_ref()
            .and_then(|artifact| artifact.budget_error.clone());
        let comparison_path = scenario_dir.join("measured_comparison.json");
        write_measured_comparison(
            &comparison_path,
            &scenario.name,
            solver_dir,
            &sim_output,
            dataset,
            config.measured_signature_key_id,
            config.measured_listener_names,
            config.measured_frequency_tolerance_hz,
            config.measured_max_magnitude_error_db,
            config.measured_max_phase_error_rad,
            report,
            budget_error.as_deref(),
            uncertainty_artifact,
        )?;
        log::info!(
            "[{solver_dir}] Wrote measured comparison to {}",
            comparison_path.display()
        );
        if let Some(error) = budget_error {
            anyhow::bail!("measured comparison exceeded configured budget: {error}");
        }
        if let Some(error) = uncertainty_budget_error {
            anyhow::bail!(
                "uncertainty-bound measured comparison exceeded configured budget: {error}"
            );
        }
    }

    let mut output =
        hf_extension::extend_to_full_range_with_room(&sim_output, &scenario.simulation);
    apply_roomeq_qa_features(scenario.name.as_str(), &mut output);

    let csv_files = csv_export::export_csvs(&output, &scenario_dir)?;
    log::info!(
        "[{solver_dir}] Exported {} CSV files to {}",
        csv_files.len(),
        scenario_dir.display()
    );

    let wav_files = wav_export::export_wavs(&output, &scenario_dir)?;
    log::info!(
        "[{solver_dir}] Exported {} WAV files to {}",
        wav_files.len(),
        scenario_dir.display()
    );

    let room_config = roomeq_config_gen::generate_config(scenario, &scenario_dir)?;
    let config_path = scenario_dir.join("config.json");
    roomeq_config_gen::write_config(&room_config, &config_path)?;
    log::info!("[{solver_dir}] Wrote config to {}", config_path.display());

    let mut metadata = roomeq_config_gen::generate_simulation_metadata_for_solver_with_overlap(
        scenario,
        config.fem_mesh.modal_damping_ratio,
        config.dense_blr.map(|opts| opts.max_freq),
        config.dense_blr.map(|opts| opts.overlap_hz),
        solver_dir,
    );
    if let Some(diagnostics) = fem_diagnostics.as_ref() {
        roomeq_config_gen::attach_fem_diagnostics(&mut metadata, diagnostics)?;
    }
    let metadata_path = scenario_dir.join("simulation_metadata.json");
    roomeq_config_gen::write_simulation_metadata(&metadata, &metadata_path)?;
    log::info!(
        "[{solver_dir}] Wrote metadata to {}",
        metadata_path.display()
    );

    if let Some(gallery_base) = gallery_base {
        let gallery_dir = gallery_base.join(solver_dir).join(&scenario.name);
        let gallery_options = gallery::GalleryOptions {
            slice_grid: config.gallery_slice_grid,
        };
        if let Some(reference) = config.gallery_reference {
            if let Some(fem_field) = fem_field.as_ref() {
                gallery::write_gallery_with_fem_field_and_loaded_reference(
                    &output,
                    &gallery_dir,
                    &scenario.name,
                    solver_dir,
                    &scenario.simulation,
                    fem_field,
                    reference,
                    config.gallery_reference_frequency_tolerance_hz,
                    config.gallery_reference_relative_floor_pa,
                    &gallery_options,
                )
                .map_err(anyhow::Error::msg)?;
            } else {
                if reference.spatial_field.is_some() {
                    anyhow::bail!(
                        "gallery reference contains a spatial field but no retained CPU FEM field is available"
                    );
                }
                gallery::write_gallery_with_loaded_reference(
                    &output,
                    &gallery_dir,
                    &scenario.name,
                    solver_dir,
                    &scenario.simulation,
                    reference,
                    config.gallery_reference_frequency_tolerance_hz,
                    config.gallery_reference_relative_floor_pa,
                    &gallery_options,
                )
                .map_err(anyhow::Error::msg)?;
            }
        } else if let Some(fem_field) = fem_field.as_ref() {
            gallery::write_gallery_with_fem_field(
                &output,
                &gallery_dir,
                &scenario.name,
                solver_dir,
                &scenario.simulation,
                fem_field,
                &gallery_options,
            )
            .map_err(anyhow::Error::msg)?;
        } else {
            gallery::write_gallery(
                &output,
                &gallery_dir,
                &scenario.name,
                solver_dir,
                &scenario.simulation,
                &gallery_options,
            )
            .map_err(anyhow::Error::msg)?;
        }
        log::info!("[{solver_dir}] Wrote gallery to {}", gallery_dir.display());
    }

    Ok(())
}

/// Add deterministic excess-phase content required by the adaptive all-pass
/// RoomEQ QA case. A noiseless simulation otherwise has valid phase and unity
/// coherence, but it does not guarantee a repeatable all-pass signature that
/// is statistically significant at every generated listening position.
fn apply_roomeq_qa_features(scenario_name: &str, output: &mut SimulationOutput) {
    if scenario_name != "medium_surround_5_2_4_multi_seat" {
        return;
    }

    // Keep the five-seat fixture physically varied while removing extremely
    // narrow FEM/image-source nulls that no bounded room-EQ filter should try
    // to fill. The retained +/-3 dB residual around a short log-frequency
    // moving average still exercises spatial robustness and seat weighting.
    for source in &mut output.pressures {
        for listener in source {
            let level_db: Vec<f64> = listener
                .iter()
                .map(|pressure| 20.0 * pressure.norm().max(1.0e-12).log10())
                .collect();
            let smoothed_db: Vec<f64> = (0..level_db.len())
                .map(|index| {
                    let start = index.saturating_sub(4);
                    let end = (index + 5).min(level_db.len());
                    level_db[start..end].iter().sum::<f64>() / (end - start) as f64
                })
                .collect();
            for ((pressure, level), smoothed) in listener.iter_mut().zip(level_db).zip(smoothed_db)
            {
                let conditioned = smoothed + (level - smoothed).clamp(-3.0, 3.0);
                *pressure *= 10.0_f64.powf((conditioned - level) / 20.0);
            }
        }
    }

    let sample_rate = 48_000.0;
    let reference_phase: Vec<f64> = output
        .source_names
        .iter()
        .position(|name| !name.starts_with("sub"))
        .map(|source_index| {
            output.pressures[source_index][0]
                .iter()
                .map(|pressure| pressure.arg())
                .collect()
        })
        .unwrap_or_default();
    for source_index in 0..output.source_names.len() {
        if output.source_names[source_index].starts_with("sub") {
            continue;
        }
        for listener in &mut output.pressures[source_index] {
            for (pressure, phase) in listener.iter_mut().zip(&reference_phase) {
                *pressure = Complex64::from_polar(pressure.norm(), *phase);
            }
        }
        let add_reference_delay = matches!(
            output.source_names[source_index].as_str(),
            "left" | "right" | "center" | "surround_left" | "surround_right"
        );
        if !add_reference_delay {
            continue;
        }
        for (frequency_index, &frequency) in output.frequencies.iter().enumerate() {
            let response = qa_allpass_response(frequency, sample_rate, 120.0, 5.0);
            for listener in &mut output.pressures[source_index] {
                listener[frequency_index] *= response;
            }
        }
    }
}

fn qa_allpass_response(frequency: f64, sample_rate: f64, center_hz: f64, q: f64) -> Complex64 {
    let omega_0 = 2.0 * std::f64::consts::PI * center_hz / sample_rate;
    let alpha = omega_0.sin() / (2.0 * q);
    let a0 = 1.0 + alpha;
    let b0 = (1.0 - alpha) / a0;
    let b1 = -2.0 * omega_0.cos() / a0;
    let b2 = 1.0;
    let a1 = b1;
    let a2 = b0;
    let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;
    let z1 = Complex64::from_polar(1.0, -omega);
    let z2 = z1 * z1;
    (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Bytes per FEM node at peak. Accounts for stiffness/mass (f64), the
/// frequency-dependent A = K - k^2 M and shifted-Laplacian P (Complex64,
/// same sparsity), and the ILUT factors at fill_factor = 20 (Complex64).
/// Element-stencil for P1 tetrahedra in the structured 6-tet decomposition
/// has ~15 non-zeros per row.
///
///   K, M (f64):        2 * 15 *  8 =   240
///   A, P (Complex64):  2 * 15 * 16 =   480
///   ILUT L+U fill 20:  20 * 15 *16 =  4800
///   Krylov subspace + RHS + soln:    ~600
///   ----------------------------------------
///   Total ~6 KB/node. Round up for safety.
const FEM_BYTES_PER_NODE: u64 = 7000;

fn elements_per_meter(max_freq: f64, c: f64) -> usize {
    let min_wavelength = c / max_freq;
    (fem_runner::ELEMENTS_PER_WAVELENGTH / min_wavelength)
        .ceil()
        .max(1.0) as usize
}

/// Approximate the count of nodes along one axis produced by the graded
/// `build_axis_coords` in `fem_runner`. POIs claim a fine-spaced interval
/// of width `2 * radius`; the remainder uses coarse spacing `h_fine *
/// grading`. Intervals are merged when they overlap.
fn axis_node_count(length: f64, eppm: usize, pois: &[f64], radius: f64, grading: f64) -> usize {
    let h_fine = 1.0 / eppm as f64;
    if grading <= 1.05 || pois.is_empty() {
        return (length / h_fine).ceil() as usize + 1;
    }
    let h_coarse = h_fine * grading;

    let mut intervals: Vec<(f64, f64)> = pois
        .iter()
        .map(|&p| ((p - radius).max(0.0), (p + radius).min(length)))
        .filter(|(a, b)| b > a)
        .collect();
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut merged: Vec<(f64, f64)> = Vec::new();
    for iv in intervals {
        if let Some(last) = merged.last_mut()
            && iv.0 <= last.1
        {
            last.1 = last.1.max(iv.1);
            continue;
        }
        merged.push(iv);
    }
    let l_fine: f64 = merged.iter().map(|(a, b)| b - a).sum();
    let l_coarse = (length - l_fine).max(0.0);
    (l_fine / h_fine + l_coarse / h_coarse).ceil() as usize + 1
}

/// Estimate peak FEM memory (bytes) for one scenario at a given max_freq.
fn estimate_fem_memory(sim: &RoomSimulation, grading: f64, fine_radius: f64, max_freq: f64) -> u64 {
    let (width, depth, height) = sim.room.dimensions();
    let symmetry = match sim.room {
        RoomGeometry::Rectangular(_) => detect_symmetry(sim),
        _ => sonium_xem::SymmetryConfig::none(),
    };
    let eff_w = if symmetry.has_x() { width / 2.0 } else { width };
    let eff_d = if symmetry.has_y() { depth / 2.0 } else { depth };

    let fold = |p: Point3D| -> (f64, f64, f64) {
        let (q, _) = map_to_reduced_domain(p, &symmetry, width, depth);
        (q.x, q.y, q.z)
    };
    let mut px = Vec::new();
    let mut py = Vec::new();
    let mut pz = Vec::new();
    for s in &sim.sources {
        let (x, y, z) = fold(s.position);
        px.push(x);
        py.push(y);
        pz.push(z);
    }
    for lp in &sim.listening_positions {
        let (x, y, z) = fold(*lp);
        px.push(x);
        py.push(y);
        pz.push(z);
    }

    let eppm = elements_per_meter(max_freq, sim.speed_of_sound);
    let nx = axis_node_count(eff_w, eppm, &px, fine_radius, grading);
    let ny = axis_node_count(eff_d, eppm, &py, fine_radius, grading);
    let nz = axis_node_count(height, eppm, &pz, fine_radius, grading);
    let nodes = (nx as u64) * (ny as u64) * (nz as u64);
    nodes * FEM_BYTES_PER_NODE
}

/// Worst-case estimated FEM memory across the selected scenarios.
fn worst_case_fem_memory(
    scenarios: &[scenarios::Scenario],
    grading: f64,
    fine_radius: f64,
    max_freq: f64,
) -> (u64, String) {
    let mut worst = 0u64;
    let mut worst_name = String::new();
    for s in scenarios {
        let bytes = estimate_fem_memory(&s.simulation, grading, fine_radius, max_freq);
        if bytes > worst {
            worst = bytes;
            worst_name = s.name.clone();
        }
    }
    (worst, worst_name)
}

/// Find the largest `max_freq <= upper` such that the worst-case FEM memory
/// fits within `limit_bytes`. Returns `None` if even the lower bound exceeds
/// the limit.
fn fit_max_freq(
    scenarios: &[scenarios::Scenario],
    grading: f64,
    fine_radius: f64,
    upper: f64,
    limit_bytes: u64,
) -> Option<f64> {
    const LOWER: f64 = 50.0;
    if worst_case_fem_memory(scenarios, grading, fine_radius, LOWER).0 > limit_bytes {
        return None;
    }
    if worst_case_fem_memory(scenarios, grading, fine_radius, upper).0 <= limit_bytes {
        return Some(upper);
    }
    let (mut lo, mut hi) = (LOWER, upper);
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if worst_case_fem_memory(scenarios, grading, fine_radius, mid).0 <= limit_bytes {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

fn detect_default_memory_limit_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    let free = sys.available_memory();
    (free as f64 * 0.8) as u64
}

struct MemoryBudget {
    bytes: u64,
    source: String,
}

#[cfg(feature = "cuda")]
fn detect_cuda_device_memory_bytes(gpu_device: usize) -> Result<(u64, u64)> {
    let backend = sonium_compute::CudaBackend::new(gpu_device)
        .map_err(|e| anyhow::anyhow!("failed to query CUDA device {gpu_device} memory: {e}"))?;
    let (free, total) = backend
        .context()
        .mem_get_info()
        .map_err(|e| anyhow::anyhow!("failed to query CUDA device {gpu_device} memory: {e:?}"))?;
    Ok((free as u64, total as u64))
}

#[cfg(not(feature = "cuda"))]
fn detect_cuda_device_memory_bytes(_gpu_device: usize) -> Result<(u64, u64)> {
    anyhow::bail!(
        "--backend cuda requires building with `--features cuda`. \
         Rebuild: `cargo build --release -p sonium-sim-room --features cuda --bin generate-roomeq-data`"
    )
}

fn resolve_memory_budget(args: &Args, backend: BackendKind) -> Result<MemoryBudget> {
    let host_or_user_budget = if args.memory_limit_gb > 0.0 {
        (args.memory_limit_gb * 1024.0 * 1024.0 * 1024.0) as u64
    } else {
        detect_default_memory_limit_bytes()
    };
    let host_or_user_source = if args.memory_limit_gb > 0.0 {
        "user-specified"
    } else {
        "auto: 80% of free host RAM at start"
    };

    if backend != BackendKind::Cuda {
        return Ok(MemoryBudget {
            bytes: host_or_user_budget,
            source: host_or_user_source.to_string(),
        });
    }

    let (gpu_free, gpu_total) = detect_cuda_device_memory_bytes(args.gpu_device)?;
    let gpu_budget = (gpu_free as f64 * 0.8) as u64;
    log::info!(
        "CUDA device {} memory: free {}, total {}, conservative budget {}",
        args.gpu_device,
        fmt_gib(gpu_free),
        fmt_gib(gpu_total),
        fmt_gib(gpu_budget),
    );

    Ok(MemoryBudget {
        bytes: host_or_user_budget.min(gpu_budget),
        source: format!(
            "min({host_or_user_source}, 80% of free CUDA device {} VRAM)",
            args.gpu_device
        ),
    })
}

fn fmt_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else if ms >= 10.0 {
        format!("{ms:.0}ms")
    } else {
        format!("{ms:.1}ms")
    }
}

fn prompt_yes_no(question: &str) -> bool {
    print!("{question} [y/N]: ");
    io::stdout().flush().ok();
    let mut buf = String::new();
    if io::stdin().read_line(&mut buf).is_err() {
        return false;
    }
    matches!(buf.trim(), "y" | "Y" | "yes" | "Yes" | "YES")
}

fn parse_measured_trust_policy(
    specifications: &[String],
) -> Result<Option<measured_io::MeasuredDatasetTrustPolicy>> {
    if specifications.is_empty() {
        return Ok(None);
    }
    let anchors = specifications
        .iter()
        .map(|specification| {
            let (key_id, public_key) = specification.split_once('=').ok_or_else(|| {
                anyhow::anyhow!("--measured-trusted-key must use KEY_ID=LOWERCASE_PUBLIC_KEY_HEX")
            })?;
            if key_id.trim().is_empty() || public_key.is_empty() {
                anyhow::bail!("--measured-trusted-key requires non-empty key ID and public key");
            }
            Ok(measured_io::MeasuredDatasetTrustAnchor {
                key_id: key_id.to_owned(),
                public_key: public_key.to_owned(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    measured_io::MeasuredDatasetTrustPolicy::new(anchors)
        .map(Some)
        .map_err(anyhow::Error::msg)
}

fn main() -> Result<()> {
    let args = Args::parse();

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(if args.verbose { "debug" } else { "info" }),
    )
    .init();

    let backend = match BackendKind::parse(&args.backend) {
        Ok(b) => b,
        Err(e) => anyhow::bail!(e),
    };
    let solver = SolverKind::parse(&args.solver).map_err(anyhow::Error::msg)?;
    let measured_trust_policy = parse_measured_trust_policy(&args.measured_trusted_keys)?;
    anyhow::ensure!(
        !args.measured_require_signature || measured_trust_policy.is_some(),
        "--measured-require-signature requires at least one --measured-trusted-key"
    );
    if matches!(
        args.shifted_laplacian,
        fem_runner::ShiftedLaplacianShift::Csl { .. }
    ) {
        log::warn!(
            "--shifted-laplacian csl is experimental and is known to be a poor ILUT/block-Jacobi target at high frequencies; use k2 for production sweeps"
        );
    }
    if backend == BackendKind::Cuda && is_point_jacobi_gpu_precond(&args.gpu_precond) {
        log::warn!(
            "--gpu-precond point-jacobi stalls around residual 1e-2 on indefinite Helmholtz; \
             use --gpu-precond auto (CUDA AMG), poly-jacobi-N, or chebyshev-N for production sweeps"
        );
    }
    if backend == BackendKind::Cpu
        && uses_serial_apply_preconditioner(args.preconditioner)
        && effective_thread_count(args.max_concurrent) > 8
    {
        log::warn!(
            "--preconditioner {} applies mostly serial triangular solves and will not use \
             {} threads; prefer block-jacobi, additive-schwarz, or amg at this width",
            args.preconditioner,
            effective_thread_count(args.max_concurrent),
        );
    }
    let dense_blr = dense_blr_options(&args)?;
    if solver == SolverKind::FastHybrid && dense_blr.is_some() {
        anyhow::bail!("--dense cannot be combined with --solver fast-hybrid");
    }
    let memory_budget = resolve_memory_budget(&args, backend)?;
    log::info!(
        "Memory budget: {} ({})",
        fmt_gib(memory_budget.bytes),
        memory_budget.source,
    );

    anyhow::ensure!(
        args.min_freq.is_finite()
            && args.max_freq.is_finite()
            && args.min_freq > 0.0
            && args.max_freq >= args.min_freq,
        "frequency range must be finite and satisfy 0 < --min-freq <= --max-freq"
    );
    anyhow::ensure!(args.num_points > 0, "--num-points must be at least 1");

    let freq_config = scenarios::FrequencyConfig {
        min_freq: args.min_freq,
        max_freq: args.max_freq,
        num_points: args.num_points,
    };

    let all_scenarios = if let Some(ref name) = args.scenario {
        match scenarios::scenario_by_name_with_config(name, &freq_config) {
            Some(s) => vec![s],
            None => {
                let available: Vec<String> = scenarios::all_scenarios()
                    .iter()
                    .map(|s| s.name.clone())
                    .collect();
                anyhow::bail!(
                    "Unknown scenario '{}'. Available: {}",
                    name,
                    available.join(", ")
                );
            }
        }
    } else {
        scenarios::all_scenarios_with_config(&freq_config)
    };
    for scenario in &all_scenarios {
        scenario
            .validate()
            .map_err(|error| anyhow::Error::msg(format!("invalid scenario definition: {error}")))?;
    }
    let measured_dataset = args
        .measured_response
        .as_ref()
        .map(|path| {
            measured_io::load_measured_response_dataset_json(path).map_err(anyhow::Error::msg)
        })
        .transpose()?;
    let (measured_dataset, measured_signature_key_id) =
        match (measured_dataset, measured_trust_policy.as_ref()) {
            (Some(_), Some(policy)) => {
                let path = args
                    .measured_response
                    .as_ref()
                    .expect("measured response path required for trust policy");
                let authenticated =
                    measured_io::load_authenticated_measured_response_dataset_json(path, policy)
                        .map_err(anyhow::Error::msg)?;
                (
                    Some(authenticated.dataset().clone()),
                    Some(authenticated.signature().key_id.clone()),
                )
            }
            (dataset, None) => (dataset, None),
            (None, Some(_)) => (None, None),
        };
    let measured_uncertainty = args
        .measured_uncertainty
        .as_ref()
        .map(|path| {
            let dataset = measured_dataset.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--measured-uncertainty requires --measured-response")
            })?;
            MeasuredUncertaintyEnvelope::load_json(path, dataset).map_err(anyhow::Error::msg)
        })
        .transpose()?;
    if !args.gallery_reference_frequency_tolerance_hz.is_finite()
        || args.gallery_reference_frequency_tolerance_hz < 0.0
    {
        anyhow::bail!("--gallery-reference-frequency-tolerance-hz must be finite and non-negative");
    }
    if !args.gallery_reference_relative_floor_pa.is_finite()
        || args.gallery_reference_relative_floor_pa <= 0.0
    {
        anyhow::bail!("--gallery-reference-relative-floor-pa must be finite and positive");
    }

    // Memory check.
    let mut effective_max_freq = args.max_freq;
    if solver == SolverKind::FastHybrid {
        log::info!("Skipping FEM peak-memory check for fast-hybrid solver");
    } else if dense_blr.is_some_and(|opts| opts.max_freq >= args.max_freq) {
        log::info!(
            "Skipping FEM peak-memory check: dense BLR covers all simulated frequencies <= {:.0} Hz",
            args.max_freq,
        );
    } else {
        let (peak, worst_name) = worst_case_fem_memory(
            &all_scenarios,
            args.mesh_grading,
            args.mesh_fine_radius,
            args.max_freq,
        );
        log::info!(
            "Estimated FEM peak memory at max_freq={:.0} Hz: {} (worst scenario: {})",
            args.max_freq,
            fmt_gib(peak),
            worst_name,
        );
        if peak > memory_budget.bytes {
            match fit_max_freq(
                &all_scenarios,
                args.mesh_grading,
                args.mesh_fine_radius,
                args.max_freq,
                memory_budget.bytes,
            ) {
                Some(f_fit) => {
                    let (peak_fit, _) = worst_case_fem_memory(
                        &all_scenarios,
                        args.mesh_grading,
                        args.mesh_fine_radius,
                        f_fit,
                    );
                    println!(
                        "Estimated FEM peak {} exceeds budget {} (scenario: {}).",
                        fmt_gib(peak),
                        fmt_gib(memory_budget.bytes),
                        worst_name,
                    );
                    println!(
                        "Lowering --max-freq to {:.0} Hz would fit (estimated {}).",
                        f_fit,
                        fmt_gib(peak_fit),
                    );
                    let accept = args.auto_reduce_freq
                        || prompt_yes_no("Reduce max_freq to fit the memory budget?");
                    if !accept {
                        anyhow::bail!(
                            "Aborting: estimated FEM memory exceeds the budget. \
                             Re-run with a higher --memory-limit-gb, a lower --max-freq, \
                             or --auto-reduce-freq."
                        );
                    }
                    effective_max_freq = f_fit;
                }
                None => {
                    anyhow::bail!(
                        "Even the minimum FEM resolution exceeds the memory budget {}. \
                         Increase --memory-limit-gb.",
                        fmt_gib(memory_budget.bytes),
                    );
                }
            }
        }
    }

    // Re-derive scenarios if max_freq was reduced.
    let all_scenarios = if (effective_max_freq - args.max_freq).abs() > f64::EPSILON {
        let cfg = scenarios::FrequencyConfig {
            min_freq: args.min_freq,
            max_freq: effective_max_freq,
            num_points: args.num_points,
        };
        if let Some(ref name) = args.scenario {
            scenarios::scenario_by_name_with_config(name, &cfg)
                .map(|s| vec![s])
                .unwrap_or_default()
        } else {
            scenarios::all_scenarios_with_config(&cfg)
        }
    } else {
        all_scenarios
    };

    let gallery_reference =
        if args.gallery_reference.is_some() || args.gallery_reference_command.is_some() {
            anyhow::ensure!(
                all_scenarios.len() == 1,
                "gallery reference comparison requires exactly one selected scenario"
            );
            load_or_execute_gallery_reference(
                &all_scenarios[0].simulation,
                &all_scenarios[0].name,
                &args.output_dir,
                args.gallery_reference.as_deref(),
                args.gallery_reference_command.as_deref(),
                args.gallery_reference_backend.as_deref(),
                &args.gallery_reference_args,
            )?
        } else {
            None
        };

    if let Some(dataset) = measured_dataset.as_ref() {
        anyhow::ensure!(
            solver == SolverKind::Fem,
            "--measured-response requires --solver fem because reduced-order outputs are not admitted"
        );
        anyhow::ensure!(
            all_scenarios.len() == 1,
            "--measured-response requires exactly one selected scenario"
        );
        anyhow::ensure!(
            args.measured_max_magnitude_error_db.is_finite()
                && args.measured_max_magnitude_error_db >= 0.0,
            "--measured-max-magnitude-error-db must be finite and non-negative"
        );
        anyhow::ensure!(
            args.measured_max_phase_error_rad.is_finite()
                && args.measured_max_phase_error_rad >= 0.0,
            "--measured-max-phase-error-rad must be finite and non-negative"
        );
        anyhow::ensure!(
            args.measured_frequency_tolerance_hz.is_finite()
                && args.measured_frequency_tolerance_hz >= 0.0,
            "--measured-frequency-tolerance-hz must be finite and non-negative"
        );
        if measured_uncertainty.is_some() {
            anyhow::ensure!(
                args.measured_max_normalized_magnitude_error.is_finite()
                    && args.measured_max_normalized_magnitude_error >= 0.0,
                "--measured-max-normalized-magnitude-error must be finite and non-negative"
            );
            anyhow::ensure!(
                args.measured_max_normalized_phase_error.is_finite()
                    && args.measured_max_normalized_phase_error >= 0.0,
                "--measured-max-normalized-phase-error must be finite and non-negative"
            );
            anyhow::ensure!(
                args.measured_minimum_phase_samples > 0,
                "--measured-minimum-phase-samples must be at least 1"
            );
        }

        let scenario = &all_scenarios[0];
        let expected_listener_count = scenario.simulation.listening_positions.len();
        anyhow::ensure!(
            args.measured_listener_names.len() == expected_listener_count,
            "--measured-listener-name must be repeated exactly {expected_listener_count} times for scenario {}",
            scenario.name
        );
        let model_source_names: Vec<String> = scenario
            .simulation
            .sources
            .iter()
            .map(|source| source.name.clone())
            .collect();
        anyhow::ensure!(
            dataset.metadata.listener_names == args.measured_listener_names,
            "measured dataset listener names must match --measured-listener-name in model tensor order"
        );
        anyhow::ensure!(
            dataset.response.source_names == model_source_names,
            "measured dataset source names must match the selected scenario"
        );
    }

    // Initialize the global rayon pool with an enlarged worker stack. The
    // nested par_iter calls in FEM assembly + GMRES + ILUT blow past the
    // default ~2 MiB worker stack. Only override num_threads when the user
    // requested it.
    {
        if let Err(e) = sonium_sim_room::runtime::install_fem_rayon_pool(args.max_concurrent) {
            anyhow::bail!(
                "Failed to install the global rayon pool with the FEM-sized worker \
                 stack: {e}. Something initialized rayon before this point; the \
                 default 2 MiB stack will overflow during nested FEM assembly."
            );
        }
    }

    if args.execution_contract {
        anyhow::ensure!(
            backend == BackendKind::Cpu,
            "--execution-contract is a host contract; run an accelerator workload to record device identity"
        );
        let evidence = sonium_xem::runtime::observe_execution_contract(
            "generate-roomeq-data",
            "cpu",
            "cpu",
            std::collections::BTreeMap::from([
                ("scenario_concurrency".to_string(), 1),
                (
                    "fem_inner_pool_budget".to_string(),
                    rayon::current_num_threads(),
                ),
            ]),
            false,
            "host-cpu",
        );
        println!("{}", serde_json::to_string_pretty(&evidence)?);
        return Ok(());
    }

    let solver_summary = if solver == SolverKind::FastHybrid {
        "solver=fast-hybrid (shoebox modes + coherent image sources)".to_string()
    } else {
        match backend {
            BackendKind::Cpu => format!("backend=cpu, preconditioner={}", args.preconditioner),
            BackendKind::Metal => format!(
                "backend=metal, gpu_precision={}, gpu_precond={}",
                args.gpu_precision, args.gpu_precond
            ),
            BackendKind::Cuda => format!(
                "backend=cuda, gpu_precision={}, gpu_precond={}",
                args.gpu_precision, args.gpu_precond
            ),
        }
    };
    let dense_summary = dense_blr
        .map(|opts| format!(", dense_blr<={:.0} Hz", opts.max_freq))
        .unwrap_or_default();

    log::info!(
        "Generating room data for {} scenarios, max_freq={:.0} Hz, {} points, {}, recycle_subspace_size={}, modal_damping_ratio={:.4}{}{}",
        all_scenarios.len(),
        effective_max_freq,
        args.num_points,
        solver_summary,
        args.recycle_subspace_size,
        args.modal_damping_ratio,
        dense_summary,
        if args.max_concurrent > 0 {
            format!(", rayon_threads={}", args.max_concurrent)
        } else {
            String::new()
        },
    );

    if !args.gmres_tolerance.is_finite() || args.gmres_tolerance <= 0.0 {
        anyhow::bail!("--gmres-tolerance must be a positive finite value");
    }
    if !args.modal_damping_ratio.is_finite() || args.modal_damping_ratio < 0.0 {
        anyhow::bail!("--modal-damping-ratio must be a finite non-negative value");
    }
    let fem_mesh = fem_runner::FemMeshConfig {
        elements_per_wavelength: fem_runner::FemMeshConfig::default().elements_per_wavelength,
        grading_factor: args.mesh_grading,
        fine_radius: args.mesh_fine_radius,
        preconditioner: args.preconditioner,
        recycle_subspace_size: args.recycle_subspace_size,
        recycler: args.recycler,
        rebuild_precond_every: args.rebuild_precond_every.max(1),
        recycle_reset_every: args.recycle_reset_every,
        gmres_tolerance: args.gmres_tolerance,
        gmres_restart: args.gmres_restart.max(1),
        modal_damping_ratio: args.modal_damping_ratio,
        shifted_laplacian_shift: args.shifted_laplacian,
        source_model: args.source_model,
        block_source_solve: args.block_source_solve,
        allow_chained_complex_crossover: args.allow_chained_complex_crossover,
    };
    let solver_config = SolverRunConfig {
        solver,
        fem_mesh: &fem_mesh,
        backend,
        gpu_precision: &args.gpu_precision,
        gpu_precond: &args.gpu_precond,
        gpu_device: args.gpu_device,
        dense_blr,
        measured_dataset: measured_dataset.as_ref(),
        measured_signature_key_id: measured_signature_key_id.as_deref(),
        measured_listener_names: &args.measured_listener_names,
        measured_max_magnitude_error_db: args.measured_max_magnitude_error_db,
        measured_max_phase_error_rad: args.measured_max_phase_error_rad,
        measured_frequency_tolerance_hz: args.measured_frequency_tolerance_hz,
        measured_uncertainty: measured_uncertainty.as_ref(),
        measured_max_normalized_magnitude_error: args.measured_max_normalized_magnitude_error,
        measured_max_normalized_phase_error: args.measured_max_normalized_phase_error,
        measured_minimum_phase_samples: args.measured_minimum_phase_samples,
        gallery_reference: gallery_reference.as_ref(),
        gallery_reference_frequency_tolerance_hz: args.gallery_reference_frequency_tolerance_hz,
        gallery_reference_relative_floor_pa: args.gallery_reference_relative_floor_pa,
        gallery_slice_grid: args.gallery_slice_grid,
    };

    // Sequential outer loop: each scenario gets the entire rayon pool for its
    // nested FEM assembly / GMRES / preconditioner par_iters. Running
    // scenarios in parallel oversubscribes the pool and starves inner
    // work-stealing, which previously left most cores idle on memory-bandwidth
    // bound SpMV.
    let mut failed_scenarios = Vec::new();
    for scenario in &all_scenarios {
        if let Err(e) = run_solver(
            scenario,
            &args.output_dir,
            &solver_config,
            args.gallery.as_deref(),
        ) {
            log::error!("Failed scenario={}: {}", scenario.name, e);
            failed_scenarios.push(format!("{}: {e}", scenario.name));
        }
    }

    if !failed_scenarios.is_empty() {
        anyhow::bail!(
            "data generation failed for {} scenario(s): {}",
            failed_scenarios.len(),
            failed_scenarios.join("; ")
        );
    }

    log::info!(
        "Data generation complete. Output: {}",
        args.output_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod cli_guard_tests {
    use super::*;

    #[test]
    fn point_jacobi_aliases_are_detected() {
        for spec in [
            "point-jacobi",
            "pointjacobi",
            "diagonal",
            "jacobi",
            " Point_Jacobi ",
            "JACOBI",
        ] {
            assert!(is_point_jacobi_gpu_precond(spec), "{spec}");
        }
        for spec in ["auto", "amg", "poly-jacobi-4", "chebyshev-8", ""] {
            assert!(!is_point_jacobi_gpu_precond(spec), "{spec}");
        }
    }

    #[test]
    fn serial_apply_preconditioners_are_flagged() {
        for kind in [
            fem_runner::FemPreconditionerKind::Ilut,
            fem_runner::FemPreconditionerKind::IluColoring,
            fem_runner::FemPreconditionerKind::IluFixedPoint,
        ] {
            assert!(uses_serial_apply_preconditioner(kind), "{kind}");
        }
        for kind in [
            fem_runner::FemPreconditionerKind::BlockJacobi,
            fem_runner::FemPreconditionerKind::AdditiveSchwarz,
            fem_runner::FemPreconditionerKind::Oras,
            fem_runner::FemPreconditionerKind::Habc,
            fem_runner::FemPreconditionerKind::Amg,
        ] {
            assert!(!uses_serial_apply_preconditioner(kind), "{kind}");
        }
    }

    #[test]
    fn effective_threads_honours_explicit_cap() {
        assert_eq!(effective_thread_count(60), 60);
        assert_eq!(effective_thread_count(1), 1);
        assert!(effective_thread_count(0) >= 1);
    }
}

#[cfg(test)]
mod qa_phase_feature_tests {
    use super::*;

    fn fixture() -> SimulationOutput {
        SimulationOutput::solved(
            vec![40.0, 120.0, 300.0],
            vec![
                vec![
                    vec![Complex64::new(1.0, 0.0); 3],
                    vec![Complex64::new(1.0, 0.0); 3],
                ],
                vec![
                    vec![Complex64::new(0.5, 0.0); 3],
                    vec![Complex64::new(0.5, 0.0); 3],
                ],
            ],
            vec!["left".to_string(), "right".to_string()],
        )
    }

    #[test]
    fn qa_allpass_has_unit_magnitude() {
        for frequency in [20.0, 120.0, 1_000.0, 20_000.0] {
            let response = qa_allpass_response(frequency, 48_000.0, 120.0, 5.0);
            assert!((response.norm() - 1.0).abs() < 1.0e-10);
        }
    }

    #[test]
    fn qa_phase_features_are_scoped_and_repeatable_across_seats() {
        let original = fixture();
        let mut untouched = original.clone();
        apply_roomeq_qa_features("medium_surround_5_1_4", &mut untouched);
        assert_eq!(untouched.pressures, original.pressures);

        let mut conditioned = original.clone();
        apply_roomeq_qa_features("medium_surround_5_2_4_multi_seat", &mut conditioned);
        assert_ne!(conditioned.pressures[0], original.pressures[0]);
        assert_ne!(conditioned.pressures[1], original.pressures[1]);
        assert_eq!(conditioned.pressures[0][0], conditioned.pressures[0][1]);
        assert_eq!(conditioned.pressures[1][0], conditioned.pressures[1][1]);
        for pressure in &conditioned.pressures[0][0] {
            assert!((pressure.norm() - 1.0).abs() < 1.0e-10);
        }
    }

    #[test]
    fn fast_hybrid_solver_gallery_flow_exports_native_contract() {
        let mut scenario = scenarios::scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.frequencies = vec![50.0, 100.0];
        let output_base = tempfile::tempdir().unwrap();
        let gallery_base = tempfile::tempdir().unwrap();
        let fem_mesh = fem_runner::FemMeshConfig::default();
        let config = SolverRunConfig {
            solver: SolverKind::FastHybrid,
            fem_mesh: &fem_mesh,
            backend: BackendKind::Cpu,
            gpu_precision: "f64",
            gpu_precond: "auto",
            gpu_device: 0,
            dense_blr: None,
            measured_dataset: None,
            measured_signature_key_id: None,
            measured_listener_names: &[],
            measured_max_magnitude_error_db: 3.0,
            measured_max_phase_error_rad: std::f64::consts::PI,
            measured_frequency_tolerance_hz: 1.0e-6,
            measured_uncertainty: None,
            measured_max_normalized_magnitude_error: 3.0,
            measured_max_normalized_phase_error: 3.0,
            measured_minimum_phase_samples: 1,
            gallery_reference: None,
            gallery_reference_frequency_tolerance_hz: 1.0e-6,
            gallery_reference_relative_floor_pa: 20.0e-6,
            gallery_slice_grid: gallery::DEFAULT_SLICE_GRID,
        };

        run_solver(
            &scenario,
            output_base.path(),
            &config,
            Some(gallery_base.path()),
        )
        .unwrap();

        let gallery_dir = gallery_base.path().join("fast-hybrid").join(&scenario.name);
        let required = [
            "index.html",
            "gallery.json",
            "gallery-provenance.json",
            "frequency_response.csv",
            "source_listener_response.csv",
            "room_geometry.csv",
            "room_layout.svg",
            "frequency_response_heatmap.png",
            "source_listener_response_heatmap.png",
            "frequency_response_plot.svg",
            "impulse_response_plot.svg",
            "energy_decay_plot.svg",
            "provenance_plot.svg",
        ];
        for filename in required {
            assert!(
                gallery_dir.join(filename).is_file(),
                "gallery is missing required asset {filename}"
            );
        }
        let source_wav = format!("{}_lp0.wav", scenario.simulation.sources[0].name);
        assert!(gallery_dir.join(&source_wav).is_file());
        sonium_xem::io::validate_gallery_manifest(&gallery_dir.join("gallery.json")).unwrap();
        assert_eq!(
            sonium_xem::io::validate_png_nonblank(
                gallery_dir.join("frequency_response_heatmap.png")
            )
            .unwrap(),
            (1920, 1080)
        );
        let provenance: serde_json::Value = serde_json::from_slice(
            &std::fs::read(gallery_dir.join("gallery-provenance.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(provenance["schema"], "sonium-sim-room-gallery/v1");
        assert!(provenance["checksums"]["index.html"].is_string());
        assert_eq!(provenance["solver"], "fast-hybrid");
        assert_eq!(provenance["viewport"]["width"], 1920);
        assert_eq!(provenance["viewport"]["height"], 1080);
        assert!(provenance["checksums"].as_object().unwrap().len() >= required.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_xem::{
        BoundaryConfig, CrossoverFilter, Directivity, Point3D, RectangularRoom, Source,
        SurfaceConfig,
    };

    #[test]
    fn dense_frequency_count_splits_at_cutoff() {
        let frequencies = [20.0, 80.0, 250.0, 400.0];

        assert_eq!(dense_frequency_count(&frequencies, 10.0), 0);
        assert_eq!(dense_frequency_count(&frequencies, 250.0), 3);
        assert_eq!(dense_frequency_count(&frequencies, 1000.0), 4);
    }

    #[test]
    fn dense_frequency_bounds_keep_exact_overlap_on_both_sides() {
        let frequencies = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0];

        assert_eq!(dense_frequency_bounds(&frequencies, 250.0, 0.0), Ok((4, 4)));
        assert_eq!(
            dense_frequency_bounds(&frequencies, 250.0, 60.0),
            Ok((5, 2))
        );
        assert!(
            dense_frequency_bounds(&[100.0, 200.0], 150.0, 1.0)
                .unwrap_err()
                .contains("exact point")
        );
    }

    #[test]
    fn dense_flag_accepts_optional_cutoff() {
        let args = Args::try_parse_from(["generate-roomeq-data", "--dense"]).unwrap();
        assert!(args.dense);
        assert_eq!(args.dense_max_freq, None);

        let args =
            Args::try_parse_from(["generate-roomeq-data", "--dense", "--dense-max-freq", "300"])
                .unwrap();
        assert_eq!(args.dense_max_freq, Some(300.0));

        let args = Args::try_parse_from([
            "generate-roomeq-data",
            "--dense",
            "--dense-overlap-hz",
            "60",
        ])
        .unwrap();
        assert_eq!(args.dense_overlap_hz, 60.0);
    }

    #[test]
    fn dense_overlap_rejects_negative_and_nonfinite_values() {
        for overlap in ["-1", "NaN"] {
            let args = Args::try_parse_from([
                "generate-roomeq-data",
                "--dense",
                "--scenario",
                "small_room",
                "--num-points",
                "8",
                &format!("--dense-overlap-hz={overlap}"),
            ])
            .unwrap();
            let error = dense_blr_options(&args).unwrap_err().to_string();
            assert!(
                error.contains("--dense-overlap-hz"),
                "unexpected error for {overlap}: {error}"
            );
        }
    }

    #[test]
    fn dense_path_is_limited_to_small_single_scenario_reference_runs() {
        let without_scenario =
            Args::try_parse_from(["generate-roomeq-data", "--dense", "--num-points", "8"]).unwrap();
        assert!(
            dense_blr_options(&without_scenario)
                .unwrap_err()
                .to_string()
                .contains("--scenario")
        );

        let too_many = Args::try_parse_from([
            "generate-roomeq-data",
            "--dense",
            "--scenario",
            "small_room_stereo",
            "--num-points",
            "9",
        ])
        .unwrap();
        assert!(
            dense_blr_options(&too_many)
                .unwrap_err()
                .to_string()
                .contains("at most 8")
        );
    }

    #[test]
    fn modal_damping_ratio_cli_overrides_default() {
        let args = Args::try_parse_from(["generate-roomeq-data", "--modal-damping-ratio", "0.012"])
            .unwrap();
        assert_eq!(args.modal_damping_ratio, 0.012);

        let args = Args::try_parse_from(["generate-roomeq-data"]).unwrap();
        assert_eq!(
            args.modal_damping_ratio,
            fem_runner::DEFAULT_MODAL_DAMPING_RATIO
        );
    }

    #[test]
    fn complex_crossover_chaining_cli_is_explicit_and_off_by_default() {
        let args = Args::try_parse_from(["generate-roomeq-data"]).unwrap();
        assert!(!args.allow_chained_complex_crossover);

        let args =
            Args::try_parse_from(["generate-roomeq-data", "--allow-chained-complex-crossover"])
                .unwrap();
        assert!(args.allow_chained_complex_crossover);
    }

    #[test]
    fn measured_comparison_options_require_the_dataset_path() {
        for (option, value) in [
            ("--measured-listener-name", "seat-a"),
            ("--measured-max-magnitude-error-db", "1"),
            ("--measured-max-phase-error-rad", "1"),
            ("--measured-frequency-tolerance-hz", "1"),
        ] {
            let error = Args::try_parse_from(["generate-roomeq-data", option, value]).unwrap_err();
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::MissingRequiredArgument
            );
        }

        let error = Args::try_parse_from([
            "generate-roomeq-data",
            "--measured-response",
            "dataset.json",
        ])
        .unwrap_err();
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn measured_uncertainty_options_require_dataset_and_parse_defaults() {
        let error = Args::try_parse_from([
            "generate-roomeq-data",
            "--measured-uncertainty",
            "uncertainty.json",
        ])
        .unwrap_err();
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let args = Args::try_parse_from([
            "generate-roomeq-data",
            "--scenario",
            "small_room_stereo",
            "--measured-response",
            "dataset.json",
            "--measured-uncertainty",
            "uncertainty.json",
        ])
        .unwrap();
        assert_eq!(args.measured_max_normalized_magnitude_error, 3.0);
        assert_eq!(args.measured_max_normalized_phase_error, 3.0);
        assert_eq!(args.measured_minimum_phase_samples, 1);

        let error = Args::try_parse_from([
            "generate-roomeq-data",
            "--measured-max-normalized-phase-error",
            "1",
        ])
        .unwrap_err();
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn measured_signature_options_require_dataset_and_parse_trust_keys() {
        let missing_dataset =
            Args::try_parse_from(["generate-roomeq-data", "--measured-trusted-key", "lab=00"])
                .unwrap_err();
        assert_eq!(
            missing_dataset.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let trusted_key = format!("lab-v1={}", "00".repeat(32));
        let args = Args::try_parse_from(vec![
            "generate-roomeq-data".to_owned(),
            "--scenario".to_owned(),
            "small_stereo_2_0".to_owned(),
            "--measured-response".to_owned(),
            "dataset.json".to_owned(),
            "--measured-trusted-key".to_owned(),
            trusted_key.clone(),
            "--measured-require-signature".to_owned(),
        ])
        .unwrap();
        assert_eq!(args.measured_trusted_keys, vec![trusted_key]);
        assert!(args.measured_require_signature);

        let policy = parse_measured_trust_policy(&args.measured_trusted_keys).unwrap();
        assert_eq!(policy.unwrap().trusted_keys[0].key_id, "lab-v1");
        let invalid = parse_measured_trust_policy(&["malformed".into()]).unwrap_err();
        assert!(
            invalid
                .to_string()
                .contains("KEY_ID=LOWERCASE_PUBLIC_KEY_HEX")
        );
    }

    #[test]
    fn gallery_reference_command_options_require_gallery_and_scenario() {
        let missing_gallery = Args::try_parse_from([
            "generate-roomeq-data",
            "--scenario",
            "small_stereo_2_0",
            "--gallery-reference-command",
            "freefem",
        ])
        .unwrap_err();
        assert_eq!(
            missing_gallery.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let missing_scenario = Args::try_parse_from([
            "generate-roomeq-data",
            "--gallery",
            "gallery",
            "--gallery-reference-command",
            "freefem",
        ])
        .unwrap_err();
        assert_eq!(
            missing_scenario.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let args = Args::try_parse_from([
            "generate-roomeq-data",
            "--scenario",
            "small_stereo_2_0",
            "--gallery",
            "gallery",
            "--gallery-reference-command",
            "freefem",
            "--gallery-reference-backend",
            "freefem-v1",
            "--gallery-reference-arg=--mesh=room.edp",
        ])
        .unwrap();
        assert_eq!(
            args.gallery_reference_command,
            Some(OsString::from("freefem"))
        );
        assert_eq!(
            args.gallery_reference_backend.as_deref(),
            Some("freefem-v1")
        );
        assert_eq!(
            args.gallery_reference_args,
            vec![OsString::from("--mesh=room.edp")]
        );

        let conflict = Args::try_parse_from([
            "generate-roomeq-data",
            "--scenario",
            "small_stereo_2_0",
            "--gallery",
            "gallery",
            "--gallery-reference",
            "reference.json",
            "--gallery-reference-command",
            "freefem",
        ])
        .unwrap_err();
        assert_eq!(conflict.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn gallery_reference_command_executes_checked_driver() {
        let simulation = RoomSimulation::with_frequencies(
            RoomGeometry::Rectangular(sonium_xem::RectangularRoom::new(1.0, 1.0, 1.0)),
            vec![sonium_xem::Source::omnidirectional(
                Point3D::new(0.25, 0.25, 0.25),
                1.0,
            )],
            vec![Point3D::new(0.75, 0.75, 0.75)],
            100.0,
            100.0,
            1,
        );
        let directory = tempfile::tempdir().unwrap();
        let manifest = external_reference::ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "sh",
            &simulation,
        )
        .unwrap();
        let response = serde_json::json!({
            "schema": gallery::REFERENCE_RESPONSE_SCHEMA_V1,
            "backend": "sh",
            "problem_sha256": manifest.content_sha256,
            "frequencies": manifest.frequencies_hz,
            "source_names": manifest.sources.iter().map(|source| source.name.clone()).collect::<Vec<_>>(),
            "pressures": [[[{"real_pa": 1.0, "imag_pa": 0.0}]]],
            "spatial_field": null,
        });
        std::fs::write(
            directory.path().join("response.json"),
            serde_json::to_vec(&response).unwrap(),
        )
        .unwrap();

        let args = vec![
            OsString::from("-c"),
            OsString::from("cp response.json \"$ROOMSIM_EXTERNAL_REFERENCE_OUTPUT\""),
        ];
        let reference = load_or_execute_gallery_reference(
            &simulation,
            "fixture-room",
            directory.path(),
            None,
            Some(OsStr::new("sh")),
            Some("sh"),
            &args,
        )
        .unwrap()
        .unwrap();

        assert_eq!(reference.backend, "sh");
        assert_eq!(reference.problem_sha256, Some(manifest.content_sha256));
        assert_eq!(reference.output.frequencies, vec![100.0]);
        assert_eq!(reference.output.source_names, vec!["Source"]);
    }

    #[test]
    fn gallery_reference_options_require_gallery_and_scenario() {
        let missing_gallery = Args::try_parse_from([
            "generate-roomeq-data",
            "--scenario",
            "small_stereo_2_0",
            "--gallery-reference",
            "reference.json",
        ])
        .unwrap_err();
        assert_eq!(
            missing_gallery.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let missing_scenario = Args::try_parse_from([
            "generate-roomeq-data",
            "--gallery",
            "gallery",
            "--gallery-reference",
            "reference.json",
        ])
        .unwrap_err();
        assert_eq!(
            missing_scenario.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let args = Args::try_parse_from([
            "generate-roomeq-data",
            "--scenario",
            "small_stereo_2_0",
            "--gallery",
            "gallery",
            "--gallery-reference",
            "reference.json",
            "--gallery-reference-frequency-tolerance-hz",
            "0.25",
            "--gallery-reference-relative-floor-pa",
            "0.001",
        ])
        .unwrap();
        assert_eq!(
            args.gallery_reference,
            Some(PathBuf::from("reference.json"))
        );
        assert_eq!(args.gallery_reference_frequency_tolerance_hz, 0.25);
        assert_eq!(args.gallery_reference_relative_floor_pa, 0.001);
    }

    #[test]
    fn measured_comparison_sidecar_is_versioned_and_records_budget_status() {
        let dataset = measured_io::MeasuredResponseDataset::from_parts(
            sonium_sim_room::measured_validation::MeasuredResponseSet::new(
                vec![100.0],
                vec!["left".into()],
                vec![vec![vec![Complex64::new(1.0, 0.0)]]],
                "fixture",
            )
            .unwrap(),
            measured_io::MeasuredDatasetMetadata {
                dataset_id: "dataset-1".into(),
                instrument_id: "instrument-1".into(),
                calibration_reference: "cal-1".into(),
                pressure_unit: "Pa".into(),
                sample_convention: "peak_complex".into(),
                phase_reference: "trigger".into(),
                listener_names: vec!["seat-a".into()],
                content_sha256: String::new(),
            },
        )
        .unwrap();
        let output = SimulationOutput::solved(
            vec![100.0],
            vec![vec![vec![Complex64::new(1.0, 0.0)]]],
            vec!["left".into()],
        );
        let report = MeasuredComparisonReport {
            sample_count: 1,
            phase_sample_count: 1,
            rms_magnitude_error_db: 0.5,
            max_abs_magnitude_error_db: 0.5,
            rms_phase_error_rad: 0.1,
            max_abs_phase_error_rad: 0.1,
        };
        let uncertainty = Some(MeasuredUncertaintyArtifact {
            max_normalized_magnitude_error: 2.0,
            max_normalized_phase_error: 2.0,
            minimum_phase_samples: 1,
            within_budget: true,
            budget_error: None,
            report: MeasuredAcceptanceReport {
                comparison: report,
                magnitude_sample_count: 1,
                phase_sample_count: 1,
                rms_normalized_magnitude_error: 0.5,
                max_normalized_magnitude_error: 0.5,
                rms_normalized_phase_error: 0.1,
                max_normalized_phase_error: 0.1,
            },
        });
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("measured_comparison.json");
        write_measured_comparison(
            &path,
            "fixture-scenario",
            "fem",
            &output,
            &dataset,
            Some("lab-key-v1"),
            &["seat-a".into()],
            1.0e-6,
            0.25,
            0.05,
            report,
            Some("phase budget exceeded"),
            uncertainty,
        )
        .unwrap();

        let document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(document["schema_version"], MEASURED_COMPARISON_SCHEMA_V1);
        assert_eq!(document["dataset"]["dataset_id"], "dataset-1");
        assert_eq!(document["measured_signature_key_id"], "lab-key-v1");
        assert_eq!(document["uncertainty"]["within_budget"], true);
        assert_eq!(document["uncertainty"]["minimum_phase_samples"], 1);
        assert_eq!(
            document["dataset"]["content_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(document["within_budget"], false);
        assert_eq!(document["budget_error"], "phase budget exceeded");
        assert_eq!(document["report"]["sample_count"], 1);
    }

    #[test]
    fn gpu_defaults_to_auto_preconditioner() {
        let args = Args::try_parse_from(["generate-roomeq-data"]).unwrap();
        assert_eq!(args.gpu_precond, "auto");
    }

    #[test]
    fn fem_backend_metal_parses() {
        assert_eq!(BackendKind::parse("metal"), Ok(BackendKind::Metal));
        assert_eq!(BackendKind::parse("apple"), Ok(BackendKind::Metal));
        assert!(
            BackendKind::parse("bogus")
                .unwrap_err()
                .contains("cpu, metal, cuda")
        );
    }

    #[test]
    fn fast_hybrid_solver_and_csl_parse() {
        let args = Args::try_parse_from([
            "generate-roomeq-data",
            "--solver",
            "fast-hybrid",
            "--shifted-laplacian",
            "csl",
        ])
        .unwrap();
        assert_eq!(SolverKind::parse(&args.solver), Ok(SolverKind::FastHybrid));
        assert_eq!(
            args.shifted_laplacian,
            fem_runner::ShiftedLaplacianShift::Csl {
                alpha: 1.0,
                beta: 0.5,
            }
        );
    }

    #[test]
    fn dense_max_freq_requires_dense() {
        let err =
            Args::try_parse_from(["generate-roomeq-data", "--dense-max-freq", "300"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn dense_backend_cuda_parses() {
        let args =
            Args::try_parse_from(["generate-roomeq-data", "--dense", "--dense-backend", "cuda"])
                .unwrap();
        assert_eq!(
            DenseBackend::parse(&args.dense_backend),
            Ok(DenseBackend::Cuda)
        );
    }

    #[test]
    fn dense_backend_requires_dense() {
        let err =
            Args::try_parse_from(["generate-roomeq-data", "--dense-backend", "cuda"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn dense_backend_error_lists_cuda() {
        let err = DenseBackend::parse("bogus").unwrap_err();
        assert!(err.contains("cpu, metal, cuda"));
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn metal_report_requires_the_metal_feature_without_starting_a_solver() {
        let simulation = continuity_test_simulation(vec![100.0]);
        let error = run_fem_metal_report(
            &simulation,
            &fem_runner::FemMeshConfig::default(),
            "f64-outer",
            "auto",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires building with `--features metal`"));
    }

    fn continuity_test_simulation(frequencies: Vec<f64>) -> RoomSimulation {
        let room = RectangularRoom::new(1.6, 1.3, 1.1);
        let source = Source {
            position: Point3D::new(0.32, 0.28, 0.45),
            orientation: Point3D::new(0.0, 1.0, 0.0),
            directivity: Directivity::Classical {
                h_angle: 360.0,
                v_angle: 360.0,
            },
            amplitude: 1.0,
            crossover: CrossoverFilter::FullRange,
            name: "test-source".to_string(),
        };
        let mut simulation = RoomSimulation::new(
            RoomGeometry::Rectangular(room),
            vec![source],
            vec![Point3D::new(1.07, 0.91, 0.62)],
        );
        simulation.frequencies = frequencies;
        simulation.boundaries = BoundaryConfig {
            floor: SurfaceConfig::Absorption { coefficient: 0.45 },
            ceiling: SurfaceConfig::Absorption { coefficient: 0.35 },
            walls: SurfaceConfig::Absorption { coefficient: 0.30 },
            front_wall: Some(SurfaceConfig::Absorption { coefficient: 0.25 }),
            back_wall: Some(SurfaceConfig::Absorption { coefficient: 0.40 }),
            left_wall: None,
            right_wall: Some(SurfaceConfig::Absorption { coefficient: 0.50 }),
        };
        simulation
    }

    #[test]
    fn dense_fem_splice_continuity_lossy_rectangular_room() {
        let dense_freq = 75.0;
        let fem_freq = 85.0;
        let dense_sim = continuity_test_simulation(vec![dense_freq]);
        let fem_sim = continuity_test_simulation(vec![fem_freq]);

        let dense = run_dense_blr(
            &dense_sim,
            DenseBlrOptions {
                max_freq: dense_freq,
                overlap_hz: 0.0,
                freq_concurrency: 1,
                tbem_threads: 1,
                #[cfg(feature = "cuda")]
                gpu_device: 0,
                backend: DenseBackend::Cpu,
            },
        )
        .expect("dense BLR solve failed");
        let fem = fem_runner::run_fem_with_mesh(
            &fem_sim,
            &fem_runner::FemMeshConfig {
                preconditioner: fem_runner::FemPreconditionerKind::Ilut,
                recycle_subspace_size: 0,
                ..fem_runner::FemMeshConfig::default()
            },
        )
        .expect("FEM solve failed");

        let dense_p = dense.pressures[0][0][0].norm().max(1.0e-15);
        let fem_p = fem.pressures[0][0][0].norm().max(1.0e-15);
        let jump_db = 20.0 * (dense_p / fem_p).abs().log10().abs();

        // Coarse P1 FEM and surface TBEM are not identical discretizations.
        // This regression guards against a gross rigid-wall dense band being
        // spliced into the lossy FEM band.
        assert!(
            jump_db < 12.0,
            "dense/FEM splice jump too large: {jump_db:.2} dB"
        );
    }
}
