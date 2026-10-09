//! Sweep benchmark for the affine-PML continuation PR (d91d2402).
//!
//! Times three ways of running a 20-point log-spaced 40-1000 Hz sweep on the
//! default resolved axisymmetric acoustic model (uniform diaphragm drive):
//!
//!   (a) `cold`  — post-PR code, pre-PR usage: a fresh
//!       `ResolvedFrequencyContinuation` per frequency, so every frequency
//!       gets a cold initial guess and a fresh ILU factorisation. Assembly
//!       still goes through the prebaked affine assembler.
//!   (b) `warm`  — post-PR intended usage: one continuation threaded through
//!       the whole sweep (warm start + ILU reuse inside
//!       `preconditioner_reuse_band`).
//!   (c) `reasm` — the pre-PR assembly path reproduced with public APIs:
//!       full `assemble_bilinear_by_region` re-assembly per frequency
//!       (wavenumber-scaled PML form) plus the same cold FGMRES+ILU solve as
//!       arm (a). This isolates the prebaked-vs-reassembly assembly delta;
//!       the solve-side knobs match arm (a).
//!
//! All arms solve the same discrete system to the same tolerance; the
//! benchmark asserts that far-field pressures from (a) and (b) and the raw
//! pressure vectors from (b) and (c) agree to 1e-6 relative, so the timing
//! comparison is apples-to-apples.
//!
//! Mesh note: the default `ResolvedAcousticConfig` mesh is 204 nodes, which
//! is under `direct_solve_max_nodes` (1000) and therefore takes the direct
//! path where warm start and ILU reuse do nothing. This benchmark refines
//! the mesh (default 10 cells/interval → ~4.3k nodes; override with the
//! `BENCH_CELLS` env var) so the sweep runs on the iterative FGMRES+ILU
//! path the PR accelerates.

use num_complex::Complex64;
use sonium_fem::assembly::{RegionMap, assemble_bilinear_by_region};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{FemSystem, apply_homogeneous_dirichlet_complex};
use sonium_fem::physics::acoustics::pressure::resolved::{
    AxisymmetricPmlHelmholtzForm, add_axisymmetric_neumann, add_axisymmetric_robin,
    positive_iomega_axisymmetric_exterior_pml,
};
use sonium_fem::physics::acoustics::pressure::thermal_wall_admittance;
use sonium_fem::spaces::H1Space;
use sonium_sim_speaker_modeling::resolved_acoustics::{
    ACOUSTIC_PML, CAVITY_WALL, FRONT_DIAPHRAGM, OUTER_PML_TRUNCATION, PHYSICAL_EXTERIOR, PORT_WALL,
    REAR_CAVITY, REAR_DIAPHRAGM, RESOLVED_PORT,
};
use sonium_sim_speaker_modeling::{
    ResolvedAcousticConfig, ResolvedAcousticModel, ResolvedAcousticSolution,
    ResolvedFrequencyContinuation,
};
use sonium_solvers::{FgmresConfig, IluPreconditioner, fgmres_preconditioned};
use std::f64::consts::PI;
use std::time::Instant;

const FREQ_POINTS: usize = 20;
const FREQ_START_HZ: f64 = 40.0;
const FREQ_END_HZ: f64 = 1_000.0;
/// Uniform diaphragm drive, RMS m/s (front = rear, zero-thickness diaphragm).
const DRIVE_VELOCITY_RMS: f64 = 1.0e-3;
/// Relative agreement gate between arms.
const MAX_RELATIVE_DISAGREEMENT: f64 = 1.0e-6;
/// Mesh refinement needed to land above `direct_solve_max_nodes` (1000) so
/// the sweep exercises the iterative path (see header). Override with the
/// `BENCH_CELLS` env var.
const DEFAULT_CELLS: usize = 10;

fn mesh_cells() -> usize {
    std::env::var("BENCH_CELLS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_CELLS)
}

fn sweep_frequencies() -> Vec<f64> {
    let ratio = (FREQ_END_HZ / FREQ_START_HZ).ln() / (FREQ_POINTS - 1) as f64;
    (0..FREQ_POINTS)
        .map(|i| FREQ_START_HZ * (ratio * i as f64).exp())
        .collect()
}

struct ArmResult {
    wall_seconds: f64,
    iterations: Vec<usize>,
    preconditioner_rebuilds: usize,
    solutions: Vec<ResolvedAcousticSolution>,
    pressures: Vec<Vec<Complex64>>,
}

/// Arm (a): fresh continuation per frequency — cold guess + fresh ILU.
fn run_cold_arm(model: &ResolvedAcousticModel, frequencies: &[f64]) -> ArmResult {
    let drive = Complex64::new(DRIVE_VELOCITY_RMS, 0.0);
    let t0 = Instant::now();
    let mut iterations = Vec::with_capacity(frequencies.len());
    let mut rebuilds = 0;
    let mut solutions = Vec::with_capacity(frequencies.len());
    let mut pressures = Vec::with_capacity(frequencies.len());
    for &frequency in frequencies {
        let mut continuation = ResolvedFrequencyContinuation::new(model);
        let solution = continuation
            .solve_frequency_with_drive(frequency, drive)
            .expect("cold-arm solve failed");
        iterations.push(solution.solve_diagnostics.iterations);
        rebuilds += usize::from(solution.solve_diagnostics.preconditioner_rebuilt);
        pressures.push(solution.pressure_rms_pa.clone());
        solutions.push(solution);
    }
    ArmResult {
        wall_seconds: t0.elapsed().as_secs_f64(),
        iterations,
        preconditioner_rebuilds: rebuilds,
        solutions,
        pressures,
    }
}

/// Arm (b): one continuation threaded through the sweep — warm start + ILU
/// reuse (the post-PR default usage in `whole-speaker`).
fn run_warm_arm(model: &ResolvedAcousticModel, frequencies: &[f64]) -> ArmResult {
    let drive = Complex64::new(DRIVE_VELOCITY_RMS, 0.0);
    let t0 = Instant::now();
    let mut iterations = Vec::with_capacity(frequencies.len());
    let mut rebuilds = 0;
    let mut solutions = Vec::with_capacity(frequencies.len());
    let mut pressures = Vec::with_capacity(frequencies.len());
    let mut continuation = ResolvedFrequencyContinuation::new(model);
    for &frequency in frequencies {
        let solution = continuation
            .solve_frequency_with_drive(frequency, drive)
            .expect("warm-arm solve failed");
        iterations.push(solution.solve_diagnostics.iterations);
        rebuilds += usize::from(solution.solve_diagnostics.preconditioner_rebuilt);
        pressures.push(solution.pressure_rms_pa.clone());
        solutions.push(solution);
    }
    ArmResult {
        wall_seconds: t0.elapsed().as_secs_f64(),
        iterations,
        preconditioner_rebuilds: rebuilds,
        solutions,
        pressures,
    }
}

/// Arm (c): pre-PR full re-assembly per frequency (wavenumber-scaled PML
/// form through `assemble_bilinear_by_region`) + cold FGMRES with a fresh
/// ILU per frequency. Mirrors the pre-PR `solve_frequency` body; uses the
/// same FGMRES parameters as the production `solve_pressure`.
fn run_reassembly_arm(model: &ResolvedAcousticModel, frequencies: &[f64]) -> ArmResult {
    let config = &model.config;
    let drive = Complex64::new(DRIVE_VELOCITY_RMS, 0.0);
    let space = H1Space::new(PolynomialDegree::P1);
    let solver = FgmresConfig {
        max_iterations: 20,
        restart: 50,
        tolerance: 1.0e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
    };
    let t0 = Instant::now();
    let mut iterations = Vec::with_capacity(frequencies.len());
    let mut pressures = Vec::with_capacity(frequencies.len());
    for &frequency in frequencies {
        let omega = 2.0 * PI * frequency;
        let wavenumber = omega / config.fluid.speed_of_sound;
        // Pre-PR: the PML stretch was rebuilt at each frequency's wavenumber
        // and the full element loop ran through the generic assembler.
        let pml = positive_iomega_axisymmetric_exterior_pml(
            config.radial_pml_start_m,
            config.radial_pml_thickness_m,
            config.upper_pml_start_m,
            config.upper_pml_thickness_m,
            config.lower_pml_start_m,
            config.lower_pml_thickness_m,
            wavenumber,
            config.pml_sigma_max,
        );
        let form = AxisymmetricPmlHelmholtzForm::new(Complex64::new(wavenumber, 0.0), pml);
        let mut forms = RegionMap::new();
        for marker in [REAR_CAVITY, RESOLVED_PORT, PHYSICAL_EXTERIOR, ACOUSTIC_PML] {
            forms.insert(marker, form.clone());
        }
        let mut matrix = assemble_bilinear_by_region(&forms, &space, &space, &model.mesh, 4)
            .expect("reassembly failed");
        let port_alpha = thermal_wall_admittance(&config.fluid, omega).conj();
        add_axisymmetric_robin(&mut matrix, &model.mesh, PORT_WALL, port_alpha);
        let cavity_alpha = config
            .cavity_wall_specific_impedance_pa_s_per_m
            .map(|impedance| Complex64::new(0.0, omega * config.fluid.density) / impedance);
        if let Some(alpha) = cavity_alpha {
            add_axisymmetric_robin(&mut matrix, &model.mesh, CAVITY_WALL, alpha);
        }
        let mut rhs = vec![Complex64::default(); model.mesh.num_nodes()];
        let i_omega_rho = Complex64::new(0.0, omega * config.fluid.density);
        add_axisymmetric_neumann(&mut rhs, &model.mesh, FRONT_DIAPHRAGM, i_omega_rho * drive);
        add_axisymmetric_neumann(&mut rhs, &model.mesh, REAR_DIAPHRAGM, -i_omega_rho * drive);
        let mut system = FemSystem::new(matrix, rhs);
        apply_homogeneous_dirichlet_complex(&mut system, &model.mesh, &space, OUTER_PML_TRUNCATION);
        let csr = system.matrix.to_csr();
        let preconditioner = IluPreconditioner::from_csr(&csr);
        let solved = fgmres_preconditioned(&csr, &preconditioner, &system.rhs, None, &solver);
        assert!(
            solved.converged,
            "reassembly-arm FGMRES did not converge at {frequency} Hz"
        );
        iterations.push(solved.iterations);
        pressures.push(solved.x);
    }
    ArmResult {
        wall_seconds: t0.elapsed().as_secs_f64(),
        iterations,
        preconditioner_rebuilds: frequencies.len(),
        solutions: Vec::new(),
        pressures,
    }
}

fn relative_pressure_error(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len(), "pressure vector lengths differ");
    let diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let scale = b.iter().map(|y| y.norm_sqr()).sum::<f64>().sqrt();
    diff / scale.max(1e-30)
}

fn main() {
    let frequencies = sweep_frequencies();
    println!("=== Resolved-acoustics sweep benchmark (PR d91d2402) ===");
    println!(
        "Sweep: {FREQ_POINTS} log-spaced frequencies {FREQ_START_HZ:.0}-{FREQ_END_HZ:.0} Hz, \
         uniform diaphragm drive {DRIVE_VELOCITY_RMS} m/s RMS"
    );

    let cells = mesh_cells();
    let mut config = ResolvedAcousticConfig {
        cells_per_physical_interval: cells,
        pml_cells: cells,
        ..ResolvedAcousticConfig::default()
    };
    // Same PML sizing rule as the whole-speaker production-QA runner.
    config
        .configure_pml_for_sweep(FREQ_START_HZ, 0.01)
        .expect("PML configuration failed");
    let t0 = Instant::now();
    let model = ResolvedAcousticModel::build(config).expect("model build failed");
    let build_seconds = t0.elapsed().as_secs_f64();
    let n_nodes = model.mesh.num_nodes();
    println!(
        "Model: {n_nodes} nodes ({cells} cells/interval), direct_solve_max_nodes = {} (built in {build_seconds:.2}s)",
        model.config.direct_solve_max_nodes
    );
    if n_nodes <= model.config.direct_solve_max_nodes {
        eprintln!("WARNING: mesh takes the direct-solve path; ILU reuse and warm start do nothing");
    }

    let cold = run_cold_arm(&model, &frequencies);
    println!(
        "(a) cold  : {:>7.2}s ({:.2}s/freq), {} ILU builds, iters/freq = {:?}",
        cold.wall_seconds,
        cold.wall_seconds / FREQ_POINTS as f64,
        cold.preconditioner_rebuilds,
        cold.iterations
    );
    let warm = run_warm_arm(&model, &frequencies);
    println!(
        "(b) warm  : {:>7.2}s ({:.2}s/freq), {} ILU builds, iters/freq = {:?}",
        warm.wall_seconds,
        warm.wall_seconds / FREQ_POINTS as f64,
        warm.preconditioner_rebuilds,
        warm.iterations
    );
    let reasm = run_reassembly_arm(&model, &frequencies);
    println!(
        "(c) reasm : {:>7.2}s ({:.2}s/freq), {} ILU builds, iters/freq = {:?}",
        reasm.wall_seconds,
        reasm.wall_seconds / FREQ_POINTS as f64,
        reasm.preconditioner_rebuilds,
        reasm.iterations
    );

    // Correctness gates: arms must produce the same answers.
    let max_far_field_rel = cold
        .solutions
        .iter()
        .zip(&warm.solutions)
        .map(|(a, b)| {
            (a.far_field_pressure_rms_pa - b.far_field_pressure_rms_pa).norm()
                / b.far_field_pressure_rms_pa.norm().max(1e-30)
        })
        .fold(0.0_f64, f64::max);
    let max_reasm_rel = reasm
        .pressures
        .iter()
        .zip(&warm.pressures)
        .map(|(a, b)| relative_pressure_error(a, b))
        .fold(0.0_f64, f64::max);
    println!(
        "Parity: max |p_ff,a - p_ff,b|/|p_ff,b| = {max_far_field_rel:.3e}, \
         max ||p_c - p_b||/||p_b|| = {max_reasm_rel:.3e}"
    );
    if max_far_field_rel > MAX_RELATIVE_DISAGREEMENT || max_reasm_rel > MAX_RELATIVE_DISAGREEMENT {
        eprintln!(
            "FAILED: arms disagree beyond {MAX_RELATIVE_DISAGREEMENT:.1e} relative — \
             the timing comparison would not be apples-to-apples"
        );
        std::process::exit(1);
    }

    println!(
        "Speedup warm-vs-cold (b vs a):      {:>5.2}×",
        cold.wall_seconds / warm.wall_seconds
    );
    println!(
        "Speedup warm-vs-reassembly (b vs c): {:>5.2}×",
        reasm.wall_seconds / warm.wall_seconds
    );
    println!("\nSUCCESS");
}
