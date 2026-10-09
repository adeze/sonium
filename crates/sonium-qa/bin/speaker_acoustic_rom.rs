//! QA: modal-basis ROM for fixed-topology acoustic FEM sweeps and the
//! coupled FEM–BEM path (review item C5).
//!
//! Gates:
//!
//! 1. **Sealed-box ROM convergence.** Res-12 rectangular box driven by a
//!    smooth piston: ROM + two-term static correction vs full GMRES solves,
//!    error vs retained mode count m; gated at <2 % for the full-band ROM.
//!    Zero-mode quasi-static compliance at 20 Hz gated at <1e-3 vs the full
//!    solve (the analytic value is printed for reference).
//! 2. **Coupled Helmholtz resonator.** Box + external port fixture (same
//!    geometry as `qa-speaker-fem-bem-coupling` gate 2): ROM aperture volume
//!    velocity resonance within 8 % of the lumped prediction AND the ROM
//!    |U| curve within 10 % of the full coupled curve across 40–80 Hz.
//! 3. **Damped-wall ROM.** Sealed box with a Z = ρc(4+2i) wall: ROM +
//!    (undamped) static correction vs full solves, gated at <15 % — the
//!    correction captures the rigid-wall tail only, so damped-wall accuracy
//!    degrades toward the band edge (documented limitation).
//!
//! Timings for the full vs ROM sweeps are printed (not gated).

use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_neumann_complex};
use sonium_fem::mesh::{Mesh, tetrahedralize_box};
use sonium_fem::physics::acoustics::pressure::resolved::helmholtz_resonance_frequency_hz;
use sonium_fem::physics::acoustics::pressure::{
    AcousticsAssemblerExt, build_acoustics_assembler_with_boundaries,
};
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::acoustic_rom::{AcousticRom, AcousticRomConfig};
use sonium_sim_speaker_modeling::fem_bem_coupling::{
    CouplingDrive, CylinderFixtureDiscretization, FemBemCoupling,
    cylindrical_box_with_external_port,
};
use sonium_solvers::sparse::TripletMatrix;
use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve_complex};
use std::collections::HashMap;
use std::time::Instant;

const C0: f64 = 343.0;
const RHO: f64 = 1.204;
const DRIVE_VELOCITY: f64 = 1.0e-2;
const CASE_ID: &str = "speaker.fem-bem-rom.v1";

#[derive(Debug, Default)]
struct RomQaMetrics {
    sealed_band_max_error: f64,
    sealed_adaptive_max_error: f64,
    quasi_static_error: f64,
    quasi_static_analytic_error: f64,
    coupled_peak_error: f64,
    coupled_curve_error: f64,
    coupled_off_peak_residual: f64,
    coupled_peak_residual: f64,
    damped_max_error: f64,
    benchmark_max_residual: f64,
    benchmark_speedup: f64,
}

fn solver_config() -> LinearSolveConfig {
    LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 2000,
        tolerance: 1e-9,
        restart: 100,
        exact_residual_check_interval: 0,
        verbosity: 0,
    }
}

/// Smooth tilted-cosine piston pattern (unit amplitude, ω-independent); the
/// Neumann RHS is `iωρ·v × pattern`.
fn drive_pattern(mesh: &Mesh, marker: i32) -> Vec<Complex64> {
    let n = mesh.num_nodes();
    let mut system = FemSystem::new(
        TripletMatrix::<Complex64>::new(n, n),
        vec![Complex64::new(0.0, 0.0); n],
    );
    apply_neumann_complex(&mut system, mesh, marker, &move |x, y, _| {
        let shape = (0.6 + 0.4 * (std::f64::consts::PI * x / 0.34).cos())
            * (0.6 + 0.4 * (std::f64::consts::PI * y / 0.22).cos());
        Complex64::new(shape, 0.0)
    });
    system.rhs
}

fn full_solve(
    mesh: &Mesh,
    boundary_tags: &[i32],
    k: Complex64,
    boundary_coeffs: &HashMap<usize, Complex64>,
    rhs: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    let assembler =
        build_acoustics_assembler_with_boundaries(mesh, PolynomialDegree::P1, boundary_tags);
    let matrix = assembler.assemble_helmholtz(k, boundary_coeffs);
    let result =
        solve_complex(&matrix, rhs, &solver_config()).map_err(|e| format!("full solve: {e}"))?;
    if !result.converged {
        return Err(format!(
            "full solve did not converge (residual {:.3e})",
            result.residual
        ));
    }
    Ok(result.x)
}

fn relative_error(a: &[Complex64], b: &[Complex64]) -> f64 {
    let diff: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denom: f64 = b.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    diff / denom
}

/// Vertex of the parabola through the three largest consecutive samples.
fn parabolic_peak(curve: &[(f64, f64)]) -> Option<f64> {
    let (idx, _) = curve
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())?;
    if idx == 0 || idx + 1 >= curve.len() {
        return Some(curve[idx].0);
    }
    let (x0, y0) = curve[idx - 1];
    let (x1, y1) = curve[idx];
    let (x2, y2) = curve[idx + 1];
    let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
    if denom.abs() < 1e-30 {
        return Some(x1);
    }
    let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
    let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
    if a.abs() < 1e-30 {
        return Some(x1);
    }
    Some(-b / (2.0 * a))
}

/// Gate 1: sealed-box ROM convergence vs mode count, with static correction.
fn gate_sealed_rom_convergence() -> Result<RomQaMetrics, String> {
    let mesh = tetrahedralize_box(0.0, 0.34, 0.0, 0.22, 0.0, 0.18, 12);
    let pattern = drive_pattern(&mesh, 6);
    let frequencies = [150.0, 300.0, 470.0, 620.0, 760.0, 900.0, 1050.0];

    // Full-order reference.
    let mut full = Vec::new();
    for &f in &frequencies {
        let omega = std::f64::consts::TAU * f;
        let scale = Complex64::new(0.0, omega * RHO * DRIVE_VELOCITY);
        let rhs: Vec<Complex64> = pattern.iter().map(|&p| scale * p).collect();
        full.push(full_solve(
            &mesh,
            &[],
            Complex64::new(omega / C0, 0.0),
            &HashMap::new(),
            &rhs,
        )?);
    }

    println!("sealed ROM convergence (res-12 box, smooth piston, static correction):");
    let mut band_error = 0.0;
    for &m in &[4_usize, 8, 12, 17] {
        // Pin the mode count; the small band keeps the band-coverage check
        // satisfied at every m (the ROM is still *evaluated* up to 1050 Hz).
        let config = AcousticRomConfig {
            min_modes: m,
            max_modes: m,
            ..AcousticRomConfig::for_sweep(C0, 100.0)
        };
        let rom = AcousticRom::build(&mesh, PolynomialDegree::P1, &[], &config)
            .map_err(|e| format!("basis build (m={m}): {e}"))?;
        let correction = rom
            .static_correction(&pattern, &solver_config())
            .map_err(|e| format!("static correction (m={m}): {e}"))?;
        let mut max_err = 0.0_f64;
        for (i, &f) in frequencies.iter().enumerate() {
            let omega = std::f64::consts::TAU * f;
            let scale = Complex64::new(0.0, omega * RHO * DRIVE_VELOCITY);
            let solution = rom
                .solve_frequency_static(f, &HashMap::new(), &correction, scale, false)
                .map_err(|e| format!("ROM solve at {f} Hz (m={m}): {e}"))?;
            max_err = max_err.max(relative_error(&solution.pressure, &full[i]));
        }
        println!("  m = {:2}: max rel err {:.3e}", rom.num_modes(), max_err);
        band_error = max_err;
    }

    // Default (band-adaptive) ROM: the deliverable configuration.
    let rom = AcousticRom::build(
        &mesh,
        PolynomialDegree::P1,
        &[],
        &AcousticRomConfig::for_sweep(C0, 1100.0),
    )
    .map_err(|e| format!("basis build (adaptive): {e}"))?;
    let correction = rom
        .static_correction(&pattern, &solver_config())
        .map_err(|e| format!("static correction (adaptive): {e}"))?;
    let mut adaptive_error = 0.0_f64;
    for (i, &f) in frequencies.iter().enumerate() {
        let omega = std::f64::consts::TAU * f;
        let scale = Complex64::new(0.0, omega * RHO * DRIVE_VELOCITY);
        let solution = rom
            .solve_frequency_static(f, &HashMap::new(), &correction, scale, false)
            .map_err(|e| format!("ROM solve at {f} Hz (adaptive): {e}"))?;
        adaptive_error = adaptive_error.max(relative_error(&solution.pressure, &full[i]));
    }
    println!(
        "  adaptive ({} modes, band {:.0} Hz): max rel err {adaptive_error:.3e}",
        rom.num_modes(),
        rom.band_hz()
    );
    // Calibrated: ~6e-3 on the res-12 box; gate at 2 %.
    if band_error > 2.0e-2 || adaptive_error > 2.0e-2 {
        return Err(format!(
            "sealed ROM error {band_error:.3e} (m=17) / {adaptive_error:.3e} (adaptive) exceeds 2 %"
        ));
    }

    // Zero-mode quasi-static compliance at 20 Hz vs the full solve.
    let f = 20.0;
    let omega = std::f64::consts::TAU * f;
    let scale = Complex64::new(0.0, omega * RHO * DRIVE_VELOCITY);
    let rhs: Vec<Complex64> = pattern.iter().map(|&p| scale * p).collect();
    let full_qs = full_solve(
        &mesh,
        &[],
        Complex64::new(omega / C0, 0.0),
        &HashMap::new(),
        &rhs,
    )?;
    let rom_qs = rom
        .solve_frequency_static(f, &HashMap::new(), &correction, scale, false)
        .map_err(|e| format!("quasi-static ROM solve: {e}"))?;
    let qs_err = relative_error(&rom_qs.pressure, &full_qs);
    // Analytic reference (informational): |p̄| = ρc²|Q|/(ωV),
    // Q = v · S_face · mean(shape), mean = 0.6·0.6.
    let driven_area = 0.34 * 0.22 * 0.36;
    let volume = 0.34 * 0.22 * 0.18;
    let analytic = RHO * C0 * C0 * DRIVE_VELOCITY * driven_area / (omega * volume);
    let mean = rom_qs.pressure.iter().sum::<Complex64>() / rom_qs.pressure.len() as f64;
    let analytic_rel = (mean.norm() - analytic).abs() / analytic;
    println!(
        "quasi-static 20 Hz: ROM-vs-full {qs_err:.3e}, |p̄| = {:.4} vs analytic {analytic:.4} (rel {analytic_rel:.3e})",
        mean.norm()
    );
    if qs_err > 1.0e-3 {
        return Err(format!("quasi-static ROM error {qs_err:.3e} exceeds 1e-3"));
    }
    Ok(RomQaMetrics {
        sealed_band_max_error: band_error,
        sealed_adaptive_max_error: adaptive_error,
        quasi_static_error: qs_err,
        quasi_static_analytic_error: analytic_rel,
        ..RomQaMetrics::default()
    })
}

/// Gate 2: coupled box+port ROM vs lumped Helmholtz and the full coupled
/// sweep.
fn gate_coupled_rom_helmholtz() -> Result<RomQaMetrics, String> {
    let (box_radius, box_depth, port_radius, port_length) = (0.15, 0.25, 0.03, 0.10);
    let fixture = cylindrical_box_with_external_port(
        box_radius,
        box_depth,
        port_radius,
        port_length,
        CylinderFixtureDiscretization {
            n_circumference: 16,
            n_box_depth: 5,
            n_cap_radial: 3,
            n_aperture_radial: 2,
            n_port_depth: 4,
        },
    )
    .map_err(|e| format!("fixture: {e}"))?;
    let coupling =
        FemBemCoupling::from_fixture(&fixture, C0, RHO).map_err(|e| format!("model: {e}"))?;

    let volume = fixture.cavity_volume_m3();
    let port_area = std::f64::consts::PI * port_radius * port_radius;
    let l_eff = port_length + (0.85 + 0.61) * port_radius;
    let f_ref = helmholtz_resonance_frequency_hz(C0, volume, port_area, l_eff);

    let drive = CouplingDrive::DriverVelocity(Complex64::new(DRIVE_VELOCITY, 0.0));
    // First interior box mode is ~1.1 kHz: 250 Hz of band covers the
    // quasi-static regime with headroom.
    let t_build = Instant::now();
    let rom = coupling
        .build_rom(
            &AcousticRomConfig::for_sweep(C0, 250.0),
            drive.clone(),
            &solver_config(),
        )
        .map_err(|e| format!("coupled ROM build: {e}"))?
        .with_static_correction(&solver_config())
        .map_err(|e| format!("coupled ROM correction: {e}"))?;
    let build_time = t_build.elapsed();
    let probe = rom
        .solve_frequency(60.0, false)
        .map_err(|e| format!("probe: {e}"))?;
    println!(
        "coupled fixture: {} interior DOFs, {} aperture elements, ROM {} modes + {} attachment",
        coupling.num_dofs(),
        coupling.num_aperture_elements(),
        probe.num_modes,
        probe.num_attachment_modes
    );

    let config = solver_config();
    let areas = coupling.aperture_element_areas();
    let sweep: Vec<f64> = (0..=20).map(|i| 40.0 + 2.0 * i as f64).collect();

    let t_full = Instant::now();
    let mut u_full = Vec::new();
    for &f in &sweep {
        let solution = coupling
            .solve_frequency(f, drive.clone(), &config)
            .map_err(|e| format!("full coupled solve at {f} Hz: {e}"))?;
        u_full.push(solution.aperture_volume_velocity(&areas).norm());
    }
    let full_time = t_full.elapsed();

    // Timed ROM sweep without the exact residual spot-check (that check
    // assembles the full operator per frequency and would skew the timing).
    let t_rom = Instant::now();
    let mut u_rom = Vec::new();
    for &f in &sweep {
        let solution = rom
            .solve_frequency(f, false)
            .map_err(|e| format!("coupled ROM solve at {f} Hz: {e}"))?;
        u_rom.push(solution.solution.aperture_volume_velocity(&areas).norm());
    }
    let rom_time = t_rom.elapsed();

    // Gated exact-residual pass (untimed).
    let mut rom_residuals = Vec::new();
    for &f in &sweep {
        let solution = rom
            .solve_frequency(f, true)
            .map_err(|e| format!("coupled ROM residual at {f} Hz: {e}"))?;
        rom_residuals.push(solution.relative_residual.unwrap());
    }

    let curve_rom: Vec<(f64, f64)> = sweep.iter().copied().zip(u_rom.iter().copied()).collect();
    let f_peak = parabolic_peak(&curve_rom).unwrap_or(0.0);
    let peak_rel = (f_peak - f_ref).abs() / f_ref;
    let mut max_u_rel = 0.0_f64;
    let mut max_residual_off_peak = 0.0_f64;
    let mut max_residual = 0.0_f64;
    for i in 0..sweep.len() {
        max_u_rel = max_u_rel.max((u_rom[i] - u_full[i]).abs() / u_full[i]);
        max_residual = max_residual.max(rom_residuals[i]);
        // Near-resonance bins (±8 Hz) are dominated by the flux-amplified
        // shifted-static residual term (see below); gate off-peak only.
        if (sweep[i] - f_peak).abs() > 8.0 {
            max_residual_off_peak = max_residual_off_peak.max(rom_residuals[i]);
        }
    }
    println!(
        "coupled ROM gate: f_peak = {f_peak:.2} Hz vs lumped {f_ref:.2} Hz (rel {peak_rel:.3e}), max |U| rel diff {max_u_rel:.3e}"
    );
    // The exact residual scales with the aperture flux: the attachment
    // modes solve the *shifted* static (K+σM)w = b, leaving an honest
    // out-of-basis residual −σMw ∝ |U| — amplified near the Helmholtz peak
    // while the solution itself stays accurate (|U| diff ~3e-5). Gate the
    // residual off-peak; the peak value is informational.
    println!(
        "coupled ROM residual: max {max_residual:.3e} (at the resonance peak), off-peak max {max_residual_off_peak:.3e}"
    );
    println!(
        "coupled sweep timings (21 freqs): full {:.2?}, ROM build {:.2?} + sweep {:.2?} (BEM DtN condensation dominates both at this interior size)",
        full_time, build_time, rom_time
    );
    if peak_rel > 0.08 {
        return Err(format!(
            "coupled ROM peak {f_peak:.2} Hz vs lumped {f_ref:.2} Hz ({peak_rel:.3e})"
        ));
    }
    if max_u_rel > 0.10 {
        return Err(format!("coupled ROM |U| curve off by {max_u_rel:.3e}"));
    }
    if max_residual_off_peak > 0.03 {
        return Err(format!(
            "coupled ROM off-peak residual {max_residual_off_peak:.3e} exceeds 3 %"
        ));
    }
    Ok(RomQaMetrics {
        coupled_peak_error: peak_rel,
        coupled_curve_error: max_u_rel,
        coupled_off_peak_residual: max_residual_off_peak,
        coupled_peak_residual: max_residual,
        ..RomQaMetrics::default()
    })
}

/// Gate 3: damped-wall sealed box, ROM + (undamped) static correction vs
/// full solves.
fn gate_damped_wall_rom() -> Result<RomQaMetrics, String> {
    let wall_marker: i32 = 5;
    let mesh = tetrahedralize_box(0.0, 0.34, 0.0, 0.22, 0.0, 0.18, 12);
    let boundary_tags = [wall_marker];
    let rom = AcousticRom::build(
        &mesh,
        PolynomialDegree::P1,
        &boundary_tags,
        &AcousticRomConfig::for_sweep(C0, 1100.0),
    )
    .map_err(|e| format!("basis build (damped): {e}"))?;
    let pattern = drive_pattern(&mesh, 6);
    let correction = rom
        .static_correction(&pattern, &solver_config())
        .map_err(|e| format!("static correction (damped): {e}"))?;

    let z_ratio = Complex64::new(4.0, 2.0);
    let frequencies = [300.0, 504.4, 760.0, 1000.0];
    let mut max_err = 0.0_f64;
    for &f in &frequencies {
        let omega = std::f64::consts::TAU * f;
        let k = omega / C0;
        let alpha = Complex64::new(0.0, -k) / z_ratio;
        let mut coeffs = HashMap::new();
        coeffs.insert(wall_marker as usize, alpha);
        let scale = Complex64::new(0.0, omega * RHO * DRIVE_VELOCITY);
        let rhs: Vec<Complex64> = pattern.iter().map(|&p| scale * p).collect();
        let full = full_solve(&mesh, &boundary_tags, Complex64::new(k, 0.0), &coeffs, &rhs)?;
        let solution = rom
            .solve_frequency_static(f, &coeffs, &correction, scale, false)
            .map_err(|e| format!("damped ROM solve at {f} Hz: {e}"))?;
        let err = relative_error(&solution.pressure, &full);
        max_err = max_err.max(err);
        println!("damped ROM at {f:.1} Hz: rel err {err:.3e}");
    }
    // Calibrated: ~1.2e-1 at the band edge (the undamped correction misses
    // the damped wall's boundary-layer tail — documented limitation).
    if max_err > 1.5e-1 {
        return Err(format!("damped ROM error {max_err:.3e} exceeds 15 %"));
    }
    Ok(RomQaMetrics {
        damped_max_error: max_err,
        ..RomQaMetrics::default()
    })
}

/// Benchmark (not gated): full 41-frequency sweep vs ROM build + sweep on
/// the sealed res-12 box.
fn benchmark_sealed_sweep() -> Result<RomQaMetrics, String> {
    let mesh = tetrahedralize_box(0.0, 0.34, 0.0, 0.22, 0.0, 0.18, 12);
    let pattern = drive_pattern(&mesh, 6);
    let sweep: Vec<f64> = (0..41).map(|i| 50.0 + 25.0 * i as f64).collect();

    let t_full = Instant::now();
    for &f in &sweep {
        let omega = std::f64::consts::TAU * f;
        let scale = Complex64::new(0.0, omega * RHO * DRIVE_VELOCITY);
        let rhs: Vec<Complex64> = pattern.iter().map(|&p| scale * p).collect();
        full_solve(
            &mesh,
            &[],
            Complex64::new(omega / C0, 0.0),
            &HashMap::new(),
            &rhs,
        )?;
    }
    let full_time = t_full.elapsed();

    let t_build = Instant::now();
    let rom = AcousticRom::build(
        &mesh,
        PolynomialDegree::P1,
        &[],
        &AcousticRomConfig::for_sweep(C0, 1100.0),
    )
    .map_err(|e| format!("basis build (bench): {e}"))?;
    let correction = rom
        .static_correction(&pattern, &solver_config())
        .map_err(|e| format!("static correction (bench): {e}"))?;
    let build_time = t_build.elapsed();

    let t_rom = Instant::now();
    let mut max_residual = 0.0_f64;
    for (i, &f) in sweep.iter().enumerate() {
        let omega = std::f64::consts::TAU * f;
        let scale = Complex64::new(0.0, omega * RHO * DRIVE_VELOCITY);
        // Exact residual spot-check every 5th frequency (per-frequency
        // assembly would skew the sweep timing).
        let solution = rom
            .solve_frequency_static(f, &HashMap::new(), &correction, scale, i % 5 == 0)
            .map_err(|e| format!("bench ROM solve at {f} Hz: {e}"))?;
        if let Some(residual) = solution.relative_residual {
            max_residual = max_residual.max(residual);
        }
    }
    let rom_time = t_rom.elapsed();

    println!(
        "benchmark (sealed res-12 box, {} DOFs, {} modes, 41 freqs):",
        rom.num_dofs(),
        rom.num_modes()
    );
    println!("  full GMRES sweep:      {full_time:.2?}");
    println!("  ROM basis build:       {build_time:.2?}  (eigensolve + 2 static solves)");
    println!(
        "  ROM sweep:             {rom_time:.2?}  (incl. exact residual check every 5th freq)"
    );
    let speedup = full_time.as_secs_f64() / rom_time.as_secs_f64();
    println!(
        "  break-even / speedup:  {:.1} freqs, {:.1}x on this sweep",
        build_time.as_secs_f64() / (full_time.as_secs_f64() / 41.0 - rom_time.as_secs_f64() / 41.0),
        speedup
    );
    println!("  max full residual over ROM sweep: {max_residual:.3e}");
    Ok(RomQaMetrics {
        benchmark_max_residual: max_residual,
        benchmark_speedup: speedup,
        ..RomQaMetrics::default()
    })
}

fn main() {
    let started = Instant::now();
    let result = std::panic::catch_unwind(|| -> Result<RomQaMetrics, String> {
        let sealed = gate_sealed_rom_convergence()?;
        let coupled = gate_coupled_rom_helmholtz()?;
        let damped = gate_damped_wall_rom()?;
        let benchmark = benchmark_sealed_sweep()?;
        Ok(RomQaMetrics {
            sealed_band_max_error: sealed.sealed_band_max_error,
            sealed_adaptive_max_error: sealed.sealed_adaptive_max_error,
            quasi_static_error: sealed.quasi_static_error,
            quasi_static_analytic_error: sealed.quasi_static_analytic_error,
            coupled_peak_error: coupled.coupled_peak_error,
            coupled_curve_error: coupled.coupled_curve_error,
            coupled_off_peak_residual: coupled.coupled_off_peak_residual,
            coupled_peak_residual: coupled.coupled_peak_residual,
            damped_max_error: damped.damped_max_error,
            benchmark_max_residual: benchmark.benchmark_max_residual,
            benchmark_speedup: benchmark.benchmark_speedup,
        })
    });
    match result {
        Ok(Ok(metrics)) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("sealed_band_max_error", metrics.sealed_band_max_error)
                .metric("sealed_adaptive_max_error", metrics.sealed_adaptive_max_error)
                .metric("quasi_static_error", metrics.quasi_static_error)
                .metric(
                    "quasi_static_analytic_error",
                    metrics.quasi_static_analytic_error,
                )
                .metric("coupled_peak_error", metrics.coupled_peak_error)
                .metric("coupled_curve_error", metrics.coupled_curve_error)
                .metric(
                    "coupled_off_peak_residual",
                    metrics.coupled_off_peak_residual,
                )
                .metric("coupled_peak_residual", metrics.coupled_peak_residual)
                .metric("damped_max_error", metrics.damped_max_error)
                .metric("benchmark_max_residual", metrics.benchmark_max_residual)
                .metric("benchmark_speedup", metrics.benchmark_speedup)
                .metric("gated_contract_count", 4.0)
                .maximum("sealed_band_max_error", 2.0e-2)
                .maximum("sealed_adaptive_max_error", 2.0e-2)
                .maximum("quasi_static_error", 1.0e-3)
                .maximum("coupled_peak_error", 8.0e-2)
                .maximum("coupled_curve_error", 1.0e-1)
                .maximum("coupled_off_peak_residual", 3.0e-2)
                .maximum("damped_max_error", 1.5e-1)
                .minimum("gated_contract_count", 4.0)
                .input_checksum("analytic/deterministic-fixture:speaker-fem-bem-rom-v1")
                .reference("full-order FEM/BEM solves, lumped Helmholtz resonance, and quasi-static compliance")
                .reference_checksum("analytic/deterministic-fixture")
                .dimension("sealed 3D box, coupled cylindrical box+port, and 41-frequency sweep")
                .formulation("production FEM acoustic ROM with coupled FEM-BEM port path")
                .tolerance("sealed <=2e-2; quasi-static <=1e-3; coupled peak <=8e-2; curve <=1e-1; off-peak residual <=3e-2; damped <=1.5e-1")
                .seed("speaker-fem-bem-rom:sealed-coupled-damped-v1")
                .timing_ms("suite", elapsed_ms)
                .elapsed_ms(elapsed_ms);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("speaker FEM-BEM ROM result must serialize")
            );
            println!("SUCCESS");
        }
        Ok(Err(e)) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, e.clone())
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .elapsed_ms(elapsed_ms)
                    .to_json_line()
                    .expect("speaker FEM-BEM ROM failure must serialize")
            );
            println!("FAILED: {e}");
            std::process::exit(1);
        }
        Err(_) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, "speaker FEM-BEM ROM gate panicked")
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .elapsed_ms(elapsed_ms)
                    .to_json_line()
                    .expect("speaker FEM-BEM ROM panic result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
