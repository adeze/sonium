//! Pre/post benchmark for the sweep-acceleration PR (commit 971c1990):
//! the OLD solver configuration vs the NEW `SolverConfig` defaults on the
//! same speaker frequency sweep.
//!
//! OLD (pre-PR behaviour, reconstructed with current knobs):
//!   - `parallel_frequency_sweep: false` (serial sweep),
//!   - `recycle_subspace_size: 0` (plain FGMRES, no GCRO-DR subspace),
//!   - `max_recycled_guess_octaves: 0.0` — this single gate controls both
//!     the warm-start initial guess *and* cross-frequency ILU reuse, so
//!     setting it to 0 is the only way to disable the ILU reuse the PR
//!     introduced. Caveat: the true pre-PR defaults kept the warm-start
//!     guess (0.25 octaves) while rebuilding the ILU every frequency;
//!     because one knob now drives both, OLD here is strictly colder than
//!     the real pre-PR default and the measured speedup is an upper bound
//!     on the parallel + GCRO-DR + ILU-reuse gain.
//!   - `verify_exact_hmatrix_residual: true` (exact O(N^2) matvec after
//!     every H-matrix solve; only exercised on the H-matrix path, i.e. the
//!     HEAVY tier — the LIGHT tier stays on the dense path).
//!
//! NEW: `SolverConfig::default()` as shipped by the PR —
//!   `parallel_frequency_sweep: true`, `recycle_subspace_size: 20`,
//!   ILU reuse + warm start within `max_recycled_guess_octaves = 0.25`,
//!   `verify_exact_hmatrix_residual: false`.
//!
//! Both passes share:
//!   - identical mesh, identical operator (dense BEM with Burton-Miller),
//!   - identical FGMRES tolerance / restart / ILU shift,
//!   - `direct_threshold: 0` on both passes so every frequency takes the
//!     iterative path — otherwise small-`f` solves route through dense LU
//!     and none of the PR mechanisms engage.
//!
//! Correctness gate: the two sweeps must agree to 1e-3 dB (SPL) and
//! 1e-4 rad (wrapped phase) on the shared output grid, and neither may
//! contain non-finite values. The speedup is recorded, and gated only
//! against a material slow-down floor (see `main`).
//!
//! Sizing knobs:
//!   - **Default (CI tier)**: `max_mesh_frequency = 800 Hz` → ~500 DOFs,
//!     `freq_points = 8` log-spaced 200-800 Hz. Sized so both passes
//!     finish in a couple minutes total. Note the 8-point log spacing is
//!     ~0.29 octaves/step, just over the 0.25-octave reuse window, so at
//!     this tier the warm start / ILU reuse / GCRO-DR subspace reset every
//!     frequency and the NEW pass mainly measures the parallel sweep.
//!   - **`SWEEP_FREQ_POINTS=N`**: override the light-tier point count
//!     (e.g. 20 gives ~0.1 octaves/step, inside the reuse window).
//!   - **`SWEEP_MAX_MESH_FREQ=HZ`**: override the light-tier mesh cap
//!     (e.g. 2000 gives ~3k DOFs, still on the dense path).
//!   - **`SWEEP_HEAVY=1`**: `max_mesh_frequency = 4 kHz` → ~10k DOFs,
//!     `freq_points = 100` log-spaced 200-4000 Hz, H-matrix path. Each
//!     pass is multi-hour at this size; gate behind the env var.

use sonium_sim_speaker::model::{
    CabinetDimensions, DriverConfig, LoudspeakerModel, SeparatorConfig, SimulationConfig,
    SpeakerMaterialAssignment, WaveguideConfig,
};
use sonium_sim_speaker::simulation::{
    SimulationResult, SolverConfig, try_run_simulation_with_config,
};
use std::time::Instant;

const FREQ_POINTS_LIGHT: usize = 8;
const FREQ_POINTS_HEAVY: usize = 100;
const FREQ_START: f64 = 200.0;
const FREQ_END_LIGHT: f64 = 800.0;
const FREQ_END_HEAVY: f64 = 4_000.0;
const MAX_MESH_FREQ_LIGHT: f64 = 800.0;
const MAX_MESH_FREQ_HEAVY: f64 = 4_000.0;

fn is_heavy() -> bool {
    std::env::var("SWEEP_HEAVY").as_deref() == Ok("1")
}

fn freq_points() -> usize {
    if is_heavy() {
        return FREQ_POINTS_HEAVY;
    }
    std::env::var("SWEEP_FREQ_POINTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(FREQ_POINTS_LIGHT)
}

fn freq_end() -> f64 {
    if is_heavy() {
        FREQ_END_HEAVY
    } else {
        FREQ_END_LIGHT
    }
}

fn max_mesh_freq() -> f64 {
    if is_heavy() {
        return MAX_MESH_FREQ_HEAVY;
    }
    std::env::var("SWEEP_MAX_MESH_FREQ")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(MAX_MESH_FREQ_LIGHT)
}

fn build_model() -> LoudspeakerModel {
    LoudspeakerModel {
        cabinet: CabinetDimensions {
            height: 0.40,
            width: 0.22,
            depth: 0.28,
        },
        woofer: DriverConfig {
            radius: 0.065,
            x_offset: 0.0,
            y_offset: -0.08,
            piston_velocity: 0.001,
            transfer_function: None,
            driver_model: None,
            cone: None,
        },
        tweeter: DriverConfig {
            radius: 0.013,
            x_offset: 0.0,
            y_offset: 0.10,
            piston_velocity: 0.001,
            transfer_function: None,
            driver_model: None,
            cone: None,
        },
        separator: SeparatorConfig {
            y_position: 0.02,
            thickness: 0.018,
        },
        waveguide: Some(WaveguideConfig {
            horizontal_spline: vec![[0.0, 0.05], [0.33, 0.04], [0.66, 0.025], [1.0, 0.013]],
            vertical_spline: vec![[0.0, 0.04], [0.33, 0.03], [0.66, 0.02], [1.0, 0.013]],
            depth: 0.025,
        }),
        materials: SpeakerMaterialAssignment::default(),
        simulation: SimulationConfig {
            freq_start: FREQ_START,
            freq_end: freq_end(),
            freq_points: freq_points(),
            speed_of_sound: 343.0,
            air_density: 1.21,
            max_mesh_frequency: Some(max_mesh_freq()),
        },
        additional_drivers: Vec::new(),
        product_design: None,
    }
}

/// Pre-PR (971c1990^) solver behaviour, reconstructed with current knobs.
/// See the file header for the warm-start caveat on
/// `max_recycled_guess_octaves`.
fn old_solver_config() -> SolverConfig {
    SolverConfig {
        // Harness knob shared by both passes: force the iterative path on
        // every frequency so the solver mechanisms are what is measured.
        direct_threshold: 0,
        parallel_frequency_sweep: false,
        recycle_subspace_size: 0,
        max_recycled_guess_octaves: 0.0,
        verify_exact_hmatrix_residual: true,
        ..SolverConfig::default()
    }
}

/// Post-PR shipped defaults (`SolverConfig::default()`), with only the
/// harness-level `direct_threshold` override.
fn new_solver_config() -> SolverConfig {
    SolverConfig {
        direct_threshold: 0,
        ..SolverConfig::default()
    }
}

fn run_pass(label: &str, config: &SolverConfig) -> Result<(f64, SimulationResult), String> {
    let model = build_model();
    // On-axis evaluation point — content of the SPL doesn't matter for the
    // benchmark, but the BEM scattered-field eval cost should match between
    // the two passes.
    let eval = [[0.0_f64, 0.0_f64, 1.0_f64]];
    let n_freq = model.simulation.frequencies().len();
    eprintln!("== {label} ({config:?}) ==");
    let t0 = Instant::now();
    let result = try_run_simulation_with_config(&model, &eval, config)
        .map_err(|e| format!("{label}: simulation failed: {e:?}"))?;
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "== {label}: {n_freq} freqs in {dt:.2}s ({:.2}s/freq) ==\n",
        dt / n_freq as f64
    );
    Ok((dt, result))
}

fn validate_pass_parity(
    old: &SimulationResult,
    new: &SimulationResult,
) -> Result<(f64, f64), String> {
    if old.frequencies != new.frequencies
        || old.spl.shape() != new.spl.shape()
        || old.phase.shape() != new.phase.shape()
    {
        return Err("old and new sweeps produced different output grids".into());
    }
    let old_spl = old.spl.as_slice();
    let new_spl = new.spl.as_slice();
    let old_phase = old.phase.as_slice();
    let new_phase = new.phase.as_slice();
    if old_spl
        .iter()
        .chain(new_spl)
        .chain(old_phase)
        .chain(new_phase)
        .any(|value| !value.is_finite())
    {
        return Err("old or new sweep produced non-finite output".into());
    }

    let max_spl_error = old_spl
        .iter()
        .zip(new_spl)
        .map(|(old, new)| (old - new).abs())
        .fold(0.0_f64, f64::max);
    let max_phase_error = old_phase
        .iter()
        .zip(new_phase)
        .map(|(old, new)| {
            let delta = old - new;
            delta.sin().atan2(delta.cos()).abs()
        })
        .fold(0.0_f64, f64::max);

    const MAX_SPL_ERROR_DB: f64 = 1.0e-3;
    const MAX_PHASE_ERROR_RAD: f64 = 1.0e-4;
    if max_spl_error > MAX_SPL_ERROR_DB || max_phase_error > MAX_PHASE_ERROR_RAD {
        return Err(format!(
            "new sweep differs from old-configuration solve: max SPL error={max_spl_error:.3e} dB (limit {MAX_SPL_ERROR_DB:.1e}), max wrapped phase error={max_phase_error:.3e} rad (limit {MAX_PHASE_ERROR_RAD:.1e})"
        ));
    }
    Ok((max_spl_error, max_phase_error))
}

fn main() {
    println!("=== Frequency-sweep benchmark: pre-PR config vs post-PR defaults ===");
    println!("Reference PR: 971c1990 (parallel sweep + GCRO-DR + ILU reuse).");
    let mode = if is_heavy() {
        "HEAVY (100 freq, ~10k DOFs, H-matrix path, multi-hour)"
    } else {
        "LIGHT (default CI tier: ~500 DOFs, dense path)"
    };
    println!("Mode: {mode}");
    println!(
        "{} log-spaced frequencies, {:.0}-{:.0} Hz, mesh capped at {:.0} Hz.",
        freq_points(),
        FREQ_START,
        freq_end(),
        max_mesh_freq(),
    );
    println!("Both passes share mesh, operator, FGMRES tolerance/restart, ILU shift.\n");

    let old = match run_pass(
        "OLD (serial, FGMRES, no reuse, verify on)",
        &old_solver_config(),
    ) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    };
    let new = match run_pass(
        "NEW (post-PR defaults: parallel, GCRO-DR k=20, reuse, verify off)",
        &new_solver_config(),
    ) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    };

    let (max_spl_error, max_phase_error) = match validate_pass_parity(&old.1, &new.1) {
        Ok(errors) => errors,
        Err(error) => {
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    };
    let n_freq = old.1.frequencies.len();
    let speedup = old.0 / new.0;
    println!(
        "OLD (serial, FGMRES, no reuse):  {:>7.2}s ({:.2}s / freq)",
        old.0,
        old.0 / n_freq as f64
    );
    println!(
        "NEW (post-PR defaults):          {:>7.2}s ({:.2}s / freq)",
        new.0,
        new.0 / n_freq as f64
    );
    println!("Speedup (old / new):             {:>7.2}×", speedup);
    println!(
        "Numerical parity: max SPL error={max_spl_error:.3e} dB, max phase error={max_phase_error:.3e} rad"
    );

    // Acceptance gate: the parity check above is the correctness gate; the
    // speedup is recorded for the operator log and gated only against a
    // material slow-down, which would indicate a real regression in the new
    // code paths (GCRO-DR projection overhead is documented in Parks–de
    // Sturler as O(k·n) extra work on top of GMRES). The light tier permits
    // timing noise; the heavy tier must show an actual speedup.
    let minimum_speedup = if is_heavy() { 1.0 } else { 0.8 };
    if !speedup.is_finite() || speedup < minimum_speedup {
        eprintln!(
            "FAILED: post-PR speedup {speedup:.2}× is below the {minimum_speedup:.2}× acceptance floor \
             (old {:.2}s vs new {:.2}s). \
             That suggests a real regression in the sweep-acceleration paths.",
            old.0, new.0
        );
        std::process::exit(1);
    }

    println!("\nSUCCESS");
}
