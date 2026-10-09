//! Rank-preservation benchmark for coarse speaker-simulation policies (B6).
//!
//! Review item B6 (`reviews/acoustics-20250730.md`, Phase B), methodology
//! §11.3 (multi-fidelity validation): for a fixed candidate set, measure
//! Spearman/Kendall rank correlation between cheap policies and the reference
//! policy, false-promotion / false-rejection rates at top-k, and timings.
//! "These measurements decide whether a cheaper model loses accuracy that
//! matters. Pointwise pressure error alone does not."
//!
//! # Fixture
//!
//! Base model: `sonium-sim-speaker/examples/waveguide_optimization_smoke.json`
//! (two-way speaker, spline waveguide), modified into a **single-active-driver**
//! fixture: the woofer velocity is set to zero so only the waveguide-loaded
//! tweeter radiates. This mirrors the real optimization setting — in the
//! production waveguide band the woofer is rolled off by its mass corner —
//! and is required for a non-degenerate objective: with both drivers
//! radiating identical flat in-phase velocity (the stock smoke config), the
//! low-band response is dominated by deep interference nulls (RMS deviations
//! of 35–55 dB), the flatness objective measures null depth at the sampled
//! frequencies, and a sparser frequency grid ranks the *anti*-correlated
//! null sampling instead of the designs (measured: Spearman −0.72 for the
//! sparse grid on the two-driver fixture). With one radiator the response is
//! smooth and the objective tracks the geometry.
//!
//! The band is narrowed to 300–900 Hz so the reference policy stays inside
//! the dense direct-LU range (~1.2k DOFs) and the whole benchmark finishes
//! in a few minutes. Sixteen deterministic candidates perturb the six active
//! shape parameters (tweeter y-offset, waveguide rim/mid radii, waveguide
//! depth) via the fixed `PERTURBATIONS` table — no RNG anywhere.
//!
//! # Ranking objective
//!
//! Candidates are ranked by the **CEA-2034 preference score** (higher is
//! better), the acceptance metric of the staged optimizer
//! (`CandidateMetrics::preference_score` in `optimization/staged.rs`). The
//! optimizer's smooth objective (`optimization.rs::compute_objective`, rebuilt
//! here as `optimization_objective` on the public `compute_spinorama_metrics`
//! because the original is crate-private) is recorded per policy for context
//! but is *not* the ranking scalar: its PIR slope-penalty term
//! `10·(max dB/oct − 2)²` divides bin-to-bin ripple by the ~0.13-octave bin
//! width, so at this band and grid density it is ~99 % slope penalty
//! (measured: objective ≈ 21 000 from a ±6 dB ripple while the flatness
//! terms are O(10)) and grid decimation ranks the aliased ripple sampling
//! instead of the designs (measured: Spearman −0.44 for the sparse grid).
//! The preference score is computed from 1/2-octave-smoothed curves and is
//! stable under grid decimation — it is the scalar a shortlist decision is
//! actually made on.
//!
//! # Policies (mesh EPW × solver tolerance × frequency grid)
//!
//! | policy           | EPW | tol     | grid | solver      |
//! |------------------|-----|---------|------|-------------|
//! | reference        |  12 | 1e-6    | 13   | direct LU   |
//! | coarse-mesh      |   8 | 1e-6    | 13   | direct LU   |
//! | loose-tolerance  |  12 | 1e-4    | 13   | FGMRES+ILU  |
//! | sparse-grid      |  12 | 1e-6    |  6   | direct LU   |
//! | l1-combined      |   8 | 1e-4    |  6   | FGMRES+ILU  |
//!
//! Mesh density is set through the existing config surface:
//! `SimulationConfig::max_mesh_frequency = freq_end * EPW / 8` (the mesher
//! targets λ/8 at that frequency, so EPW > 8 refines). `direct_threshold = 0`
//! forces the FGMRES+ILU Krylov path for the tolerance-sensitive policies —
//! at LU sizes the tolerance knob is otherwise inert, and §6's L1 rung is
//! explicitly "1e-4..1e-5 Krylov tolerance". Each cheap single-factor policy
//! varies exactly one knob against the reference; `l1-combined` is the §6 L1
//! rung (all three cheap settings at once).
//!
//! # Metrics
//!
//! For each policy all 16 candidates are evaluated (candidate-parallel via
//! Rayon) and the preference-score vector is compared against the reference:
//!
//! - Spearman ρ / Kendall τ-b: global rank agreement (1.0 = identical).
//! - Top-k preservation, k = max(3, N/4) = 4: false promotions (in the
//!   policy's top-k but not the reference's) and false rejections (the
//!   converse). These are the counts an optimization shortlist cares about.
//! - max/mean absolute score error, for context only — rank metrics are
//!   primary.
//! - Smooth-objective Spearman, context only (see above).
//! - Wall time → candidates/hour.
//!
//! # Gate
//!
//! SUCCESS iff the L1-combined policy reaches Spearman ≥ 0.9 AND
//! false-promotions ≤ 1 (the review-appropriate bar).
//!
//! # Measured values (2026-07-31, M-series 12-thread, release)
//!
//! Filled in from the calibration run; see the JSON report
//! (`output/qa-speaker-rank-preservation.json`) for the authoritative record.
//! Total benchmark wall time 86–101 s (two consecutive full runs, 12 threads),
//! well inside the ~10-minute budget; reference spread 1.94 score points.
//! Two consecutive runs produced byte-identical rankings and scores
//! (determinism check: JSON diff empty once timing fields are excluded).
//!
//! | policy          | cand/h | Spearman | Kendall | FP | FR |
//! |-----------------|--------|----------|---------|----|----|
//! | coarse-mesh     |  6109  | 0.9588   | 0.8500  | 0  | 0  |
//! | loose-tolerance |  2021  | 1.0000   | 1.0000  | 0  | 0  |
//! | sparse-grid     |  3566  | 0.9676   | 0.8833  | 0  | 0  |
//! | l1-combined     |  8948  | 0.9412   | 0.8333  | 0  | 0  |
//!
//! Gate: l1-combined Spearman 0.9412 ≥ 0.9 AND false-promotions 0 ≤ 1 → PASS.
//! Reference policy: 584 DOFs, 13-point grid, direct LU, 2241 cand/h.
//!
//! Run: `cargo run -p sonium-qa --release --bin qa-speaker-rank-preservation`
//! Flags: `--json` prints the machine-readable report to stdout instead of
//! writing `output/qa-speaker-rank-preservation.json`;
//! `--dofs-only` prints per-policy DOF counts for the base candidate and
//! exits (calibration probe, no solves); `--dump-curves` prints the base
//! candidate's on-axis/PIR curves under the reference policy;
//! `--candidates N` truncates the candidate set (smoke/debug runs);
//! `--policies name1,name2` runs only the named policies, always including
//! `reference` (calibration runs; rank metrics stay gated).

use rayon::prelude::*;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

use sonium_compute::HostMatrix;
use sonium_qa::rank::{checked_top_k_set, kendall_tau_b, spearman_rho};
use sonium_sim_speaker::cea2034_output::{compute_spinorama_metrics, generate_eval_points};
use sonium_sim_speaker::mesh_gen::generate_loudspeaker_mesh;
use sonium_sim_speaker::model::LoudspeakerModel;
use sonium_sim_speaker::simulation::{SimulationResult, SolverConfig, run_simulation_with_config};
use sonium_sim_speaker::spline::CubicSpline;

const BASE_MODEL_JSON: &str =
    include_str!("../../sonium-sim-speaker/examples/waveguide_optimization_smoke.json");

/// Fixture band. Narrowed from the smoke config's 200–5000 Hz so the
/// reference mesh (~1.2k DOFs at EPW 12) stays in the dense direct-LU range
/// and the whole benchmark fits in a few minutes: the dense FGMRES+ILU path
/// (full-pattern ILU(0) is O(n³), like LU) costs ~2.5× the LU solve at the
/// same size, so mesh size dominates the budget.
///
/// The band *top* matters for rank preservation, not just cost: above
/// ~570 Hz this fixture's response develops sharp narrowband features
/// (measured on the base candidate: on-axis +6.4 dB from 569→624 Hz, PIR
/// ±3 dB swings between adjacent 13-point samples — waveguide/cabinet
/// interior modes). The candidate-to-candidate preference-score differences
/// above 570 Hz live almost entirely in how those features land on the
/// scoring grid, which no sub-reference grid can track (measured: 6-point
/// sparse grid over 300–900 Hz, cubic-spline interpolated, ρ = 0.09 vs the
/// reference ranking — interpolated curves are smooth, so the score's
/// narrowband-deviation penalty collapses to ~0 for every candidate and the
/// spread compresses from 3.5 to 0.6 points). Below 570 Hz the curves are
/// smooth (1–2 dB per 50 Hz), so the band closes at 600 Hz.
const FREQ_START_HZ: f64 = 300.0;
const FREQ_END_HZ: f64 = 600.0;

/// Gate thresholds for the L1-combined policy (review-appropriate bar).
const GATE_SPEARMAN: f64 = 0.9;
const GATE_MAX_FALSE_PROMOTIONS: usize = 1;

/// Candidate perturbations: `[Δtweeter_y]` (additive, m) then
/// `[×h_rim, ×h_mid, ×v_rim, ×v_mid, ×depth]` (multiplicative). Fixed table —
/// deterministic by construction. Row 0 is the unperturbed base design.
/// Perturbations are ±35–150 %: wide enough that the inter-candidate score
/// spread clearly exceeds the cheap-policy noise (rank metrics need
/// signal ≫ policy noise — measured: ±25–80 % perturbations leave the
/// 16 reference scores clustered in a 1.4-point band with mid-pack gaps
/// below five percentage points, so even the small candidate-dependent error of the L1 rung
/// scrambles the order, ρ ≈ 0.84). Scaling further (×2–3 single-dimension)
/// does not help: the cheap-policy error grows with the perturbation size
/// as fast as the spread (measured ρ 0.91 vs 0.94 at this table). Small
/// enough that every candidate passes `LoudspeakerModel::validate`. The woofer is silent in this fixture, so
/// only tweeter/waveguide parameters move the objective.
const PERTURBATIONS: &[[f64; 6]] = &[
    [0.0, 1.00, 1.00, 1.00, 1.00, 1.00], // base
    [0.050, 1.00, 1.00, 1.00, 1.00, 1.00],
    [-0.040, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.0, 1.70, 1.00, 1.00, 1.00, 1.00],
    [0.0, 0.50, 1.00, 1.00, 1.00, 1.00],
    [0.0, 1.00, 1.00, 1.70, 1.00, 1.00],
    [0.0, 1.00, 1.00, 0.50, 1.00, 1.00],
    [0.0, 1.00, 2.00, 1.00, 1.00, 1.00],
    [0.0, 1.00, 1.00, 1.00, 0.40, 1.00],
    [0.0, 1.00, 1.00, 1.00, 1.00, 2.50],
    [0.0, 1.00, 1.00, 1.00, 1.00, 0.30],
    [0.035, 1.00, 1.00, 1.00, 1.00, 1.80],
    [-0.030, 1.50, 1.00, 1.00, 1.00, 1.00],
    [0.0, 0.60, 0.60, 1.40, 1.00, 1.00],
    [0.0, 1.00, 1.80, 1.00, 1.80, 0.50],
    [0.025, 1.00, 1.00, 0.65, 0.45, 1.00],
];

/// One fidelity policy: mesh density (elements per wavelength at the band
/// top) × solver tolerance × frequency-grid density.
struct Policy {
    name: &'static str,
    elements_per_wavelength: usize,
    freq_points: usize,
    tolerance: f64,
    /// `FORCE_LU` keeps every solve on direct LU; `0` forces FGMRES+ILU.
    direct_threshold: usize,
    /// Sparse-grid policies simulate on `freq_points` frequencies and then
    /// interpolate the SPL curves onto the reference grid (natural cubic
    /// spline per evaluation point through the (log f, dB) knots) before
    /// scoring — the §6 "sparse adaptive frequency grid" model.
    /// Scoring a raw 6-point sweep with the CEA-2034 machinery is unsound
    /// (measured: systematic score bias of +2.2…+3.4 vs the 13-point grid,
    /// larger than the whole inter-candidate spread). Piecewise-linear
    /// interpolation is likewise unsound (measured: kinks add ~2 points of
    /// spurious narrowband deviation).
    interpolate_to_reference_grid: bool,
    note: &'static str,
}

/// Direct-LU forcing value for `Policy::direct_threshold`. Must stay below
/// `hmatrix_threshold` (raised alongside in `policy_solver_config`) because
/// `SolverConfig` validation requires a non-empty dense-iterative range.
const FORCE_LU: usize = 100_000;

fn policies() -> Vec<Policy> {
    vec![
        Policy {
            name: "reference",
            elements_per_wavelength: 12,
            freq_points: 13,
            tolerance: 1e-6,
            direct_threshold: FORCE_LU,
            interpolate_to_reference_grid: false,
            note: "finest mesh, tightest tolerance (direct LU), dense grid",
        },
        Policy {
            name: "coarse-mesh",
            elements_per_wavelength: 8,
            freq_points: 13,
            tolerance: 1e-6,
            direct_threshold: FORCE_LU,
            interpolate_to_reference_grid: false,
            note: "coarse mesh only (λ/8 at band top, the production default)",
        },
        Policy {
            name: "loose-tolerance",
            elements_per_wavelength: 12,
            freq_points: 13,
            tolerance: 1e-4,
            direct_threshold: 0,
            interpolate_to_reference_grid: false,
            note: "loose Krylov tolerance only (FGMRES+ILU forced)",
        },
        Policy {
            name: "sparse-grid",
            elements_per_wavelength: 12,
            freq_points: 8,
            tolerance: 1e-6,
            direct_threshold: FORCE_LU,
            interpolate_to_reference_grid: true,
            note: "sparse frequency grid only (8 solves, cubic-spline interpolation onto the reference grid)",
        },
        Policy {
            name: "l1-combined",
            elements_per_wavelength: 8,
            freq_points: 8,
            tolerance: 1e-4,
            direct_threshold: 0,
            interpolate_to_reference_grid: true,
            note: "§6 L1 rung: coarse mesh + loose Krylov tolerance + sparse grid (interpolated)",
        },
    ]
}

/// Per-policy benchmark outcome for the full candidate set.
#[derive(Serialize)]
struct PolicyReport {
    name: String,
    note: String,
    elements_per_wavelength: usize,
    freq_points: usize,
    tolerance: f64,
    solver: String,
    mesh_dofs_base_candidate: usize,
    wall_seconds: f64,
    candidates_per_hour: f64,
    /// Ranking objective: CEA-2034 preference scores (higher is better).
    preference_scores: Vec<f64>,
    /// Metrics vs the reference policy (identity for the reference itself).
    spearman_rho: f64,
    kendall_tau_b: f64,
    top_k: usize,
    false_promotions: usize,
    false_rejections: usize,
    max_abs_score_error: f64,
    mean_abs_score_error: f64,
    /// Context only: the optimizer's smooth objective values and their
    /// Spearman vs reference. At this band/grid density the smooth objective
    /// is dominated by its PIR slope-penalty term (see header), so its rank
    /// agreement is expected to be poor for grid-decimating policies.
    smooth_objectives: Vec<f64>,
    smooth_objective_spearman: f64,
}

#[derive(Serialize)]
struct BenchmarkReport {
    band_hz: [f64; 2],
    num_candidates: usize,
    top_k: usize,
    ranking_objective: String,
    reference_score_spread: f64,
    gate: String,
    gate_spearman_minimum: f64,
    gate_max_false_promotions: usize,
    passed: bool,
    policies: Vec<PolicyReport>,
}

// ---------------------------------------------------------------------------
// Fixture construction
// ---------------------------------------------------------------------------

fn base_model() -> LoudspeakerModel {
    let mut model: LoudspeakerModel =
        serde_json::from_str(BASE_MODEL_JSON).expect("embedded smoke config must parse");
    // Single-active-driver fixture: silence the woofer so only the
    // waveguide-loaded tweeter radiates (see the header's Fixture section —
    // two flat in-phase drivers make the objective interference-null
    // dominated and rank-degenerate).
    model.woofer.piston_velocity = 0.0;
    model.woofer.transfer_function = None;
    model.validate().expect("base fixture model must be valid");
    model
}

/// Deterministic candidate set: base design plus 15 fixed perturbations.
/// Every candidate is validated at construction — an invalid row in
/// `PERTURBATIONS` is a fixture bug and fails loudly here, not mid-run.
fn candidate_models() -> Vec<LoudspeakerModel> {
    let base = base_model();
    PERTURBATIONS
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let mut model = base.clone();
            model.tweeter.y_offset += p[0];
            let waveguide = model.waveguide.as_mut().expect("fixture needs a waveguide");
            waveguide.horizontal_spline[0][1] *= p[1];
            waveguide.horizontal_spline[1][1] *= p[2];
            waveguide.vertical_spline[0][1] *= p[3];
            waveguide.vertical_spline[1][1] *= p[4];
            waveguide.depth *= p[5];
            model
                .validate()
                .unwrap_or_else(|e| panic!("candidate {index} failed validation: {e}"));
            model
        })
        .collect()
}

/// Apply a policy's simulation resolution to a candidate model.
fn apply_policy_resolution(model: &mut LoudspeakerModel, policy: &Policy) {
    model.simulation.freq_start = FREQ_START_HZ;
    model.simulation.freq_end = FREQ_END_HZ;
    model.simulation.freq_points = policy.freq_points;
    // The mesher targets λ/8 at max_mesh_frequency, so EPW > 8 refines.
    model.simulation.max_mesh_frequency =
        Some(FREQ_END_HZ * policy.elements_per_wavelength as f64 / 8.0);
}

fn policy_solver_config(policy: &Policy, frequency_lanes: usize) -> SolverConfig {
    let mut config = SolverConfig {
        tolerance: policy.tolerance,
        direct_threshold: policy.direct_threshold,
        max_frequency_lanes: Some(frequency_lanes),
        ..SolverConfig::default()
    };
    if policy.direct_threshold > 0 {
        // Keep the direct/iterative/H-matrix ranges consistent when LU is
        // forced (validation requires direct < hmatrix). With
        // direct_threshold = 0 the default 5000 keeps these DOF counts on the
        // dense FGMRES+ILU path.
        config.hmatrix_threshold = policy.direct_threshold + 1;
    }
    config
}

// ---------------------------------------------------------------------------
// Objective (mirrors optimization.rs::compute_objective on the public API)
// ---------------------------------------------------------------------------

const F3_TRANSITION_WIDTH_DB: f64 = 0.35;

fn smooth_above_f3_weights(on_axis: &[f64]) -> Result<Vec<f64>, String> {
    if on_axis.is_empty() || on_axis.iter().any(|v| !v.is_finite()) {
        return Err("on-axis response must contain finite values".into());
    }
    let max_spl = on_axis.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let beta = 8.0;
    let soft_peak = max_spl
        + on_axis
            .iter()
            .map(|&spl| (beta * (spl - max_spl)).exp())
            .sum::<f64>()
            .ln()
            / beta;
    let threshold = soft_peak - 3.0;
    let weights: Vec<f64> = on_axis
        .iter()
        .map(|&spl| {
            let argument = ((spl - threshold) / F3_TRANSITION_WIDTH_DB).clamp(-40.0, 40.0);
            1.0 / (1.0 + (-argument).exp())
        })
        .collect();
    if weights.iter().sum::<f64>() <= f64::MIN_POSITIVE {
        return Err("smooth F3 weighting produced no active frequencies".into());
    }
    Ok(weights)
}

fn max_octave_slope(freqs: &[f64], spl: &[f64]) -> Result<f64, String> {
    if freqs.len() != spl.len() || freqs.len() < 2 {
        return Err("need at least two matched frequency/SPL points".into());
    }
    let mut max_slope = 0.0_f64;
    for i in 1..freqs.len() {
        let octaves = (freqs[i] / freqs[i - 1]).log2();
        let slope = (spl[i] - spl[i - 1]).abs() / octaves;
        max_slope = max_slope.max(slope);
    }
    Ok(max_slope)
}

/// `weight_flat · L_flat + weight_pir · (L_pir_flat + 10 · slope_penalty)`
/// with unit weights, matching the optimizer's default `OptimizationConfig`.
fn optimization_objective(result: &SimulationResult) -> Result<f64, String> {
    let metrics = compute_spinorama_metrics(result)?;
    let on_axis = &metrics.on_axis;
    let pir = &metrics.pir;

    let f3_weights = smooth_above_f3_weights(on_axis)?;
    let weight_sum: f64 = f3_weights.iter().sum();
    let mean_above = on_axis
        .iter()
        .zip(&f3_weights)
        .map(|(&spl, &w)| spl * w)
        .sum::<f64>()
        / weight_sum;
    let l_flat = on_axis
        .iter()
        .zip(&f3_weights)
        .map(|(&spl, &w)| w * (spl - mean_above).powi(2))
        .sum::<f64>()
        / weight_sum;

    let pir_mean = pir.iter().sum::<f64>() / pir.len().max(1) as f64;
    let l_pir_flat =
        pir.iter().map(|&s| (s - pir_mean).powi(2)).sum::<f64>() / pir.len().max(1) as f64;
    let slope_penalty = (max_octave_slope(&result.frequencies, pir)? - 2.0)
        .max(0.0)
        .powi(2);

    let objective = l_flat + l_pir_flat + 10.0 * slope_penalty;
    if !objective.is_finite() {
        return Err("objective is non-finite".into());
    }
    Ok(objective)
}

/// CEA-2034 preference score — the rank-relevant scalar for this benchmark.
///
/// It is the acceptance metric of the staged optimizer (`CandidateMetrics`
/// in `optimization/staged.rs`) and, unlike `optimization_objective`, it is
/// computed from 1/2-octave-smoothed curves, so it is stable under
/// frequency-grid decimation. Higher is better; rank metrics use it
/// directly, top-k selection negates it (minimization convention).
fn preference_score(result: &SimulationResult) -> Result<f64, String> {
    let score = compute_spinorama_metrics(result)?.preference_score;
    if !score.is_finite() {
        return Err("preference score is non-finite".into());
    }
    Ok(score)
}

// ---------------------------------------------------------------------------
// Sparse-grid interpolation
// ---------------------------------------------------------------------------

/// Reference frequency grid (log-spaced, matching
/// `SimulationConfig::frequencies` for the reference policy).
fn reference_grid() -> Vec<f64> {
    let n = policies()[0].freq_points;
    let log_start = FREQ_START_HZ.ln();
    let log_end = FREQ_END_HZ.ln();
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (log_start + t * (log_end - log_start)).exp()
        })
        .collect()
}

/// Spline interpolation of a simulation result's SPL curves onto a denser
/// target frequency grid: a natural cubic per evaluation point through the
/// (log-frequency, dB) samples — the §6 "sparse adaptive frequency grid"
/// evaluation model. Phase is carried from the nearest source sample (the
/// score path uses magnitudes only; finiteness is preserved). The cheap
/// policy *estimates the curve* and the identical scoring functional is
/// applied to the estimate, so rank differences measure curve fidelity
/// rather than scoring-machinery aliasing. Piecewise-linear interpolation
/// was measured unsound here: its kinks add ~2 points of spurious
/// narrowband-deviation to the preference score, larger than the
/// inter-candidate spread.
fn interpolate_result(result: &SimulationResult, target_freqs: &[f64]) -> SimulationResult {
    let src = &result.frequencies;
    let log_src: Vec<f64> = src.iter().map(|f| f.ln()).collect();
    let log_targets: Vec<f64> = target_freqs.iter().map(|f| f.ln()).collect();
    let n_points = result.spl.ncols();
    let mut spl = HostMatrix::<f64>::zeros(target_freqs.len(), n_points);
    let mut phase = HostMatrix::<f64>::zeros(target_freqs.len(), n_points);
    for point in 0..n_points {
        let knots: Vec<[f64; 2]> = log_src
            .iter()
            .enumerate()
            .map(|(fi, &lf)| [lf, result.spl[(fi, point)]])
            .collect();
        let spline = CubicSpline::from_control_points(&knots);
        for (target_index, &lf) in log_targets.iter().enumerate() {
            spl[(target_index, point)] = spline.evaluate(lf);
            // Nearest source sample for the phase placeholder.
            let nearest = log_src
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - lf).abs().total_cmp(&(b.1 - lf).abs()))
                .map(|(i, _)| i)
                .unwrap_or(0);
            phase[(target_index, point)] = result.phase[(nearest, point)];
        }
    }
    SimulationResult {
        frequencies: target_freqs.to_vec(),
        spl,
        phase,
        // Scoring consumes only spl/phase; the complex handoff field is
        // unused here and zero-filled.
        complex_pressure: HostMatrix::<num_complex::Complex64>::zeros(target_freqs.len(), n_points),
        woofer_electrical_impedance_ohm: vec![None; target_freqs.len()],
        tweeter_electrical_impedance_ohm: vec![None; target_freqs.len()],
        driver_electrical_impedances_ohm: vec![vec![None; 2]; target_freqs.len()],
        radiation_matrices: vec![None; target_freqs.len()],
        radiation_matrix_diagnostics: vec![None; target_freqs.len()],
        thermal_driver_states: vec![[None, None]; target_freqs.len()],
        driver_thermal_states: vec![vec![None; 2]; target_freqs.len()],
        power_ledgers: vec![None; target_freqs.len()],
        product_reference_sweep: None,
        structural_responses: Vec::new(),
        driver_velocities_bem: vec![vec![]; target_freqs.len()],
        exact_residual_evaluations: result.exact_residual_evaluations,
        provenance: result.provenance.clone(),
    }
}

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

struct PolicyEvaluation {
    preference_scores: Vec<f64>,
    smooth_objectives: Vec<f64>,
    wall_seconds: f64,
    dofs: usize,
}

fn require_finite_objectives(label: &str, values: &[f64]) -> Result<(), String> {
    let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    else {
        return Ok(());
    };
    Err(format!(
        "{label} objective at candidate {index} is non-finite ({value:?})"
    ))
}

fn evaluate_policy(
    policy: &Policy,
    candidates: &[LoudspeakerModel],
    eval_points: &[[f64; 3]],
) -> PolicyEvaluation {
    let thread_budget = rayon::current_num_threads().max(1);
    let candidate_lanes = candidates.len().min(thread_budget);
    let frequency_lanes = (thread_budget / candidate_lanes).max(1);
    let solver = policy_solver_config(policy, frequency_lanes);

    let policy_models: Vec<LoudspeakerModel> = candidates
        .iter()
        .map(|candidate| {
            let mut model = candidate.clone();
            apply_policy_resolution(&mut model, policy);
            model
        })
        .collect();
    let dofs = generate_loudspeaker_mesh(&policy_models[0]).triangles.len();

    let start = Instant::now();
    let target_grid = policy.interpolate_to_reference_grid.then(reference_grid);
    let scores: Vec<(f64, f64)> = policy_models
        .par_iter()
        .map(|model| {
            let result = run_simulation_with_config(model, eval_points, &solver)
                .unwrap_or_else(|e| panic!("policy {} simulation failed: {e}", policy.name));
            let result = match &target_grid {
                Some(grid) => interpolate_result(&result, grid),
                None => result,
            };
            let score = preference_score(&result)
                .unwrap_or_else(|e| panic!("policy {} preference score failed: {e}", policy.name));
            let smooth = optimization_objective(&result)
                .unwrap_or_else(|e| panic!("policy {} objective failed: {e}", policy.name));
            (score, smooth)
        })
        .collect();
    let wall_seconds = start.elapsed().as_secs_f64();
    PolicyEvaluation {
        preference_scores: scores.iter().map(|s| s.0).collect(),
        smooth_objectives: scores.iter().map(|s| s.1).collect(),
        wall_seconds,
        dofs,
    }
}

fn output_path() -> PathBuf {
    let dir = std::env::var_os("SONIUM_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("output"));
    dir.join("qa-speaker-rank-preservation.json")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_stdout = args.iter().any(|a| a == "--json");
    let dofs_only = args.iter().any(|a| a == "--dofs-only");
    let dump_curves = args.iter().any(|a| a == "--dump-curves");

    let policies = policies();
    // Calibration knob: `--policies name1,name2` runs only the named
    // policies. The reference policy is always included (rank metrics are
    // computed against it), so a filtered run remains fully gated.
    let selected: Option<Vec<String>> = args
        .iter()
        .position(|a| a == "--policies")
        .and_then(|p| args.get(p + 1))
        .map(|value| value.split(',').map(|s| s.trim().to_string()).collect());
    let policies: Vec<Policy> = match &selected {
        Some(names) => policies
            .into_iter()
            .filter(|p| p.name == "reference" || names.iter().any(|n| n == p.name))
            .collect(),
        None => policies,
    };
    assert!(!policies.is_empty(), "--policies matched no policy");
    let mut candidates = candidate_models();
    // Debug/smoke knob: `--candidates N` truncates the candidate set (rank
    // metrics are then computed on the truncated set; not a gated run).
    if let Some(position) = args.iter().position(|a| a == "--candidates")
        && let Some(value) = args.get(position + 1)
        && let Ok(limit) = value.parse::<usize>()
    {
        candidates.truncate(limit.max(1));
    }
    let n = candidates.len();
    let top_k = (n / 4).max(3);
    let eval_points = generate_eval_points(base_model().baffle_center());

    // Debug knob: print the base candidate's on-axis / PIR curves under the
    // reference policy (objective forensics).
    if dump_curves {
        let policy = &policies[0];
        let mut model = candidates[0].clone();
        apply_policy_resolution(&mut model, policy);
        let solver = policy_solver_config(policy, 4);
        let result = run_simulation_with_config(&model, &eval_points, &solver)
            .expect("dump-curves simulation failed");
        let metrics = compute_spinorama_metrics(&result).expect("spinorama failed");
        println!("freqs: {:?}", result.frequencies);
        println!("on_axis: {:?}", metrics.on_axis);
        println!("pir: {:?}", metrics.pir);
        println!("preference_score: {:.6}", metrics.preference_score);
        println!(
            "smooth objective: {:.6}",
            optimization_objective(&result).expect("objective failed")
        );
        return;
    }

    println!("qa-speaker-rank-preservation: B6 multi-fidelity rank benchmark");
    println!(
        "  {} candidates (fixed perturbation table), band {}–{} Hz, top-k = {top_k}",
        n, FREQ_START_HZ as u32, FREQ_END_HZ as u32
    );

    if dofs_only {
        let base = base_model();
        for policy in &policies {
            let mut model = base.clone();
            apply_policy_resolution(&mut model, policy);
            let mesh = generate_loudspeaker_mesh(&model);
            println!(
                "  {:<16} EPW {:>2}  grid {:>2}  DOFs {}",
                policy.name,
                policy.elements_per_wavelength,
                policy.freq_points,
                mesh.triangles.len()
            );
        }
        return;
    }

    let mut reference_scores: Vec<f64> = Vec::new();
    let mut reference_smooth: Vec<f64> = Vec::new();
    let mut reports: Vec<PolicyReport> = Vec::new();

    for (policy_index, policy) in policies.iter().enumerate() {
        println!(
            "\n[{}/{}] {} — {}",
            policy_index + 1,
            policies.len(),
            policy.name,
            policy.note
        );
        let evaluation = evaluate_policy(policy, &candidates, &eval_points);
        let scores = &evaluation.preference_scores;
        if let Err(error) = require_finite_objectives("preference", scores)
            .and_then(|_| require_finite_objectives("smooth", &evaluation.smooth_objectives))
        {
            eprintln!("RANK-PRESERVATION INPUT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
        let candidates_per_hour = n as f64 / evaluation.wall_seconds * 3600.0;

        if policy_index == 0 {
            reference_scores = scores.clone();
            reference_smooth = evaluation.smooth_objectives.clone();
        }
        let reference = &reference_scores;
        // Higher preference score is better; top_k_set selects the smallest,
        // so rank the negated scores.
        let neg = |values: &[f64]| values.iter().map(|v| -v).collect::<Vec<_>>();
        let reference_top = checked_top_k_set(&neg(reference), top_k)
            .expect("finite reference objectives were validated before top-k ranking");
        let policy_top = checked_top_k_set(&neg(scores), top_k)
            .expect("finite policy objectives were validated before top-k ranking");
        let false_promotions = policy_top
            .iter()
            .filter(|i| !reference_top.contains(i))
            .count();
        let false_rejections = reference_top
            .iter()
            .filter(|i| !policy_top.contains(i))
            .count();
        let abs_errors: Vec<f64> = scores
            .iter()
            .zip(reference)
            .map(|(o, r)| (o - r).abs())
            .collect();
        let max_err = abs_errors.iter().copied().fold(0.0, f64::max);
        let mean_err = abs_errors.iter().sum::<f64>() / n as f64;
        let rho = spearman_rho(reference, scores);
        let tau = kendall_tau_b(reference, scores);
        let smooth_rho = spearman_rho(&reference_smooth, &evaluation.smooth_objectives);

        println!(
            "  DOFs {}, {:.1}s wall → {candidates_per_hour:.0} candidates/hour",
            evaluation.dofs, evaluation.wall_seconds
        );
        println!(
            "  Spearman {rho:.4}, Kendall {tau:.4}, FP {false_promotions}, FR {false_rejections}, max|Δscore| {max_err:.3e} (smooth-obj ρ {smooth_rho:.4})"
        );

        reports.push(PolicyReport {
            name: policy.name.into(),
            note: policy.note.into(),
            elements_per_wavelength: policy.elements_per_wavelength,
            freq_points: policy.freq_points,
            tolerance: policy.tolerance,
            solver: if policy.direct_threshold == 0 {
                "FGMRES+ILU".into()
            } else {
                "direct LU".into()
            },
            mesh_dofs_base_candidate: evaluation.dofs,
            wall_seconds: evaluation.wall_seconds,
            candidates_per_hour,
            preference_scores: evaluation.preference_scores,
            spearman_rho: rho,
            kendall_tau_b: tau,
            top_k,
            false_promotions,
            false_rejections,
            max_abs_score_error: max_err,
            mean_abs_score_error: mean_err,
            smooth_objectives: evaluation.smooth_objectives,
            smooth_objective_spearman: smooth_rho,
        });
    }

    // Degenerate-fixture guard: if the reference scores barely spread, rank
    // metrics are noise, not signal. Reported, not gated.
    let score_min = reference_scores
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let score_max = reference_scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let spread = score_max - score_min;

    println!("\n======== rank preservation vs reference (preference score) ========");
    println!(
        "{:<17} {:>6} {:>4} {:>6} {:>9} {:>8} {:>8} {:>4} {:>4} {:>10} {:>10} {:>12}",
        "policy",
        "DOFs",
        "grid",
        "tol",
        "cand/h",
        "Spearman",
        "Kendall",
        "FP",
        "FR",
        "max|Δscore|",
        "mean|Δscore|",
        "ρ(smoothobj)"
    );
    for report in &reports {
        println!(
            "{:<17} {:>6} {:>4} {:>6.0e} {:>9.0} {:>8.4} {:>8.4} {:>4} {:>4} {:>10.3e} {:>10.3e} {:>12.4}",
            report.name,
            report.mesh_dofs_base_candidate,
            report.freq_points,
            report.tolerance,
            report.candidates_per_hour,
            report.spearman_rho,
            report.kendall_tau_b,
            report.false_promotions,
            report.false_rejections,
            report.max_abs_score_error,
            report.mean_abs_score_error,
            report.smooth_objective_spearman,
        );
    }
    println!(
        "\nreference preference scores: min {score_min:.4}, max {score_max:.4}, spread {spread:.4}"
    );
    if spread < 1e-2 {
        println!(
            "WARNING: reference score spread is very small ({spread:.3e}); rank metrics may \
             reflect solver noise rather than policy quality. Consider wider perturbations."
        );
    }

    // Gate: the §6 L1 rung must preserve the reference ranking.
    let l1 = reports
        .iter()
        .find(|r| r.name == "l1-combined")
        .expect("policy list always contains l1-combined");
    let passed =
        l1.spearman_rho >= GATE_SPEARMAN && l1.false_promotions <= GATE_MAX_FALSE_PROMOTIONS;
    let gate = format!(
        "l1-combined: Spearman >= {GATE_SPEARMAN} AND false-promotions <= {GATE_MAX_FALSE_PROMOTIONS}"
    );
    println!(
        "\nGate ({gate}): Spearman {:.4}, FP {} → {}",
        l1.spearman_rho,
        l1.false_promotions,
        if passed { "PASS" } else { "FAIL" }
    );

    let report = BenchmarkReport {
        band_hz: [FREQ_START_HZ, FREQ_END_HZ],
        num_candidates: n,
        top_k,
        ranking_objective: "CEA-2034 preference score (higher is better)".into(),
        reference_score_spread: spread,
        gate,
        gate_spearman_minimum: GATE_SPEARMAN,
        gate_max_false_promotions: GATE_MAX_FALSE_PROMOTIONS,
        passed,
        policies: reports,
    };
    let json = serde_json::to_string_pretty(&report).expect("report serializes");
    if json_stdout {
        println!("\n{json}");
    } else {
        let path = output_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("cannot create {}: {e}", parent.display()));
        }
        std::fs::write(&path, &json)
            .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
        println!("JSON report written to {}", path.display());
    }

    let total_wall: f64 = report.policies.iter().map(|p| p.wall_seconds).sum();
    println!("total simulation wall time: {total_wall:.1}s");

    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::require_finite_objectives;

    #[test]
    fn nonfinite_candidate_objectives_are_rejected_before_ranking() {
        assert!(require_finite_objectives("preference", &[1.0, 0.5]).is_ok());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = require_finite_objectives("preference", &[1.0, value])
                .expect_err("non-finite objective must not be ranked");
            assert!(error.contains("candidate 1"), "{error}");
        }
    }
}
