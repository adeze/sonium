//! Step-9 convergence/promotion evidence (PORT-OPTIM.md Step 9).
//!
//! This module promotes (or demotes) the Step-7 winning profile on regenerated
//! validation meshes at independent resolutions (plan decision D6). The
//! winner runs a spatial ladder (≥ 3 mesh sizes) and a temporal ladder (≥ 2
//! timesteps) through [`run_port_flow_promotion_with_observer`](super::runtime::run_port_flow_promotion_with_observer)
//! with a [`PortFlowPromotionGate`](super::convergence::PortFlowPromotionGate)
//! tightened to the plan's 2% observable bar. The observer reduces the THD
//! and `U⁸/L³` noise proxy from the same accepted traces the phasor gate
//! validates, so the headline trio (flow phasor, THD, noise proxy) is gated
//! without re-solving.
//!
//! # Why the linear gate compares cross-tier ratios
//!
//! Tier 2 runs on the 2D slot surrogate (10 mm bore, unit depth) while the
//! Step-4 oracle
//! ([`solve_spline_port_linear_reference`](super::linear_reference::solve_spline_port_linear_reference))
//! solves the 3D round port at campaign scale (25 mm bore). Absolute
//! impedances differ by construction (the Step-0 straight-tube lumped limit
//! already sits at 9.9%), so a direct impedance comparison would fail on
//! scale, not physics. Both tiers instead predict the same dimensionless
//! winner/straight impedance ratio `Z_w/Z_s` for the same fractional
//! profile, and the Step-9 linear gate (`≤ 3%`, plus the QA
//! transfer-identity, THD `≤ 0.5%`, and Mach `≤ 0.05` bars) checks that the
//! Tier-2 low-drive limit on the final mesh agrees with the independent
//! oracle about the fractional effect of the winning dip. Both sides use
//! `exp(-i·ω·t)` phasors, so no conjugation is needed, and peak/RMS
//! conventions cancel in the ratios.
//!
//! The flow-split double ratio `H_w/H_s` is reported but deliberately not
//! gated: Tier-1 evaluates at the campaign tuning, ~1.6% from the
//! box-port Helmholtz pole (condition number ~62), while Tier-2 runs
//! off-resonance by fixture, so the two `H` ratios live in different
//! resonance regimes (see [`STEP9_MAX_FLOW_RATIO_RELATIVE_ERROR`]).
//!
//! # What runs where
//!
//! The expensive solves live in the ignored `port_optim_step9` leg
//! (release-only, like the Step-7/8 legs). This module stays solver-free
//! except for the Tier-1 oracle helper [`solve_step9_tier1`], which the
//! ignored leg calls; everything else here is configuration, reduction,
//! gating, and archiving logic covered by fast unit tests.
//!
//! Validated envelope: 2D meridian/slot convergence evidence at the
//! production envelope (ladders) plus a cross-tier linear-consistency check.
//! A [`Step9Verdict::DemoteUnconvergedIndicativeOnly`] verdict keeps every
//! Step-7/8 headline number labelled indicative; it is a successful,
//! evidence-backed outcome, not a failure.

use std::f64::consts::PI;
use std::path::Path;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::spaces::{H1Space, VectorH1Space};

use super::analysis::{coupled_complex_observables, dft_amplitude, thd_fraction};
use super::baseline::{
    BASELINE_MAX_ENERGY_PHYSICAL_CLOSURE, BASELINE_MAX_TIER1_FLUX_RELATIVE_DIFFERENCE,
    PortOptimReferenceCase, complex_pair, duct_bulk_mach_number,
};
use super::campaign::CampaignArchiverError;
use super::convergence::{PORT_FLOW_PROMOTION_SCHEMA_VERSION, PortFlowPromotionEvidence};
use super::convergence::{PortFlowPromotionGate, PortFlowPromotionReport};
use super::coupled::CoupledPortResult;
use super::error::PortFlowError;
use super::evaluate::{PortEvalDrive, PortEvalFidelity, Tier2EvalContext};
use super::geometry3d::build_spline_round_port_mesh;
use super::linear_reference::{
    ComplexPortTransfer, SplinePortLinearReferenceConfig, solve_spline_port_linear_reference,
};
use super::profile::{PROFILE_VALIDATION_GRID_POINTS, SplinePortProfile};
use super::runtime::{
    PortFlowRuntimeCase, PortFlowRuntimePlan, PortFlowRuntimeRefinement,
    run_port_flow_promotion_with_observer,
};
use super::spline_geometry::min_triangle_area_m2;
use crate::free_turbulence_power_trend;

// ---------------------------------------------------------------------------
// Thresholds and ladder schedule
// ---------------------------------------------------------------------------

/// Archive schema version for [`Step9Report`].
pub const STEP9_REPORT_SCHEMA_VERSION: u32 = 1;

/// Model lineage recorded in every [`Step9Report`].
pub const STEP9_REPORT_PROVENANCE: &str =
    "PortOptimStep9Promotion[Tier2RegeneratedLadder,SplineLRFTransferGate]";

/// Provenance recorded in the promotion evidence/plan.
pub const STEP9_PROMOTION_PROVENANCE: &str = "port-optim-step9-winner.v1";

/// Spatial ladder: coarse-to-fine regenerated mesh sizes in m.
///
/// The fine state is the production 0.01 m mesh. Every level is a credible
/// resolution (the 0.02 m mesh holds 60 nodes and drifts 2.5% in `Q_rms`,
/// so it is excluded rather than gated).
pub const STEP9_SPATIAL_MESH_SIZES_M: [f64; 3] = [0.016, 0.012, 0.01];

/// Temporal ladder: coarse-to-fine BDF samples per drive cycle.
///
/// Both values divide the drive period integrally. The fine state (504)
/// sits one dt-halving above production (252), so a passing ladder
/// brackets the Step-7/8 production numbers from the fine side; the
/// production state itself is the coarse temporal level.
pub const STEP9_TEMPORAL_SAMPLES_PER_CYCLE: [usize; 2] = [252, 504];

/// Headline convergence bar: < 2% finest-pair change (Success Criterion 4).
pub const STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE: f64 = 0.02;

/// THD resolution floor (fraction): THD pairs with both levels at or below
/// this pass regardless of relative change.
///
/// The campaign physics is linear to ~1e-5, so measured THDs (~1e-6..6e-5)
/// are DFT-leakage/solver-noise scatter, not resolved distortion: relative
/// changes across healthy levels exceed 100×. The floor is 2% of the 0.5%
/// low-drive linearity bar — differences below it are neither numerically
/// resolvable on these grids nor physically relevant. Absolute values are
/// always archived alongside the verdict so nothing is hidden.
pub const STEP9_THD_RESOLUTION_FLOOR: f64 = 1.0e-4;

/// Phasor cap-continuity health bar for Tier-2 traces.
///
/// The phasor cap plateaus at ~8e-4 on healthy runs from 0.01 to 0.016 m
/// (mesh- and dt-independent systematic, not discretization error), so it
/// cannot gate convergence — only health. The 3D-derived `1e-6` default
/// was never demonstrated on 2D slot data; `3e-3` keeps 3× margin over the
/// demonstrated ~1e-3 ladder envelope while still catching genuine
/// breakage (wrong markers, diverged caps) orders above.
pub const STEP9_MAX_CAP_CONTINUITY_RELATIVE_ERROR: f64 = 3.0e-3;

/// Cross-tier impedance-ratio bar: winner/straight `Z` within 3%.
pub const STEP9_MAX_IMPEDANCE_RATIO_RELATIVE_ERROR: f64 = 0.03;

/// Reference bar for the reported cross-tier flow-split ratio.
///
/// The winner/straight `H` double ratio is reported against 3% but NOT
/// gated: Tier-1 evaluates at the campaign tuning, which sits ~1.6% from
/// the box-port Helmholtz pole (`|H| ~ 63–94`, condition number ~62), so
/// its `H` ratio measures resonance detuning from the dip's tuning shift,
/// while Tier-2 runs off-resonance by fixture (`|H| ~ 1`, condition ~0.3).
/// Gating the two against each other would demand zero tuning shift,
/// contradicting the campaign's own ±2 Hz feasibility spec. The impedance
/// double ratio (same fractional perturbation, both tiers
/// inertance-dominated) carries the cross-tier agreement instead; the `H`
/// values, error, and Tier-1 condition numbers stay in the report for
/// transparency.
pub const STEP9_MAX_FLOW_RATIO_RELATIVE_ERROR: f64 = 0.03;

/// Tier-2 transfer-identity bar (direct `H` vs `H` from `Z`), QA value.
pub const STEP9_MAX_TRANSFER_IDENTITY_RELATIVE_ERROR: f64 = 0.02;

/// Low-drive THD bar (fraction), QA value.
pub const STEP9_MAX_THD_FRACTION: f64 = 0.005;

/// Low-drive bulk-Mach bar, QA value.
pub const STEP9_MAX_MACH: f64 = 0.05;

/// Tier-1 oracle mesh size in m (Step-0 reference mesh, not the Step-7
/// coarse re-rank mesh).
pub const STEP9_TIER1_MESH_SIZE_M: f64 = 0.01;

/// Chamber-to-duct smoothstep extent for the Tier-1 oracle in m.
pub const STEP9_TIER1_FLARE_LENGTH_M: f64 = 0.01;

/// Harmonics (including the fundamental) in the ladder THD reduction.
///
/// Matches the Step-5 harness and the Step-8 3D spot-check.
pub const STEP9_THD_HARMONICS: usize = 5;

/// Permissive structural gate: ladder shape and orchestration only.
///
/// [`run_step9_ladder`] executes under this gate so a failing ladder still
/// returns its evidence; the strict [`step9_promotion_gate`] is applied
/// afterwards via
/// [`validate_port_flow_promotion`](super::convergence::validate_port_flow_promotion),
/// and the verdict logic re-derives pass/fail from the archived numbers.
pub fn step9_structural_gate() -> PortFlowPromotionGate {
    PortFlowPromotionGate {
        min_spatial_levels: 3,
        min_temporal_levels: 2,
        max_spatial_observable_relative_change: f64::MAX,
        max_temporal_observable_relative_change: f64::MAX,
        max_cross_ladder_observable_relative_change: f64::MAX,
        max_linear_relative_residual: f64::MAX,
        max_box_conservation_relative_residual: f64::MAX,
        max_cap_continuity_relative_error: f64::MAX,
        max_energy_algebraic_relative_closure: f64::MAX,
        max_energy_physical_relative_closure: f64::MAX,
        max_energy_interface_relative_residual: f64::MAX,
    }
}

/// Step-9 promotion gate: plan bars on the shared machinery.
///
/// Observable (phasor) change at 2% per the plan; energy/cap diagnostics at
/// the demonstrated Tier-2 values (physical energy at the Step-0 bound,
/// cap as a health bar over the measured plateau); every other diagnostic
/// at the [`PortFlowPromotionGate::default`] value, which Tier-2 traces
/// clear by orders of magnitude.
pub fn step9_promotion_gate() -> PortFlowPromotionGate {
    PortFlowPromotionGate {
        min_spatial_levels: 3,
        min_temporal_levels: 2,
        max_spatial_observable_relative_change: STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE,
        max_temporal_observable_relative_change: STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE,
        max_cross_ladder_observable_relative_change: STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE,
        max_cap_continuity_relative_error: STEP9_MAX_CAP_CONTINUITY_RELATIVE_ERROR,
        max_energy_physical_relative_closure: BASELINE_MAX_ENERGY_PHYSICAL_CLOSURE,
        ..PortFlowPromotionGate::default()
    }
}

// ---------------------------------------------------------------------------
// Ladder configuration
// ---------------------------------------------------------------------------

/// Spatial + temporal ladder schedule for one Step-9 campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionLadderConfig {
    /// Coarse-to-fine regenerated mesh sizes in m (≥ 3, strictly decreasing).
    pub spatial_mesh_sizes_m: Vec<f64>,
    /// Coarse-to-fine samples per drive cycle (≥ 2, strictly increasing).
    pub temporal_samples_per_cycle: Vec<usize>,
    /// Total drive cycles simulated per level.
    pub total_cycles: usize,
    /// Settled tail cycles analysed per level (`< total_cycles`).
    pub tail_cycles: usize,
    /// Ladder drive level in V peak.
    pub voltage_peak_v: f64,
}

impl PromotionLadderConfig {
    /// Production ladder: plan schedule at the `V_max` drive.
    ///
    /// Cycles come from the frozen reference case (40 total / 8 tail); the
    /// drive is the campaign maximum, where THD and the noise proxy are
    /// largest and convergence is most demanding.
    pub fn production(case: &PortOptimReferenceCase) -> Result<Self, PortFlowError> {
        case.validate()?;
        let config = Self {
            spatial_mesh_sizes_m: STEP9_SPATIAL_MESH_SIZES_M.to_vec(),
            temporal_samples_per_cycle: STEP9_TEMPORAL_SAMPLES_PER_CYCLE.to_vec(),
            total_cycles: case.tier2_total_cycles,
            tail_cycles: case.tier2_tail_cycles,
            voltage_peak_v: case.tier2_drives_v_peak[case.tier2_drives_v_peak.len() - 1],
        };
        config.validate()?;
        Ok(config)
    }

    /// Transient cycles discarded before the analysed tail.
    pub fn discarded_cycles(&self) -> usize {
        self.total_cycles - self.tail_cycles
    }

    /// Finest (promoted) mesh size in m.
    pub fn fine_mesh_size_m(&self) -> f64 {
        self.spatial_mesh_sizes_m[self.spatial_mesh_sizes_m.len() - 1]
    }

    /// Finest (promoted) samples per cycle.
    pub fn fine_samples_per_cycle(&self) -> usize {
        self.temporal_samples_per_cycle[self.temporal_samples_per_cycle.len() - 1]
    }

    /// Reject an inconsistent schedule before any solve.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        if self.spatial_mesh_sizes_m.len() < 3 {
            return Err(PortFlowError::InvalidConfig(format!(
                "Step-9 ladder needs at least 3 spatial levels, got {}",
                self.spatial_mesh_sizes_m.len()
            )));
        }
        for window in self.spatial_mesh_sizes_m.windows(2) {
            if !(window[0].is_finite() && window[1].is_finite())
                || window[0] <= 0.0
                || window[1] <= 0.0
                || window[1] >= window[0]
            {
                return Err(PortFlowError::InvalidConfig(format!(
                    "Step-9 spatial ladder must be finite, positive, and strictly refining, \
                     got {window:?}"
                )));
            }
        }
        if self.temporal_samples_per_cycle.len() < 2 {
            return Err(PortFlowError::InvalidConfig(format!(
                "Step-9 ladder needs at least 2 temporal levels, got {}",
                self.temporal_samples_per_cycle.len()
            )));
        }
        for window in self.temporal_samples_per_cycle.windows(2) {
            // The runtime plan requires ≥ 4 samples/cycle; the THD reduction
            // additionally needs the 5th harmonic below Nyquist
            // (`spc > 2·5`), enforced again when each case is built.
            if window[0] < 4 || window[1] <= window[0] {
                return Err(PortFlowError::InvalidConfig(format!(
                    "Step-9 temporal ladder needs ≥ 4 strictly increasing samples/cycle, \
                     got {window:?}"
                )));
            }
        }
        if self.tail_cycles == 0 || self.tail_cycles >= self.total_cycles {
            return Err(PortFlowError::InvalidConfig(format!(
                "Step-9 ladder needs 0 < tail_cycles < total_cycles, got tail {} of total {}",
                self.tail_cycles, self.total_cycles
            )));
        }
        if !(self.voltage_peak_v.is_finite() && self.voltage_peak_v > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Step-9 ladder drive voltage must be finite and positive, got {}",
                self.voltage_peak_v
            )));
        }
        self.total_cycles
            .checked_mul(self.fine_samples_per_cycle())
            .ok_or_else(|| {
                PortFlowError::InvalidConfig("Step-9 ladder step count overflow".to_string())
            })?;
        Ok(())
    }

    /// Timesteps in s, coarse-to-fine, for a drive frequency in Hz.
    pub fn temporal_dt_s(&self, frequency_hz: f64) -> Result<Vec<f64>, PortFlowError> {
        self.validate()?;
        if !(frequency_hz.is_finite() && frequency_hz > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Step-9 ladder needs a finite positive drive frequency, got {frequency_hz}"
            )));
        }
        Ok(self
            .temporal_samples_per_cycle
            .iter()
            .map(|spc| 1.0 / (frequency_hz * *spc as f64))
            .collect())
    }

    /// Runtime plan for the shared promotion runner.
    pub fn to_runtime_plan(
        &self,
        case: &PortOptimReferenceCase,
        frequency_hz: f64,
    ) -> Result<PortFlowRuntimePlan, PortFlowError> {
        Ok(PortFlowRuntimePlan {
            spatial_mesh_sizes_m: self.spatial_mesh_sizes_m.clone(),
            temporal_dt_s: self.temporal_dt_s(frequency_hz)?,
            frequency_hz,
            tail_cycles: self.tail_cycles,
            discarded_cycles: self.discarded_cycles(),
            box_volume_m3: case.box_volume_m3,
            density_kg_per_m3: case.density_kg_per_m3,
            sound_speed_m_per_s: case.sound_speed_m_per_s,
            provenance: STEP9_PROMOTION_PROVENANCE.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Level case builder (regenerated validation meshes)
// ---------------------------------------------------------------------------

/// Mesh census for one ladder level, recorded by the case builder.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PromotionMeshStats {
    /// Regenerated mesh size in m.
    pub mesh_size_m: f64,
    /// Node count.
    pub mesh_nodes: usize,
    /// Element count.
    pub mesh_elements: usize,
    /// Minimum signed triangle area in m² (positivity audit).
    pub min_triangle_area_m2: f64,
}

/// Minimum duct cross-section `π·min(r)²` over the dense validation grid.
///
/// Same convention as the Step-5 harness (round-area throat even at slot
/// scale): the ladder holds the profile fixed, so the convention cancels in
/// every finest-pair change and only needs to match the harness for the
/// absolute Mach/proxy cross-checks.
pub fn min_profile_area_m2(profile: &SplinePortProfile) -> Result<f64, PortFlowError> {
    let min_radius = profile
        .sampled_profile(PROFILE_VALIDATION_GRID_POINTS)
        .iter()
        .map(|station| station[1])
        .fold(f64::INFINITY, f64::min);
    if !(min_radius.is_finite() && min_radius > 0.0) {
        return Err(PortFlowError::SolverDiverged {
            step: 0,
            time_s: 0.0,
        });
    }
    Ok(PI * min_radius * min_radius)
}

/// Build one ladder-level runtime case on a regenerated validation mesh.
///
/// The mesh is regenerated from scratch for the candidate profile at
/// `mesh_size_m` (never morphed: Step 9 is final validation at independent
/// resolutions), audited for positivity, and paired with 2D Taylor–Hood
/// P2–P1 spaces and the Step-5 coupled configuration (BDF2, stabilisation
/// off, affine box closure) at the requested drive and fidelity.
pub fn build_promotion_case(
    context: &Tier2EvalContext,
    profile: &SplinePortProfile,
    voltage_peak_v: f64,
    frequency_hz: f64,
    mesh_size_m: f64,
    fidelity: &PortEvalFidelity,
) -> Result<(PortFlowRuntimeCase, PromotionMeshStats), PortFlowError> {
    profile.validate()?;
    let leveled = context.clone().with_mesh_size_m(mesh_size_m)?;
    let geometry = leveled.candidate_geometry(profile);
    let mesh = geometry.build_validation_mesh()?;
    let min_area = min_triangle_area_m2(&mesh);
    if min_area.is_nan() || min_area <= 0.0 {
        return Err(PortFlowError::Mesh(format!(
            "Step-9 regenerated mesh failed the pre-solve positivity audit \
             (min signed area {min_area:.3e} m²)"
        )));
    }
    let stats = PromotionMeshStats {
        mesh_size_m,
        mesh_nodes: mesh.num_nodes(),
        mesh_elements: mesh.elements.len(),
        min_triangle_area_m2: min_area,
    };
    let drive = PortEvalDrive {
        voltage_peak_v,
        frequency_hz,
        second_tone: None,
    };
    let config = leveled.coupled_config(&drive, fidelity)?;
    Ok((
        PortFlowRuntimeCase {
            mesh,
            velocity_space: VectorH1Space::new(PolynomialDegree::P2, 2),
            pressure_space: H1Space::new(PolynomialDegree::P1),
            config,
        },
        stats,
    ))
}

// ---------------------------------------------------------------------------
// Per-level trace reduction (observer body)
// ---------------------------------------------------------------------------

/// Fixed inputs for reducing one ladder-level trace.
#[derive(Debug, Clone, Copy)]
pub struct PromotionReductionInputs {
    /// Settled tail cycles analysed.
    pub tail_cycles: usize,
    /// Transient cycles discarded before the tail.
    pub discarded_cycles: usize,
    /// Lumped sealed-box volume in m³.
    pub box_volume_m3: f64,
    /// Medium density in kg/m³.
    pub density_kg_per_m3: f64,
    /// Medium sound speed in m/s.
    pub sound_speed_m_per_s: f64,
    /// Profile throat area in m² (see [`min_profile_area_m2`]).
    pub throat_area_m2: f64,
    /// Reference duct length in m (noise-proxy length scale).
    pub duct_length_m: f64,
}

/// Reduced observables for one ladder level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionLevelDiagnostics {
    /// Drive frequency in Hz.
    pub frequency_hz: f64,
    /// Regenerated mesh size in m.
    pub mesh_size_m: f64,
    /// BDF timestep in s.
    pub dt_s: f64,
    /// BDF samples per drive cycle.
    pub samples_per_cycle: usize,
    /// Completed BDF steps (always the configured total on success).
    pub steps_completed: usize,
    /// Analysed settled-tail samples.
    pub tail_samples: usize,
    /// Mesh node count.
    pub mesh_nodes: usize,
    /// Mesh element count.
    pub mesh_elements: usize,
    /// Port-flow peak phasor `[re, im]` in m³/s (`exp(-i·ω·t)`).
    pub port_flow: [f64; 2],
    /// Box-pressure peak phasor `[re, im]` in Pa.
    pub box_pressure: [f64; 2],
    /// Diaphragm-flow peak phasor `[re, im]` in m³/s.
    pub diaphragm_flow: [f64; 2],
    /// Coupled port impedance `[re, im]` in Pa·s/m³.
    pub port_impedance: [f64; 2],
    /// Direct `Q_port/Q_diaphragm` transfer `[re, im]`.
    pub flow_ratio_direct: [f64; 2],
    /// Transfer reconstructed as `1/(1-i·ω·C·Z)` `[re, im]`.
    pub flow_ratio_from_impedance: [f64; 2],
    /// `|H_direct − H(Z)|/|H(Z)|`.
    pub transfer_identity_relative_error: f64,
    /// Complex cap-continuity error `|Q_ext+Q_box,out|/|Q_ext|`.
    pub cap_continuity_relative_error: f64,
    /// Algebraic energy closure over the tail (relative).
    pub energy_algebraic_relative_closure: f64,
    /// Physical-only energy closure over the tail (relative).
    pub energy_physical_relative_closure: f64,
    /// Interface energy residual over the tail (relative).
    pub energy_interface_relative_residual: f64,
    /// Energy normalisation scale over the tail in J.
    pub energy_scale_j: f64,
    /// Worst accepted affine-solve true relative residual over the tail.
    pub max_linear_relative_residual: f64,
    /// Worst box-conservation residual over the tail, normalised by `|P|`.
    pub max_box_conservation_relative_residual: f64,
    /// Time-domain port-flow RMS over the tail in m³/s.
    pub port_flow_rms_m3s: f64,
    /// Hann fundamental port-flow amplitude over the tail in m³/s.
    pub port_flow_fundamental_amplitude_m3s: f64,
    /// Hann fundamental box-pressure amplitude over the tail in Pa.
    pub box_pressure_amplitude_pa: f64,
    /// Total harmonic distortion over [`STEP9_THD_HARMONICS`] (fraction).
    pub thd_fraction: f64,
    /// Free-turbulence `U⁸/L³` noise-proxy trend from the fundamental
    /// throat velocity (same definition as the Step-5 harness).
    pub noise_proxy: f64,
    /// Bulk-velocity Mach number `max|q|/(S_min·c)` over the tail.
    pub peak_mach_bulk: f64,
}

/// Reduce one accepted ladder-level trace to diagnostics.
///
/// Mirrors the Step-5 harness reductions (Hann DFT fundamental, THD over 5
/// harmonics, `U⁸/L³` proxy, bulk Mach) and the promotion-evidence tail
/// audits (complex observables, energy summary, worst residuals) on the
/// same complete-cycle tail. Any misaligned, non-finite, or degenerate
/// tail fails closed; metrics are total on success.
pub fn reduce_promotion_trace(
    refinement: PortFlowRuntimeRefinement,
    result: &CoupledPortResult,
    inputs: PromotionReductionInputs,
    mesh_stats: PromotionMeshStats,
) -> Result<PromotionLevelDiagnostics, PortFlowError> {
    let num_steps = refinement
        .discarded_cycles
        .checked_add(refinement.tail_cycles)
        .and_then(|cycles| cycles.checked_mul(refinement.samples_per_cycle))
        .ok_or_else(|| {
            PortFlowError::InvalidConfig("Step-9 level step count overflow".to_string())
        })?;
    if refinement.tail_cycles != inputs.tail_cycles
        || refinement.discarded_cycles != inputs.discarded_cycles
    {
        return Err(PortFlowError::InvalidConfig(
            "Step-9 reduction tail does not match the executed refinement".to_string(),
        ));
    }
    let last_time = result.time_s.last().copied().unwrap_or(0.0);
    let diverged = |step: usize| PortFlowError::SolverDiverged {
        step,
        time_s: last_time,
    };
    if result.steps_completed != num_steps {
        return Err(diverged(result.steps_completed));
    }
    for length in [
        result.time_s.len(),
        result.port_flow_m3s.len(),
        result.exterior_cap_flow_m3s.len(),
        result.box_cap_outward_flow_m3s.len(),
        result.diaphragm_volume_velocity_m3s.len(),
        result.box_pressure_pa.len(),
        result.displacement_m.len(),
        result.current_a.len(),
        result.box_conservation_residual_pa.len(),
        result.energy_steps.len(),
        result.accepted_linear_relative_residual.len(),
    ] {
        if length != num_steps {
            return Err(diverged(length.min(num_steps.saturating_sub(1))));
        }
    }
    let tail_samples = refinement
        .tail_cycles
        .checked_mul(refinement.samples_per_cycle)
        .ok_or_else(|| {
            PortFlowError::InvalidConfig("Step-9 level tail sample count overflow".to_string())
        })?;
    let tail_start = num_steps - tail_samples;
    let tail_range = tail_start..num_steps;
    for series in [
        &result.port_flow_m3s[tail_range.clone()],
        &result.box_cap_outward_flow_m3s[tail_range.clone()],
        &result.box_pressure_pa[tail_range.clone()],
        &result.box_conservation_residual_pa[tail_range.clone()],
        &result.accepted_linear_relative_residual[tail_range.clone()],
    ] {
        if series.iter().any(|value| !value.is_finite()) {
            return Err(diverged(num_steps - 1));
        }
    }
    let flow_tail = &result.port_flow_m3s[tail_range.clone()];
    let pressure_tail = &result.box_pressure_pa[tail_range.clone()];
    let dt_s = refinement.dt_s;
    let frequency_hz = refinement.frequency_hz;

    let port_flow_rms_m3s =
        (flow_tail.iter().map(|q| q * q).sum::<f64>() / tail_samples as f64).sqrt();
    let max_abs_flow = flow_tail.iter().map(|q| q.abs()).fold(0.0, f64::max);
    let port_flow_fundamental_amplitude_m3s = dft_amplitude(flow_tail, dt_s, frequency_hz);
    let box_pressure_amplitude_pa = dft_amplitude(pressure_tail, dt_s, frequency_hz);
    for (name, value) in [
        ("port-flow RMS", port_flow_rms_m3s),
        (
            "port-flow fundamental amplitude",
            port_flow_fundamental_amplitude_m3s,
        ),
        ("box-pressure amplitude", box_pressure_amplitude_pa),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Step-9 level has a non-positive {name}: {value}"
            )));
        }
    }
    let thd_fraction = thd_fraction(flow_tail, dt_s, frequency_hz, STEP9_THD_HARMONICS);
    if !(thd_fraction.is_finite() && thd_fraction >= 0.0) {
        return Err(diverged(num_steps - 1));
    }
    let throat_velocity = port_flow_fundamental_amplitude_m3s / inputs.throat_area_m2;
    let noise_proxy =
        free_turbulence_power_trend(throat_velocity, inputs.duct_length_m).map_err(|error| {
            PortFlowError::InvalidConfig(format!(
                "Step-9 turbulence proxy rejected its inputs: {error}"
            ))
        })?;
    let peak_mach_bulk = duct_bulk_mach_number(
        max_abs_flow,
        inputs.throat_area_m2,
        inputs.sound_speed_m_per_s,
    );
    if !peak_mach_bulk.is_finite() {
        return Err(diverged(num_steps - 1));
    }

    let observables = coupled_complex_observables(
        result,
        dt_s,
        frequency_hz,
        refinement.samples_per_cycle,
        refinement.tail_cycles,
        refinement.discarded_cycles,
        inputs.box_volume_m3,
        inputs.density_kg_per_m3,
        inputs.sound_speed_m_per_s,
    )?;
    let energy = result.energy_summary(tail_range.clone());
    if !energy.is_finite() {
        return Err(PortFlowError::InvalidConfig(
            "Step-9 level tail energy summary is non-finite".to_string(),
        ));
    }
    let max_linear_relative_residual = result.accepted_linear_relative_residual[tail_range.clone()]
        .iter()
        .copied()
        .fold(0.0, f64::max);
    let pressure_scale = observables.box_pressure_pa.norm().max(f64::EPSILON);
    let max_box_conservation_relative_residual = result.box_conservation_residual_pa
        [tail_range.clone()]
    .iter()
    .map(|residual| residual.abs() / pressure_scale)
    .fold(0.0, f64::max);
    let transfer_identity_relative_error = complex_relative_error(
        observables.direct_box_to_port_flow_ratio,
        observables.box_to_port_flow_ratio_from_impedance,
    )?;

    Ok(PromotionLevelDiagnostics {
        frequency_hz,
        mesh_size_m: refinement.mesh_size_m,
        dt_s,
        samples_per_cycle: refinement.samples_per_cycle,
        steps_completed: num_steps,
        tail_samples,
        mesh_nodes: mesh_stats.mesh_nodes,
        mesh_elements: mesh_stats.mesh_elements,
        port_flow: complex_pair(observables.port_flow_m3s),
        box_pressure: complex_pair(observables.box_pressure_pa),
        diaphragm_flow: complex_pair(observables.diaphragm_flow_m3s),
        port_impedance: complex_pair(observables.port_impedance_pa_s_per_m3),
        flow_ratio_direct: complex_pair(observables.direct_box_to_port_flow_ratio),
        flow_ratio_from_impedance: complex_pair(observables.box_to_port_flow_ratio_from_impedance),
        transfer_identity_relative_error,
        cap_continuity_relative_error: observables.cap_continuity_relative_error,
        energy_algebraic_relative_closure: energy.algebraic_relative_closure,
        energy_physical_relative_closure: energy.physical_relative_closure,
        energy_interface_relative_residual: energy.interface_relative_residual,
        energy_scale_j: energy.energy_scale_j,
        max_linear_relative_residual,
        max_box_conservation_relative_residual,
        port_flow_rms_m3s,
        port_flow_fundamental_amplitude_m3s,
        box_pressure_amplitude_pa,
        thd_fraction,
        noise_proxy,
        peak_mach_bulk,
    })
}

fn complex_relative_error(actual: Complex64, reference: Complex64) -> Result<f64, PortFlowError> {
    if ![actual.re, actual.im, reference.re, reference.im]
        .into_iter()
        .all(f64::is_finite)
        || reference.norm() <= f64::EPSILON
    {
        return Err(PortFlowError::InvalidConfig(
            "Step-9 complex comparison has a non-finite or zero reference".to_string(),
        ));
    }
    let error = (actual - reference).norm() / reference.norm();
    if error.is_finite() {
        Ok(error)
    } else {
        Err(PortFlowError::InvalidConfig(
            "Step-9 complex relative error is non-finite".to_string(),
        ))
    }
}

fn complex_ratio(numerator: Complex64, denominator: Complex64) -> Result<Complex64, PortFlowError> {
    if ![numerator.re, numerator.im, denominator.re, denominator.im]
        .into_iter()
        .all(f64::is_finite)
        || denominator.norm() <= f64::EPSILON
    {
        return Err(PortFlowError::InvalidConfig(
            "Step-9 complex ratio has a non-finite or zero denominator".to_string(),
        ));
    }
    let ratio = numerator / denominator;
    if ratio.re.is_finite() && ratio.im.is_finite() {
        Ok(ratio)
    } else {
        Err(PortFlowError::InvalidConfig(
            "Step-9 complex ratio is non-finite".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Finest-pair extended gates (THD, noise proxy)
// ---------------------------------------------------------------------------

/// One finest-pair convergence check with its absolute context.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FinestPairChange {
    /// Coarse-level value (second-finest).
    pub coarse: f64,
    /// Fine-level value.
    pub fine: f64,
    /// `|coarse − fine|/|fine|`.
    pub relative_change: f64,
    /// Applied bar.
    pub max_allowed: f64,
    /// Resolution floor: when `Some`, the check also passes if both values
    /// sit at or below it (THD only; see [`STEP9_THD_RESOLUTION_FLOOR`]).
    pub floor: Option<f64>,
    /// Whether the check passes (bar or floor).
    pub passed: bool,
    /// Whether the pass came from the floor rather than the bar.
    pub passed_by_floor: bool,
}

/// Gate one finest-pair relative change, failing closed on bad inputs.
///
/// Both values must be finite with a strictly positive fine value; THD and
/// the noise proxy are non-negative by construction, so a non-positive fine
/// value means a degenerate reduction, not convergence.
pub fn check_finest_pair(
    name: &str,
    coarse: f64,
    fine: f64,
    max_allowed: f64,
) -> Result<FinestPairChange, PortFlowError> {
    check_finest_pair_with_floor(name, coarse, fine, max_allowed, None)
}

/// Gate one THD finest-pair change with the resolution floor.
///
/// Passes when the relative change clears the bar or when both levels sit
/// at or below [`STEP9_THD_RESOLUTION_FLOOR`] (numerically unresolvable
/// distortion). The `passed_by_floor` flag records which path passed.
pub fn check_thd_finest_pair(
    name: &str,
    coarse: f64,
    fine: f64,
    max_allowed: f64,
) -> Result<FinestPairChange, PortFlowError> {
    check_finest_pair_with_floor(
        name,
        coarse,
        fine,
        max_allowed,
        Some(STEP9_THD_RESOLUTION_FLOOR),
    )
}

fn check_finest_pair_with_floor(
    name: &str,
    coarse: f64,
    fine: f64,
    max_allowed: f64,
    floor: Option<f64>,
) -> Result<FinestPairChange, PortFlowError> {
    if !(max_allowed.is_finite() && max_allowed >= 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-9 finest-pair bar for {name} must be finite and non-negative, got {max_allowed}"
        )));
    }
    if let Some(floor) = floor
        && !(floor.is_finite() && floor >= 0.0)
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-9 finest-pair floor for {name} must be finite and non-negative, got {floor}"
        )));
    }
    if !(coarse.is_finite() && fine.is_finite() && fine > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-9 finest-pair {name} needs finite values with a positive fine value, \
             got coarse={coarse} fine={fine}"
        )));
    }
    let relative_change = (coarse - fine).abs() / fine;
    if !relative_change.is_finite() {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-9 finest-pair {name} relative change is non-finite"
        )));
    }
    let passed_by_floor = floor.is_some_and(|floor| coarse <= floor && fine <= floor);
    Ok(FinestPairChange {
        coarse,
        fine,
        relative_change,
        max_allowed,
        floor,
        passed: relative_change <= max_allowed || passed_by_floor,
        passed_by_floor: passed_by_floor && relative_change > max_allowed,
    })
}

/// THD + noise-proxy finest-pair verdicts for both ladders.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PromotionExtendedChanges {
    /// Spatial finest-pair THD change.
    pub spatial_thd: FinestPairChange,
    /// Spatial finest-pair noise-proxy change.
    pub spatial_noise_proxy: FinestPairChange,
    /// Temporal finest-pair THD change.
    pub temporal_thd: FinestPairChange,
    /// Temporal finest-pair noise-proxy change.
    pub temporal_noise_proxy: FinestPairChange,
    /// Whether all four checks pass.
    pub passed: bool,
}

impl PromotionExtendedChanges {
    /// Names of the failing checks (empty when [`Self::passed`]).
    pub fn failures(&self) -> Vec<&'static str> {
        [
            ("spatial THD", self.spatial_thd.passed),
            ("spatial noise proxy", self.spatial_noise_proxy.passed),
            ("temporal THD", self.temporal_thd.passed),
            ("temporal noise proxy", self.temporal_noise_proxy.passed),
        ]
        .into_iter()
        .filter_map(|(name, passed)| (!passed).then_some(name))
        .collect()
    }
}

fn finest_pair_levels<'a>(
    levels: &'a [PromotionLevelDiagnostics],
    ladder: &str,
) -> Result<(&'a PromotionLevelDiagnostics, &'a PromotionLevelDiagnostics), PortFlowError> {
    if levels.len() < 2 {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-9 {ladder} ladder needs at least 2 reduced levels, got {}",
            levels.len()
        )));
    }
    Ok((&levels[levels.len() - 2], &levels[levels.len() - 1]))
}

/// Gate the THD and noise-proxy finest-pair changes on both ladders.
///
/// Levels run coarse-to-fine; the phasor trio member (flow phasor) is
/// already gated on every window by
/// [`validate_port_flow_promotion`](super::convergence::validate_port_flow_promotion),
/// so only THD (floor-aware) and the proxy are checked here, on the finest
/// pair the plan names.
pub fn check_extended_changes(
    spatial: &[PromotionLevelDiagnostics],
    temporal: &[PromotionLevelDiagnostics],
    max_allowed: f64,
) -> Result<PromotionExtendedChanges, PortFlowError> {
    let (spatial_coarse, spatial_fine) = finest_pair_levels(spatial, "spatial")?;
    let (temporal_coarse, temporal_fine) = finest_pair_levels(temporal, "temporal")?;
    let spatial_thd = check_thd_finest_pair(
        "spatial THD",
        spatial_coarse.thd_fraction,
        spatial_fine.thd_fraction,
        max_allowed,
    )?;
    let spatial_noise_proxy = check_finest_pair(
        "spatial noise proxy",
        spatial_coarse.noise_proxy,
        spatial_fine.noise_proxy,
        max_allowed,
    )?;
    let temporal_thd = check_thd_finest_pair(
        "temporal THD",
        temporal_coarse.thd_fraction,
        temporal_fine.thd_fraction,
        max_allowed,
    )?;
    let temporal_noise_proxy = check_finest_pair(
        "temporal noise proxy",
        temporal_coarse.noise_proxy,
        temporal_fine.noise_proxy,
        max_allowed,
    )?;
    let passed = spatial_thd.passed
        && spatial_noise_proxy.passed
        && temporal_thd.passed
        && temporal_noise_proxy.passed;
    Ok(PromotionExtendedChanges {
        spatial_thd,
        spatial_noise_proxy,
        temporal_thd,
        temporal_noise_proxy,
        passed,
    })
}

// ---------------------------------------------------------------------------
// Ladder execution
// ---------------------------------------------------------------------------

/// Executed ladder product: evidence plus per-level reductions.
#[derive(Debug, Clone)]
pub struct Step9LadderOutcome {
    /// Runtime plan that was executed.
    pub plan: PortFlowRuntimePlan,
    /// Phasor promotion evidence (structurally validated).
    pub evidence: PortFlowPromotionEvidence,
    /// Per-level reductions, coarse-to-fine, spatial ladder.
    pub spatial_levels: Vec<PromotionLevelDiagnostics>,
    /// Per-level reductions, coarse-to-fine, temporal ladder.
    ///
    /// The fine entry reuses the spatial fine reduction (shared fine state),
    /// matching the evidence layout.
    pub temporal_levels: Vec<PromotionLevelDiagnostics>,
}

fn close_level(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= 1.0e-12 * scale
}

/// Execute the spatial + temporal ladders for one profile.
///
/// Runs the shared [`run_port_flow_promotion_with_observer`] runner with a
/// permissive structural gate (levels/orchestration only), reducing every
/// accepted trace through [`reduce_promotion_trace`]. The strict Step-9
/// gate is applied afterwards by the caller via
/// [`validate_port_flow_promotion`](super::convergence::validate_port_flow_promotion),
/// so a failing ladder still returns its evidence for the demotion report
/// instead of discarding the numbers. A non-finite or misaligned trace
/// fails closed here: that is a broken run, not a convergence verdict.
pub fn run_step9_ladder(
    context: &Tier2EvalContext,
    profile: &SplinePortProfile,
    ladder: &PromotionLadderConfig,
    frequency_hz: f64,
) -> Result<Step9LadderOutcome, PortFlowError> {
    ladder.validate()?;
    let case = context.case();
    let plan = ladder.to_runtime_plan(case, frequency_hz)?;
    let throat_area_m2 = min_profile_area_m2(profile)?;
    let reduction_inputs = PromotionReductionInputs {
        tail_cycles: ladder.tail_cycles,
        discarded_cycles: ladder.discarded_cycles(),
        box_volume_m3: case.box_volume_m3,
        density_kg_per_m3: case.density_kg_per_m3,
        sound_speed_m_per_s: case.sound_speed_m_per_s,
        throat_area_m2,
        duct_length_m: context.duct_length_m(),
    };
    // The runner is serial: build records and observer reductions land in
    // execution order and are matched to evidence levels below by (mesh, dt).
    // The builder pushes while the observer reads, so the records live
    // behind a `RefCell` (single-threaded use only).
    let build_records: std::cell::RefCell<Vec<((f64, f64), PromotionMeshStats)>> =
        std::cell::RefCell::new(Vec::new());
    let mut observed: Vec<(PortFlowRuntimeRefinement, PromotionLevelDiagnostics)> = Vec::new();
    let promotion = run_port_flow_promotion_with_observer(
        &plan,
        step9_structural_gate(),
        |refinement| {
            let fidelity = PortEvalFidelity {
                samples_per_cycle: refinement.samples_per_cycle,
                total_cycles: ladder.total_cycles,
                tail_cycles: ladder.tail_cycles,
                snapshot_interval_steps: 0,
                compression_reference: None,
            };
            let (case, stats) = build_promotion_case(
                context,
                profile,
                ladder.voltage_peak_v,
                refinement.frequency_hz,
                refinement.mesh_size_m,
                &fidelity,
            )?;
            build_records
                .borrow_mut()
                .push(((refinement.mesh_size_m, refinement.dt_s), stats));
            Ok(case)
        },
        |refinement, result| {
            let stats = build_records
                .borrow()
                .iter()
                .rev()
                .find(|((mesh, dt), _)| {
                    close_level(*mesh, refinement.mesh_size_m) && close_level(*dt, refinement.dt_s)
                })
                .map(|(_, stats)| *stats)
                .ok_or_else(|| {
                    PortFlowError::InvalidConfig(
                        "Step-9 observer saw a level the builder never recorded".to_string(),
                    )
                })?;
            observed.push((
                refinement,
                reduce_promotion_trace(refinement, result, reduction_inputs, stats)?,
            ));
            Ok(())
        },
    )?;
    let observed_for = |mesh_size_m: f64, dt_s: f64| {
        observed
            .iter()
            .find(|(refinement, _)| {
                close_level(refinement.mesh_size_m, mesh_size_m)
                    && close_level(refinement.dt_s, dt_s)
            })
            .map(|(_, diagnostics)| diagnostics.clone())
            .ok_or_else(|| {
                PortFlowError::InvalidConfig(format!(
                    "Step-9 ladder is missing reduced diagnostics for mesh {mesh_size_m} m, \
                     dt {dt_s} s"
                ))
            })
    };
    let mut spatial_levels = Vec::with_capacity(promotion.evidence.spatial_levels.len());
    for level in &promotion.evidence.spatial_levels {
        spatial_levels.push(observed_for(level.mesh_size_m, level.dt_s)?);
    }
    let mut temporal_levels = Vec::with_capacity(promotion.evidence.temporal_levels.len());
    for level in &promotion.evidence.temporal_levels {
        temporal_levels.push(observed_for(level.mesh_size_m, level.dt_s)?);
    }
    Ok(Step9LadderOutcome {
        plan,
        evidence: promotion.evidence,
        spatial_levels,
        temporal_levels,
    })
}

// ---------------------------------------------------------------------------
// Tier-1 oracle (Step-4 reference at campaign scale)
// ---------------------------------------------------------------------------

/// Archived Tier-1 transfer point for one profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tier1TransferPoint {
    /// Candidate identity: normalised free radii `u`.
    pub profile_normalized: Vec<f64>,
    /// Harmonic frequency in Hz.
    pub frequency_hz: f64,
    /// Port impedance `[re, im]` in Pa·s/m³ (`exp(-i·ω·t)`, RMS).
    pub port_impedance: [f64; 2],
    /// Compliance-derived `Q_port/Q_diaphragm` transfer `[re, im]`.
    pub flow_ratio: [f64; 2],
    /// Relative conditioning multiplier from complex `Z` perturbations to `H`.
    ///
    /// Values ≫ 1 (the campaign tuning sits near the Helmholtz pole) are
    /// why the `H` double ratio is reported, not gated.
    pub flow_ratio_condition_number: f64,
    /// True relative residual of the reference solve.
    pub relative_residual: f64,
    /// Strong/weak exterior-flux relative difference.
    pub flux_relative_difference: f64,
    /// Scalar pressure unknowns.
    pub scalar_dofs: usize,
    /// Mesh node count.
    pub mesh_nodes: usize,
    /// Mesh element count.
    pub mesh_elements: usize,
}

/// Run the Step-4 spline linear reference for one campaign-scale profile.
///
/// Meshes the campaign round port (Tier-1 envelope, reference mesh size)
/// around the profile and solves the viscothermal reference at
/// `frequency_hz`. The profile must live at campaign scale
/// (`R0 = 25 mm` bore); slot-scale Tier-2 profiles are off-envelope and
/// rejected by the geometry validation before any solve.
pub fn solve_step9_tier1(
    case: &PortOptimReferenceCase,
    profile: &SplinePortProfile,
    frequency_hz: f64,
    mesh_size_m: f64,
) -> Result<Tier1TransferPoint, PortFlowError> {
    case.validate()?;
    profile.validate()?;
    if !(frequency_hz.is_finite() && frequency_hz > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-9 Tier-1 oracle needs a finite positive frequency, got {frequency_hz}"
        )));
    }
    if !profile.normalized().iter().all(|u| u.is_finite()) {
        return Err(PortFlowError::InvalidConfig(
            "Step-9 Tier-1 oracle profile has a zero throat span, so its normalized identity is undefined"
                .to_string(),
        ));
    }
    let geometry = super::geometry3d::SplineRoundPortGeometry {
        duct_length_m: case.port_length_l_m,
        bore_radius_m: case.port_radius_r0_m,
        chamber_length_m: case.tier1_chamber_length_m,
        chamber_radius_m: case.tier1_chamber_radius_m,
        flare_length_m: STEP9_TIER1_FLARE_LENGTH_M,
        mesh_size_m,
        angular_sectors: None,
        profile: profile.clone(),
    };
    let mesh = build_spline_round_port_mesh(&geometry)?;
    let config = SplinePortLinearReferenceConfig {
        geometry,
        fluid: case.linear_reference_fluid(),
        box_volume_m3: case.box_volume_m3,
    };
    let transfer: ComplexPortTransfer =
        solve_spline_port_linear_reference(&mesh, &config, frequency_hz)?;
    Ok(Tier1TransferPoint {
        profile_normalized: profile.normalized(),
        frequency_hz: transfer.frequency_hz,
        port_impedance: complex_pair(transfer.port_impedance_pa_s_per_m3),
        flow_ratio: complex_pair(transfer.box_to_port_flow_ratio),
        flow_ratio_condition_number: transfer.box_to_port_flow_ratio_condition_number,
        relative_residual: transfer.relative_residual,
        flux_relative_difference: transfer.strong_weak_exterior_flux_relative_difference,
        scalar_dofs: transfer.scalar_dofs,
        mesh_nodes: mesh.num_nodes(),
        mesh_elements: mesh.elements.len(),
    })
}

// ---------------------------------------------------------------------------
// Low-drive linear transfer gate (final mesh)
// ---------------------------------------------------------------------------

/// Inputs for the Step-9 low-drive linear gate.
#[derive(Debug, Clone)]
pub struct Step9LinearGateInputs<'a> {
    /// Winner low-drive diagnostics on the final (fine) mesh.
    pub tier2_winner: &'a PromotionLevelDiagnostics,
    /// Straight low-drive diagnostics on the final mesh (same schedule).
    pub tier2_straight: &'a PromotionLevelDiagnostics,
    /// Step-4 oracle transfer for the winner campaign-scale profile.
    pub tier1_winner: &'a Tier1TransferPoint,
    /// Step-4 oracle transfer for the straight campaign-scale profile.
    pub tier1_straight: &'a Tier1TransferPoint,
}

/// Low-drive linear-gate verdict with its cross-tier evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step9LinearGateReport {
    /// Tier-2 drive frequency in Hz (slot tuning / 2).
    pub tier2_frequency_hz: f64,
    /// Tier-1 oracle frequency in Hz (campaign tuning).
    pub tier1_frequency_hz: f64,
    /// Tier-2 winner/straight impedance ratio `[re, im]`.
    pub tier2_impedance_ratio: [f64; 2],
    /// Tier-1 winner/straight impedance ratio `[re, im]`.
    pub tier1_impedance_ratio: [f64; 2],
    /// `|R_Z2 − R_Z1|/|R_Z1|` (bar [`STEP9_MAX_IMPEDANCE_RATIO_RELATIVE_ERROR`]).
    pub impedance_ratio_relative_error: f64,
    /// Tier-2 winner/straight flow-split ratio `[re, im]` (reported only).
    pub tier2_flow_ratio_ratio: [f64; 2],
    /// Tier-1 winner/straight flow-split ratio `[re, im]` (reported only).
    pub tier1_flow_ratio_ratio: [f64; 2],
    /// `|R_H2 − R_H1|/|R_H1|` (reported against
    /// [`STEP9_MAX_FLOW_RATIO_RELATIVE_ERROR`], not gated).
    pub flow_ratio_relative_error: f64,
    /// Winner Tier-2 transfer identity `|H − H(Z)|/|H(Z)|`.
    pub winner_transfer_identity_error: f64,
    /// Straight Tier-2 transfer identity `|H − H(Z)|/|H(Z)|`.
    pub straight_transfer_identity_error: f64,
    /// Winner low-drive THD (fraction).
    pub winner_thd: f64,
    /// Winner low-drive bulk Mach number.
    pub winner_mach: f64,
    /// Whether every check passes.
    pub passed: bool,
    /// Names of the failing checks (empty when [`Self::passed`]).
    pub failures: Vec<String>,
}

/// Gate the Tier-2 low-drive limit on the final mesh against the Step-4 oracle.
///
/// Compares the dimensionless winner/straight impedance ratio across tiers
/// (scale-free by construction), checks the Tier-2 transfer identity on
/// both low-drive runs, audits the Tier-1 reference solves, and enforces
/// the low-drive regime (THD, Mach) plus the Step-9 diagnostic bounds on
/// both Tier-2 runs. The flow-split double ratio is computed and reported
/// but excluded from the verdict (see
/// [`STEP9_MAX_FLOW_RATIO_RELATIVE_ERROR`]). Every failing check is named
/// in the report; a gate failure is evidence for demotion, not an error.
pub fn check_linear_transfer_gate(
    inputs: &Step9LinearGateInputs<'_>,
) -> Result<Step9LinearGateReport, PortFlowError> {
    let pair = |values: [f64; 2]| Complex64::new(values[0], values[1]);
    let tier2_impedance_ratio = complex_ratio(
        pair(inputs.tier2_winner.port_impedance),
        pair(inputs.tier2_straight.port_impedance),
    )?;
    let tier1_impedance_ratio = complex_ratio(
        pair(inputs.tier1_winner.port_impedance),
        pair(inputs.tier1_straight.port_impedance),
    )?;
    let impedance_ratio_relative_error =
        complex_relative_error(tier2_impedance_ratio, tier1_impedance_ratio)?;
    let tier2_flow_ratio_ratio = complex_ratio(
        pair(inputs.tier2_winner.flow_ratio_direct),
        pair(inputs.tier2_straight.flow_ratio_direct),
    )?;
    let tier1_flow_ratio_ratio = complex_ratio(
        pair(inputs.tier1_winner.flow_ratio),
        pair(inputs.tier1_straight.flow_ratio),
    )?;
    let flow_ratio_relative_error =
        complex_relative_error(tier2_flow_ratio_ratio, tier1_flow_ratio_ratio)?;

    let mut failures = Vec::new();
    let check = |failures: &mut Vec<String>, name: &str, value: f64, max: f64| {
        if !(value.is_finite() && value <= max) {
            failures.push(format!("{name}: {value:.6e} exceeds {max:.3e}"));
        }
    };
    check(
        &mut failures,
        "impedance-ratio error",
        impedance_ratio_relative_error,
        STEP9_MAX_IMPEDANCE_RATIO_RELATIVE_ERROR,
    );
    // NOTE: the H double ratio is reported, not gated (resonance-regime
    // mismatch; see STEP9_MAX_FLOW_RATIO_RELATIVE_ERROR).
    check(
        &mut failures,
        "winner transfer identity",
        inputs.tier2_winner.transfer_identity_relative_error,
        STEP9_MAX_TRANSFER_IDENTITY_RELATIVE_ERROR,
    );
    check(
        &mut failures,
        "straight transfer identity",
        inputs.tier2_straight.transfer_identity_relative_error,
        STEP9_MAX_TRANSFER_IDENTITY_RELATIVE_ERROR,
    );
    for (name, point) in [
        ("winner Tier-1", inputs.tier1_winner),
        ("straight Tier-1", inputs.tier1_straight),
    ] {
        check(
            &mut failures,
            &format!("{name} residual"),
            point.relative_residual,
            PortFlowPromotionGate::default().max_linear_relative_residual,
        );
        check(
            &mut failures,
            &format!("{name} flux difference"),
            point.flux_relative_difference,
            BASELINE_MAX_TIER1_FLUX_RELATIVE_DIFFERENCE,
        );
    }
    check(
        &mut failures,
        "winner THD",
        inputs.tier2_winner.thd_fraction,
        STEP9_MAX_THD_FRACTION,
    );
    check(
        &mut failures,
        "winner Mach",
        inputs.tier2_winner.peak_mach_bulk,
        STEP9_MAX_MACH,
    );
    for (name, level) in [
        ("winner low-drive", inputs.tier2_winner),
        ("straight low-drive", inputs.tier2_straight),
    ] {
        check(
            &mut failures,
            &format!("{name} cap continuity"),
            level.cap_continuity_relative_error,
            STEP9_MAX_CAP_CONTINUITY_RELATIVE_ERROR,
        );
        check(
            &mut failures,
            &format!("{name} energy algebraic"),
            level.energy_algebraic_relative_closure,
            PortFlowPromotionGate::default().max_energy_algebraic_relative_closure,
        );
        check(
            &mut failures,
            &format!("{name} energy physical"),
            level.energy_physical_relative_closure,
            BASELINE_MAX_ENERGY_PHYSICAL_CLOSURE,
        );
        check(
            &mut failures,
            &format!("{name} energy interface"),
            level.energy_interface_relative_residual,
            PortFlowPromotionGate::default().max_energy_interface_relative_residual,
        );
        check(
            &mut failures,
            &format!("{name} linear residual"),
            level.max_linear_relative_residual,
            PortFlowPromotionGate::default().max_linear_relative_residual,
        );
        check(
            &mut failures,
            &format!("{name} box conservation"),
            level.max_box_conservation_relative_residual,
            PortFlowPromotionGate::default().max_box_conservation_relative_residual,
        );
    }
    let passed = failures.is_empty();
    Ok(Step9LinearGateReport {
        tier2_frequency_hz: inputs.tier2_winner.frequency_hz,
        tier1_frequency_hz: inputs.tier1_winner.frequency_hz,
        tier2_impedance_ratio: complex_pair(tier2_impedance_ratio),
        tier1_impedance_ratio: complex_pair(tier1_impedance_ratio),
        impedance_ratio_relative_error,
        tier2_flow_ratio_ratio: complex_pair(tier2_flow_ratio_ratio),
        tier1_flow_ratio_ratio: complex_pair(tier1_flow_ratio_ratio),
        flow_ratio_relative_error,
        winner_transfer_identity_error: inputs.tier2_winner.transfer_identity_relative_error,
        straight_transfer_identity_error: inputs.tier2_straight.transfer_identity_relative_error,
        winner_thd: inputs.tier2_winner.thd_fraction,
        winner_mach: inputs.tier2_winner.peak_mach_bulk,
        passed,
        failures,
    })
}

// ---------------------------------------------------------------------------
// Step-9 evidence archive
// ---------------------------------------------------------------------------

/// Serde mirror of [`PortFlowPromotionGate`] (thresholds applied).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionGateMirror {
    /// Minimum spatial levels.
    pub min_spatial_levels: usize,
    /// Minimum temporal levels.
    pub min_temporal_levels: usize,
    /// Spatial observable-change bar.
    pub max_spatial_observable_relative_change: f64,
    /// Temporal observable-change bar.
    pub max_temporal_observable_relative_change: f64,
    /// Cross-ladder observable-change bar.
    pub max_cross_ladder_observable_relative_change: f64,
    /// Linear-residual bar.
    pub max_linear_relative_residual: f64,
    /// Box-conservation bar.
    pub max_box_conservation_relative_residual: f64,
    /// Cap-continuity bar.
    pub max_cap_continuity_relative_error: f64,
    /// Algebraic-energy bar.
    pub max_energy_algebraic_relative_closure: f64,
    /// Physical-energy bar.
    pub max_energy_physical_relative_closure: f64,
    /// Interface-energy bar.
    pub max_energy_interface_relative_residual: f64,
}

impl From<&PortFlowPromotionGate> for PromotionGateMirror {
    fn from(gate: &PortFlowPromotionGate) -> Self {
        Self {
            min_spatial_levels: gate.min_spatial_levels,
            min_temporal_levels: gate.min_temporal_levels,
            max_spatial_observable_relative_change: gate.max_spatial_observable_relative_change,
            max_temporal_observable_relative_change: gate.max_temporal_observable_relative_change,
            max_cross_ladder_observable_relative_change: gate
                .max_cross_ladder_observable_relative_change,
            max_linear_relative_residual: gate.max_linear_relative_residual,
            max_box_conservation_relative_residual: gate.max_box_conservation_relative_residual,
            max_cap_continuity_relative_error: gate.max_cap_continuity_relative_error,
            max_energy_algebraic_relative_closure: gate.max_energy_algebraic_relative_closure,
            max_energy_physical_relative_closure: gate.max_energy_physical_relative_closure,
            max_energy_interface_relative_residual: gate.max_energy_interface_relative_residual,
        }
    }
}

/// Serde mirror of [`PortFlowPromotionReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionReportMirror {
    /// Drive frequency in Hz.
    pub frequency_hz: f64,
    /// Spatial level count.
    pub spatial_levels: usize,
    /// Temporal level count.
    pub temporal_levels: usize,
    /// Worst spatial-window observable change.
    pub max_spatial_observable_relative_change: f64,
    /// Worst temporal-window observable change.
    pub max_temporal_observable_relative_change: f64,
    /// Fine-state cross-ladder disagreement.
    pub cross_ladder_observable_relative_change: f64,
    /// Promoted (fine) mesh size in m.
    pub promoted_mesh_size_m: f64,
    /// Promoted (fine) timestep in s.
    pub promoted_dt_s: f64,
    /// Promoted port-flow phasor `[re, im]` in m³/s.
    pub promoted_port_flow: [f64; 2],
}

impl From<&PortFlowPromotionReport> for PromotionReportMirror {
    fn from(report: &PortFlowPromotionReport) -> Self {
        Self {
            frequency_hz: report.frequency_hz,
            spatial_levels: report.spatial_levels,
            temporal_levels: report.temporal_levels,
            max_spatial_observable_relative_change: report.max_spatial_observable_relative_change,
            max_temporal_observable_relative_change: report.max_temporal_observable_relative_change,
            cross_ladder_observable_relative_change: report.cross_ladder_observable_relative_change,
            promoted_mesh_size_m: report.promoted_mesh_size_m,
            promoted_dt_s: report.promoted_dt_s,
            promoted_port_flow: [
                report.promoted_port_flow.real,
                report.promoted_port_flow.imaginary,
            ],
        }
    }
}

/// Regenerated-mesh vs archived-mesh cross-check (report-only, ungated).
///
/// The ladders run on regenerated validation meshes while Steps 0/7 ran on
/// the deterministic slot baseline (straight) and morphed meshes (winner);
/// these differences measure mesh-realisation sensitivity at production
/// fidelity. Large values are a finding, not a gate failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshIndependenceCheck {
    /// Straight low-drive `Q_rms` on the regenerated fine mesh in m³/s.
    pub straight_regenerated_q_rms_m3s: f64,
    /// Step-0 archived straight low-drive `Q_rms` in m³/s.
    pub step0_archived_q_rms_m3s: f64,
    /// Relative difference of the straight pair.
    pub straight_relative_difference: f64,
    /// Winner `V_max` `Q_rms` on the regenerated fine mesh in m³/s.
    pub winner_regenerated_q_rms_m3s: f64,
    /// Step-7 archived winner `V_max` `Q_rms` in m³/s.
    pub step7_archived_q_rms_m3s: f64,
    /// Relative difference of the winner pair.
    pub winner_relative_difference: f64,
}

impl MeshIndependenceCheck {
    /// Compare regenerated-mesh flows against the Step-0/7 archives.
    pub fn compare(
        straight_regenerated_q_rms_m3s: f64,
        step0_archived_q_rms_m3s: f64,
        winner_regenerated_q_rms_m3s: f64,
        step7_archived_q_rms_m3s: f64,
    ) -> Result<Self, PortFlowError> {
        for (name, value) in [
            ("straight regenerated Q_rms", straight_regenerated_q_rms_m3s),
            ("Step-0 archived Q_rms", step0_archived_q_rms_m3s),
            ("winner regenerated Q_rms", winner_regenerated_q_rms_m3s),
            ("Step-7 archived Q_rms", step7_archived_q_rms_m3s),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "mesh-independence check needs finite positive {name}, got {value}"
                )));
            }
        }
        Ok(Self {
            straight_regenerated_q_rms_m3s,
            step0_archived_q_rms_m3s,
            straight_relative_difference: (straight_regenerated_q_rms_m3s
                - step0_archived_q_rms_m3s)
                .abs()
                / step0_archived_q_rms_m3s,
            winner_regenerated_q_rms_m3s,
            step7_archived_q_rms_m3s,
            winner_relative_difference: (winner_regenerated_q_rms_m3s - step7_archived_q_rms_m3s)
                .abs()
                / step7_archived_q_rms_m3s,
        })
    }
}

/// Step-9 promotion verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Step9Verdict {
    /// Ladders, extended THD/proxy gates, and the linear gate all pass:
    /// the winner is converged at production fidelity.
    Pass,
    /// At least one gate fails: every Step-7/8 headline number stays
    /// labelled unconverged, indicative only. The reasons name the failing
    /// gates; this is evidence, not an error.
    DemoteUnconvergedIndicativeOnly {
        /// Failing-gate descriptions.
        reasons: Vec<String>,
    },
}

impl Step9Verdict {
    /// Whether the verdict promotes the winner.
    pub fn passed(&self) -> bool {
        matches!(self, Self::Pass)
    }
}

/// Pure verdict logic: pass iff the phasor promotion (changes and
/// per-level diagnostics), the extended THD/proxy gates, and the linear
/// gate all pass.
pub fn evaluate_step9_verdict(
    promotion: &PromotionReportMirror,
    gate: &PromotionGateMirror,
    extended: &PromotionExtendedChanges,
    linear_gate: &Step9LinearGateReport,
    spatial: &[PromotionLevelDiagnostics],
    temporal: &[PromotionLevelDiagnostics],
) -> Step9Verdict {
    let mut reasons = Vec::new();
    for (name, value, max) in [
        (
            "spatial phasor change",
            promotion.max_spatial_observable_relative_change,
            gate.max_spatial_observable_relative_change,
        ),
        (
            "temporal phasor change",
            promotion.max_temporal_observable_relative_change,
            gate.max_temporal_observable_relative_change,
        ),
        (
            "cross-ladder phasor change",
            promotion.cross_ladder_observable_relative_change,
            gate.max_cross_ladder_observable_relative_change,
        ),
    ] {
        if !(value.is_finite() && value <= max) {
            reasons.push(format!("{name}: {value:.6e} exceeds {max:.3e}"));
        }
    }
    for (ladder, levels) in [("spatial", spatial), ("temporal", temporal)] {
        for (index, level) in levels.iter().enumerate() {
            for (name, value, max) in [
                (
                    "cap continuity",
                    level.cap_continuity_relative_error,
                    gate.max_cap_continuity_relative_error,
                ),
                (
                    "energy algebraic",
                    level.energy_algebraic_relative_closure,
                    gate.max_energy_algebraic_relative_closure,
                ),
                (
                    "energy physical",
                    level.energy_physical_relative_closure,
                    gate.max_energy_physical_relative_closure,
                ),
                (
                    "energy interface",
                    level.energy_interface_relative_residual,
                    gate.max_energy_interface_relative_residual,
                ),
                (
                    "linear residual",
                    level.max_linear_relative_residual,
                    gate.max_linear_relative_residual,
                ),
                (
                    "box conservation",
                    level.max_box_conservation_relative_residual,
                    gate.max_box_conservation_relative_residual,
                ),
            ] {
                if !(value.is_finite() && value <= max) {
                    reasons.push(format!(
                        "{ladder} level {index} {name}: {value:.6e} exceeds {max:.3e}"
                    ));
                }
            }
        }
    }
    for (name, change) in [
        ("spatial THD", &extended.spatial_thd),
        ("spatial noise proxy", &extended.spatial_noise_proxy),
        ("temporal THD", &extended.temporal_thd),
        ("temporal noise proxy", &extended.temporal_noise_proxy),
    ] {
        if !change.passed {
            reasons.push(format!(
                "{name} finest-pair change: {:.6e} exceeds {:.3e}",
                change.relative_change, change.max_allowed
            ));
        }
    }
    if !linear_gate.passed {
        reasons.push(format!(
            "linear transfer gate: {}",
            linear_gate.failures.join("; ")
        ));
    }
    if reasons.is_empty() {
        Step9Verdict::Pass
    } else {
        Step9Verdict::DemoteUnconvergedIndicativeOnly { reasons }
    }
}

/// Step-9 evidence archive: ladders, gates, and the promotion verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step9Report {
    /// Must equal [`STEP9_REPORT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Model lineage; must equal [`STEP9_REPORT_PROVENANCE`].
    pub provenance: String,
    /// Frozen reference case the ladders ran on.
    pub case: PortOptimReferenceCase,
    /// Campaign tuning target in Hz (Tier-0 duct tuning of straight).
    pub f_target_hz: f64,
    /// Primary control-point count (5).
    pub n_control_points: usize,
    /// Winning `u` carried over from the Step-7 archive.
    pub winner_u: Vec<f64>,
    /// Ladder drive level in V peak (`V_max`).
    pub ladder_voltage_peak_v: f64,
    /// Linear-gate drive level in V peak (lowest campaign drive).
    pub low_drive_voltage_peak_v: f64,
    /// Executed ladder schedule.
    pub ladder_config: PromotionLadderConfig,
    /// Phasor promotion evidence (spatial + temporal levels).
    pub promotion_evidence: PortFlowPromotionEvidence,
    /// Thresholds applied to the evidence.
    pub promotion_gate: PromotionGateMirror,
    /// Phasor promotion summary.
    pub promotion: PromotionReportMirror,
    /// Per-level reductions, coarse-to-fine, spatial ladder.
    pub spatial_levels: Vec<PromotionLevelDiagnostics>,
    /// Per-level reductions, coarse-to-fine, temporal ladder.
    pub temporal_levels: Vec<PromotionLevelDiagnostics>,
    /// THD + noise-proxy finest-pair verdicts.
    pub extended: PromotionExtendedChanges,
    /// Low-drive linear-gate verdict with its Tier-1 evidence.
    pub linear_gate: Step9LinearGateReport,
    /// Winner low-drive diagnostics on the final mesh.
    pub low_drive_winner: PromotionLevelDiagnostics,
    /// Straight low-drive diagnostics on the final mesh.
    pub low_drive_straight: PromotionLevelDiagnostics,
    /// Tier-1 oracle transfer for the winner campaign-scale profile.
    pub tier1_winner: Tier1TransferPoint,
    /// Tier-1 oracle transfer for the straight campaign-scale profile.
    pub tier1_straight: Tier1TransferPoint,
    /// Regenerated-vs-archived mesh cross-check (report-only).
    pub mesh_independence: MeshIndependenceCheck,
    /// Promotion verdict.
    pub verdict: Step9Verdict,
    /// Free-form leg notes (schedule, verdict reading, absolute context).
    pub notes: String,
}

impl Step9Report {
    /// Serialize the report to pretty JSON.
    pub fn to_json_string(&self) -> Result<String, CampaignArchiverError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))
    }

    /// Write the report to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), CampaignArchiverError> {
        std::fs::write(path, self.to_json_string()?)
            .map_err(|error| CampaignArchiverError::Io(error.to_string()))
    }

    /// Parse report JSON, rejecting schema or provenance drift loudly.
    pub fn from_json_str(raw: &str) -> Result<Self, CampaignArchiverError> {
        let report: Step9Report = serde_json::from_str(raw)
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))?;
        if report.schema_version != STEP9_REPORT_SCHEMA_VERSION {
            return Err(CampaignArchiverError::Json(format!(
                "unsupported Step-9 report schema {}, expected {STEP9_REPORT_SCHEMA_VERSION}",
                report.schema_version
            )));
        }
        if report.provenance != STEP9_REPORT_PROVENANCE {
            return Err(CampaignArchiverError::Json(format!(
                "Step-9 report provenance drift: {}",
                report.provenance
            )));
        }
        Ok(report)
    }

    /// Structural + coherence validation: schedule coverage, evidence shape,
    /// and verdict agreement with the archived gate outputs.
    pub fn validate(&self) -> Result<(), CampaignArchiverError> {
        let bad = |message: String| CampaignArchiverError::Json(message);
        if self.spatial_levels.len() < 3 {
            return Err(bad(format!(
                "Step 9 needs at least 3 reduced spatial levels, got {}",
                self.spatial_levels.len()
            )));
        }
        if self.temporal_levels.len() < 2 {
            return Err(bad(format!(
                "Step 9 needs at least 2 reduced temporal levels, got {}",
                self.temporal_levels.len()
            )));
        }
        if self.promotion_evidence.schema_version != PORT_FLOW_PROMOTION_SCHEMA_VERSION {
            return Err(bad(format!(
                "Step-9 promotion evidence schema is {}, expected {PORT_FLOW_PROMOTION_SCHEMA_VERSION}",
                self.promotion_evidence.schema_version
            )));
        }
        if self.promotion_evidence.spatial_levels.len() != self.spatial_levels.len()
            || self.promotion_evidence.temporal_levels.len() != self.temporal_levels.len()
        {
            return Err(bad(
                "Step-9 promotion evidence and reduced levels disagree on ladder shape".to_string(),
            ));
        }
        for (evidence, reduced) in self
            .promotion_evidence
            .spatial_levels
            .iter()
            .chain(self.promotion_evidence.temporal_levels.iter())
            .zip(
                self.spatial_levels
                    .iter()
                    .chain(self.temporal_levels.iter()),
            )
        {
            if !close_level(evidence.mesh_size_m, reduced.mesh_size_m)
                || !close_level(evidence.dt_s, reduced.dt_s)
            {
                return Err(bad(
                    "Step-9 promotion evidence and reduced levels disagree on a (mesh, dt) schedule"
                        .to_string(),
                ));
            }
        }
        if self.promotion.spatial_levels != self.spatial_levels.len()
            || self.promotion.temporal_levels != self.temporal_levels.len()
        {
            return Err(bad(
                "Step-9 promotion summary disagrees with the reduced ladder shape".to_string(),
            ));
        }
        let expected_extended_passed = self.extended.spatial_thd.passed
            && self.extended.spatial_noise_proxy.passed
            && self.extended.temporal_thd.passed
            && self.extended.temporal_noise_proxy.passed;
        if self.extended.passed != expected_extended_passed {
            return Err(bad(
                "Step-9 extended verdict disagrees with its four finest-pair checks".to_string(),
            ));
        }
        if self.linear_gate.passed != self.linear_gate.failures.is_empty() {
            return Err(bad(
                "Step-9 linear verdict disagrees with its failure list".to_string()
            ));
        }
        let expected = evaluate_step9_verdict(
            &self.promotion,
            &self.promotion_gate,
            &self.extended,
            &self.linear_gate,
            &self.spatial_levels,
            &self.temporal_levels,
        );
        if expected.passed() != self.verdict.passed() {
            return Err(bad(format!(
                "Step-9 verdict {:?} disagrees with the archived gate outputs (recompute: {:?})",
                self.verdict, expected
            )));
        }
        if let Step9Verdict::DemoteUnconvergedIndicativeOnly { reasons } = &self.verdict
            && reasons.is_empty()
        {
            return Err(bad(
                "Step-9 demotion verdict needs at least one named reason".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    use crate::port_flow::coupled::{CoupledEnergyStep, CoupledPortResult};

    fn synthetic_trace(
        samples_per_cycle: usize,
        tail_cycles: usize,
        discarded_cycles: usize,
        frequency_hz: f64,
    ) -> (PortFlowRuntimeRefinement, CoupledPortResult) {
        let dt_s = 1.0 / (frequency_hz * samples_per_cycle as f64);
        let sample_count = (discarded_cycles + tail_cycles) * samples_per_cycle;
        let mut result = CoupledPortResult {
            time_s: Vec::with_capacity(sample_count),
            port_flow_m3s: Vec::with_capacity(sample_count),
            exterior_cap_flow_m3s: Vec::with_capacity(sample_count),
            box_cap_outward_flow_m3s: Vec::with_capacity(sample_count),
            diaphragm_volume_velocity_m3s: Vec::with_capacity(sample_count),
            box_pressure_pa: Vec::with_capacity(sample_count),
            displacement_m: vec![0.0; sample_count],
            current_a: vec![0.0; sample_count],
            box_conservation_residual_pa: vec![1.0e-10; sample_count],
            energy_steps: vec![CoupledEnergyStep::default(); sample_count],
            linear_solves: Vec::new(),
            accepted_linear_relative_residual: vec![1.0e-10; sample_count],
            field_snapshots: Vec::new(),
            final_flow_state: None,
            steps_completed: sample_count,
        };
        for index in 0..sample_count {
            let phase = TAU * frequency_hz * index as f64 * dt_s;
            let flow = 1.0e-5 * phase.sin() + 1.0e-8 * (2.0 * phase).sin();
            result.time_s.push((index + 1) as f64 * dt_s);
            result.port_flow_m3s.push(flow);
            result.exterior_cap_flow_m3s.push(flow);
            result.box_cap_outward_flow_m3s.push(-flow);
            result.diaphragm_volume_velocity_m3s.push(1.5 * flow);
            result.box_pressure_pa.push(9000.0 * flow);
        }
        let refinement = PortFlowRuntimeRefinement {
            mesh_size_m: 0.01,
            dt_s,
            frequency_hz,
            samples_per_cycle,
            tail_cycles,
            discarded_cycles,
        };
        (refinement, result)
    }

    fn reduction_inputs() -> PromotionReductionInputs {
        PromotionReductionInputs {
            tail_cycles: 2,
            discarded_cycles: 1,
            box_volume_m3: 0.005,
            density_kg_per_m3: 1.2,
            sound_speed_m_per_s: 343.0,
            throat_area_m2: PI * 0.01f64.powi(2),
            duct_length_m: 0.1,
        }
    }

    fn mesh_stats() -> PromotionMeshStats {
        PromotionMeshStats {
            mesh_size_m: 0.01,
            mesh_nodes: 100,
            mesh_elements: 150,
            min_triangle_area_m2: 1.0e-6,
        }
    }

    #[test]
    fn step9_gate_carries_plan_thresholds() {
        let gate = step9_promotion_gate();
        assert_eq!(gate.min_spatial_levels, 3);
        assert_eq!(gate.min_temporal_levels, 2);
        assert_eq!(
            gate.max_spatial_observable_relative_change,
            STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE
        );
        assert_eq!(
            gate.max_temporal_observable_relative_change,
            STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE
        );
        assert_eq!(
            gate.max_cross_ladder_observable_relative_change,
            STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE
        );
        assert_eq!(
            gate.max_cap_continuity_relative_error,
            STEP9_MAX_CAP_CONTINUITY_RELATIVE_ERROR
        );
        assert_eq!(
            gate.max_energy_physical_relative_closure,
            BASELINE_MAX_ENERGY_PHYSICAL_CLOSURE
        );
        // Untouched diagnostics stay at the shared defaults.
        let default = PortFlowPromotionGate::default();
        assert_eq!(
            gate.max_linear_relative_residual,
            default.max_linear_relative_residual
        );
        assert_eq!(
            gate.max_energy_algebraic_relative_closure,
            default.max_energy_algebraic_relative_closure
        );
    }

    #[test]
    fn production_ladder_matches_plan_schedule() {
        let case = PortOptimReferenceCase::default();
        let ladder = PromotionLadderConfig::production(&case).expect("production ladder builds");
        assert_eq!(
            ladder.spatial_mesh_sizes_m,
            STEP9_SPATIAL_MESH_SIZES_M.to_vec()
        );
        assert_eq!(
            ladder.temporal_samples_per_cycle,
            STEP9_TEMPORAL_SAMPLES_PER_CYCLE.to_vec()
        );
        assert_eq!(ladder.total_cycles, case.tier2_total_cycles);
        assert_eq!(ladder.tail_cycles, case.tier2_tail_cycles);
        assert_eq!(ladder.discarded_cycles(), 32);
        assert_eq!(ladder.fine_mesh_size_m(), 0.01);
        assert_eq!(ladder.fine_samples_per_cycle(), 504);
        assert!(ladder.voltage_peak_v > 0.0);

        let frequency_hz = case
            .tier2_drive_frequency_hz()
            .expect("Tier-2 drive resolves");
        let plan = ladder
            .to_runtime_plan(&case, frequency_hz)
            .expect("runtime plan builds");
        assert_eq!(plan.spatial_mesh_sizes_m.len(), 3);
        assert_eq!(plan.temporal_dt_s.len(), 2);
        // Timesteps divide the drive period integrally (runtime requirement).
        for (dt, spc) in plan
            .temporal_dt_s
            .iter()
            .zip(ladder.temporal_samples_per_cycle.iter())
        {
            let round_trip = 1.0 / (frequency_hz * *spc as f64);
            assert!((dt - round_trip).abs() <= 1.0e-15);
        }
        assert!(plan.temporal_dt_s[1] < plan.temporal_dt_s[0]);
    }

    #[test]
    fn ladder_config_rejects_bad_schedules() {
        let case = PortOptimReferenceCase::default();
        let good = PromotionLadderConfig::production(&case).unwrap();
        let mut bad = good.clone();
        bad.spatial_mesh_sizes_m.truncate(2);
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.spatial_mesh_sizes_m = vec![0.01, 0.014, 0.02];
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.temporal_samples_per_cycle = vec![252];
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.temporal_samples_per_cycle = vec![252, 126];
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.tail_cycles = bad.total_cycles;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.voltage_peak_v = 0.0;
        assert!(bad.validate().is_err());
        assert!(good.temporal_dt_s(f64::NAN).is_err());
    }

    #[test]
    fn finest_pair_math_and_failures() {
        let change = check_finest_pair("proxy", 1.0, 1.01, 0.02).unwrap();
        assert!(change.passed);
        assert!(!change.passed_by_floor);
        assert_eq!(change.floor, None);
        assert!((change.relative_change - 0.01 / 1.01).abs() < 1.0e-12);
        let change = check_finest_pair("proxy", 1.0, 1.05, 0.02).unwrap();
        assert!(!change.passed);
        assert!(check_finest_pair("proxy", 1.0, 0.0, 0.02).is_err());
        assert!(check_finest_pair("proxy", f64::NAN, 1.0, 0.02).is_err());
        assert!(check_finest_pair("proxy", 1.0, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn thd_floor_passes_unresolvable_pairs_only() {
        // Resolved THD obeys the bar.
        let change = check_thd_finest_pair("thd", 9.0e-3, 9.1e-3, 0.02).unwrap();
        assert!(change.passed);
        assert!(!change.passed_by_floor);
        let change = check_thd_finest_pair("thd", 9.0e-3, 1.2e-2, 0.02).unwrap();
        assert!(!change.passed);
        // Floor-level scatter passes by floor despite a huge relative change.
        let change = check_thd_finest_pair("thd", 4.6e-7, 1.0e-5, 0.02).unwrap();
        assert!(change.passed);
        assert!(change.passed_by_floor);
        assert_eq!(change.floor, Some(STEP9_THD_RESOLUTION_FLOOR));
        // One level above the floor re-arms the bar.
        let change = check_thd_finest_pair("thd", 4.6e-7, 2.0e-4, 0.02).unwrap();
        assert!(!change.passed);
        assert!(!change.passed_by_floor);
    }

    fn synthetic_diagnostics(
        mesh_size_m: f64,
        dt_s: f64,
        samples_per_cycle: usize,
        thd_fraction: f64,
        noise_proxy: f64,
    ) -> PromotionLevelDiagnostics {
        PromotionLevelDiagnostics {
            frequency_hz: 159.0,
            mesh_size_m,
            dt_s,
            samples_per_cycle,
            steps_completed: 10080,
            tail_samples: 2016,
            mesh_nodes: 1000,
            mesh_elements: 1500,
            port_flow: [1.0e-5, -2.0e-6],
            box_pressure: [0.09, 0.01],
            diaphragm_flow: [1.5e-5, -3.0e-6],
            port_impedance: [9000.0, 1000.0],
            flow_ratio_direct: [0.9, 0.1],
            flow_ratio_from_impedance: [0.901, 0.099],
            transfer_identity_relative_error: 0.002,
            cap_continuity_relative_error: 1.0e-5,
            energy_algebraic_relative_closure: 1.0e-12,
            energy_physical_relative_closure: 0.008,
            energy_interface_relative_residual: 1.0e-12,
            energy_scale_j: 1.0e-6,
            max_linear_relative_residual: 1.0e-12,
            max_box_conservation_relative_residual: 1.0e-12,
            port_flow_rms_m3s: 7.3e-6,
            port_flow_fundamental_amplitude_m3s: 1.03e-5,
            box_pressure_amplitude_pa: 0.093,
            thd_fraction,
            noise_proxy,
            peak_mach_bulk: 0.001,
        }
    }

    #[test]
    fn extended_changes_gate_names_failures() {
        // Resolved THD (above the floor) obeys the bar on the finest pair.
        let spatial = vec![
            synthetic_diagnostics(0.016, 1.25e-5, 504, 9.0e-3, 1.0),
            synthetic_diagnostics(0.012, 1.25e-5, 504, 9.05e-3, 1.005),
            synthetic_diagnostics(0.01, 1.25e-5, 504, 9.1e-3, 1.01),
        ];
        let temporal = vec![
            synthetic_diagnostics(0.01, 2.5e-5, 252, 9.2e-3, 1.02),
            synthetic_diagnostics(0.01, 1.25e-5, 504, 9.1e-3, 1.01),
        ];
        let extended = check_extended_changes(&spatial, &temporal, 0.02).unwrap();
        assert!(extended.passed);
        assert!(extended.failures().is_empty());
        assert!(!extended.spatial_thd.passed_by_floor);

        let mut bad_spatial = spatial.clone();
        bad_spatial[2].thd_fraction = 1.2e-2;
        let extended = check_extended_changes(&bad_spatial, &temporal, 0.02).unwrap();
        assert!(!extended.passed);
        assert_eq!(extended.failures(), vec!["spatial THD"]);

        // Floor-level THD scatter passes by floor.
        let mut floor_spatial = spatial.clone();
        floor_spatial[1].thd_fraction = 4.6e-7;
        floor_spatial[2].thd_fraction = 1.0e-5;
        let extended = check_extended_changes(&floor_spatial, &temporal, 0.02).unwrap();
        assert!(extended.passed);
        assert!(extended.spatial_thd.passed_by_floor);

        let short: Vec<PromotionLevelDiagnostics> = vec![];
        assert!(check_extended_changes(&short, &temporal, 0.02).is_err());
    }

    #[test]
    fn trace_reduction_recovers_synthetic_tail() {
        let (refinement, result) = synthetic_trace(16, 2, 1, 100.0);
        let diagnostics =
            reduce_promotion_trace(refinement, &result, reduction_inputs(), mesh_stats())
                .expect("synthetic settled tail reduces");
        assert_eq!(diagnostics.mesh_size_m, 0.01);
        assert_eq!(diagnostics.samples_per_cycle, 16);
        assert_eq!(diagnostics.tail_samples, 32);
        assert_eq!(diagnostics.mesh_nodes, 100);
        assert!(diagnostics.port_flow_rms_m3s > 0.0);
        assert!(diagnostics.port_flow_fundamental_amplitude_m3s > 0.0);
        assert!(diagnostics.thd_fraction.is_finite());
        assert!(diagnostics.noise_proxy > 0.0);
        assert!(diagnostics.peak_mach_bulk > 0.0);
        assert!(diagnostics.transfer_identity_relative_error.is_finite());
        // The second-harmonic content (1e-3 relative) dominates the THD.
        assert!(diagnostics.thd_fraction > 1.0e-4);

        // A truncated trace fails closed, never silently.
        let (_, mut short) = synthetic_trace(16, 2, 1, 100.0);
        short.port_flow_m3s.pop();
        assert!(
            reduce_promotion_trace(refinement, &short, reduction_inputs(), mesh_stats()).is_err()
        );
    }

    fn tier1_point(u0: f64, impedance: [f64; 2], flow_ratio: [f64; 2]) -> Tier1TransferPoint {
        Tier1TransferPoint {
            profile_normalized: vec![u0, 0.0],
            frequency_hz: 92.6,
            port_impedance: impedance,
            flow_ratio,
            flow_ratio_condition_number: 62.0,
            relative_residual: 1.0e-10,
            flux_relative_difference: 1.0e-4,
            scalar_dofs: 10000,
            mesh_nodes: 5000,
            mesh_elements: 20000,
        }
    }

    fn linear_inputs() -> (
        PromotionLevelDiagnostics,
        PromotionLevelDiagnostics,
        Tier1TransferPoint,
        Tier1TransferPoint,
    ) {
        let mut winner = synthetic_diagnostics(0.01, 2.5e-5, 252, 8.6e-6, 1.0);
        winner.port_impedance = [9090.0, 1010.0];
        winner.flow_ratio_direct = [0.909, 0.101];
        let mut straight = synthetic_diagnostics(0.01, 2.5e-5, 252, 8.6e-6, 1.0);
        straight.port_impedance = [9000.0, 1000.0];
        straight.flow_ratio_direct = [0.9, 0.1];
        let tier1_winner = tier1_point(0.015, [36000.0, 4000.0], [0.91, 0.101]);
        let tier1_straight = tier1_point(0.0, [35643.0, 3960.0], [0.9, 0.1]);
        (winner, straight, tier1_winner, tier1_straight)
    }

    #[test]
    fn linear_transfer_gate_accepts_consistent_tiers() {
        let (winner, straight, tier1_winner, tier1_straight) = linear_inputs();
        let report = check_linear_transfer_gate(&Step9LinearGateInputs {
            tier2_winner: &winner,
            tier2_straight: &straight,
            tier1_winner: &tier1_winner,
            tier1_straight: &tier1_straight,
        })
        .expect("consistent tiers gate");
        assert!(report.passed);
        assert!(report.failures.is_empty());
        assert!(report.impedance_ratio_relative_error < 0.03);
        assert!(report.flow_ratio_relative_error < 0.03);
    }

    #[test]
    fn linear_transfer_gate_names_each_failure() {
        let (winner, straight, tier1_winner, tier1_straight) = linear_inputs();
        let gate = |winner: &PromotionLevelDiagnostics,
                    straight: &PromotionLevelDiagnostics,
                    tier1_winner: &Tier1TransferPoint,
                    tier1_straight: &Tier1TransferPoint| {
            check_linear_transfer_gate(&Step9LinearGateInputs {
                tier2_winner: winner,
                tier2_straight: straight,
                tier1_winner,
                tier1_straight,
            })
            .expect("gate evaluates")
        };
        // Tier disagreement on the impedance ratio.
        let mut bad_tier1 = tier1_winner.clone();
        bad_tier1.port_impedance = [40000.0, 4000.0];
        let report = gate(&winner, &straight, &bad_tier1, &tier1_straight);
        assert!(!report.passed);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("impedance-ratio"))
        );
        // Nonlinear low-drive regime.
        let mut bad_winner = winner.clone();
        bad_winner.thd_fraction = 0.01;
        let report = gate(&bad_winner, &straight, &tier1_winner, &tier1_straight);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("THD"))
        );
        // Failed Tier-1 audit.
        let mut bad_tier1 = tier1_winner.clone();
        bad_tier1.relative_residual = 1.0e-6;
        let report = gate(&winner, &straight, &bad_tier1, &tier1_straight);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("residual"))
        );
        // Unhealthy low-drive trace.
        let mut bad_winner = winner.clone();
        bad_winner.cap_continuity_relative_error = 1.0e-2;
        let report = gate(&bad_winner, &straight, &tier1_winner, &tier1_straight);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("cap continuity"))
        );
        // H-ratio disagreement is reported, never gated (resonance-regime
        // mismatch by design).
        let mut resonant_tier1 = tier1_winner.clone();
        resonant_tier1.flow_ratio = [60.0, 70.0];
        let report = gate(&winner, &straight, &resonant_tier1, &tier1_straight);
        assert!(
            report.passed,
            "H disagreement must not fail: {:?}",
            report.failures
        );
        assert!(report.flow_ratio_relative_error > 0.03);
    }

    #[test]
    fn mesh_independence_math() {
        let check = MeshIndependenceCheck::compare(7.31e-6, 7.30e-6, 7.28e-4, 7.31e-4).unwrap();
        assert!((check.straight_relative_difference - 0.01 / 7.30).abs() < 1.0e-9);
        assert!(check.winner_relative_difference > 0.0);
        assert!(MeshIndependenceCheck::compare(0.0, 1.0, 1.0, 1.0).is_err());
        assert!(MeshIndependenceCheck::compare(1.0, 1.0, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn verdict_logic_passes_and_demotes() {
        let gate = PromotionGateMirror::from(&step9_promotion_gate());
        let mut promotion = PromotionReportMirror {
            frequency_hz: 159.0,
            spatial_levels: 3,
            temporal_levels: 2,
            max_spatial_observable_relative_change: 0.005,
            max_temporal_observable_relative_change: 0.003,
            cross_ladder_observable_relative_change: 0.0,
            promoted_mesh_size_m: 0.01,
            promoted_dt_s: 2.5e-5,
            promoted_port_flow: [1.0e-5, -2.0e-6],
        };
        let extended = PromotionExtendedChanges {
            spatial_thd: check_finest_pair("t", 9.0e-6, 9.1e-6, 0.02).unwrap(),
            spatial_noise_proxy: check_finest_pair("n", 1.0, 1.01, 0.02).unwrap(),
            temporal_thd: check_finest_pair("t", 9.2e-6, 9.1e-6, 0.02).unwrap(),
            temporal_noise_proxy: check_finest_pair("n", 1.02, 1.01, 0.02).unwrap(),
            passed: true,
        };
        let (winner, straight, tier1_winner, tier1_straight) = linear_inputs();
        let linear_gate = check_linear_transfer_gate(&Step9LinearGateInputs {
            tier2_winner: &winner,
            tier2_straight: &straight,
            tier1_winner: &tier1_winner,
            tier1_straight: &tier1_straight,
        })
        .unwrap();
        let dt_fine = 1.0 / (159.0 * 252.0);
        let dt_coarse = 1.0 / (159.0 * 126.0);
        let spatial = vec![
            synthetic_diagnostics(0.02, dt_fine, 252, 9.0e-6, 1.0),
            synthetic_diagnostics(0.014, dt_fine, 252, 9.05e-6, 1.005),
            synthetic_diagnostics(0.01, dt_fine, 252, 9.1e-6, 1.01),
        ];
        let temporal = vec![
            synthetic_diagnostics(0.01, dt_coarse, 126, 9.2e-6, 1.02),
            synthetic_diagnostics(0.01, dt_fine, 252, 9.1e-6, 1.01),
        ];
        assert!(
            evaluate_step9_verdict(
                &promotion,
                &gate,
                &extended,
                &linear_gate,
                &spatial,
                &temporal
            )
            .passed()
        );

        promotion.max_spatial_observable_relative_change = 0.05;
        let verdict = evaluate_step9_verdict(
            &promotion,
            &gate,
            &extended,
            &linear_gate,
            &spatial,
            &temporal,
        );
        assert!(!verdict.passed());
        match verdict {
            Step9Verdict::DemoteUnconvergedIndicativeOnly { reasons } => {
                assert_eq!(reasons.len(), 1);
                assert!(reasons[0].contains("spatial phasor"));
            }
            Step9Verdict::Pass => panic!("expected demotion"),
        }

        // A per-level diagnostic breach also demotes, naming the level.
        let mut bad_spatial = spatial.clone();
        bad_spatial[0].cap_continuity_relative_error = 1.0e-2;
        let verdict = evaluate_step9_verdict(
            &promotion,
            &gate,
            &extended,
            &linear_gate,
            &bad_spatial,
            &temporal,
        );
        match verdict {
            Step9Verdict::DemoteUnconvergedIndicativeOnly { reasons } => {
                assert!(
                    reasons
                        .iter()
                        .any(|reason| reason.contains("spatial level 0 cap continuity"))
                );
            }
            Step9Verdict::Pass => panic!("expected demotion"),
        }
    }

    fn synthetic_evidence_level(
        mesh_size_m: f64,
        dt_s: f64,
        scale: f64,
    ) -> crate::port_flow::convergence::PortFlowConvergenceLevel {
        use crate::port_flow::convergence::{PortFlowConvergenceLevel, PortFlowPhasor};
        PortFlowConvergenceLevel {
            mesh_size_m,
            dt_s,
            frequency_hz: 159.0,
            port_flow: PortFlowPhasor {
                real: scale,
                imaginary: -0.2 * scale,
            },
            box_pressure: PortFlowPhasor {
                real: 4.0 * scale,
                imaginary: 0.5 * scale,
            },
            diaphragm_flow: PortFlowPhasor {
                real: 1.5 * scale,
                imaginary: 0.1 * scale,
            },
            max_linear_relative_residual: 1.0e-10,
            max_box_conservation_relative_residual: 1.0e-10,
            cap_continuity_relative_error: 1.0e-8,
            energy_algebraic_relative_closure: 1.0e-10,
            energy_physical_relative_closure: 1.0e-3,
            energy_interface_relative_residual: 1.0e-10,
        }
    }

    fn synthetic_report() -> Step9Report {
        use crate::port_flow::convergence::PORT_FLOW_PROMOTION_SCHEMA_VERSION;
        let case = PortOptimReferenceCase::default();
        let ladder = PromotionLadderConfig::production(&case).unwrap();
        let dt_fine = 1.0 / (159.0 * 504.0);
        let dt_coarse = 1.0 / (159.0 * 252.0);
        let evidence = PortFlowPromotionEvidence {
            schema_version: PORT_FLOW_PROMOTION_SCHEMA_VERSION,
            provenance: STEP9_PROMOTION_PROVENANCE.to_string(),
            frequency_hz: 159.0,
            spatial_levels: vec![
                synthetic_evidence_level(0.016, dt_fine, 1.008),
                synthetic_evidence_level(0.012, dt_fine, 1.004),
                synthetic_evidence_level(0.01, dt_fine, 1.0),
            ],
            temporal_levels: vec![
                synthetic_evidence_level(0.01, dt_coarse, 1.004),
                synthetic_evidence_level(0.01, dt_fine, 1.0),
            ],
        };
        let gate = step9_promotion_gate();
        let promotion_report =
            crate::port_flow::convergence::validate_port_flow_promotion(&evidence, gate)
                .expect("synthetic evidence promotes");
        let spatial_levels = vec![
            synthetic_diagnostics(0.016, dt_fine, 504, 9.0e-6, 1.0),
            synthetic_diagnostics(0.012, dt_fine, 504, 4.6e-7, 1.005),
            synthetic_diagnostics(0.01, dt_fine, 504, 1.0e-5, 1.01),
        ];
        let temporal_levels = vec![
            synthetic_diagnostics(0.01, dt_coarse, 252, 2.4e-6, 1.02),
            synthetic_diagnostics(0.01, dt_fine, 504, 1.02e-5, 1.01),
        ];
        let extended = check_extended_changes(&spatial_levels, &temporal_levels, 0.02).unwrap();
        assert!(extended.spatial_thd.passed_by_floor);
        assert!(extended.temporal_thd.passed_by_floor);
        let (low_drive_winner, low_drive_straight, tier1_winner, tier1_straight) = linear_inputs();
        let linear_gate = check_linear_transfer_gate(&Step9LinearGateInputs {
            tier2_winner: &low_drive_winner,
            tier2_straight: &low_drive_straight,
            tier1_winner: &tier1_winner,
            tier1_straight: &tier1_straight,
        })
        .unwrap();
        let promotion = PromotionReportMirror::from(&promotion_report);
        let promotion_gate = PromotionGateMirror::from(&gate);
        let verdict = evaluate_step9_verdict(
            &promotion,
            &promotion_gate,
            &extended,
            &linear_gate,
            &spatial_levels,
            &temporal_levels,
        );
        assert!(verdict.passed());
        Step9Report {
            schema_version: STEP9_REPORT_SCHEMA_VERSION,
            provenance: STEP9_REPORT_PROVENANCE.to_string(),
            case,
            f_target_hz: 92.6,
            n_control_points: 5,
            winner_u: vec![0.015, 0.029],
            ladder_voltage_peak_v: 1.0,
            low_drive_voltage_peak_v: 0.01,
            ladder_config: ladder,
            promotion_evidence: evidence,
            promotion_gate,
            promotion,
            spatial_levels,
            temporal_levels,
            extended,
            linear_gate,
            low_drive_winner,
            low_drive_straight,
            tier1_winner,
            tier1_straight,
            mesh_independence: MeshIndependenceCheck::compare(7.31e-6, 7.30e-6, 7.3e-4, 7.31e-4)
                .unwrap(),
            verdict,
            notes: "synthetic".to_string(),
        }
    }

    #[test]
    fn report_round_trip_validates_and_catches_tampering() {
        let report = synthetic_report();
        report.validate().expect("synthetic report validates");
        let encoded = report.to_json_string().expect("report serializes");
        let decoded = Step9Report::from_json_str(&encoded).expect("report parses");
        assert_eq!(decoded, report);
        decoded.validate().expect("decoded report validates");

        // Stale verdict fails coherence.
        let mut stale = report.clone();
        stale.extended.spatial_thd.passed = false;
        assert!(stale.validate().is_err());
        // Demotion without reasons fails.
        let mut empty_demotion = report.clone();
        empty_demotion.verdict = Step9Verdict::DemoteUnconvergedIndicativeOnly {
            reasons: Vec::new(),
        };
        assert!(empty_demotion.validate().is_err());
        // Schema/provenance drift is rejected at parse.
        let mut drifted = serde_json::to_value(&report).unwrap();
        drifted["schema_version"] = serde_json::json!(STEP9_REPORT_SCHEMA_VERSION + 1);
        assert!(Step9Report::from_json_str(&serde_json::to_string(&drifted).unwrap()).is_err());
        let mut drifted = serde_json::to_value(&report).unwrap();
        drifted["provenance"] = serde_json::json!("drifted");
        assert!(Step9Report::from_json_str(&serde_json::to_string(&drifted).unwrap()).is_err());
    }
}
