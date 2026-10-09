//! Executable spatial/temporal promotion runner for resolved port flow.
//!
//! The evidence validator is deliberately independent from execution.  This
//! module supplies the missing production seam: callers provide a fresh
//! body-fitted mesh and coupled configuration for each refinement level, and
//! the runner executes the nonlinear Taylor–Hood solve, extracts one common
//! settled tail, and validates the resulting promotion artifact.

use serde::{Deserialize, Serialize};
use sonium_fem::mesh::Mesh;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use super::convergence::{
    PORT_FLOW_PROMOTION_SCHEMA_VERSION, PortFlowConvergenceLevel, PortFlowPromotionEvidence,
    PortFlowPromotionGate, PortFlowPromotionReport,
};
use super::coupled::{CoupledPortConfig, CoupledPortResult, run_coupled_port};
use super::error::PortFlowError;

/// One executable refinement point in a nonlinear port-flow study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortFlowRuntimeRefinement {
    /// Characteristic resolved mesh size (m).
    pub mesh_size_m: f64,
    /// BDF timestep (s).
    pub dt_s: f64,
    /// Fundamental drive frequency (Hz).
    pub frequency_hz: f64,
    /// Complete samples retained per drive cycle.
    pub samples_per_cycle: usize,
    /// Complete cycles retained for phasors and diagnostics.
    pub tail_cycles: usize,
    /// Complete cycles discarded as startup transient.
    pub discarded_cycles: usize,
}

impl PortFlowRuntimeRefinement {
    fn validate(self) -> Result<(), PortFlowError> {
        if !self.mesh_size_m.is_finite()
            || self.mesh_size_m <= 0.0
            || !self.dt_s.is_finite()
            || self.dt_s <= 0.0
            || !self.frequency_hz.is_finite()
            || self.frequency_hz <= 0.0
            || self.samples_per_cycle < 4
            || self.tail_cycles == 0
            || self.discarded_cycles == 0
        {
            return Err(PortFlowError::InvalidConfig(
                "runtime refinement requires finite positive mesh, timestep, frequency, at least four samples per cycle, a discarded transient, and a non-empty tail".into(),
            ));
        }
        let expected_dt = 1.0 / (self.frequency_hz * self.samples_per_cycle as f64);
        let mismatch = (self.dt_s - expected_dt).abs() / expected_dt.max(f64::MIN_POSITIVE);
        if !mismatch.is_finite() || mismatch > 1.0e-8 {
            return Err(PortFlowError::InvalidConfig(format!(
                "runtime timestep {} is inconsistent with {} Hz and {} samples/cycle",
                self.dt_s, self.frequency_hz, self.samples_per_cycle
            )));
        }
        Ok(())
    }

    fn required_steps(self) -> Result<usize, PortFlowError> {
        self.discarded_cycles
            .checked_add(self.tail_cycles)
            .and_then(|cycles| cycles.checked_mul(self.samples_per_cycle))
            .ok_or_else(|| {
                PortFlowError::InvalidConfig("runtime cycle count overflows usize".into())
            })
    }
}

/// Independent spatial and temporal refinement ladders for one drive point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortFlowRuntimePlan {
    /// Coarse-to-fine spatial mesh sizes. Values must strictly decrease.
    pub spatial_mesh_sizes_m: Vec<f64>,
    /// Coarse-to-fine timesteps. Values must strictly decrease.
    pub temporal_dt_s: Vec<f64>,
    pub frequency_hz: f64,
    pub tail_cycles: usize,
    pub discarded_cycles: usize,
    pub box_volume_m3: f64,
    pub density_kg_per_m3: f64,
    pub sound_speed_m_per_s: f64,
    /// Stable provenance copied into the promotion artifact.
    pub provenance: String,
}

impl PortFlowRuntimePlan {
    fn validate(&self) -> Result<(), PortFlowError> {
        if self.spatial_mesh_sizes_m.len() < 2 || self.temporal_dt_s.len() < 2 {
            return Err(PortFlowError::InvalidConfig(
                "runtime plan requires at least two spatial and two temporal levels".into(),
            ));
        }
        if self.provenance.trim().is_empty()
            || !self.box_volume_m3.is_finite()
            || self.box_volume_m3 <= 0.0
            || !self.density_kg_per_m3.is_finite()
            || self.density_kg_per_m3 <= 0.0
            || !self.sound_speed_m_per_s.is_finite()
            || self.sound_speed_m_per_s <= 0.0
        {
            return Err(PortFlowError::InvalidConfig(
                "runtime plan has invalid provenance or medium parameters".into(),
            ));
        }
        let fine_dt = *self.temporal_dt_s.last().ok_or_else(|| {
            PortFlowError::InvalidConfig("runtime temporal ladder is empty".into())
        })?;
        for window in self.spatial_mesh_sizes_m.windows(2) {
            if !window[0].is_finite() || !window[1].is_finite() || window[1] >= window[0] {
                return Err(PortFlowError::InvalidConfig(
                    "runtime spatial ladder must be finite and strictly refine".into(),
                ));
            }
        }
        if self.spatial_mesh_sizes_m.iter().any(|value| *value <= 0.0) {
            return Err(PortFlowError::InvalidConfig(
                "runtime spatial mesh sizes must be positive".into(),
            ));
        }
        for window in self.temporal_dt_s.windows(2) {
            if !window[0].is_finite() || !window[1].is_finite() || window[1] >= window[0] {
                return Err(PortFlowError::InvalidConfig(
                    "runtime temporal ladder must be finite and strictly refine".into(),
                ));
            }
        }
        if self.temporal_dt_s.iter().any(|value| *value <= 0.0) {
            return Err(PortFlowError::InvalidConfig(
                "runtime timesteps must be positive".into(),
            ));
        }
        let fine_dt_samples = Self::samples_per_cycle_for_dt(self.frequency_hz, fine_dt)?;
        let fine = PortFlowRuntimeRefinement {
            mesh_size_m: *self.spatial_mesh_sizes_m.last().unwrap(),
            dt_s: fine_dt,
            frequency_hz: self.frequency_hz,
            samples_per_cycle: fine_dt_samples,
            tail_cycles: self.tail_cycles,
            discarded_cycles: self.discarded_cycles,
        };
        fine.validate()?;
        for &dt_s in &self.temporal_dt_s[..self.temporal_dt_s.len() - 1] {
            Self::samples_per_cycle_for_dt(self.frequency_hz, dt_s)?;
        }
        Ok(())
    }

    fn refinement(&self, mesh_size_m: f64, dt_s: f64) -> PortFlowRuntimeRefinement {
        PortFlowRuntimeRefinement {
            mesh_size_m,
            dt_s,
            frequency_hz: self.frequency_hz,
            samples_per_cycle: Self::samples_per_cycle_for_dt(self.frequency_hz, dt_s)
                .expect("validated runtime plan must have integral cycle sampling"),
            tail_cycles: self.tail_cycles,
            discarded_cycles: self.discarded_cycles,
        }
    }

    fn samples_per_cycle_for_dt(frequency_hz: f64, dt_s: f64) -> Result<usize, PortFlowError> {
        let samples = (1.0 / (frequency_hz * dt_s)).round();
        if !samples.is_finite() || samples < 4.0 || samples > usize::MAX as f64 {
            return Err(PortFlowError::InvalidConfig(
                "runtime timestep must resolve at least four integral samples per drive cycle"
                    .into(),
            ));
        }
        let samples = samples as usize;
        let mismatch =
            (dt_s - 1.0 / (frequency_hz * samples as f64)).abs() / dt_s.max(f64::MIN_POSITIVE);
        if !mismatch.is_finite() || mismatch > 1.0e-8 {
            return Err(PortFlowError::InvalidConfig(
                "runtime timestep must divide the drive period into complete cycles".into(),
            ));
        }
        Ok(samples)
    }
}

/// Fresh FEM inputs and coupled configuration for one runtime level.
pub struct PortFlowRuntimeCase {
    /// Fresh body-fitted or resolved 3-D fluid mesh.
    pub mesh: Mesh,
    /// P2 velocity space with three components.
    pub velocity_space: VectorH1Space,
    /// P1 pressure space paired with [`Self::velocity_space`].
    pub pressure_space: H1Space,
    /// Nonlinear box/driver and BDF configuration for this level.
    pub config: CoupledPortConfig,
}

/// Validated runtime evidence and its promotion report.
#[derive(Debug, Clone, PartialEq)]
pub struct PortFlowRuntimePromotion {
    /// Versioned evidence retained for serialization or later audit.
    pub evidence: PortFlowPromotionEvidence,
    /// Gate summary for the retained evidence.
    pub report: PortFlowPromotionReport,
}

/// Solver-free artifact for replaying one runtime promotion decision.
///
/// The plan is retained alongside the evidence so a later reader can verify
/// that the serialized ladders still describe the requested mesh and timestep
/// schedule. Replaying this artifact never invokes a FEM solve; it only
/// revalidates the schedule, typed convergence evidence, and caller-selected
/// promotion gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortFlowRuntimePromotionArtifact {
    pub plan: PortFlowRuntimePlan,
    pub evidence: PortFlowPromotionEvidence,
}

fn close_runtime_value(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= 1.0e-12 * scale
}

/// Return a deterministic identity for the resolved fluid topology.
///
/// Runtime promotion must not accept a declared spatial ladder whose cases
/// all reuse one mesh. Coordinates and element connectivity are the physical
/// inputs to the Taylor–Hood spaces, so include both in the identity while
/// ignoring solver caches and refinement bookkeeping.
fn mesh_identity(mesh: &Mesh) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    mesh.dimension.hash(&mut hasher);
    mesh.nodes.len().hash(&mut hasher);
    for point in &mesh.nodes {
        point.x.to_bits().hash(&mut hasher);
        point.y.to_bits().hash(&mut hasher);
        point.z.to_bits().hash(&mut hasher);
    }
    mesh.elements.len().hash(&mut hasher);
    for element in &mesh.elements {
        std::mem::discriminant(&element.element_type).hash(&mut hasher);
        element.nodes.hash(&mut hasher);
        element.marker.hash(&mut hasher);
    }
    hasher.finish()
}

fn validate_spatial_mesh_identity(
    seen: &mut HashSet<u64>,
    mesh: &Mesh,
) -> Result<(), PortFlowError> {
    if !seen.insert(mesh_identity(mesh)) {
        return Err(PortFlowError::InvalidConfig(
            "runtime spatial ladder reused an identical mesh".into(),
        ));
    }
    Ok(())
}

fn validate_replay_schedule(
    plan: &PortFlowRuntimePlan,
    evidence: &PortFlowPromotionEvidence,
) -> Result<(), PortFlowError> {
    if !close_runtime_value(plan.frequency_hz, evidence.frequency_hz)
        || plan.provenance != evidence.provenance
        || evidence.spatial_levels.len() != plan.spatial_mesh_sizes_m.len()
        || evidence.temporal_levels.len() != plan.temporal_dt_s.len()
    {
        return Err(PortFlowError::InvalidConfig(
            "runtime promotion artifact does not match its plan metadata".into(),
        ));
    }

    let fine_mesh_size_m = *plan.spatial_mesh_sizes_m.last().ok_or_else(|| {
        PortFlowError::InvalidConfig("runtime replay plan has no fine spatial level".into())
    })?;
    let fine_dt_s = *plan.temporal_dt_s.last().ok_or_else(|| {
        PortFlowError::InvalidConfig("runtime replay plan has no fine temporal level".into())
    })?;

    for (expected_mesh_size_m, level) in plan
        .spatial_mesh_sizes_m
        .iter()
        .zip(&evidence.spatial_levels)
    {
        if !close_runtime_value(*expected_mesh_size_m, level.mesh_size_m)
            || !close_runtime_value(fine_dt_s, level.dt_s)
        {
            return Err(PortFlowError::InvalidConfig(
                "runtime replay spatial evidence does not match its plan ladder".into(),
            ));
        }
    }
    for (expected_dt_s, level) in plan.temporal_dt_s.iter().zip(&evidence.temporal_levels) {
        if !close_runtime_value(*expected_dt_s, level.dt_s)
            || !close_runtime_value(fine_mesh_size_m, level.mesh_size_m)
        {
            return Err(PortFlowError::InvalidConfig(
                "runtime replay temporal evidence does not match its plan ladder".into(),
            ));
        }
    }
    Ok(())
}

/// Replay a serialized runtime promotion artifact without solver work.
pub fn replay_port_flow_promotion(
    artifact: &PortFlowRuntimePromotionArtifact,
    gate: PortFlowPromotionGate,
) -> Result<PortFlowRuntimePromotion, PortFlowError> {
    artifact.plan.validate()?;
    validate_replay_schedule(&artifact.plan, &artifact.evidence)?;
    let report = super::convergence::validate_port_flow_promotion(&artifact.evidence, gate)?;
    Ok(PortFlowRuntimePromotion {
        evidence: artifact.evidence.clone(),
        report,
    })
}

fn validate_case(
    case: &PortFlowRuntimeCase,
    refinement: PortFlowRuntimeRefinement,
) -> Result<(), PortFlowError> {
    if case.mesh.dimension != 3
        || case.velocity_space.polynomial_degree() != 2
        || case.pressure_space.polynomial_degree() != 1
        || case.velocity_space.num_components() != case.mesh.dimension
    {
        return Err(PortFlowError::InvalidConfig(
            "runtime cases must use three-dimensional Taylor–Hood P2-P1 spaces".into(),
        ));
    }
    let dt_mismatch =
        (case.config.bdf.dt - refinement.dt_s).abs() / refinement.dt_s.max(f64::MIN_POSITIVE);
    if !dt_mismatch.is_finite() || dt_mismatch > 1.0e-10 {
        return Err(PortFlowError::InvalidConfig(
            "runtime case BDF timestep does not match its refinement level".into(),
        ));
    }
    if case.config.bdf.num_steps < refinement.required_steps()? {
        return Err(PortFlowError::InvalidConfig(
            "runtime case does not retain enough complete cycles for promotion".into(),
        ));
    }
    Ok(())
}

/// Execute and promote an independent nonlinear spatial/temporal ladder.
///
/// `build_case` must return a fresh mesh, Taylor–Hood spaces, and coupled
/// configuration for each requested level. The fine spatial result is reused
/// as the fine temporal result, making the cross-ladder comparison an exact
/// common-state check rather than two nominally equivalent reruns.
pub fn run_port_flow_promotion<F>(
    plan: &PortFlowRuntimePlan,
    gate: PortFlowPromotionGate,
    mut build_case: F,
) -> Result<PortFlowRuntimePromotion, PortFlowError>
where
    F: FnMut(PortFlowRuntimeRefinement) -> Result<PortFlowRuntimeCase, PortFlowError>,
{
    let mut spatial_levels_remaining = plan.spatial_mesh_sizes_m.len();
    let mut spatial_meshes = HashSet::new();
    run_port_flow_promotion_from_results(plan, gate, |refinement| {
        let case = build_case(refinement)?;
        validate_case(&case, refinement)?;
        // The spatial ladder is executed at the common fine timestep. The
        // temporal ladder intentionally reuses the fine spatial mesh, so only
        // check identities for this first phase.
        if spatial_levels_remaining > 0 {
            validate_spatial_mesh_identity(&mut spatial_meshes, &case.mesh)?;
            spatial_levels_remaining -= 1;
        }
        run_coupled_port(
            &case.mesh,
            &case.velocity_space,
            &case.pressure_space,
            &case.config,
        )
    })
}

/// Promote a runtime ladder from already accepted coupled traces.
///
/// This internal seam keeps the ladder orchestration independently testable:
/// production callers use [`run_port_flow_promotion`], while contract tests
/// can supply deterministic accepted traces without pretending to have run a
/// physical nonlinear campaign.
fn run_port_flow_promotion_from_results<F>(
    plan: &PortFlowRuntimePlan,
    gate: PortFlowPromotionGate,
    mut solve_result: F,
) -> Result<PortFlowRuntimePromotion, PortFlowError>
where
    F: FnMut(PortFlowRuntimeRefinement) -> Result<CoupledPortResult, PortFlowError>,
{
    plan.validate()?;
    let spatial_dt_s = *plan.temporal_dt_s.last().unwrap();
    let mut spatial_levels = Vec::with_capacity(plan.spatial_mesh_sizes_m.len());
    for &mesh_size_m in &plan.spatial_mesh_sizes_m {
        let refinement = plan.refinement(mesh_size_m, spatial_dt_s);
        refinement.validate()?;
        let result = solve_result(refinement)?;
        spatial_levels.push(PortFlowConvergenceLevel::from_coupled_result(
            &result,
            refinement.mesh_size_m,
            refinement.dt_s,
            refinement.frequency_hz,
            refinement.samples_per_cycle,
            refinement.tail_cycles,
            refinement.discarded_cycles,
            plan.box_volume_m3,
            plan.density_kg_per_m3,
            plan.sound_speed_m_per_s,
        )?);
    }

    let fine_spatial = *spatial_levels.last().ok_or_else(|| {
        PortFlowError::InvalidConfig("runtime spatial ladder produced no result".into())
    })?;
    let mut temporal_levels = Vec::with_capacity(plan.temporal_dt_s.len());
    for &dt_s in &plan.temporal_dt_s[..plan.temporal_dt_s.len() - 1] {
        let refinement = plan.refinement(fine_spatial.mesh_size_m, dt_s);
        refinement.validate()?;
        let result = solve_result(refinement)?;
        temporal_levels.push(PortFlowConvergenceLevel::from_coupled_result(
            &result,
            refinement.mesh_size_m,
            refinement.dt_s,
            refinement.frequency_hz,
            refinement.samples_per_cycle,
            refinement.tail_cycles,
            refinement.discarded_cycles,
            plan.box_volume_m3,
            plan.density_kg_per_m3,
            plan.sound_speed_m_per_s,
        )?);
    }
    temporal_levels.push(fine_spatial);

    let evidence = PortFlowPromotionEvidence {
        schema_version: PORT_FLOW_PROMOTION_SCHEMA_VERSION,
        provenance: plan.provenance.clone(),
        frequency_hz: plan.frequency_hz,
        spatial_levels,
        temporal_levels,
    };
    replay_port_flow_promotion(
        &PortFlowRuntimePromotionArtifact {
            plan: plan.clone(),
            evidence,
        },
        gate,
    )
}

/// Test-only plan helper: keep schedule validation independent from a solver.
#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::f64::consts::TAU;

    use crate::port_flow::coupled::{CoupledEnergyStep, CoupledPortResult};

    fn plan() -> PortFlowRuntimePlan {
        PortFlowRuntimePlan {
            spatial_mesh_sizes_m: vec![0.004, 0.003, 0.002],
            temporal_dt_s: vec![2.0e-5, 1.0e-5],
            frequency_hz: 12_500.0,
            tail_cycles: 2,
            discarded_cycles: 1,
            box_volume_m3: 1.0e-4,
            density_kg_per_m3: 1.2,
            sound_speed_m_per_s: 343.0,
            provenance: "runtime-test.v1".into(),
        }
    }

    #[test]
    fn plan_requires_common_fine_timestep_consistent_with_cycle_sampling() {
        assert!(plan().validate().is_ok());
        let mut invalid = plan();
        invalid.temporal_dt_s[1] = 9.5e-6;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn spatial_mesh_identity_rejects_reused_mesh() {
        let mesh = Mesh::new(3);
        let mut seen = HashSet::new();
        validate_spatial_mesh_identity(&mut seen, &mesh).expect("first mesh is new");
        let error = validate_spatial_mesh_identity(&mut seen, &mesh)
            .expect_err("reusing a spatial mesh must fail closed");
        assert!(
            error
                .to_string()
                .contains("spatial ladder reused an identical mesh")
        );
    }

    #[test]
    fn plan_rejects_non_refining_or_insufficient_ladders() {
        let mut invalid = plan();
        invalid.spatial_mesh_sizes_m = vec![0.002, 0.003];
        assert!(invalid.validate().is_err());
        invalid = plan();
        invalid.temporal_dt_s = vec![1.0e-5];
        assert!(invalid.validate().is_err());
        invalid = plan();
        invalid.tail_cycles = 0;
        assert!(invalid.validate().is_err());
        invalid = plan();
        invalid.discarded_cycles = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn refinement_counts_are_checked_without_integer_wraparound() {
        let mut invalid = plan();
        invalid.temporal_dt_s = vec![1.0e-300, 5.0e-301];
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn plan_rejects_too_few_samples_for_phasor_extraction() {
        let mut invalid = plan();
        invalid.temporal_dt_s[1] = 1.0 / (invalid.frequency_hz * 2.0);
        assert!(invalid.validate().is_err());
    }

    fn synthetic_result(refinement: PortFlowRuntimeRefinement) -> CoupledPortResult {
        let sample_count = refinement
            .required_steps()
            .expect("synthetic refinement should have finite sample count");
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
            let phase = TAU * refinement.frequency_hz * index as f64 * refinement.dt_s;
            let flow = phase.sin();
            result.time_s.push((index + 1) as f64 * refinement.dt_s);
            result.port_flow_m3s.push(flow);
            result.exterior_cap_flow_m3s.push(flow);
            result.box_cap_outward_flow_m3s.push(-flow);
            result.diaphragm_volume_velocity_m3s.push(1.5 * flow);
            result.box_pressure_pa.push(4.0 * flow);
        }
        result
    }

    #[test]
    fn result_ladder_orchestration_promotes_without_solver_workload() {
        let plan = PortFlowRuntimePlan {
            spatial_mesh_sizes_m: vec![0.004, 0.003, 0.002],
            temporal_dt_s: vec![1.0 / (80.0 * 8.0), 1.0 / (80.0 * 16.0)],
            frequency_hz: 80.0,
            tail_cycles: 2,
            discarded_cycles: 1,
            box_volume_m3: 2.0e-5,
            density_kg_per_m3: 1.2,
            sound_speed_m_per_s: 343.0,
            provenance: "runtime-synthetic-contract.v1".into(),
        };
        let mut calls = Vec::new();
        let promotion = run_port_flow_promotion_from_results(
            &plan,
            PortFlowPromotionGate::default(),
            |refinement| {
                calls.push((refinement.mesh_size_m, refinement.dt_s));
                Ok(synthetic_result(refinement))
            },
        )
        .expect("synthetic accepted traces should promote");
        assert_eq!(calls.len(), 4);
        assert_eq!(promotion.evidence.spatial_levels.len(), 3);
        assert_eq!(promotion.evidence.temporal_levels.len(), 2);
        assert_eq!(promotion.report.promoted_mesh_size_m, 0.002);
        assert_eq!(promotion.report.promoted_dt_s, 1.0 / (80.0 * 16.0));
        assert!(promotion.report.promoted_port_flow.real.is_finite());
        assert!(
            Complex64::new(
                promotion.report.promoted_port_flow.real,
                promotion.report.promoted_port_flow.imaginary,
            )
            .norm()
                > 0.0
        );

        let artifact = PortFlowRuntimePromotionArtifact {
            plan: plan.clone(),
            evidence: promotion.evidence.clone(),
        };
        let encoded = serde_json::to_string(&artifact).expect("runtime artifact should serialize");
        let decoded: PortFlowRuntimePromotionArtifact =
            serde_json::from_str(&encoded).expect("runtime artifact should deserialize");
        let replayed = replay_port_flow_promotion(&decoded, PortFlowPromotionGate::default())
            .expect("accepted runtime artifact should replay without solving");
        assert_eq!(replayed.report.frequency_hz, promotion.report.frequency_hz);
        assert_eq!(
            replayed.report.spatial_levels,
            promotion.report.spatial_levels
        );
        assert_eq!(
            replayed.report.temporal_levels,
            promotion.report.temporal_levels
        );
        for (actual, expected) in [
            (
                replayed.report.max_spatial_observable_relative_change,
                promotion.report.max_spatial_observable_relative_change,
            ),
            (
                replayed.report.max_temporal_observable_relative_change,
                promotion.report.max_temporal_observable_relative_change,
            ),
            (
                replayed.report.cross_ladder_observable_relative_change,
                promotion.report.cross_ladder_observable_relative_change,
            ),
            (
                replayed.report.promoted_mesh_size_m,
                promotion.report.promoted_mesh_size_m,
            ),
            (
                replayed.report.promoted_dt_s,
                promotion.report.promoted_dt_s,
            ),
            (
                replayed.report.promoted_port_flow.real,
                promotion.report.promoted_port_flow.real,
            ),
            (
                replayed.report.promoted_port_flow.imaginary,
                promotion.report.promoted_port_flow.imaginary,
            ),
        ] {
            assert!((actual - expected).abs() <= 1.0e-12);
        }

        let mut mismatched = decoded;
        mismatched.plan.spatial_mesh_sizes_m[2] = 0.001;
        assert!(replay_port_flow_promotion(&mismatched, PortFlowPromotionGate::default()).is_err());
    }
}
