//! Promotion evidence for resolved nonlinear port-flow studies.
//!
//! A low residual on one mesh and one timestep is not a convergence result.
//! This module keeps the spatial and temporal ladders explicit and rejects a
//! product promotion unless both ladders agree at a common fine state.

use num_complex::Complex64;
use serde::de::Error as _;
use serde::{Deserialize, Serialize};

use super::analysis::coupled_complex_observables;
use super::coupled::CoupledPortResult;
use super::error::PortFlowError;

/// Current wire format for [`PortFlowPromotionEvidence`].
pub const PORT_FLOW_PROMOTION_SCHEMA_VERSION: u32 = 1;

/// A serde-friendly complex peak phasor using the `exp(-i omega t)` convention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortFlowPhasor {
    pub real: f64,
    pub imaginary: f64,
}

impl PortFlowPhasor {
    fn from_complex(value: Complex64) -> Self {
        Self {
            real: value.re,
            imaginary: value.im,
        }
    }

    fn as_complex(self) -> Complex64 {
        Complex64::new(self.real, self.imaginary)
    }

    fn is_finite_nonzero(self) -> bool {
        self.real.is_finite()
            && self.imaginary.is_finite()
            && self.as_complex().norm() > f64::EPSILON
    }
}

/// One resolved nonlinear port-flow result retained for a convergence ladder.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortFlowConvergenceLevel {
    /// Characteristic spatial element size (m); smaller means finer.
    pub mesh_size_m: f64,
    /// BDF timestep (s); smaller means finer.
    pub dt_s: f64,
    /// Fundamental drive frequency (Hz).
    pub frequency_hz: f64,
    /// Settled port volume-flow phasor (m³/s).
    pub port_flow: PortFlowPhasor,
    /// Settled box-pressure phasor (Pa).
    pub box_pressure: PortFlowPhasor,
    /// Settled diaphragm volume-flow phasor (m³/s).
    pub diaphragm_flow: PortFlowPhasor,
    /// Maximum accepted affine-fluid true residual over the retained tail.
    pub max_linear_relative_residual: f64,
    /// Maximum box-conservation residual normalized by the settled pressure.
    pub max_box_conservation_relative_residual: f64,
    /// Cap-flow continuity residual from the same settled tail.
    pub cap_continuity_relative_error: f64,
    /// Algebraic energy closure over the retained tail.
    pub energy_algebraic_relative_closure: f64,
    /// Physical-only energy closure over the retained tail.
    pub energy_physical_relative_closure: f64,
    /// Driver/box/fluid interface energy residual over the retained tail.
    pub energy_interface_relative_residual: f64,
}

impl PortFlowConvergenceLevel {
    /// Build evidence directly from one accepted coupled result.
    ///
    /// The adapter uses the same complete-cycle tail for phasors, residuals,
    /// and energy. `discarded_cycles` proves that the selected tail is not the
    /// initial transient, while `tail_cycles` controls the retained evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn from_coupled_result(
        result: &CoupledPortResult,
        mesh_size_m: f64,
        dt_s: f64,
        frequency_hz: f64,
        samples_per_cycle: usize,
        tail_cycles: usize,
        discarded_cycles: usize,
        box_volume_m3: f64,
        density_kg_per_m3: f64,
        sound_speed_m_per_s: f64,
    ) -> Result<Self, PortFlowError> {
        if !mesh_size_m.is_finite() || mesh_size_m <= 0.0 {
            return Err(PortFlowError::InvalidConfig(
                "port-flow convergence mesh size must be finite and positive".into(),
            ));
        }
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(PortFlowError::InvalidConfig(
                "port-flow convergence timestep must be finite and positive".into(),
            ));
        }
        let observables = coupled_complex_observables(
            result,
            dt_s,
            frequency_hz,
            samples_per_cycle,
            tail_cycles,
            discarded_cycles,
            box_volume_m3,
            density_kg_per_m3,
            sound_speed_m_per_s,
        )?;
        let tail_samples = tail_cycles.checked_mul(samples_per_cycle).ok_or_else(|| {
            PortFlowError::InvalidConfig("port-flow convergence tail length overflow".into())
        })?;
        let total_required = discarded_cycles
            .checked_mul(samples_per_cycle)
            .and_then(|discarded| discarded.checked_add(tail_samples))
            .ok_or_else(|| {
                PortFlowError::InvalidConfig("port-flow convergence discard length overflow".into())
            })?;
        let sample_count = result.time_s.len();
        if total_required > sample_count
            || result.steps_completed != sample_count
            || result.energy_steps.len() != sample_count
            || result.accepted_linear_relative_residual.len() != sample_count
            || result.box_conservation_residual_pa.len() != sample_count
        {
            return Err(PortFlowError::InvalidConfig(
                "port-flow convergence evidence has insufficient or misaligned tail diagnostics"
                    .into(),
            ));
        }
        if result.time_s.iter().any(|time| !time.is_finite())
            || result.time_s.windows(2).any(|window| {
                let step = window[1] - window[0];
                !step.is_finite() || (step - dt_s).abs() > 1.0e-10 * dt_s.max(1.0)
            })
        {
            return Err(PortFlowError::InvalidConfig(
                "port-flow convergence time samples do not match the declared timestep".into(),
            ));
        }
        let tail_start = sample_count - tail_samples;
        let energy = result.energy_summary(tail_start..sample_count);
        if !energy.is_finite() {
            return Err(PortFlowError::InvalidConfig(
                "port-flow convergence tail energy summary is non-finite".into(),
            ));
        }
        if result.energy_steps[tail_start..]
            .iter()
            .any(|step| !step.is_finite() || !step.physical_dissipations_nonnegative(1.0e-12))
        {
            return Err(PortFlowError::InvalidConfig(
                "port-flow convergence tail contains invalid physical dissipation".into(),
            ));
        }
        if result.accepted_linear_relative_residual[tail_start..]
            .iter()
            .any(|residual| !residual.is_finite() || *residual < 0.0)
            || result.box_conservation_residual_pa[tail_start..]
                .iter()
                .any(|residual| !residual.is_finite())
        {
            return Err(PortFlowError::InvalidConfig(
                "port-flow convergence tail contains invalid residual diagnostics".into(),
            ));
        }
        let max_linear_relative_residual = result.accepted_linear_relative_residual[tail_start..]
            .iter()
            .copied()
            .fold(0.0, f64::max);
        let pressure_scale = observables.box_pressure_pa.norm().max(f64::EPSILON);
        let max_box_conservation_relative_residual = result.box_conservation_residual_pa
            [tail_start..]
            .iter()
            .map(|residual| residual.abs() / pressure_scale)
            .fold(0.0, f64::max);
        let level = Self {
            mesh_size_m,
            dt_s,
            frequency_hz,
            port_flow: PortFlowPhasor::from_complex(observables.port_flow_m3s),
            box_pressure: PortFlowPhasor::from_complex(observables.box_pressure_pa),
            diaphragm_flow: PortFlowPhasor::from_complex(observables.diaphragm_flow_m3s),
            max_linear_relative_residual,
            max_box_conservation_relative_residual,
            cap_continuity_relative_error: observables.cap_continuity_relative_error,
            energy_algebraic_relative_closure: energy.algebraic_relative_closure,
            energy_physical_relative_closure: energy.physical_relative_closure,
            energy_interface_relative_residual: energy.interface_relative_residual,
        };
        validate_level(&level, frequency_hz, "coupled result")?;
        Ok(level)
    }
}

/// A complete spatial and temporal convergence ladder for a candidate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PortFlowPromotionEvidence {
    pub schema_version: u32,
    pub provenance: String,
    pub frequency_hz: f64,
    /// Same timestep, strictly decreasing mesh size.
    pub spatial_levels: Vec<PortFlowConvergenceLevel>,
    /// Same mesh size, strictly decreasing timestep.
    pub temporal_levels: Vec<PortFlowConvergenceLevel>,
}

#[derive(Debug, Deserialize)]
struct PortFlowPromotionEvidenceWire {
    schema_version: u32,
    provenance: String,
    frequency_hz: f64,
    spatial_levels: Vec<PortFlowConvergenceLevel>,
    temporal_levels: Vec<PortFlowConvergenceLevel>,
}

impl<'de> Deserialize<'de> for PortFlowPromotionEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = PortFlowPromotionEvidenceWire::deserialize(deserializer)?;
        let evidence = Self {
            schema_version: wire.schema_version,
            provenance: wire.provenance,
            frequency_hz: wire.frequency_hz,
            spatial_levels: wire.spatial_levels,
            temporal_levels: wire.temporal_levels,
        };
        validate_evidence_structure(&evidence).map_err(D::Error::custom)?;
        Ok(evidence)
    }
}

/// Thresholds applied before a resolved port-flow result can be promoted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortFlowPromotionGate {
    pub min_spatial_levels: usize,
    pub min_temporal_levels: usize,
    pub max_spatial_observable_relative_change: f64,
    pub max_temporal_observable_relative_change: f64,
    pub max_cross_ladder_observable_relative_change: f64,
    pub max_linear_relative_residual: f64,
    pub max_box_conservation_relative_residual: f64,
    pub max_cap_continuity_relative_error: f64,
    pub max_energy_algebraic_relative_closure: f64,
    pub max_energy_physical_relative_closure: f64,
    pub max_energy_interface_relative_residual: f64,
}

impl Default for PortFlowPromotionGate {
    fn default() -> Self {
        Self {
            min_spatial_levels: 3,
            min_temporal_levels: 2,
            max_spatial_observable_relative_change: 1.0e-2,
            max_temporal_observable_relative_change: 1.0e-2,
            max_cross_ladder_observable_relative_change: 1.0e-2,
            max_linear_relative_residual: 1.0e-8,
            max_box_conservation_relative_residual: 1.0e-8,
            max_cap_continuity_relative_error: 1.0e-6,
            max_energy_algebraic_relative_closure: 1.0e-8,
            max_energy_physical_relative_closure: 5.0e-2,
            max_energy_interface_relative_residual: 1.0e-8,
        }
    }
}

/// Summary returned after both convergence ladders pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortFlowPromotionReport {
    pub frequency_hz: f64,
    pub spatial_levels: usize,
    pub temporal_levels: usize,
    pub max_spatial_observable_relative_change: f64,
    pub max_temporal_observable_relative_change: f64,
    pub cross_ladder_observable_relative_change: f64,
    pub promoted_mesh_size_m: f64,
    pub promoted_dt_s: f64,
    pub promoted_port_flow: PortFlowPhasor,
}

fn valid_threshold(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn close(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= 1.0e-12 * scale
}

fn relative_change(coarse: PortFlowPhasor, fine: PortFlowPhasor) -> Result<f64, PortFlowError> {
    if !coarse.is_finite_nonzero() || !fine.is_finite_nonzero() {
        return Err(PortFlowError::InvalidConfig(
            "port-flow convergence phasor is non-finite or zero".into(),
        ));
    }
    let error = (coarse.as_complex() - fine.as_complex()).norm() / fine.as_complex().norm();
    if error.is_finite() {
        Ok(error)
    } else {
        Err(PortFlowError::InvalidConfig(
            "port-flow convergence observable change is non-finite".into(),
        ))
    }
}

fn validate_level(
    level: &PortFlowConvergenceLevel,
    expected_frequency_hz: f64,
    ladder: &str,
) -> Result<(), PortFlowError> {
    if !level.mesh_size_m.is_finite()
        || level.mesh_size_m <= 0.0
        || !level.dt_s.is_finite()
        || level.dt_s <= 0.0
        || !level.frequency_hz.is_finite()
        || level.frequency_hz <= 0.0
        || !close(level.frequency_hz, expected_frequency_hz)
        || !level.port_flow.is_finite_nonzero()
        || !level.box_pressure.is_finite_nonzero()
        || !level.diaphragm_flow.is_finite_nonzero()
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "{ladder} port-flow convergence level has invalid frequency, scale, or phasor"
        )));
    }
    for (name, value) in [
        ("linear residual", level.max_linear_relative_residual),
        (
            "box-conservation residual",
            level.max_box_conservation_relative_residual,
        ),
        (
            "cap-continuity residual",
            level.cap_continuity_relative_error,
        ),
        (
            "algebraic energy closure",
            level.energy_algebraic_relative_closure,
        ),
        (
            "physical energy closure",
            level.energy_physical_relative_closure,
        ),
        (
            "interface energy residual",
            level.energy_interface_relative_residual,
        ),
    ] {
        if !valid_threshold(value) {
            return Err(PortFlowError::InvalidConfig(format!(
                "{ladder} port-flow convergence {name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

fn validate_gate(gate: PortFlowPromotionGate) -> Result<(), PortFlowError> {
    if gate.min_spatial_levels < 2 || gate.min_temporal_levels < 2 {
        return Err(PortFlowError::InvalidConfig(
            "port-flow promotion requires at least two levels on each ladder".into(),
        ));
    }
    for (name, value) in [
        (
            "spatial observable-change threshold",
            gate.max_spatial_observable_relative_change,
        ),
        (
            "temporal observable-change threshold",
            gate.max_temporal_observable_relative_change,
        ),
        (
            "cross-ladder observable-change threshold",
            gate.max_cross_ladder_observable_relative_change,
        ),
        (
            "linear residual threshold",
            gate.max_linear_relative_residual,
        ),
        (
            "box-conservation residual threshold",
            gate.max_box_conservation_relative_residual,
        ),
        (
            "cap-continuity threshold",
            gate.max_cap_continuity_relative_error,
        ),
        (
            "algebraic energy threshold",
            gate.max_energy_algebraic_relative_closure,
        ),
        (
            "physical energy threshold",
            gate.max_energy_physical_relative_closure,
        ),
        (
            "interface energy threshold",
            gate.max_energy_interface_relative_residual,
        ),
    ] {
        if !valid_threshold(value) {
            return Err(PortFlowError::InvalidConfig(format!(
                "port-flow promotion {name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

fn validate_ladder(
    levels: &[PortFlowConvergenceLevel],
    expected_frequency_hz: f64,
    min_levels: usize,
    spatial: bool,
    gate: PortFlowPromotionGate,
) -> Result<f64, PortFlowError> {
    let ladder_name = if spatial { "spatial" } else { "temporal" };
    if levels.len() < min_levels {
        return Err(PortFlowError::InvalidConfig(format!(
            "{ladder_name} port-flow promotion ladder has {} levels; requires {min_levels}",
            levels.len()
        )));
    }
    for level in levels {
        validate_level(level, expected_frequency_hz, ladder_name)?;
    }
    let first = levels[0];
    let mut maximum = 0.0_f64;
    for window in levels.windows(2) {
        let coarse = window[0];
        let fine = window[1];
        if spatial {
            if !close(coarse.dt_s, first.dt_s)
                || !close(fine.dt_s, first.dt_s)
                || fine.mesh_size_m >= coarse.mesh_size_m
            {
                return Err(PortFlowError::InvalidConfig(
                    "spatial port-flow ladder must hold dt fixed and strictly refine mesh size"
                        .into(),
                ));
            }
        } else if !close(coarse.mesh_size_m, first.mesh_size_m)
            || !close(fine.mesh_size_m, first.mesh_size_m)
            || fine.dt_s >= coarse.dt_s
        {
            return Err(PortFlowError::InvalidConfig(
                "temporal port-flow ladder must hold mesh size fixed and strictly refine dt".into(),
            ));
        }
        let observable_change = relative_change(coarse.port_flow, fine.port_flow)?
            .max(relative_change(coarse.box_pressure, fine.box_pressure)?)
            .max(relative_change(coarse.diaphragm_flow, fine.diaphragm_flow)?);
        maximum = maximum.max(observable_change);
        let threshold = if spatial {
            gate.max_spatial_observable_relative_change
        } else {
            gate.max_temporal_observable_relative_change
        };
        if observable_change > threshold {
            return Err(PortFlowError::InvalidConfig(format!(
                "{ladder_name} port-flow observable change {observable_change:e} exceeds {threshold:e}"
            )));
        }
    }
    Ok(maximum)
}

fn validate_diagnostics(
    levels: &[PortFlowConvergenceLevel],
    gate: PortFlowPromotionGate,
) -> Result<(), PortFlowError> {
    for (index, level) in levels.iter().enumerate() {
        if level.max_linear_relative_residual > gate.max_linear_relative_residual
            || level.max_box_conservation_relative_residual
                > gate.max_box_conservation_relative_residual
            || level.cap_continuity_relative_error > gate.max_cap_continuity_relative_error
            || level.energy_algebraic_relative_closure > gate.max_energy_algebraic_relative_closure
            || level.energy_physical_relative_closure > gate.max_energy_physical_relative_closure
            || level.energy_interface_relative_residual
                > gate.max_energy_interface_relative_residual
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "port-flow promotion diagnostics fail at level {index}"
            )));
        }
    }
    Ok(())
}

fn validate_evidence_structure(evidence: &PortFlowPromotionEvidence) -> Result<(), PortFlowError> {
    if evidence.schema_version != PORT_FLOW_PROMOTION_SCHEMA_VERSION
        || evidence.provenance.trim().is_empty()
        || !evidence.frequency_hz.is_finite()
        || evidence.frequency_hz <= 0.0
    {
        return Err(PortFlowError::InvalidConfig(
            "port-flow promotion evidence has invalid schema, provenance, or frequency".into(),
        ));
    }
    let structural_gate = PortFlowPromotionGate {
        min_spatial_levels: 2,
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
    };
    validate_ladder(
        &evidence.spatial_levels,
        evidence.frequency_hz,
        structural_gate.min_spatial_levels,
        true,
        structural_gate,
    )?;
    validate_ladder(
        &evidence.temporal_levels,
        evidence.frequency_hz,
        structural_gate.min_temporal_levels,
        false,
        structural_gate,
    )?;
    let Some(&spatial_fine) = evidence.spatial_levels.last() else {
        return Err(PortFlowError::InvalidConfig(
            "spatial port-flow promotion ladder is empty".into(),
        ));
    };
    let Some(&temporal_fine) = evidence.temporal_levels.last() else {
        return Err(PortFlowError::InvalidConfig(
            "temporal port-flow promotion ladder is empty".into(),
        ));
    };
    if !close(spatial_fine.mesh_size_m, temporal_fine.mesh_size_m)
        || !close(spatial_fine.dt_s, temporal_fine.dt_s)
    {
        return Err(PortFlowError::InvalidConfig(
            "spatial and temporal port-flow ladders do not share a common fine state".into(),
        ));
    }
    Ok(())
}

/// Validate resolved 3-D port-flow evidence before product promotion.
pub fn validate_port_flow_promotion(
    evidence: &PortFlowPromotionEvidence,
    gate: PortFlowPromotionGate,
) -> Result<PortFlowPromotionReport, PortFlowError> {
    validate_gate(gate)?;
    validate_evidence_structure(evidence)?;
    let max_spatial_observable_relative_change = validate_ladder(
        &evidence.spatial_levels,
        evidence.frequency_hz,
        gate.min_spatial_levels,
        true,
        gate,
    )?;
    let max_temporal_observable_relative_change = validate_ladder(
        &evidence.temporal_levels,
        evidence.frequency_hz,
        gate.min_temporal_levels,
        false,
        gate,
    )?;
    validate_diagnostics(&evidence.spatial_levels, gate)?;
    validate_diagnostics(&evidence.temporal_levels, gate)?;

    let Some(&spatial_fine) = evidence.spatial_levels.last() else {
        return Err(PortFlowError::InvalidConfig(
            "spatial port-flow promotion ladder is empty".into(),
        ));
    };
    let Some(&temporal_fine) = evidence.temporal_levels.last() else {
        return Err(PortFlowError::InvalidConfig(
            "temporal port-flow promotion ladder is empty".into(),
        ));
    };
    if !close(spatial_fine.mesh_size_m, temporal_fine.mesh_size_m)
        || !close(spatial_fine.dt_s, temporal_fine.dt_s)
    {
        return Err(PortFlowError::InvalidConfig(
            "spatial and temporal port-flow ladders do not share a common fine state".into(),
        ));
    }
    let cross_ladder_observable_relative_change =
        relative_change(spatial_fine.port_flow, temporal_fine.port_flow)?
            .max(relative_change(
                spatial_fine.box_pressure,
                temporal_fine.box_pressure,
            )?)
            .max(relative_change(
                spatial_fine.diaphragm_flow,
                temporal_fine.diaphragm_flow,
            )?);
    if cross_ladder_observable_relative_change > gate.max_cross_ladder_observable_relative_change {
        return Err(PortFlowError::InvalidConfig(format!(
            "spatial/temporal fine-state disagreement {cross_ladder_observable_relative_change:e} exceeds {:e}",
            gate.max_cross_ladder_observable_relative_change
        )));
    }

    Ok(PortFlowPromotionReport {
        frequency_hz: evidence.frequency_hz,
        spatial_levels: evidence.spatial_levels.len(),
        temporal_levels: evidence.temporal_levels.len(),
        max_spatial_observable_relative_change,
        max_temporal_observable_relative_change,
        cross_ladder_observable_relative_change,
        promoted_mesh_size_m: spatial_fine.mesh_size_m,
        promoted_dt_s: spatial_fine.dt_s,
        promoted_port_flow: temporal_fine.port_flow,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    use crate::port_flow::coupled::{CoupledEnergyStep, CoupledPortResult};

    fn level(mesh_size_m: f64, dt_s: f64, scale: f64) -> PortFlowConvergenceLevel {
        PortFlowConvergenceLevel {
            mesh_size_m,
            dt_s,
            frequency_hz: 120.0,
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

    fn evidence() -> PortFlowPromotionEvidence {
        PortFlowPromotionEvidence {
            schema_version: PORT_FLOW_PROMOTION_SCHEMA_VERSION,
            provenance: "focused-round-port-reference.v1".into(),
            frequency_hz: 120.0,
            spatial_levels: vec![
                level(0.004, 1.0e-5, 1.008),
                level(0.003, 1.0e-5, 1.004),
                level(0.002, 1.0e-5, 1.0),
            ],
            temporal_levels: vec![level(0.002, 2.0e-5, 1.004), level(0.002, 1.0e-5, 1.0)],
        }
    }

    #[test]
    fn promotion_accepts_two_independent_ladders_at_common_fine_state() {
        let report = validate_port_flow_promotion(&evidence(), PortFlowPromotionGate::default())
            .expect("focused convergence evidence should promote");
        assert_eq!(report.spatial_levels, 3);
        assert_eq!(report.temporal_levels, 2);
        assert!(report.max_spatial_observable_relative_change < 0.01);
        assert!(report.max_temporal_observable_relative_change < 0.01);
        assert_eq!(report.promoted_mesh_size_m, 0.002);
        assert_eq!(report.promoted_dt_s, 1.0e-5);
    }

    #[test]
    fn promotion_rejects_missing_or_malformed_evidence() {
        let mut missing = evidence();
        missing.spatial_levels.truncate(2);
        assert!(matches!(
            validate_port_flow_promotion(&missing, PortFlowPromotionGate::default()),
            Err(PortFlowError::InvalidConfig(_))
        ));

        let mut malformed = evidence();
        malformed.schema_version += 1;
        assert!(matches!(
            validate_port_flow_promotion(&malformed, PortFlowPromotionGate::default()),
            Err(PortFlowError::InvalidConfig(_))
        ));
        let mut non_finite = evidence();
        non_finite.temporal_levels[1].energy_physical_relative_closure = f64::NAN;
        assert!(matches!(
            validate_port_flow_promotion(&non_finite, PortFlowPromotionGate::default()),
            Err(PortFlowError::InvalidConfig(_))
        ));
    }

    #[test]
    fn promotion_rejects_refinement_change_and_fine_state_disagreement() {
        let mut changed = evidence();
        changed.spatial_levels[1].port_flow.real = 1.2;
        assert!(validate_port_flow_promotion(&changed, PortFlowPromotionGate::default()).is_err());

        let mut wrong_axis = evidence();
        wrong_axis.temporal_levels[0].mesh_size_m = 0.003;
        assert!(
            validate_port_flow_promotion(&wrong_axis, PortFlowPromotionGate::default()).is_err()
        );

        let mut disagreement = evidence();
        disagreement.temporal_levels[1].port_flow.real *= 1.05;
        assert!(
            validate_port_flow_promotion(&disagreement, PortFlowPromotionGate::default()).is_err()
        );
    }

    #[test]
    fn from_coupled_result_uses_one_complete_tail_for_all_diagnostics() {
        let frequency_hz = 120.0;
        let samples_per_cycle = 8;
        let dt_s = 1.0 / (frequency_hz * samples_per_cycle as f64);
        let sample_count = 3 * samples_per_cycle;
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
            let flow = phase.sin();
            result.time_s.push((index + 1) as f64 * dt_s);
            result.port_flow_m3s.push(flow);
            result.exterior_cap_flow_m3s.push(flow);
            result.box_cap_outward_flow_m3s.push(-flow);
            result.diaphragm_volume_velocity_m3s.push(1.5 * flow);
            result.box_pressure_pa.push(4.0 * flow);
        }
        let level = PortFlowConvergenceLevel::from_coupled_result(
            &result,
            0.002,
            dt_s,
            frequency_hz,
            samples_per_cycle,
            2,
            1,
            0.005,
            1.2,
            343.0,
        )
        .expect("synthetic settled tail should produce typed evidence");
        assert_eq!(level.mesh_size_m, 0.002);
        assert_eq!(level.dt_s, dt_s);
        assert!(level.port_flow.is_finite_nonzero());
        assert!(level.max_linear_relative_residual < 1.0e-8);
        assert!(level.max_box_conservation_relative_residual < 1.0e-8);

        let mut malformed = result;
        malformed.time_s[samples_per_cycle] += 0.25 * dt_s;
        assert!(
            PortFlowConvergenceLevel::from_coupled_result(
                &malformed,
                0.002,
                dt_s,
                frequency_hz,
                samples_per_cycle,
                2,
                1,
                0.005,
                1.2,
                343.0,
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_promotion_thresholds_fail_closed() {
        let mut gate = PortFlowPromotionGate::default();
        gate.max_spatial_observable_relative_change = f64::NAN;
        assert!(validate_port_flow_promotion(&evidence(), gate).is_err());
        let mut gate = PortFlowPromotionGate::default();
        gate.min_temporal_levels = 1;
        assert!(validate_port_flow_promotion(&evidence(), gate).is_err());
    }

    #[test]
    fn evidence_round_trips_without_losing_complex_components() {
        let original = evidence();
        let encoded = serde_json::to_string(&original).unwrap();
        let decoded: PortFlowPromotionEvidence = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, original);

        let mut malformed = serde_json::to_value(&original).unwrap();
        malformed["schema_version"] = serde_json::json!(PORT_FLOW_PROMOTION_SCHEMA_VERSION + 1);
        assert!(serde_json::from_value::<PortFlowPromotionEvidence>(malformed).is_err());
    }
}
