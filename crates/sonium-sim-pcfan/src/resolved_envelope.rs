//! Checked resolved-flow operating-envelope evaluation.
//!
//! This module evaluates every requested RPM and deterministic manufacturing
//! corner through the production CFD mesh-convergence path.  It is deliberately
//! an evaluation/reporting seam: it does not claim design derivatives and it
//! does not promote a resolved-flow optimizer candidate.

use crate::cfd::try_solve_pc_fan_cfd_mesh_convergence;
use crate::evaluation::{
    PcFanCfdEvaluationError, invalid_manufacturing_tolerance, manufacturing_corner_variations,
    try_evaluate_pc_fan_from_cfd_solution,
};
use crate::model::{
    PcFanCfdMeshConvergence, PcFanCfdMeshConvergenceConfig, PcFanCfdMeshConvergenceConfigError,
    PcFanConfig, PcFanDesignVariables, PcFanEvaluation, PcFanFlowModel,
    PcFanManufacturingVariation,
};
use std::fmt;

/// Configuration for a checked resolved-flow RPM/manufacturing sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanResolvedOperatingEnvelopeConfig {
    /// Strictly increasing RPM samples.  At least two are required so the
    /// per-manufacturing-realization monotonicity gate is meaningful.
    pub rpms: Vec<f64>,
    /// Bounded isotropic mesh sweep applied independently at every point.
    pub mesh_convergence: PcFanCfdMeshConvergenceConfig,
    /// Require at least one non-zero manufacturing tolerance axis.
    pub require_manufacturing_coverage: bool,
}

impl Default for PcFanResolvedOperatingEnvelopeConfig {
    fn default() -> Self {
        Self {
            rpms: vec![420.0, 600.0, 780.0],
            mesh_convergence: PcFanCfdMeshConvergenceConfig::default(),
            require_manufacturing_coverage: true,
        }
    }
}

/// Errors raised before or during a resolved-flow envelope sweep.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanResolvedOperatingEnvelopeError {
    UnsupportedFlowModel(PcFanFlowModel),
    NotEnoughRpms {
        actual: usize,
    },
    InvalidRpm {
        index: usize,
        value: f64,
    },
    NonIncreasingRpms {
        index: usize,
        previous: f64,
        value: f64,
    },
    InvalidTolerance {
        parameter: &'static str,
        value: f64,
    },
    InvalidGeometry(String),
    InvalidMeshConvergence(PcFanCfdMeshConvergenceConfigError),
    MeshNotConverged {
        variation_index: usize,
        rpm_index: usize,
        residual: f64,
    },
    CfdEvaluation {
        variation_index: usize,
        rpm_index: usize,
        source: PcFanCfdEvaluationError,
    },
    NonFiniteSample {
        variation_index: usize,
        rpm_index: usize,
    },
    MissingManufacturingCoverage,
}

impl fmt::Display for PcFanResolvedOperatingEnvelopeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFlowModel(model) => write!(
                formatter,
                "resolved PC-fan operating envelope requires CFD flow, got {model:?}"
            ),
            Self::NotEnoughRpms { actual } => write!(
                formatter,
                "resolved PC-fan operating envelope requires at least two RPMs, got {actual}"
            ),
            Self::InvalidRpm { index, value } => {
                write!(formatter, "resolved PC-fan RPM {index} is invalid: {value}")
            }
            Self::NonIncreasingRpms {
                index,
                previous,
                value,
            } => write!(
                formatter,
                "resolved PC-fan RPM {index} is not strictly increasing: {value} <= {previous}"
            ),
            Self::InvalidTolerance { parameter, value } => write!(
                formatter,
                "resolved PC-fan manufacturing tolerance {parameter} is invalid: {value}"
            ),
            Self::InvalidGeometry(message) => {
                write!(
                    formatter,
                    "resolved PC-fan envelope geometry is invalid: {message}"
                )
            }
            Self::InvalidMeshConvergence(error) => write!(formatter, "{error}"),
            Self::MeshNotConverged {
                variation_index,
                rpm_index,
                residual,
            } => write!(
                formatter,
                "resolved PC-fan envelope sample variation {variation_index}, RPM index {rpm_index} did not mesh-converge (residual {residual:.3e})"
            ),
            Self::CfdEvaluation {
                variation_index,
                rpm_index,
                source,
            } => write!(
                formatter,
                "resolved PC-fan envelope sample variation {variation_index}, RPM index {rpm_index} failed CFD admission: {source}"
            ),
            Self::NonFiniteSample {
                variation_index,
                rpm_index,
            } => write!(
                formatter,
                "resolved PC-fan envelope sample variation {variation_index}, RPM index {rpm_index} is non-finite"
            ),
            Self::MissingManufacturingCoverage => write!(
                formatter,
                "resolved PC-fan operating envelope requires non-zero manufacturing coverage"
            ),
        }
    }
}

impl std::error::Error for PcFanResolvedOperatingEnvelopeError {}

/// One checked resolved-flow envelope sample.
#[derive(Debug, PartialEq)]
pub struct PcFanResolvedOperatingEnvelopePoint {
    pub variation: PcFanManufacturingVariation,
    pub rpm: f64,
    pub design_variables: PcFanDesignVariables,
    pub evaluation: PcFanEvaluation,
    pub mesh_convergence: PcFanCfdMeshConvergence,
    pub shaft_power_to_fluid_w: f64,
    pub relative_mass_imbalance: f64,
    pub relative_momentum_imbalance: f64,
    pub relative_pressure_closure: f64,
    pub relative_energy_imbalance: f64,
}

/// Cross-product report for resolved CFD and manufacturing tolerances.
#[derive(Debug, PartialEq)]
pub struct PcFanResolvedOperatingEnvelope {
    /// Points are ordered by manufacturing realization, then ascending RPM.
    pub points: Vec<PcFanResolvedOperatingEnvelopePoint>,
    pub variation_count: usize,
    pub rpm_count: usize,
    pub finite: bool,
    pub all_points_feasible: bool,
    pub all_meshes_converged: bool,
    pub all_variations_monotonic: bool,
    pub has_tolerance_coverage: bool,
    pub worst_relative_mass_imbalance: f64,
    pub worst_relative_momentum_imbalance: f64,
    pub worst_relative_pressure_closure: f64,
    pub worst_relative_energy_imbalance: f64,
}

impl PcFanResolvedOperatingEnvelope {
    /// Return whether the report contains a complete finite, converged,
    /// feasible, monotonic cross-product with finite conservation ledgers.
    pub fn is_valid(&self) -> bool {
        !self.points.is_empty()
            && self.variation_count > 0
            && self.rpm_count >= 2
            && self.points.len() == self.variation_count * self.rpm_count
            && self.finite
            && self.all_points_feasible
            && self.all_meshes_converged
            && self.all_variations_monotonic
            && self.worst_relative_mass_imbalance.is_finite()
            && self.worst_relative_momentum_imbalance.is_finite()
            && self.worst_relative_pressure_closure.is_finite()
            && self.worst_relative_energy_imbalance.is_finite()
    }

    /// Return whether the valid report also exercises at least one
    /// non-nominal manufacturing-tolerance corner.
    pub fn is_robust(&self) -> bool {
        self.is_valid() && self.has_tolerance_coverage
    }
}

fn validate_config(
    config: &PcFanConfig,
    envelope: &PcFanResolvedOperatingEnvelopeConfig,
) -> Result<(), PcFanResolvedOperatingEnvelopeError> {
    if !config.cfd.flow_model.is_cfd() {
        return Err(PcFanResolvedOperatingEnvelopeError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if envelope.rpms.len() < 2 {
        return Err(PcFanResolvedOperatingEnvelopeError::NotEnoughRpms {
            actual: envelope.rpms.len(),
        });
    }
    for (index, &rpm) in envelope.rpms.iter().enumerate() {
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(PcFanResolvedOperatingEnvelopeError::InvalidRpm { index, value: rpm });
        }
        if let Some(&previous) = index
            .checked_sub(1)
            .and_then(|previous| envelope.rpms.get(previous))
            && rpm.total_cmp(&previous) != std::cmp::Ordering::Greater
        {
            return Err(PcFanResolvedOperatingEnvelopeError::NonIncreasingRpms {
                index,
                previous,
                value: rpm,
            });
        }
    }
    if let Some((parameter, value)) =
        invalid_manufacturing_tolerance(config.manufacturing.tolerance)
    {
        return Err(PcFanResolvedOperatingEnvelopeError::InvalidTolerance { parameter, value });
    }
    config
        .validate_geometry_config()
        .map_err(|error| PcFanResolvedOperatingEnvelopeError::InvalidGeometry(error.to_string()))?;
    envelope
        .mesh_convergence
        .validate()
        .map_err(PcFanResolvedOperatingEnvelopeError::InvalidMeshConvergence)
}

/// Evaluate every requested RPM and manufacturing corner through checked
/// resolved CFD mesh convergence and conservation admission.
pub fn evaluate_pc_fan_resolved_operating_envelope(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    envelope: &PcFanResolvedOperatingEnvelopeConfig,
) -> Result<PcFanResolvedOperatingEnvelope, PcFanResolvedOperatingEnvelopeError> {
    validate_config(config, envelope)?;

    let variations = manufacturing_corner_variations(config.manufacturing.tolerance);
    let has_tolerance_coverage = variations
        .iter()
        .any(|variation| *variation != PcFanManufacturingVariation::nominal());
    if envelope.require_manufacturing_coverage && !has_tolerance_coverage {
        return Err(PcFanResolvedOperatingEnvelopeError::MissingManufacturingCoverage);
    }

    let variation_count = variations.len();
    let rpm_count = envelope.rpms.len();
    let mut points = Vec::with_capacity(variation_count * rpm_count);
    let mut all_points_feasible = true;
    let all_meshes_converged = true;
    let mut all_variations_monotonic = true;
    let mut worst_relative_mass_imbalance: f64 = 0.0;
    let mut worst_relative_momentum_imbalance: f64 = 0.0;
    let mut worst_relative_pressure_closure: f64 = 0.0;
    let mut worst_relative_energy_imbalance: f64 = 0.0;

    for (variation_index, variation) in variations.into_iter().enumerate() {
        let realization = variation.apply(design);
        config.try_build_geometry(&realization).map_err(|error| {
            PcFanResolvedOperatingEnvelopeError::InvalidGeometry(error.to_string())
        })?;
        let mut local_points = Vec::with_capacity(rpm_count);

        for (rpm_index, &rpm) in envelope.rpms.iter().enumerate() {
            let mut point_config = config.clone();
            point_config.operating_point.rpm = rpm;
            // The resolved envelope is a fan-curve report, not a nominal-point
            // target test.  Bind the feasibility check to the solved airflow
            // after the CFD state is admitted so no cross-RPM target is
            // silently reused.
            point_config.weights.operating_spread_weight = 0.0;

            let mesh = try_solve_pc_fan_cfd_mesh_convergence(
                &point_config,
                &realization,
                envelope.mesh_convergence,
            )
            .map_err(PcFanResolvedOperatingEnvelopeError::InvalidMeshConvergence)?;
            if !mesh.converged {
                return Err(PcFanResolvedOperatingEnvelopeError::MeshNotConverged {
                    variation_index,
                    rpm_index,
                    residual: mesh.selected_solution.residual,
                });
            }

            point_config.operating_point.target_airflow_cfm = mesh.selected_solution.airflow_cfm;
            let evaluation = try_evaluate_pc_fan_from_cfd_solution(
                &point_config,
                &realization,
                &mesh.selected_solution,
            )
            .map_err(
                |source| PcFanResolvedOperatingEnvelopeError::CfdEvaluation {
                    variation_index,
                    rpm_index,
                    source,
                },
            )?;
            let conservation = mesh.selected_solution.conservation;
            let sample_values = [
                evaluation.airflow_cfm,
                evaluation.flow.pressure_rise_pa,
                evaluation.acoustic_level_db,
                evaluation.sound_power_level_db,
                conservation.shaft_power_to_fluid_w,
                conservation.relative_mass_imbalance,
                conservation.relative_momentum_imbalance,
                conservation.relative_pressure_closure,
                conservation.relative_energy_imbalance,
            ];
            if sample_values.iter().any(|value| !value.is_finite()) {
                return Err(PcFanResolvedOperatingEnvelopeError::NonFiniteSample {
                    variation_index,
                    rpm_index,
                });
            }
            all_points_feasible &= evaluation.is_feasible(&point_config);
            worst_relative_mass_imbalance =
                worst_relative_mass_imbalance.max(conservation.relative_mass_imbalance);
            worst_relative_momentum_imbalance =
                worst_relative_momentum_imbalance.max(conservation.relative_momentum_imbalance);
            worst_relative_pressure_closure =
                worst_relative_pressure_closure.max(conservation.relative_pressure_closure);
            worst_relative_energy_imbalance =
                worst_relative_energy_imbalance.max(conservation.relative_energy_imbalance);
            local_points.push(PcFanResolvedOperatingEnvelopePoint {
                variation,
                rpm,
                design_variables: realization.clone(),
                evaluation,
                mesh_convergence: mesh,
                shaft_power_to_fluid_w: conservation.shaft_power_to_fluid_w,
                relative_mass_imbalance: conservation.relative_mass_imbalance,
                relative_momentum_imbalance: conservation.relative_momentum_imbalance,
                relative_pressure_closure: conservation.relative_pressure_closure,
                relative_energy_imbalance: conservation.relative_energy_imbalance,
            });
        }

        all_variations_monotonic &= local_points.windows(2).all(|pair| {
            pair[1].rpm > pair[0].rpm
                && pair[1].evaluation.airflow_cfm > pair[0].evaluation.airflow_cfm
                && pair[1].evaluation.flow.pressure_rise_pa
                    > pair[0].evaluation.flow.pressure_rise_pa
        });
        points.extend(local_points);
    }

    Ok(PcFanResolvedOperatingEnvelope {
        points,
        variation_count,
        rpm_count,
        finite: true,
        all_points_feasible,
        all_meshes_converged,
        all_variations_monotonic,
        has_tolerance_coverage,
        worst_relative_mass_imbalance,
        worst_relative_momentum_imbalance,
        worst_relative_pressure_closure,
        worst_relative_energy_imbalance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{PcFanCfdConfig, PcFanMeshResolution};

    fn tiny_cfd_config() -> PcFanConfig {
        PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 8,
                nonlinear_tolerance: 1.0e-2,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::bench_120mm()
        }
    }

    #[test]
    fn resolved_envelope_rejects_reduced_flow_before_solving() {
        let config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        assert!(matches!(
            evaluate_pc_fan_resolved_operating_envelope(
                &config,
                &design,
                &PcFanResolvedOperatingEnvelopeConfig::default(),
            ),
            Err(PcFanResolvedOperatingEnvelopeError::UnsupportedFlowModel(
                PcFanFlowModel::ReducedOrder
            ))
        ));
    }

    #[test]
    fn resolved_envelope_rejects_bad_rpm_and_mesh_contracts_before_solving() {
        let mut config = tiny_cfd_config();
        let design = config.default_design_variables();
        let mut envelope = PcFanResolvedOperatingEnvelopeConfig::default();
        envelope.rpms = vec![600.0, 600.0];
        assert!(matches!(
            evaluate_pc_fan_resolved_operating_envelope(&config, &design, &envelope),
            Err(PcFanResolvedOperatingEnvelopeError::NonIncreasingRpms { .. })
        ));

        envelope.rpms = vec![600.0, 900.0];
        envelope.mesh_convergence.start_cells = 0;
        assert!(matches!(
            evaluate_pc_fan_resolved_operating_envelope(&config, &design, &envelope),
            Err(PcFanResolvedOperatingEnvelopeError::InvalidMeshConvergence(
                PcFanCfdMeshConvergenceConfigError::StartCellsMustBePositive
            ))
        ));
        config.cfd.flow_model = PcFanFlowModel::ReducedOrder;
        assert!(matches!(
            evaluate_pc_fan_resolved_operating_envelope(&config, &design, &envelope),
            Err(PcFanResolvedOperatingEnvelopeError::UnsupportedFlowModel(_))
        ));
    }

    #[test]
    fn resolved_envelope_rejects_unconverged_sample_after_checked_mesh_sweep() {
        let mut config = tiny_cfd_config();
        config.manufacturing.tolerance.blade_thickness_fraction = 0.01;
        config.manufacturing.tolerance.blade_chord_fraction = 0.0;
        config.manufacturing.tolerance.twist_deg = 0.0;
        config.manufacturing.tolerance.shroud_offset_m = 0.0;
        let design = config.default_design_variables();
        let envelope_config = PcFanResolvedOperatingEnvelopeConfig {
            rpms: vec![600.0, 900.0],
            mesh_convergence: PcFanCfdMeshConvergenceConfig {
                start_cells: 2,
                max_cells: 3,
                relative_tolerance: 2.0,
                min_levels: 2,
                max_final_residual: 1.0e-3,
            },
            require_manufacturing_coverage: true,
        };

        let result =
            evaluate_pc_fan_resolved_operating_envelope(&config, &design, &envelope_config);
        assert!(matches!(
            result,
            Err(PcFanResolvedOperatingEnvelopeError::MeshNotConverged {
                variation_index: 0,
                rpm_index: 0,
                ..
            })
        ));
    }

    #[test]
    fn resolved_envelope_requires_manufacturing_coverage_before_solving() {
        let config = tiny_cfd_config();
        let design = config.default_design_variables();
        let envelope = PcFanResolvedOperatingEnvelopeConfig {
            rpms: vec![600.0, 900.0],
            require_manufacturing_coverage: true,
            ..PcFanResolvedOperatingEnvelopeConfig::default()
        };
        assert!(matches!(
            evaluate_pc_fan_resolved_operating_envelope(&config, &design, &envelope),
            Err(PcFanResolvedOperatingEnvelopeError::MissingManufacturingCoverage)
        ));
    }

    #[test]
    fn resolved_envelope_report_gates_reject_incomplete_or_nonfinite_reports() {
        let mut report = PcFanResolvedOperatingEnvelope {
            points: Vec::new(),
            variation_count: 1,
            rpm_count: 2,
            finite: true,
            all_points_feasible: true,
            all_meshes_converged: true,
            all_variations_monotonic: true,
            has_tolerance_coverage: true,
            worst_relative_mass_imbalance: 0.0,
            worst_relative_momentum_imbalance: 0.0,
            worst_relative_pressure_closure: 0.0,
            worst_relative_energy_imbalance: 0.0,
        };
        assert!(!report.is_valid());
        assert!(!report.is_robust());

        report.worst_relative_energy_imbalance = f64::NAN;
        assert!(!report.is_valid());
        assert!(!report.is_robust());
    }
}
