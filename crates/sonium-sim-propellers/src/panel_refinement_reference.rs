//! Checked caller-supplied panel-response refinement evidence.
//!
//! This module compares observations produced by an upstream panel-flow,
//! load, or acoustic solver. It validates the refinement ledger and reports
//! adjacent response changes, but it does not execute that solver or claim
//! resolved-flow, wake, observer, or measured-data convergence.

use std::fmt;

/// Versioned identity for the caller-supplied response-convergence contract.
pub const PANEL_REFINEMENT_REFERENCE_SCHEMA_V1: &str =
    "propeller.panel-response-refinement-reference.v1";

/// One caller-supplied observation from a panel-refinement run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelRefinementObservation {
    /// Strictly increasing refinement identifier.
    pub refinement: usize,
    /// Number of panels in the corresponding run.
    pub panel_count: usize,
    /// Integrated thrust [N].
    pub thrust_n: f64,
    /// Integrated torque [N m].
    pub torque_nm: f64,
    /// Acoustic pressure metric [Pa], such as an observer peak or RMS.
    pub acoustic_pressure_pa: f64,
}

/// Controls response-refinement admission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelRefinementOptions {
    /// Maximum normalized change accepted between the last two levels.
    pub relative_tolerance: f64,
    /// Maximum number of caller-supplied levels admitted.
    pub maximum_levels: usize,
}

impl Default for PanelRefinementOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 1.0e-3,
            maximum_levels: 16,
        }
    }
}

/// Response changes associated with one refinement level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelRefinementPoint {
    /// Refinement identifier retained from the observation.
    pub refinement: usize,
    /// Panel count retained from the observation.
    pub panel_count: usize,
    /// Thrust retained from the observation [N].
    pub thrust_n: f64,
    /// Torque retained from the observation [N m].
    pub torque_nm: f64,
    /// Acoustic pressure retained from the observation [Pa].
    pub acoustic_pressure_pa: f64,
    /// Normalized thrust change from the preceding level.
    pub relative_thrust_change: f64,
    /// Normalized torque change from the preceding level.
    pub relative_torque_change: f64,
    /// Normalized acoustic-pressure change from the preceding level.
    pub relative_acoustic_pressure_change: f64,
    /// Maximum of the three normalized response changes.
    pub maximum_relative_change: f64,
}

/// Checked panel-response refinement result.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelRefinementResult {
    /// Versioned contract identity.
    pub schema: String,
    /// Response observations and adjacent changes in input order.
    pub points: Vec<PanelRefinementPoint>,
    /// Largest adjacent response change across the ladder.
    pub maximum_relative_change: f64,
    /// Adjacent response change at the finest two levels.
    pub final_relative_change: f64,
    /// Caller-declared admission tolerance.
    pub relative_tolerance: f64,
    /// Whether the final level satisfies the tolerance.
    pub converged: bool,
}

impl PanelRefinementResult {
    /// Return whether the result is finite and structurally consistent.
    pub fn is_finite(&self) -> bool {
        self.schema == PANEL_REFINEMENT_REFERENCE_SCHEMA_V1
            && self.points.len() >= 2
            && self.relative_tolerance.is_finite()
            && self.relative_tolerance > 0.0
            && self.maximum_relative_change.is_finite()
            && self.maximum_relative_change >= 0.0
            && self.final_relative_change.is_finite()
            && self.final_relative_change >= 0.0
            && self.points.iter().all(|point| {
                point.refinement > 0
                    && point.panel_count > 0
                    && point.thrust_n.is_finite()
                    && point.thrust_n >= 0.0
                    && point.torque_nm.is_finite()
                    && point.torque_nm >= 0.0
                    && point.acoustic_pressure_pa.is_finite()
                    && point.acoustic_pressure_pa >= 0.0
                    && point.relative_thrust_change.is_finite()
                    && point.relative_thrust_change >= 0.0
                    && point.relative_torque_change.is_finite()
                    && point.relative_torque_change >= 0.0
                    && point.relative_acoustic_pressure_change.is_finite()
                    && point.relative_acoustic_pressure_change >= 0.0
                    && point.maximum_relative_change.is_finite()
                    && point.maximum_relative_change >= 0.0
            })
    }

    /// Return whether the finest adjacent levels satisfy the declared gate.
    pub fn is_converged(&self) -> bool {
        self.converged && self.is_finite()
    }
}

/// Failure-closed response-refinement errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelRefinementError {
    /// Public option outside its admissible domain.
    InvalidOptions {
        /// Invalid option name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Observation outside its admissible domain.
    InvalidObservation {
        /// Zero-based observation index.
        index: usize,
        /// Invalid field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
}

impl fmt::Display for PanelRefinementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "invalid panel-refinement option {field}: {requirement}"
            ),
            Self::InvalidObservation {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid panel-refinement observation {index} field {field}: {requirement}"
            ),
        }
    }
}

impl std::error::Error for PanelRefinementError {}

/// Evaluate a caller-supplied panel-response refinement ladder.
pub fn evaluate_panel_refinement(
    observations: &[PanelRefinementObservation],
    options: PanelRefinementOptions,
) -> Result<PanelRefinementResult, PanelRefinementError> {
    if !options.relative_tolerance.is_finite() || options.relative_tolerance <= 0.0 {
        return Err(PanelRefinementError::InvalidOptions {
            field: "relative_tolerance",
            requirement: "finite and positive",
        });
    }
    if options.maximum_levels < 2 {
        return Err(PanelRefinementError::InvalidOptions {
            field: "maximum_levels",
            requirement: "at least two",
        });
    }
    if observations.len() < 2 || observations.len() > options.maximum_levels {
        return Err(PanelRefinementError::InvalidOptions {
            field: "observations",
            requirement: "at least two and within maximum_levels",
        });
    }

    for (index, observation) in observations.iter().enumerate() {
        if observation.refinement == 0 {
            return Err(PanelRefinementError::InvalidObservation {
                index,
                field: "refinement",
                requirement: "positive",
            });
        }
        if observation.panel_count == 0 {
            return Err(PanelRefinementError::InvalidObservation {
                index,
                field: "panel_count",
                requirement: "positive",
            });
        }
        if index > 0
            && (observation.refinement <= observations[index - 1].refinement
                || observation.panel_count <= observations[index - 1].panel_count)
        {
            return Err(PanelRefinementError::InvalidObservation {
                index,
                field: "refinement/panel_count",
                requirement: "strictly increasing",
            });
        }
        for (field, value) in [
            ("thrust_n", observation.thrust_n),
            ("torque_nm", observation.torque_nm),
            ("acoustic_pressure_pa", observation.acoustic_pressure_pa),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PanelRefinementError::InvalidObservation {
                    index,
                    field,
                    requirement: "finite and non-negative",
                });
            }
        }
    }

    let mut points = Vec::with_capacity(observations.len());
    for (index, observation) in observations.iter().enumerate() {
        let previous = index
            .checked_sub(1)
            .and_then(|previous| observations.get(previous));
        let relative_thrust_change = previous
            .map(|previous| normalized_change(observation.thrust_n, previous.thrust_n))
            .unwrap_or(0.0);
        let relative_torque_change = previous
            .map(|previous| normalized_change(observation.torque_nm, previous.torque_nm))
            .unwrap_or(0.0);
        let relative_acoustic_pressure_change = previous
            .map(|previous| {
                normalized_change(
                    observation.acoustic_pressure_pa,
                    previous.acoustic_pressure_pa,
                )
            })
            .unwrap_or(0.0);
        let maximum_relative_change = relative_thrust_change
            .max(relative_torque_change)
            .max(relative_acoustic_pressure_change);
        points.push(PanelRefinementPoint {
            refinement: observation.refinement,
            panel_count: observation.panel_count,
            thrust_n: observation.thrust_n,
            torque_nm: observation.torque_nm,
            acoustic_pressure_pa: observation.acoustic_pressure_pa,
            relative_thrust_change,
            relative_torque_change,
            relative_acoustic_pressure_change,
            maximum_relative_change,
        });
    }

    let maximum_relative_change = points
        .iter()
        .map(|point| point.maximum_relative_change)
        .fold(0.0, f64::max);
    let final_relative_change = points
        .last()
        .map(|point| point.maximum_relative_change)
        .ok_or(PanelRefinementError::InvalidOptions {
            field: "observations",
            requirement: "at least two and within maximum_levels",
        })?;
    let result = PanelRefinementResult {
        schema: PANEL_REFINEMENT_REFERENCE_SCHEMA_V1.to_owned(),
        points,
        maximum_relative_change,
        final_relative_change,
        relative_tolerance: options.relative_tolerance,
        converged: final_relative_change <= options.relative_tolerance,
    };
    debug_assert!(result.is_finite());
    Ok(result)
}

fn normalized_change(current: f64, previous: f64) -> f64 {
    if current == 0.0 && previous == 0.0 {
        0.0
    } else {
        (current - previous).abs() / current.abs().max(previous.abs()).max(f64::MIN_POSITIVE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observations() -> [PanelRefinementObservation; 3] {
        [
            PanelRefinementObservation {
                refinement: 1,
                panel_count: 96,
                thrust_n: 10.0,
                torque_nm: 0.25,
                acoustic_pressure_pa: 0.8,
            },
            PanelRefinementObservation {
                refinement: 2,
                panel_count: 384,
                thrust_n: 10.01,
                torque_nm: 0.2501,
                acoustic_pressure_pa: 0.8005,
            },
            PanelRefinementObservation {
                refinement: 4,
                panel_count: 1536,
                thrust_n: 10.0101,
                torque_nm: 0.250101,
                acoustic_pressure_pa: 0.800501,
            },
        ]
    }

    #[test]
    fn response_refinement_is_finite_and_converged() {
        let result = evaluate_panel_refinement(
            &observations(),
            PanelRefinementOptions {
                relative_tolerance: 1.0e-3,
                ..PanelRefinementOptions::default()
            },
        )
        .unwrap();
        assert!(result.is_converged());
        assert_eq!(result.points.len(), 3);
        assert!(result.final_relative_change < 1.0e-3);
    }

    #[test]
    fn response_refinement_rejects_non_converged_finest_change() {
        let mut levels = observations();
        levels[2].acoustic_pressure_pa = 1.2;
        let result = evaluate_panel_refinement(
            &levels,
            PanelRefinementOptions {
                relative_tolerance: 1.0e-2,
                ..PanelRefinementOptions::default()
            },
        )
        .unwrap();
        assert!(!result.is_converged());
        assert!(result.final_relative_change > 1.0e-2);
    }

    #[test]
    fn response_refinement_rejects_order_non_finite_and_budget_errors() {
        let mut levels = observations();
        levels[1].panel_count = levels[0].panel_count;
        assert!(matches!(
            evaluate_panel_refinement(&levels, PanelRefinementOptions::default()),
            Err(PanelRefinementError::InvalidObservation {
                field: "refinement/panel_count",
                ..
            })
        ));

        let mut levels = observations();
        levels[1].torque_nm = f64::NAN;
        assert!(matches!(
            evaluate_panel_refinement(&levels, PanelRefinementOptions::default()),
            Err(PanelRefinementError::InvalidObservation {
                field: "torque_nm",
                ..
            })
        ));

        assert!(matches!(
            evaluate_panel_refinement(
                &observations(),
                PanelRefinementOptions {
                    maximum_levels: 2,
                    ..PanelRefinementOptions::default()
                }
            ),
            Err(PanelRefinementError::InvalidOptions {
                field: "observations",
                ..
            })
        ));
    }

    #[test]
    fn response_refinement_replay_is_deterministic() {
        let first =
            evaluate_panel_refinement(&observations(), PanelRefinementOptions::default()).unwrap();
        let second =
            evaluate_panel_refinement(&observations(), PanelRefinementOptions::default()).unwrap();
        assert_eq!(first, second);
    }
}
