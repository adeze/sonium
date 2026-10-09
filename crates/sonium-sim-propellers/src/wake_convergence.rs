//! Checked convergence evidence for the steady free-wake reference.
//!
//! The ladder varies wake revolutions and azimuthal segment resolution, then
//! compares the induced velocity at the BEM station observer plane. This is a
//! bounded spatial/revolution reference for the steady wake; it is not an
//! unsteady time-step or physical shedding-convergence solver.

use std::fmt;

use crate::{
    BladeElementBemResult, BladeElementModel, Point3, SteadyFreeWakeError, SteadyFreeWakeOptions,
    SteadyFreeWakeResult, evaluate_steady_free_wake,
};

/// One wake-turn/azimuthal-resolution level in a convergence ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SteadyFreeWakeDiscretization {
    /// Number of wake revolutions retained.
    pub wake_turns: usize,
    /// Azimuthal segments per revolution.
    pub segments_per_turn: usize,
}

/// Controls a bounded steady free-wake discretization ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyFreeWakeConvergenceOptions {
    /// Ordered, non-decreasing wake-turn/segment refinement levels.
    pub levels: Vec<SteadyFreeWakeDiscretization>,
    /// Maximum relative induced-velocity change admitted at the final level.
    pub tolerance: f64,
    /// Base wake solver options, including node/kernel budgets.
    pub wake_options: SteadyFreeWakeOptions,
}

impl Default for SteadyFreeWakeConvergenceOptions {
    fn default() -> Self {
        Self {
            levels: vec![
                SteadyFreeWakeDiscretization {
                    wake_turns: 1,
                    segments_per_turn: 8,
                },
                SteadyFreeWakeDiscretization {
                    wake_turns: 1,
                    segments_per_turn: 16,
                },
                SteadyFreeWakeDiscretization {
                    wake_turns: 2,
                    segments_per_turn: 16,
                },
            ],
            tolerance: 0.25,
            wake_options: SteadyFreeWakeOptions::default(),
        }
    }
}

/// Evidence retained for one steady-wake refinement level.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyFreeWakeConvergenceLevel {
    /// Discretization used for this level.
    pub discretization: SteadyFreeWakeDiscretization,
    /// Number of retained wake nodes.
    pub node_count: usize,
    /// Number of Biot--Savart evaluations performed by the wake solve.
    pub biot_savart_evaluation_count: usize,
    /// Maximum sampled induced speed at the station observer plane [m/s].
    pub maximum_induced_speed_m_s: f64,
    /// Relative change from the preceding level, when one exists [-].
    pub relative_induced_velocity_change: Option<f64>,
    /// Whether this level's comparison met the configured tolerance.
    pub converged: bool,
}

/// Result from a checked steady free-wake refinement ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyFreeWakeConvergence {
    /// Per-level wake/work evidence in caller order.
    pub levels: Vec<SteadyFreeWakeConvergenceLevel>,
    /// Retained wake results in the same order as [`Self::levels`].
    pub wakes: Vec<SteadyFreeWakeResult>,
    /// Maximum relative change across adjacent levels [-].
    pub max_relative_induced_velocity_change: f64,
    /// Relative change between the final two levels [-].
    pub final_relative_induced_velocity_change: f64,
    /// Caller-declared convergence tolerance [-].
    pub tolerance: f64,
    /// Whether the final adjacent levels met the tolerance.
    pub converged: bool,
}

impl SteadyFreeWakeConvergence {
    /// Whether all retained convergence evidence is finite and consistent.
    pub fn is_finite(&self) -> bool {
        self.tolerance.is_finite()
            && self.tolerance > 0.0
            && self.max_relative_induced_velocity_change.is_finite()
            && self.max_relative_induced_velocity_change >= 0.0
            && self.final_relative_induced_velocity_change.is_finite()
            && self.final_relative_induced_velocity_change >= 0.0
            && self.levels.len() >= 2
            && self.levels.len() == self.wakes.len()
            && self.levels.iter().all(|level| {
                level.discretization.wake_turns > 0
                    && level.discretization.segments_per_turn >= 2
                    && level.node_count > 0
                    && level.biot_savart_evaluation_count > 0
                    && level.maximum_induced_speed_m_s.is_finite()
                    && level.maximum_induced_speed_m_s >= 0.0
                    && level
                        .relative_induced_velocity_change
                        .map(|change| change.is_finite() && change >= 0.0)
                        .unwrap_or(true)
            })
            && self.wakes.iter().all(SteadyFreeWakeResult::is_finite)
    }
}

/// Failure-closed error from the steady-wake convergence ladder.
#[derive(Debug, Clone, PartialEq)]
pub enum SteadyFreeWakeConvergenceError {
    /// Ladder configuration is invalid.
    InvalidConfiguration {
        /// Configuration field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A wake level failed.
    WakeEvaluation {
        /// Failed level index.
        level_index: usize,
        /// Underlying wake error.
        error: SteadyFreeWakeError,
    },
    /// A sampled observer value was non-finite.
    NonFiniteObservable {
        /// Level index.
        level_index: usize,
        /// Station index.
        station: usize,
        /// Invalid field name.
        field: &'static str,
    },
}

impl fmt::Display for SteadyFreeWakeConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, requirement } => {
                write!(
                    f,
                    "steady free-wake convergence {field} must be {requirement}"
                )
            }
            Self::WakeEvaluation { level_index, error } => write!(
                f,
                "steady free-wake convergence level {level_index} failed: {error}"
            ),
            Self::NonFiniteObservable {
                level_index,
                station,
                field,
            } => write!(
                f,
                "steady free-wake convergence level {level_index} station {station} {field} is non-finite"
            ),
        }
    }
}

impl std::error::Error for SteadyFreeWakeConvergenceError {}

/// Evaluate a bounded steady free-wake wake-turn/segment convergence ladder.
pub fn evaluate_steady_free_wake_convergence(
    model: &BladeElementModel,
    bem: &BladeElementBemResult,
    options: SteadyFreeWakeConvergenceOptions,
) -> Result<SteadyFreeWakeConvergence, SteadyFreeWakeConvergenceError> {
    validate_options(&options)?;
    let mut levels = Vec::with_capacity(options.levels.len());
    let mut wakes = Vec::with_capacity(options.levels.len());
    let mut sampled_previous: Option<Vec<Point3>> = None;
    let mut max_relative_change: f64 = 0.0;

    for (level_index, discretization) in options.levels.iter().copied().enumerate() {
        let wake_options = SteadyFreeWakeOptions {
            wake_turns: discretization.wake_turns,
            segments_per_turn: discretization.segments_per_turn,
            ..options.wake_options
        };
        let wake = evaluate_steady_free_wake(model, bem, wake_options).map_err(|error| {
            SteadyFreeWakeConvergenceError::WakeEvaluation { level_index, error }
        })?;
        let sampled = sample_induced_velocities(model, &wake, level_index)?;
        let maximum_induced_speed_m_s =
            sampled.iter().map(|value| norm(*value)).fold(0.0, f64::max);
        let relative_change = sampled_previous
            .as_ref()
            .map(|previous| relative_change(previous, &sampled));
        if let Some(change) = relative_change {
            max_relative_change = max_relative_change.max(change);
        }
        levels.push(SteadyFreeWakeConvergenceLevel {
            discretization,
            node_count: wake.node_count,
            biot_savart_evaluation_count: wake.biot_savart_evaluation_count,
            maximum_induced_speed_m_s,
            relative_induced_velocity_change: relative_change,
            converged: relative_change
                .map(|change| change <= options.tolerance)
                .unwrap_or(false),
        });
        sampled_previous = Some(sampled);
        wakes.push(wake);
    }

    let final_relative_change = levels
        .last()
        .and_then(|level| level.relative_induced_velocity_change)
        .ok_or(SteadyFreeWakeConvergenceError::InvalidConfiguration {
            field: "levels",
            requirement: "at least two levels",
        })?;
    let result = SteadyFreeWakeConvergence {
        levels,
        wakes,
        max_relative_induced_velocity_change: max_relative_change,
        final_relative_induced_velocity_change: final_relative_change,
        tolerance: options.tolerance,
        converged: final_relative_change <= options.tolerance,
    };
    if !result.is_finite() {
        return Err(SteadyFreeWakeConvergenceError::NonFiniteObservable {
            level_index: result.levels.len().saturating_sub(1),
            station: 0,
            field: "convergence_result",
        });
    }
    Ok(result)
}

fn sample_induced_velocities(
    model: &BladeElementModel,
    wake: &SteadyFreeWakeResult,
    level_index: usize,
) -> Result<Vec<Point3>, SteadyFreeWakeConvergenceError> {
    model
        .stations()
        .iter()
        .enumerate()
        .map(|(station, geometry)| {
            let value =
                wake.induced_velocity_at(Point3 {
                    x: geometry.radius_m,
                    y: 0.0,
                    z: 0.0,
                })
                .map_err(|error| {
                    SteadyFreeWakeConvergenceError::WakeEvaluation { level_index, error }
                })?;
            if !value.x.is_finite() {
                return Err(SteadyFreeWakeConvergenceError::NonFiniteObservable {
                    level_index,
                    station,
                    field: "x_velocity_m_s",
                });
            }
            if !value.y.is_finite() {
                return Err(SteadyFreeWakeConvergenceError::NonFiniteObservable {
                    level_index,
                    station,
                    field: "y_velocity_m_s",
                });
            }
            if !value.z.is_finite() {
                return Err(SteadyFreeWakeConvergenceError::NonFiniteObservable {
                    level_index,
                    station,
                    field: "z_velocity_m_s",
                });
            }
            Ok(value)
        })
        .collect()
}

fn relative_change(previous: &[Point3], current: &[Point3]) -> f64 {
    previous
        .iter()
        .zip(current)
        .map(|(previous, current)| {
            let difference = Point3 {
                x: current.x - previous.x,
                y: current.y - previous.y,
                z: current.z - previous.z,
            };
            norm(difference) / norm(*previous).max(norm(*current)).max(1.0e-12)
        })
        .fold(0.0, f64::max)
}

fn norm(value: Point3) -> f64 {
    value.x.hypot(value.y).hypot(value.z)
}

fn validate_options(
    options: &SteadyFreeWakeConvergenceOptions,
) -> Result<(), SteadyFreeWakeConvergenceError> {
    if options.levels.len() < 2 {
        return Err(SteadyFreeWakeConvergenceError::InvalidConfiguration {
            field: "levels",
            requirement: "at least two levels",
        });
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(SteadyFreeWakeConvergenceError::InvalidConfiguration {
            field: "tolerance",
            requirement: "finite positive",
        });
    }
    for (index, level) in options.levels.iter().enumerate() {
        if level.wake_turns == 0 || level.segments_per_turn < 2 {
            return Err(SteadyFreeWakeConvergenceError::InvalidConfiguration {
                field: "levels",
                requirement: "positive wake turns and at least two segments per turn",
            });
        }
        if let Some(previous) = index.checked_sub(1).and_then(|i| options.levels.get(i))
            && (level.wake_turns < previous.wake_turns
                || level.segments_per_turn < previous.segments_per_turn
                || (level.wake_turns == previous.wake_turns
                    && level.segments_per_turn == previous.segments_per_turn))
        {
            return Err(SteadyFreeWakeConvergenceError::InvalidConfiguration {
                field: "levels",
                requirement: "strictly increasing non-decreasing refinement",
            });
        }
    }
    Ok(())
}
