//! Bounded load-coupled steady free-wake reference.
//!
//! This module couples the checked stationwise BEM reference to the existing
//! fixed-circulation steady free-wake reference with explicit outer, wake, and
//! Biot--Savart budgets. It is intentionally a steady reference contract: it
//! does not model unsteady shedding, dynamic stall, viscous-core evolution,
//! CFD, FW-H acoustics, or aeroelastic deformation.

use std::fmt;

use crate::{
    BladeElementBemOptions, BladeElementBemResult, BladeElementError, BladeElementExternalInflow,
    BladeElementModel, BladeElementOperatingPoint, Point3, SteadyFreeWakeError,
    SteadyFreeWakeOptions, SteadyFreeWakeResult, evaluate_steady_free_wake,
};

/// Controls the bounded load-coupled BEM/free-wake fixed-point solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadCoupledFreeWakeOptions {
    /// Maximum number of outer inflow-coupling checks.
    pub max_outer_iterations: usize,
    /// Under-relaxation applied to sampled wake velocity in `(0, 1]`.
    pub inflow_relaxation: f64,
    /// Maximum station inflow update residual [m/s].
    pub convergence_tolerance_m_s: f64,
    /// Bounded fixed-circulation wake solver options.
    pub wake_options: SteadyFreeWakeOptions,
}

impl Default for LoadCoupledFreeWakeOptions {
    fn default() -> Self {
        Self {
            max_outer_iterations: 12,
            inflow_relaxation: 0.4,
            convergence_tolerance_m_s: 1.0e-5,
            wake_options: SteadyFreeWakeOptions::default(),
        }
    }
}

/// Converged load-coupled BEM and steady-wake reference result.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadCoupledFreeWakeResult {
    /// Final BEM result evaluated with `station_external_inflow`.
    pub bem: BladeElementBemResult,
    /// Fixed-circulation wake generated from the final BEM result.
    pub wake: SteadyFreeWakeResult,
    /// Relaxed wake-induced station velocities used by the final BEM solve.
    pub station_external_inflow: Vec<BladeElementExternalInflow>,
    /// Number of outer checks performed, including the converged check.
    pub outer_iterations: usize,
    /// Number of BEM evaluations, including the initial zero-inflow solve.
    pub bem_evaluation_count: usize,
    /// Number of wake evaluations, including the initial wake.
    pub wake_evaluation_count: usize,
    /// Final maximum station inflow update [m/s].
    pub final_residual_m_s: f64,
    /// Whether the station inflow update met the requested tolerance.
    pub converged: bool,
}

impl LoadCoupledFreeWakeResult {
    /// Return whether the result contains a finite, converged, internally
    /// consistent state.
    pub fn is_finite(&self) -> bool {
        self.converged
            && self.bem.converged
            && self.wake.is_finite()
            && self.outer_iterations > 0
            && self.bem_evaluation_count > 0
            && self.wake_evaluation_count > 0
            && self.final_residual_m_s.is_finite()
            && self.final_residual_m_s >= 0.0
            && self.station_external_inflow.iter().all(|inflow| {
                inflow.axial_velocity_m_s.is_finite() && inflow.tangential_velocity_m_s.is_finite()
            })
    }
}

/// Failure-closed error from the bounded load-coupled reference.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadCoupledFreeWakeError {
    /// Outer coupling options are invalid.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A BEM evaluation failed; `outer_iteration = 0` identifies the initial
    /// zero-external-inflow solve.
    BemEvaluation {
        /// Outer iteration at which the BEM solve failed.
        outer_iteration: usize,
        /// Underlying checked BEM error.
        error: Box<BladeElementError>,
    },
    /// A wake evaluation failed.
    WakeEvaluation {
        /// Outer iteration at which the wake solve failed.
        outer_iteration: usize,
        /// Underlying checked wake error.
        error: SteadyFreeWakeError,
    },
    /// A sampled wake velocity was non-finite.
    NonFiniteWakeVelocity {
        /// Rotor station index.
        station: usize,
        /// Invalid velocity component.
        field: &'static str,
    },
    /// The outer inflow fixed-point budget was exhausted.
    NotConverged {
        /// Number of outer checks performed.
        iterations: usize,
        /// Final maximum station inflow update [m/s].
        residual_m_s: f64,
    },
}

impl fmt::Display for LoadCoupledFreeWakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(f, "load-coupled free wake {field} must be {requirement}")
            }
            Self::BemEvaluation {
                outer_iteration,
                error,
            } => write!(
                f,
                "load-coupled free wake BEM evaluation at outer iteration {outer_iteration} failed: {error}"
            ),
            Self::WakeEvaluation {
                outer_iteration,
                error,
            } => write!(
                f,
                "load-coupled free wake wake evaluation at outer iteration {outer_iteration} failed: {error}"
            ),
            Self::NonFiniteWakeVelocity { station, field } => write!(
                f,
                "load-coupled free wake station {station} {field} is non-finite"
            ),
            Self::NotConverged {
                iterations,
                residual_m_s,
            } => write!(
                f,
                "load-coupled free wake did not converge in {iterations} iterations (residual {residual_m_s:e} m/s)"
            ),
        }
    }
}

impl std::error::Error for LoadCoupledFreeWakeError {}

/// Solve a bounded load-coupled steady BEM/free-wake reference.
///
/// The initial BEM solve uses zero external inflow. Each outer iteration
/// samples the wake-induced velocity at the rotor stations, relaxes that
/// velocity, re-solves BEM, and rebuilds the wake. The returned BEM and wake
/// are therefore from the same final outer state; convergence is reported only
/// when the next sampled station inflow would change by no more than the
/// configured tolerance.
pub fn evaluate_load_coupled_steady_free_wake(
    model: &BladeElementModel,
    operating_point: BladeElementOperatingPoint,
    bem_options: BladeElementBemOptions,
    options: LoadCoupledFreeWakeOptions,
) -> Result<LoadCoupledFreeWakeResult, LoadCoupledFreeWakeError> {
    validate_options(options)?;

    let station_count = model.stations().len();
    let mut station_external_inflow = vec![
        BladeElementExternalInflow {
            axial_velocity_m_s: 0.0,
            tangential_velocity_m_s: 0.0,
        };
        station_count
    ];

    let mut bem = model
        .evaluate_bem(operating_point, bem_options)
        .map_err(|error| LoadCoupledFreeWakeError::BemEvaluation {
            outer_iteration: 0,
            error: Box::new(error),
        })?;

    let mut wake =
        evaluate_steady_free_wake(model, &bem, options.wake_options).map_err(|error| {
            LoadCoupledFreeWakeError::WakeEvaluation {
                outer_iteration: 0,
                error,
            }
        })?;
    let mut final_residual_m_s = f64::INFINITY;

    for outer_index in 0..options.max_outer_iterations {
        let outer_iteration = outer_index + 1;
        let sampled_inflow = sample_wake_inflow(model, &wake, outer_iteration)?;
        let relaxed_inflow = relax_inflow(
            &station_external_inflow,
            &sampled_inflow,
            options.inflow_relaxation,
        )?;
        final_residual_m_s = max_inflow_change(&station_external_inflow, &relaxed_inflow);

        if final_residual_m_s <= options.convergence_tolerance_m_s {
            let result = LoadCoupledFreeWakeResult {
                bem,
                wake,
                station_external_inflow,
                outer_iterations: outer_iteration,
                bem_evaluation_count: outer_iteration,
                wake_evaluation_count: outer_iteration,
                final_residual_m_s,
                converged: true,
            };
            if !result.is_finite() {
                return Err(LoadCoupledFreeWakeError::NonFiniteWakeVelocity {
                    station: 0,
                    field: "coupled_result",
                });
            }
            return Ok(result);
        }

        station_external_inflow = relaxed_inflow;
        bem = model
            .evaluate_bem_with_external_inflow(
                operating_point,
                bem_options,
                &station_external_inflow,
            )
            .map_err(|error| LoadCoupledFreeWakeError::BemEvaluation {
                outer_iteration,
                error: Box::new(error),
            })?;

        wake = evaluate_steady_free_wake(model, &bem, options.wake_options).map_err(|error| {
            LoadCoupledFreeWakeError::WakeEvaluation {
                outer_iteration,
                error,
            }
        })?;
    }

    Err(LoadCoupledFreeWakeError::NotConverged {
        iterations: options.max_outer_iterations,
        residual_m_s: final_residual_m_s,
    })
}

fn sample_wake_inflow(
    model: &BladeElementModel,
    wake: &SteadyFreeWakeResult,
    outer_iteration: usize,
) -> Result<Vec<BladeElementExternalInflow>, LoadCoupledFreeWakeError> {
    model
        .stations()
        .iter()
        .enumerate()
        .map(|(station, geometry)| {
            let induced = wake
                .induced_velocity_at(Point3 {
                    x: geometry.radius_m,
                    y: 0.0,
                    z: 0.0,
                })
                .map_err(|error| LoadCoupledFreeWakeError::WakeEvaluation {
                    outer_iteration,
                    error,
                })?;
            if !induced.x.is_finite() {
                return Err(LoadCoupledFreeWakeError::NonFiniteWakeVelocity {
                    station,
                    field: "x_velocity_m_s",
                });
            }
            if !induced.y.is_finite() {
                return Err(LoadCoupledFreeWakeError::NonFiniteWakeVelocity {
                    station,
                    field: "tangential_velocity_m_s",
                });
            }
            if !induced.z.is_finite() {
                return Err(LoadCoupledFreeWakeError::NonFiniteWakeVelocity {
                    station,
                    field: "axial_velocity_m_s",
                });
            }
            Ok(BladeElementExternalInflow {
                axial_velocity_m_s: induced.z,
                tangential_velocity_m_s: induced.y,
            })
        })
        .collect()
}

fn relax_inflow(
    current: &[BladeElementExternalInflow],
    sampled: &[BladeElementExternalInflow],
    relaxation: f64,
) -> Result<Vec<BladeElementExternalInflow>, LoadCoupledFreeWakeError> {
    if current.len() != sampled.len() {
        return Err(LoadCoupledFreeWakeError::NonFiniteWakeVelocity {
            station: 0,
            field: "station_count",
        });
    }
    current
        .iter()
        .zip(sampled)
        .enumerate()
        .map(|(station, (current, sampled))| {
            let axial_velocity_m_s = current.axial_velocity_m_s
                + relaxation * (sampled.axial_velocity_m_s - current.axial_velocity_m_s);
            let tangential_velocity_m_s = current.tangential_velocity_m_s
                + relaxation * (sampled.tangential_velocity_m_s - current.tangential_velocity_m_s);
            if !axial_velocity_m_s.is_finite() {
                return Err(LoadCoupledFreeWakeError::NonFiniteWakeVelocity {
                    station,
                    field: "axial_velocity_m_s",
                });
            }
            if !tangential_velocity_m_s.is_finite() {
                return Err(LoadCoupledFreeWakeError::NonFiniteWakeVelocity {
                    station,
                    field: "tangential_velocity_m_s",
                });
            }
            Ok(BladeElementExternalInflow {
                axial_velocity_m_s,
                tangential_velocity_m_s,
            })
        })
        .collect()
}

fn max_inflow_change(
    previous: &[BladeElementExternalInflow],
    next: &[BladeElementExternalInflow],
) -> f64 {
    previous
        .iter()
        .zip(next)
        .map(|(previous, next)| {
            (next.axial_velocity_m_s - previous.axial_velocity_m_s)
                .abs()
                .max((next.tangential_velocity_m_s - previous.tangential_velocity_m_s).abs())
        })
        .fold(0.0, f64::max)
}

fn validate_options(options: LoadCoupledFreeWakeOptions) -> Result<(), LoadCoupledFreeWakeError> {
    if options.max_outer_iterations == 0 {
        return Err(LoadCoupledFreeWakeError::InvalidOptions {
            field: "max_outer_iterations",
            requirement: "non-zero",
        });
    }
    if !options.inflow_relaxation.is_finite()
        || !(0.0..=1.0).contains(&options.inflow_relaxation)
        || options.inflow_relaxation == 0.0
    {
        return Err(LoadCoupledFreeWakeError::InvalidOptions {
            field: "inflow_relaxation",
            requirement: "finite and in (0, 1]",
        });
    }
    if !options.convergence_tolerance_m_s.is_finite() || options.convergence_tolerance_m_s <= 0.0 {
        return Err(LoadCoupledFreeWakeError::InvalidOptions {
            field: "convergence_tolerance_m_s",
            requirement: "finite and positive",
        });
    }
    Ok(())
}
