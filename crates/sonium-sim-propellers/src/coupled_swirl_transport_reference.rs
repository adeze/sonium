//! Checked outer coupling of meridional flow and transported axisymmetric swirl.
//!
//! This is a bounded steady Picard reference, not a three-dimensional rotating
//! CFD solver.  The meridional FEM solve supplies the advector for the scalar
//! azimuthal equation, and the resulting `u_theta` field supplies the radial
//! centrifugal body force `rho*u_theta^2/r` on the next meridional solve.

use crate::body_fitted_flow_reference::{
    BodyFittedDuctStation, BodyFittedFlowError, BodyFittedFlowOptions, BodyFittedFlowResult,
    solve_body_fitted_duct_flow, solve_body_fitted_duct_flow_with_centrifugal_source,
};
use crate::swirl_transport_reference::{
    BodyFittedSwirlError, BodyFittedSwirlOptions, BodyFittedSwirlResult,
    solve_body_fitted_swirl_transport_with_meridional,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Versioned identity for the two-way meridional/swirl reference.
pub const BODY_FITTED_COUPLED_SWIRL_TRANSPORT_SCHEMA_V1: &str =
    "propeller.body-fitted-axisymmetric-coupled-swirl-transport.v1";

/// Controls the bounded outer meridional/swirl Picard iteration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyFittedCoupledSwirlOptions {
    /// Maximum meridional/swirl coupling iterations.
    pub max_outer_iterations: usize,
    /// Minimum iterations before the outer residual may be accepted.
    pub min_outer_iterations: usize,
    /// Relative nodal tangential-velocity change accepted as convergence.
    pub outer_tolerance: f64,
    /// Under-relaxation applied to the tangential field used as the next source.
    pub outer_relaxation: f64,
    /// Reject a finite but unconverged outer state when true.
    pub require_convergence: bool,
}

impl Default for BodyFittedCoupledSwirlOptions {
    fn default() -> Self {
        Self {
            max_outer_iterations: 20,
            min_outer_iterations: 2,
            outer_tolerance: 1.0e-7,
            outer_relaxation: 0.7,
            require_convergence: true,
        }
    }
}

/// Result of a checked two-way steady meridional/swirl reference solve.
#[derive(Debug, PartialEq)]
pub struct BodyFittedCoupledSwirlResult {
    /// Contract schema.
    pub schema: String,
    /// Final meridional state.
    pub meridional: BodyFittedFlowResult,
    /// Final transported swirl state, using `meridional` as its advector.
    pub swirl: BodyFittedSwirlResult,
    /// Number of outer coupling updates performed.
    pub outer_iterations: usize,
    /// Final relative nodal tangential-velocity change.
    pub outer_residual: f64,
    /// Whether the outer fixed-point criterion was met.
    pub converged: bool,
    /// Maximum radial acceleration from the final transported field [m/s²].
    pub maximum_centrifugal_acceleration_m_s2: f64,
}

impl BodyFittedCoupledSwirlResult {
    /// Whether all coupled fields and diagnostics are finite and coherent.
    pub fn is_finite(&self) -> bool {
        self.schema == BODY_FITTED_COUPLED_SWIRL_TRANSPORT_SCHEMA_V1
            && self.meridional.is_finite()
            && self.swirl.is_finite()
            && self.swirl.meridional == self.meridional
            && self.outer_iterations > 0
            && self.outer_residual.is_finite()
            && self.outer_residual >= 0.0
            && self.maximum_centrifugal_acceleration_m_s2.is_finite()
            && self.maximum_centrifugal_acceleration_m_s2 >= 0.0
            && self.swirl.nodes.len() == self.meridional.nodes.len()
    }
}

/// Failure-closed errors from the two-way reference coupling.
#[derive(Debug, Clone, PartialEq)]
pub enum BodyFittedCoupledSwirlError {
    /// The underlying meridional solve rejected its input or state.
    Meridional(BodyFittedFlowError),
    /// The underlying scalar swirl solve rejected its input or state.
    Swirl(BodyFittedSwirlError),
    /// An outer option is outside the admitted bounded domain.
    InvalidOption {
        /// Invalid option name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The source field could not be aligned with the body-fitted mesh.
    InvalidSourceField,
    /// The fixed-point iteration did not meet its requested tolerance.
    OuterSolverDidNotConverge { iterations: usize, residual: f64 },
    /// A published coupled result failed its finite-state contract.
    InvalidResult(&'static str),
}

impl fmt::Display for BodyFittedCoupledSwirlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meridional(error) => write!(formatter, "meridional flow failed: {error}"),
            Self::Swirl(error) => write!(formatter, "swirl transport failed: {error}"),
            Self::InvalidOption { field, requirement } => {
                write!(
                    formatter,
                    "invalid coupled swirl option {field}: {requirement}"
                )
            }
            Self::InvalidSourceField => {
                formatter.write_str("coupled swirl source field is invalid")
            }
            Self::OuterSolverDidNotConverge {
                iterations,
                residual,
            } => write!(
                formatter,
                "coupled swirl iteration did not converge after {iterations} iterations (residual {residual})"
            ),
            Self::InvalidResult(field) => {
                write!(formatter, "coupled swirl result field {field} is invalid")
            }
        }
    }
}

impl std::error::Error for BodyFittedCoupledSwirlError {}

impl From<BodyFittedFlowError> for BodyFittedCoupledSwirlError {
    fn from(error: BodyFittedFlowError) -> Self {
        Self::Meridional(error)
    }
}

impl From<BodyFittedSwirlError> for BodyFittedCoupledSwirlError {
    fn from(error: BodyFittedSwirlError) -> Self {
        Self::Swirl(error)
    }
}

/// Solve the bounded two-way steady axisymmetric meridional/swirl reference.
pub fn solve_body_fitted_coupled_swirl_transport(
    profile: &[BodyFittedDuctStation],
    flow_options: BodyFittedFlowOptions,
    swirl_options: BodyFittedSwirlOptions,
    coupled_options: BodyFittedCoupledSwirlOptions,
) -> Result<BodyFittedCoupledSwirlResult, BodyFittedCoupledSwirlError> {
    validate_options(coupled_options)?;

    // Establish a finite initial transported state using the existing
    // rotating-wall source (or zero source when no model is configured).
    let mut meridional = solve_body_fitted_duct_flow(profile, flow_options)?;
    let mut swirl = solve_body_fitted_swirl_transport_with_meridional(
        profile,
        flow_options,
        swirl_options,
        meridional.clone(),
    )?;
    let expected_nodes = (flow_options.axial_elements + 1)
        .checked_mul(flow_options.radial_elements + 1)
        .ok_or(BodyFittedCoupledSwirlError::InvalidSourceField)?;
    if swirl.nodes.len() != expected_nodes || meridional.nodes.len() != expected_nodes {
        return Err(BodyFittedCoupledSwirlError::InvalidSourceField);
    }

    let mut source_tangential = swirl
        .nodes
        .iter()
        .map(|node| node.tangential_velocity_m_s)
        .collect::<Vec<_>>();
    let mut outer_residual = f64::INFINITY;
    let mut converged = false;
    let mut outer_iterations = 0;

    for iteration in 1..=coupled_options.max_outer_iterations {
        let source_values = source_tangential.clone();
        let source_profile = profile.to_vec();
        let source = Arc::new(move |position: [f64; 3]| {
            let tangential = interpolate_tangential_velocity(
                &source_profile,
                flow_options.axial_elements,
                flow_options.radial_elements,
                &source_values,
                position[0],
                position[1],
            );
            if position[0] <= 0.0 {
                f64::NAN
            } else {
                tangential * tangential / position[0]
            }
        });
        let next_meridional = solve_body_fitted_duct_flow_with_centrifugal_source(
            profile,
            flow_options,
            Some(source),
        )?;
        let next_swirl = solve_body_fitted_swirl_transport_with_meridional(
            profile,
            flow_options,
            swirl_options,
            next_meridional.clone(),
        )?;
        if next_swirl.nodes.len() != source_tangential.len()
            || next_swirl.nodes.iter().any(|node| {
                !node.tangential_velocity_m_s.is_finite()
                    || !node.radial_position_m.is_finite()
                    || node.radial_position_m <= 0.0
            })
        {
            return Err(BodyFittedCoupledSwirlError::InvalidSourceField);
        }

        let old_norm = source_tangential
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let new_values = next_swirl
            .nodes
            .iter()
            .map(|node| node.tangential_velocity_m_s)
            .collect::<Vec<_>>();
        let delta_norm = source_tangential
            .iter()
            .zip(new_values.iter())
            .map(|(old, new)| (new - old) * (new - old))
            .sum::<f64>()
            .sqrt();
        outer_residual = delta_norm
            / old_norm
                .max(new_values.iter().map(|v| v * v).sum::<f64>().sqrt())
                .max(1.0);
        source_tangential
            .iter_mut()
            .zip(new_values.iter())
            .for_each(|(old, new)| {
                *old += coupled_options.outer_relaxation * (new - *old);
            });

        meridional = next_meridional;
        swirl = next_swirl;
        outer_iterations = iteration;
        if iteration >= coupled_options.min_outer_iterations
            && outer_residual <= coupled_options.outer_tolerance
        {
            converged = true;
            break;
        }
    }

    if coupled_options.require_convergence && !converged {
        return Err(BodyFittedCoupledSwirlError::OuterSolverDidNotConverge {
            iterations: outer_iterations,
            residual: outer_residual,
        });
    }
    let maximum_centrifugal_acceleration_m_s2 = swirl
        .nodes
        .iter()
        .map(|node| {
            node.tangential_velocity_m_s * node.tangential_velocity_m_s / node.radial_position_m
        })
        .fold(0.0, f64::max);
    let result = BodyFittedCoupledSwirlResult {
        schema: BODY_FITTED_COUPLED_SWIRL_TRANSPORT_SCHEMA_V1.to_owned(),
        meridional,
        swirl,
        outer_iterations,
        outer_residual,
        converged,
        maximum_centrifugal_acceleration_m_s2,
    };
    if !result.is_finite() {
        return Err(BodyFittedCoupledSwirlError::InvalidResult("finite_state"));
    }
    Ok(result)
}

fn validate_options(
    options: BodyFittedCoupledSwirlOptions,
) -> Result<(), BodyFittedCoupledSwirlError> {
    if options.max_outer_iterations == 0
        || options.min_outer_iterations == 0
        || options.min_outer_iterations > options.max_outer_iterations
    {
        return Err(BodyFittedCoupledSwirlError::InvalidOption {
            field: "outer_iterations",
            requirement: "positive and internally ordered",
        });
    }
    if !options.outer_tolerance.is_finite() || options.outer_tolerance <= 0.0 {
        return Err(BodyFittedCoupledSwirlError::InvalidOption {
            field: "outer_tolerance",
            requirement: "finite positive",
        });
    }
    if !(0.0 < options.outer_relaxation && options.outer_relaxation <= 1.0) {
        return Err(BodyFittedCoupledSwirlError::InvalidOption {
            field: "outer_relaxation",
            requirement: "in (0, 1]",
        });
    }
    Ok(())
}

fn interpolate_tangential_velocity(
    profile: &[BodyFittedDuctStation],
    axial_elements: usize,
    radial_elements: usize,
    values: &[f64],
    radius: f64,
    axial: f64,
) -> f64 {
    let stride = radial_elements + 1;
    let z_min = profile[0].axial_position_m;
    let z_max = profile[profile.len() - 1].axial_position_m;
    let z_fraction = ((axial - z_min) / (z_max - z_min)).clamp(0.0, 1.0);
    let axial_position = z_fraction * axial_elements as f64;
    let axial_index = (axial_position.floor() as usize).min(axial_elements.saturating_sub(1));
    let axial_fraction = axial_position - axial_index as f64;
    let z0 = z_min + (axial_index as f64 / axial_elements as f64) * (z_max - z_min);
    let z1 = z_min + ((axial_index + 1) as f64 / axial_elements as f64) * (z_max - z_min);
    let (inner0, outer0) = interpolate_profile(profile, z0);
    let (inner1, outer1) = interpolate_profile(profile, z1);
    let inner = inner0 + axial_fraction * (inner1 - inner0);
    let outer = outer0 + axial_fraction * (outer1 - outer0);
    let radial_fraction = ((radius - inner) / (outer - inner)).clamp(0.0, 1.0);
    let radial_position = radial_fraction * radial_elements as f64;
    let radial_index = (radial_position.floor() as usize).min(radial_elements.saturating_sub(1));
    let radial_weight = radial_position - radial_index as f64;
    let index00 = axial_index * stride + radial_index;
    let index01 = index00 + 1;
    let index10 = (axial_index + 1) * stride + radial_index;
    let index11 = index10 + 1;
    let lower = values[index00] + radial_weight * (values[index01] - values[index00]);
    let upper = values[index10] + radial_weight * (values[index11] - values[index10]);
    lower + axial_fraction * (upper - lower)
}

fn interpolate_profile(profile: &[BodyFittedDuctStation], axial: f64) -> (f64, f64) {
    if axial <= profile[0].axial_position_m {
        return (profile[0].inner_radius_m, profile[0].outer_radius_m);
    }
    if axial >= profile[profile.len() - 1].axial_position_m {
        let station = profile[profile.len() - 1];
        return (station.inner_radius_m, station.outer_radius_m);
    }
    for pair in profile.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if axial <= right.axial_position_m {
            let fraction =
                (axial - left.axial_position_m) / (right.axial_position_m - left.axial_position_m);
            return (
                left.inner_radius_m + fraction * (right.inner_radius_m - left.inner_radius_m),
                left.outer_radius_m + fraction * (right.outer_radius_m - left.outer_radius_m),
            );
        }
    }
    unreachable!("profile bounds handled above")
}
