//! Axisymmetric weighted-flow and rotating-wall reference contracts.
//!
//! The weighted-flow path consumes the checked production body-fitted
//! meridional FEM result and evaluates the physical `2πr` section measure for
//! volume flow, mass flow, and kinetic-energy flux.  The rotating-wall path is
//! the exact laminar annular Couette solution at each supplied axial station.
//! These are useful verification and coupling seams; they are not a claim of
//! fully coupled three-dimensional rotating-wall CFD, turbulence, or FW-H.

use crate::body_fitted_flow_reference::{
    BodyFittedDuctStation, BodyFittedFlowError, BodyFittedFlowOptions, BodyFittedFlowResult,
    solve_body_fitted_duct_flow,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Versioned identity for axisymmetric weighted-flow diagnostics.
pub const AXISYMMETRIC_FLOW_REFERENCE_SCHEMA_V1: &str =
    "propeller.axisymmetric-weighted-flow-reference.v1";

/// Versioned identity for the annular rotating-wall reference.
pub const AXISYMMETRIC_ROTATING_WALL_SCHEMA_V1: &str =
    "propeller.axisymmetric-rotating-wall-reference.v1";

/// Options for physical axisymmetric section integration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricFlowOptions {
    /// Fluid density used for mass flow and kinetic-energy flux [kg/m³].
    pub density_kg_m3: f64,
    /// Axial tolerance used to select inlet and outlet nodal sections [m].
    pub axial_tolerance_m: f64,
}

impl Default for AxisymmetricFlowOptions {
    fn default() -> Self {
        Self {
            density_kg_m3: 1.2,
            axial_tolerance_m: 1.0e-10,
        }
    }
}

/// Weighted diagnostics for one inlet or outlet section.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricFlowSection {
    /// Section axial position [m].
    pub axial_position_m: f64,
    /// Inner and outer section radii [m].
    pub inner_radius_m: f64,
    pub outer_radius_m: f64,
    /// Geometric annular area [m²].
    pub area_m2: f64,
    /// `2π∫r u_z dr` [m³/s].
    pub volume_flow_m3_s: f64,
    /// Density-weighted volume flow [kg/s].
    pub mass_flow_kg_s: f64,
    /// Volume-flow weighted axial mean velocity [m/s].
    pub weighted_mean_axial_velocity_m_s: f64,
    /// Signed kinetic-energy flux `1/2 ρ 2π∫r u_z³ dr` [W].
    pub kinetic_energy_flux_w: f64,
}

/// Axisymmetric weighted-flow result over a production FEM state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricFlowResult {
    /// Versioned contract identity.
    pub schema: String,
    /// Original checked meridional production FEM result.
    pub meridional_flow: BodyFittedFlowResult,
    /// Weighted inlet and outlet diagnostics.
    pub inlet: AxisymmetricFlowSection,
    pub outlet: AxisymmetricFlowSection,
    /// Relative volume-flow mismatch between selected sections.
    pub relative_volume_flow_mismatch: f64,
    /// Relative mass-flow mismatch; equal to the volume mismatch for constant
    /// density but retained explicitly for downstream ledgers.
    pub relative_mass_flow_mismatch: f64,
}

impl AxisymmetricFlowResult {
    /// Whether all published weighted diagnostics are finite and coherent.
    pub fn is_finite(&self) -> bool {
        self.schema == AXISYMMETRIC_FLOW_REFERENCE_SCHEMA_V1
            && self.meridional_flow.is_finite()
            && self.inlet.is_finite()
            && self.outlet.is_finite()
            && self.relative_volume_flow_mismatch.is_finite()
            && self.relative_volume_flow_mismatch >= 0.0
            && self.relative_mass_flow_mismatch.is_finite()
            && self.relative_mass_flow_mismatch >= 0.0
    }
}

impl AxisymmetricFlowSection {
    fn is_finite(&self) -> bool {
        self.axial_position_m.is_finite()
            && self.inner_radius_m.is_finite()
            && self.inner_radius_m >= 0.0
            && self.outer_radius_m.is_finite()
            && self.outer_radius_m > self.inner_radius_m
            && self.area_m2.is_finite()
            && self.area_m2 > 0.0
            && self.volume_flow_m3_s.is_finite()
            && self.mass_flow_kg_s.is_finite()
            && self.weighted_mean_axial_velocity_m_s.is_finite()
            && self.kinetic_energy_flux_w.is_finite()
    }
}

/// Exact laminar annular Couette solution options.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricRotatingWallOptions {
    /// Inner-wall angular velocity [rad/s].
    pub inner_angular_velocity_rad_s: f64,
    /// Outer-wall angular velocity [rad/s].
    pub outer_angular_velocity_rad_s: f64,
    /// Dynamic viscosity [Pa·s].
    pub dynamic_viscosity_pa_s: f64,
    /// Number of radial samples used for bounded maximum-speed diagnostics.
    pub radial_samples: usize,
    /// Maximum total station/sample work.
    pub maximum_work: usize,
}

impl Default for AxisymmetricRotatingWallOptions {
    fn default() -> Self {
        Self {
            inner_angular_velocity_rad_s: 0.0,
            outer_angular_velocity_rad_s: 0.0,
            dynamic_viscosity_pa_s: 1.8e-5,
            radial_samples: 16,
            maximum_work: 100_000,
        }
    }
}

/// One exact annular Couette station.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricRotatingWallStation {
    /// Axial coordinate [m].
    pub axial_position_m: f64,
    /// Annular radii [m].
    pub inner_radius_m: f64,
    pub outer_radius_m: f64,
    /// Solution coefficients for `u_theta(r) = a r + b/r`.
    pub coefficient_a_s_inv: f64,
    pub coefficient_b_m2_s_inv: f64,
    /// Tangential wall speeds [m/s].
    pub inner_wall_speed_m_s: f64,
    pub outer_wall_speed_m_s: f64,
    /// Signed torque exerted by the fluid on the inner wall per axial length
    /// [N·m/m].
    pub fluid_torque_on_inner_wall_n_m_per_m: f64,
    /// Viscous dissipation per axial length [W/m].
    pub viscous_dissipation_w_per_m: f64,
    /// Bounded sampled maximum tangential speed [m/s].
    pub maximum_tangential_speed_m_s: f64,
}

/// Exact annular rotating-wall result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricRotatingWallResult {
    /// Versioned contract identity.
    pub schema: String,
    /// Station-wise Couette solutions.
    pub stations: Vec<AxisymmetricRotatingWallStation>,
    /// Maximum absolute torque per axial length [N·m/m].
    pub maximum_absolute_torque_n_m_per_m: f64,
    /// Total station-wise dissipation integral [W·m/m].
    pub dissipation_sum_w_per_m: f64,
}

impl AxisymmetricRotatingWallResult {
    /// Whether all published station diagnostics are finite and coherent.
    pub fn is_finite(&self) -> bool {
        self.schema == AXISYMMETRIC_ROTATING_WALL_SCHEMA_V1
            && !self.stations.is_empty()
            && self.maximum_absolute_torque_n_m_per_m.is_finite()
            && self.maximum_absolute_torque_n_m_per_m >= 0.0
            && self.dissipation_sum_w_per_m.is_finite()
            && self.dissipation_sum_w_per_m >= 0.0
            && self
                .stations
                .iter()
                .all(AxisymmetricRotatingWallStation::is_finite)
    }
}

impl AxisymmetricRotatingWallStation {
    fn is_finite(&self) -> bool {
        self.axial_position_m.is_finite()
            && self.inner_radius_m.is_finite()
            && self.inner_radius_m > 0.0
            && self.outer_radius_m.is_finite()
            && self.outer_radius_m > self.inner_radius_m
            && self.coefficient_a_s_inv.is_finite()
            && self.coefficient_b_m2_s_inv.is_finite()
            && self.inner_wall_speed_m_s.is_finite()
            && self.outer_wall_speed_m_s.is_finite()
            && self.fluid_torque_on_inner_wall_n_m_per_m.is_finite()
            && self.viscous_dissipation_w_per_m.is_finite()
            && self.viscous_dissipation_w_per_m >= -1.0e-10
            && self.maximum_tangential_speed_m_s.is_finite()
            && self.maximum_tangential_speed_m_s >= 0.0
    }
}

/// Errors produced by either bounded reference path.
#[derive(Debug, Clone, PartialEq)]
pub enum AxisymmetricFlowError {
    /// The production meridional solve failed.
    Meridional(BodyFittedFlowError),
    /// A scalar option is outside its finite positive domain.
    InvalidOption {
        field: &'static str,
        requirement: &'static str,
    },
    /// The supplied production result is not a valid finite state.
    InvalidFlowResult { field: &'static str },
    /// The requested inlet/outlet section was not represented by enough nodes.
    MissingSection { side: &'static str },
    /// Section radius or ordering is invalid.
    InvalidSection {
        side: &'static str,
        requirement: &'static str,
    },
    /// The rotating-wall profile is invalid.
    InvalidStation {
        index: usize,
        field: &'static str,
        requirement: &'static str,
    },
    /// A station/sample work budget would be exceeded.
    WorkBudgetExceeded { work: usize, maximum: usize },
}

impl fmt::Display for AxisymmetricFlowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meridional(error) => write!(formatter, "meridional flow rejected: {error}"),
            Self::InvalidOption { field, requirement } => {
                write!(
                    formatter,
                    "invalid axisymmetric option {field}: {requirement}"
                )
            }
            Self::InvalidFlowResult { field } => {
                write!(formatter, "invalid axisymmetric flow result field {field}")
            }
            Self::MissingSection { side } => {
                write!(
                    formatter,
                    "axisymmetric {side} section is missing or undersampled"
                )
            }
            Self::InvalidSection { side, requirement } => {
                write!(
                    formatter,
                    "invalid axisymmetric {side} section: {requirement}"
                )
            }
            Self::InvalidStation {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid rotating-wall station {index} field {field}: {requirement}"
            ),
            Self::WorkBudgetExceeded { work, maximum } => write!(
                formatter,
                "axisymmetric rotating-wall work {work} exceeds budget {maximum}"
            ),
        }
    }
}

impl std::error::Error for AxisymmetricFlowError {}

impl From<BodyFittedFlowError> for AxisymmetricFlowError {
    fn from(error: BodyFittedFlowError) -> Self {
        Self::Meridional(error)
    }
}

/// Solve the production body-fitted meridional state and publish weighted
/// axisymmetric inlet/outlet diagnostics.
pub fn solve_axisymmetric_weighted_flow(
    profile: &[BodyFittedDuctStation],
    flow_options: BodyFittedFlowOptions,
    options: AxisymmetricFlowOptions,
) -> Result<AxisymmetricFlowResult, AxisymmetricFlowError> {
    let meridional = solve_body_fitted_duct_flow(profile, flow_options)?;
    evaluate_axisymmetric_weighted_flow(&meridional, options)
}

/// Evaluate physical `2πr` section diagnostics over a checked production FEM
/// result.
pub fn evaluate_axisymmetric_weighted_flow(
    meridional_flow: &BodyFittedFlowResult,
    options: AxisymmetricFlowOptions,
) -> Result<AxisymmetricFlowResult, AxisymmetricFlowError> {
    validate_weighted_options(options)?;
    if !meridional_flow.is_finite() {
        return Err(AxisymmetricFlowError::InvalidFlowResult {
            field: "meridional_flow",
        });
    }

    let minimum_z = meridional_flow
        .nodes
        .iter()
        .map(|node| node.axial_position_m)
        .fold(f64::INFINITY, f64::min);
    let maximum_z = meridional_flow
        .nodes
        .iter()
        .map(|node| node.axial_position_m)
        .fold(f64::NEG_INFINITY, f64::max);
    let inlet = integrate_section(
        &meridional_flow.nodes,
        minimum_z,
        options.axial_tolerance_m,
        options.density_kg_m3,
        "inlet",
    )?;
    let outlet = integrate_section(
        &meridional_flow.nodes,
        maximum_z,
        options.axial_tolerance_m,
        options.density_kg_m3,
        "outlet",
    )?;
    let denominator = inlet
        .volume_flow_m3_s
        .abs()
        .max(outlet.volume_flow_m3_s.abs())
        .max(f64::MIN_POSITIVE);
    let relative_volume_flow_mismatch =
        (outlet.volume_flow_m3_s - inlet.volume_flow_m3_s).abs() / denominator;
    let mass_denominator = inlet
        .mass_flow_kg_s
        .abs()
        .max(outlet.mass_flow_kg_s.abs())
        .max(f64::MIN_POSITIVE);
    let relative_mass_flow_mismatch =
        (outlet.mass_flow_kg_s - inlet.mass_flow_kg_s).abs() / mass_denominator;
    let result = AxisymmetricFlowResult {
        schema: AXISYMMETRIC_FLOW_REFERENCE_SCHEMA_V1.to_owned(),
        meridional_flow: meridional_flow.clone(),
        inlet,
        outlet,
        relative_volume_flow_mismatch,
        relative_mass_flow_mismatch,
    };
    if !result.is_finite() {
        return Err(AxisymmetricFlowError::InvalidFlowResult {
            field: "weighted_diagnostics",
        });
    }
    Ok(result)
}

/// Evaluate the exact laminar annular Couette profile at each duct station.
pub fn evaluate_axisymmetric_rotating_wall(
    profile: &[BodyFittedDuctStation],
    options: AxisymmetricRotatingWallOptions,
) -> Result<AxisymmetricRotatingWallResult, AxisymmetricFlowError> {
    validate_rotating_options(options)?;
    if profile.is_empty() {
        return Err(AxisymmetricFlowError::InvalidSection {
            side: "profile",
            requirement: "at least one station",
        });
    }
    let work = profile.len().checked_mul(options.radial_samples).ok_or(
        AxisymmetricFlowError::WorkBudgetExceeded {
            work: usize::MAX,
            maximum: options.maximum_work,
        },
    )?;
    if work > options.maximum_work {
        return Err(AxisymmetricFlowError::WorkBudgetExceeded {
            work,
            maximum: options.maximum_work,
        });
    }

    let mut stations = Vec::with_capacity(profile.len());
    for (index, station) in profile.iter().enumerate() {
        validate_rotating_station(index, station)?;
        let inner_squared = station.inner_radius_m * station.inner_radius_m;
        let outer_squared = station.outer_radius_m * station.outer_radius_m;
        let denominator = outer_squared - inner_squared;
        let coefficient_b = (options.inner_angular_velocity_rad_s
            - options.outer_angular_velocity_rad_s)
            * inner_squared
            * outer_squared
            / denominator;
        let coefficient_a = options.inner_angular_velocity_rad_s - coefficient_b / inner_squared;
        let inner_wall_speed = options.inner_angular_velocity_rad_s * station.inner_radius_m;
        let outer_wall_speed = options.outer_angular_velocity_rad_s * station.outer_radius_m;
        let torque = -4.0 * std::f64::consts::PI * options.dynamic_viscosity_pa_s * coefficient_b;
        let dissipation = 4.0
            * std::f64::consts::PI
            * options.dynamic_viscosity_pa_s
            * coefficient_b
            * (options.inner_angular_velocity_rad_s - options.outer_angular_velocity_rad_s);
        let maximum_tangential_speed = (0..options.radial_samples)
            .map(|sample| {
                let fraction = sample as f64 / (options.radial_samples - 1) as f64;
                let radius = station.inner_radius_m
                    + fraction * (station.outer_radius_m - station.inner_radius_m);
                (coefficient_a * radius + coefficient_b / radius).abs()
            })
            .fold(0.0, f64::max);
        stations.push(AxisymmetricRotatingWallStation {
            axial_position_m: station.axial_position_m,
            inner_radius_m: station.inner_radius_m,
            outer_radius_m: station.outer_radius_m,
            coefficient_a_s_inv: coefficient_a,
            coefficient_b_m2_s_inv: coefficient_b,
            inner_wall_speed_m_s: inner_wall_speed,
            outer_wall_speed_m_s: outer_wall_speed,
            fluid_torque_on_inner_wall_n_m_per_m: torque,
            viscous_dissipation_w_per_m: dissipation.max(0.0),
            maximum_tangential_speed_m_s: maximum_tangential_speed,
        });
    }
    let result = AxisymmetricRotatingWallResult {
        schema: AXISYMMETRIC_ROTATING_WALL_SCHEMA_V1.to_owned(),
        maximum_absolute_torque_n_m_per_m: stations
            .iter()
            .map(|station| station.fluid_torque_on_inner_wall_n_m_per_m.abs())
            .fold(0.0, f64::max),
        dissipation_sum_w_per_m: stations
            .iter()
            .map(|station| station.viscous_dissipation_w_per_m)
            .sum(),
        stations,
    };
    if !result.is_finite() {
        return Err(AxisymmetricFlowError::InvalidFlowResult {
            field: "rotating_wall_diagnostics",
        });
    }
    Ok(result)
}

fn integrate_section(
    nodes: &[crate::body_fitted_flow_reference::BodyFittedFlowNode],
    axial_position_m: f64,
    axial_tolerance_m: f64,
    density_kg_m3: f64,
    side: &'static str,
) -> Result<AxisymmetricFlowSection, AxisymmetricFlowError> {
    let mut section = nodes
        .iter()
        .filter(|node| (node.axial_position_m - axial_position_m).abs() <= axial_tolerance_m)
        .map(|node| (node.radial_position_m, node.axial_velocity_m_s))
        .collect::<Vec<_>>();
    if section.len() < 2 {
        return Err(AxisymmetricFlowError::MissingSection { side });
    }
    section.sort_by(|left, right| left.0.total_cmp(&right.0));
    if section
        .iter()
        .any(|(radius, velocity)| !radius.is_finite() || *radius < 0.0 || !velocity.is_finite())
    {
        return Err(AxisymmetricFlowError::InvalidSection {
            side,
            requirement: "finite non-negative radii and finite axial velocity",
        });
    }
    let mut volume_integral = 0.0;
    let mut energy_integral = 0.0;
    for pair in section.windows(2) {
        let (radius_a, velocity_a) = pair[0];
        let (radius_b, velocity_b) = pair[1];
        let delta_radius = radius_b - radius_a;
        if delta_radius <= 0.0 {
            return Err(AxisymmetricFlowError::InvalidSection {
                side,
                requirement: "strictly increasing radial nodes",
            });
        }
        volume_integral += 0.5 * delta_radius * (radius_a * velocity_a + radius_b * velocity_b);
        energy_integral +=
            0.5 * delta_radius * (radius_a * velocity_a.powi(3) + radius_b * velocity_b.powi(3));
    }
    let inner_radius_m = section.first().expect("section length checked").0;
    let outer_radius_m = section.last().expect("section length checked").0;
    if outer_radius_m <= inner_radius_m {
        return Err(AxisymmetricFlowError::InvalidSection {
            side,
            requirement: "positive annular area",
        });
    }
    let area_m2 =
        std::f64::consts::PI * (outer_radius_m * outer_radius_m - inner_radius_m * inner_radius_m);
    let volume_flow_m3_s = 2.0 * std::f64::consts::PI * volume_integral;
    let mass_flow_kg_s = density_kg_m3 * volume_flow_m3_s;
    let weighted_mean_axial_velocity_m_s = volume_flow_m3_s / area_m2;
    let kinetic_energy_flux_w = density_kg_m3 * std::f64::consts::PI * energy_integral;
    let result = AxisymmetricFlowSection {
        axial_position_m,
        inner_radius_m,
        outer_radius_m,
        area_m2,
        volume_flow_m3_s,
        mass_flow_kg_s,
        weighted_mean_axial_velocity_m_s,
        kinetic_energy_flux_w,
    };
    if !result.is_finite() {
        return Err(AxisymmetricFlowError::InvalidSection {
            side,
            requirement: "finite weighted section diagnostics",
        });
    }
    Ok(result)
}

fn validate_weighted_options(
    options: AxisymmetricFlowOptions,
) -> Result<(), AxisymmetricFlowError> {
    if !options.density_kg_m3.is_finite() || options.density_kg_m3 <= 0.0 {
        return Err(AxisymmetricFlowError::InvalidOption {
            field: "density_kg_m3",
            requirement: "finite and positive",
        });
    }
    if !options.axial_tolerance_m.is_finite() || options.axial_tolerance_m <= 0.0 {
        return Err(AxisymmetricFlowError::InvalidOption {
            field: "axial_tolerance_m",
            requirement: "finite and positive",
        });
    }
    Ok(())
}

fn validate_rotating_options(
    options: AxisymmetricRotatingWallOptions,
) -> Result<(), AxisymmetricFlowError> {
    for (field, value) in [
        (
            "inner_angular_velocity_rad_s",
            options.inner_angular_velocity_rad_s,
        ),
        (
            "outer_angular_velocity_rad_s",
            options.outer_angular_velocity_rad_s,
        ),
        ("dynamic_viscosity_pa_s", options.dynamic_viscosity_pa_s),
    ] {
        if !value.is_finite() {
            return Err(AxisymmetricFlowError::InvalidOption {
                field,
                requirement: "finite",
            });
        }
    }
    if options.dynamic_viscosity_pa_s <= 0.0 {
        return Err(AxisymmetricFlowError::InvalidOption {
            field: "dynamic_viscosity_pa_s",
            requirement: "positive",
        });
    }
    if options.radial_samples < 2 {
        return Err(AxisymmetricFlowError::InvalidOption {
            field: "radial_samples",
            requirement: "at least two",
        });
    }
    if options.maximum_work == 0 {
        return Err(AxisymmetricFlowError::InvalidOption {
            field: "maximum_work",
            requirement: "positive",
        });
    }
    Ok(())
}

fn validate_rotating_station(
    index: usize,
    station: &BodyFittedDuctStation,
) -> Result<(), AxisymmetricFlowError> {
    if !station.axial_position_m.is_finite() {
        return Err(AxisymmetricFlowError::InvalidStation {
            index,
            field: "axial_position_m",
            requirement: "finite",
        });
    }
    if !station.inner_radius_m.is_finite() || station.inner_radius_m <= 0.0 {
        return Err(AxisymmetricFlowError::InvalidStation {
            index,
            field: "inner_radius_m",
            requirement: "finite and positive for an annular rotating-wall solution",
        });
    }
    if !station.outer_radius_m.is_finite() || station.outer_radius_m <= station.inner_radius_m {
        return Err(AxisymmetricFlowError::InvalidStation {
            index,
            field: "outer_radius_m",
            requirement: "finite and greater than inner radius",
        });
    }
    Ok(())
}
