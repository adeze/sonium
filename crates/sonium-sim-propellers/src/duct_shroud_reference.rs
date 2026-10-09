//! Bounded quasi-one-dimensional duct/shroud mean-flow reference.
//!
//! The reference treats a prescribed geometry-derived station ladder as a constant-
//! density streamtube. It enforces continuity, accounts for Darcy wall loss
//! and a caller-supplied minor-loss coefficient, and reports Reynolds and Mach
//! diagnostics. It is a reduced-order duct/shroud coupling contract, not a
//! three-dimensional body-fitted CFD, compressible-flow, or acoustic solver.

use std::fmt;

/// One axial station in a prescribed duct or shroud centerline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuctShroudStation {
    /// Axial station position [m].
    pub axial_position_m: f64,
    /// Flow area normal to the station [m²].
    pub flow_area_m2: f64,
    /// Hydraulic diameter of the station section [m].
    pub hydraulic_diameter_m: f64,
}

/// Controls for the bounded quasi-one-dimensional duct/shroud reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuctShroudOptions {
    /// Positive mass flow in the increasing axial direction [kg/s].
    pub mass_flow_kg_s: f64,
    /// Constant reference density [kg/m³].
    pub density_kg_m3: f64,
    /// Dynamic viscosity used for Reynolds number and wall loss [Pa·s].
    pub dynamic_viscosity_pa_s: f64,
    /// Static pressure at the first station [Pa].
    pub inlet_static_pressure_pa: f64,
    /// Sound speed used only for the reported Mach number [m/s].
    pub speed_of_sound_m_s: f64,
    /// Dimensionless minor-loss coefficient applied at each downstream step.
    pub minor_loss_coefficient: f64,
    /// Maximum admitted station velocity [m/s].
    pub maximum_velocity_m_s: f64,
    /// Maximum admitted station Mach number [-].
    pub maximum_mach: f64,
    /// Maximum admitted station count [-].
    pub maximum_stations: usize,
}

impl Default for DuctShroudOptions {
    fn default() -> Self {
        Self {
            mass_flow_kg_s: 0.5,
            density_kg_m3: 1.2,
            dynamic_viscosity_pa_s: 1.8e-5,
            inlet_static_pressure_pa: 101_325.0,
            speed_of_sound_m_s: 343.0,
            minor_loss_coefficient: 0.02,
            maximum_velocity_m_s: 200.0,
            maximum_mach: 0.6,
            maximum_stations: 4096,
        }
    }
}

/// Evaluated mean-flow state at one duct/shroud station.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuctShroudFlowStation {
    /// Zero-based station index.
    pub station_index: usize,
    /// Axial station position [m].
    pub axial_position_m: f64,
    /// Flow area [m²].
    pub flow_area_m2: f64,
    /// Hydraulic diameter [m].
    pub hydraulic_diameter_m: f64,
    /// Positive axial mean velocity [m/s].
    pub velocity_m_s: f64,
    /// Static pressure [Pa].
    pub static_pressure_pa: f64,
    /// Local dynamic pressure [Pa].
    pub dynamic_pressure_pa: f64,
    /// Reynolds number based on hydraulic diameter [-].
    pub reynolds_number: f64,
    /// Local Mach number [-].
    pub mach_number: f64,
    /// Smooth-pipe Darcy friction factor [-].
    pub darcy_friction_factor: f64,
    /// Continuity residual `rho A v - mdot` [kg/s].
    pub continuity_residual_kg_s: f64,
}

/// Result of the bounded duct/shroud station evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct DuctShroudFlowResult {
    /// Evaluated station states in input order.
    pub stations: Vec<DuctShroudFlowStation>,
    /// Accumulated Darcy wall loss [Pa].
    pub total_wall_loss_pa: f64,
    /// Accumulated minor-transition loss [Pa].
    pub total_minor_loss_pa: f64,
    /// Static pressure at the last station [Pa].
    pub outlet_static_pressure_pa: f64,
    /// Maximum station velocity [m/s].
    pub maximum_velocity_m_s: f64,
    /// Maximum station Mach number [-].
    pub maximum_mach: f64,
    /// Maximum absolute continuity residual [kg/s].
    pub maximum_continuity_residual_kg_s: f64,
}

impl DuctShroudFlowResult {
    /// Whether all station states are finite, ordered, and physically bounded.
    pub fn is_finite(&self) -> bool {
        self.total_wall_loss_pa.is_finite()
            && self.total_wall_loss_pa >= 0.0
            && self.total_minor_loss_pa.is_finite()
            && self.total_minor_loss_pa >= 0.0
            && self.outlet_static_pressure_pa.is_finite()
            && self.outlet_static_pressure_pa >= 0.0
            && self.maximum_velocity_m_s.is_finite()
            && self.maximum_velocity_m_s > 0.0
            && self.maximum_mach.is_finite()
            && self.maximum_mach > 0.0
            && self.maximum_continuity_residual_kg_s.is_finite()
            && self.maximum_continuity_residual_kg_s >= 0.0
            && !self.stations.is_empty()
            && self
                .stations
                .windows(2)
                .all(|pair| pair[1].axial_position_m > pair[0].axial_position_m)
            && self.stations.iter().all(|station| {
                station.axial_position_m.is_finite()
                    && station.flow_area_m2.is_finite()
                    && station.flow_area_m2 > 0.0
                    && station.hydraulic_diameter_m.is_finite()
                    && station.hydraulic_diameter_m > 0.0
                    && station.velocity_m_s.is_finite()
                    && station.velocity_m_s > 0.0
                    && station.static_pressure_pa.is_finite()
                    && station.static_pressure_pa >= 0.0
                    && station.dynamic_pressure_pa.is_finite()
                    && station.dynamic_pressure_pa > 0.0
                    && station.reynolds_number.is_finite()
                    && station.reynolds_number > 0.0
                    && station.mach_number.is_finite()
                    && station.mach_number > 0.0
                    && station.darcy_friction_factor.is_finite()
                    && station.darcy_friction_factor > 0.0
                    && station.continuity_residual_kg_s.is_finite()
            })
    }
}

/// Failure-closed errors from the duct/shroud reference.
#[derive(Debug, Clone, PartialEq)]
pub enum DuctShroudError {
    /// An option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No station geometry was supplied.
    EmptyStations,
    /// The station count exceeds the declared work budget.
    TooManyStations { count: usize, maximum: usize },
    /// A station field is malformed.
    InvalidStation {
        /// Zero-based station index.
        index: usize,
        /// Station field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Axial station positions must increase strictly.
    NonMonotonicAxial { index: usize },
    /// Static pressure became nonphysical under the reduced-order losses.
    NegativeStaticPressure { index: usize, pressure_pa: f64 },
    /// A station velocity exceeds the declared bound.
    VelocityExceeded {
        /// Zero-based station index.
        index: usize,
        /// Computed velocity [m/s].
        velocity_m_s: f64,
        /// Maximum admitted velocity [m/s].
        maximum_m_s: f64,
    },
    /// A station Mach number exceeds the declared bound.
    MachExceeded {
        /// Zero-based station index.
        index: usize,
        /// Computed Mach number [-].
        mach: f64,
        /// Maximum admitted Mach number [-].
        maximum: f64,
    },
    /// A derived state became non-finite.
    NonFiniteResult { field: &'static str },
}

impl fmt::Display for DuctShroudError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "invalid duct/shroud option {field}: {requirement}"
                )
            }
            Self::EmptyStations => {
                write!(formatter, "duct/shroud station history must not be empty")
            }
            Self::TooManyStations { count, maximum } => write!(
                formatter,
                "duct/shroud station count {count} exceeds {maximum}"
            ),
            Self::InvalidStation {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid duct/shroud station {index} field {field}: {requirement}"
            ),
            Self::NonMonotonicAxial { index } => {
                write!(
                    formatter,
                    "duct/shroud axial position is not increasing at {index}"
                )
            }
            Self::NegativeStaticPressure { index, pressure_pa } => write!(
                formatter,
                "duct/shroud static pressure at {index} is {pressure_pa} Pa"
            ),
            Self::VelocityExceeded {
                index,
                velocity_m_s,
                maximum_m_s,
            } => write!(
                formatter,
                "duct/shroud velocity {velocity_m_s} m/s at {index} exceeds {maximum_m_s} m/s"
            ),
            Self::MachExceeded {
                index,
                mach,
                maximum,
            } => write!(
                formatter,
                "duct/shroud Mach {mach} at {index} exceeds {maximum}"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite duct/shroud result field {field}")
            }
        }
    }
}

impl std::error::Error for DuctShroudError {}

/// Evaluate a prescribed quasi-one-dimensional duct/shroud station ladder.
pub fn evaluate_duct_shroud_reference(
    stations: &[DuctShroudStation],
    options: DuctShroudOptions,
) -> Result<DuctShroudFlowResult, DuctShroudError> {
    validate_options(options)?;
    if stations.is_empty() {
        return Err(DuctShroudError::EmptyStations);
    }
    if stations.len() > options.maximum_stations {
        return Err(DuctShroudError::TooManyStations {
            count: stations.len(),
            maximum: options.maximum_stations,
        });
    }

    let mut evaluated: Vec<DuctShroudFlowStation> = Vec::with_capacity(stations.len());
    let mut total_wall_loss_pa = 0.0;
    let mut total_minor_loss_pa = 0.0;

    for (index, &station) in stations.iter().enumerate() {
        validate_station(index, station)?;
        if index > 0 && station.axial_position_m <= stations[index - 1].axial_position_m {
            return Err(DuctShroudError::NonMonotonicAxial { index });
        }

        let velocity_m_s = options.mass_flow_kg_s / (options.density_kg_m3 * station.flow_area_m2);
        let dynamic_pressure_pa = 0.5 * options.density_kg_m3 * velocity_m_s.powi(2);
        let reynolds_number = options.density_kg_m3 * velocity_m_s * station.hydraulic_diameter_m
            / options.dynamic_viscosity_pa_s;
        let mach_number = velocity_m_s / options.speed_of_sound_m_s;
        let darcy_friction_factor = darcy_friction_factor(reynolds_number);
        let continuity_residual_kg_s =
            options.density_kg_m3 * station.flow_area_m2 * velocity_m_s - options.mass_flow_kg_s;

        if !velocity_m_s.is_finite()
            || !dynamic_pressure_pa.is_finite()
            || !reynolds_number.is_finite()
            || !mach_number.is_finite()
            || !darcy_friction_factor.is_finite()
            || !continuity_residual_kg_s.is_finite()
        {
            return Err(DuctShroudError::NonFiniteResult { field: "station" });
        }
        if velocity_m_s > options.maximum_velocity_m_s {
            return Err(DuctShroudError::VelocityExceeded {
                index,
                velocity_m_s,
                maximum_m_s: options.maximum_velocity_m_s,
            });
        }
        if mach_number > options.maximum_mach {
            return Err(DuctShroudError::MachExceeded {
                index,
                mach: mach_number,
                maximum: options.maximum_mach,
            });
        }

        let static_pressure_pa = if index == 0 {
            options.inlet_static_pressure_pa
        } else {
            let previous = evaluated[index - 1];
            let length_m = station.axial_position_m - previous.axial_position_m;
            let wall_loss_pa = darcy_friction_factor * length_m / station.hydraulic_diameter_m
                * dynamic_pressure_pa;
            let minor_loss_pa = options.minor_loss_coefficient * dynamic_pressure_pa;
            total_wall_loss_pa += wall_loss_pa;
            total_minor_loss_pa += minor_loss_pa;
            previous.static_pressure_pa + previous.dynamic_pressure_pa
                - dynamic_pressure_pa
                - wall_loss_pa
                - minor_loss_pa
        };
        if !static_pressure_pa.is_finite() {
            return Err(DuctShroudError::NonFiniteResult {
                field: "static_pressure_pa",
            });
        }
        if static_pressure_pa < 0.0 {
            return Err(DuctShroudError::NegativeStaticPressure {
                index,
                pressure_pa: static_pressure_pa,
            });
        }

        evaluated.push(DuctShroudFlowStation {
            station_index: index,
            axial_position_m: station.axial_position_m,
            flow_area_m2: station.flow_area_m2,
            hydraulic_diameter_m: station.hydraulic_diameter_m,
            velocity_m_s,
            static_pressure_pa,
            dynamic_pressure_pa,
            reynolds_number,
            mach_number,
            darcy_friction_factor,
            continuity_residual_kg_s,
        });
    }

    let result = DuctShroudFlowResult {
        outlet_static_pressure_pa: evaluated
            .last()
            .ok_or(DuctShroudError::NonFiniteResult { field: "stations" })?
            .static_pressure_pa,
        maximum_velocity_m_s: evaluated
            .iter()
            .map(|station| station.velocity_m_s)
            .fold(0.0, f64::max),
        maximum_mach: evaluated
            .iter()
            .map(|station| station.mach_number)
            .fold(0.0, f64::max),
        maximum_continuity_residual_kg_s: evaluated
            .iter()
            .map(|station| station.continuity_residual_kg_s.abs())
            .fold(0.0, f64::max),
        stations: evaluated,
        total_wall_loss_pa,
        total_minor_loss_pa,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(DuctShroudError::NonFiniteResult {
            field: "duct_shroud_result",
        })
    }
}

fn darcy_friction_factor(reynolds_number: f64) -> f64 {
    if reynolds_number < 2_300.0 {
        64.0 / reynolds_number
    } else {
        0.3164 / reynolds_number.powf(0.25)
    }
}

fn validate_options(options: DuctShroudOptions) -> Result<(), DuctShroudError> {
    for (field, value, requirement) in [
        (
            "mass_flow_kg_s",
            options.mass_flow_kg_s,
            "finite and positive",
        ),
        (
            "density_kg_m3",
            options.density_kg_m3,
            "finite and positive",
        ),
        (
            "dynamic_viscosity_pa_s",
            options.dynamic_viscosity_pa_s,
            "finite and positive",
        ),
        (
            "inlet_static_pressure_pa",
            options.inlet_static_pressure_pa,
            "finite and positive",
        ),
        (
            "speed_of_sound_m_s",
            options.speed_of_sound_m_s,
            "finite and positive",
        ),
        (
            "maximum_velocity_m_s",
            options.maximum_velocity_m_s,
            "finite and positive",
        ),
        ("maximum_mach", options.maximum_mach, "finite and positive"),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(DuctShroudError::InvalidOptions { field, requirement });
        }
    }
    if !options.minor_loss_coefficient.is_finite() || options.minor_loss_coefficient < 0.0 {
        return Err(DuctShroudError::InvalidOptions {
            field: "minor_loss_coefficient",
            requirement: "finite and non-negative",
        });
    }
    if options.maximum_stations == 0 {
        return Err(DuctShroudError::InvalidOptions {
            field: "maximum_stations",
            requirement: "positive",
        });
    }
    Ok(())
}

fn validate_station(index: usize, station: DuctShroudStation) -> Result<(), DuctShroudError> {
    if !station.axial_position_m.is_finite() || station.axial_position_m < 0.0 {
        return Err(DuctShroudError::InvalidStation {
            index,
            field: "axial_position_m",
            requirement: "finite and non-negative",
        });
    }
    if !station.flow_area_m2.is_finite() || station.flow_area_m2 <= 0.0 {
        return Err(DuctShroudError::InvalidStation {
            index,
            field: "flow_area_m2",
            requirement: "finite and positive",
        });
    }
    if !station.hydraulic_diameter_m.is_finite() || station.hydraulic_diameter_m <= 0.0 {
        return Err(DuctShroudError::InvalidStation {
            index,
            field: "hydraulic_diameter_m",
            requirement: "finite and positive",
        });
    }
    Ok(())
}
