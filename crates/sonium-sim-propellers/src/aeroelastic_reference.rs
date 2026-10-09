//! Bounded small-deflection aeroelastic, modal-separation, and fatigue screen.
//!
//! This reference turns checked steady BEM station loads into a transparent
//! cantilever-beam estimate. It uses a rectangular section, an effective
//! first flapwise Euler--Bernoulli mode, a small-deflection static response,
//! and a caller-declared cyclic-load fraction for a Basquin fatigue screen.
//! It is not a nonlinear aeroelastic, centrifugal-stiffened, transient, or
//! manufactured-section solver.

use std::fmt;

use crate::{BladeElementBemResult, BladeElementModel, BladeElementOperatingPoint};

const FIRST_CANTILEVER_ROOT: f64 = 1.875_104_068_711_961;

/// Options for the bounded small-deflection blade dynamics reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeAeroelasticOptions {
    /// Material density [kg/m³].
    pub material_density_kg_m3: f64,
    /// Isotropic Young's modulus used for flapwise bending [Pa].
    pub youngs_modulus_pa: f64,
    /// Rectangular section thickness as a fraction of local chord [-].
    pub thickness_fraction: f64,
    /// Stress at the reference fatigue life [Pa].
    pub fatigue_reference_stress_pa: f64,
    /// Reference cycles in the Basquin fatigue law [-].
    pub fatigue_reference_cycles: f64,
    /// Positive Basquin exponent [-].
    pub fatigue_exponent: f64,
    /// Declared cyclic fraction of the steady aerodynamic bending stress [-].
    pub aerodynamic_load_amplitude_fraction: f64,
    /// Exposure duration used to accumulate blade-passing cycles [h].
    pub operating_hours: f64,
    /// Minimum relative separation from every checked BPF harmonic [-].
    pub minimum_modal_separation_ratio: f64,
    /// Highest blade-passing harmonic checked for modal separation.
    pub max_harmonic: usize,
    /// Maximum allowed small-deflection estimate [m].
    pub max_tip_deflection_m: f64,
}

impl Default for BladeAeroelasticOptions {
    fn default() -> Self {
        Self {
            material_density_kg_m3: 1_600.0,
            youngs_modulus_pa: 70.0e9,
            thickness_fraction: 0.12,
            fatigue_reference_stress_pa: 150.0e6,
            fatigue_reference_cycles: 1.0e6,
            fatigue_exponent: 5.0,
            aerodynamic_load_amplitude_fraction: 0.1,
            operating_hours: 1.0,
            minimum_modal_separation_ratio: 0.1,
            max_harmonic: 3,
            max_tip_deflection_m: 5.0e-3,
        }
    }
}

/// Per-station result from the bounded blade dynamics screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeAeroelasticStation {
    /// Radial station location [m].
    pub radius_m: f64,
    /// Rectangular section area [m²].
    pub section_area_m2: f64,
    /// Flapwise second moment of area [m⁴].
    pub second_moment_m4: f64,
    /// Flapwise bending stiffness `E I` [N m²].
    pub bending_stiffness_nm2: f64,
    /// Per-blade aerodynamic point-load approximation [N].
    pub aerodynamic_force_n: f64,
    /// Estimated static flapwise bending stress [Pa].
    pub bending_stress_pa: f64,
    /// Estimated static flapwise deflection at the station [m].
    pub static_deflection_m: f64,
    /// Declared cyclic stress amplitude [Pa].
    pub alternating_stress_pa: f64,
    /// Estimated Basquin fatigue life [cycles].
    pub fatigue_life_cycles: f64,
    /// Miner-style damage for the configured exposure [-].
    pub fatigue_damage: f64,
}

/// Result from the bounded small-deflection aeroelastic reference.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeAeroelasticResult {
    /// Operating point inherited from BEM.
    pub operating_point: BladeElementOperatingPoint,
    /// Effective first flapwise natural frequency [Hz].
    pub first_flapwise_natural_frequency_hz: f64,
    /// Blade-passing frequency [Hz].
    pub blade_passing_frequency_hz: f64,
    /// Harmonic nearest the first flapwise mode.
    pub nearest_harmonic: usize,
    /// Frequency of the nearest checked BPF harmonic [Hz].
    pub nearest_harmonic_frequency_hz: f64,
    /// Smallest relative frequency separation over checked harmonics [-].
    pub minimum_relative_modal_separation: f64,
    /// Whether the modal separation threshold passed.
    pub modal_separation_passed: bool,
    /// Maximum estimated static deflection [m].
    pub max_static_deflection_m: f64,
    /// Whether the deflection threshold passed.
    pub deflection_passed: bool,
    /// Maximum estimated alternating stress [Pa].
    pub max_alternating_stress_pa: f64,
    /// Applied blade-passing cycles over the configured exposure [-].
    pub applied_cycles: f64,
    /// Minimum estimated fatigue life over stations [cycles].
    pub minimum_fatigue_life_cycles: f64,
    /// Maximum station fatigue damage [-].
    pub maximum_fatigue_damage: f64,
    /// Whether the fatigue screen passed.
    pub fatigue_passed: bool,
    /// Combined modal, deflection, and fatigue admission.
    pub feasible: bool,
    /// Station results in model order.
    pub stations: Vec<BladeAeroelasticStation>,
}

impl BladeAeroelasticResult {
    /// Whether this result is finite and internally consistent.
    pub fn is_finite(&self) -> bool {
        self.operating_point.rpm.is_finite()
            && self.operating_point.rpm > 0.0
            && self.operating_point.freestream_m_s.is_finite()
            && self.operating_point.freestream_m_s > 0.0
            && self.operating_point.density_kg_m3.is_finite()
            && self.operating_point.density_kg_m3 > 0.0
            && self.first_flapwise_natural_frequency_hz.is_finite()
            && self.first_flapwise_natural_frequency_hz > 0.0
            && self.blade_passing_frequency_hz.is_finite()
            && self.blade_passing_frequency_hz > 0.0
            && self.nearest_harmonic > 0
            && self.nearest_harmonic_frequency_hz.is_finite()
            && self.nearest_harmonic_frequency_hz > 0.0
            && self.minimum_relative_modal_separation.is_finite()
            && self.minimum_relative_modal_separation >= 0.0
            && self.max_static_deflection_m.is_finite()
            && self.max_static_deflection_m >= 0.0
            && self.max_alternating_stress_pa.is_finite()
            && self.max_alternating_stress_pa >= 0.0
            && self.applied_cycles.is_finite()
            && self.applied_cycles >= 0.0
            && self.minimum_fatigue_life_cycles.is_finite()
            && self.minimum_fatigue_life_cycles > 0.0
            && self.maximum_fatigue_damage.is_finite()
            && self.maximum_fatigue_damage >= 0.0
            && self.stations.iter().all(|station| {
                station.radius_m.is_finite()
                    && station.radius_m > 0.0
                    && station.section_area_m2.is_finite()
                    && station.section_area_m2 > 0.0
                    && station.second_moment_m4.is_finite()
                    && station.second_moment_m4 > 0.0
                    && station.bending_stiffness_nm2.is_finite()
                    && station.bending_stiffness_nm2 > 0.0
                    && station.aerodynamic_force_n.is_finite()
                    && station.aerodynamic_force_n >= 0.0
                    && station.bending_stress_pa.is_finite()
                    && station.bending_stress_pa >= 0.0
                    && station.static_deflection_m.is_finite()
                    && station.static_deflection_m >= 0.0
                    && station.alternating_stress_pa.is_finite()
                    && station.alternating_stress_pa >= 0.0
                    && station.fatigue_life_cycles.is_finite()
                    && station.fatigue_life_cycles > 0.0
                    && station.fatigue_damage.is_finite()
                    && station.fatigue_damage >= 0.0
            })
    }

    /// Whether all configured admission gates passed.
    pub fn is_feasible(&self) -> bool {
        self.is_finite()
            && self.modal_separation_passed
            && self.deflection_passed
            && self.fatigue_passed
            && self.feasible
    }
}

/// Failure-closed error from the bounded aeroelastic reference.
#[derive(Debug, Clone, PartialEq)]
pub enum BladeAeroelasticError {
    /// A reference option is invalid.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The supplied BEM result is not converged or finite.
    InvalidBemResult {
        /// Invalid BEM field.
        field: &'static str,
    },
    /// Model and BEM station arrays differ.
    StationCountMismatch {
        /// Number of model stations.
        model_count: usize,
        /// Number of BEM stations.
        result_count: usize,
    },
    /// A derived quantity became non-finite or left its physical domain.
    NonFiniteResult {
        /// Derived field name.
        field: &'static str,
    },
}

impl fmt::Display for BladeAeroelasticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(f, "blade aeroelastic option {field} must be {requirement}")
            }
            Self::InvalidBemResult { field } => {
                write!(f, "blade aeroelastic BEM result {field} is invalid")
            }
            Self::StationCountMismatch {
                model_count,
                result_count,
            } => write!(
                f,
                "blade aeroelastic station counts differ: model={model_count}, BEM={result_count}"
            ),
            Self::NonFiniteResult { field } => {
                write!(
                    f,
                    "blade aeroelastic result {field} is non-finite or invalid"
                )
            }
        }
    }
}

impl std::error::Error for BladeAeroelasticError {}

/// Evaluate a bounded small-deflection modal and fatigue reference from BEM.
pub fn evaluate_blade_aeroelastic_reference(
    model: &BladeElementModel,
    bem: &BladeElementBemResult,
    options: BladeAeroelasticOptions,
) -> Result<BladeAeroelasticResult, BladeAeroelasticError> {
    validate_options(options)?;
    if !bem.converged || bem.max_iterations_used == 0 {
        return Err(BladeAeroelasticError::InvalidBemResult { field: "converged" });
    }
    let model_stations = model.stations();
    if model_stations.len() != bem.loads.strips.len() || model_stations.len() != bem.strips.len() {
        return Err(BladeAeroelasticError::StationCountMismatch {
            model_count: model_stations.len(),
            result_count: bem.loads.strips.len(),
        });
    }
    let operating_point = bem.loads.operating_point;
    if !operating_point.rpm.is_finite()
        || operating_point.rpm <= 0.0
        || !operating_point.freestream_m_s.is_finite()
        || operating_point.freestream_m_s <= 0.0
        || !operating_point.density_kg_m3.is_finite()
        || operating_point.density_kg_m3 <= 0.0
    {
        return Err(BladeAeroelasticError::InvalidBemResult {
            field: "operating_point",
        });
    }

    let span_m = model.tip_radius_m() - model.hub_radius_m();
    let widths = station_widths(model);
    let blade_count = model.n_blades() as f64;
    let mut sections = Vec::with_capacity(model_stations.len());
    let mut total_mass_kg = 0.0;
    let mut total_bending_stiffness_integral = 0.0;
    for ((station, width_m), load) in model_stations
        .iter()
        .zip(widths.iter().copied())
        .zip(&bem.loads.strips)
    {
        if !load.axial_force_per_m_n.is_finite() {
            return Err(BladeAeroelasticError::InvalidBemResult {
                field: "axial_force_per_m_n",
            });
        }
        let thickness_m = station.chord_m * options.thickness_fraction;
        let section_area_m2 = station.chord_m * thickness_m;
        let second_moment_m4 = station.chord_m * thickness_m.powi(3) / 12.0;
        let bending_stiffness_nm2 = options.youngs_modulus_pa * second_moment_m4;
        let mass_per_m_kg_m = options.material_density_kg_m3 * section_area_m2;
        let aerodynamic_force_n = load.axial_force_per_m_n.abs() * width_m / blade_count;
        for (field, value) in [
            ("section_area_m2", section_area_m2),
            ("second_moment_m4", second_moment_m4),
            ("bending_stiffness_nm2", bending_stiffness_nm2),
            ("mass_per_m_kg_m", mass_per_m_kg_m),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(BladeAeroelasticError::NonFiniteResult { field });
            }
        }
        if !aerodynamic_force_n.is_finite() || aerodynamic_force_n < 0.0 {
            return Err(BladeAeroelasticError::NonFiniteResult {
                field: "aerodynamic_force_n",
            });
        }
        total_mass_kg += mass_per_m_kg_m * width_m;
        total_bending_stiffness_integral += bending_stiffness_nm2 * width_m;
        sections.push(SectionData {
            radius_m: station.radius_m,
            section_area_m2,
            thickness_m,
            second_moment_m4,
            bending_stiffness_nm2,
            aerodynamic_force_n,
        });
    }
    if !total_mass_kg.is_finite() || total_mass_kg <= 0.0 {
        return Err(BladeAeroelasticError::NonFiniteResult {
            field: "total_mass_kg",
        });
    }
    if !total_bending_stiffness_integral.is_finite() || total_bending_stiffness_integral <= 0.0 {
        return Err(BladeAeroelasticError::NonFiniteResult {
            field: "total_bending_stiffness_integral",
        });
    }

    let average_mass_per_m_kg_m = total_mass_kg / span_m;
    let average_bending_stiffness_nm2 = total_bending_stiffness_integral / span_m;
    let first_flapwise_natural_frequency_hz = FIRST_CANTILEVER_ROOT.powi(2)
        / (2.0 * std::f64::consts::PI * span_m.powi(2))
        * (average_bending_stiffness_nm2 / average_mass_per_m_kg_m).sqrt();
    let blade_passing_frequency_hz = model.n_blades() as f64 * operating_point.rpm / 60.0;
    if !first_flapwise_natural_frequency_hz.is_finite()
        || first_flapwise_natural_frequency_hz <= 0.0
        || !blade_passing_frequency_hz.is_finite()
        || blade_passing_frequency_hz <= 0.0
    {
        return Err(BladeAeroelasticError::NonFiniteResult {
            field: "modal_frequency",
        });
    }

    let mut nearest_harmonic = 0;
    let mut nearest_harmonic_frequency_hz = 0.0;
    let mut minimum_relative_modal_separation = f64::INFINITY;
    for harmonic in 1..=options.max_harmonic {
        let harmonic_frequency_hz = blade_passing_frequency_hz * harmonic as f64;
        let relative_separation = (first_flapwise_natural_frequency_hz - harmonic_frequency_hz)
            .abs()
            / harmonic_frequency_hz;
        if relative_separation < minimum_relative_modal_separation {
            minimum_relative_modal_separation = relative_separation;
            nearest_harmonic = harmonic;
            nearest_harmonic_frequency_hz = harmonic_frequency_hz;
        }
    }

    let effective_ei = average_bending_stiffness_nm2;
    let mut station_results = Vec::with_capacity(sections.len());
    let mut max_static_deflection_m: f64 = 0.0;
    let mut max_alternating_stress_pa: f64 = 0.0;
    let mut minimum_fatigue_life_cycles = f64::MAX;
    let applied_cycles = blade_passing_frequency_hz * 3_600.0 * options.operating_hours;
    let mut maximum_fatigue_damage: f64 = 0.0;
    for section in &sections {
        let observation_x_m = section.radius_m - model.hub_radius_m();
        let mut bending_moment_nm = 0.0;
        let mut static_deflection_m = 0.0;
        for load_section in &sections {
            let load_x_m = load_section.radius_m - model.hub_radius_m();
            if load_x_m > observation_x_m {
                bending_moment_nm +=
                    load_section.aerodynamic_force_n * (load_x_m - observation_x_m);
            }
            let contribution = if observation_x_m <= load_x_m {
                load_section.aerodynamic_force_n
                    * observation_x_m.powi(2)
                    * (3.0 * load_x_m - observation_x_m)
                    / (6.0 * effective_ei)
            } else {
                load_section.aerodynamic_force_n
                    * load_x_m.powi(2)
                    * (3.0 * observation_x_m - load_x_m)
                    / (6.0 * effective_ei)
            };
            static_deflection_m += contribution;
        }
        let bending_stress_pa =
            bending_moment_nm * (section.thickness_m / 2.0) / section.second_moment_m4;
        let alternating_stress_pa = bending_stress_pa * options.aerodynamic_load_amplitude_fraction;
        let fatigue_life_cycles = if alternating_stress_pa <= f64::EPSILON {
            f64::MAX
        } else {
            options.fatigue_reference_cycles
                * (options.fatigue_reference_stress_pa / alternating_stress_pa)
                    .powf(options.fatigue_exponent)
        };
        if !bending_stress_pa.is_finite()
            || bending_stress_pa < 0.0
            || !static_deflection_m.is_finite()
            || static_deflection_m < 0.0
            || !alternating_stress_pa.is_finite()
            || alternating_stress_pa < 0.0
            || !fatigue_life_cycles.is_finite()
            || fatigue_life_cycles <= 0.0
        {
            return Err(BladeAeroelasticError::NonFiniteResult {
                field: "station_response",
            });
        }
        let fatigue_damage = if alternating_stress_pa <= f64::EPSILON {
            0.0
        } else {
            applied_cycles / fatigue_life_cycles
        };
        if !fatigue_damage.is_finite() || fatigue_damage < 0.0 {
            return Err(BladeAeroelasticError::NonFiniteResult {
                field: "fatigue_damage",
            });
        }
        max_static_deflection_m = max_static_deflection_m.max(static_deflection_m);
        max_alternating_stress_pa = max_alternating_stress_pa.max(alternating_stress_pa);
        minimum_fatigue_life_cycles = minimum_fatigue_life_cycles.min(fatigue_life_cycles);
        maximum_fatigue_damage = maximum_fatigue_damage.max(fatigue_damage);
        station_results.push(BladeAeroelasticStation {
            radius_m: section.radius_m,
            section_area_m2: section.section_area_m2,
            second_moment_m4: section.second_moment_m4,
            bending_stiffness_nm2: section.bending_stiffness_nm2,
            aerodynamic_force_n: section.aerodynamic_force_n,
            bending_stress_pa,
            static_deflection_m,
            alternating_stress_pa,
            fatigue_life_cycles,
            fatigue_damage,
        });
    }

    let modal_separation_passed =
        minimum_relative_modal_separation >= options.minimum_modal_separation_ratio;
    let deflection_passed = max_static_deflection_m <= options.max_tip_deflection_m;
    let fatigue_passed = maximum_fatigue_damage <= 1.0;
    let result = BladeAeroelasticResult {
        operating_point,
        first_flapwise_natural_frequency_hz,
        blade_passing_frequency_hz,
        nearest_harmonic,
        nearest_harmonic_frequency_hz,
        minimum_relative_modal_separation,
        modal_separation_passed,
        max_static_deflection_m,
        deflection_passed,
        max_alternating_stress_pa,
        applied_cycles,
        minimum_fatigue_life_cycles,
        maximum_fatigue_damage,
        fatigue_passed,
        feasible: modal_separation_passed && deflection_passed && fatigue_passed,
        stations: station_results,
    };
    if !result.is_finite() {
        return Err(BladeAeroelasticError::NonFiniteResult { field: "result" });
    }
    Ok(result)
}

#[derive(Debug, Clone, Copy)]
struct SectionData {
    radius_m: f64,
    section_area_m2: f64,
    thickness_m: f64,
    second_moment_m4: f64,
    bending_stiffness_nm2: f64,
    aerodynamic_force_n: f64,
}

fn station_widths(model: &BladeElementModel) -> Vec<f64> {
    let stations = model.stations();
    stations
        .iter()
        .enumerate()
        .map(|(index, station)| {
            let lower = if index == 0 {
                model.hub_radius_m()
            } else {
                0.5 * (stations[index - 1].radius_m + station.radius_m)
            };
            let upper = if index + 1 == stations.len() {
                model.tip_radius_m()
            } else {
                0.5 * (station.radius_m + stations[index + 1].radius_m)
            };
            upper - lower
        })
        .collect()
}

fn validate_options(options: BladeAeroelasticOptions) -> Result<(), BladeAeroelasticError> {
    for (field, value) in [
        ("material_density_kg_m3", options.material_density_kg_m3),
        ("youngs_modulus_pa", options.youngs_modulus_pa),
        (
            "fatigue_reference_stress_pa",
            options.fatigue_reference_stress_pa,
        ),
        ("fatigue_reference_cycles", options.fatigue_reference_cycles),
        ("fatigue_exponent", options.fatigue_exponent),
        (
            "minimum_modal_separation_ratio",
            options.minimum_modal_separation_ratio,
        ),
        ("max_tip_deflection_m", options.max_tip_deflection_m),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(BladeAeroelasticError::InvalidOptions {
                field,
                requirement: "finite positive",
            });
        }
    }
    if !options.thickness_fraction.is_finite()
        || options.thickness_fraction <= 0.0
        || options.thickness_fraction > 1.0
    {
        return Err(BladeAeroelasticError::InvalidOptions {
            field: "thickness_fraction",
            requirement: "finite in (0, 1]",
        });
    }
    if !options.aerodynamic_load_amplitude_fraction.is_finite()
        || !(0.0..=1.0).contains(&options.aerodynamic_load_amplitude_fraction)
    {
        return Err(BladeAeroelasticError::InvalidOptions {
            field: "aerodynamic_load_amplitude_fraction",
            requirement: "finite in [0, 1]",
        });
    }
    if !options.operating_hours.is_finite() || options.operating_hours < 0.0 {
        return Err(BladeAeroelasticError::InvalidOptions {
            field: "operating_hours",
            requirement: "finite and non-negative",
        });
    }
    if options.max_harmonic == 0 {
        return Err(BladeAeroelasticError::InvalidOptions {
            field: "max_harmonic",
            requirement: "non-zero",
        });
    }
    Ok(())
}
