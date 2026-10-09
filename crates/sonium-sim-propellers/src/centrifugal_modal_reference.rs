//! Bounded centrifugal-stiffened first-mode blade reference.
//!
//! This module uses a smooth cantilever trial mode and deterministic
//! spanwise quadrature to estimate bending modal mass, elastic stiffness,
//! centrifugal geometric stiffness, and blade-passing-frequency separation.
//! It is a transparent structural/modal screen, not a manufactured-section
//! finite-element, nonlinear, flutter, or measured structural solver.

use std::fmt;

use crate::{BladeElementModel, BladeElementOperatingPoint};

/// Inputs for the centrifugal-stiffened modal reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CentrifugalModalOptions {
    /// Material density [kg/m³].
    pub material_density_kg_m3: f64,
    /// Isotropic Young's modulus [Pa].
    pub youngs_modulus_pa: f64,
    /// Rectangular thickness as a fraction of local chord [-].
    pub thickness_fraction: f64,
    /// Highest blade-passing harmonic included in the separation scan.
    pub max_harmonic: usize,
    /// Minimum relative separation from every scanned harmonic [-].
    pub minimum_relative_separation_ratio: f64,
    /// Number of deterministic spanwise quadrature nodes.
    pub integration_points: usize,
}

impl Default for CentrifugalModalOptions {
    fn default() -> Self {
        Self {
            material_density_kg_m3: 1_600.0,
            youngs_modulus_pa: 70.0e9,
            thickness_fraction: 0.12,
            max_harmonic: 8,
            minimum_relative_separation_ratio: 0.10,
            integration_points: 128,
        }
    }
}

/// Centrifugal-stiffened first-mode and Campbell-separation evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CentrifugalModalResult {
    /// Operating point used for rotational speed and blade count.
    pub operating_point: BladeElementOperatingPoint,
    /// Unstiffened first-mode frequency [Hz].
    pub unstiffened_natural_frequency_hz: f64,
    /// Centrifugal-stiffened first-mode frequency [Hz].
    pub stiffened_natural_frequency_hz: f64,
    /// Blade-passing frequency [Hz].
    pub blade_passing_frequency_hz: f64,
    /// Root centrifugal load from the modeled blade mass [N].
    pub root_centrifugal_load_n: f64,
    /// Elastic modal stiffness before centrifugal stiffening [N/m].
    pub elastic_modal_stiffness_n_m: f64,
    /// Centrifugal geometric modal stiffness [N/m].
    pub centrifugal_modal_stiffness_n_m: f64,
    /// Ratio of geometric to elastic modal stiffness [-].
    pub centrifugal_stiffening_ratio: f64,
    /// Harmonic nearest the stiffened mode.
    pub nearest_harmonic: usize,
    /// Frequency of the nearest scanned harmonic [Hz].
    pub nearest_harmonic_frequency_hz: f64,
    /// Minimum relative mode/harmonic separation over the scan [-].
    pub minimum_relative_modal_separation: f64,
    /// Whether the configured modal-separation screen passed.
    pub modal_separation_passed: bool,
    /// Whether all finite/modal admission checks passed.
    pub feasible: bool,
}

impl CentrifugalModalResult {
    /// Returns whether the result is finite and internally admissible.
    pub fn is_finite(&self) -> bool {
        self.operating_point.rpm.is_finite()
            && self.operating_point.rpm > 0.0
            && self.operating_point.freestream_m_s.is_finite()
            && self.operating_point.freestream_m_s >= 0.0
            && self.operating_point.density_kg_m3.is_finite()
            && self.operating_point.density_kg_m3 > 0.0
            && self.unstiffened_natural_frequency_hz.is_finite()
            && self.unstiffened_natural_frequency_hz > 0.0
            && self.stiffened_natural_frequency_hz.is_finite()
            && self.stiffened_natural_frequency_hz >= self.unstiffened_natural_frequency_hz
            && self.blade_passing_frequency_hz.is_finite()
            && self.blade_passing_frequency_hz > 0.0
            && self.root_centrifugal_load_n.is_finite()
            && self.root_centrifugal_load_n >= 0.0
            && self.elastic_modal_stiffness_n_m.is_finite()
            && self.elastic_modal_stiffness_n_m > 0.0
            && self.centrifugal_modal_stiffness_n_m.is_finite()
            && self.centrifugal_modal_stiffness_n_m >= 0.0
            && self.centrifugal_stiffening_ratio.is_finite()
            && self.centrifugal_stiffening_ratio >= 0.0
            && self.nearest_harmonic > 0
            && self.nearest_harmonic_frequency_hz.is_finite()
            && self.nearest_harmonic_frequency_hz > 0.0
            && self.minimum_relative_modal_separation.is_finite()
            && self.minimum_relative_modal_separation >= 0.0
    }

    /// Whether finite modal and configured separation gates passed.
    pub fn is_feasible(&self) -> bool {
        self.is_finite() && self.modal_separation_passed && self.feasible
    }
}

/// Failure-closed errors from the centrifugal modal reference.
#[derive(Debug, Clone, PartialEq)]
pub enum CentrifugalModalError {
    /// An option is outside its declared domain.
    InvalidOptions {
        /// Option field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The operating point is invalid for a rotating modal reference.
    InvalidOperatingPoint {
        /// Operating-point field.
        field: &'static str,
    },
    /// Model geometry cannot form a positive spanwise beam.
    InvalidGeometry {
        /// Geometry field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A derived quadrature or modal quantity is non-finite.
    NonFiniteResult {
        /// Derived field.
        field: &'static str,
    },
}

impl fmt::Display for CentrifugalModalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(formatter, "centrifugal modal option {field}: {requirement}")
            }
            Self::InvalidOperatingPoint { field } => {
                write!(
                    formatter,
                    "centrifugal modal operating point {field} is invalid"
                )
            }
            Self::InvalidGeometry { field, requirement } => write!(
                formatter,
                "centrifugal modal geometry {field}: {requirement}"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "centrifugal modal result {field} is non-finite")
            }
        }
    }
}

impl std::error::Error for CentrifugalModalError {}

/// Evaluate the bounded centrifugal-stiffened first-mode reference.
pub fn evaluate_centrifugal_stiffened_modal_reference(
    model: &BladeElementModel,
    operating_point: BladeElementOperatingPoint,
    options: CentrifugalModalOptions,
) -> Result<CentrifugalModalResult, CentrifugalModalError> {
    validate_options(options)?;
    if !operating_point.rpm.is_finite() || operating_point.rpm <= 0.0 {
        return Err(CentrifugalModalError::InvalidOperatingPoint { field: "rpm" });
    }
    if !operating_point.freestream_m_s.is_finite() || operating_point.freestream_m_s < 0.0 {
        return Err(CentrifugalModalError::InvalidOperatingPoint {
            field: "freestream_m_s",
        });
    }
    if !operating_point.density_kg_m3.is_finite() || operating_point.density_kg_m3 <= 0.0 {
        return Err(CentrifugalModalError::InvalidOperatingPoint {
            field: "density_kg_m3",
        });
    }
    let span_m = model.tip_radius_m() - model.hub_radius_m();
    if !span_m.is_finite() || span_m <= 0.0 {
        return Err(CentrifugalModalError::InvalidGeometry {
            field: "span",
            requirement: "finite and positive",
        });
    }
    if model.n_blades() == 0 || model.stations().is_empty() {
        return Err(CentrifugalModalError::InvalidGeometry {
            field: "model",
            requirement: "positive blade count and non-empty stations",
        });
    }

    let node_count = options.integration_points;
    let step_m = span_m / (node_count - 1) as f64;
    let omega_rad_s = 2.0 * std::f64::consts::PI * operating_point.rpm / 60.0;
    let mut radii = Vec::with_capacity(node_count);
    let mut mass_per_length = Vec::with_capacity(node_count);
    let mut bending_stiffness = Vec::with_capacity(node_count);
    for index in 0..node_count {
        let x_m = index as f64 * step_m;
        let radius_m = model.hub_radius_m() + x_m;
        let chord_m = chord_at_radius(model, radius_m)?;
        let thickness_m = chord_m * options.thickness_fraction;
        let area_m2 = chord_m * thickness_m;
        let second_moment_m4 = chord_m * thickness_m.powi(3) / 12.0;
        let mass_per_length_kg_m = options.material_density_kg_m3 * area_m2;
        let elastic_stiffness_nm2 = options.youngs_modulus_pa * second_moment_m4;
        for (field, value) in [
            ("radius_m", radius_m),
            ("area_m2", area_m2),
            ("second_moment_m4", second_moment_m4),
            ("mass_per_length_kg_m", mass_per_length_kg_m),
            ("elastic_stiffness_nm2", elastic_stiffness_nm2),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(CentrifugalModalError::NonFiniteResult { field });
            }
        }
        radii.push(radius_m);
        mass_per_length.push(mass_per_length_kg_m);
        bending_stiffness.push(elastic_stiffness_nm2);
    }

    let mut tension_n = vec![0.0; node_count];
    for index in (0..node_count - 1).rev() {
        let load_left = mass_per_length[index] * radii[index] * omega_rad_s.powi(2);
        let load_right = mass_per_length[index + 1] * radii[index + 1] * omega_rad_s.powi(2);
        tension_n[index] = tension_n[index + 1] + 0.5 * (load_left + load_right) * step_m;
    }

    let mut modal_mass_kg = 0.0;
    let mut elastic_modal_stiffness_n_m = 0.0;
    let mut centrifugal_modal_stiffness_n_m = 0.0;
    for index in 0..node_count - 1 {
        let s_left = index as f64 / (node_count - 1) as f64;
        let s_right = (index + 1) as f64 / (node_count - 1) as f64;
        let phi_left = trial_mode(s_left);
        let phi_right = trial_mode(s_right);
        let phi_prime_left = trial_mode_prime(s_left, span_m);
        let phi_prime_right = trial_mode_prime(s_right, span_m);
        let phi_second_left = trial_mode_second(s_left, span_m);
        let phi_second_right = trial_mode_second(s_right, span_m);
        modal_mass_kg += 0.5
            * (mass_per_length[index] * phi_left.powi(2)
                + mass_per_length[index + 1] * phi_right.powi(2))
            * step_m;
        elastic_modal_stiffness_n_m += 0.5
            * (bending_stiffness[index] * phi_second_left.powi(2)
                + bending_stiffness[index + 1] * phi_second_right.powi(2))
            * step_m;
        centrifugal_modal_stiffness_n_m += 0.5
            * (tension_n[index] * phi_prime_left.powi(2)
                + tension_n[index + 1] * phi_prime_right.powi(2))
            * step_m;
    }
    if !modal_mass_kg.is_finite()
        || modal_mass_kg <= 0.0
        || !elastic_modal_stiffness_n_m.is_finite()
        || elastic_modal_stiffness_n_m <= 0.0
        || !centrifugal_modal_stiffness_n_m.is_finite()
        || centrifugal_modal_stiffness_n_m < 0.0
    {
        return Err(CentrifugalModalError::NonFiniteResult {
            field: "modal_integrals",
        });
    }

    let unstiffened_natural_frequency_hz =
        (elastic_modal_stiffness_n_m / modal_mass_kg).sqrt() / (2.0 * std::f64::consts::PI);
    let stiffened_natural_frequency_hz =
        ((elastic_modal_stiffness_n_m + centrifugal_modal_stiffness_n_m) / modal_mass_kg).sqrt()
            / (2.0 * std::f64::consts::PI);
    let blade_passing_frequency_hz = operating_point.rpm / 60.0 * model.n_blades() as f64;
    let mut nearest_harmonic = 0;
    let mut nearest_relative_separation = f64::INFINITY;
    let mut nearest_harmonic_frequency_hz = f64::NAN;
    let mut minimum_relative_modal_separation = f64::INFINITY;
    for harmonic in 1..=options.max_harmonic {
        let harmonic_frequency_hz = blade_passing_frequency_hz * harmonic as f64;
        let relative_separation = (stiffened_natural_frequency_hz - harmonic_frequency_hz).abs()
            / stiffened_natural_frequency_hz;
        minimum_relative_modal_separation =
            minimum_relative_modal_separation.min(relative_separation);
        if relative_separation < nearest_relative_separation {
            nearest_relative_separation = relative_separation;
            nearest_harmonic = harmonic;
            nearest_harmonic_frequency_hz = harmonic_frequency_hz;
        }
    }
    let root_centrifugal_load_n = tension_n[0];
    let centrifugal_stiffening_ratio =
        centrifugal_modal_stiffness_n_m / elastic_modal_stiffness_n_m;
    let result = CentrifugalModalResult {
        operating_point,
        unstiffened_natural_frequency_hz,
        stiffened_natural_frequency_hz,
        blade_passing_frequency_hz,
        root_centrifugal_load_n,
        elastic_modal_stiffness_n_m,
        centrifugal_modal_stiffness_n_m,
        centrifugal_stiffening_ratio,
        nearest_harmonic,
        nearest_harmonic_frequency_hz,
        minimum_relative_modal_separation,
        modal_separation_passed: minimum_relative_modal_separation
            >= options.minimum_relative_separation_ratio,
        feasible: minimum_relative_modal_separation >= options.minimum_relative_separation_ratio,
    };
    if !result.is_finite() {
        return Err(CentrifugalModalError::NonFiniteResult {
            field: "modal_result",
        });
    }
    Ok(result)
}

fn validate_options(options: CentrifugalModalOptions) -> Result<(), CentrifugalModalError> {
    for (field, value) in [
        ("material_density_kg_m3", options.material_density_kg_m3),
        ("youngs_modulus_pa", options.youngs_modulus_pa),
        ("thickness_fraction", options.thickness_fraction),
        (
            "minimum_relative_separation_ratio",
            options.minimum_relative_separation_ratio,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(CentrifugalModalError::InvalidOptions {
                field,
                requirement: "finite and positive",
            });
        }
    }
    if options.thickness_fraction >= 1.0 {
        return Err(CentrifugalModalError::InvalidOptions {
            field: "thickness_fraction",
            requirement: "< 1",
        });
    }
    if options.max_harmonic == 0 {
        return Err(CentrifugalModalError::InvalidOptions {
            field: "max_harmonic",
            requirement: "non-zero",
        });
    }
    if options.integration_points < 4 {
        return Err(CentrifugalModalError::InvalidOptions {
            field: "integration_points",
            requirement: "at least 4",
        });
    }
    Ok(())
}

fn chord_at_radius(model: &BladeElementModel, radius_m: f64) -> Result<f64, CentrifugalModalError> {
    let stations = model.stations();
    let chord_m = if radius_m <= stations[0].radius_m {
        stations[0].chord_m
    } else if radius_m >= stations[stations.len() - 1].radius_m {
        stations[stations.len() - 1].chord_m
    } else {
        let window = stations
            .windows(2)
            .find(|window| radius_m >= window[0].radius_m && radius_m <= window[1].radius_m)
            .ok_or(CentrifugalModalError::InvalidGeometry {
                field: "station_radius",
                requirement: "ordered and spanning the blade",
            })?;
        let fraction = (radius_m - window[0].radius_m) / (window[1].radius_m - window[0].radius_m);
        window[0].chord_m + fraction * (window[1].chord_m - window[0].chord_m)
    };
    if !chord_m.is_finite() || chord_m <= 0.0 {
        return Err(CentrifugalModalError::InvalidGeometry {
            field: "station_chord_m",
            requirement: "finite and positive",
        });
    }
    Ok(chord_m)
}

fn trial_mode(s: f64) -> f64 {
    s * s * (3.0 - 2.0 * s)
}

fn trial_mode_prime(s: f64, span_m: f64) -> f64 {
    6.0 * s * (1.0 - s) / span_m
}

fn trial_mode_second(s: f64, span_m: f64) -> f64 {
    6.0 * (1.0 - 2.0 * s) / span_m.powi(2)
}
