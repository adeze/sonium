//! Bounded compact stationary-surface FW-H-style observer reference.
//!
//! This module integrates prescribed harmonic surface pressure and normal
//! velocity phasors into free-field monopole and dipole pressure terms. The
//! convention is peak phasors with `exp(i omega t)` and outgoing propagation
//! factor `exp(-i k r)`. It is a source-observer contract only: no CFD,
//! moving-surface kinematics, mean-flow correction, retarded-time history, or
//! full Ffowcs Williams--Hawkings solver is provided.

use std::fmt;

use crate::Point3;

/// One prescribed stationary surface element at one harmonic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwhSurfaceElement {
    /// Element centroid [m].
    pub position: Point3,
    /// Outward surface normal; normalization is checked and applied internally.
    pub normal: Point3,
    /// Element area [m²].
    pub area_m2: f64,
    /// Surface pressure peak phasor real component [Pa].
    pub pressure_real_pa: f64,
    /// Surface pressure peak phasor imaginary component [Pa].
    pub pressure_imaginary_pa: f64,
    /// Surface-normal velocity peak phasor real component [m/s].
    pub normal_velocity_real_m_s: f64,
    /// Surface-normal velocity peak phasor imaginary component [m/s].
    pub normal_velocity_imaginary_m_s: f64,
}

/// Controls for the bounded compact stationary-surface observer reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwhOptions {
    /// Harmonic frequency [Hz].
    pub frequency_hz: f64,
    /// Quiescent medium density [kg/m³].
    pub density_kg_m3: f64,
    /// Quiescent medium sound speed [m/s].
    pub sound_speed_m_s: f64,
    /// Observer position [m].
    pub observer_position: Point3,
    /// Minimum admitted element-observer distance [m].
    pub minimum_observer_distance_m: f64,
    /// Maximum admitted surface element count [-].
    pub maximum_elements: usize,
}

impl Default for FwhOptions {
    fn default() -> Self {
        Self {
            frequency_hz: 1_000.0,
            density_kg_m3: 1.2,
            sound_speed_m_s: 343.0,
            observer_position: Point3 {
                x: 0.0,
                y: 0.0,
                z: 10.0,
            },
            minimum_observer_distance_m: 1.0,
            maximum_elements: 16_384,
        }
    }
}

/// Complex peak-pressure phasor returned by the FW-H-style reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwhComplexPressure {
    /// Real component [Pa].
    pub real_pa: f64,
    /// Imaginary component [Pa].
    pub imaginary_pa: f64,
}

impl FwhComplexPressure {
    /// Peak-pressure magnitude [Pa].
    pub fn magnitude_pa(self) -> f64 {
        self.real_pa.hypot(self.imaginary_pa)
    }
}

/// Result of the compact stationary-surface observer evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwhResult {
    /// Integrated monopole/thickness pressure phasor [Pa].
    pub monopole_pressure_pa: FwhComplexPressure,
    /// Integrated dipole/loading pressure phasor [Pa].
    pub dipole_pressure_pa: FwhComplexPressure,
    /// Coherent total pressure phasor [Pa].
    pub total_pressure_pa: FwhComplexPressure,
    /// Total supplied surface area [m²].
    pub total_surface_area_m2: f64,
    /// Nearest element-observer distance [m].
    pub minimum_observer_distance_m: f64,
    /// Farthest element-observer distance [m].
    pub maximum_observer_distance_m: f64,
}

impl FwhResult {
    /// Whether all phasors and geometry diagnostics are finite and bounded.
    pub fn is_finite(&self) -> bool {
        pressure_is_finite(self.monopole_pressure_pa)
            && pressure_is_finite(self.dipole_pressure_pa)
            && pressure_is_finite(self.total_pressure_pa)
            && self.total_surface_area_m2.is_finite()
            && self.total_surface_area_m2 > 0.0
            && self.minimum_observer_distance_m.is_finite()
            && self.minimum_observer_distance_m > 0.0
            && self.maximum_observer_distance_m.is_finite()
            && self.maximum_observer_distance_m >= self.minimum_observer_distance_m
    }
}

/// Failure-closed errors from the compact FW-H-style reference.
#[derive(Debug, Clone, PartialEq)]
pub enum FwhError {
    /// An option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No surface elements were supplied.
    EmptySurface,
    /// The element count exceeds the declared work budget.
    TooManyElements { count: usize, maximum: usize },
    /// A surface element field is malformed.
    InvalidElement {
        /// Zero-based element index.
        index: usize,
        /// Element field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// An observer is too close for the far-field reference.
    ObserverTooClose {
        /// Zero-based element index.
        index: usize,
        /// Actual distance [m].
        distance_m: f64,
        /// Minimum admitted distance [m].
        minimum_m: f64,
    },
    /// A derived result became non-finite.
    NonFiniteResult { field: &'static str },
}

impl fmt::Display for FwhError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(formatter, "invalid FW-H option {field}: {requirement}")
            }
            Self::EmptySurface => write!(formatter, "FW-H surface must not be empty"),
            Self::TooManyElements { count, maximum } => {
                write!(formatter, "FW-H element count {count} exceeds {maximum}")
            }
            Self::InvalidElement {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid FW-H element {index} field {field}: {requirement}"
            ),
            Self::ObserverTooClose {
                index,
                distance_m,
                minimum_m,
            } => write!(
                formatter,
                "FW-H element {index} observer distance {distance_m} m is below {minimum_m} m"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite FW-H result field {field}")
            }
        }
    }
}

impl std::error::Error for FwhError {}

/// Integrate prescribed harmonic surface sources at one free-field observer.
pub fn evaluate_fwh_reference(
    elements: &[FwhSurfaceElement],
    options: FwhOptions,
) -> Result<FwhResult, FwhError> {
    validate_options(options)?;
    if elements.is_empty() {
        return Err(FwhError::EmptySurface);
    }
    if elements.len() > options.maximum_elements {
        return Err(FwhError::TooManyElements {
            count: elements.len(),
            maximum: options.maximum_elements,
        });
    }

    let omega_rad_s = 2.0 * std::f64::consts::PI * options.frequency_hz;
    let wavenumber_rad_m = omega_rad_s / options.sound_speed_m_s;
    let mut monopole = FwhComplexPressure {
        real_pa: 0.0,
        imaginary_pa: 0.0,
    };
    let mut dipole = monopole;
    let mut total_surface_area_m2 = 0.0;
    let mut minimum_observer_distance_m = f64::INFINITY;
    let mut maximum_observer_distance_m: f64 = 0.0;

    for (index, &element) in elements.iter().enumerate() {
        validate_element(index, element)?;
        let path = subtract(options.observer_position, element.position);
        let distance_m = norm(path);
        if !distance_m.is_finite() {
            return Err(FwhError::NonFiniteResult {
                field: "distance_m",
            });
        }
        if distance_m < options.minimum_observer_distance_m {
            return Err(FwhError::ObserverTooClose {
                index,
                distance_m,
                minimum_m: options.minimum_observer_distance_m,
            });
        }
        let normal_magnitude = norm(element.normal);
        let direction = scale(path, 1.0 / distance_m);
        let normal = scale(element.normal, 1.0 / normal_magnitude);
        let directional_cosine = dot(normal, direction);
        let phase = FwhComplexPressure {
            real_pa: (-wavenumber_rad_m * distance_m).cos(),
            imaginary_pa: (-wavenumber_rad_m * distance_m).sin(),
        };
        let pressure = FwhComplexPressure {
            real_pa: element.pressure_real_pa,
            imaginary_pa: element.pressure_imaginary_pa,
        };
        let normal_velocity = FwhComplexPressure {
            real_pa: element.normal_velocity_real_m_s,
            imaginary_pa: element.normal_velocity_imaginary_m_s,
        };
        let monopole_factor = omega_rad_s * options.density_kg_m3 * element.area_m2
            / (4.0 * std::f64::consts::PI * distance_m);
        let dipole_factor = omega_rad_s * element.area_m2 * directional_cosine
            / (4.0 * std::f64::consts::PI * options.sound_speed_m_s * distance_m);
        monopole = add(
            monopole,
            multiply(multiply_by_i(normal_velocity, monopole_factor), phase),
        );
        dipole = add(
            dipole,
            multiply(multiply_by_i(pressure, dipole_factor), phase),
        );
        total_surface_area_m2 += element.area_m2;
        minimum_observer_distance_m = minimum_observer_distance_m.min(distance_m);
        maximum_observer_distance_m = maximum_observer_distance_m.max(distance_m);
    }

    let result = FwhResult {
        monopole_pressure_pa: monopole,
        dipole_pressure_pa: dipole,
        total_pressure_pa: add(monopole, dipole),
        total_surface_area_m2,
        minimum_observer_distance_m,
        maximum_observer_distance_m,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(FwhError::NonFiniteResult {
            field: "fwh_result",
        })
    }
}

fn validate_options(options: FwhOptions) -> Result<(), FwhError> {
    for (field, value, requirement) in [
        ("frequency_hz", options.frequency_hz, "finite and positive"),
        (
            "density_kg_m3",
            options.density_kg_m3,
            "finite and positive",
        ),
        (
            "sound_speed_m_s",
            options.sound_speed_m_s,
            "finite and positive",
        ),
        (
            "minimum_observer_distance_m",
            options.minimum_observer_distance_m,
            "finite and positive",
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(FwhError::InvalidOptions { field, requirement });
        }
    }
    if !point_is_finite(options.observer_position) {
        return Err(FwhError::InvalidOptions {
            field: "observer_position",
            requirement: "finite",
        });
    }
    if options.maximum_elements == 0 {
        return Err(FwhError::InvalidOptions {
            field: "maximum_elements",
            requirement: "positive",
        });
    }
    Ok(())
}

fn validate_element(index: usize, element: FwhSurfaceElement) -> Result<(), FwhError> {
    if !point_is_finite(element.position) {
        return Err(FwhError::InvalidElement {
            index,
            field: "position",
            requirement: "finite",
        });
    }
    if !point_is_finite(element.normal) || norm(element.normal) <= 0.0 {
        return Err(FwhError::InvalidElement {
            index,
            field: "normal",
            requirement: "finite and non-zero",
        });
    }
    if !element.area_m2.is_finite() || element.area_m2 <= 0.0 {
        return Err(FwhError::InvalidElement {
            index,
            field: "area_m2",
            requirement: "finite and positive",
        });
    }
    for (field, value) in [
        ("pressure_real_pa", element.pressure_real_pa),
        ("pressure_imaginary_pa", element.pressure_imaginary_pa),
        ("normal_velocity_real_m_s", element.normal_velocity_real_m_s),
        (
            "normal_velocity_imaginary_m_s",
            element.normal_velocity_imaginary_m_s,
        ),
    ] {
        if !value.is_finite() {
            return Err(FwhError::InvalidElement {
                index,
                field,
                requirement: "finite",
            });
        }
    }
    Ok(())
}

fn point_is_finite(point: Point3) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

fn subtract(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn scale(point: Point3, factor: f64) -> Point3 {
    Point3 {
        x: point.x * factor,
        y: point.y * factor,
        z: point.z * factor,
    }
}

fn dot(left: Point3, right: Point3) -> f64 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn norm(point: Point3) -> f64 {
    point.x.hypot(point.y).hypot(point.z)
}

fn add(left: FwhComplexPressure, right: FwhComplexPressure) -> FwhComplexPressure {
    FwhComplexPressure {
        real_pa: left.real_pa + right.real_pa,
        imaginary_pa: left.imaginary_pa + right.imaginary_pa,
    }
}

fn multiply(left: FwhComplexPressure, right: FwhComplexPressure) -> FwhComplexPressure {
    FwhComplexPressure {
        real_pa: left.real_pa * right.real_pa - left.imaginary_pa * right.imaginary_pa,
        imaginary_pa: left.real_pa * right.imaginary_pa + left.imaginary_pa * right.real_pa,
    }
}

fn multiply_by_i(value: FwhComplexPressure, factor: f64) -> FwhComplexPressure {
    FwhComplexPressure {
        real_pa: -value.imaginary_pa * factor,
        imaginary_pa: value.real_pa * factor,
    }
}

fn pressure_is_finite(pressure: FwhComplexPressure) -> bool {
    pressure.real_pa.is_finite() && pressure.imaginary_pa.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::physics::acoustics::aeroacoustics::{FwhSurfacePatch, stationary_fwh_pressure};

    /// Audit: the far-field-only reference agrees with the full-kernel
    /// FEM evaluator at `kr >> 1`, where the `1/R` near-field loading
    /// term is a sub-percent correction.
    #[test]
    fn far_field_reference_matches_full_kernel_fem_evaluator() {
        // Six-element disk source with mixed thickness/loading phasors.
        let elements: Vec<FwhSurfaceElement> = (0..6)
            .map(|index| {
                let angle = index as f64 * std::f64::consts::TAU / 6.0;
                FwhSurfaceElement {
                    position: Point3 {
                        x: 0.05 * angle.cos(),
                        y: 0.05 * angle.sin(),
                        z: 0.0,
                    },
                    normal: Point3 {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                    },
                    area_m2: 0.001,
                    pressure_real_pa: 100.0 + 10.0 * index as f64,
                    pressure_imaginary_pa: -20.0,
                    normal_velocity_real_m_s: 0.01 * index as f64,
                    normal_velocity_imaginary_m_s: 0.05,
                }
            })
            .collect();
        let options = FwhOptions {
            frequency_hz: 1_000.0,
            observer_position: Point3 {
                x: 8.0,
                y: 0.0,
                z: 6.0,
            },
            ..FwhOptions::default()
        };
        let reference = evaluate_fwh_reference(&elements, options).unwrap();

        let patches: Vec<FwhSurfacePatch> = elements
            .iter()
            .map(|element| {
                FwhSurfacePatch::new(
                    [element.position.x, element.position.y, element.position.z],
                    [element.normal.x, element.normal.y, element.normal.z],
                    element.area_m2,
                    num_complex::Complex64::new(
                        element.pressure_real_pa,
                        element.pressure_imaginary_pa,
                    ),
                    num_complex::Complex64::new(
                        element.normal_velocity_real_m_s,
                        element.normal_velocity_imaginary_m_s,
                    ),
                )
                .unwrap()
            })
            .collect();
        let full = stationary_fwh_pressure(
            &patches,
            [
                options.observer_position.x,
                options.observer_position.y,
                options.observer_position.z,
            ],
            std::f64::consts::TAU * options.frequency_hz,
            options.density_kg_m3,
            options.sound_speed_m_s,
        )
        .unwrap();

        let relative = |a: FwhComplexPressure, b: num_complex::Complex64| {
            ((a.real_pa - b.re).hypot(a.imaginary_pa - b.im)) / b.norm().max(1e-30)
        };
        // kr ~ 180: thickness matches to roundoff (identical closed
        // form); loading carries the 1/(kr) near-field difference.
        assert!(
            relative(reference.monopole_pressure_pa, full.thickness) < 1e-12,
            "thickness mismatch"
        );
        assert!(
            relative(reference.dipole_pressure_pa, full.loading) < 2e-2,
            "loading mismatch"
        );
        assert!(
            relative(reference.total_pressure_pa, full.total) < 2e-2,
            "total mismatch"
        );
    }
}
