//! Checked compact-source installation image reference.
//!
//! This module adds one explicit installation effect to the analytic rotating
//! dipole: a horizontal-plane image source with a caller-supplied complex
//! reflection coefficient. Direct and reflected paths are summed coherently,
//! including their distance-dependent phase. It is a bounded ground/body
//! image reference, not a body-fitted, ducted, porous, or FW-H installation
//! solver.

use std::fmt;

use crate::{Point3, RotatingDipoleComplexPressure, RotatingDipoleReference};

/// Complex pressure reflection coefficient for the image plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageReflectionCoefficient {
    /// Real part of the pressure coefficient.
    pub real: f64,
    /// Imaginary part of the pressure coefficient.
    pub imaginary: f64,
}

impl ImageReflectionCoefficient {
    /// Construct a passive coefficient with magnitude at most one.
    pub fn try_new(real: f64, imaginary: f64) -> Result<Self, InstallationReferenceError> {
        let magnitude = real.hypot(imaginary);
        if !real.is_finite() || !imaginary.is_finite() || !magnitude.is_finite() {
            return Err(InstallationReferenceError::InvalidReflection {
                requirement: "finite",
            });
        }
        if magnitude > 1.0 {
            return Err(InstallationReferenceError::InvalidReflection {
                requirement: "magnitude at most one",
            });
        }
        Ok(Self { real, imaginary })
    }

    /// Perfectly rigid pressure-reflecting plane.
    pub const fn rigid() -> Self {
        Self {
            real: 1.0,
            imaginary: 0.0,
        }
    }

    /// Pressure-release plane.
    pub const fn pressure_release() -> Self {
        Self {
            real: -1.0,
            imaginary: 0.0,
        }
    }
}

/// Coherent direct, image, and total observer pressure for one harmonic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstallationDirectivityResult {
    /// Direct compact-source pressure phasor [Pa].
    pub direct_pressure_pa: RotatingDipoleComplexPressure,
    /// Unreflected image-source pressure phasor [Pa].
    pub image_pressure_pa: RotatingDipoleComplexPressure,
    /// Image pressure after applying the reflection coefficient [Pa].
    pub reflected_pressure_pa: RotatingDipoleComplexPressure,
    /// Coherent direct-plus-reflected pressure phasor [Pa].
    pub total_pressure_pa: RotatingDipoleComplexPressure,
    /// Direct source-to-observer distance [m].
    pub direct_distance_m: f64,
    /// Image source-to-observer distance [m].
    pub image_distance_m: f64,
}

impl InstallationDirectivityResult {
    /// Whether all path and pressure values are finite and positive-distance.
    pub fn is_finite(&self) -> bool {
        self.direct_distance_m.is_finite()
            && self.direct_distance_m > 0.0
            && self.image_distance_m.is_finite()
            && self.image_distance_m > 0.0
            && pressure_is_finite(self.direct_pressure_pa)
            && pressure_is_finite(self.image_pressure_pa)
            && pressure_is_finite(self.reflected_pressure_pa)
            && pressure_is_finite(self.total_pressure_pa)
    }
}

/// Failure-closed installation-reference error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InstallationReferenceError {
    /// A point is non-finite or on an invalid side of the image plane.
    InvalidPoint {
        /// Point field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Reflection coefficient is non-finite or active.
    InvalidReflection {
        /// Required coefficient domain.
        requirement: &'static str,
    },
    /// Source and observer coincide for one path.
    CoincidentPath {
        /// Path name.
        path: &'static str,
    },
    /// Compact dipole reference could not evaluate a path.
    InvalidReference,
    /// The final coherent pressure became non-finite.
    NonFiniteResult,
}

impl fmt::Display for InstallationReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPoint { field, requirement } => {
                write!(f, "installation reference {field} must be {requirement}")
            }
            Self::InvalidReflection { requirement } => {
                write!(f, "installation reflection must have {requirement}")
            }
            Self::CoincidentPath { path } => {
                write!(
                    f,
                    "installation reference {path} source and observer must differ"
                )
            }
            Self::InvalidReference => write!(f, "installation reference dipole evaluation failed"),
            Self::NonFiniteResult => write!(f, "installation reference result is non-finite"),
        }
    }
}

impl std::error::Error for InstallationReferenceError {}

/// Evaluate a compact rotating dipole above a horizontal image plane.
///
/// The source and observer must both have positive `z` coordinates. The image
/// source is reflected to `(x, y, -z)` and multiplied by `reflection`. Direct
/// and image pressures use the same analytic dipole orientation and are summed
/// coherently with their own path lengths and observer angles.
pub fn evaluate_ground_image_directivity(
    reference: RotatingDipoleReference,
    source_position: Point3,
    observer_position: Point3,
    reflection: ImageReflectionCoefficient,
) -> Result<InstallationDirectivityResult, InstallationReferenceError> {
    validate_point("source_position", source_position)?;
    validate_point("observer_position", observer_position)?;
    if source_position.z <= 0.0 {
        return Err(InstallationReferenceError::InvalidPoint {
            field: "source_position.z",
            requirement: "strictly above the image plane",
        });
    }
    if observer_position.z <= 0.0 {
        return Err(InstallationReferenceError::InvalidPoint {
            field: "observer_position.z",
            requirement: "strictly above the image plane",
        });
    }

    let image_position = Point3 {
        x: source_position.x,
        y: source_position.y,
        z: -source_position.z,
    };
    let direct_vector = subtract(observer_position, source_position);
    let image_vector = subtract(observer_position, image_position);
    let direct_distance_m = norm(direct_vector);
    let image_distance_m = norm(image_vector);
    if !direct_distance_m.is_finite() || direct_distance_m <= 0.0 {
        return Err(InstallationReferenceError::CoincidentPath { path: "direct" });
    }
    if !image_distance_m.is_finite() || image_distance_m <= 0.0 {
        return Err(InstallationReferenceError::CoincidentPath { path: "image" });
    }

    let direct_pressure_pa = pressure_at_path(reference, direct_vector, direct_distance_m)?;
    let image_pressure_pa = pressure_at_path(reference, image_vector, image_distance_m)?;
    let reflected_pressure_pa = multiply(image_pressure_pa, reflection);
    let total_pressure_pa = add(direct_pressure_pa, reflected_pressure_pa);
    let result = InstallationDirectivityResult {
        direct_pressure_pa,
        image_pressure_pa,
        reflected_pressure_pa,
        total_pressure_pa,
        direct_distance_m,
        image_distance_m,
    };
    if !result.is_finite() {
        return Err(InstallationReferenceError::NonFiniteResult);
    }
    Ok(result)
}

fn pressure_at_path(
    reference: RotatingDipoleReference,
    vector: Point3,
    distance_m: f64,
) -> Result<RotatingDipoleComplexPressure, InstallationReferenceError> {
    let polar_angle_rad = (vector.z / distance_m).clamp(-1.0, 1.0).acos();
    let azimuth_angle_rad = vector.y.atan2(vector.x);
    reference
        .complex_pressure_pa(distance_m, polar_angle_rad, azimuth_angle_rad)
        .map_err(|_| InstallationReferenceError::InvalidReference)
}

fn validate_point(field: &'static str, point: Point3) -> Result<(), InstallationReferenceError> {
    if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
        return Err(InstallationReferenceError::InvalidPoint {
            field,
            requirement: "finite",
        });
    }
    Ok(())
}

fn subtract(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn norm(point: Point3) -> f64 {
    point.x.hypot(point.y).hypot(point.z)
}

fn add(
    left: RotatingDipoleComplexPressure,
    right: RotatingDipoleComplexPressure,
) -> RotatingDipoleComplexPressure {
    RotatingDipoleComplexPressure {
        real_pa: left.real_pa + right.real_pa,
        imaginary_pa: left.imaginary_pa + right.imaginary_pa,
    }
}

fn multiply(
    left: RotatingDipoleComplexPressure,
    right: ImageReflectionCoefficient,
) -> RotatingDipoleComplexPressure {
    RotatingDipoleComplexPressure {
        real_pa: left.real_pa * right.real - left.imaginary_pa * right.imaginary,
        imaginary_pa: left.real_pa * right.imaginary + left.imaginary_pa * right.real,
    }
}

fn pressure_is_finite(value: RotatingDipoleComplexPressure) -> bool {
    value.real_pa.is_finite() && value.imaginary_pa.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> RotatingDipoleReference {
        RotatingDipoleReference::try_from_equatorial_amplitude(2, 240.0, 1.0, 2.0, 1.2, 343.0)
            .expect("valid installation dipole")
    }

    fn source() -> Point3 {
        Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.25,
        }
    }

    fn observer() -> Point3 {
        Point3 {
            x: 1.0,
            y: 0.2,
            z: 0.8,
        }
    }

    #[test]
    fn image_reference_is_finite_and_zero_reflection_recovers_direct_path() {
        let direct = evaluate_ground_image_directivity(
            reference(),
            source(),
            observer(),
            ImageReflectionCoefficient::try_new(0.0, 0.0).unwrap(),
        )
        .expect("zero-reflection image reference");
        assert!(direct.is_finite());
        assert_eq!(direct.total_pressure_pa, direct.direct_pressure_pa);
        assert!(direct.image_distance_m > direct.direct_distance_m);
    }

    #[test]
    fn image_reference_applies_coherent_complex_reflection() {
        let result = evaluate_ground_image_directivity(
            reference(),
            source(),
            observer(),
            ImageReflectionCoefficient::try_new(0.3, -0.4).unwrap(),
        )
        .expect("complex-reflection image reference");
        let expected = multiply(
            result.image_pressure_pa,
            ImageReflectionCoefficient {
                real: 0.3,
                imaginary: -0.4,
            },
        );
        assert_eq!(result.reflected_pressure_pa, expected);
        assert_eq!(
            result.total_pressure_pa,
            add(result.direct_pressure_pa, result.reflected_pressure_pa)
        );
    }

    #[test]
    fn image_reference_rejects_invalid_points_and_reflections() {
        assert!(matches!(
            ImageReflectionCoefficient::try_new(1.0, 1.0),
            Err(InstallationReferenceError::InvalidReflection { .. })
        ));
        assert!(matches!(
            evaluate_ground_image_directivity(
                reference(),
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                observer(),
                ImageReflectionCoefficient::rigid(),
            ),
            Err(InstallationReferenceError::InvalidPoint { .. })
        ));
        assert!(matches!(
            evaluate_ground_image_directivity(
                reference(),
                source(),
                Point3 {
                    x: f64::NAN,
                    y: 0.0,
                    z: 1.0,
                },
                ImageReflectionCoefficient::rigid(),
            ),
            Err(InstallationReferenceError::InvalidPoint { .. })
        ));
    }

    #[test]
    fn image_reference_pressure_follows_inverse_distance() {
        let first = evaluate_ground_image_directivity(
            reference(),
            source(),
            Point3 {
                x: 1.0,
                y: 0.0,
                z: 0.25,
            },
            ImageReflectionCoefficient::try_new(0.0, 0.0).unwrap(),
        )
        .unwrap();
        let second = evaluate_ground_image_directivity(
            reference(),
            source(),
            Point3 {
                x: 2.0,
                y: 0.0,
                z: 0.25,
            },
            ImageReflectionCoefficient::try_new(0.0, 0.0).unwrap(),
        )
        .unwrap();
        assert!(
            (first.direct_pressure_pa.magnitude_pa() / second.direct_pressure_pa.magnitude_pa()
                - 2.0)
                .abs()
                < 1.0e-12
        );
    }
}
