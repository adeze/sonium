//! Shared units and convention metadata for exchange-boundary APIs.
//!
//! Legacy configuration fields remain scalar for compatibility. New
//! exchange-facing APIs should use these small wrappers so units and phasor
//! conventions are explicit at the boundary.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Schema version emitted by the current XEM configuration serializer.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Version carried by a serialized exchange document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    /// Construct the current schema version.
    pub const fn current() -> Self {
        Self(CURRENT_SCHEMA_VERSION)
    }

    /// Whether this version can be consumed without a migration.
    pub const fn is_supported(self) -> bool {
        self.0 == CURRENT_SCHEMA_VERSION
    }
}

impl Default for SchemaVersion {
    fn default() -> Self {
        Self::current()
    }
}

macro_rules! positive_quantity {
    ($name:ident, $unit:literal) => {
        #[doc = concat!("A finite scalar measured in ", $unit, ".")]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub f64);

        impl $name {
            /// Construct the quantity without changing its numeric value.
            pub const fn new(value: f64) -> Self {
                Self(value)
            }

            /// Return the scalar value in the documented unit.
            pub const fn get(self) -> f64 {
                self.0
            }

            /// Construct a finite, positive quantity.
            pub fn try_new(value: f64) -> Result<Self, &'static str> {
                if value.is_finite() && value > 0.0 {
                    Ok(Self(value))
                } else {
                    Err(concat!(stringify!($name), " must be finite and positive"))
                }
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                Self::try_new(self.0)
                    .map_err(|error| <S::Error as serde::ser::Error>::custom(error))?;
                self.0.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = f64::deserialize(deserializer)?;
                Self::try_new(value).map_err(|error| <D::Error as serde::de::Error>::custom(error))
            }
        }
    };
}

positive_quantity!(FrequencyHz, "Hz");
positive_quantity!(LengthM, "m");
positive_quantity!(DensityKgM3, "kg/m^3");
positive_quantity!(SpeedOfSoundMPerS, "m/s");
positive_quantity!(FlowResistivityPaSPerm2, "Pa s/m^2");

macro_rules! finite_quantity {
    ($name:ident, $unit:literal) => {
        #[doc = concat!("A finite scalar measured in ", $unit, ".")]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub f64);

        impl $name {
            /// Construct the quantity without changing its numeric value.
            pub const fn new(value: f64) -> Self {
                Self(value)
            }

            /// Return the scalar value in the documented unit.
            pub const fn get(self) -> f64 {
                self.0
            }

            /// Construct a finite quantity. Signed values are valid for
            /// pressure, impedance components, and phase.
            pub fn try_new(value: f64) -> Result<Self, &'static str> {
                if value.is_finite() {
                    Ok(Self(value))
                } else {
                    Err(concat!(stringify!($name), " must be finite"))
                }
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                Self::try_new(self.0)
                    .map_err(|error| <S::Error as serde::ser::Error>::custom(error))?;
                self.0.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = f64::deserialize(deserializer)?;
                Self::try_new(value).map_err(|error| <D::Error as serde::de::Error>::custom(error))
            }
        }
    };
}

finite_quantity!(PressurePa, "Pa");
finite_quantity!(ImpedancePaSPerM, "Pa s/m");
finite_quantity!(PhaseRad, "rad");

/// Coordinate frame used by a point, vector, or field payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CoordinateFrame {
    /// Right-handed world frame shared by solver meshes.
    #[default]
    WorldRightHanded,
    /// Right-handed frame local to a source or component.
    SourceLocalRightHanded,
}

/// Harmonic time convention used by complex phasors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HarmonicConvention {
    /// `p(t) = Re(P exp(-i ω t))`.
    #[default]
    ExpMinusIwt,
    /// `p(t) = Re(P exp(+i ω t))`.
    ExpPlusIwt,
}

impl HarmonicConvention {
    /// Return `(time_sign, spatial_sign)` for the outgoing acoustic factor.
    ///
    /// A phasor using `exp(-iωt)` radiates with `exp(+ikr)`, while the
    /// conjugate time convention radiates with `exp(-ikr)`. Keeping this
    /// relationship in XEM prevents BEM and FEM adapters from drifting.
    pub const fn signs(self) -> (i8, i8) {
        match self {
            Self::ExpMinusIwt => (-1, 1),
            Self::ExpPlusIwt => (1, -1),
        }
    }
}

/// Conventions that travel with a coupled BEM/FEM exchange payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConventionRegistry {
    /// Complex harmonic time convention.
    pub harmonic_time: HarmonicConvention,
    /// Coordinate handedness/frame convention.
    pub coordinate_frame: CoordinateFrame,
    /// Whether normals point out of the physical domain.
    pub outward_normals: bool,
    /// Whether amplitudes are RMS (`true`) or peak (`false`).
    pub rms_amplitudes: bool,
}

impl Default for ConventionRegistry {
    fn default() -> Self {
        Self {
            harmonic_time: HarmonicConvention::ExpMinusIwt,
            coordinate_frame: CoordinateFrame::WorldRightHanded,
            outward_normals: true,
            rms_amplitudes: true,
        }
    }
}

/// Typed acoustic medium exchanged by BEM/FEM and application frontends.
///
/// Keeping frequency and material properties together prevents one consumer
/// from deriving a wavenumber with a different unit or phasor convention than
/// another.  The scalar accessors remain available for compatibility APIs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcousticMedium {
    /// Frequency of the harmonic solve.
    pub frequency: FrequencyHz,
    /// Speed of sound in metres per second.
    pub speed_of_sound: SpeedOfSoundMPerS,
    /// Fluid density in kilograms per cubic metre.
    pub density: DensityKgM3,
    /// Harmonic and orientation contract attached to the medium.
    pub conventions: ConventionRegistry,
}

impl AcousticMedium {
    /// Construct a medium with the canonical exchange convention.
    pub fn try_new(
        frequency_hz: f64,
        speed_of_sound_m_per_s: f64,
        density_kg_m3: f64,
    ) -> Result<Self, &'static str> {
        Self::try_new_with_conventions(
            FrequencyHz::try_new(frequency_hz)?,
            SpeedOfSoundMPerS::try_new(speed_of_sound_m_per_s)?,
            DensityKgM3::try_new(density_kg_m3)?,
            ConventionRegistry::default(),
        )
    }

    /// Construct a medium from already validated quantities and conventions.
    pub fn try_new_with_conventions(
        frequency: FrequencyHz,
        speed_of_sound: SpeedOfSoundMPerS,
        density: DensityKgM3,
        conventions: ConventionRegistry,
    ) -> Result<Self, &'static str> {
        if !conventions.outward_normals {
            return Err("acoustic medium requires outward normals");
        }
        if !matches!(
            conventions.coordinate_frame,
            CoordinateFrame::WorldRightHanded
        ) {
            return Err("acoustic medium requires the world right-handed frame");
        }
        Ok(Self {
            frequency,
            speed_of_sound,
            density,
            conventions,
        })
    }

    /// Angular frequency in radians per second.
    pub fn angular_frequency(self) -> f64 {
        2.0 * std::f64::consts::PI * self.frequency.get()
    }

    /// Signed outgoing wavenumber under the attached harmonic convention.
    pub fn wavenumber(self) -> f64 {
        let (_, spatial_sign) = self.conventions.harmonic_time.signs();
        f64::from(spatial_sign) * self.angular_frequency() / self.speed_of_sound.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_and_convention_defaults_are_explicit() {
        assert_eq!(SchemaVersion::default(), SchemaVersion::current());
        assert!(SchemaVersion::current().is_supported());
        assert!(!SchemaVersion(CURRENT_SCHEMA_VERSION + 1).is_supported());

        let conventions = ConventionRegistry::default();
        assert_eq!(conventions.harmonic_time, HarmonicConvention::ExpMinusIwt);
        assert_eq!(conventions.harmonic_time.signs(), (-1, 1));
        assert_eq!(HarmonicConvention::ExpPlusIwt.signs(), (1, -1));
        assert_eq!(
            conventions.coordinate_frame,
            CoordinateFrame::WorldRightHanded
        );
        assert!(conventions.outward_normals);
        assert!(conventions.rms_amplitudes);
    }

    #[test]
    fn quantity_constructors_reject_nonphysical_values() {
        assert_eq!(FrequencyHz::try_new(1_000.0).unwrap().get(), 1_000.0);
        assert_eq!(LengthM::try_new(0.25).unwrap().get(), 0.25);
        assert_eq!(DensityKgM3::try_new(1.2).unwrap().get(), 1.2);
        assert_eq!(PressurePa::try_new(-2.0).unwrap().get(), -2.0);
        assert_eq!(ImpedancePaSPerM::try_new(0.0).unwrap().get(), 0.0);
        assert_eq!(
            PhaseRad::try_new(-std::f64::consts::PI).unwrap().get(),
            -std::f64::consts::PI
        );

        assert!(FrequencyHz::try_new(0.0).is_err());
        assert!(LengthM::try_new(f64::NAN).is_err());
        assert!(DensityKgM3::try_new(f64::INFINITY).is_err());
        assert!(FlowResistivityPaSPerm2::try_new(12_000.0).is_ok());
        assert!(FlowResistivityPaSPerm2::try_new(0.0).is_err());
        assert!(PressurePa::try_new(f64::NEG_INFINITY).is_err());
        assert!(PhaseRad::try_new(f64::NAN).is_err());
    }

    #[test]
    fn typed_acoustic_medium_derives_convention_bound_wavenumber() {
        let medium = AcousticMedium::try_new(1_000.0, 343.0, 1.21).expect("valid medium");
        assert!((medium.angular_frequency() - 2.0 * std::f64::consts::PI * 1_000.0).abs() < 1e-12);
        assert!(medium.wavenumber() > 0.0);

        let mut conventions = ConventionRegistry {
            harmonic_time: HarmonicConvention::ExpPlusIwt,
            ..ConventionRegistry::default()
        };
        assert!(
            AcousticMedium::try_new_with_conventions(
                FrequencyHz::new(1_000.0),
                SpeedOfSoundMPerS::new(343.0),
                DensityKgM3::new(1.21),
                conventions,
            )
            .expect("valid convention")
            .wavenumber()
                < 0.0
        );
        conventions.outward_normals = false;
        assert!(
            AcousticMedium::try_new_with_conventions(
                FrequencyHz::new(1_000.0),
                SpeedOfSoundMPerS::new(343.0),
                DensityKgM3::new(1.21),
                conventions,
            )
            .is_err()
        );
    }

    #[test]
    fn serde_quantity_boundary_rejects_invalid_values() {
        assert!(serde_json::from_str::<FrequencyHz>("0.0").is_err());
        assert!(serde_json::from_str::<FrequencyHz>("-1.0").is_err());
        assert!(serde_json::from_str::<DensityKgM3>("null").is_err());
        assert!(serde_json::from_str::<PressurePa>("NaN").is_err());
        assert!(serde_json::from_str::<LengthM>("1.25").is_ok());

        let invalid = FrequencyHz::new(-1.0);
        assert!(serde_json::to_string(&invalid).is_err());
        let valid = FrequencyHz::try_new(1_000.0).expect("valid frequency");
        let encoded = serde_json::to_string(&valid).expect("serialize valid quantity");
        assert_eq!(encoded, "1000.0");
    }
}
