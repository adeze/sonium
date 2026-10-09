//! Bounded compressibility and compact-source admission reference.
//!
//! The compact observer references in this crate (for example
//! [`crate::fwh_reference`]) assume a subsonic rotor and an acoustically
//! compact source (`kL << 1`). This module makes that assumption
//! machine-checkable for one operating condition: it derives the tip Mach
//! number from shaft speed and tip radius, and the Helmholtz number from
//! frequency and caller-declared source extent. Supersonic tips fail closed;
//! non-compact sources are reported and optionally rejected so compact
//! propagation must not be silently applied where it does not hold. This is
//! an admission contract only, not a transonic-flow, broadband-closure, CFD,
//! or measured-validation solver.

use std::fmt;

/// Controls for the bounded compressibility/compactness check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressibilityOptions {
    /// Shaft speed [RPM], finite and positive.
    pub rpm: f64,
    /// Blade tip radius [m], finite and positive.
    pub tip_radius_m: f64,
    /// Quiescent medium sound speed [m/s], finite and positive.
    pub sound_speed_m_s: f64,
    /// Harmonic frequency under test [Hz], finite and positive.
    pub frequency_hz: f64,
    /// Caller-declared acoustic source extent [m], finite and positive.
    /// Use the rotor diameter when the whole disk radiates.
    pub source_extent_m: f64,
    /// Maximum admitted tip Mach number [-], finite and positive.
    /// The default (`1.0`) rejects sonic and supersonic tips.
    pub maximum_tip_mach: f64,
    /// Maximum admitted Helmholtz number `kL` [-], finite and positive.
    /// The default (`1.0`) marks `kL > 1` as non-compact.
    pub maximum_compact_helmholtz: f64,
    /// When true, a non-compact source is a failure instead of a report.
    pub require_compact: bool,
}

impl Default for CompressibilityOptions {
    fn default() -> Self {
        Self {
            rpm: 1_800.0,
            tip_radius_m: 0.15,
            sound_speed_m_s: 343.0,
            frequency_hz: 100.0,
            source_extent_m: 0.3,
            maximum_tip_mach: 1.0,
            maximum_compact_helmholtz: 1.0,
            require_compact: false,
        }
    }
}

/// Result of the bounded compressibility/compactness evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressibilityResult {
    /// Rotor tip speed `omega R` [m/s].
    pub tip_speed_m_s: f64,
    /// Tip Mach number `omega R / c` [-].
    pub tip_mach: f64,
    /// Acoustic wavenumber `2 pi f / c` [rad/m].
    pub wavenumber_rad_m: f64,
    /// Helmholtz number `kL` for the declared source extent [-].
    pub helmholtz_number: f64,
    /// Whether `helmholtz_number <= maximum_compact_helmholtz`.
    pub is_compact: bool,
}

impl CompressibilityResult {
    /// Whether all diagnostics are finite and internally consistent.
    pub fn is_finite(&self) -> bool {
        self.tip_speed_m_s.is_finite()
            && self.tip_speed_m_s > 0.0
            && self.tip_mach.is_finite()
            && self.tip_mach >= 0.0
            && self.wavenumber_rad_m.is_finite()
            && self.wavenumber_rad_m > 0.0
            && self.helmholtz_number.is_finite()
            && self.helmholtz_number > 0.0
    }
}

/// Failure-closed errors from the compressibility/compactness reference.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressibilityError {
    /// An option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The rotor tip is sonic or supersonic under the declared admission.
    SupersonicTip {
        /// Computed tip Mach number [-].
        mach: f64,
        /// Maximum admitted tip Mach number [-].
        maximum: f64,
    },
    /// The source is non-compact and `require_compact` was set.
    NonCompactSource {
        /// Computed Helmholtz number [-].
        helmholtz: f64,
        /// Maximum admitted Helmholtz number [-].
        maximum: f64,
    },
    /// A derived quantity became non-finite.
    NonFiniteResult {
        /// Derived field name.
        field: &'static str,
    },
}

impl fmt::Display for CompressibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "invalid compressibility option {field}: {requirement}"
                )
            }
            Self::SupersonicTip { mach, maximum } => {
                write!(
                    formatter,
                    "supersonic rotor tip Mach {mach} exceeds admitted {maximum}"
                )
            }
            Self::NonCompactSource { helmholtz, maximum } => {
                write!(
                    formatter,
                    "non-compact source Helmholtz number {helmholtz} exceeds admitted {maximum}"
                )
            }
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite compressibility result field {field}")
            }
        }
    }
}

impl std::error::Error for CompressibilityError {}

/// Evaluate tip Mach and Helmholtz compactness for one operating condition.
///
/// The tip speed follows rigid rotation (`omega R`); the wavenumber follows
/// the quiescent-medium dispersion (`2 pi f / c`). Supersonic tips always
/// fail closed. Non-compact sources fail closed only when
/// `require_compact` is set, so screening sweeps can record the flag while
/// compact observer paths must opt into rejection.
pub fn evaluate_compressibility_reference(
    options: CompressibilityOptions,
) -> Result<CompressibilityResult, CompressibilityError> {
    validate_options(options)?;

    let tip_speed_m_s = options.rpm * 2.0 * std::f64::consts::PI / 60.0 * options.tip_radius_m;
    let tip_mach = tip_speed_m_s / options.sound_speed_m_s;
    let wavenumber_rad_m =
        2.0 * std::f64::consts::PI * options.frequency_hz / options.sound_speed_m_s;
    let helmholtz_number = wavenumber_rad_m * options.source_extent_m;
    for (field, value) in [
        ("tip_speed_m_s", tip_speed_m_s),
        ("tip_mach", tip_mach),
        ("wavenumber_rad_m", wavenumber_rad_m),
        ("helmholtz_number", helmholtz_number),
    ] {
        if !value.is_finite() {
            return Err(CompressibilityError::NonFiniteResult { field });
        }
    }

    if tip_mach >= options.maximum_tip_mach {
        return Err(CompressibilityError::SupersonicTip {
            mach: tip_mach,
            maximum: options.maximum_tip_mach,
        });
    }

    let is_compact = helmholtz_number <= options.maximum_compact_helmholtz;
    if !is_compact && options.require_compact {
        return Err(CompressibilityError::NonCompactSource {
            helmholtz: helmholtz_number,
            maximum: options.maximum_compact_helmholtz,
        });
    }

    let result = CompressibilityResult {
        tip_speed_m_s,
        tip_mach,
        wavenumber_rad_m,
        helmholtz_number,
        is_compact,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(CompressibilityError::NonFiniteResult {
            field: "compressibility_result",
        })
    }
}

fn validate_options(options: CompressibilityOptions) -> Result<(), CompressibilityError> {
    for (field, value) in [
        ("rpm", options.rpm),
        ("tip_radius_m", options.tip_radius_m),
        ("sound_speed_m_s", options.sound_speed_m_s),
        ("frequency_hz", options.frequency_hz),
        ("source_extent_m", options.source_extent_m),
        ("maximum_tip_mach", options.maximum_tip_mach),
        (
            "maximum_compact_helmholtz",
            options.maximum_compact_helmholtz,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(CompressibilityError::InvalidOptions {
                field,
                requirement: "finite and positive",
            });
        }
    }
    Ok(())
}
