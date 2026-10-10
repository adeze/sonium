//! Rotor-alone tones through the axisymmetric FW-H analogy.
//!
//! This module propagates PC-fan blade loading to free-field observers
//! with revolved FW-H patches: the rotor disk (hub to tip annulus at
//! `z = 0`) carries a spinning pressure jump whose azimuthal order is
//! `m = -h·B` for BPF harmonic `h` and blade count `B`. Each radial
//! strip radiates as a dipole layer with the modal loading amplitude;
//! thickness noise is neglected (thin blades at low tip Mach).
//!
//! The loading model is deliberately minimal: blade-local pressure jump
//! `Δp/σ` (thrust-conserving over the `B` top-hat blades covering a
//! pitch fraction `σ`, with `Δp = thrust / disk area`), whose Fourier
//! coefficient at order `hB` is `Δp·sin(hπσ)/(hπσ)`. A full disk
//! (`σ = 1`) is steady and axisymmetric, so all `h ≥ 1` tones vanish,
//! as they must.
//!
//! This path validates and audits the empirical tone model in
//! [`crate::evaluation`]; it does not replace the optimizer hot path.

use num_complex::Complex64;
use sonium_fem::physics::acoustics::aeroacoustics::{
    AzimuthalMode, RevolvedQuadrature, revolved_patch_from_modal_data, stationary_fwh_ring_pressure,
};
use std::fmt;

/// Failure while building or evaluating annular rotor tones.
#[derive(Debug, Clone, PartialEq)]
pub enum AnnularFwhError {
    /// A specification field is outside its declared domain.
    InvalidSpec {
        /// Specification field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The analogy evaluator rejected the assembled source.
    Analogy(String),
}

impl fmt::Display for AnnularFwhError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec { field, requirement } => {
                write!(
                    formatter,
                    "invalid annular FW-H spec {field}: {requirement}"
                )
            }
            Self::Analogy(error) => write!(formatter, "annular FW-H evaluation failed: {error}"),
        }
    }
}

impl std::error::Error for AnnularFwhError {}

/// Rotor and flow inputs for the annular FW-H tone path.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnularToneSpec {
    /// Blade count (≥ 1).
    pub blade_count: usize,
    /// Shaft speed [rev/min].
    pub rpm: f64,
    /// Steady rotor thrust [N] (≥ 0).
    pub thrust_n: f64,
    /// Hub radius [m] (≥ 0, below tip).
    pub hub_radius_m: f64,
    /// Tip radius [m] (> hub).
    pub tip_radius_m: f64,
    /// Blade pitch solidity in (0, 1]: fraction of each blade pitch
    /// covered by blade chord.
    pub blade_solidity: f64,
    /// Radial strip count over the annulus (≥ 1).
    pub strip_count: usize,
    /// One-based BPF harmonics to evaluate (non-empty).
    pub harmonics: Vec<usize>,
    /// Medium density [kg/m³].
    pub density_kg_m3: f64,
    /// Medium sound speed [m/s].
    pub sound_speed_m_s: f64,
}

/// One evaluated rotor-alone tone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnularTone {
    /// One-based BPF harmonic.
    pub harmonic: usize,
    /// Tone frequency [Hz].
    pub frequency_hz: f64,
    /// Spinning order in the `exp(+imφ)` basis (`-h·B`).
    pub azimuthal_order: i32,
    /// Complex peak pressure phasor [Pa].
    pub pressure_pa: Complex64,
}

/// Blade-passing frequency [Hz].
pub fn blade_passing_frequency_hz(blade_count: usize, rpm: f64) -> f64 {
    blade_count as f64 * rpm / 60.0
}

/// Modal pressure-jump amplitude [Pa] of one BPF harmonic.
///
/// Fourier coefficient of the `B`-blade top-hat train at spinning
/// order `-h·B`, real-positive for blades centered on `φ = 0`. Blades
/// covering a pitch fraction `σ` carry the blade-local jump `Δp/σ`
/// (thrust-conserving), whose train coefficient is
/// `(Δp/σ)·σ·sin(hπσ)/(hπσ) = Δp·sinc(hπσ)`.
pub fn modal_loading_amplitude_pa(
    thrust_n: f64,
    disk_area_m2: f64,
    solidity: f64,
    harmonic: usize,
) -> f64 {
    let pressure_jump = thrust_n / disk_area_m2;
    let argument = harmonic as f64 * std::f64::consts::PI * solidity;
    pressure_jump * argument.sin() / argument
}

fn require_positive(field: &'static str, value: f64) -> Result<(), AnnularFwhError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(AnnularFwhError::InvalidSpec {
            field,
            requirement: "finite and positive",
        })
    }
}

fn validate_spec(spec: &AnnularToneSpec) -> Result<(), AnnularFwhError> {
    if spec.blade_count < 1 {
        return Err(AnnularFwhError::InvalidSpec {
            field: "blade_count",
            requirement: "at least one blade",
        });
    }
    require_positive("rpm", spec.rpm)?;
    if !spec.thrust_n.is_finite() || spec.thrust_n < 0.0 {
        return Err(AnnularFwhError::InvalidSpec {
            field: "thrust_n",
            requirement: "finite and non-negative",
        });
    }
    if !spec.hub_radius_m.is_finite() || spec.hub_radius_m < 0.0 {
        return Err(AnnularFwhError::InvalidSpec {
            field: "hub_radius_m",
            requirement: "finite and non-negative",
        });
    }
    require_positive("tip_radius_m", spec.tip_radius_m)?;
    if spec.hub_radius_m >= spec.tip_radius_m {
        return Err(AnnularFwhError::InvalidSpec {
            field: "hub_radius_m",
            requirement: "below tip_radius_m",
        });
    }
    if !spec.blade_solidity.is_finite() || spec.blade_solidity <= 0.0 || spec.blade_solidity > 1.0 {
        return Err(AnnularFwhError::InvalidSpec {
            field: "blade_solidity",
            requirement: "in (0, 1]",
        });
    }
    if spec.strip_count < 1 {
        return Err(AnnularFwhError::InvalidSpec {
            field: "strip_count",
            requirement: "at least one radial strip",
        });
    }
    if spec.harmonics.is_empty() || spec.harmonics.iter().any(|harmonic| *harmonic < 1) {
        return Err(AnnularFwhError::InvalidSpec {
            field: "harmonics",
            requirement: "non-empty one-based harmonic list",
        });
    }
    require_positive("density_kg_m3", spec.density_kg_m3)?;
    require_positive("sound_speed_m_s", spec.sound_speed_m_s)?;
    Ok(())
}

/// Evaluate rotor-alone tones at one free-field observer.
///
/// Builds uniform radial strips over the rotor annulus with the modal
/// loading of each requested BPF harmonic and sums the revolved-patch
/// FW-H response. The observer must clear the disk plane singularity
/// (the evaluator rejects coincident points).
pub fn annular_loading_tones(
    spec: &AnnularToneSpec,
    observer_m: [f64; 3],
    quadrature: RevolvedQuadrature,
) -> Result<Vec<AnnularTone>, AnnularFwhError> {
    validate_spec(spec)?;
    if observer_m.iter().any(|value| !value.is_finite()) {
        return Err(AnnularFwhError::InvalidSpec {
            field: "observer_m",
            requirement: "finite position",
        });
    }
    let disk_area = std::f64::consts::PI * (spec.tip_radius_m.powi(2) - spec.hub_radius_m.powi(2));
    let blade_passing = blade_passing_frequency_hz(spec.blade_count, spec.rpm);
    let mut tones = Vec::with_capacity(spec.harmonics.len());
    for &harmonic in &spec.harmonics {
        let order = -((harmonic.saturating_mul(spec.blade_count)) as i32);
        let azimuthal = AzimuthalMode::new(order);
        let amplitude =
            modal_loading_amplitude_pa(spec.thrust_n, disk_area, spec.blade_solidity, harmonic);
        let frequency_hz = harmonic as f64 * blade_passing;
        let omega = std::f64::consts::TAU * frequency_hz;
        let mut total = Complex64::new(0.0, 0.0);
        for strip in 0..spec.strip_count {
            let fraction = spec.strip_count as f64;
            let inner = spec.hub_radius_m
                + (spec.tip_radius_m - spec.hub_radius_m) * strip as f64 / fraction;
            let outer = spec.hub_radius_m
                + (spec.tip_radius_m - spec.hub_radius_m) * (strip + 1) as f64 / fraction;
            let patch = revolved_patch_from_modal_data(
                [inner, 0.0],
                [outer, 0.0],
                [0.0, 1.0],
                azimuthal,
                Complex64::new(amplitude, 0.0),
                Complex64::new(0.0, 0.0),
            )
            .map_err(|error| AnnularFwhError::Analogy(error.to_string()))?;
            total += stationary_fwh_ring_pressure(
                std::slice::from_ref(&patch),
                observer_m,
                omega,
                spec.density_kg_m3,
                spec.sound_speed_m_s,
                quadrature,
            )
            .map_err(|error| AnnularFwhError::Analogy(error.to_string()))?
            .total;
        }
        tones.push(AnnularTone {
            harmonic,
            frequency_hz,
            azimuthal_order: order,
            pressure_pa: total,
        });
    }
    Ok(tones)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spec() -> AnnularToneSpec {
        AnnularToneSpec {
            blade_count: 7,
            rpm: 3_000.0,
            thrust_n: 2.0,
            hub_radius_m: 0.018,
            tip_radius_m: 0.054,
            blade_solidity: 0.35,
            strip_count: 6,
            harmonics: vec![1, 2],
            density_kg_m3: 1.2,
            sound_speed_m_s: 343.0,
        }
    }

    fn quadrature() -> RevolvedQuadrature {
        RevolvedQuadrature::new(1e-8, 8).unwrap()
    }

    #[test]
    fn unloaded_rotor_is_silent() {
        let mut spec = test_spec();
        spec.thrust_n = 0.0;
        let tones = annular_loading_tones(&spec, [1.0, 0.0, 0.0], quadrature()).unwrap();
        assert!(tones.iter().all(|tone| tone.pressure_pa.norm() == 0.0));
    }

    #[test]
    fn full_disk_carries_no_bpf_tones() {
        // Solidity 1 is steady and axisymmetric: h >= 1 vanish.
        let mut spec = test_spec();
        spec.blade_solidity = 1.0;
        let tones = annular_loading_tones(&spec, [1.0, 0.0, 0.0], quadrature()).unwrap();
        assert!(
            tones.iter().all(|tone| tone.pressure_pa.norm() < 1e-12),
            "{tones:?}"
        );
    }

    #[test]
    fn spinning_tones_vanish_on_axis_and_beam_obliquely() {
        // Axial-dipole layers null in the disc plane (R.z = 0) and
        // spinning orders null on the axis: lobes live at oblique angles.
        let spec = test_spec();
        let on_axis = annular_loading_tones(&spec, [0.0, 0.0, 1.0], quadrature()).unwrap();
        assert!(
            on_axis.iter().all(|tone| tone.pressure_pa.norm() < 1e-9),
            "{on_axis:?}"
        );
        let oblique = annular_loading_tones(&spec, [1.0, 0.0, 1.0], quadrature()).unwrap();
        assert_eq!(oblique[0].azimuthal_order, -7);
        assert_eq!(oblique[1].azimuthal_order, -14);
        assert!(oblique.iter().all(|tone| tone.pressure_pa.norm() > 0.0));
        assert!((oblique[0].frequency_hz - 350.0).abs() < 1e-9);
    }

    #[test]
    fn tones_decay_between_harmonics_and_reject_bad_specs() {
        let spec = test_spec();
        let tones = annular_loading_tones(&spec, [1.0, 0.0, 1.0], quadrature()).unwrap();
        // sinc envelope: harmonic 2 below harmonic 1 for solidity 0.35.
        assert!(tones[1].pressure_pa.norm() < tones[0].pressure_pa.norm());
        let mut bad = spec;
        bad.strip_count = 0;
        assert!(annular_loading_tones(&bad, [1.0, 0.0, 0.0], quadrature()).is_err());
        let mut bad = test_spec();
        bad.blade_solidity = 1.5;
        assert!(annular_loading_tones(&bad, [1.0, 0.0, 0.0], quadrature()).is_err());
        let mut bad = test_spec();
        bad.harmonics.clear();
        assert!(annular_loading_tones(&bad, [1.0, 0.0, 0.0], quadrature()).is_err());
    }
}
