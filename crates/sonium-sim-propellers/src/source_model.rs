//! Source-side aeroacoustic model for realistic propeller design studies.
//!
//! The axisymmetric shroud solver can propagate sound through a meridional duct
//! profile, but it cannot geometrically represent a toroidal loop blade. This
//! module therefore models loop blades and serrations as source-generation
//! changes before axisymmetric propagation:
//!
//! - toroidal loops reduce open-tip vortex shedding and add structural rigidity;
//! - leading-edge serrations pre-condition incoming turbulence;
//! - trailing-edge serrations break coherent wake eddies into smaller vortices.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

/// Empirical interval for an observable band reduction, in decibels.
///
/// These are sensitivity priors rather than confidence intervals. Their
/// applicability and references are documented in `CALIBRATION.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalReductionPriorDb {
    pub center_db: f64,
    pub min_db: f64,
    pub max_db: f64,
}

impl EmpiricalReductionPriorDb {
    pub fn contains(self, reduction_db: f64) -> bool {
        (self.min_db..=self.max_db).contains(&reduction_db)
    }
}

/// Observer-specific toroidal-loop reduction prior.
pub const TOROIDAL_REDUCTION_PRIOR_DB: EmpiricalReductionPriorDb = EmpiricalReductionPriorDb {
    center_db: 5.5,
    min_db: 2.0,
    max_db: 8.0,
};
/// Leading-edge serration prior in its effective turbulent-inflow band.
pub const LE_SERRATION_REDUCTION_PRIOR_DB: EmpiricalReductionPriorDb = EmpiricalReductionPriorDb {
    center_db: 4.0,
    min_db: 0.0,
    max_db: 7.0,
};
/// Trailing-edge serration prior at fixed RPM.
pub const TE_SERRATION_FIXED_RPM_PRIOR_DB: EmpiricalReductionPriorDb = EmpiricalReductionPriorDb {
    center_db: 3.5,
    min_db: 0.0,
    max_db: 6.0,
};
/// Trailing-edge serration prior when compared at matched thrust.
pub const TE_SERRATION_MATCHED_THRUST_PRIOR_DB: EmpiricalReductionPriorDb =
    EmpiricalReductionPriorDb {
        center_db: 1.5,
        min_db: 0.0,
        max_db: 4.0,
    };
/// Toroidal plus leading/trailing-edge treatment prior.
pub const COMBINED_REDUCTION_PRIOR_DB: EmpiricalReductionPriorDb = EmpiricalReductionPriorDb {
    center_db: 7.0,
    min_db: 4.0,
    max_db: 10.0,
};

pub const SERRATION_AMPLITUDE_CHORD_FRACTION_RANGE: [f64; 2] = [0.006, 0.035];
pub const SERRATION_WAVELENGTH_CHORD_FRACTION_RANGE: [f64; 2] = [0.035, 0.120];
pub const SERRATION_SPAN_FRACTION_RANGE: [f64; 2] = [0.40, 0.95];

/// Propeller planform family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropellerPlanform {
    /// Conventional open-tip blades.
    Conventional,
    /// Closed loop/toroidal blades that remove the free open tip.
    ToroidalLoop,
}

/// Bio-inspired edge treatment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeTreatment {
    Smooth,
    LeadingEdgeSerrations,
    TrailingEdgeSerrations,
    LeadingAndTrailingSerrations,
}

/// Evaluation fidelity for the source model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationMode {
    /// Fast deterministic engineering surrogate for optimization loops.
    FastSurrogate,
    /// Higher-order quadrature and coherence-control estimate.
    ResearchControl,
}

/// Serration geometry in nondimensional blade-relative terms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SerrationGeometry {
    /// Tooth amplitude as a fraction of local chord.
    pub amplitude_chord_fraction: f64,
    /// Tooth wavelength as a fraction of local chord.
    pub wavelength_chord_fraction: f64,
    /// Fraction of blade span treated by serrations.
    pub span_fraction: f64,
}

impl SerrationGeometry {
    pub fn owl_like_micro_airfoil() -> Self {
        Self {
            amplitude_chord_fraction: 0.018,
            wavelength_chord_fraction: 0.055,
            span_fraction: 0.72,
        }
    }

    fn effectiveness(&self) -> f64 {
        let amp = smooth_window(self.amplitude_chord_fraction, 0.006, 0.035);
        let wavelength = 1.0 - smooth_window(self.wavelength_chord_fraction, 0.09, 0.22);
        let span = self.span_fraction.clamp(0.0, 1.0);
        (0.15 + 0.85 * amp * wavelength) * span
    }
}

impl Default for SerrationGeometry {
    fn default() -> Self {
        Self::owl_like_micro_airfoil()
    }
}

/// High-level propeller design knobs used by the source model.
#[derive(Debug, Clone)]
pub struct PropellerDesign {
    pub source: PropellerSource,
    pub planform: PropellerPlanform,
    pub edge_treatment: EdgeTreatment,
    pub serrations: SerrationGeometry,
}

impl PropellerDesign {
    pub fn conventional(source: PropellerSource) -> Self {
        Self {
            source,
            planform: PropellerPlanform::Conventional,
            edge_treatment: EdgeTreatment::Smooth,
            serrations: SerrationGeometry::default(),
        }
    }

    pub fn toroidal_serrated(source: PropellerSource) -> Self {
        Self {
            source,
            planform: PropellerPlanform::ToroidalLoop,
            edge_treatment: EdgeTreatment::LeadingAndTrailingSerrations,
            serrations: SerrationGeometry::default(),
        }
    }

    /// Build a harmonic spectrum at the requested fidelity.
    pub fn spectrum(&self, n_harmonics: usize, mode: EvaluationMode) -> PropellerSpectrum {
        match mode {
            EvaluationMode::FastSurrogate => self.fast_spectrum(n_harmonics),
            EvaluationMode::ResearchControl => self.research_control_spectrum(n_harmonics),
        }
    }

    fn fast_spectrum(&self, n_harmonics: usize) -> PropellerSpectrum {
        let baseline = self.source.harmonic_amplitudes(n_harmonics);
        let harmonics = baseline
            .into_iter()
            .enumerate()
            .map(|(i, base_pressure_pa)| {
                let harmonic = i + 1;
                let frequency_hz = self.source.harmonic_frequency(harmonic);
                let reduction = self.fast_amplitude_reduction(frequency_hz, harmonic);
                HarmonicNoise {
                    harmonic,
                    frequency_hz,
                    base_pressure_pa,
                    pressure_pa: base_pressure_pa * reduction,
                    amplitude_reduction: reduction,
                }
            })
            .collect();

        PropellerSpectrum::new(EvaluationMode::FastSurrogate, harmonics)
    }

    fn research_control_spectrum(&self, n_harmonics: usize) -> PropellerSpectrum {
        let baseline = self.source.harmonic_amplitudes(n_harmonics);
        // Indexed collect preserves harmonic order on both paths.
        let evaluate_harmonic = |(i, base_pressure_pa): (usize, f64)| {
            let harmonic = i + 1;
            let frequency_hz = self.source.harmonic_frequency(harmonic);
            let reduction = self.research_control_reduction(frequency_hz, harmonic);
            HarmonicNoise {
                harmonic,
                frequency_hz,
                base_pressure_pa,
                pressure_pa: base_pressure_pa * reduction,
                amplitude_reduction: reduction,
            }
        };
        let harmonics: Vec<HarmonicNoise> = if crate::parallelism::use_propeller_parallel(
            baseline.len(),
            crate::parallelism::HARMONICS_PER_THREAD,
        ) {
            baseline
                .into_par_iter()
                .enumerate()
                .map(evaluate_harmonic)
                .collect()
        } else {
            baseline
                .into_iter()
                .enumerate()
                .map(evaluate_harmonic)
                .collect()
        };

        PropellerSpectrum::new(EvaluationMode::ResearchControl, harmonics)
    }

    fn fast_amplitude_reduction(&self, frequency_hz: f64, harmonic: usize) -> f64 {
        let mut reduction = 1.0;

        if self.planform == PropellerPlanform::ToroidalLoop {
            let tip_vortex_band = octave_like_band(frequency_hz, 1_000.0, 5_000.0);
            let broadband_tip_relief = 0.08 + 0.27 * tip_vortex_band;
            let rigidity_relief = 0.08 * high_frequency_ramp(frequency_hz, 700.0);
            reduction *= 1.0 - broadband_tip_relief - rigidity_relief;
        }

        let serration_effect = self.serrations.effectiveness();
        match self.edge_treatment {
            EdgeTreatment::Smooth => {}
            EdgeTreatment::LeadingEdgeSerrations => {
                let inflow =
                    0.18 * serration_effect * octave_like_band(frequency_hz, 800.0, 4_500.0);
                reduction *= 1.0 - inflow;
            }
            EdgeTreatment::TrailingEdgeSerrations => {
                let wake =
                    0.24 * serration_effect * octave_like_band(frequency_hz, 1_200.0, 6_000.0);
                reduction *= 1.0 - wake;
            }
            EdgeTreatment::LeadingAndTrailingSerrations => {
                let inflow =
                    0.18 * serration_effect * octave_like_band(frequency_hz, 800.0, 4_500.0);
                let wake =
                    0.24 * serration_effect * octave_like_band(frequency_hz, 1_200.0, 6_000.0);
                let decoherence = 0.05 * serration_effect * (harmonic as f64).ln_1p().min(2.0);
                reduction *= 1.0 - inflow - wake - decoherence;
            }
        }

        reduction.clamp(0.18, 1.0)
    }

    fn research_control_reduction(&self, frequency_hz: f64, harmonic: usize) -> f64 {
        let radial_nodes = 256usize;
        let azimuth_nodes = 512usize;
        let tip_radius = self.source.tip_radius;
        let hub_radius = self.source.hub_radius;
        let span = (tip_radius - hub_radius).max(f64::EPSILON);
        let omega = self.source.omega();

        // Evaluate radial rows in parallel, but combine them in their stable
        // radial order.  A free Rayon reduction would make the floating-point
        // summation order worker-count dependent and would invalidate the
        // deterministic artifact/checksum contract.
        // Indexed collect preserves radial order; the fold below then sums
        // rows in stable order on both paths.
        let integrate_row = |ir: usize| {
            let eta = (ir as f64 + 0.5) / radial_nodes as f64;
            let radius = hub_radius + eta * span;
            let local_mach = omega * radius / self.source.speed_of_sound;
            let radial_weight = eta.powi(3);
            let tip_weight = eta.powi(4);
            let edge_weight = (0.25 + 0.75 * eta).powi(3);

            let mut sum = 0.0;
            let mut baseline_sum = 0.0;
            for ia in 0..azimuth_nodes {
                let theta = (ia as f64 + 0.5) / azimuth_nodes as f64;
                let phase = harmonic as f64 * std::f64::consts::TAU * theta;
                let loop_tip_relief = if self.planform == PropellerPlanform::ToroidalLoop {
                    0.66 * tip_weight * octave_like_band(frequency_hz, 1_000.0, 5_000.0)
                } else {
                    0.0
                };
                let serration_relief = self.local_serration_relief(frequency_hz, eta, phase);
                let rigidity_relief = if self.planform == PropellerPlanform::ToroidalLoop {
                    0.13 * high_frequency_ramp(frequency_hz, 700.0)
                } else {
                    0.0
                };

                let local =
                    (1.0 - loop_tip_relief - serration_relief - rigidity_relief).clamp(0.12, 1.0);
                let mach_scaling = (local_mach / self.source.tip_mach().max(1e-9)).powi(2);
                let weight = radial_weight * edge_weight * mach_scaling;
                sum += local * local * weight;
                baseline_sum += weight;
            }
            (
                sum / azimuth_nodes as f64,
                baseline_sum / azimuth_nodes as f64,
            )
        };
        let radial_contributions: Vec<(f64, f64)> = if crate::parallelism::use_propeller_parallel(
            radial_nodes.saturating_mul(azimuth_nodes),
            crate::parallelism::INTEGRATION_POINTS_PER_THREAD,
        ) {
            (0..radial_nodes)
                .into_par_iter()
                .map(integrate_row)
                .collect()
        } else {
            (0..radial_nodes).map(integrate_row).collect()
        };
        let (coherent_energy, baseline_energy) = radial_contributions
            .into_iter()
            .fold((0.0, 0.0), |(coherent, baseline), (row, row_baseline)| {
                (coherent + row, baseline + row_baseline)
            });

        (coherent_energy / baseline_energy.max(f64::EPSILON))
            .sqrt()
            .clamp(0.15, 1.0)
    }

    fn local_serration_relief(&self, frequency_hz: f64, eta: f64, phase: f64) -> f64 {
        if self.edge_treatment == EdgeTreatment::Smooth {
            return 0.0;
        }

        let serration = self.serrations.effectiveness();
        let treated_span_start = 1.0 - self.serrations.span_fraction.clamp(0.0, 1.0);
        let span_mask = smooth_window(eta, treated_span_start, 1.0);
        let phase_decoherence = 0.65 + 0.35 * phase.sin().abs();

        let leading = match self.edge_treatment {
            EdgeTreatment::LeadingEdgeSerrations | EdgeTreatment::LeadingAndTrailingSerrations => {
                0.30 * octave_like_band(frequency_hz, 800.0, 4_500.0)
            }
            _ => 0.0,
        };
        let trailing = match self.edge_treatment {
            EdgeTreatment::TrailingEdgeSerrations | EdgeTreatment::LeadingAndTrailingSerrations => {
                0.36 * octave_like_band(frequency_hz, 1_200.0, 6_000.0)
            }
            _ => 0.0,
        };

        serration * span_mask * phase_decoherence * (leading + trailing)
    }
}

/// Convention used for the pressure magnitudes in a [`PropellerSpectrum`].
///
/// The source model does not retain a complex observer phasor.  It produces
/// positive peak-pressure magnitudes at the one-metre reference distance.
/// Consumers that need coherent summation must obtain phase and directivity
/// from a resolved propagation model rather than treating these magnitudes as
/// complex pressures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectrumPressureConvention {
    /// Positive peak-pressure magnitude at the source model's one-metre
    /// reference distance.
    PeakMagnitudeAtOneMetre,
}

/// Directivity convention attached to a source-model spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectrumDirectivityConvention {
    /// The magnitude has unit-mean-power isotropic reference normalization;
    /// it is not an observer-direction prediction.
    UnitMeanPowerIsotropicReference,
}

/// Phase convention attached to a source-model spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectrumPhaseConvention {
    /// The source model reports magnitudes only.  There is no observer phase
    /// to use for coherent cross-harmonic or multi-rotor summation.
    MagnitudeOnly,
}

/// Explicit fidelity and physical conventions for a [`PropellerSpectrum`].
///
/// This provenance prevents source-modifier spectra from being mislabelled as
/// resolved wake, observer-directivity, or complex-phase results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerSpectrumProvenance {
    pub evaluation_mode: EvaluationMode,
    pub pressure: SpectrumPressureConvention,
    pub directivity: SpectrumDirectivityConvention,
    pub phase: SpectrumPhaseConvention,
}

impl PropellerSpectrumProvenance {
    fn source_modifier(evaluation_mode: EvaluationMode) -> Self {
        Self {
            evaluation_mode,
            pressure: SpectrumPressureConvention::PeakMagnitudeAtOneMetre,
            directivity: SpectrumDirectivityConvention::UnitMeanPowerIsotropicReference,
            phase: SpectrumPhaseConvention::MagnitudeOnly,
        }
    }
}

impl Default for PropellerSpectrumProvenance {
    fn default() -> Self {
        // Preserve serde compatibility with older magnitude-only spectra.
        Self::source_modifier(EvaluationMode::FastSurrogate)
    }
}

/// Harmonic pressure result for one BPF multiple.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarmonicNoise {
    pub harmonic: usize,
    pub frequency_hz: f64,
    /// Unmodified positive peak-pressure magnitude at the source reference
    /// distance. This is not a complex phasor.
    pub base_pressure_pa: f64,
    /// Modified positive peak-pressure magnitude at the source reference
    /// distance. This is not a complex phasor.
    pub pressure_pa: f64,
    pub amplitude_reduction: f64,
}

/// Full propeller source spectrum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropellerSpectrum {
    pub harmonics: Vec<HarmonicNoise>,
    /// Fidelity and physical conventions under which the magnitudes were
    /// produced. Defaults when reading historical serialized spectra.
    #[serde(default)]
    pub provenance: PropellerSpectrumProvenance,
}

/// Why a spectrum cannot be used for a physical band metric.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropellerSpectrumError {
    InvalidBand,
    InvalidHarmonic { harmonic: usize },
}

impl PropellerSpectrum {
    fn new(evaluation_mode: EvaluationMode, harmonics: Vec<HarmonicNoise>) -> Self {
        Self {
            harmonics,
            provenance: PropellerSpectrumProvenance::source_modifier(evaluation_mode),
        }
    }

    /// Build a source-modifier spectrum from harmonics using the historical
    /// fast-surrogate provenance. Prefer [`PropellerDesign::spectrum`] for
    /// production evaluations.
    pub fn from_harmonics(harmonics: Vec<HarmonicNoise>) -> Self {
        Self::new(EvaluationMode::FastSurrogate, harmonics)
    }

    fn validate_band(&self, lo_hz: f64, hi_hz: f64) -> Result<(), PropellerSpectrumError> {
        if !lo_hz.is_finite() || !hi_hz.is_finite() || lo_hz < 0.0 || hi_hz < lo_hz {
            return Err(PropellerSpectrumError::InvalidBand);
        }
        for harmonic in &self.harmonics {
            if harmonic.harmonic == 0
                || !harmonic.frequency_hz.is_finite()
                || harmonic.frequency_hz <= 0.0
                || !harmonic.base_pressure_pa.is_finite()
                || harmonic.base_pressure_pa < 0.0
                || !harmonic.pressure_pa.is_finite()
                || harmonic.pressure_pa < 0.0
                || !harmonic.amplitude_reduction.is_finite()
                || !(0.0..=1.0).contains(&harmonic.amplitude_reduction)
            {
                return Err(PropellerSpectrumError::InvalidHarmonic {
                    harmonic: harmonic.harmonic,
                });
            }
        }
        Ok(())
    }

    /// A-weighted time-mean-square pressure in a frequency range.
    ///
    /// Harmonic pressures are positive peak magnitudes, so each contributes
    /// `p_peak² / 2` before applying the squared A-weighting amplitude factor.
    pub fn try_a_weighted_mean_square_pressure_in_band(
        &self,
        lo_hz: f64,
        hi_hz: f64,
    ) -> Result<f64, PropellerSpectrumError> {
        self.validate_band(lo_hz, hi_hz)?;
        Ok(self
            .harmonics
            .iter()
            .filter(|h| h.frequency_hz >= lo_hz && h.frequency_hz <= hi_hz)
            .map(|h| {
                let aw = sonium_fem::physics::acoustics::pressure::sound_metrics::a_weighting(
                    h.frequency_hz,
                );
                0.5 * h.pressure_pa * h.pressure_pa * aw * aw
            })
            .sum())
    }

    /// A-weighted time-mean-square pressure in a frequency range.
    ///
    /// The infallible compatibility API returns `NaN` for malformed bands or
    /// nonphysical public harmonic data; use the checked variant where an
    /// invalid input must be reported to a caller.
    pub fn a_weighted_pressure_power_in_band(&self, lo_hz: f64, hi_hz: f64) -> f64 {
        self.try_a_weighted_mean_square_pressure_in_band(lo_hz, hi_hz)
            .unwrap_or(f64::NAN)
    }

    /// A-weighted sound-pressure level in dB re 20 µPa for the selected band.
    /// A silent valid spectrum yields negative infinity.
    pub fn try_a_weighted_spl_in_band(
        &self,
        lo_hz: f64,
        hi_hz: f64,
    ) -> Result<f64, PropellerSpectrumError> {
        let mean_square = self.try_a_weighted_mean_square_pressure_in_band(lo_hz, hi_hz)?;
        Ok(10.0 * (mean_square / 20.0e-6_f64.powi(2)).log10())
    }

    /// Band reduction based on A-weighted time-mean-square pressure.
    ///
    /// The infallible compatibility API returns `NaN` when either spectrum or
    /// band is malformed; use [`Self::try_band_reduction_db_against`] when
    /// evaluating public data.
    pub fn band_reduction_db_against(&self, baseline: &Self, lo_hz: f64, hi_hz: f64) -> f64 {
        self.try_band_reduction_db_against(baseline, lo_hz, hi_hz)
            .unwrap_or(f64::NAN)
    }

    /// Checked variant of [`Self::band_reduction_db_against`].
    pub fn try_band_reduction_db_against(
        &self,
        baseline: &Self,
        lo_hz: f64,
        hi_hz: f64,
    ) -> Result<f64, PropellerSpectrumError> {
        let p = self
            .try_a_weighted_mean_square_pressure_in_band(lo_hz, hi_hz)?
            .max(1e-30);
        let p0 = baseline
            .try_a_weighted_mean_square_pressure_in_band(lo_hz, hi_hz)?
            .max(1e-30);
        Ok(10.0 * (p0 / p).log10())
    }
}

fn octave_like_band(frequency_hz: f64, lo_hz: f64, hi_hz: f64) -> f64 {
    if frequency_hz <= 0.0 {
        return 0.0;
    }
    let f = frequency_hz.log10();
    let lo = lo_hz.log10();
    let hi = hi_hz.log10();
    let rise = smooth_window(f, (lo_hz * 0.75).log10(), lo);
    let fall = 1.0 - smooth_window(f, hi, (hi_hz * 1.4).log10());
    rise * fall
}

fn high_frequency_ramp(frequency_hz: f64, knee_hz: f64) -> f64 {
    let x = (frequency_hz / knee_hz).max(0.0);
    x / (1.0 + x)
}

fn smooth_window(x: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo || x <= lo {
        return 0.0;
    }
    if x >= hi {
        return 1.0;
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toroidal_loop_reduces_sensitive_band() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let baseline = PropellerDesign::conventional(source.clone())
            .spectrum(12, EvaluationMode::FastSurrogate);
        let toroidal = PropellerDesign {
            source,
            planform: PropellerPlanform::ToroidalLoop,
            edge_treatment: EdgeTreatment::Smooth,
            serrations: SerrationGeometry::default(),
        }
        .spectrum(12, EvaluationMode::FastSurrogate);

        assert!(toroidal.band_reduction_db_against(&baseline, 1_000.0, 5_000.0) > 1.0);
    }

    #[test]
    fn combined_design_is_quieter_than_conventional() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let baseline = PropellerDesign::conventional(source.clone())
            .spectrum(12, EvaluationMode::ResearchControl);
        let combined = PropellerDesign::toroidal_serrated(source)
            .spectrum(12, EvaluationMode::ResearchControl);

        assert!(combined.band_reduction_db_against(&baseline, 1_000.0, 5_000.0) > 3.0);
    }

    #[test]
    fn reference_reductions_remain_inside_empirical_sensitivity_priors() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let baseline = PropellerDesign::conventional(source.clone())
            .spectrum(20, EvaluationMode::ResearchControl);
        let toroidal = PropellerDesign {
            source: source.clone(),
            planform: PropellerPlanform::ToroidalLoop,
            edge_treatment: EdgeTreatment::Smooth,
            serrations: SerrationGeometry::default(),
        }
        .spectrum(20, EvaluationMode::ResearchControl);
        let combined = PropellerDesign::toroidal_serrated(source)
            .spectrum(20, EvaluationMode::ResearchControl);

        let toroidal_db = toroidal.band_reduction_db_against(&baseline, 1_000.0, 5_000.0);
        let combined_db = combined.band_reduction_db_against(&baseline, 1_000.0, 5_000.0);
        assert!(TOROIDAL_REDUCTION_PRIOR_DB.contains(toroidal_db));
        assert!(COMBINED_REDUCTION_PRIOR_DB.contains(combined_db));
    }

    #[test]
    fn band_reduction_is_positive_for_a_quieter_spectrum() {
        let baseline = PropellerSpectrum::from_harmonics(vec![HarmonicNoise {
            harmonic: 1,
            frequency_hz: 2_000.0,
            base_pressure_pa: 2.0,
            pressure_pa: 2.0,
            amplitude_reduction: 1.0,
        }]);
        let quieter = PropellerSpectrum::from_harmonics(vec![HarmonicNoise {
            harmonic: 1,
            frequency_hz: 2_000.0,
            base_pressure_pa: 2.0,
            pressure_pa: 1.0,
            amplitude_reduction: 0.5,
        }]);
        assert!(
            (quieter.band_reduction_db_against(&baseline, 1_000.0, 5_000.0)
                - 20.0 * 2.0_f64.log10())
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn peak_pressure_band_metrics_use_rms_and_publish_source_provenance() {
        let spectrum = PropellerDesign::conventional(PropellerSource::drone_5inch(18_000.0))
            .spectrum(1, EvaluationMode::ResearchControl);
        let harmonic = &spectrum.harmonics[0];
        let aw = sonium_fem::physics::acoustics::pressure::sound_metrics::a_weighting(
            harmonic.frequency_hz,
        );
        let expected_mean_square = 0.5 * harmonic.pressure_pa.powi(2) * aw.powi(2);
        let mean_square = spectrum
            .try_a_weighted_mean_square_pressure_in_band(
                harmonic.frequency_hz,
                harmonic.frequency_hz,
            )
            .unwrap();
        assert!(
            (mean_square - expected_mean_square).abs()
                <= expected_mean_square.abs().max(f64::MIN_POSITIVE) * 1.0e-12
        );

        let expected_spl = 10.0 * (expected_mean_square / 20.0e-6_f64.powi(2)).log10();
        assert!(
            (spectrum
                .try_a_weighted_spl_in_band(harmonic.frequency_hz, harmonic.frequency_hz)
                .unwrap()
                - expected_spl)
                .abs()
                <= 1.0e-12
        );
        assert_eq!(
            spectrum.provenance.evaluation_mode,
            EvaluationMode::ResearchControl
        );
        assert_eq!(
            spectrum.provenance.pressure,
            SpectrumPressureConvention::PeakMagnitudeAtOneMetre
        );
        assert_eq!(
            spectrum.provenance.directivity,
            SpectrumDirectivityConvention::UnitMeanPowerIsotropicReference
        );
        assert_eq!(
            spectrum.provenance.phase,
            SpectrumPhaseConvention::MagnitudeOnly
        );
    }

    #[test]
    fn checked_band_metrics_reject_malformed_public_spectra() {
        let malformed = PropellerSpectrum::from_harmonics(vec![HarmonicNoise {
            harmonic: 1,
            frequency_hz: 2_000.0,
            base_pressure_pa: 1.0,
            pressure_pa: f64::NAN,
            amplitude_reduction: 1.0,
        }]);
        assert_eq!(
            malformed.try_a_weighted_mean_square_pressure_in_band(1_000.0, 5_000.0),
            Err(PropellerSpectrumError::InvalidHarmonic { harmonic: 1 })
        );
        assert!(
            malformed
                .a_weighted_pressure_power_in_band(1_000.0, 5_000.0)
                .is_nan()
        );

        let valid = PropellerSpectrum::from_harmonics(vec![]);
        assert_eq!(
            valid.try_band_reduction_db_against(&valid, 5_000.0, 1_000.0),
            Err(PropellerSpectrumError::InvalidBand)
        );
    }
}
