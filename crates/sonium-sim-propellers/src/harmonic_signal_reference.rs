//! Independent harmonic time-domain, RMS, band-power, and SPL reference.
//!
//! The evaluator synthesizes caller-supplied integer harmonics over an integer
//! number of fundamental cycles. It compares sampled mean-square pressure with
//! the orthogonal analytical result, reports band-limited RMS/SPL, and rejects
//! sampling plans that could alias the squared signal. This is a deterministic
//! signal-contract reference, not a measured-data, propagation, or FW-H model.

use std::fmt;

/// One signed harmonic pressure component before directivity scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicSignalComponent {
    /// Positive integer harmonic number.
    pub harmonic: usize,
    /// Positive peak pressure magnitude [Pa].
    pub peak_pressure_pa: f64,
    /// Component phase [rad].
    pub phase_rad: f64,
    /// Signed directivity/observer gain [-].
    pub directivity_gain: f64,
}

/// Controls deterministic harmonic synthesis and reference comparisons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicSignalOptions {
    /// Fundamental frequency [Hz].
    pub fundamental_frequency_hz: f64,
    /// Number of complete fundamental cycles synthesized.
    pub cycles: usize,
    /// Uniform samples per fundamental cycle.
    pub samples_per_cycle: usize,
    /// SPL reference pressure [Pa].
    pub reference_pressure_pa: f64,
    /// Inclusive lower band edge [Hz].
    pub band_low_hz: f64,
    /// Inclusive upper band edge [Hz].
    pub band_high_hz: f64,
    /// Maximum analytical-versus-sampled mean-square mismatch [Pa²].
    pub closure_tolerance_pa2: f64,
}

impl Default for HarmonicSignalOptions {
    fn default() -> Self {
        Self {
            fundamental_frequency_hz: 600.0,
            cycles: 8,
            samples_per_cycle: 256,
            reference_pressure_pa: 20.0e-6,
            band_low_hz: 1_000.0,
            band_high_hz: 5_000.0,
            closure_tolerance_pa2: 1.0e-12,
        }
    }
}

/// Per-component frequency and RMS evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicSignalComponentResult {
    /// Harmonic number.
    pub harmonic: usize,
    /// Component frequency [Hz].
    pub frequency_hz: f64,
    /// Signed effective peak pressure after directivity [Pa].
    pub effective_peak_pressure_pa: f64,
    /// Component phase [rad].
    pub phase_rad: f64,
    /// Component RMS pressure [Pa].
    pub rms_pressure_pa: f64,
    /// Whether this component lies in the requested band.
    pub in_band: bool,
}

/// Analytical and sampled harmonic aggregation evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicSignalResult {
    /// Component results in caller order.
    pub components: Vec<HarmonicSignalComponentResult>,
    /// Synthesis sample rate [Hz].
    pub sample_rate_hz: f64,
    /// Number of time samples synthesized.
    pub sample_count: usize,
    /// Analytical total mean-square pressure [Pa²].
    pub analytical_total_mean_square_pa2: f64,
    /// Sampled total mean-square pressure [Pa²].
    pub sampled_total_mean_square_pa2: f64,
    /// Analytical band mean-square pressure [Pa²].
    pub analytical_band_mean_square_pa2: f64,
    /// Sampled band mean-square pressure [Pa²].
    pub sampled_band_mean_square_pa2: f64,
    /// Total RMS pressure [Pa].
    pub total_rms_pressure_pa: f64,
    /// Band RMS pressure, absent when no component lies in the band [Pa].
    pub band_rms_pressure_pa: Option<f64>,
    /// Total SPL, absent for a zero signal [dB re reference pressure].
    pub total_spl_db: Option<f64>,
    /// Band SPL, absent when the band has no component or zero pressure.
    pub band_spl_db: Option<f64>,
    /// Total mean-square closure residual [Pa²].
    pub total_closure_error_pa2: f64,
    /// Band mean-square closure residual [Pa²].
    pub band_closure_error_pa2: f64,
    /// Number of components selected by the band.
    pub band_component_count: usize,
    /// Whether both analytical/sample closures meet the configured tolerance.
    pub closure_passed: bool,
}

impl HarmonicSignalResult {
    /// Returns whether all retained signal and closure values are finite.
    pub fn is_finite(&self) -> bool {
        self.sample_rate_hz.is_finite()
            && self.sample_rate_hz > 0.0
            && self.sample_count > 0
            && self.components.iter().all(|component| {
                component.frequency_hz.is_finite()
                    && component.frequency_hz > 0.0
                    && component.effective_peak_pressure_pa.is_finite()
                    && component.phase_rad.is_finite()
                    && component.rms_pressure_pa.is_finite()
                    && component.rms_pressure_pa >= 0.0
            })
            && self.analytical_total_mean_square_pa2.is_finite()
            && self.analytical_total_mean_square_pa2 >= 0.0
            && self.sampled_total_mean_square_pa2.is_finite()
            && self.sampled_total_mean_square_pa2 >= 0.0
            && self.analytical_band_mean_square_pa2.is_finite()
            && self.analytical_band_mean_square_pa2 >= 0.0
            && self.sampled_band_mean_square_pa2.is_finite()
            && self.sampled_band_mean_square_pa2 >= 0.0
            && self.total_rms_pressure_pa.is_finite()
            && self.total_rms_pressure_pa >= 0.0
            && self
                .band_rms_pressure_pa
                .map(|value| value.is_finite() && value >= 0.0)
                .unwrap_or(true)
            && self
                .total_spl_db
                .map(|value| value.is_finite())
                .unwrap_or(true)
            && self
                .band_spl_db
                .map(|value| value.is_finite())
                .unwrap_or(true)
            && self.total_closure_error_pa2.is_finite()
            && self.band_closure_error_pa2.is_finite()
    }
}

/// Failure-closed errors for harmonic signal aggregation.
#[derive(Debug, Clone, PartialEq)]
pub enum HarmonicSignalError {
    /// No harmonic components were supplied.
    EmptyComponents,
    /// An options field is outside its admitted domain.
    InvalidOptions {
        /// Options field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A component is malformed or not strictly ordered.
    InvalidComponent {
        /// Component index.
        component_index: usize,
        /// Component field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The chosen sampling plan can alias the squared harmonic signal.
    AliasingRisk {
        /// Highest supplied harmonic number.
        maximum_harmonic: usize,
        /// Samples per fundamental cycle.
        samples_per_cycle: usize,
    },
    /// The requested sample count overflows `usize`.
    SampleCountOverflow,
}

impl fmt::Display for HarmonicSignalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyComponents => write!(formatter, "at least one harmonic is required"),
            Self::InvalidOptions { field, requirement } => {
                write!(formatter, "invalid harmonic option {field}: {requirement}")
            }
            Self::InvalidComponent {
                component_index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid harmonic {component_index} field {field}: {requirement}"
            ),
            Self::AliasingRisk {
                maximum_harmonic,
                samples_per_cycle,
            } => write!(
                formatter,
                "harmonic {maximum_harmonic} requires more than half the {samples_per_cycle} samples/cycle"
            ),
            Self::SampleCountOverflow => write!(formatter, "harmonic sample count overflows usize"),
        }
    }
}

impl std::error::Error for HarmonicSignalError {}

/// Synthesize and aggregate an integer harmonic signal with analytical closure.
pub fn evaluate_harmonic_signal_reference(
    components: &[HarmonicSignalComponent],
    options: HarmonicSignalOptions,
) -> Result<HarmonicSignalResult, HarmonicSignalError> {
    validate_options(&options)?;
    if components.is_empty() {
        return Err(HarmonicSignalError::EmptyComponents);
    }
    validate_components(components)?;
    let maximum_harmonic = components
        .last()
        .expect("non-empty components validated")
        .harmonic;
    if maximum_harmonic
        .checked_mul(2)
        .is_none_or(|value| value >= options.samples_per_cycle)
    {
        return Err(HarmonicSignalError::AliasingRisk {
            maximum_harmonic,
            samples_per_cycle: options.samples_per_cycle,
        });
    }
    let sample_count = options
        .cycles
        .checked_mul(options.samples_per_cycle)
        .ok_or(HarmonicSignalError::SampleCountOverflow)?;
    let sample_rate_hz = options.fundamental_frequency_hz * options.samples_per_cycle as f64;
    let components = components
        .iter()
        .map(|component| {
            let frequency_hz = options.fundamental_frequency_hz * component.harmonic as f64;
            let effective_peak_pressure_pa =
                component.peak_pressure_pa * component.directivity_gain;
            HarmonicSignalComponentResult {
                harmonic: component.harmonic,
                frequency_hz,
                effective_peak_pressure_pa,
                phase_rad: component.phase_rad,
                rms_pressure_pa: effective_peak_pressure_pa.abs() / 2.0_f64.sqrt(),
                in_band: frequency_hz >= options.band_low_hz
                    && frequency_hz <= options.band_high_hz,
            }
        })
        .collect::<Vec<_>>();
    let analytical_total_mean_square_pa2 = components
        .iter()
        .map(|component| component.effective_peak_pressure_pa.powi(2) / 2.0)
        .sum::<f64>();
    let analytical_band_mean_square_pa2 = components
        .iter()
        .filter(|component| component.in_band)
        .map(|component| component.effective_peak_pressure_pa.powi(2) / 2.0)
        .sum::<f64>();
    let band_component_count = components
        .iter()
        .filter(|component| component.in_band)
        .count();

    let mut sampled_total_sum = 0.0;
    let mut sampled_band_sum = 0.0;
    for sample_index in 0..sample_count {
        let phase_cycle =
            2.0 * std::f64::consts::PI * sample_index as f64 / options.samples_per_cycle as f64;
        let mut total_pressure_pa = 0.0;
        let mut band_pressure_pa = 0.0;
        for component in &components {
            let pressure_pa = component.effective_peak_pressure_pa
                * (phase_cycle * component.harmonic as f64 + component.phase_rad).cos();
            total_pressure_pa += pressure_pa;
            if component.in_band {
                band_pressure_pa += pressure_pa;
            }
        }
        sampled_total_sum += total_pressure_pa.powi(2);
        sampled_band_sum += band_pressure_pa.powi(2);
    }
    let sampled_total_mean_square_pa2 = sampled_total_sum / sample_count as f64;
    let sampled_band_mean_square_pa2 = sampled_band_sum / sample_count as f64;
    let total_rms_pressure_pa = analytical_total_mean_square_pa2.sqrt();
    let band_rms_pressure_pa =
        (band_component_count > 0).then_some(analytical_band_mean_square_pa2.sqrt());
    let total_spl_db = spl_db(total_rms_pressure_pa, options.reference_pressure_pa);
    let band_spl_db =
        band_rms_pressure_pa.and_then(|value| spl_db(value, options.reference_pressure_pa));
    let total_closure_error_pa2 = sampled_total_mean_square_pa2 - analytical_total_mean_square_pa2;
    let band_closure_error_pa2 = sampled_band_mean_square_pa2 - analytical_band_mean_square_pa2;
    let closure_passed = total_closure_error_pa2.abs() <= options.closure_tolerance_pa2
        && band_closure_error_pa2.abs() <= options.closure_tolerance_pa2;
    let result = HarmonicSignalResult {
        components,
        sample_rate_hz,
        sample_count,
        analytical_total_mean_square_pa2,
        sampled_total_mean_square_pa2,
        analytical_band_mean_square_pa2,
        sampled_band_mean_square_pa2,
        total_rms_pressure_pa,
        band_rms_pressure_pa,
        total_spl_db,
        band_spl_db,
        total_closure_error_pa2,
        band_closure_error_pa2,
        band_component_count,
        closure_passed,
    };
    debug_assert!(result.is_finite());
    Ok(result)
}

fn spl_db(rms_pressure_pa: f64, reference_pressure_pa: f64) -> Option<f64> {
    (rms_pressure_pa > 0.0).then(|| 20.0 * (rms_pressure_pa / reference_pressure_pa).log10())
}

fn validate_options(options: &HarmonicSignalOptions) -> Result<(), HarmonicSignalError> {
    if !options.fundamental_frequency_hz.is_finite() || options.fundamental_frequency_hz <= 0.0 {
        return Err(HarmonicSignalError::InvalidOptions {
            field: "fundamental_frequency_hz",
            requirement: "finite and > 0",
        });
    }
    if options.cycles == 0 {
        return Err(HarmonicSignalError::InvalidOptions {
            field: "cycles",
            requirement: "> 0",
        });
    }
    if options.samples_per_cycle < 8 {
        return Err(HarmonicSignalError::InvalidOptions {
            field: "samples_per_cycle",
            requirement: ">= 8",
        });
    }
    if !options.reference_pressure_pa.is_finite() || options.reference_pressure_pa <= 0.0 {
        return Err(HarmonicSignalError::InvalidOptions {
            field: "reference_pressure_pa",
            requirement: "finite and > 0",
        });
    }
    if !options.band_low_hz.is_finite()
        || !options.band_high_hz.is_finite()
        || options.band_low_hz < 0.0
        || options.band_high_hz <= options.band_low_hz
    {
        return Err(HarmonicSignalError::InvalidOptions {
            field: "band_low_hz/band_high_hz",
            requirement: "finite with 0 <= low < high",
        });
    }
    if !options.closure_tolerance_pa2.is_finite() || options.closure_tolerance_pa2 <= 0.0 {
        return Err(HarmonicSignalError::InvalidOptions {
            field: "closure_tolerance_pa2",
            requirement: "finite and > 0",
        });
    }
    Ok(())
}

fn validate_components(components: &[HarmonicSignalComponent]) -> Result<(), HarmonicSignalError> {
    let mut previous_harmonic = 0;
    for (component_index, component) in components.iter().enumerate() {
        if component.harmonic == 0 || component.harmonic <= previous_harmonic {
            return Err(HarmonicSignalError::InvalidComponent {
                component_index,
                field: "harmonic",
                requirement: "positive and strictly increasing",
            });
        }
        if !component.peak_pressure_pa.is_finite() || component.peak_pressure_pa <= 0.0 {
            return Err(HarmonicSignalError::InvalidComponent {
                component_index,
                field: "peak_pressure_pa",
                requirement: "finite and > 0",
            });
        }
        if !component.phase_rad.is_finite() {
            return Err(HarmonicSignalError::InvalidComponent {
                component_index,
                field: "phase_rad",
                requirement: "finite",
            });
        }
        if !component.directivity_gain.is_finite() {
            return Err(HarmonicSignalError::InvalidComponent {
                component_index,
                field: "directivity_gain",
                requirement: "finite",
            });
        }
        previous_harmonic = component.harmonic;
    }
    Ok(())
}
