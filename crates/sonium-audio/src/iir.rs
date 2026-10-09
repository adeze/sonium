//! IIR filters: biquads, parametric EQ, SVF, Kautz systems, IIR crossovers.

mod biquad;
mod biquad_bank;
pub mod kautz;
mod peq;
pub mod warped_biquad;

/// Selectable second- and fourth-order IIR crossovers.
pub mod crossover;
/// Forward-reverse (zero-phase) filtering for offline signal processing.
pub mod filtfilt;
/// Linkwitz-Riley 4th-order IIR crossover.
pub mod lr4_crossover;
/// Linkwitz-Riley 8th-order IIR crossover (48 dB/octave).
pub mod lr8_crossover;
pub(crate) mod phase_smooth;
/// Zero-Delay Feedback State Variable Filter (Zavalishin TPT topology).
pub mod svf;

pub use biquad::{Biquad, BiquadCoefficients, BiquadFilterType, Peq};
pub use biquad_bank::BiquadBank;
pub use crossover::{Crossover, CrossoverFamily, CrossoverOrder};
pub use kautz::{
    BandLimit, FitConfig, FitDiagnostics, FitStatus, FitWeights, GuardGrid, GuardReport,
    KAUTZ_CORRECTION_FORMAT, KautzBankSpec, KautzCorrection, KautzError, KautzResult,
    KautzSectionSpec, MAX_POLE_RADIUS, MAX_SUPPORTED_RATE_HZ, MIN_Q_FACTOR, MIN_SUPPORTED_RATE_HZ,
    NormalizationPolicy, check_bounds, fit_correction, parse_bank_spec,
};
pub use kautz::{KautzFilter, KautzSection};
pub use lr4_crossover::{CROSSOVER_PRESETS, Lr4Crossover, MultibandLr4Crossover};
pub use lr8_crossover::{Lr8Crossover, MultibandLr8Crossover};
pub use peq::{
    FilterRow, compute_peq_response, compute_peq_response_into, peq_allpass,
    peq_butterworth_highpass, peq_butterworth_lowpass, peq_butterworth_q, peq_equal,
    peq_format_apo, peq_format_aupreset, peq_format_camilladsp, peq_format_easyeffects,
    peq_format_pipewire, peq_format_rme_channel, peq_format_rme_room, peq_format_roon,
    peq_format_wavelet, peq_linkwitzriley_highpass, peq_linkwitzriley_lowpass, peq_linkwitzriley_q,
    peq_loudness_gain, peq_preamp_gain, peq_preamp_gain_max, peq_print, peq_spl, peq_spl_into,
};
pub use phase_smooth::{interpolate_phase_complex, smooth_phase_via_group_delay, unwrap_phase};
pub use svf::{SvfFilter, SvfFilterType};
pub use warped_biquad::{WarpedBiquad, bark_lambda, unwarp_frequency, warp_frequency};

// Shared numeric helpers re-exported for downstream generic code.
/// Flush-to-zero guards for denormal-free real-time processing.
pub use crate::common::denormals;
pub use crate::common::{
    AUDIBLE_MAX_FREQ, AUDIBLE_MIN_FREQ, DEFAULT_Q_HIGH_LOW_PASS, DEFAULT_Q_HIGH_LOW_SHELF,
    FilterFloat, FirError, FirResult, IirError, Result, bw2q, logspace, q2bw,
};

/// 32-bit biquad filter.
pub type BiquadF32 = Biquad<f32>;
/// 32-bit biquad coefficients.
pub type BiquadCoefficientsF32 = BiquadCoefficients<f32>;
/// 32-bit parametric EQ chain.
pub type PeqF32 = Peq<f32>;
/// 32-bit biquad bank.
pub type BiquadBankF32 = BiquadBank<f32>;
/// 32-bit SVF filter.
pub type SvfFilterF32 = SvfFilter<f32>;
/// 32-bit LR4 crossover.
pub type Lr4CrossoverF32 = Lr4Crossover<f32>;
/// 32-bit multiband LR4 crossover.
pub type MultibandLr4CrossoverF32 = MultibandLr4Crossover<f32>;
/// 32-bit LR8 crossover.
pub type Lr8CrossoverF32 = Lr8Crossover<f32>;
/// 32-bit multiband LR8 crossover.
pub type MultibandLr8CrossoverF32 = MultibandLr8Crossover<f32>;
