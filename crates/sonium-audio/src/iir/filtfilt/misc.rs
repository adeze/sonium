use crate::common::traits::FilterFloat;
use crate::iir::{BiquadCoefficients, Peq};

/// Extract [`BiquadCoefficients`] from a [`Peq`] chain, folding per-stage gains
/// into the feedforward coefficients.
///
/// This bridges the `Peq`-based filter design functions
/// ([`peq_butterworth_lowpass`](crate::peq_butterworth_lowpass), etc.)
/// with the coefficient-based [`sosfilt`] / [`filtfilt`] functions.
pub fn peq_to_coefficients<T: FilterFloat>(peq: &Peq<T>) -> Vec<BiquadCoefficients<T>> {
    peq.iter()
        .map(|(gain, biquad): &(T, crate::iir::Biquad<T>)| {
            let c = biquad.coefficients();
            BiquadCoefficients {
                b0: c.b0 * *gain,
                b1: c.b1 * *gain,
                b2: c.b2 * *gain,
                a1: c.a1,
                a2: c.a2,
            }
        })
        .collect()
}
