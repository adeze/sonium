use super::Fir;
use crate::common::traits::FilterFloat;

/// Type alias for a collection of weighted FIR filters
pub type FirBank<T = f64> = Vec<(T, Fir<T>)>;

/// Compute the combined FIR bank response (in dB) on a given frequency grid.
///
/// # Arguments
/// * `freqs` - Frequency points for evaluation (Hz)
/// * `fir_bank` - Collection of weighted FIR filters
///
/// # Returns
/// Frequency response in dB SPL at the specified frequency points
pub fn compute_fir_bank_response<T: FilterFloat>(freqs: &[T], fir_bank: &FirBank<T>) -> Vec<T> {
    let mut response = vec![T::zero(); freqs.len()];
    let mut filter_scratch = vec![T::zero(); freqs.len()];
    compute_fir_bank_response_into(freqs, fir_bank, &mut response, &mut filter_scratch);
    response
}

/// Compute a combined FIR bank response into caller-owned reusable buffers.
///
/// `response` and `filter_scratch` must both have the same length as `freqs`.
/// Each filter is evaluated with [`Fir::np_log_result_into`], so no per-filter
/// `Array1` is allocated in the loop; keeping both buffers outside this
/// function lets callers reuse them across evaluations (mirrors
/// `compute_peq_response_into` for IIR banks).
pub fn compute_fir_bank_response_into<T: FilterFloat>(
    freqs: &[T],
    fir_bank: &FirBank<T>,
    response: &mut [T],
    filter_scratch: &mut [T],
) {
    assert_eq!(
        response.len(),
        freqs.len(),
        "response length must match frequency grid"
    );
    assert_eq!(
        filter_scratch.len(),
        freqs.len(),
        "filter scratch length must match frequency grid"
    );

    response.fill(T::zero());

    for (weight, filter) in fir_bank {
        filter.np_log_result_into(freqs, filter_scratch);
        for (sum, value) in response.iter_mut().zip(filter_scratch.iter()) {
            *sum += *value * *weight;
        }
    }
}
