//! DCT-II/III and DST-II/III transforms with explicit normalization.
//!
//! The forward DCT-II and DST-II are unnormalised. Their matching III
//! transforms include the inverse scale, so `dct3(dct2(x))` and
//! `dst3(dst2(x))` reconstruct the input.
//!
//! Two tiers: the free functions below are direct O(N^2) summations kept as
//! the readable reference (and the fast path for tiny lengths). The
//! [`DctPlan`]/[`DstPlan`] types dispatch lengths above
//! [`DIRECT_THRESHOLD`] to an O(N log N) FFT kernel ([`FftDctKernel`]): one
//! length-N complex FFT plus O(N) pre/post shuffles.

use num_complex::Complex;
use std::sync::Arc;

use crate::FftDirection;
use crate::FftPlan;
use crate::cpu::planner::CpuFftPlanner;
use crate::scalar::FftScalar;
use crate::sincos::fill_sincos;

/// Largest length that uses direct O(N^2) summation inside the plans.
///
/// For tiny transforms the FFT machinery (reorder, plan dispatch, twiddle
/// multiply) costs more than the naive loops, and the direct path allocates
/// nothing. Larger lengths go through the FFT kernel. The free functions
/// below ([`dct2`], [`dct3`], [`dst2`], [`dst3`] and their adjoints) always
/// use the direct summation regardless of length, so treat this constant as
/// the application guard: for `n > DIRECT_THRESHOLD` prefer [`DctPlan`] /
/// [`DstPlan`], which select the O(N log N) path automatically.
pub const DIRECT_THRESHOLD: usize = 64;

/// Unnormalised DCT-II:
/// `X[k] = sum_n x[n] cos(pi/N * (n+1/2) * k)`.
///
/// # Cost
///
/// Direct O(N^2) summation at every length. For `n > DIRECT_THRESHOLD`
/// prefer [`DctPlan::forward`], which routes through the O(N log N) FFT
/// kernel with identical output up to floating-point rounding.
pub fn dct2<T: FftScalar>(input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "DCT input/output length mismatch"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = T::zero();
        for (index, &sample) in input.iter().enumerate() {
            let angle = std::f64::consts::PI * (index as f64 + 0.5) * k as f64 / n as f64;
            sum = sum + sample * T::from_f64(angle.cos()).unwrap();
        }
        *out = sum;
    }
}

/// Inverse of [`dct2`], using the standard DCT-III scale.
///
/// # Cost
///
/// Direct O(N^2) summation at every length. For `n > DIRECT_THRESHOLD`
/// prefer [`DctPlan::inverse`], which routes through the O(N log N) FFT
/// kernel with identical output up to floating-point rounding.
pub fn dct3<T: FftScalar>(input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "DCT input/output length mismatch"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    let scale = T::from_f64(2.0 / n as f64).unwrap();
    for (index, out) in output.iter_mut().enumerate() {
        let mut sum = input[0] * T::from_f64(0.5).unwrap();
        for (k, &value) in input.iter().enumerate().skip(1) {
            let angle = std::f64::consts::PI * k as f64 * (index as f64 + 0.5) / n as f64;
            sum = sum + value * T::from_f64(angle.cos()).unwrap();
        }
        *out = sum * scale;
    }
}

/// Euclidean adjoint of [`dct2`].
///
/// This is intentionally distinct from [`dct3`]: DCT-III is the normalized
/// inverse, while this routine applies the transpose of the unnormalised
/// DCT-II matrix. It is the operation required by reverse-mode spectral
/// objectives: `<dct2(x), y> = <x, dct2_adjoint(y)>`.
///
/// # Cost
///
/// Direct O(N^2) summation at every length. For `n > DIRECT_THRESHOLD`
/// prefer [`DctPlan::forward_adjoint`].
pub fn dct2_adjoint<T: FftScalar>(input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "DCT input/output length mismatch"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    for (index, out) in output.iter_mut().enumerate() {
        let mut sum = input[0];
        for (k, &value) in input.iter().enumerate().skip(1) {
            let angle = std::f64::consts::PI * k as f64 * (index as f64 + 0.5) / n as f64;
            sum = sum + value * T::from_f64(angle.cos()).unwrap();
        }
        *out = sum;
    }
}

/// Euclidean adjoint of [`dct3`].
///
/// DCT-III is the normalized inverse of DCT-II, so its adjoint is the
/// corresponding frequency-weighted DCT-II, not [`dct2`] itself.
///
/// # Cost
///
/// Direct O(N^2) summation at every length. For `n > DIRECT_THRESHOLD`
/// prefer [`DctPlan::inverse_adjoint`].
pub fn dct3_adjoint<T: FftScalar>(input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "DCT input/output length mismatch"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    let scale = T::from_f64(2.0 / n as f64).unwrap();
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = T::zero();
        for (index, &sample) in input.iter().enumerate() {
            let angle = std::f64::consts::PI * (index as f64 + 0.5) * k as f64 / n as f64;
            sum = sum + sample * T::from_f64(angle.cos()).unwrap();
        }
        let weight = if k == 0 {
            T::from_f64(0.5).unwrap()
        } else {
            T::one()
        };
        *out = sum * scale * weight;
    }
}

/// Unnormalised DST-II:
/// `X[k] = sum_n x[n] sin(pi/N * (n+1/2) * (k+1))`.
///
/// # Cost
///
/// Direct O(N^2) summation at every length. For `n > DIRECT_THRESHOLD`
/// prefer [`DstPlan::forward`], which routes through the O(N log N) FFT
/// kernel with identical output up to floating-point rounding.
pub fn dst2<T: FftScalar>(input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "DST input/output length mismatch"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = T::zero();
        for (index, &sample) in input.iter().enumerate() {
            let angle = std::f64::consts::PI * (index as f64 + 0.5) * (k as f64 + 1.0) / n as f64;
            sum = sum + sample * T::from_f64(angle.sin()).unwrap();
        }
        *out = sum;
    }
}

/// Inverse of [`dst2`], using the standard DST-III scale.
///
/// # Cost
///
/// Direct O(N^2) summation at every length. For `n > DIRECT_THRESHOLD`
/// prefer [`DstPlan::inverse`], which routes through the O(N log N) FFT
/// kernel with identical output up to floating-point rounding.
pub fn dst3<T: FftScalar>(input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "DST input/output length mismatch"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    for (index, out) in output.iter_mut().enumerate() {
        let mut sum = T::zero();
        for (k, &value) in input.iter().enumerate() {
            let angle = std::f64::consts::PI * (index as f64 + 0.5) * (k as f64 + 1.0) / n as f64;
            // The highest DST-II row has norm N rather than N/2; all
            // preceding rows have norm N/2. This is the DST-III weighting.
            let row_scale = if k + 1 == n {
                1.0 / n as f64
            } else {
                2.0 / n as f64
            };
            sum = sum + value * T::from_f64(angle.sin() * row_scale).unwrap();
        }
        *out = sum;
    }
}

/// Euclidean adjoint of [`dst2`].
///
/// DST-III is the normalized inverse of DST-II and therefore uses different
/// row weights. This routine instead applies the transpose of the
/// unnormalised DST-II matrix for gradient/adjoint calculations.
///
/// # Cost
///
/// Direct O(N^2) summation at every length. For `n > DIRECT_THRESHOLD`
/// prefer [`DstPlan::forward_adjoint`].
pub fn dst2_adjoint<T: FftScalar>(input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "DST input/output length mismatch"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    for (index, out) in output.iter_mut().enumerate() {
        let mut sum = T::zero();
        for (k, &value) in input.iter().enumerate() {
            let angle = std::f64::consts::PI * (index as f64 + 0.5) * (k as f64 + 1.0) / n as f64;
            sum = sum + value * T::from_f64(angle.sin()).unwrap();
        }
        *out = sum;
    }
}

/// Euclidean adjoint of [`dst3`].
///
/// # Cost
///
/// Direct O(N^2) summation at every length. For `n > DIRECT_THRESHOLD`
/// prefer [`DstPlan::inverse_adjoint`].
pub fn dst3_adjoint<T: FftScalar>(input: &[T], output: &mut [T]) {
    assert_eq!(
        input.len(),
        output.len(),
        "DST input/output length mismatch"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = T::zero();
        for (index, &sample) in input.iter().enumerate() {
            let angle = std::f64::consts::PI * (index as f64 + 0.5) * (k as f64 + 1.0) / n as f64;
            sum = sum + sample * T::from_f64(angle.sin()).unwrap();
        }
        let row_scale = if k + 1 == n {
            1.0 / n as f64
        } else {
            2.0 / n as f64
        };
        *out = sum * T::from_f64(row_scale).unwrap();
    }
}

/// O(N log N) cosine/sine kernel shared by every plan transform.
///
/// A single length-N complex FFT computes DCT-II; every other transform
/// reduces to it with O(N) shuffles:
///
/// - DCT-II: reorder `x` into evens-then-reversed-odds `v`,
///   take `V = FFT(v)`, then `X[k] = Re(W[k] * V[k])` with
///   `W[k] = exp(-i pi k / (2N))`.
/// - DCT-III: form `V[k] = y[k] * exp(-i pi k / (2N))`, take the unnormalised
///   inverse FFT `a = IFFT(V)`, then `S[0] = Re(a[0]) - y[0]/2`,
///   `S[2t] = Re(a[N-t]) - y[0]/2`, `S[2t+1] = Re(a[t+1]) - y[0]/2`,
///   scaled by `2/N`. (The inverse FFT evaluates cosines at the odd
///   residues `4j-1 mod 4N`, which cover every odd multiple `2m+1` exactly
///   once up to the evenness of cosine; the `y[0]/2` term restores the
///   DCT-III `k = 0` weight of one half.)
/// - DST-II: `DST-II(x)[k] = DCT-II(w)[N-1-k]` with `w[j] = (-1)^j x[j]`,
///   since `cos(pi (j+1/2) - a) = (-1)^j sin(a)`.
/// - DST-III: `DST-III(y)[n] = (-1)^n * DCT-III(reverse(y))[n]`
///   (same reversal duality as the DCT-III case).
/// - Adjoints are exact matrix transposes routed through the same kernels:
///   `dct2_adjoint(y) = (N/2) * DCT-III([2*y0, y1..])`,
///   `dct3_adjoint(y)[k] = w_k * (2/N) * DCT-II(y)[k]` (`w_0 = 1/2`),
///   `dst2_adjoint(y) = DST-III([(N/2)*y_.. , N*y_{N-1}])`,
///   `dst3_adjoint(y)[k] = row_k * DST-II(y)[k]`
///   (`row_k = 2/N`, except `1/N` for the last row).
struct FftDctKernel<T: FftScalar> {
    n: usize,
    plan: Arc<dyn FftPlan<T>>,
    /// Unnormalised inverse plan of the same length, for DCT-III.
    plan_inv: Arc<dyn FftPlan<T>>,
    /// Post-FFT twiddle `kernel[k] = exp(-i pi k / (2N))`.
    kernel: Vec<Complex<T>>,
}

impl<T: FftScalar> FftDctKernel<T> {
    fn new(n: usize) -> Self {
        assert!(n > 0, "FFT cosine kernel requires a positive length");
        let planner = CpuFftPlanner::<T>::new();
        let plan = planner.plan(n, FftDirection::Forward);
        let plan_inv = planner.plan(n, FftDirection::Inverse);
        let thetas: Vec<f64> = (0..n)
            .map(|k| std::f64::consts::PI * k as f64 / (2.0 * n as f64))
            .collect();
        let mut sins = vec![0.0_f64; n];
        let mut coss = vec![0.0_f64; n];
        fill_sincos(&thetas, &mut sins, &mut coss);
        let kernel = sins
            .into_iter()
            .zip(coss)
            .map(|(sin, cos)| {
                Complex::new(
                    T::from_f64(cos).expect("FFT scalar must represent cosine values"),
                    T::from_f64(-sin).expect("FFT scalar must represent sine values"),
                )
            })
            .collect();
        Self {
            n,
            plan,
            plan_inv,
            kernel,
        }
    }

    /// Unnormalised DCT-II via one length-N FFT.
    fn dct2(&self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n);
        let n = self.n;
        let mut work = vec![Complex::new(T::zero(), T::zero()); n + self.plan.scratch_len()];
        let (spectrum, scratch) = work.split_at_mut(n);
        // Even-odd reorder: evens ascending, then odds descending.
        for (i, slot) in spectrum.iter_mut().enumerate() {
            let source = if i < n.div_ceil(2) {
                input[2 * i]
            } else {
                input[2 * (n - 1 - i) + 1]
            };
            *slot = Complex::new(source, T::zero());
        }
        self.plan.process_with_scratch(spectrum, scratch);
        for ((out, value), twiddle) in output
            .iter_mut()
            .zip(spectrum.iter())
            .zip(self.kernel.iter())
        {
            *out = twiddle.re * value.re - twiddle.im * value.im;
        }
    }

    /// DCT-III (the scaled inverse of DCT-II) via one length-N inverse FFT.
    fn dct3(&self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n);
        let n = self.n;
        let mut work = vec![Complex::new(T::zero(), T::zero()); n + self.plan_inv.scratch_len()];
        let (spectrum, scratch) = work.split_at_mut(n);
        // `V[k] = y[k] * exp(-i pi k / (2N))`, exactly the stored kernel.
        for ((slot, sample), twiddle) in spectrum
            .iter_mut()
            .zip(input.iter())
            .zip(self.kernel.iter())
        {
            *slot = Complex::new(*sample * twiddle.re, *sample * twiddle.im);
        }
        self.plan_inv.process_with_scratch(spectrum, scratch);
        let half_dc = input[0] / T::from_f64(2.0).expect("FFT scalar must represent 2");
        let scale = T::from_f64(2.0 / n as f64).expect("FFT scalar must represent 2/N");
        // Odd residues `4j-1 mod 4N` cover every `2m+1` exactly once: even
        // outputs read the spectrum from the top, odd outputs from position
        // `(m+1)/2`.
        for (m, out) in output.iter_mut().enumerate() {
            let source = if m == 0 {
                spectrum[0].re
            } else if m % 2 == 0 {
                spectrum[n - m / 2].re
            } else {
                spectrum[m.div_ceil(2)].re
            };
            *out = (source - half_dc) * scale;
        }
    }

    /// Unnormalised DST-II via [`Self::dct2`].
    fn dst2(&self, input: &[T], output: &mut [T]) {
        let n = self.n;
        let mut weighted = vec![T::zero(); n];
        for (j, (slot, sample)) in weighted.iter_mut().zip(input.iter()).enumerate() {
            *slot = if j % 2 == 1 { -*sample } else { *sample };
        }
        let mut spectrum = vec![T::zero(); n];
        self.dct2(&weighted, &mut spectrum);
        for (k, out) in output.iter_mut().enumerate() {
            *out = spectrum[n - 1 - k];
        }
    }

    /// DST-III (the scaled inverse of DST-II) via [`Self::dct3`].
    fn dst3(&self, input: &[T], output: &mut [T]) {
        let n = self.n;
        let mut reversed = vec![T::zero(); n];
        for (i, slot) in reversed.iter_mut().enumerate() {
            *slot = input[n - 1 - i];
        }
        let mut spectrum = vec![T::zero(); n];
        self.dct3(&reversed, &mut spectrum);
        for (i, (out, value)) in output.iter_mut().zip(spectrum.iter()).enumerate() {
            *out = if i % 2 == 1 { -*value } else { *value };
        }
    }

    /// Transpose of the unnormalised DCT-II matrix, via [`Self::dct3`].
    fn dct2_adjoint(&self, input: &[T], output: &mut [T]) {
        let n = self.n;
        let mut weighted = vec![T::zero(); n];
        weighted[0] = input[0] + input[0];
        for (slot, sample) in weighted.iter_mut().skip(1).zip(input.iter().skip(1)) {
            *slot = *sample;
        }
        let mut spectrum = vec![T::zero(); n];
        self.dct3(&weighted, &mut spectrum);
        let scale = T::from_f64(n as f64 / 2.0).expect("FFT scalar must represent N/2");
        for (out, value) in output.iter_mut().zip(spectrum.iter()) {
            *out = *value * scale;
        }
    }

    /// Transpose of the scaled DCT-III matrix, via [`Self::dct2`].
    fn dct3_adjoint(&self, input: &[T], output: &mut [T]) {
        let n = self.n;
        let mut spectrum = vec![T::zero(); n];
        self.dct2(input, &mut spectrum);
        let scale = T::from_f64(2.0 / n as f64).expect("FFT scalar must represent 2/N");
        let half = T::from_f64(0.5).expect("FFT scalar must represent 1/2");
        for (k, (out, value)) in output.iter_mut().zip(spectrum.iter()).enumerate() {
            let weight = if k == 0 { half } else { T::one() };
            *out = *value * scale * weight;
        }
    }

    /// Transpose of the unnormalised DST-II matrix, via [`Self::dst3`].
    fn dst2_adjoint(&self, input: &[T], output: &mut [T]) {
        let n = self.n;
        let half_n = T::from_f64(n as f64 / 2.0).expect("FFT scalar must represent N/2");
        let full_n = T::from_f64(n as f64).expect("FFT scalar must represent N");
        let mut weighted = vec![T::zero(); n];
        for (k, (slot, sample)) in weighted.iter_mut().zip(input.iter()).enumerate() {
            *slot = *sample * if k + 1 == n { full_n } else { half_n };
        }
        self.dst3(&weighted, output);
    }

    /// Transpose of the scaled DST-III matrix, via the forward [`Self::dst2`]:
    /// `(S3^T y)[k] = row_k * DST-II(y)[k]` because the DST-III row weights
    /// factor out of the transposed sum.
    fn dst3_adjoint(&self, input: &[T], output: &mut [T]) {
        let n = self.n;
        let mut spectrum = vec![T::zero(); n];
        self.dst2(input, &mut spectrum);
        let full = T::from_f64(2.0 / n as f64).expect("FFT scalar must represent 2/N");
        let last = T::from_f64(1.0 / n as f64).expect("FFT scalar must represent 1/N");
        for (k, (out, value)) in output.iter_mut().zip(spectrum.iter()).enumerate() {
            *out = *value * if k + 1 == n { last } else { full };
        }
    }
}

/// Reusable DCT-II/III plan.
///
/// Lengths at or below [`DIRECT_THRESHOLD`] use the direct O(N^2) reference
/// summation (allocation-free); larger lengths use the O(N log N) FFT kernel,
/// which caches its complex FFT plan and post-FFT twiddles across calls but
/// allocates O(N) workspace per call.
pub struct DctPlan<T: FftScalar> {
    n: usize,
    kernel: Option<FftDctKernel<T>>,
}

impl<T: FftScalar> DctPlan<T> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            kernel: (n > DIRECT_THRESHOLD).then(|| FftDctKernel::new(n)),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Whether this plan executes through the FFT kernel (`true`) or the
    /// direct reference summation (`false`).
    pub fn uses_fft(&self) -> bool {
        self.kernel.is_some()
    }

    pub fn forward(&self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);
        match &self.kernel {
            Some(kernel) => kernel.dct2(input, output),
            None => dct2(input, output),
        }
    }

    pub fn inverse(&self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);
        match &self.kernel {
            Some(kernel) => kernel.dct3(input, output),
            None => dct3(input, output),
        }
    }

    /// Apply the Euclidean adjoint of [`Self::forward`].
    pub fn forward_adjoint(&self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);
        match &self.kernel {
            Some(kernel) => kernel.dct2_adjoint(input, output),
            None => dct2_adjoint(input, output),
        }
    }

    /// Apply the Euclidean adjoint of [`Self::inverse`].
    pub fn inverse_adjoint(&self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);
        match &self.kernel {
            Some(kernel) => kernel.dct3_adjoint(input, output),
            None => dct3_adjoint(input, output),
        }
    }
}

/// Reusable DST-II/III plan.
///
/// Same two-tier strategy as [`DctPlan`]: direct O(N^2) summation at or below
/// [`DIRECT_THRESHOLD`], O(N log N) FFT kernel above.
pub struct DstPlan<T: FftScalar> {
    n: usize,
    kernel: Option<FftDctKernel<T>>,
}

impl<T: FftScalar> DstPlan<T> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            kernel: (n > DIRECT_THRESHOLD).then(|| FftDctKernel::new(n)),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Whether this plan executes through the FFT kernel (`true`) or the
    /// direct reference summation (`false`).
    pub fn uses_fft(&self) -> bool {
        self.kernel.is_some()
    }

    pub fn forward(&self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);
        match &self.kernel {
            Some(kernel) => kernel.dst2(input, output),
            None => dst2(input, output),
        }
    }

    pub fn inverse(&self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);
        match &self.kernel {
            Some(kernel) => kernel.dst3(input, output),
            None => dst3(input, output),
        }
    }

    /// Apply the Euclidean adjoint of [`Self::forward`].
    pub fn forward_adjoint(&self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);
        match &self.kernel {
            Some(kernel) => kernel.dst2_adjoint(input, output),
            None => dst2_adjoint(input, output),
        }
    }

    /// Apply the Euclidean adjoint of [`Self::inverse`].
    pub fn inverse_adjoint(&self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);
        match &self.kernel {
            Some(kernel) => kernel.dst3_adjoint(input, output),
            None => dst3_adjoint(input, output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cosine_and_sine_round_trip() {
        for n in [1usize, 2, 7, 16] {
            let input: Vec<f64> = (0..n).map(|i| (0.17 * i as f64).sin()).collect();
            let mut transformed = vec![0.0; n];
            let mut recovered = vec![0.0; n];
            dct2(&input, &mut transformed);
            dct3(&transformed, &mut recovered);
            for (actual, expected) in recovered.iter().zip(&input) {
                assert_relative_eq!(actual, expected, max_relative = 1e-11, epsilon = 1e-11);
            }
            dst2(&input, &mut transformed);
            dst3(&transformed, &mut recovered);
            for (actual, expected) in recovered.iter().zip(&input) {
                assert_relative_eq!(actual, expected, max_relative = 1e-11, epsilon = 1e-11);
            }
        }
    }

    #[test]
    fn cosine_and_sine_parseval_identities_hold() {
        for n in [1usize, 2, 7, 16] {
            let input: Vec<f64> = (0..n)
                .map(|i| (0.17 * i as f64).sin() + 0.3 * (0.11 * i as f64).cos())
                .collect();
            let mut transformed = vec![0.0; n];

            dct2(&input, &mut transformed);
            let time_energy: f64 = input.iter().map(|value| value * value).sum();
            let cosine_energy = transformed
                .iter()
                .enumerate()
                .map(|(k, value)| {
                    let weight = if k == 0 { 1.0 } else { 2.0 };
                    weight * value * value
                })
                .sum::<f64>()
                / n as f64;
            assert_relative_eq!(cosine_energy, time_energy, epsilon = 1e-10);

            dst2(&input, &mut transformed);
            let sine_energy = transformed
                .iter()
                .enumerate()
                .map(|(k, value)| {
                    let weight = if k + 1 == n { 1.0 } else { 2.0 };
                    weight * value * value
                })
                .sum::<f64>()
                / n as f64;
            assert_relative_eq!(sine_energy, time_energy, epsilon = 1e-10);
        }
    }

    #[test]
    fn cosine_and_sine_adjoint_identities_hold() {
        for n in [0usize, 1, 2, 7, 16] {
            let x: Vec<f64> = (0..n)
                .map(|index| (0.17 * index as f64).sin() + 0.3)
                .collect();
            let y: Vec<f64> = (0..n)
                .map(|index| (0.23 * index as f64).cos() - 0.2)
                .collect();
            let inner_product = |left: &[f64], right: &[f64]| {
                left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>()
            };
            let mut forward_x = vec![0.0; n];
            let mut forward_adjoint_y = vec![0.0; n];
            dct2(&x, &mut forward_x);
            dct2_adjoint(&y, &mut forward_adjoint_y);
            assert_relative_eq!(
                inner_product(&forward_x, &y),
                inner_product(&x, &forward_adjoint_y),
                epsilon = 2e-11
            );
            dct3(&x, &mut forward_x);
            dct3_adjoint(&y, &mut forward_adjoint_y);
            assert_relative_eq!(
                inner_product(&forward_x, &y),
                inner_product(&x, &forward_adjoint_y),
                epsilon = 2e-11
            );

            dst2(&x, &mut forward_x);
            dst2_adjoint(&y, &mut forward_adjoint_y);
            assert_relative_eq!(
                inner_product(&forward_x, &y),
                inner_product(&x, &forward_adjoint_y),
                epsilon = 2e-11
            );
            dst3(&x, &mut forward_x);
            dst3_adjoint(&y, &mut forward_adjoint_y);
            assert_relative_eq!(
                inner_product(&forward_x, &y),
                inner_product(&x, &forward_adjoint_y),
                epsilon = 2e-11
            );

            let dct_plan = DctPlan::<f64>::new(n);
            dct_plan.forward(&x, &mut forward_x);
            dct_plan.forward_adjoint(&y, &mut forward_adjoint_y);
            assert_relative_eq!(
                inner_product(&forward_x, &y),
                inner_product(&x, &forward_adjoint_y),
                epsilon = 2e-11
            );
            let dst_plan = DstPlan::<f64>::new(n);
            dst_plan.inverse(&x, &mut forward_x);
            dst_plan.inverse_adjoint(&y, &mut forward_adjoint_y);
            assert_relative_eq!(
                inner_product(&forward_x, &y),
                inner_product(&x, &forward_adjoint_y),
                epsilon = 2e-11
            );
        }
    }

    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    /// The FFT kernel must reproduce the direct reference on both sides of
    /// the direct/FFT threshold, for even, odd, smooth, and prime lengths.
    #[test]
    fn fft_kernel_matches_direct_reference() {
        for n in [1usize, 2, 3, 5, 7, 8, 16, 63, 64, 65, 100, 127, 200, 256] {
            let input = pseudo_random(n, n as u64 * 31 + 7);
            let kernel = FftDctKernel::<f64>::new(n);
            let mut direct = vec![0.0; n];
            let mut fast = vec![0.0; n];
            let tol = (n as f64 * 1e-13).max(1e-12);

            dct2(&input, &mut direct);
            kernel.dct2(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = tol);
            }
            dct3(&input, &mut direct);
            kernel.dct3(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = tol);
            }
            dst2(&input, &mut direct);
            kernel.dst2(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = tol);
            }
            dst3(&input, &mut direct);
            kernel.dst3(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = tol);
            }
            dct2_adjoint(&input, &mut direct);
            kernel.dct2_adjoint(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = tol);
            }
            dct3_adjoint(&input, &mut direct);
            kernel.dct3_adjoint(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = tol);
            }
            dst2_adjoint(&input, &mut direct);
            kernel.dst2_adjoint(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = tol);
            }
            dst3_adjoint(&input, &mut direct);
            kernel.dst3_adjoint(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = tol);
            }
        }
    }

    #[test]
    fn plans_dispatch_by_threshold_and_round_trip() {
        assert!(!DctPlan::<f64>::new(0).uses_fft());
        assert!(!DctPlan::<f64>::new(64).uses_fft());
        assert!(DctPlan::<f64>::new(65).uses_fft());
        assert!(!DstPlan::<f64>::new(64).uses_fft());
        assert!(DstPlan::<f64>::new(65).uses_fft());

        for n in [65usize, 100, 127, 200, 256] {
            let input = pseudo_random(n, n as u64 * 17 + 3);
            let dct = DctPlan::<f64>::new(n);
            let dst = DstPlan::<f64>::new(n);
            assert!(dct.uses_fft() && dst.uses_fft());
            let mut transformed = vec![0.0; n];
            let mut recovered = vec![0.0; n];
            dct.forward(&input, &mut transformed);
            dct.inverse(&transformed, &mut recovered);
            for (actual, expected) in recovered.iter().zip(&input) {
                assert_relative_eq!(actual, expected, max_relative = 1e-9, epsilon = 1e-9);
            }
            dst.forward(&input, &mut transformed);
            dst.inverse(&transformed, &mut recovered);
            for (actual, expected) in recovered.iter().zip(&input) {
                assert_relative_eq!(actual, expected, max_relative = 1e-9, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn fft_adjoints_satisfy_inner_product_identities() {
        for n in [65usize, 100, 127] {
            let x = pseudo_random(n, n as u64 + 1);
            let y = pseudo_random(n, n as u64 + 999);
            let inner = |left: &[f64], right: &[f64]| {
                left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>()
            };
            let dct = DctPlan::<f64>::new(n);
            let dst = DstPlan::<f64>::new(n);
            let mut fx = vec![0.0; n];
            let mut ay = vec![0.0; n];
            dct.forward(&x, &mut fx);
            dct.forward_adjoint(&y, &mut ay);
            assert_relative_eq!(inner(&fx, &y), inner(&x, &ay), epsilon = 1e-9);
            dct.inverse(&x, &mut fx);
            dct.inverse_adjoint(&y, &mut ay);
            assert_relative_eq!(inner(&fx, &y), inner(&x, &ay), epsilon = 1e-9);
            dst.forward(&x, &mut fx);
            dst.forward_adjoint(&y, &mut ay);
            assert_relative_eq!(inner(&fx, &y), inner(&x, &ay), epsilon = 1e-9);
            dst.inverse(&x, &mut fx);
            dst.inverse_adjoint(&y, &mut ay);
            assert_relative_eq!(inner(&fx, &y), inner(&x, &ay), epsilon = 1e-9);
        }
    }

    #[test]
    fn fft_kernel_matches_reference_in_f32() {
        for n in [65usize, 100, 127] {
            let input: Vec<f32> = pseudo_random(n, n as u64 * 5 + 1)
                .into_iter()
                .map(|v| v as f32)
                .collect();
            let kernel = FftDctKernel::<f32>::new(n);
            let mut direct = vec![0.0f32; n];
            let mut fast = vec![0.0f32; n];
            let tol = 1e-4f32;
            dct2(&input, &mut direct);
            kernel.dct2(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert!((a - b).abs() <= tol * b.abs().max(1.0), "dct2 f32 n={n}");
            }
            dct3(&input, &mut direct);
            kernel.dct3(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert!((a - b).abs() <= tol * b.abs().max(1.0), "dct3 f32 n={n}");
            }
            dst2(&input, &mut direct);
            kernel.dst2(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert!((a - b).abs() <= tol * b.abs().max(1.0), "dst2 f32 n={n}");
            }
            dst3(&input, &mut direct);
            kernel.dst3(&input, &mut fast);
            for (a, b) in fast.iter().zip(direct.iter()) {
                assert!((a - b).abs() <= tol * b.abs().max(1.0), "dst3 f32 n={n}");
            }
        }
    }
}
