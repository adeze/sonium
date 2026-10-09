//! Bluestein chirp-Z transform.
//!
//! Computes a length-`n` DFT via a length-`m` convolution where `m` is
//! the next power of two `≥ 2n - 1`. This handles every length, in
//! particular primes that aren't supported by the small-radix Stockham
//! kernel (factors of 2/3/4/5).
//!
//! Identity used:
//!     n·k = (n² + k² − (n − k)²) / 2
//! so
//!     X[k] = c[k] · ∑_n (x[n] · c[n]) · b[k − n]
//! with `c[j] = w^(j²/2)` and `b[m] = w^(−m²/2)`, where
//! `w = exp(dir_sign · 2π i / n)` and `j²/2` is computed mod `2n` so it
//! is well-defined integer-valued for the lookup.

use num_complex::Complex;
use std::f64::consts::TAU;

use crate::CpuFftDispatch;
use crate::FftDirection;
use crate::scalar::FftScalar;
use crate::stockham;
use crate::twiddles::{StageTwiddles, build_stage_twiddles};

/// Precomputed data for a length-`n` Bluestein transform.
///
/// The inner length is `m = next_pow2(2n − 1)`, which is always a
/// product of `4`s and possibly one `2` — handled by the mixed-radix
/// Stockham kernel.
pub struct BluesteinPlan<T: FftScalar> {
    pub n: usize,
    pub m: usize,
    pub direction: FftDirection,
    /// `c[j] = w^(j²/2)` for `j ∈ [0, n)` — the per-sample chirp.
    chirp: Vec<Complex<T>>,
    /// FFT of the symmetric kernel `B` (length `m`).
    kernel_fft: Vec<Complex<T>>,
    /// Per-stage forward / inverse twiddles for the inner length-`m` FFT.
    stages_fwd: Vec<StageTwiddles<T>>,
    stages_inv: Vec<StageTwiddles<T>>,
    dispatch: CpuFftDispatch,
}

impl<T: FftScalar> BluesteinPlan<T> {
    pub fn new(n: usize, direction: FftDirection) -> Self {
        Self::new_with_dispatch(n, direction, CpuFftDispatch::Auto)
    }

    /// Checked construction for untrusted lengths: the arithmetic overflow
    /// that panics in [`Self::new`] is reported as
    /// [`super::factor::PlanLengthError::LengthOverflow`] instead.
    /// Lengths below 2 remain a programmer error (Bluestein only handles
    /// non-trivial lengths) and panic in both entry points.
    pub fn try_new(
        n: usize,
        direction: FftDirection,
    ) -> Result<Self, super::factor::PlanLengthError> {
        Self::try_new_with_dispatch(n, direction, CpuFftDispatch::Auto)
    }

    pub(crate) fn new_with_dispatch(
        n: usize,
        direction: FftDirection,
        dispatch: CpuFftDispatch,
    ) -> Self {
        Self::try_new_with_dispatch(n, direction, dispatch)
            .unwrap_or_else(|error| panic!("cannot build Bluestein plan: {error}"))
    }

    pub(crate) fn try_new_with_dispatch(
        n: usize,
        direction: FftDirection,
        dispatch: CpuFftDispatch,
    ) -> Result<Self, super::factor::PlanLengthError> {
        assert!(n > 1, "Bluestein requires n > 1");
        let convolution_len = n
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(super::factor::PlanLengthError::LengthOverflow { length: n })?;
        let m = super::factor::try_next_pow2_pub(convolution_len)?;

        let dir_sign = match direction {
            FftDirection::Forward => -1.0_f64,
            FftDirection::Inverse => 1.0_f64,
        };

        // Build chirp c[j] = w^(j²/2). w = exp(dir_sign · 2π i / n). We
        // compute the exponent in float modulo (2n) to keep precision.
        //
        // Both this chirp and the B-kernel below sweep `n` sin/cos calls,
        // and they share the same `jj = (j·j) mod 2n` integer pattern —
        // only the sign of `dir_sign` differs. Compute the chirp's `theta`
        // array once, fan out to (sin, cos) via the SIMD-batched
        // `fill_sincos`, and reuse for the B-kernel by negating the
        // sin component.
        let two_n = n
            .checked_mul(2)
            .ok_or(super::factor::PlanLengthError::LengthOverflow { length: n })?;
        let mut thetas = Vec::with_capacity(n);
        for j in 0..n {
            let jj = ((j as u128 * j as u128) % two_n as u128) as usize;
            thetas.push(dir_sign * TAU * (jj as f64) / (two_n as f64));
        }
        let mut sins = vec![0.0_f64; n];
        let mut coss = vec![0.0_f64; n];
        crate::sincos::fill_sincos(&thetas, &mut sins, &mut coss);

        let mut chirp = Vec::with_capacity(n);
        for j in 0..n {
            chirp.push(Complex::new(
                T::from_f64(coss[j]).unwrap(),
                T::from_f64(sins[j]).unwrap(),
            ));
        }

        // Kernel B of length m: B[m'] = w^(−m'²/2) for m' near 0 (both
        // positive and negative). Symmetric around 0. Same magnitudes as
        // the chirp but with `dir_sign` flipped, so `cos` is unchanged
        // and `sin` is negated.
        let mut b = vec![Complex::<T>::new(T::zero(), T::zero()); m];
        for j in 0..n {
            let val = Complex::new(
                T::from_f64(coss[j]).unwrap(),
                T::from_f64(-sins[j]).unwrap(),
            );
            b[j] = val;
            if j != 0 {
                b[m - j] = val;
            }
        }

        // Forward FFT of B (length m).
        let radices = match super::factor::plan_length(m) {
            super::factor::Strategy::MixedRadix(v) => v,
            super::factor::Strategy::Bluestein { .. } => {
                unreachable!("Bluestein inner length must be smooth (next_pow2)")
            }
        };
        let stages_fwd = build_stage_twiddles::<T>(m, &radices, FftDirection::Forward);
        let stages_inv = build_stage_twiddles::<T>(m, &radices, FftDirection::Inverse);

        let mut kernel_fft = b;
        let mut scratch = vec![Complex::<T>::new(T::zero(), T::zero()); m];
        stockham::run_single_with_dispatch(
            &stages_fwd,
            T::from_f64(-1.0).unwrap(),
            &mut kernel_fft,
            &mut scratch,
            dispatch,
        );

        Ok(Self {
            n,
            m,
            direction,
            chirp,
            kernel_fft,
            stages_fwd,
            stages_inv,
            dispatch,
        })
    }

    /// Forward transform of a single `n`-element buffer.
    pub fn process(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let n = self.n;
        let m = self.m;
        // Release-mode preconditions (not `debug_assert!`): a short buffer
        // or scratch would otherwise proceed to unchecked indexing below.
        assert!(
            buffer.len() == n,
            "Bluestein buffer len must equal plan len"
        );
        assert!(
            scratch.len() >= 2 * m,
            "Bluestein scratch too small for inner length"
        );
        let (a, work) = scratch.split_at_mut(m);
        let work = &mut work[..m];

        // 1. Build A: A[j] = x[j] · c[j], pad zeros up to m.
        for (aj, (xj, cj)) in a[..n].iter_mut().zip(buffer.iter().zip(self.chirp.iter())) {
            *aj = *xj * *cj;
        }
        for aj in a[n..m].iter_mut() {
            *aj = Complex::new(T::zero(), T::zero());
        }

        // 2. FFT(A) into A in place.
        stockham::run_single_with_dispatch(
            &self.stages_fwd,
            T::from_f64(-1.0).unwrap(),
            a,
            work,
            self.dispatch,
        );

        // 3. Pointwise multiply A := A · kernel_fft.
        for (aj, kj) in a.iter_mut().zip(self.kernel_fft.iter()) {
            *aj = *aj * *kj;
        }

        // 4. IFFT into A in place. Result is unnormalised (factor of m).
        stockham::run_single_with_dispatch(
            &self.stages_inv,
            T::from_f64(1.0).unwrap(),
            a,
            work,
            self.dispatch,
        );

        // 5. Output: X[k] = (1/m) · A[k] · c[k] for k in 0..n.
        let inv_m = T::from_f64(1.0 / m as f64).unwrap();
        for ((xk, ak), ck) in buffer
            .iter_mut()
            .zip(a.iter())
            .zip(self.chirp.iter())
            .take(n)
        {
            let scaled = Complex::new(ak.re * inv_m, ak.im * inv_m);
            *xk = scaled * *ck;
        }
    }

    /// Length of the scratch buffer needed by [`Self::process`].
    pub fn scratch_len(&self) -> usize {
        2 * self.m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn naive_dft(input: &[Complex64], dir_sign: f64) -> Vec<Complex64> {
        let n = input.len();
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        for (k, ok) in out.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, ij) in input.iter().enumerate() {
                let theta = dir_sign * TAU * (j as f64) * (k as f64) / (n as f64);
                acc += ij * Complex64::new(theta.cos(), theta.sin());
            }
            *ok = acc;
        }
        out
    }

    fn random(seed: u64, n: usize) -> Vec<Complex64> {
        let mut s = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let re = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let im = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                Complex64::new(re, im)
            })
            .collect()
    }

    fn check(n: usize, dir: FftDirection) {
        let plan = BluesteinPlan::<f64>::new(n, dir);
        let inp = random(n as u64 * 31 + 7, n);
        let mut buf = inp.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
        plan.process(&mut buf, &mut scratch);
        let dir_sign = match dir {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };
        let want = naive_dft(&inp, dir_sign);
        let tol = (n as f64 * 1e-12).max(1e-12);
        for (g, w) in buf.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = tol, epsilon = tol);
            assert_relative_eq!(g.im, w.im, max_relative = tol, epsilon = tol);
        }
    }

    #[test]
    fn bluestein_primes() {
        for &n in &[7usize, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] {
            check(n, FftDirection::Forward);
            check(n, FftDirection::Inverse);
        }
    }

    #[test]
    fn bluestein_non_smooth_composites() {
        // composites containing a non-{2,3,4,5} factor
        for &n in &[14usize, 21, 22, 26, 33, 35, 49, 55, 77, 91] {
            check(n, FftDirection::Forward);
        }
    }

    #[test]
    #[should_panic(expected = "buffer len")]
    fn process_rejects_short_buffer() {
        // Release-mode precondition: must panic even with debug
        // assertions disabled (this is a `should_panic` test, so it
        // passes in both debug and release profiles).
        let plan = BluesteinPlan::<f64>::new(7, FftDirection::Forward);
        let mut buf = vec![Complex64::new(0.0, 0.0); 6];
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
        plan.process(&mut buf, &mut scratch);
    }

    #[test]
    #[should_panic(expected = "scratch")]
    fn process_rejects_short_scratch() {
        let plan = BluesteinPlan::<f64>::new(7, FftDirection::Forward);
        let mut buf = vec![Complex64::new(0.0, 0.0); 7];
        let mut scratch = vec![Complex64::new(0.0, 0.0); 1];
        plan.process(&mut buf, &mut scratch);
    }

    #[test]
    fn try_new_reports_overflow_without_panicking() {
        // `2n - 1` is unrepresentable here, so even the convolution length
        // cannot be formed.
        assert!(matches!(
            BluesteinPlan::<f64>::try_new(usize::MAX, FftDirection::Forward),
            Err(crate::factor::PlanLengthError::LengthOverflow { .. })
        ));
        // Representable lengths build exactly like `new`.
        let checked = BluesteinPlan::<f64>::try_new(7, FftDirection::Forward)
            .expect("prime length 7 must build");
        assert_eq!(
            checked.scratch_len(),
            BluesteinPlan::<f64>::new(7, FftDirection::Forward).scratch_len()
        );
    }
}
