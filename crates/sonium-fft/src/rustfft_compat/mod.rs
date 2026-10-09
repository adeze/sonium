//! Partial-shim of the `rustfft` 6.x API surface.
//!
//! Call sites that read
//!
//! ```rust,ignore
//! use rustfft::{Fft, FftPlanner};
//! let mut planner = FftPlanner::<f64>::new();
//! let plan = planner.plan_fft_forward(1024);
//! plan.process(&mut buffer);
//! ```
//!
//! migrate by changing the `use` to
//! `use sonium_fft::rustfft_compat::{Fft, FftPlanner};` — for the common
//! mutator + immutable + scratch surface. The shim is CPU-only; for
//! backend-aware planning use the native [`crate::FftPlanner`] /
//! [`crate::FftPlan`] API instead.
//!
//! # What is covered
//!
//! - [`FftPlanner<T>`] with `new`, `plan_fft`, `plan_fft_forward`, `plan_fft_inverse`.
//! - [`Fft<T>`] trait extending [`Length`] + [`Direction`] + `Send + Sync`.
//! - In-place: `process`, `process_with_scratch`.
//! - Out-of-place: `process_outofplace_with_scratch`.
//! - Immutable: `process_immutable_with_scratch` (rustfft 6.4+).
//! - Scratch-length queries: `get_inplace_scratch_len`,
//!   `get_outofplace_scratch_len`, `get_immutable_scratch_len`.
//! - [`FftNum`] marker for `f32` / `f64`.
//!
//! # What is *not* covered
//!
//! - Algorithm-specific concrete types (`Radix4`, `MixedRadix`,
//!   `BluesteinAlgorithm`, …). These are rustfft implementation details
//!   that few callers reach for; if you do, you are already past "drop-in"
//!   territory.
//! - The `RealFftPlanner` from the `realfft` companion crate.
//! - Backwards-compatibility with rustfft <= 6.3 trait surfaces.

use num_complex::Complex;
use num_traits::Float;
use std::sync::Arc;

use crate::FftDirection;
use crate::FftPlan as NativeFftPlan;
use crate::cpu::planner::CpuFftPlanner;
use crate::scalar::FftScalar;

/// Marker trait for the real types supported by [`Fft`]. Implemented for
/// `f32` and `f64`, matching `rustfft::FftNum`.
pub trait FftNum: FftScalar + Float {}
impl<T: FftScalar + Float> FftNum for T {}

/// Mirror of `rustfft::Length` — separated out so generic code that
/// constrains on `Length` alone keeps compiling.
#[allow(clippy::len_without_is_empty)]
pub trait Length {
    fn len(&self) -> usize;
}

/// Mirror of `rustfft::Direction`.
pub trait Direction {
    fn fft_direction(&self) -> FftDirection;
}

/// Trait mirroring `rustfft::Fft` (6.4+ shape). Inherits [`Length`] and
/// [`Direction`] so generic callers that constrain on the supertraits
/// alone keep compiling.
///
/// The transform convention is the unnormalised DFT:
/// `X[k] = sum_j x[j] exp(sign * 2πi jk/N)`, with `sign = -1` for
/// [`FftDirection::Forward`] and `sign = +1` for inverse. Callers scale an
/// inverse result by `1/N` for a round trip. Buffers contain contiguous
/// `Complex<T>` samples; a buffer longer than `len()` is interpreted as
/// independent consecutive batches.
///
/// Every mutating method requires a non-empty buffer whose length is a
/// positive multiple of `len()`. The scratch slice must be at least the
/// reported query length. Out-of-place input and output must have equal
/// lengths and must not overlap; the immutable variant preserves its input.
pub trait Fft<T: FftNum>: Length + Direction + Send + Sync {
    fn process(&self, buffer: &mut [Complex<T>]);
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);
    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );
    /// Read-only input variant (rustfft 6.4+). `input` is treated as
    /// immutable; the data is copied through `scratch` / `output` so the
    /// caller's buffer is preserved on exit.
    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );
    fn get_inplace_scratch_len(&self) -> usize;
    fn get_outofplace_scratch_len(&self) -> usize;
    fn get_immutable_scratch_len(&self) -> usize;
}

struct FftAdapter<T: FftNum> {
    inner: Arc<dyn NativeFftPlan<T>>,
}

/// rustfft 6.x semantics: panic when `buffer.len() < self.len()` or when
/// the buffer is not a positive integer multiple of `self.len()`.
#[inline]
fn assert_rustfft_buffer<T: FftNum>(n: usize, buffer: &[Complex<T>], label: &str) {
    assert!(
        !buffer.is_empty(),
        "rustfft_compat: {label} buffer must be non-empty (rustfft panics on empty input)"
    );
    assert!(
        buffer.len() >= n,
        "rustfft_compat: {label} buffer length {len} is shorter than plan length {n}",
        len = buffer.len(),
    );
    assert_eq!(
        buffer.len() % n,
        0,
        "rustfft_compat: {label} buffer length {len} is not a multiple of plan length {n}",
        len = buffer.len(),
    );
}

#[inline]
fn assert_rustfft_scratch<T: FftNum>(required: usize, scratch: &[Complex<T>], label: &str) {
    assert!(
        scratch.len() >= required,
        "rustfft_compat: {label} scratch length {} is shorter than required {}",
        scratch.len(),
        required
    );
}

impl<T: FftNum> Length for FftAdapter<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T: FftNum> Direction for FftAdapter<T> {
    fn fft_direction(&self) -> FftDirection {
        self.inner.direction()
    }
}

impl<T: FftNum> Fft<T> for FftAdapter<T> {
    fn process(&self, buffer: &mut [Complex<T>]) {
        assert_rustfft_buffer(self.inner.len(), buffer, "process");
        self.inner.process(buffer);
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        assert_rustfft_buffer(self.inner.len(), buffer, "process_with_scratch");
        assert_rustfft_scratch(self.inner.scratch_len(), scratch, "process_with_scratch");
        self.inner.process_with_scratch(buffer, scratch);
    }

    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        // rustfft contract: input may be clobbered, output receives the
        // result. We emulate by copying input → output and running in
        // place. The pre-check matches rustfft's panic shape.
        let n = self.inner.len();
        assert_rustfft_buffer(n, input, "process_outofplace_with_scratch(input)");
        assert_eq!(
            input.len(),
            output.len(),
            "rustfft_compat: process_outofplace_with_scratch input/output lengths differ"
        );
        assert_rustfft_scratch(
            self.inner.scratch_len(),
            scratch,
            "process_outofplace_with_scratch",
        );
        output.copy_from_slice(input);
        self.inner.process_with_scratch(output, scratch);
    }

    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let n = self.inner.len();
        assert_rustfft_buffer(n, input, "process_immutable_with_scratch(input)");
        assert_eq!(
            input.len(),
            output.len(),
            "rustfft_compat: process_immutable_with_scratch input/output lengths differ"
        );
        assert_rustfft_scratch(
            self.inner.scratch_len(),
            scratch,
            "process_immutable_with_scratch",
        );
        output.copy_from_slice(input);
        self.inner.process_with_scratch(output, scratch);
    }

    fn get_inplace_scratch_len(&self) -> usize {
        self.inner.scratch_len()
    }

    fn get_outofplace_scratch_len(&self) -> usize {
        self.inner.scratch_len()
    }

    fn get_immutable_scratch_len(&self) -> usize {
        self.inner.scratch_len()
    }
}

/// `rustfft::FftPlanner` shim. Holds a [`CpuFftPlanner`] internally.
pub struct FftPlanner<T: FftNum> {
    inner: CpuFftPlanner<T>,
}

impl<T: FftNum> Default for FftPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FftNum> FftPlanner<T> {
    pub fn new() -> Self {
        Self {
            inner: CpuFftPlanner::<T>::new(),
        }
    }

    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        Arc::new(FftAdapter {
            inner: self.inner.plan(len, direction),
        })
    }

    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn forward_round_trip_inverse() {
        let mut planner = FftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(64);
        let inv = planner.plan_fft_inverse(64);

        let input: Vec<Complex64> = (0..64)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut buf = input.clone();
        fwd.process(&mut buf);
        inv.process(&mut buf);
        // Inverse is unnormalised: buf == 64 * input.
        for (b, x) in buf.iter().zip(input.iter()) {
            assert_relative_eq!(b.re / 64.0, x.re, max_relative = 1e-12, epsilon = 1e-12);
            assert_relative_eq!(b.im / 64.0, x.im, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    #[test]
    fn shape_matches_rustfft_for_smooth_size() {
        let mut ours = FftPlanner::<f64>::new();
        let mut theirs = rustfft::FftPlanner::<f64>::new();
        for n in [4usize, 8, 12, 24, 36, 48, 64, 120, 1024] {
            let p_ours = ours.plan_fft_forward(n);
            let p_theirs = theirs.plan_fft_forward(n);
            let input: Vec<Complex64> = (0..n)
                .map(|i| Complex64::new(i as f64, -(i as f64) * 0.3))
                .collect();
            let mut a = input.clone();
            let mut b = input.clone();
            p_ours.process(&mut a);
            p_theirs.process(&mut b);
            for (ai, bi) in a.iter().zip(b.iter()) {
                assert_relative_eq!(ai.re, bi.re, max_relative = 1e-10, epsilon = 1e-10);
                assert_relative_eq!(ai.im, bi.im, max_relative = 1e-10, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn randomized_round_trip_many_sizes() {
        use rand::Rng;
        let mut rng = rand::rng();
        let mut planner = FftPlanner::<f64>::new();
        // Test a mix of smooth and non-smooth lengths.
        for &n in &[
            7usize, 16, 23, 32, 55, 64, 89, 128, 211, 256, 337, 512, 1024,
        ] {
            let fwd = planner.plan_fft_forward(n);
            let inv = planner.plan_fft_inverse(n);
            let input: Vec<Complex64> = (0..n)
                .map(|_| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                .collect();
            let mut buf = input.clone();
            fwd.process(&mut buf);
            inv.process(&mut buf);
            for (b, x) in buf.iter().zip(input.iter()) {
                assert_relative_eq!(b.re / n as f64, x.re, max_relative = 1e-10, epsilon = 1e-10);
                assert_relative_eq!(b.im / n as f64, x.im, max_relative = 1e-10, epsilon = 1e-10);
            }
        }
    }
}
