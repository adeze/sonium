//! Twiddle table generation.
//!
//! For an FFT of length `N` and direction `dir`, the twiddle root is
//! `w = exp(dir_sign · 2π i / N)`. Inter-stage twiddles needed by the
//! Stockham driver are `w^(l · k)` where `l ∈ [0, m)` and `k ∈ [1, r)`
//! for a stage going from sub-FFT size `m` to `m · r`.
//!
//! The public [`TwiddleTable`] retains a full table for compatibility, while
//! [`CompressedTwiddleTable`] stores only one quadrant when `N` is divisible
//! by four and reconstructs the remaining roots by symmetry. Stage planning
//! uses the compressed representation.
//!
//! `sin/cos` evaluation goes through [`crate::sincos::fill_sincos`] —
//! which routes to `sonium-fastmath` SIMD kernels (NEON 2-lane on aarch64,
//! AVX2 4-lane on x86_64) with a scalar fallback. Plan-build cost on
//! large transforms drops by ~1.5–2× versus libm `f64::sin_cos`.

use num_complex::Complex;
use std::f64::consts::TAU;

use crate::FftDirection;
use crate::scalar::FftScalar;
use crate::sincos::fill_sincos;

/// Precomputed twiddle table of size `N`. Entry `j` holds
/// `exp(dir_sign · 2π i · j / N)`.
pub struct TwiddleTable<T: FftScalar> {
    pub roots: Vec<Complex<T>>,
}

/// Quarter-table twiddles with symmetry reconstruction for lengths divisible
/// by four. Non-divisible lengths use a compact full-table fallback because
/// their quadrants do not land on integer table indices.
pub struct CompressedTwiddleTable<T: FftScalar> {
    n: usize,
    conjugate: bool,
    quarter: Option<Vec<Complex<T>>>,
    full: Option<Vec<Complex<T>>>,
}

impl<T: FftScalar> CompressedTwiddleTable<T> {
    pub fn new(n: usize, direction: FftDirection) -> Self {
        assert!(n > 0, "twiddle length must be positive");
        // Build one positive-angle table, then conjugate it for the forward
        // DFT whose root is exp(-i 2π/N).
        let conjugate = matches!(direction, FftDirection::Forward);
        if n.is_multiple_of(4) {
            let quarter_len = n / 4;
            let phases: Vec<f64> = (0..=quarter_len)
                .map(|j| std::f64::consts::TAU * j as f64 / n as f64)
                .collect();
            let mut sins = vec![0.0; phases.len()];
            let mut coss = vec![0.0; phases.len()];
            sonium_fastmath::sincos_f64_slice(&phases, &mut sins, &mut coss);
            let quarter = phases
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    Complex::new(T::from_f64(coss[i]).unwrap(), T::from_f64(sins[i]).unwrap())
                })
                .collect();
            Self {
                n,
                conjugate,
                quarter: Some(quarter),
                full: None,
            }
        } else {
            let phases: Vec<f64> = (0..n)
                .map(|j| std::f64::consts::TAU * j as f64 / n as f64)
                .collect();
            let mut sins = vec![0.0; phases.len()];
            let mut coss = vec![0.0; phases.len()];
            sonium_fastmath::sincos_f64_slice(&phases, &mut sins, &mut coss);
            let full = phases
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    Complex::new(T::from_f64(coss[i]).unwrap(), T::from_f64(sins[i]).unwrap())
                })
                .collect();
            Self {
                n,
                conjugate,
                quarter: None,
                full: Some(full),
            }
        }
    }

    /// Return `exp(sign · 2π i · j / N)` with a bounds-checked index.
    #[inline]
    pub fn at(&self, j: usize) -> Complex<T> {
        assert!(j < self.n, "twiddle index out of bounds");
        let value = if let Some(full) = &self.full {
            full[j]
        } else {
            let quarter = self.n / 4;
            let group = j / quarter;
            let remainder = j % quarter;
            let base = self.quarter.as_ref().expect("quarter table missing");
            match group {
                0 => base[remainder],
                1 => {
                    let v = base[remainder];
                    Complex::new(-v.im, v.re)
                }
                2 => {
                    let v = base[remainder];
                    Complex::new(-v.re, -v.im)
                }
                3 => {
                    let v = base[remainder];
                    Complex::new(v.im, -v.re)
                }
                _ => unreachable!("twiddle quadrant is in 0..4"),
            }
        };
        if self.conjugate {
            Complex::new(value.re, -value.im)
        } else {
            value
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn stored_len(&self) -> usize {
        self.quarter
            .as_ref()
            .map_or_else(|| self.full.as_ref().map_or(0, Vec::len), Vec::len)
    }
}

impl<T: FftScalar> TwiddleTable<T> {
    pub fn new(n: usize, direction: FftDirection) -> Self {
        let dir_sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };
        // Build the input phase array, then sincos-batch into a paired
        // output. Single allocation; the SIMD batch runs through
        // sonium-fastmath's `fill_sincos` helper.
        let mut thetas = Vec::with_capacity(n);
        for j in 0..n {
            thetas.push(dir_sign * TAU * (j as f64) / (n as f64));
        }
        let mut sins = vec![0.0_f64; n];
        let mut coss = vec![0.0_f64; n];
        fill_sincos(&thetas, &mut sins, &mut coss);

        let mut roots = Vec::with_capacity(n);
        for j in 0..n {
            roots.push(Complex::new(
                T::from_f64(coss[j]).unwrap(),
                T::from_f64(sins[j]).unwrap(),
            ));
        }
        Self { roots }
    }

    /// `w^j` for `j ∈ [0, N)`.
    #[inline(always)]
    pub fn at(&self, j: usize) -> Complex<T> {
        // Callers normally compute `j` modulo N, but this is a public helper
        // and must remain safe if a future caller violates that invariant.
        assert!(j < self.roots.len(), "twiddle index out of bounds");
        self.roots[j]
    }
}

/// Per-stage twiddle table laid out for stride-1 SIMD loads.
///
/// For a radix-r Stockham stage with sub-FFT size `m` and group stride
/// `s = N / (r·m)`, the scalar inner loop multiplies `x_k[l]` by
/// `roots[k · l · s mod N]` for `k ∈ [1, r)` and `l ∈ [0, m)`. The
/// strided access pattern defeats SIMD twiddle loads on the unsorted
/// [`TwiddleTable`]; this struct flattens those `(r-1) × m` complex
/// values into a single contiguous `Vec<Complex<T>>` so each stage's
/// inner loop reads consecutive twiddles.
///
/// Memory: `(radix-1) × m` complex per stage. Summed across all stages
/// of an N-point FFT this is ≈ N complex — roughly the same total as
/// the legacy single-N-array layout.
pub struct StageTwiddles<T: FftScalar> {
    /// Stage radix. 2, 3, 4, or 5 today.
    pub radix: usize,
    /// Sub-FFT size entering this stage. Stride within each per-k array.
    pub m: usize,
    /// Group count: `n / (radix · m)`. Cached for the stage loop.
    pub s: usize,
    /// Flat storage: `[tw1[0..m], tw2[0..m], …, tw_{r-1}[0..m]]`.
    /// `tw_k[l]` lives at `flat[(k-1)·m + l]`.
    flat: Vec<Complex<T>>,
    /// Radix-4 twiddles packed by adjacent `l` pairs as
    /// `[k1.re, k1.im, k2.re, k2.im, k3.re, k3.im]`, with two values per
    /// component. This avoids deinterleaving in the AArch64 fused tail.
    #[allow(dead_code)]
    radix4_pairs: Vec<T>,
}

impl<T: FftScalar> StageTwiddles<T> {
    /// Contiguous slice for the `k`-th radix multiplier (`k ∈ [1, radix)`),
    /// length `m`. Lane-parallel SIMD loads (`vld2q_f64`) read pairs of
    /// consecutive twiddles in one instruction.
    #[inline(always)]
    pub fn k_slice(&self, k: usize) -> &[Complex<T>] {
        debug_assert!(k >= 1 && k < self.radix);
        let base = (k - 1) * self.m;
        &self.flat[base..base + self.m]
    }

    /// Check the layout invariant required by a direct radix stage.
    ///
    /// SIMD entry points are unsafe because their loop bounds are derived
    /// from these relationships. Keep the checks in one place so debug and
    /// test builds fail at the call site instead of much later on an indexed
    /// load. Release builds retain the documented caller precondition.
    #[inline]
    pub(crate) fn debug_assert_stage_shape(
        &self,
        radix: usize,
        stride: usize,
        src_len: usize,
        dst_len: usize,
    ) {
        debug_assert_eq!(self.radix, radix);
        debug_assert_eq!(src_len, dst_len);
        debug_assert_eq!(stride, self.m.saturating_mul(self.s));
        debug_assert_eq!(src_len, radix.saturating_mul(stride));
        debug_assert_eq!(self.flat.len(), (radix - 1).saturating_mul(self.m));
    }

    /// Construct from raw parts (test fixture helper). The caller must
    /// ensure `flat.len() == (radix - 1) * m`.
    pub fn from_parts(radix: usize, m: usize, s: usize, flat: Vec<Complex<T>>) -> Self {
        assert!(radix >= 2, "stage radix must be at least two");
        assert_eq!(flat.len(), (radix - 1) * m);
        let radix4_pairs = pack_radix4_pairs(radix, m, &flat);
        Self {
            radix,
            m,
            s,
            flat,
            radix4_pairs,
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn radix4_pair_ptr(&self, l: usize) -> *const T {
        assert_eq!(self.radix, 4, "radix-4 packed twiddles required");
        assert_eq!(l & 1, 0, "packed twiddle offset must be even");
        assert!(l + 1 < self.m, "packed twiddle offset out of range");
        assert!(
            !self.radix4_pairs.is_empty(),
            "packed twiddles are unavailable"
        );
        unsafe { self.radix4_pairs.as_ptr().add((l / 2) * 12) }
    }
}

fn pack_radix4_pairs<T: FftScalar>(radix: usize, m: usize, flat: &[Complex<T>]) -> Vec<T> {
    // Only the AArch64 4096 fused tail consumes this layout. Restrict it to
    // the two stage widths used there so other plans and architectures do not
    // pay the duplicate twiddle-storage cost.
    if !cfg!(target_arch = "aarch64") || radix != 4 || !matches!(m, 256 | 1024) {
        return Vec::new();
    }
    let mut packed = Vec::with_capacity((m / 2) * 12);
    for pair in 0..m / 2 {
        let l = pair * 2;
        for k in 0..3 {
            packed.push(flat[k * m + l].re);
            packed.push(flat[k * m + l + 1].re);
            packed.push(flat[k * m + l].im);
            packed.push(flat[k * m + l + 1].im);
        }
    }
    packed
}

/// Build per-stage twiddle tables for the Stockham factorisation. Calls
/// scalar `f64::sin_cos` once per entry; precision is the same as the
/// legacy [`TwiddleTable`].
pub fn build_stage_twiddles<T: FftScalar>(
    n: usize,
    radices: &[usize],
    direction: FftDirection,
) -> Vec<StageTwiddles<T>> {
    if n <= 1 || radices.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(radices.len());
    let compressed = CompressedTwiddleTable::<T>::new(n, direction);
    let mut m: usize = 1;
    for &r in radices {
        let m_new = m.checked_mul(r).expect("FFT stage size overflow");
        let s = n / m_new;
        let count = (r - 1) * m;
        let mut flat: Vec<Complex<T>> = Vec::with_capacity(count);
        for k in 1..r {
            for l in 0..m {
                flat.push(compressed.at((k * l * s) % n));
            }
        }
        let radix4_pairs = pack_radix4_pairs(r, m, &flat);
        out.push(StageTwiddles {
            radix: r,
            m,
            s,
            flat,
            radix4_pairs,
        });
        m = m_new;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn compressed_quadrants_match_direct_phases() {
        for &n in &[4usize, 8, 16, 60, 1024, 7, 15] {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let table = CompressedTwiddleTable::<f64>::new(n, direction);
                assert!(table.stored_len() <= n);
                for j in 0..n {
                    let sign = match direction {
                        FftDirection::Forward => -1.0,
                        FftDirection::Inverse => 1.0,
                    };
                    let (sin, cos) = (sign * TAU * j as f64 / n as f64).sin_cos();
                    let got = table.at(j);
                    assert_relative_eq!(got.re, cos, epsilon = 2e-14, max_relative = 2e-14);
                    assert_relative_eq!(got.im, sin, epsilon = 2e-14, max_relative = 2e-14);
                }
            }
        }
    }

    #[test]
    fn quarter_storage_is_used_for_four_smooth_lengths() {
        let table = CompressedTwiddleTable::<f64>::new(1024, FftDirection::Forward);
        assert_eq!(table.stored_len(), 257);
    }
}
