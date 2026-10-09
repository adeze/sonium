//! # sonium-fastmath
//!
//! Vectorised math primitives for the rest of the workspace. Phase, phasor,
//! and derivative functions share one contract; the crate exists so it can be reused by
//! `sonium-compute` (BEM Helmholtz kernel), `sonium-fft` (phase rotators
//! and twiddle generation), and any future audio / acoustic code that needs
//! `exp(i·k·r)`-style work without paying scalar libm per element.
//!
//! ## Algorithm
//!
//! Two-term Cody–Waite range reduction `x = k·π/2 + r` with `r ∈ [−π/4,
//! π/4]`, then Sleef u10 minimax polynomials evaluated as Horner FMAs, then
//! quadrant select via bitwise masks. The fast path is used for
//! `|x| <= [`CODY_WAITE_LIMIT`]`; `CODY_WAITE_LIMIT < |x| <=
//! [`MID_RANGE_LIMIT`]` uses a three-term Cody–Waite reduction with the same
//! polynomial; larger finite values use the fixed-point Payne–Hanek slow
//! path and non-finite inputs delegate to `f64::sin_cos` so quadrants are
//! never silently lost.
//!
//! ## Lane width
//!
//! - aarch64 (mandatory NEON): 2-lane `f64` (`float64x2_t`) and 4-lane
//!   `f32` (`float32x4_t`).
//! - x86_64: AVX-512 `f64x8`/`f32x16` when available, then AVX2 + FMA
//!   `f64x4`/`f32x8`.
//! - wasm32: `f32x4` through `simd128` when that target feature is enabled.
//! - All other targets / unavailable SIMD features: scalar via
//!   [`sincos_f64`].
//!
//! For reproducible native-x86 evidence, set `SONIUM_FASTMATH_FORCE_SCALAR=1`
//! before process start for the scalar baseline or
//! `SONIUM_FASTMATH_FORCE_AVX2=1` to select AVX2 even on an AVX-512 host. The
//! latter still requires AVX2+FMA; the focused x86 validation tests use
//! `SONIUM_FASTMATH_REQUIRE_AVX2=1` to fail closed when that hardware is absent.
//!
//! ARM SVE/SVE2 is intentionally not enabled yet. Stable Rust recognizes the
//! target feature but its scalable SVE vector types and intrinsics remain
//! unstable, so the scalar/NEON paths remain the sound fallback until a
//! supported implementation and cross-target test toolchain are available.
//!
//! ## API levels
//!
//! - [`sincos_f64`] — single scalar `(sin, cos)`. Uses the same Cody–Waite
//!   reduction as the SIMD versions. The AVX2 path evaluates the full Sleef
//!   u10 chain and matches scalar output bit-for-bit per lane; the NEON path
//!   uses a compensated shortened chain (dropping `SIN_S0/S1` and `COS_C0`
//!   with derived `COS_C*_COMP` constants) and agrees only within the ULP
//!   envelope documented below.
//! - [`sincos_f64x2`] — lane-parallel pair (aarch64 only). Caller already
//!   has data in a `float64x2_t`.
//!
//! See the `tests/sincos_ulp.rs` harness for the four-suite validation
//! against `f64::sin_cos` (uniform sweep, quadrant boundary, large
//! argument, special inputs).
//!
//! ## Published accuracy contract
//!
//! | Domain | Contract |
//! | --- | --- |
//! | `|x| <= 10³`, ordinary finite inputs | maximum 2 and mean 0.5 `EPSILON`-at-one units on the uniform sweep |
//! | reduction boundaries near `k·π/2` | maximum 3 and mean 1 `EPSILON`-at-one units |
//! | `10³`-scale finite fast-path inputs | maximum 8 and mean 1.5 `EPSILON`-at-one units |
//! | `10³ < |x| <= 10⁸` (three-term mid-range) | maximum 8 `EPSILON`-at-one units; quadrant-preserving reference comparison |
//! | finite inputs outside [`MID_RANGE_LIMIT`] | Payne–Hanek scalar fallback; quadrant-preserving reference comparison |
//! | `±0` | sine preserves the input signed zero and cosine is `+1` |
//! | `±∞` and NaN | both outputs are NaN, matching `f64::sin_cos` semantics |
//!
//! `accuracy_report` v2 reports absolute error both in units of `EPSILON` at
//! magnitude one and as conventional representable-float ULP distance. The
//! latter excludes reference outputs with magnitude at most `1e-12`, where
//! result-spacing ULP counts become uninformative near roots; the report gives
//! separate sample counts for sine and cosine. The near-root error budget is
//! absolute (`≤ 64·ε` in the full finite-domain sweep) because relative and
//! result-spacing errors at a true zero are undefined. Legacy Rust fields
//! ending in `_ulp` remain as deprecated aliases for the `EPSILON`-at-one
//! metric; new code should use `*_error_eps` or `*_result_ulp` explicitly.
//! The checked-in comparison harness and versioned report are the evidence for
//! these partitions.
//!
//! ## std requirement
//!
//! The crate links against `std::f64` (for `round`) and `libm`'s `sin_cos`
//! is reached via `f64::sin_cos`, both of which live in `std`. If we ever
//! need to use this in a `no_std` consumer, swap the `round` call for the
//! `libm` crate's `round`; the polynomial path itself is `no_std`-clean.

mod api;
pub mod coeffs;
mod payne_hanek;
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(target_arch = "wasm32")]
pub mod wasm_simd;

pub use api::{
    AccuracyProfile, AccuracyReport, SimdBackend, SinCosDerivatives, accuracy_report,
    dispatch_features, exp_i, exp_i_f32, exp_i_f32_slice, exp_i_f32_with_derivative_slice,
    exp_i_f32_with_profile, exp_i_slice, exp_i_with_derivative, exp_i_with_derivative_profile,
    exp_i_with_derivative_slice, exp_i_with_derivative_slice_with_profile, exp_i_with_profile,
    mid_path_count, reset_mid_path_count, reset_slow_path_count, selected_backend, sincos_f32,
    sincos_f32_slice, sincos_f32_with_derivatives, sincos_f64_slice, sincos_with_derivatives,
    sincos_with_profile, slow_path_count,
};
pub use scalar::{CODY_WAITE_LIMIT, MID_RANGE_LIMIT, sincos_scalar};

#[cfg(target_arch = "aarch64")]
pub use neon::sincos_f32x4;

#[cfg(target_arch = "x86_64")]
pub use avx2::sincos_f32x8;

/// Scalar `(sin(x), cos(x))` via the same Cody–Waite + Sleef u10 polynomial
/// as the SIMD kernels. Convenience re-export of [`scalar::sincos_scalar`].
#[inline]
pub fn sincos_f64(x: f64) -> (f64, f64) {
    scalar::sincos_scalar(x)
}

/// Lane-parallel `(sin, cos)` for two f64 inputs packed in a
/// `float64x2_t`. AArch64-only.
///
/// See [`neon::sincos_f64x2`] for accuracy notes. Lanes outside
/// [`CODY_WAITE_LIMIT`] use the scalar fallback (mid-range reduction,
/// Payne–Hanek, or `f64::sin_cos` for non-finite inputs).
///
/// # Safety
///
/// AArch64 implies NEON, so this is always safe to call on a supported
/// target. The unsafe block is required only because `target_feature`
/// attributes propagate the unsafe contract.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn sincos_f64x2(
    x: core::arch::aarch64::float64x2_t,
) -> (
    core::arch::aarch64::float64x2_t,
    core::arch::aarch64::float64x2_t,
) {
    // SAFETY: AArch64 mandates NEON. Caller can rely on the cfg gate.
    unsafe { neon::sincos_f64x2(x) }
}

/// Lane-parallel `(sin, cos)` for four f64 inputs packed in an
/// `__m256d`. x86_64-only.
///
/// See [`avx2::sincos_f64x4`] for accuracy notes. Lanes outside
/// [`CODY_WAITE_LIMIT`] use the scalar fallback (mid-range reduction,
/// Payne–Hanek, or `f64::sin_cos` for non-finite inputs).
///
/// # Safety
///
/// Caller must guarantee the host CPU has AVX2 + FMA support. Use
/// `std::is_x86_feature_detected!("avx2")` and
/// `std::is_x86_feature_detected!("fma")` for runtime dispatch.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn sincos_f64x4(
    x: core::arch::x86_64::__m256d,
) -> (core::arch::x86_64::__m256d, core::arch::x86_64::__m256d) {
    // SAFETY: caller-asserted AVX2 + FMA. The cfg gate restricts the
    // function to x86_64 build targets; runtime detection is the
    // caller's responsibility.
    unsafe { avx2::sincos_f64x4(x) }
}
