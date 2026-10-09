//! Public scalar and slice-level phase primitives.
//!
//! This module keeps the sign convention and derivative relationship for
//! oscillatory kernels in one place. `exp_i(x)` means `exp(+i x)`, so its
//! derivative is `i exp(+i x)`. The slice APIs validate all output lengths and
//! dispatch by architecture once per slice, leaving only a scalar tail.

use num_complex::{Complex32, Complex64};
use std::sync::OnceLock;

use crate::scalar;
use crate::scalar::sincos_scalar;
pub use crate::scalar::{
    mid_path_count, reset_mid_path_count, reset_slow_path_count, slow_path_count,
};

/// SIMD implementation selected by the public slice APIs.
///
/// The selection is made once per slice from runtime CPU feature detection on
/// x86_64. `Scalar` is always available and is also the fallback for targets
/// without a supported vector implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdBackend {
    Scalar,
    Neon,
    Avx2,
    Avx512,
    WasmSimd,
}

impl SimdBackend {
    /// Stable machine-readable name for reports and benchmark artifacts.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::WasmSimd => "wasm-simd128",
        }
    }
}

/// Return the backend currently selected by the public slice APIs.
pub fn selected_backend() -> SimdBackend {
    if force_scalar_dispatch() {
        return SimdBackend::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        return SimdBackend::Neon;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 is useful as an independently reproducible baseline on hosts
        // that also expose AVX-512. This override is opt-in and is read once
        // at process start so normal dispatch does not inspect the
        // environment in the hot path.
        if force_avx2_dispatch()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            return SimdBackend::Avx2;
        }
        if !force_scalar_dispatch()
            && !force_avx2_dispatch()
            && std::is_x86_feature_detected!("avx512f")
        {
            return SimdBackend::Avx512;
        }
        if !force_scalar_dispatch()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            return SimdBackend::Avx2;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return SimdBackend::WasmSimd;
    }
    SimdBackend::Scalar
}

fn force_scalar_dispatch() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| {
        std::env::var_os("SONIUM_FASTMATH_FORCE_SCALAR")
            .is_some_and(|value| value == "1" || value == "true")
    })
}

fn force_avx2_dispatch() -> bool {
    static FORCE_AVX2: OnceLock<bool> = OnceLock::new();
    *FORCE_AVX2.get_or_init(|| {
        std::env::var_os("SONIUM_FASTMATH_FORCE_AVX2")
            .is_some_and(|value| value == "1" || value == "true")
    })
}

/// Return the runtime feature facts used by the public slice dispatcher.
///
/// The string is stable and machine-readable so accuracy artifacts can
/// distinguish a forced scalar baseline from an AVX2 run. The
/// `target_features` field in [`AccuracyReport`] describes compile-time flags;
/// this function also describes the runtime capability checks.
/// Cached x86_64 runtime capability bits. `is_x86_feature_detected!` reads
/// CPUID on every call; `accuracy_report` invokes [`dispatch_features`]
/// once per report, so cache the probe in a `OnceLock` instead of
/// re-executing it per call.
#[cfg(target_arch = "x86_64")]
fn x86_capabilities() -> (bool, bool, bool) {
    static CAPS: OnceLock<(bool, bool, bool)> = OnceLock::new();
    *CAPS.get_or_init(|| {
        (
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("avx512f"),
        )
    })
}

pub fn dispatch_features() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let (avx2, fma, avx512f) = x86_capabilities();
        format!(
            "arch=x86_64;avx2={avx2};fma={fma};avx512f={avx512f};force_scalar={};force_avx2={}",
            force_scalar_dispatch(),
            force_avx2_dispatch(),
        )
    }
    #[cfg(target_arch = "aarch64")]
    {
        return format!(
            "arch=aarch64;neon=true;force_scalar={};force_avx2={}",
            force_scalar_dispatch(),
            force_avx2_dispatch(),
        );
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return format!(
            "arch=wasm32;simd128=true;force_scalar={};force_avx2={}",
            force_scalar_dispatch(),
            force_avx2_dispatch(),
        );
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        "arch=unknown;scalar=true".to_owned()
    }
}

/// Accuracy policy for elementary oscillatory functions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccuracyProfile {
    /// Use the Cody–Waite/Sleef fast path within its published domain and the
    /// robust fallback outside it.
    Fast,
    /// Use the platform `f64::sin_cos` / `f32::sin_cos` implementation for
    /// reference-quality values.
    Reference,
}

/// A sine/cosine pair together with its analytic first derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinCosDerivatives {
    pub sin: f64,
    pub cos: f64,
    pub d_sin: f64,
    pub d_cos: f64,
}

/// Machine-readable accuracy summary for a supplied phase sample set.
#[derive(Clone, Debug, PartialEq)]
pub struct AccuracyReport {
    pub architecture: &'static str,
    pub compiler: &'static str,
    pub target_features: &'static str,
    /// Runtime CPU feature detection and explicit dispatch overrides.
    pub runtime_features: String,
    pub dispatch: &'static str,
    pub samples: usize,
    /// Maximum error rounded to units of `EPSILON` at magnitude one. This is
    /// a normalized absolute error metric, not result-spacing ULP distance.
    pub max_sin_error_eps: u64,
    /// Maximum error rounded to units of `EPSILON` at magnitude one.
    pub max_cos_error_eps: u64,
    /// 95th percentile normalized absolute error in `EPSILON` units.
    pub p95_sin_error_eps: u64,
    /// 95th percentile normalized absolute error in `EPSILON` units.
    pub p95_cos_error_eps: u64,
    /// Maximum conventional representable-float distance for reference
    /// values whose magnitude exceeds `RESULT_ULP_ROOT_FLOOR`.
    pub max_sin_result_ulp: u64,
    /// Maximum conventional representable-float distance for reference
    /// values whose magnitude exceeds `RESULT_ULP_ROOT_FLOOR`.
    pub max_cos_result_ulp: u64,
    /// 95th percentile conventional ULP distance, excluding reference values
    /// at or below `RESULT_ULP_ROOT_FLOOR` in magnitude.
    pub p95_sin_result_ulp: u64,
    /// 95th percentile conventional ULP distance, excluding reference values
    /// at or below `RESULT_ULP_ROOT_FLOOR` in magnitude.
    pub p95_cos_result_ulp: u64,
    /// Number of sine reference observations used for result-ULP stats.
    pub sin_result_ulp_samples: usize,
    /// Number of cosine reference observations used for result-ULP stats.
    pub cos_result_ulp_samples: usize,
    /// Legacy aliases for the normalized absolute-error metrics. Use the
    /// `*_error_eps` fields instead.
    #[deprecated(note = "use max_sin_error_eps; this is not result-spacing ULP")]
    pub max_sin_ulp: u64,
    #[deprecated(note = "use max_cos_error_eps; this is not result-spacing ULP")]
    pub max_cos_ulp: u64,
    #[deprecated(note = "use p95_sin_error_eps; this is not result-spacing ULP")]
    pub p95_sin_ulp: u64,
    #[deprecated(note = "use p95_cos_error_eps; this is not result-spacing ULP")]
    pub p95_cos_ulp: u64,
    pub max_sin_abs_error: f64,
    pub max_cos_abs_error: f64,
    pub max_sin_relative_error: f64,
    pub max_cos_relative_error: f64,
    /// Derivative errors for `d sin(x)/dx = cos(x)` and
    /// `d cos(x)/dx = -sin(x)` over the same sample set.
    pub max_d_sin_abs_error: f64,
    pub max_d_cos_abs_error: f64,
    pub max_d_sin_relative_error: f64,
    pub max_d_cos_relative_error: f64,
    pub near_root_samples: usize,
    pub slow_path_calls: u64,
}

/// Escape a short internal string for embedding in [`AccuracyReport::to_json`].
///
/// All current producers emit `[A-Za-z0-9_=;.,+-]*`, so this is purely
/// defensive: it keeps the hand-rolled serializer well-formed if a future
/// compiler or feature string ever carries a quote, backslash, or control
/// character, without pulling `serde_json` into the hot-path crate.
fn escape_json_str(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

impl AccuracyReport {
    /// Serialize the report as a single JSON object without requiring a
    /// serialization dependency in the hot-path crate.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"schema\":\"sonium.fastmath.accuracy.v2\",\"architecture\":\"{}\",\"compiler\":\"{}\",\"target_features\":\"{}\",\"runtime_features\":\"{}\",\"dispatch\":\"{}\",\"samples\":{},\"max_sin_error_eps\":{},\"max_cos_error_eps\":{},\"p95_sin_error_eps\":{},\"p95_cos_error_eps\":{},\"max_sin_result_ulp\":{},\"max_cos_result_ulp\":{},\"p95_sin_result_ulp\":{},\"p95_cos_result_ulp\":{},\"result_ulp_root_floor\":{:.1e},\"sin_result_ulp_samples\":{},\"cos_result_ulp_samples\":{},\"max_sin_abs_error\":{:.17e},\"max_cos_abs_error\":{:.17e},\"max_sin_relative_error\":{:.17e},\"max_cos_relative_error\":{:.17e},\"max_d_sin_abs_error\":{:.17e},\"max_d_cos_abs_error\":{:.17e},\"max_d_sin_relative_error\":{:.17e},\"max_d_cos_relative_error\":{:.17e},\"near_root_samples\":{},\"slow_path_calls\":{}}}",
            escape_json_str(self.architecture),
            escape_json_str(self.compiler),
            escape_json_str(self.target_features),
            escape_json_str(&self.runtime_features),
            escape_json_str(self.dispatch),
            self.samples,
            self.max_sin_error_eps,
            self.max_cos_error_eps,
            self.p95_sin_error_eps,
            self.p95_cos_error_eps,
            self.max_sin_result_ulp,
            self.max_cos_result_ulp,
            self.p95_sin_result_ulp,
            self.p95_cos_result_ulp,
            RESULT_ULP_ROOT_FLOOR,
            self.sin_result_ulp_samples,
            self.cos_result_ulp_samples,
            self.max_sin_abs_error,
            self.max_cos_abs_error,
            self.max_sin_relative_error,
            self.max_cos_relative_error,
            self.max_d_sin_abs_error,
            self.max_d_cos_abs_error,
            self.max_d_sin_relative_error,
            self.max_d_cos_relative_error,
            self.near_root_samples,
            self.slow_path_calls,
        )
    }
}

/// Measure the fast API against platform libm and return a report suitable
/// for CI artifacts or benchmark logs.
pub fn accuracy_report(input: &[f64]) -> AccuracyReport {
    // Exercise the same once-per-slice architecture dispatch used by callers
    // rather than reporting scalar-only measurements under an architecture
    // label. The report is an offline diagnostic, so these two work buffers
    // are intentional; the hot-path slice APIs remain allocation-free.
    let mut sins = vec![0.0_f64; input.len()];
    let mut coss = vec![0.0_f64; input.len()];
    sincos_f64_slice(input, &mut sins, &mut coss);

    let mut sin_error_eps = Vec::with_capacity(input.len());
    let mut cos_error_eps = Vec::with_capacity(input.len());
    let mut sin_result_ulp = Vec::with_capacity(input.len());
    let mut cos_result_ulp = Vec::with_capacity(input.len());
    let mut max_sin_abs_error = 0.0_f64;
    let mut max_cos_abs_error = 0.0_f64;
    let mut max_sin_relative_error = 0.0_f64;
    let mut max_cos_relative_error = 0.0_f64;
    let mut max_d_sin_abs_error = 0.0_f64;
    let mut max_d_cos_abs_error = 0.0_f64;
    let mut max_d_sin_relative_error = 0.0_f64;
    let mut max_d_cos_relative_error = 0.0_f64;
    let mut near_root_samples = 0;
    let mut slow_path_calls = 0;
    for (index, &x) in input.iter().enumerate() {
        if x != 0.0 && (!x.is_finite() || x.abs() > scalar::CODY_WAITE_LIMIT) {
            slow_path_calls += 1;
        }
        let sin = sins[index];
        let cos = coss[index];
        let (sin_ref, cos_ref) = x.sin_cos();
        sin_error_eps.push(error_in_eps_at_one(sin, sin_ref));
        cos_error_eps.push(error_in_eps_at_one(cos, cos_ref));
        if let Some(distance) = result_ulp_distance(sin, sin_ref) {
            sin_result_ulp.push(distance);
        }
        if let Some(distance) = result_ulp_distance(cos, cos_ref) {
            cos_result_ulp.push(distance);
        }
        if sin.is_finite() && sin_ref.is_finite() {
            let abs_error = (sin - sin_ref).abs();
            max_sin_abs_error = max_sin_abs_error.max(abs_error);
            if sin_ref.abs() > 1.0e-12 {
                max_sin_relative_error = max_sin_relative_error.max(abs_error / sin_ref.abs());
            }
        }
        if cos.is_finite() && cos_ref.is_finite() {
            let abs_error = (cos - cos_ref).abs();
            max_cos_abs_error = max_cos_abs_error.max(abs_error);
            max_d_sin_abs_error = max_d_sin_abs_error.max(abs_error);
            if cos_ref.abs() > 1.0e-12 {
                let relative_error = abs_error / cos_ref.abs();
                max_cos_relative_error = max_cos_relative_error.max(relative_error);
                max_d_sin_relative_error = max_d_sin_relative_error.max(relative_error);
            }
        }
        if sin.is_finite() && sin_ref.is_finite() {
            let abs_error = (-sin - -sin_ref).abs();
            max_d_cos_abs_error = max_d_cos_abs_error.max(abs_error);
            if sin_ref.abs() > 1.0e-12 {
                max_d_cos_relative_error = max_d_cos_relative_error.max(abs_error / sin_ref.abs());
            }
        }
        if sin_ref.abs() <= 1.0e-12 || cos_ref.abs() <= 1.0e-12 {
            near_root_samples += 1;
        }
    }
    sin_error_eps.sort_unstable();
    cos_error_eps.sort_unstable();
    sin_result_ulp.sort_unstable();
    cos_result_ulp.sort_unstable();
    let percentile = |values: &[u64], p: usize| -> u64 {
        if values.is_empty() {
            0
        } else {
            values[((values.len() - 1) * p) / 100]
        }
    };
    #[allow(deprecated)]
    AccuracyReport {
        architecture: std::env::consts::ARCH,
        compiler: env!("SONIUM_RUSTC_VERSION"),
        target_features: env!("SONIUM_TARGET_FEATURES"),
        runtime_features: dispatch_features(),
        dispatch: selected_backend().as_str(),
        samples: input.len(),
        max_sin_error_eps: sin_error_eps.last().copied().unwrap_or(0),
        max_cos_error_eps: cos_error_eps.last().copied().unwrap_or(0),
        p95_sin_error_eps: percentile(&sin_error_eps, 95),
        p95_cos_error_eps: percentile(&cos_error_eps, 95),
        max_sin_result_ulp: sin_result_ulp.last().copied().unwrap_or(0),
        max_cos_result_ulp: cos_result_ulp.last().copied().unwrap_or(0),
        p95_sin_result_ulp: percentile(&sin_result_ulp, 95),
        p95_cos_result_ulp: percentile(&cos_result_ulp, 95),
        sin_result_ulp_samples: sin_result_ulp.len(),
        cos_result_ulp_samples: cos_result_ulp.len(),
        max_sin_ulp: sin_error_eps.last().copied().unwrap_or(0),
        max_cos_ulp: cos_error_eps.last().copied().unwrap_or(0),
        p95_sin_ulp: percentile(&sin_error_eps, 95),
        p95_cos_ulp: percentile(&cos_error_eps, 95),
        max_sin_abs_error,
        max_cos_abs_error,
        max_sin_relative_error,
        max_cos_relative_error,
        max_d_sin_abs_error,
        max_d_cos_abs_error,
        max_d_sin_relative_error,
        max_d_cos_relative_error,
        near_root_samples,
        slow_path_calls,
    }
}

/// Evaluate sine and cosine using the selected accuracy profile.
#[inline]
pub fn sincos_with_profile(x: f64, profile: AccuracyProfile) -> (f64, f64) {
    match profile {
        AccuracyProfile::Fast => sincos_scalar(x),
        AccuracyProfile::Reference => x.sin_cos(),
    }
}

/// Evaluate `sin(x)`, `cos(x)`, `d sin(x)/dx`, and `d cos(x)/dx` from one
/// shared phase evaluation.
#[inline]
pub fn sincos_with_derivatives(x: f64) -> SinCosDerivatives {
    let (sin, cos) = sincos_scalar(x);
    SinCosDerivatives {
        sin,
        cos,
        d_sin: cos,
        d_cos: -sin,
    }
}

/// f32 counterpart of [`sincos_with_derivatives`].
#[inline]
pub fn sincos_f32_with_derivatives(x: f32) -> (f32, f32, f32, f32) {
    let (sin, cos) = sincos_f32(x);
    (sin, cos, cos, -sin)
}

/// Evaluate sine and cosine for an f32 phase.
///
/// The f64 fast path is used as the normative implementation and rounded
/// once to f32. This gives deterministic behaviour on targets without a
/// native f32 vector kernel while preserving f32 input semantics.
#[inline]
pub fn sincos_f32(x: f32) -> (f32, f32) {
    let (sin, cos) = sincos_scalar(x as f64);
    (sin as f32, cos as f32)
}

/// Compute `exp(+i x)` as `(cos(x), sin(x))`.
#[inline]
pub fn exp_i(x: f64) -> Complex64 {
    exp_i_with_profile(x, AccuracyProfile::Fast)
}

/// Compute `exp(+i x)` using the requested accuracy profile.
///
/// Assembly callers can retain the vectorized fast profile, while residual
/// verification and adjoint/Taylor checks can select the platform reference
/// path without implementing a second phase convention.
#[inline]
pub fn exp_i_with_profile(x: f64, profile: AccuracyProfile) -> Complex64 {
    let (sin, cos) = sincos_with_profile(x, profile);
    Complex64::new(cos, sin)
}

/// f32 counterpart of [`exp_i`].
#[inline]
pub fn exp_i_f32(x: f32) -> Complex32 {
    exp_i_f32_with_profile(x, AccuracyProfile::Fast)
}

/// f32 counterpart of [`exp_i_with_profile`].
#[inline]
pub fn exp_i_f32_with_profile(x: f32, profile: AccuracyProfile) -> Complex32 {
    let (sin, cos) = match profile {
        AccuracyProfile::Fast => sincos_f32(x),
        AccuracyProfile::Reference => x.sin_cos(),
    };
    Complex32::new(cos, sin)
}

/// Return `exp(+i x)` and its derivative with respect to `x`.
#[inline]
pub fn exp_i_with_derivative(x: f64) -> (Complex64, Complex64) {
    exp_i_with_derivative_profile(x, AccuracyProfile::Fast)
}

/// Return `exp(+i x)` and its derivative using one selected phase contract.
#[inline]
pub fn exp_i_with_derivative_profile(x: f64, profile: AccuracyProfile) -> (Complex64, Complex64) {
    let value = exp_i_with_profile(x, profile);
    (value, Complex64::new(-value.im, value.re))
}

/// Fill f64 phasors and their phase derivatives without allocating.
///
/// Each lane evaluates sine and cosine exactly once. The derivative is
/// `d exp(+ix)/dx = i exp(+ix)`, so callers can use the same dispatched phase
/// evaluation for values and analytic kernel derivatives.
pub fn exp_i_with_derivative_slice(
    input: &[f64],
    values: &mut [Complex64],
    derivatives: &mut [Complex64],
) {
    assert_eq!(input.len(), values.len(), "phasor output length mismatch");
    assert_eq!(
        input.len(),
        derivatives.len(),
        "phasor derivative output length mismatch"
    );
    for_each_sincos_f64(input, |i, sin, cos| {
        values[i] = Complex64::new(cos, sin);
        derivatives[i] = Complex64::new(-sin, cos);
    });
}

/// Fill f64 phasors and derivatives using a selected accuracy profile.
///
/// `Fast` retains the architecture-dispatched allocation-free path. `Reference`
/// uses platform `sin_cos` for residual and derivative verification, while
/// preserving the same positive-exponent phasor and derivative convention.
pub fn exp_i_with_derivative_slice_with_profile(
    input: &[f64],
    values: &mut [Complex64],
    derivatives: &mut [Complex64],
    profile: AccuracyProfile,
) {
    assert_eq!(input.len(), values.len(), "phasor output length mismatch");
    assert_eq!(
        input.len(),
        derivatives.len(),
        "phasor derivative output length mismatch"
    );
    match profile {
        AccuracyProfile::Fast => exp_i_with_derivative_slice(input, values, derivatives),
        AccuracyProfile::Reference => {
            for ((&phase, value), derivative) in input.iter().zip(values).zip(derivatives) {
                let (sin, cos) = phase.sin_cos();
                *value = Complex64::new(cos, sin);
                *derivative = Complex64::new(-sin, cos);
            }
        }
    }
}

/// Fill f32 phasors and their phase derivatives without allocating.
///
/// This is the f32 counterpart of [`exp_i_with_derivative_slice`].
pub fn exp_i_f32_with_derivative_slice(
    input: &[f32],
    values: &mut [Complex32],
    derivatives: &mut [Complex32],
) {
    assert_eq!(input.len(), values.len(), "phasor output length mismatch");
    assert_eq!(
        input.len(),
        derivatives.len(),
        "phasor derivative output length mismatch"
    );
    for_each_sincos_f32(input, |i, sin, cos| {
        values[i] = Complex32::new(cos, sin);
        derivatives[i] = Complex32::new(-sin, cos);
    });
}

/// Fill f64 sine/cosine arrays without allocating.
pub fn sincos_f64_slice(input: &[f64], sins: &mut [f64], coss: &mut [f64]) {
    assert_eq!(input.len(), sins.len(), "sine output length mismatch");
    assert_eq!(input.len(), coss.len(), "cosine output length mismatch");
    for_each_sincos_f64(input, |i, sin, cos| {
        sins[i] = sin;
        coss[i] = cos;
    });
}

/// Fill f32 sine/cosine arrays without allocating.
pub fn sincos_f32_slice(input: &[f32], sins: &mut [f32], coss: &mut [f32]) {
    assert_eq!(input.len(), sins.len(), "sine output length mismatch");
    assert_eq!(input.len(), coss.len(), "cosine output length mismatch");
    for_each_sincos_f32(input, |i, sin, cos| {
        sins[i] = sin;
        coss[i] = cos;
    });
}

fn for_each_sincos_f32<F>(input: &[f32], mut visit: F)
where
    F: FnMut(usize, f32, f32),
{
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        while i + 4 <= input.len() {
            // SAFETY: AArch64 mandates NEON.
            unsafe {
                let x = vld1q_f32(input.as_ptr().add(i));
                let (sv, cv) = crate::sincos_f32x4(x);
                visit(i, vgetq_lane_f32::<0>(sv), vgetq_lane_f32::<0>(cv));
                visit(i + 1, vgetq_lane_f32::<1>(sv), vgetq_lane_f32::<1>(cv));
                visit(i + 2, vgetq_lane_f32::<2>(sv), vgetq_lane_f32::<2>(cv));
                visit(i + 3, vgetq_lane_f32::<3>(sv), vgetq_lane_f32::<3>(cv));
            }
            i += 4;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !force_scalar_dispatch()
            && !force_avx2_dispatch()
            && std::is_x86_feature_detected!("avx512f")
        {
            use std::arch::x86_64::*;
            while i + 16 <= input.len() {
                // SAFETY: Runtime feature check establishes AVX-512F.
                unsafe {
                    let x = _mm512_loadu_ps(input.as_ptr().add(i));
                    let (sv, cv) = crate::avx512::sincos_f32x16(x);
                    let mut sins = [0.0_f32; 16];
                    let mut coss = [0.0_f32; 16];
                    _mm512_storeu_ps(sins.as_mut_ptr(), sv);
                    _mm512_storeu_ps(coss.as_mut_ptr(), cv);
                    for lane in 0..16 {
                        visit(i + lane, sins[lane], coss[lane]);
                    }
                }
                i += 16;
            }
        }
        if !force_scalar_dispatch()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            use std::arch::x86_64::*;
            while i + 8 <= input.len() {
                // SAFETY: Runtime feature checks establish AVX2 + FMA.
                unsafe {
                    let x = _mm256_loadu_ps(input.as_ptr().add(i));
                    let (sv, cv) = crate::sincos_f32x8(x);
                    let mut sins = [0.0_f32; 8];
                    let mut coss = [0.0_f32; 8];
                    _mm256_storeu_ps(sins.as_mut_ptr(), sv);
                    _mm256_storeu_ps(coss.as_mut_ptr(), cv);
                    for lane in 0..8 {
                        visit(i + lane, sins[lane], coss[lane]);
                    }
                }
                i += 8;
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use std::arch::wasm32::*;
        while i + 4 <= input.len() {
            // SAFETY: The cfg gate proves wasm simd128 support.
            unsafe {
                let mut x = f32x4_splat(input[i]);
                x = f32x4_replace_lane::<1>(x, input[i + 1]);
                x = f32x4_replace_lane::<2>(x, input[i + 2]);
                x = f32x4_replace_lane::<3>(x, input[i + 3]);
                let (sv, cv) = crate::wasm_simd::sincos_f32x4(x);
                visit(i, f32x4_extract_lane::<0>(sv), f32x4_extract_lane::<0>(cv));
                visit(
                    i + 1,
                    f32x4_extract_lane::<1>(sv),
                    f32x4_extract_lane::<1>(cv),
                );
                visit(
                    i + 2,
                    f32x4_extract_lane::<2>(sv),
                    f32x4_extract_lane::<2>(cv),
                );
                visit(
                    i + 3,
                    f32x4_extract_lane::<3>(sv),
                    f32x4_extract_lane::<3>(cv),
                );
            }
            i += 4;
        }
    }
    while i < input.len() {
        let (sin, cos) = sincos_f32(input[i]);
        visit(i, sin, cos);
        i += 1;
    }
}

/// Fill a complex f64 phasor array with `exp(+i input[i])` without allocating.
pub fn exp_i_slice(input: &[f64], output: &mut [Complex64]) {
    assert_eq!(input.len(), output.len(), "phasor output length mismatch");
    for_each_sincos_f64(input, |i, sin, cos| {
        output[i] = Complex64::new(cos, sin);
    });
}

/// Fill a complex f32 phasor array with `exp(+i input[i])` without allocating.
pub fn exp_i_f32_slice(input: &[f32], output: &mut [Complex32]) {
    assert_eq!(input.len(), output.len(), "phasor output length mismatch");
    for_each_sincos_f32(input, |i, sin, cos| {
        output[i] = Complex32::new(cos, sin);
    });
}

fn for_each_sincos_f64<F>(input: &[f64], mut visit: F)
where
    F: FnMut(usize, f64, f64),
{
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        while i + 2 <= input.len() {
            // SAFETY: AArch64 mandates NEON and the public SIMD wrapper has
            // the corresponding target-feature contract.
            unsafe {
                let x = vsetq_lane_f64(input[i + 1], vdupq_n_f64(input[i]), 1);
                let (sv, cv) = crate::sincos_f64x2(x);
                visit(i, vgetq_lane_f64(sv, 0), vgetq_lane_f64(cv, 0));
                visit(i + 1, vgetq_lane_f64(sv, 1), vgetq_lane_f64(cv, 1));
            }
            i += 2;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !force_scalar_dispatch()
            && !force_avx2_dispatch()
            && std::is_x86_feature_detected!("avx512f")
        {
            use std::arch::x86_64::*;
            while i + 8 <= input.len() {
                // SAFETY: Runtime feature detection establishes AVX-512F.
                unsafe {
                    let x = _mm512_loadu_pd(input.as_ptr().add(i));
                    let (sv, cv) = crate::avx512::sincos_f64x8(x);
                    let mut sin = [0.0; 8];
                    let mut cos = [0.0; 8];
                    _mm512_storeu_pd(sin.as_mut_ptr(), sv);
                    _mm512_storeu_pd(cos.as_mut_ptr(), cv);
                    for lane in 0..8 {
                        visit(i + lane, sin[lane], cos[lane]);
                    }
                }
                i += 8;
            }
        }
        if !force_scalar_dispatch()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            use std::arch::x86_64::*;
            while i + 4 <= input.len() {
                // SAFETY: Runtime feature checks establish AVX2 + FMA.
                unsafe {
                    let x = _mm256_set_pd(input[i + 3], input[i + 2], input[i + 1], input[i]);
                    let (sv, cv) = crate::sincos_f64x4(x);
                    let mut sin = [0.0; 4];
                    let mut cos = [0.0; 4];
                    _mm256_storeu_pd(sin.as_mut_ptr(), sv);
                    _mm256_storeu_pd(cos.as_mut_ptr(), cv);
                    for lane in 0..4 {
                        visit(i + lane, sin[lane], cos[lane]);
                    }
                }
                i += 4;
            }
        }
    }
    while i < input.len() {
        let (sin, cos) = sincos_scalar(input[i]);
        visit(i, sin, cos);
        i += 1;
    }
}

/// Root values make result-spacing ULP error unbounded and uninformative.
const RESULT_ULP_ROOT_FLOOR: f64 = 1.0e-12;

/// Absolute error rounded to units of `EPSILON` at magnitude one.
fn error_in_eps_at_one(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return 0;
    }
    if !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    ((a - b).abs() / f64::EPSILON).round() as u64
}

/// Conventional representable-float distance, omitted near roots and for
/// non-finite values. The root exclusion is part of the serialized contract.
fn result_ulp_distance(a: f64, b: f64) -> Option<u64> {
    if !a.is_finite() || !b.is_finite() || b.abs() <= RESULT_ULP_ROOT_FLOOR {
        return None;
    }
    let ordered = |value: f64| {
        let bits = value.to_bits();
        if bits >> 63 == 0 {
            bits | (1 << 63)
        } else {
            !bits
        }
    };
    Some(ordered(a).abs_diff(ordered(b)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{reset_slow_path_count, slow_path_count};
    use std::f64::consts::PI;

    #[test]
    fn profile_and_derivative_contracts() {
        let fast = sincos_with_profile(0.7, AccuracyProfile::Fast);
        let reference = sincos_with_profile(0.7, AccuracyProfile::Reference);
        assert!((fast.0 - reference.0).abs() < 1e-14);
        let d = sincos_with_derivatives(0.7);
        assert_eq!(d.d_sin, d.cos);
        assert_eq!(d.d_cos, -d.sin);
    }

    #[test]
    fn phasor_and_derivative_use_positive_i_sign() {
        let (value, derivative) = exp_i_with_derivative(PI / 2.0);
        assert!(value.re.abs() < 1e-14);
        assert!((value.im - 1.0).abs() < 1e-14);
        assert!((derivative.re + 1.0).abs() < 1e-14);
        assert!(derivative.im.abs() < 1e-14);
    }

    #[test]
    fn phasor_profiles_keep_value_and_derivative_contracts_aligned() {
        let phase = 1.0e20_f64;
        let (sin, cos) = phase.sin_cos();
        let value = exp_i_with_profile(phase, AccuracyProfile::Reference);
        assert_eq!(value, Complex64::new(cos, sin));
        let (value, derivative) = exp_i_with_derivative_profile(phase, AccuracyProfile::Reference);
        assert_eq!(derivative, Complex64::new(-value.im, value.re));

        let phase_f32 = 1.0e10_f32;
        let (sin_f32, cos_f32) = phase_f32.sin_cos();
        assert_eq!(
            exp_i_f32_with_profile(phase_f32, AccuracyProfile::Reference),
            Complex32::new(cos_f32, sin_f32)
        );

        let input = [-1.0, 0.0, 0.7, 1.0e20];
        let mut values = [Complex64::default(); 4];
        let mut derivatives = [Complex64::default(); 4];
        exp_i_with_derivative_slice_with_profile(
            &input,
            &mut values,
            &mut derivatives,
            AccuracyProfile::Reference,
        );
        for ((&phase, value), derivative) in input.iter().zip(values).zip(derivatives) {
            let (sin, cos) = phase.sin_cos();
            assert_eq!(value, Complex64::new(cos, sin));
            assert_eq!(derivative, Complex64::new(-sin, cos));
        }

        exp_i_with_derivative_slice_with_profile(
            &input,
            &mut values,
            &mut derivatives,
            AccuracyProfile::Fast,
        );
        for ((&phase, value), derivative) in input.iter().zip(values).zip(derivatives) {
            let (expected, expected_derivative) = exp_i_with_derivative(phase);
            assert_eq!(value, expected);
            assert_eq!(derivative, expected_derivative);
        }
    }

    #[test]
    fn f32_and_slice_apis_are_allocation_free_contracts() {
        let input = [-0.5_f64, 0.0, 0.5, 1.0e6];
        let mut sins = [0.0; 4];
        let mut coss = [0.0; 4];
        let mut phasors = [Complex64::new(0.0, 0.0); 4];
        let mut derivatives = [Complex64::new(0.0, 0.0); 4];
        sincos_f64_slice(&input, &mut sins, &mut coss);
        exp_i_slice(&input, &mut phasors);
        exp_i_with_derivative_slice(&input, &mut phasors, &mut derivatives);
        for i in 0..input.len() {
            let (s, c) = input[i].sin_cos();
            assert!((sins[i] - s).abs() < 1e-12);
            assert!((coss[i] - c).abs() < 1e-12);
            assert!((phasors[i].re - c).abs() < 1e-12);
            assert!((phasors[i].im - s).abs() < 1e-12);
            assert!((derivatives[i].re + s).abs() < 1e-12);
            assert!((derivatives[i].im - c).abs() < 1e-12);
        }

        let f32_input: Vec<f32> = (0..17).map(|i| -0.5 + i as f32 * 0.37).collect();
        let mut f32_sin = vec![0.0; f32_input.len()];
        let mut f32_cos = vec![0.0; f32_input.len()];
        let mut f32_phasors = vec![Complex32::new(0.0, 0.0); f32_input.len()];
        let mut f32_derivatives = vec![Complex32::new(0.0, 0.0); f32_input.len()];
        sincos_f32_slice(&f32_input, &mut f32_sin, &mut f32_cos);
        exp_i_f32_slice(&f32_input, &mut f32_phasors);
        exp_i_f32_with_derivative_slice(&f32_input, &mut f32_phasors, &mut f32_derivatives);
        for i in 0..f32_input.len() {
            let (s, c) = f32_input[i].sin_cos();
            assert!((f32_sin[i] - s).abs() < 2e-6);
            assert!((f32_cos[i] - c).abs() < 2e-6);
            assert!((f32_phasors[i].re - c).abs() < 2e-6);
            assert!((f32_phasors[i].im - s).abs() < 2e-6);
            assert!((f32_derivatives[i].re + s).abs() < 2e-6);
            assert!((f32_derivatives[i].im - c).abs() < 2e-6);
        }
    }

    #[test]
    fn fallback_counter_tracks_huge_phases() {
        reset_slow_path_count();
        let _ = sincos_scalar(1.0e20);
        assert!(slow_path_count() >= 1);
        reset_slow_path_count();
    }

    #[test]
    fn accuracy_report_is_machine_readable() {
        let report = accuracy_report(&[-1.0, 0.0, 1.0, 1.0e20]);
        assert_eq!(report.samples, 4);
        assert_eq!(report.slow_path_calls, 1);
        assert!(report.max_sin_abs_error.is_finite());
        assert!(report.max_sin_relative_error.is_finite());
        assert!(report.max_d_sin_abs_error.is_finite());
        assert!(report.max_d_cos_relative_error.is_finite());
        assert!(report.to_json().starts_with("{\"schema\":"));
        assert!(report.to_json().contains("runtime_features"));
        assert!(report.to_json().contains("max_d_sin_abs_error"));
        assert!(report.to_json().contains("sonium.fastmath.accuracy.v2"));
        assert!(report.to_json().contains("max_sin_error_eps"));
        assert!(report.to_json().contains("max_sin_result_ulp"));
        assert!(!report.to_json().contains("\"max_sin_ulp\""));
    }

    #[test]
    #[allow(deprecated)]
    fn accuracy_report_reports_scaled_absolute_error_and_excludes_roots_from_ulp_stats() {
        let input: Vec<f64> = (-32..=32).map(|k| k as f64 * PI / 2.0 + 1.0e-12).collect();
        let report = accuracy_report(&input);
        assert!(report.near_root_samples > 0);
        assert!(report.max_sin_error_eps <= 4);
        assert!(report.max_cos_error_eps <= 4);
        assert!(report.sin_result_ulp_samples < input.len());
        assert!(report.cos_result_ulp_samples < input.len());
        assert_eq!(report.max_sin_ulp, report.max_sin_error_eps);
        assert_eq!(report.max_cos_ulp, report.max_cos_error_eps);
    }

    #[test]
    fn result_ulp_distance_uses_adjacent_representable_values() {
        let next = f64::from_bits(1.0_f64.to_bits() + 1);
        assert_eq!(result_ulp_distance(next, 1.0), Some(1));
        assert_eq!(result_ulp_distance(1.0, next), Some(1));
        assert_eq!(result_ulp_distance(f64::EPSILON, 0.0), None);
        assert_eq!(result_ulp_distance(f64::INFINITY, 1.0), None);
    }

    #[test]
    fn error_in_eps_at_one_is_scaled_absolute_error() {
        assert_eq!(error_in_eps_at_one(1.0 + 2.0 * f64::EPSILON, 1.0), 2);
        assert_eq!(error_in_eps_at_one(0.0, f64::EPSILON), 1);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn wasm_simd_slice_dispatch_covers_vector_and_scalar_tail() {
        let input: Vec<f32> = (0..9).map(|i| -0.7 + i as f32 * 0.31).collect();
        let mut sins = vec![0.0_f32; input.len()];
        let mut coss = vec![0.0_f32; input.len()];
        sincos_f32_slice(&input, &mut sins, &mut coss);
        for ((&input, &sin), &cos) in input.iter().zip(&sins).zip(&coss) {
            let (expected_sin, expected_cos) = input.sin_cos();
            assert!((sin - expected_sin).abs() < 4.0e-6);
            assert!((cos - expected_cos).abs() < 4.0e-6);
        }
    }
}
