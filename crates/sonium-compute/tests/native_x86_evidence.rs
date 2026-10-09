//! Machine-readable native-architecture SIMD parity evidence.
//!
//! The test deliberately calls the public dispatched kernels and their scalar
//! references for every supported scalar family.  It covers tails, padded
//! leading dimensions, transpose/conjugate-transpose contracts, CSR gathers,
//! and the BEM regular-pair path.  The final line is stable enough for CI or
//! the QA summarizer to archive without parsing human-oriented test output.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::cell::Cell;

use num_complex::{Complex32, Complex64};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sonium_compute::BemRegularPairIn;
use sonium_compute::backend::Transpose;
use sonium_compute::bem_cpu::{
    RegularPairInF64, regular_pair_batch, regular_pair_c32_batch_from_gpu_abi, regular_pair_scalar,
};
use sonium_compute::cpu::{blas1, blas2, blas3, spmv_simd};

const EVIDENCE_SCHEMA: u32 = 2;
const CROSS_ARCH_ABS_TOLERANCE: f64 = 5.0e-5;

thread_local! {
    static MAX_NORM_ERROR: Cell<f64> = const { Cell::new(0.0) };
    static MAX_ULP_ERROR: Cell<u64> = const { Cell::new(0) };
}

#[derive(Debug, Deserialize, Serialize)]
struct NativeCpuEvidenceArtifact {
    schema: u32,
    status: String,
    target_arch: String,
    cpu_model: String,
    microcode: String,
    compiler: String,
    features: String,
    selected_kernel: String,
    #[serde(default)]
    force_scalar: bool,
    runtime_dispatch: String,
    os: String,
    thread_policy: String,
    affinity_policy: String,
    numa_policy: String,
    deterministic_mode: String,
    kernels: String,
    bem_elements: String,
    bem_phase_regimes: String,
    bem_precisions: String,
    norm: String,
    max_norm_error: f64,
    max_ulp_error: u64,
    #[serde(default)]
    aarch64_comparison_max_abs_error: Option<f64>,
    #[serde(default)]
    physical_validation: PhysicalValidation,
    cases_passed: usize,
    cases_skipped: usize,
    cases_failed: usize,
    observables: Vec<f64>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct PhysicalValidation {
    status: String,
    budget: String,
    reason: String,
}

fn reset_accuracy_summary() {
    MAX_NORM_ERROR.with(|value| value.set(0.0));
    MAX_ULP_ERROR.with(|value| value.set(0));
}

fn ordered_f64_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) != 0 {
        !bits
    } else {
        bits | (1_u64 << 63)
    }
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & (1_u32 << 31) != 0 {
        !bits
    } else {
        bits | (1_u32 << 31)
    }
}

fn ulp_distance_f64(actual: f64, expected: f64) -> u64 {
    if !actual.is_finite() || !expected.is_finite() {
        return u64::MAX;
    }
    ordered_f64_bits(actual).abs_diff(ordered_f64_bits(expected))
}

fn ulp_distance_f32(actual: f32, expected: f32) -> u64 {
    if !actual.is_finite() || !expected.is_finite() {
        return u64::MAX;
    }
    u64::from(ordered_f32_bits(actual).abs_diff(ordered_f32_bits(expected)))
}

fn record_accuracy_with_ulp(actual: f64, expected: f64, ulp: u64) {
    if actual.is_finite() && expected.is_finite() {
        MAX_NORM_ERROR.with(|value| value.set(value.get().max((actual - expected).abs())));
    }
    MAX_ULP_ERROR.with(|value| value.set(value.get().max(ulp)));
}

fn accuracy_summary() -> (f64, u64) {
    (
        MAX_NORM_ERROR.with(Cell::get),
        MAX_ULP_ERROR.with(Cell::get),
    )
}

trait AccuracyScalar: Copy + std::fmt::Debug {
    fn as_f64(self) -> f64;
    fn ulp_distance(self, expected: Self) -> u64;
}

impl AccuracyScalar for f32 {
    fn as_f64(self) -> f64 {
        f64::from(self)
    }

    fn ulp_distance(self, expected: Self) -> u64 {
        ulp_distance_f32(self, expected)
    }
}

impl AccuracyScalar for f64 {
    fn as_f64(self) -> f64 {
        self
    }

    fn ulp_distance(self, expected: Self) -> u64 {
        ulp_distance_f64(self, expected)
    }
}

fn assert_close<T: AccuracyScalar>(actual: T, expected: T, tolerance: f64, label: &str) {
    let actual_f64 = actual.as_f64();
    let expected_f64 = expected.as_f64();
    record_accuracy_with_ulp(actual_f64, expected_f64, actual.ulp_distance(expected));
    assert!(
        (actual_f64 - expected_f64).abs() <= tolerance,
        "{label}: actual={actual:?}, expected={expected:?}, tolerance={tolerance}"
    );
}

trait AccuracyComplex: Copy + std::fmt::Debug {
    fn error_norm(self, expected: Self) -> f64;
    fn parts(self) -> (f64, f64);
    fn part_ulp_distance(self, expected: Self) -> (u64, u64);
}

impl AccuracyComplex for Complex32 {
    fn error_norm(self, expected: Self) -> f64 {
        f64::from((self - expected).norm())
    }

    fn parts(self) -> (f64, f64) {
        (f64::from(self.re), f64::from(self.im))
    }

    fn part_ulp_distance(self, expected: Self) -> (u64, u64) {
        (
            ulp_distance_f32(self.re, expected.re),
            ulp_distance_f32(self.im, expected.im),
        )
    }
}

impl AccuracyComplex for Complex64 {
    fn error_norm(self, expected: Self) -> f64 {
        (self - expected).norm()
    }

    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }

    fn part_ulp_distance(self, expected: Self) -> (u64, u64) {
        (
            ulp_distance_f64(self.re, expected.re),
            ulp_distance_f64(self.im, expected.im),
        )
    }
}

fn assert_complex_close<T: AccuracyComplex>(actual: T, expected: T, tolerance: f64, label: &str) {
    let (actual_re, actual_im) = actual.parts();
    let (expected_re, expected_im) = expected.parts();
    let (re_ulp, im_ulp) = actual.part_ulp_distance(expected);
    record_accuracy_with_ulp(actual_re, expected_re, re_ulp);
    record_accuracy_with_ulp(actual_im, expected_im, im_ulp);
    assert!(
        actual.error_norm(expected) <= tolerance,
        "{label}: actual={actual:?}, expected={expected:?}, error={}, tolerance={tolerance}",
        actual.error_norm(expected)
    );
}

fn selected_kernel(native_avx2_fma: bool, force_scalar: bool) -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "neon"
    } else if force_scalar {
        "scalar-fallback"
    } else if native_avx2_fma {
        "avx2-fma"
    } else {
        "scalar-fallback"
    }
}

fn stored_matrix(rows: usize, _cols: usize, lda: usize, scale: f64) -> Vec<f64> {
    (0..rows * lda)
        .map(|index| ((index * 17 + 3) % 29) as f64 * scale - 0.5)
        .collect()
}

fn cpuinfo_value(field: &str) -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == field).then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unknown".into())
}

fn compiler_version() -> String {
    if let Some(version) = option_env!("RUSTC_VERSION") {
        return version.to_owned();
    }
    std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.lines().next().unwrap_or("unknown").to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn require_avx2_fma_if_requested() {
    if std::env::var("SONIUM_REQUIRE_AVX2_FMA").as_deref() != Ok("1") {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        assert_ne!(
            std::env::var("SONIUM_COMPUTE_FORCE_SCALAR").as_deref(),
            Ok("1"),
            "SONIUM_REQUIRE_AVX2_FMA=1 cannot be combined with forced scalar dispatch"
        );
        assert!(
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            "SONIUM_REQUIRE_AVX2_FMA=1 but runtime AVX2/FMA is unavailable"
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    panic!("SONIUM_REQUIRE_AVX2_FMA=1 requires an x86_64 test target");
}

fn append_f32(observables: &mut Vec<f64>, values: &[f32]) {
    observables.extend(values.iter().map(|&value| value as f64));
}

fn append_f64(observables: &mut Vec<f64>, values: &[f64]) {
    observables.extend(values.iter().copied());
}

fn append_c32(observables: &mut Vec<f64>, values: &[Complex32]) {
    observables.extend(
        values
            .iter()
            .flat_map(|value| [value.re as f64, value.im as f64]),
    );
}

fn append_c64(observables: &mut Vec<f64>, values: &[Complex64]) {
    observables.extend(values.iter().flat_map(|value| [value.re, value.im]));
}

fn append_pair_outputs(
    observables: &mut Vec<f64>,
    value: &(Complex64, Complex64, Complex64, Complex64),
) {
    append_c64(observables, &[value.0, value.1, value.2, value.3]);
}

fn assert_pair_batch_parity(
    observables: &mut Vec<f64>,
    pairs: &[RegularPairInF64],
    wave_number: f64,
    harmonic_factor: f64,
    label: &str,
) {
    let actual = regular_pair_batch(pairs, wave_number, harmonic_factor);
    assert_eq!(actual.len(), pairs.len(), "{label}: output length");
    for (index, (actual, pair)) in actual.iter().zip(pairs).enumerate() {
        let expected = regular_pair_scalar(pair, wave_number, harmonic_factor);
        for (component, (actual, expected)) in [actual.0, actual.1, actual.2, actual.3]
            .into_iter()
            .zip([expected.0, expected.1, expected.2, expected.3])
            .enumerate()
        {
            let tolerance = 8.0e-12 * expected.norm().max(1.0);
            assert_complex_close(
                actual,
                expected,
                tolerance,
                &format!("{label} pair {index} component {component}"),
            );
        }
        append_pair_outputs(observables, actual);
    }
}

fn artifact_checksum(artifact: &NativeCpuEvidenceArtifact) -> String {
    let bytes = serde_json::to_vec(artifact).expect("native CPU evidence serializes");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn compare_aarch64(
    artifact: &NativeCpuEvidenceArtifact,
    path: &std::path::Path,
) -> Result<f64, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read accepted aarch64 evidence {}: {error}", path.display()))?;
    let reference: NativeCpuEvidenceArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode accepted aarch64 evidence: {error}"))?;
    if reference.schema != EVIDENCE_SCHEMA
        || reference.target_arch != "aarch64"
        || reference.bem_elements != artifact.bem_elements
        || reference.bem_phase_regimes != artifact.bem_phase_regimes
        || reference.bem_precisions != artifact.bem_precisions
        || reference.norm != artifact.norm
        || reference.observables.len() != artifact.observables.len()
    {
        return Err("accepted aarch64 evidence schema or observable shape mismatch".into());
    }
    let maximum = artifact
        .observables
        .iter()
        .zip(&reference.observables)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    if maximum > CROSS_ARCH_ABS_TOLERANCE {
        return Err(format!(
            "native x86/aarch64 observable error {maximum:.6e} exceeds {CROSS_ARCH_ABS_TOLERANCE:.6e}"
        ));
    }
    Ok(maximum)
}

#[test]
fn runtime_dispatch_exposes_scalar_fallback_without_avx2_fma() {
    #[cfg(target_arch = "x86_64")]
    assert_eq!(selected_kernel(false, false), "scalar-fallback");
    #[cfg(target_arch = "aarch64")]
    assert_eq!(selected_kernel(false, false), "neon");
}

#[test]
fn native_cpu_simd_parity_evidence_v2() {
    require_avx2_fma_if_requested();
    reset_accuracy_summary();
    let mut observables = Vec::new();
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2");
    #[cfg(target_arch = "x86_64")]
    let fma = std::is_x86_feature_detected!("fma");
    #[cfg(target_arch = "x86_64")]
    let native_avx2_fma = avx2 && fma;
    #[cfg(target_arch = "aarch64")]
    let native_avx2_fma = true;

    let force_scalar = std::env::var("SONIUM_COMPUTE_FORCE_SCALAR")
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);

    // BLAS-1: odd lengths exercise all AVX2 tails and all four scalar types.
    let x_f32: Vec<f32> = (0..19).map(|i| i as f32 * 0.125 - 1.0).collect();
    let mut y_f32 = x_f32.iter().map(|v| v * 0.75).collect::<Vec<_>>();
    let mut y_f32_ref = y_f32.clone();
    blas1::axpy_f32(-0.375, &x_f32, &mut y_f32);
    blas1::scalar::axpy_f32(-0.375, &x_f32, &mut y_f32_ref);
    for (i, (&actual, &expected)) in y_f32.iter().zip(&y_f32_ref).enumerate() {
        assert_close(actual, expected, 2.0e-6, &format!("blas1 f32 axpy[{i}]"));
    }

    let x_f64: Vec<f64> = (0..19).map(|i| i as f64 * 0.125 - 1.0).collect();
    let mut y_f64 = x_f64.iter().map(|v| v * 0.75).collect::<Vec<_>>();
    let mut y_f64_ref = y_f64.clone();
    blas1::axpy_f64(-0.375, &x_f64, &mut y_f64);
    blas1::scalar::axpy_f64(-0.375, &x_f64, &mut y_f64_ref);
    for (i, (&actual, &expected)) in y_f64.iter().zip(&y_f64_ref).enumerate() {
        assert_close(actual, expected, 2.0e-14, &format!("blas1 f64 axpy[{i}]"));
    }

    let x_c32: Vec<Complex32> = x_f32
        .iter()
        .enumerate()
        .map(|(i, &value)| Complex32::new(value, i as f32 * 0.03125))
        .collect();
    let mut y_c32 = x_c32.clone();
    let mut y_c32_ref = y_c32.clone();
    blas1::axpy_c32(Complex32::new(-0.375, 0.125), &x_c32, &mut y_c32);
    blas1::scalar::axpy_c32(Complex32::new(-0.375, 0.125), &x_c32, &mut y_c32_ref);
    for (i, (&actual, &expected)) in y_c32.iter().zip(&y_c32_ref).enumerate() {
        assert_complex_close(actual, expected, 2.0e-5, &format!("blas1 c32 axpy[{i}]"));
    }

    let x_c64: Vec<Complex64> = x_f64
        .iter()
        .enumerate()
        .map(|(i, &value)| Complex64::new(value, i as f64 * 0.03125))
        .collect();
    let mut y_c64 = x_c64.clone();
    let mut y_c64_ref = y_c64.clone();
    blas1::axpy_c64(Complex64::new(-0.375, 0.125), &x_c64, &mut y_c64);
    blas1::scalar::axpy_c64(Complex64::new(-0.375, 0.125), &x_c64, &mut y_c64_ref);
    for (i, (&actual, &expected)) in y_c64.iter().zip(&y_c64_ref).enumerate() {
        assert_complex_close(actual, expected, 2.0e-14, &format!("blas1 c64 axpy[{i}]"));
    }
    append_f32(&mut observables, &y_f32);
    append_f64(&mut observables, &y_f64);
    append_c32(&mut observables, &y_c32);
    append_c64(&mut observables, &y_c64);
    let dot_f64 = blas1::dot_f64(&x_f64, &y_f64);
    assert_close(
        dot_f64,
        blas1::scalar::dot_f64(&x_f64, &y_f64),
        2.0e-13,
        "blas1 f64 dot",
    );
    let dotc_c64 = blas1::dotc_c64(&x_c64, &y_c64);
    assert_complex_close(
        dotc_c64,
        blas1::scalar::dotc_c64(&x_c64, &y_c64),
        2.0e-13,
        "blas1 c64 dotc",
    );
    append_f64(&mut observables, &[dot_f64]);
    append_c64(&mut observables, &[dotc_c64]);

    // Complete BLAS-1 coverage: scale, norm, and both real/complex dot
    // contracts for every dispatched scalar family.
    let mut scaled_f32 = x_f32.clone();
    let mut scaled_f32_ref = scaled_f32.clone();
    blas1::scal_f32(-0.625, &mut scaled_f32);
    blas1::scalar::scal_f32(-0.625, &mut scaled_f32_ref);
    for (i, (&actual, &expected)) in scaled_f32.iter().zip(&scaled_f32_ref).enumerate() {
        assert_close(actual, expected, 2.0e-6, &format!("blas1 f32 scal[{i}]"));
    }
    let mut scaled_f64 = x_f64.clone();
    let mut scaled_f64_ref = scaled_f64.clone();
    blas1::scal_f64(-0.625, &mut scaled_f64);
    blas1::scalar::scal_f64(-0.625, &mut scaled_f64_ref);
    for (i, (&actual, &expected)) in scaled_f64.iter().zip(&scaled_f64_ref).enumerate() {
        assert_close(actual, expected, 2.0e-14, &format!("blas1 f64 scal[{i}]"));
    }
    let mut scaled_c32 = x_c32.clone();
    let mut scaled_c32_ref = scaled_c32.clone();
    let scale_c32 = Complex32::new(-0.625, 0.1875);
    blas1::scal_c32(scale_c32, &mut scaled_c32);
    blas1::scalar::scal_c32(scale_c32, &mut scaled_c32_ref);
    for (i, (&actual, &expected)) in scaled_c32.iter().zip(&scaled_c32_ref).enumerate() {
        assert_complex_close(actual, expected, 2.0e-5, &format!("blas1 c32 scal[{i}]"));
    }
    let mut scaled_c64 = x_c64.clone();
    let mut scaled_c64_ref = scaled_c64.clone();
    let scale_c64 = Complex64::new(-0.625, 0.1875);
    blas1::scal_c64(scale_c64, &mut scaled_c64);
    blas1::scalar::scal_c64(scale_c64, &mut scaled_c64_ref);
    for (i, (&actual, &expected)) in scaled_c64.iter().zip(&scaled_c64_ref).enumerate() {
        assert_complex_close(actual, expected, 2.0e-14, &format!("blas1 c64 scal[{i}]"));
    }
    append_f32(&mut observables, &scaled_f32);
    append_f64(&mut observables, &scaled_f64);
    append_c32(&mut observables, &scaled_c32);
    append_c64(&mut observables, &scaled_c64);

    for (actual, expected, tolerance, label) in [
        (
            blas1::nrm2_f32(&x_f32) as f64,
            blas1::scalar::nrm2_f32(&x_f32) as f64,
            2.0e-6,
            "blas1 f32 nrm2",
        ),
        (
            blas1::nrm2_f64(&x_f64),
            blas1::scalar::nrm2_f64(&x_f64),
            2.0e-14,
            "blas1 f64 nrm2",
        ),
        (
            blas1::nrm2_c32(&x_c32) as f64,
            blas1::scalar::nrm2_c32(&x_c32) as f64,
            2.0e-5,
            "blas1 c32 nrm2",
        ),
        (
            blas1::nrm2_c64(&x_c64),
            blas1::scalar::nrm2_c64(&x_c64),
            2.0e-14,
            "blas1 c64 nrm2",
        ),
    ] {
        assert_close(actual, expected, tolerance, label);
        observables.push(actual);
    }
    let dot_f32 = blas1::dot_f32(&x_f32, &y_f32);
    assert_close(
        dot_f32,
        blas1::scalar::dot_f32(&x_f32, &y_f32),
        2.0e-6,
        "blas1 f32 dot",
    );
    let dotc_c32 = blas1::dotc_c32(&x_c32, &y_c32);
    assert_complex_close(
        Complex64::new(dotc_c32.re as f64, dotc_c32.im as f64),
        Complex64::new(
            blas1::scalar::dotc_c32(&x_c32, &y_c32).re as f64,
            blas1::scalar::dotc_c32(&x_c32, &y_c32).im as f64,
        ),
        2.0e-5,
        "blas1 c32 dotc",
    );
    append_f32(&mut observables, &[dot_f32]);
    observables.extend([dotc_c32.re as f64, dotc_c32.im as f64]);

    // BLAS-2: test all transpose modes with a padded row-major leading stride.
    for &op in &[Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
        let m = 3;
        let n = 5;
        let (a_rows, a_cols) = match op {
            Transpose::NoTrans => (m, n),
            Transpose::Trans | Transpose::ConjTrans => (n, m),
        };
        let lda = a_cols + 2;
        let a = stored_matrix(a_rows, a_cols, lda, 0.03125);
        let x = x_f64[..n].to_vec();
        let initial = vec![0.25, -0.5, 0.75];
        let mut actual = initial.clone();
        let mut expected = initial.clone();
        blas2::gemv_f64(op, m, n, 0.75, &a, lda, &x, -0.25, &mut actual);
        blas2::scalar::gemv_f64(op, m, n, 0.75, &a, lda, &x, -0.25, &mut expected);
        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_close(actual, expected, 2.0e-13, &format!("blas2 f64 {op:?}[{i}]"));
        }
        append_f64(&mut observables, &actual);

        let a_f32: Vec<f32> = a.iter().map(|&value| value as f32).collect();
        let x_f32_blas2: Vec<f32> = x_f32[..n].to_vec();
        let mut actual_f32 = initial
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>();
        let mut expected_f32 = actual_f32.clone();
        blas2::gemv_f32(
            op,
            m,
            n,
            0.75,
            &a_f32,
            lda,
            &x_f32_blas2,
            -0.25,
            &mut actual_f32,
        );
        blas2::scalar::gemv_f32(
            op,
            m,
            n,
            0.75,
            &a_f32,
            lda,
            &x_f32_blas2,
            -0.25,
            &mut expected_f32,
        );
        for (i, (&actual, &expected)) in actual_f32.iter().zip(&expected_f32).enumerate() {
            assert_close(actual, expected, 2.0e-5, &format!("blas2 f32 {op:?}[{i}]"));
        }
        append_f32(&mut observables, &actual_f32);

        let a_complex: Vec<Complex64> = a
            .iter()
            .enumerate()
            .map(|(i, &value)| Complex64::new(value, (i as f64 + 1.0) * 0.0078125))
            .collect();
        let x_complex = x_c64[..n].to_vec();
        let mut actual = x_complex[..m].to_vec();
        let mut expected = actual.clone();
        blas2::gemv_c64(
            op,
            m,
            n,
            Complex64::new(0.75, -0.125),
            &a_complex,
            lda,
            &x_complex,
            Complex64::new(-0.25, 0.0625),
            &mut actual,
        );
        blas2::scalar::gemv_c64(
            op,
            m,
            n,
            Complex64::new(0.75, -0.125),
            &a_complex,
            lda,
            &x_complex,
            Complex64::new(-0.25, 0.0625),
            &mut expected,
        );
        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_complex_close(actual, expected, 2.0e-13, &format!("blas2 c64 {op:?}[{i}]"));
        }
        append_c64(&mut observables, &actual);

        let a_c32: Vec<Complex32> = a
            .iter()
            .enumerate()
            .map(|(i, &value)| Complex32::new(value as f32, (i as f32 + 1.0) * 0.0078125))
            .collect();
        let x_c32_blas2 = x_c32[..n].to_vec();
        let alpha_c32 = Complex32::new(0.75, -0.125);
        let beta_c32 = Complex32::new(-0.25, 0.0625);
        let mut actual_c32 = x_c32_blas2[..m].to_vec();
        let mut expected_c32 = actual_c32.clone();
        blas2::gemv_c32(
            op,
            m,
            n,
            alpha_c32,
            &a_c32,
            lda,
            &x_c32_blas2,
            beta_c32,
            &mut actual_c32,
        );
        blas2::scalar::gemv_c32(
            op,
            m,
            n,
            alpha_c32,
            &a_c32,
            lda,
            &x_c32_blas2,
            beta_c32,
            &mut expected_c32,
        );
        for (i, (&actual, &expected)) in actual_c32.iter().zip(&expected_c32).enumerate() {
            assert_complex_close(actual, expected, 3.0e-5, &format!("blas2 c32 {op:?}[{i}]"));
        }
        append_c32(&mut observables, &actual_c32);
    }

    // BLAS-3: all transpose combinations and complex conjugation.
    for &transa in &[Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
        for &transb in &[Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            let m = 3;
            let n = 4;
            let k = 5;
            let (a_rows, a_cols) = match transa {
                Transpose::NoTrans => (m, k),
                Transpose::Trans | Transpose::ConjTrans => (k, m),
            };
            let (b_rows, b_cols) = match transb {
                Transpose::NoTrans => (k, n),
                Transpose::Trans | Transpose::ConjTrans => (n, k),
            };
            let lda = a_cols + 1;
            let ldb = b_cols + 1;
            let ldc = n + 2;
            let a = stored_matrix(a_rows, a_cols, lda, 0.015625);
            let b = stored_matrix(b_rows, b_cols, ldb, -0.0234375);
            let mut actual = stored_matrix(m, n, ldc, 0.0078125);
            let mut expected = actual.clone();
            blas3::gemm_f64(
                transa,
                transb,
                m,
                n,
                k,
                0.75,
                &a,
                lda,
                &b,
                ldb,
                -0.25,
                &mut actual,
                ldc,
            );
            blas3::scalar::gemm_f64(
                transa,
                transb,
                m,
                n,
                k,
                0.75,
                &a,
                lda,
                &b,
                ldb,
                -0.25,
                &mut expected,
                ldc,
            );
            for row in 0..m {
                for column in 0..n {
                    let index = row * ldc + column;
                    assert_close(
                        actual[index],
                        expected[index],
                        3.0e-13,
                        &format!("blas3 f64 {transa:?}/{transb:?}[{row},{column}]"),
                    );
                }
            }
            append_f64(&mut observables, &actual);

            let a_f32: Vec<f32> = a.iter().map(|&value| value as f32).collect();
            let b_f32: Vec<f32> = b.iter().map(|&value| value as f32).collect();
            let mut actual_f32: Vec<f32> = stored_matrix(m, n, ldc, 0.0078125)
                .into_iter()
                .map(|value| value as f32)
                .collect();
            let mut expected_f32 = actual_f32.clone();
            blas3::gemm_f32(
                transa,
                transb,
                m,
                n,
                k,
                0.75,
                &a_f32,
                lda,
                &b_f32,
                ldb,
                -0.25,
                &mut actual_f32,
                ldc,
            );
            blas3::scalar::gemm_f32(
                transa,
                transb,
                m,
                n,
                k,
                0.75,
                &a_f32,
                lda,
                &b_f32,
                ldb,
                -0.25,
                &mut expected_f32,
                ldc,
            );
            for row in 0..m {
                for column in 0..n {
                    let index = row * ldc + column;
                    assert_close(
                        actual_f32[index],
                        expected_f32[index],
                        4.0e-5,
                        &format!("blas3 f32 {transa:?}/{transb:?}[{row},{column}]"),
                    );
                }
            }
            append_f32(&mut observables, &actual_f32);

            let a_complex: Vec<Complex64> = a
                .iter()
                .enumerate()
                .map(|(i, &value)| Complex64::new(value, i as f64 * 0.00390625))
                .collect();
            let b_complex: Vec<Complex64> = b
                .iter()
                .enumerate()
                .map(|(i, &value)| Complex64::new(value, -(i as f64) * 0.005859375))
                .collect();
            let mut actual = vec![Complex64::new(0.125, -0.25); m * ldc];
            let mut expected = actual.clone();
            blas3::gemm_c64(
                transa,
                transb,
                m,
                n,
                k,
                Complex64::new(0.75, -0.125),
                &a_complex,
                lda,
                &b_complex,
                ldb,
                Complex64::new(-0.25, 0.0625),
                &mut actual,
                ldc,
            );
            blas3::scalar::gemm_c64(
                transa,
                transb,
                m,
                n,
                k,
                Complex64::new(0.75, -0.125),
                &a_complex,
                lda,
                &b_complex,
                ldb,
                Complex64::new(-0.25, 0.0625),
                &mut expected,
                ldc,
            );
            for row in 0..m {
                for column in 0..n {
                    let index = row * ldc + column;
                    assert_complex_close(
                        actual[index],
                        expected[index],
                        3.0e-13,
                        &format!("blas3 c64 {transa:?}/{transb:?}[{row},{column}]"),
                    );
                }
            }
            append_c64(&mut observables, &actual);

            let a_complex32: Vec<Complex32> = a_complex
                .iter()
                .map(|&value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let b_complex32: Vec<Complex32> = b_complex
                .iter()
                .map(|&value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let mut actual_c32 = vec![Complex32::new(0.125, -0.25); m * ldc];
            let mut expected_c32 = actual_c32.clone();
            let alpha_c32 = Complex32::new(0.75, -0.125);
            let beta_c32 = Complex32::new(-0.25, 0.0625);
            blas3::gemm_c32(
                transa,
                transb,
                m,
                n,
                k,
                alpha_c32,
                &a_complex32,
                lda,
                &b_complex32,
                ldb,
                beta_c32,
                &mut actual_c32,
                ldc,
            );
            blas3::scalar::gemm_c32(
                transa,
                transb,
                m,
                n,
                k,
                alpha_c32,
                &a_complex32,
                lda,
                &b_complex32,
                ldb,
                beta_c32,
                &mut expected_c32,
                ldc,
            );
            for row in 0..m {
                for column in 0..n {
                    let index = row * ldc + column;
                    assert_complex_close(
                        actual_c32[index],
                        expected_c32[index],
                        5.0e-5,
                        &format!("blas3 c32 {transa:?}/{transb:?}[{row},{column}]"),
                    );
                }
            }
            append_c32(&mut observables, &actual_c32);
        }
    }

    // CSR gather reducer: both f32 and f64, with a non-multiple-of-vector tail.
    let col_idx = [0, 7, 2, 5, 1, 6, 3, 4, 0, 7, 2];
    let values_f32: Vec<f32> = (0..col_idx.len()).map(|i| i as f32 * 0.125 - 0.5).collect();
    let values_f64: Vec<f64> = values_f32.iter().map(|&value| value as f64).collect();
    let x_f32 = (0..8).map(|i| i as f32 * 0.25 - 0.75).collect::<Vec<_>>();
    let x_f64 = x_f32.iter().map(|&value| value as f64).collect::<Vec<_>>();
    let spmv_f32 = spmv_simd::row_f32(&col_idx, &values_f32, &x_f32, 0, col_idx.len());
    assert_close(
        spmv_f32,
        spmv_simd::row_f32_scalar_reference(&col_idx, &values_f32, &x_f32, 0, col_idx.len()),
        2.0e-6,
        "spmv f32",
    );
    let spmv_f64 = spmv_simd::row_f64(&col_idx, &values_f64, &x_f64, 0, col_idx.len());
    assert_close(
        spmv_f64,
        spmv_simd::row_f64_scalar_reference(&col_idx, &values_f64, &x_f64, 0, col_idx.len()),
        2.0e-14,
        "spmv f64",
    );
    append_f64(&mut observables, &[spmv_f32 as f64, spmv_f64]);

    // BEM pair kernel: Tri3 is the only accelerated element family currently
    // exposed by this CPU/GPU ABI. Exercise every AVX2/FMA lane remainder,
    // several geometry variants, low/acoustic/high phase regimes, and both
    // supported precision contracts (f64/C64 and f32/C32).
    let pair = RegularPairInF64 {
        source_point: [0.1, -0.2, 0.3],
        source_normal: [0.0, 0.0, 1.0],
        v0: [-0.8, -0.6, 0.0],
        v1: [0.9, -0.5, 0.1],
        v2: [-0.2, 0.8, -0.15],
    };
    let tri3_variants = [
        pair,
        RegularPairInF64 {
            source_point: [-2.25, 1.75, -0.5],
            source_normal: [0.0, 1.0, 0.0],
            v0: [3.0, -1.0, 0.25],
            v1: [3.000_001, -1.0, 0.25],
            v2: [3.0, -0.25, 0.75],
        },
        RegularPairInF64 {
            source_point: [0.01, -0.02, 0.03],
            source_normal: [0.5773502691896258; 3],
            v0: [-20.0, 10.0, -5.0],
            v1: [-18.0, 10.5, -4.5],
            v2: [-19.0, 12.0, -5.25],
        },
    ];
    for &length in &[0_usize, 1, 2, 3, 4, 5, 7, 9] {
        let pairs: Vec<_> = (0..length)
            .map(|index| tri3_variants[index % tri3_variants.len()])
            .collect();
        assert_pair_batch_parity(
            &mut observables,
            &pairs,
            2.75,
            1.0,
            &format!("bem Tri3 lane-tail length {length}"),
        );
    }
    for &(wave_number, harmonic_factor, label) in &[
        (0.0, 1.0, "static phase"),
        (0.125, 1.0, "low phase"),
        (2.75, 1.0, "acoustic phase"),
        (1_000.0, -1.0, "high phase"),
    ] {
        assert_pair_batch_parity(
            &mut observables,
            &tri3_variants,
            wave_number,
            harmonic_factor,
            &format!("bem Tri3 {label}"),
        );
    }

    let pairs_c32: Vec<BemRegularPairIn> = tri3_variants
        .iter()
        .map(|value| {
            BemRegularPairIn::from_triangle(
                value.source_point.map(|coordinate| coordinate as f32),
                value.source_normal.map(|coordinate| coordinate as f32),
                value.v0.map(|coordinate| coordinate as f32),
                value.v1.map(|coordinate| coordinate as f32),
                value.v2.map(|coordinate| coordinate as f32),
            )
        })
        .collect();
    let actual_c32 = regular_pair_c32_batch_from_gpu_abi(&pairs_c32, 2.75, 1.0);
    for (index, (actual, input)) in actual_c32.iter().zip(&pairs_c32).enumerate() {
        let input_f64 = RegularPairInF64 {
            source_point: input.source_point.map(f64::from),
            source_normal: input.source_normal.map(f64::from),
            v0: input.v0.map(f64::from),
            v1: input.v1.map(f64::from),
            v2: input.v2.map(f64::from),
        };
        let expected = regular_pair_scalar(&input_f64, 2.75, 1.0);
        let actual_values = [actual.g, actual.h, actual.ht, actual.e];
        let expected_values = [expected.0, expected.1, expected.2, expected.3];
        for (component, (actual, expected)) in
            actual_values.into_iter().zip(expected_values).enumerate()
        {
            assert_complex_close(
                actual,
                Complex32::new(expected.re as f32, expected.im as f32),
                3.0e-6,
                &format!("bem Tri3 C32 pair {index} component {component}"),
            );
        }
        observables.extend(
            [actual.g, actual.h, actual.ht, actual.e]
                .into_iter()
                .flat_map(|value| [value.re as f64, value.im as f64]),
        );
    }

    let status = if force_scalar {
        "scalar-baseline"
    } else if native_avx2_fma {
        "pass"
    } else {
        "skip"
    }
    .to_owned();
    #[cfg(target_arch = "x86_64")]
    let feature_set = format!(
        "avx2={},fma={},sse2={},force_scalar={}",
        avx2,
        fma,
        std::is_x86_feature_detected!("sse2"),
        force_scalar
    );
    #[cfg(target_arch = "aarch64")]
    let feature_set = "neon=true".to_owned();
    let selected_kernel = selected_kernel(native_avx2_fma, force_scalar).to_owned();
    let (max_norm_error, max_ulp_error) = accuracy_summary();
    let mut artifact = NativeCpuEvidenceArtifact {
        schema: EVIDENCE_SCHEMA,
        status,
        target_arch: std::env::consts::ARCH.to_owned(),
        cpu_model: cpuinfo_value("model name"),
        microcode: cpuinfo_value("microcode"),
        compiler: compiler_version(),
        features: feature_set,
        selected_kernel: selected_kernel.clone(),
        force_scalar,
        runtime_dispatch: format!("selected_kernel={selected_kernel},force_scalar={force_scalar}"),
        os: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        thread_policy: std::env::var("SONIUM_QA_THREAD_POLICY")
            .unwrap_or_else(|_| "unspecified".into()),
        affinity_policy: std::env::var("SONIUM_QA_AFFINITY_POLICY")
            .unwrap_or_else(|_| "unspecified".into()),
        numa_policy: std::env::var("SONIUM_QA_NUMA_POLICY")
            .unwrap_or_else(|_| "unspecified".into()),
        deterministic_mode: std::env::var("SONIUM_QA_DETERMINISTIC_MODE")
            .unwrap_or_else(|_| "unspecified".into()),
        kernels: "blas1,blas2,blas3,spmv,bem".to_owned(),
        bem_elements: "Tri3-constant:f64/C64,Tri3-constant:f32/C32,Quad4-and-higher:unsupported"
            .to_owned(),
        bem_phase_regimes: "static,low,acoustic,high".to_owned(),
        bem_precisions: "f64/C64,f32/C32".to_owned(),
        norm: "maximum absolute scalar/component error".to_owned(),
        max_norm_error,
        max_ulp_error,
        aarch64_comparison_max_abs_error: None,
        physical_validation: PhysicalValidation {
            status: "not-run".into(),
            budget: "not-applicable".into(),
            reason: "B4 proves compute/operator dispatch parity only; physical entry, field, power, and derivative budgets belong to B3 and are not inferred here.".into(),
        },
        cases_passed: 5,
        cases_skipped: usize::from(!native_avx2_fma && !force_scalar),
        cases_failed: 0,
        observables,
    };
    if std::env::var("SONIUM_REQUIRE_AVX2_FMA").as_deref() == Ok("1") {
        for (name, value) in [
            ("cpu_model", &artifact.cpu_model),
            ("microcode", &artifact.microcode),
            ("compiler", &artifact.compiler),
            ("thread_policy", &artifact.thread_policy),
            ("affinity_policy", &artifact.affinity_policy),
            ("numa_policy", &artifact.numa_policy),
            ("deterministic_mode", &artifact.deterministic_mode),
        ] {
            assert!(
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "unknown" | "unspecified" | "unavailable" | "n/a"
                ),
                "required native x86 evidence metadata {name} is unavailable"
            );
        }
    }
    let comparison = std::env::var_os("SONIUM_AARCH64_REFERENCE")
        .map(|path| compare_aarch64(&artifact, &std::path::PathBuf::from(path)))
        .transpose()
        .unwrap_or_else(|error| panic!("native x86/aarch64 comparison failed: {error}"));
    artifact.aarch64_comparison_max_abs_error = comparison;
    let checksum = artifact_checksum(&artifact);
    let artifact_json = serde_json::to_vec_pretty(&artifact).expect("native CPU evidence JSON");
    let comparison_error = artifact
        .aarch64_comparison_max_abs_error
        .map(|value| format!("{value:.6e}"))
        .unwrap_or_else(|| "null".to_owned());
    eprintln!(
        "sonium_native_x86_evidence={{\"schema\":{},\"status\":\"{}\",\"target_arch\":\"{}\",\"selected_kernel\":\"{}\",\"features\":\"{}\",\"kernels\":\"{}\",\"cases_passed\":{},\"cases_skipped\":{},\"cases_failed\":0,\"observables\":{},\"max_norm_error\":{:.9e},\"max_ulp_error\":{},\"artifact_sha256\":\"{}\",\"aarch64_comparison\":\"{}\",\"aarch64_comparison_max_abs_error\":{}}}",
        artifact.schema,
        artifact.status,
        json_escape(&artifact.target_arch),
        json_escape(&artifact.selected_kernel),
        json_escape(&artifact.features),
        artifact.kernels,
        artifact.cases_passed,
        artifact.cases_skipped,
        artifact.observables.len(),
        artifact.max_norm_error,
        artifact.max_ulp_error,
        checksum,
        if artifact.aarch64_comparison_max_abs_error.is_some() {
            "passed"
        } else {
            "not_requested"
        },
        comparison_error,
    );
    if let Some(output_dir) = std::env::var_os("SONIUM_OUTPUT_DIR") {
        let output_dir = std::path::PathBuf::from(output_dir);
        std::fs::create_dir_all(&output_dir).expect("create native x86 evidence directory");
        std::fs::write(
            output_dir.join("sonium_native_x86_evidence.json"),
            artifact_json,
        )
        .expect("write native x86 evidence artifact");
    }
}

#[cfg(test)]
mod comparison_tests {
    use super::*;

    fn artifact(target_arch: &str, observables: Vec<f64>) -> NativeCpuEvidenceArtifact {
        NativeCpuEvidenceArtifact {
            schema: EVIDENCE_SCHEMA,
            status: "pass".into(),
            target_arch: target_arch.into(),
            cpu_model: "test-cpu".into(),
            microcode: "test-microcode".into(),
            compiler: "test-compiler".into(),
            features: "test-features".into(),
            selected_kernel: "test-kernel".into(),
            force_scalar: false,
            runtime_dispatch: "selected_kernel=test-kernel".into(),
            os: "linux-x86_64".into(),
            thread_policy: "single".into(),
            affinity_policy: "none".into(),
            numa_policy: "default".into(),
            deterministic_mode: "enabled".into(),
            kernels: "blas1,blas2,blas3,spmv,bem".into(),
            bem_elements:
                "Tri3-constant:f64/C64,Tri3-constant:f32/C32,Quad4-and-higher:unsupported".into(),
            bem_phase_regimes: "static,low,acoustic,high".into(),
            bem_precisions: "f64/C64,f32/C32".into(),
            norm: "maximum absolute scalar/component error".into(),
            max_norm_error: 0.0,
            max_ulp_error: 0,
            aarch64_comparison_max_abs_error: None,
            physical_validation: PhysicalValidation {
                status: "not-run".into(),
                budget: "not-applicable".into(),
                reason: "operator-only test fixture".into(),
            },
            cases_passed: 5,
            cases_skipped: 0,
            cases_failed: 0,
            observables,
        }
    }

    #[test]
    fn accepted_aarch64_reference_comparison_is_checked() {
        let actual = artifact("x86_64", vec![0.0, 1.0, -2.0]);
        let reference = artifact("aarch64", actual.observables.clone());
        let path = std::env::temp_dir().join(format!(
            "sonium-native-cpu-evidence-reference-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            serde_json::to_vec(&reference).expect("reference artifact serializes"),
        )
        .expect("reference artifact writes");
        let maximum = compare_aarch64(&actual, &path).expect("matching reference is accepted");
        std::fs::remove_file(path).expect("reference artifact removes");
        assert_eq!(maximum, 0.0);
    }
}
