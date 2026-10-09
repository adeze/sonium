//! Lane-by-lane SIMD equivalence vectors.
//!
//! The vectors deliberately cover subnormals, signed zero, exact reduction
//! boundaries, both neighbours of each boundary, the Cody–Waite cutover,
//! overflow-scale finite values, and non-finite inputs.

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
use sonium_fastmath::sincos_f32;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use sonium_fastmath::{CODY_WAITE_LIMIT, sincos_scalar};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn edge_vectors() -> Vec<f64> {
    let mut values = vec![
        -0.0,
        0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(0x3ff921fb54442d17),
        f64::from_bits(0x3ff921fb54442d18),
        f64::from_bits(0x3ff921fb54442d19),
        f64::from_bits(0x400921fb54442d17),
        f64::from_bits(0x400921fb54442d18),
        f64::from_bits(0x400921fb54442d19),
        CODY_WAITE_LIMIT - f64::EPSILON,
        CODY_WAITE_LIMIT,
        f64::from_bits(CODY_WAITE_LIMIT.to_bits() + 1),
        -CODY_WAITE_LIMIT,
        f64::from_bits((-CODY_WAITE_LIMIT).to_bits() + 1),
        1.0e12,
        -1.0e12,
        1.0e20,
        -1.0e20,
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    values.extend((-32..=32).map(|k| k as f64 * std::f64::consts::FRAC_PI_2));
    values
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn assert_lane(input: f64, actual_sin: f64, actual_cos: f64) {
    let (expected_sin, expected_cos) = sincos_scalar(input);
    if expected_sin.is_nan() {
        assert!(actual_sin.is_nan() && actual_cos.is_nan());
        return;
    }
    if input == 0.0 {
        assert_eq!(actual_sin.to_bits(), expected_sin.to_bits());
        assert_eq!(actual_cos.to_bits(), expected_cos.to_bits());
        return;
    }
    assert!(
        (actual_sin - expected_sin).abs() <= 8.0 * f64::EPSILON,
        "sin lane mismatch at {input:e}: {actual_sin:?} vs {expected_sin:?}"
    );
    assert!(
        (actual_cos - expected_cos).abs() <= 8.0 * f64::EPSILON,
        "cos lane mismatch at {input:e}: {actual_cos:?} vs {expected_cos:?}"
    );
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
fn assert_lane_f32(input: f32, actual_sin: f32, actual_cos: f32) {
    let (expected_sin, expected_cos) = sincos_f32(input);
    if expected_sin.is_nan() {
        assert!(actual_sin.is_nan() && actual_cos.is_nan());
        return;
    }
    if input == 0.0 {
        assert_eq!(actual_sin.to_bits(), expected_sin.to_bits());
        assert_eq!(actual_cos.to_bits(), expected_cos.to_bits());
        return;
    }
    assert!((actual_sin - expected_sin).abs() <= 8.0e-6);
    assert!((actual_cos - expected_cos).abs() <= 8.0e-6);
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
fn edge_vectors_f32() -> [f32; 16] {
    [
        -0.0,
        0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        -1000.0,
        -31.0,
        -std::f32::consts::FRAC_PI_2,
        0.0,
        std::f32::consts::FRAC_PI_2,
        31.0,
        1000.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_f64_lanes_match_scalar_edge_vectors() {
    use sonium_fastmath::sincos_f64x2;
    use std::arch::aarch64::*;

    let values = edge_vectors();
    for chunk in values.chunks(2) {
        let a = chunk[0];
        let b = *chunk.get(1).unwrap_or(&a);
        // SAFETY: AArch64 guarantees the NEON instruction set.
        let input = unsafe { vsetq_lane_f64(b, vdupq_n_f64(a), 1) };
        // SAFETY: The target architecture provides the required SIMD type.
        let (sin, cos) = unsafe { sincos_f64x2(input) };
        let actual_sin = [unsafe { vgetq_lane_f64(sin, 0) }, unsafe {
            vgetq_lane_f64(sin, 1)
        }];
        let actual_cos = [unsafe { vgetq_lane_f64(cos, 0) }, unsafe {
            vgetq_lane_f64(cos, 1)
        }];
        assert_lane(a, actual_sin[0], actual_cos[0]);
        assert_lane(b, actual_sin[1], actual_cos[1]);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_f64_lanes_match_scalar_edge_vectors() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    use sonium_fastmath::sincos_f64x4;
    use std::arch::x86_64::*;

    let values = edge_vectors();
    for chunk in values.chunks(4) {
        let mut input = [0.0; 4];
        input[..chunk.len()].copy_from_slice(chunk);
        let input_vector = unsafe { _mm256_loadu_pd(input.as_ptr()) };
        // SAFETY: The runtime checks above establish AVX2 + FMA support.
        let (sin, cos) = unsafe { sincos_f64x4(input_vector) };
        let mut actual_sin = [0.0; 4];
        let mut actual_cos = [0.0; 4];
        unsafe {
            _mm256_storeu_pd(actual_sin.as_mut_ptr(), sin);
            _mm256_storeu_pd(actual_cos.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            assert_lane(chunk[lane], actual_sin[lane], actual_cos[lane]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_mixed_batches_blend_per_lane() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    use sonium_fastmath::sincos_f64x4;
    use std::arch::x86_64::*;

    // One outlier per batch must not disturb the ordinary lanes: fallback
    // lanes take the scalar path (bitwise equal), vector lanes keep the
    // polynomial path (within 1e-14 of scalar).
    let batches: [[f64; 4]; 5] = [
        [0.5, f64::NAN, 0.25, -0.0],
        [f64::INFINITY, 1.0, 1.0e20, 2.0],
        [CODY_WAITE_LIMIT + 1.0, -1.0, 0.0, 3.0],
        [f64::MAX, f64::NEG_INFINITY, -f64::MAX, 0.1],
        [1.0e8, 1001.0, -1001.0, -1.0e8],
    ];
    for batch in batches {
        // SAFETY: the runtime checks above establish AVX2 + FMA support.
        let input = unsafe { _mm256_set_pd(batch[3], batch[2], batch[1], batch[0]) };
        // SAFETY: the runtime checks above establish AVX2 + FMA support.
        let (sin, cos) = unsafe { sincos_f64x4(input) };
        let mut actual_sin = [0.0; 4];
        let mut actual_cos = [0.0; 4];
        unsafe {
            _mm256_storeu_pd(actual_sin.as_mut_ptr(), sin);
            _mm256_storeu_pd(actual_cos.as_mut_ptr(), cos);
        }
        for (lane, &x) in batch.iter().enumerate() {
            let (want_s, want_c) = sincos_scalar(x);
            if want_s.is_nan() {
                assert!(
                    actual_sin[lane].is_nan() && actual_cos[lane].is_nan(),
                    "NaN lane {lane} at {x}"
                );
            } else if x == 0.0 || !x.is_finite() || x.abs() > CODY_WAITE_LIMIT {
                assert_eq!(actual_sin[lane], want_s, "fallback sin lane {lane} at {x}");
                assert_eq!(actual_cos[lane], want_c, "fallback cos lane {lane} at {x}");
            } else {
                assert!(
                    (actual_sin[lane] - want_s).abs() < 1e-14,
                    "vector sin lane {lane} at {x}"
                );
                assert!(
                    (actual_cos[lane] - want_c).abs() < 1e-14,
                    "vector cos lane {lane} at {x}"
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_mixed_batches_blend_per_lane() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }
    use sonium_fastmath::avx512::sincos_f64x8;
    use std::arch::x86_64::*;

    let batch: [f64; 8] = [
        0.5,
        f64::NAN,
        -0.0,
        1.0e20,
        2.0,
        f64::INFINITY,
        1001.0,
        -3.0,
    ];
    // SAFETY: the runtime check above establishes AVX-512F support.
    let input = unsafe { _mm512_loadu_pd(batch.as_ptr()) };
    // SAFETY: the runtime check above establishes AVX-512F support.
    let (sin, cos) = unsafe { sincos_f64x8(input) };
    let mut actual_sin = [0.0; 8];
    let mut actual_cos = [0.0; 8];
    unsafe {
        _mm512_storeu_pd(actual_sin.as_mut_ptr(), sin);
        _mm512_storeu_pd(actual_cos.as_mut_ptr(), cos);
    }
    for (lane, &x) in batch.iter().enumerate() {
        let (want_s, want_c) = sincos_scalar(x);
        if want_s.is_nan() {
            assert!(
                actual_sin[lane].is_nan() && actual_cos[lane].is_nan(),
                "NaN lane {lane} at {x}"
            );
        } else if x == 0.0 || !x.is_finite() || x.abs() > CODY_WAITE_LIMIT {
            assert_eq!(actual_sin[lane], want_s, "fallback sin lane {lane} at {x}");
            assert_eq!(actual_cos[lane], want_c, "fallback cos lane {lane} at {x}");
        } else {
            assert!(
                (actual_sin[lane] - want_s).abs() <= 4.0 * f64::EPSILON,
                "vector sin lane {lane} at {x}"
            );
            assert!(
                (actual_cos[lane] - want_c).abs() <= 4.0 * f64::EPSILON,
                "vector cos lane {lane} at {x}"
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_f32_lanes_match_scalar_edge_vectors() {
    use sonium_fastmath::sincos_f32x4;
    use std::arch::aarch64::*;

    let values = edge_vectors_f32();
    for chunk in values.chunks(4) {
        let input = unsafe { vld1q_f32(chunk.as_ptr()) };
        // SAFETY: AArch64 guarantees NEON.
        let (sin, cos) = unsafe { sincos_f32x4(input) };
        let mut actual_sin = [0.0; 4];
        let mut actual_cos = [0.0; 4];
        unsafe {
            vst1q_f32(actual_sin.as_mut_ptr(), sin);
            vst1q_f32(actual_cos.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            assert_lane_f32(chunk[lane], actual_sin[lane], actual_cos[lane]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_f32_lanes_match_scalar_edge_vectors() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    use sonium_fastmath::avx2::sincos_f32x8;
    use std::arch::x86_64::*;

    let values = edge_vectors_f32();
    for chunk in values.chunks(8) {
        let input = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
        // SAFETY: The runtime checks establish AVX2 + FMA support.
        let (sin, cos) = unsafe { sincos_f32x8(input) };
        let mut actual_sin = [0.0; 8];
        let mut actual_cos = [0.0; 8];
        unsafe {
            _mm256_storeu_ps(actual_sin.as_mut_ptr(), sin);
            _mm256_storeu_ps(actual_cos.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            assert_lane_f32(chunk[lane], actual_sin[lane], actual_cos[lane]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_f64_lanes_match_scalar_edge_vectors() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }
    use sonium_fastmath::avx512::sincos_f64x8;
    use std::arch::x86_64::*;

    let values = edge_vectors();
    for chunk in values.chunks(8) {
        let mut input = [0.0; 8];
        input[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: The runtime check establishes AVX-512F support.
        let input_vector = unsafe { _mm512_loadu_pd(input.as_ptr()) };
        // SAFETY: The runtime check establishes AVX-512F support.
        let (sin, cos) = unsafe { sincos_f64x8(input_vector) };
        let mut actual_sin = [0.0; 8];
        let mut actual_cos = [0.0; 8];
        unsafe {
            _mm512_storeu_pd(actual_sin.as_mut_ptr(), sin);
            _mm512_storeu_pd(actual_cos.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            assert_lane(chunk[lane], actual_sin[lane], actual_cos[lane]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_f32_lanes_match_scalar_edge_vectors() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }
    use sonium_fastmath::avx512::sincos_f32x16;
    use std::arch::x86_64::*;

    let values = edge_vectors_f32();
    for chunk in values.chunks(16) {
        let mut input = [0.0; 16];
        input[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: The runtime check establishes AVX-512F support.
        let input_vector = unsafe { _mm512_loadu_ps(input.as_ptr()) };
        // SAFETY: The runtime check establishes AVX-512F support.
        let (sin, cos) = unsafe { sincos_f32x16(input_vector) };
        let mut actual_sin = [0.0; 16];
        let mut actual_cos = [0.0; 16];
        unsafe {
            _mm512_storeu_ps(actual_sin.as_mut_ptr(), sin);
            _mm512_storeu_ps(actual_cos.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            assert_lane_f32(chunk[lane], actual_sin[lane], actual_cos[lane]);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm_simd_f32_lanes_match_scalar_edge_vectors() {
    use sonium_fastmath::wasm_simd::sincos_f32x4;
    use std::arch::wasm32::*;

    for chunk in edge_vectors_f32().chunks(4) {
        let mut input = f32x4_splat(chunk[0]);
        if chunk.len() > 1 {
            input = f32x4_replace_lane::<1>(input, chunk[1]);
        }
        if chunk.len() > 2 {
            input = f32x4_replace_lane::<2>(input, chunk[2]);
        }
        if chunk.len() > 3 {
            input = f32x4_replace_lane::<3>(input, chunk[3]);
        }
        // SAFETY: The cfg gate establishes wasm simd128 support.
        let (sin, cos) = unsafe { sincos_f32x4(input) };
        let actual_sin = [
            f32x4_extract_lane::<0>(sin),
            f32x4_extract_lane::<1>(sin),
            f32x4_extract_lane::<2>(sin),
            f32x4_extract_lane::<3>(sin),
        ];
        let actual_cos = [
            f32x4_extract_lane::<0>(cos),
            f32x4_extract_lane::<1>(cos),
            f32x4_extract_lane::<2>(cos),
            f32x4_extract_lane::<3>(cos),
        ];
        for lane in 0..chunk.len() {
            assert_lane_f32(chunk[lane], actual_sin[lane], actual_cos[lane]);
        }
    }
}
