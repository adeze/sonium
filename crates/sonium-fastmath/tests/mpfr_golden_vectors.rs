//! Checked-in MPFR 4.2.2 (256-bit) sine/cosine references.
//!
//! Inputs and outputs are stored as IEEE-754 bit patterns so the test does
//! not depend on decimal-parser spelling or on the host C library.  The
//! boundary triplets include both exact floating-point neighbours of the
//! reduction points and the Cody–Waite cutover probes.

use sonium_fastmath::{sincos_f32, sincos_scalar};

#[derive(Clone, Copy)]
struct Golden {
    input: u64,
    sin: u64,
    cos: u64,
    max_scaled_ulp: f64,
}

const GOLDEN: &[Golden] = &[
    Golden {
        input: 0x0000000000000000,
        sin: 0x0000000000000000,
        cos: 0x3ff0000000000000,
        max_scaled_ulp: 0.0,
    },
    Golden {
        input: 0x8000000000000000,
        sin: 0x8000000000000000,
        cos: 0x3ff0000000000000,
        max_scaled_ulp: 0.0,
    },
    Golden {
        input: 0x3ff0000000000000,
        sin: 0x3feaed548f090cee,
        cos: 0x3fe14a280fb5068c,
        max_scaled_ulp: 2.0,
    },
    Golden {
        input: 0xbff0000000000000,
        sin: 0xbfeaed548f090cee,
        cos: 0x3fe14a280fb5068c,
        max_scaled_ulp: 2.0,
    },
    Golden {
        input: 0x01a56e1fc2f8f359,
        sin: 0x01a56e1fc2f8f359,
        cos: 0x3ff0000000000000,
        max_scaled_ulp: 0.0,
    },
    Golden {
        input: 0x81a56e1fc2f8f359,
        sin: 0x81a56e1fc2f8f359,
        cos: 0x3ff0000000000000,
        max_scaled_ulp: 0.0,
    },
    Golden {
        input: 0x0010000000000000,
        sin: 0x0010000000000000,
        cos: 0x3ff0000000000000,
        max_scaled_ulp: 0.0,
    },
    Golden {
        input: 0x8010000000000000,
        sin: 0x8010000000000000,
        cos: 0x3ff0000000000000,
        max_scaled_ulp: 0.0,
    },
    Golden {
        input: 0x408f400000000000,
        sin: 0x3fea75cc150a206b,
        cos: 0x3fe1ff026793f1bb,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xc08f400000000000,
        sin: 0xbfea75cc150a206b,
        cos: 0x3fe1ff026793f1bb,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x426d1a94a2000000,
        sin: 0xbfe38f4477984352,
        cos: 0x3fe9538731dff223,
        max_scaled_ulp: 8.0,
    },
    Golden {
        input: 0xc26d1a94a2000000,
        sin: 0x3fe38f4477984352,
        cos: 0x3fe9538731dff223,
        max_scaled_ulp: 8.0,
    },
    Golden {
        input: 0x4415af1d78b58c40,
        sin: 0xbfe4a5e605fd6450,
        cos: 0x3fe872720fc60d3d,
        max_scaled_ulp: 64.0,
    },
    Golden {
        input: 0xc415af1d78b58c40,
        sin: 0x3fe4a5e605fd6450,
        cos: 0x3fe872720fc60d3d,
        max_scaled_ulp: 64.0,
    },
    Golden {
        input: 0x7fefffffffffffff,
        sin: 0x3f7452fc98b34e97,
        cos: 0xbfefffe62ecfab75,
        max_scaled_ulp: 64.0,
    },
    Golden {
        input: 0xffefffffffffffff,
        sin: 0xbf7452fc98b34e97,
        cos: 0xbfefffe62ecfab75,
        max_scaled_ulp: 64.0,
    },
    Golden {
        input: 0x3ff921fb54442d18,
        sin: 0x3ff0000000000000,
        cos: 0x3c91a62633145c07,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xbff921fb54442d18,
        sin: 0xbff0000000000000,
        cos: 0x3c91a62633145c07,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x400921fb54442d18,
        sin: 0x3ca1a62633145c07,
        cos: 0xbff0000000000000,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xc00921fb54442d18,
        sin: 0xbca1a62633145c07,
        cos: 0xbff0000000000000,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x3ff921fb54442d17,
        sin: 0x3ff0000000000000,
        cos: 0x3cb469898cc51702,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x3ff921fb54442d18,
        sin: 0x3ff0000000000000,
        cos: 0x3c91a62633145c07,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x3ff921fb54442d19,
        sin: 0x3ff0000000000000,
        cos: 0xbca72cece675d1fd,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xbff921fb54442d19,
        sin: 0xbff0000000000000,
        cos: 0xbca72cece675d1fd,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xbff921fb54442d18,
        sin: 0xbff0000000000000,
        cos: 0x3c91a62633145c07,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xbff921fb54442d17,
        sin: 0xbff0000000000000,
        cos: 0x3cb469898cc51702,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x400921fb54442d17,
        sin: 0x3cc469898cc51702,
        cos: 0xbff0000000000000,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x400921fb54442d18,
        sin: 0x3ca1a62633145c07,
        cos: 0xbff0000000000000,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x400921fb54442d19,
        sin: 0xbcb72cece675d1fd,
        cos: 0xbff0000000000000,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xc00921fb54442d19,
        sin: 0x3cb72cece675d1fd,
        cos: 0xbff0000000000000,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xc00921fb54442d18,
        sin: 0xbca1a62633145c07,
        cos: 0xbff0000000000000,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xc00921fb54442d17,
        sin: 0xbcc469898cc51702,
        cos: 0xbff0000000000000,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x408f3fffffffffffu64,
        sin: 0x3fea75cc150a1e2b,
        cos: 0x3fe1ff026793f50a,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x408f400000000000,
        sin: 0x3fea75cc150a206b,
        cos: 0x3fe1ff026793f1bb,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0x408f400000000001,
        sin: 0x3fea75cc150a22ab,
        cos: 0x3fe1ff026793ee6d,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xc08f400000000001,
        sin: 0xbfea75cc150a22ab,
        cos: 0x3fe1ff026793ee6d,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xc08f400000000000,
        sin: 0xbfea75cc150a206b,
        cos: 0x3fe1ff026793f1bb,
        max_scaled_ulp: 4.0,
    },
    Golden {
        input: 0xc08f3fffffffffffu64,
        sin: 0xbfea75cc150a1e2b,
        cos: 0x3fe1ff026793f50a,
        max_scaled_ulp: 4.0,
    },
];

fn scaled_ulp_distance(actual: f64, expected: f64) -> f64 {
    if actual == expected {
        return 0.0;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return f64::INFINITY;
    }
    (actual - expected).abs() / f64::EPSILON
}

fn assert_golden_output(index: usize, vector: Golden, actual_sin: f64, actual_cos: f64) {
    let expected_sin = f64::from_bits(vector.sin);
    let expected_cos = f64::from_bits(vector.cos);
    assert_eq!(
        actual_sin.is_sign_negative(),
        expected_sin.is_sign_negative(),
        "sine sign at vector {index}"
    );
    assert_eq!(
        actual_cos.is_sign_negative(),
        expected_cos.is_sign_negative(),
        "cosine sign at vector {index}"
    );
    let sin_error = scaled_ulp_distance(actual_sin, expected_sin);
    let cos_error = scaled_ulp_distance(actual_cos, expected_cos);
    assert!(
        sin_error <= vector.max_scaled_ulp,
        "MPFR sine vector {index}: {sin_error} > {}",
        vector.max_scaled_ulp
    );
    assert!(
        cos_error <= vector.max_scaled_ulp,
        "MPFR cosine vector {index}: {cos_error} > {}",
        vector.max_scaled_ulp
    );
}

#[test]
fn mpfr_vectors_preserve_specials_boundaries_and_huge_quadrants() {
    for (index, &vector) in GOLDEN.iter().enumerate() {
        let input = f64::from_bits(vector.input);
        let (sin, cos) = sincos_scalar(input);
        assert_golden_output(index, vector, sin, cos);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_checked_in_mpfr_vectors() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        if std::env::var_os("SONIUM_FASTMATH_REQUIRE_AVX2").is_some() {
            panic!("SONIUM_FASTMATH_REQUIRE_AVX2=1 but the runner lacks AVX2+FMA");
        }
        return;
    }
    use sonium_fastmath::avx2::sincos_f64x4;
    use std::arch::x86_64::*;

    for (base, chunk) in GOLDEN.chunks(4).enumerate() {
        let mut input = [0.0_f64; 4];
        for (lane, vector) in chunk.iter().enumerate() {
            input[lane] = f64::from_bits(vector.input);
        }
        let fill = input[0];
        for value in &mut input[chunk.len()..] {
            *value = fill;
        }
        // SAFETY: the feature checks above establish AVX2 + FMA support.
        let vector = unsafe { _mm256_loadu_pd(input.as_ptr()) };
        // SAFETY: the feature checks above establish AVX2 + FMA support.
        let (sin, cos) = unsafe { sincos_f64x4(vector) };
        let mut sins = [0.0; 4];
        let mut coss = [0.0; 4];
        // SAFETY: the output arrays have four f64 lanes.
        unsafe {
            _mm256_storeu_pd(sins.as_mut_ptr(), sin);
            _mm256_storeu_pd(coss.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            assert_golden_output(base * 4 + lane, chunk[lane], sins[lane], coss[lane]);
        }
    }
}

/// The f32 AVX2 kernel widens each lane and narrows exactly once. Exercise it
/// with the same deterministic MPFR corpus (rounded to f32 input semantics)
/// and require bit-for-bit agreement with the public scalar f32 path. The
/// f64 vectors remain the high-precision oracle; this test additionally
/// catches f32 lane packing, narrowing, signed-zero, and non-finite handling.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_f32_matches_scalar_on_the_mpfr_corpus() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        if std::env::var_os("SONIUM_FASTMATH_REQUIRE_AVX2").is_some() {
            panic!("SONIUM_FASTMATH_REQUIRE_AVX2=1 but the runner lacks AVX2+FMA");
        }
        return;
    }
    use sonium_fastmath::avx2::sincos_f32x8;
    use std::arch::x86_64::*;

    let values: Vec<f32> = GOLDEN
        .iter()
        .map(|vector| f64::from_bits(vector.input) as f32)
        .chain([
            -0.0_f32,
            0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ])
        .collect();
    for chunk in values.chunks(8) {
        let mut input = [0.0_f32; 8];
        input[..chunk.len()].copy_from_slice(chunk);
        let fill = input[0];
        for value in &mut input[chunk.len()..] {
            *value = fill;
        }
        // SAFETY: the feature checks above establish AVX2 + FMA support.
        let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
        // SAFETY: the feature checks above establish AVX2 + FMA support.
        let (sin, cos) = unsafe { sincos_f32x8(vector) };
        let mut sins = [0.0_f32; 8];
        let mut coss = [0.0_f32; 8];
        // SAFETY: the output arrays have eight f32 lanes.
        unsafe {
            _mm256_storeu_ps(sins.as_mut_ptr(), sin);
            _mm256_storeu_ps(coss.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            let (want_sin, want_cos) = sincos_f32(chunk[lane]);
            if want_sin.is_nan() {
                assert!(sins[lane].is_nan() && coss[lane].is_nan());
            } else {
                assert_eq!(sins[lane].to_bits(), want_sin.to_bits(), "sin lane {lane}");
                assert_eq!(coss[lane].to_bits(), want_cos.to_bits(), "cos lane {lane}");
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_matches_checked_in_mpfr_vectors_when_required_hardware_is_present() {
    if !std::is_x86_feature_detected!("avx512f") {
        if std::env::var_os("SONIUM_FASTMATH_REQUIRE_AVX512").is_some() {
            panic!("SONIUM_FASTMATH_REQUIRE_AVX512=1 but the runner lacks AVX-512F");
        }
        return;
    }
    use sonium_fastmath::avx512::sincos_f64x8;
    use std::arch::x86_64::*;

    for (base, chunk) in GOLDEN.chunks(8).enumerate() {
        let mut input = [0.0_f64; 8];
        for (lane, vector) in chunk.iter().enumerate() {
            input[lane] = f64::from_bits(vector.input);
        }
        let fill = input[0];
        for value in &mut input[chunk.len()..] {
            *value = fill;
        }
        // SAFETY: the feature check above establishes AVX-512F support.
        let vector = unsafe { _mm512_loadu_pd(input.as_ptr()) };
        // SAFETY: the feature check above establishes AVX-512F support.
        let (sin, cos) = unsafe { sincos_f64x8(vector) };
        let mut sins = [0.0; 8];
        let mut coss = [0.0; 8];
        // SAFETY: the output arrays have eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(sins.as_mut_ptr(), sin);
            _mm512_storeu_pd(coss.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            assert_golden_output(base * 8 + lane, chunk[lane], sins[lane], coss[lane]);
        }
    }
}
