//! Throughput comparison: `sonium-fastmath` scalar vs SIMD (NEON or scalar
//! fallback) vs libm `f64::sin_cos`. Synthetic input array; we measure
//! kernel cost only (no allocation amortisation).
//!
//! Run:
//!
//! ```sh
//! cargo bench -p sonium-fastmath --bench sincos_bench
//! ```

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(not(target_arch = "wasm32"))]
use sonium_fastmath::{CODY_WAITE_LIMIT, sincos_f64, sincos_scalar};
#[cfg(not(target_arch = "wasm32"))]
use std::hint::black_box;

#[cfg(target_arch = "aarch64")]
use sonium_fastmath::sincos_f64x2;

#[cfg(target_arch = "x86_64")]
use sonium_fastmath::sincos_f64x4;

#[cfg(target_arch = "x86_64")]
use sonium_fastmath::avx512::sincos_f64x8;

#[cfg(not(target_arch = "wasm32"))]
const SIZES: &[usize] = &[64, 1024, 16_384, 262_144];
#[cfg(not(target_arch = "wasm32"))]
const REGIMES: &[&str] = &[
    "reduced",
    "coherent_ramp",
    "random_full_range",
    "reduction_boundaries",
    "mid_range",
    "huge_fallback",
];

#[cfg(not(target_arch = "wasm32"))]
fn build_inputs(n: usize, regime: &str) -> Vec<f64> {
    match regime {
        "reduced" => (0..n)
            .map(|i| {
                (i as f64 / n as f64) * 2.0 * std::f64::consts::FRAC_PI_4
                    - std::f64::consts::FRAC_PI_4
            })
            .collect(),
        "coherent_ramp" => (0..n)
            .map(|i| {
                (i as f64 / n as f64) * 20.0 * std::f64::consts::PI - 10.0 * std::f64::consts::PI
            })
            .collect(),
        "random_full_range" => {
            let mut state = 0x5eed_u64;
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (2.0 * state as f64 / u64::MAX as f64 - 1.0) * CODY_WAITE_LIMIT
                })
                .collect()
        }
        "reduction_boundaries" => (0..n)
            .map(|i| {
                let k = (i % 257) as f64 - 128.0;
                let offset = (i as f64 % 32.0 - 16.0) * 1.0e-12;
                k * std::f64::consts::FRAC_PI_2 + offset
            })
            .collect(),
        // Three-term Cody–Waite domain: log-uniform in (1e3, 1e8]
        // across both signs. Measures the mid-range path (no heap
        // allocation) against libm and the old bigint fallback.
        "mid_range" => (0..n)
            .map(|i| {
                let t = (i as f64 + 1.0) / (n as f64 + 1.0);
                let mag = 10f64.powf(3.0 + 5.0 * t);
                if i % 2 == 0 { mag } else { -mag }
            })
            .collect(),
        "huge_fallback" => (0..n)
            .map(|i| {
                let magnitude = if i % 2 == 0 {
                    CODY_WAITE_LIMIT + 1.0
                } else {
                    1.0e20 + i as f64
                };
                if i % 4 == 0 { magnitude } else { -magnitude }
            })
            .collect(),
        _ => panic!("unknown fastmath benchmark regime: {regime}"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_libm(xs: &[f64]) -> Vec<(f64, f64)> {
    xs.iter().map(|&x| x.sin_cos()).collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_fastmath_scalar(xs: &[f64]) -> Vec<(f64, f64)> {
    xs.iter().map(|&x| sincos_f64(x)).collect()
}

#[cfg(target_arch = "aarch64")]
fn bench_fastmath_neon(xs: &[f64]) -> Vec<(f64, f64)> {
    use std::arch::aarch64::*;
    let mut out = Vec::with_capacity(xs.len());
    let mut i = 0;
    while i + 2 <= xs.len() {
        // SAFETY: aarch64 + NEON is mandatory on this arch.
        let v = unsafe { vsetq_lane_f64(xs[i + 1], vdupq_n_f64(xs[i]), 1) };
        let (s, c) = unsafe { sincos_f64x2(v) };
        unsafe {
            out.push((vgetq_lane_f64(s, 0), vgetq_lane_f64(c, 0)));
            out.push((vgetq_lane_f64(s, 1), vgetq_lane_f64(c, 1)));
        }
        i += 2;
    }
    while i < xs.len() {
        out.push(sincos_scalar(xs[i]));
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
fn bench_fastmath_avx2(xs: &[f64]) -> Vec<(f64, f64)> {
    use std::arch::x86_64::*;
    let mut out = Vec::with_capacity(xs.len());
    let mut i = 0;
    let mut sout = [0.0_f64; 4];
    let mut cout = [0.0_f64; 4];
    while i + 4 <= xs.len() {
        // SAFETY: caller verified AVX2+FMA via is_x86_feature_detected!
        let v = unsafe { _mm256_set_pd(xs[i + 3], xs[i + 2], xs[i + 1], xs[i]) };
        let (sv, cv) = unsafe { sincos_f64x4(v) };
        unsafe {
            _mm256_storeu_pd(sout.as_mut_ptr(), sv);
            _mm256_storeu_pd(cout.as_mut_ptr(), cv);
        }
        out.push((sout[0], cout[0]));
        out.push((sout[1], cout[1]));
        out.push((sout[2], cout[2]));
        out.push((sout[3], cout[3]));
        i += 4;
    }
    while i < xs.len() {
        out.push(sincos_scalar(xs[i]));
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bench_fastmath_avx512(xs: &[f64]) -> Vec<(f64, f64)> {
    use std::arch::x86_64::*;

    let mut out = Vec::with_capacity(xs.len());
    let mut i = 0;
    let mut sout = [0.0_f64; 8];
    let mut cout = [0.0_f64; 8];
    while i + 8 <= xs.len() {
        let v = _mm512_set_pd(
            xs[i + 7],
            xs[i + 6],
            xs[i + 5],
            xs[i + 4],
            xs[i + 3],
            xs[i + 2],
            xs[i + 1],
            xs[i],
        );
        // SAFETY: this function is only registered when the runtime AVX-512F
        // check in `bench_sincos` succeeds, and the output buffers contain
        // eight f64 lanes.
        let (sv, cv) = unsafe { sincos_f64x8(v) };
        unsafe {
            _mm512_storeu_pd(sout.as_mut_ptr(), sv);
            _mm512_storeu_pd(cout.as_mut_ptr(), cv);
        }
        for lane in 0..8 {
            out.push((sout[lane], cout[lane]));
        }
        i += 8;
    }
    while i < xs.len() {
        out.push(sincos_scalar(xs[i]));
        i += 1;
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_sincos(c: &mut Criterion) {
    let mut group = c.benchmark_group("sincos_f64");

    for &n in SIZES {
        for &regime in REGIMES {
            let xs = build_inputs(n, regime);
            group.throughput(Throughput::Elements(n as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("{regime}/libm"), n),
                &xs,
                |bencher, xs| {
                    bencher.iter(|| black_box(bench_libm(black_box(xs))));
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{regime}/fastmath_scalar"), n),
                &xs,
                |bencher, xs| {
                    bencher.iter(|| black_box(bench_fastmath_scalar(black_box(xs))));
                },
            );

            #[cfg(target_arch = "aarch64")]
            group.bench_with_input(
                BenchmarkId::new(format!("{regime}/fastmath_neon"), n),
                &xs,
                |bencher, xs| {
                    bencher.iter(|| black_box(bench_fastmath_neon(black_box(xs))));
                },
            );

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                group.bench_with_input(
                    BenchmarkId::new(format!("{regime}/fastmath_avx2"), n),
                    &xs,
                    |bencher, xs| {
                        bencher.iter(|| black_box(bench_fastmath_avx2(black_box(xs))));
                    },
                );
            }

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                group.bench_with_input(
                    BenchmarkId::new(format!("{regime}/fastmath_avx512"), n),
                    &xs,
                    |bencher, xs| {
                        bencher.iter(|| unsafe { black_box(bench_fastmath_avx512(black_box(xs))) });
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(benches, bench_sincos);
#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);
