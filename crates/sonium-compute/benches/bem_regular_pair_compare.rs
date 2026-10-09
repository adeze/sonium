//! BEM regular-pair Helmholtz kernel comparison:
//! CPU scalar f64 (serial) vs CPU scalar f64 (rayon) vs Metal C32 vs CUDA C32.
//!
//! Each pair is one source point + outward normal + a Tri3 field element. The
//! kernel produces four complex integrals `(G, H, H^T, E)` per pair using the
//! same fixed 4-point Gauss-tri quadrature on all paths. Pairs are
//! well-separated (`distance/sqrt(area) >= 3`) so a single 4-point quad is
//! the correct rule — matches what the production room-acoustics dense BLR
//! path dispatches to its far-field tail.
//!
//! Run:
//!
//! ```sh
//! cargo bench -p sonium-compute --bench bem_regular_pair_compare
//! # filter to one size:
//! cargo bench -p sonium-compute --bench bem_regular_pair_compare -- /1024
//! ```
//!
//! Sweeps `n_pairs ∈ {64, 1_024, 16_384, 262_144, 1_048_576}` so the
//! comparison spans the GPU launch-overhead regime up to large room-mesh
//! workloads (~10⁶ pairs at 200 Hz for `small_stereo_2_1`).
//!
//! Notes:
//! - The Metal/CUDA bars include kernel launch + host readback. They do NOT
//!   include backend construction (one-time pipeline state setup).
//! - The CPU bars are pure f64 scalar arithmetic — the production
//!   `regular_integration_with_normals` path also runs scalar f64, so the
//!   speedups reported here track the production speedup for the
//!   `assemble` slice of the dense-BLR log.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex64;
use rayon::prelude::*;
use sonium_compute::BemRegularPairIn;
#[cfg(any(feature = "cuda", feature = "metal"))]
use sonium_compute::BemRegularPairOut;
use sonium_compute::bem_cpu::{RegularPairInF64, regular_pair_batch};
use std::f64::consts::PI;
use std::hint::black_box;

/// 4-point Tri3 quadrature, weights pre-scaled by 0.5 (reference triangle
/// area). Must match `SONIUM_BEM_TRI4_QUAD` in `kernels.metal`, the CUDA
/// PTX kernel, and `GAUCORWEI_TR4` in `sonium-bem`'s `gauss.rs`.
const TRI4_QUAD: [(f64, f64, f64); 4] = [
    (0.333_333_333_333_333, 0.333_333_333_333_333, -0.281_250),
    (0.6, 0.2, 0.260_416_666_666_667),
    (0.2, 0.6, 0.260_416_666_666_667),
    (0.2, 0.2, 0.260_416_666_666_667),
];

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale_v(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn axpy_v(a: [f64; 3], t: f64, b: [f64; 3]) -> [f64; 3] {
    [a[0] + t * b[0], a[1] + t * b[1], a[2] + t * b[2]]
}

/// f64 reference for the four Helmholtz kernel integrals over a single
/// well-separated Tri3 field element. Identical math to `kernels.metal`
/// and the CUDA PTX kernel; this is the canonical CPU path.
#[inline]
fn cpu_regular_pair_f64(
    source_point: [f64; 3],
    source_normal: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    wave_number: f64,
    harmonic_factor: f64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let wavruim = harmonic_factor * wave_number;
    let k2 = wave_number * wave_number;

    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let raw_n = cross(e1, e2);
    let jac = norm(raw_n);
    let n_y = if jac > 1e-15 {
        scale_v(raw_n, 1.0 / jac)
    } else {
        [0.0; 3]
    };

    let mut g = Complex64::new(0.0, 0.0);
    let mut h = Complex64::new(0.0, 0.0);
    let mut ht = Complex64::new(0.0, 0.0);
    let mut e = Complex64::new(0.0, 0.0);

    for &(xi, eta, w) in TRI4_QUAD.iter() {
        let n0 = 1.0 - xi - eta;
        let crd = axpy_v(axpy_v(scale_v(v0, n0), xi, v1), eta, v2);
        let wga = w * jac;
        let diff = sub(crd, source_point);
        let r = norm(diff);
        if r < 1e-15 {
            continue;
        }
        let ur = scale_v(diff, 1.0 / r);
        let re1 = wavruim * r;
        let re2 = wga / (4.0 * PI * r);
        let zg = Complex64::new(re1.cos() * re2, re1.sin() * re2);
        let z1 = Complex64::new(-1.0 / r, wavruim);
        let zhh_base = zg * z1;
        let r1h = dot(ur, n_y);
        let r2h = -dot(ur, source_normal);
        let zhh = zhh_base * r1h;
        let zht = zhh_base * r2h;
        let rq = r1h * r2h;
        let nx_dot_ny = dot(source_normal, n_y);
        let dq = r * r;
        let ze_factor = Complex64::new(
            (3.0 / dq - k2) * rq + nx_dot_ny / dq,
            -wavruim / r * (3.0 * rq + nx_dot_ny),
        );
        let ze = zg * ze_factor;
        g += zg;
        h += zhh;
        ht += zht;
        e += ze;
    }
    (g, h, ht, e)
}

/// A workload of `n` well-separated Tri3 pairs synthesised on a deterministic
/// pseudo-random grid. Returns the GPU-side input batch (f32) plus parallel
/// f64 arrays for the CPU paths (no f32→f64 conversion on the hot path).
struct PairFixture {
    #[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
    gpu: Vec<BemRegularPairIn>,
    cpu: Vec<RegularPairInF64>,
    sources: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
    v0s: Vec<[f64; 3]>,
    v1s: Vec<[f64; 3]>,
    v2s: Vec<[f64; 3]>,
}

fn build_fixture(n: usize) -> PairFixture {
    // Deterministic pseudo-random pair generator. Spreads source points and
    // triangle centroids over a [-5, 5]³ box with triangle edge sizes ~0.3,
    // guaranteeing `dist/sqrt(area)` well above the quasi-singular threshold
    // of 3 used by the production assembler.
    let mut gpu = Vec::with_capacity(n);
    let mut cpu = Vec::with_capacity(n);
    let mut sources = Vec::with_capacity(n);
    let mut normals = Vec::with_capacity(n);
    let mut v0s = Vec::with_capacity(n);
    let mut v1s = Vec::with_capacity(n);
    let mut v2s = Vec::with_capacity(n);

    let frac = |idx: usize, salt: u64| -> f64 {
        let x = (idx as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(salt);
        let x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let x = x ^ (x >> 31);
        (x as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    for i in 0..n {
        let sp = [5.0 * frac(i, 1), 5.0 * frac(i, 2), 5.0 * frac(i, 3)];
        let mut sn = [frac(i, 4), frac(i, 5), frac(i, 6)];
        let snorm = norm(sn).max(1e-9);
        sn = scale_v(sn, 1.0 / snorm);

        let base = [
            5.0 * frac(i, 7) + (if frac(i, 7) > 0.0 { -8.0 } else { 8.0 }),
            5.0 * frac(i, 8) + (if frac(i, 8) > 0.0 { -8.0 } else { 8.0 }),
            5.0 * frac(i, 9) + (if frac(i, 9) > 0.0 { -8.0 } else { 8.0 }),
        ];
        let v0 = base;
        let v1 = [base[0] + 0.3, base[1] + 0.05 * frac(i, 10), base[2]];
        let v2 = [base[0] + 0.05 * frac(i, 11), base[1] + 0.3, base[2]];

        gpu.push(BemRegularPairIn::from_triangle(
            [sp[0] as f32, sp[1] as f32, sp[2] as f32],
            [sn[0] as f32, sn[1] as f32, sn[2] as f32],
            [v0[0] as f32, v0[1] as f32, v0[2] as f32],
            [v1[0] as f32, v1[1] as f32, v1[2] as f32],
            [v2[0] as f32, v2[1] as f32, v2[2] as f32],
        ));
        cpu.push(RegularPairInF64 {
            source_point: sp,
            source_normal: sn,
            v0,
            v1,
            v2,
        });
        sources.push(sp);
        normals.push(sn);
        v0s.push(v0);
        v1s.push(v1);
        v2s.push(v2);
    }

    PairFixture {
        gpu,
        cpu,
        sources,
        normals,
        v0s,
        v1s,
        v2s,
    }
}

const WAVE_NUMBER: f64 = 1.5;
const HARMONIC_FACTOR: f64 = 1.0;

fn cpu_serial(fx: &PairFixture) -> Vec<(Complex64, Complex64, Complex64, Complex64)> {
    fx.sources
        .iter()
        .zip(fx.normals.iter())
        .zip(fx.v0s.iter())
        .zip(fx.v1s.iter())
        .zip(fx.v2s.iter())
        .map(|((((sp, sn), v0), v1), v2)| {
            cpu_regular_pair_f64(*sp, *sn, *v0, *v1, *v2, WAVE_NUMBER, HARMONIC_FACTOR)
        })
        .collect()
}

fn cpu_parallel(fx: &PairFixture) -> Vec<(Complex64, Complex64, Complex64, Complex64)> {
    (0..fx.sources.len())
        .into_par_iter()
        .map(|i| {
            cpu_regular_pair_f64(
                fx.sources[i],
                fx.normals[i],
                fx.v0s[i],
                fx.v1s[i],
                fx.v2s[i],
                WAVE_NUMBER,
                HARMONIC_FACTOR,
            )
        })
        .collect()
}

fn bench_bem(c: &mut Criterion) {
    let sizes: &[usize] = &[64, 1_024, 16_384, 262_144, 1_048_576];

    let mut group = c.benchmark_group("bem_regular_pair");

    #[cfg(feature = "metal")]
    let metal = match sonium_compute::MetalBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("Metal backend unavailable: {e:?}");
            None
        }
    };

    #[cfg(feature = "cuda")]
    let cuda = match sonium_compute::cuda::CudaBackend::new_default() {
        Ok(b) if b.has_bem_regular_pair_c32() => Some(b),
        Ok(_) => {
            eprintln!("CUDA backend present but BEM regular-pair kernel missing");
            None
        }
        Err(e) => {
            eprintln!("CUDA backend unavailable: {e:?}");
            None
        }
    };

    for &n in sizes {
        let fx = build_fixture(n);
        group.throughput(Throughput::Elements(n as u64));

        // 16 384 serial f64 pair-evals is ~milliseconds; 1 048 576 is
        // seconds. Above 16 384 we skip serial so criterion's sample cap
        // doesn't push individual iterations over the warning threshold.
        if n <= 16_384 {
            group.bench_with_input(BenchmarkId::new("cpu_serial_f64", n), &fx, |bencher, fx| {
                bencher.iter(|| black_box(cpu_serial(black_box(fx))));
            });
        }

        group.bench_with_input(
            BenchmarkId::new("cpu_parallel_f64", n),
            &fx,
            |bencher, fx| {
                bencher.iter(|| black_box(cpu_parallel(black_box(fx))));
            },
        );

        // SIMD bar: NEON-vectorised across 2 pairs on aarch64 (scalar
        // elsewhere). Serial driver — apples-to-apples vs cpu_serial_f64.
        if n <= 16_384 {
            group.bench_with_input(BenchmarkId::new("cpu_simd_f64", n), &fx, |bencher, fx| {
                bencher.iter(|| {
                    let out = regular_pair_batch(black_box(&fx.cpu), WAVE_NUMBER, HARMONIC_FACTOR);
                    black_box(out)
                });
            });
        }

        // SIMD + rayon: chunk pairs across worker threads, each thread runs
        // the NEON-vectorised batch on its chunk. Mirrors what production
        // assembly would dispatch once `regular_integration_with_normals`
        // is taught to call through `bem_cpu::regular_pair_batch`. Chunk
        // size targets ~1k pairs per task so rayon's dispatch overhead is
        // amortised even at moderate N — smaller chunks lose to scalar at
        // these problem sizes.
        let chunk = (n / (rayon::current_num_threads() * 4)).clamp(1024, 16_384);
        group.bench_with_input(
            BenchmarkId::new("cpu_simd_parallel_f64", n),
            &fx,
            |bencher, fx| {
                bencher.iter(|| {
                    let out: Vec<_> = fx
                        .cpu
                        .par_chunks(chunk)
                        .flat_map_iter(|c| {
                            regular_pair_batch(c, WAVE_NUMBER, HARMONIC_FACTOR).into_iter()
                        })
                        .collect();
                    black_box(out)
                });
            },
        );

        #[cfg(feature = "metal")]
        if let Some(backend) = metal.as_ref() {
            group.bench_with_input(BenchmarkId::new("metal_c32", n), &fx, |bencher, fx| {
                bencher.iter(|| {
                    let out: Vec<BemRegularPairOut> = backend.bem_regular_pair_c32_batch(
                        black_box(&fx.gpu),
                        WAVE_NUMBER as f32,
                        HARMONIC_FACTOR as f32,
                    );
                    black_box(out)
                });
            });
        }

        #[cfg(feature = "cuda")]
        if let Some(backend) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda_c32", n), &fx, |bencher, fx| {
                bencher.iter(|| {
                    let out: Vec<BemRegularPairOut> = backend.bem_regular_pair_c32_batch(
                        black_box(&fx.gpu),
                        WAVE_NUMBER as f32,
                        HARMONIC_FACTOR as f32,
                    );
                    black_box(out)
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_bem);
criterion_main!(benches);
