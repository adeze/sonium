//! BLAS-1 comparison across implementations:
//! sonium-compute's in-house CPU kernels vs. platform
//! vendor BLAS (Accelerate / OpenBLAS) vs. accelerator backends
//! (Metal / cuBLAS).
//!
//! The `sonium` bar is the raw CPU kernel call; the `metal` and
//! `cublas` bars go through `MetalBackend` / `CudaBackend` with their
//! inputs pre-staged on the device, so they measure kernel + stream
//! sync only (no host upload).
//!
//! Covers exactly the BLAS-1 surface sonium uses today:
//!
//! - `axpy`  for f32, f64, Complex32, Complex64
//! - `scal`  for f32, f64, Complex32, Complex64
//! - `dot`   (real)  for f32, f64
//! - `dotc`  (Hermitian, complex) for Complex32, Complex64
//! - `nrm2`  for f32, f64, Complex32, Complex64
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p sonium-compute --bench blas1_compare
//! # filter to one op:
//! cargo bench -p sonium-compute --bench blas1_compare -- axpy_f32
//! ```
//!
//! Sweeps `n ∈ {64, 1_024, 16_384, 262_144}` — small/medium/large/
//! L2-spill regimes — so the comparison shows where each implementation
//! wins.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::{Complex32, Complex64};
use std::hint::black_box;

use sonium_compute::cpu::blas1 as sonium_blas;
use sonium_compute::perf;

mod perf_report;

/// Vector sizes swept by every benchmark group. Chosen to span:
///   64       — L1-resident, dispatch-cost dominates
///   1 024    — comfortably L1
///   16 384   — L2 territory
///   262 144  — DRAM-bound (≥ 1 MiB even at f32)
const SIZES: &[usize] = &[64, 1_024, 16_384, 262_144];

// ---------------------------------------------------------------------
// Input generators (deterministic so all three impls see the same data)
// ---------------------------------------------------------------------

fn ramp_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.0123).sin()).collect()
}

fn ramp_f64(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.0123).sin()).collect()
}

fn ramp_c32(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|i| Complex32::new((i as f32 * 0.0123).sin(), (i as f32 * 0.0177).cos()))
        .collect()
}

fn ramp_c64(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new((i as f64 * 0.0123).sin(), (i as f64 * 0.0177).cos()))
        .collect()
}

fn report_precision_check(
    operation: &str,
    scalar: &str,
    shape: &str,
    max_abs: f64,
    max_rel: f64,
    tolerance: f64,
) {
    println!(
        "sonium_precision_check bench=blas1_compare operation={operation} scalar={scalar} \
         shape={shape} max_abs={max_abs:.9e} max_rel={max_rel:.9e} tolerance={tolerance:.9e}"
    );
    assert!(
        max_rel <= tolerance,
        "BLAS-1 {operation} {scalar} precision check failed for {shape}: \
         max_abs={max_abs:.9e}, max_rel={max_rel:.9e}, tolerance={tolerance:.9e}"
    );
}

/// Run independent odd-sized references before Criterion starts timing.
///
/// A benchmark result is not admissible performance evidence unless the exact
/// production kernels in this executable first pass their numerical contract.
fn validate_blas1_correctness(_: &mut Criterion) {
    const N: usize = 37;

    let x = ramp_f32(N);
    let y0 = (0..N)
        .map(|index| ((index as f32) * 0.0191).cos())
        .collect::<Vec<_>>();
    let alpha = 0.7_f32;
    let expected = x
        .iter()
        .zip(&y0)
        .map(|(&x, &y)| (f64::from(y) + f64::from(alpha) * f64::from(x)) as f32)
        .collect::<Vec<_>>();
    let mut got = y0;
    sonium_blas::axpy_f32(alpha, &x, &mut got);
    let (max_abs, max_rel) = got.iter().zip(&expected).fold(
        (0.0_f64, 0.0_f64),
        |(max_abs, max_rel), (&got, &expected)| {
            let abs = f64::from((got - expected).abs());
            let rel = abs / f64::from(expected.abs().max(1.0e-30));
            (max_abs.max(abs), max_rel.max(rel))
        },
    );
    report_precision_check("axpy", "f32", "n=37", max_abs, max_rel, 2.0e-6);

    let mut scaled = ramp_f64(N);
    let expected = scaled
        .iter()
        .map(|value| -0.375 * value)
        .collect::<Vec<_>>();
    sonium_blas::scal_f64(-0.375, &mut scaled);
    let (max_abs, max_rel) = scaled.iter().zip(&expected).fold(
        (0.0_f64, 0.0_f64),
        |(max_abs, max_rel), (&got, &expected)| {
            let abs = (got - expected).abs();
            let rel = abs / expected.abs().max(1.0e-300);
            (max_abs.max(abs), max_rel.max(rel))
        },
    );
    report_precision_check("scal", "f64", "n=37", max_abs, max_rel, 2.0e-14);

    let x = ramp_c64(N);
    let y = (0..N)
        .map(|index| {
            Complex64::new(
                ((index as f64) * 0.0213).cos(),
                ((index as f64) * 0.0097).sin(),
            )
        })
        .collect::<Vec<_>>();
    let expected_dot = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| x.conj() * y)
        .sum::<Complex64>();
    let got_dot = sonium_blas::dotc_c64(&x, &y);
    let abs = (got_dot - expected_dot).norm();
    let rel = abs / expected_dot.norm().max(1.0e-300);
    report_precision_check("dotc", "c64", "n=37", abs, rel, 5.0e-14);

    let expected_norm = x.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    let got_norm = sonium_blas::nrm2_c64(&x);
    let abs = (got_norm - expected_norm).abs();
    let rel = abs / expected_norm.max(1.0e-300);
    report_precision_check("nrm2", "c64", "n=37", abs, rel, 5.0e-14);

    // Keep the comparator's correctness gate representative of every
    // supported complex precision, not just the outer Complex64 path.
    let c32_x = ramp_c32(N);
    let c32_y = (0..N)
        .map(|index| {
            Complex32::new(
                ((index as f32) * 0.0213).cos(),
                ((index as f32) * 0.0097).sin(),
            )
        })
        .collect::<Vec<_>>();
    let c32_alpha = Complex32::new(0.3, -0.7);

    let mut c32_axpy = c32_y.clone();
    sonium_blas::axpy_c32(c32_alpha, &c32_x, &mut c32_axpy);
    let c32_axpy_expected = c32_x
        .iter()
        .zip(&c32_y)
        .map(|(&x, &y)| c32_alpha * x + y)
        .collect::<Vec<_>>();
    let (abs, rel) = c32_axpy.iter().zip(&c32_axpy_expected).fold(
        (0.0_f64, 0.0_f64),
        |(max_abs, max_rel), (actual, expected)| {
            let abs = f64::from((*actual - *expected).norm());
            let rel = abs / f64::from(expected.norm().max(1.0e-30));
            (max_abs.max(abs), max_rel.max(rel))
        },
    );
    report_precision_check("axpy", "c32", "n=37", abs, rel, 5.0e-6);

    let mut c32_scaled = c32_x.clone();
    sonium_blas::scal_c32(c32_alpha, &mut c32_scaled);
    let (abs, rel) = c32_scaled.iter().zip(&c32_x).fold(
        (0.0_f64, 0.0_f64),
        |(max_abs, max_rel), (actual, expected)| {
            let expected = c32_alpha * *expected;
            let abs = f64::from((*actual - expected).norm());
            let rel = abs / f64::from(expected.norm().max(1.0e-30));
            (max_abs.max(abs), max_rel.max(rel))
        },
    );
    report_precision_check("scal", "c32", "n=37", abs, rel, 5.0e-6);

    let expected_dot = c32_x
        .iter()
        .zip(&c32_y)
        .map(|(&x, &y)| x.conj() * y)
        .sum::<Complex32>();
    let got_dot = sonium_blas::dotc_c32(&c32_x, &c32_y);
    let abs = f64::from((got_dot - expected_dot).norm());
    let rel = abs / f64::from(expected_dot.norm().max(1.0e-30));
    report_precision_check("dotc", "c32", "n=37", abs, rel, 5.0e-6);

    let expected_norm = c32_x
        .iter()
        .map(|value| f64::from(value.norm_sqr()))
        .sum::<f64>()
        .sqrt();
    let got_norm = f64::from(sonium_blas::nrm2_c32(&c32_x));
    let abs = (got_norm - expected_norm).abs();
    let rel = abs / expected_norm.max(1.0e-30);
    report_precision_check("nrm2", "c32", "n=37", abs, rel, 5.0e-6);
}

// ---------------------------------------------------------------------
// Apple Accelerate bindings (macOS only — cblas surface).
// ---------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod accelerate {
    use std::os::raw::{c_double, c_float, c_int};

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_saxpy(
            n: c_int,
            alpha: c_float,
            x: *const c_float,
            incx: c_int,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_daxpy(
            n: c_int,
            alpha: c_double,
            x: *const c_double,
            incx: c_int,
            y: *mut c_double,
            incy: c_int,
        );
        pub fn cblas_caxpy(
            n: c_int,
            alpha: *const c_float,
            x: *const c_float,
            incx: c_int,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_zaxpy(
            n: c_int,
            alpha: *const c_double,
            x: *const c_double,
            incx: c_int,
            y: *mut c_double,
            incy: c_int,
        );

        pub fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
        pub fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);
        pub fn cblas_cscal(n: c_int, alpha: *const c_float, x: *mut c_float, incx: c_int);
        pub fn cblas_zscal(n: c_int, alpha: *const c_double, x: *mut c_double, incx: c_int);

        pub fn cblas_sdot(
            n: c_int,
            x: *const c_float,
            incx: c_int,
            y: *const c_float,
            incy: c_int,
        ) -> c_float;
        pub fn cblas_ddot(
            n: c_int,
            x: *const c_double,
            incx: c_int,
            y: *const c_double,
            incy: c_int,
        ) -> c_double;
        pub fn cblas_cdotc_sub(
            n: c_int,
            x: *const c_float,
            incx: c_int,
            y: *const c_float,
            incy: c_int,
            out: *mut c_float,
        );
        pub fn cblas_zdotc_sub(
            n: c_int,
            x: *const c_double,
            incx: c_int,
            y: *const c_double,
            incy: c_int,
            out: *mut c_double,
        );

        pub fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
        pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;
        pub fn cblas_scnrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
        pub fn cblas_dznrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;
    }
}

// ---------------------------------------------------------------------
// OpenBLAS bindings (Linux). The link directive is gated so the bench
// only attempts to link `libopenblas.so` when the user opts in.
// ---------------------------------------------------------------------

#[cfg(all(feature = "bench-openblas", target_os = "linux"))]
mod openblas {
    use std::os::raw::{c_double, c_float, c_int};

    #[link(name = "openblas")]
    unsafe extern "C" {
        pub fn cblas_saxpy(
            n: c_int,
            alpha: c_float,
            x: *const c_float,
            incx: c_int,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_daxpy(
            n: c_int,
            alpha: c_double,
            x: *const c_double,
            incx: c_int,
            y: *mut c_double,
            incy: c_int,
        );
        pub fn cblas_caxpy(
            n: c_int,
            alpha: *const c_float,
            x: *const c_float,
            incx: c_int,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_zaxpy(
            n: c_int,
            alpha: *const c_double,
            x: *const c_double,
            incx: c_int,
            y: *mut c_double,
            incy: c_int,
        );

        pub fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
        pub fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);
        pub fn cblas_cscal(n: c_int, alpha: *const c_float, x: *mut c_float, incx: c_int);
        pub fn cblas_zscal(n: c_int, alpha: *const c_double, x: *mut c_double, incx: c_int);

        pub fn cblas_sdot(
            n: c_int,
            x: *const c_float,
            incx: c_int,
            y: *const c_float,
            incy: c_int,
        ) -> c_float;
        pub fn cblas_ddot(
            n: c_int,
            x: *const c_double,
            incx: c_int,
            y: *const c_double,
            incy: c_int,
        ) -> c_double;
        pub fn cblas_cdotc_sub(
            n: c_int,
            x: *const c_float,
            incx: c_int,
            y: *const c_float,
            incy: c_int,
            out: *mut c_float,
        );
        pub fn cblas_zdotc_sub(
            n: c_int,
            x: *const c_double,
            incx: c_int,
            y: *const c_double,
            incy: c_int,
            out: *mut c_double,
        );

        pub fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
        pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;
        pub fn cblas_scnrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
        pub fn cblas_dznrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;
    }
}

// ---------------------------------------------------------------------
// cuBLAS BLAS-1 harness. The CUDA backend's `axpy` / `dot` / `nrm2` /
// `scale` route through `cublasSaxpy` / `cublasSdot` / etc. The bench
// pre-stages inputs on the device and times the dispatch + synchronize
// only — so the figures reflect kernel cost, not h2d transfer cost.
//
// At BLAS-1 sizes the GPU is usually *slower* than CPU because launch
// latency (~10 µs) dominates a few-µs kernel. Useful as a baseline so
// callers don't try to "accelerate" BLAS-1 onto cuBLAS by accident.
// ---------------------------------------------------------------------

// Metal harnesses for BLAS-1. Same purpose as the cuBLAS ones: bench
// the kernel + sync only, not the host upload. On Apple Silicon the
// MetalBackend currently routes BLAS-1 through native MSL kernels for
// f32/c32 and host-staged compute for f64/c64.

#[cfg(feature = "metal")]
fn run_metal_axpy<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
    alpha: T,
    x_host: &[T],
    y_host: &[T],
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, MetalBackend};
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable, skipping axpy: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    let mut y = pool.try_alloc::<T>(n).unwrap();
    pool.backend().copy_h2d(x_host, x.raw_mut());
    pool.backend().copy_h2d(y_host, y.raw_mut());
    group.bench_with_input(BenchmarkId::new("metal", n), &n, |b, _| {
        b.iter(|| {
            pool.backend().axpy(black_box(alpha), x.raw(), y.raw_mut());
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
        });
    });
}

#[cfg(feature = "metal")]
fn run_metal_scale<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
    alpha: T,
    x_host: &[T],
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, MetalBackend};
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable, skipping scale: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    pool.backend().copy_h2d(x_host, x.raw_mut());
    group.bench_with_input(BenchmarkId::new("metal", n), &n, |b, _| {
        b.iter(|| {
            pool.backend().scale(black_box(alpha), x.raw_mut());
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
        });
    });
}

#[cfg(feature = "metal")]
fn run_metal_dot<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
    x_host: &[T],
    y_host: &[T],
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, MetalBackend};
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable, skipping dot: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    let mut y = pool.try_alloc::<T>(n).unwrap();
    pool.backend().copy_h2d(x_host, x.raw_mut());
    pool.backend().copy_h2d(y_host, y.raw_mut());
    group.bench_with_input(BenchmarkId::new("metal", n), &n, |b, _| {
        b.iter(|| {
            let r = pool.backend().dot(x.raw(), y.raw());
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
            black_box(r)
        });
    });
}

#[cfg(feature = "metal")]
fn run_metal_nrm2<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
    x_host: &[T],
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, MetalBackend};
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable, skipping nrm2: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    pool.backend().copy_h2d(x_host, x.raw_mut());
    group.bench_with_input(BenchmarkId::new("metal", n), &n, |b, _| {
        b.iter(|| {
            let r = pool.backend().nrm2(x.raw());
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
            black_box(r)
        });
    });
}

#[cfg(feature = "cuda")]
fn run_cuda_axpy<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
    alpha: T,
    x_host: &[T],
    y_host: &[T],
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping cuBLAS axpy: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    let mut y = pool.try_alloc::<T>(n).unwrap();
    pool.backend().copy_h2d(x_host, x.raw_mut());
    pool.backend().copy_h2d(y_host, y.raw_mut());
    group.bench_with_input(BenchmarkId::new("cublas", n), &n, |b, _| {
        b.iter(|| {
            pool.backend().axpy(black_box(alpha), x.raw(), y.raw_mut());
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
        });
    });
}

#[cfg(feature = "cuda")]
fn run_cuda_scale<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
    alpha: T,
    x_host: &[T],
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping cuBLAS scale: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    pool.backend().copy_h2d(x_host, x.raw_mut());
    group.bench_with_input(BenchmarkId::new("cublas", n), &n, |b, _| {
        b.iter(|| {
            pool.backend().scale(black_box(alpha), x.raw_mut());
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
        });
    });
}

#[cfg(feature = "cuda")]
fn run_cuda_dot<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
    x_host: &[T],
    y_host: &[T],
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping cuBLAS dot: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    let mut y = pool.try_alloc::<T>(n).unwrap();
    pool.backend().copy_h2d(x_host, x.raw_mut());
    pool.backend().copy_h2d(y_host, y.raw_mut());
    group.bench_with_input(BenchmarkId::new("cublas", n), &n, |b, _| {
        b.iter(|| {
            let r = pool.backend().dot(x.raw(), y.raw());
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
            black_box(r)
        });
    });
}

#[cfg(feature = "cuda")]
fn run_cuda_nrm2<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
    x_host: &[T],
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping cuBLAS nrm2: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    pool.backend().copy_h2d(x_host, x.raw_mut());
    group.bench_with_input(BenchmarkId::new("cublas", n), &n, |b, _| {
        b.iter(|| {
            let r = pool.backend().nrm2(x.raw());
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
            black_box(r)
        });
    });
}

// ---------------------------------------------------------------------
// Each helper takes setup + run closures; criterion re-creates `y`
// per-sample for write-targeting ops (axpy, scal) to keep the comparison
// fair (none of the three impls reads stale output).
// ---------------------------------------------------------------------

fn set_blas1_model(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    scalar: &str,
    n: usize,
    model: perf_report::PerfModel,
) {
    perf_report::set_bytes_throughput(group, op, scalar, &n.to_string(), model);
}

// ---------------------------------------------------------------------
// axpy
// ---------------------------------------------------------------------

fn bench_axpy_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy_f32");
    for &n in SIZES {
        let x = ramp_f32(n);
        let y0 = ramp_f32(n);
        set_blas1_model(&mut group, "axpy", "f32", n, perf::axpy_model::<f32>(n));

        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| {
                sonium_blas::axpy_f32(black_box(0.7), black_box(&x), black_box(&mut y));
            });
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| unsafe {
                accelerate::cblas_saxpy(n as i32, black_box(0.7), x.as_ptr(), 1, y.as_mut_ptr(), 1);
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| unsafe {
                openblas::cblas_saxpy(n as i32, black_box(0.7), x.as_ptr(), 1, y.as_mut_ptr(), 1);
            });
        });
        #[cfg(feature = "metal")]
        run_metal_axpy::<f32>(&mut group, n, 0.7, &x, &y0);
        #[cfg(feature = "cuda")]
        run_cuda_axpy::<f32>(&mut group, n, 0.7, &x, &y0);
    }
    group.finish();
}

fn bench_axpy_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy_f64");
    for &n in SIZES {
        let x = ramp_f64(n);
        let y0 = ramp_f64(n);
        set_blas1_model(&mut group, "axpy", "f64", n, perf::axpy_model::<f64>(n));

        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| {
                sonium_blas::axpy_f64(black_box(0.7), black_box(&x), black_box(&mut y));
            });
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| unsafe {
                accelerate::cblas_daxpy(n as i32, black_box(0.7), x.as_ptr(), 1, y.as_mut_ptr(), 1);
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| unsafe {
                openblas::cblas_daxpy(n as i32, black_box(0.7), x.as_ptr(), 1, y.as_mut_ptr(), 1);
            });
        });
        #[cfg(feature = "metal")]
        run_metal_axpy::<f64>(&mut group, n, 0.7, &x, &y0);
        #[cfg(feature = "cuda")]
        run_cuda_axpy::<f64>(&mut group, n, 0.7, &x, &y0);
    }
    group.finish();
}

fn bench_axpy_c32(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy_c32");
    let alpha = Complex32::new(0.3, -0.7);
    for &n in SIZES {
        let x = ramp_c32(n);
        let y0 = ramp_c32(n);
        set_blas1_model(
            &mut group,
            "axpy",
            "c32",
            n,
            perf::axpy_model::<Complex32>(n),
        );

        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| {
                sonium_blas::axpy_c32(black_box(alpha), black_box(&x), black_box(&mut y));
            });
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            let mut y = y0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            b.iter(|| unsafe {
                accelerate::cblas_caxpy(
                    n as i32,
                    alpha_buf.as_ptr(),
                    x.as_ptr() as *const f32,
                    1,
                    y.as_mut_ptr() as *mut f32,
                    1,
                );
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            let mut y = y0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            b.iter(|| unsafe {
                openblas::cblas_caxpy(
                    n as i32,
                    alpha_buf.as_ptr(),
                    x.as_ptr() as *const f32,
                    1,
                    y.as_mut_ptr() as *mut f32,
                    1,
                );
            });
        });
        #[cfg(feature = "metal")]
        run_metal_axpy::<Complex32>(&mut group, n, alpha, &x, &y0);
        #[cfg(feature = "cuda")]
        run_cuda_axpy::<Complex32>(&mut group, n, alpha, &x, &y0);
    }
    group.finish();
}

fn bench_axpy_c64(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy_c64");
    let alpha = Complex64::new(0.3, -0.7);
    for &n in SIZES {
        let x = ramp_c64(n);
        let y0 = ramp_c64(n);
        set_blas1_model(
            &mut group,
            "axpy",
            "c64",
            n,
            perf::axpy_model::<Complex64>(n),
        );

        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| {
                sonium_blas::axpy_c64(black_box(alpha), black_box(&x), black_box(&mut y));
            });
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            let mut y = y0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            b.iter(|| unsafe {
                accelerate::cblas_zaxpy(
                    n as i32,
                    alpha_buf.as_ptr(),
                    x.as_ptr() as *const f64,
                    1,
                    y.as_mut_ptr() as *mut f64,
                    1,
                );
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            let mut y = y0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            b.iter(|| unsafe {
                openblas::cblas_zaxpy(
                    n as i32,
                    alpha_buf.as_ptr(),
                    x.as_ptr() as *const f64,
                    1,
                    y.as_mut_ptr() as *mut f64,
                    1,
                );
            });
        });
        #[cfg(feature = "metal")]
        run_metal_axpy::<Complex64>(&mut group, n, alpha, &x, &y0);
        #[cfg(feature = "cuda")]
        run_cuda_axpy::<Complex64>(&mut group, n, alpha, &x, &y0);
    }
    group.finish();
}

// ---------------------------------------------------------------------
// scal
// ---------------------------------------------------------------------

fn bench_scal_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("scal_f32");
    for &n in SIZES {
        let x0 = ramp_f32(n);
        set_blas1_model(&mut group, "scale", "f32", n, perf::scale_model::<f32>(n));
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut x = x0.clone();
            b.iter(|| sonium_blas::scal_f32(black_box(0.7), black_box(&mut x)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            let mut x = x0.clone();
            b.iter(|| unsafe {
                accelerate::cblas_sscal(n as i32, black_box(0.7), x.as_mut_ptr(), 1);
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            let mut x = x0.clone();
            b.iter(|| unsafe {
                openblas::cblas_sscal(n as i32, black_box(0.7), x.as_mut_ptr(), 1);
            });
        });
        #[cfg(feature = "metal")]
        run_metal_scale::<f32>(&mut group, n, 0.7, &x0);
        #[cfg(feature = "cuda")]
        run_cuda_scale::<f32>(&mut group, n, 0.7, &x0);
    }
    group.finish();
}

fn bench_scal_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("scal_f64");
    for &n in SIZES {
        let x0 = ramp_f64(n);
        set_blas1_model(&mut group, "scale", "f64", n, perf::scale_model::<f64>(n));
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut x = x0.clone();
            b.iter(|| sonium_blas::scal_f64(black_box(0.7), black_box(&mut x)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            let mut x = x0.clone();
            b.iter(|| unsafe {
                accelerate::cblas_dscal(n as i32, black_box(0.7), x.as_mut_ptr(), 1);
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            let mut x = x0.clone();
            b.iter(|| unsafe {
                openblas::cblas_dscal(n as i32, black_box(0.7), x.as_mut_ptr(), 1);
            });
        });
        #[cfg(feature = "metal")]
        run_metal_scale::<f64>(&mut group, n, 0.7, &x0);
        #[cfg(feature = "cuda")]
        run_cuda_scale::<f64>(&mut group, n, 0.7, &x0);
    }
    group.finish();
}

fn bench_scal_c32(c: &mut Criterion) {
    let mut group = c.benchmark_group("scal_c32");
    let alpha = Complex32::new(0.3, -0.7);
    for &n in SIZES {
        let x0 = ramp_c32(n);
        set_blas1_model(
            &mut group,
            "scale",
            "c32",
            n,
            perf::scale_model::<Complex32>(n),
        );
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut x = x0.clone();
            b.iter(|| sonium_blas::scal_c32(black_box(alpha), black_box(&mut x)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            let mut x = x0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            b.iter(|| unsafe {
                accelerate::cblas_cscal(
                    n as i32,
                    alpha_buf.as_ptr(),
                    x.as_mut_ptr() as *mut f32,
                    1,
                );
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            let mut x = x0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            b.iter(|| unsafe {
                openblas::cblas_cscal(n as i32, alpha_buf.as_ptr(), x.as_mut_ptr() as *mut f32, 1);
            });
        });
        #[cfg(feature = "metal")]
        run_metal_scale::<Complex32>(&mut group, n, alpha, &x0);
        #[cfg(feature = "cuda")]
        run_cuda_scale::<Complex32>(&mut group, n, alpha, &x0);
    }
    group.finish();
}

fn bench_scal_c64(c: &mut Criterion) {
    let mut group = c.benchmark_group("scal_c64");
    let alpha = Complex64::new(0.3, -0.7);
    for &n in SIZES {
        let x0 = ramp_c64(n);
        set_blas1_model(
            &mut group,
            "scale",
            "c64",
            n,
            perf::scale_model::<Complex64>(n),
        );
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut x = x0.clone();
            b.iter(|| sonium_blas::scal_c64(black_box(alpha), black_box(&mut x)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            let mut x = x0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            b.iter(|| unsafe {
                accelerate::cblas_zscal(
                    n as i32,
                    alpha_buf.as_ptr(),
                    x.as_mut_ptr() as *mut f64,
                    1,
                );
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            let mut x = x0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            b.iter(|| unsafe {
                openblas::cblas_zscal(n as i32, alpha_buf.as_ptr(), x.as_mut_ptr() as *mut f64, 1);
            });
        });
        #[cfg(feature = "metal")]
        run_metal_scale::<Complex64>(&mut group, n, alpha, &x0);
        #[cfg(feature = "cuda")]
        run_cuda_scale::<Complex64>(&mut group, n, alpha, &x0);
    }
    group.finish();
}

// ---------------------------------------------------------------------
// dot / dotc
// ---------------------------------------------------------------------

fn bench_dot_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f32");
    for &n in SIZES {
        let x = ramp_f32(n);
        let y = ramp_f32(n);
        set_blas1_model(&mut group, "dot", "f32", n, perf::dot_model::<f32>(n));
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            b.iter(|| sonium_blas::dot_f32(black_box(&x), black_box(&y)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            b.iter(|| unsafe { accelerate::cblas_sdot(n as i32, x.as_ptr(), 1, y.as_ptr(), 1) });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            b.iter(|| unsafe { openblas::cblas_sdot(n as i32, x.as_ptr(), 1, y.as_ptr(), 1) });
        });
        #[cfg(feature = "metal")]
        run_metal_dot::<f32>(&mut group, n, &x, &y);
        #[cfg(feature = "cuda")]
        run_cuda_dot::<f32>(&mut group, n, &x, &y);
    }
    group.finish();
}

fn bench_dot_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f64");
    for &n in SIZES {
        let x = ramp_f64(n);
        let y = ramp_f64(n);
        set_blas1_model(&mut group, "dot", "f64", n, perf::dot_model::<f64>(n));
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            b.iter(|| sonium_blas::dot_f64(black_box(&x), black_box(&y)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            b.iter(|| unsafe { accelerate::cblas_ddot(n as i32, x.as_ptr(), 1, y.as_ptr(), 1) });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            b.iter(|| unsafe { openblas::cblas_ddot(n as i32, x.as_ptr(), 1, y.as_ptr(), 1) });
        });
        #[cfg(feature = "metal")]
        run_metal_dot::<f64>(&mut group, n, &x, &y);
        #[cfg(feature = "cuda")]
        run_cuda_dot::<f64>(&mut group, n, &x, &y);
    }
    group.finish();
}

fn bench_dotc_c32(c: &mut Criterion) {
    let mut group = c.benchmark_group("dotc_c32");
    for &n in SIZES {
        let x = ramp_c32(n);
        let y = ramp_c32(n);
        set_blas1_model(
            &mut group,
            "dotc",
            "c32",
            n,
            perf::dot_model::<Complex32>(n),
        );
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            b.iter(|| sonium_blas::dotc_c32(black_box(&x), black_box(&y)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            b.iter(|| {
                let mut out = [0.0_f32; 2];
                unsafe {
                    accelerate::cblas_cdotc_sub(
                        n as i32,
                        x.as_ptr() as *const f32,
                        1,
                        y.as_ptr() as *const f32,
                        1,
                        out.as_mut_ptr(),
                    );
                }
                Complex32::new(out[0], out[1])
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            b.iter(|| {
                let mut out = [0.0_f32; 2];
                unsafe {
                    openblas::cblas_cdotc_sub(
                        n as i32,
                        x.as_ptr() as *const f32,
                        1,
                        y.as_ptr() as *const f32,
                        1,
                        out.as_mut_ptr(),
                    );
                }
                Complex32::new(out[0], out[1])
            });
        });
        #[cfg(feature = "metal")]
        run_metal_dot::<Complex32>(&mut group, n, &x, &y);
        #[cfg(feature = "cuda")]
        run_cuda_dot::<Complex32>(&mut group, n, &x, &y);
    }
    group.finish();
}

fn bench_dotc_c64(c: &mut Criterion) {
    let mut group = c.benchmark_group("dotc_c64");
    for &n in SIZES {
        let x = ramp_c64(n);
        let y = ramp_c64(n);
        set_blas1_model(
            &mut group,
            "dotc",
            "c64",
            n,
            perf::dot_model::<Complex64>(n),
        );
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            b.iter(|| sonium_blas::dotc_c64(black_box(&x), black_box(&y)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            b.iter(|| {
                let mut out = [0.0_f64; 2];
                unsafe {
                    accelerate::cblas_zdotc_sub(
                        n as i32,
                        x.as_ptr() as *const f64,
                        1,
                        y.as_ptr() as *const f64,
                        1,
                        out.as_mut_ptr(),
                    );
                }
                Complex64::new(out[0], out[1])
            });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            b.iter(|| {
                let mut out = [0.0_f64; 2];
                unsafe {
                    openblas::cblas_zdotc_sub(
                        n as i32,
                        x.as_ptr() as *const f64,
                        1,
                        y.as_ptr() as *const f64,
                        1,
                        out.as_mut_ptr(),
                    );
                }
                Complex64::new(out[0], out[1])
            });
        });
        #[cfg(feature = "metal")]
        run_metal_dot::<Complex64>(&mut group, n, &x, &y);
        #[cfg(feature = "cuda")]
        run_cuda_dot::<Complex64>(&mut group, n, &x, &y);
    }
    group.finish();
}

// ---------------------------------------------------------------------
// nrm2
// ---------------------------------------------------------------------

fn bench_nrm2_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("nrm2_f32");
    for &n in SIZES {
        let x = ramp_f32(n);
        set_blas1_model(&mut group, "nrm2", "f32", n, perf::nrm2_model::<f32>(n));
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            b.iter(|| sonium_blas::nrm2_f32(black_box(&x)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            b.iter(|| unsafe { accelerate::cblas_snrm2(n as i32, x.as_ptr(), 1) });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            b.iter(|| unsafe { openblas::cblas_snrm2(n as i32, x.as_ptr(), 1) });
        });
        #[cfg(feature = "metal")]
        run_metal_nrm2::<f32>(&mut group, n, &x);
        #[cfg(feature = "cuda")]
        run_cuda_nrm2::<f32>(&mut group, n, &x);
    }
    group.finish();
}

fn bench_nrm2_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("nrm2_f64");
    for &n in SIZES {
        let x = ramp_f64(n);
        set_blas1_model(&mut group, "nrm2", "f64", n, perf::nrm2_model::<f64>(n));
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            b.iter(|| sonium_blas::nrm2_f64(black_box(&x)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            b.iter(|| unsafe { accelerate::cblas_dnrm2(n as i32, x.as_ptr(), 1) });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            b.iter(|| unsafe { openblas::cblas_dnrm2(n as i32, x.as_ptr(), 1) });
        });
        #[cfg(feature = "metal")]
        run_metal_nrm2::<f64>(&mut group, n, &x);
        #[cfg(feature = "cuda")]
        run_cuda_nrm2::<f64>(&mut group, n, &x);
    }
    group.finish();
}

fn bench_nrm2_c32(c: &mut Criterion) {
    let mut group = c.benchmark_group("nrm2_c32");
    for &n in SIZES {
        let x = ramp_c32(n);
        set_blas1_model(
            &mut group,
            "nrm2",
            "c32",
            n,
            perf::nrm2_model::<Complex32>(n),
        );
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            b.iter(|| sonium_blas::nrm2_c32(black_box(&x)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            b.iter(|| unsafe { accelerate::cblas_scnrm2(n as i32, x.as_ptr() as *const f32, 1) });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            b.iter(|| unsafe { openblas::cblas_scnrm2(n as i32, x.as_ptr() as *const f32, 1) });
        });
        #[cfg(feature = "metal")]
        run_metal_nrm2::<Complex32>(&mut group, n, &x);
        #[cfg(feature = "cuda")]
        run_cuda_nrm2::<Complex32>(&mut group, n, &x);
    }
    group.finish();
}

fn bench_nrm2_c64(c: &mut Criterion) {
    let mut group = c.benchmark_group("nrm2_c64");
    for &n in SIZES {
        let x = ramp_c64(n);
        set_blas1_model(
            &mut group,
            "nrm2",
            "c64",
            n,
            perf::nrm2_model::<Complex64>(n),
        );
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            b.iter(|| sonium_blas::nrm2_c64(black_box(&x)));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
            b.iter(|| unsafe { accelerate::cblas_dznrm2(n as i32, x.as_ptr() as *const f64, 1) });
        });
        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", n), &n, |b, _| {
            b.iter(|| unsafe { openblas::cblas_dznrm2(n as i32, x.as_ptr() as *const f64, 1) });
        });
        #[cfg(feature = "metal")]
        run_metal_nrm2::<Complex64>(&mut group, n, &x);
        #[cfg(feature = "cuda")]
        run_cuda_nrm2::<Complex64>(&mut group, n, &x);
    }
    group.finish();
}

criterion_group!(
    blas1_compare,
    validate_blas1_correctness,
    bench_axpy_f32,
    bench_axpy_f64,
    bench_axpy_c32,
    bench_axpy_c64,
    bench_scal_f32,
    bench_scal_f64,
    bench_scal_c32,
    bench_scal_c64,
    bench_dot_f32,
    bench_dot_f64,
    bench_dotc_c32,
    bench_dotc_c64,
    bench_nrm2_f32,
    bench_nrm2_f64,
    bench_nrm2_c32,
    bench_nrm2_c64,
);
criterion_main!(blas1_compare);
