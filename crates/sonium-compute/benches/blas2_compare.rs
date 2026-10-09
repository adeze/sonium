//! BLAS-2 (`gemv`) comparison across implementations.
//!
//! Compares sonium-compute's in-house kernels against external BLAS
//! libraries for the operation `y := α·op(A)·x + β·y`:
//!
//! - **sonium** — `sonium_compute::cpu::blas2::gemv_*` (row-major,
//!   forwards real `NoTrans` to SME2 on AArch64).
//! - **oxiblas** — pure-Rust BLAS-2 from `oxiblas-blas`
//!   (column-major via `MatRef`).
//! - **accelerate** — Apple Accelerate `cblas_?gemv` (macOS only).
//! - **openblas** — Linux dynamic-linked `libopenblas.so` cblas surface
//!   (gated on the `bench-openblas` feature).
//! - **metal_mps** — experimental MPSMatrixVectorMultiplication f32 GEMV,
//!   measured beside the default Metal MSL kernel (gated on `metal`).
//! - **metal_mps_packed_trans** — MPS f32 Trans GEMV with `Aᵀ` prepacked as
//!   a row-major matrix, making the required layout cost explicit.
//! - **cublas** — `sonium_compute::CudaBackend::gemv` measured with the
//!   workload pre-staged on the device (gated on the `cuda` feature).
//!
//! Each comparator gets its native storage layout (row-major for sonium,
//! column-major for the cblas family) so the comparison is honest: each
//! library does the work it would do at a real call site.
//!
//! ```sh
//! cargo bench -p sonium-compute --bench blas2_compare
//! # filter:
//! cargo bench -p sonium-compute --bench blas2_compare -- gemv_f64
//! # with OpenBLAS (Linux):
//! cargo bench -p sonium-compute --bench blas2_compare --features bench-openblas
//! # with cuBLAS:
//! cargo bench -p sonium-compute --bench blas2_compare --features cuda
//! ```

#![allow(clippy::too_many_arguments)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::{Complex32, Complex64};
use std::hint::black_box;

use sonium_compute::cpu::blas2 as sonium_blas2;
use sonium_compute::scalar::DeviceScalar;
use sonium_compute::{Transpose, perf};

mod perf_report;

/// Matrix shapes `(m, n)` swept by every group.
///
///   64×64       — L1-resident, dispatch-cost dominates
///   256×256     — L2 territory
///   1024×1024   — DRAM bound (~8 MiB at f64)
///   4096×1024   — tall-skinny, common in BEM right-hand sides
const SIZES: &[(usize, usize)] = &[(64, 64), (256, 256), (1024, 1024), (4096, 1024)];

// ---------------------------------------------------------------------
// Data generators. We allocate both row-major (sonium) and column-major
// (everyone else) copies of A so each library reads from its preferred
// stride. `to_col_major` transposes in place — `A_cm[i + j*lda] = A[i,j]`.
// ---------------------------------------------------------------------

fn ramp_f32(m: usize, n: usize) -> Vec<f32> {
    (0..m * n).map(|k| ((k as f32) * 0.0123).sin()).collect()
}

fn ramp_f64(m: usize, n: usize) -> Vec<f64> {
    (0..m * n).map(|k| ((k as f64) * 0.0123).sin()).collect()
}

fn ramp_c32(m: usize, n: usize) -> Vec<Complex32> {
    (0..m * n)
        .map(|k| Complex32::new((k as f32 * 0.0123).sin(), (k as f32 * 0.0177).cos()))
        .collect()
}

fn ramp_c64(m: usize, n: usize) -> Vec<Complex64> {
    (0..m * n)
        .map(|k| Complex64::new((k as f64 * 0.0123).sin(), (k as f64 * 0.0177).cos()))
        .collect()
}

fn to_col_major<T: Copy + Default>(rm: &[T], m: usize, n: usize) -> Vec<T> {
    let mut cm = vec![T::default(); m * n];
    for i in 0..m {
        for j in 0..n {
            cm[i + j * m] = rm[i * n + j];
        }
    }
    cm
}

fn report_precision_check(scalar: &str, shape: &str, max_abs: f64, max_rel: f64, tolerance: f64) {
    println!(
        "sonium_precision_check bench=blas2_compare operation=gemv scalar={scalar} \
         shape={shape} variant=no_trans max_abs={max_abs:.9e} max_rel={max_rel:.9e} \
         tolerance={tolerance:.9e}"
    );
    assert!(
        max_rel <= tolerance,
        "BLAS-2 GEMV {scalar} precision check failed for {shape}: \
         max_abs={max_abs:.9e}, max_rel={max_rel:.9e}, tolerance={tolerance:.9e}"
    );
}

/// Check an odd rectangular GEMV against an independent f64 accumulator
/// before collecting any timing samples.
fn validate_blas2_correctness(_: &mut Criterion) {
    const M: usize = 13;
    const N: usize = 7;
    let alpha = 0.7_f32;
    let beta = -0.25_f32;
    let a = ramp_f32(M, N);
    let x = (0..N)
        .map(|index| ((index as f32) * 0.0317).cos())
        .collect::<Vec<_>>();
    let y0 = (0..M)
        .map(|index| ((index as f32) * 0.0179).sin())
        .collect::<Vec<_>>();
    let expected = (0..M)
        .map(|row| {
            let product = (0..N)
                .map(|col| f64::from(a[row * N + col]) * f64::from(x[col]))
                .sum::<f64>();
            (f64::from(alpha) * product + f64::from(beta) * f64::from(y0[row])) as f32
        })
        .collect::<Vec<_>>();
    let mut got = y0;
    sonium_blas2::gemv_f32(Transpose::NoTrans, M, N, alpha, &a, N, &x, beta, &mut got);
    let (max_abs, max_rel) = got.iter().zip(&expected).fold(
        (0.0_f64, 0.0_f64),
        |(max_abs, max_rel), (&got, &expected)| {
            let abs = f64::from((got - expected).abs());
            let rel = abs / f64::from(expected.abs().max(1.0e-30));
            (max_abs.max(abs), max_rel.max(rel))
        },
    );
    report_precision_check("f32", "13x7", max_abs, max_rel, 4.0e-6);

    // Exercise the Complex32 comparator path as well. The benchmark matrix
    // already times this type; the pre-timing correctness gate must cover it.
    let alpha = Complex32::new(0.7, -0.2);
    let beta = Complex32::new(-0.25, 0.1);
    let a = ramp_c32(M, N);
    let x = (0..N)
        .map(|index| {
            Complex32::new(
                ((index as f32) * 0.0317).cos(),
                ((index as f32) * 0.0119).sin(),
            )
        })
        .collect::<Vec<_>>();
    let y0 = (0..M)
        .map(|index| {
            Complex32::new(
                ((index as f32) * 0.0179).sin(),
                ((index as f32) * 0.0231).cos(),
            )
        })
        .collect::<Vec<_>>();
    let expected = (0..M)
        .map(|row| {
            let product = (0..N)
                .map(|column| a[row * N + column] * x[column])
                .sum::<Complex32>();
            alpha * product + beta * y0[row]
        })
        .collect::<Vec<_>>();
    let mut got = y0;
    sonium_blas2::gemv_c32(Transpose::NoTrans, M, N, alpha, &a, N, &x, beta, &mut got);
    let (max_abs, max_rel) = got.iter().zip(&expected).fold(
        (0.0_f64, 0.0_f64),
        |(max_abs, max_rel), (actual, expected)| {
            let abs = f64::from((*actual - *expected).norm());
            let rel = abs / f64::from(expected.norm().max(1.0e-30));
            (max_abs.max(abs), max_rel.max(rel))
        },
    );
    report_precision_check("c32", "13x7", max_abs, max_rel, 5.0e-6);
}

fn set_gemv_model<T: DeviceScalar>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scalar: &str,
    label: &str,
    m: usize,
    n: usize,
) {
    perf_report::set_flop_throughput(group, "gemv", scalar, label, perf::gemv_model::<T>(m, n));
}

// ---------------------------------------------------------------------
// Apple Accelerate bindings (macOS).
// ---------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod accelerate {
    use std::os::raw::{c_double, c_float, c_int};
    #[repr(C)]
    #[allow(non_camel_case_types, dead_code)]
    #[derive(Clone, Copy)]
    pub enum CBLAS_ORDER {
        RowMajor = 101,
        ColMajor = 102,
    }
    #[repr(C)]
    #[allow(non_camel_case_types, dead_code)]
    #[derive(Clone, Copy)]
    pub enum CBLAS_TRANSPOSE {
        NoTrans = 111,
        Trans = 112,
        ConjTrans = 113,
    }

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemv(
            order: CBLAS_ORDER,
            trans: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            incx: c_int,
            beta: c_float,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_dgemv(
            order: CBLAS_ORDER,
            trans: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            incx: c_int,
            beta: c_double,
            y: *mut c_double,
            incy: c_int,
        );
        pub fn cblas_cgemv(
            order: CBLAS_ORDER,
            trans: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            alpha: *const c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            incx: c_int,
            beta: *const c_float,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_zgemv(
            order: CBLAS_ORDER,
            trans: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            alpha: *const c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            incx: c_int,
            beta: *const c_double,
            y: *mut c_double,
            incy: c_int,
        );
    }
}

// ---------------------------------------------------------------------
// OpenBLAS bindings (Linux, opt-in via `bench-openblas` feature).
//
// OpenBLAS ships the same cblas surface as Accelerate. Link directive is
// gated so the bench compiles without `libopenblas.so` on the host.
// ---------------------------------------------------------------------

#[cfg(all(feature = "bench-openblas", target_os = "linux"))]
mod openblas {
    use std::os::raw::{c_double, c_float, c_int};
    #[repr(C)]
    #[allow(non_camel_case_types, dead_code)]
    #[derive(Clone, Copy)]
    pub enum CBLAS_ORDER {
        RowMajor = 101,
        ColMajor = 102,
    }
    #[repr(C)]
    #[allow(non_camel_case_types, dead_code)]
    #[derive(Clone, Copy)]
    pub enum CBLAS_TRANSPOSE {
        NoTrans = 111,
        Trans = 112,
        ConjTrans = 113,
    }

    #[link(name = "openblas")]
    unsafe extern "C" {
        pub fn cblas_sgemv(
            order: CBLAS_ORDER,
            trans: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            incx: c_int,
            beta: c_float,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_dgemv(
            order: CBLAS_ORDER,
            trans: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            incx: c_int,
            beta: c_double,
            y: *mut c_double,
            incy: c_int,
        );
        pub fn cblas_cgemv(
            order: CBLAS_ORDER,
            trans: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            alpha: *const c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            incx: c_int,
            beta: *const c_float,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_zgemv(
            order: CBLAS_ORDER,
            trans: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            alpha: *const c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            incx: c_int,
            beta: *const c_double,
            y: *mut c_double,
            incy: c_int,
        );
    }
}

// ---------------------------------------------------------------------
// cuBLAS gemv harness (feature = "cuda"). The CUDA-side setup (allocate
// buffers + upload A,x,y) runs once outside the timed region so the
// reported figures reflect kernel + synchronize cost, not h2d transfer.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Metal gemv harness (feature = "metal"). The MetalBackend's gemv
// routes f32/Complex32 through native MSL kernels and keeps f64/Complex64
// on the CPU-over-shared-memory policy, so these bars compare the actual
// production dispatch path with the data already resident on the backend.
// ---------------------------------------------------------------------

#[cfg(feature = "metal")]
fn run_metal_gemv<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: Transpose,
    m: usize,
    n: usize,
    a_rm: &[T],
    x_host: &[T],
    y_host: &[T],
    alpha: T,
    beta: T,
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, MetalBackend};
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable, skipping gemv bench: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let mut a_dev = pool.try_alloc::<T>(a_rm.len()).unwrap();
    let mut x_dev = pool.try_alloc::<T>(x_host.len()).unwrap();
    let mut y_dev = pool.try_alloc::<T>(y_host.len()).unwrap();
    pool.backend().copy_h2d(a_rm, a_dev.raw_mut());
    pool.backend().copy_h2d(x_host, x_dev.raw_mut());
    pool.backend().copy_h2d(y_host, y_dev.raw_mut());

    group.bench_with_input(
        BenchmarkId::new("metal", format!("{m}x{n}")),
        &(m, n),
        |b, _| {
            b.iter(|| {
                pool.backend().gemv(
                    op,
                    m,
                    n,
                    black_box(alpha),
                    a_dev.raw(),
                    n,
                    x_dev.raw(),
                    black_box(beta),
                    y_dev.raw_mut(),
                );
                pool.backend()
                    .synchronize()
                    .expect("backend synchronization failed");
            });
        },
    );
}

#[cfg(feature = "metal")]
fn run_metal_mps_gemv_f32(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    label: &str,
    op: Transpose,
    m: usize,
    n: usize,
    a_rm: &[f32],
    x_host: &[f32],
    y_host: &[f32],
    alpha: f32,
    beta: f32,
) {
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, MetalBackend};
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable, skipping MPS gemv bench: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let mut a_dev = pool.try_alloc::<f32>(a_rm.len()).unwrap();
    let mut x_dev = pool.try_alloc::<f32>(x_host.len()).unwrap();
    let mut y_dev = pool.try_alloc::<f32>(y_host.len()).unwrap();
    pool.backend().copy_h2d(a_rm, a_dev.raw_mut());
    pool.backend().copy_h2d(x_host, x_dev.raw_mut());
    pool.backend().copy_h2d(y_host, y_dev.raw_mut());

    group.bench_with_input(BenchmarkId::new(bench_name, label), &(m, n), |b, _| {
        b.iter(|| {
            pool.backend().gemv_f32_mps_experimental(
                op,
                m,
                n,
                black_box(alpha),
                a_dev.raw(),
                n,
                x_dev.raw(),
                black_box(beta),
                y_dev.raw_mut(),
            );
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
        });
    });
}

#[cfg(feature = "cuda")]
fn run_cuda_gemv<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: Transpose,
    m: usize,
    n: usize,
    a_rm: &[T],
    x_host: &[T],
    y_host: &[T],
    alpha: T,
    beta: T,
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping gemv bench: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let mut a_dev = pool.try_alloc::<T>(a_rm.len()).unwrap();
    let mut x_dev = pool.try_alloc::<T>(x_host.len()).unwrap();
    let mut y_dev = pool.try_alloc::<T>(y_host.len()).unwrap();
    pool.backend().copy_h2d(a_rm, a_dev.raw_mut());
    pool.backend().copy_h2d(x_host, x_dev.raw_mut());
    pool.backend().copy_h2d(y_host, y_dev.raw_mut());

    // Stored row stride: NoTrans stores A as m×n, Trans/ConjTrans as n×m.
    let lda = match op {
        Transpose::NoTrans => n,
        Transpose::Trans | Transpose::ConjTrans => m,
    };
    group.bench_with_input(
        BenchmarkId::new("cublas", format!("{m}x{n}")),
        &(m, n),
        |b, _| {
            b.iter(|| {
                pool.backend().gemv(
                    op,
                    m,
                    n,
                    black_box(alpha),
                    a_dev.raw(),
                    lda,
                    x_dev.raw(),
                    black_box(beta),
                    y_dev.raw_mut(),
                );
                pool.backend()
                    .synchronize()
                    .expect("backend synchronization failed");
            });
        },
    );
}

// ---------------------------------------------------------------------
// gemv f32
// ---------------------------------------------------------------------

fn bench_gemv_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_f32");
    let alpha = 0.7_f32;
    let beta = 0.5_f32;
    for &(m, n) in SIZES {
        let a_rm = ramp_f32(m, n);
        let a_cm = to_col_major(&a_rm, m, n);
        let x = ramp_f32(n, 1);
        let y0 = ramp_f32(m, 1);
        let label = format!("{m}x{n}");
        set_gemv_model::<f32>(&mut group, "f32", &label, m, n);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| {
                sonium_blas2::gemv_f32(
                    Transpose::NoTrans,
                    m,
                    n,
                    black_box(alpha),
                    black_box(&a_rm),
                    n,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", &label), &(m, n), |bb, _| {
            use oxiblas_blas::level2::{GemvTrans, gemv};
            use oxiblas_matrix::MatRef;
            let mut y = y0.clone();
            // SAFETY: `a_cm` owns exactly `m * n` initialized values in
            // column-major order and outlives the benchmark closure.
            let mat = unsafe { MatRef::<f32>::new(a_cm.as_ptr(), m, n, m) };
            bb.iter(|| {
                gemv::<f32>(
                    GemvTrans::NoTrans,
                    black_box(alpha),
                    mat,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| unsafe {
                accelerate::cblas_sgemv(
                    accelerate::CBLAS_ORDER::ColMajor,
                    accelerate::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    black_box(alpha),
                    a_cm.as_ptr(),
                    m as i32,
                    x.as_ptr(),
                    1,
                    black_box(beta),
                    y.as_mut_ptr(),
                    1,
                );
            });
        });

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| unsafe {
                openblas::cblas_sgemv(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    black_box(alpha),
                    a_cm.as_ptr(),
                    m as i32,
                    x.as_ptr(),
                    1,
                    black_box(beta),
                    y.as_mut_ptr(),
                    1,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemv::<f32>(
            &mut group,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
        #[cfg(feature = "metal")]
        run_metal_mps_gemv_f32(
            &mut group,
            "metal_mps",
            &label,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_gemv::<f32>(
            &mut group,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// gemv f32 transpose mode
// ---------------------------------------------------------------------

fn bench_gemv_f32_trans(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_f32_trans");
    let alpha = 0.7_f32;
    let beta = 0.5_f32;
    for &(m, n) in SIZES {
        // Sonium's Trans contract stores A physically as n rows by m cols,
        // then computes y_m := A_stored^T * x_n. The column-major copy is
        // also the row-major prepacked A_stored^T used by the MPS best-case.
        let a_stored = ramp_f32(n, m);
        let a_stored_cm = to_col_major(&a_stored, n, m);
        let x = ramp_f32(n, 1);
        let y0 = ramp_f32(m, 1);
        let label = format!("{m}x{n}");
        set_gemv_model::<f32>(&mut group, "f32", &label, m, n);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| {
                sonium_blas2::gemv_f32(
                    Transpose::Trans,
                    m,
                    n,
                    black_box(alpha),
                    black_box(&a_stored),
                    m,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", &label), &(m, n), |bb, _| {
            use oxiblas_blas::level2::{GemvTrans, gemv};
            use oxiblas_matrix::MatRef;
            let mut y = y0.clone();
            // SAFETY: `a_stored_cm` owns exactly `n * m` initialized values in
            // column-major order and outlives the benchmark closure.
            let mat = unsafe { MatRef::<f32>::new(a_stored_cm.as_ptr(), n, m, n) };
            bb.iter(|| {
                gemv::<f32>(
                    GemvTrans::Trans,
                    black_box(alpha),
                    mat,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| unsafe {
                accelerate::cblas_sgemv(
                    accelerate::CBLAS_ORDER::ColMajor,
                    accelerate::CBLAS_TRANSPOSE::Trans,
                    n as i32,
                    m as i32,
                    black_box(alpha),
                    a_stored_cm.as_ptr(),
                    n as i32,
                    x.as_ptr(),
                    1,
                    black_box(beta),
                    y.as_mut_ptr(),
                    1,
                );
            });
        });

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| unsafe {
                openblas::cblas_sgemv(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::Trans,
                    n as i32,
                    m as i32,
                    black_box(alpha),
                    a_stored_cm.as_ptr(),
                    n as i32,
                    x.as_ptr(),
                    1,
                    black_box(beta),
                    y.as_mut_ptr(),
                    1,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemv::<f32>(
            &mut group,
            Transpose::Trans,
            m,
            n,
            &a_stored,
            &x,
            &y0,
            alpha,
            beta,
        );
        #[cfg(feature = "metal")]
        run_metal_mps_gemv_f32(
            &mut group,
            "metal_mps_packed_trans",
            &label,
            Transpose::NoTrans,
            m,
            n,
            &a_stored_cm,
            &x,
            &y0,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_gemv::<f32>(
            &mut group,
            Transpose::Trans,
            m,
            n,
            &a_stored,
            &x,
            &y0,
            alpha,
            beta,
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// gemv f64
// ---------------------------------------------------------------------

fn bench_gemv_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_f64");
    let alpha = 0.7_f64;
    let beta = 0.5_f64;
    for &(m, n) in SIZES {
        let a_rm = ramp_f64(m, n);
        let a_cm = to_col_major(&a_rm, m, n);
        let x = ramp_f64(n, 1);
        let y0 = ramp_f64(m, 1);
        let label = format!("{m}x{n}");
        set_gemv_model::<f64>(&mut group, "f64", &label, m, n);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| {
                sonium_blas2::gemv_f64(
                    Transpose::NoTrans,
                    m,
                    n,
                    black_box(alpha),
                    black_box(&a_rm),
                    n,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", &label), &(m, n), |bb, _| {
            use oxiblas_blas::level2::{GemvTrans, gemv};
            use oxiblas_matrix::MatRef;
            let mut y = y0.clone();
            // SAFETY: `a_cm` owns exactly `m * n` initialized values in
            // column-major order and outlives the benchmark closure.
            let mat = unsafe { MatRef::<f64>::new(a_cm.as_ptr(), m, n, m) };
            bb.iter(|| {
                gemv::<f64>(
                    GemvTrans::NoTrans,
                    black_box(alpha),
                    mat,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| unsafe {
                accelerate::cblas_dgemv(
                    accelerate::CBLAS_ORDER::ColMajor,
                    accelerate::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    black_box(alpha),
                    a_cm.as_ptr(),
                    m as i32,
                    x.as_ptr(),
                    1,
                    black_box(beta),
                    y.as_mut_ptr(),
                    1,
                );
            });
        });

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| unsafe {
                openblas::cblas_dgemv(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    black_box(alpha),
                    a_cm.as_ptr(),
                    m as i32,
                    x.as_ptr(),
                    1,
                    black_box(beta),
                    y.as_mut_ptr(),
                    1,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemv::<f64>(
            &mut group,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_gemv::<f64>(
            &mut group,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// gemv c32
// ---------------------------------------------------------------------

fn bench_gemv_c32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_c32");
    let alpha = Complex32::new(0.7, -0.3);
    let beta = Complex32::new(0.5, 0.1);
    for &(m, n) in SIZES {
        let a_rm = ramp_c32(m, n);
        let a_cm = to_col_major(&a_rm, m, n);
        let x = ramp_c32(n, 1);
        let y0 = ramp_c32(m, 1);
        let label = format!("{m}x{n}");
        set_gemv_model::<Complex32>(&mut group, "c32", &label, m, n);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| {
                sonium_blas2::gemv_c32(
                    Transpose::NoTrans,
                    m,
                    n,
                    black_box(alpha),
                    black_box(&a_rm),
                    n,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", &label), &(m, n), |bb, _| {
            use oxiblas_blas::level2::{GemvTrans, gemv};
            use oxiblas_matrix::MatRef;
            let mut y = y0.clone();
            // SAFETY: `a_cm` owns exactly `m * n` initialized values in
            // column-major order and outlives the benchmark closure.
            let mat = unsafe { MatRef::<Complex32>::new(a_cm.as_ptr(), m, n, m) };
            bb.iter(|| {
                gemv::<Complex32>(
                    GemvTrans::NoTrans,
                    black_box(alpha),
                    mat,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            let beta_buf = [beta.re, beta.im];
            bb.iter(|| unsafe {
                accelerate::cblas_cgemv(
                    accelerate::CBLAS_ORDER::ColMajor,
                    accelerate::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    alpha_buf.as_ptr(),
                    a_cm.as_ptr() as *const f32,
                    m as i32,
                    x.as_ptr() as *const f32,
                    1,
                    beta_buf.as_ptr(),
                    y.as_mut_ptr() as *mut f32,
                    1,
                );
            });
        });

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            let beta_buf = [beta.re, beta.im];
            bb.iter(|| unsafe {
                openblas::cblas_cgemv(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    alpha_buf.as_ptr(),
                    a_cm.as_ptr() as *const f32,
                    m as i32,
                    x.as_ptr() as *const f32,
                    1,
                    beta_buf.as_ptr(),
                    y.as_mut_ptr() as *mut f32,
                    1,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemv::<Complex32>(
            &mut group,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_gemv::<Complex32>(
            &mut group,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// gemv c64
// ---------------------------------------------------------------------

fn bench_gemv_c64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_c64");
    let alpha = Complex64::new(0.7, -0.3);
    let beta = Complex64::new(0.5, 0.1);
    for &(m, n) in SIZES {
        let a_rm = ramp_c64(m, n);
        let a_cm = to_col_major(&a_rm, m, n);
        let x = ramp_c64(n, 1);
        let y0 = ramp_c64(m, 1);
        let label = format!("{m}x{n}");
        set_gemv_model::<Complex64>(&mut group, "c64", &label, m, n);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            bb.iter(|| {
                sonium_blas2::gemv_c64(
                    Transpose::NoTrans,
                    m,
                    n,
                    black_box(alpha),
                    black_box(&a_rm),
                    n,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", &label), &(m, n), |bb, _| {
            use oxiblas_blas::level2::{GemvTrans, gemv};
            use oxiblas_matrix::MatRef;
            let mut y = y0.clone();
            // SAFETY: `a_cm` owns exactly `m * n` initialized values in
            // column-major order and outlives the benchmark closure.
            let mat = unsafe { MatRef::<Complex64>::new(a_cm.as_ptr(), m, n, m) };
            bb.iter(|| {
                gemv::<Complex64>(
                    GemvTrans::NoTrans,
                    black_box(alpha),
                    mat,
                    black_box(&x),
                    black_box(beta),
                    &mut y,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            let beta_buf = [beta.re, beta.im];
            bb.iter(|| unsafe {
                accelerate::cblas_zgemv(
                    accelerate::CBLAS_ORDER::ColMajor,
                    accelerate::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    alpha_buf.as_ptr(),
                    a_cm.as_ptr() as *const f64,
                    m as i32,
                    x.as_ptr() as *const f64,
                    1,
                    beta_buf.as_ptr(),
                    y.as_mut_ptr() as *mut f64,
                    1,
                );
            });
        });

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n), |bb, _| {
            let mut y = y0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            let beta_buf = [beta.re, beta.im];
            bb.iter(|| unsafe {
                openblas::cblas_zgemv(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    alpha_buf.as_ptr(),
                    a_cm.as_ptr() as *const f64,
                    m as i32,
                    x.as_ptr() as *const f64,
                    1,
                    beta_buf.as_ptr(),
                    y.as_mut_ptr() as *mut f64,
                    1,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemv::<Complex64>(
            &mut group,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_gemv::<Complex64>(
            &mut group,
            Transpose::NoTrans,
            m,
            n,
            &a_rm,
            &x,
            &y0,
            alpha,
            beta,
        );
    }
    group.finish();
}

criterion_group!(
    blas2_compare,
    validate_blas2_correctness,
    bench_gemv_f32,
    bench_gemv_f32_trans,
    bench_gemv_f64,
    bench_gemv_c32,
    bench_gemv_c64,
);
criterion_main!(blas2_compare);
