//! BLAS-3 (`gemm`) comparison across implementations.
//!
//! Computes `C := α·op(A)·op(B) + β·C` with `op = NoTrans` across:
//!
//! - **sonium** — `sonium_compute::cpu::blas3::gemm_*` (row-major,
//!   forwards real `(NoTrans, NoTrans)` to the SME2 micro-kernel).
//! - **accelerate** — Apple Accelerate `cblas_?gemm` (macOS).
//! - **openblas** — Linux `libopenblas.so` cblas surface (feature
//!   `bench-openblas`).
//! - **cublas** — `sonium_compute::CudaBackend::gemm` with the
//!   workload pre-staged on the device (feature `cuda`).
//!
//! Sizes span small/medium/large square gemm plus a tall-skinny case
//! representative of BEM right-hand-side assembly.
//!
//! ```sh
//! cargo bench -p sonium-compute --bench blas3_compare
//! cargo bench -p sonium-compute --bench blas3_compare -- gemm_f64/256x256x256
//! cargo bench -p sonium-compute --bench blas3_compare --features bench-openblas
//! cargo bench -p sonium-compute --bench blas3_compare --features cuda
//! ```

#![allow(clippy::too_many_arguments)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::{Complex32, Complex64};

use sonium_compute::cpu::blas3 as sonium_blas3;
use sonium_compute::scalar::DeviceScalar;
use sonium_compute::{Transpose, perf};

mod perf_report;

/// `(m, n, k)` shapes.
///
///   64×64×64       — fits in L1; dispatch overhead matters
///   256×256×256    — L2-bound
///   1024×1024×1024 — full compute regime (≈2 GF)
///   2048×128×2048  — tall-skinny BEM-like
const SIZES: &[(usize, usize, usize)] = &[
    (64, 64, 64),
    (256, 256, 256),
    (1024, 1024, 1024),
    (2048, 128, 2048),
];

/// `(rows, cols)` shapes for FlashLib-style dense products:
/// `gram_gemm` / `cov_gemm` compute X^T X.
const DENSE_PRODUCT_SIZES: &[(usize, usize)] = &[(4096, 64), (16384, 64), (4096, 256)];

fn ramp_f32(n: usize) -> Vec<f32> {
    (0..n).map(|k| ((k as f32) * 0.0123).sin()).collect()
}
fn ramp_f64(n: usize) -> Vec<f64> {
    (0..n).map(|k| ((k as f64) * 0.0123).sin()).collect()
}
fn ramp_c32(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|k| Complex32::new((k as f32 * 0.0123).sin(), (k as f32 * 0.0177).cos()))
        .collect()
}
fn ramp_c64(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|k| Complex64::new((k as f64 * 0.0123).sin(), (k as f64 * 0.0177).cos()))
        .collect()
}

/// Reinterpret a row-major `m × n` slice as the corresponding column-major
/// buffer `cm[i + j*m] = rm[i*n + j]`.
#[cfg(any(
    target_os = "macos",
    all(feature = "bench-openblas", target_os = "linux")
))]
fn to_col_major<T: Copy + Default>(rm: &[T], m: usize, n: usize) -> Vec<T> {
    let mut cm = vec![T::default(); m * n];
    for i in 0..m {
        for j in 0..n {
            cm[i + j * m] = rm[i * n + j];
        }
    }
    cm
}

/// Canonical FLOP count: `2 * m * n * k` for real gemm.
fn flops_gemm(m: usize, n: usize, k: usize) -> u64 {
    (2 * m * n * k) as u64
}

fn bytes_gemm(m: usize, n: usize, k: usize, elem_bytes: usize) -> u64 {
    ((m * k + k * n + 2 * m * n) * elem_bytes) as u64
}

fn set_gemm_model(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scalar: &str,
    label: &str,
    model: perf_report::PerfModel,
) {
    perf_report::set_flop_throughput(group, "gemm", scalar, label, model);
}

fn set_device_gemm_model<T: DeviceScalar>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scalar: &str,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
) {
    set_gemm_model(group, scalar, label, perf::gemm_model::<T>(m, n, k));
}

fn set_dense_gemm_model(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench: &str,
    scalar: &str,
    label: &str,
    variant: &str,
    model: perf::PerfModel,
) {
    perf_report::set_flop_throughput_with_variant(
        group, bench, scalar, label, variant, "exact", model,
    );
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn reference_gemm_f32(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c0: &[f32],
    ldc: usize,
) -> Vec<f32> {
    let mut c = c0.to_vec();
    sonium_blas3::gemm_f32(
        transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &mut c, ldc,
    );
    c
}

#[cfg(feature = "cuda")]
fn max_abs_rel_f32(got: &[f32], want: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let abs = (g - w).abs();
        let rel = abs / w.abs().max(1.0e-30);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

#[cfg(feature = "cuda")]
fn report_precision_check(
    bench: &str,
    scalar: &str,
    shape: &str,
    variant: &str,
    precision: &str,
    got: &[f32],
    want: &[f32],
) {
    let (max_abs, max_rel) = max_abs_rel_f32(got, want);
    eprintln!(
        "sonium_precision_check bench={bench} scalar={scalar} shape={shape} variant={variant} precision={precision} max_abs={max_abs:.9e} max_rel={max_rel:.9e} metadata=\"{}\"",
        perf_report::host_metadata()
    );
}

// ---------------------------------------------------------------------
// Accelerate FFI (macOS).
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
        pub fn cblas_sgemm(
            order: CBLAS_ORDER,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        pub fn cblas_dgemm(
            order: CBLAS_ORDER,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
        pub fn cblas_cgemm(
            order: CBLAS_ORDER,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: *const c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        pub fn cblas_zgemm(
            order: CBLAS_ORDER,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: *const c_double,
            c: *mut c_double,
            ldc: c_int,
        );
    }
}

// ---------------------------------------------------------------------
// OpenBLAS FFI (Linux, opt-in).
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
        pub fn cblas_sgemm(
            order: CBLAS_ORDER,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        pub fn cblas_dgemm(
            order: CBLAS_ORDER,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
        pub fn cblas_cgemm(
            order: CBLAS_ORDER,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: *const c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        pub fn cblas_zgemm(
            order: CBLAS_ORDER,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: *const c_double,
            c: *mut c_double,
            ldc: c_int,
        );
    }
}

// ---------------------------------------------------------------------
// cuBLAS gemm — pre-stages A,B,C on the device so the bench measures
// kernel + synchronize, not h2d transfers.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Metal gemm harness (feature = "metal"). Reads the unified-memory
// shared buffer via host_slice() and runs the CPU blas3 kernel — the
// bar therefore measures CPU work plus per-call `flush_pending` and
// MetalBackend dispatch overhead, and is the slot where native MSL
// `?gemm` kernels would land.
// ---------------------------------------------------------------------

#[cfg(feature = "metal")]
fn run_metal_gemm<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    m: usize,
    n: usize,
    k: usize,
    a_rm: &[T],
    b_rm: &[T],
    c_rm0: &[T],
    alpha: T,
    beta: T,
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, MetalBackend};
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable, skipping gemm bench: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let mut a_dev = pool.try_alloc::<T>(a_rm.len()).unwrap();
    let mut b_dev = pool.try_alloc::<T>(b_rm.len()).unwrap();
    let mut c_dev = pool.try_alloc::<T>(c_rm0.len()).unwrap();
    pool.backend().copy_h2d(a_rm, a_dev.raw_mut());
    pool.backend().copy_h2d(b_rm, b_dev.raw_mut());
    pool.backend().copy_h2d(c_rm0, c_dev.raw_mut());

    group.bench_with_input(
        BenchmarkId::new("metal", format!("{m}x{n}x{k}")),
        &(m, n, k),
        |bb, _| {
            bb.iter(|| {
                pool.backend().gemm(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    m,
                    n,
                    k,
                    black_box(alpha),
                    a_dev.raw(),
                    k,
                    b_dev.raw(),
                    n,
                    black_box(beta),
                    c_dev.raw_mut(),
                    n,
                );
                pool.backend()
                    .synchronize()
                    .expect("backend synchronization failed");
            });
        },
    );
}

#[cfg(feature = "metal")]
fn run_metal_exact_gemm<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    variant: &str,
    label: &str,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_rm: &[T],
    lda: usize,
    b_rm: &[T],
    ldb: usize,
    c_rm0: &[T],
    ldc: usize,
    alpha: T,
    beta: T,
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, MetalBackend};
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable, skipping {variant} bench: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let mut a_dev = pool.try_alloc::<T>(a_rm.len()).unwrap();
    let mut b_dev = pool.try_alloc::<T>(b_rm.len()).unwrap();
    let mut c_dev = pool.try_alloc::<T>(c_rm0.len()).unwrap();
    pool.backend().copy_h2d(a_rm, a_dev.raw_mut());
    pool.backend().copy_h2d(b_rm, b_dev.raw_mut());
    pool.backend().copy_h2d(c_rm0, c_dev.raw_mut());

    group.bench_with_input(BenchmarkId::new(variant, label), &(m, n, k), |bb, _| {
        bb.iter(|| {
            pool.backend().gemm(
                transa,
                transb,
                m,
                n,
                k,
                black_box(alpha),
                a_dev.raw(),
                lda,
                b_dev.raw(),
                ldb,
                black_box(beta),
                c_dev.raw_mut(),
                ldc,
            );
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
        });
    });
}

#[cfg(feature = "cuda")]
fn run_cuda_gemm<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    m: usize,
    n: usize,
    k: usize,
    a_rm: &[T],
    b_rm: &[T],
    c_rm0: &[T],
    alpha: T,
    beta: T,
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping gemm bench: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let mut a_dev = pool.try_alloc::<T>(a_rm.len()).unwrap();
    let mut b_dev = pool.try_alloc::<T>(b_rm.len()).unwrap();
    let mut c_dev = pool.try_alloc::<T>(c_rm0.len()).unwrap();
    pool.backend().copy_h2d(a_rm, a_dev.raw_mut());
    pool.backend().copy_h2d(b_rm, b_dev.raw_mut());
    pool.backend().copy_h2d(c_rm0, c_dev.raw_mut());

    group.bench_with_input(
        BenchmarkId::new("cublas", format!("{m}x{n}x{k}")),
        &(m, n, k),
        |b, _| {
            b.iter(|| {
                pool.backend().gemm(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    m,
                    n,
                    k,
                    black_box(alpha),
                    a_dev.raw(),
                    k,
                    b_dev.raw(),
                    n,
                    black_box(beta),
                    c_dev.raw_mut(),
                    n,
                );
                pool.backend()
                    .synchronize()
                    .expect("backend synchronization failed");
            });
        },
    );
}

#[cfg(feature = "cuda")]
fn run_cuda_exact_gemm<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    variant: &str,
    label: &str,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_rm: &[T],
    lda: usize,
    b_rm: &[T],
    ldb: usize,
    c_rm0: &[T],
    ldc: usize,
    alpha: T,
    beta: T,
) where
    T: sonium_compute::DeviceScalar,
{
    use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping {variant} bench: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let mut a_dev = pool.try_alloc::<T>(a_rm.len()).unwrap();
    let mut b_dev = pool.try_alloc::<T>(b_rm.len()).unwrap();
    let mut c_dev = pool.try_alloc::<T>(c_rm0.len()).unwrap();
    pool.backend().copy_h2d(a_rm, a_dev.raw_mut());
    pool.backend().copy_h2d(b_rm, b_dev.raw_mut());
    pool.backend().copy_h2d(c_rm0, c_dev.raw_mut());

    group.bench_with_input(BenchmarkId::new(variant, label), &(m, n, k), |bb, _| {
        bb.iter(|| {
            pool.backend().gemm(
                transa,
                transb,
                m,
                n,
                k,
                black_box(alpha),
                a_dev.raw(),
                lda,
                b_dev.raw(),
                ldb,
                black_box(beta),
                c_dev.raw_mut(),
                ldc,
            );
            pool.backend()
                .synchronize()
                .expect("backend synchronization failed");
        });
    });
}

#[cfg(feature = "cuda")]
fn cuda_gemm_op(t: Transpose) -> cudarc::cublas::sys::cublasOperation_t {
    use cudarc::cublas::sys::cublasOperation_t;
    match t {
        Transpose::NoTrans => cublasOperation_t::CUBLAS_OP_N,
        Transpose::Trans => cublasOperation_t::CUBLAS_OP_T,
        Transpose::ConjTrans => cublasOperation_t::CUBLAS_OP_C,
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_cuda_gemm_ex_f32(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench: &str,
    variant: &str,
    precision: &str,
    label: &str,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_rm: &[f32],
    lda: usize,
    b_rm: &[f32],
    ldb: usize,
    c_rm0: &[f32],
    ldc: usize,
    alpha: f32,
    beta: f32,
    compute_type: cudarc::cublas::sys::cublasComputeType_t,
    algo: cudarc::cublas::sys::cublasGemmAlgo_t,
) {
    use cudarc::cublas::sys;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping {variant} bench: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let mut a_dev = pool.try_alloc::<f32>(a_rm.len()).unwrap();
    let mut b_dev = pool.try_alloc::<f32>(b_rm.len()).unwrap();
    let mut c_dev = pool.try_alloc::<f32>(c_rm0.len()).unwrap();
    pool.backend().copy_h2d(a_rm, a_dev.raw_mut());
    pool.backend().copy_h2d(b_rm, b_dev.raw_mut());
    pool.backend().copy_h2d(c_rm0, c_dev.raw_mut());

    let blas = pool.backend().blas_handle();
    let stream = pool.backend().stream().clone();
    let m_i = i32::try_from(m).expect("cuda gemm_ex: m exceeds i32");
    let n_i = i32::try_from(n).expect("cuda gemm_ex: n exceeds i32");
    let k_i = i32::try_from(k).expect("cuda gemm_ex: k exceeds i32");
    let lda_i = i32::try_from(lda).expect("cuda gemm_ex: lda exceeds i32");
    let ldb_i = i32::try_from(ldb).expect("cuda gemm_ex: ldb exceeds i32");
    let ldc_i = i32::try_from(ldc).expect("cuda gemm_ex: ldc exceeds i32");
    let benchmark_variant = format!("{variant}_{bench}");

    {
        {
            let (ap, _ag) = a_dev
                .raw()
                .raw()
                .expect("cuda gemm_ex A")
                .device_ptr(&stream);
            let (bp, _bg) = b_dev
                .raw()
                .raw()
                .expect("cuda gemm_ex B")
                .device_ptr(&stream);
            let (cp, _cg) = c_dev
                .raw_mut()
                .raw_mut()
                .expect("cuda gemm_ex C")
                .device_ptr_mut(&stream);
            let status = unsafe {
                sys::cublasGemmEx(
                    blas,
                    cuda_gemm_op(transb),
                    cuda_gemm_op(transa),
                    n_i,
                    m_i,
                    k_i,
                    (&alpha as *const f32).cast(),
                    bp as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    ldb_i,
                    ap as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    lda_i,
                    (&beta as *const f32).cast(),
                    cp as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    ldc_i,
                    compute_type,
                    algo,
                )
            };
            assert_eq!(
                status,
                sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                "cublasGemmEx {variant} precision preflight failed with {status:?}"
            );
        }
        pool.backend()
            .synchronize()
            .expect("backend synchronization failed");
        let mut got = vec![0.0_f32; c_rm0.len()];
        pool.backend().copy_d2h(c_dev.raw(), &mut got);
        let want = reference_gemm_f32(
            transa, transb, m, n, k, alpha, a_rm, lda, b_rm, ldb, beta, c_rm0, ldc,
        );
        report_precision_check(bench, "f32", label, variant, precision, &got, &want);
        pool.backend().copy_h2d(c_rm0, c_dev.raw_mut());
    }

    group.bench_with_input(
        BenchmarkId::new(benchmark_variant, label),
        &(m, n, k),
        |bb, _| {
            bb.iter(|| {
                let (ap, _ag) = a_dev
                    .raw()
                    .raw()
                    .expect("cuda gemm_ex A")
                    .device_ptr(&stream);
                let (bp, _bg) = b_dev
                    .raw()
                    .raw()
                    .expect("cuda gemm_ex B")
                    .device_ptr(&stream);
                let (cp, _cg) = c_dev
                    .raw_mut()
                    .raw_mut()
                    .expect("cuda gemm_ex C")
                    .device_ptr_mut(&stream);

                // Same row-major swap trick as CudaBackend::gemm:
                // C^T = B^T * A^T in cuBLAS column-major terms.
                let status = unsafe {
                    sys::cublasGemmEx(
                        blas,
                        cuda_gemm_op(transb),
                        cuda_gemm_op(transa),
                        n_i,
                        m_i,
                        k_i,
                        (&alpha as *const f32).cast(),
                        bp as *const _,
                        sys::cudaDataType_t::CUDA_R_32F,
                        ldb_i,
                        ap as *const _,
                        sys::cudaDataType_t::CUDA_R_32F,
                        lda_i,
                        (&beta as *const f32).cast(),
                        cp as *mut _,
                        sys::cudaDataType_t::CUDA_R_32F,
                        ldc_i,
                        compute_type,
                        algo,
                    )
                };
                assert_eq!(
                    status,
                    sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                    "cublasGemmEx {variant} failed with {status:?}"
                );
                pool.backend()
                    .synchronize()
                    .expect("backend synchronization failed");
            });
        },
    );
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_cuda_precision_candidates_f32(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench: &str,
    label: &str,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_rm: &[f32],
    lda: usize,
    b_rm: &[f32],
    ldb: usize,
    c_rm0: &[f32],
    ldc: usize,
    alpha: f32,
    beta: f32,
) {
    use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t};

    let model = perf::gemm_model::<f32>(m, n, k);
    perf_report::report_model_with_variant(
        bench,
        "f32",
        label,
        "cublas_gemmex_fp32_pedantic",
        "fp32_pedantic",
        model,
    );
    perf_report::report_model_with_variant(
        bench,
        "f32",
        label,
        "cublas_gemmex_tf32",
        "tf32",
        model,
    );

    run_cuda_gemm_ex_f32(
        group,
        bench,
        "cublas_gemmex_fp32_pedantic",
        "fp32_pedantic",
        label,
        transa,
        transb,
        m,
        n,
        k,
        a_rm,
        lda,
        b_rm,
        ldb,
        c_rm0,
        ldc,
        alpha,
        beta,
        cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
    );
    run_cuda_gemm_ex_f32(
        group,
        bench,
        "cublas_gemmex_tf32",
        "tf32",
        label,
        transa,
        transb,
        m,
        n,
        k,
        a_rm,
        lda,
        b_rm,
        ldb,
        c_rm0,
        ldc,
        alpha,
        beta,
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
}

// ---------------------------------------------------------------------
// gemm f32
// ---------------------------------------------------------------------

fn bench_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f32");
    let alpha = 0.7_f32;
    let beta = 0.0_f32;
    for &(m, n, k) in SIZES {
        let a_rm = ramp_f32(m * k);
        let b_rm = ramp_f32(k * n);
        let c_rm0 = ramp_f32(m * n);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let a_cm = to_col_major(&a_rm, m, k);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let b_cm = to_col_major(&b_rm, k, n);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let c_cm0 = to_col_major(&c_rm0, m, n);
        let label = format!("{m}x{n}x{k}");
        set_device_gemm_model::<f32>(&mut group, "f32", &label, m, n, k);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n, k), |bb, _| {
            let mut cm = c_rm0.clone();
            bb.iter(|| {
                sonium_blas3::gemm_f32(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    m,
                    n,
                    k,
                    black_box(alpha),
                    black_box(&a_rm),
                    k,
                    black_box(&b_rm),
                    n,
                    black_box(beta),
                    &mut cm,
                    n,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(
            BenchmarkId::new("accelerate", &label),
            &(m, n, k),
            |bb, _| {
                let mut cm = c_cm0.clone();
                bb.iter(|| unsafe {
                    accelerate::cblas_sgemm(
                        accelerate::CBLAS_ORDER::ColMajor,
                        accelerate::CBLAS_TRANSPOSE::NoTrans,
                        accelerate::CBLAS_TRANSPOSE::NoTrans,
                        m as i32,
                        n as i32,
                        k as i32,
                        black_box(alpha),
                        a_cm.as_ptr(),
                        m as i32,
                        b_cm.as_ptr(),
                        k as i32,
                        black_box(beta),
                        cm.as_mut_ptr(),
                        m as i32,
                    );
                });
            },
        );

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n, k), |bb, _| {
            let mut cm = c_cm0.clone();
            bb.iter(|| unsafe {
                openblas::cblas_sgemm(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    k as i32,
                    black_box(alpha),
                    a_cm.as_ptr(),
                    m as i32,
                    b_cm.as_ptr(),
                    k as i32,
                    black_box(beta),
                    cm.as_mut_ptr(),
                    m as i32,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemm::<f32>(&mut group, m, n, k, &a_rm, &b_rm, &c_rm0, alpha, beta);
        #[cfg(feature = "cuda")]
        run_cuda_gemm::<f32>(&mut group, m, n, k, &a_rm, &b_rm, &c_rm0, alpha, beta);
        #[cfg(feature = "cuda")]
        run_cuda_precision_candidates_f32(
            &mut group,
            "gemm",
            &label,
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            &a_rm,
            k,
            &b_rm,
            n,
            &c_rm0,
            n,
            alpha,
            beta,
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// gemm f64
// ---------------------------------------------------------------------

fn bench_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f64");
    let alpha = 0.7_f64;
    let beta = 0.0_f64;
    for &(m, n, k) in SIZES {
        let a_rm = ramp_f64(m * k);
        let b_rm = ramp_f64(k * n);
        let c_rm0 = ramp_f64(m * n);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let a_cm = to_col_major(&a_rm, m, k);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let b_cm = to_col_major(&b_rm, k, n);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let c_cm0 = to_col_major(&c_rm0, m, n);
        let label = format!("{m}x{n}x{k}");
        set_device_gemm_model::<f64>(&mut group, "f64", &label, m, n, k);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n, k), |bb, _| {
            let mut cm = c_rm0.clone();
            bb.iter(|| {
                sonium_blas3::gemm_f64(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    m,
                    n,
                    k,
                    black_box(alpha),
                    black_box(&a_rm),
                    k,
                    black_box(&b_rm),
                    n,
                    black_box(beta),
                    &mut cm,
                    n,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(
            BenchmarkId::new("accelerate", &label),
            &(m, n, k),
            |bb, _| {
                let mut cm = c_cm0.clone();
                bb.iter(|| unsafe {
                    accelerate::cblas_dgemm(
                        accelerate::CBLAS_ORDER::ColMajor,
                        accelerate::CBLAS_TRANSPOSE::NoTrans,
                        accelerate::CBLAS_TRANSPOSE::NoTrans,
                        m as i32,
                        n as i32,
                        k as i32,
                        black_box(alpha),
                        a_cm.as_ptr(),
                        m as i32,
                        b_cm.as_ptr(),
                        k as i32,
                        black_box(beta),
                        cm.as_mut_ptr(),
                        m as i32,
                    );
                });
            },
        );

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n, k), |bb, _| {
            let mut cm = c_cm0.clone();
            bb.iter(|| unsafe {
                openblas::cblas_dgemm(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    k as i32,
                    black_box(alpha),
                    a_cm.as_ptr(),
                    m as i32,
                    b_cm.as_ptr(),
                    k as i32,
                    black_box(beta),
                    cm.as_mut_ptr(),
                    m as i32,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemm::<f64>(&mut group, m, n, k, &a_rm, &b_rm, &c_rm0, alpha, beta);
        #[cfg(feature = "cuda")]
        run_cuda_gemm::<f64>(&mut group, m, n, k, &a_rm, &b_rm, &c_rm0, alpha, beta);
    }
    group.finish();
}

// ---------------------------------------------------------------------
// gemm c32
// ---------------------------------------------------------------------

fn bench_gemm_c32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_c32");
    let alpha = Complex32::new(0.7, -0.3);
    let beta = Complex32::new(0.0, 0.0);
    for &(m, n, k) in SIZES {
        let a_rm = ramp_c32(m * k);
        let b_rm = ramp_c32(k * n);
        let c_rm0 = ramp_c32(m * n);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let a_cm = to_col_major(&a_rm, m, k);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let b_cm = to_col_major(&b_rm, k, n);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let c_cm0 = to_col_major(&c_rm0, m, n);
        let label = format!("{m}x{n}x{k}");
        set_device_gemm_model::<Complex32>(&mut group, "c32", &label, m, n, k);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n, k), |bb, _| {
            let mut cm = c_rm0.clone();
            bb.iter(|| {
                sonium_blas3::gemm_c32(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    m,
                    n,
                    k,
                    black_box(alpha),
                    black_box(&a_rm),
                    k,
                    black_box(&b_rm),
                    n,
                    black_box(beta),
                    &mut cm,
                    n,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(
            BenchmarkId::new("accelerate", &label),
            &(m, n, k),
            |bb, _| {
                let mut cm = c_cm0.clone();
                let alpha_buf = [alpha.re, alpha.im];
                let beta_buf = [beta.re, beta.im];
                bb.iter(|| unsafe {
                    accelerate::cblas_cgemm(
                        accelerate::CBLAS_ORDER::ColMajor,
                        accelerate::CBLAS_TRANSPOSE::NoTrans,
                        accelerate::CBLAS_TRANSPOSE::NoTrans,
                        m as i32,
                        n as i32,
                        k as i32,
                        alpha_buf.as_ptr(),
                        a_cm.as_ptr() as *const f32,
                        m as i32,
                        b_cm.as_ptr() as *const f32,
                        k as i32,
                        beta_buf.as_ptr(),
                        cm.as_mut_ptr() as *mut f32,
                        m as i32,
                    );
                });
            },
        );

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n, k), |bb, _| {
            let mut cm = c_cm0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            let beta_buf = [beta.re, beta.im];
            bb.iter(|| unsafe {
                openblas::cblas_cgemm(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha_buf.as_ptr(),
                    a_cm.as_ptr() as *const f32,
                    m as i32,
                    b_cm.as_ptr() as *const f32,
                    k as i32,
                    beta_buf.as_ptr(),
                    cm.as_mut_ptr() as *mut f32,
                    m as i32,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemm::<Complex32>(&mut group, m, n, k, &a_rm, &b_rm, &c_rm0, alpha, beta);
        #[cfg(feature = "cuda")]
        run_cuda_gemm::<Complex32>(&mut group, m, n, k, &a_rm, &b_rm, &c_rm0, alpha, beta);
    }
    group.finish();
}

// ---------------------------------------------------------------------
// gemm c64
// ---------------------------------------------------------------------

fn bench_gemm_c64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_c64");
    let alpha = Complex64::new(0.7, -0.3);
    let beta = Complex64::new(0.0, 0.0);
    for &(m, n, k) in SIZES {
        let a_rm = ramp_c64(m * k);
        let b_rm = ramp_c64(k * n);
        let c_rm0 = ramp_c64(m * n);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let a_cm = to_col_major(&a_rm, m, k);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let b_cm = to_col_major(&b_rm, k, n);
        #[cfg(any(
            target_os = "macos",
            all(feature = "bench-openblas", target_os = "linux")
        ))]
        let c_cm0 = to_col_major(&c_rm0, m, n);
        let label = format!("{m}x{n}x{k}");
        set_device_gemm_model::<Complex64>(&mut group, "c64", &label, m, n, k);

        group.bench_with_input(BenchmarkId::new("sonium", &label), &(m, n, k), |bb, _| {
            let mut cm = c_rm0.clone();
            bb.iter(|| {
                sonium_blas3::gemm_c64(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    m,
                    n,
                    k,
                    black_box(alpha),
                    black_box(&a_rm),
                    k,
                    black_box(&b_rm),
                    n,
                    black_box(beta),
                    &mut cm,
                    n,
                );
            });
        });

        #[cfg(target_os = "macos")]
        group.bench_with_input(
            BenchmarkId::new("accelerate", &label),
            &(m, n, k),
            |bb, _| {
                let mut cm = c_cm0.clone();
                let alpha_buf = [alpha.re, alpha.im];
                let beta_buf = [beta.re, beta.im];
                bb.iter(|| unsafe {
                    accelerate::cblas_zgemm(
                        accelerate::CBLAS_ORDER::ColMajor,
                        accelerate::CBLAS_TRANSPOSE::NoTrans,
                        accelerate::CBLAS_TRANSPOSE::NoTrans,
                        m as i32,
                        n as i32,
                        k as i32,
                        alpha_buf.as_ptr(),
                        a_cm.as_ptr() as *const f64,
                        m as i32,
                        b_cm.as_ptr() as *const f64,
                        k as i32,
                        beta_buf.as_ptr(),
                        cm.as_mut_ptr() as *mut f64,
                        m as i32,
                    );
                });
            },
        );

        #[cfg(all(feature = "bench-openblas", target_os = "linux"))]
        group.bench_with_input(BenchmarkId::new("openblas", &label), &(m, n, k), |bb, _| {
            let mut cm = c_cm0.clone();
            let alpha_buf = [alpha.re, alpha.im];
            let beta_buf = [beta.re, beta.im];
            bb.iter(|| unsafe {
                openblas::cblas_zgemm(
                    openblas::CBLAS_ORDER::ColMajor,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    openblas::CBLAS_TRANSPOSE::NoTrans,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha_buf.as_ptr(),
                    a_cm.as_ptr() as *const f64,
                    m as i32,
                    b_cm.as_ptr() as *const f64,
                    k as i32,
                    beta_buf.as_ptr(),
                    cm.as_mut_ptr() as *mut f64,
                    m as i32,
                );
            });
        });

        #[cfg(feature = "metal")]
        run_metal_gemm::<Complex64>(&mut group, m, n, k, &a_rm, &b_rm, &c_rm0, alpha, beta);
        #[cfg(feature = "cuda")]
        run_cuda_gemm::<Complex64>(&mut group, m, n, k, &a_rm, &b_rm, &c_rm0, alpha, beta);
    }
    group.finish();
}

// ---------------------------------------------------------------------
// FlashLib-style dense products: ab_gemm, gram_gemm, cov_gemm
// ---------------------------------------------------------------------

fn bench_dense_products_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_products_f32");
    let alpha = 1.0_f32;
    let beta = 0.0_f32;

    for &(m, n, k) in SIZES {
        let a_rm = ramp_f32(m * k);
        let b_rm = ramp_f32(k * n);
        let c_rm0 = vec![0.0_f32; m * n];
        let label = format!("{m}x{n}x{k}");
        set_dense_gemm_model(
            &mut group,
            "ab_gemm",
            "f32",
            &label,
            "exact",
            perf::ab_gemm_model::<f32>(m, n, k),
        );

        group.bench_with_input(
            BenchmarkId::new("sonium_ab_gemm", &label),
            &(m, n, k),
            |bb, _| {
                let mut out = c_rm0.clone();
                bb.iter(|| {
                    sonium_blas3::gemm_f32(
                        Transpose::NoTrans,
                        Transpose::NoTrans,
                        m,
                        n,
                        k,
                        black_box(alpha),
                        black_box(&a_rm),
                        k,
                        black_box(&b_rm),
                        n,
                        black_box(beta),
                        &mut out,
                        n,
                    );
                });
            },
        );

        #[cfg(feature = "metal")]
        run_metal_exact_gemm::<f32>(
            &mut group,
            "metal_ab_gemm",
            &label,
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            &a_rm,
            k,
            &b_rm,
            n,
            &c_rm0,
            n,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_exact_gemm::<f32>(
            &mut group,
            "cublas_ab_gemm",
            &label,
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            &a_rm,
            k,
            &b_rm,
            n,
            &c_rm0,
            n,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_precision_candidates_f32(
            &mut group,
            "ab_gemm",
            &label,
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            &a_rm,
            k,
            &b_rm,
            n,
            &c_rm0,
            n,
            alpha,
            beta,
        );
    }

    for &(rows, cols) in DENSE_PRODUCT_SIZES {
        let x = ramp_f32(rows * cols);
        let out0 = vec![0.0_f32; cols * cols];
        let label = format!("{rows}x{cols}");
        set_dense_gemm_model(
            &mut group,
            "gram_gemm",
            "f32",
            &label,
            "exact",
            perf::gram_gemm_model::<f32>(rows, cols),
        );

        group.bench_with_input(
            BenchmarkId::new("sonium_gram_gemm", &label),
            &(rows, cols),
            |bb, _| {
                let mut gram = out0.clone();
                bb.iter(|| {
                    sonium_blas3::gemm_f32(
                        Transpose::Trans,
                        Transpose::NoTrans,
                        cols,
                        cols,
                        rows,
                        black_box(alpha),
                        black_box(&x),
                        cols,
                        black_box(&x),
                        cols,
                        black_box(beta),
                        &mut gram,
                        cols,
                    );
                });
            },
        );

        #[cfg(feature = "metal")]
        run_metal_exact_gemm::<f32>(
            &mut group,
            "metal_gram_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_exact_gemm::<f32>(
            &mut group,
            "cublas_gram_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_precision_candidates_f32(
            &mut group,
            "gram_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );

        set_dense_gemm_model(
            &mut group,
            "cov_gemm",
            "f32",
            &label,
            "exact",
            perf::cov_gemm_model::<f32>(rows, cols),
        );
        group.bench_with_input(
            BenchmarkId::new("sonium_cov_gemm", &label),
            &(rows, cols),
            |bb, _| {
                let mut cov = out0.clone();
                bb.iter(|| {
                    sonium_blas3::gemm_f32(
                        Transpose::Trans,
                        Transpose::NoTrans,
                        cols,
                        cols,
                        rows,
                        black_box(alpha),
                        black_box(&x),
                        cols,
                        black_box(&x),
                        cols,
                        black_box(beta),
                        &mut cov,
                        cols,
                    );
                });
            },
        );

        #[cfg(feature = "metal")]
        run_metal_exact_gemm::<f32>(
            &mut group,
            "metal_cov_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_exact_gemm::<f32>(
            &mut group,
            "cublas_cov_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_precision_candidates_f32(
            &mut group,
            "cov_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );
    }
    group.finish();
}

fn bench_dense_products_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_products_f64");
    let alpha = 1.0_f64;
    let beta = 0.0_f64;

    for &(m, n, k) in SIZES {
        let a_rm = ramp_f64(m * k);
        let b_rm = ramp_f64(k * n);
        let c_rm0 = vec![0.0_f64; m * n];
        let label = format!("{m}x{n}x{k}");
        set_dense_gemm_model(
            &mut group,
            "ab_gemm",
            "f64",
            &label,
            "exact",
            perf::ab_gemm_model::<f64>(m, n, k),
        );

        group.bench_with_input(
            BenchmarkId::new("sonium_ab_gemm", &label),
            &(m, n, k),
            |bb, _| {
                let mut out = c_rm0.clone();
                bb.iter(|| {
                    sonium_blas3::gemm_f64(
                        Transpose::NoTrans,
                        Transpose::NoTrans,
                        m,
                        n,
                        k,
                        black_box(alpha),
                        black_box(&a_rm),
                        k,
                        black_box(&b_rm),
                        n,
                        black_box(beta),
                        &mut out,
                        n,
                    );
                });
            },
        );

        #[cfg(feature = "metal")]
        run_metal_exact_gemm::<f64>(
            &mut group,
            "metal_ab_gemm",
            &label,
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            &a_rm,
            k,
            &b_rm,
            n,
            &c_rm0,
            n,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_exact_gemm::<f64>(
            &mut group,
            "cublas_ab_gemm",
            &label,
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            &a_rm,
            k,
            &b_rm,
            n,
            &c_rm0,
            n,
            alpha,
            beta,
        );
    }

    for &(rows, cols) in DENSE_PRODUCT_SIZES {
        let x = ramp_f64(rows * cols);
        let out0 = vec![0.0_f64; cols * cols];
        let label = format!("{rows}x{cols}");
        set_dense_gemm_model(
            &mut group,
            "gram_gemm",
            "f64",
            &label,
            "exact",
            perf::gram_gemm_model::<f64>(rows, cols),
        );

        group.bench_with_input(
            BenchmarkId::new("sonium_gram_gemm", &label),
            &(rows, cols),
            |bb, _| {
                let mut gram = out0.clone();
                bb.iter(|| {
                    sonium_blas3::gemm_f64(
                        Transpose::Trans,
                        Transpose::NoTrans,
                        cols,
                        cols,
                        rows,
                        black_box(alpha),
                        black_box(&x),
                        cols,
                        black_box(&x),
                        cols,
                        black_box(beta),
                        &mut gram,
                        cols,
                    );
                });
            },
        );

        #[cfg(feature = "metal")]
        run_metal_exact_gemm::<f64>(
            &mut group,
            "metal_gram_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_exact_gemm::<f64>(
            &mut group,
            "cublas_gram_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );

        set_dense_gemm_model(
            &mut group,
            "cov_gemm",
            "f64",
            &label,
            "exact",
            perf::cov_gemm_model::<f64>(rows, cols),
        );
        group.bench_with_input(
            BenchmarkId::new("sonium_cov_gemm", &label),
            &(rows, cols),
            |bb, _| {
                let mut cov = out0.clone();
                bb.iter(|| {
                    sonium_blas3::gemm_f64(
                        Transpose::Trans,
                        Transpose::NoTrans,
                        cols,
                        cols,
                        rows,
                        black_box(alpha),
                        black_box(&x),
                        cols,
                        black_box(&x),
                        cols,
                        black_box(beta),
                        &mut cov,
                        cols,
                    );
                });
            },
        );

        #[cfg(feature = "metal")]
        run_metal_exact_gemm::<f64>(
            &mut group,
            "metal_cov_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );
        #[cfg(feature = "cuda")]
        run_cuda_exact_gemm::<f64>(
            &mut group,
            "cublas_cov_gemm",
            &label,
            Transpose::Trans,
            Transpose::NoTrans,
            cols,
            cols,
            rows,
            &x,
            cols,
            &x,
            cols,
            &out0,
            cols,
            alpha,
            beta,
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// bf16 → f32 GEMM
// ---------------------------------------------------------------------
//
// Compares two paths at the same shape:
//   • sonium_bf16  — `sme2::gemm_bf16::gemm_bf16` (widens A,B to f32 once
//                    into a TLS buffer, then routes through sgemm).
//   • sonium_f32   — `sgemm` directly on pre-widened f32 buffers. Tells
//                    us how much overhead the per-call bf16→f32 conversion
//                    costs versus the FMOPA work.
// We don't compare to Accelerate here — Accelerate doesn't expose a
// public bf16 GEMM with f32 accumulation.

fn bench_gemm_bf16(c: &mut Criterion) {
    use half::bf16;
    use sonium_compute::sme2::gemm as sme_gemm;
    use sonium_compute::sme2::gemm_bf16;

    let mut group = c.benchmark_group("gemm_bf16");
    let alpha = 0.7_f32;
    let beta = 0.0_f32;
    for &(m, n, k) in SIZES {
        let a_f32 = ramp_f32(m * k);
        let b_f32 = ramp_f32(k * n);
        let a_bf: Vec<bf16> = a_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let b_bf: Vec<bf16> = b_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let c0 = vec![0.0f32; m * n];
        let label = format!("{m}x{n}x{k}");
        set_gemm_model(
            &mut group,
            "bf16",
            &label,
            perf_report::PerfModel::new(bytes_gemm(m, n, k, 2), flops_gemm(m, n, k)),
        );

        group.bench_with_input(
            BenchmarkId::new("sonium_bf16", &label),
            &(m, n, k),
            |bb, _| {
                let mut cm = c0.clone();
                bb.iter(|| {
                    gemm_bf16::gemm_bf16(
                        m,
                        n,
                        k,
                        black_box(alpha),
                        black_box(&a_bf),
                        k,
                        black_box(&b_bf),
                        n,
                        black_box(beta),
                        &mut cm,
                        n,
                    );
                });
            },
        );

        // Baseline: same shape, but inputs are already f32 — isolates the
        // bf16-widening cost in the bench above.
        group.bench_with_input(
            BenchmarkId::new("sonium_f32_baseline", &label),
            &(m, n, k),
            |bb, _| {
                let mut cm = c0.clone();
                bb.iter(|| {
                    sme_gemm::sgemm(
                        m,
                        n,
                        k,
                        black_box(alpha),
                        black_box(&a_f32),
                        k,
                        black_box(&b_f32),
                        n,
                        black_box(beta),
                        &mut cm,
                        n,
                        false,
                    );
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = blas3_compare;
    config = Criterion::default().sample_size(20);
    targets = bench_gemm_f32, bench_gemm_f64, bench_gemm_c32, bench_gemm_c64,
        bench_dense_products_f32, bench_dense_products_f64, bench_gemm_bf16,
}
criterion_main!(blas3_compare);
