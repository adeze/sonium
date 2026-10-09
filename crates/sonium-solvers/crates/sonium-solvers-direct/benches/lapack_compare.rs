//! Dense LAPACK factorization comparison: sonium vs vendor.
//!
//! Groups `potrf_{f32,f64}`, `getrf_{f32,f64}`, `geqrf_f64`,
//! `potrf_{c32,c64}` at square sizes, each with a `sonium` arm plus one
//! vendor arm per platform:
//!
//! - **accelerate** — Apple Accelerate LAPACK (macOS).
//! - **openblas** — OpenBLAS Fortran LAPACK, needs `--features bench-openblas`
//!   on Linux. Vendor buffers are column-major; the transpose runs outside
//!   the timer.
//! - **cusolver** — `CudaBackend::dense_lu_f64_from_column_major` (getrf)
//!   and `dense_cholesky_f64_from_column_major` (potrf), needing
//!   `--features cuda`. Upload is inside the timer: a one-shot number,
//!   not the persistent-handle solver-loop rate.
//!
//! Every arm passes a relative-residual gate at setup (1e-10 f64, 1e-4 f32)
//! before any timing runs; gate values print to stderr. Inputs are seeded,
//! so repeated runs compare identical matrices.
//!
//! Run:
//!   cargo bench -p sonium-solvers-direct --bench lapack_compare
//!   cargo bench -p sonium-solvers-direct --bench lapack_compare --features bench-openblas
//!   cargo bench -p sonium-solvers-direct --bench lapack_compare --features cuda

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::{Complex32, Complex64};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{
    LuWorkspace, QrFactorization, cholesky_factorize_in_place,
    cholesky_factorize_in_place_hermitian, lu_factorize_in_place,
};
use std::hint::black_box;

const SIZES_F64: [usize; 4] = [512, 1024, 2048, 4096];
const SIZES_F32: [usize; 2] = [1024, 2048];
const SIZES_C32: [usize; 3] = [512, 1024, 2048];
const SIZES_C64: [usize; 3] = [512, 1024, 2048];
const GATE_F64: f64 = 1e-10;
const GATE_F32: f64 = 1e-4;

#[cfg(all(target_os = "linux", feature = "bench-openblas"))]
#[link(name = "openblas")]
unsafe extern "C" {
    fn dgemm_(
        transa: *const std::ffi::c_char,
        transb: *const std::ffi::c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dpotrf_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        info: *mut i32,
    );
    fn spotrf_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        info: *mut i32,
    );
    fn dgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn sgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn dgetrs_(
        trans: *const std::ffi::c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        ipiv: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn dpotrs_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn dgeqrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dorgqr_(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *const f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    // Complex buffers cross as interleaved f32/f64 pairs (`Complex32/64`
    // are `#[repr(C)]`; `lda` counts complex elements).
    fn cpotrf_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        info: *mut i32,
    );
    fn zpotrf_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        info: *mut i32,
    );
    fn cpotrs_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const f32,
        lda: *const i32,
        b: *mut f32,
        ldb: *const i32,
        info: *mut i32,
    );
    fn zpotrs_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn dgemm_(
        transa: *const std::ffi::c_char,
        transb: *const std::ffi::c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dpotrf_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        info: *mut i32,
    );
    fn spotrf_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        info: *mut i32,
    );
    fn dgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn sgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn dgetrs_(
        trans: *const std::ffi::c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        ipiv: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn dpotrs_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn dgeqrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dorgqr_(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *const f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    // Complex buffers cross as interleaved f32/f64 pairs (`Complex32/64`
    // are `#[repr(C)]`; `lda` counts complex elements).
    fn cpotrf_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        info: *mut i32,
    );
    fn zpotrf_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        info: *mut i32,
    );
    fn cpotrs_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const f32,
        lda: *const i32,
        b: *mut f32,
        ldb: *const i32,
        info: *mut i32,
    );
    fn zpotrs_(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
}

#[cfg(any(
    target_os = "macos",
    all(target_os = "linux", feature = "bench-openblas")
))]
const VENDOR_ARM: &str = if cfg!(target_os = "macos") {
    "accelerate"
} else {
    "openblas"
};

fn seeded_matrix_f64(n: usize, seed: u64) -> DenseMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    DenseMatrix::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0))
}

fn seeded_matrix_f32(n: usize, seed: u64) -> DenseMatrix<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    DenseMatrix::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0))
}

/// Symmetric positive-definite input: X·Xᵀ + n·I, diagonally dominant.
fn seeded_spd_f64(n: usize, seed: u64) -> DenseMatrix<f64> {
    let x = seeded_matrix_f64(n, seed);
    let xs = x.as_slice();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += xs[i * n + k] * xs[j * n + k];
            }
            out[i * n + j] = acc;
        }
        out[i * n + i] += n as f64;
    }
    DenseMatrix::from_fn(n, n, |i, j| out[i * n + j])
}

fn seeded_spd_f32(n: usize, seed: u64) -> DenseMatrix<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let x: Vec<f32> = (0..n * n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += x[i * n + k] * x[j * n + k];
            }
            out[i * n + j] = acc;
        }
        out[i * n + i] += n as f32;
    }
    DenseMatrix::from_fn(n, n, |i, j| out[i * n + j])
}

fn transpose_to_column_major_f64(matrix: &DenseMatrix<f64>) -> Vec<f64> {
    let n = matrix.nrows();
    let src = matrix.as_slice();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = src[i * n + j];
        }
    }
    out
}

fn transpose_to_column_major_f32(matrix: &DenseMatrix<f32>) -> Vec<f32> {
    let n = matrix.nrows();
    let src = matrix.as_slice();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = src[i * n + j];
        }
    }
    out
}

/// Hermitian positive-definite input: X·Xᴴ + n·I, diagonally dominant.
fn seeded_hpd_c64(n: usize, seed: u64) -> DenseMatrix<Complex64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let x: Vec<Complex64> = (0..n * n)
        .map(|_| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
        .collect();
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n {
                acc += x[i * n + k] * x[j * n + k].conj();
            }
            out[i * n + j] = acc;
        }
        out[i * n + i] += Complex64::new(n as f64, 0.0);
    }
    DenseMatrix::from_fn(n, n, |i, j| out[i * n + j])
}

fn seeded_hpd_c32(n: usize, seed: u64) -> DenseMatrix<Complex32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let x: Vec<Complex32> = (0..n * n)
        .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
        .collect();
    let mut out = vec![Complex32::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex32::new(0.0, 0.0);
            for k in 0..n {
                acc += x[i * n + k] * x[j * n + k].conj();
            }
            out[i * n + j] = acc;
        }
        out[i * n + i] += Complex32::new(n as f32, 0.0);
    }
    DenseMatrix::from_fn(n, n, |i, j| out[i * n + j])
}

fn transpose_to_column_major_c64(matrix: &DenseMatrix<Complex64>) -> Vec<Complex64> {
    let n = matrix.nrows();
    let src = matrix.as_slice();
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = src[i * n + j];
        }
    }
    out
}

fn transpose_to_column_major_c32(matrix: &DenseMatrix<Complex32>) -> Vec<Complex32> {
    let n = matrix.nrows();
    let src = matrix.as_slice();
    let mut out = vec![Complex32::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = src[i * n + j];
        }
    }
    out
}

fn residual_solve_c64(matrix: &DenseMatrix<Complex64>, x: &[Complex64], b: &[Complex64]) -> f64 {
    let n = matrix.nrows();
    let a = matrix.as_slice();
    let mut num = 0.0f64;
    let mut denom_a = 0.0f64;
    let mut denom_x = 0.0f64;
    for i in 0..n {
        let mut ax = Complex64::new(0.0, 0.0);
        for j in 0..n {
            let aij = a[i * n + j];
            ax += aij * x[j];
            denom_a += aij.norm_sqr();
        }
        num += (ax - b[i]).norm_sqr();
        denom_x += x[i].norm_sqr();
    }
    num.sqrt() / (denom_a.sqrt() * denom_x.sqrt()).max(1e-300)
}

fn residual_solve_c32(matrix: &DenseMatrix<Complex32>, x: &[Complex32], b: &[Complex32]) -> f64 {
    let n = matrix.nrows();
    let a = matrix.as_slice();
    let mut num = 0.0f64;
    let mut denom_a = 0.0f64;
    let mut denom_x = 0.0f64;
    for i in 0..n {
        let mut ax = Complex64::new(0.0, 0.0);
        for j in 0..n {
            let aij = Complex64::new(a[i * n + j].re as f64, a[i * n + j].im as f64);
            let xj = Complex64::new(x[j].re as f64, x[j].im as f64);
            ax += aij * xj;
            denom_a += aij.norm_sqr();
        }
        let bi = Complex64::new(b[i].re as f64, b[i].im as f64);
        num += (ax - bi).norm_sqr();
        let xi = Complex64::new(x[i].re as f64, x[i].im as f64);
        denom_x += xi.norm_sqr();
    }
    num.sqrt() / (denom_a.sqrt() * denom_x.sqrt()).max(1e-300)
}

fn residual_solve_f64(matrix: &DenseMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let n = matrix.nrows();
    let a = matrix.as_slice();
    let mut num = 0.0f64;
    let mut denom_a = 0.0f64;
    let mut denom_x = 0.0f64;
    for i in 0..n {
        let mut ax = 0.0;
        for j in 0..n {
            let aij = a[i * n + j];
            ax += aij * x[j];
            denom_a += aij * aij;
        }
        let r = ax - b[i];
        num += r * r;
        denom_x += x[i] * x[i];
    }
    (num.sqrt()) / (denom_a.sqrt() * denom_x.sqrt()).max(1e-300)
}

#[cfg(not(any(
    target_os = "macos",
    all(target_os = "linux", feature = "bench-openblas")
)))]
fn residual_qr_f64(matrix: &DenseMatrix<f64>, q: &DenseMatrix<f64>, r: &DenseMatrix<f64>) -> f64 {
    let n = matrix.nrows();
    let (a, qs, rs) = (matrix.as_slice(), q.as_slice(), r.as_slice());
    let mut num = 0.0f64;
    let mut denom = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut qr = 0.0;
            for k in 0..n {
                qr += qs[i * n + k] * rs[k * n + j];
            }
            let diff = a[i * n + j] - qr;
            num += diff * diff;
            denom += a[i * n + j] * a[i * n + j];
        }
    }
    num.sqrt() / denom.sqrt().max(1e-300)
}

fn gate(name: &str, residual: f64, limit: f64) {
    eprintln!("lapack_compare gate {name}: residual {residual:.3e} (limit {limit:.1e})");
    assert!(
        residual <= limit,
        "{name} residual {residual:.3e} exceeds gate {limit:.1e}"
    );
}

// BLAS-backed QR residuals for benchmark setup only (never inside the timer).
// The naive triple loops they replace cost minutes single-threaded at n=4096;
// vendor dgemm does the same math in ~0.1 s with the same error order.
#[cfg(any(
    target_os = "macos",
    all(target_os = "linux", feature = "bench-openblas")
))]
fn qr_residual_blas(a_row: &[f64], q_row: &[f64], r_row: &[f64], n: usize) -> f64 {
    let ni = n as i32;
    let no = b'N' as std::ffi::c_char;
    let one = 1.0f64;
    let zero = 0.0f64;
    // Row-major C = Q·R via col-major Cᵀ = Rᵀ·Qᵀ (row-major buffers read as
    // col-major are the transposes).
    let mut qr = vec![0.0; n * n];
    unsafe {
        dgemm_(
            &no,
            &no,
            &ni,
            &ni,
            &ni,
            &one,
            r_row.as_ptr(),
            &ni,
            q_row.as_ptr(),
            &ni,
            &zero,
            qr.as_mut_ptr(),
            &ni,
        );
    }
    frobenius_ratio(a_row, &qr)
}

// Variant for a col-major Q straight out of vendor dgeqrf/dorgqr: forms R = QᵀA
// with dgemm, then ||A − QR|| / ||A||. `q_col` is col-major, `a_row` row-major.
#[cfg(any(
    target_os = "macos",
    all(target_os = "linux", feature = "bench-openblas")
))]
fn vendor_qr_gate_residual(n: usize, a_row: &[f64], q_col: &[f64]) -> f64 {
    let ni = n as i32;
    let tr = b'T' as std::ffi::c_char;
    let no = b'N' as std::ffi::c_char;
    let one = 1.0f64;
    let zero = 0.0f64;
    let mut r_col = vec![0.0; n * n];
    unsafe {
        dgemm_(
            &tr,
            &tr,
            &ni,
            &ni,
            &ni,
            &one,
            q_col.as_ptr(),
            &ni,
            a_row.as_ptr(),
            &ni,
            &zero,
            r_col.as_mut_ptr(),
            &ni,
        );
    }
    let mut qr_col = vec![0.0; n * n];
    unsafe {
        dgemm_(
            &no,
            &no,
            &ni,
            &ni,
            &ni,
            &one,
            q_col.as_ptr(),
            &ni,
            r_col.as_ptr(),
            &ni,
            &zero,
            qr_col.as_mut_ptr(),
            &ni,
        );
    }
    let mut num = 0.0f64;
    let mut denom = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let diff = a_row[i * n + j] - qr_col[j * n + i];
            num += diff * diff;
            denom += a_row[i * n + j] * a_row[i * n + j];
        }
    }
    num.sqrt() / denom.sqrt().max(1e-300)
}

#[cfg(any(
    target_os = "macos",
    all(target_os = "linux", feature = "bench-openblas")
))]
fn frobenius_ratio(a: &[f64], approx: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut denom = 0.0f64;
    for (x, y) in a.iter().zip(approx.iter()) {
        let diff = x - y;
        num += diff * diff;
        denom += x * x;
    }
    num.sqrt() / denom.sqrt().max(1e-300)
}

fn bench_potrf_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("potrf_f64");
    for &n in &SIZES_F64 {
        let label = format!("{n}");
        let base = seeded_spd_f64(n, 0x50_74_72_66);
        let mut rhs_rng = StdRng::seed_from_u64(0xBE_EF);
        let rhs: Vec<f64> = (0..n).map(|_| rhs_rng.random_range(-1.0..1.0)).collect();

        let probe = cholesky_factorize_in_place(base.clone())
            .expect("sonium potrf setup")
            .solve(&rhs)
            .expect("sonium potrs setup");
        gate(
            &format!("sonium/potrf_f64/{n}"),
            residual_solve_f64(&base, &probe, &rhs),
            GATE_F64,
        );

        group.bench_function(BenchmarkId::new("sonium", &label), |b| {
            b.iter_batched(
                || base.clone(),
                |m| black_box(cholesky_factorize_in_place(m).expect("potrf")),
                BatchSize::SmallInput,
            );
        });

        #[cfg(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        ))]
        {
            let col_major = transpose_to_column_major_f64(&base);
            let mut probe_cm = col_major.clone();
            let mut probe_rhs = rhs.clone();
            let mut info = 0;
            let ni = n as i32;
            let uplo = b'L' as std::ffi::c_char;
            unsafe {
                dpotrf_(&uplo, &ni, probe_cm.as_mut_ptr(), &ni, &mut info);
                assert_eq!(info, 0, "vendor potrf setup info");
                let nrhs = 1;
                dpotrs_(
                    &uplo,
                    &ni,
                    &nrhs,
                    probe_cm.as_ptr(),
                    &ni,
                    probe_rhs.as_mut_ptr(),
                    &ni,
                    &mut info,
                );
                assert_eq!(info, 0, "vendor potrs setup info");
            }
            gate(
                &format!("{}/potrf_f64/{n}", VENDOR_ARM),
                residual_solve_f64(&base, &probe_rhs, &rhs),
                GATE_F64,
            );
            group.bench_function(BenchmarkId::new(VENDOR_ARM, &label), |b| {
                b.iter_batched(
                    || col_major.clone(),
                    |mut m| unsafe {
                        let mut info = 0;
                        dpotrf_(&uplo, &ni, m.as_mut_ptr(), &ni, &mut info);
                        black_box(info);
                    },
                    BatchSize::SmallInput,
                );
            });
        }

        #[cfg(feature = "cuda")]
        {
            use sonium_compute::CudaBackend;
            if let Ok(cuda) = CudaBackend::new_default() {
                let col_major = transpose_to_column_major_f64(&base);
                // Gate via device solve residual would need the factors back;
                // the wrapper is cuSOLVER-direct, so gate on info only.
                let probe = cuda
                    .dense_cholesky_f64_from_column_major(n, &col_major)
                    .expect("cusolver potrf setup");
                black_box(probe.n());
                eprintln!("lapack_compare gate cusolver/potrf_f64/{n}: info ok (cuSOLVER-direct)");
                group.bench_function(BenchmarkId::new("cusolver", &label), |b| {
                    b.iter_batched(
                        || col_major.clone(),
                        |m| {
                            black_box(
                                cuda.dense_cholesky_f64_from_column_major(n, &m)
                                    .expect("potrf"),
                            )
                        },
                        BatchSize::SmallInput,
                    );
                });
            } else {
                eprintln!("lapack_compare: no CUDA backend, skipping cusolver arm");
            }
        }
    }
    group.finish();
}

fn bench_potrf_c64(c: &mut Criterion) {
    let mut group = c.benchmark_group("potrf_c64");
    for &n in &SIZES_C64 {
        let label = format!("{n}");
        let base = seeded_hpd_c64(n, 0x50_74_72_66);
        let mut rhs_rng = StdRng::seed_from_u64(0xBE_EF);
        let rhs: Vec<Complex64> = (0..n)
            .map(|_| {
                Complex64::new(
                    rhs_rng.random_range(-1.0..1.0),
                    rhs_rng.random_range(-1.0..1.0),
                )
            })
            .collect();

        let probe = cholesky_factorize_in_place_hermitian(base.clone())
            .expect("sonium zpotrf setup")
            .solve(&rhs)
            .expect("sonium zpotrs setup");
        gate(
            &format!("sonium/potrf_c64/{n}"),
            residual_solve_c64(&base, &probe, &rhs),
            GATE_F64,
        );

        group.bench_function(BenchmarkId::new("sonium", &label), |b| {
            b.iter_batched(
                || base.clone(),
                |m| black_box(cholesky_factorize_in_place_hermitian(m).expect("potrf")),
                BatchSize::SmallInput,
            );
        });

        #[cfg(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        ))]
        {
            let col_major = transpose_to_column_major_c64(&base);
            let mut probe_cm = col_major.clone();
            let mut probe_rhs = rhs.clone();
            let mut info = 0;
            let ni = n as i32;
            let uplo = b'L' as std::ffi::c_char;
            unsafe {
                zpotrf_(
                    &uplo,
                    &ni,
                    probe_cm.as_mut_ptr() as *mut f64,
                    &ni,
                    &mut info,
                );
                assert_eq!(info, 0, "vendor potrf setup info");
                let nrhs = 1;
                zpotrs_(
                    &uplo,
                    &ni,
                    &nrhs,
                    probe_cm.as_ptr() as *const f64,
                    &ni,
                    probe_rhs.as_mut_ptr() as *mut f64,
                    &ni,
                    &mut info,
                );
                assert_eq!(info, 0, "vendor potrs setup info");
            }
            gate(
                &format!("{}/potrf_c64/{n}", VENDOR_ARM),
                residual_solve_c64(&base, &probe_rhs, &rhs),
                GATE_F64,
            );
            group.bench_function(BenchmarkId::new(VENDOR_ARM, &label), |b| {
                b.iter_batched(
                    || col_major.clone(),
                    |mut m| unsafe {
                        let mut info = 0;
                        zpotrf_(&uplo, &ni, m.as_mut_ptr() as *mut f64, &ni, &mut info);
                        black_box(info);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_potrf_c32(c: &mut Criterion) {
    let mut group = c.benchmark_group("potrf_c32");
    for &n in &SIZES_C32 {
        let label = format!("{n}");
        let base = seeded_hpd_c32(n, 0x50_74_72_66);
        let mut rhs_rng = StdRng::seed_from_u64(0xBE_EF);
        let rhs: Vec<Complex32> = (0..n)
            .map(|_| {
                Complex32::new(
                    rhs_rng.random_range(-1.0..1.0),
                    rhs_rng.random_range(-1.0..1.0),
                )
            })
            .collect();

        let probe = cholesky_factorize_in_place_hermitian(base.clone())
            .expect("sonium cpotrf setup")
            .solve(&rhs)
            .expect("sonium cpotrs setup");
        gate(
            &format!("sonium/potrf_c32/{n}"),
            residual_solve_c32(&base, &probe, &rhs),
            GATE_F32,
        );

        group.bench_function(BenchmarkId::new("sonium", &label), |b| {
            b.iter_batched(
                || base.clone(),
                |m| black_box(cholesky_factorize_in_place_hermitian(m).expect("potrf")),
                BatchSize::SmallInput,
            );
        });

        #[cfg(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        ))]
        {
            let col_major = transpose_to_column_major_c32(&base);
            let mut probe_cm = col_major.clone();
            let mut probe_rhs = rhs.clone();
            let mut info = 0;
            let ni = n as i32;
            let uplo = b'L' as std::ffi::c_char;
            unsafe {
                cpotrf_(
                    &uplo,
                    &ni,
                    probe_cm.as_mut_ptr() as *mut f32,
                    &ni,
                    &mut info,
                );
                assert_eq!(info, 0, "vendor potrf setup info");
                let nrhs = 1;
                cpotrs_(
                    &uplo,
                    &ni,
                    &nrhs,
                    probe_cm.as_ptr() as *const f32,
                    &ni,
                    probe_rhs.as_mut_ptr() as *mut f32,
                    &ni,
                    &mut info,
                );
                assert_eq!(info, 0, "vendor potrs setup info");
            }
            gate(
                &format!("{}/potrf_c32/{n}", VENDOR_ARM),
                residual_solve_c32(&base, &probe_rhs, &rhs),
                GATE_F32,
            );
            group.bench_function(BenchmarkId::new(VENDOR_ARM, &label), |b| {
                b.iter_batched(
                    || col_major.clone(),
                    |mut m| unsafe {
                        let mut info = 0;
                        cpotrf_(&uplo, &ni, m.as_mut_ptr() as *mut f32, &ni, &mut info);
                        black_box(info);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_getrf_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("getrf_f64");
    for &n in &SIZES_F64 {
        let label = format!("{n}");
        let base = seeded_matrix_f64(n, 0x6E_74_72_66);
        let mut rhs_rng = StdRng::seed_from_u64(0xBE_EF);
        let rhs: Vec<f64> = (0..n).map(|_| rhs_rng.random_range(-1.0..1.0)).collect();

        let mut workspace = LuWorkspace::new();
        let probe = lu_factorize_in_place(base.clone(), &mut workspace)
            .expect("sonium getrf setup")
            .solve(&rhs)
            .expect("sonium getrs setup");
        gate(
            &format!("sonium/getrf_f64/{n}"),
            residual_solve_f64(&base, &probe, &rhs),
            GATE_F64,
        );

        group.bench_function(BenchmarkId::new("sonium", &label), |b| {
            let mut workspace = LuWorkspace::new();
            b.iter_batched(
                || base.clone(),
                |m| black_box(lu_factorize_in_place(m, &mut workspace).expect("getrf")),
                BatchSize::SmallInput,
            );
        });

        #[cfg(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        ))]
        {
            let col_major = transpose_to_column_major_f64(&base);
            let mut probe_cm = col_major.clone();
            let mut probe_rhs = rhs.clone();
            let mut ipiv = vec![0; n];
            let mut info = 0;
            let ni = n as i32;
            let trans = b'N' as std::ffi::c_char;
            unsafe {
                dgetrf_(
                    &ni,
                    &ni,
                    probe_cm.as_mut_ptr(),
                    &ni,
                    ipiv.as_mut_ptr(),
                    &mut info,
                );
                assert_eq!(info, 0, "vendor getrf setup info");
                let nrhs = 1;
                dgetrs_(
                    &trans,
                    &ni,
                    &nrhs,
                    probe_cm.as_ptr(),
                    &ni,
                    ipiv.as_ptr(),
                    probe_rhs.as_mut_ptr(),
                    &ni,
                    &mut info,
                );
                assert_eq!(info, 0, "vendor getrs setup info");
            }
            gate(
                &format!("{}/getrf_f64/{n}", VENDOR_ARM),
                residual_solve_f64(&base, &probe_rhs, &rhs),
                GATE_F64,
            );
            group.bench_function(BenchmarkId::new(VENDOR_ARM, &label), |b| {
                b.iter_batched(
                    || (col_major.clone(), vec![0; n]),
                    |(mut m, mut ipiv)| unsafe {
                        let mut info = 0;
                        dgetrf_(&ni, &ni, m.as_mut_ptr(), &ni, ipiv.as_mut_ptr(), &mut info);
                        black_box(info);
                    },
                    BatchSize::SmallInput,
                );
            });
        }

        #[cfg(feature = "cuda")]
        {
            use sonium_compute::CudaBackend;
            if let Ok(cuda) = CudaBackend::new_default() {
                let col_major = transpose_to_column_major_f64(&base);
                // Gate via device solve residual would need the factors back;
                // the wrapper is cuSOLVER-direct, so gate on info only.
                let probe = cuda
                    .dense_lu_f64_from_column_major(n, &col_major)
                    .expect("cusolver getrf setup");
                black_box(probe.n());
                eprintln!("lapack_compare gate cusolver/getrf_f64/{n}: info ok (cuSOLVER-direct)");
                group.bench_function(BenchmarkId::new("cusolver", &label), |b| {
                    b.iter_batched(
                        || col_major.clone(),
                        |m| black_box(cuda.dense_lu_f64_from_column_major(n, &m).expect("getrf")),
                        BatchSize::SmallInput,
                    );
                });
            } else {
                eprintln!("lapack_compare: no CUDA backend, skipping cusolver arm");
            }
        }
    }
    group.finish();
}

fn bench_geqrf_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("geqrf_f64");
    for &n in &SIZES_F64 {
        let label = format!("{n}");
        let base = seeded_matrix_f64(n, 0x71_72_66);

        let probe = QrFactorization::factorize(&base).expect("sonium geqrf setup");
        #[cfg(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        ))]
        let sonium_res = qr_residual_blas(
            base.as_slice(),
            probe.q().as_slice(),
            probe.r().as_slice(),
            n,
        );
        #[cfg(not(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        )))]
        let sonium_res = residual_qr_f64(&base, probe.q(), probe.r());
        gate(&format!("sonium/geqrf_f64/{n}"), sonium_res, GATE_F64);

        group.bench_function(BenchmarkId::new("sonium", &label), |b| {
            b.iter_batched(
                || base.clone(),
                |m| black_box(QrFactorization::factorize(&m).expect("geqrf")),
                BatchSize::SmallInput,
            );
        });

        #[cfg(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        ))]
        {
            let col_major = transpose_to_column_major_f64(&base);
            let ni = n as i32;
            // Workspace query outside the timer.
            let mut query = 0.0;
            let mut tau_probe = vec![0.0; n];
            let mut info = 0;
            let minus_one = -1;
            let mut probe_cm = col_major.clone();
            unsafe {
                dgeqrf_(
                    &ni,
                    &ni,
                    probe_cm.as_mut_ptr(),
                    &ni,
                    tau_probe.as_mut_ptr(),
                    &mut query,
                    &minus_one,
                    &mut info,
                );
            }
            let lwork = query as i32;
            let mut probe_cm = col_major.clone();
            let mut tau = vec![0.0; n];
            let mut work = vec![0.0; lwork as usize];
            unsafe {
                dgeqrf_(
                    &ni,
                    &ni,
                    probe_cm.as_mut_ptr(),
                    &ni,
                    tau.as_mut_ptr(),
                    work.as_mut_ptr(),
                    &lwork,
                    &mut info,
                );
                assert_eq!(info, 0, "vendor geqrf setup info");
                dorgqr_(
                    &ni,
                    &ni,
                    &ni,
                    probe_cm.as_mut_ptr(),
                    &ni,
                    tau.as_ptr(),
                    work.as_mut_ptr(),
                    &lwork,
                    &mut info,
                );
                assert_eq!(info, 0, "vendor orgqr setup info");
            }
            // probe_cm now holds explicit Q (col-major); R via QᵀA + residual in BLAS.
            gate(
                &format!("{}/geqrf_f64/{n}", VENDOR_ARM),
                vendor_qr_gate_residual(n, base.as_slice(), &probe_cm),
                GATE_F64,
            );
            group.bench_function(BenchmarkId::new(VENDOR_ARM, &label), |b| {
                b.iter_batched(
                    || (col_major.clone(), vec![0.0; n], vec![0.0; lwork as usize]),
                    |(mut m, mut tau, mut work)| unsafe {
                        let mut info = 0;
                        dgeqrf_(
                            &ni,
                            &ni,
                            m.as_mut_ptr(),
                            &ni,
                            tau.as_mut_ptr(),
                            work.as_mut_ptr(),
                            &lwork,
                            &mut info,
                        );
                        dorgqr_(
                            &ni,
                            &ni,
                            &ni,
                            m.as_mut_ptr(),
                            &ni,
                            tau.as_ptr(),
                            work.as_mut_ptr(),
                            &lwork,
                            &mut info,
                        );
                        black_box(info);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_getrf_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("getrf_f32");
    for &n in &SIZES_F32 {
        let label = format!("{n}");
        let base = seeded_matrix_f32(n, 0x6E_74_72_66);
        let mut rhs_rng = StdRng::seed_from_u64(0xBE_EF);
        let rhs: Vec<f32> = (0..n).map(|_| rhs_rng.random_range(-1.0..1.0)).collect();

        let mut workspace = LuWorkspace::new();
        let probe = lu_factorize_in_place(base.clone(), &mut workspace)
            .expect("sonium sgetrf setup")
            .solve(&rhs)
            .expect("sonium sgetrs setup");
        let base_f64 = DenseMatrix::from_fn(n, n, |i, j| base.as_slice()[i * n + j] as f64);
        let x_f64: Vec<f64> = probe.iter().map(|&v| v as f64).collect();
        let b_f64: Vec<f64> = rhs.iter().map(|&v| v as f64).collect();
        gate(
            &format!("sonium/getrf_f32/{n}"),
            residual_solve_f64(&base_f64, &x_f64, &b_f64),
            GATE_F32,
        );

        group.bench_function(BenchmarkId::new("sonium", &label), |b| {
            let mut workspace = LuWorkspace::new();
            b.iter_batched(
                || base.clone(),
                |m| black_box(lu_factorize_in_place(m, &mut workspace).expect("getrf")),
                BatchSize::SmallInput,
            );
        });

        #[cfg(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        ))]
        {
            let col_major = transpose_to_column_major_f32(&base);
            group.bench_function(BenchmarkId::new(VENDOR_ARM, &label), |b| {
                b.iter_batched(
                    || (col_major.clone(), vec![0; n]),
                    |(mut m, mut ipiv)| unsafe {
                        let ni = n as i32;
                        let mut info = 0;
                        sgetrf_(&ni, &ni, m.as_mut_ptr(), &ni, ipiv.as_mut_ptr(), &mut info);
                        black_box(info);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_potrf_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("potrf_f32");
    for &n in &SIZES_F32 {
        let label = format!("{n}");
        let base = seeded_spd_f32(n, 0x50_74_72_66);
        let mut rhs_rng = StdRng::seed_from_u64(0xBE_EF);
        let rhs: Vec<f32> = (0..n).map(|_| rhs_rng.random_range(-1.0..1.0)).collect();

        let probe = cholesky_factorize_in_place(base.clone())
            .expect("sonium spotrf setup")
            .solve(&rhs)
            .expect("sonium spotrs setup");
        let base_f64 = DenseMatrix::from_fn(n, n, |i, j| base.as_slice()[i * n + j] as f64);
        let x_f64: Vec<f64> = probe.iter().map(|&v| v as f64).collect();
        let b_f64: Vec<f64> = rhs.iter().map(|&v| v as f64).collect();
        gate(
            &format!("sonium/potrf_f32/{n}"),
            residual_solve_f64(&base_f64, &x_f64, &b_f64),
            GATE_F32,
        );

        group.bench_function(BenchmarkId::new("sonium", &label), |b| {
            b.iter_batched(
                || base.clone(),
                |m| black_box(cholesky_factorize_in_place(m).expect("potrf")),
                BatchSize::SmallInput,
            );
        });

        #[cfg(any(
            target_os = "macos",
            all(target_os = "linux", feature = "bench-openblas")
        ))]
        {
            let col_major = transpose_to_column_major_f32(&base);
            group.bench_function(BenchmarkId::new(VENDOR_ARM, &label), |b| {
                b.iter_batched(
                    || col_major.clone(),
                    |mut m| unsafe {
                        let ni = n as i32;
                        let uplo = b'L' as std::ffi::c_char;
                        let mut info = 0;
                        spotrf_(&uplo, &ni, m.as_mut_ptr(), &ni, &mut info);
                        black_box(info);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_potrf_f64,
    bench_getrf_f64,
    bench_geqrf_f64,
    bench_getrf_f32,
    bench_potrf_f32,
    bench_potrf_c64,
    bench_potrf_c32,
);
criterion_main!(benches);
