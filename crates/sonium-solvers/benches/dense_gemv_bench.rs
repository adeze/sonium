//! Dense GEMV benchmark (`DenseMatrix::matvec`).
//!
//! Two groups:
//!
//! - `dense_gemv`: the blas2-dispatched matvec (SME2 via
//!   `sonium_compute::cpu::blas2` for f32/f64 panels of at least 128×128
//!   elements; `blas2::complex_native` NEON kernels for Complex64 panels of
//!   at least 64×64) against the scalar row-loop baseline it replaced.
//! - `blas2_complex_gemv`: kernel-level comparison of the blas2 scalar
//!   complex reference against the native interleaved kernel
//!   (`blas2::complex_native`) for `NoTrans` and `ConjTrans`. Measured on
//!   Apple M4 Pro: the native kernel wins at every size from 64² up —
//!   Complex64 1.7×/2.1× (NoTrans/ConjTrans) at 64², 1.2×/1.4× at 4096²;
//!   Complex32 ~2-3× across the range. The older `complex4m` path, which
//!   re-deinterleaves the matrix per call, measured slower than scalar
//!   (17.4 ms vs 10.2 ms at 4096²) and is not used by `mat_vec_op`.
//!
//! Run with:
//! `cargo bench -p sonium-solvers --bench dense_gemv_bench`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::{Complex32, Complex64};
use sonium_compute::backend::Transpose;
use sonium_compute::cpu::blas2::{complex_native, scalar as blas2_scalar};
use sonium_solvers::dense::DenseMatrix;
use std::hint::black_box;

fn value(k: usize) -> f64 {
    ((k.wrapping_mul(6364136223846793005) >> 11) % 2000) as f64 / 1000.0 - 1.0
}

fn matrix_f64(n: usize) -> DenseMatrix<f64> {
    DenseMatrix::from_fn(n, n, |i, j| value(i * n + j))
}

fn matrix_c64(n: usize) -> DenseMatrix<Complex64> {
    DenseMatrix::from_fn(n, n, |i, j| {
        Complex64::new(value(i * n + j), value(n * n + i * n + j))
    })
}

/// Direct kernel-level comparison: the blas2 scalar reference (what
/// `mat_vec_op_scalar` does for complex types) against the native
/// interleaved NEON/AVX2 kernel in `blas2::complex_native`, for the two ops
/// the solvers actually use (`NoTrans` = matvec, `ConjTrans` = hermitian).
fn bench_blas2_complex_gemv(c: &mut Criterion) {
    let one64 = Complex64::new(1.0, 0.0);
    let zero64 = Complex64::new(0.0, 0.0);
    let one32 = Complex32::new(1.0, 0.0);
    let zero32 = Complex32::new(0.0, 0.0);
    let mut group = c.benchmark_group("blas2_complex_gemv");
    group.sample_size(10);
    for n in [64_usize, 128, 256, 512, 1024, 2048, 4096] {
        // Complex GEMV performs 8 n^2 real floating-point operations.
        group.throughput(Throughput::Elements((8 * n * n) as u64));

        let a64: Vec<Complex64> = (0..n * n)
            .map(|k| Complex64::new(value(k), value(n * n + k)))
            .collect();
        let x64: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(value(2 * n * n + k), value(3 * n * n + k)))
            .collect();
        let mut y64 = vec![zero64; n];
        for op in [Transpose::NoTrans, Transpose::ConjTrans] {
            group.bench_with_input(
                BenchmarkId::new(format!("c64/{op:?}/scalar"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        blas2_scalar::gemv_c64(
                            op,
                            n,
                            n,
                            one64,
                            black_box(&a64),
                            n,
                            black_box(&x64),
                            zero64,
                            black_box(&mut y64),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("c64/{op:?}/native"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        complex_native::gemv_c64(
                            op,
                            n,
                            n,
                            one64,
                            black_box(&a64),
                            n,
                            black_box(&x64),
                            zero64,
                            black_box(&mut y64),
                        )
                    });
                },
            );
        }

        let a32: Vec<Complex32> = (0..n * n)
            .map(|k| Complex32::new(value(k) as f32, value(n * n + k) as f32))
            .collect();
        let x32: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new(value(2 * n * n + k) as f32, value(3 * n * n + k) as f32))
            .collect();
        let mut y32 = vec![zero32; n];
        for op in [Transpose::NoTrans, Transpose::ConjTrans] {
            group.bench_with_input(
                BenchmarkId::new(format!("c32/{op:?}/scalar"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        blas2_scalar::gemv_c32(
                            op,
                            n,
                            n,
                            one32,
                            black_box(&a32),
                            n,
                            black_box(&x32),
                            zero32,
                            black_box(&mut y32),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("c32/{op:?}/native"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        complex_native::gemv_c32(
                            op,
                            n,
                            n,
                            one32,
                            black_box(&a32),
                            n,
                            black_box(&x32),
                            zero32,
                            black_box(&mut y32),
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

/// The scalar row-loop implementation replaced by the blas2 dispatch.
fn matvec_scalar_f64(a: &DenseMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for (i, yi) in y.iter_mut().enumerate() {
        let row = a.row(i);
        let mut acc = 0.0;
        for (av, bv) in row.iter().zip(x.iter()) {
            acc += av * bv;
        }
        *yi = acc;
    }
}

/// The scalar row-loop implementation replaced by the blas2 dispatch.
fn matvec_scalar_c64(a: &DenseMatrix<Complex64>, x: &[Complex64], y: &mut [Complex64]) {
    for (i, yi) in y.iter_mut().enumerate() {
        let row = a.row(i);
        let mut acc = Complex64::new(0.0, 0.0);
        for (av, bv) in row.iter().zip(x.iter()) {
            acc += av * bv;
        }
        *yi = acc;
    }
}

fn bench_dense_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_gemv");
    group.sample_size(10);
    for n in [64_usize, 128, 256, 1024, 4096] {
        // GEMV performs 2 n^2 floating-point operations (per complex scalar
        // for c64, which the throughput element count treats as one unit).
        group.throughput(Throughput::Elements((2 * n * n) as u64));

        let a = matrix_f64(n);
        let x: Vec<f64> = (0..n).map(|k| value(n * n + k)).collect();
        let mut y = vec![0.0; n];
        group.bench_with_input(BenchmarkId::new("f64/blas2_dispatch", n), &n, |b, _| {
            b.iter(|| black_box(a.matvec(black_box(&x))));
        });
        group.bench_with_input(BenchmarkId::new("f64/scalar_baseline", n), &n, |b, _| {
            b.iter(|| matvec_scalar_f64(black_box(&a), black_box(&x), black_box(&mut y)));
        });

        let a = matrix_c64(n);
        let x: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(value(2 * n * n + k), value(3 * n * n + k)))
            .collect();
        let mut y = vec![Complex64::new(0.0, 0.0); n];
        group.bench_with_input(BenchmarkId::new("c64/blas2_dispatch", n), &n, |b, _| {
            b.iter(|| black_box(a.matvec(black_box(&x))));
        });
        group.bench_with_input(BenchmarkId::new("c64/scalar_baseline", n), &n, |b, _| {
            b.iter(|| matvec_scalar_c64(black_box(&a), black_box(&x), black_box(&mut y)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dense_gemv, bench_blas2_complex_gemv);
criterion_main!(benches);
