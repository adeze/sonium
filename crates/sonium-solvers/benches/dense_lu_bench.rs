//! Dense LU factorization benchmark.
//!
//! Compares the blocked Sonium factorizer with the former unblocked
//! right-looking kernel and, on macOS, Accelerate's optimized LAPACK `dgetrf`.
//! The input matrix is cloned outside the measured factorization where the API
//! permits it; the Sonium timing includes its documented input clone because
//! `lu_factorize` is a non-mutating public API.
//!
//! Run a short comparison with:
//! `cargo bench -p sonium-solvers --bench dense_lu_bench -- --quick`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_factorize;
use std::hint::black_box;

fn matrix(n: usize) -> DenseMatrix<f64> {
    DenseMatrix::from_fn(n, n, |i, j| {
        let value = (((i * 37 + j * 17 + 11) % 101) as f64 - 50.0) / 101.0;
        if i == j { value + n as f64 } else { value }
    })
}

/// The scalar right-looking implementation replaced by the blocked kernel.
fn unblocked_lu(mut a: DenseMatrix<f64>) -> bool {
    let n = a.nrows();
    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[[k, k]].abs();
        for i in (k + 1)..n {
            let candidate = a[[i, k]].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }
        if pivot_abs < 1e-20 {
            return false;
        }
        a.swap_rows(k, pivot_row);
        let pivot_inv = 1.0 / a[[k, k]];
        for i in (k + 1)..n {
            let multiplier = a[[i, k]] * pivot_inv;
            a[[i, k]] = multiplier;
            for j in (k + 1)..n {
                a[[i, j]] -= multiplier * a[[k, j]];
            }
        }
    }
    black_box(a);
    true
}

#[cfg(target_os = "macos")]
mod platform_lapack {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn dgetrf_(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            info: *mut i32,
        );
    }

    pub fn factorize(row_major: &[f64], n: usize) -> bool {
        // LAPACK consumes column-major storage.
        let mut column_major = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                column_major[j * n + i] = row_major[i * n + j];
            }
        }
        let order = i32::try_from(n).expect("benchmark matrix exceeds LAPACK i32 ABI");
        let mut pivots = vec![0_i32; n];
        let mut info = 0_i32;
        unsafe {
            dgetrf_(
                &order,
                &order,
                column_major.as_mut_ptr(),
                &order,
                pivots.as_mut_ptr(),
                &mut info,
            );
        }
        std::hint::black_box((column_major, pivots));
        info == 0
    }
}

fn bench_dense_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_lu/factorize_f64");
    group.sample_size(10);
    for n in [64_usize, 128, 256, 512, 1024] {
        let input = matrix(n);
        // GETRF performs 2/3 n^3 floating-point operations asymptotically.
        group.throughput(Throughput::Elements(
            ((2.0 / 3.0) * (n as f64).powi(3)) as u64,
        ));
        group.bench_with_input(BenchmarkId::new("sonium_blocked", n), &input, |b, a| {
            b.iter(|| {
                let factor = lu_factorize(black_box(a)).expect("factorization");
                black_box(factor);
            });
        });
        group.bench_with_input(BenchmarkId::new("sonium_unblocked", n), &input, |b, a| {
            b.iter(|| assert!(unblocked_lu(black_box(a.clone()))));
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("accelerate_dgetrf", n), &input, |b, a| {
            b.iter(|| assert!(platform_lapack::factorize(black_box(a.as_slice()), n)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dense_lu);
criterion_main!(benches);
