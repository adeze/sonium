//! Sparse-LU factorization and threshold-pivoting benchmark.
//!
//! Run with:
//! `cargo bench -p sonium-solvers --bench sparse_lu_bench -- --quick`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::SparseLuFactorization;
use sonium_solvers::sparse::CsrMatrix;
use std::hint::black_box;

fn convection_diffusion(nx: usize) -> CsrMatrix<f64> {
    let n = nx * nx;
    let mut dense = DenseMatrix::zeros(n, n);
    for y in 0..nx {
        for x in 0..nx {
            let i = y * nx + x;
            dense[[i, i]] = 4.01;
            if x > 0 {
                dense[[i, i - 1]] = -1.2;
            }
            if x + 1 < nx {
                dense[[i, i + 1]] = -0.8;
            }
            if y > 0 {
                dense[[i, i - nx]] = -1.1;
            }
            if y + 1 < nx {
                dense[[i, i + nx]] = -0.9;
            }
        }
    }
    CsrMatrix::from_dense(&dense, 0.0)
}

fn bench_sparse_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_lu/factor_convection_diffusion");
    group.sample_size(10);
    for nx in [12_usize, 20, 28] {
        let matrix = convection_diffusion(nx);
        let symbolic = SparseLuFactorization::analyze_supernodal(&matrix).unwrap();
        group.bench_with_input(
            BenchmarkId::new("partial", nx * nx),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let factor = SparseLuFactorization::factorize(black_box(matrix)).unwrap();
                    black_box(factor.factor_nonzeros());
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("threshold_0.1", nx * nx),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let factor = SparseLuFactorization::factorize_with_pivot_threshold(
                        black_box(matrix),
                        0.1,
                    )
                    .unwrap();
                    black_box(factor.factor_nonzeros());
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("supernodal_partial", nx * nx),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let factor =
                        SparseLuFactorization::factorize_supernodal(black_box(matrix)).unwrap();
                    black_box((factor.factor_nonzeros(), factor.supernode_count()));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("supernodal_threshold_0.1", nx * nx),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let factor = SparseLuFactorization::factorize_supernodal_with_pivot_threshold(
                        black_box(matrix),
                        0.1,
                    )
                    .unwrap();
                    black_box((factor.factor_nonzeros(), factor.supernode_count()));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("supernodal_numeric_only", nx * nx),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let factor = SparseLuFactorization::factorize_supernodal_with_symbolic(
                        black_box(matrix),
                        black_box(&symbolic),
                        0.1,
                    )
                    .unwrap();
                    black_box(factor.factor_nonzeros());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_sparse_lu);
criterion_main!(benches);
