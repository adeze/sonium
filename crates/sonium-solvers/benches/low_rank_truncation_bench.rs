//! Randomized range-finder versus unblocked RRQR scaling benchmark.
//!
//! Run with:
//! `cargo bench -p sonium-solvers --bench low_rank_truncation_bench -- --quick`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::multifrontal_blr::truncation::{truncate_randomized, truncate_rrqr};
use std::hint::black_box;

fn rank_eight_block(m: usize, n: usize) -> DenseMatrix<f64> {
    let rank = 8;
    let u = DenseMatrix::from_fn(m, rank, |i, k| {
        (((i * 17 + k * 31 + i * k * 7 + 3) % 101) as f64 - 50.0) / 51.0
    });
    let v = DenseMatrix::from_fn(n, rank, |j, k| {
        (((j * 29 + k * 11 + j * k * 13 + 5) % 103) as f64 - 51.0) / 52.0
    });
    u.matmul(&v.transposed())
}

fn bench_truncation(c: &mut Criterion) {
    let mut group = c.benchmark_group("low_rank_truncation/rank8_f64");
    group.sample_size(10);
    for n in [64_usize, 128, 256, 512] {
        let block = rank_eight_block(n, n);
        group.bench_with_input(BenchmarkId::new("randomized", n), &block, |b, block| {
            b.iter(|| {
                let result = truncate_randomized(black_box(block), 1e-10);
                black_box((result.u, result.v, result.rank));
            });
        });
        group.bench_with_input(BenchmarkId::new("rrqr", n), &block, |b, block| {
            b.iter(|| {
                let result = truncate_rrqr(black_box(block), 1e-10);
                black_box((result.u, result.v, result.rank));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_truncation);
criterion_main!(benches);
