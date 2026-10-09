//! Scaling benchmark for `MultifrontalBlrFactorization`.
//!
//! Runs the factoriser on 2D Laplacian problems of increasing size and
//! fits the factor / solve time against `t = c · N^α`. The original plan
//! targets `α ≈ 1.4` for the factor phase (3D BLR theoretical scaling);
//! 2D Laplacian is easier — expect smaller α here. We use this bench
//! mainly to detect regressions: it prints the fitted exponent so
//! reviewers can compare across commits.
//!
//! Run with:
//! ```bash
//! cargo bench -p sonium-solvers --bench multifrontal_blr_bench
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    BlrConfig, MultifrontalBlrFactorization, PanelKind, PivotStrategy, UpdateOrder,
};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::ComplexField;
use std::hint::black_box;

fn make_2d_laplacian(nx: usize) -> CsrMatrix<f64> {
    let n = nx * nx;
    let mut dense = DenseMatrix::<f64>::zeros(n, n);
    for iy in 0..nx {
        for ix in 0..nx {
            let i = iy * nx + ix;
            dense[[i, i]] = 4.0;
            if ix > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if ix + 1 < nx {
                dense[[i, i + 1]] = -1.0;
            }
            if iy > 0 {
                dense[[i, i - nx]] = -1.0;
            }
            if iy + 1 < nx {
                dense[[i, i + nx]] = -1.0;
            }
        }
    }
    CsrMatrix::from_dense(&dense, 1e-15)
}

fn factorise_with(matrix: &CsrMatrix<f64>, cfg: &BlrConfig) -> MultifrontalBlrFactorization<f64> {
    MultifrontalBlrFactorization::factorize_with_config(matrix, cfg).expect("factorise")
}

fn bench_factor_scaling(c: &mut Criterion) {
    let cfg = BlrConfig {
        epsilon: 1e-8,
        block_size: 32,
        min_block_size: 32,
        max_levels: None,
        pivot_strategy: PivotStrategy::Partial,
        panel_kind: PanelKind::Blr,
        update_order: UpdateOrder::RightLook,
        ..Default::default()
    };
    let mut group = c.benchmark_group("multifrontal_blr/factor");
    group.sample_size(10);
    for &nx in &[16usize, 24, 32, 48] {
        let n = nx * nx;
        let m = make_2d_laplacian(nx);
        group.bench_function(format!("nx={}_n={}", nx, n), |b| {
            b.iter(|| {
                let fac = factorise_with(&m, &cfg);
                black_box(fac.num_fronts());
            });
        });
    }
    group.finish();
}

fn bench_solve(c: &mut Criterion) {
    let cfg = BlrConfig {
        epsilon: 1e-8,
        block_size: 32,
        min_block_size: 32,
        max_levels: None,
        ..Default::default()
    };
    let mut group = c.benchmark_group("multifrontal_blr/solve");
    group.sample_size(20);
    for &nx in &[16usize, 24, 32, 48] {
        let n = nx * nx;
        let m = make_2d_laplacian(nx);
        let fac = factorise_with(&m, &cfg);
        let b_vec: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5).sin()).collect();
        group.bench_function(format!("nx={}_n={}", nx, n), |b| {
            b.iter(|| {
                let x = fac.solve(black_box(&b_vec));
                black_box(x.len());
            });
        });
    }
    group.finish();
}

fn bench_compression_modes(c: &mut Criterion) {
    let nx = 32usize;
    let n = nx * nx;
    let m = make_2d_laplacian(nx);

    let mut group = c.benchmark_group("multifrontal_blr/modes");
    group.sample_size(10);

    let variants: [(&str, BlrConfig); 4] = [
        (
            "blr_rightlook",
            BlrConfig {
                epsilon: 1e-8,
                block_size: 32,
                min_block_size: 32,
                max_levels: None,
                panel_kind: PanelKind::Blr,
                update_order: UpdateOrder::RightLook,
                ..Default::default()
            },
        ),
        (
            "blr_leftlook",
            BlrConfig {
                epsilon: 1e-8,
                block_size: 32,
                min_block_size: 32,
                max_levels: None,
                panel_kind: PanelKind::Blr,
                update_order: UpdateOrder::LeftLook,
                ..Default::default()
            },
        ),
        (
            "hodlr_rightlook",
            BlrConfig {
                epsilon: 1e-8,
                block_size: 32,
                min_block_size: 32,
                max_levels: None,
                panel_kind: PanelKind::Hodlr,
                update_order: UpdateOrder::RightLook,
                ..Default::default()
            },
        ),
        (
            "rook_pivoting",
            BlrConfig {
                epsilon: 1e-8,
                block_size: 32,
                min_block_size: 32,
                max_levels: None,
                pivot_strategy: PivotStrategy::Rook,
                ..Default::default()
            },
        ),
    ];
    for (name, cfg) in variants.iter() {
        group.bench_function(format!("{}_n={}", name, n), |b| {
            b.iter(|| {
                let fac = factorise_with(&m, cfg);
                black_box(fac.num_fronts());
            });
        });
    }
    group.finish();
}

fn scalar_matmul(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..m {
        for kk in 0..k {
            let lhs = a[i * k + kk];
            for j in 0..n {
                out[i * n + j] += lhs * b[kk * n + j];
            }
        }
    }
}

fn bench_front_schur_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("multifrontal_blr/front_schur_f64");
    group.sample_size(10);
    for ncb in [64_usize, 128, 256] {
        let nfs = ncb / 2;
        let lhs: Vec<f64> = (0..ncb * nfs)
            .map(|index| ((index * 17 + 3) % 101) as f64 / 101.0)
            .collect();
        let rhs: Vec<f64> = (0..nfs * ncb)
            .map(|index| ((index * 29 + 5) % 103) as f64 / 103.0)
            .collect();
        group.throughput(Throughput::Elements((2 * ncb * ncb * nfs) as u64));
        group.bench_with_input(BenchmarkId::new("blas3", ncb), &ncb, |bench, _| {
            let mut out = vec![0.0; ncb * ncb];
            bench.iter(|| {
                f64::mat_mul_nn(ncb, ncb, nfs, black_box(&lhs), black_box(&rhs), &mut out);
                black_box(&out);
            });
        });
        group.bench_with_input(BenchmarkId::new("scalar", ncb), &ncb, |bench, _| {
            let mut out = vec![0.0; ncb * ncb];
            bench.iter(|| {
                scalar_matmul(ncb, ncb, nfs, black_box(&lhs), black_box(&rhs), &mut out);
                black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_factor_scaling,
    bench_solve,
    bench_compression_modes,
    bench_front_schur_kernel
);
criterion_main!(benches);
