//! Audit `reviews/audit-sonium-wave-20260921.md`: batched observation cost.
//!
//! The audit permits measuring vectorized/batched observation evaluation only
//! after the reference error is bounded. That bound exists:
//! `tests/series_convergence.rs` proves the suggested truncations hold the
//! series tail below `1e-9` for `ka <= 20`, and `tests/scattering_golden.rs`
//! pins the coefficients against an independent oracle. This bench therefore
//! measures the batch interfaces consumers use for observation grids —
//! contiguous spherical-Bessel batches and per-pair Helmholtz kernel
//! assembly — as a sizing input, not a speed claim.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_wave::Point;
use sonium_wave::special::{all_kernels_3d, try_spherical_bessel_j_batch_into};

fn bench_bessel_batch(criterion: &mut Criterion) {
    for num_points in [256usize, 4096] {
        let arguments: Vec<f64> = (0..num_points)
            .map(|i| 0.5 + 19.5 * i as f64 / num_points as f64)
            .collect();
        let order = 32usize;
        let mut output = vec![0.0; order * num_points];
        let mut workspace = Vec::new();
        criterion.bench_with_input(
            BenchmarkId::new("spherical_bessel_j_batch/order32", num_points),
            &arguments,
            |bencher, arguments| {
                bencher.iter(|| {
                    try_spherical_bessel_j_batch_into(order, arguments, &mut output, &mut workspace)
                        .expect("bench batch must evaluate")
                })
            },
        );
    }
}

fn bench_kernel_assembly(criterion: &mut Criterion) {
    let source = Point::new_3d(0.0, 0.0, 0.0);
    let normal = [0.0, 0.0, 1.0];
    let k = 2.0 * std::f64::consts::PI / 0.5;
    for num_points in [256usize, 4096] {
        let fields: Vec<Point> = (0..num_points)
            .map(|i| {
                let t = i as f64 / num_points as f64;
                Point::new_3d(1.5 + t, -0.3 + 0.5 * t, 0.5 - t)
            })
            .collect();
        criterion.bench_with_input(
            BenchmarkId::new("all_kernels_3d", num_points),
            &fields,
            |bencher, fields| {
                bencher.iter(|| {
                    let mut acc = 0.0;
                    for field in fields {
                        let (g, _, _, _) = all_kernels_3d(&source, field, &normal, &normal, k);
                        acc += g.norm();
                    }
                    acc
                })
            },
        );
    }
}

criterion_group!(benches, bench_bessel_batch, bench_kernel_assembly);
criterion_main!(benches);
