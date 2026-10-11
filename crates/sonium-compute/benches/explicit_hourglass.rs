//! Hourglass-control and bulk-viscosity throughput benches.
//!
//! Phase 1 lane 1C kernel benches over the production
//! `sonium_compute::explicit_hourglass` API: Gamma assembly, hourglass
//! force evaluation, and batched bulk viscosity. Each group sweeps
//! element counts spanning L1-resident to DRAM-bound regimes and
//! reports elements/second via Criterion throughput.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p sonium-compute --bench explicit_hourglass
//! # filter to one kernel:
//! cargo bench -p sonium-compute --bench explicit_hourglass -- hexa8_gamma
//! ```

// Rust guideline compliant 2026-02-21

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_compute::explicit_hourglass::{
    bulk_viscosity_q, bulk_viscosity_q_batch, hexa8_gamma, hexa8_hourglass_forces,
};
use std::hint::black_box;

/// Element counts swept by every group: L1-resident, L2 territory,
/// and DRAM-bound at 24+32 doubles per element.
const SIZES: &[usize] = &[64, 1_024, 16_384];

/// Stack of translated unit cubes: element `e` spans `[e, e+1]^3`.
fn cube_stack(n: usize) -> Vec<f64> {
    let mut coords = vec![0.0; 24 * n];
    for e in 0..n {
        let base = e as f64;
        for node in 0..8 {
            let nx = f64::from((node == 1 || node == 2 || node == 5 || node == 6) as u8);
            let ny = f64::from((node == 2 || node == 3 || node == 6 || node == 7) as u8);
            let nz = f64::from((node >= 4) as u8);
            coords[24 * e + 3 * node] = base + nx;
            coords[24 * e + 3 * node + 1] = ny;
            coords[24 * e + 3 * node + 2] = nz;
        }
    }
    coords
}

/// Deterministic modal velocity field with hourglass content.
fn modal_rates(n: usize) -> Vec<f64> {
    let mut rates = vec![0.0; 24 * n];
    for e in 0..n {
        let scale = 1.0 + 0.125 * (e % 7) as f64;
        for node in 0..8 {
            // Mode-1 pattern plus a linear shear plus translation.
            let h1 = if node % 2 == 0 { 1.0 } else { -1.0 };
            rates[24 * e + 3 * node] = scale * (0.5 * h1 + 1.0);
            rates[24 * e + 3 * node + 1] = scale * (0.25 * node as f64 - 0.5);
            rates[24 * e + 3 * node + 2] = scale * 0.125;
        }
    }
    rates
}

/// Compression-heavy strain-rate ladder spanning linear to quadratic regimes.
fn strain_rate_ladder(n: usize) -> Vec<f64> {
    const LADDER: [f64; 5] = [-50_000.0, -1_000.0, -100.0, 0.0, 500.0];
    (0..n).map(|e| LADDER[e % LADDER.len()]).collect()
}

fn report_precision_check(
    operation: &str,
    shape: &str,
    max_abs: f64,
    max_rel: f64,
    tolerance: f64,
) {
    println!(
        "sonium_precision_check bench=explicit_hourglass operation={operation} \
         shape={shape} max_abs={max_abs:.9e} max_rel={max_rel:.9e} tolerance={tolerance:.9e}"
    );
    assert!(
        max_rel <= tolerance,
        "hourglass {operation} precision check failed for {shape}: \
         max_abs={max_abs:.9e}, max_rel={max_rel:.9e}, tolerance={tolerance:.9e}"
    );
}

/// Correctness gate: the production kernels must pass their numerical
/// contract before Criterion starts timing.
fn validate_hourglass_correctness(_: &mut Criterion) {
    // Gamma of the unit cube equals the exact base modes.
    let coords = cube_stack(1);
    let mut gamma = vec![0.0; 32];
    hexa8_gamma(&coords, &mut gamma);
    let base: [f64; 32] = [
        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
        1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    ];
    let max_abs = gamma
        .iter()
        .zip(base.iter())
        .map(|(got, exact)| (got - exact).abs())
        .fold(0.0f64, f64::max);
    report_precision_check("hexa8_gamma", "unit-cube", max_abs, max_abs, 0.0);

    // Viscous power identity P = kappa * sum(q^2) on a mixed field.
    let rates = modal_rates(1);
    let mut forces = vec![0.0; 24];
    hexa8_hourglass_forces(&gamma, &rates, 0.1, &mut forces);
    let power: f64 = rates.iter().zip(forces.iter()).map(|(v, f)| v * f).sum();
    let mut modal = 0.0;
    for i in 0..3 {
        for a in 0..4 {
            let mut q = 0.0;
            for node in 0..8 {
                q += gamma[a * 8 + node] * rates[3 * node + i];
            }
            modal += q * q;
        }
    }
    let expected = 0.1 * modal;
    assert!(power > 0.0, "mixed field must dissipate");
    let abs = (power - expected).abs();
    let rel = abs / expected.abs().max(1.0e-300);
    report_precision_check("hexa8_hourglass_forces", "mixed-field", abs, rel, 1.0e-12);

    // Bulk viscosity hand value plus expansion cutoff.
    let q = bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, -1000.0);
    let abs = (q - 18_408_000.0).abs();
    let rel = abs / 18_408_000.0;
    report_precision_check("bulk_viscosity_q", "steel-shock", abs, rel, 1.0e-12);
    assert_eq!(
        bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, 500.0),
        0.0
    );
}

fn bench_hexa8_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("hexa8_gamma");
    for &n in SIZES {
        let coords = cube_stack(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut gamma = vec![0.0; 32 * n];
            b.iter(|| {
                for e in 0..n {
                    hexa8_gamma(
                        black_box(&coords[24 * e..24 * e + 24]),
                        black_box(&mut gamma[32 * e..32 * e + 32]),
                    );
                }
                black_box(&gamma);
            });
        });
    }
    group.finish();
}

fn bench_hexa8_hourglass_forces(c: &mut Criterion) {
    let mut group = c.benchmark_group("hexa8_hourglass_forces");
    for &n in SIZES {
        let coords = cube_stack(n);
        let rates = modal_rates(n);
        let mut gamma = vec![0.0; 32 * n];
        for e in 0..n {
            hexa8_gamma(
                &coords[24 * e..24 * e + 24],
                &mut gamma[32 * e..32 * e + 32],
            );
        }
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut forces = vec![0.0; 24 * n];
            b.iter(|| {
                for e in 0..n {
                    hexa8_hourglass_forces(
                        black_box(&gamma[32 * e..32 * e + 32]),
                        black_box(&rates[24 * e..24 * e + 24]),
                        black_box(0.1),
                        black_box(&mut forces[24 * e..24 * e + 24]),
                    );
                }
                black_box(&forces);
            });
        });
    }
    group.finish();
}

fn bench_bulk_viscosity_q(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_viscosity_q");
    for &n in SIZES {
        let rho = vec![7800.0; n];
        let sound = vec![4500.0; n];
        let len = vec![0.01; n];
        let rates = strain_rate_ladder(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("batch", n), &n, |b, _| {
            let mut q = vec![0.0; n];
            b.iter(|| {
                bulk_viscosity_q_batch(
                    black_box(&rho),
                    black_box(&sound),
                    black_box(&len),
                    black_box(1.1),
                    black_box(0.05),
                    black_box(&rates),
                    black_box(&mut q),
                );
                black_box(&q);
            });
        });
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            let mut q = vec![0.0; n];
            b.iter(|| {
                for e in 0..n {
                    q[e] = bulk_viscosity_q(
                        black_box(rho[e]),
                        black_box(sound[e]),
                        black_box(len[e]),
                        black_box(1.1),
                        black_box(0.05),
                        black_box(rates[e]),
                    );
                }
                black_box(&q);
            });
        });
    }
    group.finish();
}

criterion_group!(
    explicit_hourglass_bench,
    validate_hourglass_correctness,
    bench_hexa8_gamma,
    bench_hexa8_hourglass_forces,
    bench_bulk_viscosity_q,
);
criterion_main!(explicit_hourglass_bench);
