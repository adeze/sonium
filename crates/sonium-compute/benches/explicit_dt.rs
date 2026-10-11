//! Critical-dt and mass-scaling throughput (Phase 1 lane 1B).
//!
//! Covers the `sonium_compute::explicit_dt` surface:
//!
//! - `critical_dt` (serial) vs `critical_dt_parallel` min-reduce
//! - `mass_scale_uniform` and `mass_scale_selective` in-place scaling
//! - `cd_amplification` batched over a stability sweep
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p sonium-compute --bench explicit_dt
//! # filter to one kernel:
//! cargo bench -p sonium-compute --bench explicit_dt -- critical_dt
//! ```
//!
//! Sweeps `n ∈ {1_024, 65_536, 1_048_576}` elements — L1-resident,
//! LLC-resident, and DRAM-bound — so the serial/parallel crossover is
//! visible.

// Rust guideline compliant 2026-02-21

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use sonium_compute::explicit_dt;

/// Element counts swept by every benchmark group.
const SIZES: &[usize] = &[1_024, 65_536, 1_048_576];

/// Deterministic characteristic lengths in `[0.005, 0.105)` m.
fn lengths(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.005 + 0.1 * ((i as f64 * 0.0137).sin() * 0.5 + 0.5))
        .collect()
}

/// Deterministic sound speeds in `[4500, 5500)` m/s.
fn speeds(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 4500.0 + 1000.0 * ((i as f64 * 0.0213).cos() * 0.5 + 0.5))
        .collect()
}

/// Deterministic lumped masses in `[0.5, 1.5)` kg.
fn masses(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 + ((i as f64 * 0.0171).sin() * 0.5 + 0.5))
        .collect()
}

/// Numerical contract check before Criterion starts timing.
///
/// A benchmark result is not admissible performance evidence unless the
/// exact production kernels in this executable first agree with each
/// other and with closed forms.
fn validate_explicit_dt_correctness(_: &mut Criterion) {
    const N: usize = 1021;
    let les = lengths(N);
    let css = speeds(N);
    let serial = explicit_dt::critical_dt(&les, &css);
    let parallel = explicit_dt::critical_dt_parallel(&les, &css);
    println!(
        "sonium_precision_check bench=explicit_dt kernel=critical_dt serial={serial:.17e} parallel={parallel:.17e}"
    );
    assert_eq!(serial, parallel);
    let reference = les
        .iter()
        .zip(&css)
        .map(|(&le, &c)| le / c)
        .fold(f64::INFINITY, f64::min);
    assert_eq!(serial, reference);

    let (abs, arg) = explicit_dt::cd_amplification(1.0);
    println!(
        "sonium_precision_check bench=explicit_dt kernel=cd_amplification abs={abs:.17e} arg={arg:.17e}"
    );
    assert!((abs - 1.0).abs() <= 1.0e-15);
    assert!((arg - std::f64::consts::FRAC_PI_3).abs() <= 1.0e-15);

    let mut ms = masses(N);
    explicit_dt::mass_scale_uniform(&mut ms, 2.0);
    let expected = masses(N);
    assert!(
        ms.iter()
            .zip(&expected)
            .all(|(&got, &want)| got == 4.0 * want)
    );
    println!("sonium_precision_check bench=explicit_dt kernel=mass_scale_uniform ok=true");
}

fn bench_critical_dt(c: &mut Criterion) {
    let mut group = c.benchmark_group("critical_dt");
    for &n in SIZES {
        let les = lengths(n);
        let css = speeds(n);
        group.bench_with_input(BenchmarkId::new("serial", n), &n, |b, _| {
            b.iter(|| explicit_dt::critical_dt(black_box(&les), black_box(&css)));
        });
        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |b, _| {
            b.iter(|| explicit_dt::critical_dt_parallel(black_box(&les), black_box(&css)));
        });
    }
    group.finish();
}

fn bench_mass_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("mass_scale");
    for &n in SIZES {
        let initial = masses(n);
        group.bench_with_input(BenchmarkId::new("uniform", n), &n, |b, _| {
            b.iter_batched(
                || initial.clone(),
                |mut ms| explicit_dt::mass_scale_uniform(black_box(&mut ms), black_box(2.0)),
                criterion::BatchSize::SmallInput,
            );
        });
        let les = lengths(n);
        let css = speeds(n);
        // Target above every element step so the hook takes the
        // scale-up branch on all lanes.
        let target = 1.0e-3;
        group.bench_with_input(BenchmarkId::new("selective", n), &n, |b, _| {
            b.iter_batched(
                || initial.clone(),
                |mut ms| {
                    explicit_dt::mass_scale_selective(
                        black_box(&mut ms),
                        black_box(&les),
                        black_box(&css),
                        black_box(target),
                    )
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_cd_amplification(c: &mut Criterion) {
    // Stability sweep 0..=2.5 in 1024 steps, timed as one batch.
    let omegas: Vec<f64> = (0..=1024).map(|i| f64::from(i) * 2.5 / 1024.0).collect();
    c.bench_function("cd_amplification/batch_1025", |b| {
        b.iter(|| {
            let mut checksum = 0.0;
            for &omega in &omegas {
                let (abs, arg) = explicit_dt::cd_amplification(black_box(omega));
                checksum += abs + arg;
            }
            black_box(checksum)
        });
    });
}

criterion_group!(
    explicit_dt_bench,
    validate_explicit_dt_correctness,
    bench_critical_dt,
    bench_mass_scale,
    bench_cd_amplification,
);
criterion_main!(explicit_dt_bench);
