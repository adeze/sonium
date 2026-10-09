//! Frequency-lane sweep-planning bench.
//!
//! Benchmarks the lane-gate decision ([`use_frequency_lanes`]) and the
//! lane planner over a grid of sweep sizes. The gate itself is nanoseconds;
//! the value is pinning the serial/parallel crossover documented in
//! `simulation.rs` so a gate regression shows up as a bench failure
//! (wrong branch), not just a timing drift.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_sim_speaker::simulation::{use_frequency_lanes, use_frequency_lanes_with};

/// (frequencies, dofs) probe points from laptop sweeps to product runs.
const SWEEP_GRID: &[(usize, usize)] = &[(2, 16), (8, 256), (32, 1024), (64, 4096), (256, 4096)];

fn bench_lane_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("lane_gate");
    for &(n_freqs, n_dofs) in SWEEP_GRID {
        group.bench_with_input(
            BenchmarkId::new("use_frequency_lanes", format!("{n_freqs}x{n_dofs}")),
            &(n_freqs, n_dofs),
            |b, &(n_freqs, n_dofs)| {
                b.iter(|| use_frequency_lanes(n_freqs, n_dofs));
            },
        );
    }
    group.finish();
}

fn bench_lane_gate_pure(c: &mut Criterion) {
    // Pure helper over explicit thread counts: documents the
    // machine-adaptive crossover (grain × threads) independent of the
    // ambient machine.
    let mut group = c.benchmark_group("lane_gate_pure");
    for threads in [1usize, 2, 8, 32] {
        group.bench_with_input(
            BenchmarkId::new("use_frequency_lanes_with", format!("threads={threads}")),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    for &(n_freqs, n_dofs) in SWEEP_GRID {
                        std::hint::black_box(use_frequency_lanes_with(
                            n_freqs, n_dofs, false, None, threads,
                        ));
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_lane_gate, bench_lane_gate_pure);
criterion_main!(benches);
