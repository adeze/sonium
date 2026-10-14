//! Frequency-lane sweep-planning bench.
//!
//! Benchmarks the lane-gate decision ([`use_frequency_lanes`]) and the
//! lane planner over a grid of sweep sizes. The gate itself is nanoseconds;
//! the value is pinning the serial/parallel crossover documented in
//! `simulation.rs` so a gate regression shows up as a bench failure
//! (wrong branch), not just a timing drift.
//!
//! `bench_lane_solve_time` additionally times real end-to-end sweeps at
//! fixed lane caps over the bundled two-way model, with a serial-parity
//! gate per case (SIM-PERF-20261013.md section 6, recommendation 6:
//! the gate-only benches above cannot catch a lane-scaling collapse).
//!
//! Measured Oct 2026 (8 freqs × 1102 DOFs, direct LU, 128-thread box):
//! lanes=1/2/4 took 16.0/31.9/16.3 s — lanes add nothing here because one
//! frequency's assembly already saturates memory bandwidth, so lanes only
//! repartition a fixed bandwidth budget (plus overhead at 2 lanes). Lanes
//! pay off where per-frequency work cannot saturate the machine (small
//! problems on few threads, or serial-section-dominated sweeps); the gate
//! question of exactly where the crossover sits needs a multi-size study
//! and is deliberately left untouched by this bench.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_sim_speaker::mesh_gen::generate_loudspeaker_mesh;
use sonium_sim_speaker::mlg_loader::load_model;
use sonium_sim_speaker::model::LoudspeakerModel;
use sonium_sim_speaker::simulation::{
    SimulationResult, SolverConfig, try_run_simulation_with_config, use_frequency_lanes,
    use_frequency_lanes_with,
};
use std::time::Duration;

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

/// Max relative complex-pressure difference between two sweeps.
fn field_parity_error(result: &SimulationResult, reference: &SimulationResult) -> f64 {
    let scale = reference
        .complex_pressure
        .as_slice()
        .iter()
        .map(|p| p.norm())
        .fold(0.0_f64, f64::max);
    result
        .complex_pressure
        .as_slice()
        .iter()
        .zip(reference.complex_pressure.as_slice())
        .map(|(p, r)| (p - r).norm() / r.norm().max(scale * 1e-6).max(1e-30))
        .fold(0.0, f64::max)
}

/// Small two-way sweep sized so the lane gate opens on any multi-threaded
/// machine while one serial iteration stays in the tens of seconds.
fn small_sweep_fixture() -> (LoudspeakerModel, Vec<[f64; 3]>, usize) {
    let mut model = load_model(
        include_str!("../examples/two_way_speaker.mlg"),
        "two_way_speaker.mlg",
    )
    .expect("bundled two-way model must load");
    model.simulation.freq_start = 200.0;
    model.simulation.freq_end = 900.0;
    model.simulation.freq_points = 8;
    model.simulation.max_mesh_frequency = Some(1250.0);
    let mesh = generate_loudspeaker_mesh(&model);
    let n_dofs = mesh.triangles.len();
    let front = model.baffle_center()[2];
    let points = vec![[0.0, 0.0, front + 1.0]];
    (model, points, n_dofs)
}

fn bench_lane_solve_time(c: &mut Criterion) {
    let (model, points, n_dofs) = small_sweep_fixture();
    let n_freqs = model.simulation.freq_points;
    // Fail closed if the fixture no longer opens the gate: timing
    // single-lane runs under several lane caps would report a flat
    // (meaningless) scaling curve instead of a bench failure.
    assert!(
        use_frequency_lanes(n_freqs, n_dofs),
        "lane-gate shut for {n_freqs}x{n_dofs}: raise the fixture mesh cap",
    );
    let reference = try_run_simulation_with_config(
        &model,
        &points,
        &SolverConfig {
            parallel_frequency_sweep: false,
            ..Default::default()
        },
    )
    .expect("serial reference sweep");
    let mut group = c.benchmark_group("lane_solve");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));
    for lanes in [1usize, 2, 4] {
        let config = SolverConfig {
            max_frequency_lanes: Some(lanes),
            ..Default::default()
        };
        let trial =
            try_run_simulation_with_config(&model, &points, &config).expect("lane trial sweep");
        let error = field_parity_error(&trial, &reference);
        assert!(
            error.is_finite() && error < 1e-6,
            "lanes={lanes}: serial-parity gate failed with error {error:.3e}",
        );
        group.bench_with_input(
            BenchmarkId::new("sweep_lanes", format!("lanes={lanes}")),
            &config,
            |b, config| {
                b.iter(|| {
                    std::hint::black_box(
                        try_run_simulation_with_config(&model, &points, config)
                            .expect("bench sweep"),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_lane_gate,
    bench_lane_gate_pure,
    bench_lane_solve_time
);
criterion_main!(benches);
