//! Scatter/gather and lumped-mass throughput (Phase 1 lane 1A).
//!
//! Steady-state accumulation throughput for [`gather_nodal`],
//! [`scatter_add_nodal`], [`scatter_add_nodal_parallel`], and
//! [`assemble_lumped_mass`] over a synthetic hex8 chain mesh where
//! element `e` owns nodes `(e + k) % N`, so every node is shared by 8
//! elements (the race case scatter must handle).
//!
//! Sweeps `N in {256, 4_096, 65_536, 262_144}` nodes with `E = N`
//! elements, from L1-resident to DRAM-bound. Scatter and assembly
//! targets accumulate across iterations, matching the `blas1_compare`
//! precedent of measuring the hot loop rather than setup; values stay
//! far from overflow for any realistic iteration count.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p sonium-compute --bench explicit_scatter
//! # filter to one kernel:
//! cargo bench -p sonium-compute --bench explicit_scatter -- scatter_parallel
//! ```
//!
//! [`gather_nodal`]: sonium_compute::explicit_scatter::gather_nodal
//! [`scatter_add_nodal`]: sonium_compute::explicit_scatter::scatter_add_nodal
//! [`scatter_add_nodal_parallel`]:
//!     sonium_compute::explicit_scatter::scatter_add_nodal_parallel
//! [`assemble_lumped_mass`]:
//!     sonium_compute::explicit_scatter::assemble_lumped_mass

// Rust guideline compliant 2026-02-21

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use sonium_compute::explicit_scatter::{
    ElementTopology, assemble_lumped_mass, gather_nodal, scatter_add_nodal,
    scatter_add_nodal_parallel,
};

/// Node counts swept by every benchmark group. Chosen to span:
///   256      — L1-resident, dispatch cost dominates
///   4 096    — comfortably L2
///   65 536   — last-level-cache territory
///   262 144  — DRAM-bound (multi-MiB fields)
const SIZES: &[usize] = &[256, 4_096, 65_536, 262_144];

/// Nodes per synthetic element (hex8 chain).
const NODES_PER_ELEMENT: usize = 8;

// ---------------------------------------------------------------------
// Input generators (deterministic so every kernel sees the same data)
// ---------------------------------------------------------------------

/// Hex8 chain mesh with `elements` elements over `nodes` nodes; element
/// `e` owns nodes `(e + k) % nodes` for `k` in `0..8`.
fn chain_hex_mesh(nodes: usize, elements: usize) -> (Vec<usize>, Vec<usize>) {
    let mut connectivity = Vec::with_capacity(NODES_PER_ELEMENT * elements);
    let mut offsets = Vec::with_capacity(elements + 1);
    for element in 0..elements {
        offsets.push(connectivity.len());
        for k in 0..NODES_PER_ELEMENT {
            connectivity.push((element + k) % nodes);
        }
    }
    offsets.push(connectivity.len());
    (connectivity, offsets)
}

/// Deterministic fractional ramp over `len` values.
fn ramp(len: usize) -> Vec<f64> {
    (0..len).map(|i| ((i as f64) * 0.0123).sin()).collect()
}

/// Deterministic strictly positive ramp (volumes and densities).
fn positive_ramp(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as f64) * 0.0177).sin() + 2.0)
        .collect()
}

/// Run exactness checks before Criterion starts timing.
///
/// A benchmark result is not admissible performance evidence unless the
/// production kernels in this executable first pass their contract:
/// serial and parallel scatter agree bit-exactly on an odd-sized mesh,
/// and lumped-mass assembly conserves the element-mass total.
fn validate_scatter_correctness(c: &mut Criterion) {
    c.bench_function("validate_scatter_correctness", |b| {
        b.iter(|| {
            let nodes = 37;
            let (connectivity, offsets) = chain_hex_mesh(nodes, 64);
            let values = ramp(3 * connectivity.len());
            let mut serial = vec![0.0; 3 * nodes];
            let mut parallel = vec![0.0; 3 * nodes];
            scatter_add_nodal(&mut serial, &connectivity, &values);
            scatter_add_nodal_parallel(&mut parallel, &connectivity, &values);
            assert_eq!(serial, parallel, "serial/parallel scatter disagree");
            let mut gathered = vec![0.0; values.len()];
            gather_nodal(&serial, &connectivity, &mut gathered);
            assert_eq!(gathered.len(), values.len());

            let topology = ElementTopology::new(&connectivity, &offsets);
            let volumes = positive_ramp(64);
            let densities = positive_ramp(64);
            let mut masses = vec![0.0; nodes];
            assemble_lumped_mass(&mut masses, topology, &volumes, &densities);
            let total: f64 = masses.iter().sum();
            let want: f64 = volumes
                .iter()
                .zip(densities.iter())
                .map(|(&volume, &density)| density * volume)
                .sum();
            let rel = (total - want).abs() / want.abs().max(1.0e-300);
            println!(
                "sonium_precision_check bench=explicit_scatter operation=lumped_mass \
                 shape=n=37 rel={rel:.9e} tolerance=1.000000000e-12"
            );
            assert!(rel <= 1.0e-12, "lumped-mass row sum failed: rel={rel:.9e}");
            black_box((serial, parallel, masses));
        });
    });
}

fn bench_gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("gather_nodal");
    for &nodes in SIZES {
        let field = ramp(3 * nodes);
        let (connectivity, _) = chain_hex_mesh(nodes, nodes);
        let mut out = vec![0.0; 3 * connectivity.len()];
        group.bench_with_input(BenchmarkId::new("gather", nodes), &nodes, |b, _| {
            b.iter(|| {
                gather_nodal(
                    black_box(&field),
                    black_box(&connectivity),
                    black_box(&mut out),
                );
            });
        });
    }
    group.finish();
}

fn bench_scatter_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("scatter_serial");
    for &nodes in SIZES {
        let (connectivity, _) = chain_hex_mesh(nodes, nodes);
        let values = ramp(3 * connectivity.len());
        let mut target = vec![0.0; 3 * nodes];
        group.bench_with_input(BenchmarkId::new("serial", nodes), &nodes, |b, _| {
            b.iter(|| {
                scatter_add_nodal(
                    black_box(&mut target),
                    black_box(&connectivity),
                    black_box(&values),
                );
            });
        });
    }
    group.finish();
}

fn bench_scatter_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("scatter_parallel");
    for &nodes in SIZES {
        let (connectivity, _) = chain_hex_mesh(nodes, nodes);
        let values = ramp(3 * connectivity.len());
        let mut target = vec![0.0; 3 * nodes];
        group.bench_with_input(BenchmarkId::new("parallel", nodes), &nodes, |b, _| {
            b.iter(|| {
                scatter_add_nodal_parallel(
                    black_box(&mut target),
                    black_box(&connectivity),
                    black_box(&values),
                );
            });
        });
    }
    group.finish();
}

fn bench_lumped_mass(c: &mut Criterion) {
    let mut group = c.benchmark_group("lumped_mass");
    for &nodes in SIZES {
        let (connectivity, offsets) = chain_hex_mesh(nodes, nodes);
        let topology = ElementTopology::new(&connectivity, &offsets);
        let volumes = positive_ramp(nodes);
        let densities = positive_ramp(nodes);
        let mut masses = vec![0.0; nodes];
        group.bench_with_input(BenchmarkId::new("assemble", nodes), &nodes, |b, _| {
            b.iter(|| {
                assemble_lumped_mass(
                    black_box(&mut masses),
                    black_box(topology),
                    black_box(&volumes),
                    black_box(&densities),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(
    explicit_scatter_bench,
    validate_scatter_correctness,
    bench_gather,
    bench_scatter_serial,
    bench_scatter_parallel,
    bench_lumped_mass,
);
criterion_main!(explicit_scatter_bench);
