//! Restart round-trip throughput vs node count.
//!
//! Times [`encode_restart`](sonium_compute::explicit_restart::encode_restart),
//! [`decode_restart`](sonium_compute::explicit_restart::decode_restart), and
//! the full encode+decode round trip over deterministic states, reporting
//! bytes per second against the encoded document size.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p sonium-compute --bench explicit_restart
//! # filter to one leg:
//! cargo bench -p sonium-compute --bench explicit_restart -- encode
//! ```
//!
//! Sweeps `n ∈ {64, 1_024, 16_384}` nodes — kilobyte / tens-of-kilobyte /
//! megabyte restart documents.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use sonium_compute::explicit_restart::{decode_restart, encode_restart};
use sonium_compute::explicit_state::ExplicitState;

/// Node counts swept by every benchmark group.
const SIZES: &[usize] = &[64, 1_024, 16_384];

/// Builds the deterministic state every leg of the bench replays.
fn bench_state(nodes: usize) -> ExplicitState {
    let mut state = ExplicitState::new(nodes);
    for (index, value) in state.positions.iter_mut().enumerate() {
        *value = (f64::from(u32::try_from(index).unwrap_or(u32::MAX)) * 0.0123).sin();
    }
    for (index, value) in state.velocities.iter_mut().enumerate() {
        *value = (f64::from(u32::try_from(index).unwrap_or(u32::MAX)) * 0.0177).cos() * 10.0;
    }
    for (index, value) in state.masses.iter_mut().enumerate() {
        *value = 1.0 + f64::from(u32::try_from(index).unwrap_or(u32::MAX)) * 0.001;
    }
    for (index, value) in state.forces.iter_mut().enumerate() {
        *value = -f64::from(u32::try_from(index).unwrap_or(u32::MAX)) * 0.25;
    }
    state.time = 0.125;
    state.step = 7;
    state.dt = 1.0e-6;
    state
}

/// Proves the timed path round-trips before Criterion starts measuring.
///
/// A benchmark result is not admissible performance evidence unless the
/// exact production functions in this executable first agree bit-wise.
fn validate_restart_round_trip(_: &mut Criterion) {
    const NODES: usize = 37;
    let state = bench_state(NODES);
    let bytes = encode_restart(&state).expect("bench state must encode");
    let back = decode_restart(&bytes).expect("bench bytes must decode");
    assert_eq!(back, state, "restart round trip must be bit-exact");
    println!(
        "sonium_precision_check bench=explicit_restart operation=round_trip \
         shape=n={NODES} bytes={} status=bit_exact",
        bytes.len()
    );
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    for &nodes in SIZES {
        let state = bench_state(nodes);
        let bytes = encode_restart(&state).expect("bench state must encode");
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(BenchmarkId::new("encode", nodes), &nodes, |b, _| {
            b.iter(|| encode_restart(black_box(&state)));
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    for &nodes in SIZES {
        let bytes = encode_restart(&bench_state(nodes)).expect("bench state must encode");
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(BenchmarkId::new("decode", nodes), &nodes, |b, _| {
            b.iter(|| decode_restart(black_box(&bytes)));
        });
    }
    group.finish();
}

fn bench_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_trip");
    for &nodes in SIZES {
        let state = bench_state(nodes);
        let bytes = encode_restart(&state).expect("bench state must encode");
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(BenchmarkId::new("round_trip", nodes), &nodes, |b, _| {
            b.iter(|| {
                let bytes = encode_restart(black_box(&state)).expect("bench state must encode");
                decode_restart(black_box(&bytes))
            });
        });
    }
    group.finish();
}

criterion_group!(
    explicit_restart_bench,
    validate_restart_round_trip,
    bench_encode,
    bench_decode,
    bench_round_trip,
);
criterion_main!(explicit_restart_bench);
