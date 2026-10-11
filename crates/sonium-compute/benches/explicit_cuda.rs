//! CUDA vs CPU explicit-kernel admission comparison (Phase 1 lane 1E).
//!
//! Steady-state throughput for device [`CudaExplicitKernels`] launches
//! against the serial CPU reference kernels over the same synthetic hex8
//! chain mesh lane 1A uses (element `e` owns nodes `(e + k) % N`, so every
//! node is shared by 8 elements). Each group benches the CPU kernel, the
//! CUDA kernel with pre-staged device buffers (launch + stream sync only,
//! the `blas1_compare` precedent for resident data), and a `cuda_e2e`
//! end-to-end offload (validate + pool-allocate + upload + launch + sync +
//! fetch per iteration) at two sizes to inform the admission note.
//!
//! Sweeps `N in {256, 4_096, 65_536, 262_144, 1_048_576}` nodes with
//! `E = N` elements, from dispatch-dominated to DRAM-bound. CUDA benches
//! skip gracefully when no driver is present; CPU benches always run.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p sonium-compute --features cuda --bench explicit_cuda
//! # filter to one kernel:
//! cargo bench -p sonium-compute --features cuda --bench explicit_cuda -- scatter
//! ```
//!
//! [`CudaExplicitKernels`]: sonium_compute::cuda::explicit::CudaExplicitKernels

// Rust guideline compliant 2026-02-21

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use sonium_compute::cuda::explicit::{
    CudaExplicitKernels, cast_indices_i32, finish_min_dt, min_dt_block_count, scatter_csr,
};
use sonium_compute::explicit_dt::critical_dt;
use sonium_compute::explicit_hourglass::{HEXA8_HOURGLASS_BASE, hexa8_hourglass_forces};
use sonium_compute::explicit_scatter::{
    ElementTopology, assemble_lumped_mass, gather_nodal, scatter_add_nodal,
};
use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};

/// Node counts swept by every benchmark group. Chosen to span:
///   256        — dispatch-dominated
///   4 096      — comfortably L2
///   65 536     — last-level-cache territory
///   262 144    — DRAM-bound (multi-MiB fields)
///   1 048 576  — large offload (tens of MiB per field)
const SIZES: &[usize] = &[256, 4_096, 65_536, 262_144, 1_048_576];

/// Sizes Runner for the per-iteration end-to-end offload variant.
const E2E_SIZES: &[usize] = &[4_096, 262_144];

/// Nodes per synthetic element (hex8 chain).
const NODES_PER_ELEMENT: usize = 8;

/// Device quota for staged bench buffers (1 GiB; the 1M-node scatter stages
/// ~230 MiB across values, indices, CSR, and target at once).
const POOL_CAP_BYTES: usize = 1 << 30;

// ---------------------------------------------------------------------
// Input generators (deterministic so CPU and CUDA see the same data)
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

/// Deterministic strictly positive ramp (lengths, speeds, volumes).
fn positive_ramp(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as f64) * 0.0177).sin() + 2.0)
        .collect()
}

/// CUDA stack shared by one benchmark group: pool plus loaded kernels.
struct CudaStack {
    pool: MemoryPool<CudaBackend>,
    kernels: CudaExplicitKernels,
}

/// Build the CUDA stack, or `None` when no driver is present.
fn cuda_stack() -> Option<CudaStack> {
    let backend = CudaBackend::new_default().ok()?;
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(POOL_CAP_BYTES)).ok()?;
    let kernels = CudaExplicitKernels::load(pool.backend()).ok()?;
    Some(CudaStack { pool, kernels })
}

/// Prove device-vs-CPU parity once before Criterion starts timing.
///
/// A benchmark result is not admissible admission evidence unless the device
/// kernels in this executable first reproduce the CPU references bit-exactly
/// on an odd-sized mesh. Skips (with a notice) when no driver is present.
fn validate_cuda_parity(c: &mut Criterion) {
    let Some(stack) = cuda_stack() else {
        eprintln!("CUDA unavailable, skipping explicit_cuda parity validation");
        c.bench_function("validate_cuda_parity_skipped", |b| {
            b.iter(|| black_box(1 + 1))
        });
        return;
    };
    let nodes = 37;
    let (connectivity, offsets) = chain_hex_mesh(nodes, 64);
    // Gather and scatter at scalar and vector widths.
    for width in [1, 3] {
        let field = ramp(width * nodes);
        let device = stack
            .kernels
            .gather_nodal_device(&stack.pool, &field, &connectivity, width)
            .expect("parity gather");
        let mut expected = vec![0.0; width * connectivity.len()];
        gather_nodal(&field, &connectivity, &mut expected);
        assert_eq!(device, expected, "gather parity failed at width {width}");

        let values = ramp(width * connectivity.len());
        let mut expected = ramp(width * nodes);
        scatter_add_nodal(&mut expected, &connectivity, &values);
        let mut device = ramp(width * nodes);
        stack
            .kernels
            .scatter_add_nodal_device(&stack.pool, &mut device, &connectivity, &values, width)
            .expect("parity scatter");
        assert_eq!(device, expected, "scatter parity failed at width {width}");
    }
    // Critical dt over single-block and multi-block shapes.
    for elements in [1, 1000] {
        let lengths = positive_ramp(elements);
        let speeds: Vec<f64> = positive_ramp(elements)
            .iter()
            .map(|speed| speed * 1000.0)
            .collect();
        let device = stack
            .kernels
            .critical_dt_device(&stack.pool, &lengths, &speeds)
            .expect("parity dt");
        assert_eq!(device, critical_dt(&lengths, &speeds));
    }
    // Batched hourglass forces over two elements.
    let base: Vec<f64> = HEXA8_HOURGLASS_BASE.iter().flatten().copied().collect();
    let gamma: Vec<f64> = base.iter().copied().chain(base.iter().copied()).collect();
    let rates = ramp(48);
    let mut expected = vec![0.0; 48];
    for element in 0..2 {
        hexa8_hourglass_forces(
            &gamma[32 * element..32 * element + 32],
            &rates[24 * element..24 * element + 24],
            0.1,
            &mut expected[24 * element..24 * element + 24],
        );
    }
    let mut device = vec![0.0; 48];
    stack
        .kernels
        .hexa8_hourglass_forces_device(&stack.pool, &gamma, &rates, 0.1, &mut device)
        .expect("parity hourglass");
    assert_eq!(device, expected);
    // Lumped-mass assembly onto nonzero initial masses.
    let topology = ElementTopology::new(&connectivity, &offsets);
    let volumes = positive_ramp(64);
    let densities = positive_ramp(64);
    let mut expected = positive_ramp(nodes);
    assemble_lumped_mass(&mut expected, topology, &volumes, &densities);
    let mut device = positive_ramp(nodes);
    stack
        .kernels
        .assemble_lumped_mass_device(
            &stack.pool,
            &mut device,
            &connectivity,
            &offsets,
            &volumes,
            &densities,
        )
        .expect("parity lumped mass");
    assert_eq!(device, expected);
    println!("sonium_precision_check bench=explicit_cuda status=device_cpu_bit_exact");
    c.bench_function("validate_cuda_parity", |b| b.iter(|| black_box(1 + 1)));
}

fn bench_gather(c: &mut Criterion) {
    let stack = cuda_stack();
    if stack.is_none() {
        eprintln!("CUDA unavailable, skipping cuda gather benches");
    }
    let mut group = c.benchmark_group("gather_cuda");
    for &nodes in SIZES {
        let field = ramp(3 * nodes);
        let (connectivity, _) = chain_hex_mesh(nodes, nodes);
        let mut out = vec![0.0; 3 * connectivity.len()];
        group.bench_with_input(BenchmarkId::new("cpu", nodes), &nodes, |b, _| {
            b.iter(|| {
                gather_nodal(
                    black_box(&field),
                    black_box(&connectivity),
                    black_box(&mut out),
                );
            });
        });
        if let Some(stack) = &stack {
            let device_ids =
                cast_indices_i32(&connectivity, nodes).expect("bench indices in range");
            let mut field_dev = stack.pool.try_alloc::<f64>(field.len()).expect("field");
            let mut ids_dev = stack.pool.try_alloc::<i32>(device_ids.len()).expect("ids");
            let mut out_dev = stack.pool.try_alloc::<f64>(out.len()).expect("out");
            stack
                .pool
                .backend()
                .try_copy_h2d(&field, field_dev.raw_mut())
                .expect("upload");
            stack
                .pool
                .backend()
                .try_copy_h2d(&device_ids, ids_dev.raw_mut())
                .expect("upload");
            group.bench_with_input(BenchmarkId::new("cuda", nodes), &nodes, |b, _| {
                b.iter(|| {
                    stack
                        .kernels
                        .try_gather(
                            stack.pool.backend(),
                            black_box(field_dev.raw()),
                            black_box(ids_dev.raw()),
                            black_box(out_dev.raw_mut()),
                            black_box(3),
                        )
                        .expect("gather launch");
                    stack.pool.backend().synchronize().expect("sync");
                });
            });
            if E2E_SIZES.contains(&nodes) {
                group.bench_with_input(BenchmarkId::new("cuda_e2e", nodes), &nodes, |b, _| {
                    b.iter(|| {
                        let out = stack
                            .kernels
                            .gather_nodal_device(
                                black_box(&stack.pool),
                                black_box(&field),
                                black_box(&connectivity),
                                black_box(3),
                            )
                            .expect("e2e gather");
                        black_box(out);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_scatter(c: &mut Criterion) {
    let stack = cuda_stack();
    if stack.is_none() {
        eprintln!("CUDA unavailable, skipping cuda scatter benches");
    }
    let mut group = c.benchmark_group("scatter_cuda");
    for &nodes in SIZES {
        let (connectivity, _) = chain_hex_mesh(nodes, nodes);
        let values = ramp(3 * connectivity.len());
        let mut target = vec![0.0; 3 * nodes];
        group.bench_with_input(BenchmarkId::new("cpu", nodes), &nodes, |b, _| {
            b.iter(|| {
                scatter_add_nodal(
                    black_box(&mut target),
                    black_box(&connectivity),
                    black_box(&values),
                );
            });
        });
        if let Some(stack) = &stack {
            let (offsets, slots) = scatter_csr(nodes, &connectivity).expect("bench CSR builds");
            let mut values_dev = stack.pool.try_alloc::<f64>(values.len()).expect("values");
            let mut slots_dev = stack.pool.try_alloc::<i32>(slots.len()).expect("slots");
            let mut offsets_dev = stack.pool.try_alloc::<i32>(offsets.len()).expect("offsets");
            let mut target_dev = stack.pool.try_alloc::<f64>(3 * nodes).expect("target");
            stack
                .pool
                .backend()
                .try_copy_h2d(&values, values_dev.raw_mut())
                .expect("upload");
            stack
                .pool
                .backend()
                .try_copy_h2d(&slots, slots_dev.raw_mut())
                .expect("upload");
            stack
                .pool
                .backend()
                .try_copy_h2d(&offsets, offsets_dev.raw_mut())
                .expect("upload");
            stack
                .pool
                .backend()
                .try_copy_h2d(&target, target_dev.raw_mut())
                .expect("upload");
            group.bench_with_input(BenchmarkId::new("cuda", nodes), &nodes, |b, _| {
                b.iter(|| {
                    stack
                        .kernels
                        .try_scatter_add(
                            stack.pool.backend(),
                            black_box(values_dev.raw()),
                            black_box(slots_dev.raw()),
                            black_box(offsets_dev.raw()),
                            black_box(target_dev.raw_mut()),
                            black_box(3),
                        )
                        .expect("scatter launch");
                    stack.pool.backend().synchronize().expect("sync");
                });
            });
            if E2E_SIZES.contains(&nodes) {
                let mut e2e_target = vec![0.0; 3 * nodes];
                group.bench_with_input(BenchmarkId::new("cuda_e2e", nodes), &nodes, |b, _| {
                    b.iter(|| {
                        stack
                            .kernels
                            .scatter_add_nodal_device(
                                black_box(&stack.pool),
                                black_box(&mut e2e_target),
                                black_box(&connectivity),
                                black_box(&values),
                                black_box(3),
                            )
                            .expect("e2e scatter");
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_min_dt(c: &mut Criterion) {
    let stack = cuda_stack();
    if stack.is_none() {
        eprintln!("CUDA unavailable, skipping cuda min-dt benches");
    }
    let mut group = c.benchmark_group("min_dt_cuda");
    for &nodes in SIZES {
        let lengths = positive_ramp(nodes);
        let speeds: Vec<f64> = positive_ramp(nodes)
            .iter()
            .map(|speed| speed * 1000.0)
            .collect();
        group.bench_with_input(BenchmarkId::new("cpu", nodes), &nodes, |b, _| {
            b.iter(|| black_box(critical_dt(black_box(&lengths), black_box(&speeds))));
        });
        if let Some(stack) = &stack {
            let blocks = min_dt_block_count(nodes);
            let mut lengths_dev = stack.pool.try_alloc::<f64>(lengths.len()).expect("lengths");
            let mut speeds_dev = stack.pool.try_alloc::<f64>(speeds.len()).expect("speeds");
            let mut partials_dev = stack.pool.try_alloc::<f64>(blocks).expect("partials");
            stack
                .pool
                .backend()
                .try_copy_h2d(&lengths, lengths_dev.raw_mut())
                .expect("upload");
            stack
                .pool
                .backend()
                .try_copy_h2d(&speeds, speeds_dev.raw_mut())
                .expect("upload");
            let mut partials = vec![0.0; blocks];
            group.bench_with_input(BenchmarkId::new("cuda", nodes), &nodes, |b, _| {
                b.iter(|| {
                    stack
                        .kernels
                        .try_min_dt_partials(
                            stack.pool.backend(),
                            black_box(lengths_dev.raw()),
                            black_box(speeds_dev.raw()),
                            black_box(partials_dev.raw_mut()),
                        )
                        .expect("min-dt launch");
                    stack.pool.backend().synchronize().expect("sync");
                    stack
                        .pool
                        .backend()
                        .try_copy_d2h(partials_dev.raw(), black_box(&mut partials))
                        .expect("fetch");
                    black_box(finish_min_dt(&partials));
                });
            });
            if E2E_SIZES.contains(&nodes) {
                group.bench_with_input(BenchmarkId::new("cuda_e2e", nodes), &nodes, |b, _| {
                    b.iter(|| {
                        let dt = stack
                            .kernels
                            .critical_dt_device(
                                black_box(&stack.pool),
                                black_box(&lengths),
                                black_box(&speeds),
                            )
                            .expect("e2e dt");
                        black_box(dt);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_lumped_mass(c: &mut Criterion) {
    let stack = cuda_stack();
    if stack.is_none() {
        eprintln!("CUDA unavailable, skipping cuda lumped-mass benches");
    }
    let mut group = c.benchmark_group("lumped_mass_cuda");
    for &nodes in SIZES {
        let (connectivity, offsets) = chain_hex_mesh(nodes, nodes);
        let topology = ElementTopology::new(&connectivity, &offsets);
        let volumes = positive_ramp(nodes);
        let densities = positive_ramp(nodes);
        let mut masses = vec![0.0; nodes];
        group.bench_with_input(BenchmarkId::new("cpu", nodes), &nodes, |b, _| {
            b.iter(|| {
                assemble_lumped_mass(
                    black_box(&mut masses),
                    black_box(topology),
                    black_box(&volumes),
                    black_box(&densities),
                );
            });
        });
        // No pre-staged variant: the device path reuses scatter-add after a
        // host-side share expansion, so end-to-end is the honest comparison.
        if let Some(stack) = &stack
            && E2E_SIZES.contains(&nodes)
        {
            let mut e2e_masses = vec![0.0; nodes];
            group.bench_with_input(BenchmarkId::new("cuda_e2e", nodes), &nodes, |b, _| {
                b.iter(|| {
                    stack
                        .kernels
                        .assemble_lumped_mass_device(
                            black_box(&stack.pool),
                            black_box(&mut e2e_masses),
                            black_box(&connectivity),
                            black_box(&offsets),
                            black_box(&volumes),
                            black_box(&densities),
                        )
                        .expect("e2e lumped mass");
                });
            });
        }
    }
    group.finish();
}

fn bench_hourglass(c: &mut Criterion) {
    let stack = cuda_stack();
    if stack.is_none() {
        eprintln!("CUDA unavailable, skipping cuda hourglass benches");
    }
    let mut group = c.benchmark_group("hourglass_cuda");
    let base: Vec<f64> = HEXA8_HOURGLASS_BASE.iter().flatten().copied().collect();
    for &elements in SIZES {
        let mut gamma = Vec::with_capacity(32 * elements);
        for element in 0..elements {
            let scale = 1.0 + 0.01 * element as f64;
            gamma.extend(base.iter().map(|value| value * scale));
        }
        let rates = ramp(24 * elements);
        let mut forces = vec![0.0; 24 * elements];
        group.bench_with_input(BenchmarkId::new("cpu", elements), &elements, |b, _| {
            b.iter(|| {
                let gamma = black_box(&gamma);
                let rates = black_box(&rates);
                let forces = black_box(&mut forces);
                for element in 0..elements {
                    hexa8_hourglass_forces(
                        &gamma[32 * element..32 * element + 32],
                        &rates[24 * element..24 * element + 24],
                        0.1,
                        &mut forces[24 * element..24 * element + 24],
                    );
                }
            });
        });
        if let Some(stack) = &stack {
            let mut gamma_dev = stack.pool.try_alloc::<f64>(gamma.len()).expect("gamma");
            let mut rates_dev = stack.pool.try_alloc::<f64>(rates.len()).expect("rates");
            let mut forces_dev = stack.pool.try_alloc::<f64>(forces.len()).expect("forces");
            stack
                .pool
                .backend()
                .try_copy_h2d(&gamma, gamma_dev.raw_mut())
                .expect("upload");
            stack
                .pool
                .backend()
                .try_copy_h2d(&rates, rates_dev.raw_mut())
                .expect("upload");
            group.bench_with_input(BenchmarkId::new("cuda", elements), &elements, |b, _| {
                b.iter(|| {
                    stack
                        .kernels
                        .try_hourglass_forces(
                            stack.pool.backend(),
                            black_box(gamma_dev.raw()),
                            black_box(rates_dev.raw()),
                            black_box(forces_dev.raw_mut()),
                            black_box(0.1),
                            black_box(elements),
                        )
                        .expect("hourglass launch");
                    stack.pool.backend().synchronize().expect("sync");
                });
            });
            if E2E_SIZES.contains(&elements) {
                let mut e2e_forces = vec![0.0; 24 * elements];
                group.bench_with_input(
                    BenchmarkId::new("cuda_e2e", elements),
                    &elements,
                    |b, _| {
                        b.iter(|| {
                            stack
                                .kernels
                                .hexa8_hourglass_forces_device(
                                    black_box(&stack.pool),
                                    black_box(&gamma),
                                    black_box(&rates),
                                    black_box(0.1),
                                    black_box(&mut e2e_forces),
                                )
                                .expect("e2e hourglass");
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(
    explicit_cuda_bench,
    validate_cuda_parity,
    bench_gather,
    bench_scatter,
    bench_min_dt,
    bench_lumped_mass,
    bench_hourglass,
);
criterion_main!(explicit_cuda_bench);
