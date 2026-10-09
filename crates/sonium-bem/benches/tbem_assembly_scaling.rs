use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex64;
use rayon::{ThreadPool, ThreadPoolBuilder};
use sonium_bem::core::assembly::tbem::build_tbem_system_with_beta_parallel;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, Element, PhysicsParams};

fn velocity_mesh(subdivisions: usize) -> (Vec<Element>, Vec<[f64; 3]>) {
    let mesh = generate_icosphere_mesh(1.0, subdivisions);
    let mut elements = mesh.elements;
    for (dof, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![dof];
    }
    (elements, mesh.nodes)
}

fn pool(threads: usize) -> ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn bench_tbem_assembly_scaling(c: &mut Criterion) {
    let parallel_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let serial_pool = pool(1);
    let parallel_pool = pool(parallel_threads);
    let physics = PhysicsParams::new(500.0, 343.0, 1.21, false);
    let beta = physics.burton_miller_beta();

    let mut group = c.benchmark_group("tbem_assembly_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    for subdivisions in [0, 1, 2] {
        let (elements, nodes) = velocity_mesh(subdivisions);
        let element_count = elements.len();
        group.throughput(Throughput::Elements((element_count * element_count) as u64));

        for (label, thread_pool) in [("1-thread", &serial_pool), ("all-threads", &parallel_pool)] {
            group.bench_with_input(
                BenchmarkId::new(label, element_count),
                &element_count,
                |b, _| {
                    b.iter(|| {
                        thread_pool.install(|| {
                            build_tbem_system_with_beta_parallel(&elements, &nodes, &physics, beta)
                        })
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_tbem_assembly_scaling);
criterion_main!(benches);
