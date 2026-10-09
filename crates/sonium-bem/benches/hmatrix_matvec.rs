//! H-matrix matvec benchmarks: forward vs transpose vs Hermitian.
//!
//! The transpose and Hermitian paths run over the field-cluster partition
//! schedule built at H-matrix construction time; this bench tracks their
//! cost relative to the forward (row-partition) matvec. Transpose should
//! stay within ~2x of forward.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex64;
use sonium_bem::core::assembly::hmatrix::{HMatrix, build_hmatrix};
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::mesh::octree::Octree;
use sonium_bem::core::types::{BoundaryCondition, Element, PhysicsParams};
use sonium_bem::room_acoustics::build_clusters_from_octree;

fn velocity_mesh(subdivisions: usize) -> (Vec<Element>, Vec<[f64; 3]>) {
    let mesh = generate_icosphere_mesh(1.0, subdivisions);
    let mut elements = mesh.elements;
    for (dof, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![dof];
    }
    (elements, mesh.nodes)
}

fn build_fixture(subdivisions: usize) -> HMatrix {
    let (elements, nodes) = velocity_mesh(subdivisions);
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let mut octree = Octree::build(&centers, 32, 8);
    octree.compute_interaction_lists(2.0);
    let clusters = build_clusters_from_octree(&octree, &elements);
    let physics = PhysicsParams::new(500.0, 343.0, 1.21, false);
    build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-8, 32)
}

fn bench_hmatrix_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmatrix_matvec");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    // Icosphere subdivisions 3/4 -> 1280 / 5120 DOFs.
    for subdivisions in [3, 4] {
        let hmat = build_fixture(subdivisions);
        let n = hmat.num_dofs;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((0.37 * (i + 1) as f64).sin(), (0.19 * (i + 1) as f64).cos()))
            .collect();
        let mut y = vec![Complex64::new(0.0, 0.0); n];

        group.bench_with_input(BenchmarkId::new("forward", n), &n, |b, _| {
            b.iter(|| hmat.matvec_into(&x, &mut y));
        });
        group.bench_with_input(BenchmarkId::new("transpose", n), &n, |b, _| {
            b.iter(|| hmat.matvec_transpose_into(&x, &mut y));
        });
        group.bench_with_input(BenchmarkId::new("hermitian", n), &n, |b, _| {
            b.iter(|| hmat.matvec_hermitian_into(&x, &mut y));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hmatrix_matvec);
criterion_main!(benches);
