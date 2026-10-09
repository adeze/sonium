use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_mesher::mesh::refinement::{
    coarsen_elements_checked, refine_elements_checked, refine_elements_with_history_checked,
};
use sonium_mesher::mesh::tetrahedralize::{
    TetrahedralizeConfig, tetrahedralize_points, tetrahedralize_surface,
};
use sonium_mesher::mesh::types::Point;
use sonium_mesher::mesh::unit_square_triangles;
use sonium_mesher::partition::{PartitionOptions, try_build_element_graph, try_partition_mesh};
use std::hint::black_box;

const POINT_SIZES: &[usize] = &[64, 256, 1024];

fn bench_config(preserve_surface: bool) -> TetrahedralizeConfig {
    TetrahedralizeConfig {
        max_smooth_iterations: 0,
        add_interior_points: false,
        lloyd_iterations: 0,
        preserve_surface,
        max_ratio: 1.5,
    }
}

fn point_cloud(n: usize) -> Vec<Point> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    };

    let mut points = Vec::with_capacity(n.max(8));
    points.extend([
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(1.0, 1.0, 0.0),
        Point::new_3d(1.0, 0.0, 1.0),
        Point::new_3d(0.0, 1.0, 1.0),
        Point::new_3d(1.0, 1.0, 1.0),
    ]);
    while points.len() < n {
        points.push(Point::new_3d(next(), next(), next()));
    }
    points
}

fn cube_surface() -> (Vec<Point>, Vec<[usize; 3]>) {
    let points = vec![
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(1.0, 1.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(1.0, 0.0, 1.0),
        Point::new_3d(1.0, 1.0, 1.0),
        Point::new_3d(0.0, 1.0, 1.0),
    ];
    let facets = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    (points, facets)
}

fn refinement_fixture() -> sonium_mesher::mesh::types::Mesh {
    // Use the deterministic structured fixture for checked-mutation baselines.
    // Random Delaunay point clouds deliberately include poor-quality cells and
    // therefore conflate recovery/quality failures with refinement throughput.
    unit_square_triangles(8)
}

fn bench_tetrahedralize(c: &mut Criterion) {
    let mut group = c.benchmark_group("tetrahedralize_points");
    let config = bench_config(false);
    for &n in POINT_SIZES {
        let points = point_cloud(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &points,
            |bencher, points| {
                bencher.iter(|| {
                    black_box(tetrahedralize_points(black_box(points), black_box(&config)))
                });
            },
        );
    }
    group.finish();

    let mut constrained = c.benchmark_group("tetrahedralize_surface");
    let config = bench_config(true);
    let (points, facets) = cube_surface();
    constrained.throughput(Throughput::Elements(facets.len() as u64));
    constrained.bench_function("constrained_cube", |bencher| {
        bencher.iter(|| {
            black_box(tetrahedralize_surface(
                black_box(&points),
                black_box(&facets),
                None,
                black_box(&config),
            ))
        });
    });
    constrained.finish();

    let base_mesh = refinement_fixture();
    let selection: Vec<usize> = (0..base_mesh.num_elements().min(16)).collect();
    assert!(
        !selection.is_empty(),
        "benchmark fixture must contain elements"
    );
    let mut refinement = c.benchmark_group("checked_refinement");
    refinement.throughput(Throughput::Elements(selection.len() as u64));
    refinement.bench_function("first_order_selected_elements", |bencher| {
        bencher.iter_batched(
            || base_mesh.clone(),
            |mut mesh| {
                let result = refine_elements_checked(&mut mesh, &selection)
                    .expect("benchmark refinement fixture must remain valid");
                black_box((mesh, result))
            },
            BatchSize::SmallInput,
        );
    });
    refinement.finish();

    let mut refined_mesh = base_mesh.clone();
    let (_, history) = refine_elements_with_history_checked(&mut refined_mesh, &selection)
        .expect("benchmark refinement history fixture must remain valid");
    let parent_ids: Vec<usize> = history
        .parents
        .iter()
        .map(|parent| parent.element.id)
        .collect();
    let mut coarsening = c.benchmark_group("checked_coarsening");
    coarsening.throughput(Throughput::Elements(parent_ids.len() as u64));
    coarsening.bench_function("first_order_sibling_families", |bencher| {
        bencher.iter_batched(
            || refined_mesh.clone(),
            |mut mesh| {
                let result = coarsen_elements_checked(&mut mesh, &history, &parent_ids)
                    .expect("benchmark coarsening fixture must remain valid");
                black_box((mesh, result))
            },
            BatchSize::SmallInput,
        );
    });
    coarsening.finish();

    let partition_mesh = unit_square_triangles(32);
    let partition_options = PartitionOptions {
        num_partitions: 4,
        refinement: 5,
        ..PartitionOptions::default()
    };
    let mut partitioning = c.benchmark_group("checked_partitioning");
    partitioning.throughput(Throughput::Elements(partition_mesh.num_elements() as u64));
    partitioning.bench_function("element_graph", |bencher| {
        bencher.iter(|| {
            black_box(
                try_build_element_graph(black_box(&partition_mesh))
                    .expect("benchmark partition fixture must remain valid"),
            )
        });
    });
    partitioning.bench_function("four_way_partition", |bencher| {
        bencher.iter(|| {
            black_box(
                try_partition_mesh(black_box(&partition_mesh), black_box(&partition_options))
                    .expect("benchmark partition fixture must remain valid"),
            )
        });
    });
    partitioning.finish();

    let partition_graph = try_build_element_graph(&partition_mesh)
        .expect("benchmark partition graph fixture must remain valid");
    let partition = try_partition_mesh(&partition_mesh, &partition_options)
        .expect("benchmark partition fixture must remain valid");
    let mut ghosts = c.benchmark_group("checked_partition_ghost_layout");
    ghosts.throughput(Throughput::Elements(partition_mesh.num_elements() as u64));
    ghosts.bench_function("four_way_all_partitions", |bencher| {
        bencher.iter(|| {
            let layouts = (0..partition_options.num_partitions)
                .map(|partition_id| {
                    partition
                        .ghost_layout(&partition_mesh, &partition_graph, partition_id)
                        .expect("benchmark ghost-layout fixture must remain valid")
                })
                .collect::<Vec<_>>();
            black_box(layouts)
        });
    });
    ghosts.finish();
}

criterion_group!(benches, bench_tetrahedralize);
criterion_main!(benches);
