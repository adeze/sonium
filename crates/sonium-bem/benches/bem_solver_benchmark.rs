use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_bem::core::{BemProblem, BemSolver};
use sonium_wave::analytical::sphere_scattering_3d;
use std::f64::consts::PI;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Group 1: Analytical solution baseline
// ---------------------------------------------------------------------------
//
// Measures the cost of the Mie-series reference computation used in
// accuracy checks.  These should be sub-millisecond; any regression here
// indicates a change to the special-function math.

fn bench_analytical_solutions(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_analytical_solutions");

    // 37 theta points spaced uniformly on [0, π]
    let theta_37: Vec<f64> = (0..37).map(|i| PI * i as f64 / 36.0).collect();
    let theta_37_clone = theta_37.clone();

    group.bench_function("sphere_scattering_ka1_20terms", |b| {
        b.iter(|| {
            sphere_scattering_3d(
                1.0,       // wave_number (ka = 1 with radius = 1)
                1.0,       // radius
                20,        // num_terms
                vec![1.5], // r_points — just outside the surface
                theta_37.clone(),
            )
        })
    });

    group.bench_function("sphere_scattering_ka5_60terms", |b| {
        b.iter(|| sphere_scattering_3d(5.0, 1.0, 60, vec![1.5], theta_37_clone.clone()))
    });

    // 37 points reused for the high-frequency case
    let theta_37_hf: Vec<f64> = (0..37).map(|i| PI * i as f64 / 36.0).collect();
    group.bench_function("sphere_scattering_ka10_100terms", |b| {
        b.iter(|| sphere_scattering_3d(10.0, 1.0, 100, vec![1.5], theta_37_hf.clone()))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: Mesh generation
// ---------------------------------------------------------------------------
//
// Benchmarks Fibonacci-hull sphere construction at four resolutions.
// Element count ≈ 2 * n_theta * n_phi  (polar caps + interior quads).

fn bench_mesh_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_mesh_generation");

    // (n_theta, n_phi) pairs covering coarse → fine
    let configs: &[(usize, usize)] = &[(8, 16), (12, 24), (16, 32), (24, 48)];

    for &(n_theta, n_phi) in configs {
        group.bench_with_input(
            BenchmarkId::new("sphere_mesh", format!("{}x{}", n_theta, n_phi)),
            &(n_theta, n_phi),
            |b, &(nt, np)| {
                b.iter(|| {
                    // BemProblem::rigid_sphere_scattering_custom calls generate_sphere_mesh
                    // internally; using it here keeps us benchmarking the public API.
                    BemProblem::rigid_sphere_scattering_custom(
                        1.0,   // radius
                        100.0, // frequency (low → ka << 1, assembly is not the bottleneck)
                        343.0, 1.21, nt, np,
                    )
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: BEM system assembly  (O(N²) bottleneck)
// ---------------------------------------------------------------------------
//
// The assembly step dominates for moderate mesh sizes.  Because the
// assembly function is not separately exposed in the public API we time
// the full `solver.solve()` path and note that it includes both assembly
// and the direct LU solve.  For the sizes tested the LU factorization is
// O(N³) but with N ≤ 528 elements its constant is small relative to the
// dense O(N²) quadrature loop.
//
// Sizes:
//   coarse  4× 8 →  ~48 elements
//   medium  8×16 → ~224 elements
//   fine   12×24 → ~528 elements

fn bench_bem_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_bem_assembly");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Use a very low frequency so that ka << 1 and assembly dominates over
    // the iterative convergence cost.
    let frequency = 50.0; // Hz — ka ≈ 0.001 for r = 0.1 m

    let configs: &[(usize, usize, &str)] = &[(4, 8, "coarse"), (8, 16, "medium"), (12, 24, "fine")];

    let solver = BemSolver::new();

    for &(n_theta, n_phi, label) in configs {
        let problem =
            BemProblem::rigid_sphere_scattering_custom(1.0, frequency, 343.0, 1.21, n_theta, n_phi);

        group.bench_with_input(
            BenchmarkId::new("bem_assembly_plus_solve", label),
            &problem,
            |b, prob| b.iter(|| solver.solve(prob).unwrap()),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: End-to-end BEM solve
// ---------------------------------------------------------------------------
//
// Tracks total wall time for the full pipeline (mesh → assemble → solve)
// at physically meaningful ka values.

fn bench_bem_solve_total(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_bem_solve_total");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // (n_theta, n_phi, ka, frequency_hz)
    // ka = k * r = (2π f / c) * r  →  f = ka * c / (2π * r)  with r = 1 m
    let c_sound = 343.0_f64;
    let radius = 1.0_f64;

    let configs: &[(usize, usize, f64)] = &[
        (4, 8, 0.5),   // small  — expected ~5 ms
        (8, 16, 1.0),  // medium — expected ~50 ms
        (12, 24, 1.0), // large  — expected ~500 ms
    ];

    let solver = BemSolver::new();

    for &(n_theta, n_phi, ka) in configs {
        let frequency = ka * c_sound / (2.0 * PI * radius);
        let label = format!("{}x{}_ka{}", n_theta, n_phi, ka);

        let problem = BemProblem::rigid_sphere_scattering_custom(
            radius, frequency, c_sound, 1.21, n_theta, n_phi,
        );

        group.bench_with_input(
            BenchmarkId::new("bem_solve", &label),
            &problem,
            |b, prob| b.iter(|| solver.solve(prob).unwrap()),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 5: Field evaluation (post-processing)
// ---------------------------------------------------------------------------
//
// Solves a fixed small problem once in setup, then measures the cost of
// evaluating the scattered + incident pressure at N exterior points.

fn bench_field_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_field_evaluation");

    // Solve once outside the benchmark loop
    let problem = BemProblem::rigid_sphere_scattering_custom(1.0, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new();
    let solution = solver.solve(&problem).expect("setup solve failed");

    let point_counts: &[usize] = &[10, 100, 1000];

    for &n_pts in point_counts {
        let mut points: Vec<[f64; 3]> = vec![[0.0; 3]; n_pts];
        for i in 0..n_pts {
            let r = 1.5 + 3.5 * i as f64 / (n_pts.max(1) - 1).max(1) as f64;
            points[i][2] = r;
        }

        group.bench_with_input(
            BenchmarkId::new("field_eval_points", n_pts),
            &points,
            |b, pts| b.iter(|| solution.evaluate_pressure_field(pts)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(analytical_benches, bench_analytical_solutions);
criterion_group!(mesh_benches, bench_mesh_generation);
criterion_group!(
    name = assembly_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(30));
    targets = bench_bem_assembly
);
criterion_group!(
    name = solve_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(30));
    targets = bench_bem_solve_total
);
criterion_group!(field_benches, bench_field_evaluation);

criterion_main!(
    analytical_benches,
    mesh_benches,
    assembly_benches,
    solve_benches,
    field_benches
);
