//! Optimisation benchmarks: extended Rosenbrock vs SciPy L-BFGS-B.
//!
//! `f(x) = Σ_{i even} [(1−x_i)² + 100·(x_{i+1} − x_i²)²]`, `n` even, start
//! `(−1.2, 1)` per pair. Variant `lbfgs-rosen` runs [`lbfgs`] with defaults
//! (memory 10, Strong Wolfe, grad tol 1e-6); compare against
//! `benches/scipy_rosen.py` (SciPy `L-BFGS-B`, analytic Jacobian, gtol 1e-6)
//! for the vendor table.
//!
//! Each case prints one JSON line
//! `{variant, n, ms, iters, fun_evals, grad_evals, fun, gnorm}` to stdout.
//!
//! Run:
//!   cargo bench -p sonium-solvers-optimization --bench optim_bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_solvers_optimization::unconstrained::de::differential_evolution::differential_evolution;
use sonium_solvers_optimization::{DEConfigBuilder, LbfgsConfig, lbfgs};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for pair in x.as_chunks::<2>().0 {
        let (xi, xj) = (pair[0], pair[1]);
        let a = 1.0 - xi;
        let b = xj - xi * xi;
        s += a * a + 100.0 * b * b;
    }
    s
}

fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let mut g = vec![0.0; x.len()];
    for (k, pair) in x.as_chunks::<2>().0.iter().enumerate() {
        let (xi, xj) = (pair[0], pair[1]);
        g[2 * k] = -2.0 * (1.0 - xi) - 400.0 * xi * (xj - xi * xi);
        g[2 * k + 1] = 200.0 * (xj - xi * xi);
    }
    g
}

fn rosenbrock_start(n: usize) -> Vec<f64> {
    let mut x0 = vec![0.0; n];
    for (k, pair) in x0.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let _ = k;
        pair[0] = -1.2;
        pair[1] = 1.0;
    }
    x0
}

fn gnorm_inf(g: &[f64]) -> f64 {
    g.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

struct OptimRow {
    variant: &'static str,
    n: usize,
    ms: f64,
    iters: usize,
    fun_evals: usize,
    grad_evals: usize,
    fun: f64,
    gnorm: f64,
}

fn solve_lbfgs(n: usize) -> OptimRow {
    let x0 = rosenbrock_start(n);
    let config = LbfgsConfig::<f64>::default();
    let start = Instant::now();
    let sol = lbfgs(&rosenbrock, &rosenbrock_grad, &x0, &config).expect("bench must converge");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let gnorm = sol.grad.as_ref().map(|g| gnorm_inf(g)).unwrap_or(f64::NAN);
    OptimRow {
        variant: "lbfgs-rosen",
        n,
        ms,
        iters: sol.iterations,
        fun_evals: sol.fun_evals,
        grad_evals: sol.grad_evals,
        fun: sol.fun,
        gnorm,
    }
}

fn print_row(r: &OptimRow) {
    println!(
        "{{\"variant\":\"{}\",\"n\":{},\"ms\":{:.3},\"iters\":{},\"fun_evals\":{},\
         \"grad_evals\":{},\"fun\":{:.3e},\"gnorm\":{:.3e}}}",
        r.variant, r.n, r.ms, r.iters, r.fun_evals, r.grad_evals, r.fun, r.gnorm
    );
}

fn bench_lbfgs(c: &mut Criterion) {
    let mut group = c.benchmark_group("optim/lbfgs-rosen");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);
    for n in [100usize, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let row = solve_lbfgs(n);
                    assert!(row.gnorm < 1e-4, "bench lbfgs must converge");
                    total += Duration::from_secs_f64(row.ms / 1000.0);
                }
                total
            })
        });
        print_row(&solve_lbfgs(n));
    }
    group.finish();
}

fn bench_de_sphere(c: &mut Criterion) {
    let dim = 30;
    let bounds = vec![(-5.0, 5.0); dim];
    let objective = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum::<f64>();

    c.bench_function("de_sphere_30d", |b| {
        b.iter(|| {
            differential_evolution(
                black_box(&objective),
                black_box(&bounds),
                // DEConfig is not Clone; move a fresh copy each iteration via builder.
                DEConfigBuilder::new()
                    .maxiter(100)
                    .seed(42)
                    .build()
                    .unwrap(),
            )
            .unwrap()
        })
    });
}

criterion_group!(optim, bench_lbfgs, bench_de_sphere);
criterion_main!(optim);
