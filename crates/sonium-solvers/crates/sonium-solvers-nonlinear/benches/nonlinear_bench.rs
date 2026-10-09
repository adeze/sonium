//! Nonlinear solver benchmarks: 1D Bratu problem vs PETSc SNES.
//!
//! `-u'' = λ·e^u` on (0,1) with `u(0) = u(1) = 0`, second-order finite
//! differences with `n` interior points, `λ = 1.0`. Two solver paths:
//!
//! * `newton-exact-thomas`: [`newton_raphson_inexact`] with an analytic
//!   tridiagonal Jacobian solved exactly in O(n) (Thomas). Isolates
//!   Newton-loop efficiency; inner-Krylov tuning is covered by the
//!   iterative crate's own benches.
//! * `anderson-picard`: [`anderson_acceleration`] on damped-Jacobi Picard
//!   fixed-point iteration for the same discretization.
//!
//! Each case prints one JSON line
//! `{variant, n, ms, newton_iters, linear_iters, residual, converged}` to
//! stdout; compare against `benches/snes_bratu.py` (petsc4py SNES on the
//! identical discretization) for the vendor table.
//!
//! Run:
//!   cargo bench -p sonium-solvers-nonlinear --bench nonlinear_bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_solvers_core::traits::LinearOperator;
use sonium_solvers_nonlinear::{
    AndersonConfig, InexactNewtonConfig, anderson_acceleration, newton_raphson_inexact,
};
use std::time::{Duration, Instant};

/// 1D Bratu residual with Dirichlet BCs folded in.
fn bratu_residual(x: &[f64], lambda: f64) -> Vec<f64> {
    let n = x.len();
    let h = 1.0 / (n as f64 + 1.0);
    let h2 = h * h;
    let mut f = vec![0.0; n];
    for i in 0..n {
        let xm = if i == 0 { 0.0 } else { x[i - 1] };
        let xp = if i + 1 == n { 0.0 } else { x[i + 1] };
        f[i] = (-xm + 2.0 * x[i] - xp) / h2 - lambda * x[i].exp();
    }
    f
}

/// Analytic tridiagonal Jacobian of [`bratu_residual`].
struct BratuJacobian {
    lower: Vec<f64>,
    diag: Vec<f64>,
    upper: Vec<f64>,
}

impl BratuJacobian {
    fn new(x: &[f64], lambda: f64) -> Self {
        let n = x.len();
        let h = 1.0 / (n as f64 + 1.0);
        let inv_h2 = 1.0 / (h * h);
        Self {
            lower: vec![-inv_h2; n.saturating_sub(1)],
            diag: x
                .iter()
                .map(|&xi| 2.0 * inv_h2 - lambda * xi.exp())
                .collect(),
            upper: vec![-inv_h2; n.saturating_sub(1)],
        }
    }
}

/// Exact O(n) Thomas solve for the tridiagonal Jacobian: the bench isolates
/// Newton-loop efficiency (same math on both sides of the vendor table),
/// not inner-Krylov tuning.
fn thomas_solve(jac: &BratuJacobian, rhs: &[f64]) -> Vec<f64> {
    let n = jac.diag.len();
    let mut c = vec![0.0; n];
    let mut d = rhs.to_vec();
    c[0] = jac.upper[0] / jac.diag[0];
    d[0] /= jac.diag[0];
    for i in 1..n {
        let m = 1.0 / (jac.diag[i] - jac.lower[i - 1] * c[i - 1]);
        if i + 1 < n {
            c[i] = jac.upper[i] * m;
        }
        d[i] = (d[i] - jac.lower[i - 1] * d[i - 1]) * m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    x
}

impl LinearOperator<f64> for BratuJacobian {
    fn num_rows(&self) -> usize {
        self.diag.len()
    }
    fn num_cols(&self) -> usize {
        self.diag.len()
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        let n = self.diag.len();
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        if n == 0 {
            return;
        }
        if n == 1 {
            y[0] = self.diag[0] * x[0];
            return;
        }
        y[0] = self.diag[0] * x[0] + self.upper[0] * x[1];
        for i in 1..n - 1 {
            y[i] = self.lower[i - 1] * x[i - 1] + self.diag[i] * x[i] + self.upper[i] * x[i + 1];
        }
        y[n - 1] = self.lower[n - 2] * x[n - 2] + self.diag[n - 1] * x[n - 1];
    }
    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        // Symmetric tridiagonal: transpose equals the operator.
        self.apply_into(x, y);
    }
}

struct NewtonRow {
    variant: &'static str,
    n: usize,
    ms: f64,
    newton_iters: usize,
    linear_iters: usize,
    residual: f64,
    converged: bool,
}

fn solve_newton(n: usize, lambda: f64) -> NewtonRow {
    let x0 = vec![0.0_f64; n];
    let config = InexactNewtonConfig {
        newton: sonium_solvers_nonlinear::NewtonConfig {
            // Production globalization (mirrors PETSc NEWTONLS): Armijo
            // backtracking so the bench compares solver efficiency, not
            // robustness setup.
            line_search: sonium_solvers_nonlinear::LineSearch::backtracking(),
            // rel 1e-6 (not 1e-8): at n = 32000 the O(1e9)-scaled Jacobian
            // puts the attainable rounding floor near 2.4e-6 absolute, and
            // both solvers stall there — verified independently with PETSc
            // SNES. A gate below the floor would benchmark rounding.
            relative_tolerance: 1e-6,
            ..Default::default()
        },
        ..Default::default()
    };
    let start = Instant::now();
    let sol = newton_raphson_inexact(
        &x0,
        |x| bratu_residual(x, lambda),
        |x, rhs, _eta| {
            // Exact O(n) inner solve: isolates Newton-loop efficiency (same
            // math on both sides of the vendor table). Inner-Krylov tuning
            // belongs to the iterative crate's own benches.
            Ok(thomas_solve(&BratuJacobian::new(x, lambda), rhs))
        },
        &config,
    )
    .expect("newton bench solve must succeed");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    NewtonRow {
        variant: "newton-exact-thomas",
        n,
        ms,
        newton_iters: sol.iterations,
        // One exact tridiagonal solve per Newton step.
        linear_iters: sol.iterations,
        residual: sol.residual,
        converged: sol.converged,
    }
}

/// Synthetic contractive fixed-point map for benchmarking the Anderson
/// machinery itself: `G_i(x) = ρ·x_i + (1−ρ)·φ_i(x)` with
/// `φ_i(x) = sin((x_{i−1} + x_{i+1})/4 + 0.01·i)/2` (periodic indices) and
/// `ρ = 0.95`. Cheap O(n) evals make framework overhead (history MGS-QR,
/// BLAS-1 throughput) visible instead of hiding it behind an expensive `G`;
/// Picard crawls (factor ≈ ρ) while Anderson(5) flies. Same map on both
/// sides of the vendor table.
struct ContractionMap {
    n: usize,
    rho: f64,
}

impl ContractionMap {
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut out = vec![0.0; n];
        for i in 0..n {
            let xm = x[(i + n - 1) % n];
            let xp = x[(i + 1) % n];
            let phi = ((xm + xp) * 0.25 + 0.01 * i as f64).sin() * 0.5;
            out[i] = self.rho * x[i] + (1.0 - self.rho) * phi;
        }
        out
    }
}

fn solve_anderson(n: usize) -> NewtonRow {
    let prob = ContractionMap { n, rho: 0.95 };
    let x0 = vec![0.0_f64; n];
    let config = AndersonConfig {
        depth: 5,
        max_iterations: 500,
        ..Default::default()
    };
    let evals = std::cell::Cell::new(0usize);
    let start = Instant::now();
    let sol = anderson_acceleration(
        &x0,
        |x: &[f64]| {
            evals.set(evals.get() + 1);
            prob.apply(x)
        },
        &config,
    )
    .expect("anderson bench solve must succeed");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    NewtonRow {
        variant: "anderson-contract",
        n,
        ms,
        newton_iters: sol.iterations,
        linear_iters: evals.get(),
        residual: sol.residual,
        converged: sol.converged,
    }
}

fn print_row(r: &NewtonRow) {
    println!(
        "{{\"variant\":\"{}\",\"n\":{},\"ms\":{:.3},\"newton_iters\":{},\"linear_iters\":{},\
         \"residual\":{:.3e},\"converged\":{}}}",
        r.variant, r.n, r.ms, r.newton_iters, r.linear_iters, r.residual, r.converged
    );
}

fn bench_newton(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear/bratu-newton");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);
    for n in [2000usize, 8000, 32000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let row = solve_newton(n, 1.0);
                    assert!(row.converged, "bench newton must converge");
                    total += Duration::from_secs_f64(row.ms / 1000.0);
                }
                total
            })
        });
        print_row(&solve_newton(n, 1.0));
    }
    group.finish();
}

fn bench_anderson(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear/contract-anderson");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);
    for n in [2000usize, 8000, 32000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let row = solve_anderson(n);
                    assert!(row.converged, "bench anderson must converge");
                    total += Duration::from_secs_f64(row.ms / 1000.0);
                }
                total
            })
        });
        print_row(&solve_anderson(n));
    }
    group.finish();
}

criterion_group!(nonlinear, bench_newton, bench_anderson);
criterion_main!(nonlinear);
