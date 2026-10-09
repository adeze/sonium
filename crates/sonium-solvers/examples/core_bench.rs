use sonium_solvers::*;
use std::time::Instant;

fn poisson_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut triplets = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            triplets.push((row, row, 4.0));
            if i > 0 {
                triplets.push((row, (i - 1) * n + j, -1.0));
            }
            if i + 1 < n {
                triplets.push((row, (i + 1) * n + j, -1.0));
            }
            if j > 0 {
                triplets.push((row, i * n + j - 1, -1.0));
            }
            if j + 1 < n {
                triplets.push((row, i * n + j + 1, -1.0));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

fn bench_spmv(matrix: &CsrMatrix<f64>, x: &[f64], iters: usize) -> f64 {
    for _ in 0..3 {
        let _ = matrix.matvec(x);
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = matrix.matvec(x);
    }
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_amg_solve(matrix: &CsrMatrix<f64>, rhs: &[f64]) -> (f64, f64, usize) {
    let config_amg = AmgConfig::for_energy_minimizing();
    let t0 = Instant::now();
    let amg = AmgPreconditioner::from_csr(matrix, config_amg);
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let config_gmres = GmresConfig {
        max_iterations: 500,
        restart: 30,
        tolerance: 1e-6,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };

    // Warm up
    let _ = gmres_preconditioned(matrix, &amg, rhs, &config_gmres);

    let t = Instant::now();
    let sol = gmres_preconditioned(matrix, &amg, rhs, &config_gmres);
    let solve_ms = t.elapsed().as_secs_f64() * 1000.0;
    (setup_ms, solve_ms, sol.iterations)
}

fn run_bench(label: &str, matrix: &CsrMatrix<f64>, x: &[f64], rhs: &[f64]) {
    let nn = matrix.num_rows;
    eprintln!("{} (n={})", label, nn);
    eprintln!(
        "{:<14} {:>10} {:>10} {:>10} {:>6} {:>10}",
        "threads", "SpMV(ms)", "setup(ms)", "solve(ms)", "iters", "total(ms)"
    );

    for &threads in &[1, 4, 10, 14] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        pool.install(|| {
            let spmv_iters = if nn > 500_000 { 10 } else { 50 };
            let spmv_ms = bench_spmv(matrix, x, spmv_iters);
            let (setup_ms, solve_ms, iters) = bench_amg_solve(matrix, rhs);

            eprintln!(
                "{:<14} {:>10.3} {:>10.1} {:>10.1} {:>6} {:>10.1}",
                format!("{} threads", threads),
                spmv_ms,
                setup_ms,
                solve_ms,
                iters,
                setup_ms + solve_ms
            );
        });
    }
    eprintln!();
}

fn main() {
    // 100K DOFs
    let n = 316;
    let m100k = poisson_2d(n);
    let x100k = vec![1.0_f64; m100k.num_rows];
    let r100k = vec![1.0_f64; m100k.num_rows];
    run_bench("100K DOFs", &m100k, &x100k, &r100k);

    // 1M DOFs
    let n = 1000;
    let m1m = poisson_2d(n);
    let x1m = vec![1.0_f64; m1m.num_rows];
    let r1m = vec![1.0_f64; m1m.num_rows];
    run_bench("1M DOFs", &m1m, &x1m, &r1m);
}
