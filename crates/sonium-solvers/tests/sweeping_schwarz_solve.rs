// Facade-level proof that the sweeping preconditioner from Algo review
// step 2 (SIM-PERF-20261013.md section 2, gap 2) is reachable and
// effective through the public `sonium_solvers` API only.
use num_complex::Complex64;
use sonium_solvers::{
    AdditiveSchwarzPreconditioner, CsrMatrix, GmresConfig, SweepingSchwarzPreconditioner,
    coordinate_layers, gmres_preconditioned,
};

fn helmholtz_2d(m: usize, k: f64) -> (CsrMatrix<Complex64>, Vec<[f64; 3]>) {
    let n = m * m;
    let diag = 4.0 - k * k;
    let mut triplets = Vec::with_capacity(5 * n);
    for iy in 0..m {
        for ix in 0..m {
            let i = iy * m + ix;
            triplets.push((i, i, Complex64::new(diag, 0.0)));
            if ix > 0 {
                triplets.push((i, i - 1, Complex64::new(-1.0, 0.0)));
            }
            if ix + 1 < m {
                triplets.push((i, i + 1, Complex64::new(-1.0, 0.0)));
            }
            if iy > 0 {
                triplets.push((i, i - m, Complex64::new(-1.0, 0.0)));
            }
            if iy + 1 < m {
                triplets.push((i, i + m, Complex64::new(-1.0, 0.0)));
            }
        }
    }
    let h = 1.0 / (m as f64 + 1.0);
    let coords: Vec<[f64; 3]> = (0..n)
        .map(|i| [((i % m) as f64 + 1.0) * h, ((i / m) as f64 + 1.0) * h, 0.0])
        .collect();
    (CsrMatrix::from_triplets(n, n, triplets), coords)
}

#[test]
fn facade_sweeping_schwarz_beats_additive_oras() {
    let m = 16;
    let k = 0.5;
    let (matrix, coords) = helmholtz_2d(m, k);
    let n = m * m;
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos()))
        .collect();
    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-8,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };
    let layers = coordinate_layers(&coords, 4, 0).expect("valid layers");
    let sweeping = SweepingSchwarzPreconditioner::try_from_csr_layers(&matrix, &layers, 1, k)
        .expect("sweeping setup through facade");
    let mut blocks = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for i in 0..n {
        blocks[(i / m) % 4].push(i);
    }
    let additive = AdditiveSchwarzPreconditioner::from_csr_partitions_oras(&matrix, &blocks, 1, k);
    let sol_sweep = gmres_preconditioned(&matrix, &sweeping, &b, &config);
    let sol_add = gmres_preconditioned(&matrix, &additive, &b, &config);
    assert!(sol_add.converged, "additive baseline converges");
    assert!(sol_sweep.converged, "sweeping converges via facade");
    assert!(
        sol_sweep.iterations <= sol_add.iterations,
        "sweeping ({} iters) must not exceed additive ORAS ({} iters)",
        sol_sweep.iterations,
        sol_add.iterations
    );
}
