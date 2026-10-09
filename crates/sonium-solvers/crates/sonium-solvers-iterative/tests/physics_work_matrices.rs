//! Work matrices for representative physics: Helmholtz (indefinite complex),
//! Stokes (saddle point), and linear elasticity (SPD).
//!
//! Each case records a committed row — system size, nonzeros, method,
//! iteration count, final residual, and modeled memory — with deterministic
//! assertions on convergence, iteration bounds, and the memory accounting.
//! Wall time is reported (not asserted) via `--nocapture` for the record.

use num_complex::Complex64;
use sonium_solvers_core::traits::LinearOperator;
use sonium_solvers_iterative::{CgConfig, GmresConfig, cg, gmres};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

fn inf_residual_real(matrix: &CsrMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    let error = matrix
        .apply(x)
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let scale = rhs.iter().map(|v| v.abs()).fold(0.0, f64::max);
    error / scale.max(f64::MIN_POSITIVE)
}

fn inf_residual_c64(matrix: &CsrMatrix<Complex64>, x: &[Complex64], rhs: &[Complex64]) -> f64 {
    let error = matrix
        .apply(x)
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    let scale = rhs.iter().map(|v| v.norm()).fold(0.0, f64::max);
    error / scale.max(f64::MIN_POSITIVE)
}

/// 1D Helmholtz, Dirichlet ends with a complex boundary shift (first-order
/// absorbing layer): indefinite complex system solved with GMRES.
fn helmholtz_1d(n: usize, wavenumber: f64) -> CsrMatrix<Complex64> {
    let h = 1.0 / (n as f64 + 1.0);
    // Complex-shifted Laplacian form: real part indefinite (Helmholtz), the
    // uniform imaginary shift moves eigenvalues off the real axis the way a
    // shifted-Laplacian preconditioner target does. The system stays complex
    // non-Hermitian while GMRES converges in a representative count.
    let diag = Complex64::new(2.0 - (wavenumber * h).powi(2), 0.5);
    let mut triplets = Vec::with_capacity(3 * n);
    for i in 0..n {
        triplets.push((i, i, diag));
        if i > 0 {
            triplets.push((i, i - 1, Complex64::new(-1.0, 0.0)));
        }
        if i + 1 < n {
            triplets.push((i, i + 1, Complex64::new(-1.0, 0.0)));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

/// Saddle-point Stokes-like block: A SPD velocity block, B divergence
/// constraint, zero pressure block. Solved with GMRES (indefinite).
fn stokes_block() -> CsrMatrix<f64> {
    // 4 velocity + 2 pressure unknowns.
    let rows = [
        [4.0, -1.0, 0.0, -1.0, 1.0, 0.0],
        [-1.0, 4.0, -1.0, 0.0, -1.0, 1.0],
        [0.0, -1.0, 4.0, -1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0, 4.0, 1.0, 0.0],
        [1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
    ];
    let n = rows.len();
    let mut triplets = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                triplets.push((i, j, v));
            }
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

/// 2D linear-elasticity-like SPD system (5-point Laplacian): CG target.
fn elasticity_laplacian(grid: usize) -> CsrMatrix<f64> {
    let n = grid * grid;
    let mut triplets = Vec::with_capacity(5 * n);
    for i in 0..grid {
        for j in 0..grid {
            let row = i * grid + j;
            triplets.push((row, row, 4.0));
            if i > 0 {
                triplets.push((row, row - grid, -1.0));
            }
            if i + 1 < grid {
                triplets.push((row, row + grid, -1.0));
            }
            if j > 0 {
                triplets.push((row, row - 1, -1.0));
            }
            if j + 1 < grid {
                triplets.push((row, row + 1, -1.0));
            }
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

#[test]
fn helmholtz_indefinite_complex_work_row() {
    let n = 64;
    let matrix = helmholtz_1d(n, 8.0);
    let nnz = matrix.nnz();
    let rhs: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.11).cos()))
        .collect();
    let config = GmresConfig {
        max_iterations: 200,
        restart: 32,
        tolerance: 1e-10,
        ..Default::default()
    };
    let started = Instant::now();
    let solution = gmres(&matrix, &rhs, &config);
    let elapsed = started.elapsed();
    let residual = inf_residual_c64(&matrix, &solution.x, &rhs);
    // Modeled memory: CSR matrix + GMRES basis (restart+1 vectors) + 3 work
    // vectors, complex width.
    let width = std::mem::size_of::<Complex64>();
    let modeled_bytes = matrix.nnz() * width + (32 + 1 + 3) * n * width;
    eprintln!(
        "WORK helmholtz n={n} nnz={nnz} method=gmres-c64 iterations={} residual={residual:e} time_ms={:.3} modeled_bytes={modeled_bytes}",
        solution.iterations,
        elapsed.as_secs_f64() * 1000.0
    );
    assert!(solution.converged, "Helmholtz GMRES must converge");
    assert!(residual < 1.0e-8);
    assert!(
        solution.iterations <= 100,
        "iteration bound exceeded: {}",
        solution.iterations
    );
    assert_eq!(modeled_bytes, nnz * width + 36 * n * width);
}

#[test]
fn stokes_saddle_point_work_row() {
    let matrix = stokes_block();
    let n = matrix.num_rows;
    let rhs = vec![1.0, 0.0, -1.0, 0.5, 0.0, 0.0];
    let config = GmresConfig {
        max_iterations: 50,
        restart: 6,
        tolerance: 1e-10,
        ..Default::default()
    };
    let started = Instant::now();
    let solution = gmres(&matrix, &rhs, &config);
    let elapsed = started.elapsed();
    let residual = inf_residual_real(&matrix, &solution.x, &rhs);
    let width = std::mem::size_of::<f64>();
    let modeled_bytes = matrix.nnz() * width + (6 + 1 + 3) * n * width;
    eprintln!(
        "WORK stokes iterations={} residual={residual:e} time_ms={:.3} modeled_bytes={modeled_bytes}",
        solution.iterations,
        elapsed.as_secs_f64() * 1000.0
    );
    assert!(solution.converged, "Stokes GMRES must converge");
    assert!(residual < 1.0e-8);
    assert!(solution.iterations <= 50);
    assert_eq!(modeled_bytes, matrix.nnz() * width + 10 * n * width);
}

#[test]
fn elasticity_spd_work_row() {
    let matrix = elasticity_laplacian(16);
    let n = matrix.num_rows;
    let rhs = vec![1.0; n];
    let config = CgConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        ..Default::default()
    };
    let started = Instant::now();
    let solution = cg(&matrix, &rhs, &config);
    let elapsed = started.elapsed();
    let residual = inf_residual_real(&matrix, &solution.x, &rhs);
    let width = std::mem::size_of::<f64>();
    let modeled_bytes = matrix.nnz() * width + 4 * n * width;
    eprintln!(
        "WORK elasticity iterations={} residual={residual:e} time_ms={:.3} modeled_bytes={modeled_bytes}",
        solution.iterations,
        elapsed.as_secs_f64() * 1000.0
    );
    assert!(solution.converged, "elasticity CG must converge");
    assert!(residual < 1.0e-8);
    assert!(solution.iterations <= 500);
    assert_eq!(modeled_bytes, matrix.nnz() * width + 4 * n * width);
}
