//! Benchmark matrix generators
//!
//! Reusable test matrices shared across all benchmark files.

#![allow(dead_code)]

use num_complex::Complex64;
use sonium_solvers::{CsrBuilder, CsrMatrix};

/// 2D Poisson equation (5-point stencil) on an n×n grid.
///
/// Produces an SPD matrix of dimension n² with ~5n² nonzeros.
/// Suitable for CG, GMRES, and all preconditioners.
pub fn poisson_2d(n: usize) -> CsrMatrix<f64> {
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

/// 2D Helmholtz equation on an n×n grid with wavenumber k.
///
/// Discretizes -Δu - k²u = f with mesh spacing h = 1/(n+1).
/// Produces a complex, non-symmetric (with absorbing BCs), indefinite matrix.
pub fn helmholtz_2d(n: usize, k: f64) -> CsrMatrix<Complex64> {
    let nn = n * n;
    let h = 1.0 / (n as f64 + 1.0);
    let k2h2 = k * k * h * h;

    let mut triplets: Vec<(usize, usize, Complex64)> = Vec::with_capacity(5 * nn);

    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;

            // Diagonal: 4 - k²h² (+ small imaginary shift for stability)
            let diag = Complex64::new(4.0 - k2h2, 0.1 * k2h2);
            triplets.push((row, row, diag));

            if i > 0 {
                triplets.push((row, (i - 1) * n + j, Complex64::new(-1.0, 0.0)));
            }
            if i + 1 < n {
                triplets.push((row, (i + 1) * n + j, Complex64::new(-1.0, 0.0)));
            }
            if j > 0 {
                triplets.push((row, i * n + j - 1, Complex64::new(-1.0, 0.0)));
            }
            if j + 1 < n {
                triplets.push((row, i * n + j + 1, Complex64::new(-1.0, 0.0)));
            }
        }
    }

    CsrMatrix::from_triplets(nn, nn, triplets)
}

/// 3D Laplacian (7-point stencil) on an n×n×n grid.
///
/// Produces an SPD matrix of dimension n³ with ~7n³ nonzeros.
pub fn laplacian_3d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n * n;
    let mut triplets = Vec::with_capacity(7 * nn);

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let row = i * n * n + j * n + k;
                triplets.push((row, row, 6.0));

                if i > 0 {
                    triplets.push((row, (i - 1) * n * n + j * n + k, -1.0));
                }
                if i + 1 < n {
                    triplets.push((row, (i + 1) * n * n + j * n + k, -1.0));
                }
                if j > 0 {
                    triplets.push((row, i * n * n + (j - 1) * n + k, -1.0));
                }
                if j + 1 < n {
                    triplets.push((row, i * n * n + (j + 1) * n + k, -1.0));
                }
                if k > 0 {
                    triplets.push((row, i * n * n + j * n + k - 1, -1.0));
                }
                if k + 1 < n {
                    triplets.push((row, i * n * n + j * n + k + 1, -1.0));
                }
            }
        }
    }

    CsrMatrix::from_triplets(nn, nn, triplets)
}

/// Random diagonally dominant banded matrix.
///
/// Uses a fixed seed for deterministic benchmarks.
pub fn random_diag_dominant(n: usize, bandwidth: usize) -> CsrMatrix<f64> {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    let mut rng = SmallRng::seed_from_u64(12345);
    let mut triplets = Vec::with_capacity(n * (2 * bandwidth + 1));

    for i in 0..n {
        let mut row_sum = 0.0_f64;
        let lo = i.saturating_sub(bandwidth);
        let hi = (i + bandwidth + 1).min(n);

        for j in lo..hi {
            if j != i {
                let val: f64 = rng.random_range(-1.0..1.0);
                row_sum += val.abs();
                triplets.push((i, j, val));
            }
        }

        // Diagonal dominance: |a_ii| > sum of off-diag
        triplets.push((i, i, row_sum + 1.0));
    }

    CsrMatrix::from_triplets(n, n, triplets)
}

/// Deterministic matrix with a fixed number of nonzeros per row.
///
/// This is intentionally synthetic: the roofline preflight uses it to separate
/// short/medium/long row effects from PDE-specific sparsity patterns.
pub fn fixed_row_nnz(n: usize, row_nnz: usize) -> CsrMatrix<f64> {
    let row_nnz = row_nnz.clamp(1, n.max(1));
    let mut triplets = Vec::with_capacity(n * row_nnz);

    for row in 0..n {
        let half = row_nnz / 2;
        let mut cols = Vec::with_capacity(row_nnz);
        cols.push(row);
        for offset in 1..=half {
            cols.push((row + offset) % n);
        }
        for offset in 1.. {
            if cols.len() >= row_nnz {
                break;
            }
            cols.push((row + n - offset % n) % n);
        }
        cols.sort_unstable();
        cols.dedup();
        while cols.len() < row_nnz {
            let next = (row + cols.len() * 17 + 1) % n;
            if !cols.contains(&next) {
                cols.push(next);
            }
        }

        for col in cols {
            let value = if col == row {
                row_nnz as f64 + 1.0
            } else {
                -1.0 / row_nnz as f64
            };
            triplets.push((row, col, value));
        }
    }

    CsrMatrix::from_triplets(n, n, triplets)
}

/// Create an all-ones RHS vector (f64).
pub fn ones_rhs(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// Create an all-ones RHS vector (f32).
pub fn ones_rhs_f32(n: usize) -> Vec<f32> {
    vec![1.0_f32; n]
}

/// Create an all-ones RHS vector (Complex64).
pub fn ones_rhs_c64(n: usize) -> Vec<Complex64> {
    vec![Complex64::new(1.0, 0.0); n]
}

/// 2D Poisson equation (5-point stencil) on an n×n grid, in f32.
pub fn poisson_2d_f32(n: usize) -> CsrMatrix<f32> {
    poisson_2d(n).cast::<f32>()
}

/// 2D Poisson equation (5-point stencil) on an n×n grid, in f16.
pub fn poisson_2d_f16(n: usize) -> CsrMatrix<half::f16> {
    poisson_2d(n).cast::<half::f16>()
}

/// Create an all-ones RHS vector (f16).
pub fn ones_rhs_f16(n: usize) -> Vec<half::f16> {
    vec![half::f16::ONE; n]
}

/// Write a real-valued CSR matrix to MatrixMarket coordinate format.
pub fn write_matrix_market(matrix: &CsrMatrix<f64>, path: &std::path::Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Cannot create .mtx file");

    writeln!(f, "%%MatrixMarket matrix coordinate real general").unwrap();
    writeln!(f, "% Generated by sonium-solvers benchmarks").unwrap();
    writeln!(
        f,
        "{} {} {}",
        matrix.num_rows,
        matrix.num_cols,
        matrix.nnz()
    )
    .unwrap();

    for i in 0..matrix.num_rows {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            let v = matrix.values[idx];
            writeln!(f, "{} {} {:.17e}", i + 1, j + 1, v).unwrap();
        }
    }
}

/// Read a real-valued MatrixMarket coordinate file into a CSR matrix.
pub fn read_matrix_market(path: &std::path::Path) -> CsrMatrix<f64> {
    let content = std::fs::read_to_string(path).expect("Cannot read .mtx file");
    let mut lines = content.lines();

    // Parse header
    let header = lines.next().expect("Empty .mtx file");
    let is_symmetric = header.contains("symmetric");

    // Skip comments
    let mut size_line = String::new();
    for line in lines.by_ref() {
        if !line.starts_with('%') {
            size_line = line.to_string();
            break;
        }
    }

    // Parse dimensions
    let parts: Vec<&str> = size_line.split_whitespace().collect();
    let nrows: usize = parts[0].parse().unwrap();
    let ncols: usize = parts[1].parse().unwrap();
    let nnz: usize = parts[2].parse().unwrap();

    let mut triplets = Vec::with_capacity(if is_symmetric { 2 * nnz } else { nnz });

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let p: Vec<&str> = line.split_whitespace().collect();
        let i: usize = p[0].parse::<usize>().unwrap() - 1; // 1-indexed to 0-indexed
        let j: usize = p[1].parse::<usize>().unwrap() - 1;
        let v: f64 = p[2].parse().unwrap();

        triplets.push((i, j, v));
        if is_symmetric && i != j {
            triplets.push((j, i, v));
        }
    }

    CsrMatrix::from_triplets(nrows, ncols, triplets)
}

/// Grid size n that produces approximately `target` DOFs for 2D (n²).
pub fn grid_size_2d(target: usize) -> usize {
    (target as f64).sqrt() as usize
}

/// Grid size n that produces approximately `target` DOFs for 3D (n³).
pub fn grid_size_3d(target: usize) -> usize {
    (target as f64).cbrt() as usize
}

/// Contract-scale 4x4 KKT saddle fixture with an external Schur approximation.
///
/// Same block structure as `tests/saddle_point_contract.rs` (2+2 blocks,
/// manufactured solution `[1.0, -0.5, 0.75, 1.5]`), shared here so the
/// selection-evidence bench exercises the explicit `solve_saddle_point`
/// facade path on identical data.
pub fn saddle_kkt_4x4() -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let rows = [
        vec![(0, 4.0), (1, 1.0), (2, 1.0)],
        vec![(0, 1.0), (1, 3.0), (3, 1.0)],
        vec![(0, 1.0), (2, 2.0)],
        vec![(1, 1.0), (3, 1.5)],
    ];
    let mut builder = CsrBuilder::new(4, 4);
    for row in rows {
        builder.add_row_entries(row.into_iter());
    }
    let mut schur = CsrBuilder::new(2, 2);
    schur.add_row_entries([(0, 2.0)].into_iter());
    schur.add_row_entries([(1, 1.5)].into_iter());
    (builder.finish(), schur.finish())
}

/// 2D steady convection-diffusion on an n×n grid with first-order upwind
/// convection and velocity field (vx, vy) = (2, 1).
///
/// Discretizes -Δu + v·∇u = f with mesh spacing h = 1/(n+1). Upwinding keeps
/// the matrix (nonsymmetric) diagonally dominant, so it exercises the general
/// nonsymmetric GMRES+ILU facade path without an SPD assumption.
pub fn convection_diffusion_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let h = 1.0 / (n as f64 + 1.0);
    let vx = 2.0;
    let vy = 1.0;
    let mut triplets = Vec::with_capacity(5 * nn);

    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            triplets.push((row, row, 4.0 + (vx + vy) * h));
            if i > 0 {
                triplets.push((row, (i - 1) * n + j, -1.0 - vy * h));
            }
            if i + 1 < n {
                triplets.push((row, (i + 1) * n + j, -1.0));
            }
            if j > 0 {
                triplets.push((row, i * n + j - 1, -1.0 - vx * h));
            }
            if j + 1 < n {
                triplets.push((row, i * n + j + 1, -1.0));
            }
        }
    }

    CsrMatrix::from_triplets(nn, nn, triplets)
}
