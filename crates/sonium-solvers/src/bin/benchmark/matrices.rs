use num_complex::Complex64;
use sonium_solvers::CsrMatrix;
use std::path::Path;

/// Compute grid size n such that n*n ~ target_dofs for a 2D problem.
fn grid_size_2d(target_dofs: usize) -> usize {
    (target_dofs as f64).sqrt() as usize
}

/// Compute grid size n such that n*n*n ~ target_dofs for a 3D problem.
fn grid_size_3d(target_dofs: usize) -> usize {
    (target_dofs as f64).cbrt() as usize
}

/// 2D Poisson equation (5-point stencil) on an n*n grid.
///
/// Produces an SPD matrix of dimension n^2 with ~5n^2 nonzeros.
/// Returns the matrix and the actual number of DOFs (n^2).
pub fn poisson_2d(target_dofs: usize) -> (CsrMatrix<f64>, usize) {
    let n = grid_size_2d(target_dofs);
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

    (CsrMatrix::from_triplets(nn, nn, triplets), nn)
}

/// 2D anisotropic diffusion on an n*n grid.
///
/// Discretizes `-d/dx(ax du/dx) - d/dy(ay du/dy)` with constant,
/// strongly stretched coefficients. This keeps the matrix SPD while giving
/// AMG and Krylov benchmarks a harder SOTA-style elliptic case than isotropic
/// Poisson.
pub fn anisotropic_diffusion_2d(target_dofs: usize) -> (CsrMatrix<f64>, usize) {
    let n = grid_size_2d(target_dofs);
    let nn = n * n;
    let ax = 1.0;
    let ay = 0.01;
    let mut triplets = Vec::with_capacity(5 * nn);

    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            let mut diag = 0.0;

            if i > 0 {
                diag += ay;
                triplets.push((row, (i - 1) * n + j, -ay));
            }
            if i + 1 < n {
                diag += ay;
                triplets.push((row, (i + 1) * n + j, -ay));
            }
            if j > 0 {
                diag += ax;
                triplets.push((row, i * n + j - 1, -ax));
            }
            if j + 1 < n {
                diag += ax;
                triplets.push((row, i * n + j + 1, -ax));
            }

            triplets.push((row, row, diag + 1.0));
        }
    }

    (CsrMatrix::from_triplets(nn, nn, triplets), nn)
}

/// Regularized 2D saddle-point system with two velocity fields and pressure.
///
/// The block form is `[A 0 Bx^T; 0 A By^T; Bx By -eps I]`, where `A` is a
/// shifted 2D Poisson operator. It is a compact Stokes-like benchmark: real,
/// sparse, symmetric indefinite, and nonsingular.
pub fn saddle_point_2d(target_dofs: usize) -> (CsrMatrix<f64>, usize) {
    let n = ((target_dofs as f64 / 3.0).sqrt() as usize).max(2);
    let nodes = n * n;
    let u_offset = 0;
    let v_offset = nodes;
    let p_offset = 2 * nodes;
    let total = 3 * nodes;
    let pressure_shift = 0.05;
    let coupling = 0.25;
    let mut triplets = Vec::with_capacity(15 * nodes);

    for i in 0..n {
        for j in 0..n {
            let node = i * n + j;
            add_poisson_row(&mut triplets, n, u_offset, i, j, 4.1);
            add_poisson_row(&mut triplets, n, v_offset, i, j, 4.1);

            let p = p_offset + node;
            triplets.push((p, p, -pressure_shift));

            add_symmetric_coupling(&mut triplets, p, u_offset + node, coupling);
            if j > 0 {
                add_symmetric_coupling(&mut triplets, p, u_offset + node - 1, -coupling);
            }
            add_symmetric_coupling(&mut triplets, p, v_offset + node, coupling);
            if i > 0 {
                add_symmetric_coupling(&mut triplets, p, v_offset + node - n, -coupling);
            }
        }
    }

    (CsrMatrix::from_triplets(total, total, triplets), total)
}

fn add_poisson_row(
    triplets: &mut Vec<(usize, usize, f64)>,
    n: usize,
    offset: usize,
    i: usize,
    j: usize,
    diagonal: f64,
) {
    let row = offset + i * n + j;
    triplets.push((row, row, diagonal));

    if i > 0 {
        triplets.push((row, offset + (i - 1) * n + j, -1.0));
    }
    if i + 1 < n {
        triplets.push((row, offset + (i + 1) * n + j, -1.0));
    }
    if j > 0 {
        triplets.push((row, offset + i * n + j - 1, -1.0));
    }
    if j + 1 < n {
        triplets.push((row, offset + i * n + j + 1, -1.0));
    }
}

fn add_symmetric_coupling(
    triplets: &mut Vec<(usize, usize, f64)>,
    pressure_row: usize,
    velocity_col: usize,
    value: f64,
) {
    triplets.push((pressure_row, velocity_col, value));
    triplets.push((velocity_col, pressure_row, value));
}

/// 2D Helmholtz equation on an n*n grid with the given wavenumber.
///
/// Discretizes -Delta u - k^2 u = f with mesh spacing h = 1/(n+1).
/// Produces a complex, non-symmetric (with absorbing BCs), indefinite matrix.
/// Returns the matrix and the actual number of DOFs (n^2).
pub fn helmholtz_2d(target_dofs: usize, wavenumber: f64) -> (CsrMatrix<Complex64>, usize) {
    let n = grid_size_2d(target_dofs);
    let nn = n * n;
    let h = 1.0 / (n as f64 + 1.0);
    let k2h2 = wavenumber * wavenumber * h * h;

    let mut triplets: Vec<(usize, usize, Complex64)> = Vec::with_capacity(5 * nn);

    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;

            // Diagonal: 4 - k^2 h^2 (+ small imaginary shift for stability)
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

    (CsrMatrix::from_triplets(nn, nn, triplets), nn)
}

/// 3D Laplacian (7-point stencil) on an n*n*n grid.
///
/// Produces an SPD matrix of dimension n^3 with ~7n^3 nonzeros.
/// Returns the matrix and the actual number of DOFs (n^3).
pub fn laplacian_3d(target_dofs: usize) -> (CsrMatrix<f64>, usize) {
    let n = grid_size_3d(target_dofs);
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

    (CsrMatrix::from_triplets(nn, nn, triplets), nn)
}

/// Write a real-valued CSR matrix to MatrixMarket coordinate format.
pub fn write_matrix_market_real(matrix: &CsrMatrix<f64>, path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Cannot create .mtx file");

    writeln!(f, "%%MatrixMarket matrix coordinate real general").unwrap();
    writeln!(f, "% Generated by sonium-benchmark").unwrap();
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

/// Write a complex-valued CSR matrix to MatrixMarket coordinate format.
#[allow(dead_code)]
pub fn write_matrix_market_complex(matrix: &CsrMatrix<Complex64>, path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Cannot create .mtx file");

    writeln!(f, "%%MatrixMarket matrix coordinate complex general").unwrap();
    writeln!(f, "% Generated by sonium-benchmark").unwrap();
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
            writeln!(f, "{} {} {:.17e} {:.17e}", i + 1, j + 1, v.re, v.im).unwrap();
        }
    }
}

/// Create an all-ones RHS vector (f64).
pub fn ones_rhs_real(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// Create an all-ones RHS vector (Complex64).
pub fn ones_rhs_complex(n: usize) -> Vec<Complex64> {
    vec![Complex64::new(1.0, 0.0); n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_2d_sizes() {
        let (mat, actual_n) = poisson_2d(100);
        // sqrt(100) = 10, so 10x10 = 100 DOFs
        assert_eq!(actual_n, 100);
        assert_eq!(mat.num_rows, 100);
        assert_eq!(mat.num_cols, 100);
    }

    #[test]
    fn test_helmholtz_2d_sizes() {
        let (mat, actual_n) = helmholtz_2d(100, 5.0);
        assert_eq!(actual_n, 100);
        assert_eq!(mat.num_rows, 100);
        assert_eq!(mat.num_cols, 100);
    }

    #[test]
    fn test_anisotropic_diffusion_2d_sizes() {
        let (mat, actual_n) = anisotropic_diffusion_2d(100);
        assert_eq!(actual_n, 100);
        assert_eq!(mat.num_rows, 100);
        assert_eq!(mat.num_cols, 100);
    }

    #[test]
    fn test_saddle_point_2d_sizes() {
        let (mat, actual_n) = saddle_point_2d(300);
        assert_eq!(actual_n, 300);
        assert_eq!(mat.num_rows, 300);
        assert_eq!(mat.num_cols, 300);
    }

    #[test]
    fn test_laplacian_3d_sizes() {
        let (mat, actual_n) = laplacian_3d(1000);
        // cbrt(1000) = 10, so 10^3 = 1000
        assert_eq!(actual_n, 1000);
        assert_eq!(mat.num_rows, 1000);
        assert_eq!(mat.num_cols, 1000);
    }

    #[test]
    fn test_grid_size_rounding() {
        // For target 1000, sqrt(1000) = 31.6... => 31, so 31^2 = 961
        let (_, actual_n) = poisson_2d(1000);
        assert_eq!(actual_n, 31 * 31);
    }

    #[test]
    fn test_ones_rhs() {
        let rhs = ones_rhs_real(10);
        assert_eq!(rhs.len(), 10);
        assert!(rhs.iter().all(|&v| v == 1.0));

        let rhs_c = ones_rhs_complex(10);
        assert_eq!(rhs_c.len(), 10);
        assert!(rhs_c.iter().all(|&v| v == Complex64::new(1.0, 0.0)));
    }

    #[test]
    fn test_poisson_2d_spd() {
        // Check diagonal dominance (SPD proxy): each diagonal entry should be >= sum of off-diags
        let (mat, _) = poisson_2d(100);
        for i in 0..mat.num_rows {
            let mut diag = 0.0;
            let mut off_sum = 0.0;
            for idx in mat.row_ptrs[i]..mat.row_ptrs[i + 1] {
                if mat.col_indices[idx] == i {
                    diag = mat.values[idx];
                } else {
                    off_sum += mat.values[idx].abs();
                }
            }
            assert!(diag >= off_sum, "Row {i}: diag={diag}, off_sum={off_sum}");
        }
    }

    #[test]
    fn test_anisotropic_diffusion_2d_is_strictly_diagonally_dominant() {
        let (mat, _) = anisotropic_diffusion_2d(100);
        for i in 0..mat.num_rows {
            let mut diag = 0.0;
            let mut off_sum = 0.0;
            for idx in mat.row_ptrs[i]..mat.row_ptrs[i + 1] {
                if mat.col_indices[idx] == i {
                    diag = mat.values[idx];
                } else {
                    off_sum += mat.values[idx].abs();
                }
            }
            assert!(diag > off_sum, "Row {i}: diag={diag}, off_sum={off_sum}");
        }
    }

    #[test]
    fn test_saddle_point_2d_has_indefinite_diagonal_blocks() {
        let (mat, _) = saddle_point_2d(300);
        let pressure_offset = mat.num_rows / 3 * 2;
        let mut velocity_diag = 0.0;
        let mut pressure_diag = 0.0;

        for idx in mat.row_ptrs[0]..mat.row_ptrs[1] {
            if mat.col_indices[idx] == 0 {
                velocity_diag = mat.values[idx];
            }
        }
        for idx in mat.row_ptrs[pressure_offset]..mat.row_ptrs[pressure_offset + 1] {
            if mat.col_indices[idx] == pressure_offset {
                pressure_diag = mat.values[idx];
            }
        }

        assert!(velocity_diag > 0.0);
        assert!(pressure_diag < 0.0);
    }
}
