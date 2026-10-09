//! Parity validation of `MultifrontalBlrFactorization` against the exact
//! `SparseLuFactorization`.
//!
//! This lives as an integration test (instead of a sonium-qa binary) because
//! the sonium-qa crate transitively pulls in sonium-bem and sonium-fem,
//! which are mid-refactor and not currently buildable in this branch.

use num_complex::Complex64;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{BlrConfig, MultifrontalBlrFactorization, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn make_2d_laplacian_f64(nx: usize) -> CsrMatrix<f64> {
    let n = nx * nx;
    let mut dense = DenseMatrix::<f64>::zeros(n, n);
    for iy in 0..nx {
        for ix in 0..nx {
            let i = iy * nx + ix;
            dense[[i, i]] = 4.0;
            if ix > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if ix + 1 < nx {
                dense[[i, i + 1]] = -1.0;
            }
            if iy > 0 {
                dense[[i, i - nx]] = -1.0;
            }
            if iy + 1 < nx {
                dense[[i, i + nx]] = -1.0;
            }
        }
    }
    CsrMatrix::from_dense(&dense, 1e-15)
}

fn make_banded_nonsymmetric(n: usize, bandwidth: usize) -> CsrMatrix<f64> {
    let mut dense = DenseMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        dense[[i, i]] = (n as f64) + (i as f64).sin();
        for k in 1..=bandwidth.min(n - 1) {
            if i + k < n {
                dense[[i, i + k]] = -1.0 / (k as f64);
            }
            if i >= k {
                dense[[i, i - k]] = -1.0 / (k as f64 + 0.5);
            }
        }
    }
    CsrMatrix::from_dense(&dense, 1e-15)
}

fn make_complex_helmholtz(nx: usize) -> CsrMatrix<Complex64> {
    let n = nx * nx;
    let mut dense = DenseMatrix::<Complex64>::zeros(n, n);
    let shift = Complex64::new(0.0, 0.1);
    for iy in 0..nx {
        for ix in 0..nx {
            let i = iy * nx + ix;
            dense[[i, i]] = Complex64::new(4.0, 0.0) + shift;
            if ix > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if ix + 1 < nx {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
            if iy > 0 {
                dense[[i, i - nx]] = Complex64::new(-1.0, 0.0);
            }
            if iy + 1 < nx {
                dense[[i, i + nx]] = Complex64::new(-1.0, 0.0);
            }
        }
    }
    CsrMatrix::from_dense(&dense, 1e-15)
}

#[test]
fn multifrontal_blr_rejects_malformed_csr_before_factorization() {
    let mut matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]);
    matrix.row_ptrs[2] = 1;
    assert!(MultifrontalBlrFactorization::factorize(&matrix).is_err());
}

fn rel_residual_f64(matrix: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut res_sq = 0.0;
    let mut b_sq = 0.0;
    for (i, &bi) in b.iter().enumerate() {
        let mut s = 0.0;
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            s += matrix.values[idx] * x[matrix.col_indices[idx]];
        }
        let r = bi - s;
        res_sq += r * r;
        b_sq += bi * bi;
    }
    if b_sq == 0.0 {
        res_sq.sqrt()
    } else {
        (res_sq / b_sq).sqrt()
    }
}

fn rel_residual_complex(matrix: &CsrMatrix<Complex64>, x: &[Complex64], b: &[Complex64]) -> f64 {
    let mut res_sq = 0.0;
    let mut b_sq = 0.0;
    for (i, &bi) in b.iter().enumerate() {
        let mut s = Complex64::new(0.0, 0.0);
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            s += matrix.values[idx] * x[matrix.col_indices[idx]];
        }
        let r = bi - s;
        res_sq += r.norm_sqr();
        b_sq += bi.norm_sqr();
    }
    if b_sq == 0.0 {
        res_sq.sqrt()
    } else {
        (res_sq / b_sq).sqrt()
    }
}

#[test]
fn laplacian_parity_eps_zero() {
    // ε=0 BLR should match exact LU to numerical round-off.
    for nx in [8, 12] {
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

        let exact = SparseLuFactorization::factorize(&m).expect("exact LU");
        let x_exact = exact.solve(&b).expect("sparse LU solve");
        let res_exact = rel_residual_f64(&m, &x_exact, &b);

        let blr = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 32,
                min_block_size: 32,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("BLR factorise");
        let x_blr = blr.solve(&b);
        let res_blr = rel_residual_f64(&m, &x_blr, &b);

        let bound = (100.0 * res_exact).max(1e-10);
        assert!(
            res_blr < bound,
            "[laplacian nx={}] blr res {:.3e} exceeds bound {:.3e} (exact {:.3e})",
            nx,
            res_blr,
            bound,
            res_exact
        );
    }
}

#[test]
fn laplacian_parity_eps_compressed() {
    // ε=1e-8 BLR should still solve to ~1e-5.
    for nx in [12, 16] {
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.3).cos()).collect();

        let blr = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 1e-8,
                block_size: 32,
                min_block_size: 32,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("BLR factorise");
        let x = blr.solve(&b);
        let res = rel_residual_f64(&m, &x, &b);
        assert!(
            res < 1e-4,
            "[laplacian nx={}] blr(ε=1e-8) residual {:.3e} too large",
            nx,
            res
        );
    }
}

#[test]
fn banded_nonsymmetric_parity() {
    for &(n, bw) in &[(80, 3), (160, 5)] {
        let m = make_banded_nonsymmetric(n, bw);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.3).cos()).collect();

        let exact = SparseLuFactorization::factorize(&m).expect("exact LU");
        let x_exact = exact.solve(&b).expect("sparse LU solve");
        let res_exact = rel_residual_f64(&m, &x_exact, &b);

        let blr = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 32,
                min_block_size: 32,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("BLR factorise");
        let x_blr = blr.solve(&b);
        let res_blr = rel_residual_f64(&m, &x_blr, &b);

        let bound = (100.0 * res_exact).max(1e-10);
        assert!(
            res_blr < bound,
            "[banded n={} bw={}] blr {:.3e} > bound {:.3e} (exact {:.3e})",
            n,
            bw,
            res_blr,
            bound,
            res_exact
        );
    }
}

#[test]
fn complex_helmholtz_runs() {
    for nx in [8, 12, 16] {
        let m = make_complex_helmholtz(nx);
        let n = m.num_rows;
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 + 1.0).sin(), 0.3))
            .collect();
        let blr = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 32,
                min_block_size: 32,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("BLR factorise");
        let x = blr.solve(&b);
        let res = rel_residual_complex(&m, &x, &b);
        assert!(
            res < 1e-8,
            "[complex helmholtz nx={}] residual {:.3e}",
            nx,
            res
        );
    }
}

#[test]
fn f32_laplacian_parity() {
    let nx = 16;
    let n = nx * nx;
    let mut dense = DenseMatrix::<f32>::zeros(n, n);
    for iy in 0..nx {
        for ix in 0..nx {
            let i = iy * nx + ix;
            dense[[i, i]] = 4.0;
            if ix > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if ix + 1 < nx {
                dense[[i, i + 1]] = -1.0;
            }
            if iy > 0 {
                dense[[i, i - nx]] = -1.0;
            }
            if iy + 1 < nx {
                dense[[i, i + nx]] = -1.0;
            }
        }
    }
    let m = CsrMatrix::from_dense(&dense, 1e-7);
    let b: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5).sin()).collect();
    let blr = MultifrontalBlrFactorization::factorize_with_config(
        &m,
        &BlrConfig {
            epsilon: 0.0,
            block_size: 32,
            min_block_size: 32,
            max_levels: None,
            ..Default::default()
        },
    )
    .expect("BLR factorise (f32)");
    let x = blr.solve(&b);
    let mut res_sq = 0.0_f32;
    let mut b_sq = 0.0_f32;
    for (i, &bi) in b.iter().enumerate() {
        let mut s = 0.0_f32;
        for idx in m.row_ptrs[i]..m.row_ptrs[i + 1] {
            s += m.values[idx] * x[m.col_indices[idx]];
        }
        res_sq += (bi - s) * (bi - s);
        b_sq += bi * bi;
    }
    let rel = (res_sq / b_sq).sqrt();
    assert!(rel < 1e-3, "f32 residual {} too large", rel);
}
