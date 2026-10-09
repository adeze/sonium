//! Multifrontal BLR validation on a 2D FEM Helmholtz plane-wave problem.
//!
//! Mirrors the `test_2d_plane_wave` case in `fem_analytical_validation`,
//! but swaps the GMRES solve for `MultifrontalBlrFactorization` so we
//! exercise the BLR direct path on a complex symmetric (Helmholtz) system.
//!
//! Gates:
//!  1. `||A·x − b|| / ||b||` ≤ scenario residual bound.
//!  2. `||x_blr − x_lu|| / ||x_lu||` ≤ scenario parity bound.
//!  3. L² error vs analytical plane wave matches the same bar as the
//!     iterative `fem_analytical_validation` test (≤ 0.01).
//!
//! The Helmholtz K − k²M matrix is symmetric (complex-symmetric, not
//! Hermitian) and sparse, with the natural sparsity pattern of a P1
//! triangle mesh. This is exactly the regime BLR was built for, so
//! `epsilon = 1e-10` is expected to recover the LU solve to ~1e-8.

use num_complex::Complex64;
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{DirichletBC, apply_dirichlet};
use sonium_fem::mesh::{Mesh, unit_square_triangles};
use sonium_solvers::direct::{
    BlrConfig, MultifrontalBlrFactorization, PanelKind, SparseLuFactorization,
};
use sonium_solvers::sparse::CsrMatrix;
use std::f64::consts::PI;

fn helmholtz_to_csr(problem: &HelmholtzProblem) -> CsrMatrix<Complex64> {
    let compressed = problem.matrix.to_compressed();
    let n = compressed.dim;
    let triplets: Vec<(usize, usize, Complex64)> = (0..compressed.nnz())
        .map(|i| (compressed.rows[i], compressed.cols[i], compressed.values[i]))
        .collect();
    CsrMatrix::from_triplets(n, n, triplets)
}

fn rel_residual(matrix: &CsrMatrix<Complex64>, x: &[Complex64], b: &[Complex64]) -> f64 {
    let n = b.len();
    let mut res_sq = 0.0;
    let mut b_sq = 0.0;
    for i in 0..n {
        let mut row = Complex64::new(0.0, 0.0);
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            row += matrix.values[idx] * x[matrix.col_indices[idx]];
        }
        let r = b[i] - row;
        res_sq += r.norm_sqr();
        b_sq += b[i].norm_sqr();
    }
    if b_sq == 0.0 {
        res_sq.sqrt()
    } else {
        (res_sq / b_sq).sqrt()
    }
}

fn rel_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    let mut d = 0.0;
    let mut n = 0.0;
    for i in 0..a.len() {
        d += (a[i] - b[i]).norm_sqr();
        n += b[i].norm_sqr();
    }
    if n == 0.0 { d.sqrt() } else { (d / n).sqrt() }
}

fn l2_error<F>(mesh: &Mesh, x: &[Complex64], analytical: F) -> f64
where
    F: Fn(f64, f64, f64) -> Complex64,
{
    let mut err_sq = 0.0;
    let mut norm_sq = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact = analytical(node.x, node.y, node.z);
        let diff = x[i] - exact;
        err_sq += diff.norm_sqr();
        norm_sq += exact.norm_sqr();
    }
    if norm_sq > 1e-15 {
        (err_sq / norm_sq).sqrt()
    } else {
        err_sq.sqrt()
    }
}

struct Scenario {
    label: &'static str,
    cfg: BlrConfig,
    res_bound: f64,
    parity_bound: f64,
    /// L² discretisation-error bound. Same value across scenarios — BLR
    /// shouldn't degrade the analytical comparison vs the exact solve.
    l2_bound: f64,
}

fn main() {
    // Plane wave at k = 2.0, 45° — exact solution of homogeneous Helmholtz.
    let k = 2.0;
    let theta = PI / 4.0;
    let kx = k * theta.cos();
    let ky = k * theta.sin();
    let plane_wave = move |x: f64, y: f64, _z: f64| {
        let phase = kx * x + ky * y;
        Complex64::new(phase.cos(), phase.sin())
    };

    let mesh = unit_square_triangles(16);
    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::new(0.0, 0.0)
        });

    // Dirichlet plane wave on all four edge tags.
    let bcs: Vec<DirichletBC> = (1..=4)
        .map(|tag| DirichletBC::new(tag, plane_wave))
        .collect();
    apply_dirichlet(&mut problem, &mesh, &bcs);

    let matrix = helmholtz_to_csr(&problem);
    let b = problem.rhs.clone();
    let n = matrix.num_rows;

    println!(
        "=== BLR-FEM Helmholtz plane-wave (n = {n} DOFs, nnz = {}, k = {k}) ===",
        matrix.values.len()
    );

    let lu = SparseLuFactorization::factorize(&matrix).expect("exact sparse LU");
    let x_lu = lu.solve(&b).expect("sparse LU solve");
    let res_lu = rel_residual(&matrix, &x_lu, &b);
    let l2_lu = l2_error(&mesh, &x_lu, plane_wave);
    println!(
        "Exact SparseLU reference: residual {:.3e}, L² vs plane wave {:.3e}",
        res_lu, l2_lu
    );
    assert!(
        l2_lu < 0.01,
        "FAILED: exact-LU L² error {l2_lu:.3e} above iterative bar 0.01 — mesh discretisation problem"
    );

    let scenarios = vec![
        Scenario {
            label: "BLR eps=0 (no compression)",
            cfg: BlrConfig {
                epsilon: 0.0,
                block_size: 32,
                min_block_size: 32,
                ..Default::default()
            },
            res_bound: (100.0 * res_lu).max(1e-10),
            parity_bound: 1e-8,
            l2_bound: 0.01,
        },
        Scenario {
            label: "BLR eps=1e-10",
            cfg: BlrConfig {
                epsilon: 1e-10,
                block_size: 32,
                min_block_size: 32,
                ..Default::default()
            },
            res_bound: 1e-6,
            parity_bound: 1e-5,
            l2_bound: 0.01,
        },
        Scenario {
            label: "HSS eps=1e-10",
            cfg: BlrConfig {
                epsilon: 1e-10,
                block_size: 32,
                min_block_size: 32,
                panel_kind: PanelKind::Hss,
                ..Default::default()
            },
            res_bound: 1e-6,
            parity_bound: 1e-5,
            l2_bound: 0.01,
        },
    ];

    for scn in &scenarios {
        let blr = MultifrontalBlrFactorization::factorize_with_config(&matrix, &scn.cfg)
            .expect("BLR factorise");
        let x_blr = blr.solve(&b);
        let res = rel_residual(&matrix, &x_blr, &b);
        let parity = rel_diff(&x_blr, &x_lu);
        let l2 = l2_error(&mesh, &x_blr, plane_wave);
        let avg_rank = blr.average_rank();
        let bytes = blr.factor_memory_bytes();
        println!(
            "{:<28} | res {:.2e} (≤ {:.2e}) | parity {:.2e} (≤ {:.2e}) | L² {:.3e} (≤ {:.3e}) | fronts {} avg_rank {:.1} factor {} KB",
            scn.label,
            res,
            scn.res_bound,
            parity,
            scn.parity_bound,
            l2,
            scn.l2_bound,
            blr.num_fronts(),
            avg_rank,
            bytes / 1024,
        );
        assert!(
            res < scn.res_bound,
            "FAILED: {}: residual {:.2e} exceeds bound {:.2e}",
            scn.label,
            res,
            scn.res_bound
        );
        assert!(
            parity < scn.parity_bound,
            "FAILED: {}: BLR-vs-LU diff {:.2e} exceeds bound {:.2e}",
            scn.label,
            parity,
            scn.parity_bound
        );
        assert!(
            l2 < scn.l2_bound,
            "FAILED: {}: L² error {:.3e} exceeds bound {:.3e}",
            scn.label,
            l2,
            scn.l2_bound
        );
    }

    println!("SUCCESS");
}
