//! Profile the nested-dissection ordering step vs end-to-end sparse Cholesky
//! factorization on 2D/3D Laplacian grids, and report separator/fill quality
//! so ordering changes can be validated for parity.
//!
//! Run: cargo run --release -p sonium-solvers --example nd_ordering_profile

use sonium_solvers::CsrMatrix;
use sonium_solvers::direct::sparse_cholesky::{
    CholeskyOrdering, SparseCholeskyFactorization, SparseCholeskySymbolic,
};
use sonium_solvers::sparse::partition::nested_dissection;
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

fn laplacian_3d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n * n;
    let mut triplets = Vec::with_capacity(7 * nn);
    let idx = |i: usize, j: usize, k: usize| (i * n + j) * n + k;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let row = idx(i, j, k);
                triplets.push((row, row, 6.0));
                for (di, dj, dk) in [
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    let (ni, nj, nk) = (i as isize + di, j as isize + dj, k as isize + dk);
                    if ni >= 0
                        && nj >= 0
                        && nk >= 0
                        && ni < n as isize
                        && nj < n as isize
                        && nk < n as isize
                    {
                        triplets.push((row, idx(ni as usize, nj as usize, nk as usize), -1.0));
                    }
                }
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

/// Symmetric adjacency matching `sparse_cholesky::symmetrized_adjacency`.
fn symmetrized_adjacency<T: sonium_solvers::ComplexField>(
    matrix: &CsrMatrix<T>,
) -> Vec<Vec<usize>> {
    let n = matrix.num_rows;
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            if j != i {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

fn best_of<F: FnMut() -> R, R>(iters: usize, mut f: F) -> (f64, R) {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..iters {
        let t = Instant::now();
        let r = f();
        let dt = t.elapsed().as_secs_f64() * 1e3;
        if dt < best {
            best = dt;
        }
        out = Some(r);
    }
    (best, out.unwrap())
}

fn run_case(name: &str, matrix: &CsrMatrix<f64>, min_block: usize) {
    let n = matrix.num_rows;
    let adjacency = symmetrized_adjacency(matrix);
    let all: Vec<usize> = (0..n).collect();
    let max_levels = ((n as f64) / (min_block as f64)).log2().ceil() as usize + 1;

    // Ordering step alone: nested_dissection over the full adjacency.
    let (t_order, nodes) = best_of(5, || {
        let mut nodes = Vec::new();
        nested_dissection(&adjacency, &all, min_block, max_levels, 0, &mut nodes);
        nodes
    });

    // Quality: total separator size (interior nodes) and factor fill.
    let sep_total: usize = nodes
        .iter()
        .filter(|nd| nd.left.is_some())
        .map(|nd| nd.global_indices.len())
        .sum();
    let symbolic = SparseCholeskySymbolic::analyze(matrix, CholeskyOrdering::NestedDissection)
        .expect("symbolic analysis");
    let fill = symbolic.factor_nonzeros();

    let (t_factorize, _) = best_of(5, || {
        SparseCholeskyFactorization::factorize(matrix).expect("factorize")
    });

    println!(
        "{name}: n={n} nnz={} order={t_order:.2}ms factorize={t_factorize:.2}ms \
         order_share={:.1}% sep_total={sep_total} fill={fill}",
        matrix.nnz(),
        100.0 * t_order / t_factorize
    );
}

fn main() {
    run_case("grid2d-64", &poisson_2d(64), 32);
    run_case("grid2d-128", &poisson_2d(128), 32);
    run_case("grid2d-256", &poisson_2d(256), 32);
    run_case("grid3d-16", &laplacian_3d(16), 32);
    run_case("grid3d-24", &laplacian_3d(24), 32);
    run_case("grid3d-32", &laplacian_3d(32), 32);
}
