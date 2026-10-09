//! Graph coloring for multicolor Gauss-Seidel smoothing.
//!
//! Greedy coloring of the adjacency graph of a CSR matrix so that no two
//! rows sharing a nonzero off-diagonal entry receive the same color. Rows
//! with the same color are independent and can be updated in parallel.

use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;

/// Greedy graph coloring of a CSR matrix.
///
/// Returns `(color_of_row, rows_per_color)` where:
/// - `color_of_row[i]` is the color assigned to row `i`
/// - `rows_per_color[c]` is the sorted list of row indices with color `c`
///
/// The algorithm is O(nnz) and typically produces 5-15 colors for FEM
/// matrices arising from low-order discretisations on unstructured meshes.
pub(crate) fn greedy_graph_coloring<T: ComplexField>(
    matrix: &CsrMatrix<T>,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    matrix
        .validate_structure()
        .expect("AMG coloring requires valid CSR storage");
    assert_eq!(
        matrix.num_rows, matrix.num_cols,
        "AMG coloring requires a square matrix"
    );
    let n = matrix.num_rows;
    // A Gauss-Seidel update reads outgoing neighbors and writes its own row.
    // Conflicts therefore follow A union A^T, even for nonsymmetric storage.
    // Record each edge at its later endpoint so that either orientation
    // forbids the earlier endpoint's color. Duplicate edges are harmless.
    let mut earlier_neighbors = vec![Vec::new(); n];
    for i in 0..n {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            if i != j {
                earlier_neighbors[i.max(j)].push(i.min(j));
            }
        }
    }
    let mut color_of_row = vec![0usize; n];
    let mut num_colors = 0usize;

    // Scratch buffer: tracks which colors are used by neighbors of the
    // current row.  Indexed by color; grows as new colors are allocated.
    let mut forbidden = Vec::<bool>::new();

    for i in 0..n {
        // Mark colors used by already-colored neighbors
        for idx in forbidden.iter_mut() {
            *idx = false;
        }

        for &j in &earlier_neighbors[i] {
            let c = color_of_row[j];
            if c < forbidden.len() {
                forbidden[c] = true;
            }
        }

        // Find the smallest available color
        let color = match forbidden.iter().position(|&f| !f) {
            Some(c) => c,
            None => {
                // All existing colors are taken; allocate a new one
                forbidden.push(false);
                forbidden.len() - 1
            }
        };

        color_of_row[i] = color;
        if color >= num_colors {
            num_colors = color + 1;
        }
    }

    // Build rows_per_color
    let mut rows_per_color = vec![Vec::new(); num_colors];
    for (i, &c) in color_of_row.iter().enumerate() {
        rows_per_color[c].push(i);
    }

    (color_of_row, rows_per_color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn nonsymmetric_edges_never_share_a_color() {
        // Upper-only edges were invisible to the old lower-neighbor scan.
        // That made the parallel smoother read slots another worker wrote.
        for edges in [vec![(0, 1), (1, 2), (0, 2)], vec![(1, 0), (1, 2)]] {
            let mut entries = vec![(0, 0, 4.0), (1, 1, 4.0), (2, 2, 4.0)];
            entries.extend(edges.iter().map(|&(i, j)| (i, j, -1.0)));
            let matrix = CsrMatrix::from_triplets(3, 3, entries);
            let (colors, sets) = greedy_graph_coloring(&matrix);
            for &(i, j) in &edges {
                assert_ne!(colors[i], colors[j], "conflicting rows {i}, {j}");
            }
            let mut rows: Vec<_> = sets.into_iter().flatten().collect();
            rows.sort_unstable();
            assert_eq!(rows, vec![0, 1, 2]);
        }
    }

    #[test]
    fn test_greedy_coloring_valid() {
        // 2D Poisson 8x8 = 64 unknowns
        let nx = 8;
        let n = nx * nx;
        let mut triplets: Vec<(usize, usize, Complex64)> = Vec::new();
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                triplets.push((i, i, Complex64::new(4.0, 0.0)));
                if ix > 0 {
                    triplets.push((i, i - 1, Complex64::new(-1.0, 0.0)));
                }
                if ix < nx - 1 {
                    triplets.push((i, i + 1, Complex64::new(-1.0, 0.0)));
                }
                if iy > 0 {
                    triplets.push((i, i - nx, Complex64::new(-1.0, 0.0)));
                }
                if iy < nx - 1 {
                    triplets.push((i, i + nx, Complex64::new(-1.0, 0.0)));
                }
            }
        }
        let matrix = CsrMatrix::from_triplets(n, n, triplets);

        let (color_of_row, rows_per_color) = greedy_graph_coloring(&matrix);

        // Every row must be assigned
        assert_eq!(color_of_row.len(), n);

        // No two adjacent rows may share a color
        for i in 0..n {
            let c_i = color_of_row[i];
            for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                let j = matrix.col_indices[idx];
                if j != i {
                    assert_ne!(
                        c_i, color_of_row[j],
                        "Adjacent rows {} and {} share color {}",
                        i, j, c_i
                    );
                }
            }
        }

        // rows_per_color must cover all rows exactly once
        let total: usize = rows_per_color.iter().map(|v| v.len()).sum();
        assert_eq!(total, n);

        // For 5-point stencil we expect a modest number of colors
        assert!(
            rows_per_color.len() <= 20,
            "Too many colors: {}",
            rows_per_color.len()
        );
    }

    #[test]
    fn test_greedy_coloring_diagonal_matrix() {
        // Diagonal matrix: every row is independent, should need only 1 color
        let n = 10;
        let triplets: Vec<(usize, usize, Complex64)> =
            (0..n).map(|i| (i, i, Complex64::new(2.0, 0.0))).collect();
        let matrix = CsrMatrix::from_triplets(n, n, triplets);

        let (_, rows_per_color) = greedy_graph_coloring(&matrix);
        assert_eq!(
            rows_per_color.len(),
            1,
            "Diagonal matrix needs only 1 color"
        );
        assert_eq!(rows_per_color[0].len(), n);
    }
}
