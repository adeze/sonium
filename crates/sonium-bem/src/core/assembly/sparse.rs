//! Sparse matrix structures for BEM
//!
//! Re-exports [`CsrMatrix`](sonium_solvers::CsrMatrix) from sonium-solvers for
//! general-purpose CSR storage. Provides [`BlockedCsr`] for hierarchical matrix
//! leaf-level storage.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;

// Re-export the canonical CsrMatrix from solvers
pub use sonium_solvers::CsrMatrix;
pub use sonium_solvers::sparse::CsrBuilder;

/// Blocked CSR format for hierarchical matrices
///
/// Stores the matrix as a collection of dense blocks at the leaf level
/// of a hierarchical decomposition.
#[derive(Debug, Clone)]
pub struct BlockedCsr {
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_cols: usize,
    /// Block size (rows and columns per block)
    pub block_size: usize,
    /// Number of block rows
    pub num_block_rows: usize,
    /// Number of block columns
    pub num_block_cols: usize,
    /// Dense blocks stored in CSR-like format
    /// Each block is a dense matrix
    pub blocks: Vec<DenseMatrix<Complex64>>,
    /// Block column indices
    pub block_col_indices: Vec<usize>,
    /// Block row pointers
    pub block_row_ptrs: Vec<usize>,
}

impl BlockedCsr {
    /// Create a new blocked CSR matrix
    pub fn new(num_rows: usize, num_cols: usize, block_size: usize) -> Self {
        let num_block_rows = num_rows.div_ceil(block_size);
        let num_block_cols = num_cols.div_ceil(block_size);

        Self {
            num_rows,
            num_cols,
            block_size,
            num_block_rows,
            num_block_cols,
            blocks: Vec::new(),
            block_col_indices: Vec::new(),
            block_row_ptrs: vec![0; num_block_rows + 1],
        }
    }

    /// Matrix-vector product using blocked structure
    pub fn matvec(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");

        let mut y = vec![Complex64::new(0.0, 0.0); self.num_rows];

        for block_i in 0..self.num_block_rows {
            let row_start = block_i * self.block_size;
            let row_end = (row_start + self.block_size).min(self.num_rows);
            let local_rows = row_end - row_start;

            for idx in self.block_row_ptrs[block_i]..self.block_row_ptrs[block_i + 1] {
                let block_j = self.block_col_indices[idx];
                let block = &self.blocks[idx];

                let col_start = block_j * self.block_size;
                let col_end = (col_start + self.block_size).min(self.num_cols);
                let local_cols = col_end - col_start;

                let x_local: Vec<Complex64> = (col_start..col_end).map(|j| x[j]).collect();

                for i in 0..local_rows {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for j in 0..local_cols {
                        sum += block[[i, j]] * x_local[j];
                    }
                    y[row_start + i] += sum;
                }
            }
        }

        y
    }
}
