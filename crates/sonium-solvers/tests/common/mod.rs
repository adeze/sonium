//! Seeded solver fixtures shared by integration property tests.

use rand::{Rng, SeedableRng, rngs::SmallRng};
use sonium_solvers::sparse::CsrMatrix;

/// Generate a dense row-major matrix with strict diagonal dominance.
#[allow(dead_code)]
pub fn random_diagonally_dominant(seed: u64, size: usize) -> CsrMatrix<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(size * size);
    for row in 0..size {
        let mut off_diagonal_sum = 0.0_f64;
        let mut row_values = vec![0.0_f64; size];
        for (col, value) in row_values.iter_mut().enumerate() {
            if row != col {
                *value = rng.random_range(-0.5..=0.5);
                off_diagonal_sum += (*value).abs();
            }
        }
        row_values[row] = off_diagonal_sum + rng.random_range(1.0..=2.0);
        values.extend(row_values);
    }
    let row_ptrs = (0..=size).map(|row| row * size).collect();
    let col_indices = (0..size).flat_map(|_| 0..size).collect();
    CsrMatrix::from_raw_parts(size, size, row_ptrs, col_indices, values)
}

#[allow(dead_code)]
pub fn random_spd(seed: u64, size: usize, regularization: f64) -> CsrMatrix<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut m = vec![0.0; size * size];
    for value in &mut m {
        *value = rng.random_range(-1.0..=1.0);
    }
    let mut values = vec![0.0; size * size];
    for row in 0..size {
        for col in 0..size {
            let mut value = 0.0;
            for k in 0..size {
                value += m[k * size + row] * m[k * size + col];
            }
            values[row * size + col] = value + if row == col { regularization } else { 0.0 };
        }
    }
    let row_ptrs = (0..=size).map(|row| row * size).collect();
    let col_indices = (0..size).flat_map(|_| 0..size).collect();
    CsrMatrix::from_raw_parts(size, size, row_ptrs, col_indices, values)
}
