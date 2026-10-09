//! Deterministic generators shared by compute property tests.
//!
//! The generators deliberately use `SmallRng` and take an explicit seed so a
//! failing property can be reproduced without depending on global RNG state.

use rand::{Rng, SeedableRng, rngs::SmallRng};

#[allow(dead_code)]
pub fn random_complex64(seed: u64, len: usize) -> Vec<num_complex::Complex64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..len)
        .map(|_| {
            num_complex::Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))
        })
        .collect()
}

#[allow(dead_code)]
pub fn random_dense_f64(seed: u64, rows: usize, cols: usize) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| rng.random_range(-1.0..=1.0))
        .collect()
}

/// Construct a row-major SPD matrix `MᵀM + regularization*I`.
#[allow(dead_code)]
pub fn random_spd_f64(seed: u64, size: usize, regularization: f64) -> Vec<f64> {
    let m = random_dense_f64(seed, size, size);
    let mut matrix = vec![0.0; size * size];
    for row in 0..size {
        for col in 0..size {
            let mut value = 0.0;
            for k in 0..size {
                value += m[k * size + row] * m[k * size + col];
            }
            if row == col {
                value += regularization;
            }
            matrix[row * size + col] = value;
        }
    }
    matrix
}

/// Generate sorted, unique indices suitable for a gather/scatter property.
#[allow(dead_code)]
pub fn unique_indices(seed: u64, count: usize, upper_bound: usize) -> Vec<usize> {
    assert!(count <= upper_bound);
    let mut indices: Vec<_> = (0..upper_bound).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    for index in (1..indices.len()).rev() {
        let swap = rng.random_range(0..=index);
        indices.swap(index, swap);
    }
    indices.truncate(count);
    indices.sort_unstable();
    indices
}
