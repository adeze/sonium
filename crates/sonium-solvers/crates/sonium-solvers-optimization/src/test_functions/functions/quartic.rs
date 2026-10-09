//! Quartic test function

/// Quartic function with noise - unimodal with added random noise
/// Global minimum: f(x) ≈ 0 at x = (0, 0, ..., 0)
/// Bounds: x_i in [-1.28, 1.28]
pub fn quartic(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, &xi)| (i as f64 + 1.0) * xi.powi(4))
        .sum::<f64>()
    // Note: Original includes random noise, but we omit it for deterministic testing
}
