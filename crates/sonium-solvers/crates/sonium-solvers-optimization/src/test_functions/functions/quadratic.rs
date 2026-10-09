//! Quadratic test function

/// Simple quadratic function for basic testing
/// f(x) = sum(x\[i\]^2)
/// Global minimum at (0, 0, ..., 0) with f = 0
pub fn quadratic(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}
