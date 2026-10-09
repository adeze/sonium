//! Alpine N1 test function

/// Alpine N.1 function - multimodal with many local minima
/// Global minimum: f(x) = 0 at x = (0, 0, ..., 0)
/// Bounds: x_i in [-10, 10]
pub fn alpine_n1(x: &[f64]) -> f64 {
    x.iter().map(|&xi| (xi * xi.sin() + 0.1 * xi).abs()).sum()
}
