//! Step test function

/// Step function - discontinuous, multimodal
/// Global minimum: f(x) = 0 for every coordinate in [-0.5, 0.5)
/// Bounds: x_i in [-100, 100]
pub fn step(x: &[f64]) -> f64 {
    x.iter().map(|&xi| (xi + 0.5).floor().powi(2)).sum::<f64>()
}
