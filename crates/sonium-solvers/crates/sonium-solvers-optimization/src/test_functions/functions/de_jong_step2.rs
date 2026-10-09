//! De Jong Step2 test function

/// De Jong step function (variant)
pub fn de_jong_step2(x: &[f64]) -> f64 {
    x.iter().map(|&xi| (xi + 0.5).floor().powi(2)).sum()
}
