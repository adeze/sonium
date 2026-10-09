//! Holder Table test function

/// Holder table function - 2D multimodal
/// Global minimum: f(x) = -19.2085 at x = (±8.05502, ±9.66459)
/// Bounds: x_i in [-10, 10]
pub fn holder_table(x: &[f64]) -> f64 {
    assert!(
        x.len() == 2,
        "holder_table requires 2 dimensions, got {}",
        x.len()
    );
    let x1 = x[0];
    let x2 = x[1];
    let exp_term = (1.0 - (x1.powi(2) + x2.powi(2)).sqrt() / std::f64::consts::PI).abs();
    -(x1.sin() * x2.cos()).abs() * exp_term.exp()
}
