//! Cross In Tray test function

/// Cross-in-tray function - 2D multimodal function
/// Global minimum: f(x) = -2.06261 at x = (±1.34941, ±1.34941)
/// Bounds: x_i in [-10, 10]
pub fn cross_in_tray(x: &[f64]) -> f64 {
    assert!(
        x.len() == 2,
        "cross_in_tray requires 2 dimensions, got {}",
        x.len()
    );
    let x1 = x[0];
    let x2 = x[1];
    let exp_term = (100.0 - (x1.powi(2) + x2.powi(2)).sqrt() / std::f64::consts::PI).abs();
    -0.0001 * ((x1.sin() * x2.sin()).abs() * exp_term.exp() + 1.0).powf(0.1)
}
