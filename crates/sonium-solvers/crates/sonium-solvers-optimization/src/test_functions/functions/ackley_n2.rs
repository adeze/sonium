//! Ackley N2 test function

/// Ackley N.2 function - challenging multimodal function
/// Global minimum: f(x*)=-200 at x=(0,0)
/// Bounds: x_i in [-32, 32]
pub fn ackley_n2(x: &[f64]) -> f64 {
    assert!(
        x.len() == 2,
        "ackley_n2 requires 2 dimensions, got {}",
        x.len()
    );
    let x1 = x[0];
    let x2 = x[1];
    -200.0
        * (-0.02 * (x1.powi(2) + x2.powi(2)).sqrt()).exp()
        * (2.0 * std::f64::consts::PI * x1).cos()
        * (2.0 * std::f64::consts::PI * x2).cos()
}
