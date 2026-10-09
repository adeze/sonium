//! Binh Korn Weighted test function

/// Binh-Korn weighted objective function
pub fn binh_korn_weighted(x: &[f64]) -> f64 {
    assert!(
        x.len() == 2,
        "binh_korn_weighted requires 2 dimensions, got {}",
        x.len()
    );
    4.0 * x[0].powi(2) + 4.0 * x[1].powi(2)
}
