//! Alpine N2 test function

/// Alpine N.2 function - multimodal with single global minimum
/// Global minimum on `[0, 10]^N`: each dimension is near `7.917`, with
/// 2D value approximately `-7.885`.
/// Bounds: x_i in [0, 10]
pub fn alpine_n2(x: &[f64]) -> f64 {
    -x.iter()
        .map(|&xi| xi.max(0.0).sqrt() * xi.sin())
        .product::<f64>()
}
