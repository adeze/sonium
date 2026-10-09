//! Internal sincos batching helper backed by `sonium-fastmath`.
//!
//! Twiddle and Bluestein-chirp generation both compute `sin/cos` over a
//! contiguous array of phases. Going through SIMD (`sincos_f64x2` on NEON,
//! `sincos_f64x4` on AVX2 + FMA) compresses plan-build by ~1.5–2× versus
//! per-element libm `f64::sin_cos`. Falls back to the scalar polynomial
//! path elsewhere — still 1.5× faster than libm on Apple silicon because
//! the scalar fastmath path is Sleef u10 minimax + Cody-Waite, not the
//! correctly-rounded libm.

use sonium_fastmath::sincos_f64_slice;

/// Fill `sins[j]` and `coss[j]` with `(sin(thetas[j]), cos(thetas[j]))`
/// using whatever SIMD width is available. Lengths must match.
pub(crate) fn fill_sincos(thetas: &[f64], sins: &mut [f64], coss: &mut [f64]) {
    assert_eq!(thetas.len(), sins.len());
    assert_eq!(thetas.len(), coss.len());
    sincos_f64_slice(thetas, sins, coss);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn matches_libm_within_tolerance() {
        let n = 257; // not a multiple of 2 or 4 — exercises tail too
        let thetas: Vec<f64> = (0..n).map(|j| TAU * (j as f64) / (n as f64)).collect();
        let mut sins = vec![0.0; n];
        let mut coss = vec![0.0; n];
        fill_sincos(&thetas, &mut sins, &mut coss);
        for (j, &t) in thetas.iter().enumerate() {
            let (sref, cref) = t.sin_cos();
            assert!(
                (sins[j] - sref).abs() < 1e-13,
                "sin diff at j={j}, theta={t}: {} vs {sref}",
                sins[j]
            );
            assert!(
                (coss[j] - cref).abs() < 1e-13,
                "cos diff at j={j}, theta={t}: {} vs {cref}",
                coss[j]
            );
        }
    }

    #[test]
    fn empty_is_noop() {
        let thetas: Vec<f64> = vec![];
        let mut sins: Vec<f64> = vec![];
        let mut coss: Vec<f64> = vec![];
        fill_sincos(&thetas, &mut sins, &mut coss);
        assert!(sins.is_empty());
        assert!(coss.is_empty());
    }
}
