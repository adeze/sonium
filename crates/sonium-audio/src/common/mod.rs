//! Shared numeric traits, error types, and filter helpers.

pub mod denormals;
pub mod error;
pub mod traits;

pub use error::{FirError, FirResult, IirError, Result};
pub use traits::FilterFloat;
use traits::lit;

/// Logarithmically spaced grid from `base^start` to `base^end` (inclusive).
///
/// Matches `ndarray::Array1::logspace` semantics for the sizes used here: an
/// empty grid for `n == 0`, and `[base^start]` for `n == 1`.
pub fn logspace<T: FilterFloat>(base: T, start: T, end: T, n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![base.powf(start)];
    }
    let step = (end - start) / lit::<T>(n as f64 - 1.0);
    (0..n)
        .map(|i| base.powf(start + lit::<T>(i as f64) * step))
        .collect()
}

/// Converts bandwidth in octaves to a Q factor.
pub fn bw2q(bw: f64) -> f64 {
    let two_pow_bw = 2.0_f64.powf(bw);
    two_pow_bw.sqrt() / (two_pow_bw - 1.0)
}

/// Converts a Q factor to bandwidth in octaves.
pub fn q2bw(q: f64) -> f64 {
    let q2 = (2.0 * q * q + 1.0) / (2.0 * q * q);
    (q2 + (q2 * q2 - 1.0).sqrt()).log(2.0)
}

/// Default Q factor for high/low pass filters
pub const DEFAULT_Q_HIGH_LOW_PASS: f64 = 1.0 / std::f64::consts::SQRT_2;
/// Default Q factor for high/low shelf filters
pub const DEFAULT_Q_HIGH_LOW_SHELF: f64 = 1.0668676536332304; // Value of bw2q(0.9)

/// Lower bound of human hearing (Hz).
pub const AUDIBLE_MIN_FREQ: f64 = 20.0;
/// Upper bound of human hearing (Hz).
pub const AUDIBLE_MAX_FREQ: f64 = 20_000.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_bw_q_roundtrip() {
        let qs = [0.5, 1.0, 2.0, 5.0];
        for &q in &qs {
            let bw = q2bw(q);
            let q2 = bw2q(bw);
            assert!(
                approx_eq(q, q2, 1e-9),
                "roundtrip failed: q={} -> bw={} -> q2={}",
                q,
                bw,
                q2
            );
        }
    }
}
