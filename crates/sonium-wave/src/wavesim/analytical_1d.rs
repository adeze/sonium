//! 1D free-space analytical reference matching `wavesim_py.utilities.analytical_solution`.
//!
//! The reference accounts for the truncated sinc Laplace kernel used by
//! WaveSim, so it is **not** simply the continuous Helmholtz Green's function.
//! The closed form, due to Vellekoop, is:
//!
//! ```text
//! u(x) = i·h/(2k) · exp(i·k·|x|)
//!      − h/(4π·k) · [ exp(i·k·|x|)·(E1(i(k − π/h)|x|) − E1(i(k + π/h)|x|))
//!                   − exp(−i·k·|x|)·(−E1(−i(k − π/h)|x|) + E1(−i(k + π/h)|x|)) ]
//! ```
//!
//! with the special value at `x = 0`:
//!
//! ```text
//! u(0) = i·h/(2k) · (1 + 2j·atanh(h·k/π)/π)
//! ```

use num_complex::Complex64;

use crate::array::Array1;

/// Build the WaveSim 1D analytical reference field on the propagation axis.
///
/// * `x_um[i]` — signed positions relative to the source. Spacing must be uniform.
/// * `wavelength` — vacuum wavelength.
pub fn analytical_solution(x_um: &[f64], wavelength: f64) -> Array1<Complex64> {
    assert!(x_um.len() >= 2, "need at least two grid points");
    let h = (x_um[1] - x_um[0]).abs();
    let k = std::f64::consts::TAU / wavelength;
    let pi = std::f64::consts::PI;

    let mut out: Array1<Complex64> = vec![Complex64::new(0.0, 0.0); x_um.len()];
    for (i, &xi) in x_um.iter().enumerate() {
        let x = xi.abs();
        let phi = k * x;
        if (k * x).abs() < 1.0e-10 {
            // Special case at x = 0.
            let arg = h * k / pi;
            let atanh_val = 0.5 * ((1.0 + arg) / (1.0 - arg)).ln();
            out[i] = Complex64::new(0.0, h / (2.0 * k))
                * (Complex64::new(1.0, 0.0) + Complex64::new(0.0, 2.0 * atanh_val / pi));
            continue;
        }

        let exp_pos = Complex64::from_polar(1.0, phi);
        let exp_neg = exp_pos.conj();

        let arg1 = Complex64::new(0.0, (k - pi / h) * x);
        let arg2 = Complex64::new(0.0, (k + pi / h) * x);
        let arg1_neg = -arg1;
        let arg2_neg = -arg2;

        let bracket =
            exp_pos * (exp1(arg1) - exp1(arg2)) - exp_neg * (-exp1(arg1_neg) + exp1(arg2_neg));

        let leading = Complex64::new(0.0, h / (2.0 * k)) * exp_pos;
        let correction = Complex64::new(-h / (4.0 * pi * k), 0.0) * bracket;
        out[i] = leading + correction;
    }
    out
}

/// Relative error matching `wavesim_py.utilities.relative_error`:
///
/// `mean(|e − e_true|²) / mean(|e_true|²)`.
pub fn relative_error(u: &[Complex64], u_ref: &[Complex64]) -> f64 {
    assert_eq!(u.len(), u_ref.len());
    let n = u.len() as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (a, b) in u.iter().zip(u_ref.iter()) {
        num += (a - b).norm_sqr();
        den += b.norm_sqr();
    }
    (num / n) / (den / n)
}

/// Numpy-style allclose with relative tolerance.
pub fn all_close(a: &[Complex64], b: &[Complex64], rtol: f64) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).norm() <= rtol * y.norm().max(1e-30))
}

/// Complex exponential integral `E1(z) = ∫_z^∞ e^(-t)/t dt`.
///
/// Branch cut on the negative real axis. We use:
/// * series expansion near the origin (`|z| < 4`)
/// * continued-fraction expansion otherwise (asymptotic-like, valid for
///   `Re(z) ≥ 0` and most of the upper/lower half-plane).
pub fn exp1(z: Complex64) -> Complex64 {
    if z == Complex64::new(0.0, 0.0) {
        return Complex64::new(f64::INFINITY, 0.0);
    }
    if z.norm() < 4.0 {
        exp1_series(z)
    } else {
        exp1_cfrac(z)
    }
}

/// Series: `E1(z) = -γ - ln(z) - Σ_{n≥1} (-z)^n / (n · n!)`.
fn exp1_series(z: Complex64) -> Complex64 {
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = Complex64::new(0.0, 0.0);
    for n in 1..200 {
        term *= -z / Complex64::new(n as f64, 0.0);
        let add = term / Complex64::new(n as f64, 0.0);
        sum += add;
        if add.norm() < 1e-18 * sum.norm().max(1e-30) {
            break;
        }
    }
    -Complex64::new(EULER_GAMMA, 0.0) - z.ln() - sum
}

/// Numerical Recipes-style continued fraction for E_n(z) with n=1:
/// `E1(z) = e^(-z) · h` where `h` is built from the recurrence with
/// `a_i = -i², b_0 = z+1, b_{i+1} = b_i + 2`.
fn exp1_cfrac(z: Complex64) -> Complex64 {
    let tiny = 1.0e-300;
    let mut b = z + Complex64::new(1.0, 0.0);
    let mut c = Complex64::new(1.0 / tiny, 0.0);
    let mut d = b.inv();
    let mut h = d;
    for i in 1..400 {
        let a = Complex64::new(-((i * i) as f64), 0.0);
        b += Complex64::new(2.0, 0.0);
        let denom = a * d + b;
        d = if denom.norm() < tiny {
            Complex64::new(1.0 / tiny, 0.0)
        } else {
            denom.inv()
        };
        c = b + a / c;
        if c.norm() < tiny {
            c = Complex64::new(tiny, 0.0);
        }
        let delta = c * d;
        h *= delta;
        if (delta - Complex64::new(1.0, 0.0)).norm() < 1e-15 {
            break;
        }
    }
    h * (-z).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp1_small_real() {
        // E1(1) ≈ 0.21938393439552
        let v = exp1(Complex64::new(1.0, 0.0));
        assert!((v.re - 0.219_383_934_395_52).abs() < 1e-9, "got {v:?}");
        assert!(v.im.abs() < 1e-12);
    }

    #[test]
    fn exp1_large_real() {
        // E1(10) ≈ 4.156968929e-6
        let v = exp1(Complex64::new(10.0, 0.0));
        assert!((v.re - 4.156_968_929_435_22e-6).abs() < 1e-13, "got {v:?}");
    }

    #[test]
    fn exp1_pure_imaginary() {
        // E1(i) = -Ci(1) + i(Si(1) - π/2) ≈ -0.337403922900968 − 0.624713256427568 i
        let v = exp1(Complex64::new(0.0, 1.0));
        assert!((v.re - (-0.337_403_922_900_968)).abs() < 1e-9, "got {v:?}");
        assert!((v.im - (-0.624_713_256_427_568)).abs() < 1e-9, "got {v:?}");
    }

    #[test]
    fn analytical_at_zero_matches_special_value() {
        let n = 5;
        let pixel = 0.05;
        let wavelength = 0.5;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 - 2.0) * pixel).collect();
        let u = analytical_solution(&x, wavelength);
        // Special value at x=0
        let k = std::f64::consts::TAU / wavelength;
        let pi = std::f64::consts::PI;
        let arg = pixel * k / pi;
        let atanh_val = 0.5 * ((1.0 + arg) / (1.0 - arg)).ln();
        let expected = Complex64::new(0.0, pixel / (2.0 * k))
            * (Complex64::new(1.0, 0.0) + Complex64::new(0.0, 2.0 * atanh_val / pi));
        assert!(
            (u[2] - expected).norm() < 1e-12,
            "got {:?} vs {expected:?}",
            u[2]
        );
    }
}
