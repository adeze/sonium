//! Spherical Bessel and Hankel functions
//!
//! Direct port of NC_SphericalHankel from NC_3dFunctions.cpp.
//! These functions are critical for FMM translation operators.
//!
//! ## Definitions
//!
//! Spherical Bessel function of first kind:
//! ```text
//! j_n(x) = √(π/2x) * J_{n+1/2}(x)
//! ```
//!
//! Spherical Bessel function of second kind (Neumann):
//! ```text
//! y_n(x) = √(π/2x) * Y_{n+1/2}(x)
//! ```
//!
//! Spherical Hankel function of first kind:
//! ```text
//! h_n^(1)(x) = j_n(x) + i * y_n(x)
//! ```

use num_complex::Complex64;
use std::fmt;

/// Validation failures from the checked spherical-function API.
#[derive(Debug, Clone, PartialEq)]
pub enum SphericalFunctionError {
    /// No terms were requested.
    EmptyOrder,
    /// The real argument must be finite and strictly positive for the
    /// outgoing-wave functions and the Neumann branch.
    InvalidArgument(f64),
    /// The harmonic sign must be exactly +1 or -1.
    InvalidHarmonicFactor(f64),
    /// A caller-owned output slice does not hold all requested terms.
    OutputLength { expected: usize, actual: usize },
    /// The requested batch shape cannot be represented by a slice length.
    OutputLengthOverflow { order: usize, rows: usize },
    /// A recurrence overflowed or otherwise produced a non-finite result.
    NonFiniteResult { order: usize, index: usize },
}

impl fmt::Display for SphericalFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOrder => write!(f, "spherical-function order must be at least one"),
            Self::InvalidArgument(x) => {
                write!(
                    f,
                    "spherical-function argument {x} must be finite and positive"
                )
            }
            Self::InvalidHarmonicFactor(sign) => {
                write!(f, "harmonic factor {sign} must be +1 or -1")
            }
            Self::OutputLength { expected, actual } => {
                write!(
                    f,
                    "spherical-function output length {actual} does not match requested order {expected}"
                )
            }
            Self::OutputLengthOverflow { order, rows } => {
                write!(
                    f,
                    "spherical-function batch shape {rows} × {order} exceeds the maximum slice length"
                )
            }
            Self::NonFiniteResult { order, index } => {
                write!(
                    f,
                    "spherical-function order {order} produced a non-finite term {index}"
                )
            }
        }
    }
}

impl std::error::Error for SphericalFunctionError {}

fn validate_real_result(values: &[f64], order: usize) -> Result<(), SphericalFunctionError> {
    values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
        .map_or(Ok(()), |(index, _)| {
            Err(SphericalFunctionError::NonFiniteResult { order, index })
        })
}

fn validate_complex_result(
    values: &[Complex64],
    order: usize,
) -> Result<(), SphericalFunctionError> {
    values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.re.is_finite() || !value.im.is_finite())
        .map_or(Ok(()), |(index, _)| {
            Err(SphericalFunctionError::NonFiniteResult { order, index })
        })
}

fn validate_order_argument(order: usize, x: f64) -> Result<(), SphericalFunctionError> {
    if order == 0 {
        return Err(SphericalFunctionError::EmptyOrder);
    }
    if !x.is_finite() || x <= 0.0 {
        return Err(SphericalFunctionError::InvalidArgument(x));
    }
    Ok(())
}

/// Checked spherical Bessel `j_n` values for `n = 0..order - 1`.
pub fn try_spherical_bessel_j(order: usize, x: f64) -> Result<Vec<f64>, SphericalFunctionError> {
    validate_order_argument(order, x)?;
    let values = spherical_bessel_j(order, x);
    validate_real_result(&values, order)?;
    Ok(values)
}

/// Checked spherical Bessel `j_n` evaluation using caller-owned recurrence
/// storage.  Reuse `workspace` and `output` across a frequency sweep to avoid
/// per-sample allocation while preserving the reference recurrence.
pub fn try_spherical_bessel_j_into(
    order: usize,
    x: f64,
    output: &mut [f64],
    workspace: &mut Vec<f64>,
) -> Result<(), SphericalFunctionError> {
    validate_order_argument(order, x)?;
    if output.len() != order {
        return Err(SphericalFunctionError::OutputLength {
            expected: order,
            actual: output.len(),
        });
    }
    spherical_bessel_j_into_unchecked(order, x, output, workspace);
    validate_real_result(output, order)
}

/// Evaluate spherical Bessel `j_n` terms for several real arguments into a
/// caller-owned row-major buffer.
///
/// `output` must contain exactly `arguments.len() * order` values. Row `i`
/// holds `j_0(arguments[i])` through `j_{order - 1}(arguments[i])`. The
/// function validates every argument before changing `output`; reuse the same
/// `workspace` for frequency or parameter sweeps to avoid a recurrence-buffer
/// allocation for each sample.
///
/// This preserves the scalar Miller reference recurrence. It is a contiguous
/// batch interface for consumers that schedule vector or parallel work, not a
/// claim that the recurrence itself is SIMD-vectorized.
pub fn try_spherical_bessel_j_batch_into(
    order: usize,
    arguments: &[f64],
    output: &mut [f64],
    workspace: &mut Vec<f64>,
) -> Result<(), SphericalFunctionError> {
    if order == 0 {
        return Err(SphericalFunctionError::EmptyOrder);
    }
    let expected =
        order
            .checked_mul(arguments.len())
            .ok_or(SphericalFunctionError::OutputLengthOverflow {
                order,
                rows: arguments.len(),
            })?;
    if output.len() != expected {
        return Err(SphericalFunctionError::OutputLength {
            expected,
            actual: output.len(),
        });
    }
    for &x in arguments {
        validate_order_argument(order, x)?;
    }
    for (&x, row) in arguments.iter().zip(output.chunks_exact_mut(order)) {
        spherical_bessel_j_into_unchecked(order, x, row, workspace);
        validate_real_result(row, order)?;
    }
    Ok(())
}

/// Checked spherical Neumann `y_n` values for `n = 0..order`.
pub fn try_spherical_bessel_y(order: usize, x: f64) -> Result<Vec<f64>, SphericalFunctionError> {
    validate_order_argument(order, x)?;
    let values = spherical_bessel_y(order, x);
    validate_real_result(&values, order)?;
    Ok(values)
}

/// Checked spherical Hankel values for `n = 0..order`.
pub fn try_spherical_hankel_first_kind(
    order: usize,
    x: f64,
    harmonic_factor: f64,
) -> Result<Vec<Complex64>, SphericalFunctionError> {
    if order < 2 {
        return Err(SphericalFunctionError::EmptyOrder);
    }
    if !x.is_finite() || x <= 0.0 {
        return Err(SphericalFunctionError::InvalidArgument(x));
    }
    if harmonic_factor != 1.0 && harmonic_factor != -1.0 {
        return Err(SphericalFunctionError::InvalidHarmonicFactor(
            harmonic_factor,
        ));
    }
    let values = spherical_hankel_first_kind(order, x, harmonic_factor);
    validate_complex_result(&values, order)?;
    Ok(values)
}

/// Checked derivative of spherical Bessel `j_n`.
pub fn try_spherical_bessel_j_derivative(
    order: usize,
    x: f64,
) -> Result<Vec<f64>, SphericalFunctionError> {
    validate_order_argument(order, x)?;
    let values = spherical_bessel_j_derivative(order, x);
    validate_real_result(&values, order)?;
    Ok(values)
}

/// Checked derivative of spherical Neumann `y_n`.
pub fn try_spherical_bessel_y_derivative(
    order: usize,
    x: f64,
) -> Result<Vec<f64>, SphericalFunctionError> {
    validate_order_argument(order, x)?;
    let values = spherical_bessel_y_derivative(order, x);
    validate_real_result(&values, order)?;
    Ok(values)
}

/// Compute spherical Bessel functions j_n(x) for n = 0, 1, ..., order-1
///
/// Uses Miller's downward recurrence for numerical stability, which is
/// essential when n > x. The recurrence relation is:
/// ```text
/// j_{n-1}(x) = (2n+1)/x * j_n(x) - j_{n+1}(x)
/// ```
///
/// Normalization uses j_0(x) = sin(x)/x.
///
/// # Arguments
/// * `order` - Number of terms (returns j_0 through j_{order-1})
/// * `x` - Argument (must be > 0)
///
/// # Panics
///
/// Panics if `order < 1`. Use a checked wrapper where the argument is
/// user-controlled.
///
/// # Example
/// ```
/// use sonium_wave::special::spherical::spherical_bessel_j;
/// let j = spherical_bessel_j(5, 1.0);
/// // j[0] = sin(1)/1 ≈ 0.8414709848
/// ```
pub fn spherical_bessel_j(order: usize, x: f64) -> Vec<f64> {
    assert!(order >= 1, "Order must be at least 1");

    let mut result = vec![0.0; order];
    let mut workspace = Vec::new();
    spherical_bessel_j_into_unchecked(order, x, &mut result, &mut workspace);
    result
}

fn spherical_bessel_j_into_unchecked(
    order: usize,
    x: f64,
    result: &mut [f64],
    workspace: &mut Vec<f64>,
) {
    debug_assert_eq!(result.len(), order);
    // The legacy wrapper starts from a newly zeroed vector.  Callers of the
    // reusable API do not, so establish the same result invariant before any
    // small-argument early return.
    result.fill(0.0);

    // Handle very small x
    if x.abs() < 1e-15 {
        result[0] = 1.0;
        return;
    }

    if x.abs() < 1e-10 {
        // Series expansion for small x
        result[0] = 1.0 - x * x / 6.0;
        if order > 1 {
            result[1] = x / 3.0;
        }
        for item in result.iter_mut().take(order).skip(2) {
            *item = 0.0;
        }
        return;
    }

    // Miller's downward recurrence algorithm
    // Start from N >> order and x
    let start_n = order + (x.abs() as usize) + 20;

    workspace.resize(start_n + 2, 0.0);
    workspace[start_n + 1] = 0.0;
    workspace[start_n] = 1e-30; // Arbitrary small starting value

    // Downward recurrence: j_{n-1} = (2n+1)/x * j_n - j_{n+1}
    for k in (0..start_n).rev() {
        workspace[k] = (2 * k + 3) as f64 / x * workspace[k + 1] - workspace[k + 2];
    }

    // Normalize with whichever of j_0 or j_1 is farther from zero. Using
    // j_0 alone is singular at every non-zero root of sin(x), even though
    // higher-order spherical Bessel functions remain well-defined there.
    let true_j0 = x.sin() / x;
    let true_j1 = x.sin() / (x * x) - x.cos() / x;
    let scale = if true_j0.abs() >= true_j1.abs() {
        true_j0 / workspace[0]
    } else {
        true_j1 / workspace[1]
    };

    for n in 0..order {
        result[n] = workspace[n] * scale;
    }
}

/// Compute spherical Bessel functions y_n(x) (Neumann functions) for n = 0, 1, ..., order-1
///
/// Uses upward recurrence, which is stable for y_n:
/// ```text
/// y_{n+1}(x) = (2n+1)/x * y_n(x) - y_{n-1}(x)
/// ```
///
/// Starting values:
/// ```text
/// y_0(x) = -cos(x)/x
/// y_1(x) = -cos(x)/x² - sin(x)/x
/// ```
///
/// # Arguments
/// * `order` - Number of terms (returns y_0 through y_{order-1})
/// * `x` - Argument (must be > 0)
pub fn spherical_bessel_y(order: usize, x: f64) -> Vec<f64> {
    assert!(order >= 1, "Order must be at least 1");

    let mut result = vec![0.0; order];

    if x.abs() < 1e-15 {
        // y_n → -∞ as x → 0
        for item in result.iter_mut().take(order) {
            *item = f64::NEG_INFINITY;
        }
        return result;
    }

    let cos_x = x.cos();
    let sin_x = x.sin();

    // y_0(x) = -cos(x)/x
    result[0] = -cos_x / x;

    if order == 1 {
        return result;
    }

    // y_1(x) = -cos(x)/x² - sin(x)/x
    result[1] = -cos_x / (x * x) - sin_x / x;

    // Upward recurrence: y_{n+1} = (2n+1)/x * y_n - y_{n-1}
    for n in 2..order {
        result[n] = (2 * n - 1) as f64 / x * result[n - 1] - result[n - 2];
    }

    result
}

/// Compute spherical Hankel functions of the first kind h_n^(1)(x) for n = 0, 1, ..., order-1
///
/// Uses the algorithm from NC_SphericalHankel in NC_3dFunctions.cpp.
///
/// The spherical Hankel function is:
/// ```text
/// h_n^(1)(x) = j_n(x) + i * y_n(x)
/// ```
///
/// For the real part, uses Giebermann's continued fraction method which is
/// more numerically stable than direct computation for large orders.
///
/// # Arguments
/// * `order` - Number of terms (must be >= 2)
/// * `x` - Argument (must be > 0)
/// * `harmonic_factor` - +1 or -1 for time convention
///
/// # Returns
/// Vector of Complex64 values h_n^(1)(x) for n = 0, ..., order-1
///
/// # Panics
///
/// Panics if `order < 2` or `x <= 0.0`. Use a checked wrapper where the
/// argument is user-controlled.
pub fn spherical_hankel_first_kind(order: usize, x: f64, harmonic_factor: f64) -> Vec<Complex64> {
    assert!(order >= 2, "Order must be at least 2");
    assert!(x > 0.0, "Argument must be positive");

    let mut result = vec![Complex64::new(0.0, 0.0); order];

    // Compute imaginary part (y_n) using direct formula
    let cos_x = x.cos();
    let sin_x = x.sin();

    let mut y_n = vec![0.0; order];
    y_n[0] = -cos_x / x;
    y_n[1] = -(cos_x / x + sin_x) / x;

    for n in 2..order {
        y_n[n] = (2 * n - 1) as f64 / x * y_n[n - 1] - y_n[n - 2];
    }

    // Compute real part (j_n) using Giebermann's continued fraction method
    // This is the algorithm from NC_SphericalHankel
    let nu = (order - 1) as f64;
    let eps_gn = 1e-9;

    // Continued fraction for g_N = j'_N / j_N
    let mut di = (2.0 * (nu + 1.0) + 1.0) / x;
    let mut cj = di;
    let mut dj = 0.0;
    let mut err_gn = 1.0;
    let mut j = 1;

    while err_gn > eps_gn {
        let aj = -1.0;
        let bj = (2.0 * (nu + j as f64 + 1.0) + 1.0) / x;

        dj = bj + aj * dj;
        if dj == 0.0 {
            dj = 1e-30;
        }
        dj = 1.0 / dj;

        cj = bj + aj / cj;
        if cj == 0.0 {
            cj = 1e-30;
        }

        di = di * cj * dj;
        err_gn = (cj * dj - 1.0).abs();
        j += 1;

        if j > 1000 {
            // Safety limit to prevent infinite loop
            break;
        }
    }

    let gnu = nu / x - 1.0 / di;

    // Compute g_n and j_n by downward recurrence
    let mut gg_n = vec![0.0; order];
    let mut dg_n = vec![0.0; order];

    gg_n[order - 1] = 1.0;
    dg_n[order - 1] = gnu;

    for i in (0..order - 1).rev() {
        let di = i as f64;
        gg_n[i] = (di + 2.0) / x * gg_n[i + 1] + dg_n[i + 1];
        dg_n[i] = di / x * gg_n[i] - gg_n[i + 1];
    }

    // Normalize using known value
    let dp = if gg_n[0].abs() > 1e-5 {
        sin_x / x / gg_n[0]
    } else {
        (cos_x - sin_x / x) / x / dg_n[0]
    };

    // Assemble complex result
    for n in 0..order {
        result[n] = Complex64::new(dp * gg_n[n], harmonic_factor * y_n[n]);
    }

    result
}

/// Compute derivative of spherical Bessel j_n'(x)
///
/// Uses the recurrence relation:
/// ```text
/// j_n'(x) = j_{n-1}(x) - (n+1)/x * j_n(x)
/// ```
///
/// Requires `x != 0`: at `x = 0` the `(n+1)/x` term yields inf/NaN for
/// `n >= 1`. Validate the argument (or use a checked wrapper) before calling.
pub fn spherical_bessel_j_derivative(order: usize, x: f64) -> Vec<f64> {
    debug_assert!(
        x != 0.0,
        "spherical_bessel_j_derivative requires x != 0 (division by x)"
    );
    let j = spherical_bessel_j(order + 1, x);
    let mut result = vec![0.0; order];

    for n in 0..order {
        if n == 0 {
            // j_0' = -j_1
            result[0] = -j[1];
        } else {
            result[n] = j[n - 1] - (n + 1) as f64 / x * j[n];
        }
    }

    result
}

/// Compute derivative of spherical Bessel y_n'(x)
///
/// Uses the recurrence relation:
/// ```text
/// y_n'(x) = y_{n-1}(x) - (n+1)/x * y_n(x)
/// ```
///
/// Requires `x != 0`: at `x = 0` the `(n+1)/x` term yields inf/NaN for
/// `n >= 1`. Validate the argument (or use a checked wrapper) before calling.
pub fn spherical_bessel_y_derivative(order: usize, x: f64) -> Vec<f64> {
    debug_assert!(
        x != 0.0,
        "spherical_bessel_y_derivative requires x != 0 (division by x)"
    );
    let y = spherical_bessel_y(order + 1, x);
    let mut result = vec![0.0; order];

    for n in 0..order {
        if n == 0 {
            // y_0' = -y_1
            result[0] = -y[1];
        } else {
            result[n] = y[n - 1] - (n + 1) as f64 / x * y[n];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_spherical_bessel_j0() {
        // j_0(x) = sin(x)/x
        let j = spherical_bessel_j(1, 1.0);
        let expected = 1.0_f64.sin() / 1.0;
        assert!((j[0] - expected).abs() < EPSILON);

        let j = spherical_bessel_j(1, PI);
        let expected = PI.sin() / PI;
        assert!((j[0] - expected).abs() < EPSILON);
    }

    #[test]
    fn test_spherical_bessel_j1() {
        // j_1(x) = sin(x)/x² - cos(x)/x
        let x = 2.0;
        let j = spherical_bessel_j(2, x);
        let expected = x.sin() / (x * x) - x.cos() / x;
        assert!((j[1] - expected).abs() < EPSILON);
    }

    #[test]
    fn checked_bessel_j_into_matches_wrapper_and_reuses_workspace() {
        let order = 12;
        let mut output = vec![0.0; order];
        let mut workspace = Vec::new();

        try_spherical_bessel_j_into(order, 2.5, &mut output, &mut workspace)
            .expect("checked recurrence");
        assert_eq!(output, spherical_bessel_j(order, 2.5));
        let capacity = workspace.capacity();

        try_spherical_bessel_j_into(order, 1.25, &mut output, &mut workspace)
            .expect("reused checked recurrence");
        assert_eq!(output, spherical_bessel_j(order, 1.25));
        assert_eq!(workspace.capacity(), capacity);
    }

    #[test]
    fn checked_bessel_j_into_rejects_wrong_output_length() {
        let mut output = [0.0; 3];
        assert_eq!(
            try_spherical_bessel_j_into(4, 1.0, &mut output, &mut Vec::new()),
            Err(SphericalFunctionError::OutputLength {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn checked_bessel_j_into_clears_reused_output_at_tiny_argument() {
        let order = 5;
        let mut output = vec![0.0; order];
        let mut workspace = Vec::new();
        try_spherical_bessel_j_into(order, 2.5, &mut output, &mut workspace)
            .expect("nonzero initial evaluation");

        try_spherical_bessel_j_into(order, 1e-16, &mut output, &mut workspace)
            .expect("tiny-argument evaluation");
        assert_eq!(output, vec![1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn checked_bessel_j_batch_into_has_row_major_parity_and_reuses_workspace() {
        let order = 6;
        let arguments = [0.25, 1.25, 2.5];
        let mut output = vec![0.0; arguments.len() * order];
        let mut workspace = Vec::new();

        try_spherical_bessel_j_batch_into(order, &arguments, &mut output, &mut workspace)
            .expect("batch recurrence");
        for (&x, row) in arguments.iter().zip(output.chunks_exact(order)) {
            assert_eq!(row, spherical_bessel_j(order, x));
        }
        let capacity = workspace.capacity();

        try_spherical_bessel_j_batch_into(order, &[1.5], &mut output[..order], &mut workspace)
            .expect("reused batch recurrence");
        assert_eq!(workspace.capacity(), capacity);
    }

    #[test]
    fn checked_bessel_j_batch_into_is_fail_closed_before_writing_output() {
        let mut output = [42.0; 4];
        assert_eq!(
            try_spherical_bessel_j_batch_into(2, &[1.0, 0.0], &mut output, &mut Vec::new()),
            Err(SphericalFunctionError::InvalidArgument(0.0))
        );
        assert_eq!(output, [42.0; 4]);
        assert_eq!(
            try_spherical_bessel_j_batch_into(2, &[1.0], &mut output, &mut Vec::new()),
            Err(SphericalFunctionError::OutputLength {
                expected: 2,
                actual: 4,
            })
        );
    }

    #[test]
    fn test_spherical_bessel_y0() {
        // y_0(x) = -cos(x)/x
        let x = 1.0;
        let y = spherical_bessel_y(1, x);
        let expected = -x.cos() / x;
        assert!((y[0] - expected).abs() < EPSILON);
    }

    #[test]
    fn test_spherical_bessel_y1() {
        // y_1(x) = -cos(x)/x² - sin(x)/x
        let x = 2.0;
        let y = spherical_bessel_y(2, x);
        let expected = -x.cos() / (x * x) - x.sin() / x;
        assert!((y[1] - expected).abs() < EPSILON);
    }

    #[test]
    fn test_spherical_hankel_consistency() {
        // h_n^(1) = j_n + i*y_n
        let x = 3.0;
        let order = 5;
        let j = spherical_bessel_j(order, x);
        let y = spherical_bessel_y(order, x);
        let h = spherical_hankel_first_kind(order, x, 1.0);

        for n in 0..order {
            assert!(
                (h[n].re - j[n]).abs() < 1e-8,
                "Real part mismatch at n={}: {} vs {}",
                n,
                h[n].re,
                j[n]
            );
            assert!(
                (h[n].im - y[n]).abs() < 1e-8,
                "Imag part mismatch at n={}: {} vs {}",
                n,
                h[n].im,
                y[n]
            );
        }
    }

    #[test]
    fn test_hankel_asymptotic() {
        // For large x, h_n^(1)(x) → (-i)^{n+1} * exp(ix)/x
        let x = 50.0;
        let h = spherical_hankel_first_kind(3, x, 1.0);

        // h_0(x) → -i * exp(ix)/x = (sin(x) - i*cos(x))/x for large x
        let expected_re = x.sin() / x;
        let expected_im = -x.cos() / x;

        assert!(
            (h[0].re - expected_re).abs() < 0.01,
            "Asymptotic real mismatch"
        );
        assert!(
            (h[0].im - expected_im).abs() < 0.01,
            "Asymptotic imag mismatch"
        );
    }

    #[test]
    fn test_bessel_derivative_j0() {
        // j_0'(x) = -j_1(x)
        let x = 2.0;
        let jp = spherical_bessel_j_derivative(1, x);
        let j = spherical_bessel_j(2, x);
        assert!((jp[0] + j[1]).abs() < EPSILON);
    }

    #[test]
    fn test_recurrence_stability() {
        // Test that computation is stable for order > x
        let x = 5.0;
        let order = 20;
        let j = spherical_bessel_j(order, x);

        // All values should be finite
        for (n, &jn) in j.iter().enumerate().take(order) {
            assert!(jn.is_finite(), "j_{} is not finite", n);
        }

        // Values should decrease for n >> x
        assert!(j[15].abs() < j[5].abs());
    }

    #[test]
    fn miller_normalization_remains_finite_at_j0_root() {
        let x = PI;
        let values = try_spherical_bessel_j(6, x).expect("valid root-regime argument");
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values[0].abs() < 1e-14);
        assert!((values[1] - 1.0 / PI).abs() < 1e-12);
        assert!((values[2] - 3.0 / (PI * PI)).abs() < 1e-12);
    }

    #[test]
    fn checked_api_rejects_invalid_order_argument_and_sign() {
        assert_eq!(
            try_spherical_bessel_j(0, 1.0),
            Err(SphericalFunctionError::EmptyOrder)
        );
        assert_eq!(
            try_spherical_bessel_y(2, 0.0),
            Err(SphericalFunctionError::InvalidArgument(0.0))
        );
        assert_eq!(
            try_spherical_hankel_first_kind(2, 1.0, 0.0),
            Err(SphericalFunctionError::InvalidHarmonicFactor(0.0))
        );
    }

    #[test]
    fn checked_api_matches_legacy_values() {
        let j = try_spherical_bessel_j(8, 2.5).unwrap();
        assert_eq!(j, spherical_bessel_j(8, 2.5));
        let y = try_spherical_bessel_y(8, 2.5).unwrap();
        assert_eq!(y, spherical_bessel_y(8, 2.5));
        let h = try_spherical_hankel_first_kind(8, 2.5, -1.0).unwrap();
        assert_eq!(h, spherical_hankel_first_kind(8, 2.5, -1.0));
    }
}
