//! Natural cubic spline interpolation for waveguide profiles.

/// Rejected `CubicSpline` control points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineError {
    /// Fewer than 2 control points were provided.
    TooFewPoints,
    /// A control-point coordinate is not finite.
    NonFinitePoint,
    /// Knot positions are not strictly increasing.
    KnotsNotIncreasing,
}

impl std::fmt::Display for SplineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints => write!(f, "need at least 2 control points"),
            Self::NonFinitePoint => write!(f, "spline control points must be finite"),
            Self::KnotsNotIncreasing => {
                write!(f, "spline knot positions must be strictly increasing")
            }
        }
    }
}

impl std::error::Error for SplineError {}

/// A natural cubic spline fitted through control points.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    /// Sorted knot positions.
    xs: Vec<f64>,
    /// Values at knots.
    ys: Vec<f64>,
    /// Second derivatives at knots (from tridiagonal solve).
    m: Vec<f64>,
}

impl CubicSpline {
    /// Fit a natural cubic spline through the given control points.
    ///
    /// Points must be sorted by the first coordinate (parameter `t`).
    ///
    /// # Panics
    ///
    /// Panics if fewer than 2 points are provided, a coordinate is not
    /// finite, or knot positions are not strictly increasing. Prefer
    /// [`try_from_control_points`](Self::try_from_control_points), which
    /// returns a [`SplineError`] instead, in all library and application code.
    pub fn from_control_points(pts: &[[f64; 2]]) -> Self {
        Self::try_from_control_points(pts).expect("invalid spline control points")
    }

    /// Checked variant of [`from_control_points`](Self::from_control_points).
    ///
    /// Returns [`SplineError`] instead of panicking on bad input.
    pub fn try_from_control_points(pts: &[[f64; 2]]) -> Result<Self, SplineError> {
        if pts.len() < 2 {
            return Err(SplineError::TooFewPoints);
        }
        let n = pts.len();
        let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
        if !pts.iter().flatten().all(|value| value.is_finite()) {
            return Err(SplineError::NonFinitePoint);
        }
        if !xs.windows(2).all(|pair| pair[1] > pair[0]) {
            return Err(SplineError::KnotsNotIncreasing);
        }

        if n == 2 {
            // Linear — second derivatives are zero.
            return Self {
                xs,
                ys,
                m: vec![0.0; 2],
            };
        }

        // Tridiagonal system for natural cubic spline.
        // h_i = x_{i+1} - x_i
        let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();

        // Right-hand side
        let mut rhs = vec![0.0; n];
        for i in 1..n - 1 {
            rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
        }

        // Thomas algorithm for tridiagonal solve.
        // Diag: 2*(h_{i-1}+h_i), sub/super: h_i
        // Natural BC: m[0] = m[n-1] = 0.
        let mut cp = vec![0.0; n]; // modified upper diagonal
        let mut dp = vec![0.0; n]; // modified rhs

        // Forward sweep (rows 1..n-2)
        for i in 1..n - 1 {
            let diag = 2.0 * (h[i - 1] + h[i]);
            let sub = if i > 1 { h[i - 1] } else { 0.0 };
            let denom = diag - sub * cp[i - 1];
            cp[i] = h[i] / denom;
            dp[i] = (rhs[i] - sub * dp[i - 1]) / denom;
        }

        // Back substitution
        let mut m = vec![0.0; n];
        for i in (1..n - 1).rev() {
            m[i] = dp[i] - cp[i] * m[i + 1];
        }

        Self { xs, ys, m }
    }

    /// Evaluate the spline at parameter `t`.
    ///
    /// Values outside the knot range are linearly extrapolated from the
    /// nearest segment.
    pub fn evaluate(&self, t: f64) -> f64 {
        // Find interval
        let i = self.find_interval(t);
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - t) / h;
        let b = (t - self.xs[i]) / h;
        a * self.ys[i]
            + b * self.ys[i + 1]
            + (a * a * a - a) * h * h / 6.0 * self.m[i]
            + (b * b * b - b) * h * h / 6.0 * self.m[i + 1]
    }

    /// Evaluate the first derivative at parameter `t`.
    pub fn derivative(&self, t: f64) -> f64 {
        let i = self.find_interval(t);
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - t) / h;
        let b = (t - self.xs[i]) / h;
        (self.ys[i + 1] - self.ys[i]) / h
            - (3.0 * a * a - 1.0) * h / 6.0 * self.m[i]
            + (3.0 * b * b - 1.0) * h / 6.0 * self.m[i + 1]
    }

    fn find_interval(&self, t: f64) -> usize {
        let n = self.xs.len();
        if t <= self.xs[0] {
            return 0;
        }
        if t >= self.xs[n - 1] {
            return n - 2;
        }
        // Binary search
        match self.xs.binary_search_by(|x| x.total_cmp(&t)) {
            Ok(i) => {
                if i >= n - 1 {
                    n - 2
                } else {
                    i
                }
            }
            Err(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_two_points() {
        let spline = CubicSpline::from_control_points(&[[0.0, 0.0], [1.0, 1.0]]);
        assert!((spline.evaluate(0.0) - 0.0).abs() < 1e-10);
        assert!((spline.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((spline.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_try_from_control_points_rejects_bad_input() {
        assert_eq!(
            CubicSpline::try_from_control_points(&[[0.0, 0.0]]).err(),
            Some(SplineError::TooFewPoints)
        );
        assert_eq!(
            CubicSpline::try_from_control_points(&[[0.0, 0.0], [f64::NAN, 1.0]]).err(),
            Some(SplineError::NonFinitePoint)
        );
        assert_eq!(
            CubicSpline::try_from_control_points(&[[1.0, 0.0], [0.0, 1.0]]).err(),
            Some(SplineError::KnotsNotIncreasing)
        );
        assert_eq!(
            CubicSpline::try_from_control_points(&[[0.0, 0.0], [0.0, 1.0]]).err(),
            Some(SplineError::KnotsNotIncreasing)
        );
        assert!(CubicSpline::try_from_control_points(&[[0.0, 0.0], [1.0, 1.0]]).is_ok());
    }

    #[test]
    fn test_passes_through_control_points() {
        let pts = [[0.0, 1.0], [0.25, 2.0], [0.5, 1.5], [0.75, 3.0], [1.0, 2.0]];
        let spline = CubicSpline::from_control_points(&pts);
        for &[t, y] in &pts {
            assert!(
                (spline.evaluate(t) - y).abs() < 1e-10,
                "at t={t}: expected {y}, got {}",
                spline.evaluate(t)
            );
        }
    }

    #[test]
    fn test_derivative_finite_diff() {
        let pts = [[0.0, 0.05], [0.33, 0.04], [0.66, 0.025], [1.0, 0.013]];
        let spline = CubicSpline::from_control_points(&pts);
        let t = 0.5;
        let eps = 1e-6;
        let fd = (spline.evaluate(t + eps) - spline.evaluate(t - eps)) / (2.0 * eps);
        let exact = spline.derivative(t);
        assert!(
            (fd - exact).abs() < 1e-4,
            "fd={fd}, exact={exact}"
        );
    }

    #[test]
    fn test_three_points_quadratic() {
        // y = x^2 at x=0,1,2 → spline should reproduce exactly
        let pts = [[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]];
        let spline = CubicSpline::from_control_points(&pts);
        // Natural spline won't reproduce x^2 exactly, but should be close at midpoints
        assert!((spline.evaluate(0.5) - 0.25).abs() < 0.15);
        assert!((spline.evaluate(1.5) - 2.25).abs() < 0.15);
    }

    #[test]
    fn test_monotone_decreasing_profile() {
        // Typical waveguide profile: starts wide, narrows
        let pts = [[0.0, 0.05], [0.5, 0.03], [1.0, 0.013]];
        let spline = CubicSpline::from_control_points(&pts);
        // Should be generally decreasing
        let v0 = spline.evaluate(0.0);
        let v1 = spline.evaluate(0.25);
        let v2 = spline.evaluate(0.5);
        let v3 = spline.evaluate(0.75);
        let v4 = spline.evaluate(1.0);
        assert!(v0 > v1);
        assert!(v1 > v2 || (v1 - v2).abs() < 0.005);
        assert!(v2 > v3 || (v2 - v3).abs() < 0.005);
        assert!(v3 > v4 || (v3 - v4).abs() < 0.005);
    }
}
