//! Piecewise-linear function and table interpolation for crash laws.
//!
//! Contains the frozen Phase-3 readers for Radioss `/FUNCT` (1-D) and
//! `/TABLE` (2-D) curves: strict validation at construction, monotone
//! abscissae, clamped ends, and bilinear evaluation. Material lanes
//! consume this module read-only; deck-text parsing lives in the
//! lane-owned extension module built on top of these types.
//!
//! # Conventions
//!
//! - SI units flow through untouched; this module is unit-agnostic.
//! - Abscissae must be finite and strictly increasing.
//! - Evaluation outside the tabulated range holds the endpoint value
//!   (constant extension, the Radioss default).

// Rust guideline compliant 2026-02-21

/// One-dimensional piecewise-linear function (`/FUNCT` semantics).
///
/// Stores validated node pairs and evaluates with clamped ends.
/// Slopes are exact per segment for hardening-tangent use.
#[derive(Debug, Clone)]
pub struct Funct1D {
    /// Strictly increasing abscissae.
    x: Vec<f64>,
    /// Ordinates, one per abscissa.
    y: Vec<f64>,
}

impl Funct1D {
    /// Builds a function from node pairs.
    ///
    /// Rejects fewer than two nodes, non-finite values, and
    /// non-increasing abscissae so evaluation never divides by zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
    ///
    /// let f = Funct1D::new(vec![0.0, 1.0], vec![0.0, 2.0]).unwrap();
    /// assert_eq!(f.eval(0.25), 0.5);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a message when the node vectors are too short, differ
    /// in length, contain non-finite values, or abscissae repeat.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, String> {
        if x.len() != y.len() {
            return Err(format!(
                "funct node mismatch: {} abscissae vs {} ordinates",
                x.len(),
                y.len()
            ));
        }
        if x.len() < 2 {
            return Err("funct needs at least two nodes".to_string());
        }
        if !x.iter().chain(y.iter()).all(|v| v.is_finite()) {
            return Err("funct nodes must be finite".to_string());
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err("funct abscissae must strictly increase".to_string());
        }
        Ok(Self { x, y })
    }

    /// Evaluates the function with clamped ends.
    ///
    /// Interior points interpolate linearly; inputs below the first
    /// or above the last abscissa hold the endpoint ordinate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
    ///
    /// let f = Funct1D::new(vec![0.0, 1.0], vec![10.0, 20.0]).unwrap();
    /// assert_eq!(f.eval(-5.0), 10.0);
    /// assert_eq!(f.eval(5.0), 20.0);
    /// ```
    pub fn eval(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        if x <= self.x[0] {
            return self.y[0];
        }
        let last = self.x.len() - 1;
        if x >= self.x[last] {
            return self.y[last];
        }
        let upper = self.x.partition_point(|node| *node <= x);
        let lo = upper - 1;
        let span = self.x[upper] - self.x[lo];
        let frac = (x - self.x[lo]) / span;
        self.y[lo] + frac * (self.y[upper] - self.y[lo])
    }

    /// Returns the segment slope at the input.
    ///
    /// Uses the adjacent segment at and beyond the ends so tangent
    /// queries never fail; NaN input yields NaN.
    pub fn slope(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        let last = self.x.len() - 1;
        let lo = if x <= self.x[0] {
            0
        } else if x >= self.x[last] {
            last - 1
        } else {
            self.x.partition_point(|node| *node <= x) - 1
        };
        (self.y[lo + 1] - self.y[lo]) / (self.x[lo + 1] - self.x[lo])
    }

    /// Scales all ordinates by the factor.
    ///
    /// Supports strain-rate or temperature scaling of hardening curves
    /// without rebuilding the abscissa grid.
    pub fn scaled_ordinates(&self, factor: f64) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.iter().map(|v| v * factor).collect(),
        }
    }
}

/// Two-dimensional bilinear table (`/TABLE` semantics).
///
/// Row-major values over an (x, y) grid with clamped edges.
/// Consumed by laws with biaxial or rate-table hardening.
#[derive(Debug, Clone)]
pub struct Table2D {
    /// Strictly increasing x abscissae (columns).
    x: Vec<f64>,
    /// Strictly increasing y abscissae (rows).
    y: Vec<f64>,
    /// Row-major values, `values[row * nx + col]`.
    values: Vec<f64>,
}

impl Table2D {
    /// Builds a table from grid vectors and row-major values.
    ///
    /// Both grids need at least two strictly increasing finite nodes
    /// and exactly `nx * ny` finite values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::funct::Table2D;
    ///
    /// let t = Table2D::new(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0, 1.0, 2.0]).unwrap();
    /// assert_eq!(t.eval(0.5, 0.5), 1.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a message when grids are too short, not strictly
    /// increasing, non-finite, or the value count mismatches.
    pub fn new(x: Vec<f64>, y: Vec<f64>, values: Vec<f64>) -> Result<Self, String> {
        if x.len() < 2 || y.len() < 2 {
            return Err("table grids need at least two nodes each".to_string());
        }
        if !x.iter().chain(y.iter()).all(|v| v.is_finite()) {
            return Err("table grids must be finite".to_string());
        }
        if x.windows(2).any(|w| w[1] <= w[0]) || y.windows(2).any(|w| w[1] <= w[0]) {
            return Err("table grids must strictly increase".to_string());
        }
        if values.len() != x.len() * y.len() {
            return Err(format!(
                "table needs {} values, got {}",
                x.len() * y.len(),
                values.len()
            ));
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err("table values must be finite".to_string());
        }
        Ok(Self { x, y, values })
    }

    /// Evaluates the table with clamped edges.
    ///
    /// Bilinear within the grid; inputs outside hold the edge row,
    /// column, or corner value.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        let nx = self.x.len();
        let ny = self.y.len();
        let col = bracket(&self.x, x);
        let row = bracket(&self.y, y);
        let fx = if x <= self.x[0] || x >= self.x[nx - 1] {
            if x <= self.x[0] { 0.0 } else { 1.0 }
        } else {
            (x - self.x[col]) / (self.x[col + 1] - self.x[col])
        };
        let fy = if y <= self.y[0] || y >= self.y[ny - 1] {
            if y <= self.y[0] { 0.0 } else { 1.0 }
        } else {
            (y - self.y[row]) / (self.y[row + 1] - self.y[row])
        };
        let v00 = self.values[row * nx + col];
        let v10 = self.values[row * nx + col + 1];
        let v01 = self.values[(row + 1) * nx + col];
        let v11 = self.values[(row + 1) * nx + col + 1];
        v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy
    }
}

/// Locates the lower bracket index with clamped ends.
fn bracket(grid: &[f64], v: f64) -> usize {
    if v <= grid[0] {
        0
    } else if v >= grid[grid.len() - 1] {
        grid.len() - 2
    } else {
        grid.partition_point(|node| *node <= v) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn funct_rejects_bad_nodes() {
        assert!(Funct1D::new(vec![0.0], vec![1.0]).is_err());
        assert!(Funct1D::new(vec![0.0, 1.0], vec![1.0]).is_err());
        assert!(Funct1D::new(vec![0.0, 0.0], vec![1.0, 2.0]).is_err());
        assert!(Funct1D::new(vec![1.0, 0.0], vec![1.0, 2.0]).is_err());
        assert!(Funct1D::new(vec![0.0, f64::NAN], vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn funct_interpolates_and_clamps() {
        let f = Funct1D::new(vec![0.0, 1.0, 3.0], vec![0.0, 2.0, 3.0]).unwrap();
        assert_eq!(f.eval(0.5), 1.0);
        assert_eq!(f.eval(2.0), 2.5);
        assert_eq!(f.eval(-1.0), 0.0);
        assert_eq!(f.eval(9.0), 3.0);
        assert_eq!(f.slope(0.5), 2.0);
        assert_eq!(f.slope(2.0), 0.5);
        assert!(f.eval(f64::NAN).is_nan());
    }

    #[test]
    fn funct_scales_ordinates() {
        let f = Funct1D::new(vec![0.0, 1.0], vec![1.0, 3.0]).unwrap();
        let scaled = f.scaled_ordinates(2.0);
        assert_eq!(scaled.eval(0.5), 4.0);
        assert_eq!(f.eval(0.5), 2.0);
    }

    #[test]
    fn table_bilinear_and_edges() {
        let t = Table2D::new(vec![0.0, 2.0], vec![0.0, 4.0], vec![0.0, 2.0, 4.0, 6.0]).unwrap();
        assert_eq!(t.eval(1.0, 2.0), 3.0);
        assert_eq!(t.eval(0.0, 0.0), 0.0);
        assert_eq!(t.eval(-9.0, -9.0), 0.0);
        assert_eq!(t.eval(9.0, 9.0), 6.0);
        assert_eq!(t.eval(9.0, 2.0), 4.0);
        assert!(Table2D::new(vec![0.0], vec![0.0, 1.0], vec![0.0, 0.0]).is_err());
        assert!(Table2D::new(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0]).is_err());
    }
}
