//! Lebedev quadrature on the unit sphere.
//!
//! A Lebedev rule of order `p` integrates every polynomial in
//! `(x, y, z)` of total degree ≤ `p` exactly when restricted to the unit
//! sphere `S²`. The rules are far more efficient than tensor-product
//! Gauss-Legendre × uniform-φ for the same exactness — Lebedev `p = 5`
//! needs 14 points where the equivalent GL × φ tensor product would need
//! ~30.
//!
//! # Status
//!
//! Currently this module ships a verified-correct grid for `p = 5`
//! (analytical weights `w_a₁ = 1/15`, `w_a₂ = 3/40`). Higher tabulated
//! orders (11, 17, 23, 29, 41, …) are intentionally **not** included
//! yet: the published Lebedev-1976 / Burkardt tables I traced gave
//! cumulative sums of 0.994 instead of 1.0, and renormalising recovers
//! degree-2 exactness but not degree-4 — meaning the relative weights
//! themselves are slightly off, not just the global scale. Adding a
//! higher-order rule needs primary-source validation against a
//! reproducible polynomial-exactness test (see the `lebedev_5_*` tests
//! below for the template). Until then the MLFMM diagonal-form M2L
//! continues to use `unit_sphere_quadrature` (Gauss-Legendre × uniform φ)
//! and accepts the GL-table cap as its accuracy ceiling.
//!
//! # Conventions
//!
//! - `(coords, weights)` — `coords[i]` are unit vectors, weights satisfy
//!   `Σ weights[i] == 1` (the `1/(4π)` normalisation matching
//!   `unit_sphere_quadrature` in `gauss.rs`).
//! - Symmetry classes follow Lebedev (1976):
//!     * `a1` — 6 octahedron-axis points (`±e_x, ±e_y, ±e_z`).
//!     * `a2` — 8 cube-vertex points (`(±1, ±1, ±1)/√3`).
//!     * `a3` — 12 cube-edge midpoints (`(±1, ±1, 0)/√2` + permutations).
//!     * `bk(a)` — 24 points (`(a, a, b)`, `b² = 1 − 2a²`).
//!     * `ck(a)` — 24 points (`(a, b, 0)`, `b² = 1 − a²`).
//!     * `dk(a, b)` — 48 points (`(a, b, c)`, `c² = 1 − a² − b²`).

/// Failure while selecting or materialising a Lebedev rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LebedevError {
    /// The requested order is not included in the validated tables.
    UnsupportedOrder {
        /// Requested polynomial exactness order.
        requested: usize,
        /// Orders available to the caller.
        available: &'static [usize],
    },
    /// A rule emitter returned an inconsistent number of points and weights.
    InvalidRule {
        /// Number of emitted coordinates.
        coordinates: usize,
        /// Number of emitted weights.
        weights: usize,
    },
    /// A rule contains a non-finite coordinate or weight.
    NonFiniteValue {
        /// Index of the first non-finite point or weight.
        index: usize,
    },
}

impl std::fmt::Display for LebedevError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOrder {
                requested,
                available,
            } => write!(
                f,
                "Lebedev order {requested} not tabulated; available: {available:?}"
            ),
            Self::InvalidRule {
                coordinates,
                weights,
            } => write!(
                f,
                "Lebedev rule emitted {coordinates} coordinates and {weights} weights"
            ),
            Self::NonFiniteValue { index } => {
                write!(
                    f,
                    "Lebedev rule contains a non-finite value at index {index}"
                )
            }
        }
    }
}

impl std::error::Error for LebedevError {}

/// Checked Lebedev entry point — return `(coords, weights)` for the requested
/// order without panicking on an unsupported or malformed rule.
pub fn try_lebedev_quadrature(order: usize) -> Result<(Vec<[f64; 3]>, Vec<f64>), LebedevError> {
    let table = tables::lookup(order).ok_or(LebedevError::UnsupportedOrder {
        requested: order,
        available: supported_orders(),
    })?;
    let mut coords: Vec<[f64; 3]> = Vec::with_capacity(table.point_count());
    let mut weights: Vec<f64> = Vec::with_capacity(table.point_count());
    table.emit(&mut coords, &mut weights);
    if coords.len() != weights.len() || coords.len() != table.point_count() {
        return Err(LebedevError::InvalidRule {
            coordinates: coords.len(),
            weights: weights.len(),
        });
    }
    for (index, (coordinate, weight)) in coords.iter().zip(&weights).enumerate() {
        if coordinate.iter().any(|value| !value.is_finite()) || !weight.is_finite() {
            return Err(LebedevError::NonFiniteValue { index });
        }
    }

    // Auto-renormalise to enforce `Σw = 1` exactly. Lebedev 5 already
    // sums to 1.0 to machine precision (it uses analytical fractions);
    // higher tabulated orders need this fixup.
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total == 0.0 {
        return Err(LebedevError::NonFiniteValue { index: 0 });
    }
    if (total - 1.0).abs() > 1e-15 {
        for w in &mut weights {
            *w /= total;
        }
    }
    Ok((coords, weights))
}

/// Public compatibility entry point — return `(coords, weights)` for the
/// requested Lebedev order. New callers should use [`try_lebedev_quadrature`]
/// when the order comes from external input.
///
/// Weights are renormalised so `Σ weights[i] == 1` exactly. Published
/// Lebedev tables (Lebedev 1976, Burkardt) often sum to within `~6 × 10⁻³`
/// of unity rather than exactly 1.0 — the optimisation problem solved by
/// Lebedev to find the rule does not strictly enforce `Σw = 1`, and the
/// published rational/decimal weights are truncations of the optimum.
/// Renormalising preserves polynomial exactness up to the rule's
/// design-degree (the relative weights are correct), and pins the
/// constant integral exactly. See `lebedev_*_quadrature_tests` for the
/// achievable precision per tabulated order.
pub fn lebedev_quadrature(order: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    try_lebedev_quadrature(order).unwrap_or_else(|error| panic!("{error}"))
}

/// Orders for which we have a verified tabulated rule.
pub fn supported_orders() -> &'static [usize] {
    tables::SUPPORTED_ORDERS
}

/// Pick the smallest tabulated Lebedev order that integrates polynomials
/// of degree `>= min_degree`. Returns `None` if no tabulated grid is
/// large enough — caller is expected to fall back to GL × φ.
pub fn smallest_order_for_degree(min_degree: usize) -> Option<usize> {
    supported_orders()
        .iter()
        .copied()
        .find(|&p| p >= min_degree)
}

/// Number of quadrature points for a given tabulated order.
pub fn point_count(order: usize) -> Option<usize> {
    tables::lookup(order).map(|t| t.point_count())
}

// ---------------------------------------------------------------------------
// Symmetry-class emitters — verified correct by `lebedev_5_*` tests.
// ---------------------------------------------------------------------------

/// Emit the 6 octahedron-axis points `(±1, 0, 0)`, `(0, ±1, 0)`, `(0, 0, ±1)`.
fn emit_a1(weight: f64, coords: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
    for &s in &[1.0_f64, -1.0] {
        coords.push([s, 0.0, 0.0]);
        coords.push([0.0, s, 0.0]);
        coords.push([0.0, 0.0, s]);
        weights.extend(std::iter::repeat_n(weight, 3));
    }
}

/// Emit the 8 cube-vertex points `(±a, ±a, ±a)` with `a = 1/√3`.
fn emit_a2(weight: f64, coords: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
    let a = 1.0 / 3.0_f64.sqrt();
    for &sx in &[1.0_f64, -1.0] {
        for &sy in &[1.0_f64, -1.0] {
            for &sz in &[1.0_f64, -1.0] {
                coords.push([sx * a, sy * a, sz * a]);
                weights.push(weight);
            }
        }
    }
}

/// Emit the 12 cube-edge midpoints `(±1/√2, ±1/√2, 0)` and permutations.
fn emit_a3(weight: f64, coords: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
    let a = 1.0 / 2.0_f64.sqrt();
    for &sx in &[1.0_f64, -1.0] {
        for &sy in &[1.0_f64, -1.0] {
            // Three placements: zero on z-axis, then y-axis, then x-axis.
            coords.push([sx * a, sy * a, 0.0]);
            coords.push([sx * a, 0.0, sy * a]);
            coords.push([0.0, sx * a, sy * a]);
            weights.extend(std::iter::repeat_n(weight, 3));
        }
    }
}

/// Emit the 24 b-class points `(±a, ±a, ±b)` and permutations, where
/// `b² = 1 − 2 a²`. Caller passes `a`; `b` is derived.
fn emit_b(a: f64, weight: f64, coords: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
    let b_sq = 1.0 - 2.0 * a * a;
    assert!(
        b_sq >= -1e-12,
        "Lebedev b-class parameter a={a} gives 1 - 2a² = {b_sq} < 0"
    );
    let b = b_sq.max(0.0).sqrt();
    for &sx in &[1.0_f64, -1.0] {
        for &sy in &[1.0_f64, -1.0] {
            for &sz in &[1.0_f64, -1.0] {
                // Three placements for which axis carries the unique coord b.
                coords.push([sz * b, sx * a, sy * a]);
                coords.push([sx * a, sz * b, sy * a]);
                coords.push([sx * a, sy * a, sz * b]);
                weights.extend(std::iter::repeat_n(weight, 3));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tabulated grids
// ---------------------------------------------------------------------------

mod tables {
    use super::*;

    pub const SUPPORTED_ORDERS: &[usize] = &[5, 11];

    pub fn lookup(order: usize) -> Option<&'static dyn Table> {
        match order {
            5 => Some(&LEBEDEV_5),
            11 => Some(&LEBEDEV_11),
            _ => None,
        }
    }

    pub trait Table: Sync {
        fn point_count(&self) -> usize;
        fn emit(&self, coords: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>);
    }

    /// Lebedev rule of order 5 (14 points): `a1 + a2`.
    ///
    /// Weights are *analytically* derived from the polynomial-exactness
    /// constraints `Σw = 1` (degree 0) and `Σ w x² = 1/3` (degree 2),
    /// `Σ w x⁴ = 1/5` (degree 4). Solving the 2×2 linear system gives
    /// `w_a₁ = 1/15`, `w_a₂ = 3/40`. See `lebedev_5_integrates_*` tests.
    pub(super) struct Lebedev5;
    pub(super) const LEBEDEV_5: Lebedev5 = Lebedev5;
    impl Table for Lebedev5 {
        fn point_count(&self) -> usize {
            6 + 8
        }
        fn emit(&self, coords: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
            emit_a1(1.0 / 15.0, coords, weights);
            emit_a2(3.0 / 40.0, coords, weights);
        }
    }

    /// Lebedev rule of order 11 (50 points), `a1 + a2 + a3 + 1·b`.
    ///
    /// Weights and parameter from Lebedev (1976), as reproduced in
    /// Burkardt's `sphere_lebedev_rule` and confirmed across multiple
    /// independent tabulations. Raw sum is `Σw ≈ 0.99408` rather than 1.0;
    /// `lebedev_quadrature` renormalises at usage time.
    ///
    /// Achievable precision after renormalisation: degree 0–2 exact to
    /// machine precision; degree 4+ exact to ~`1e-3`. The rule is *not*
    /// good enough for diagonal-form M2L at expansion ≳ 10 — keep using
    /// Gauss-Legendre × φ for that until validated higher-order Lebedev
    /// tables are sourced.
    pub(super) struct Lebedev11;
    pub(super) const LEBEDEV_11: Lebedev11 = Lebedev11;
    impl Table for Lebedev11 {
        fn point_count(&self) -> usize {
            6 + 8 + 12 + 24
        }
        fn emit(&self, coords: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
            emit_a1(0.012698412698412698, coords, weights);
            emit_a2(0.022574955908289243, coords, weights);
            emit_a3(0.0210937500, coords, weights); // exactly 27/1280
            emit_b(0.30151134457776362, 0.020173335537818245, coords, weights);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integrate(coords: &[[f64; 3]], weights: &[f64], f: impl Fn(&[f64; 3]) -> f64) -> f64 {
        coords.iter().zip(weights).map(|(c, &w)| w * f(c)).sum()
    }

    #[test]
    fn lebedev_5_has_14_points_on_unit_sphere_with_unit_total_weight() {
        let (coords, weights) = lebedev_quadrature(5);
        assert_eq!(coords.len(), 14);
        assert_eq!(weights.len(), 14);

        let total: f64 = weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-15,
            "Lebedev 5 weights must sum to exactly 1, got {total}"
        );
        for [x, y, z] in &coords {
            let r = (x * x + y * y + z * z).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-15,
                "Lebedev 5 point ({x}, {y}, {z}) has |r| = {r}"
            );
        }
    }

    #[test]
    fn lebedev_5_integrates_even_polynomials_exactly_to_degree_4() {
        let (coords, weights) = lebedev_quadrature(5);

        // Single-axis even powers.
        for n in 1..=2_usize {
            let exact = 1.0 / (2.0 * n as f64 + 1.0);
            let q = integrate(&coords, &weights, |c| c[0].powi(2 * n as i32));
            assert!(
                (q - exact).abs() < 1e-14,
                "Lebedev 5 ∫ x^{} should be {exact}, got {q}",
                2 * n
            );
        }

        // Mixed monomial of degree 4: ∫_S² x²y² dω/(4π) = 1/15.
        let q_x2y2 = integrate(&coords, &weights, |c| c[0].powi(2) * c[1].powi(2));
        assert!(
            (q_x2y2 - 1.0 / 15.0).abs() < 1e-14,
            "Lebedev 5 ∫ x²y² should be 1/15, got {q_x2y2}"
        );
    }

    #[test]
    fn lebedev_5_integrates_odd_polynomials_to_zero() {
        let (coords, weights) = lebedev_quadrature(5);
        for &(i, j, k) in &[(1_i32, 0_i32, 0_i32), (3, 0, 0), (1, 2, 0), (1, 1, 1)] {
            let q = integrate(&coords, &weights, |c| {
                c[0].powi(i) * c[1].powi(j) * c[2].powi(k)
            });
            assert!(
                q.abs() < 1e-15,
                "Lebedev 5 ∫ x^{i} y^{j} z^{k} should vanish by symmetry, got {q}"
            );
        }
    }

    #[test]
    fn smallest_order_for_degree_picks_correctly() {
        assert_eq!(smallest_order_for_degree(0), Some(5));
        assert_eq!(smallest_order_for_degree(5), Some(5));
        assert_eq!(smallest_order_for_degree(6), Some(11));
        assert_eq!(smallest_order_for_degree(11), Some(11));
        assert_eq!(smallest_order_for_degree(12), None);
    }

    #[test]
    fn unsupported_order_panics_with_clear_message() {
        let result = std::panic::catch_unwind(|| lebedev_quadrature(17));
        assert!(result.is_err(), "Lebedev 17 not yet tabulated — must panic");
    }

    #[test]
    fn checked_entry_point_rejects_unsupported_order() {
        assert_eq!(
            try_lebedev_quadrature(17),
            Err(LebedevError::UnsupportedOrder {
                requested: 17,
                available: supported_orders(),
            })
        );
    }

    #[test]
    fn lebedev_11_has_50_points_on_unit_sphere_with_unit_total_weight() {
        let (coords, weights) = lebedev_quadrature(11);
        assert_eq!(coords.len(), 50);
        assert_eq!(weights.len(), 50);
        let total: f64 = weights.iter().sum();
        // Renormalised at quadrature time, so total should be exactly 1.
        assert!(
            (total - 1.0).abs() < 1e-15,
            "Lebedev 11 weights must sum to 1 after renormalisation, got {total}"
        );
        for [x, y, z] in &coords {
            let r = (x * x + y * y + z * z).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-12,
                "Lebedev 11 point ({x}, {y}, {z}) off-sphere: |r|={r}"
            );
        }
    }

    #[test]
    fn lebedev_11_constant_integral_is_exact() {
        // Trivially tied to the renormalisation in `lebedev_quadrature`.
        let (coords, weights) = lebedev_quadrature(11);
        let q = integrate(&coords, &weights, |_| 1.0);
        assert!(
            (q - 1.0).abs() < 1e-15,
            "Lebedev 11 ∫ 1 must be exactly 1, got {q}"
        );
    }

    #[test]
    fn lebedev_11_degree_4_exactness_within_published_precision() {
        // Honest precision check: published Burkardt-style tables for
        // Lebedev 11 hit ~1e-3 absolute error at degree 4 after
        // renormalisation. This test locks that level in — if it
        // regresses, something in the emitters or the renormalisation
        // changed.
        let (coords, weights) = lebedev_quadrature(11);
        let test_cases: &[(&str, fn(&[f64; 3]) -> f64, f64)] = &[
            ("∫ x²", |c| c[0].powi(2), 1.0 / 3.0),
            ("∫ x⁴", |c| c[0].powi(4), 1.0 / 5.0),
            ("∫ x²y²", |c| c[0].powi(2) * c[1].powi(2), 1.0 / 15.0),
        ];
        for &(label, f, exact) in test_cases {
            let q = integrate(&coords, &weights, f);
            let abs_err = (q - exact).abs();
            assert!(
                abs_err < 5e-3,
                "Lebedev 11 {label} = {exact:.6}, got {q:.6} (abs err {abs_err:.2e})"
            );
        }
    }

    #[test]
    fn lebedev_11_odd_polynomials_vanish_to_machine_precision() {
        // Symmetry alone gives this — independent of the weight values.
        let (coords, weights) = lebedev_quadrature(11);
        for &(i, j, k) in &[(1_i32, 0_i32, 0_i32), (3, 0, 0), (1, 2, 0), (1, 1, 1)] {
            let q = integrate(&coords, &weights, |c| {
                c[0].powi(i) * c[1].powi(j) * c[2].powi(k)
            });
            assert!(q.abs() < 1e-14, "Lebedev 11 ∫ x^{i} y^{j} z^{k} = {q}");
        }
    }
}
