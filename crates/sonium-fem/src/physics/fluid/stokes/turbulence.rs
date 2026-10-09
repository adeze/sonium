//! Turbulence closure for the Navier-Stokes Picard solver.
//!
//! Two models are supported:
//!
//! - [`TurbulenceModel::Laminar`] — no eddy viscosity, the classic
//!   `μ_eff = μ` used by every existing test.
//! - [`TurbulenceModel::KOmegaSstAlgebraic`] — an algebraic surrogate for
//!   the Menter k-ω SST eddy viscosity. Computes a globally-enhanced
//!   `μ_eff = μ + ρ·ν_t` from the current velocity-field strain-rate
//!   magnitude and a characteristic length scale, mimicking the SST
//!   blending behaviour (low ν_t in laminar regions of low strain, high
//!   ν_t in high-shear regions).
//!
//! The algebraic variant intentionally does NOT solve k and ω transport
//! equations — those would require two new scalar function spaces, two
//! more SUPG-stabilised assemblies, and Menter's blended wall functions
//! (~600 LOC in total). The algebraic model captures the first-order
//! effect of turbulence on bulk flow (reduced effective Reynolds number,
//! dampened spurious oscillations, stabilised Picard iteration) at a
//! fraction of the cost, and exposes the same configuration interface
//! (`TurbulenceModel`) so a full transport-equation k-ω SST can be
//! dropped in later as a new enum variant without churning callers. Its
//! current Newton tangent is exact only for a unique, non-zero global peak;
//! tied or zero peaks remain fail-closed at the resolved-adjoint boundary.

/// Active turbulence model used by the Navier-Stokes Picard solver.
///
/// `Laminar` matches the legacy behaviour exactly. `KOmegaSstAlgebraic`
/// computes an enhanced effective viscosity from the velocity field and
/// applies it as a scalar `μ_eff` at each Picard iteration.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum TurbulenceModel {
    /// No turbulence closure. `μ_eff = μ`.
    #[default]
    Laminar,
    /// Algebraic k-ω SST surrogate. Computes `ν_t = c1 · L · |U|` from
    /// the velocity field magnitude and a characteristic length scale,
    /// clamped to `ν_t ≤ ν_t_max_ratio · ν` to avoid runaway viscosity
    /// when the Picard iteration overshoots.
    KOmegaSstAlgebraic {
        /// Calibration coefficient. SST ranges typical: 0.05–0.15.
        c1: f64,
        /// Characteristic flow length (e.g. shroud radius or hub-to-tip
        /// span for a fan). Used to dimensionalise `ν_t` against the
        /// velocity scale extracted from the current iterate.
        characteristic_length_m: f64,
        /// Hard upper bound on `ν_t / ν`. Prevents pathological blow-up
        /// when `|U|` peaks in the first Picard iterations. Default is
        /// 200 (turbulent-to-laminar viscosity ratios above this are
        /// unphysical for sub-supersonic internal flow).
        nu_t_max_ratio: f64,
    },
}

impl TurbulenceModel {
    /// k-ω SST defaults that work for typical PC-fan-scale internal flows
    /// (Re ≈ 10⁴, characteristic length ≈ 0.05 m).
    pub fn pcfan_default() -> Self {
        Self::KOmegaSstAlgebraic {
            c1: 0.09,                       // SST a₁ value, reused as our magnitude scaler
            characteristic_length_m: 0.058, // shroud inner radius
            nu_t_max_ratio: 200.0,
        }
    }

    /// Compute the effective dynamic viscosity `μ_eff = μ + ρ·ν_t` for the
    /// current Picard iterate. `velocity_magnitude_max` is the peak
    /// velocity (m/s) in the current velocity field — typically computed
    /// once per iteration over the FE node values.
    pub fn effective_mu(&self, mu: f64, rho: f64, velocity_magnitude_max: f64) -> f64 {
        self.effective_mu_and_speed_derivative(mu, rho, velocity_magnitude_max)
            .0
    }

    /// Return effective viscosity and its derivative with respect to the
    /// supplied velocity scale.
    ///
    /// The algebraic PC-fan closure uses a scalar velocity scale.  Keeping
    /// the clamp derivative explicit lets a Newton assembly publish an exact
    /// tangent on either side of the cap; at the cap the closure is locally
    /// constant and the derivative is zero.
    pub fn effective_mu_and_speed_derivative(
        &self,
        mu: f64,
        rho: f64,
        velocity_magnitude_max: f64,
    ) -> (f64, f64) {
        match *self {
            Self::Laminar => (mu, 0.0),
            Self::KOmegaSstAlgebraic {
                c1,
                characteristic_length_m,
                nu_t_max_ratio,
            } => {
                let nu = mu / rho.max(f64::EPSILON);
                let nu_t = c1 * characteristic_length_m * velocity_magnitude_max.abs();
                let cap = nu_t_max_ratio * nu;
                let nu_t_clamped = nu_t.min(cap).max(0.0);
                let derivative = if nu_t > 0.0 && nu_t < cap {
                    rho * c1 * characteristic_length_m
                } else {
                    0.0
                };
                (mu + rho * nu_t_clamped, derivative)
            }
        }
    }

    /// Compute the unique active nodal velocity peak and its subgradient.
    ///
    /// The global maximum is intentionally retained for compatibility with
    /// the algebraic closure.  A Newton tangent is only valid away from a
    /// tied or zero peak, so those cases return `None` and callers must keep
    /// the resolved derivative path fail-closed.
    pub fn unique_peak_velocity_subgradient(
        velocity: &[f64],
        dimension: usize,
    ) -> Option<(f64, Vec<f64>)> {
        if !(1..=3).contains(&dimension)
            || velocity.is_empty()
            || !velocity.len().is_multiple_of(dimension)
        {
            return None;
        }

        let mut peak: f64 = 0.0;
        let mut peak_node = None;
        let mut peak_count = 0usize;
        for (node, values) in velocity.chunks_exact(dimension).enumerate() {
            let magnitude = values.iter().map(|value| value * value).sum::<f64>().sqrt();
            if !magnitude.is_finite() {
                return None;
            }
            let tolerance = 1.0e-12 * peak.max(magnitude).max(1.0);
            if magnitude > peak + tolerance {
                peak = magnitude;
                peak_node = Some(node);
                peak_count = 1;
            } else if (magnitude - peak).abs() <= tolerance {
                peak_count += 1;
            }
        }

        let peak_node = peak_node?;
        if peak <= f64::EPSILON || peak_count != 1 {
            return None;
        }

        let mut gradient = vec![0.0; velocity.len()];
        let offset = peak_node * dimension;
        for component in 0..dimension {
            gradient[offset + component] = velocity[offset + component] / peak;
        }
        Some((peak, gradient))
    }

    /// Diagnostic: `ν_t / ν` for the current iterate. Useful for plots.
    pub fn turbulent_viscosity_ratio(&self, mu: f64, rho: f64, velocity_magnitude_max: f64) -> f64 {
        let mu_eff = self.effective_mu(mu, rho, velocity_magnitude_max);
        (mu_eff - mu) / mu.max(f64::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laminar_returns_mu_unchanged() {
        let m = TurbulenceModel::Laminar;
        assert_eq!(m.effective_mu(1.8e-5, 1.2, 10.0), 1.8e-5);
        assert_eq!(m.turbulent_viscosity_ratio(1.8e-5, 1.2, 10.0), 0.0);
    }

    #[test]
    fn algebraic_increases_with_velocity() {
        let m = TurbulenceModel::pcfan_default();
        let mu_low = m.effective_mu(1.8e-5, 1.2, 0.5);
        let mu_high = m.effective_mu(1.8e-5, 1.2, 5.0);
        assert!(mu_high > mu_low);
    }

    #[test]
    fn algebraic_clamped_at_max_ratio() {
        let m = TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.09,
            characteristic_length_m: 1.0,
            nu_t_max_ratio: 100.0,
        };
        // Push velocity high enough to saturate the cap.
        let mu = 1.8e-5;
        let rho = 1.2;
        let nu = mu / rho;
        let mu_eff = m.effective_mu(mu, rho, 1.0e6);
        let ratio = (mu_eff - mu) / (rho * nu);
        assert!(
            (ratio - 100.0).abs() < 1.0e-6,
            "expected clamp at 100, got {ratio}"
        );
    }

    #[test]
    fn zero_velocity_gives_no_eddy_viscosity() {
        let m = TurbulenceModel::pcfan_default();
        let mu = 1.8e-5;
        assert_eq!(m.effective_mu(mu, 1.2, 0.0), mu);
    }

    #[test]
    fn ratio_diagnostic_matches_definition() {
        let m = TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.1,
            characteristic_length_m: 0.05,
            nu_t_max_ratio: 1000.0,
        };
        let mu = 1.8e-5;
        let rho = 1.2;
        let u = 5.0;
        let mu_eff = m.effective_mu(mu, rho, u);
        let ratio = m.turbulent_viscosity_ratio(mu, rho, u);
        let expected = (mu_eff - mu) / mu;
        assert!((ratio - expected).abs() < 1.0e-12);
    }

    #[test]
    fn viscosity_speed_derivative_matches_centered_difference() {
        let model = TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.1,
            characteristic_length_m: 0.05,
            nu_t_max_ratio: 1_000.0,
        };
        let mu = 1.8e-5;
        let rho = 1.2;
        let speed = 5.0;
        let step = 1.0e-6;
        let (_, derivative) = model.effective_mu_and_speed_derivative(mu, rho, speed);
        let centered = (model.effective_mu(mu, rho, speed + step)
            - model.effective_mu(mu, rho, speed - step))
            / (2.0 * step);
        assert!((derivative - centered).abs() < 1.0e-10);
    }

    #[test]
    fn viscosity_speed_derivative_is_zero_at_clamp() {
        let model = TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.1,
            characteristic_length_m: 1.0,
            nu_t_max_ratio: 10.0,
        };
        let (_, derivative) = model.effective_mu_and_speed_derivative(1.8e-5, 1.2, 1.0e6);
        assert_eq!(derivative, 0.0);
    }

    #[test]
    fn unique_peak_subgradient_matches_centered_directional_difference() {
        let velocity = vec![1.0, 0.0, 0.0, 0.3, 0.4, 0.2];
        let (peak, gradient) =
            TurbulenceModel::unique_peak_velocity_subgradient(&velocity, 3).unwrap();
        let direction = vec![0.2, -0.4, 0.1, 0.0, 0.0, 0.0];
        let step = 1.0e-7;
        let plus = velocity
            .iter()
            .zip(&direction)
            .map(|(value, delta)| value + step * delta)
            .collect::<Vec<_>>();
        let minus = velocity
            .iter()
            .zip(&direction)
            .map(|(value, delta)| value - step * delta)
            .collect::<Vec<_>>();
        let plus_peak = plus
            .chunks_exact(3)
            .map(|values| values.iter().map(|value| value * value).sum::<f64>().sqrt())
            .fold(0.0, f64::max);
        let minus_peak = minus
            .chunks_exact(3)
            .map(|values| values.iter().map(|value| value * value).sum::<f64>().sqrt())
            .fold(0.0, f64::max);
        let centered = (plus_peak - minus_peak) / (2.0 * step);
        let directional = gradient
            .iter()
            .zip(&direction)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        assert!((peak - 1.0).abs() < 1.0e-12);
        assert!((centered - directional).abs() < 1.0e-8);
    }

    #[test]
    fn unique_peak_subgradient_rejects_ties_and_zero_state() {
        assert!(TurbulenceModel::unique_peak_velocity_subgradient(&[0.0; 6], 3).is_none());
        assert!(
            TurbulenceModel::unique_peak_velocity_subgradient(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3)
                .is_none()
        );
    }
}
