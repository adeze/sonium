//! Analytical lubrication benchmarks from Reynolds, Sommerfeld, and Stefan.

use std::f64::consts::PI;

/// Infinite-width linear slider-bearing pressure, relative to ambient.
pub fn linear_slider_pressure(
    x: f64,
    length: f64,
    inlet_gap: f64,
    outlet_gap: f64,
    viscosity: f64,
    surface_speed: f64,
) -> f64 {
    assert!(length > 0.0 && inlet_gap > 0.0 && outlet_gap > 0.0 && viscosity > 0.0);
    if (outlet_gap - inlet_gap).abs() <= f64::EPSILON * inlet_gap.max(outlet_gap) {
        return 0.0;
    }
    let x = x.clamp(0.0, length);
    let slope = (outlet_gap - inlet_gap) / length;
    let h = inlet_gap + slope * x;
    let harmonic_half = inlet_gap * outlet_gap / (inlet_gap + outlet_gap);
    (6.0 * viscosity * surface_speed / slope)
        * (-1.0 / h
            + 1.0 / inlet_gap
            + harmonic_half * (1.0 / (h * h) - 1.0 / (inlet_gap * inlet_gap)))
}

/// Exact load per unit bearing width for the linear slider solution.
pub fn linear_slider_load(
    length: f64,
    inlet_gap: f64,
    outlet_gap: f64,
    viscosity: f64,
    surface_speed: f64,
) -> f64 {
    assert!(length > 0.0 && inlet_gap > 0.0 && outlet_gap > 0.0 && viscosity > 0.0);
    if (outlet_gap - inlet_gap).abs() <= f64::EPSILON * inlet_gap.max(outlet_gap) {
        return 0.0;
    }
    let slope = (outlet_gap - inlet_gap) / length;
    let c = inlet_gap * outlet_gap / (inlet_gap + outlet_gap);
    let integral_h = -(outlet_gap / inlet_gap).ln()
        + (outlet_gap - inlet_gap) / inlet_gap
        + c * (-1.0 / outlet_gap + 1.0 / inlet_gap)
        - c * (outlet_gap - inlet_gap) / (inlet_gap * inlet_gap);
    6.0 * viscosity * surface_speed * integral_h / (slope * slope)
}

/// Full-Sommerfeld pressure relative to its arbitrary mean for an infinitely
/// long journal bearing with `h = clearance * (1 + eccentricity*cos(theta))`.
pub fn sommerfeld_journal_pressure(
    theta: f64,
    journal_radius: f64,
    clearance: f64,
    eccentricity_ratio: f64,
    viscosity: f64,
    surface_speed: f64,
) -> f64 {
    assert!(journal_radius > 0.0 && clearance > 0.0 && viscosity > 0.0);
    assert!((0.0..1.0).contains(&eccentricity_ratio));
    let e = eccentricity_ratio;
    let cosine = theta.cos();
    let denominator = (2.0 + e * e) * (1.0 + e * cosine).powi(2);
    6.0 * viscosity * surface_speed * journal_radius / clearance.powi(2)
        * e
        * theta.sin()
        * (2.0 + e * cosine)
        / denominator
}

/// Stefan squeeze-film pressure relative to ambient between circular disks.
pub fn circular_squeeze_pressure(
    radius: f64,
    disk_radius: f64,
    gap: f64,
    gap_rate: f64,
    viscosity: f64,
) -> f64 {
    assert!(disk_radius > 0.0 && gap > 0.0 && viscosity > 0.0);
    let r = radius.clamp(0.0, disk_radius);
    -3.0 * viscosity * gap_rate * (disk_radius * disk_radius - r * r) / gap.powi(3)
}

/// Total Stefan squeeze force; closing motion (`gap_rate < 0`) is positive.
pub fn circular_squeeze_force(disk_radius: f64, gap: f64, gap_rate: f64, viscosity: f64) -> f64 {
    assert!(disk_radius > 0.0 && gap > 0.0 && viscosity > 0.0);
    -3.0 * PI * viscosity * disk_radius.powi(4) * gap_rate / (2.0 * gap.powi(3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reynolds_slider_has_zero_end_pressure_and_exact_load() {
        let (length, h0, h1, mu, speed) = (0.1, 100e-6, 50e-6, 0.08, 3.0);
        assert_eq!(linear_slider_pressure(0.0, length, h0, h1, mu, speed), 0.0);
        assert!(linear_slider_pressure(0.5 * length, length, h0, h1, mu, speed) > 0.0);
        assert!(linear_slider_pressure(length, length, h0, h1, mu, speed).abs() < 1e-7);

        let n = 20_000;
        let dx = length / n as f64;
        let integrated = (0..=n)
            .map(|i| {
                let factor = if i == 0 || i == n { 0.5 } else { 1.0 };
                factor * linear_slider_pressure(i as f64 * dx, length, h0, h1, mu, speed)
            })
            .sum::<f64>()
            * dx;
        let exact = linear_slider_load(length, h0, h1, mu, speed);
        assert!((integrated - exact).abs() / exact < 1e-8);
    }

    #[test]
    fn sommerfeld_pressure_is_periodic_and_antisymmetric() {
        let args = (0.05, 50e-6, 0.4, 0.02, 2.0);
        let p = sommerfeld_journal_pressure(0.7, args.0, args.1, args.2, args.3, args.4);
        let opposite = sommerfeld_journal_pressure(-0.7, args.0, args.1, args.2, args.3, args.4);
        assert!((p + opposite).abs() < 1e-10 * p.abs().max(1.0));
        let periodic =
            sommerfeld_journal_pressure(0.7 + 2.0 * PI, args.0, args.1, args.2, args.3, args.4);
        assert!((p - periodic).abs() < 1e-10 * p.abs().max(1.0));
    }

    #[test]
    fn stefan_pressure_integrates_to_squeeze_force() {
        let (radius, gap, rate, mu) = (0.02, 100e-6, -2e-4, 0.1);
        let n = 20_000;
        let dr = radius / n as f64;
        let force = (0..=n)
            .map(|i| {
                let r = i as f64 * dr;
                let factor = if i == 0 || i == n { 0.5 } else { 1.0 };
                factor * circular_squeeze_pressure(r, radius, gap, rate, mu) * 2.0 * PI * r
            })
            .sum::<f64>()
            * dr;
        let exact = circular_squeeze_force(radius, gap, rate, mu);
        assert!((force - exact).abs() / exact < 1e-8);
        assert!(exact > 0.0);
    }

    #[test]
    fn cavitation_clamps_pressure_and_bounds_mass_fraction() {
        use crate::physics::fluid::lubrication::CavitationState;
        let state = CavitationState::from_trial_pressure(20_000.0, 100_000.0);
        assert_eq!(state.pressure, 100_000.0);
        assert_eq!(state.fraction, 0.2);
        assert!(state.is_cavitated());
    }
}
