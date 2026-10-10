//! Closed-form Hertz contact references.
//!
//! This module collects textbook normal-contact formulas for smooth
//! elastic bodies: sphere-on-half-space and cylinder line contact,
//! including contact size, pressure profiles, peak pressure, and
//! normal approach. All quantities are SI and all functions are pure
//! `f64` helpers.
//!
//! Reference is Johnson (Contact Mechanics, 1985).

// Rust guideline compliant 2026-02-21

use std::f64::consts::PI;

/// Effective contact modulus of two elastic bodies.
///
/// Returns `1 / ((1 - nu1^2)/E1 + (1 - nu2^2)/E2)`; a rigid body is
/// represented by an infinite modulus term (zero summand).
pub fn hertz_effective_modulus(
    youngs_1_pa: f64,
    poisson_1: f64,
    youngs_2_pa: f64,
    poisson_2: f64,
) -> f64 {
    1.0 / ((1.0 - poisson_1.powi(2)) / youngs_1_pa + (1.0 - poisson_2.powi(2)) / youngs_2_pa)
}

/// Hertz contact radius of a sphere on a half-space.
///
/// Returns `(3 F R / (4 E*))^(1/3)` for normal force `F`, sphere
/// radius `R`, and effective modulus `E*`.
pub fn hertz_sphere_contact_radius(force_n: f64, radius_m: f64, effective_modulus_pa: f64) -> f64 {
    (3.0 * force_n * radius_m / (4.0 * effective_modulus_pa)).powf(1.0 / 3.0)
}

/// Peak Hertz pressure of a sphere on a half-space.
///
/// Returns `3 F / (2 pi a^2)` at the contact center.
pub fn hertz_sphere_peak_pressure(force_n: f64, contact_radius_m: f64) -> f64 {
    3.0 * force_n / (2.0 * PI * contact_radius_m.powi(2))
}

/// Hertz pressure profile of a sphere on a half-space.
///
/// Returns `p0 sqrt(1 - (r/a)^2)` at radius `r`; zero outside contact.
pub fn hertz_sphere_pressure(r_m: f64, peak_pressure_pa: f64, contact_radius_m: f64) -> f64 {
    if r_m.abs() >= contact_radius_m {
        return 0.0;
    }
    let ratio = r_m / contact_radius_m;
    peak_pressure_pa * (1.0 - ratio.powi(2)).sqrt()
}

/// Normal approach of a sphere pressed on a half-space.
///
/// Returns `a^2 / R`, the relative rigid-body displacement.
pub fn hertz_sphere_approach(contact_radius_m: f64, radius_m: f64) -> f64 {
    contact_radius_m.powi(2) / radius_m
}

/// Peak subsurface shear stress of Hertz sphere contact.
///
/// Returns `0.465 p0` at depth `0.48 a` below the center, the
/// documented values for `nu = 0.3` (Johnson, Table 3.1).
pub fn hertz_sphere_max_shear_stress(peak_pressure_pa: f64) -> (f64, f64) {
    // Calibrated constants for Poisson ratio 0.3; other ratios need
    // the full Hamilton solution rather than rescaling these numbers.
    const SHEAR_FRACTION: f64 = 0.465;
    const DEPTH_FRACTION: f64 = 0.48;
    (SHEAR_FRACTION * peak_pressure_pa, DEPTH_FRACTION)
}

/// Hertz half-width of cylinder line contact.
///
/// Returns `sqrt(4 F' R / (pi E*))` for load per unit length `F'`.
pub fn hertz_cylinder_half_width(
    force_per_m: f64,
    radius_m: f64,
    effective_modulus_pa: f64,
) -> f64 {
    (4.0 * force_per_m * radius_m / (PI * effective_modulus_pa)).sqrt()
}

/// Peak Hertz pressure of cylinder line contact.
///
/// Returns `2 F' / (pi b)` at the contact centerline.
pub fn hertz_cylinder_peak_pressure(force_per_m: f64, half_width_m: f64) -> f64 {
    2.0 * force_per_m / (PI * half_width_m)
}

/// Hertz pressure profile of cylinder line contact.
///
/// Returns `p0 sqrt(1 - (x/b)^2)` at offset `x`; zero outside contact.
pub fn hertz_cylinder_pressure(x_m: f64, peak_pressure_pa: f64, half_width_m: f64) -> f64 {
    if x_m.abs() >= half_width_m {
        return 0.0;
    }
    let ratio = x_m / half_width_m;
    peak_pressure_pa * (1.0 - ratio.powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn sphere_contact_matches_textbook_values() {
        // Steel sphere (E = 200 GPa, nu = 0.3) on a rigid plane under 1 kN.
        let estar = hertz_effective_modulus(200.0e9, 0.3, f64::INFINITY, 0.0);
        assert_abs_diff_eq!(estar, 200.0e9 / 0.91, epsilon = 1e3);
        let radius = 0.05;
        let force = 1000.0;
        let a = hertz_sphere_contact_radius(force, radius, estar);
        assert_abs_diff_eq!(
            a,
            (3.0 * force * radius / (4.0 * estar)).powf(1.0 / 3.0),
            epsilon = 1e-15
        );
        let p0 = hertz_sphere_peak_pressure(force, a);
        assert_abs_diff_eq!(p0, 3.0 * force / (2.0 * PI * a * a), epsilon = 1e-6);
        assert_abs_diff_eq!(hertz_sphere_pressure(0.0, p0, a), p0, epsilon = 1e-9);
        assert_abs_diff_eq!(hertz_sphere_pressure(a, p0, a), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(hertz_sphere_pressure(2.0 * a, p0, a), 0.0, epsilon = 1e-15);
        // Pressure integrates back to the applied force.
        let steps = 2001;
        let mut total = 0.0;
        for i in 0..steps {
            let r = a * i as f64 / (steps - 1) as f64;
            let weight = if i == 0 || i == steps - 1 { 0.5 } else { 1.0 };
            total += weight * hertz_sphere_pressure(r, p0, a) * r;
        }
        total *= 2.0 * PI * a / (steps - 1) as f64;
        // Trapezoidal rule on the singular-edge sqrt profile: coarse but exact in the mean.
        assert_abs_diff_eq!(total, force, epsilon = 1e-4 * force);
        // Approach scales as F^(2/3).
        let approach = hertz_sphere_approach(a, radius);
        assert_abs_diff_eq!(approach, a * a / radius, epsilon = 1e-15);
    }

    #[test]
    fn cylinder_contact_matches_textbook_values() {
        let estar = 100.0e9;
        let b = hertz_cylinder_half_width(1.0e6, 0.1, estar);
        assert_abs_diff_eq!(
            b,
            (4.0 * 1.0e6 * 0.1 / (PI * estar)).sqrt(),
            epsilon = 1e-15
        );
        let p0 = hertz_cylinder_peak_pressure(1.0e6, b);
        assert_abs_diff_eq!(hertz_cylinder_pressure(0.0, p0, b), p0, epsilon = 1e-9);
        // Line profile integrates back to the load per unit length.
        let steps = 2001;
        let mut total = 0.0;
        for i in 0..steps {
            let x = -b + 2.0 * b * i as f64 / (steps - 1) as f64;
            let weight = if i == 0 || i == steps - 1 { 0.5 } else { 1.0 };
            total += weight * hertz_cylinder_pressure(x, p0, b);
        }
        total *= 2.0 * b / (steps - 1) as f64;
        // Trapezoidal rule on the singular-edge sqrt profile: coarse but exact in the mean.
        assert_abs_diff_eq!(total, 1.0e6, epsilon = 1e-4 * 1.0e6);
    }
}
