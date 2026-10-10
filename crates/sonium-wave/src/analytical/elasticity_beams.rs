//! Closed-form beam, torsion, and buckling references.
//!
//! This module collects textbook strength-of-materials formulas used to
//! validate beam, shaft, and column discretizations: Euler-Bernoulli and
//! Timoshenko cantilever deflection, propped-cantilever reactions, circular
//! shaft torsion, and Euler column buckling loads. All quantities are SI
//! (metres, newtons, pascals) and all functions are pure `f64` helpers.
//!
//! References are Timoshenko and Gere (Mechanics of Materials) and
//! Timoshenko and Woinowsky-Krieger (Theory of Plates and Shells).

// Rust guideline compliant 2026-02-21

use std::f64::consts::PI;

/// Tip deflection of a cantilever under a tip load.
///
/// Returns `F L^3 / (3 E I)` for a prismatic Euler-Bernoulli beam
/// clamped at `x = 0` with a transverse tip load `F` at `x = L`.
///
/// # Examples
///
/// ```rust
/// use sonium_wave::analytical::manufactured_elasticity::beams::euler_bernoulli_cantilever_tip_point;
/// let tip = euler_bernoulli_cantilever_tip_point(100.0, 1.0, 200.0e9, 1.0e-8);
/// assert!((tip - 100.0 / (3.0 * 200.0e9 * 1.0e-8)).abs() < 1e-15);
/// ```
pub fn euler_bernoulli_cantilever_tip_point(
    load_n: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
) -> f64 {
    load_n * length_m.powi(3) / (3.0 * youngs_modulus_pa * second_moment_m4)
}

/// Tip slope of a cantilever under a tip load.
///
/// Returns `F L^2 / (2 E I)` in radians for the beam described in
/// [`euler_bernoulli_cantilever_tip_point`].
///
/// # Examples
///
/// ```rust
/// use sonium_wave::analytical::manufactured_elasticity::beams::euler_bernoulli_cantilever_slope_point;
/// let slope = euler_bernoulli_cantilever_slope_point(100.0, 1.0, 200.0e9, 1.0e-8);
/// assert!((slope - 100.0 / (2.0 * 200.0e9 * 1.0e-8)).abs() < 1e-15);
/// ```
pub fn euler_bernoulli_cantilever_slope_point(
    load_n: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
) -> f64 {
    load_n * length_m.powi(2) / (2.0 * youngs_modulus_pa * second_moment_m4)
}

/// Deflection field of a cantilever under a tip load.
///
/// Returns `F x^2 (3L - x) / (6 E I)` at station `x` measured from the
/// clamped end.
pub fn euler_bernoulli_cantilever_deflection_point(
    x_m: f64,
    load_n: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
) -> f64 {
    load_n * x_m.powi(2) * (3.0 * length_m - x_m) / (6.0 * youngs_modulus_pa * second_moment_m4)
}

/// Tip deflection of a cantilever under uniform load.
///
/// Returns `q L^4 / (8 E I)` for a distributed load `q` in N/m.
pub fn euler_bernoulli_cantilever_tip_uniform(
    load_per_m: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
) -> f64 {
    load_per_m * length_m.powi(4) / (8.0 * youngs_modulus_pa * second_moment_m4)
}

/// Deflection field of a cantilever under uniform load.
///
/// Returns `q x^2 (6L^2 - 4Lx + x^2) / (24 E I)` at station `x`.
pub fn euler_bernoulli_cantilever_deflection_uniform(
    x_m: f64,
    load_per_m: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
) -> f64 {
    load_per_m * x_m.powi(2) * (6.0 * length_m.powi(2) - 4.0 * length_m * x_m + x_m.powi(2))
        / (24.0 * youngs_modulus_pa * second_moment_m4)
}

/// Prop reaction of a propped cantilever under uniform load.
///
/// Returns `3 q L / 8` for a beam clamped at `x = 0` and simply
/// supported at `x = L`.
pub fn euler_bernoulli_propped_reaction_uniform(load_per_m: f64, length_m: f64) -> f64 {
    3.0 * load_per_m * length_m / 8.0
}

/// Deflection field of a propped cantilever under uniform load.
///
/// Returns `q x^2 (L - x)(3L - 2x) / (48 E I)` at station `x`.
pub fn euler_bernoulli_propped_deflection_uniform(
    x_m: f64,
    load_per_m: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
) -> f64 {
    load_per_m * x_m.powi(2) * (length_m - x_m) * (3.0 * length_m - 2.0 * x_m)
        / (48.0 * youngs_modulus_pa * second_moment_m4)
}

/// Prop reaction of a propped cantilever under a point load.
///
/// Returns `P a^2 (3L - a) / (2 L^3)` for a transverse load `P` at
/// station `a` on a beam clamped at `x = 0` and pinned at `x = L`.
/// Derived by releasing the prop and enforcing zero deflection there.
pub fn euler_bernoulli_propped_reaction_point(load_n: f64, load_x_m: f64, length_m: f64) -> f64 {
    load_n * load_x_m.powi(2) * (3.0 * length_m - load_x_m) / (2.0 * length_m.powi(3))
}

/// Deflection field of a propped cantilever under a point load.
///
/// Superposes the released-cantilever response to `P` at `a` with the
/// redundant prop reaction, evaluated through the point-load field.
pub fn euler_bernoulli_propped_deflection_point(
    x_m: f64,
    load_n: f64,
    load_x_m: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
) -> f64 {
    // Released cantilever: load P at station a.
    let released = if x_m <= load_x_m {
        load_n * x_m.powi(2) * (3.0 * load_x_m - x_m)
    } else {
        load_n * load_x_m.powi(2) * (3.0 * x_m - load_x_m)
    } / (6.0 * youngs_modulus_pa * second_moment_m4);
    let reaction = euler_bernoulli_propped_reaction_point(load_n, load_x_m, length_m);
    let redundant = euler_bernoulli_cantilever_deflection_point(
        x_m,
        reaction,
        length_m,
        youngs_modulus_pa,
        second_moment_m4,
    );
    released - redundant
}

/// Tip deflection of a Timoshenko cantilever under a tip load.
///
/// Returns `F L^3 / (3 E I) + F L / (kappa G A)`, the bending term
/// plus the shear term with shear correction factor `kappa`.
pub fn timoshenko_cantilever_tip_point(
    load_n: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
    shear_modulus_pa: f64,
    area_m2: f64,
    shear_factor: f64,
) -> f64 {
    let bending =
        euler_bernoulli_cantilever_tip_point(load_n, length_m, youngs_modulus_pa, second_moment_m4);
    bending + load_n * length_m / (shear_factor * shear_modulus_pa * area_m2)
}

/// Tip rotation of a Timoshenko cantilever under a tip load.
///
/// Returns `F L^2 / (2 E I)`; transverse shear adds deflection but no
/// cross-section rotation for a tip point load.
pub fn timoshenko_cantilever_rotation_point(
    load_n: f64,
    length_m: f64,
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
) -> f64 {
    euler_bernoulli_cantilever_slope_point(load_n, length_m, youngs_modulus_pa, second_moment_m4)
}

/// Twist of a circular shaft under pure torsion.
///
/// Returns `T L / (G J)` in radians for torque `T`, length `L`,
/// shear modulus `G`, and torsional constant `J`.
pub fn torsion_circular_twist(
    torque_n_m: f64,
    length_m: f64,
    shear_modulus_pa: f64,
    torsional_constant_m4: f64,
) -> f64 {
    torque_n_m * length_m / (shear_modulus_pa * torsional_constant_m4)
}

/// Peak shear stress in a circular shaft under torsion.
///
/// Returns `T r / J` at the outer radius `r`.
pub fn torsion_circular_max_shear_stress(
    torque_n_m: f64,
    radius_m: f64,
    torsional_constant_m4: f64,
) -> f64 {
    torque_n_m * radius_m / torsional_constant_m4
}

/// Torsional constant of a solid circular shaft.
///
/// Returns `pi r^4 / 2`.
pub fn torsional_constant_solid_shaft(radius_m: f64) -> f64 {
    PI * radius_m.powi(4) / 2.0
}

/// Euler buckling load of a prismatic column.
///
/// Returns `pi^2 E I / (K L)^2` with effective-length factor `K`
/// (`K = 1` pinned-pinned, `2` fixed-free, `0.5` fixed-fixed,
/// `0.7` fixed-pinned).
pub fn euler_buckling_load(
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
    length_m: f64,
    effective_length_factor: f64,
) -> f64 {
    let effective = effective_length_factor * length_m;
    PI.powi(2) * youngs_modulus_pa * second_moment_m4 / effective.powi(2)
}

/// Euler buckling load of a pinned-pinned column.
///
/// Convenience wrapper with `K = 1`.
pub fn euler_buckling_pinned(youngs_modulus_pa: f64, second_moment_m4: f64, length_m: f64) -> f64 {
    euler_buckling_load(youngs_modulus_pa, second_moment_m4, length_m, 1.0)
}

/// Euler buckling load of a fixed-free (cantilever) column.
///
/// Convenience wrapper with `K = 2`.
pub fn euler_buckling_cantilever(
    youngs_modulus_pa: f64,
    second_moment_m4: f64,
    length_m: f64,
) -> f64 {
    euler_buckling_load(youngs_modulus_pa, second_moment_m4, length_m, 2.0)
}

/// Euler buckling load of a fixed-fixed column.
///
/// Convenience wrapper with `K = 0.5`.
pub fn euler_buckling_clamped(youngs_modulus_pa: f64, second_moment_m4: f64, length_m: f64) -> f64 {
    euler_buckling_load(youngs_modulus_pa, second_moment_m4, length_m, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn cantilever_tip_matches_textbook_value() {
        // Timoshenko and Gere, Example 9-6 style: F = 100 N, L = 1 m,
        // E = 200 GPa, I = 1e-8 m^4 gives 100/(3*2e3) = 1/60 m.
        let tip = euler_bernoulli_cantilever_tip_point(100.0, 1.0, 200.0e9, 1.0e-8);
        assert_abs_diff_eq!(tip, 1.0 / 60.0, epsilon = 1e-15);
        let field = euler_bernoulli_cantilever_deflection_point(1.0, 100.0, 1.0, 200.0e9, 1.0e-8);
        assert_abs_diff_eq!(field, tip, epsilon = 1e-15);
        assert_abs_diff_eq!(
            euler_bernoulli_cantilever_deflection_point(0.0, 100.0, 1.0, 200.0e9, 1.0e-8),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn timoshenko_recovers_euler_bernoulli_for_slender_beams() {
        let euler = euler_bernoulli_cantilever_tip_point(50.0, 2.0, 70.0e9, 2.0e-8);
        let timoshenko =
            timoshenko_cantilever_tip_point(50.0, 2.0, 70.0e9, 2.0e-8, 26.0e9, 1.0e-4, 5.0 / 6.0);
        let shear = 50.0 * 2.0 / ((5.0 / 6.0) * 26.0e9 * 1.0e-4);
        assert_abs_diff_eq!(timoshenko - euler, shear, epsilon = 1e-15);
        assert!(
            shear / euler < 0.05,
            "slender beam must be bending dominated"
        );
    }

    #[test]
    fn propped_cantilever_satisfies_both_supports() {
        let (e, i, l, q) = (200.0e9, 1.0e-8, 1.5, 500.0);
        assert_abs_diff_eq!(
            euler_bernoulli_propped_reaction_uniform(q, l),
            3.0 * q * l / 8.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            euler_bernoulli_propped_deflection_uniform(0.0, q, l, e, i),
            0.0,
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(
            euler_bernoulli_propped_deflection_uniform(l, q, l, e, i),
            0.0,
            epsilon = 1e-12
        );
        // Point-load variant: zero deflection at the prop by construction.
        let p = 800.0;
        let a = 0.9;
        let reaction = euler_bernoulli_propped_reaction_point(p, a, l);
        assert_abs_diff_eq!(
            reaction,
            p * a * a * (3.0 * l - a) / (2.0 * l.powi(3)),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            euler_bernoulli_propped_deflection_point(l, p, a, l, e, i),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn torsion_and_buckling_match_textbook_values() {
        let j = torsional_constant_solid_shaft(0.02);
        assert_abs_diff_eq!(j, PI * 0.02f64.powi(4) / 2.0, epsilon = 1e-18);
        let twist = torsion_circular_twist(500.0, 1.2, 79.0e9, j);
        assert_abs_diff_eq!(twist, 500.0 * 1.2 / (79.0e9 * j), epsilon = 1e-15);
        let pcr = euler_buckling_pinned(200.0e9, 1.0e-8, 1.0);
        assert_abs_diff_eq!(pcr, PI.powi(2) * 200.0e9 * 1.0e-8, epsilon = 1e-6);
        assert_abs_diff_eq!(
            euler_buckling_cantilever(200.0e9, 1.0e-8, 1.0),
            pcr / 4.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            euler_buckling_clamped(200.0e9, 1.0e-8, 1.0),
            4.0 * pcr,
            epsilon = 1e-6
        );
    }
}
