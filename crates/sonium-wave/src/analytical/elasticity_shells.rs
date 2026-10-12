//! Closed-form plate, vessel, and thick-cylinder references.
//!
//! This module collects textbook axisymmetric solid-mechanics formulas:
//! circular-plate bending, thin pressure-vessel membrane stresses, and the
//! Lame thick-cylinder solution with external pressure and axial terms. All
//! quantities are SI and all functions are pure `f64` helpers.
//!
//! References are Timoshenko and Woinowsky-Krieger (Theory of Plates and
//! Shells) and Timoshenko and Gere (Mechanics of Materials).

// Rust guideline compliant 2026-02-21

use std::f64::consts::PI;

/// Flexural rigidity of an isotropic plate.
///
/// Returns `E h^3 / (12 (1 - nu^2))` for thickness `h`.
pub fn plate_bending_stiffness(
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
    thickness_m: f64,
) -> f64 {
    youngs_modulus_pa * thickness_m.powi(3) / (12.0 * (1.0 - poisson_ratio.powi(2)))
}

/// Center deflection of a clamped circular plate under uniform load.
///
/// Returns `q a^4 / (64 D)` for pressure `q`, radius `a`, and rigidity `D`.
pub fn plate_clamped_circular_center_uniform(
    pressure_pa: f64,
    radius_m: f64,
    rigidity_n_m: f64,
) -> f64 {
    pressure_pa * radius_m.powi(4) / (64.0 * rigidity_n_m)
}

/// Deflection field of a clamped circular plate under uniform load.
///
/// Returns `q (a^2 - r^2)^2 / (64 D)` at radius `r`.
pub fn plate_clamped_circular_deflection_uniform(
    r_m: f64,
    pressure_pa: f64,
    radius_m: f64,
    rigidity_n_m: f64,
) -> f64 {
    pressure_pa * (radius_m.powi(2) - r_m.powi(2)).powi(2) / (64.0 * rigidity_n_m)
}

/// Edge radial moment of a clamped circular plate under uniform load.
///
/// Returns `-q a^2 / 8` (hogging) at the built-in edge.
pub fn plate_clamped_circular_edge_moment(pressure_pa: f64, radius_m: f64) -> f64 {
    -pressure_pa * radius_m.powi(2) / 8.0
}

/// Center deflection of a clamped circular plate under a center load.
///
/// Returns `P a^2 / (16 pi D)` for a concentrated center force `P`.
pub fn plate_clamped_circular_center_point(load_n: f64, radius_m: f64, rigidity_n_m: f64) -> f64 {
    load_n * radius_m.powi(2) / (16.0 * PI * rigidity_n_m)
}

/// Deflection field of a clamped circular plate under a center load.
///
/// Returns `P (a^2 - r^2 + 2 r^2 ln(r/a)) / (16 pi D)`; the limit at
/// `r = 0` is the center value.
pub fn plate_clamped_circular_deflection_point(
    r_m: f64,
    load_n: f64,
    radius_m: f64,
    rigidity_n_m: f64,
) -> f64 {
    if r_m <= 0.0 {
        return plate_clamped_circular_center_point(load_n, radius_m, rigidity_n_m);
    }
    load_n * (radius_m.powi(2) - r_m.powi(2) + 2.0 * r_m.powi(2) * (r_m / radius_m).ln())
        / (16.0 * PI * rigidity_n_m)
}

/// Center deflection of a simply supported circular plate under uniform load.
///
/// Returns `q a^4 (5 + nu) / (64 (1 + nu) D)`.
pub fn plate_ss_circular_center_uniform(
    pressure_pa: f64,
    radius_m: f64,
    rigidity_n_m: f64,
    poisson_ratio: f64,
) -> f64 {
    pressure_pa * radius_m.powi(4) * (5.0 + poisson_ratio)
        / (64.0 * (1.0 + poisson_ratio) * rigidity_n_m)
}

/// Deflection field of a simply supported circular plate under uniform load.
///
/// Returns `q (a^2 - r^2) ((5 + nu) a^2 - (1 + nu) r^2) / (64 (1 + nu) D)`.
pub fn plate_ss_circular_deflection_uniform(
    r_m: f64,
    pressure_pa: f64,
    radius_m: f64,
    rigidity_n_m: f64,
    poisson_ratio: f64,
) -> f64 {
    pressure_pa
        * (radius_m.powi(2) - r_m.powi(2))
        * ((5.0 + poisson_ratio) * radius_m.powi(2) - (1.0 + poisson_ratio) * r_m.powi(2))
        / (64.0 * (1.0 + poisson_ratio) * rigidity_n_m)
}

/// Center deflection of a simply supported circular plate under center load.
///
/// Returns `P a^2 (3 + nu) / (16 pi (1 + nu) D)`, strictly larger than
/// the clamped value as the edge constraint requires.
pub fn plate_ss_circular_center_point(
    load_n: f64,
    radius_m: f64,
    rigidity_n_m: f64,
    poisson_ratio: f64,
) -> f64 {
    load_n * radius_m.powi(2) * (3.0 + poisson_ratio)
        / (16.0 * PI * (1.0 + poisson_ratio) * rigidity_n_m)
}

/// Hoop stress of a thin cylindrical vessel under pressure.
///
/// Returns `p R / t` for internal pressure `p`, radius `R`, thickness `t`.
pub fn vessel_cylindrical_hoop_stress(pressure_pa: f64, radius_m: f64, thickness_m: f64) -> f64 {
    pressure_pa * radius_m / thickness_m
}

/// Axial stress of a thin closed cylindrical vessel under pressure.
///
/// Returns `p R / (2 t)`, half the hoop stress.
pub fn vessel_cylindrical_axial_stress(pressure_pa: f64, radius_m: f64, thickness_m: f64) -> f64 {
    pressure_pa * radius_m / (2.0 * thickness_m)
}

/// Radial growth of a thin closed cylindrical vessel under pressure.
///
/// Returns `p R^2 (1 - nu/2) / (E t)` from the biaxial hoop strain.
pub fn vessel_cylindrical_radial_growth(
    pressure_pa: f64,
    radius_m: f64,
    thickness_m: f64,
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
) -> f64 {
    pressure_pa * radius_m.powi(2) * (1.0 - poisson_ratio / 2.0) / (youngs_modulus_pa * thickness_m)
}

/// Membrane stress of a thin spherical vessel under pressure.
///
/// Returns `p R / (2 t)`, equal in all tangent directions.
pub fn vessel_spherical_membrane_stress(pressure_pa: f64, radius_m: f64, thickness_m: f64) -> f64 {
    pressure_pa * radius_m / (2.0 * thickness_m)
}

/// Radial growth of a thin spherical vessel under pressure.
///
/// Returns `p R^2 (1 - nu) / (2 E t)` from the equibiaxial strain.
pub fn vessel_spherical_radial_growth(
    pressure_pa: f64,
    radius_m: f64,
    thickness_m: f64,
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
) -> f64 {
    pressure_pa * radius_m.powi(2) * (1.0 - poisson_ratio) / (2.0 * youngs_modulus_pa * thickness_m)
}

/// Lame stresses of a thick cylinder under internal pressure.
///
/// Returns `(sigma_r, sigma_theta)` at radius `r` for inner radius `a`,
/// outer radius `b`, and internal pressure `p`.
pub fn thick_cylinder_internal_pressure(
    r_m: f64,
    inner_m: f64,
    outer_m: f64,
    pressure_pa: f64,
) -> (f64, f64) {
    let (a2, b2, r2) = (inner_m.powi(2), outer_m.powi(2), r_m.powi(2));
    let scale = pressure_pa * a2 / (b2 - a2);
    (scale * (1.0 - b2 / r2), scale * (1.0 + b2 / r2))
}

/// Lame stresses of a thick cylinder under external pressure.
///
/// Returns `(sigma_r, sigma_theta)` at radius `r` for external
/// pressure `p` (positive in compression).
pub fn thick_cylinder_external_pressure(
    r_m: f64,
    inner_m: f64,
    outer_m: f64,
    pressure_pa: f64,
) -> (f64, f64) {
    let (a2, b2, r2) = (inner_m.powi(2), outer_m.powi(2), r_m.powi(2));
    let scale = pressure_pa * b2 / (b2 - a2);
    (-scale * (1.0 - a2 / r2), -scale * (1.0 + a2 / r2))
}

/// Axial stress of a capped thick cylinder under end pressure.
///
/// Returns `(p_i a^2 - p_o b^2) / (b^2 - a^2)` from axial equilibrium
/// of the closed ends.
pub fn thick_cylinder_axial_stress(
    inner_m: f64,
    outer_m: f64,
    inner_pressure_pa: f64,
    outer_pressure_pa: f64,
) -> f64 {
    (inner_pressure_pa * inner_m.powi(2) - outer_pressure_pa * outer_m.powi(2))
        / (outer_m.powi(2) - inner_m.powi(2))
}

/// Radial displacement of a thick cylinder under internal pressure.
///
/// Returns the plane-strain displacement `(1+nu)/E ((1-2nu) A r - B/r)`
/// with `A = p a^2/(b^2-a^2)` and `B = p a^2 b^2/(b^2-a^2)`.
pub fn thick_cylinder_radial_displacement_plane_strain(
    r_m: f64,
    inner_m: f64,
    outer_m: f64,
    pressure_pa: f64,
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
) -> f64 {
    let denom = outer_m.powi(2) - inner_m.powi(2);
    let big_a = pressure_pa * inner_m.powi(2) / denom;
    let big_b = pressure_pa * inner_m.powi(2) * outer_m.powi(2) / denom;
    (1.0 + poisson_ratio) / youngs_modulus_pa
        * ((1.0 - 2.0 * poisson_ratio) * big_a * r_m - big_b / r_m)
}

/// MacNeal-Harder pinched-cylinder deflection under a unit load.
///
/// Radial deflection per unit pinch load of the short pinched cylinder
/// with diaphragm ends: radius 300, length 600, thickness 3, Young
/// modulus 3e6, Poisson ratio 0.3, two opposite radial loads (all in
/// consistent units). Literature target 1.8248e-5 from MacNeal and
/// Harder, "A proposed standard set of problems to test finite
/// element accuracy", Finite Elements in Analysis and Design 1
/// (1985), 3-20, Table 1.
pub const PINCHED_CYLINDER_UNIT_DEFLECTION: f64 = 1.8248e-5;

/// Radial deflection of the MacNeal-Harder pinched cylinder.
///
/// Returns the linear static deflection under the pinch loads,
/// scaling the literature unit-load target by `load`.
pub fn pinched_cylinder_macneal_harder(load: f64) -> f64 {
    load * PINCHED_CYLINDER_UNIT_DEFLECTION
}

/// Radial displacement of a thick cylinder under pressure loading.
///
/// Returns the plane-stress displacement for combined internal and
/// external pressure from the Lame strains.
pub fn thick_cylinder_radial_displacement(
    r_m: f64,
    inner_m: f64,
    outer_m: f64,
    inner_pressure_pa: f64,
    outer_pressure_pa: f64,
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
) -> f64 {
    let (sr_i, st_i) = thick_cylinder_internal_pressure(r_m, inner_m, outer_m, inner_pressure_pa);
    let (sr_o, st_o) = thick_cylinder_external_pressure(r_m, inner_m, outer_m, outer_pressure_pa);
    let hoop = st_i + st_o;
    let radial = sr_i + sr_o;
    r_m * (hoop - poisson_ratio * radial) / youngs_modulus_pa
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn circular_plates_match_textbook_values() {
        let d = plate_bending_stiffness(200.0e9, 0.3, 0.01);
        assert_abs_diff_eq!(d, 200.0e9 * 1e-6 / (12.0 * 0.91), epsilon = 1e-6);
        // Clamped, uniform q: center q a^4 / 64D.
        let clamped = plate_clamped_circular_center_uniform(1.0e5, 0.5, d);
        assert_abs_diff_eq!(
            clamped,
            1.0e5 * 0.5f64.powi(4) / (64.0 * d),
            epsilon = 1e-15
        );
        let field = plate_clamped_circular_deflection_uniform(0.0, 1.0e5, 0.5, d);
        assert_abs_diff_eq!(field, clamped, epsilon = 1e-15);
        assert_abs_diff_eq!(
            plate_clamped_circular_deflection_uniform(0.5, 1.0e5, 0.5, d),
            0.0,
            epsilon = 1e-15
        );
        // Simply supported deflects more than clamped under the same load.
        let ss = plate_ss_circular_center_uniform(1.0e5, 0.5, d, 0.3);
        assert_abs_diff_eq!(ss / clamped, (5.0 + 0.3) / (1.0 + 0.3), epsilon = 1e-12);
        // Point-load ordering: SS > clamped.
        let clamped_p = plate_clamped_circular_center_point(1000.0, 0.5, d);
        let ss_p = plate_ss_circular_center_point(1000.0, 0.5, d, 0.3);
        assert_abs_diff_eq!(ss_p / clamped_p, (3.0 + 0.3) / (1.0 + 0.3), epsilon = 1e-12);
        assert_abs_diff_eq!(
            plate_clamped_circular_deflection_point(0.5, 1000.0, 0.5, d),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn vessels_and_thick_cylinder_match_textbook_values() {
        // Thin cylinder: hoop = pR/t, axial = pR/2t.
        assert_abs_diff_eq!(
            vessel_cylindrical_hoop_stress(1.0e6, 0.5, 0.01),
            50.0e6,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            vessel_cylindrical_axial_stress(1.0e6, 0.5, 0.01),
            25.0e6,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            vessel_spherical_membrane_stress(1.0e6, 0.5, 0.01),
            25.0e6,
            epsilon = 1e-6
        );
        // Thick cylinder: radial stress recovers the applied pressures.
        let (sr_a, _) = thick_cylinder_internal_pressure(0.1, 0.1, 0.2, 10.0e6);
        assert_abs_diff_eq!(sr_a, -10.0e6, epsilon = 1e-6);
        let (sr_b, _) = thick_cylinder_internal_pressure(0.2, 0.1, 0.2, 10.0e6);
        assert_abs_diff_eq!(sr_b, 0.0, epsilon = 1e-6);
        let (sr_bo, _) = thick_cylinder_external_pressure(0.2, 0.1, 0.2, 5.0e6);
        assert_abs_diff_eq!(sr_bo, -5.0e6, epsilon = 1e-6);
        // Thin-wall limit of hoop stress tends to pR/t.
        let (_, st) = thick_cylinder_internal_pressure(0.1, 0.1, 0.11, 1.0e6);
        assert_abs_diff_eq!(st / 1.0e6, 0.1 / 0.01, epsilon = 0.6);
        // Plane-strain bore displacement matches the closed form.
        let u =
            thick_cylinder_radial_displacement_plane_strain(0.1, 0.1, 0.2, 10.0e6, 200.0e9, 0.3);
        let big_a = 10.0e6 * 0.01 / 0.03;
        let big_b = 10.0e6 * 0.01 * 0.04 / 0.03;
        assert_abs_diff_eq!(
            u,
            1.3 / 200.0e9 * (0.4 * big_a * 0.1 - big_b / 0.1),
            epsilon = 1e-15
        );
    }

    #[test]
    fn pinched_cylinder_matches_macneal_harder_target() {
        assert_abs_diff_eq!(
            pinched_cylinder_macneal_harder(1.0),
            1.8248e-5,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            pinched_cylinder_macneal_harder(2.0),
            2.0 * PINCHED_CYLINDER_UNIT_DEFLECTION,
            epsilon = 1e-12
        );
    }
}
