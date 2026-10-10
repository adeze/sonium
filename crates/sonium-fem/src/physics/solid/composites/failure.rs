//! Two-dimensional ply failure criteria and progressive damage.
//!
//! Provides Hashin (1980) and Puck (action-plane search) failure exposures
//! evaluated in ply material coordinates `(1 = fiber, 2 = transverse)`,
//! plus stateful damage variables with sudden or gradual stiffness
//! degradation. Puck is preferred over LaRC05 here because this module is
//! strictly two-dimensional plane stress: Puck's inter-fiber-fracture
//! criterion was formulated and validated in 2-D, while LaRC05's value
//! (fiber kinking with a 3-D misalignment frame) needs through-thickness
//! stresses this module does not carry.
//!
//! # References
//!
//! - Hashin, Z. (1980). Failure criteria for unidirectional fiber
//!   composites. J. Appl. Mech. 47, 329-334.
//! - Puck, A. and Schuermann, H. (1998). Failure analysis of FRP laminates
//!   by means of physically based phenomenological models. Comp. Sci.
//!   Tech. 58, 1045-1067.
//! - Puck, A. and Schuermann, H. (2002). Failure analysis of FRP laminates
//!   by means of physically based phenomenological models (Part B).
//!   Comp. Sci. Tech. 62, 1633-1653.
//! - Deuschle, H. M. (2010). 3D failure analysis of UD fibre reinforced
//!   composites. PhD thesis, Universitaet Stuttgart (Puck's fracture
//!   exposure derivation, inclination parameters, fracture-angle search).
//! - Lapczyk, I. and Hurtado, J. A. (2007). Progressive damage modeling in
//!   fiber-reinforced materials. Comp. Part A 38, 2333-2341 (damage
//!   operator-split pattern this module follows).

// Rust guideline compliant 2026-02-21

use super::materials::OrthotropicPly;

/// Ply allowables in material coordinates, all positive, in Pa.
///
/// `s23` is the transverse (through-thickness) shear strength needed by the
/// Hashin matrix-compression mode; when unmeasured, `s23 = s12` is the
/// common approximation (state it wherever used).
#[derive(Debug, Clone, Copy)]
pub struct PlyStrengths {
    /// Fiber-direction tensile strength Xt.
    pub xt: f64,
    /// Fiber-direction compressive strength Xc.
    pub xc: f64,
    /// Transverse tensile strength Yt.
    pub yt: f64,
    /// Transverse compressive strength Yc.
    pub yc: f64,
    /// In-plane shear strength S12.
    pub s12: f64,
    /// Transverse shear strength S23.
    pub s23: f64,
}

/// Puck inclination (slope) parameters, dimensionless.
///
/// Recommended values (Deuschle Table 4.1): carbon/epoxy
/// `(0.35, 0.30, 0.25-0.30, 0.25-0.30)`, glass/epoxy
/// `(0.30, 0.25, 0.20-0.25, 0.20-0.25)`.
#[derive(Debug, Clone, Copy)]
pub struct PuckParams {
    /// Slope of the longitudinal-shear fracture curve, tension side.
    pub p_t_par: f64,
    /// Slope of the longitudinal-shear fracture curve, compression side.
    pub p_c_par: f64,
    /// Slope of the transverse-shear fracture curve, tension side.
    pub p_t_perp: f64,
    /// Slope of the transverse-shear fracture curve, compression side.
    pub p_c_perp: f64,
}

impl PuckParams {
    /// Carbon/epoxy defaults from Deuschle Table 4.1.
    pub fn carbon_epoxy() -> Self {
        Self {
            p_t_par: 0.35,
            p_c_par: 0.30,
            p_t_perp: 0.30,
            p_c_perp: 0.25,
        }
    }

    /// Glass/epoxy defaults from Deuschle Table 4.1.
    pub fn glass_epoxy() -> Self {
        Self {
            p_t_par: 0.30,
            p_c_par: 0.25,
            p_t_perp: 0.25,
            p_c_perp: 0.20,
        }
    }
}

/// Hashin 1980 failure exposures, one per mode.
///
/// Each exposure is the square root of its criterion value, so exposure
/// scales linearly with proportional loading and 1.0 marks initiation.
#[derive(Debug, Clone, Copy)]
pub struct HashinExposures {
    /// Fiber tension exposure (sigma1 >= 0 branch).
    pub fiber_tension: f64,
    /// Fiber compression exposure (sigma1 < 0 branch).
    pub fiber_compression: f64,
    /// Matrix tension exposure (sigma2 >= 0 branch).
    pub matrix_tension: f64,
    /// Matrix compression exposure (sigma2 < 0 branch).
    pub matrix_compression: f64,
}

impl HashinExposures {
    /// Largest exposure over the four modes.
    pub fn max(&self) -> f64 {
        self.fiber_tension
            .max(self.fiber_compression)
            .max(self.matrix_tension)
            .max(self.matrix_compression)
    }
}

/// Evaluate Hashin 1980 exposures for plane stress.
///
/// # Arguments
///
/// * `stress` - `[sigma1, sigma2, tau12]` in ply material coordinates (Pa).
/// * `strengths` - ply allowables.
///
/// # Returns
///
/// Per-mode exposures; values `>= 1.0` mark initiation of that mode.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::solid::composites::{PlyStrengths, hashin_exposures};
/// let s = PlyStrengths { xt: 1.0e9, xc: 1.0e9, yt: 5.0e7, yc: 2.0e8, s12: 8.0e7, s23: 8.0e7 };
/// let e = hashin_exposures([1.0e9, 0.0, 0.0], &s);
/// assert!((e.fiber_tension - 1.0).abs() < 1e-12);
/// ```
pub fn hashin_exposures(stress: [f64; 3], strengths: &PlyStrengths) -> HashinExposures {
    let (s1, s2, t12) = (stress[0], stress[1], stress[2]);
    let s = strengths;
    let shear = (t12 / s.s12).powi(2);
    let fiber_tension = if s1 >= 0.0 {
        ((s1 / s.xt).powi(2) + shear).max(0.0).sqrt()
    } else {
        0.0
    };
    let fiber_compression = if s1 < 0.0 { (-s1 / s.xc).max(0.0) } else { 0.0 };
    let matrix_tension = if s2 >= 0.0 {
        ((s2 / s.yt).powi(2) + shear).max(0.0).sqrt()
    } else {
        0.0
    };
    let matrix_compression = if s2 < 0.0 {
        let bulk = (s.yc / (2.0 * s.s23)).powi(2) - 1.0;
        (bulk * (s2 / s.yc) + (s2 / (2.0 * s.s23)).powi(2) + shear)
            .max(0.0)
            .sqrt()
    } else {
        0.0
    };
    HashinExposures {
        fiber_tension,
        fiber_compression,
        matrix_tension,
        matrix_compression,
    }
}

/// Puck failure exposures with the inter-fiber fracture angle.
///
/// `fiber` is the max-stress fiber exposure; `iff` is the action-plane
/// maximum over the fracture-angle search.
#[derive(Debug, Clone, Copy)]
pub struct PuckExposures {
    /// Fiber exposure (sigma1/Xt or -sigma1/Xc).
    pub fiber: f64,
    /// Inter-fiber-fracture exposure (action-plane maximum).
    pub iff: f64,
    /// Fracture angle in degrees within [0, 90].
    pub fracture_angle_deg: f64,
}

impl PuckExposures {
    /// Largest exposure over fiber and inter-fiber modes.
    pub fn max(&self) -> f64 {
        self.fiber.max(self.iff)
    }
}

/// Action-plane fracture resistances derived from strengths.
fn puck_resistances(s: &PlyStrengths, p: &PuckParams) -> (f64, f64, f64) {
    let r_perp_t = s.yt;
    let r_perp_par = s.s12;
    let r_perp_perp = s.yc / (2.0 * (1.0 + p.p_c_perp));
    (r_perp_t, r_perp_perp, r_perp_par)
}

/// Evaluate Puck exposures for plane stress with fracture-angle search.
///
/// # Arguments
///
/// * `stress` - `[sigma1, sigma2, tau12]` in ply material coordinates (Pa).
/// * `strengths` - ply allowables.
/// * `params` - inclination parameters.
///
/// # Returns
///
/// Fiber and inter-fiber exposures plus the maximizing fracture angle.
/// The 2-D action-plane stresses are symmetric in the fracture angle, so
/// the search covers `[0, 90]` degrees in 0.5-degree steps.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::solid::composites::{PlyStrengths, PuckParams, puck_exposures};
/// let s = PlyStrengths { xt: 1.0e9, xc: 1.0e9, yt: 5.0e7, yc: 2.0e8, s12: 8.0e7, s23: 8.0e7 };
/// let e = puck_exposures([0.0, 5.0e7, 0.0], &s, &PuckParams::carbon_epoxy());
/// assert!((e.iff - 1.0).abs() < 1e-9);
/// assert!(e.fracture_angle_deg.abs() < 1.0);
/// ```
pub fn puck_exposures(
    stress: [f64; 3],
    strengths: &PlyStrengths,
    params: &PuckParams,
) -> PuckExposures {
    let (s1, s2, t12) = (stress[0], stress[1], stress[2]);
    let s = strengths;
    let p = params;
    let fiber = if s1 >= 0.0 {
        (s1 / s.xt).max(0.0)
    } else {
        (-s1 / s.xc).max(0.0)
    };
    let (r_t, r_pp, r_pl) = puck_resistances(s, p);
    let mut best = 0.0;
    let mut best_angle = 0.0;
    let mut theta_deg: f64 = 0.0;
    while theta_deg <= 90.0 {
        let theta = theta_deg.to_radians();
        let (sn, cos) = theta.sin_cos();
        let cos2 = cos * cos;
        // 2-D action-plane stresses (sigma3 = tau23 = tau31 = 0).
        let sigma_n = s2 * cos2;
        let tau_nt = -s2 * sn * cos;
        let tau_n1 = t12 * cos;
        let shear_sq = (tau_nt / r_pp).powi(2) + (tau_n1 / r_pl).powi(2);
        // Inclination ratio interpolated by shear angle psi; pure normal
        // stress defaults to the transverse-shear (psi = 0) branch.
        let shear_mag_sq = tau_nt * tau_nt + tau_n1 * tau_n1;
        let exposure = if sigma_n >= 0.0 {
            let ratio = if shear_mag_sq > 0.0 {
                (p.p_t_perp / r_pp) * tau_nt * tau_nt / shear_mag_sq
                    + (p.p_t_par / r_pl) * tau_n1 * tau_n1 / shear_mag_sq
            } else {
                p.p_t_perp / r_pp
            };
            let linear = 2.0 * ratio * sigma_n;
            let c2 = 1.0 - 2.0 * ratio * r_t;
            let quad = shear_sq + c2 * (sigma_n / r_t).powi(2);
            0.5 * (linear + (linear * linear + 4.0 * quad).max(0.0).sqrt())
        } else {
            let ratio = if shear_mag_sq > 0.0 {
                (p.p_c_perp / r_pp) * tau_nt * tau_nt / shear_mag_sq
                    + (p.p_c_par / r_pl) * tau_n1 * tau_n1 / shear_mag_sq
            } else {
                p.p_c_perp / r_pp
            };
            let linear = 2.0 * ratio * sigma_n;
            0.5 * (linear + (linear * linear + 4.0 * shear_sq).max(0.0).sqrt())
        };
        if exposure > best {
            best = exposure;
            best_angle = theta_deg;
        }
        theta_deg += 0.5;
    }
    PuckExposures {
        fiber,
        iff: best,
        fracture_angle_deg: best_angle,
    }
}

/// Failure criterion selector for ply assessment.
#[derive(Debug, Clone, Copy)]
pub enum FailureCriterion {
    /// Hashin 1980 four-mode criterion.
    Hashin,
    /// Puck criterion with inclination parameters.
    Puck(PuckParams),
}

/// Per-criterion exposure detail behind a unified assessment.
#[derive(Debug, Clone, Copy)]
pub enum FailureDetail {
    /// Hashin per-mode exposures.
    Hashin(HashinExposures),
    /// Puck exposures with fracture angle.
    Puck(PuckExposures),
}

/// Unified ply failure assessment driving damage.
///
/// `fiber` and `matrix` are the maximum exposures of the fiber-controlled
/// and matrix-controlled modes; either reaching 1.0 initiates damage.
#[derive(Debug, Clone, Copy)]
pub struct PlyFailureAssessment {
    /// Fiber-mode exposure (fiber breakage/kinking driver).
    pub fiber: f64,
    /// Matrix-mode exposure (matrix cracking/crushing driver).
    pub matrix: f64,
    /// Per-criterion detail (modes, Puck fracture angle).
    pub detail: FailureDetail,
}

/// Assess ply failure exposures under one criterion.
///
/// # Arguments
///
/// * `stress` - `[sigma1, sigma2, tau12]` in ply material coordinates (Pa).
/// * `strengths` - ply allowables.
/// * `criterion` - Hashin or Puck selector.
///
/// # Returns
///
/// Unified fiber/matrix exposures with per-criterion detail.
pub fn assess_ply_failure(
    stress: [f64; 3],
    strengths: &PlyStrengths,
    criterion: FailureCriterion,
) -> PlyFailureAssessment {
    match criterion {
        FailureCriterion::Hashin => {
            let e = hashin_exposures(stress, strengths);
            PlyFailureAssessment {
                fiber: e.fiber_tension.max(e.fiber_compression),
                matrix: e.matrix_tension.max(e.matrix_compression),
                detail: FailureDetail::Hashin(e),
            }
        }
        FailureCriterion::Puck(params) => {
            let e = puck_exposures(stress, strengths, &params);
            PlyFailureAssessment {
                fiber: e.fiber,
                matrix: e.iff,
                detail: FailureDetail::Puck(e),
            }
        }
    }
}

/// Stiffness degradation rule applied past initiation.
#[derive(Debug, Clone, Copy)]
pub enum DegradationRule {
    /// Sudden death: damage jumps to `1 - residual` at initiation.
    SuddenDeath {
        /// Retained stiffness fraction (keeps tangents positive definite).
        residual: f64,
    },
    /// Gradual: `d = max(d, 1 - 1/exposure)` past initiation, capped.
    Gradual {
        /// Retained stiffness fraction bounding the growth.
        residual: f64,
    },
}

impl DegradationRule {
    /// Sudden death with 1e-6 residual stiffness.
    pub fn sudden() -> Self {
        Self::SuddenDeath { residual: 1e-6 }
    }

    /// Gradual degradation with 1e-6 residual stiffness.
    pub fn gradual() -> Self {
        Self::Gradual { residual: 1e-6 }
    }

    fn residual(&self) -> f64 {
        match *self {
            Self::SuddenDeath { residual } | Self::Gradual { residual } => residual,
        }
    }
}

/// Stateful ply damage variables, all within `[0, 1)`.
///
/// `df` degrades the fiber modulus E1, `dm` the transverse modulus E2, and
/// `ds` the shear modulus G12. Shear follows the worse of fiber and matrix
/// damage (`ds = max(df, dm)`): a fiber break or a matrix crack both kill
/// shear transfer. States only ever advance (no healing).
#[derive(Debug, Clone, Copy)]
pub struct DamageState {
    /// Fiber damage variable.
    pub df: f64,
    /// Matrix damage variable.
    pub dm: f64,
    /// Shear damage variable (derived as `max(df, dm)`).
    pub ds: f64,
}

impl DamageState {
    /// Virgin (undamaged) state.
    pub fn virgin() -> Self {
        Self {
            df: 0.0,
            dm: 0.0,
            ds: 0.0,
        }
    }

    /// Largest damage variable (0 virgin, approaching 1 fully failed).
    pub fn max(&self) -> f64 {
        self.df.max(self.dm).max(self.ds)
    }

    /// Advance damage from mode exposures without mutating (pure).
    ///
    /// Damage is irreversible: each variable is the maximum of its current
    /// value and the rule applied to the current exposure. Shear follows
    /// the worse of the advanced fiber and matrix variables.
    pub fn advance(
        &self,
        fiber_exposure: f64,
        matrix_exposure: f64,
        rule: DegradationRule,
    ) -> Self {
        let cap = 1.0 - rule.residual().clamp(1e-12, 1.0);
        let grow = |current: f64, exposure: f64| -> f64 {
            if exposure < 1.0 {
                return current;
            }
            let target = match rule {
                DegradationRule::SuddenDeath { .. } => cap,
                DegradationRule::Gradual { .. } => (1.0 - 1.0 / exposure).min(cap),
            };
            current.max(target)
        };
        let df = grow(self.df, fiber_exposure);
        let dm = grow(self.dm, matrix_exposure);
        Self {
            df,
            dm,
            ds: df.max(dm),
        }
    }
}

/// Rotate laminate stress to ply material coordinates.
///
/// # Arguments
///
/// * `stress_xy` - `[sigmax, sigmay, tauxy]` in laminate coordinates (Pa).
/// * `angle_rad` - ply angle from laminate x to material 1-axis.
///
/// # Returns
///
/// `[sigma1, sigma2, tau12]` in material coordinates.
pub fn laminate_to_material_stress(stress_xy: [f64; 3], angle_rad: f64) -> [f64; 3] {
    let (sx, sy, txy) = (stress_xy[0], stress_xy[1], stress_xy[2]);
    let (sn, m) = angle_rad.sin_cos();
    let (m2, n2, mn) = (m * m, sn * sn, m * sn);
    [
        m2 * sx + n2 * sy + 2.0 * mn * txy,
        n2 * sx + m2 * sy - 2.0 * mn * txy,
        -mn * sx + mn * sy + (m2 - n2) * txy,
    ]
}

/// Degrade orthotropic moduli by damage variables.
///
/// Returns a copy of the ply with E1 scaled by `(1 - df)`, E2 by
/// `(1 - dm)`, and G12 by `(1 - ds)`; nu12, thickness, and angle are
/// unchanged. Factors clamp at 1e-12 to keep Q valid.
pub fn degraded_ply(ply: &OrthotropicPly, damage: &DamageState) -> OrthotropicPly {
    let clamp = |d: f64| (1.0 - d).clamp(1e-12, 1.0);
    OrthotropicPly {
        e1: ply.e1 * clamp(damage.df),
        e2: ply.e2 * clamp(damage.dm),
        g12: ply.g12 * clamp(damage.ds),
        nu12: ply.nu12,
        thickness: ply.thickness,
        angle_rad: ply.angle_rad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn carbon_strengths() -> PlyStrengths {
        PlyStrengths {
            xt: 1.95e9,
            xc: 1.48e9,
            yt: 4.8e7,
            yc: 2.0e8,
            s12: 7.9e7,
            s23: 7.9e7,
        }
    }

    #[test]
    fn hashin_uniaxial_anchors_are_exact() {
        let s = carbon_strengths();
        let ft = hashin_exposures([s.xt, 0.0, 0.0], &s);
        assert!((ft.fiber_tension - 1.0).abs() < 1e-12);
        assert_eq!(ft.matrix_tension, 0.0);
        let fc = hashin_exposures([-s.xc, 0.0, 0.0], &s);
        assert!((fc.fiber_compression - 1.0).abs() < 1e-12);
        let mt = hashin_exposures([0.0, s.yt, 0.0], &s);
        assert!((mt.matrix_tension - 1.0).abs() < 1e-12);
        let mc = hashin_exposures([0.0, -s.yc, 0.0], &s);
        assert!((mc.matrix_compression - 1.0).abs() < 1e-9);
        let sh = hashin_exposures([0.0, 0.0, s.s12], &s);
        assert!((sh.fiber_tension - 1.0).abs() < 1e-12);
        assert!((sh.matrix_tension - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hashin_biaxial_matches_closed_form() {
        let s = carbon_strengths();
        // Fiber tension plus half shear strength: sqrt(0.6^2 + 0.5^2).
        let e = hashin_exposures([0.6 * s.xt, 0.0, 0.5 * s.s12], &s);
        assert!((e.fiber_tension - 0.781025).abs() < 1e-5);
        assert_eq!(e.fiber_compression, 0.0);
    }

    #[test]
    fn puck_uniaxial_anchors_are_exact() {
        let s = carbon_strengths();
        let p = PuckParams::carbon_epoxy();
        let ft = puck_exposures([s.xt, 0.0, 0.0], &s, &p);
        assert!((ft.fiber - 1.0).abs() < 1e-12);
        let fc = puck_exposures([-s.xc, 0.0, 0.0], &s, &p);
        assert!((fc.fiber - 1.0).abs() < 1e-12);
        let mt = puck_exposures([0.0, s.yt, 0.0], &s, &p);
        assert!((mt.iff - 1.0).abs() < 1e-9);
        assert!(mt.fracture_angle_deg.abs() < 1.0);
    }

    #[test]
    fn puck_transverse_compression_fractures_oblique() {
        // Pure transverse compression must reach exposure 1.0 on an oblique
        // plane near the experimental 53-54 degrees (Mode C).
        let s = carbon_strengths();
        let p = PuckParams::carbon_epoxy();
        let e = puck_exposures([0.0, -s.yc, 0.0], &s, &p);
        assert!((e.iff - 1.0).abs() < 0.02, "iff = {}", e.iff);
        assert!(
            (e.fracture_angle_deg - 53.5).abs() < 3.0,
            "angle = {}",
            e.fracture_angle_deg
        );
    }

    #[test]
    fn puck_pure_shear_fractures_at_zero_degrees() {
        let s = carbon_strengths();
        let p = PuckParams::carbon_epoxy();
        let e = puck_exposures([0.0, 0.0, s.s12], &s, &p);
        assert!((e.iff - 1.0).abs() < 1e-9);
        assert!(e.fracture_angle_deg.abs() < 1.0);
    }

    #[test]
    fn puck_exposure_scales_linearly_in_tension() {
        // Mode-A exposure is homogeneous of degree one in tension.
        let s = carbon_strengths();
        let p = PuckParams::carbon_epoxy();
        let half = puck_exposures([0.0, 0.5 * s.yt, 0.25 * s.s12], &s, &p);
        let full = puck_exposures([0.0, s.yt, 0.5 * s.s12], &s, &p);
        assert!((full.iff - 2.0 * half.iff).abs() < 1e-9);
    }

    #[test]
    fn stress_rotation_recovers_material_axes() {
        // 90-degree ply: laminate x maps to material 2 (with shear sign flip).
        let m = laminate_to_material_stress([1.0e6, 2.0e6, 3.0e6], std::f64::consts::FRAC_PI_2);
        assert!((m[0] - 2.0e6).abs() < 1.0);
        assert!((m[1] - 1.0e6).abs() < 1.0);
        assert!((m[2] + 3.0e6).abs() < 1.0);
        // 0-degree ply is the identity.
        let z = laminate_to_material_stress([1.0e6, 2.0e6, 3.0e6], 0.0);
        assert!((z[0] - 1.0e6).abs() < 1e-6);
        assert!((z[1] - 2.0e6).abs() < 1e-6);
        assert!((z[2] - 3.0e6).abs() < 1e-6);
    }

    #[test]
    fn damage_advance_is_irreversible_and_capped() {
        let virgin = DamageState::virgin();
        assert_eq!(virgin.max(), 0.0);
        let failed = virgin.advance(1.5, 0.5, DegradationRule::sudden());
        assert!((failed.df - (1.0 - 1e-6)).abs() < 1e-12);
        assert_eq!(failed.dm, 0.0);
        assert_eq!(failed.ds, failed.df);
        // Unloading below initiation never heals.
        let unloaded = failed.advance(0.2, 0.1, DegradationRule::sudden());
        assert_eq!(unloaded.df, failed.df);
        // Gradual law grows smoothly: exposure 2 -> d = 0.5.
        let gradual = virgin.advance(2.0, 4.0, DegradationRule::gradual());
        assert!((gradual.df - 0.5).abs() < 1e-12);
        assert!((gradual.dm - 0.75).abs() < 1e-12);
        assert_eq!(gradual.ds, gradual.dm);
    }

    #[test]
    fn degraded_ply_scales_moduli_selectively() {
        let ply = OrthotropicPly {
            e1: 1.4e11,
            e2: 1.0e10,
            g12: 5.0e9,
            nu12: 0.3,
            thickness: 1.25e-4,
            angle_rad: 0.0,
        };
        let damage = DamageState {
            df: 0.0,
            dm: 0.5,
            ds: 0.5,
        };
        let deg = degraded_ply(&ply, &damage);
        assert!((deg.e1 - ply.e1).abs() < 1.0);
        assert!((deg.e2 - 0.5 * ply.e2).abs() < 1.0);
        assert!((deg.g12 - 0.5 * ply.g12).abs() < 1.0);
        assert_eq!(deg.nu12, ply.nu12);
        let q = deg.reduced_stiffness();
        assert!(q[0][0] > 0.0 && q[1][1] > 0.0 && q[2][2] > 0.0);
    }
}
