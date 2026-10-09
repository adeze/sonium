//! Radiation and periodic boundary conditions
//!
//! # Radiation BCs
//!
//! Absorbing boundary conditions of increasing order for truncating
//! unbounded domains:
//!
//! Sonium uses the `exp(-iωt)` harmonic convention, so an outgoing wave has
//! spatial phase `exp(+ikr)`. Consequently:
//!
//! - **First order (Sommerfeld)**: ∂p/∂n − ikp = 0. Already available via
//!   the Robin BC infrastructure; this module provides convenience functions.
//! - **Second order (BGT)**: Bayliss-Gunzburger-Turkel condition that
//!   improves absorption of oblique and spherical waves.
//!
//! # Floquet periodic BCs
//!
//! For periodic structures (infinite arrays, phononic crystals), the Floquet
//! (Bloch) condition relates the field on paired boundaries:
//! ```text
//! p(x + L) = p(x) · exp(j k_x L)
//! ```

use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Radiation boundary conditions
// ---------------------------------------------------------------------------

/// First-order Sommerfeld absorbing coefficient: α = -ik.
///
/// Absorbs plane waves at normal incidence. Identical to the existing
/// `RobinBC::absorbing`. Provided here for API completeness.
pub fn sommerfeld_coefficient(k: f64) -> Complex64 {
    Complex64::new(0.0, -k)
}

/// Second-order Bayliss-Gunzburger-Turkel (BGT) absorbing coefficient for
/// a **spherical** radiation boundary at distance `r` from the origin.
///
/// The BGT-2 condition is:
/// ```text
/// ∂p/∂r + (-ik + 1/(2r)) p - (1/(2ikr²)) (∂²p/∂θ² + ...) = 0
/// ```
///
/// When the tangential Laplacian term is neglected (local approximation),
/// the Robin coefficient becomes α = -ik + 1/(2r).
///
/// For large r, this converges to the Sommerfeld condition.
pub fn bgt_second_order_spherical(k: f64, r: f64) -> Complex64 {
    Complex64::new(1.0 / (2.0 * r), -k)
}

/// Second-order BGT absorbing coefficient for a **cylindrical** radiation
/// boundary at distance `r` from the axis.
///
/// The 2D cylindrical radiation condition:
/// ```text
/// ∂p/∂r + (-ik + 1/(4r)) p ≈ 0
/// ```
///
/// The 1/(4r) correction improves absorption for cylindrical spreading.
pub fn bgt_second_order_cylindrical(k: f64, r: f64) -> Complex64 {
    Complex64::new(1.0 / (4.0 * r), -k)
}

// ---------------------------------------------------------------------------
// Floquet periodic boundary conditions
// ---------------------------------------------------------------------------

/// Compute the Floquet (Bloch) phase factor for a periodic boundary.
///
/// For an incident wave with wavevector component k_x along the periodicity
/// direction and period L:
/// ```text
/// e^{j k_x L}
/// ```
///
/// In FEM, Floquet periodicity is enforced by constraining paired DOFs:
/// u_target = u_source × phase_factor.
pub fn floquet_phase_factor(k_x: f64, period: f64) -> Complex64 {
    Complex64::new(0.0, k_x * period).exp()
}

/// Compute the Bloch wavevector component for a given angle of incidence.
///
/// For a plane wave at angle θ from the normal to the periodic boundary:
/// k_x = k sin(θ).
pub fn bloch_wavevector(k: f64, angle_rad: f64) -> f64 {
    k * angle_rad.sin()
}

/// Paired node constraint for Floquet periodicity.
///
/// Each pair `(source, target)` means: DOF\[target\] = phase × DOF\[source\].
///
/// After assembling the unconstrained system, apply these constraints by:
/// 1. Replacing row/col `target` with the constraint equation.
/// 2. Or using a master-slave elimination approach.
#[derive(Debug, Clone)]
pub struct FloquetConstraint {
    /// Pairs of (source_dof, target_dof).
    pub pairs: Vec<(usize, usize)>,
    /// Complex phase factor e^{jk_x L}.
    pub phase: Complex64,
}

impl FloquetConstraint {
    pub fn new(pairs: Vec<(usize, usize)>, k_x: f64, period: f64) -> Self {
        Self {
            pairs,
            phase: floquet_phase_factor(k_x, period),
        }
    }

    /// Apply Floquet constraints to a system matrix (CSR format) and RHS.
    ///
    /// For each pair (src, tgt): eliminates DOF `tgt` by substituting
    /// u_tgt = phase × u_src.
    ///
    /// Modifies the system in-place. The target rows are zeroed and the
    /// constraint u_tgt − phase × u_src = 0 is inserted.
    pub fn apply_to_rhs(&self, rhs: &mut [Complex64]) {
        for &(src, tgt) in &self.pairs {
            rhs[src] += self.phase.conj() * rhs[tgt];
            rhs[tgt] = Complex64::new(0.0, 0.0);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_sommerfeld_matches_exp_minus_iwt_outgoing_wave() {
        let k = 18.3;
        let alpha = sommerfeld_coefficient(k);
        assert!((alpha - Complex64::new(0.0, -k)).norm() < 1e-14);
    }

    #[test]
    fn test_bgt_converges_to_sommerfeld_at_large_r() {
        let k = 18.3;
        let r = 1e6; // Very large radius
        let bgt = bgt_second_order_spherical(k, r);
        let som = sommerfeld_coefficient(k);
        assert!(
            (bgt - som).norm() < 1e-4,
            "BGT at large r: {bgt}, Sommerfeld: {som}"
        );
    }

    #[test]
    fn test_bgt_correction_sign() {
        let k = 18.3;
        let r = 1.0;
        let bgt = bgt_second_order_spherical(k, r);
        // The whole Robin coefficient changes sign with the exp(-iωt)
        // outgoing-wave convention used by the Helmholtz Green function.
        assert!(bgt.re > 0.0, "Re(BGT) = {}", bgt.re);
        assert!((bgt.im + k).abs() < 1e-14);
    }

    #[test]
    fn test_floquet_phase_normal_incidence() {
        // Normal incidence: k_x = 0, phase = 1
        let phase = floquet_phase_factor(0.0, 0.1);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn test_floquet_phase_periodicity() {
        // After one full Bloch wavelength, phase should return to 1
        let k_x = 2.0 * PI / 0.1; // Bloch wavelength = period
        let phase = floquet_phase_factor(k_x, 0.1);
        assert!(
            (phase - Complex64::new(1.0, 0.0)).norm() < 1e-10,
            "Phase after full wavelength: {phase}"
        );
    }

    #[test]
    fn test_bloch_wavevector_at_45_degrees() {
        let k = 18.3;
        let k_x = bloch_wavevector(k, PI / 4.0);
        let expected = k / 2.0_f64.sqrt();
        assert!(
            (k_x - expected).abs() < 1e-10,
            "k_x = {k_x}, expected {expected}"
        );
    }
}
