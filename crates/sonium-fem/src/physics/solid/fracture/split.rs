//! Spectral tensile/compressive strain-energy split (Miehe).
//!
//! Isotropic energy `psi = lambda/2 (tr e)^2 + mu tr(e^2)` split by
//! principal-strain sign: the positive part drives phase-field damage,
//! the negative part never does (no cracking in compression).

// Rust guideline compliant 2026-09-28

/// Positive ramp `<x>+ = max(x, 0)`.
fn ramp_positive(value: f64) -> f64 {
    value.max(0.0)
}

/// Negative ramp `<x>- = min(x, 0)`.
fn ramp_negative(value: f64) -> f64 {
    value.min(0.0)
}

/// Principal strains of a symmetric 2x2 tensor, descending.
///
/// Inputs are `exx`, `eyy`, and the tensor shear component `exy`
/// (half the engineering shear `gamma_xy`).
pub fn principal_strains_2d(exx: f64, eyy: f64, exy: f64) -> [f64; 2] {
    let mean = 0.5 * (exx + eyy);
    let radius = (0.25 * (exx - eyy) * (exx - eyy) + exy * exy).sqrt();
    [mean + radius, mean - radius]
}

/// Miehe spectral split from principal strains.
///
/// Returns `(tensile, compressive)` energy densities for Lame constants
/// `lambda`/`mu`: `psi+- = lambda/2 <tr e>+-^2 + mu tr(<e>+-^2)`.
/// The parts always sum to the total isotropic energy.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::spectral_energy_split;
///
/// let (tensile, compressive) = spectral_energy_split([0.02, -0.006, -0.006], 5.0e5, 3.0e5);
/// assert!(tensile > 0.0 && compressive > 0.0);
/// assert!((tensile + compressive - 0.5 * 5.0e5 * 0.008f64.powi(2)
///     - 3.0e5 * (0.02f64.powi(2) + 2.0 * 0.006f64.powi(2)))
/// .abs()
///     < 1.0e-9);
/// ```
pub fn spectral_energy_split(principal: [f64; 3], lambda: f64, mu: f64) -> (f64, f64) {
    let trace = principal[0] + principal[1] + principal[2];
    let positive_trace = ramp_positive(trace);
    let negative_trace = ramp_negative(trace);
    let mut positive_squares = 0.0;
    let mut negative_squares = 0.0;
    for value in principal {
        positive_squares += ramp_positive(value).powi(2);
        negative_squares += ramp_negative(value).powi(2);
    }
    let tensile = 0.5 * lambda * positive_trace.powi(2) + mu * positive_squares;
    let compressive = 0.5 * lambda * negative_trace.powi(2) + mu * negative_squares;
    (tensile, compressive)
}

/// Tensile/compressive split of a plane-stress state.
///
/// Completes the out-of-plane strain from `ezz = -nu/(1-nu) (exx+eyy)`
/// (traction-free faces) and applies the Miehe spectral split with
/// Lame constants derived from `youngs_modulus`/`poisson_ratio`.
/// Shear input is engineering shear `gamma_xy`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::plane_stress_energy_split;
///
/// let (tensile, compressive) = plane_stress_energy_split(0.01, -0.003, 0.0, 1.0e6, 0.3);
/// assert!(tensile > compressive);
/// ```
pub fn plane_stress_energy_split(
    exx: f64,
    eyy: f64,
    gamma_xy: f64,
    youngs_modulus: f64,
    poisson_ratio: f64,
) -> (f64, f64) {
    let in_plane = principal_strains_2d(exx, eyy, 0.5 * gamma_xy);
    let ezz = -poisson_ratio / (1.0 - poisson_ratio) * (exx + eyy);
    let lambda =
        youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    spectral_energy_split([in_plane[0], in_plane[1], ezz], lambda, mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lame constants of the E = 1 MPa, nu = 0.3 test material.
    fn lame() -> (f64, f64) {
        (1.0e6 * 0.3 / (1.3 * 0.4), 1.0e6 / 2.6)
    }

    #[test]
    fn uniaxial_tension_drives_damage_only() {
        // Uniaxial x-strain e: principals (e, -nu e, -nu e), trace
        // e(1-2nu) > 0. Tensile part is exact in closed form.
        let (lambda, mu) = lame();
        let e = 0.05;
        let nu = 0.3;
        let (tensile, compressive) = spectral_energy_split([e, -nu * e, -nu * e], lambda, mu);
        let expected = e * e * (0.5 * lambda * (1.0 - 2.0 * nu).powi(2) + mu);
        assert!((tensile - expected).abs() < 1.0e-9 * expected);
        let expected_negative = e * e * (mu * 2.0 * nu * nu);
        assert!((compressive - expected_negative).abs() < 1.0e-9 * expected_negative);
    }

    #[test]
    fn uniaxial_compression_keeps_transverse_tension() {
        // Sign-flipped principals swap the energy parts exactly.
        let (lambda, mu) = lame();
        let (tension_pos, tension_neg) = spectral_energy_split([0.05, -0.015, -0.015], lambda, mu);
        let (compression_pos, compression_neg) =
            spectral_energy_split([-0.05, 0.015, 0.015], lambda, mu);
        assert!((compression_pos - tension_neg).abs() < 1.0e-12);
        assert!((compression_neg - tension_pos).abs() < 1.0e-12);
        // Transverse Poisson tension still drives (splitting), weakly.
        assert!(compression_pos > 0.0);
        assert!(compression_pos < 0.2 * tension_pos);
    }

    #[test]
    fn pure_shear_splits_energy_evenly() {
        // Principals (+g/2, -g/2, 0), zero trace: halves match exactly.
        let (lambda, mu) = lame();
        let (tensile, compressive) = spectral_energy_split([0.01, -0.01, 0.0], lambda, mu);
        assert!((tensile - compressive).abs() < 1.0e-12);
        let expected = mu * 0.01f64.powi(2);
        assert!((tensile - expected).abs() < 1.0e-12);
    }

    #[test]
    fn hydrostatic_states_stay_unsplit() {
        let (lambda, mu) = lame();
        let (tension_pos, tension_neg) = spectral_energy_split([0.01; 3], lambda, mu);
        assert_eq!(tension_neg, 0.0);
        assert!(tension_pos > 0.0);
        let (compression_pos, compression_neg) = spectral_energy_split([-0.01; 3], lambda, mu);
        assert_eq!(compression_pos, 0.0);
        assert!(compression_neg > 0.0);
    }

    #[test]
    fn plane_stress_split_matches_closed_form() {
        // Plane-stress uniaxial tension: ezz completes to -nu e.
        let e = 0.05;
        let nu = 0.3;
        let (tensile, compressive) = plane_stress_energy_split(e, -nu * e, 0.0, 1.0e6, nu);
        let (lambda, mu) = lame();
        let expected = e * e * (0.5 * lambda * (1.0 - 2.0 * nu).powi(2) + mu);
        assert!((tensile - expected).abs() < 1.0e-9 * expected);
        assert!(compressive > 0.0);
        assert!(tensile + compressive > 0.0);
    }
}
