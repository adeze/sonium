//! Boundary conditions for aeroacoustics
//!
//! Provides acoustic liner impedance conditions for duct acoustics
//! in the presence of a grazing mean flow.
//!
//! # Liner models
//!
//! ## Ingard-Myers condition
//!
//! The classical impedance condition for a locally-reacting liner
//! with an infinitely thin boundary layer:
//!
//! ```text
//! p' = Z_n * (i omega + U_0 . nabla) * (u' . n) / (i omega)
//! ```
//!
//! where Z_n is the normal impedance of the liner and n is the
//! wall-normal direction.
//!
//! This condition is well-posed for subsonic flows but can exhibit
//! an ill-posed instability (the Kelvin-Helmholtz type) when the
//! boundary layer is not resolved.
//!
//! ## Brambley condition
//!
//! A modified impedance condition that accounts for a finite-thickness
//! boundary layer with profile U_0(y):
//!
//! ```text
//! p' = Z_n * [(i omega + U_0 . nabla)(u' . n) / (i omega)
//!      - delta * d^2(u' . n) / dn^2]
//! ```
//!
//! where delta is the boundary layer displacement thickness. This
//! regularises the Ingard-Myers instability and gives physically
//! correct attenuation rates.
//!
//! # References
//!
//! - Ingard (1959), "Influence of fluid motion past a plane boundary
//!   on sound reflection, absorption, and transmission"
//! - Myers (1980), "On the acoustic boundary condition in the presence
//!   of flow"
//! - Brambley (2009), "Fundamental problems with the model of uniform
//!   flow over acoustic linings"

use num_complex::Complex64;

use super::mean_flow::MeanFlow;

/// Liner impedance model specification.
#[derive(Debug, Clone, Copy)]
pub enum LinerModel {
    /// Classical Ingard-Myers condition (infinitely thin BL).
    IngardMyers,
    /// Brambley condition with finite BL displacement thickness.
    Brambley {
        /// Boundary layer displacement thickness delta [m].
        displacement_thickness: f64,
    },
}

/// Acoustic liner properties.
pub struct AcousticLiner {
    /// Normal impedance Z_n as a function of frequency.
    /// Takes angular frequency omega, returns complex impedance.
    impedance_fn: Box<dyn Fn(f64) -> Complex64 + Send + Sync>,
    /// Liner model (Ingard-Myers or Brambley).
    pub model: LinerModel,
}

impl AcousticLiner {
    /// Create a liner with frequency-dependent impedance.
    pub fn new(
        impedance_fn: impl Fn(f64) -> Complex64 + Send + Sync + 'static,
        model: LinerModel,
    ) -> Self {
        Self {
            impedance_fn: Box::new(impedance_fn),
            model,
        }
    }

    /// Create a liner with constant (frequency-independent) impedance.
    pub fn constant_impedance(impedance: Complex64, model: LinerModel) -> Self {
        Self::new(move |_| impedance, model)
    }

    /// Create a single-degree-of-freedom (SDOF) resonant liner.
    ///
    /// Z_n = R + i(omega L - S/omega)
    ///
    /// where R is resistance, L is inductance (mass reactance),
    /// and S is compliance (stiffness reactance).
    pub fn sdof(resistance: f64, inductance: f64, compliance: f64, model: LinerModel) -> Self {
        Self::new(
            move |omega| Complex64::new(resistance, omega * inductance - compliance / omega),
            model,
        )
    }

    /// Evaluate the normal impedance at angular frequency omega.
    pub fn impedance(&self, omega: f64) -> Complex64 {
        (self.impedance_fn)(omega)
    }

    /// Compute the effective admittance coefficient for the Robin BC.
    ///
    /// For the Ingard-Myers condition applied to the LPF (convected
    /// Helmholtz), the boundary condition becomes a Robin condition:
    ///
    /// `d(phi)/dn + alpha * phi = 0` on Gamma_liner
    ///
    /// where alpha depends on the impedance, frequency, and Mach number.
    ///
    /// Returns the complex admittance coefficient alpha for boundary
    /// assembly (e.g., as a boundary mass coefficient in PrebakedAssembler).
    pub fn admittance_coefficient(&self, mean_flow: &MeanFlow, omega: f64) -> Complex64 {
        let z = self.impedance(omega);
        let rho0 = mean_flow.density;
        let c0 = mean_flow.speed_of_sound;
        let k = Complex64::new(omega / c0, 0.0);
        let iw = Complex64::new(0.0, omega);

        // Specific admittance y = rho_0 c_0 / Z_n
        let y = Complex64::new(rho0 * c0, 0.0) / z;

        match self.model {
            LinerModel::IngardMyers => {
                // For LPF, the Ingard-Myers BC gives:
                // alpha = -i k y / (1 - M_n)^2
                // where M_n is the Mach component normal to the wall.
                // For a wall-parallel flow (typical case), M_n = 0:
                // alpha = -i k y
                -Complex64::new(0.0, 1.0) * k * y
            }
            LinerModel::Brambley {
                displacement_thickness,
            } => {
                // Brambley correction adds a regularising term:
                // alpha = -i k y * (1 + delta * k^2 / (i k))
                // Simplified for the dominant correction:
                let delta = displacement_thickness;
                let correction = Complex64::new(1.0, 0.0)
                    + Complex64::new(delta, 0.0) * iw / (Complex64::new(c0, 0.0));
                -Complex64::new(0.0, 1.0) * k * y * correction
            }
        }
    }
}

/// Compute the modal wavenumber for a duct mode in the presence of
/// uniform mean flow.
///
/// For a rectangular duct of width `a` x height `b`, mode (m, n):
///
/// ```text
/// k_z^2 = (k^2 - k_m^2 - k_n^2) / (1 - M_z^2) - k^2 M_z^2 / (1 - M_z^2)
/// ```
///
/// where k_m = m*pi/a, k_n = n*pi/b, and M_z is the axial Mach number.
///
/// Returns (k_z_plus, k_z_minus) for downstream and upstream propagating modes.
/// Evanescent modes have purely imaginary k_z.
pub fn duct_modal_wavenumber(
    omega: f64,
    mean_flow: &MeanFlow,
    mode_m: usize,
    mode_n: usize,
    duct_width: f64,
    duct_height: f64,
) -> (Complex64, Complex64) {
    let c0 = mean_flow.speed_of_sound;
    let k = omega / c0;
    let mz = mean_flow.velocity[2] / c0; // axial Mach in z-direction

    let km = mode_m as f64 * std::f64::consts::PI / duct_width;
    let kn = mode_n as f64 * std::f64::consts::PI / duct_height;
    let k_transverse_sq = km * km + kn * kn;

    let one_minus_m_sq = 1.0 - mz * mz;
    let discriminant = k * k - k_transverse_sq * one_minus_m_sq;

    if discriminant >= 0.0 {
        // Propagating mode
        let k_axial = discriminant.sqrt() / one_minus_m_sq;
        let shift = -k * mz / one_minus_m_sq;
        (
            Complex64::new(shift + k_axial, 0.0), // downstream
            Complex64::new(shift - k_axial, 0.0), // upstream
        )
    } else {
        // Evanescent mode
        let k_axial = (-discriminant).sqrt() / one_minus_m_sq;
        let shift = -k * mz / one_minus_m_sq;
        (
            Complex64::new(shift, -k_axial), // downstream (decaying)
            Complex64::new(shift, k_axial),  // upstream (decaying)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_impedance_liner() {
        let z = Complex64::new(500.0, -200.0);
        let liner = AcousticLiner::constant_impedance(z, LinerModel::IngardMyers);
        assert_eq!(liner.impedance(1000.0), z);
        assert_eq!(liner.impedance(5000.0), z);
    }

    #[test]
    fn test_sdof_liner_resonance() {
        // At resonance: omega_r = sqrt(S/L), reactance = 0
        let r = 100.0;
        let l = 0.01;
        let s = 1e6;
        let liner = AcousticLiner::sdof(r, l, s, LinerModel::IngardMyers);

        let omega_r = (s / l).sqrt();
        let z_r = liner.impedance(omega_r);
        assert!((z_r.re - r).abs() < 1e-6);
        assert!(z_r.im.abs() < 1e-6, "Reactance should be ~0 at resonance");
    }

    #[test]
    fn test_admittance_coefficient_ingard_myers() {
        let flow = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        let rho_c = flow.density * flow.speed_of_sound;
        // Use exact rho*c as impedance for a perfectly matched liner
        let z = Complex64::new(rho_c, 0.0);
        let liner = AcousticLiner::constant_impedance(z, LinerModel::IngardMyers);
        let omega = 1000.0;

        let alpha = liner.admittance_coefficient(&flow, omega);
        // For matched impedance: y = 1, alpha = -ik
        let k = omega / flow.speed_of_sound;
        assert!((alpha.re).abs() < 1e-10);
        assert!(
            (alpha.im + k).abs() < 1e-10,
            "alpha.im = {}, expected {}",
            alpha.im,
            -k
        );
    }

    #[test]
    fn test_duct_mode_plane_wave() {
        let flow = MeanFlow::air_standard([0.0, 0.0, 34.3]); // M_z = 0.1
        let omega = 1000.0;

        // Plane wave mode (0,0) should always propagate
        let (kp, km) = duct_modal_wavenumber(omega, &flow, 0, 0, 0.1, 0.1);
        assert!(kp.im.abs() < 1e-10, "Plane wave should propagate");
        assert!(km.im.abs() < 1e-10, "Plane wave should propagate");

        // With flow: downstream k should be smaller than upstream k
        assert!(kp.re.abs() < km.re.abs());
    }

    #[test]
    fn test_duct_mode_evanescent() {
        let flow = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        let omega = 100.0; // low frequency

        // High-order mode at low frequency should be evanescent
        let (kp, km) = duct_modal_wavenumber(omega, &flow, 10, 10, 0.01, 0.01);
        assert!(kp.im.abs() > 0.1, "High-order mode should be evanescent");
        assert!(km.im.abs() > 0.1, "High-order mode should be evanescent");
    }
}
