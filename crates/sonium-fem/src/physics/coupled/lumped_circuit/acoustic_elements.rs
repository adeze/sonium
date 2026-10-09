//! Lumped acoustic elements for circuit-analogy models
//!
//! Lumped elements represent distributed acoustic components (cavities,
//! tubes, ports) as equivalent circuits with acoustic impedance Z_a.
//!
//! Convention: Z_a = p / U where p is pressure \[Pa\] and U is volume
//! velocity \[m³/s\].
//!
//! These elements can be combined in series/parallel or cascaded as
//! two-port transfer matrices.

use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Fundamental elements
// ---------------------------------------------------------------------------

/// Acoustic compliance: a closed cavity of volume V₀.
///
/// C_a = V₀ / (ρ₀ c₀²) \[m³/Pa\].
/// Impedance: Z_a = 1 / (jω C_a).
#[derive(Debug, Clone, Copy)]
pub struct AcousticCompliance {
    pub volume: f64,
    pub density: f64,
    pub speed_of_sound: f64,
}

impl AcousticCompliance {
    pub fn new(volume: f64, density: f64, speed_of_sound: f64) -> Self {
        Self {
            volume,
            density,
            speed_of_sound,
        }
    }

    /// Compliance value C_a \[m³/Pa\].
    pub fn compliance(&self) -> f64 {
        self.volume / (self.density * self.speed_of_sound * self.speed_of_sound)
    }

    /// Impedance at angular frequency ω.
    pub fn impedance(&self, omega: f64) -> Complex64 {
        Complex64::new(1.0, 0.0) / Complex64::new(0.0, omega * self.compliance())
    }
}

/// Acoustic mass: a tube or port of length L and cross-section S.
///
/// M_a = ρ₀ L / S \[kg/m⁴\].
/// Impedance: Z_a = jω M_a.
///
/// For a port with end corrections, use `effective_length`.
#[derive(Debug, Clone, Copy)]
pub struct AcousticMass {
    pub length: f64,
    pub cross_section: f64,
    pub density: f64,
}

impl AcousticMass {
    pub fn new(length: f64, cross_section: f64, density: f64) -> Self {
        Self {
            length,
            cross_section,
            density,
        }
    }

    /// Acoustic mass value M_a \[kg/m⁴\].
    pub fn mass(&self) -> f64 {
        self.density * self.length / self.cross_section
    }

    /// Impedance at angular frequency ω.
    pub fn impedance(&self, omega: f64) -> Complex64 {
        Complex64::new(0.0, omega * self.mass())
    }

    /// Create with end correction for a flanged circular port.
    ///
    /// Effective length: L_eff = L + 0.85 × d (one flanged end).
    pub fn with_flange_correction(physical_length: f64, radius: f64, density: f64) -> Self {
        let diameter = 2.0 * radius;
        let area = PI * radius * radius;
        Self {
            length: physical_length + 0.85 * diameter,
            cross_section: area,
            density,
        }
    }
}

/// Acoustic resistance: viscous or radiation losses.
///
/// R_a \[Pa·s/m³\].
/// Impedance: Z_a = R_a (real, frequency-independent).
#[derive(Debug, Clone, Copy)]
pub struct AcousticResistance {
    pub resistance: f64,
}

impl AcousticResistance {
    pub fn new(resistance: f64) -> Self {
        Self { resistance }
    }

    pub fn impedance(&self) -> Complex64 {
        Complex64::new(self.resistance, 0.0)
    }
}

// ---------------------------------------------------------------------------
// Composite elements
// ---------------------------------------------------------------------------

/// Helmholtz resonator: neck (mass + resistance) + cavity (compliance).
///
/// The series impedance: Z = R_a + jωM_a + 1/(jωC_a).
/// Resonance: f₀ = (1/2π)√(S / (ρ₀ L_eff V₀)).
#[derive(Debug, Clone, Copy)]
pub struct HelmholtzResonator {
    pub neck_mass: AcousticMass,
    pub cavity_compliance: AcousticCompliance,
    pub neck_resistance: AcousticResistance,
}

impl HelmholtzResonator {
    /// Create from physical dimensions.
    pub fn new(
        neck_length: f64,
        neck_radius: f64,
        cavity_volume: f64,
        density: f64,
        speed_of_sound: f64,
    ) -> Self {
        Self {
            neck_mass: AcousticMass::with_flange_correction(neck_length, neck_radius, density),
            cavity_compliance: AcousticCompliance::new(cavity_volume, density, speed_of_sound),
            neck_resistance: AcousticResistance::new(0.0),
        }
    }

    /// Add viscous resistance to the neck.
    pub fn with_resistance(mut self, resistance: f64) -> Self {
        self.neck_resistance = AcousticResistance::new(resistance);
        self
    }

    /// Series impedance: Z = R + jωM + 1/(jωC).
    pub fn impedance(&self, omega: f64) -> Complex64 {
        self.neck_resistance.impedance()
            + self.neck_mass.impedance(omega)
            + self.cavity_compliance.impedance(omega)
    }

    /// Resonance frequency \[Hz\].
    pub fn resonance_frequency(&self) -> f64 {
        let m = self.neck_mass.mass();
        let c = self.cavity_compliance.compliance();
        1.0 / (2.0 * PI * (m * c).sqrt())
    }
}

// ---------------------------------------------------------------------------
// Nonlinear port
// ---------------------------------------------------------------------------

/// Bass reflex port with nonlinear (velocity-dependent) resistance.
///
/// At low volume velocities, the port behaves linearly: Z = jωM_a + R_linear.
/// At high velocities, turbulent losses add a resistance proportional to |U|:
///
/// ```text
/// Z_port(ω, U) = R_linear + R_nl·|U|/S_port + jωM_a
/// ```
///
/// where `U` is the volume velocity (m³/s) and `S_port` is the port cross-section.
///
/// The nonlinear coefficient R_nl is empirical. Typical values: 0.5–2.0
/// for well-rounded ports, up to 5+ for sharp-edged ports.
///
/// Reference: Roozen et al., "Reduction of bass-reflex port nonlinearities", JASA 2008.
#[derive(Debug, Clone, Copy)]
pub struct NonlinearPort {
    /// Linear acoustic mass (port inertance).
    pub mass: AcousticMass,
    /// Linear viscous resistance (wall friction + end radiation).
    pub resistance: AcousticResistance,
    /// Nonlinear resistance coefficient (dimensionless).
    /// R_turb = nonlinear_coeff · ρ₀ · |v_port| / (2 · S_port)
    pub nonlinear_coeff: f64,
    /// Port cross-section area (m²).
    pub area: f64,
    /// Air density (kg/m³).
    pub density: f64,
}

impl NonlinearPort {
    /// Create a circular port with end correction and empirical nonlinear coefficient.
    ///
    /// `length`: physical port length (m)
    /// `radius`: port radius (m)
    /// `density`: air density (kg/m³)
    /// `nonlinear_coeff`: turbulence coefficient (typically 1.0)
    pub fn circular(length: f64, radius: f64, density: f64, nonlinear_coeff: f64) -> Self {
        let area = PI * radius * radius;
        // Viscous resistance estimate: R_v ≈ 8·μ·L_eff / (π·r⁴) for laminar flow
        // For typical port dimensions this is very small; use a nominal value.
        let mu_air = 1.81e-5; // dynamic viscosity of air (Pa·s)
        let l_eff = length + 0.85 * 2.0 * radius; // flanged end correction
        let r_viscous = 8.0 * mu_air * l_eff / (PI * radius.powi(4));
        Self {
            mass: AcousticMass::with_flange_correction(length, radius, density),
            resistance: AcousticResistance::new(r_viscous),
            area,
            nonlinear_coeff,
            density,
        }
    }

    /// Linear impedance (ignoring nonlinear effects).
    pub fn impedance_linear(&self, omega: f64) -> Complex64 {
        self.mass.impedance(omega) + self.resistance.impedance()
    }

    /// Nonlinear impedance at a given volume velocity amplitude.
    ///
    /// `omega`: angular frequency (rad/s)
    /// `volume_velocity_amplitude`: peak |U| (m³/s)
    pub fn impedance(&self, omega: f64, volume_velocity_amplitude: f64) -> Complex64 {
        let z_linear = self.impedance_linear(omega);
        // Turbulent resistance: R_turb = K · ρ · |v| / (2·S)
        // where |v| = |U| / S is the air velocity in the port
        let v_port = volume_velocity_amplitude / self.area;
        let r_turb = self.nonlinear_coeff * self.density * v_port / 2.0;
        z_linear + Complex64::new(r_turb, 0.0)
    }

    /// Port resonance frequency with the back cavity.
    ///
    /// `cavity_compliance`: acoustic compliance of the back volume (m³/Pa).
    pub fn resonance_frequency(&self, cavity_compliance: f64) -> f64 {
        let m = self.mass.mass();
        1.0 / (2.0 * PI * (m * cavity_compliance).sqrt())
    }

    /// Estimate the volume velocity at which nonlinear effects become significant.
    ///
    /// Defined as |U| where R_turb = R_linear (i.e., nonlinear resistance
    /// equals the linear viscous resistance).
    pub fn onset_velocity(&self) -> f64 {
        // R_turb = K·ρ·|U|/(2·S²) = R_linear
        // |U| = 2·S²·R_linear / (K·ρ)
        2.0 * self.area * self.area * self.resistance.resistance
            / (self.nonlinear_coeff * self.density)
    }
}

// ---------------------------------------------------------------------------
// Impedance combination
// ---------------------------------------------------------------------------

/// Series combination of acoustic impedances.
pub fn series_impedance(impedances: &[Complex64]) -> Complex64 {
    impedances.iter().copied().sum()
}

/// Parallel combination of acoustic impedances.
pub fn parallel_impedance(impedances: &[Complex64]) -> Complex64 {
    let admittance_sum: Complex64 = impedances
        .iter()
        .map(|z| Complex64::new(1.0, 0.0) / z)
        .sum();
    Complex64::new(1.0, 0.0) / admittance_sum
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const RHO: f64 = 1.204;
    const C: f64 = 343.0;

    #[test]
    fn test_acoustic_compliance() {
        let cav = AcousticCompliance::new(1e-3, RHO, C); // 1 cm³
        let c_a = cav.compliance();
        let expected = 1e-3 / (RHO * C * C);
        assert!((c_a - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_compliance_impedance_capacitive() {
        let cav = AcousticCompliance::new(1e-3, RHO, C);
        let omega = 2.0 * PI * 1000.0;
        let z = cav.impedance(omega);
        // Capacitive: Im(Z) < 0
        assert!(z.im < 0.0, "Compliance should be capacitive");
    }

    #[test]
    fn test_acoustic_mass_impedance_inductive() {
        let port = AcousticMass::new(0.05, 1e-4, RHO);
        let omega = 2.0 * PI * 1000.0;
        let z = port.impedance(omega);
        // Inductive: Im(Z) > 0
        assert!(z.im > 0.0, "Mass should be inductive");
    }

    #[test]
    fn test_helmholtz_resonance() {
        // 1 cm neck radius, 5 cm neck, 500 cm³ cavity
        let hr = HelmholtzResonator::new(0.05, 0.01, 500e-6, RHO, C);
        let f0 = hr.resonance_frequency();
        // Should be in the hundreds of Hz range
        assert!(f0 > 50.0 && f0 < 500.0, "f₀ = {f0:.1} Hz");

        // At resonance, impedance should be minimum (only resistance)
        let omega0 = 2.0 * PI * f0;
        let z = hr.impedance(omega0);
        // Imaginary part should be near zero at resonance
        assert!(z.im.abs() < z.re.abs() * 0.1 + 0.1, "At resonance: Z = {z}");
    }

    #[test]
    fn test_nonlinear_port_matches_linear_at_low_velocity() {
        let port = NonlinearPort::circular(0.1, 0.025, RHO, 1.0);
        let omega = 2.0 * PI * 50.0;
        let u_small = 1e-6; // very small volume velocity

        let z_lin = port.impedance_linear(omega);
        let z_nl = port.impedance(omega, u_small);

        // At tiny velocity, nonlinear and linear should match
        let diff = (z_nl - z_lin).norm();
        assert!(
            diff < 0.01 * z_lin.norm(),
            "At low U: Z_nl={z_nl}, Z_lin={z_lin}, diff={diff}"
        );
    }

    #[test]
    fn test_nonlinear_port_resistance_increases_with_velocity() {
        let port = NonlinearPort::circular(0.1, 0.025, RHO, 1.0);
        let omega = 2.0 * PI * 50.0;

        let z_low = port.impedance(omega, 0.001);
        let z_high = port.impedance(omega, 0.01);

        // Real part (resistance) should increase with velocity
        assert!(
            z_high.re > z_low.re,
            "R should increase: R(low)={:.4}, R(high)={:.4}",
            z_low.re,
            z_high.re
        );

        // Imaginary part (mass reactance) should be unchanged
        assert!(
            (z_high.im - z_low.im).abs() < 1e-10,
            "X should be unchanged"
        );
    }

    #[test]
    fn test_nonlinear_port_resonance() {
        let port = NonlinearPort::circular(0.1, 0.025, RHO, 1.0);
        // Back volume: 20 liters
        let c_a = 20e-3 / (RHO * C * C);
        let fb = port.resonance_frequency(c_a);
        // Typical bass reflex: 30–80 Hz
        assert!(fb > 20.0 && fb < 100.0, "fb = {fb:.1} Hz");
    }

    #[test]
    fn test_nonlinear_port_onset_velocity() {
        let port = NonlinearPort::circular(0.1, 0.025, RHO, 1.0);
        let u_onset = port.onset_velocity();
        // Should be a small but positive volume velocity
        assert!(u_onset > 0.0, "Onset velocity should be positive");
    }

    #[test]
    fn test_series_parallel() {
        let z1 = Complex64::new(100.0, 0.0);
        let z2 = Complex64::new(100.0, 0.0);
        let zs = series_impedance(&[z1, z2]);
        assert!((zs - Complex64::new(200.0, 0.0)).norm() < 1e-10);
        let zp = parallel_impedance(&[z1, z2]);
        assert!((zp - Complex64::new(50.0, 0.0)).norm() < 1e-10);
    }
}
