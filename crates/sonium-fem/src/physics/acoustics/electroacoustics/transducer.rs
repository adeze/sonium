//! Multi-domain transducer coupling: electrical ↔ mechanical ↔ acoustic
//!
//! Couples a lumped electrical domain (voice coil or electrostatic) to
//! distributed mechanical and acoustic FEM domains via a full two-way
//! coupling. The block system at frequency ω:
//!
//! ```text
//! | Z_e        Φ^T         0      | | I |   | V  |
//! | Φ          K_m − ω²M_m   C_fs | | u | = | 0  |
//! | 0          C_sf         K_a   | | p |   | 0  |
//! ```
//!
//! where Φ is the electromechanical coupling (BL for voice coil,
//! electrostatic for condenser), and C is the fluid-structure coupling.
//!
//! # Electrodynamic (voice coil) driver
//!
//! The voice coil couples current I to force BL·I on the coil DOFs,
//! and back-EMF BL·v from coil velocity feeds back to the circuit.
//!
//! # Electrostatic (condenser) driver
//!
//! The electrostatic gap couples signal voltage v to force η·v on the
//! membrane DOFs, and membrane displacement u feeds back to the
//! circuit via capacitance change.

use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Transducer response
// ---------------------------------------------------------------------------

/// Frequency-domain transducer response at a single frequency.
#[derive(Debug, Clone)]
pub struct TransducerResponse {
    /// Electrical current \[A\] (complex phasor).
    pub current: Complex64,
    /// Electrical impedance Z_e = V/I \[Ω\].
    pub impedance: Complex64,
    /// Mechanical velocity at reference point \[m/s\].
    pub velocity: Complex64,
    /// Average membrane/diaphragm displacement \[m\].
    pub displacement: Complex64,
    /// Radiated acoustic power \[W\].
    pub radiated_power: f64,
    /// Angular frequency \[rad/s\].
    pub omega: f64,
}

// ---------------------------------------------------------------------------
// Electrodynamic (voice coil) driver coupling
// ---------------------------------------------------------------------------

/// Coupling definition for an electrodynamic (moving-coil) driver.
///
/// The voice coil is a lumped circuit with impedance Z_e = R + jωL.
/// It couples to the mechanical FEM DOFs via the force factor BL:
///
/// - **Forward**: Lorentz force F = BL × I applied to coil DOFs
/// - **Backward**: Back-EMF e = BL × v from coil velocity
///
/// The coil velocity is the weighted average of DOF velocities:
/// v_coil = Σ w_i × v_i (where w_i are the coil DOF weights).
#[derive(Debug, Clone)]
pub struct ElectrodynamicCoupling {
    /// DC resistance R_e \[Ω\].
    pub resistance: f64,
    /// Voice coil inductance L_e \[H\].
    pub inductance: f64,
    /// Force factor BL \[T·m\].
    pub bl: f64,
    /// Coil DOF weights: (mechanical_dof_index, weight).
    /// The weights sum to 1 and represent the coil's participation
    /// in each mechanical DOF.
    pub coil_dofs: Vec<(usize, f64)>,
}

impl ElectrodynamicCoupling {
    /// Electrical impedance Z_e = R + jωL.
    pub fn electrical_impedance(&self, omega: f64) -> Complex64 {
        Complex64::new(self.resistance, omega * self.inductance)
    }

    /// Force per unit current on each DOF: F_i = BL × w_i.
    pub fn force_distribution(&self) -> Vec<(usize, f64)> {
        self.coil_dofs
            .iter()
            .map(|&(dof, w)| (dof, self.bl * w))
            .collect()
    }

    /// Compute the electrical impedance including the motional contribution.
    ///
    /// Z_total = Z_e + Z_mot where Z_mot = BL² / Z_mech and
    /// Z_mech is the mechanical impedance at the coil.
    pub fn total_impedance(&self, omega: f64, z_mech_at_coil: Complex64) -> Complex64 {
        let z_e = self.electrical_impedance(omega);
        let z_mot = Complex64::new(self.bl * self.bl, 0.0) / z_mech_at_coil;
        z_e + z_mot
    }

    /// Solve the coupled system for a given voltage drive at frequency ω.
    ///
    /// Given the mechanical impedance at the coil Z_mech, returns the
    /// current, velocity, and displacement.
    pub fn solve_frequency(
        &self,
        voltage: Complex64,
        omega: f64,
        z_mech_at_coil: Complex64,
    ) -> (Complex64, Complex64, Complex64) {
        let z_total = self.total_impedance(omega, z_mech_at_coil);
        let current = voltage / z_total;
        let force = Complex64::new(self.bl, 0.0) * current;
        let velocity = force / z_mech_at_coil;
        let displacement = velocity / Complex64::new(0.0, omega);
        (current, velocity, displacement)
    }
}

// ---------------------------------------------------------------------------
// Electrostatic (condenser) driver coupling
// ---------------------------------------------------------------------------

/// Coupling definition for an electrostatic (condenser) driver.
///
/// The electrostatic gap couples signal voltage to pressure on the
/// membrane via the linearised coupling factor η = εV₀/d₀².
///
/// - **Forward**: Electrostatic pressure Δp = η × v_signal on membrane
/// - **Backward**: Capacitance change from displacement feeds current:
///   i_mot = Γ × jω × u_avg (where Γ = C₀V₀/d₀)
///
/// The negative electrostatic spring k_e = εV₀²/d₀³ must be subtracted
/// from the mechanical stiffness (softening effect).
#[derive(Debug, Clone)]
pub struct ElectrostaticCoupling {
    /// Coupling factor η \[Pa/V\].
    pub coupling_factor: f64,
    /// Motional transduction coefficient Γ \[A·s/m\].
    pub transduction_coefficient: f64,
    /// Electrostatic spring per unit area k_e \[Pa/m\].
    pub spring_per_area: f64,
    /// Blocked electrical capacitance C₀ \[F\].
    pub capacitance: f64,
    /// Membrane DOF indices (on boundary mass matrix).
    pub membrane_dofs: Vec<usize>,
}

impl ElectrostaticCoupling {
    /// Build from physical parameters.
    pub fn from_gap(
        bias_voltage: f64,
        gap_distance: f64,
        permittivity: f64,
        membrane_area: f64,
        membrane_dofs: Vec<usize>,
    ) -> Self {
        let c0 = permittivity * membrane_area / gap_distance;
        Self {
            coupling_factor: permittivity * bias_voltage / (gap_distance * gap_distance),
            transduction_coefficient: c0 * bias_voltage / gap_distance,
            spring_per_area: permittivity * bias_voltage * bias_voltage
                / (gap_distance * gap_distance * gap_distance),
            capacitance: c0,
            membrane_dofs,
        }
    }

    /// Blocked electrical impedance: Z_e = 1/(jωC₀).
    pub fn blocked_impedance(&self, omega: f64) -> Complex64 {
        Complex64::new(1.0, 0.0) / Complex64::new(0.0, omega * self.capacitance)
    }

    /// Sensitivity: output voltage per unit pressure for a microphone.
    ///
    /// For a condenser microphone at low frequencies:
    /// M = V₀ × C_a / (C_a + C₀) × S / k_mech
    ///
    /// This simplified version assumes the membrane compliance dominates.
    pub fn microphone_sensitivity(
        &self,
        bias_voltage: f64,
        gap_distance: f64,
        membrane_area: f64,
        mechanical_compliance: f64,
    ) -> f64 {
        // Open-circuit sensitivity: Δv/Δp ≈ V₀ × compliance × area / d₀
        bias_voltage * mechanical_compliance * membrane_area / gap_distance
    }
}

// ---------------------------------------------------------------------------
// Nonlinear transducer time-domain step
// ---------------------------------------------------------------------------

/// State for nonlinear large-signal transducer simulation.
#[derive(Debug, Clone)]
pub struct NonlinearTransducerState {
    /// Electrical current \[A\].
    pub current: f64,
    /// Coil/membrane displacement \[m\].
    pub displacement: f64,
    /// Coil/membrane velocity \[m/s\].
    pub velocity: f64,
}

/// Perform one nonlinear time step for a voice coil driver.
///
/// Updates the electrical state using the coupled equation:
/// V(t) = R·i + L(x)·di/dt + BL(x)·v + L'(x)·v·i
///
/// Returns the mechanical force: F = BL(x)·i + ½L'(x)·i²
///
/// The `bl_fn` and `le_fn` provide displacement-dependent parameters.
pub fn voice_coil_nonlinear_step<BL, LE, DLE>(
    state: &mut NonlinearTransducerState,
    voltage: f64,
    dt: f64,
    resistance: f64,
    bl_fn: &BL,
    le_fn: &LE,
    dle_fn: &DLE,
) -> f64
where
    BL: Fn(f64) -> f64,
    LE: Fn(f64) -> f64,
    DLE: Fn(f64) -> f64,
{
    let x = state.displacement;
    let v = state.velocity;
    let bl = bl_fn(x);
    let le = le_fn(x);
    let dle = dle_fn(x);

    // Implicit Euler for di/dt:
    // V = R·i_{n+1} + (L/dt)·(i_{n+1} - i_n) + BL·v + L'·v·i_n
    let back_emf = bl * v + dle * v * state.current;
    let effective_v = voltage - back_emf + le * state.current / dt;
    let effective_r = resistance + le / dt;
    state.current = effective_v / effective_r;

    // Total force: Lorentz + reluctance
    bl * state.current + 0.5 * dle * state.current * state.current
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_electrodynamic_impedance() {
        let driver = ElectrodynamicCoupling {
            resistance: 6.0,
            inductance: 0.5e-3,
            bl: 5.0,
            coil_dofs: vec![(0, 1.0)],
        };
        let omega = 2.0 * PI * 1000.0;
        let z_e = driver.electrical_impedance(omega);
        assert!((z_e.re - 6.0).abs() < 1e-10);
        assert!((z_e.im - omega * 0.5e-3).abs() < 1e-6);
    }

    #[test]
    fn test_motional_impedance_at_resonance() {
        let driver = ElectrodynamicCoupling {
            resistance: 6.0,
            inductance: 0.5e-3,
            bl: 5.0,
            coil_dofs: vec![(0, 1.0)],
        };
        // At mechanical resonance: Z_mech is purely resistive
        let z_mech = Complex64::new(1.0, 0.0); // 1 kg/s mechanical resistance
        let omega = 2.0 * PI * 100.0;
        let z_total = driver.total_impedance(omega, z_mech);
        // Z_mot = BL²/R_mech = 25/1 = 25 Ω (adds to Re)
        let z_e = driver.electrical_impedance(omega);
        assert!(
            z_total.re > z_e.re,
            "Motional impedance should increase Re at resonance"
        );
        assert!(
            (z_total.re - z_e.re - 25.0).abs() < 1e-6,
            "Z_mot = {} Ω",
            z_total.re - z_e.re
        );
    }

    #[test]
    fn test_electrostatic_blocked_impedance() {
        let coupling = ElectrostaticCoupling::from_gap(
            200.0,     // V₀
            25e-6,     // d₀
            8.854e-12, // ε₀
            1e-4,      // Area
            vec![0],
        );
        let omega = 2.0 * PI * 1000.0;
        let z = coupling.blocked_impedance(omega);
        // Should be large (MΩ range for typical condenser mic)
        assert!(z.norm() > 1e6, "|Z| = {} Ω", z.norm());
        // Capacitive: Im < 0
        assert!(z.im < 0.0);
    }

    #[test]
    fn test_electrostatic_coupling_factor() {
        let coupling = ElectrostaticCoupling::from_gap(200.0, 25e-6, 8.854e-12, 1e-4, vec![0]);
        // η = εV₀/d₀² should be ~2.83 Pa/V
        assert!(
            coupling.coupling_factor > 1.0 && coupling.coupling_factor < 10.0,
            "η = {} Pa/V",
            coupling.coupling_factor
        );
    }

    #[test]
    fn test_voice_coil_force_balance() {
        // At DC: V = R·I, F = BL·I
        let mut state = NonlinearTransducerState {
            current: 0.0,
            displacement: 0.0,
            velocity: 0.0,
        };
        let bl = |_x: f64| 5.0;
        let le = |_x: f64| 0.5e-3;
        let dle = |_x: f64| 0.0;
        // Run many small steps to reach steady state
        for _ in 0..10000 {
            voice_coil_nonlinear_step(&mut state, 6.0, 1e-4, 6.0, &bl, &le, &dle);
        }
        // DC: i = V/R = 1 A, F = BL·i = 5 N
        assert!(
            (state.current - 1.0).abs() < 0.01,
            "i = {} A",
            state.current
        );
    }

    #[test]
    fn test_solve_frequency() {
        let driver = ElectrodynamicCoupling {
            resistance: 6.0,
            inductance: 0.5e-3,
            bl: 5.0,
            coil_dofs: vec![(0, 1.0)],
        };
        let omega = 2.0 * PI * 100.0;
        let z_mech = Complex64::new(2.0, 10.0);
        let voltage = Complex64::new(1.0, 0.0);
        let (current, velocity, displacement) = driver.solve_frequency(voltage, omega, z_mech);
        // Current should be non-zero
        assert!(current.norm() > 0.0);
        // Displacement = velocity / (jω)
        let expected_disp = velocity / Complex64::new(0.0, omega);
        assert!((displacement - expected_disp).norm() < 1e-12);
    }
}
