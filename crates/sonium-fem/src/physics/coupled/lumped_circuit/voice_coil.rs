//! Voice coil electrical circuit model
//!
//! Models the electrical side of a loudspeaker motor:
//!
//! ```text
//! V(t) = R·i + L(x)·di/dt + BL(x)·v + L'(x)·v·i
//! ```
//!
//! where:
//! - `V(t)`: input voltage
//! - `R`: DC resistance (ohms)
//! - `L(x)`: voice coil inductance as function of displacement (henries)
//! - `BL(x)`: force factor as function of displacement (T·m)
//! - `v = dx/dt`: voice coil velocity
//! - `i`: voice coil current
//! - `L'(x) = dL/dx`: inductance gradient (generates reluctance force)
//!
//! The force applied to the mechanical system is `F = BL(x)·i`.

/// Voice coil parameters.
///
/// All nonlinear parameters (BL, Le) are functions of displacement `x` from rest.
/// For linear models, use constant functions.
pub struct VoiceCoilParams {
    /// DC resistance (ohms).
    pub resistance: f64,
    /// Voice coil inductance L(x) in henries. Takes displacement in meters.
    pub inductance: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Derivative of inductance dL/dx (H/m). For reluctance force.
    pub inductance_derivative: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Force factor BL(x) in T·m. Takes displacement in meters.
    pub bl: Box<dyn Fn(f64) -> f64 + Send + Sync>,
}

impl std::fmt::Debug for VoiceCoilParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceCoilParams")
            .field("resistance", &self.resistance)
            .finish()
    }
}

impl VoiceCoilParams {
    /// Linear (constant-parameter) voice coil.
    pub fn linear(resistance: f64, inductance: f64, bl: f64) -> Self {
        Self {
            resistance,
            inductance: Box::new(move |_x| inductance),
            inductance_derivative: Box::new(|_x| 0.0),
            bl: Box::new(move |_x| bl),
        }
    }

    /// Nonlinear voice coil with displacement-dependent BL and Le.
    ///
    /// `bl_fn`: BL(x) in T·m
    /// `le_fn`: L(x) in henries
    /// `dle_fn`: dL/dx in H/m
    pub fn nonlinear<B, L, D>(resistance: f64, bl_fn: B, le_fn: L, dle_fn: D) -> Self
    where
        B: Fn(f64) -> f64 + Send + Sync + 'static,
        L: Fn(f64) -> f64 + Send + Sync + 'static,
        D: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        Self {
            resistance,
            inductance: Box::new(le_fn),
            inductance_derivative: Box::new(dle_fn),
            bl: Box::new(bl_fn),
        }
    }

    /// Evaluate BL at displacement x.
    pub fn bl_at(&self, x: f64) -> f64 {
        (self.bl)(x)
    }

    /// Evaluate inductance at displacement x.
    pub fn le_at(&self, x: f64) -> f64 {
        (self.inductance)(x)
    }

    /// Evaluate all constitutive quantities at one displacement.
    pub fn evaluate(&self, displacement_m: f64) -> VoiceCoilOperatingPoint {
        VoiceCoilOperatingPoint {
            resistance_ohm: self.resistance,
            inductance_h: (self.inductance)(displacement_m),
            inductance_gradient_h_per_m: (self.inductance_derivative)(displacement_m),
            force_factor_tm: (self.bl)(displacement_m),
        }
    }
}

/// Immutable voice-coil constitutive evaluation at one displacement.
#[derive(Debug, Clone, Copy)]
pub struct VoiceCoilOperatingPoint {
    /// DC winding resistance at the evaluated state (ohm).
    pub resistance_ohm: f64,
    /// Differential voice-coil inductance at the evaluated displacement (H).
    pub inductance_h: f64,
    /// Displacement derivative of inductance, `dL/dx` (H/m).
    pub inductance_gradient_h_per_m: f64,
    /// Signed force factor `Bl`; positive current produces positive force (T·m).
    pub force_factor_tm: f64,
}

/// Exact inputs and outputs used by one semi-implicit circuit step.
#[derive(Debug, Clone, Copy)]
pub struct CircuitStepAudit {
    /// Current at the beginning of the accepted step (A).
    pub current_old_a: f64,
    /// Current at the end of the accepted step (A).
    pub current_new_a: f64,
    /// Applied endpoint voltage (V).
    pub voltage_v: f64,
    /// Mechanical predictor displacement used for constitutive evaluation (m).
    pub predictor_x_m: f64,
    /// Mechanical predictor velocity, positive in the displacement direction (m/s).
    pub predictor_v_m_per_s: f64,
    /// Immutable constitutive values used by the accepted circuit step.
    pub operating_point: VoiceCoilOperatingPoint,
    /// Electromagnetic force on the mechanical subsystem (N).
    pub force_em_n: f64,
}

impl CircuitStepAudit {
    /// Electrical energy supplied during the endpoint implicit-Euler step.
    pub fn source_energy_j(self, dt: f64) -> f64 {
        dt * self.voltage_v * self.current_new_a
    }

    pub fn copper_loss_j(self, dt: f64) -> f64 {
        dt * self.operating_point.resistance_ohm * self.current_new_a.powi(2)
    }

    pub fn electromagnetic_electrical_work_j(self, dt: f64) -> f64 {
        let op = self.operating_point;
        dt * (op.force_factor_tm * self.predictor_v_m_per_s * self.current_new_a
            + op.inductance_gradient_h_per_m
                * self.predictor_v_m_per_s
                * self.current_old_a
                * self.current_new_a)
    }
}

/// State of the voice coil circuit at a point in time.
#[derive(Debug, Clone)]
pub struct CircuitState {
    /// Current through voice coil (amperes).
    pub current: f64,
    /// Voice coil displacement from rest (meters). Tracked for nonlinear parameter lookup.
    pub displacement: f64,
    /// Voice coil velocity (m/s).
    pub velocity: f64,
}

impl CircuitState {
    /// At-rest initial state.
    pub fn at_rest() -> Self {
        Self {
            current: 0.0,
            displacement: 0.0,
            velocity: 0.0,
        }
    }
}

/// Advance the voice coil circuit by one time step using implicit trapezoidal rule.
///
/// Given the mechanical state (displacement, velocity) at time t_{n+1},
/// computes the new current i_{n+1} and returns the force F = BL(x)·i to apply
/// to the mechanical system.
///
/// The circuit equation `V = R·i + L·di/dt + BL·v` is discretized as:
/// ```text
/// V_{n+1} = R·i_{n+1} + L(x_{n+1}) · (i_{n+1} - i_n)/dt + BL(x_{n+1})·v_{n+1}
/// ```
///
/// Solving for i_{n+1}:
/// ```text
/// i_{n+1} = (V - BL·v - L'·v·i_n + L/dt·i_n) / (R + L/dt)
/// ```
///
/// The `L'(x)·v·i` reluctance term is linearized semi-implicitly using `i_n`.
pub fn circuit_step(
    params: &VoiceCoilParams,
    state: &mut CircuitState,
    voltage: f64,
    displacement: f64,
    velocity: f64,
    dt: f64,
) -> f64 {
    circuit_step_audited(params, state, voltage, displacement, velocity, dt).force_em_n
}

/// Audited variant of [`circuit_step`] preserving its exact numerics.
pub fn circuit_step_audited(
    params: &VoiceCoilParams,
    state: &mut CircuitState,
    voltage: f64,
    displacement: f64,
    velocity: f64,
    dt: f64,
) -> CircuitStepAudit {
    let current_old = state.current;
    state.displacement = displacement;
    state.velocity = velocity;

    let operating_point = params.evaluate(displacement);
    let le = operating_point.inductance_h;
    let bl = operating_point.force_factor_tm;
    let dle = operating_point.inductance_gradient_h_per_m;

    let back_emf = bl * velocity;
    let reluctance_emf = dle * velocity * current_old; // L'(x)·v·i_n (semi-implicit)

    // Implicit Euler for the inductance term
    let l_over_dt = le / dt;
    let i_new = (voltage - back_emf - reluctance_emf + l_over_dt * current_old)
        / (params.resistance + l_over_dt);

    state.current = i_new;

    // Force on the mechanical system (includes reluctance force)
    // F_total = BL(x)·i + 0.5·L'(x)·i^2 (the reluctance force)
    // For the staggered coupling, we return the total force.
    let reluctance_force = 0.5 * dle * i_new * i_new;
    CircuitStepAudit {
        current_old_a: current_old,
        current_new_a: i_new,
        voltage_v: voltage,
        predictor_x_m: displacement,
        predictor_v_m_per_s: velocity,
        operating_point,
        force_em_n: bl * i_new + reluctance_force,
    }
}

/// Compute the electrical impedance Z(f) of the voice coil (blocked, no motion).
///
/// `Z(f) = R + i·2·pi·f·L`
///
/// Returns `(real, imag)` at each frequency.
pub fn blocked_impedance(params: &VoiceCoilParams, frequencies: &[f64]) -> Vec<(f64, f64)> {
    let x = 0.0; // blocked at rest position
    let le = params.le_at(x);
    let r = params.resistance;

    frequencies
        .iter()
        .map(|&f| {
            let omega = 2.0 * std::f64::consts::PI * f;
            (r, omega * le)
        })
        .collect()
}

/// Compute the motional impedance contribution.
///
/// For a linear system, the total impedance is:
/// ```text
/// Z_total = R + jwL + (BL)^2 / Z_mech
/// ```
/// where `Z_mech = jw·Mms + Rms + Kms/(jw)` is the mechanical impedance.
///
/// Returns the motional impedance `(BL)^2 / Z_mech` at each frequency.
pub fn motional_impedance(
    bl: f64,
    mms: f64,
    rms: f64,
    kms: f64,
    frequencies: &[f64],
) -> Vec<(f64, f64)> {
    let bl2 = bl * bl;

    frequencies
        .iter()
        .map(|&f| {
            let omega = 2.0 * std::f64::consts::PI * f;
            // Z_mech = Rms + j(omega*Mms - Kms/omega)
            let z_re = rms;
            let z_im = omega * mms - kms / omega;
            let z_mag_sq = z_re * z_re + z_im * z_im;

            // (BL)^2 / Z_mech = BL^2 * conj(Z_mech) / |Z_mech|^2
            let mot_re = bl2 * z_re / z_mag_sq;
            let mot_im = -bl2 * z_im / z_mag_sq;

            (mot_re, mot_im)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_voice_coil_step() {
        let params = VoiceCoilParams::linear(8.0, 0.001, 10.0);
        let mut state = CircuitState::at_rest();
        let dt = 1e-5;

        // Apply 1V step, no motion
        let force = circuit_step(&params, &mut state, 1.0, 0.0, 0.0, dt);

        // Current should ramp up toward V/R = 1/8 = 0.125 A
        assert!(state.current > 0.0);
        assert!(state.current < 0.125);

        // Force = BL * i
        assert!((force - 10.0 * state.current).abs() < 1e-14);
    }

    #[test]
    fn test_steady_state_current() {
        let params = VoiceCoilParams::linear(8.0, 0.001, 10.0);
        let mut state = CircuitState::at_rest();
        let dt = 1e-5;

        // Run many steps to reach steady state (no motion)
        for _ in 0..10000 {
            circuit_step(&params, &mut state, 1.0, 0.0, 0.0, dt);
        }

        // Steady state: i = V/R = 1/8
        assert!(
            (state.current - 0.125).abs() < 1e-4,
            "Steady state current = {}, expected 0.125",
            state.current
        );
    }

    #[test]
    fn test_back_emf_reduces_current() {
        let params = VoiceCoilParams::linear(8.0, 0.001, 10.0);

        let mut state_no_motion = CircuitState::at_rest();
        let mut state_with_motion = CircuitState::at_rest();
        let dt = 1e-5;

        // Both start with same voltage
        for _ in 0..1000 {
            circuit_step(&params, &mut state_no_motion, 1.0, 0.0, 0.0, dt);
            circuit_step(&params, &mut state_with_motion, 1.0, 0.0, 0.1, dt); // v=0.1 m/s
        }

        // Back-EMF = BL*v = 10*0.1 = 1.0V — this equals the input voltage,
        // so current with motion should be near zero
        assert!(
            state_with_motion.current < state_no_motion.current,
            "Back-EMF should reduce current"
        );
    }

    #[test]
    fn test_blocked_impedance() {
        let params = VoiceCoilParams::linear(8.0, 0.001, 10.0);
        let freqs = vec![100.0, 1000.0, 10000.0];
        let z = blocked_impedance(&params, &freqs);

        // At DC: Z = R = 8
        // Real part is always R
        for &(re, _im) in &z {
            assert!((re - 8.0).abs() < 1e-14);
        }

        // Imaginary part increases with frequency
        assert!(z[2].1 > z[1].1);
        assert!(z[1].1 > z[0].1);
    }

    #[test]
    fn test_motional_impedance_resonance_peak() {
        let bl = 10.0_f64;
        let mms = 0.01_f64;
        let rms = 1.0_f64;
        let kms = 1000.0_f64;
        let fs = (kms / mms).sqrt() / (2.0 * std::f64::consts::PI);

        let freqs: Vec<f64> = (1..200).map(|i| i as f64 * 1.0).collect();
        let z_mot = motional_impedance(bl, mms, rms, kms, &freqs);

        // At resonance, Z_mech is purely real (= Rms), so motional impedance peaks
        let fs_idx = freqs.iter().position(|&f| (f - fs).abs() < 1.0).unwrap();
        let peak_re = z_mot[fs_idx].0;

        // Peak should be BL^2/Rms = 100/1 = 100
        assert!(
            (peak_re - 100.0).abs() < 10.0,
            "Motional impedance peak = {peak_re}, expected ~100"
        );
    }

    #[test]
    fn test_reluctance_term_affects_force() {
        // Nonlinear inductance: Le(x) = 0.001 + 0.01*x, dLe/dx = 0.01
        let params = VoiceCoilParams::nonlinear(
            8.0,
            |_x| 10.0,            // BL constant
            |x| 0.001 + 0.01 * x, // Le(x)
            |_x| 0.01,            // dLe/dx
        );

        let mut state_linear = CircuitState::at_rest();
        let mut state_nonlinear = CircuitState::at_rest();
        let dt = 1e-5;

        let params_linear = VoiceCoilParams::linear(8.0, 0.001, 10.0);

        // Ramp up current with no motion, then apply motion at x=0.01
        for _ in 0..1000 {
            circuit_step(&params_linear, &mut state_linear, 1.0, 0.0, 0.0, dt);
            circuit_step(&params, &mut state_nonlinear, 1.0, 0.0, 0.0, dt);
        }

        // Now apply velocity while displaced
        let force_linear = circuit_step(&params_linear, &mut state_linear, 1.0, 0.01, 0.5, dt);
        let force_nonlinear = circuit_step(&params, &mut state_nonlinear, 1.0, 0.01, 0.5, dt);

        // Nonlinear force should differ due to reluctance term
        assert!(
            (force_linear - force_nonlinear).abs() > 1e-6,
            "Reluctance term should affect force: linear={force_linear}, nonlinear={force_nonlinear}"
        );
    }

    #[test]
    fn audited_blocked_coil_closes_implicit_euler_energy() {
        let params = VoiceCoilParams::linear(6.0, 0.002, 9.0);
        let mut state = CircuitState::at_rest();
        let dt = 2.0e-5;
        let audit = circuit_step_audited(&params, &mut state, 3.0, 0.0, 0.0, dt);
        let delta_magnetic = 0.5
            * audit.operating_point.inductance_h
            * (audit.current_new_a.powi(2) - audit.current_old_a.powi(2));
        let implicit_euler_dissipation = 0.5
            * audit.operating_point.inductance_h
            * (audit.current_new_a - audit.current_old_a).powi(2);
        let residual = audit.source_energy_j(dt)
            - audit.copper_loss_j(dt)
            - delta_magnetic
            - audit.electromagnetic_electrical_work_j(dt);
        assert!((residual - implicit_euler_dissipation).abs() < 1.0e-15);
    }

    #[test]
    fn audited_variable_inductance_records_reluctance_channels_exactly() {
        let params =
            VoiceCoilParams::nonlinear(0.0, |x| 8.0 - 3.0 * x, |x| 0.001 + 0.02 * x, |_x| 0.02);
        let mut state = CircuitState {
            current: 0.4,
            displacement: 0.0,
            velocity: 0.0,
        };
        let audit = circuit_step_audited(&params, &mut state, 2.0, 0.01, 0.3, 1.0e-4);
        assert_eq!(audit.current_old_a, 0.4);
        assert_eq!(audit.operating_point.inductance_gradient_h_per_m, 0.02);
        let expected_force = audit.operating_point.force_factor_tm * audit.current_new_a
            + 0.5 * audit.operating_point.inductance_gradient_h_per_m * audit.current_new_a.powi(2);
        assert!((audit.force_em_n - expected_force).abs() < 1.0e-14);
        let expected_work = 1.0e-4
            * (audit.operating_point.force_factor_tm * 0.3 * audit.current_new_a
                + 0.02 * 0.3 * 0.4 * audit.current_new_a);
        assert!((audit.electromagnetic_electrical_work_j(1.0e-4) - expected_work).abs() < 1.0e-16);
    }

    #[test]
    fn variable_inductance_staggering_residual_converges_under_dt_halving() {
        let params =
            VoiceCoilParams::nonlinear(0.0, |x| 8.0 - 3.0 * x, |x| 0.001 + 0.02 * x, |_x| 0.02);
        let physical_window = 1.0e-4;
        let prescribed_velocity = 0.3;
        let accumulated_residual = |dt: f64| {
            let mut state = CircuitState {
                current: 0.4,
                displacement: 0.01,
                velocity: prescribed_velocity,
            };
            let mut x = state.displacement;
            let mut residual = 0.0;
            for _ in 0..(physical_window / dt).round() as usize {
                let inductance_old = params.evaluate(x).inductance_h;
                x += prescribed_velocity * dt;
                let audit =
                    circuit_step_audited(&params, &mut state, 2.0, x, prescribed_velocity, dt);
                // Changing L stores half of the L'(x) v i² electrical work
                // magnetically; only the other half is the reluctance-force
                // work. Excluding that storage term gives this test a
                // timestep-independent O(dt) floor instead of measuring the
                // semi-implicit staggering error it is intended to isolate.
                let magnetic_geometry_delta = 0.5
                    * (audit.operating_point.inductance_h - inductance_old)
                    * audit.current_new_a.powi(2);
                residual += audit.electromagnetic_electrical_work_j(dt)
                    - audit.force_em_n * prescribed_velocity * dt
                    - magnetic_geometry_delta;
            }
            residual.abs()
        };
        let coarse = accumulated_residual(physical_window);
        let fine = accumulated_residual(physical_window / 2.0);
        assert!(coarse > 0.0);
        assert!(
            coarse >= 1.8 * fine,
            "EM staggering residual improved only {}x under dt/2",
            coarse / fine
        );
    }
}
