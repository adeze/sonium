//! Nonlinear time-domain coupled transducer simulation
//!
//! Staggered predictor-corrector coupling of electrical (voice coil) and
//! mechanical (lumped or FEM) domains using Newmark-β time integration.
//!
//! # Coupling algorithm
//!
//! At each time step n → n+1:
//! 1. **Predict** mechanical state from current acceleration
//! 2. **Electrical step**: given predicted velocity/displacement, advance
//!    voice coil circuit → get current i_{n+1} and force F_{n+1}
//! 3. **Mechanical corrector**: solve Newmark effective system with F_{n+1}
//! 4. **Correct** displacement and velocity
//!
//! # Lumped model
//!
//! For a single-DOF lumped transducer:
//! ```text
//! Mms·x'' + Rms·x' + Kms·x = BL(x)·i + ½·L'(x)·i²
//! V(t) = Re·i + Le(x)·di/dt + BL(x)·x' + Le'(x)·x'·i
//! ```
//!
//! The mechanical system is a scalar Newmark integration and the electrical
//! circuit is advanced via [`circuit_step`](crate::physics::coupled::lumped_circuit::circuit_step).

use crate::physics::coupled::lumped_circuit::{CircuitState, VoiceCoilParams, circuit_step};

/// Configuration for lumped time-domain transducer simulation.
pub struct LumpedTransducerConfig {
    /// Moving mass Mms (kg).
    pub mms: f64,
    /// Mechanical resistance Rms (kg/s).
    pub rms: f64,
    /// Mechanical stiffness Kms (N/m).
    /// For nonlinear compliance, set this to the nominal (small-signal) value
    /// and provide the full restoring force via `compliance_force_fn`.
    /// The Newmark effective system uses this linear value in the denominator,
    /// so the corrector is only first-order accurate for nonlinear springs.
    /// Strongly hardening springs (effective stiffness >> Kms) can cause
    /// instability — reduce dt or increase Kms toward the expected maximum.
    pub kms: f64,
    /// Voice coil parameters (R, L(x), BL(x)).
    pub voice_coil: VoiceCoilParams,
    /// Nonlinear compliance force: f(x) returning the restoring force.
    /// If `None`, linear stiffness Kms·x is used.
    pub compliance_force_fn: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
    /// Input voltage waveform V(t).
    pub voltage_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Newmark β parameter (default: 0.25 = average acceleration).
    pub beta: f64,
    /// Newmark γ parameter (default: 0.5).
    pub gamma: f64,
    /// Time step (s).
    pub dt: f64,
    /// Number of time steps.
    pub num_steps: usize,
}

impl LumpedTransducerConfig {
    /// Create a linear lumped transducer configuration.
    pub fn linear(
        mms: f64,
        rms: f64,
        kms: f64,
        voice_coil: VoiceCoilParams,
        voltage_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        dt: f64,
        num_steps: usize,
    ) -> Self {
        Self {
            mms,
            rms,
            kms,
            voice_coil,
            compliance_force_fn: None,
            voltage_fn,
            beta: 0.25,
            gamma: 0.5,
            dt,
            num_steps,
        }
    }
}

/// Time-domain simulation result.
#[derive(Debug, Clone)]
pub struct TimeDomainResult {
    /// Time points (s).
    pub time: Vec<f64>,
    /// Displacement history (m).
    pub displacement: Vec<f64>,
    /// Velocity history (m/s).
    pub velocity: Vec<f64>,
    /// Voice coil current history (A).
    pub current: Vec<f64>,
    /// Applied force history (N): BL·i + reluctance force.
    pub force: Vec<f64>,
    /// Number of steps completed.
    pub steps_completed: usize,
}

impl TimeDomainResult {
    /// Acceleration at each time step, derived from the force balance.
    pub fn acceleration(&self, mms: f64, rms: f64, kms: f64) -> Vec<f64> {
        self.displacement
            .iter()
            .zip(self.velocity.iter())
            .zip(self.force.iter())
            .map(|((&x, &v), &f)| (f - rms * v - kms * x) / mms)
            .collect()
    }

    /// Kinetic + potential energy at each time step.
    pub fn energy(&self, mms: f64, kms: f64) -> Vec<f64> {
        self.displacement
            .iter()
            .zip(self.velocity.iter())
            .map(|(&x, &v)| 0.5 * mms * v * v + 0.5 * kms * x * x)
            .collect()
    }

    /// RMS displacement over the last `n` samples.
    pub fn rms_displacement(&self, n: usize) -> f64 {
        let start = self.displacement.len().saturating_sub(n);
        let slice = &self.displacement[start..];
        let sum_sq: f64 = slice.iter().map(|x| x * x).sum();
        (sum_sq / slice.len() as f64).sqrt()
    }

    /// RMS current over the last `n` samples.
    pub fn rms_current(&self, n: usize) -> f64 {
        let start = self.current.len().saturating_sub(n);
        let slice = &self.current[start..];
        let sum_sq: f64 = slice.iter().map(|i| i * i).sum();
        (sum_sq / slice.len() as f64).sqrt()
    }
}

/// Run a lumped single-DOF transducer time-domain simulation.
///
/// Integrates the coupled electrical-mechanical system using Newmark-β
/// with staggered predictor-corrector coupling to the voice coil circuit.
pub fn simulate_lumped_transducer(config: &LumpedTransducerConfig) -> TimeDomainResult {
    let dt = config.dt;
    let beta = config.beta;
    let gamma = config.gamma;

    // Newmark effective "stiffness" for the scalar 1-DOF system:
    // A_eff = M + gamma*dt*C + beta*dt²*K
    let kms_eff = config.kms;
    let a_eff = config.mms + gamma * dt * config.rms + beta * dt * dt * kms_eff;

    // Initial state: at rest
    let mut x = 0.0_f64; // displacement
    let mut v = 0.0_f64; // velocity
    let mut a = 0.0_f64; // acceleration
    let mut circuit = CircuitState::at_rest();

    // History storage
    let capacity = config.num_steps + 1;
    let mut time_hist = Vec::with_capacity(capacity);
    let mut x_hist = Vec::with_capacity(capacity);
    let mut v_hist = Vec::with_capacity(capacity);
    let mut i_hist = Vec::with_capacity(capacity);
    let mut f_hist = Vec::with_capacity(capacity);

    // Record initial state
    time_hist.push(0.0);
    x_hist.push(x);
    v_hist.push(v);
    i_hist.push(0.0);
    f_hist.push(0.0);

    for step in 0..config.num_steps {
        let t_next = (step + 1) as f64 * dt;

        // --- Newmark predictor ---
        let x_pred = x + dt * v + (0.5 - beta) * dt * dt * a;
        let v_pred = v + (1.0 - gamma) * dt * a;

        // --- Electrical step ---
        // Use predicted mechanical state to advance voice coil circuit
        let voltage = (config.voltage_fn)(t_next);
        let force_em = circuit_step(
            &config.voice_coil,
            &mut circuit,
            voltage,
            x_pred,
            v_pred,
            dt,
        );

        // --- Mechanical RHS ---
        // f_{n+1} - C*v_pred - K*u_pred
        let spring_force = match &config.compliance_force_fn {
            Some(f) => f(x_pred),
            None => kms_eff * x_pred,
        };
        let rhs = force_em - config.rms * v_pred - spring_force;

        // --- Solve for acceleration ---
        let a_new = rhs / a_eff;

        // --- Newmark corrector ---
        x = x_pred + beta * dt * dt * a_new;
        v = v_pred + gamma * dt * a_new;
        a = a_new;

        // Record
        time_hist.push(t_next);
        x_hist.push(x);
        v_hist.push(v);
        i_hist.push(circuit.current);
        f_hist.push(force_em);
    }

    TimeDomainResult {
        time: time_hist,
        displacement: x_hist,
        velocity: v_hist,
        current: i_hist,
        force: f_hist,
        steps_completed: config.num_steps,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Reference Thiele-Small parameters for a typical 6.5" woofer.
    fn test_driver() -> (f64, f64, f64, VoiceCoilParams) {
        let mms = 0.015; // 15 g
        let rms = 1.0; // 1 kg/s
        let kms = 2400.0; // N/m → fs ≈ 64 Hz
        let re = 6.0;
        let le = 0.5e-3;
        let bl = 5.0;
        let vc = VoiceCoilParams::linear(re, le, bl);
        (mms, rms, kms, vc)
    }

    #[test]
    fn test_free_vibration_frequency() {
        // No voltage input, initial displacement → should oscillate at fs
        let (mms, _rms, kms, vc) = test_driver();
        let fs = (kms / mms).sqrt() / (2.0 * PI);

        // Zero voltage, but we need to push the cone initially.
        // We'll use a short impulse voltage to set the system in motion,
        // then let it ring free.
        let dt = 1.0 / (fs * 200.0); // 200 samples per period
        let impulse_duration = dt * 5.0;
        let config = LumpedTransducerConfig::linear(
            mms,
            0.0,
            kms, // zero damping for clean oscillation
            VoiceCoilParams::linear(vc.resistance, 0.0, vc.bl_at(0.0)), // zero L for fast response
            Box::new(move |t| if t < impulse_duration { 10.0 } else { 0.0 }),
            dt,
            (5.0 / fs / dt) as usize, // 5 periods
        );

        let result = simulate_lumped_transducer(&config);

        // Find zero crossings after the impulse to estimate frequency
        let start_idx = (impulse_duration / dt) as usize + 10;
        let mut crossings = Vec::new();
        for i in start_idx..result.displacement.len() - 1 {
            if result.displacement[i] * result.displacement[i + 1] < 0.0 {
                // Linear interpolation for exact crossing time
                let t0 = result.time[i];
                let t1 = result.time[i + 1];
                let x0 = result.displacement[i];
                let x1 = result.displacement[i + 1];
                let t_cross = t0 - x0 * (t1 - t0) / (x1 - x0);
                crossings.push(t_cross);
            }
        }

        // Need at least 2 crossings (half-period) to estimate frequency
        assert!(
            crossings.len() >= 4,
            "Not enough zero crossings: {}",
            crossings.len()
        );

        // Frequency from zero crossing intervals (2 crossings = 1 half-period)
        let half_periods: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
        let avg_half_period: f64 = half_periods.iter().sum::<f64>() / half_periods.len() as f64;
        let measured_fs = 1.0 / (2.0 * avg_half_period);

        // Allow ~8% tolerance: voice coil back-EMF shifts resonance slightly
        let error_pct = ((measured_fs - fs) / fs * 100.0).abs();
        assert!(
            error_pct < 8.0,
            "Measured fs = {measured_fs:.2} Hz, expected {fs:.2} Hz (error {error_pct:.1}%)"
        );
    }

    #[test]
    fn test_dc_steady_state() {
        // Apply constant voltage → steady state: i = V/Re, x = BL·i/Kms
        let (mms, rms, kms, vc) = test_driver();
        let v_dc = 1.0;
        let bl = vc.bl_at(0.0);
        let re = vc.resistance;

        let dt = 1e-5;
        // Run long enough for transients to die (> 5 mechanical time constants)
        let tau_mech = mms / rms;
        let num_steps = (10.0 * tau_mech / dt) as usize;

        let config = LumpedTransducerConfig::linear(
            mms,
            rms,
            kms,
            vc,
            Box::new(move |_t| v_dc),
            dt,
            num_steps,
        );

        let result = simulate_lumped_transducer(&config);

        // DC steady state: V = Re·i + BL·0 (velocity=0 in steady state)
        let expected_current = v_dc / re;
        let expected_displacement = bl * expected_current / kms;

        let final_i = *result.current.last().unwrap();
        let final_x = *result.displacement.last().unwrap();

        assert!(
            (final_i - expected_current).abs() / expected_current < 0.01,
            "Steady-state current: {final_i:.4} A, expected {expected_current:.4} A"
        );
        assert!(
            (final_x - expected_displacement).abs() / expected_displacement < 0.05,
            "Steady-state displacement: {final_x:.6} m, expected {expected_displacement:.6} m"
        );
    }

    #[test]
    fn test_energy_decay_with_damping() {
        // With damping and no input: energy should monotonically decrease
        let (mms, rms, kms, vc) = test_driver();
        let fs = (kms / mms).sqrt() / (2.0 * PI);
        let dt = 1.0 / (fs * 100.0);
        let impulse_end = dt * 10.0;

        let config = LumpedTransducerConfig::linear(
            mms,
            rms,
            kms,
            vc,
            Box::new(move |t| if t < impulse_end { 5.0 } else { 0.0 }),
            dt,
            (3.0 / fs / dt) as usize,
        );

        let result = simulate_lumped_transducer(&config);
        let energy = result.energy(mms, kms);

        // After the impulse, energy should decay (allow electrical losses too)
        let start = (impulse_end / dt) as usize + 50;
        for i in start + 1..energy.len() {
            // Allow small numerical tolerance
            assert!(
                energy[i] <= energy[i - 1] * 1.001,
                "Energy not decaying at step {i}: E[{i}]={:.6e} > E[{}]={:.6e}",
                energy[i],
                i - 1,
                energy[i - 1]
            );
        }

        // Final energy should be much smaller than peak
        let peak_energy = energy[start..].iter().cloned().fold(0.0_f64, f64::max);
        let final_energy = *energy.last().unwrap();
        assert!(
            final_energy < peak_energy * 0.1,
            "Energy should decay significantly: peak={peak_energy:.4e}, final={final_energy:.4e}"
        );
    }

    #[test]
    fn test_small_signal_matches_frequency_domain() {
        // Small-signal sine → compare RMS displacement against frequency-domain prediction
        let (mms, rms, kms, vc) = test_driver();
        let bl = vc.bl_at(0.0);
        let re = vc.resistance;
        let le = vc.le_at(0.0);

        let f_test = 100.0; // Hz
        let omega = 2.0 * PI * f_test;
        let v_amplitude = 0.1; // small signal

        // Frequency-domain prediction
        use num_complex::Complex64;
        let z_e = Complex64::new(re, omega * le);
        let z_mech = Complex64::new(rms, omega * mms - kms / omega);
        let z_total = z_e + Complex64::new(bl * bl, 0.0) / z_mech;
        let i_phasor = Complex64::new(v_amplitude, 0.0) / z_total;
        let f_phasor = Complex64::new(bl, 0.0) * i_phasor;
        let v_phasor = f_phasor / z_mech;
        let x_phasor = v_phasor / Complex64::new(0.0, omega); // displacement = velocity / (jω)
        let x_rms_expected = x_phasor.norm() / std::f64::consts::SQRT_2;

        // Time-domain simulation
        let dt = 1.0 / (f_test * 100.0);
        let n_settle = (10.0 / f_test / dt) as usize; // 10 periods to settle
        let n_measure = (5.0 / f_test / dt) as usize; // 5 periods to measure
        let total_steps = n_settle + n_measure;

        let config = LumpedTransducerConfig::linear(
            mms,
            rms,
            kms,
            vc,
            Box::new(move |t| v_amplitude * (omega * t).sin()),
            dt,
            total_steps,
        );

        let result = simulate_lumped_transducer(&config);
        let x_rms_measured = result.rms_displacement(n_measure);

        let error_pct = ((x_rms_measured - x_rms_expected) / x_rms_expected * 100.0).abs();
        assert!(
            error_pct < 5.0,
            "x_rms: measured={x_rms_measured:.6e}, expected={x_rms_expected:.6e} (error {error_pct:.1}%)"
        );
    }

    #[test]
    fn test_back_emf_limits_velocity() {
        // At resonance with zero damping, back-EMF is the only velocity limiter.
        // The steady-state velocity should be bounded by V/(BL).
        let mms = 0.01_f64;
        let kms = 1000.0_f64;
        let fs = (kms / mms).sqrt() / (2.0 * PI);
        let bl = 10.0;
        let re = 8.0;

        let vc = VoiceCoilParams::linear(re, 0.0, bl); // zero inductance for simplicity
        let v_amp = 1.0;
        let omega = 2.0 * PI * fs;
        let dt = 1.0 / (fs * 200.0);
        let num_steps = (20.0 / fs / dt) as usize;

        let config = LumpedTransducerConfig::linear(
            mms,
            0.0,
            kms,
            vc, // zero mechanical damping
            Box::new(move |t| v_amp * (omega * t).sin()),
            dt,
            num_steps,
        );

        let result = simulate_lumped_transducer(&config);

        // Peak velocity should be finite and bounded
        let v_peak = result
            .velocity
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        // With only electrical damping (back-EMF): v_peak ≈ V/(BL²/Re) at resonance
        // BL²/Re = 100/8 = 12.5 → equivalent Rms from back-EMF
        // v_peak ≈ BL·V / (Re·BL²/Re) = V/BL = 0.1 m/s
        let v_expected_order = v_amp / bl;
        assert!(
            v_peak < v_expected_order * 5.0,
            "Peak velocity {v_peak:.4} m/s should be bounded near {v_expected_order:.4} m/s"
        );
        assert!(v_peak > 0.0, "System should oscillate");
    }

    #[test]
    fn test_nonlinear_bl_produces_harmonics() {
        // Symmetric BL(x) = BL0*(1 - x²/x_max²) → should produce odd harmonics
        let mms = 0.01;
        let rms = 2.0;
        let kms = 1000.0;
        let re = 6.0;
        let bl0 = 5.0;
        let x_max = 0.005;

        let vc = VoiceCoilParams::nonlinear(
            re,
            move |x| bl0 * (1.0 - (x / x_max).powi(2)).max(0.1),
            |_x| 0.5e-3,
            |_x| 0.0,
        );

        let f_test = 80.0;
        let omega = 2.0 * PI * f_test;
        let dt = 1.0 / (f_test * 200.0);
        let num_steps = (20.0 / f_test / dt) as usize;

        let config = LumpedTransducerConfig {
            mms,
            rms,
            kms,
            voice_coil: vc,
            compliance_force_fn: None,
            voltage_fn: Box::new(move |t| 2.0 * (omega * t).sin()), // moderate drive
            beta: 0.25,
            gamma: 0.5,
            dt,
            num_steps,
        };

        let result = simulate_lumped_transducer(&config);

        // Verify the system ran without blowing up
        let x_peak = result
            .displacement
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            x_peak < x_max * 3.0,
            "Displacement {x_peak:.4} m should be bounded"
        );
        assert!(x_peak > 0.0, "System should respond");

        // Simple DFT at fundamental and 3rd harmonic to verify nonlinear content
        let n_fft = (5.0 / f_test / dt) as usize;
        let start = result.displacement.len() - n_fft;
        let (h1, h3) = dft_harmonics(&result.displacement[start..], dt, f_test);

        // With nonlinear BL, 3rd harmonic should be non-zero
        // (much smaller than fundamental but measurable)
        let thd3 = h3 / h1;
        assert!(
            thd3 > 1e-6,
            "3rd harmonic should be present with nonlinear BL: H1={h1:.4e}, H3={h3:.4e}"
        );
    }

    /// Simple DFT to extract amplitude at fundamental and 3rd harmonic.
    fn dft_harmonics(signal: &[f64], dt: f64, fundamental_freq: f64) -> (f64, f64) {
        let n = signal.len() as f64;
        let omega1 = 2.0 * PI * fundamental_freq;
        let omega3 = 3.0 * omega1;

        let mut re1 = 0.0;
        let mut im1 = 0.0;
        let mut re3 = 0.0;
        let mut im3 = 0.0;

        for (k, &x) in signal.iter().enumerate() {
            let t = k as f64 * dt;
            re1 += x * (omega1 * t).cos();
            im1 += x * (omega1 * t).sin();
            re3 += x * (omega3 * t).cos();
            im3 += x * (omega3 * t).sin();
        }

        let scale = 2.0 / n;
        let h1 = (re1 * re1 + im1 * im1).sqrt() * scale;
        let h3 = (re3 * re3 + im3 * im3).sqrt() * scale;
        (h1, h3)
    }
}
