//! Lumped box-compliance + nonlinear driver ODE, stepped in lockstep with the
//! Navier–Stokes port solve.
//!
//! [`BoxDriverModel::step`] preserves the legacy sequential update algorithm
//! of
//! `sonium_fem::physics::acoustics::electroacoustics::time_domain::simulate_lumped_transducer`
//! (Newmark-β mechanics + `circuit_step` voice coil), and adds the sealed-box
//! acoustic compliance on top:
//!
//! ```text
//! p_box = ρc²/V · (Sd·x − ∫q_port dt)      (q_port positive box → exterior)
//! F_box = −p_box·Sd                        (restoring, on the mechanical RHS)
//! ```
//!
//! The box pressure is updated once per step, after the Newmark corrector,
//! using the step-average (trapezoidal) diaphragm velocity:
//!
//! ```text
//! p_box += ρc²/V · (Sd·v_avg·dt − q_port·dt)
//! ```
//!
//! Free air (`box_volume_m3: None`) skips the box-pressure term entirely.
//! Front-side radiation loading is out of M1 scope (see Task 11 provenance).

use sonium_fem::physics::coupled::lumped_circuit::{
    CircuitState, CircuitStepAudit, VoiceCoilParams, circuit_step_audited,
};
use sonium_solvers::time_integration::{
    acceleration_effective_scalar, correct_displacement_from_acceleration,
    correct_velocity_from_acceleration, predict_displacement_scalar, predict_velocity_scalar,
};

/// Lumped driver + optional sealed-box compliance, one DOF.
///
/// The voice coil carries the nonlinear BL(x) / Le(x) behaviour; the moving
/// assembly is linear Mms/Rms/Kms unless `compliance_force_fn` overrides the
/// spring restoring force.
pub struct BoxDriverModel {
    /// Moving mass Mms (kg).
    pub mms_kg: f64,
    /// Mechanical resistance Rms (kg/s).
    pub rms_kg_per_s: f64,
    /// Mechanical stiffness Kms (N/m). Nominal small-signal value when
    /// `compliance_force_fn` is set (the Newmark effective system always uses
    /// this linear value in the denominator).
    pub kms_n_per_m: f64,
    /// Voice coil parameters (R, L(x), BL(x)).
    pub voice_coil: VoiceCoilParams,
    /// Nonlinear compliance restoring force f(x). `None` = linear Kms·x.
    pub compliance_force_fn: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
    /// Potential energy U(x) paired with `compliance_force_fn`. Audited
    /// nonlinear runs require this to preserve an energy-consistent model.
    pub compliance_potential_fn: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
    /// Effective diaphragm area Sd (m²).
    pub sd_m2: f64,
    /// Sealed box volume (m³). `None` = free air (no box-pressure term).
    pub box_volume_m3: Option<f64>,
    /// Ambient medium density ρ (kg/m³).
    pub medium_density_kg_per_m3: f64,
    /// Ambient medium sound speed c (m/s).
    pub medium_sound_speed_m_per_s: f64,
    /// Newmark β (0.25 = average acceleration).
    pub beta: f64,
    /// Newmark γ (0.5).
    pub gamma: f64,
}

/// State of the lumped box + driver system at a point in time.
pub struct BoxDriverState {
    /// Diaphragm displacement from rest (m), positive outward.
    pub x_m: f64,
    /// Diaphragm velocity (m/s).
    pub v_m_per_s: f64,
    /// Diaphragm acceleration (m/s²).
    pub a_m_per_s2: f64,
    /// Box gauge pressure (Pa): positive when the box is compressed.
    pub p_box_pa: f64,
    /// Voice coil circuit state.
    pub circuit: CircuitState,
}

/// Driver-side signed energy channels for one completed, pressure-corrected
/// step. Energies are joules; positive work enters the driver/box subsystem.
#[derive(Debug, Clone, Copy)]
pub struct BoxDriverStepAudit {
    pub x_old_m: f64,
    pub v_old_m_per_s: f64,
    pub p_box_old_pa: f64,
    pub circuit: CircuitStepAudit,
    pub predictor_spring_force_n: f64,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct BoxDriverEnergyStep {
    pub source_j: f64,
    pub copper_loss_j: f64,
    pub inductive_storage_old_j: f64,
    pub inductive_storage_new_j: f64,
    pub electrical_numerical_j: f64,
    /// Electrical-minus-mechanical electromagnetic transfer work caused by
    /// predictor-velocity/end-displacement constitutive staggering (J).
    pub electromagnetic_staggering_j: f64,
    pub mechanical_kinetic_old_j: f64,
    pub mechanical_kinetic_new_j: f64,
    pub suspension_storage_old_j: f64,
    pub suspension_storage_new_j: f64,
    pub suspension_numerical_j: f64,
    pub mechanical_damping_j: f64,
    pub mechanical_damping_midpoint_j: f64,
    pub mechanical_residual_j: f64,
    pub box_storage_old_j: f64,
    pub box_storage_new_j: f64,
    pub driver_to_box_work_j: f64,
    pub box_on_driver_work_j: f64,
}

impl Default for BoxDriverState {
    fn default() -> Self {
        Self {
            x_m: 0.0,
            v_m_per_s: 0.0,
            a_m_per_s2: 0.0,
            p_box_pa: 0.0,
            circuit: CircuitState::at_rest(),
        }
    }
}

impl BoxDriverModel {
    /// Install an energy-consistent nonlinear suspension law. The force and
    /// potential are paired by construction so audited runs can account for
    /// conservative suspension storage without numerical quadrature.
    pub fn with_nonlinear_suspension<F, U>(mut self, force_n: F, potential_j: U) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
        U: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        self.compliance_force_fn = Some(Box::new(force_n));
        self.compliance_potential_fn = Some(Box::new(potential_j));
        self
    }

    /// At-rest initial state.
    pub fn new_state(&self) -> BoxDriverState {
        BoxDriverState::default()
    }

    /// Advance one Newmark lockstep.
    ///
    /// `voltage` is the terminal voltage at t_{n+1} (V). `q_port_m3s` is the
    /// port volume flow OUT of the box (positive box → exterior), taken from
    /// the previous N-S step.
    pub fn step(&self, st: &mut BoxDriverState, voltage: f64, q_port_m3s: f64, dt: f64) {
        self.step_pressure_base(st, voltage, dt);
        st.p_box_pa -= self.port_pressure_coefficient(dt) * q_port_m3s;
    }

    /// Advance the electrical/mechanical state and diaphragm contribution to
    /// box pressure, leaving the port-flow contribution for the coupled
    /// end-of-step interface solve.
    pub(crate) fn step_pressure_base(&self, st: &mut BoxDriverState, voltage: f64, dt: f64) {
        let _ = self.step_pressure_base_impl(st, voltage, dt, false);
    }

    pub(crate) fn step_pressure_base_audited(
        &self,
        st: &mut BoxDriverState,
        voltage: f64,
        dt: f64,
    ) -> BoxDriverStepAudit {
        self.step_pressure_base_impl(st, voltage, dt, true)
    }

    fn step_pressure_base_impl(
        &self,
        st: &mut BoxDriverState,
        voltage: f64,
        dt: f64,
        require_potential: bool,
    ) -> BoxDriverStepAudit {
        assert!(
            !require_potential
                || self.compliance_force_fn.is_none()
                || self.compliance_potential_fn.is_some(),
            "audited nonlinear suspension requires a matching potential-energy function"
        );
        let x_old = st.x_m;
        let v_old = st.v_m_per_s;
        let p_box_old = st.p_box_pa;
        let beta = self.beta;
        let gamma = self.gamma;

        // Newmark effective "stiffness": A_eff = M + γ·dt·C + β·dt²·K
        let a_eff = acceleration_effective_scalar(
            self.mms_kg,
            self.rms_kg_per_s,
            self.kms_n_per_m,
            beta,
            gamma,
            dt,
        );

        // --- Newmark predictor ---
        let x_pred = predict_displacement_scalar(beta, dt, st.x_m, st.v_m_per_s, st.a_m_per_s2);
        let v_pred = predict_velocity_scalar(gamma, dt, st.v_m_per_s, st.a_m_per_s2);

        // --- Electrical step ---
        let circuit = circuit_step_audited(
            &self.voice_coil,
            &mut st.circuit,
            voltage,
            x_pred,
            v_pred,
            dt,
        );
        let force_em = circuit.force_em_n;

        // --- Mechanical RHS ---
        // f_{n+1} + F_box − C·v_pred − K·u_pred
        let spring_force = match &self.compliance_force_fn {
            Some(f) => f(x_pred),
            None => self.kms_n_per_m * x_pred,
        };
        let f_acoustic = if self.box_volume_m3.is_some() {
            -st.p_box_pa * self.sd_m2
        } else {
            0.0
        };
        let rhs = force_em + f_acoustic - self.rms_kg_per_s * v_pred - spring_force;

        // --- Solve for acceleration ---
        let a_new = rhs / a_eff;

        // --- Newmark corrector ---
        st.x_m = correct_displacement_from_acceleration(beta, dt, x_pred, a_new);
        st.v_m_per_s = correct_velocity_from_acceleration(gamma, dt, v_pred, a_new);
        st.a_m_per_s2 = a_new;

        // --- Box-pressure update (trapezoidal diaphragm velocity) ---
        if let Some(volume) = self.box_volume_m3 {
            let v_avg = 0.5 * (v_old + st.v_m_per_s);
            let compression =
                self.medium_density_kg_per_m3 * self.medium_sound_speed_m_per_s.powi(2) / volume;
            st.p_box_pa += compression * self.sd_m2 * v_avg * dt;
        }
        BoxDriverStepAudit {
            x_old_m: x_old,
            v_old_m_per_s: v_old,
            p_box_old_pa: p_box_old,
            circuit,
            predictor_spring_force_n: spring_force,
        }
    }

    fn suspension_potential_j(&self, x_m: f64) -> f64 {
        self.compliance_potential_fn
            .as_ref()
            .map_or(0.5 * self.kms_n_per_m * x_m * x_m, |potential| {
                potential(x_m)
            })
    }

    pub(crate) fn finalize_energy_step(
        &self,
        audit: BoxDriverStepAudit,
        final_state: &BoxDriverState,
        dt: f64,
    ) -> BoxDriverEnergyStep {
        let dx = final_state.x_m - audit.x_old_m;
        let circuit = audit.circuit;
        let old_op = self.voice_coil.evaluate(audit.x_old_m);
        let new_op = self.voice_coil.evaluate(final_state.x_m);
        let inductive_old = 0.5 * old_op.inductance_h * circuit.current_old_a.powi(2);
        let inductive_new = 0.5 * new_op.inductance_h * circuit.current_new_a.powi(2);
        let em_electrical = circuit.electromagnetic_electrical_work_j(dt);
        let em_mechanical = circuit.force_em_n * dx;
        let spring_algorithmic = (audit.predictor_spring_force_n
            + self.kms_n_per_m * (final_state.x_m - circuit.predictor_x_m))
            * dx;
        let suspension_old = self.suspension_potential_j(audit.x_old_m);
        let suspension_new = self.suspension_potential_j(final_state.x_m);
        let kinetic_old = 0.5 * self.mms_kg * audit.v_old_m_per_s.powi(2);
        let kinetic_new = 0.5 * self.mms_kg * final_state.v_m_per_s.powi(2);
        let damping = self.rms_kg_per_s * dx * final_state.v_m_per_s;
        let box_on_driver = -final_state.p_box_pa * self.sd_m2 * dx;
        let compliance = self.box_volume_m3.map_or(0.0, |volume| {
            volume / (self.medium_density_kg_per_m3 * self.medium_sound_speed_m_per_s.powi(2))
        });
        BoxDriverEnergyStep {
            source_j: circuit.source_energy_j(dt),
            copper_loss_j: circuit.copper_loss_j(dt),
            inductive_storage_old_j: inductive_old,
            inductive_storage_new_j: inductive_new,
            electrical_numerical_j: circuit.source_energy_j(dt)
                - circuit.copper_loss_j(dt)
                - (inductive_new - inductive_old)
                - em_electrical,
            electromagnetic_staggering_j: em_electrical - em_mechanical,
            mechanical_kinetic_old_j: kinetic_old,
            mechanical_kinetic_new_j: kinetic_new,
            suspension_storage_old_j: suspension_old,
            suspension_storage_new_j: suspension_new,
            suspension_numerical_j: spring_algorithmic - (suspension_new - suspension_old),
            mechanical_damping_j: damping,
            mechanical_damping_midpoint_j: dt
                * self.rms_kg_per_s
                * (0.5 * (audit.v_old_m_per_s + final_state.v_m_per_s)).powi(2),
            mechanical_residual_j: em_mechanical + box_on_driver
                - (kinetic_new - kinetic_old)
                - damping
                - spring_algorithmic,
            box_storage_old_j: 0.5 * compliance * audit.p_box_old_pa.powi(2),
            box_storage_new_j: 0.5 * compliance * final_state.p_box_pa.powi(2),
            driver_to_box_work_j: final_state.p_box_pa * self.sd_m2 * dx,
            box_on_driver_work_j: box_on_driver,
        }
    }

    /// Pressure decrement per unit end-of-step port volume flow.
    pub(crate) fn port_pressure_coefficient(&self, dt: f64) -> f64 {
        self.box_volume_m3.map_or(0.0, |volume| {
            self.medium_density_kg_per_m3 * self.medium_sound_speed_m_per_s.powi(2) / volume * dt
        })
    }

    /// Dimensionless change in diaphragm-generated box pressure per pascal of
    /// end-of-step box pressure fed back through the Newmark acceleration.
    pub(crate) fn diaphragm_pressure_feedback(&self, dt: f64) -> f64 {
        let alpha = self.port_pressure_coefficient(dt);
        let a_eff = self.mms_kg
            + self.gamma * dt * self.rms_kg_per_s
            + self.beta * dt * dt * self.kms_n_per_m;
        alpha * self.sd_m2 * 0.5 * self.gamma * dt * self.sd_m2 / a_eff
    }

    /// Apply the affine Newmark response to an end-of-step acoustic-pressure
    /// correction. The electrical predictor/corrector has already advanced
    /// once and is deliberately not repeated.
    pub(crate) fn apply_pressure_correction(
        &self,
        st: &mut BoxDriverState,
        pressure_delta_from_old_pa: f64,
        dt: f64,
    ) {
        let a_eff = self.mms_kg
            + self.gamma * dt * self.rms_kg_per_s
            + self.beta * dt * dt * self.kms_n_per_m;
        let acceleration_delta = -self.sd_m2 * pressure_delta_from_old_pa / a_eff;
        st.x_m += self.beta * dt * dt * acceleration_delta;
        st.v_m_per_s += self.gamma * dt * acceleration_delta;
        st.a_m_per_s2 += acceleration_delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::physics::acoustics::electroacoustics::time_domain::{
        LumpedTransducerConfig, simulate_lumped_transducer,
    };
    use std::f64::consts::TAU;

    #[test]
    fn free_air_matches_reference_transducer() {
        // Identical linear parameters as a sonium-fem LumpedTransducerConfig run;
        // our lockstep replicates time_domain.rs's algorithm, so displacement
        // trajectories must agree to 1e-12 over 100 steps.
        let dt = 1e-5;
        let model = BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 1.0,
            kms_n_per_m: 4000.0,
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
            compliance_force_fn: None,
            compliance_potential_fn: None,
            sd_m2: 0.01,
            box_volume_m3: None,
            medium_density_kg_per_m3: 1.2,
            medium_sound_speed_m_per_s: 343.0,
            beta: 0.25,
            gamma: 0.5,
        };
        let reference = simulate_lumped_transducer(&LumpedTransducerConfig {
            mms: 0.02,
            rms: 1.0,
            kms: 4000.0,
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
            compliance_force_fn: None,
            voltage_fn: Box::new(|t| 2.83 * (TAU * 100.0 * t).sin()),
            beta: 0.25,
            gamma: 0.5,
            dt,
            num_steps: 100,
        });
        let mut st = model.new_state();
        // displacement[0] is the at-rest initial state; compare per-step states.
        for (k, &x_ref) in reference.displacement[1..].iter().enumerate() {
            let t = (k + 1) as f64 * dt;
            model.step(&mut st, 2.83 * (TAU * 100.0 * t).sin(), 0.0, dt);
            assert!(
                (st.x_m - x_ref).abs() < 1e-12,
                "step {k}: {} vs {x_ref}",
                st.x_m
            );
        }
    }

    /// Impulse ringdown helper: zero initial state, 10 steps of 1 V drive,
    /// then free decay; frequency from upward zero crossings of `x_m`.
    fn ringdown_frequency(model: &BoxDriverModel, dt: f64, num_steps: usize) -> f64 {
        let mut st = model.new_state();
        let mut xs = Vec::with_capacity(num_steps);
        for k in 0..num_steps {
            let voltage = if k < 10 { 1.0 } else { 0.0 };
            model.step(&mut st, voltage, 0.0, dt);
            xs.push(st.x_m); // xs[k] is the state at t = (k + 1)·dt
        }

        // Upward zero crossings after the drive pulse, linearly interpolated.
        let mut crossings = Vec::new();
        for i in 10..xs.len() - 1 {
            if xs[i] < 0.0 && xs[i + 1] >= 0.0 {
                let frac = -xs[i] / (xs[i + 1] - xs[i]);
                crossings.push((i as f64 + 1.0 + frac) * dt);
            }
        }
        assert!(
            crossings.len() >= 3,
            "not enough upward zero crossings: {}",
            crossings.len()
        );
        let periods: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
        let avg_period = periods.iter().sum::<f64>() / periods.len() as f64;
        1.0 / avg_period
    }

    #[test]
    fn sealed_box_raises_resonance_by_acoustic_stiffness() {
        // Sealed box adds K_box = ρc²Sd²/V to the spring; measure the free
        // resonance of model+box by impulse ringdown (count zero crossings) and
        // compare with f0' = (1/2π)√((Kms + K_box)/Mms) within 2%.
        let sd = 0.01;
        let v_box = 0.005; // K_box = 1.2·343²·1e-4/0.005 ≈ 2823 N/m
        let model = BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 0.5,
            kms_n_per_m: 4000.0,
            // Small BL so back-EMF damping (≈ BL²/Re) does not shift the
            // ringdown frequency away from the mechanical f0'.
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 1.0),
            compliance_force_fn: None,
            compliance_potential_fn: None,
            sd_m2: sd,
            box_volume_m3: Some(v_box),
            medium_density_kg_per_m3: 1.2,
            medium_sound_speed_m_per_s: 343.0,
            beta: 0.25,
            gamma: 0.5,
        };
        let f_measured = ringdown_frequency(&model, 1e-5, 6000);
        let f_exact = ((4000.0 + 1.2 * 343.0_f64.powi(2) * sd * sd / v_box) / 0.02).sqrt() / TAU;
        assert!(
            (f_measured - f_exact).abs() / f_exact < 0.02,
            "f_measured = {f_measured:.2} Hz, f_exact = {f_exact:.2} Hz"
        );
    }

    #[test]
    fn nonlinear_suspension_audit_requires_matching_potential() {
        let model = BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 1.0,
            kms_n_per_m: 4000.0,
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
            compliance_force_fn: Some(Box::new(|x| 4000.0 * x + 1.0e8 * x.powi(3))),
            compliance_potential_fn: None,
            sd_m2: 0.01,
            box_volume_m3: Some(0.005),
            medium_density_kg_per_m3: 1.2,
            medium_sound_speed_m_per_s: 343.0,
            beta: 0.25,
            gamma: 0.5,
        };
        let mut state = model.new_state();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                model.step_pressure_base_audited(&mut state, 1.0, 1.0e-5)
            }))
            .is_err()
        );
    }

    #[test]
    fn legacy_step_accepts_force_only_nonlinear_suspension() {
        let model = BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 1.0,
            kms_n_per_m: 4000.0,
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
            compliance_force_fn: Some(Box::new(|x| 4000.0 * x + 1.0e8 * x.powi(3))),
            compliance_potential_fn: None,
            sd_m2: 0.01,
            box_volume_m3: None,
            medium_density_kg_per_m3: 1.2,
            medium_sound_speed_m_per_s: 343.0,
            beta: 0.25,
            gamma: 0.5,
        };
        let mut state = model.new_state();
        model.step(&mut state, 1.0, 0.0, 1.0e-5);
        assert!(state.x_m.is_finite());
    }

    #[test]
    fn driver_energy_audit_cancels_internal_box_work() {
        let model = BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 1.0,
            kms_n_per_m: 4000.0,
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
            compliance_force_fn: None,
            compliance_potential_fn: None,
            sd_m2: 0.01,
            box_volume_m3: Some(0.005),
            medium_density_kg_per_m3: 1.2,
            medium_sound_speed_m_per_s: 343.0,
            beta: 0.25,
            gamma: 0.5,
        };
        let mut state = model.new_state();
        let audit = model.step_pressure_base_audited(&mut state, 1.0, 1.0e-5);
        let energy = model.finalize_energy_step(audit, &state, 1.0e-5);
        assert!((energy.driver_to_box_work_j + energy.box_on_driver_work_j).abs() < 1.0e-20);
        assert!(energy.copper_loss_j >= 0.0);
        assert!(energy.mechanical_damping_midpoint_j >= 0.0);
    }

    #[test]
    fn cubic_suspension_audit_uses_the_paired_analytic_potential() {
        let stiffness = 4000.0;
        let cubic = 1.0e8;
        let model = BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 0.0,
            kms_n_per_m: stiffness,
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
            compliance_force_fn: None,
            compliance_potential_fn: None,
            sd_m2: 0.01,
            box_volume_m3: None,
            medium_density_kg_per_m3: 1.2,
            medium_sound_speed_m_per_s: 343.0,
            beta: 0.25,
            gamma: 0.5,
        }
        .with_nonlinear_suspension(
            move |x| stiffness * x + cubic * x.powi(3),
            move |x| 0.5 * stiffness * x.powi(2) + 0.25 * cubic * x.powi(4),
        );
        let mut state = model.new_state();
        state.x_m = 1.0e-3;
        state.v_m_per_s = 0.2;
        let x_old = state.x_m;
        let audit = model.step_pressure_base_audited(&mut state, 2.0, 1.0e-4);
        let energy = model.finalize_energy_step(audit, &state, 1.0e-4);
        let potential = |x: f64| 0.5 * stiffness * x.powi(2) + 0.25 * cubic * x.powi(4);
        assert!((energy.suspension_storage_old_j - potential(x_old)).abs() < 1.0e-15);
        assert!((energy.suspension_storage_new_j - potential(state.x_m)).abs() < 1.0e-15);
        let dx = state.x_m - x_old;
        let algorithmic_work = (audit.predictor_spring_force_n
            + stiffness * (state.x_m - audit.circuit.predictor_x_m))
            * dx;
        assert!(
            (energy.suspension_numerical_j
                - (algorithmic_work - (potential(state.x_m) - potential(x_old))))
            .abs()
                < 1.0e-15
        );
    }

    #[test]
    fn average_acceleration_newmark_preserves_undamped_linear_energy() {
        let model = BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 0.0,
            kms_n_per_m: 4000.0,
            voice_coil: VoiceCoilParams::linear(6.0, 0.0, 0.0),
            compliance_force_fn: None,
            compliance_potential_fn: None,
            sd_m2: 0.01,
            box_volume_m3: None,
            medium_density_kg_per_m3: 1.2,
            medium_sound_speed_m_per_s: 343.0,
            beta: 0.25,
            gamma: 0.5,
        };
        let mut state = model.new_state();
        state.x_m = 1.0e-3;
        state.a_m_per_s2 = -model.kms_n_per_m * state.x_m / model.mms_kg;
        let energy = |state: &BoxDriverState| {
            0.5 * model.mms_kg * state.v_m_per_s.powi(2)
                + 0.5 * model.kms_n_per_m * state.x_m.powi(2)
        };
        let initial = energy(&state);
        let period = TAU * (model.mms_kg / model.kms_n_per_m).sqrt();
        let dt = period / 100.0;
        for _ in 0..10_000 {
            model.step(&mut state, 0.0, 0.0, dt);
        }
        assert!((energy(&state) - initial).abs() / initial < 1.0e-11);
    }
}
