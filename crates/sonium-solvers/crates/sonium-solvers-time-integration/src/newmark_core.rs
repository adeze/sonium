//! Shared Newmark predictor–corrector algebra and energy accounting.
//!
//! The structural nonlinear-transient FEM path and the speaker box-driver
//! lockstep previously each hand-rolled the same average-acceleration update.
//! This module is the single reusable owner of that mechanics; application
//! crates keep their study schemas, loading, boundary conditions, and
//! engineering acceptance policy.

// Rust guideline compliant 2026-02-21

use sonium_solvers_core::traits::SolverError;

/// Checked Newmark coefficients for one fixed step size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewmarkCoefficients {
    beta: f64,
    gamma: f64,
    dt: f64,
    acceleration: f64,
    velocity: f64,
}

impl NewmarkCoefficients {
    /// Build coefficients, rejecting non-finite inputs and overflowing
    /// derived constants instead of silently integrating with them.
    pub fn new(beta: f64, gamma: f64, dt: f64) -> Result<Self, SolverError> {
        for (name, value) in [("beta", beta), ("gamma", gamma), ("dt", dt)] {
            if !value.is_finite() {
                return Err(SolverError::OperationFailed {
                    operation: "Newmark coefficients",
                    reason: format!("{name} must be finite"),
                });
            }
        }
        if beta <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Newmark coefficients",
                reason: "beta must be positive".to_string(),
            });
        }
        if gamma < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Newmark coefficients",
                reason: "gamma must be non-negative".to_string(),
            });
        }
        if dt <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Newmark coefficients",
                reason: "dt must be positive".to_string(),
            });
        }
        let acceleration = 1.0 / (beta * dt * dt);
        let velocity = gamma / (beta * dt);
        if !acceleration.is_finite() || !velocity.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "Newmark coefficients",
                reason: "derived coefficients overflow for this beta/gamma/dt".to_string(),
            });
        }
        Ok(Self {
            beta,
            gamma,
            dt,
            acceleration,
            velocity,
        })
    }

    /// Average-acceleration coefficients (`beta = 1/4`, `gamma = 1/2`).
    pub fn average_acceleration(dt: f64) -> Result<Self, SolverError> {
        Self::new(0.25, 0.5, dt)
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// `1 / (beta dt^2)`: displacement increment to acceleration.
    pub fn acceleration_coefficient(&self) -> f64 {
        self.acceleration
    }

    /// `gamma / (beta dt)`: displacement increment to velocity.
    pub fn velocity_coefficient(&self) -> f64 {
        self.velocity
    }

    /// Effective mass multiplier `1/(beta dt^2) + c gamma/(beta dt)` for
    /// mass-proportional damping `c` (units 1/s).
    pub fn effective_mass_coefficient(&self, damping_per_mass: f64) -> Result<f64, SolverError> {
        if !damping_per_mass.is_finite() || damping_per_mass < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Newmark effective mass",
                reason: "damping per unit mass must be finite and non-negative".to_string(),
            });
        }
        let value = self.acceleration + damping_per_mass * self.velocity;
        if !value.is_finite() || value <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Newmark effective mass",
                reason: "effective coefficient is non-finite or non-positive".to_string(),
            });
        }
        Ok(value)
    }

    /// Acceleration-form effective scalar `m + gamma dt c + beta dt^2 k`
    /// for single-DOF locksteps that solve for the new acceleration.
    pub fn acceleration_effective(&self, mass: f64, damping: f64, stiffness: f64) -> f64 {
        acceleration_effective_scalar(mass, damping, stiffness, self.beta, self.gamma, self.dt)
    }

    pub fn predict_displacement(&self, displacement: f64, velocity: f64, acceleration: f64) -> f64 {
        predict_displacement_scalar(self.beta, self.dt, displacement, velocity, acceleration)
    }

    pub fn predict_velocity(&self, velocity: f64, acceleration: f64) -> f64 {
        predict_velocity_scalar(self.gamma, self.dt, velocity, acceleration)
    }

    /// Corrector from a solved displacement: returns `(u, v, a)`.
    pub fn correct_from_displacement(
        &self,
        predicted_displacement: f64,
        predicted_velocity: f64,
        solved_displacement: f64,
    ) -> (f64, f64, f64) {
        let delta = solved_displacement - predicted_displacement;
        (
            solved_displacement,
            predicted_velocity + self.velocity * delta,
            self.acceleration * delta,
        )
    }

    /// Corrector from a solved acceleration: returns `(u, v)`.
    pub fn correct_from_acceleration(
        &self,
        predicted_displacement: f64,
        predicted_velocity: f64,
        solved_acceleration: f64,
    ) -> (f64, f64) {
        (
            correct_displacement_from_acceleration(
                self.beta,
                self.dt,
                predicted_displacement,
                solved_acceleration,
            ),
            correct_velocity_from_acceleration(
                self.gamma,
                self.dt,
                predicted_velocity,
                solved_acceleration,
            ),
        )
    }
}

/// `u + dt v + dt^2 (1/2 - beta) a`.
pub fn predict_displacement_scalar(
    beta: f64,
    dt: f64,
    displacement: f64,
    velocity: f64,
    acceleration: f64,
) -> f64 {
    displacement + dt * velocity + dt * dt * (0.5 - beta) * acceleration
}

/// `v + dt (1 - gamma) a`.
pub fn predict_velocity_scalar(gamma: f64, dt: f64, velocity: f64, acceleration: f64) -> f64 {
    velocity + dt * (1.0 - gamma) * acceleration
}

/// `u_pred + beta dt^2 a_new`.
pub fn correct_displacement_from_acceleration(
    beta: f64,
    dt: f64,
    predicted_displacement: f64,
    solved_acceleration: f64,
) -> f64 {
    predicted_displacement + beta * dt * dt * solved_acceleration
}

/// `v_pred + gamma dt a_new`.
pub fn correct_velocity_from_acceleration(
    gamma: f64,
    dt: f64,
    predicted_velocity: f64,
    solved_acceleration: f64,
) -> f64 {
    predicted_velocity + gamma * dt * solved_acceleration
}

/// `m + gamma dt c + beta dt^2 k`.
pub fn acceleration_effective_scalar(
    mass: f64,
    damping: f64,
    stiffness: f64,
    beta: f64,
    gamma: f64,
    dt: f64,
) -> f64 {
    mass + gamma * dt * damping + beta * dt * dt * stiffness
}

fn check_lengths(expected: usize, got: usize, what: &'static str) -> Result<(), SolverError> {
    if expected != got {
        return Err(SolverError::DimensionMismatch { expected, got });
    }
    let _ = what;
    Ok(())
}

/// Predictor for displacement histories: `out[i] = predict(u[i], v[i], a[i])`.
pub fn predict_displacement_slice(
    coefficients: &NewmarkCoefficients,
    displacement: &[f64],
    velocity: &[f64],
    acceleration: &[f64],
    out: &mut [f64],
) -> Result<(), SolverError> {
    check_lengths(displacement.len(), velocity.len(), "velocity")?;
    check_lengths(displacement.len(), acceleration.len(), "acceleration")?;
    check_lengths(displacement.len(), out.len(), "output")?;
    for (((output, &u), &v), &a) in out
        .iter_mut()
        .zip(displacement)
        .zip(velocity)
        .zip(acceleration)
    {
        *output = coefficients.predict_displacement(u, v, a);
    }
    Ok(())
}

/// Predictor for velocity histories.
pub fn predict_velocity_slice(
    coefficients: &NewmarkCoefficients,
    velocity: &[f64],
    acceleration: &[f64],
    out: &mut [f64],
) -> Result<(), SolverError> {
    check_lengths(velocity.len(), acceleration.len(), "acceleration")?;
    check_lengths(velocity.len(), out.len(), "output")?;
    for ((output, &v), &a) in out.iter_mut().zip(velocity).zip(acceleration) {
        *output = coefficients.predict_velocity(v, a);
    }
    Ok(())
}

/// Corrector from a solved displacement history; fills `velocity` and
/// `acceleration` in place and returns the displacement increment norm.
pub fn correct_from_displacement_slice(
    coefficients: &NewmarkCoefficients,
    predicted_displacement: &[f64],
    predicted_velocity: &[f64],
    solved_displacement: &[f64],
    velocity: &mut [f64],
    acceleration: &mut [f64],
) -> Result<f64, SolverError> {
    check_lengths(
        solved_displacement.len(),
        predicted_displacement.len(),
        "predicted displacement",
    )?;
    check_lengths(
        solved_displacement.len(),
        predicted_velocity.len(),
        "predicted velocity",
    )?;
    check_lengths(solved_displacement.len(), velocity.len(), "velocity")?;
    check_lengths(
        solved_displacement.len(),
        acceleration.len(),
        "acceleration",
    )?;
    let mut increment_norm_squared = 0.0;
    for ((((up, vp), u_new), v), a) in predicted_displacement
        .iter()
        .zip(predicted_velocity)
        .zip(solved_displacement)
        .zip(velocity.iter_mut())
        .zip(acceleration.iter_mut())
    {
        let (next_u, next_v, next_a) = coefficients.correct_from_displacement(*up, *vp, *u_new);
        let _ = next_u;
        *v = next_v;
        *a = next_a;
        let delta = *u_new - *up;
        increment_norm_squared += delta * delta;
    }
    Ok(increment_norm_squared.sqrt())
}

/// Trapezoidal external-work increment `1/2 (f_prev + f) . (u - u_prev)`.
pub fn trapezoidal_force_work_increment(
    previous_force: &[f64],
    force: &[f64],
    previous_displacement: &[f64],
    displacement: &[f64],
) -> Result<f64, SolverError> {
    check_lengths(previous_force.len(), force.len(), "force")?;
    check_lengths(
        previous_force.len(),
        previous_displacement.len(),
        "displacement",
    )?;
    check_lengths(previous_force.len(), displacement.len(), "displacement")?;
    let mut work = 0.0;
    for (((&fp, &f), &up), &u) in previous_force
        .iter()
        .zip(force)
        .zip(previous_displacement)
        .zip(displacement)
    {
        work += 0.5 * (fp + f) * (u - up);
    }
    Ok(work)
}

/// Trapezoidal dissipation increment `1/2 dt (p_prev + p)`.
pub fn trapezoidal_power_increment(dt: f64, previous_power: f64, power: f64) -> f64 {
    0.5 * dt * (previous_power + power)
}

/// Damping power `c v^T (M v)` for mass-proportional damping `c`.
pub fn mass_proportional_damping_power(
    damping_per_mass: f64,
    velocity: &[f64],
    mass_times_velocity: &[f64],
) -> Result<f64, SolverError> {
    check_lengths(velocity.len(), mass_times_velocity.len(), "mass-velocity")?;
    let mut power = 0.0;
    for (&v, &mv) in velocity.iter().zip(mass_times_velocity) {
        power += v * mv;
    }
    Ok(damping_per_mass * power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_match_hand_computation() {
        let coefficients = NewmarkCoefficients::new(0.25, 0.5, 0.1).expect("valid");
        assert!(
            (coefficients.acceleration_coefficient() - 400.0).abs() < 1.0e-9,
            "accel = {}",
            coefficients.acceleration_coefficient()
        );
        assert!(
            (coefficients.velocity_coefficient() - 20.0).abs() < 1.0e-12,
            "vel = {}",
            coefficients.velocity_coefficient()
        );
        assert!(
            (coefficients.predict_displacement(1.0, 2.0, 3.0) - 1.2075).abs() < 1.0e-14,
            "upred = {}",
            coefficients.predict_displacement(1.0, 2.0, 3.0)
        );
        assert!(
            (coefficients.predict_velocity(2.0, 3.0) - 2.15).abs() < 1.0e-14,
            "vpred = {}",
            coefficients.predict_velocity(2.0, 3.0)
        );
        let (u, v, a) = coefficients.correct_from_displacement(1.2075, 2.15, 1.3);
        assert_eq!(u, 1.3);
        assert!((v - 4.0).abs() < 1.0e-12, "v = {v}");
        assert!((a - 37.0).abs() < 1.0e-9, "a = {a}");
        let (u2, v2) = coefficients.correct_from_acceleration(1.2075, 2.15, 37.0);
        assert!((u2 - 1.3).abs() < 1.0e-12, "u2 = {u2}");
        assert!((v2 - 4.0).abs() < 1.0e-12, "v2 = {v2}");
    }

    #[test]
    fn effective_coefficients_match_closed_forms() {
        let coefficients = NewmarkCoefficients::average_acceleration(1.0e-3).expect("valid");
        let effective = coefficients.effective_mass_coefficient(2.0).expect("valid");
        assert!(
            (effective - (4.0e6 + 2.0 * 2.0e3)).abs() < 1.0e-6,
            "effective = {effective}"
        );
        let lockstep = coefficients.acceleration_effective(1.0, 0.5, 2.0);
        assert!(
            (lockstep - (1.0 + 0.5 * 1.0e-3 * 0.5 + 0.25 * 1.0e-6 * 2.0)).abs() < 1.0e-15,
            "lockstep = {lockstep}"
        );
    }

    #[test]
    fn invalid_coefficients_are_rejected() {
        assert!(NewmarkCoefficients::new(0.0, 0.5, 0.1).is_err());
        assert!(NewmarkCoefficients::new(0.25, -0.1, 0.1).is_err());
        assert!(NewmarkCoefficients::new(0.25, 0.5, 0.0).is_err());
        assert!(NewmarkCoefficients::new(f64::NAN, 0.5, 0.1).is_err());
        assert!(NewmarkCoefficients::new(0.25, 0.5, 1.0e-200).is_err());
    }

    #[test]
    fn slice_helpers_agree_with_scalar_helpers() {
        let coefficients = NewmarkCoefficients::new(0.3, 0.6, 0.02).expect("valid");
        let u = vec![0.5, -1.25, 2.0];
        let v = vec![0.1, 0.2, -0.3];
        let a = vec![1.0, -1.0, 0.5];
        let mut up = vec![0.0; 3];
        let mut vp = vec![0.0; 3];
        predict_displacement_slice(&coefficients, &u, &v, &a, &mut up).expect("valid");
        predict_velocity_slice(&coefficients, &v, &a, &mut vp).expect("valid");
        for i in 0..3 {
            assert_eq!(up[i], coefficients.predict_displacement(u[i], v[i], a[i]));
            assert_eq!(vp[i], coefficients.predict_velocity(v[i], a[i]));
        }
        let solved = vec![0.55, -1.2, 2.1];
        let mut next_v = vec![0.0; 3];
        let mut next_a = vec![0.0; 3];
        let norm = correct_from_displacement_slice(
            &coefficients,
            &up,
            &vp,
            &solved,
            &mut next_v,
            &mut next_a,
        )
        .expect("valid");
        assert!(norm > 0.0);
        for i in 0..3 {
            let (_, v_check, a_check) =
                coefficients.correct_from_displacement(up[i], vp[i], solved[i]);
            assert_eq!(next_v[i], v_check);
            assert_eq!(next_a[i], a_check);
        }
        assert!(predict_velocity_slice(&coefficients, &v, &a[..2], &mut vp).is_err());
    }

    #[test]
    fn energy_helpers_are_exact_on_linear_ramps() {
        let work =
            trapezoidal_force_work_increment(&[0.0], &[10.0], &[0.0], &[0.5]).expect("valid");
        assert_eq!(work, 2.5);
        assert_eq!(trapezoidal_power_increment(0.1, 1.0, 3.0), 0.2);
        let power = mass_proportional_damping_power(2.0, &[1.0, 2.0], &[3.0, 4.0]).expect("valid");
        assert_eq!(power, 22.0);
        assert!(trapezoidal_force_work_increment(&[0.0], &[1.0, 2.0], &[0.0], &[0.0]).is_err());
    }
}
