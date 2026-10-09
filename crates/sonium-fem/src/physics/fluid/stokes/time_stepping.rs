//! Time integration for unsteady Navier-Stokes
//!
//! Provides BDF (Backward Differentiation Formula) time steppers for
//! the transient incompressible Navier-Stokes equations:
//!
//! ```text
//! ρ ∂u/∂t + ρ(u·∇)u - μΔu + ∇p = f
//!                          div(u) = 0
//! ```
//!
//! # BDF1 (Implicit Euler)
//!
//! ```text
//! ρ(u^{n+1} - u^n)/Δt + ρ(u^{n+1}·∇)u^{n+1} - μΔu^{n+1} + ∇p^{n+1} = f^{n+1}
//! ```
//!
//! # BDF2 (Second-order backward differentiation)
//!
//! ```text
//! ρ(3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + ρ(u*·∇)u^{n+1} - μΔu^{n+1} + ∇p^{n+1} = f^{n+1}
//! ```
//!
//! where `u* = 2u^n - u^{n-1}` is the extrapolated advection velocity.
//!
//! At each time step, a linearised (Oseen) system is solved using the
//! existing Picard machinery.

/// BDF time-stepping configuration.
#[derive(Debug, Clone)]
pub struct BdfConfig {
    /// Time step Δt \[s\].
    pub dt: f64,
    /// BDF order (1 or 2).
    pub order: usize,
    /// Number of time steps.
    pub num_steps: usize,
    /// Output interval (write VTK every N steps, 0 = never).
    pub output_interval: usize,
}

impl BdfConfig {
    /// BDF1 (implicit Euler).
    pub fn bdf1(dt: f64, num_steps: usize) -> Self {
        Self {
            dt,
            order: 1,
            num_steps,
            output_interval: 0,
        }
    }

    /// BDF2 (second-order, recommended for accuracy).
    pub fn bdf2(dt: f64, num_steps: usize) -> Self {
        Self {
            dt,
            order: 2,
            num_steps,
            output_interval: 0,
        }
    }

    /// Set VTK output interval.
    pub fn with_output(mut self, interval: usize) -> Self {
        self.output_interval = interval;
        self
    }
}

/// State for the BDF time stepper.
#[derive(Debug, Clone)]
pub struct BdfState {
    /// Current velocity DOFs (time level n).
    pub velocity: Vec<f64>,
    /// Previous velocity DOFs (time level n-1, for BDF2).
    pub velocity_prev: Vec<f64>,
    /// Current pressure DOFs.
    pub pressure: Vec<f64>,
    /// Current simulation time.
    pub time: f64,
    /// Current step number.
    pub step: usize,
}

impl BdfState {
    /// Create initial state from rest (zero velocity, zero pressure).
    pub fn from_rest(n_velocity_dofs: usize, n_pressure_dofs: usize) -> Self {
        Self {
            velocity: vec![0.0; n_velocity_dofs],
            velocity_prev: vec![0.0; n_velocity_dofs],
            pressure: vec![0.0; n_pressure_dofs],
            time: 0.0,
            step: 0,
        }
    }

    /// Create initial state from a steady solution.
    pub fn from_steady(velocity: Vec<f64>, pressure: Vec<f64>) -> Self {
        let vel_prev = velocity.clone();
        Self {
            velocity,
            velocity_prev: vel_prev,
            pressure,
            time: 0.0,
            step: 0,
        }
    }
}

/// BDF time derivative coefficients.
///
/// The time derivative is approximated as:
/// `∂u/∂t ≈ (a0 u^{n+1} + a1 u^n + a2 u^{n-1}) / Δt`
///
/// This adds `ρ a0 / Δt × M` to the system matrix and
/// `ρ / Δt × M × (-a1 u^n - a2 u^{n-1})` to the RHS.
pub struct BdfCoefficients {
    /// Coefficient for u^{n+1} (appears in the system matrix).
    pub a0: f64,
    /// Coefficient for u^n (appears in the RHS).
    pub a1: f64,
    /// Coefficient for u^{n-1} (appears in the RHS, zero for BDF1).
    pub a2: f64,
}

impl BdfCoefficients {
    /// BDF1 coefficients: (u^{n+1} - u^n) / Δt
    pub fn bdf1() -> Self {
        Self {
            a0: 1.0,
            a1: -1.0,
            a2: 0.0,
        }
    }

    /// BDF2 coefficients: (3u^{n+1} - 4u^n + u^{n-1}) / (2Δt)
    pub fn bdf2() -> Self {
        Self {
            a0: 1.5,
            a1: -2.0,
            a2: 0.5,
        }
    }

    /// Get coefficients for the given BDF order.
    pub fn for_order(order: usize) -> Self {
        match order {
            2 => Self::bdf2(),
            _ => Self::bdf1(),
        }
    }
}

/// Extrapolate the advection velocity for the BDF scheme.
///
/// - BDF1: `u* = u^n` (first-order extrapolation)
/// - BDF2: `u* = 2u^n - u^{n-1}` (second-order extrapolation)
pub fn extrapolate_velocity(state: &BdfState, order: usize) -> Vec<f64> {
    match order {
        2 if state.step >= 1 => {
            // u* = 2 u^n - u^{n-1}
            state
                .velocity
                .iter()
                .zip(&state.velocity_prev)
                .map(|(&un, &unm1)| 2.0 * un - unm1)
                .collect()
        }
        _ => {
            // u* = u^n
            state.velocity.clone()
        }
    }
}

/// Compute the BDF RHS contribution from previous time levels.
///
/// Returns `ρ/Δt × (−a1 u^n − a2 u^{n-1})` which is added to the
/// momentum RHS after multiplication by the mass matrix.
pub fn bdf_rhs_velocity(state: &BdfState, coeffs: &BdfCoefficients, rho: f64, dt: f64) -> Vec<f64> {
    let factor = rho / dt;
    state
        .velocity
        .iter()
        .zip(&state.velocity_prev)
        .map(|(&un, &unm1)| factor * (-coeffs.a1 * un - coeffs.a2 * unm1))
        .collect()
}

/// Monitor for tracking time-dependent quantities.
#[derive(Debug, Default)]
pub struct TimeMonitor {
    /// Time stamps.
    pub times: Vec<f64>,
    /// Drag coefficient history.
    pub drag: Vec<f64>,
    /// Lift coefficient history.
    pub lift: Vec<f64>,
    /// Kinetic energy: E_k = 0.5 ∫ |u|² dΩ
    pub kinetic_energy: Vec<f64>,
}

impl TimeMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, time: f64, drag: f64, lift: f64, energy: f64) {
        self.times.push(time);
        self.drag.push(drag);
        self.lift.push(lift);
        self.kinetic_energy.push(energy);
    }

    /// Estimate Strouhal number from lift coefficient oscillation.
    ///
    /// St = f L / U where f is the dominant frequency.
    /// Uses zero-crossing counting for frequency estimation.
    pub fn strouhal_number(&self, char_length: f64, char_velocity: f64) -> Option<f64> {
        if self.lift.len() < 10 {
            return None;
        }

        // Count zero crossings in the last half of the signal
        let start = self.lift.len() / 2;
        let mut crossings = 0;
        for i in start + 1..self.lift.len() {
            if self.lift[i] * self.lift[i - 1] < 0.0 {
                crossings += 1;
            }
        }

        if crossings < 2 {
            return None;
        }

        let duration = self.times.last()? - self.times[start];
        let frequency = crossings as f64 / (2.0 * duration); // half-periods → full periods
        Some(frequency * char_length / char_velocity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bdf_coefficients() {
        let c1 = BdfCoefficients::bdf1();
        assert_eq!(c1.a0, 1.0);
        assert_eq!(c1.a1, -1.0);
        assert_eq!(c1.a2, 0.0);

        let c2 = BdfCoefficients::bdf2();
        assert_eq!(c2.a0, 1.5);
        assert_eq!(c2.a1, -2.0);
        assert_eq!(c2.a2, 0.5);
    }

    #[test]
    fn test_extrapolate_velocity_bdf1() {
        let state = BdfState {
            velocity: vec![1.0, 2.0],
            velocity_prev: vec![0.5, 1.5],
            pressure: vec![],
            time: 0.0,
            step: 0,
        };
        let u_star = extrapolate_velocity(&state, 1);
        assert_eq!(u_star, vec![1.0, 2.0]); // BDF1: u* = u^n
    }

    #[test]
    fn test_extrapolate_velocity_bdf2() {
        let state = BdfState {
            velocity: vec![1.0, 2.0],
            velocity_prev: vec![0.5, 1.5],
            pressure: vec![],
            time: 0.0,
            step: 1, // need at least 1 previous step for BDF2
        };
        let u_star = extrapolate_velocity(&state, 2);
        // u* = 2*u^n - u^{n-1} = 2*[1,2] - [0.5,1.5] = [1.5, 2.5]
        assert_eq!(u_star, vec![1.5, 2.5]);
    }

    #[test]
    fn test_bdf_rhs() {
        let state = BdfState {
            velocity: vec![1.0, 2.0],
            velocity_prev: vec![0.5, 1.5],
            pressure: vec![],
            time: 0.0,
            step: 1,
        };
        let coeffs = BdfCoefficients::bdf2();
        let rhs = bdf_rhs_velocity(&state, &coeffs, 1.0, 0.1);
        // rhs = (1/0.1) * (-(-2)*[1,2] - 0.5*[0.5,1.5])
        //     = 10 * (2*[1,2] - 0.5*[0.5,1.5])
        //     = 10 * ([2,4] - [0.25,0.75])
        //     = 10 * [1.75, 3.25]
        //     = [17.5, 32.5]
        assert!((rhs[0] - 17.5).abs() < 1e-10);
        assert!((rhs[1] - 32.5).abs() < 1e-10);
    }

    #[test]
    fn test_strouhal_detection() {
        let mut monitor = TimeMonitor::new();
        let dt = 0.01;
        let freq = 5.0; // 5 Hz
        for i in 0..200 {
            let t = i as f64 * dt;
            let cl = (2.0 * std::f64::consts::PI * freq * t).sin();
            monitor.record(t, 0.0, cl, 0.0);
        }
        let st = monitor.strouhal_number(1.0, 1.0);
        assert!(st.is_some());
        let st = st.unwrap();
        // Should detect frequency ≈ 5 Hz → St ≈ 5
        assert!((st - freq).abs() < 1.0, "St = {st}, expected ~{freq}");
    }
}
