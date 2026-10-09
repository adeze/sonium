//! Explicit central-difference time integration for second-order ODEs
//!
//! Solves `M u'' = F(u, v, t)` with the explicit central-difference scheme,
//! i.e. Newmark with `beta = 0, gamma = 1/2`. This is the standard explicit
//! integrator for transient wave propagation (acoustics, elastodynamics)
//! where lumped-mass explicit stepping beats an implicit solve.
//!
//! ## Algorithm
//!
//! One force evaluation per step, with the half-step velocity used for the
//! force/damping evaluation (velocity-Verlet form):
//!
//! ```text
//! u_{n+1}   = u_n + dt * v_n + (dt^2 / 2) * a_n
//! v_{n+1/2} = v_n + (dt / 2) * a_n
//! a_{n+1}   = M^{-1} F(u_{n+1}, v_{n+1/2}, t_{n+1})
//! v_{n+1}   = v_{n+1/2} + (dt / 2) * a_{n+1}
//! ```
//!
//! The scheme is second-order accurate. Evaluating damping forces at the
//! half-step velocity keeps the update fully explicit while preserving
//! second-order accuracy for `C v` terms.
//!
//! ## Conditional stability (CFL contract)
//!
//! Unlike the implicit schemes in [`super::newmark`], this integrator is
//! only **conditionally stable**: the time step must satisfy
//!
//! ```text
//! dt <= dt_crit = 2 / omega_max                                    (undamped)
//! dt <= dt_crit = 2 (sqrt(xi^2 + 1) - xi) / omega_max              (damped)
//! ```
//!
//! where `omega_max` is the highest natural frequency of the discretised
//! system (`max sqrt(eig(M^{-1} K))`) and `xi` the modal damping ratio of
//! that highest mode (for Rayleigh damping `C = alpha M + beta K`,
//! `xi = (alpha / omega_max + beta * omega_max) / 2`). See
//! [`critical_time_step`] and [`critical_time_step_damped`]. `omega_max`
//! itself can come from an element-level estimate, the Gershgorin bound
//! [`gershgorin_omega_max`], or the power-iteration estimate
//! [`power_iteration_omega_max`]. Above `dt_crit` the highest-frequency mode
//! grows geometrically and the solution blows up.
//!
//! ## Mass solve and force closures
//!
//! The caller supplies two closures, mirroring the closure style of
//! [`super::newmark`]:
//!
//! - `mass_solve(r) -> M^{-1} r`: trivial for a diagonal (lumped) mass
//!   matrix — divide componentwise. Any cheap solver works for non-diagonal
//!   M, but then the scheme is no longer truly "explicit-cheap".
//! - `force_fn(u, v, t) -> F(u, v, t)`: the full residual force
//!   `f_ext(t) - f_int(u, v, t)`. Internal forces (`-K u`), damping
//!   (`-C v`, including Rayleigh/proportional damping), and nonlinear terms
//!   all live inside this closure — no special-casing is needed. The
//!   velocity passed in is the half-step velocity `v_{n+1/2}` (and `v_0` for
//!   the initial acceleration).
//!
//! ## Scalar type
//!
//! Generic over [`ComplexField`](crate::ComplexField): works with f32, f64,
//! Complex32, and Complex64. The `dt_crit` helper functions are f64-only
//! (stability estimates are scalar quantities).

use num_traits::{Float, FromPrimitive};

use sonium_solvers_core::traits::{ComplexField, LinearOperator, SolverError};
use sonium_solvers_sparse::CsrMatrix;
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::newmark::{NewmarkInitial, NewmarkResult, validate_initial_state, validate_result};

/// Explicit central-difference integration parameters.
#[derive(Debug, Clone)]
pub struct ExplicitConfig {
    /// Time step size. Must satisfy the conditional-stability bound
    /// `dt <= 2/omega_max` (undamped); see the module docs.
    pub dt: f64,
    /// Number of time steps.
    pub num_steps: usize,
    /// Store full history (all time steps) or only final state.
    pub store_history: bool,
}

impl ExplicitConfig {
    /// Central-difference parameters for `num_steps` steps of size `dt`.
    pub fn new(dt: f64, num_steps: usize) -> Self {
        Self {
            dt,
            num_steps,
            store_history: false,
        }
    }

    /// Store full time history.
    pub fn with_history(mut self) -> Self {
        self.store_history = true;
        self
    }

    /// Validate the public explicit-integration configuration before any
    /// callback or allocation is entered.
    pub fn validate(&self) -> Result<(), SolverError> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "explicit integration configuration",
                reason: "dt must be finite and positive".to_string(),
            });
        }
        if self.num_steps.checked_add(1).is_none() || !(self.num_steps as f64 * self.dt).is_finite()
        {
            return Err(SolverError::OperationFailed {
                operation: "explicit integration configuration",
                reason: "num_steps and final time overflow".to_string(),
            });
        }
        Ok(())
    }
}

/// Fallible critical time step of the undamped central-difference scheme:
/// `dt_crit = 2 / omega_max`, where `omega_max` is the highest natural
/// frequency of the discretised system (`max sqrt(eig(M^{-1} K))`).
///
/// The integration is unstable (geometric growth of the highest mode) for
/// `dt > dt_crit`. A safety factor of 0.8–0.95 is customary in practice.
pub fn try_critical_time_step(omega_max: f64) -> Result<f64, SolverError> {
    if !omega_max.is_finite() || omega_max <= 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "critical_time_step",
            reason: format!("omega_max must be finite and positive, got {omega_max}"),
        });
    }
    Ok(2.0 / omega_max)
}

/// Critical time step of the undamped central-difference scheme.
///
/// Panicking wrapper around [`try_critical_time_step`].
pub fn critical_time_step(omega_max: f64) -> f64 {
    try_critical_time_step(omega_max).expect("omega_max must be positive")
}

/// Fallible critical time step with (proportional/Rayleigh) damping:
/// `dt_crit = 2 (sqrt(xi^2 + 1) - xi) / omega_max`, where `xi` is the modal
/// damping ratio of the highest-frequency mode. `xi = 0` recovers
/// [`critical_time_step`].
pub fn try_critical_time_step_damped(omega_max: f64, xi: f64) -> Result<f64, SolverError> {
    if !omega_max.is_finite() || omega_max <= 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "critical_time_step_damped",
            reason: format!("omega_max must be finite and positive, got {omega_max}"),
        });
    }
    if !xi.is_finite() || xi < 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "critical_time_step_damped",
            reason: format!("damping ratio must be finite and non-negative, got {xi}"),
        });
    }
    Ok(2.0 * ((xi * xi + 1.0).sqrt() - xi) / omega_max)
}

/// Critical time step with (proportional/Rayleigh) damping.
///
/// Panicking wrapper around [`try_critical_time_step_damped`].
pub fn critical_time_step_damped(omega_max: f64, xi: f64) -> f64 {
    try_critical_time_step_damped(omega_max, xi).expect("omega_max/xi must be valid")
}

/// Gershgorin upper bound on `omega_max` for the generalized eigenproblem
/// `K phi = omega^2 M phi` with a diagonal (lumped) mass matrix:
///
/// ```text
/// omega_max^2 <= max_i ( sum_j |K_ij| ) / m_i
/// ```
///
/// This is a bound, not an estimate: the true `omega_max` is never larger,
/// so [`critical_time_step`] with this value is always conservative
/// (for the undamped problem with symmetric positive semi-definite K).
pub fn try_gershgorin_omega_max(
    stiffness: &CsrMatrix<f64>,
    lumped_mass: &[f64],
) -> Result<f64, SolverError> {
    if stiffness.num_rows != stiffness.num_cols {
        return Err(SolverError::DimensionMismatch {
            expected: stiffness.num_rows,
            got: stiffness.num_cols,
        });
    }
    if stiffness.num_rows != lumped_mass.len() {
        return Err(SolverError::DimensionMismatch {
            expected: stiffness.num_rows,
            got: lumped_mass.len(),
        });
    }
    let mut lambda_max = 0.0_f64;
    for (i, &m_i) in lumped_mass.iter().enumerate() {
        if !m_i.is_finite() || m_i <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "gershgorin_omega_max",
                reason: format!("lumped mass entry {i} must be finite and positive, got {m_i}"),
            });
        }
        let row_abs_sum: f64 = stiffness.row_entries(i).map(|(_, v)| v.abs()).sum();
        lambda_max = lambda_max.max(row_abs_sum / m_i);
    }
    Ok(lambda_max.sqrt())
}

/// Gershgorin upper bound on `omega_max`.
///
/// Panicking wrapper around [`try_gershgorin_omega_max`].
pub fn gershgorin_omega_max(stiffness: &CsrMatrix<f64>, lumped_mass: &[f64]) -> f64 {
    try_gershgorin_omega_max(stiffness, lumped_mass).expect("stiffness/mass must be valid")
}

/// Power-iteration estimate of `omega_max = sqrt(lambda_max(M^{-1} K))`.
///
/// Iterates `x <- M^{-1} K x / ‖·‖` from a fixed deterministic start vector
/// and returns `sqrt(‖y‖/‖x‖)` after `num_iters` iterations. Converges to
/// the dominant eigenvalue from below; for chain/FE stiffness matrices a few
/// dozen iterations are typically enough for a percent-level estimate. Use
/// the slightly more expensive [`gershgorin_omega_max`] when a guaranteed
/// upper bound is required.
/// Fallible power-iteration estimate of `omega_max`. Rejects `dim == 0` and
/// `num_iters == 0` (which would silently return 0 and panic downstream in
/// [`critical_time_step`]) instead of asserting.
pub fn try_power_iteration_omega_max<K, MS>(
    stiffness: &K,
    mass_solve: MS,
    dim: usize,
    num_iters: usize,
) -> Result<f64, SolverError>
where
    K: LinearOperator<f64>,
    MS: Fn(&[f64]) -> Vec<f64>,
{
    if dim == 0 {
        return Err(SolverError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if num_iters == 0 {
        return Err(SolverError::OperationFailed {
            operation: "power_iteration_omega_max",
            reason: "num_iters must be at least 1".into(),
        });
    }
    Ok(power_iteration_omega_max(
        stiffness, mass_solve, dim, num_iters,
    ))
}

pub fn power_iteration_omega_max<K, MS>(
    stiffness: &K,
    mass_solve: MS,
    dim: usize,
    num_iters: usize,
) -> f64
where
    K: LinearOperator<f64>,
    MS: Fn(&[f64]) -> Vec<f64>,
{
    assert!(dim > 0, "dim must be positive");
    assert!(num_iters > 0, "num_iters must be at least 1");
    // Deterministic non-symmetric start vector (excites all modes of
    // chain-like operators).
    let mut x: Vec<f64> = (0..dim).map(|i| 1.0 + 0.01 * (i % 7) as f64).collect();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let x_norm = norm(&x);
    for xi in x.iter_mut() {
        *xi /= x_norm;
    }
    let mut lambda = 0.0;
    // Reused operator-action scratch: `apply_into` instead of the allocating
    // `apply` (the default `apply` forwards to `apply_into`, so identical).
    let mut kx = vec![0.0; dim];
    for _ in 0..num_iters {
        stiffness.apply_into(&x, &mut kx);
        let y = mass_solve(&kx);
        let y_norm = norm(&y);
        assert!(y_norm > 0.0, "M^{{-1}} K x = 0: no positive eigenvalue");
        lambda = y_norm; // ‖x‖ = 1 throughout
        let inv = 1.0 / y_norm;
        for (xi, &yi) in x.iter_mut().zip(y.iter()) {
            *xi = yi * inv;
        }
    }
    lambda.sqrt()
}

/// Integrate `M u'' = F(u, v, t)` with the explicit central-difference
/// scheme (Newmark `beta = 0, gamma = 1/2`).
///
/// Generic over any [`ComplexField`] type: works with f32, f64, Complex32,
/// and Complex64.
///
/// **Stability is the caller's responsibility**: `config.dt` must satisfy
/// the conditional-stability bound (see the module docs and
/// [`critical_time_step`]).
///
/// # Arguments
///
/// - `force_fn`: residual force `F(u, v, t) = f_ext(t) - f_int(u, v, t)`.
///   Damping (`-C v`, Rayleigh or otherwise) belongs inside this closure;
///   it is evaluated with the half-step velocity.
/// - `mass_solve`: closure solving `M a = r` for `a`. Trivial for a lumped
///   (diagonal) mass matrix.
/// - `initial`: initial displacement and velocity. The `acceleration` field
///   is **ignored** — a consistent `a_0 = M^{-1} F(u_0, v_0, 0)` is computed
///   internally so the first step never inherits an inconsistent state.
/// - `config`: time step, step count, history flag.
pub fn explicit_integrate<T, F, S>(
    force_fn: F,
    mass_solve: S,
    initial: &NewmarkInitial<T>,
    config: &ExplicitConfig,
) -> NewmarkResult<T>
where
    T: ComplexField,
    F: Fn(&[T], &[T], f64) -> Vec<T>,
    S: Fn(&[T]) -> Vec<T>,
{
    let dt = config.dt;
    let dt_t = T::from_real(T::Real::from_f64(dt).unwrap());
    let half_dt = T::from_real(T::Real::from_f64(0.5 * dt).unwrap());
    let half_dt2 = T::from_real(T::Real::from_f64(0.5 * dt * dt).unwrap());

    let mut u = initial.displacement.clone();
    let mut v = initial.velocity.clone();
    // Consistent initial acceleration: a_0 = M^{-1} F(u_0, v_0, 0).
    let mut a = mass_solve(&force_fn(&u, &v, 0.0));

    let n = u.len();
    assert_eq!(v.len(), n, "displacement/velocity size mismatch");
    assert_eq!(a.len(), n, "force/mass-solve size mismatch");

    // Pre-allocate buffers (reused across time steps, zero allocation in loop).
    let mut u_next = vec![T::zero(); n];
    let mut v_half = vec![T::zero(); n];

    let mut u_history = if config.store_history {
        Some(vec![u.clone()])
    } else {
        None
    };
    let mut v_history = if config.store_history {
        Some(vec![v.clone()])
    } else {
        None
    };
    let mut a_history = if config.store_history {
        Some(vec![a.clone()])
    } else {
        None
    };

    for step in 0..config.num_steps {
        let t_next = (step + 1) as f64 * dt;

        // u_{n+1} = u_n + dt*v_n + (dt^2/2)*a_n
        u_next.copy_from_slice(&u);
        T::vec_axpy(dt_t, &v, &mut u_next);
        T::vec_axpy(half_dt2, &a, &mut u_next);
        // v_{n+1/2} = v_n + (dt/2)*a_n
        v_half.copy_from_slice(&v);
        T::vec_axpy(half_dt, &a, &mut v_half);

        // a_{n+1} = M^{-1} F(u_{n+1}, v_{n+1/2}, t_{n+1})
        let a_new = mass_solve(&force_fn(&u_next, &v_half, t_next));

        // v_{n+1} = v_{n+1/2} + (dt/2)*a_{n+1}
        T::vec_axpy(half_dt, &a_new, &mut v_half);

        std::mem::swap(&mut u, &mut u_next);
        std::mem::swap(&mut v, &mut v_half);
        a = a_new;

        if let Some(ref mut hist) = u_history {
            hist.push(u.clone());
        }
        if let Some(ref mut hist) = v_history {
            hist.push(v.clone());
        }
        if let Some(ref mut hist) = a_history {
            hist.push(a.clone());
        }
    }

    NewmarkResult {
        displacement: u,
        velocity: v,
        acceleration: a,
        displacement_history: u_history,
        velocity_history: v_history,
        acceleration_history: a_history,
        steps_completed: config.num_steps,
        final_time: config.num_steps as f64 * dt,
    }
}

/// Checked, panic-contained explicit central-difference integration.
///
/// The legacy [`explicit_integrate`] entry point remains available for
/// trusted callers. This boundary validates the time step and initial vector
/// dimensions, validates generated force and mass-solve vectors, contains
/// callback/solver panics, and rejects a non-finite or dimensionally malformed
/// final trajectory instead of returning it as a successful result.
pub fn try_explicit_integrate<T, F, S>(
    force_fn: F,
    mass_solve: S,
    initial: &NewmarkInitial<T>,
    config: &ExplicitConfig,
) -> Result<NewmarkResult<T>, SolverError>
where
    T: ComplexField,
    F: Fn(&[T], &[T], f64) -> Vec<T>,
    S: Fn(&[T]) -> Vec<T>,
{
    config.validate()?;
    validate_initial_state(initial)?;
    // Mirror the Newmark `try_` path: reject dt coefficients the target
    // scalar cannot represent before the raw integrator's `unwrap()`s run.
    super::newmark::validate_scalar_coefficients_for_explicit::<T>(
        "explicit integration",
        config.dt,
    )?;
    let dimension = initial.displacement.len();
    let callback_error = RefCell::new(None);
    let checked_force = |displacement: &[T], velocity: &[T], time: f64| {
        let values = force_fn(displacement, velocity, time);
        if let Err(error) = validate_callback_vector(
            &values,
            dimension,
            "explicit integration force callback",
            "force output",
        ) {
            store_callback_error(&callback_error, error);
        }
        values
    };
    let checked_mass = |rhs: &[T]| {
        if let Err(error) = validate_callback_vector(
            rhs,
            dimension,
            "explicit integration mass callback",
            "mass-solve input",
        ) {
            store_callback_error(&callback_error, error);
        }
        let values = mass_solve(rhs);
        if let Err(error) = validate_callback_vector(
            &values,
            dimension,
            "explicit integration mass callback",
            "mass-solve output",
        ) {
            store_callback_error(&callback_error, error);
        }
        values
    };
    let result = match catch_unwind(AssertUnwindSafe(|| {
        explicit_integrate(checked_force, checked_mass, initial, config)
    })) {
        Ok(result) => {
            if let Some(error) = callback_error.into_inner() {
                return Err(error);
            }
            result
        }
        Err(payload) => {
            if let Some(error) = callback_error.into_inner() {
                return Err(error);
            }
            return Err(SolverError::OperationFailed {
                operation: "explicit integration",
                reason: panic_reason(payload),
            });
        }
    };
    validate_result(&result, config.num_steps, "explicit integration")?;
    Ok(result)
}

fn validate_callback_vector<T: ComplexField>(
    values: &[T],
    expected: usize,
    operation: &'static str,
    role: &'static str,
) -> Result<(), SolverError> {
    if values.len() != expected {
        return Err(SolverError::DimensionMismatch {
            expected,
            got: values.len(),
        });
    }
    if let Some(index) = values
        .iter()
        .position(|value| !value.re().is_finite() || !value.im().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!("{role} is non-finite at index {index}"),
        });
    }
    Ok(())
}

fn store_callback_error(slot: &RefCell<Option<SolverError>>, error: SolverError) -> ! {
    *slot.borrow_mut() = Some(error);
    panic!("checked explicit integration callback output rejected");
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "callback panicked with a non-string payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_sparse::CsrMatrix;

    /// Build the 1-D fixed-fixed spring chain stiffness (tridiagonal
    /// [-k, 2k, -k]) for `n` free interior masses.
    fn chain_stiffness(n: usize, k: f64) -> CsrMatrix<f64> {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 2.0 * k;
            if i > 0 {
                data[i * n + i - 1] = -k;
            }
            if i + 1 < n {
                data[i * n + i + 1] = -k;
            }
        }
        let dense = sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(n, n, data);
        CsrMatrix::from_dense(&dense, 0.0)
    }

    /// Exact highest natural frequency of the fixed-fixed unit chain:
    /// `omega_j = 2 sqrt(k/m) sin(j pi / (2 (n+1)))`, `j = n` is the max.
    fn chain_omega_max(n: usize, k: f64, m: f64) -> f64 {
        2.0 * (k / m).sqrt() * ((n as f64 * std::f64::consts::PI) / (2.0 * (n + 1) as f64)).sin()
    }

    #[test]
    fn checked_explicit_boundary_validates_config_dimensions_and_panics() {
        let stiffness = CsrMatrix::from_dense(&sonium_solvers_core::array![[1.0_f64]], 0.0);
        let initial = NewmarkInitial {
            displacement: vec![0.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };
        let invalid = ExplicitConfig::new(0.0, 1);
        assert!(matches!(
            try_explicit_integrate(
                |_u: &[f64], _v: &[f64], _t| vec![0.0],
                |_rhs: &[f64]| vec![0.0],
                &initial,
                &invalid,
            ),
            Err(SolverError::OperationFailed { .. })
        ));

        let mismatched = NewmarkInitial {
            displacement: vec![0.0],
            velocity: vec![],
            acceleration: vec![0.0],
        };
        assert!(matches!(
            try_explicit_integrate(
                |_u: &[f64], _v: &[f64], _t| vec![0.0],
                |_rhs: &[f64]| vec![0.0],
                &mismatched,
                &ExplicitConfig::new(0.1, 1),
            ),
            Err(SolverError::DimensionMismatch { .. })
        ));

        let error = try_explicit_integrate(
            |_u: &[f64], _v: &[f64], _t| panic!("force callback failure"),
            |_rhs: &[f64]| vec![0.0],
            &initial,
            &ExplicitConfig::new(0.1, 1),
        )
        .expect_err("callback panic must be contained");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let result = try_explicit_integrate(
            move |u: &[f64], _v: &[f64], _t| vec![-u[0]],
            |_rhs: &[f64]| vec![0.0],
            &initial,
            &ExplicitConfig::new(0.1, 1),
        )
        .expect("finite trajectory should pass the checked boundary");
        assert_eq!(
            result.solver_status(),
            sonium_solvers_core::traits::SolverStatus::Converged
        );
        assert_eq!(
            result.accuracy_status(),
            sonium_solvers_core::traits::AccuracyStatus::NotChecked
        );
        let _ = stiffness;
    }

    /// Free vibration of a single DOF: m*u'' + k*u = 0.
    /// Exact solution: u(t) = u0 * cos(omega*t), omega = sqrt(k/m).
    #[test]
    fn test_single_dof_free_vibration() {
        let m_val = 1.0_f64;
        let k_val = 4.0_f64;
        let omega = (k_val / m_val).sqrt();

        // dt well below dt_crit = 2/omega = 1.
        let period = 2.0 * std::f64::consts::PI / omega;
        let dt = period / 100.0;
        let num_steps = 500; // 5 periods
        let config = ExplicitConfig::new(dt, num_steps);

        let initial = NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.0],
            acceleration: vec![0.0], // ignored: recomputed consistently
        };

        let result = explicit_integrate(
            move |u: &[f64], _v: &[f64], _t| vec![-k_val * u[0]],
            move |r: &[f64]| vec![r[0] / m_val],
            &initial,
            &config,
        );

        let t_final = result.final_time;
        let u_exact = (omega * t_final).cos();
        assert!(
            (result.displacement[0] - u_exact).abs() < 0.02,
            "u({t_final:.4}) = {:.6}, expected {u_exact:.6}",
            result.displacement[0]
        );
        // Consistent a_0 must have been computed internally.
        assert!(
            (result.acceleration[0]).abs() < k_val * 1.1,
            "acceleration should be bounded, got {}",
            result.acceleration[0]
        );
    }

    /// Second-order accuracy: dt refinement study on the undamped SDOF
    /// oscillator; the error at a fixed time must decrease ~4x per halving.
    #[test]
    fn test_single_dof_second_order() {
        let m_val = 1.0_f64;
        let k_val = 4.0_f64;
        let omega = 2.0_f64;
        let period = 2.0 * std::f64::consts::PI / omega;
        let t_final = 1.25 * period;
        let u_exact = (omega * t_final).cos();

        let initial = NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };

        let mut errors = Vec::new();
        for &num_steps in &[40_usize, 80, 160] {
            let dt = t_final / num_steps as f64;
            let config = ExplicitConfig::new(dt, num_steps);
            let result = explicit_integrate(
                move |u: &[f64], _v: &[f64], _t| vec![-k_val * u[0]],
                move |r: &[f64]| vec![r[0] / m_val],
                &initial,
                &config,
            );
            errors.push((result.displacement[0] - u_exact).abs());
        }
        let ratio1 = errors[0] / errors[1];
        let ratio2 = errors[1] / errors[2];
        assert!(
            (3.5..=4.5).contains(&ratio1),
            "expected 2nd order (ratio ~4), got {ratio1:.2} ({errors:?})"
        );
        assert!(
            (3.5..=4.5).contains(&ratio2),
            "expected 2nd order (ratio ~4), got {ratio2:.2} ({errors:?})"
        );
    }

    /// The dt_crit helpers against the known spectrum of the spring chain.
    #[test]
    fn test_dt_crit_helpers_on_chain() {
        let n = 10;
        let k = 1.0_f64;
        let m = 1.0_f64;
        let stiff = chain_stiffness(n, k);
        let mass = vec![m; n];
        let omega_max_exact = chain_omega_max(n, k, m);

        // Power iteration: percent-level estimate of the exact value.
        let omega_pi = power_iteration_omega_max(&stiff, |r: &[f64]| r.to_vec(), n, 200);
        assert!(
            (omega_pi - omega_max_exact).abs() / omega_max_exact < 0.01,
            "power iteration {omega_pi} vs exact {omega_max_exact}"
        );

        // Gershgorin: guaranteed upper bound, interior rows give 4k/m.
        let omega_g = gershgorin_omega_max(&stiff, &mass);
        assert!(
            omega_g >= omega_max_exact,
            "Gershgorin {omega_g} must bound exact {omega_max_exact}"
        );
        assert!(
            (omega_g - 2.0 * (k / m).sqrt()).abs() < 1e-12,
            "chain Gershgorin bound should be exactly 2 sqrt(k/m), got {omega_g}"
        );

        // dt_crit formulas.
        let dt_crit = critical_time_step(omega_max_exact);
        assert!((dt_crit - 2.0 / omega_max_exact).abs() < 1e-15);
        let xi = 0.1;
        let dt_crit_d = critical_time_step_damped(omega_max_exact, xi);
        let expect = 2.0 * ((xi * xi + 1.0).sqrt() - xi) / omega_max_exact;
        assert!((dt_crit_d - expect).abs() < 1e-15);
        assert!(dt_crit_d < dt_crit, "damping lowers dt_crit");
    }

    /// Stability contract: a unit chain integrated below dt_crit stays
    /// bounded over long times; integrated above dt_crit it blows up.
    #[test]
    fn test_stability_bounded_below_dt_crit() {
        let n = 10;
        let k = 1.0_f64;
        let m = 1.0_f64;
        let stiff = chain_stiffness(n, k);
        let omega_max = chain_omega_max(n, k, m);
        let dt_crit = critical_time_step(omega_max);

        // Non-symmetric initial state exciting all modes.
        let u0: Vec<f64> = (0..n)
            .map(|i| 0.1 * (i as f64 + 1.0).sin() + 0.05)
            .collect();
        let initial = NewmarkInitial {
            displacement: u0.clone(),
            velocity: vec![0.0; n],
            acceleration: vec![0.0; n],
        };
        let force = move |u: &[f64], _v: &[f64], _t| {
            sonium_solvers_core::vec_ops::vec_scale(&stiff.apply(u), -1.0)
        };
        let mass_solve = move |r: &[f64]| r.iter().map(|x| x / m).collect::<Vec<f64>>();

        // Below dt_crit: bounded for 20_000 steps (>> any beat period).
        let config = ExplicitConfig::new(0.9 * dt_crit, 20_000).with_history();
        let result = explicit_integrate(&force, mass_solve, &initial, &config);
        let u_max = result
            .displacement_history
            .unwrap()
            .iter()
            .map(|u| u.iter().fold(0.0_f64, |a, &b| a.max(b.abs())))
            .fold(0.0_f64, f64::max);
        let u0_max = u0.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(
            u_max < 10.0 * u0_max,
            "dt < dt_crit must stay bounded, max |u| = {u_max}"
        );

        // Above dt_crit: the highest mode grows geometrically (factor ~2.4
        // per step here). 200 steps suffice to reach ~1e77 while staying
        // short of floating-point overflow.
        let config = ExplicitConfig::new(1.1 * dt_crit, 200);
        let result = explicit_integrate(&force, mass_solve, &initial, &config);
        let u_max = result
            .displacement
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(
            u_max > 1e3 * u0_max,
            "dt > dt_crit must diverge, max |u| = {u_max}"
        );
    }

    /// Damped SDOF free vibration through the force closure (Rayleigh-style
    /// damping needs no special-casing): matches the analytical underdamped
    /// decay and stays within the damped dt_crit bound.
    #[test]
    fn test_damped_free_vibration() {
        let m_val = 1.0_f64;
        let c_val = 0.5;
        let k_val = 4.0_f64;
        let omega0 = (k_val / m_val).sqrt();
        let zeta = c_val / (2.0 * m_val * omega0);
        let omega_d = omega0 * (1.0 - zeta * zeta).sqrt();

        // dt well below the damped critical step (central difference has
        // O(dt^2) phase error; omega*dt must be small for an accurate phase
        // over 5 periods).
        let dt_crit = critical_time_step_damped(omega0, zeta);
        let period = 2.0 * std::f64::consts::PI / omega_d;
        let dt = period / 200.0;
        assert!(dt < dt_crit, "test dt must respect the stability bound");
        let num_steps = 1000; // 5 periods
        let config = ExplicitConfig::new(dt, num_steps);

        let initial = NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };
        let result = explicit_integrate(
            move |u: &[f64], v: &[f64], _t| vec![-c_val * v[0] - k_val * u[0]],
            move |r: &[f64]| vec![r[0] / m_val],
            &initial,
            &config,
        );

        let t = result.final_time;
        let u_exact = (-zeta * omega0 * t).exp()
            * ((omega_d * t).cos() + (zeta * omega0 / omega_d) * (omega_d * t).sin());
        assert!(
            (result.displacement[0] - u_exact).abs() < 5e-3,
            "u({t:.4}) = {:.6}, expected {u_exact:.6}",
            result.displacement[0]
        );
    }

    /// Wave-propagation sanity: a Gaussian displacement pulse released in a
    /// long fixed-fixed chain splits into two pulses travelling at the
    /// lattice wave speed c = a*sqrt(k/m) (a = 1 here).
    #[test]
    fn test_wave_propagation_1d_chain() {
        let n = 200;
        let k = 1.0_f64;
        let m = 1.0_f64;
        let c = (k / m).sqrt(); // lattice wave speed = 1
        let stiff = chain_stiffness(n, k);
        let omega_max = chain_omega_max(n, k, m);
        let dt = 0.9 * critical_time_step(omega_max);

        // Gaussian pulse centred at node 50, zero velocity.
        let center = 50.0;
        let width = 5.0;
        let u0: Vec<f64> = (0..n)
            .map(|i| (-((i as f64 - center).powi(2)) / (2.0 * width * width)).exp())
            .collect();
        let initial = NewmarkInitial {
            displacement: u0,
            velocity: vec![0.0; n],
            acceleration: vec![0.0; n],
        };

        // Travel time before reflections from the fixed ends matter:
        // right-moving pulse centre should reach ~50 + c*t.
        let t_final = 30.0;
        let num_steps = (t_final / dt) as usize;
        let config = ExplicitConfig::new(dt, num_steps);
        let result = explicit_integrate(
            move |u: &[f64], _v: &[f64], _t| {
                sonium_solvers_core::vec_ops::vec_scale(&stiff.apply(u), -1.0)
            },
            move |r: &[f64]| r.iter().map(|x| x / m).collect::<Vec<f64>>(),
            &initial,
            &config,
        );

        // Locate the peak of |u| in the right half of the chain.
        let (peak_idx, peak_val) = result.displacement[60..]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i + 60, v.abs()))
            .fold((0, 0.0_f64), |a, b| if b.1 > a.1 { b } else { a });
        let expected = 50.0 + c * t_final; // = 80
        assert!(
            peak_val > 0.2,
            "pulse amplitude should survive, peak = {peak_val}"
        );
        assert!(
            (peak_idx as f64 - expected).abs() < 8.0,
            "pulse peak at node {peak_idx}, expected near {expected:.0}"
        );
    }
}
