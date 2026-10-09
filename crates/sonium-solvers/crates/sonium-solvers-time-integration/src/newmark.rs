//! Newmark-beta time integration for second-order ODEs
//!
//! Solves `M u'' + C u' + K u = f(t)` with the implicit Newmark scheme.
//!
//! ## Algorithm
//!
//! Predictor:
//! ```text
//! u_pred    = u_n + dt * v_n + (0.5 - beta) * dt^2 * a_n
//! v_pred    = v_n + (1 - gamma) * dt * a_n
//! ```
//!
//! Effective system (solve for a_{n+1}):
//! ```text
//! (M + gamma*dt*C + beta*dt^2*K) a_{n+1} = f_{n+1} - C*v_pred - K*u_pred
//! ```
//!
//! Corrector:
//! ```text
//! u_{n+1} = u_pred + beta * dt^2 * a_{n+1}
//! v_{n+1} = v_pred + gamma * dt * a_{n+1}
//! ```
//!
//! ## Parameters
//!
//! - `beta = 1/4, gamma = 1/2`: Average acceleration (unconditionally stable, no dissipation)
//! - `beta = 1/6, gamma = 1/2`: Linear acceleration (conditionally stable)
//!
//! ## Generalized-alpha (Chung & Hulbert, 1993)
//!
//! [`generalized_alpha_integrate`] implements the true generalized-alpha
//! method: the residual is evaluated at the intermediate point
//! `t_{n+1-alpha_f}` with alpha-weighted states,
//!
//! ```text
//! M a_{n+1-alpha_m} + C v_{n+1-alpha_f} + K u_{n+1-alpha_f} = f(t_{n+1-alpha_f})
//! x_{n+1-alpha} = (1 - alpha) * x_{n+1} + alpha * x_n
//! ```
//!
//! which leads to the effective system (solved for `a_{n+1}`)
//!
//! ```text
//! ((1-alpha_m)*M + (1-alpha_f)*gamma*dt*C + (1-alpha_f)*beta*dt^2*K) a_{n+1} = rhs
//! ```
//!
//! Coefficients are parameterized by the spectral radius at infinity
//! `rho_inf` in `[0, 1]` (user-controlled high-frequency dissipation):
//!
//! ```text
//! alpha_m = (2*rho_inf - 1) / (rho_inf + 1)
//! alpha_f = rho_inf / (rho_inf + 1)
//! beta    = (1 - alpha_m + alpha_f)^2 / 4
//! gamma   = 1/2 - alpha_m + alpha_f
//! ```
//!
//! `rho_inf = 1` recovers the average-acceleration Newmark method (no
//! dissipation); `rho_inf = 0` annihilates the highest-frequency mode in one
//! step. HHT-alpha is the special case `alpha_m = 0`, `alpha_f = alpha`.
//!
//! ## Scalar type
//!
//! All state vectors are generic over [`ComplexField`](crate::ComplexField),
//! so f32, f64, Complex32, and Complex64 are all supported.

use num_traits::{Float, FromPrimitive};

use super::revolve::RevolveSchedule;
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, SolverError,
    SolverResultContract, SolverStatus,
};
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Newmark-beta integration parameters.
#[derive(Debug, Clone)]
pub struct NewmarkConfig {
    /// Newmark beta parameter (controls displacement update).
    pub beta: f64,
    /// Newmark gamma parameter (controls velocity update).
    pub gamma: f64,
    /// Time step size.
    pub dt: f64,
    /// Number of time steps.
    pub num_steps: usize,
    /// Store full history (all time steps) or only final state.
    pub store_history: bool,
}

impl NewmarkConfig {
    /// Validate parameters before entering the integration loop.
    pub fn validate(&self) -> Result<(), SolverError> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Newmark configuration",
                reason: "dt must be finite and positive".to_string(),
            });
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Newmark configuration",
                reason: "beta must be finite and positive".to_string(),
            });
        }
        if !self.gamma.is_finite() || self.gamma < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Newmark configuration",
                reason: "gamma must be finite and non-negative".to_string(),
            });
        }
        if self.num_steps.checked_add(1).is_none() || !(self.num_steps as f64 * self.dt).is_finite()
        {
            return Err(SolverError::OperationFailed {
                operation: "Newmark configuration",
                reason: "num_steps and final time overflow".to_string(),
            });
        }
        Ok(())
    }

    /// Build a bounded-checkpoint reverse schedule for this trajectory.
    ///
    /// A transient adjoint driver can execute the returned actions by
    /// restoring a state, recomputing a segment, and applying one reverse
    /// recurrence. Full history storage remains available through
    /// [`Self::with_history`] for small problems and compatibility callers.
    pub fn checkpoint_schedule(
        &self,
        max_checkpoints: usize,
    ) -> Result<RevolveSchedule, SolverError> {
        RevolveSchedule::new(self.num_steps, max_checkpoints).map_err(|error| {
            SolverError::OperationFailed {
                operation: "Newmark checkpoint schedule",
                reason: error.to_string(),
            }
        })
    }

    /// Average acceleration method (unconditionally stable, second-order accurate).
    pub fn average_acceleration(dt: f64, num_steps: usize) -> Self {
        Self {
            beta: 0.25,
            gamma: 0.5,
            dt,
            num_steps,
            store_history: false,
        }
    }

    /// Newmark with HHT-alpha parameters (`beta = (1+alpha)^2/4`,
    /// `gamma = 1/2 + alpha`).
    ///
    /// **Note:** this preset only sets Newmark parameters — the residual is
    /// still evaluated fully at `t_{n+1}`, so it is *not* the true HHT-alpha
    /// method (which requires alpha-weighting of the residual) and with
    /// `alpha < 0` yields `gamma < 1/2`. It is kept unchanged for backward
    /// compatibility. For the true HHT-alpha method use
    /// [`GeneralizedAlphaConfig::hht_alpha`] with
    /// [`generalized_alpha_integrate`].
    #[deprecated(
        note = "parameter-only approximation, not true HHT-alpha; use GeneralizedAlphaConfig::hht_alpha with generalized_alpha_integrate"
    )]
    pub fn hht_alpha(alpha: f64, dt: f64, num_steps: usize) -> Self {
        Self {
            beta: (1.0 + alpha).powi(2) / 4.0,
            gamma: 0.5 + alpha,
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
}

/// Convert real-valued integration coefficients without allowing a target
/// scalar conversion to panic or silently produce an infinity.
/// Explicit-integrator coefficient check shared with `explicit.rs`: `dt`,
/// `dt/2`, and `dt^2/2` must all be representable and finite in the target
/// scalar, matching what the raw integrator converts with `unwrap()`.
pub(crate) fn validate_scalar_coefficients_for_explicit<T: ComplexField>(
    operation: &'static str,
    dt: f64,
) -> Result<(), SolverError> {
    validate_scalar_coefficients::<T>(
        operation,
        &[("dt", dt), ("dt/2", 0.5 * dt), ("dt^2/2", 0.5 * dt * dt)],
    )
}

fn validate_scalar_coefficients<T: ComplexField>(
    operation: &'static str,
    coefficients: &[(&'static str, f64)],
) -> Result<(), SolverError> {
    for &(name, value) in coefficients {
        let Some(converted) = T::Real::from_f64(value) else {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!("{name} cannot be represented by the target scalar"),
            });
        };
        if !converted.is_finite() {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!("{name} is non-finite in the target scalar"),
            });
        }
    }
    Ok(())
}

/// Compatibility kernels return a result value rather than `Result`; use a
/// non-finite coefficient on an impossible conversion so the checked wrapper
/// can report the failure through its normal result-contract validation.
#[inline]
fn scalar_from_f64_or_nan<T: ComplexField>(value: f64) -> T {
    T::from_real(T::Real::from_f64(value).unwrap_or_else(T::Real::nan))
}

/// Forward state passed through a checkpointed Newmark trajectory.
#[derive(Debug, Clone)]
pub struct NewmarkState<T: ComplexField> {
    pub displacement: Vec<T>,
    pub velocity: Vec<T>,
    pub acceleration: Vec<T>,
}

impl<T: ComplexField> NewmarkState<T> {
    /// Construct a state from displacement, velocity, and acceleration.
    pub fn new(displacement: Vec<T>, velocity: Vec<T>, acceleration: Vec<T>) -> Self {
        Self {
            displacement,
            velocity,
            acceleration,
        }
    }
}

/// Result of a Newmark time integration, generic over the scalar type.
#[derive(Debug, Clone)]
pub struct NewmarkResult<T: ComplexField> {
    /// Final displacement.
    pub displacement: Vec<T>,
    /// Final velocity.
    pub velocity: Vec<T>,
    /// Final acceleration.
    pub acceleration: Vec<T>,
    /// Displacement history (if `store_history` was enabled): `[step][dof]`.
    pub displacement_history: Option<Vec<Vec<T>>>,
    /// Velocity history (if `store_history` was enabled).
    pub velocity_history: Option<Vec<Vec<T>>>,
    /// Acceleration history (if `store_history` was enabled).
    pub acceleration_history: Option<Vec<Vec<T>>>,
    /// Number of time steps completed.
    pub steps_completed: usize,
    /// Final time.
    pub final_time: f64,
}

impl<T: ComplexField> NewmarkResult<T> {
    /// Return the canonical solver status for the completed trajectory.
    ///
    /// Time integration has no scalar residual tolerance of its own: a
    /// completed trajectory is therefore `Converged` when all exposed state
    /// and complete history values are finite, and `NumericalError` otherwise.
    /// Consumers that have an equation-specific residual should retain that
    /// residual as a separate accuracy gate.
    pub fn solver_status(&self) -> SolverStatus {
        let finite_state = self
            .displacement
            .iter()
            .chain(self.velocity.iter())
            .chain(self.acceleration.iter())
            .all(|value| value.re().is_finite() && value.im().is_finite());
        let matching_state = self.displacement.len() == self.velocity.len()
            && self.displacement.len() == self.acceleration.len();
        let histories = [
            self.displacement_history.as_ref(),
            self.velocity_history.as_ref(),
            self.acceleration_history.as_ref(),
        ];
        let present = histories.iter().filter(|history| history.is_some()).count();
        let histories_are_consistent = match present {
            0 => true,
            3 => {
                let Some(expected_len) = self.steps_completed.checked_add(1) else {
                    return SolverStatus::NumericalError;
                };
                histories.iter().all(|history| {
                    let Some(history) = history else {
                        return false;
                    };
                    history.len() == expected_len
                        && history.iter().all(|state| {
                            state.len() == self.displacement.len()
                                && state
                                    .iter()
                                    .all(|value| value.re().is_finite() && value.im().is_finite())
                        })
                })
            }
            _ => false,
        };
        if finite_state && matching_state && histories_are_consistent && self.final_time.is_finite()
        {
            SolverStatus::Converged
        } else {
            SolverStatus::NumericalError
        }
    }

    /// Return the canonical accuracy status for this trajectory.
    ///
    /// The time integrator does not know the caller's discretization or
    /// equation residual tolerance, so a finite trajectory is explicitly
    /// reported as `NotChecked` rather than being mistaken for an attained
    /// PDE accuracy target.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::NotChecked,
            _ => AccuracyStatus::NotAttained,
        }
    }

    /// Alias matching the checked result vocabulary used by the other solver
    /// families.
    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    /// Alias matching the checked result vocabulary used by the other solver
    /// families.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

/// Portable checkpoint format version for [`NewmarkResult<f64>`].
pub const NEWMARK_CHECKPOINT_FORMAT_VERSION: u16 = 1;

/// Magic prefix for [`NewmarkResult<f64>`] checkpoints.
pub const NEWMARK_CHECKPOINT_MAGIC: [u8; 8] = *b"NMRKCP01";
const NEWMARK_CHECKPOINT_MAX_ELEMENTS: usize = 16 * 1024 * 1024;

impl NewmarkResult<f64> {
    /// Encode a finite Newmark trajectory into a strict little-endian packet.
    ///
    /// The packet is self-describing: vector lengths, optional history
    /// presence, and every history row length are encoded so malformed or
    /// truncated state cannot be mistaken for a valid restart.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        validate_newmark_checkpoint_state(self)?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&NEWMARK_CHECKPOINT_MAGIC);
        bytes.extend_from_slice(&NEWMARK_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes()); // reserved flags
        bytes.extend_from_slice(&(self.steps_completed as u64).to_le_bytes());
        bytes.extend_from_slice(&self.final_time.to_le_bytes());
        write_checkpoint_vector(&mut bytes, &self.displacement);
        write_checkpoint_vector(&mut bytes, &self.velocity);
        write_checkpoint_vector(&mut bytes, &self.acceleration);
        write_checkpoint_history(&mut bytes, self.displacement_history.as_deref());
        write_checkpoint_history(&mut bytes, self.velocity_history.as_deref());
        write_checkpoint_history(&mut bytes, self.acceleration_history.as_deref());
        Ok(bytes)
    }

    /// Restore a Newmark trajectory from [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = CheckpointReader::new(bytes);
        if reader.take(NEWMARK_CHECKPOINT_MAGIC.len())? != NEWMARK_CHECKPOINT_MAGIC {
            return Err(newmark_checkpoint_error("invalid magic"));
        }
        let version = reader.u16()?;
        if version != NEWMARK_CHECKPOINT_FORMAT_VERSION {
            return Err(newmark_checkpoint_error("unsupported checkpoint version"));
        }
        if reader.u16()? != 0 {
            return Err(newmark_checkpoint_error("reserved flags are non-zero"));
        }
        let steps_completed = usize::try_from(reader.u64()?)
            .map_err(|_| newmark_checkpoint_error("step count overflows usize"))?;
        let final_time = reader.f64()?;
        let displacement = reader.vector()?;
        let velocity = reader.vector()?;
        let acceleration = reader.vector()?;
        let displacement_history = reader.history()?;
        let velocity_history = reader.history()?;
        let acceleration_history = reader.history()?;
        if !reader.is_empty() {
            return Err(newmark_checkpoint_error("trailing bytes"));
        }

        let result = Self {
            displacement,
            velocity,
            acceleration,
            displacement_history,
            velocity_history,
            acceleration_history,
            steps_completed,
            final_time,
        };
        validate_newmark_checkpoint_state(&result)?;
        Ok(result)
    }
}

fn newmark_checkpoint_error(reason: &str) -> SolverError {
    SolverError::OperationFailed {
        operation: "Newmark checkpoint",
        reason: reason.into(),
    }
}

fn validate_newmark_checkpoint_state(state: &NewmarkResult<f64>) -> Result<(), SolverError> {
    let dimension = state.displacement.len();
    if state.velocity.len() != dimension || state.acceleration.len() != dimension {
        return Err(newmark_checkpoint_error("state vector dimensions differ"));
    }
    if !state.final_time.is_finite()
        || state
            .displacement
            .iter()
            .chain(&state.velocity)
            .chain(&state.acceleration)
            .any(|value| !value.is_finite())
    {
        return Err(newmark_checkpoint_error("state contains non-finite values"));
    }
    let histories = [
        state.displacement_history.as_ref(),
        state.velocity_history.as_ref(),
        state.acceleration_history.as_ref(),
    ];
    let present = histories.iter().filter(|history| history.is_some()).count();
    if present != 0 && present != histories.len() {
        return Err(newmark_checkpoint_error(
            "history presence flags do not match",
        ));
    }
    if present == histories.len() {
        let expected_len = state
            .steps_completed
            .checked_add(1)
            .ok_or_else(|| newmark_checkpoint_error("history length overflows usize"))?;
        if histories.iter().any(|history| {
            history
                .as_ref()
                .is_none_or(|history| history.len() != expected_len)
        }) {
            return Err(newmark_checkpoint_error(
                "history lengths do not match completed steps",
            ));
        }
    }
    for history in [
        state.displacement_history.as_ref(),
        state.velocity_history.as_ref(),
        state.acceleration_history.as_ref(),
    ] {
        let invalid_history = history.is_some_and(|history| {
            history.len() > NEWMARK_CHECKPOINT_MAX_ELEMENTS
                || history
                    .iter()
                    .any(|row| row.len() != dimension || row.iter().any(|value| !value.is_finite()))
        });
        if invalid_history {
            return Err(newmark_checkpoint_error(
                "history shape or values are invalid",
            ));
        }
    }
    Ok(())
}

fn write_checkpoint_vector(bytes: &mut Vec<u8>, values: &[f64]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn write_checkpoint_history(bytes: &mut Vec<u8>, history: Option<&[Vec<f64>]>) {
    match history {
        None => bytes.push(0),
        Some(history) => {
            bytes.push(1);
            bytes.extend_from_slice(&(history.len() as u64).to_le_bytes());
            for row in history {
                write_checkpoint_vector(bytes, row);
            }
        }
    }
}

struct CheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> CheckpointReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| newmark_checkpoint_error("packet offset overflow"))?;
        if end > self.bytes.len() {
            return Err(newmark_checkpoint_error("truncated packet"));
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16, SolverError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("length checked"),
        ))
    }

    fn u64(&mut self) -> Result<u64, SolverError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("length checked"),
        ))
    }

    fn f64(&mut self) -> Result<f64, SolverError> {
        Ok(f64::from_le_bytes(
            self.take(8)?.try_into().expect("length checked"),
        ))
    }

    fn vector(&mut self) -> Result<Vec<f64>, SolverError> {
        let length = usize::try_from(self.u64()?)
            .map_err(|_| newmark_checkpoint_error("vector length overflows usize"))?;
        if length > NEWMARK_CHECKPOINT_MAX_ELEMENTS || length > self.bytes.len() / 8 {
            return Err(newmark_checkpoint_error("vector length is unreasonable"));
        }
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            values.push(self.f64()?);
        }
        Ok(values)
    }

    fn history(&mut self) -> Result<Option<Vec<Vec<f64>>>, SolverError> {
        match self.take(1)?[0] {
            0 => Ok(None),
            1 => {
                let rows = usize::try_from(self.u64()?)
                    .map_err(|_| newmark_checkpoint_error("history length overflows usize"))?;
                if rows > NEWMARK_CHECKPOINT_MAX_ELEMENTS {
                    return Err(newmark_checkpoint_error("history length is unreasonable"));
                }
                let mut history = Vec::with_capacity(rows);
                for _ in 0..rows {
                    history.push(self.vector()?);
                }
                Ok(Some(history))
            }
            _ => Err(newmark_checkpoint_error("invalid history presence flag")),
        }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

impl<T: ComplexField> SolverResultContract for NewmarkResult<T> {
    fn solver_status(&self) -> SolverStatus {
        NewmarkResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        NewmarkResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if NewmarkResult::solver_status(self) == SolverStatus::NumericalError {
            return Err(SolverError::OperationFailed {
                operation: "Newmark result contract",
                reason: "trajectory is non-finite or dimensionally inconsistent".into(),
            });
        }
        Ok(())
    }
}

/// Initial conditions for Newmark integration, generic over the scalar type.
#[derive(Debug, Clone)]
pub struct NewmarkInitial<T: ComplexField> {
    /// Initial displacement u(0).
    pub displacement: Vec<T>,
    /// Initial velocity u'(0).
    pub velocity: Vec<T>,
    /// Initial acceleration u''(0). Typically computed as `M^{-1}(f(0) - C*v0 - K*u0)`.
    pub acceleration: Vec<T>,
}

pub(crate) fn validate_initial_state<T: ComplexField>(
    initial: &NewmarkInitial<T>,
) -> Result<(), SolverError> {
    let n = initial.displacement.len();
    if initial.velocity.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: initial.velocity.len(),
        });
    }
    if initial.acceleration.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: initial.acceleration.len(),
        });
    }
    for (role, values) in [
        ("initial displacement", &initial.displacement),
        ("initial velocity", &initial.velocity),
        ("initial acceleration", &initial.acceleration),
    ] {
        if let Some(index) = values
            .iter()
            .position(|value| !value.re().is_finite() || !value.im().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "time integration initial state",
                reason: format!("{role} is non-finite at index {index}"),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_result<T: ComplexField>(
    result: &NewmarkResult<T>,
    expected_steps: usize,
    operation: &'static str,
) -> Result<(), SolverError> {
    let expected_history_len =
        expected_steps
            .checked_add(1)
            .ok_or_else(|| SolverError::OperationFailed {
                operation,
                reason: "history length overflows usize".to_string(),
            })?;
    if result.steps_completed != expected_steps {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "reported {} steps, expected {expected_steps}",
                result.steps_completed
            ),
        });
    }
    for (role, history) in [
        ("displacement", &result.displacement_history),
        ("velocity", &result.velocity_history),
        ("acceleration", &result.acceleration_history),
    ] {
        if let Some(history) = history
            && history.len() != expected_history_len
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "{role} history has {} states, expected {expected_history_len}",
                    history.len()
                ),
            });
        }
    }
    if result.solver_status() != SolverStatus::Converged {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "integration returned a non-finite or dimensionally invalid trajectory"
                .to_string(),
        });
    }
    Ok(())
}

/// Generalized-alpha integration parameters (Chung & Hulbert, 1993).
///
/// The residual is evaluated at the intermediate point with
/// `alpha_f`-weighted displacement/velocity/force and `alpha_m`-weighted
/// acceleration; `beta`/`gamma` play their usual Newmark role in the
/// kinematic update. See the module docs for the full scheme.
#[derive(Debug, Clone)]
pub struct GeneralizedAlphaConfig {
    /// Acceleration weighting parameter `alpha_m`.
    pub alpha_m: f64,
    /// Displacement/velocity/force weighting parameter `alpha_f`.
    pub alpha_f: f64,
    /// Newmark beta parameter (controls displacement update).
    pub beta: f64,
    /// Newmark gamma parameter (controls velocity update).
    pub gamma: f64,
    /// Time step size.
    pub dt: f64,
    /// Number of time steps.
    pub num_steps: usize,
    /// Store full history (all time steps) or only final state.
    pub store_history: bool,
}

impl GeneralizedAlphaConfig {
    /// Validate generalized-alpha coefficients before integration.
    pub fn validate(&self) -> Result<(), SolverError> {
        for (name, value) in [
            ("alpha_m", self.alpha_m),
            ("alpha_f", self.alpha_f),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("dt", self.dt),
        ] {
            if !value.is_finite() {
                return Err(SolverError::OperationFailed {
                    operation: "generalized-alpha configuration",
                    reason: format!("{name} must be finite"),
                });
            }
        }
        if self.dt <= 0.0 || self.beta <= 0.0 || self.gamma < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "generalized-alpha configuration",
                reason: "dt and beta must be positive; gamma must be non-negative".to_string(),
            });
        }
        if self.num_steps.checked_add(1).is_none() || !(self.num_steps as f64 * self.dt).is_finite()
        {
            return Err(SolverError::OperationFailed {
                operation: "generalized-alpha configuration",
                reason: "num_steps and final time overflow".to_string(),
            });
        }
        Ok(())
    }

    /// Build second-order, unconditionally stable coefficients from the
    /// spectral radius at infinity `rho_inf` in `[0, 1]`.
    ///
    /// - `rho_inf = 1`: no algorithmic dissipation (average acceleration).
    /// - `rho_inf = 0`: maximal dissipation (high-frequency modes annihilated
    ///   in one step).
    ///
    /// Uses the standard formulas (Chung & Hulbert, 1993):
    /// `alpha_m = (2*rho_inf - 1)/(rho_inf + 1)`,
    /// `alpha_f = rho_inf/(rho_inf + 1)`,
    /// `gamma = 1/2 - alpha_m + alpha_f`,
    /// `beta = (1 - alpha_m + alpha_f)^2/4`.
    /// Build a bounded-checkpoint schedule for a generalized-alpha adjoint.
    pub fn checkpoint_schedule(
        &self,
        max_checkpoints: usize,
    ) -> Result<RevolveSchedule, SolverError> {
        RevolveSchedule::new(self.num_steps, max_checkpoints).map_err(|error| {
            SolverError::OperationFailed {
                operation: "generalized-alpha checkpoint schedule",
                reason: error.to_string(),
            }
        })
    }

    pub fn from_spectral_radius(rho_inf: f64, dt: f64, num_steps: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&rho_inf),
            "rho_inf must be in [0, 1], got {rho_inf}"
        );
        let alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0);
        let alpha_f = rho_inf / (rho_inf + 1.0);
        let gamma = 0.5 - alpha_m + alpha_f;
        let beta = 0.25 * (gamma + 0.5).powi(2);
        Self {
            alpha_m,
            alpha_f,
            beta,
            gamma,
            dt,
            num_steps,
            store_history: false,
        }
    }

    /// True HHT-alpha method: generalized-alpha with `alpha_m = 0`,
    /// `alpha_f = alpha`, where `alpha` in `[0, 1/3]` controls the numerical
    /// dissipation (larger = more dissipation; `alpha = 0` is average
    /// acceleration). The spectral radius at infinity is
    /// `rho_inf = (1 - alpha)/(1 + alpha)`.
    ///
    /// Note this uses the convention `alpha >= 0`; it differs from
    /// [`NewmarkConfig::hht_alpha`], which only sets Newmark parameters
    /// without alpha-weighting the residual.
    pub fn hht_alpha(alpha: f64, dt: f64, num_steps: usize) -> Self {
        assert!(
            (0.0..=1.0 / 3.0).contains(&alpha),
            "alpha must be in [0, 1/3], got {alpha}"
        );
        let alpha_m = 0.0;
        let alpha_f = alpha;
        let gamma = 0.5 - alpha_m + alpha_f;
        let beta = 0.25 * (gamma + 0.5).powi(2);
        Self {
            alpha_m,
            alpha_f,
            beta,
            gamma,
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
}

/// Integrate `M u'' + C u' + K u = f(t)` using the Newmark-beta method.
///
/// Generic over any [`ComplexField`] type: works with f32, f64, Complex32,
/// and Complex64.
///
/// # Arguments
///
/// - `damping`: damping matrix C (LinearOperator), or `None` for undamped
/// - `stiffness`: stiffness matrix K (LinearOperator)
/// - `force_fn`: time-dependent force `f(t)` returning an Array1
/// - `initial`: initial displacement, velocity, and acceleration
/// - `linear_solve`: closure that solves `A_eff x = b` for x, where
///   `A_eff = M + gamma*dt*C + beta*dt^2*K` is the effective Newmark matrix.
///   The caller pre-builds and factorizes this matrix.
/// - `config`: Newmark parameters
pub fn newmark_integrate<T, C, K, F, S>(
    damping: Option<&C>,
    stiffness: &K,
    force_fn: F,
    initial: &NewmarkInitial<T>,
    linear_solve: S,
    config: &NewmarkConfig,
) -> NewmarkResult<T>
where
    T: ComplexField,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    F: Fn(f64) -> Vec<T>,
    S: Fn(&[T]) -> Vec<T>,
{
    let u0 = &initial.displacement;
    let v0 = &initial.velocity;
    let a0 = &initial.acceleration;
    let dt = config.dt;
    let beta = config.beta;
    let gamma = config.gamma;
    let dt2 = dt * dt;

    let dt_t = scalar_from_f64_or_nan::<T>(dt);
    let half_minus_beta_dt2 = scalar_from_f64_or_nan::<T>((0.5 - beta) * dt2);
    let beta_dt2 = scalar_from_f64_or_nan::<T>(beta * dt2);
    let one_minus_gamma_dt = scalar_from_f64_or_nan::<T>((1.0 - gamma) * dt);
    let gamma_dt = scalar_from_f64_or_nan::<T>(gamma * dt);

    let mut u = u0.clone();
    let mut v = v0.clone();
    let mut a = a0.clone();

    let n = u.len();
    // Pre-allocate predictor buffers (reused across time steps, zero allocation in loop)
    let mut u_pred = vec![T::zero(); n];
    let mut v_pred = vec![T::zero(); n];
    // RHS assembly scratch: `op_buf` holds each operator action in turn
    // (`apply_into`, no per-step temporaries) and `rhs_buf` the assembled
    // right-hand side. Same operations in the same order as the previous
    // `vec_sub(force, stiffness.apply(..))` formulation, so bitwise identical.
    let mut op_buf = vec![T::zero(); n];
    let mut rhs_buf = vec![T::zero(); n];

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

        // Predictor (zero allocation via pre-allocated buffers)
        // u_pred = u + dt*v + (0.5-beta)*dt^2*a
        u_pred.copy_from_slice(&u);
        T::vec_axpy(dt_t, &v, &mut u_pred);
        T::vec_axpy(half_minus_beta_dt2, &a, &mut u_pred);
        // v_pred = v + (1-gamma)*dt*a
        v_pred.copy_from_slice(&v);
        T::vec_axpy(one_minus_gamma_dt, &a, &mut v_pred);

        // RHS: f_{n+1} - C*v_pred - K*u_pred
        let f_next = force_fn(t_next);
        stiffness.apply_into(&u_pred, &mut op_buf);
        rhs_buf.copy_from_slice(&f_next);
        sonium_solvers_core::vec_ops::vec_sub_assign(&mut rhs_buf, &op_buf);
        if let Some(c) = damping {
            c.apply_into(&v_pred, &mut op_buf);
            sonium_solvers_core::vec_ops::vec_sub_assign(&mut rhs_buf, &op_buf);
        }

        // Solve effective system for a_{n+1}
        let a_new = linear_solve(&rhs_buf);

        // Corrector (zero allocation: reuse u_pred/v_pred as u/v)
        u.copy_from_slice(&u_pred);
        T::vec_axpy(beta_dt2, &a_new, &mut u);
        v.copy_from_slice(&v_pred);
        T::vec_axpy(gamma_dt, &a_new, &mut v);
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

/// Integrate `M u'' + C u' + K u = f(t)` using the generalized-alpha method
/// Checked, panic-contained Newmark integration entry point.
#[allow(clippy::too_many_arguments)]
pub fn try_newmark_integrate<T, C, K, F, S>(
    damping: Option<&C>,
    stiffness: &K,
    force_fn: F,
    initial: &NewmarkInitial<T>,
    linear_solve: S,
    config: &NewmarkConfig,
) -> Result<NewmarkResult<T>, SolverError>
where
    T: ComplexField,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    F: Fn(f64) -> Vec<T>,
    S: Fn(&[T]) -> Vec<T>,
{
    config.validate()?;
    validate_initial_state(initial)?;
    let dt2 = config.dt * config.dt;
    validate_scalar_coefficients::<T>(
        "Newmark coefficient conversion",
        &[
            ("dt", config.dt),
            ("(0.5 - beta) * dt^2", (0.5 - config.beta) * dt2),
            ("beta * dt^2", config.beta * dt2),
            ("(1.0 - gamma) * dt", (1.0 - config.gamma) * config.dt),
            ("gamma * dt", config.gamma * config.dt),
        ],
    )?;
    let n = initial.displacement.len();
    if stiffness.num_rows() != n || stiffness.num_cols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: stiffness.num_rows().max(stiffness.num_cols()),
        });
    }
    if let Some(damping) = damping
        && (damping.num_rows() != n || damping.num_cols() != n)
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: damping.num_rows().max(damping.num_cols()),
        });
    }
    let callback_error = RefCell::new(None);
    let dimension = initial.displacement.len();
    let checked_force = |time: f64| {
        let values = force_fn(time);
        if let Err(error) = validate_callback_vector(
            &values,
            dimension,
            "Newmark integration force callback",
            "force output",
        ) {
            store_callback_error(&callback_error, error);
        }
        values
    };
    let checked_solve = |rhs: &[T]| {
        if let Err(error) = validate_callback_vector(
            rhs,
            dimension,
            "Newmark integration linear-solve callback",
            "linear-solve input",
        ) {
            store_callback_error(&callback_error, error);
        }
        let values = linear_solve(rhs);
        if let Err(error) = validate_callback_vector(
            &values,
            dimension,
            "Newmark integration linear-solve callback",
            "linear-solve output",
        ) {
            store_callback_error(&callback_error, error);
        }
        values
    };
    let result = match catch_unwind(AssertUnwindSafe(|| {
        newmark_integrate(
            damping,
            stiffness,
            checked_force,
            initial,
            checked_solve,
            config,
        )
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
                operation: "Newmark integration",
                reason: panic_reason(payload),
            });
        }
    };
    validate_result(&result, config.num_steps, "Newmark integration")?;
    Ok(result)
}

/// Checked, panic-contained generalized-alpha integration entry point.
#[allow(clippy::too_many_arguments)]
pub fn try_generalized_alpha_integrate<T, M, C, K, F, S>(
    mass: &M,
    damping: Option<&C>,
    stiffness: &K,
    force_fn: F,
    initial: &NewmarkInitial<T>,
    linear_solve: S,
    config: &GeneralizedAlphaConfig,
) -> Result<NewmarkResult<T>, SolverError>
where
    T: ComplexField,
    M: LinearOperator<T>,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    F: Fn(f64) -> Vec<T>,
    S: Fn(&[T]) -> Vec<T>,
{
    config.validate()?;
    validate_initial_state(initial)?;
    let dt2 = config.dt * config.dt;
    validate_scalar_coefficients::<T>(
        "generalized-alpha coefficient conversion",
        &[
            ("dt", config.dt),
            ("(0.5 - beta) * dt^2", (0.5 - config.beta) * dt2),
            ("beta * dt^2", config.beta * dt2),
            ("(1.0 - gamma) * dt", (1.0 - config.gamma) * config.dt),
            ("gamma * dt", config.gamma * config.dt),
            ("(1.0 - alpha_f) * dt", (1.0 - config.alpha_f) * config.dt),
            (
                "(1.0 - alpha_f) * (0.5 - beta) * dt^2",
                (1.0 - config.alpha_f) * (0.5 - config.beta) * dt2,
            ),
            (
                "(1.0 - alpha_f) * (1.0 - gamma) * dt",
                (1.0 - config.alpha_f) * (1.0 - config.gamma) * config.dt,
            ),
            ("alpha_m", config.alpha_m),
        ],
    )?;
    let n = initial.displacement.len();
    if mass.num_rows() != n || mass.num_cols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: mass.num_rows().max(mass.num_cols()),
        });
    }
    if stiffness.num_rows() != n || stiffness.num_cols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: stiffness.num_rows().max(stiffness.num_cols()),
        });
    }
    if let Some(damping) = damping
        && (damping.num_rows() != n || damping.num_cols() != n)
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: damping.num_rows().max(damping.num_cols()),
        });
    }
    let callback_error = RefCell::new(None);
    let dimension = initial.displacement.len();
    let checked_force = |time: f64| {
        let values = force_fn(time);
        if let Err(error) = validate_callback_vector(
            &values,
            dimension,
            "generalized-alpha integration force callback",
            "force output",
        ) {
            store_callback_error(&callback_error, error);
        }
        values
    };
    let checked_solve = |rhs: &[T]| {
        if let Err(error) = validate_callback_vector(
            rhs,
            dimension,
            "generalized-alpha integration linear-solve callback",
            "linear-solve input",
        ) {
            store_callback_error(&callback_error, error);
        }
        let values = linear_solve(rhs);
        if let Err(error) = validate_callback_vector(
            &values,
            dimension,
            "generalized-alpha integration linear-solve callback",
            "linear-solve output",
        ) {
            store_callback_error(&callback_error, error);
        }
        values
    };
    let result = match catch_unwind(AssertUnwindSafe(|| {
        generalized_alpha_integrate(
            mass,
            damping,
            stiffness,
            checked_force,
            initial,
            checked_solve,
            config,
        )
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
                operation: "generalized-alpha integration",
                reason: panic_reason(payload),
            });
        }
    };
    validate_result(&result, config.num_steps, "generalized-alpha integration")?;
    Ok(result)
}

/// Reverse-mode sensitivities for a stored Newmark trajectory.
#[derive(Debug, Clone)]
pub struct NewmarkAdjointResult<T: ComplexField> {
    /// Cotangent of each force sample `f_{n+1}`.
    pub force_gradient: Vec<Vec<T>>,
    /// Cotangent of the initial displacement.
    pub initial_displacement_gradient: Vec<T>,
    /// Cotangent of the initial velocity.
    pub initial_velocity_gradient: Vec<T>,
    /// Cotangent of the initial acceleration.
    pub initial_acceleration_gradient: Vec<T>,
}

impl<T: ComplexField> NewmarkAdjointResult<T> {
    /// Return whether every adjoint gradient entry is finite.
    pub fn is_finite(&self) -> bool {
        self.force_gradient
            .iter()
            .flat_map(|gradient| gradient.iter())
            .chain(self.initial_displacement_gradient.iter())
            .chain(self.initial_velocity_gradient.iter())
            .chain(self.initial_acceleration_gradient.iter())
            .all(|value| value.re().is_finite() && value.im().is_finite())
    }

    /// Return the canonical status for the completed adjoint recurrence.
    ///
    /// The adjoint driver has no iterative stopping tolerance of its own;
    /// finite output therefore means that the requested recurrence completed,
    /// while non-finite output is a numerical failure.
    pub fn solver_status(&self) -> SolverStatus {
        if self.is_finite() {
            SolverStatus::Converged
        } else {
            SolverStatus::NumericalError
        }
    }

    /// Alias matching the checked result vocabulary used by other families.
    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    /// Adjoint sensitivities do not carry an independent residual tolerance.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::NotChecked,
            _ => AccuracyStatus::NotAttained,
        }
    }

    /// Alias matching the checked result vocabulary used by other families.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: ComplexField> SolverResultContract for NewmarkAdjointResult<T> {
    fn solver_status(&self) -> SolverStatus {
        NewmarkAdjointResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        NewmarkAdjointResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if self.is_finite() {
            Ok(())
        } else {
            Err(SolverError::OperationFailed {
                operation: "Newmark adjoint result contract",
                reason: "adjoint gradient contains a non-finite value".into(),
            })
        }
    }
}

/// The generalized-alpha reverse pass has the same cotangent layout as the
/// Newmark reverse pass. The alias keeps transient clients independent of the
/// particular second-order time integrator they selected.
pub type GeneralizedAlphaAdjointResult<T> = NewmarkAdjointResult<T>;

/// Run the exact reverse recurrence of [`newmark_integrate`].
#[allow(clippy::too_many_arguments)]
pub fn newmark_adjoint<T, C, K, S>(
    damping: Option<&C>,
    stiffness: &K,
    forward: &NewmarkResult<T>,
    terminal_displacement: &[T],
    terminal_velocity: &[T],
    terminal_acceleration: &[T],
    adjoint_solve: S,
    config: &NewmarkConfig,
) -> Result<NewmarkAdjointResult<T>, SolverError>
where
    T: ComplexField,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    S: Fn(&[T]) -> Result<Vec<T>, SolverError>,
{
    let displacement_history =
        forward
            .displacement_history
            .as_ref()
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "Newmark adjoint",
                reason: "forward displacement history was not stored".to_string(),
            })?;
    let velocity_history =
        forward
            .velocity_history
            .as_ref()
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "Newmark adjoint",
                reason: "forward velocity history was not stored".to_string(),
            })?;
    let acceleration_history =
        forward
            .acceleration_history
            .as_ref()
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "Newmark adjoint",
                reason: "forward acceleration history was not stored".to_string(),
            })?;
    let steps = config.num_steps;
    let n = terminal_displacement.len();
    if terminal_velocity.len() != n || terminal_acceleration.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: terminal_velocity.len().max(terminal_acceleration.len()),
        });
    }
    if forward.steps_completed != steps
        || displacement_history.len() != steps + 1
        || velocity_history.len() != steps + 1
        || acceleration_history.len() != steps + 1
    {
        return Err(SolverError::OperationFailed {
            operation: "Newmark adjoint",
            reason: "forward history length does not match configuration".to_string(),
        });
    }
    // Validate-once: Hermitian capability and square dimensions are admitted
    // here so the reverse loop below runs the raw action (finiteness is
    // validated once over the finished outputs).
    admit_adjoint_hermitian(stiffness, n)?;
    if let Some(damping) = damping {
        admit_adjoint_hermitian(damping, n)?;
    }

    let real = |value: f64| {
        T::Real::from_f64(value)
            .map(T::from_real)
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "Newmark adjoint",
                reason: format!("coefficient {value} cannot be represented"),
            })
    };
    let dt = config.dt;
    let dt_t = real(dt)?;
    let half_minus_beta_dt2 = real((0.5 - config.beta) * dt * dt)?;
    let beta_dt2 = real(config.beta * dt * dt)?;
    let one_minus_gamma_dt = real((1.0 - config.gamma) * dt)?;
    let gamma_dt = real(config.gamma * dt)?;
    let zero = T::zero();
    let mut lambda_u = terminal_displacement.to_vec();
    let mut lambda_v = terminal_velocity.to_vec();
    let mut lambda_a = terminal_acceleration.to_vec();
    let mut force_gradient = vec![vec![zero; n]; steps];

    for step in (0..steps).rev() {
        let mut lambda_a_next = lambda_a.clone();
        T::vec_axpy(beta_dt2, &lambda_u, &mut lambda_a_next);
        T::vec_axpy(gamma_dt, &lambda_v, &mut lambda_a_next);
        let lambda_rhs = catch_unwind(AssertUnwindSafe(|| adjoint_solve(&lambda_a_next)))
            .map_err(|payload| SolverError::OperationFailed {
                operation: "Newmark adjoint solve",
                reason: panic_reason(payload),
            })??;
        if lambda_rhs.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: lambda_rhs.len(),
            });
        }
        force_gradient[step] = lambda_rhs.clone();

        // Pullback through the corrector `u' = u_pred + beta*dt^2*a'`,
        // `v' = v_pred + gamma*dt*a'`: the predicted-state cotangents keep
        // the direct seeds `lambda_u` / `lambda_v` and subtract the
        // stiffness/damping contributions arriving via the solve.
        // (Seeding with `-lambda_rhs` instead loses the direct corrector
        // dependence and zeroes every multi-step gradient.)
        let mut lambda_u_pred = lambda_u.clone();
        let mut lambda_v_pred = lambda_v.clone();
        if let Some(damping) = damping {
            let damping_adjoint = apply_adjoint_hermitian(damping, &lambda_rhs, n)?;
            for (target, contribution) in lambda_v_pred.iter_mut().zip(damping_adjoint) {
                *target -= contribution;
            }
        }
        let stiffness_adjoint = apply_adjoint_hermitian(stiffness, &lambda_rhs, n)?;
        for (target, contribution) in lambda_u_pred.iter_mut().zip(stiffness_adjoint) {
            *target -= contribution;
        }

        let mut lambda_u_previous = lambda_u_pred.clone();
        let mut lambda_v_previous = lambda_v_pred.clone();
        let mut lambda_a_previous = vec![zero; n];
        T::vec_axpy(dt_t, &lambda_u_pred, &mut lambda_v_previous);
        T::vec_axpy(half_minus_beta_dt2, &lambda_u_pred, &mut lambda_a_previous);
        T::vec_axpy(one_minus_gamma_dt, &lambda_v_pred, &mut lambda_a_previous);
        lambda_u = std::mem::take(&mut lambda_u_previous);
        lambda_v = std::mem::take(&mut lambda_v_previous);
        lambda_a = std::mem::take(&mut lambda_a_previous);
    }

    validate_adjoint_outputs(
        "Newmark adjoint",
        &force_gradient,
        &lambda_u,
        &lambda_v,
        &lambda_a,
    )?;
    Ok(NewmarkAdjointResult {
        force_gradient,
        initial_displacement_gradient: lambda_u,
        initial_velocity_gradient: lambda_v,
        initial_acceleration_gradient: lambda_a,
    })
}

/// Default planning budget for checkpointed adjoint storage (256 MiB).
/// This excludes caller-owned operators, callbacks and allocator overhead.
pub const DEFAULT_CHECKPOINTED_ADJOINT_MEMORY_BYTES: usize = 256 * 1024 * 1024;

/// Conservative peak-storage estimate for materialized force gradients,
/// forward checkpoint states, reverse vectors, and the retained schedule.
/// Saturation represents an inadmissible size, never a wrapped small size.
pub fn checkpointed_adjoint_memory_bytes<T>(
    dofs: usize,
    num_steps: usize,
    max_checkpoints: usize,
) -> u128 {
    let steps = num_steps as u128;
    let slots = max_checkpoints.min(num_steps) as u128;
    let vector_bytes = (dofs as u128).saturating_mul(std::mem::size_of::<T>() as u128);
    let gradient =
        steps.saturating_mul(vector_bytes.saturating_add(std::mem::size_of::<Vec<T>>() as u128));
    // Binary-split schedules have <7 actions per step. Account for Vec
    // growth capacity and the validation reverse-step set as well.
    let schedule =
        steps.saturating_mul(16 * std::mem::size_of::<crate::RevolveAction>() as u128 + 128);
    gradient
        .saturating_add(vector_bytes.saturating_mul(slots.saturating_mul(3).saturating_add(24)))
        .saturating_add(slots.saturating_mul(256))
        .saturating_add(schedule)
}

fn admit_checkpointed_adjoint<T>(
    dofs: usize,
    steps: usize,
    checkpoints: usize,
    budget: usize,
) -> Result<(), SolverError> {
    let required = checkpointed_adjoint_memory_bytes::<T>(dofs, steps, checkpoints);
    if required > budget as u128 {
        return Err(SolverError::OperationFailed {
            operation: "checkpointed adjoint memory admission",
            reason: format!("estimated storage {required} bytes exceeds budget {budget} bytes"),
        });
    }
    Ok(())
}

/// Run a Newmark reverse pass with bounded forward checkpoints.
///
/// `advance_step` recomputes one forward step and receives the zero-based
/// step index and state at its beginning. `adjoint_solve` receives the
/// reverse step index, the restored forward state at the end of that step,
/// and the Hermitian effective-system right-hand side. This lets nonlinear
/// or state-dependent callers rebuild the same operator during recomputation
/// without retaining the complete forward trajectory.
///
/// The returned force gradient is still materialized for every step; only
/// the forward state storage is bounded by `max_checkpoints`.
/// Total planned storage is admitted against
/// [`DEFAULT_CHECKPOINTED_ADJOINT_MEMORY_BYTES`] before schedule construction
/// or callbacks. Use [`newmark_adjoint_checkpointed_with_memory_budget`] to
/// explicitly admit a different storage budget.
#[allow(clippy::too_many_arguments)]
pub fn newmark_adjoint_checkpointed<T, C, K, F, S>(
    damping: Option<&C>,
    stiffness: &K,
    initial_state: NewmarkState<T>,
    terminal_displacement: &[T],
    terminal_velocity: &[T],
    terminal_acceleration: &[T],
    advance_step: F,
    adjoint_solve: S,
    config: &NewmarkConfig,
    max_checkpoints: usize,
) -> Result<NewmarkAdjointResult<T>, SolverError>
where
    T: ComplexField,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    F: FnMut(usize, &NewmarkState<T>) -> Result<NewmarkState<T>, SolverError>,
    S: Fn(usize, &NewmarkState<T>, &[T]) -> Result<Vec<T>, SolverError>,
{
    newmark_adjoint_checkpointed_with_memory_budget(
        damping,
        stiffness,
        initial_state,
        terminal_displacement,
        terminal_velocity,
        terminal_acceleration,
        advance_step,
        adjoint_solve,
        config,
        max_checkpoints,
        DEFAULT_CHECKPOINTED_ADJOINT_MEMORY_BYTES,
    )
}

/// [`newmark_adjoint_checkpointed`] with an explicit peak-storage planning
/// budget in bytes. Callback/operator allocations remain the caller's responsibility.
#[allow(clippy::too_many_arguments)]
pub fn newmark_adjoint_checkpointed_with_memory_budget<T, C, K, F, S>(
    damping: Option<&C>,
    stiffness: &K,
    initial_state: NewmarkState<T>,
    terminal_displacement: &[T],
    terminal_velocity: &[T],
    terminal_acceleration: &[T],
    mut advance_step: F,
    adjoint_solve: S,
    config: &NewmarkConfig,
    max_checkpoints: usize,
    memory_budget_bytes: usize,
) -> Result<NewmarkAdjointResult<T>, SolverError>
where
    T: ComplexField,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    F: FnMut(usize, &NewmarkState<T>) -> Result<NewmarkState<T>, SolverError>,
    S: Fn(usize, &NewmarkState<T>, &[T]) -> Result<Vec<T>, SolverError>,
{
    config.validate()?;
    admit_checkpointed_adjoint::<T>(
        terminal_displacement.len(),
        config.num_steps,
        max_checkpoints,
        memory_budget_bytes,
    )?;
    let schedule = config.checkpoint_schedule(max_checkpoints)?;
    let n = terminal_displacement.len();
    if terminal_velocity.len() != n || terminal_acceleration.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: terminal_velocity.len().max(terminal_acceleration.len()),
        });
    }
    if initial_state.displacement.len() != n
        || initial_state.velocity.len() != n
        || initial_state.acceleration.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: initial_state
                .displacement
                .len()
                .max(initial_state.velocity.len())
                .max(initial_state.acceleration.len()),
        });
    }
    // Validate-once: Hermitian capability and square dimensions are admitted
    // here so the reverse closure below runs the raw action (finiteness is
    // validated once over the finished outputs).
    admit_adjoint_hermitian(stiffness, n)?;
    if let Some(damping) = damping {
        admit_adjoint_hermitian(damping, n)?;
    }

    let real = |value: f64| {
        T::Real::from_f64(value)
            .map(T::from_real)
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "Newmark checkpointed adjoint",
                reason: format!("coefficient {value} cannot be represented"),
            })
    };
    let dt_t = real(config.dt)?;
    let half_minus_beta_dt2 = real((0.5 - config.beta) * config.dt * config.dt)?;
    let beta_dt2 = real(config.beta * config.dt * config.dt)?;
    let one_minus_gamma_dt = real((1.0 - config.gamma) * config.dt)?;
    let gamma_dt = real(config.gamma * config.dt)?;
    let zero = T::zero();
    let mut lambda_u = terminal_displacement.to_vec();
    let mut lambda_v = terminal_velocity.to_vec();
    let mut lambda_a = terminal_acceleration.to_vec();
    let mut force_gradient = vec![vec![zero; n]; config.num_steps];

    let advance = |from: usize,
                   to: usize,
                   mut state: NewmarkState<T>|
     -> Result<NewmarkState<T>, SolverError> {
        for step in from..to {
            state = advance_step(step, &state)?;
            if state.displacement.len() != n
                || state.velocity.len() != n
                || state.acceleration.len() != n
            {
                return Err(SolverError::DimensionMismatch {
                    expected: n,
                    got: state
                        .displacement
                        .len()
                        .max(state.velocity.len())
                        .max(state.acceleration.len()),
                });
            }
        }
        Ok(state)
    };

    let reverse = |step: usize, state: &NewmarkState<T>| -> Result<(), SolverError> {
        if state.displacement.len() != n
            || state.velocity.len() != n
            || state.acceleration.len() != n
        {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: state
                    .displacement
                    .len()
                    .max(state.velocity.len())
                    .max(state.acceleration.len()),
            });
        }
        let mut lambda_a_next = lambda_a.clone();
        T::vec_axpy(beta_dt2, &lambda_u, &mut lambda_a_next);
        T::vec_axpy(gamma_dt, &lambda_v, &mut lambda_a_next);
        let lambda_rhs = adjoint_solve(step, state, &lambda_a_next)?;
        if lambda_rhs.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: lambda_rhs.len(),
            });
        }
        force_gradient[step] = lambda_rhs.clone();

        // Pullback through the corrector `u' = u_pred + beta*dt^2*a'`,
        // `v' = v_pred + gamma*dt*a'`: the predicted-state cotangents keep
        // the direct seeds `lambda_u` / `lambda_v` and subtract the
        // stiffness/damping contributions arriving via the solve.
        // (Seeding with `-lambda_rhs` instead loses the direct corrector
        // dependence and zeroes every multi-step gradient.)
        let mut lambda_u_pred = lambda_u.clone();
        let mut lambda_v_pred = lambda_v.clone();
        if let Some(damping) = damping {
            let damping_adjoint = apply_adjoint_hermitian(damping, &lambda_rhs, n)?;
            for (target, contribution) in lambda_v_pred.iter_mut().zip(damping_adjoint) {
                *target -= contribution;
            }
        }
        let stiffness_adjoint = apply_adjoint_hermitian(stiffness, &lambda_rhs, n)?;
        for (target, contribution) in lambda_u_pred.iter_mut().zip(stiffness_adjoint) {
            *target -= contribution;
        }

        let mut lambda_u_previous = lambda_u_pred.clone();
        let mut lambda_v_previous = lambda_v_pred.clone();
        let mut lambda_a_previous = vec![zero; n];
        T::vec_axpy(dt_t, &lambda_u_pred, &mut lambda_v_previous);
        T::vec_axpy(half_minus_beta_dt2, &lambda_u_pred, &mut lambda_a_previous);
        T::vec_axpy(one_minus_gamma_dt, &lambda_v_pred, &mut lambda_a_previous);
        lambda_u = std::mem::take(&mut lambda_u_previous);
        lambda_v = std::mem::take(&mut lambda_v_previous);
        lambda_a = std::mem::take(&mut lambda_a_previous);
        Ok(())
    };

    schedule
        .execute(initial_state, advance, reverse)
        .map_err(|error| SolverError::OperationFailed {
            operation: "Newmark checkpointed adjoint",
            reason: error.to_string(),
        })?;

    validate_adjoint_outputs(
        "Newmark checkpointed adjoint",
        &force_gradient,
        &lambda_u,
        &lambda_v,
        &lambda_a,
    )?;
    Ok(NewmarkAdjointResult {
        force_gradient,
        initial_displacement_gradient: lambda_u,
        initial_velocity_gradient: lambda_v,
        initial_acceleration_gradient: lambda_a,
    })
}

/// Reverse-mode sensitivities for a stored generalized-alpha trajectory.
///
/// `adjoint_solve` must solve the Hermitian/transpose effective system used
/// by the forward acceleration solve. The recurrence includes both
/// alpha-weighted predictor states and the `alpha_m M a_n` residual term.
#[allow(clippy::too_many_arguments)]
pub fn generalized_alpha_adjoint<T, M, C, K, S>(
    mass: &M,
    damping: Option<&C>,
    stiffness: &K,
    forward: &NewmarkResult<T>,
    terminal_displacement: &[T],
    terminal_velocity: &[T],
    terminal_acceleration: &[T],
    adjoint_solve: S,
    config: &GeneralizedAlphaConfig,
) -> Result<GeneralizedAlphaAdjointResult<T>, SolverError>
where
    T: ComplexField,
    M: LinearOperator<T>,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    S: Fn(&[T]) -> Result<Vec<T>, SolverError>,
{
    config.validate()?;
    let displacement_history =
        forward
            .displacement_history
            .as_ref()
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "generalized-alpha adjoint",
                reason: "forward displacement history not stored".to_string(),
            })?;
    let velocity_history =
        forward
            .velocity_history
            .as_ref()
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "generalized-alpha adjoint",
                reason: "forward velocity history not stored".to_string(),
            })?;
    let acceleration_history =
        forward
            .acceleration_history
            .as_ref()
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "generalized-alpha adjoint",
                reason: "forward acceleration history not stored".to_string(),
            })?;
    let steps = config.num_steps;
    let n = terminal_displacement.len();
    if terminal_velocity.len() != n || terminal_acceleration.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: terminal_velocity.len().max(terminal_acceleration.len()),
        });
    }
    if forward.steps_completed != steps
        || displacement_history.len() != steps + 1
        || velocity_history.len() != steps + 1
        || acceleration_history.len() != steps + 1
    {
        return Err(SolverError::OperationFailed {
            operation: "generalized-alpha adjoint",
            reason: "forward history length does not match configuration".to_string(),
        });
    }
    // Validate-once: Hermitian capability and square dimensions are admitted
    // here so the reverse loop below runs the raw action (finiteness is
    // validated once over the finished outputs). Mass is admitted only when
    // its term is active, matching the conditional application below.
    admit_adjoint_hermitian(stiffness, n)?;
    if let Some(damping) = damping {
        admit_adjoint_hermitian(damping, n)?;
    }
    if config.alpha_m != 0.0 {
        admit_adjoint_hermitian(mass, n)?;
    }

    let real = |value: f64| {
        T::Real::from_f64(value)
            .map(T::from_real)
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "generalized-alpha adjoint",
                reason: format!("coefficient {value} cannot be represented"),
            })
    };
    let dt = config.dt;
    let dt_t = real(dt)?;
    let predictor_u_a = real((1.0 - config.alpha_f) * (0.5 - config.beta) * dt * dt)?;
    let predictor_u_v = real((1.0 - config.alpha_f) * dt)?;
    let predictor_v_a = real((1.0 - config.alpha_f) * (1.0 - config.gamma) * dt)?;
    let corrector_u_a = real((0.5 - config.beta) * dt * dt)?;
    let corrector_u_new = real(config.beta * dt * dt)?;
    let corrector_v_a = real((1.0 - config.gamma) * dt)?;
    let corrector_v_new = real(config.gamma * dt)?;
    let alpha_m = real(config.alpha_m)?;
    let zero = T::zero();
    let mut lambda_u = terminal_displacement.to_vec();
    let mut lambda_v = terminal_velocity.to_vec();
    let mut lambda_a = terminal_acceleration.to_vec();
    let mut force_gradient = vec![vec![zero; n]; steps];

    for step in (0..steps).rev() {
        // a_{n+1} contributes to displacement, velocity, and the explicit
        // acceleration state returned by the integrator.
        let mut lambda_a_new = lambda_a.clone();
        T::vec_axpy(corrector_u_new, &lambda_u, &mut lambda_a_new);
        T::vec_axpy(corrector_v_new, &lambda_v, &mut lambda_a_new);

        let lambda_rhs = catch_unwind(AssertUnwindSafe(|| adjoint_solve(&lambda_a_new))).map_err(
            |payload| SolverError::OperationFailed {
                operation: "generalized-alpha adjoint solve",
                reason: panic_reason(payload),
            },
        )??;
        if lambda_rhs.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: lambda_rhs.len(),
            });
        }
        force_gradient[step] = lambda_rhs.clone();

        // Solve-mediated predictor pullback only: the direct corrector
        // seeds are added to `lambda_u_previous` below, so seeding
        // `-lambda_rhs` here would corrupt every multi-step gradient.
        let mut lambda_u_pred = vec![zero; n];
        let mut lambda_v_pred = vec![zero; n];
        if let Some(damping) = damping {
            let contribution = apply_adjoint_hermitian(damping, &lambda_rhs, n)?;
            for (target, value) in lambda_v_pred.iter_mut().zip(contribution) {
                *target -= value;
            }
        }
        let stiffness_contribution = apply_adjoint_hermitian(stiffness, &lambda_rhs, n)?;
        for (target, value) in lambda_u_pred.iter_mut().zip(stiffness_contribution) {
            *target -= value;
        }

        let mut lambda_u_previous = lambda_u.clone();
        let mut lambda_v_previous = lambda_v.clone();
        let mut lambda_a_previous = vec![zero; n];
        T::vec_axpy(dt_t, &lambda_u, &mut lambda_v_previous);
        T::vec_axpy(corrector_u_a, &lambda_u, &mut lambda_a_previous);
        T::vec_axpy(corrector_v_a, &lambda_v, &mut lambda_a_previous);
        T::vec_axpy(predictor_u_v, &lambda_u_pred, &mut lambda_v_previous);
        T::vec_axpy(predictor_u_a, &lambda_u_pred, &mut lambda_a_previous);
        T::vec_axpy(predictor_v_a, &lambda_v_pred, &mut lambda_a_previous);
        for (target, value) in lambda_u_previous.iter_mut().zip(lambda_u_pred) {
            *target += value;
        }

        // The mass contribution is -alpha_m M a_n in the forward residual.
        if config.alpha_m != 0.0 {
            let mass_contribution = apply_adjoint_hermitian(mass, &lambda_rhs, n)?;
            T::vec_axpy(-alpha_m, &mass_contribution, &mut lambda_a_previous);
        }

        lambda_u = std::mem::take(&mut lambda_u_previous);
        lambda_v = std::mem::take(&mut lambda_v_previous);
        lambda_a = std::mem::take(&mut lambda_a_previous);
    }

    validate_adjoint_outputs(
        "generalized-alpha adjoint",
        &force_gradient,
        &lambda_u,
        &lambda_v,
        &lambda_a,
    )?;
    Ok(NewmarkAdjointResult {
        force_gradient,
        initial_displacement_gradient: lambda_u,
        initial_velocity_gradient: lambda_v,
        initial_acceleration_gradient: lambda_a,
    })
}

/// Run a generalized-alpha reverse pass with bounded forward checkpoints.
///
/// The callbacks have the same state-recomputation contract as
/// [`newmark_adjoint_checkpointed`], while the reverse recurrence includes
/// the generalized-alpha mass, damping, and stiffness terms.
/// Materialized output and schedule storage are subject to the same default
/// memory budget; use [`generalized_alpha_adjoint_checkpointed_with_memory_budget`]
/// to explicitly admit another budget.
#[allow(clippy::too_many_arguments)]
pub fn generalized_alpha_adjoint_checkpointed<T, M, C, K, F, S>(
    mass: &M,
    damping: Option<&C>,
    stiffness: &K,
    initial_state: NewmarkState<T>,
    terminal_displacement: &[T],
    terminal_velocity: &[T],
    terminal_acceleration: &[T],
    advance_step: F,
    adjoint_solve: S,
    config: &GeneralizedAlphaConfig,
    max_checkpoints: usize,
) -> Result<GeneralizedAlphaAdjointResult<T>, SolverError>
where
    T: ComplexField,
    M: LinearOperator<T>,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    F: FnMut(usize, &NewmarkState<T>) -> Result<NewmarkState<T>, SolverError>,
    S: Fn(usize, &NewmarkState<T>, &[T]) -> Result<Vec<T>, SolverError>,
{
    generalized_alpha_adjoint_checkpointed_with_memory_budget(
        mass,
        damping,
        stiffness,
        initial_state,
        terminal_displacement,
        terminal_velocity,
        terminal_acceleration,
        advance_step,
        adjoint_solve,
        config,
        max_checkpoints,
        DEFAULT_CHECKPOINTED_ADJOINT_MEMORY_BYTES,
    )
}

/// [`generalized_alpha_adjoint_checkpointed`] with an explicit peak-storage
/// planning budget. Callback/operator allocations are excluded.
#[allow(clippy::too_many_arguments)]
pub fn generalized_alpha_adjoint_checkpointed_with_memory_budget<T, M, C, K, F, S>(
    mass: &M,
    damping: Option<&C>,
    stiffness: &K,
    initial_state: NewmarkState<T>,
    terminal_displacement: &[T],
    terminal_velocity: &[T],
    terminal_acceleration: &[T],
    mut advance_step: F,
    adjoint_solve: S,
    config: &GeneralizedAlphaConfig,
    max_checkpoints: usize,
    memory_budget_bytes: usize,
) -> Result<GeneralizedAlphaAdjointResult<T>, SolverError>
where
    T: ComplexField,
    M: LinearOperator<T>,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    F: FnMut(usize, &NewmarkState<T>) -> Result<NewmarkState<T>, SolverError>,
    S: Fn(usize, &NewmarkState<T>, &[T]) -> Result<Vec<T>, SolverError>,
{
    config.validate()?;
    admit_checkpointed_adjoint::<T>(
        terminal_displacement.len(),
        config.num_steps,
        max_checkpoints,
        memory_budget_bytes,
    )?;
    let schedule = config.checkpoint_schedule(max_checkpoints)?;
    let n = terminal_displacement.len();
    if terminal_velocity.len() != n || terminal_acceleration.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: terminal_velocity.len().max(terminal_acceleration.len()),
        });
    }
    if initial_state.displacement.len() != n
        || initial_state.velocity.len() != n
        || initial_state.acceleration.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: initial_state
                .displacement
                .len()
                .max(initial_state.velocity.len())
                .max(initial_state.acceleration.len()),
        });
    }
    // Validate-once: Hermitian capability and square dimensions are admitted
    // here so the reverse closure below runs the raw action (finiteness is
    // validated once over the finished outputs). Mass is admitted only when
    // its term is active, matching the conditional application below.
    admit_adjoint_hermitian(stiffness, n)?;
    if let Some(damping) = damping {
        admit_adjoint_hermitian(damping, n)?;
    }
    if config.alpha_m != 0.0 {
        admit_adjoint_hermitian(mass, n)?;
    }

    let real = |value: f64| {
        T::Real::from_f64(value)
            .map(T::from_real)
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "generalized-alpha checkpointed adjoint",
                reason: format!("coefficient {value} cannot be represented"),
            })
    };
    let dt_t = real(config.dt)?;
    let predictor_u_a = real((1.0 - config.alpha_f) * (0.5 - config.beta) * config.dt * config.dt)?;
    let predictor_u_v = real((1.0 - config.alpha_f) * config.dt)?;
    let predictor_v_a = real((1.0 - config.alpha_f) * (1.0 - config.gamma) * config.dt)?;
    let corrector_u_a = real((0.5 - config.beta) * config.dt * config.dt)?;
    let corrector_u_new = real(config.beta * config.dt * config.dt)?;
    let corrector_v_a = real((1.0 - config.gamma) * config.dt)?;
    let corrector_v_new = real(config.gamma * config.dt)?;
    let alpha_m = real(config.alpha_m)?;
    let zero = T::zero();
    let mut lambda_u = terminal_displacement.to_vec();
    let mut lambda_v = terminal_velocity.to_vec();
    let mut lambda_a = terminal_acceleration.to_vec();
    let mut force_gradient = vec![vec![zero; n]; config.num_steps];

    let advance = |from: usize,
                   to: usize,
                   mut state: NewmarkState<T>|
     -> Result<NewmarkState<T>, SolverError> {
        for step in from..to {
            state = advance_step(step, &state)?;
            if state.displacement.len() != n
                || state.velocity.len() != n
                || state.acceleration.len() != n
            {
                return Err(SolverError::DimensionMismatch {
                    expected: n,
                    got: state
                        .displacement
                        .len()
                        .max(state.velocity.len())
                        .max(state.acceleration.len()),
                });
            }
        }
        Ok(state)
    };

    let reverse = |step: usize, state: &NewmarkState<T>| -> Result<(), SolverError> {
        if state.displacement.len() != n
            || state.velocity.len() != n
            || state.acceleration.len() != n
        {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: state
                    .displacement
                    .len()
                    .max(state.velocity.len())
                    .max(state.acceleration.len()),
            });
        }

        let mut lambda_a_new = lambda_a.clone();
        T::vec_axpy(corrector_u_new, &lambda_u, &mut lambda_a_new);
        T::vec_axpy(corrector_v_new, &lambda_v, &mut lambda_a_new);
        let lambda_rhs = catch_unwind(AssertUnwindSafe(|| {
            adjoint_solve(step, state, &lambda_a_new)
        }))
        .map_err(|payload| SolverError::OperationFailed {
            operation: "generalized-alpha checkpointed adjoint solve",
            reason: panic_reason(payload),
        })??;
        if lambda_rhs.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: lambda_rhs.len(),
            });
        }
        force_gradient[step] = lambda_rhs.clone();

        // Solve-mediated predictor pullback only: the direct corrector
        // seeds are added to `lambda_u_previous` below, so seeding
        // `-lambda_rhs` here would corrupt every multi-step gradient.
        let mut lambda_u_pred = vec![zero; n];
        let mut lambda_v_pred = vec![zero; n];
        if let Some(damping) = damping {
            let contribution = apply_adjoint_hermitian(damping, &lambda_rhs, n)?;
            for (target, value) in lambda_v_pred.iter_mut().zip(contribution) {
                *target -= value;
            }
        }
        let stiffness_contribution = apply_adjoint_hermitian(stiffness, &lambda_rhs, n)?;
        for (target, value) in lambda_u_pred.iter_mut().zip(stiffness_contribution) {
            *target -= value;
        }

        let mut lambda_u_previous = lambda_u.clone();
        let mut lambda_v_previous = lambda_v.clone();
        let mut lambda_a_previous = vec![zero; n];
        T::vec_axpy(dt_t, &lambda_u, &mut lambda_v_previous);
        T::vec_axpy(corrector_u_a, &lambda_u, &mut lambda_a_previous);
        T::vec_axpy(corrector_v_a, &lambda_v, &mut lambda_a_previous);
        T::vec_axpy(predictor_u_v, &lambda_u_pred, &mut lambda_v_previous);
        T::vec_axpy(predictor_u_a, &lambda_u_pred, &mut lambda_a_previous);
        T::vec_axpy(predictor_v_a, &lambda_v_pred, &mut lambda_a_previous);
        for (target, value) in lambda_u_previous.iter_mut().zip(lambda_u_pred) {
            *target += value;
        }
        if config.alpha_m != 0.0 {
            let mass_contribution = apply_adjoint_hermitian(mass, &lambda_rhs, n)?;
            T::vec_axpy(-alpha_m, &mass_contribution, &mut lambda_a_previous);
        }
        lambda_u = std::mem::take(&mut lambda_u_previous);
        lambda_v = std::mem::take(&mut lambda_v_previous);
        lambda_a = std::mem::take(&mut lambda_a_previous);
        Ok(())
    };

    schedule
        .execute(initial_state, advance, reverse)
        .map_err(|error| SolverError::OperationFailed {
            operation: "generalized-alpha checkpointed adjoint",
            reason: error.to_string(),
        })?;

    validate_adjoint_outputs(
        "generalized-alpha checkpointed adjoint",
        &force_gradient,
        &lambda_u,
        &lambda_v,
        &lambda_a,
    )?;
    Ok(GeneralizedAlphaAdjointResult {
        force_gradient,
        initial_displacement_gradient: lambda_u,
        initial_velocity_gradient: lambda_v,
        initial_acceleration_gradient: lambda_a,
    })
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "adjoint callback panicked".to_string()
    }
}

/// Validate-once gate for the Hermitian operator actions in an adjoint
/// reverse pass.
///
/// `LinearOperator::try_apply_hermitian` re-checks capability, dimensions,
/// and input/output finiteness on every call — two O(n) scans plus an unwind
/// guard per operator per reverse step. Adjoint trajectories are linear
/// recurrences in fixed operators, so capability and (square) dimensions are
/// admitted once here and finiteness is validated once over the finished
/// outputs (see [`validate_adjoint_outputs`]).
fn admit_adjoint_hermitian<T, Op>(operator: &Op, n: usize) -> Result<(), SolverError>
where
    T: ComplexField,
    Op: LinearOperator<T> + ?Sized,
{
    if operator.hermitian_capability() != AdjointCapability::Exact {
        return Err(SolverError::UnsupportedOperatorOperation {
            operation: "operator Hermitian transpose",
            operator: std::any::type_name::<Op>(),
        });
    }
    if operator.num_rows() != n || operator.num_cols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: operator.num_rows().max(operator.num_cols()),
        });
    }
    Ok(())
}

/// Single reverse-step Hermitian action with the per-call validation hoisted.
///
/// Capability and squareness come from [`admit_adjoint_hermitian`] and
/// finiteness from [`validate_adjoint_outputs`]. The output-length check stays
/// per step: one integer compare preserving the `DimensionMismatch` contract
/// for operators whose raw action mis-reports its length. Panics stay
/// contained per call so a faulty operator action still surfaces as
/// `OperationFailed` rather than an unwind.
fn apply_adjoint_hermitian<T, Op>(operator: &Op, x: &[T], n: usize) -> Result<Vec<T>, SolverError>
where
    T: ComplexField,
    Op: LinearOperator<T> + ?Sized,
{
    debug_assert_eq!(x.len(), n);
    let applied =
        catch_unwind(AssertUnwindSafe(|| operator.apply_hermitian(x))).map_err(|payload| {
            SolverError::OperationFailed {
                operation: "operator Hermitian transpose",
                reason: panic_reason(payload),
            }
        })?;
    if applied.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: applied.len(),
        });
    }
    Ok(applied)
}

/// Validate-once finiteness gate over finished adjoint outputs.
///
/// Every reverse-step value flows linearly into `force_gradient` (each
/// `adjoint_solve` output is stored verbatim) or the terminal cotangents, and
/// NaN/Inf can only propagate through the `+`/`*` recurrence — never cancel
/// back to finite — so one scan here detects exactly what per-step
/// `try_apply_hermitian` scans detected, in a single pass.
fn validate_adjoint_outputs<T: ComplexField>(
    operation: &'static str,
    force_gradient: &[Vec<T>],
    initial_displacement_gradient: &[T],
    initial_velocity_gradient: &[T],
    initial_acceleration_gradient: &[T],
) -> Result<(), SolverError> {
    let finite = force_gradient
        .iter()
        .flatten()
        .all(|value| value.norm().is_finite())
        && [
            initial_displacement_gradient,
            initial_velocity_gradient,
            initial_acceleration_gradient,
        ]
        .iter()
        .all(|gradient| gradient.iter().all(|value| value.norm().is_finite()));
    if finite {
        Ok(())
    } else {
        Err(SolverError::OperationFailed {
            operation,
            reason: "adjoint trajectory contains NaN or infinity".to_string(),
        })
    }
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
    panic!("checked time-integration callback output rejected");
}

/// (Chung & Hulbert, 1993).
///
/// Unlike [`newmark_integrate`], the residual is evaluated at the
/// intermediate point `t_{n+1-alpha_f}` with alpha-weighted states:
///
/// ```text
/// M a_{n+1-alpha_m} + C v_{n+1-alpha_f} + K u_{n+1-alpha_f} = f(t_{n+1-alpha_f})
/// x_{n+1-alpha} = (1 - alpha) * x_{n+1} + alpha * x_n
/// ```
///
/// This gives true user-controlled numerical dissipation parameterized by the
/// spectral radius at infinity (see
/// [`GeneralizedAlphaConfig::from_spectral_radius`]); HHT-alpha is the
/// special case `alpha_m = 0` ([`GeneralizedAlphaConfig::hht_alpha`]).
///
/// Generic over any [`ComplexField`] type: works with f32, f64, Complex32,
/// and Complex64.
///
/// # Arguments
///
/// - `mass`: mass matrix M (LinearOperator). Unlike [`newmark_integrate`],
///   the mass matrix is needed explicitly to form the `alpha_m * M a_n`
///   term of the residual.
/// - `damping`: damping matrix C (LinearOperator), or `None` for undamped
/// - `stiffness`: stiffness matrix K (LinearOperator)
/// - `force_fn`: time-dependent force `f(t)` returning an Array1; it is
///   evaluated at the intermediate time `t_{n+1-alpha_f}`.
/// - `initial`: initial displacement, velocity, and acceleration
/// - `linear_solve`: closure that solves `A_eff x = b` for x, where
///   `A_eff = (1-alpha_m)*M + (1-alpha_f)*gamma*dt*C + (1-alpha_f)*beta*dt^2*K`
///   is the effective generalized-alpha matrix. The caller pre-builds and
///   factorizes this matrix.
/// - `config`: generalized-alpha parameters
pub fn generalized_alpha_integrate<T, M, C, K, F, S>(
    mass: &M,
    damping: Option<&C>,
    stiffness: &K,
    force_fn: F,
    initial: &NewmarkInitial<T>,
    linear_solve: S,
    config: &GeneralizedAlphaConfig,
) -> NewmarkResult<T>
where
    T: ComplexField,
    M: LinearOperator<T>,
    C: LinearOperator<T>,
    K: LinearOperator<T>,
    F: Fn(f64) -> Vec<T>,
    S: Fn(&[T]) -> Vec<T>,
{
    let u0 = &initial.displacement;
    let v0 = &initial.velocity;
    let a0 = &initial.acceleration;
    let dt = config.dt;
    let alpha_m = config.alpha_m;
    let alpha_f = config.alpha_f;
    let beta = config.beta;
    let gamma = config.gamma;
    let dt2 = dt * dt;

    let to_t = |x: f64| scalar_from_f64_or_nan::<T>(x);
    let dt_t = to_t(dt);
    let half_minus_beta_dt2 = to_t((0.5 - beta) * dt2);
    let beta_dt2 = to_t(beta * dt2);
    let one_minus_gamma_dt = to_t((1.0 - gamma) * dt);
    let gamma_dt = to_t(gamma * dt);
    // Coefficients of the alpha-weighted predictor states.
    let one_m_af_dt = to_t((1.0 - alpha_f) * dt);
    let one_m_af_half_minus_beta_dt2 = to_t((1.0 - alpha_f) * (0.5 - beta) * dt2);
    let one_m_af_one_minus_gamma_dt = to_t((1.0 - alpha_f) * (1.0 - gamma) * dt);
    let alpha_m_t = to_t(alpha_m);

    let mut u = u0.clone();
    let mut v = v0.clone();
    let mut a = a0.clone();

    let n = u.len();
    // Pre-allocate buffers (reused across time steps, zero allocation in loop)
    let mut u_pred = vec![T::zero(); n];
    let mut v_pred = vec![T::zero(); n];
    let mut u_next = vec![T::zero(); n];
    let mut v_next = vec![T::zero(); n];
    // RHS assembly scratch: `op_buf` holds each operator action in turn
    // (`apply_into`, no per-step temporaries) and `rhs_buf` the assembled
    // right-hand side. Same operations in the same order as the previous
    // allocating formulation, so bitwise identical.
    let mut op_buf = vec![T::zero(); n];
    let mut rhs_buf = vec![T::zero(); n];

    let mut u_history = if config.store_history {
        Some(vec![u.clone()])
    } else {
        None
    };
    let mut a_history = if config.store_history {
        Some(vec![a.clone()])
    } else {
        None
    };
    let mut v_history = if config.store_history {
        Some(vec![v.clone()])
    } else {
        None
    };

    for step in 0..config.num_steps {
        // Intermediate time t_{n+1-alpha_f} for the force evaluation.
        let t_mid = (step + 1) as f64 * dt - alpha_f * dt;

        // Predictor: constant (a_{n+1}-free) part of the alpha-weighted states.
        // u_pred = u + (1-alpha_f)*dt*v + (1-alpha_f)*(0.5-beta)*dt^2*a
        u_pred.copy_from_slice(&u);
        T::vec_axpy(one_m_af_dt, &v, &mut u_pred);
        T::vec_axpy(one_m_af_half_minus_beta_dt2, &a, &mut u_pred);
        // v_pred = v + (1-alpha_f)*(1-gamma)*dt*a
        v_pred.copy_from_slice(&v);
        T::vec_axpy(one_m_af_one_minus_gamma_dt, &a, &mut v_pred);

        // RHS: f(t_{n+1-alpha_f}) - K*u_pred - C*v_pred - alpha_m*M*a
        let f_mid = force_fn(t_mid);
        stiffness.apply_into(&u_pred, &mut op_buf);
        rhs_buf.copy_from_slice(&f_mid);
        sonium_solvers_core::vec_ops::vec_sub_assign(&mut rhs_buf, &op_buf);
        if let Some(c) = damping {
            c.apply_into(&v_pred, &mut op_buf);
            sonium_solvers_core::vec_ops::vec_sub_assign(&mut rhs_buf, &op_buf);
        }
        if alpha_m != 0.0 {
            mass.apply_into(&a, &mut op_buf);
            // In-place `vec_scale(op_buf, alpha_m_t)`, then subtract —
            // no temporary. (`*=` lowers to the same multiply, so this
            // matches the previous `value * alpha_m_t` order bitwise.)
            for value in op_buf.iter_mut() {
                *value *= alpha_m_t;
            }
            sonium_solvers_core::vec_ops::vec_sub_assign(&mut rhs_buf, &op_buf);
        }

        // Solve effective system for a_{n+1}
        let a_new = linear_solve(&rhs_buf);

        // Corrector: full Newmark update at t_{n+1}.
        // u_{n+1} = u + dt*v + (0.5-beta)*dt^2*a + beta*dt^2*a_{n+1}
        u_next.copy_from_slice(&u);
        T::vec_axpy(dt_t, &v, &mut u_next);
        T::vec_axpy(half_minus_beta_dt2, &a, &mut u_next);
        T::vec_axpy(beta_dt2, &a_new, &mut u_next);
        // v_{n+1} = v + (1-gamma)*dt*a + gamma*dt*a_{n+1}
        v_next.copy_from_slice(&v);
        T::vec_axpy(one_minus_gamma_dt, &a, &mut v_next);
        T::vec_axpy(gamma_dt, &a_new, &mut v_next);

        std::mem::swap(&mut u, &mut u_next);
        std::mem::swap(&mut v, &mut v_next);
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

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn adjoint_result_reports_canonical_status_and_non_finite_failure() {
        let finite = NewmarkAdjointResult {
            force_gradient: vec![vec![1.0_f64]],
            initial_displacement_gradient: vec![2.0],
            initial_velocity_gradient: vec![3.0],
            initial_acceleration_gradient: vec![4.0],
        };
        assert!(finite.is_finite());
        assert_eq!(finite.status(), SolverStatus::Converged);
        assert_eq!(finite.accuracy_status(), AccuracyStatus::NotChecked);
        assert_eq!(finite.attainable_accuracy(), AccuracyStatus::NotChecked);

        let non_finite = NewmarkAdjointResult {
            force_gradient: vec![vec![f64::NAN]],
            ..finite
        };
        assert!(!non_finite.is_finite());
        assert_eq!(non_finite.status(), SolverStatus::NumericalError);
        assert_eq!(non_finite.accuracy_status(), AccuracyStatus::NotAttained);
    }

    #[test]
    fn generalized_alpha_adjoint_matches_one_step_force_derivative() {
        let mass = CsrMatrix::from_dense(&sonium_solvers_core::array![[1.0_f64]], 0.0);
        let stiffness = CsrMatrix::from_dense(&sonium_solvers_core::array![[0.0_f64]], 0.0);
        let config = GeneralizedAlphaConfig::from_spectral_radius(0.5, 0.2, 1).with_history();
        let initial = NewmarkInitial {
            displacement: vec![0.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };
        let forward = generalized_alpha_integrate(
            &mass,
            None::<&CsrMatrix<f64>>,
            &stiffness,
            |_| vec![1.0],
            &initial,
            |rhs: &[f64]| rhs.to_vec(),
            &config,
        );
        let adjoint = generalized_alpha_adjoint(
            &mass,
            None::<&CsrMatrix<f64>>,
            &stiffness,
            &forward,
            &[1.0],
            &[0.0],
            &[0.0],
            |rhs: &[f64]| Ok(rhs.to_vec()),
            &config,
        )
        .expect("generalized-alpha adjoint should converge");
        let expected = config.beta * config.dt * config.dt;
        assert!((adjoint.force_gradient[0][0] - expected).abs() < 1.0e-12);
    }

    #[test]
    fn checkpointed_generalized_alpha_adjoint_matches_stored_recurrence() {
        let mass = CsrMatrix::from_dense(&sonium_solvers_core::array![[1.0_f64]], 0.0);
        let stiffness = CsrMatrix::from_dense(&sonium_solvers_core::array![[0.0_f64]], 0.0);
        let config = GeneralizedAlphaConfig::from_spectral_radius(0.5, 0.2, 4).with_history();
        let initial = NewmarkInitial {
            displacement: vec![0.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };
        let forward = generalized_alpha_integrate(
            &mass,
            None::<&CsrMatrix<f64>>,
            &stiffness,
            |_| vec![0.0],
            &initial,
            |rhs: &[f64]| rhs.to_vec(),
            &config,
        );
        let stored = generalized_alpha_adjoint(
            &mass,
            None::<&CsrMatrix<f64>>,
            &stiffness,
            &forward,
            &[1.0],
            &[0.25],
            &[0.5],
            |rhs: &[f64]| Ok(rhs.to_vec()),
            &config,
        )
        .expect("stored generalized-alpha adjoint should succeed");

        let checkpointed = generalized_alpha_adjoint_checkpointed(
            &mass,
            None::<&CsrMatrix<f64>>,
            &stiffness,
            NewmarkState::new(vec![0.0], vec![0.0], vec![0.0]),
            &[1.0],
            &[0.25],
            &[0.5],
            |_step, state| {
                Ok(NewmarkState::new(
                    vec![state.displacement[0] + 1.0],
                    vec![state.velocity[0] + 2.0],
                    vec![state.acceleration[0] + 3.0],
                ))
            },
            |step, state, rhs| {
                assert_eq!(state.displacement[0], (step + 1) as f64);
                Ok(rhs.to_vec())
            },
            &config,
            2,
        )
        .expect("checkpointed generalized-alpha adjoint should succeed");

        assert_eq!(checkpointed.force_gradient, stored.force_gradient);
        assert_eq!(
            checkpointed.initial_displacement_gradient,
            stored.initial_displacement_gradient
        );
        assert_eq!(
            checkpointed.initial_velocity_gradient,
            stored.initial_velocity_gradient
        );
        assert_eq!(
            checkpointed.initial_acceleration_gradient,
            stored.initial_acceleration_gradient
        );
    }

    /// Free vibration of a single DOF: m*u'' + k*u = 0
    /// Exact solution: u(t) = u0 * cos(omega*t), omega = sqrt(k/m)
    #[test]
    fn test_single_dof_free_vibration() {
        let m_val = 1.0_f64;
        let k_val = 4.0_f64;
        let omega = (k_val / m_val).sqrt();

        let k = CsrMatrix::from_dense(&sonium_solvers_core::array![[k_val]], 0.0);

        let initial = NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.0],
            acceleration: vec![-k_val], // a_0 = -k*u_0/m
        };

        let period = 2.0 * std::f64::consts::PI / omega;
        let dt = period / 100.0;
        let num_steps = 500;
        let config = NewmarkConfig::average_acceleration(dt, num_steps);

        let eff_val = m_val + 0.25 * dt * dt * k_val;

        let result = newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &k,
            |_t| vec![0.0],
            &initial,
            |rhs: &[f64]| sonium_solvers_core::array![rhs[0] / eff_val],
            &config,
        );

        let t_final = result.final_time;
        let u_exact = (omega * t_final).cos();

        assert!(
            (result.displacement[0] - u_exact).abs() < 0.01,
            "u({t_final:.4}) = {:.6}, expected {u_exact:.6}",
            result.displacement[0]
        );
    }

    /// Verify f32 works (single precision).
    #[test]
    fn test_single_dof_f32() {
        let k_val = 4.0_f32;
        let k = CsrMatrix::from_dense(&sonium_solvers_core::array![[k_val]], 0.0);

        let initial = NewmarkInitial {
            displacement: vec![1.0_f32],
            velocity: vec![0.0_f32],
            acceleration: vec![-k_val],
        };

        let dt = 0.01;
        let config = NewmarkConfig::average_acceleration(dt, 100);
        let eff_val: f32 = 1.0 + 0.25 * (dt * dt) as f32 * k_val;

        let result = newmark_integrate(
            None::<&CsrMatrix<f32>>,
            &k,
            |_t| vec![0.0_f32],
            &initial,
            |rhs: &[f32]| sonium_solvers_core::array![rhs[0] / eff_val],
            &config,
        );

        assert!(result.displacement[0].abs() < 2.0, "f32 solve should work");
    }

    /// Forced vibration: m*u'' + k*u = F0*sin(omega_f*t)
    #[test]
    fn test_forced_vibration_bounded() {
        let m_val = 1.0_f64;
        let k_val = 100.0_f64;
        let omega = (k_val / m_val).sqrt();
        let omega_f = omega * 0.9;
        let f0 = 1.0;

        let k = CsrMatrix::from_dense(&sonium_solvers_core::array![[k_val]], 0.0);

        let period = 2.0 * std::f64::consts::PI / omega;
        let dt = period / 50.0;
        let num_steps = 500;
        let config = NewmarkConfig::average_acceleration(dt, num_steps);
        let eff_val = m_val + 0.25 * dt * dt * k_val;

        let initial = NewmarkInitial {
            displacement: vec![0.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };

        let result = newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &k,
            |t| vec![f0 * (omega_f * t).sin()],
            &initial,
            |rhs: &[f64]| sonium_solvers_core::array![rhs[0] / eff_val],
            &config,
        );

        assert!(
            result.displacement[0].abs() < 100.0,
            "Displacement should be bounded, got {}",
            result.displacement[0]
        );
    }

    /// Verify that with damping, energy decays.
    #[test]
    fn test_damped_free_vibration() {
        let m_val = 1.0_f64;
        let c_val = 0.5;
        let k_val = 4.0_f64;

        let c = CsrMatrix::from_dense(&sonium_solvers_core::array![[c_val]], 0.0);
        let k = CsrMatrix::from_dense(&sonium_solvers_core::array![[k_val]], 0.0);

        let dt = 0.05;
        let num_steps = 200;
        let config = NewmarkConfig::average_acceleration(dt, num_steps);
        let eff_val = m_val + 0.5 * dt * c_val + 0.25 * dt * dt * k_val;

        let initial = NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.0],
            acceleration: vec![-k_val],
        };

        let result = newmark_integrate(
            Some(&c),
            &k,
            |_t| vec![0.0],
            &initial,
            |rhs: &[f64]| sonium_solvers_core::array![rhs[0] / eff_val],
            &config,
        );

        assert!(
            result.displacement[0].abs() < 0.5,
            "Damped displacement should decay, got {}",
            result.displacement[0]
        );
    }

    /// Verify history storage works.
    #[test]
    fn test_history_storage() {
        let k = CsrMatrix::from_dense(&sonium_solvers_core::array![[1.0_f64]], 0.0);

        let dt = 0.1;
        let num_steps = 10;
        let config = NewmarkConfig::average_acceleration(dt, num_steps).with_history();
        let eff_val = 1.0 + 0.25 * dt * dt;

        let initial = NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.0],
            acceleration: vec![-1.0],
        };

        let result = newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &k,
            |_t| vec![0.0],
            &initial,
            |rhs: &[f64]| sonium_solvers_core::array![rhs[0] / eff_val],
            &config,
        );

        let hist = result.displacement_history.unwrap();
        assert_eq!(hist.len(), num_steps + 1);
        assert!((hist[0][0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn newmark_adjoint_matches_one_step_force_derivative() {
        let stiffness = CsrMatrix::from_dense(&sonium_solvers_core::array![[0.0_f64]], 0.0);
        let initial = NewmarkInitial {
            displacement: vec![0.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };
        let config = NewmarkConfig::average_acceleration(0.1, 1).with_history();
        let forward = newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &stiffness,
            |_t| vec![2.0],
            &initial,
            |rhs: &[f64]| rhs.to_vec(),
            &config,
        );
        let adjoint = newmark_adjoint(
            None::<&CsrMatrix<f64>>,
            &stiffness,
            &forward,
            &[1.0],
            &[0.0],
            &[0.0],
            |rhs: &[f64]| Ok(rhs.to_vec()),
            &config,
        )
        .expect("one-step adjoint should succeed");
        assert!((adjoint.force_gradient[0][0] - 0.0025).abs() < 1.0e-14);
    }

    #[test]
    fn checkpointed_newmark_adjoint_matches_stored_recurrence() {
        let stiffness = CsrMatrix::from_dense(&sonium_solvers_core::array![[0.0_f64]], 0.0);
        let config = NewmarkConfig::average_acceleration(0.1, 4).with_history();
        let initial = NewmarkInitial {
            displacement: vec![0.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };
        let forward = newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &stiffness,
            |_t| vec![0.0],
            &initial,
            |rhs: &[f64]| rhs.to_vec(),
            &config,
        );
        let stored = newmark_adjoint(
            None::<&CsrMatrix<f64>>,
            &stiffness,
            &forward,
            &[1.0],
            &[0.25],
            &[0.5],
            |rhs: &[f64]| Ok(rhs.to_vec()),
            &config,
        )
        .expect("stored-history adjoint should succeed");

        let observed_reverse_steps = std::cell::Cell::new(0usize);
        let checkpointed = newmark_adjoint_checkpointed(
            None::<&CsrMatrix<f64>>,
            &stiffness,
            NewmarkState::new(vec![0.0], vec![0.0], vec![0.0]),
            &[1.0],
            &[0.25],
            &[0.5],
            |_step, state| {
                Ok(NewmarkState::new(
                    vec![state.displacement[0] + 1.0],
                    vec![state.velocity[0] + 2.0],
                    vec![state.acceleration[0] + 3.0],
                ))
            },
            |step, state, rhs| {
                assert_eq!(state.displacement[0], (step + 1) as f64);
                observed_reverse_steps.set(observed_reverse_steps.get() + 1);
                Ok(rhs.to_vec())
            },
            &config,
            2,
        )
        .expect("checkpointed adjoint should succeed");

        assert_eq!(observed_reverse_steps.get(), config.num_steps);
        assert_eq!(checkpointed.force_gradient, stored.force_gradient);
        assert_eq!(
            checkpointed.initial_displacement_gradient,
            stored.initial_displacement_gradient
        );
        assert_eq!(
            checkpointed.initial_velocity_gradient,
            stored.initial_velocity_gradient
        );
        assert_eq!(
            checkpointed.initial_acceleration_gradient,
            stored.initial_acceleration_gradient
        );
    }

    #[test]
    fn newmark_config_exposes_bounded_checkpoint_schedule() {
        let config = NewmarkConfig::average_acceleration(0.01, 9);
        let schedule = config
            .checkpoint_schedule(3)
            .expect("checkpoint budget should be valid");
        assert_eq!(schedule.num_steps(), 9);
        assert_eq!(schedule.max_checkpoints(), 3);
        assert_eq!(schedule.reverse_steps(), 9);

        let error = config
            .checkpoint_schedule(0)
            .expect_err("non-empty trajectory needs a checkpoint");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_newmark_rejects_invalid_config_and_contains_callback_panics() {
        let stiffness = CsrMatrix::from_dense(&sonium_solvers_core::array![[0.0_f64]], 0.0);
        let initial = NewmarkInitial {
            displacement: vec![0.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };
        let invalid = NewmarkConfig {
            beta: 0.25,
            gamma: 0.5,
            dt: 0.0,
            num_steps: 1,
            store_history: false,
        };
        let error = try_newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &stiffness,
            |_t| vec![0.0],
            &initial,
            |rhs: &[f64]| rhs.to_vec(),
            &invalid,
        )
        .expect_err("invalid dt must fail before integration");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let valid = NewmarkConfig::average_acceleration(0.1, 1);
        let error = try_newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &stiffness,
            |_t| panic!("force callback failure"),
            &initial,
            |rhs: &[f64]| rhs.to_vec(),
            &valid,
        )
        .expect_err("force callback panic must be contained");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    /// Run a scalar `u'' + c u' + k u = 0` (unit mass) with generalized-alpha.
    fn run_sdof_ga(
        rho_inf: f64,
        c: f64,
        k: f64,
        dt: f64,
        num_steps: usize,
        u0: f64,
        v0: f64,
    ) -> NewmarkResult<f64> {
        let m = 1.0;
        let config = GeneralizedAlphaConfig::from_spectral_radius(rho_inf, dt, num_steps);
        let mass = CsrMatrix::from_dense(&sonium_solvers_core::array![[m]], 0.0);
        let damp = CsrMatrix::from_dense(&sonium_solvers_core::array![[c]], 0.0);
        let stiff = CsrMatrix::from_dense(&sonium_solvers_core::array![[k]], 0.0);
        let initial = NewmarkInitial {
            displacement: vec![u0],
            velocity: vec![v0],
            acceleration: vec![-(c * v0 + k * u0) / m],
        };
        let eff = (1.0 - config.alpha_m) * m
            + (1.0 - config.alpha_f) * config.gamma * dt * c
            + (1.0 - config.alpha_f) * config.beta * dt * dt * k;
        generalized_alpha_integrate(
            &mass,
            Some(&damp),
            &stiff,
            |_t| vec![0.0],
            &initial,
            |rhs: &[f64]| sonium_solvers_core::array![rhs[0] / eff],
            &config,
        )
    }

    /// Coefficient formulas (Chung & Hulbert, 1993) and the HHT special case.
    #[test]
    fn test_generalized_alpha_coefficients() {
        // rho_inf = 1 -> alpha_m = alpha_f = 1/2: midpoint evaluation of the
        // residual, the non-dissipative trapezoidal limit (gamma = 1/2,
        // beta = 1/4 as in average acceleration).
        let cfg = GeneralizedAlphaConfig::from_spectral_radius(1.0, 0.01, 1);
        assert!((cfg.alpha_m - 0.5).abs() < 1e-15);
        assert!((cfg.alpha_f - 0.5).abs() < 1e-15);
        assert!((cfg.beta - 0.25).abs() < 1e-15);
        assert!((cfg.gamma - 0.5).abs() < 1e-15);

        // rho_inf = 0 -> alpha_m = -1, alpha_f = 0, gamma = 3/2, beta = 1
        let cfg = GeneralizedAlphaConfig::from_spectral_radius(0.0, 0.01, 1);
        assert!((cfg.alpha_m + 1.0).abs() < 1e-15);
        assert!(cfg.alpha_f.abs() < 1e-15);
        assert!((cfg.gamma - 1.5).abs() < 1e-15);
        assert!((cfg.beta - 1.0).abs() < 1e-15);

        // HHT-alpha is the special case alpha_m = 0, alpha_f = alpha
        let alpha = 0.1;
        let cfg = GeneralizedAlphaConfig::hht_alpha(alpha, 0.01, 1);
        assert!(cfg.alpha_m.abs() < 1e-15);
        assert!((cfg.alpha_f - alpha).abs() < 1e-15);
        assert!((cfg.gamma - (0.5 + alpha)).abs() < 1e-15);
        assert!((cfg.beta - (1.0 + alpha).powi(2) / 4.0).abs() < 1e-15);
    }

    /// With rho_inf = 1 (alpha_m = alpha_f = 1/2), generalized-alpha
    /// evaluates the residual at the midpoint; for a linear autonomous
    /// problem this coincides algebraically with average-acceleration
    /// Newmark, so the free-vibration trajectories must match
    /// `newmark_integrate` to roundoff.
    #[test]
    fn test_generalized_alpha_matches_newmark_at_rho1() {
        let m_val = 1.0_f64;
        let c_val = 0.5;
        let k_val = 4.0_f64;

        let dt = 0.03;
        let num_steps = 100;

        let c = CsrMatrix::from_dense(&sonium_solvers_core::array![[c_val]], 0.0);
        let k = CsrMatrix::from_dense(&sonium_solvers_core::array![[k_val]], 0.0);
        let initial = NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.2],
            acceleration: vec![-(c_val * 0.2 + k_val) / m_val],
        };

        let config_nm = NewmarkConfig::average_acceleration(dt, num_steps).with_history();
        let eff_nm = m_val + 0.5 * dt * c_val + 0.25 * dt * dt * k_val;
        let res_nm = newmark_integrate(
            Some(&c),
            &k,
            |_t| vec![0.0],
            &initial,
            |rhs: &[f64]| sonium_solvers_core::array![rhs[0] / eff_nm],
            &config_nm,
        );

        let m_op = CsrMatrix::from_dense(&sonium_solvers_core::array![[m_val]], 0.0);
        let config_ga =
            GeneralizedAlphaConfig::from_spectral_radius(1.0, dt, num_steps).with_history();
        let eff_ga = (1.0 - config_ga.alpha_m) * m_val
            + (1.0 - config_ga.alpha_f) * config_ga.gamma * dt * c_val
            + (1.0 - config_ga.alpha_f) * config_ga.beta * dt * dt * k_val;
        let res_ga = generalized_alpha_integrate(
            &m_op,
            Some(&c),
            &k,
            |_t| vec![0.0],
            &initial,
            |rhs: &[f64]| sonium_solvers_core::array![rhs[0] / eff_ga],
            &config_ga,
        );

        let u_nm = res_nm.displacement_history.unwrap();
        let u_ga = res_ga.displacement_history.unwrap();
        let v_nm = res_nm.velocity_history.unwrap();
        let v_ga = res_ga.velocity_history.unwrap();
        for i in 0..=num_steps {
            assert!(
                (u_nm[i][0] - u_ga[i][0]).abs() < 1e-11,
                "step {i}: u {} vs {}",
                u_nm[i][0],
                u_ga[i][0]
            );
            assert!(
                (v_nm[i][0] - v_ga[i][0]).abs() < 1e-11,
                "step {i}: v {} vs {}",
                v_nm[i][0],
                v_ga[i][0]
            );
        }
    }

    /// Second-order accuracy: dt refinement study on an undamped SDOF
    /// oscillator. The error at a fixed time must decrease by ~4x per
    /// halving of dt. (The final time is 1.25 periods so the phase error —
    /// the dominant O(dt^2) error — is visible in the displacement.)
    #[test]
    fn test_generalized_alpha_second_order() {
        let k_val = 4.0_f64;
        let omega = 2.0_f64;
        let period = 2.0 * std::f64::consts::PI / omega;
        let t_final = 1.25 * period;
        let u_exact = (omega * t_final).cos();

        let mut errors = Vec::new();
        for &num_steps in &[40_usize, 80, 160] {
            let dt = t_final / num_steps as f64;
            let res = run_sdof_ga(0.9, 0.0, k_val, dt, num_steps, 1.0, 0.0);
            errors.push((res.displacement[0] - u_exact).abs());
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

    /// Spectral radius (max |eigenvalue|) of a 3x3 real matrix, via the
    /// characteristic polynomial and the cubic formula.
    fn spectral_radius_3x3(a: [[f64; 3]; 3]) -> f64 {
        let tr = a[0][0] + a[1][1] + a[2][2];
        let q = a[0][0] * a[1][1] - a[0][1] * a[1][0] + a[0][0] * a[2][2] - a[0][2] * a[2][0]
            + a[1][1] * a[2][2]
            - a[1][2] * a[2][1];
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        // lambda^3 - tr*lambda^2 + q*lambda - det = 0; depress with
        // lambda = mu + tr/3: mu^3 + p*mu + qq = 0.
        let p = q - tr * tr / 3.0;
        let qq = -2.0 * tr.powi(3) / 27.0 + tr * q / 3.0 - det;
        let disc = qq * qq / 4.0 + p.powi(3) / 27.0;
        let shift = tr / 3.0;
        if disc >= 0.0 {
            // One real root (Cardano) + a complex pair; pair modulus from
            // the product of roots (= det).
            let sq = disc.sqrt();
            let cubert = |x: f64| x.abs().cbrt().copysign(x);
            let mu = cubert(-qq / 2.0 + sq) + cubert(-qq / 2.0 - sq);
            let real_root = mu + shift;
            let pair_modulus = (det / real_root).abs().sqrt();
            real_root.abs().max(pair_modulus)
        } else {
            // Three real roots (trigonometric method).
            let r = 2.0 * (-p / 3.0).sqrt();
            let theta = ((3.0 * qq / (2.0 * p)) * (-3.0 / p).sqrt()).acos() / 3.0;
            (0..3)
                .map(|k| {
                    (r * (theta - 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos() + shift).abs()
                })
                .fold(0.0, f64::max)
        }
    }

    /// Numerical damping matches theory: build the 3x3 amplification matrix
    /// of the integrator by one-step runs from basis states, and check its
    /// spectral radius against the Chung-Hulbert theory — the high-frequency
    /// mode (omega*dt >> 1) dissipates toward `rho_inf`, the low-frequency
    /// mode (omega*dt << 1) is nearly undamped.
    #[test]
    fn test_generalized_alpha_high_frequency_damping() {
        let dt = 0.05;
        let rho_inf = 0.5;
        let m_val = 1.0_f64;
        // Mode 1: omega*dt = 0.05 (low), mode 2: omega*dt = 200 (high).
        let omega1 = 1.0_f64;
        let omega2 = 4000.0_f64;
        let k1 = omega1 * omega1;
        let k2 = omega2 * omega2;

        let mass = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[m_val, 0.0], [0.0, m_val]],
            0.0,
        );
        let stiff = CsrMatrix::from_dense(&sonium_solvers_core::array![[k1, 0.0], [0.0, k2]], 0.0);
        let config = GeneralizedAlphaConfig::from_spectral_radius(rho_inf, dt, 1);
        let eff1 =
            (1.0 - config.alpha_m) * m_val + (1.0 - config.alpha_f) * config.beta * dt * dt * k1;
        let eff2 =
            (1.0 - config.alpha_m) * m_val + (1.0 - config.alpha_f) * config.beta * dt * dt * k2;

        // One-step map on each mode's (u, v, a) state, column by column:
        // the homogeneous scheme is linear in the full state, so one-step
        // runs from basis states give the columns of the 3x3 amplification
        // matrix of each (decoupled, diagonal) mode.
        let mut amp = [[[0.0_f64; 3]; 3]; 2];
        for dof in 0..2 {
            for (col, basis) in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
                .iter()
                .enumerate()
            {
                let mut initial = NewmarkInitial {
                    displacement: vec![0.0; 2],
                    velocity: vec![0.0; 2],
                    acceleration: vec![0.0; 2],
                };
                initial.displacement[dof] = basis.0;
                initial.velocity[dof] = basis.1;
                initial.acceleration[dof] = basis.2;
                let res = generalized_alpha_integrate(
                    &mass,
                    None::<&CsrMatrix<f64>>,
                    &stiff,
                    |_t| vec![0.0; 2],
                    &initial,
                    |rhs: &[f64]| vec![rhs[0] / eff1, rhs[1] / eff2],
                    &config,
                );
                amp[dof][0][col] = res.displacement[dof];
                amp[dof][1][col] = res.velocity[dof];
                amp[dof][2][col] = res.acceleration[dof];
            }
        }

        let rho_low = spectral_radius_3x3(amp[0]);
        let rho_high = spectral_radius_3x3(amp[1]);
        assert!(
            rho_low > 0.9999,
            "low-frequency mode should be nearly undamped, rho = {rho_low}"
        );
        assert!(
            (rho_high - rho_inf).abs() < 0.03,
            "high-frequency mode should dissipate per rho_inf = {rho_inf}, got {rho_high}"
        );
    }

    /// Free vibration of a damped SDOF with rho_inf = 1 (no algorithmic
    /// damping) matches the analytical underdamped decay.
    /// Checked entry points must reject coefficients that overflow the target
    /// scalar instead of entering the compatibility kernel with infinities.
    #[test]
    fn checked_time_integration_rejects_target_scalar_coefficient_overflow() {
        let stiffness = CsrMatrix::from_dense(&sonium_solvers_core::array![[0.0_f32]], 0.0);
        let initial = NewmarkInitial {
            displacement: vec![0.0_f32],
            velocity: vec![0.0_f32],
            acceleration: vec![0.0_f32],
        };
        let config = NewmarkConfig {
            beta: 0.25,
            gamma: 0.5,
            dt: f64::MAX,
            num_steps: 0,
            store_history: false,
        };

        let error = try_newmark_integrate(
            None::<&CsrMatrix<f32>>,
            &stiffness,
            |_| vec![0.0_f32],
            &initial,
            |_| vec![0.0_f32],
            &config,
        )
        .expect_err("f64::MAX cannot be represented as a finite f32 coefficient");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let generalized_config = GeneralizedAlphaConfig {
            alpha_m: 0.0,
            alpha_f: 0.0,
            beta: 0.25,
            gamma: 0.5,
            dt: f64::MAX,
            num_steps: 0,
            store_history: false,
        };
        let generalized_error = try_generalized_alpha_integrate(
            &stiffness,
            None::<&CsrMatrix<f32>>,
            &stiffness,
            |_| vec![0.0_f32],
            &initial,
            |_| vec![0.0_f32],
            &generalized_config,
        )
        .expect_err("generalized-alpha must reject an overflowing f32 coefficient");
        assert!(matches!(
            generalized_error,
            SolverError::OperationFailed { .. }
        ));
    }

    #[test]
    fn test_generalized_alpha_free_vibration_decay() {
        let m_val = 1.0_f64;
        let c_val = 0.2;
        let k_val = 4.0_f64;
        let omega0 = (k_val / m_val).sqrt();
        let zeta = c_val / (2.0 * m_val * omega0);
        let omega_d = omega0 * (1.0 - zeta * zeta).sqrt();

        let period = 2.0 * std::f64::consts::PI / omega_d;
        let dt = period / 200.0;
        let num_steps = 1000; // 5 periods
        let res = run_sdof_ga(1.0, c_val, k_val, dt, num_steps, 1.0, 0.0);

        let t = res.final_time;
        let u_exact = (-zeta * omega0 * t).exp()
            * ((omega_d * t).cos() + (zeta * omega0 / omega_d) * (omega_d * t).sin());
        assert!(
            (res.displacement[0] - u_exact).abs() < 5e-4,
            "u({t:.4}) = {:.6}, expected {u_exact:.6}",
            res.displacement[0]
        );
    }
}
