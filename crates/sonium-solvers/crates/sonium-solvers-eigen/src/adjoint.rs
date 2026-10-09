//! Sensitivity primitives for generalized eigenvalue adjoints.

use num_traits::{Float, FromPrimitive, Zero};
use sonium_solvers_core::traits::{ComplexField, LinearOperator, SolverError};

fn validate_finite_slice<T: ComplexField>(
    operation: &'static str,
    values: &[T],
) -> Result<(), SolverError> {
    if values.iter().all(|value| value.norm().is_finite()) {
        Ok(())
    } else {
        Err(SolverError::OperationFailed {
            operation,
            reason: "input contains NaN or infinity".into(),
        })
    }
}

fn validate_finite_scalar<T: ComplexField>(
    operation: &'static str,
    value: T,
) -> Result<(), SolverError> {
    if value.norm().is_finite() {
        Ok(())
    } else {
        Err(SolverError::OperationFailed {
            operation,
            reason: "value contains NaN or infinity".into(),
        })
    }
}

/// A checked point on a generalized eigenvalue branch.
///
/// The vectors are not implicitly normalized: callers may use the
/// normalization convention required by their eigensolver. The predictor
/// preserves that convention by applying the gauge-fixed eigenvector
/// sensitivity returned by [`generalized_eigenvector_sensitivity`].
#[derive(Debug, Clone)]
pub struct EigenContinuationPoint<T: ComplexField> {
    /// Continuation parameter at this point.
    pub parameter: T::Real,
    /// Eigenvalue satisfying `A(parameter) v = lambda B(parameter) v`.
    pub eigenvalue: T,
    /// Right eigenvector at this point.
    pub right_eigenvector: Vec<T>,
    /// Left eigenvector used for the adjoint and gauge condition.
    pub left_eigenvector: Vec<T>,
}

/// First-order tangent prediction for one generalized eigenpair step.
#[derive(Debug, Clone)]
pub struct EigenContinuationPrediction<T: ComplexField> {
    /// Predicted continuation parameter.
    pub parameter: T::Real,
    /// First-order predicted eigenvalue.
    pub eigenvalue: T,
    /// First-order predicted right eigenvector.
    pub right_eigenvector: Vec<T>,
    /// `d lambda / d parameter` at the source point.
    pub eigenvalue_sensitivity: T,
    /// Gauge-fixed `d v / d parameter` at the source point.
    pub eigenvector_sensitivity: Vec<T>,
    /// Signed parameter increment used by the predictor.
    pub step_size: T::Real,
}

/// Adaptive step policy for [`continue_generalized_eigenpair`].
#[derive(Debug, Clone, Copy)]
pub struct EigenContinuationConfig {
    /// Smallest absolute step accepted after correction retries.
    pub min_step_size: f64,
    /// Largest absolute step used by the predictor.
    pub max_step_size: f64,
    /// Multiplicative shrink applied after a corrector failure.
    pub shrink_factor: f64,
    /// Maximum number of corrector retries after the initial attempt.
    pub max_retries: usize,
}

impl Default for EigenContinuationConfig {
    fn default() -> Self {
        Self {
            min_step_size: 1.0e-6,
            max_step_size: 1.0,
            shrink_factor: 0.5,
            max_retries: 4,
        }
    }
}

/// Accepted adaptive continuation step and its corrected eigenpair.
#[derive(Debug, Clone)]
pub struct EigenContinuationStep<T: ComplexField> {
    /// First-order predictor supplied to the corrector callback.
    pub prediction: EigenContinuationPrediction<T>,
    /// Corrected eigenpair at `prediction.parameter`.
    pub corrected_point: EigenContinuationPoint<T>,
    /// Absolute number of retries used after the initial corrector attempt.
    pub retries: usize,
}

/// Classification emitted by pseudo-arclength continuation.
///
/// `TurningPoint` means that the continuation parameter changed direction
/// while the branch tangent kept its orientation. `BranchReversal` is a
/// stronger signal: the full `(parameter, eigenvalue)` secant reversed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenContinuationEvent {
    /// The branch tangent kept its orientation and parameter direction.
    Regular,
    /// The continuation parameter changed direction while the branch stayed coherent.
    TurningPoint,
    /// The full parameter/eigenvalue secant reversed orientation.
    BranchReversal,
}

/// Secant predictor used by pseudo-arclength continuation.
#[derive(Debug, Clone)]
pub struct EigenPseudoArclengthPrediction<T: ComplexField> {
    /// Predicted continuation parameter.
    pub parameter: T::Real,
    /// Predicted eigenvalue (including complex components when applicable).
    pub eigenvalue: T,
    /// Secant-extrapolated right eigenvector.
    pub right_eigenvector: Vec<T>,
    /// Signed arclength requested for this attempt.
    pub step_size: T::Real,
    /// Parameter component of the unit secant tangent.
    pub tangent_parameter: T::Real,
    /// Eigenvalue component of the unit secant tangent.
    pub tangent_eigenvalue: T,
}

/// Result of one checked pseudo-arclength continuation step.
#[derive(Debug, Clone)]
pub struct EigenPseudoArclengthStep<T: ComplexField> {
    /// Secant prediction supplied to the corrector callback.
    pub prediction: EigenPseudoArclengthPrediction<T>,
    /// Corrected point returned by the callback.
    pub corrected_point: EigenContinuationPoint<T>,
    /// Number of retries after the initial callback attempt.
    pub retries: usize,
    /// Event detected from the previous/current/corrected secants.
    pub event: EigenContinuationEvent,
}

/// Validated continuation step bounds: `(min_step_size, max_step_size, shrink_factor)`.
type ContinuationStepBounds<R> = (R, R, R);

fn validate_continuation_config<T: ComplexField>(
    config: &EigenContinuationConfig,
) -> Result<ContinuationStepBounds<T::Real>, SolverError> {
    if !config.min_step_size.is_finite()
        || !config.max_step_size.is_finite()
        || !config.shrink_factor.is_finite()
        || config.min_step_size <= 0.0
        || config.max_step_size < config.min_step_size
        || config.shrink_factor <= 0.0
        || config.shrink_factor >= 1.0
    {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair continuation config",
            reason: "step bounds must finite positive with 0 < shrink_factor < 1".into(),
        });
    }
    Ok((
        real_from_f64::<T>(
            config.min_step_size,
            "generalized eigenpair continuation config",
        )?,
        real_from_f64::<T>(
            config.max_step_size,
            "generalized eigenpair continuation config",
        )?,
        real_from_f64::<T>(
            config.shrink_factor,
            "generalized eigenpair continuation config",
        )?,
    ))
}

fn validate_point_shape<T: ComplexField>(
    point: &EigenContinuationPoint<T>,
) -> Result<(), SolverError> {
    if !point.parameter.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair pseudo-arclength",
            reason: "continuation point parameter NaN or infinity".into(),
        });
    }
    validate_finite_scalar("generalized eigenpair pseudo-arclength", point.eigenvalue)?;
    if point.left_eigenvector.len() != point.right_eigenvector.len() {
        return Err(SolverError::DimensionMismatch {
            expected: point.right_eigenvector.len(),
            got: point.left_eigenvector.len(),
        });
    }
    validate_finite_slice(
        "generalized eigenpair pseudo-arclength",
        &point.right_eigenvector,
    )?;
    validate_finite_slice(
        "generalized eigenpair pseudo-arclength",
        &point.left_eigenvector,
    )
}

/// Classify a branch change from three consecutive corrected points.
///
/// The test is independent of eigenvector phase and works for real and
/// complex eigenvalues. A parameter-direction sign change is a turning point;
/// a negative secant dot product is reported as the stronger branch-reversal
/// event. Degenerate (zero-length) secants are rejected instead of silently
/// producing an ambiguous event.
pub fn classify_eigen_continuation_event<T: ComplexField>(
    previous: &EigenContinuationPoint<T>,
    current: &EigenContinuationPoint<T>,
    next: &EigenContinuationPoint<T>,
) -> Result<EigenContinuationEvent, SolverError> {
    validate_point_shape(previous)?;
    validate_point_shape(current)?;
    validate_point_shape(next)?;
    if previous.right_eigenvector.len() != current.right_eigenvector.len()
        || current.right_eigenvector.len() != next.right_eigenvector.len()
    {
        return Err(SolverError::DimensionMismatch {
            expected: previous.right_eigenvector.len(),
            got: current
                .right_eigenvector
                .len()
                .max(next.right_eigenvector.len()),
        });
    }
    let previous_parameter_delta = current.parameter - previous.parameter;
    let next_parameter_delta = next.parameter - current.parameter;
    let previous_eigenvalue_delta = current.eigenvalue - previous.eigenvalue;
    let next_eigenvalue_delta = next.eigenvalue - current.eigenvalue;
    let previous_norm = (previous_parameter_delta * previous_parameter_delta
        + previous_eigenvalue_delta.norm_sqr())
    .sqrt();
    let next_norm =
        (next_parameter_delta * next_parameter_delta + next_eigenvalue_delta.norm_sqr()).sqrt();
    if !previous_norm.is_finite()
        || !next_norm.is_finite()
        || previous_norm == T::Real::zero()
        || next_norm == T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair pseudo-arclength",
            reason: "branch secant must be finite and non-zero".into(),
        });
    }
    let dot = previous_parameter_delta * next_parameter_delta
        + (previous_eigenvalue_delta.conj() * next_eigenvalue_delta).re();
    if dot < T::Real::zero() {
        Ok(EigenContinuationEvent::BranchReversal)
    } else if previous_parameter_delta * next_parameter_delta <= T::Real::zero() {
        Ok(EigenContinuationEvent::TurningPoint)
    } else {
        Ok(EigenContinuationEvent::Regular)
    }
}

fn validate_continuation_point<T: ComplexField>(
    point: &EigenContinuationPoint<T>,
    rows: usize,
    cols: usize,
) -> Result<(), SolverError> {
    if !point.parameter.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair continuation",
            reason: "continuation point parameter is NaN or infinity".into(),
        });
    }
    validate_finite_scalar("generalized eigenpair continuation", point.eigenvalue)?;
    let dimension = point.right_eigenvector.len();
    if point.left_eigenvector.len() != dimension {
        return Err(SolverError::DimensionMismatch {
            expected: dimension,
            got: point.left_eigenvector.len(),
        });
    }
    if rows != dimension || cols != dimension {
        return Err(SolverError::DimensionMismatch {
            expected: dimension,
            got: rows.max(cols),
        });
    }
    validate_finite_slice(
        "generalized eigenpair continuation",
        &point.right_eigenvector,
    )?;
    validate_finite_slice(
        "generalized eigenpair continuation",
        &point.left_eigenvector,
    )
}

fn real_from_f64<T: ComplexField>(
    value: f64,
    operation: &'static str,
) -> Result<T::Real, SolverError> {
    T::Real::from_f64(value).ok_or_else(|| SolverError::OperationFailed {
        operation,
        reason: "continuation configuration is not representable in the scalar type".into(),
    })
}

/// Continue a generalized eigenpair through a fold using a secant
/// pseudo-arclength predictor.
///
/// The corrector owns the nonlinear/eigensolver work and may return a point
/// whose parameter differs from the predictor. Callback failures and panics
/// are retried with the same signed shrink policy as
/// [`continue_generalized_eigenpair`]. The returned event is computed from
/// the previous, current, and corrected points, allowing callers to stop or
/// switch branches at a detected turning point.
#[allow(clippy::too_many_arguments)]
pub fn continue_generalized_eigenpair_pseudo_arclength<T, B, C>(
    previous: &EigenContinuationPoint<T>,
    current: &EigenContinuationPoint<T>,
    requested_step: T::Real,
    mass: &B,
    config: &EigenContinuationConfig,
    corrector: C,
) -> Result<EigenPseudoArclengthStep<T>, SolverError>
where
    T: ComplexField,
    B: LinearOperator<T>,
    C: Fn(&EigenPseudoArclengthPrediction<T>) -> Result<EigenContinuationPoint<T>, SolverError>,
{
    let (min_step, max_step, shrink) = validate_continuation_config::<T>(config)?;
    if !requested_step.is_finite() || requested_step == T::Real::zero() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair pseudo-arclength",
            reason: "requested step must be finite and non-zero".into(),
        });
    }
    validate_continuation_point(previous, mass.num_rows(), mass.num_cols())?;
    validate_continuation_point(current, mass.num_rows(), mass.num_cols())?;
    if previous.right_eigenvector.len() != current.right_eigenvector.len() {
        return Err(SolverError::DimensionMismatch {
            expected: previous.right_eigenvector.len(),
            got: current.right_eigenvector.len(),
        });
    }
    let delta_parameter = current.parameter - previous.parameter;
    let delta_eigenvalue = current.eigenvalue - previous.eigenvalue;
    let secant_norm = (delta_parameter * delta_parameter + delta_eigenvalue.norm_sqr()).sqrt();
    if !secant_norm.is_finite() || secant_norm == T::Real::zero() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair pseudo-arclength",
            reason: "previous and current points must define a finite non-zero secant".into(),
        });
    }
    let magnitude = requested_step.abs().max(min_step).min(max_step);
    let mut step = if requested_step < T::Real::zero() {
        -magnitude
    } else {
        magnitude
    };
    let mut retries = 0;
    loop {
        let secant_scale = step / secant_norm;
        let scale = T::from_real(secant_scale);
        let parameter = current.parameter + secant_scale * delta_parameter;
        let eigenvalue = current.eigenvalue + scale * delta_eigenvalue;
        let mut right_eigenvector = current.right_eigenvector.clone();
        let vector_delta: Vec<T> = current
            .right_eigenvector
            .iter()
            .zip(&previous.right_eigenvector)
            .map(|(current_value, previous_value)| *current_value - *previous_value)
            .collect();
        T::vec_axpy(scale, &vector_delta, &mut right_eigenvector);
        validate_finite_scalar("generalized eigenpair pseudo-arclength", eigenvalue)?;
        validate_finite_slice("generalized eigenpair pseudo-arclength", &right_eigenvector)?;
        if !parameter.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "generalized eigenpair pseudo-arclength",
                reason: "predicted continuation parameter NaN or infinity".into(),
            });
        }
        let prediction = EigenPseudoArclengthPrediction {
            parameter,
            eigenvalue,
            right_eigenvector,
            step_size: step,
            tangent_parameter: delta_parameter / secant_norm,
            tangent_eigenvalue: delta_eigenvalue / T::from_real(secant_norm),
        };
        let corrected_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| corrector(&prediction)))
                .map_err(|payload| SolverError::OperationFailed {
                    operation: "generalized eigenpair pseudo-arclength corrector",
                    reason: if let Some(message) = payload.downcast_ref::<&'static str>() {
                        (*message).to_string()
                    } else if let Some(message) = payload.downcast_ref::<String>() {
                        message.clone()
                    } else {
                        "continuation corrector panicked".to_string()
                    },
                })?;
        match corrected_result {
            Ok(corrected_point) => {
                validate_continuation_point(&corrected_point, mass.num_rows(), mass.num_cols())?;
                let event = classify_eigen_continuation_event(previous, current, &corrected_point)?;
                return Ok(EigenPseudoArclengthStep {
                    prediction,
                    corrected_point,
                    retries,
                    event,
                });
            }
            Err(error) => {
                let next_magnitude = step.abs() * shrink;
                if retries >= config.max_retries || next_magnitude < min_step {
                    return Err(error);
                }
                step = if step < T::Real::zero() {
                    -next_magnitude
                } else {
                    next_magnitude
                };
                retries += 1;
            }
        }
    }
}

/// Advance one generalized eigenpair with a bounded adaptive corrector loop.
///
/// `corrector` receives each first-order prediction and owns the problem-
/// specific eigenvalue solve at the predicted parameter. A callback error or
/// panic causes the signed step to shrink by `shrink_factor` and retry until
/// `max_retries` or `min_step_size` is reached. Successful corrections must
/// return finite, dimensionally compatible vectors at exactly the predicted
/// parameter; malformed corrected points are rejected without being retried.
/// This contract intentionally leaves pseudo-arclength and branch/bifurcation
/// policies to a higher-level controller.
pub fn continue_generalized_eigenpair<T, B, DB, DA, S, C>(
    point: &EigenContinuationPoint<T>,
    requested_step: T::Real,
    mass: &B,
    mass_derivative: Option<&DB>,
    operator_derivative: &DA,
    solve_singular: S,
    config: &EigenContinuationConfig,
    corrector: C,
) -> Result<EigenContinuationStep<T>, SolverError>
where
    T: ComplexField,
    B: LinearOperator<T>,
    DB: LinearOperator<T>,
    DA: LinearOperator<T>,
    S: Fn(&[T]) -> Result<Vec<T>, SolverError>,
    C: Fn(&EigenContinuationPrediction<T>) -> Result<EigenContinuationPoint<T>, SolverError>,
{
    if !config.min_step_size.is_finite()
        || !config.max_step_size.is_finite()
        || !config.shrink_factor.is_finite()
        || config.min_step_size <= 0.0
        || config.max_step_size < config.min_step_size
        || config.shrink_factor <= 0.0
        || config.shrink_factor >= 1.0
    {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair continuation config",
            reason: "step bounds must be finite and positive with 0 < shrink_factor < 1".into(),
        });
    }
    if !requested_step.is_finite() || requested_step == T::Real::zero() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair continuation",
            reason: "requested step must be finite and non-zero".into(),
        });
    }
    validate_continuation_point(point, mass.num_rows(), mass.num_cols())?;

    let min_step = real_from_f64::<T>(
        config.min_step_size,
        "generalized eigenpair continuation config",
    )?;
    let max_step = real_from_f64::<T>(
        config.max_step_size,
        "generalized eigenpair continuation config",
    )?;
    let shrink = real_from_f64::<T>(
        config.shrink_factor,
        "generalized eigenpair continuation config",
    )?;
    let magnitude = requested_step.abs().max(min_step).min(max_step);
    let mut step = if requested_step < T::Real::zero() {
        -magnitude
    } else {
        magnitude
    };
    let mut retries = 0;

    loop {
        let prediction = predict_generalized_eigenpair(
            point,
            step,
            mass,
            mass_derivative,
            operator_derivative,
            &solve_singular,
        )?;
        let corrected_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| corrector(&prediction)))
                .map_err(|payload| SolverError::OperationFailed {
                    operation: "generalized eigenpair continuation corrector",
                    reason: if let Some(message) = payload.downcast_ref::<&'static str>() {
                        (*message).to_string()
                    } else if let Some(message) = payload.downcast_ref::<String>() {
                        message.clone()
                    } else {
                        "continuation corrector panicked".to_string()
                    },
                })?;

        match corrected_result {
            Ok(corrected_point) => {
                validate_continuation_point(&corrected_point, mass.num_rows(), mass.num_cols())?;
                if corrected_point.parameter != prediction.parameter {
                    return Err(SolverError::OperationFailed {
                        operation: "generalized eigenpair continuation corrector",
                        reason: "corrected point parameter does not match prediction".into(),
                    });
                }
                return Ok(EigenContinuationStep {
                    prediction,
                    corrected_point,
                    retries,
                });
            }
            Err(error) => {
                let next_magnitude = step.abs() * shrink;
                if retries >= config.max_retries || next_magnitude < min_step {
                    return Err(error);
                }
                step = if step < T::Real::zero() {
                    -next_magnitude
                } else {
                    next_magnitude
                };
                retries += 1;
            }
        }
    }
}

/// Predict one generalized eigenpair along a scalar continuation parameter.
///
/// This is deliberately a predictor-only contract. It computes analytic
/// first-order tangents from matrix-free `dA`/`dB` operators and advances the
/// supplied point by `step_size`; a caller may then pass the prediction to a
/// problem-specific eigensolver as a corrector. A zero, non-finite, or
/// dimensionally invalid step is rejected before the singular-solve callback
/// is invoked. Adaptive step control, pseudo-arclength, and branch/bifurcation
/// tracking remain higher-level policies.
#[allow(clippy::too_many_arguments)]
pub fn predict_generalized_eigenpair<T, B, DB, DA, S>(
    point: &EigenContinuationPoint<T>,
    step_size: T::Real,
    mass: &B,
    mass_derivative: Option<&DB>,
    operator_derivative: &DA,
    solve_singular: S,
) -> Result<EigenContinuationPrediction<T>, SolverError>
where
    T: ComplexField,
    B: LinearOperator<T>,
    DB: LinearOperator<T>,
    DA: LinearOperator<T>,
    S: Fn(&[T]) -> Result<Vec<T>, SolverError>,
{
    if !point.parameter.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair continuation",
            reason: "continuation parameter is NaN or infinity".into(),
        });
    }
    if !step_size.is_finite() || step_size == T::Real::zero() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair continuation",
            reason: "step size must be finite and non-zero".into(),
        });
    }

    let eigenvalue_sensitivity = generalized_eigenvalue_sensitivity(
        point.eigenvalue,
        &point.right_eigenvector,
        &point.left_eigenvector,
        mass,
        mass_derivative,
        operator_derivative,
    )?;
    let eigenvector_sensitivity = generalized_eigenvector_sensitivity(
        point.eigenvalue,
        &point.right_eigenvector,
        &point.left_eigenvector,
        mass,
        mass_derivative,
        operator_derivative,
        solve_singular,
    )?;

    let step = T::from_real(step_size);
    let eigenvalue = point.eigenvalue + step * eigenvalue_sensitivity;
    let mut right_eigenvector = point.right_eigenvector.clone();
    T::vec_axpy(step, &eigenvector_sensitivity, &mut right_eigenvector);
    validate_finite_scalar("generalized eigenpair continuation", eigenvalue)?;
    validate_finite_slice("generalized eigenpair continuation", &right_eigenvector)?;

    let parameter = point.parameter + step_size;
    if !parameter.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenpair continuation",
            reason: "predicted continuation parameter is NaN or infinity".into(),
        });
    }

    Ok(EigenContinuationPrediction {
        parameter,
        eigenvalue,
        right_eigenvector,
        eigenvalue_sensitivity,
        eigenvector_sensitivity,
        step_size,
    })
}

/// Compute the first-order sensitivity of a simple generalized eigenvalue.
///
/// For `A v = λ B v` and a left eigenvector `w`, this evaluates
///
/// `dλ = wᴴ (dA - λ dB) v / (wᴴ B v)`.
///
/// The derivative operators are matrix-free and may be backed by sparse,
/// device, or assembled operators. `mass_derivative = None` means `dB = 0`.
/// The denominator is checked before division so a defective or improperly
/// normalized eigenpair returns a typed error instead of a non-finite value.
#[allow(clippy::too_many_arguments)]
pub fn generalized_eigenvalue_sensitivity<T, B, DB, DA>(
    eigenvalue: T,
    right_eigenvector: &[T],
    left_eigenvector: &[T],
    mass: &B,
    mass_derivative: Option<&DB>,
    operator_derivative: &DA,
) -> Result<T, SolverError>
where
    T: ComplexField,
    B: LinearOperator<T>,
    DB: LinearOperator<T>,
    DA: LinearOperator<T>,
{
    validate_finite_scalar("generalized eigenvalue sensitivity", eigenvalue)?;
    validate_finite_slice("generalized eigenvalue sensitivity", right_eigenvector)?;
    validate_finite_slice("generalized eigenvalue sensitivity", left_eigenvector)?;
    let n = right_eigenvector.len();
    if left_eigenvector.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: left_eigenvector.len(),
        });
    }
    if mass.num_cols() != n || mass.num_rows() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: mass.num_cols().max(mass.num_rows()),
        });
    }
    if operator_derivative.num_cols() != n || operator_derivative.num_rows() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: operator_derivative
                .num_cols()
                .max(operator_derivative.num_rows()),
        });
    }
    if let Some(mass_derivative) = mass_derivative
        && (mass_derivative.num_cols() != n || mass_derivative.num_rows() != n)
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: mass_derivative.num_cols().max(mass_derivative.num_rows()),
        });
    }

    let mass_right = mass.try_apply(right_eigenvector)?;
    let denominator = T::vec_dot(left_eigenvector, &mass_right);
    if denominator.norm_sqr() <= T::Real::epsilon() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenvalue sensitivity",
            reason: "left/right eigenvectors have zero B-inner product".to_string(),
        });
    }
    let operator_right = operator_derivative.try_apply(right_eigenvector)?;
    let mut numerator = T::vec_dot(left_eigenvector, &operator_right);
    if let Some(mass_derivative) = mass_derivative {
        let mass_derivative_right = mass_derivative.try_apply(right_eigenvector)?;
        numerator -= eigenvalue * T::vec_dot(left_eigenvector, &mass_derivative_right);
    }
    let sensitivity = numerator / denominator;
    validate_finite_scalar("generalized eigenvalue sensitivity", sensitivity)?;
    Ok(sensitivity)
}

/// Compute a gauge-fixed first-order sensitivity of a generalized right
/// eigenvector. `solve_singular` must solve the bordered/reduced system
/// `(A - lambda B) dv = rhs`; the returned vector is projected to satisfy
/// `wᴴ B dv = 0`, which fixes the otherwise arbitrary eigenvector scaling.
#[allow(clippy::too_many_arguments)]
pub fn generalized_eigenvector_sensitivity<T, B, DB, DA, S>(
    eigenvalue: T,
    right_eigenvector: &[T],
    left_eigenvector: &[T],
    mass: &B,
    mass_derivative: Option<&DB>,
    operator_derivative: &DA,
    solve_singular: S,
) -> Result<Vec<T>, SolverError>
where
    T: ComplexField,
    B: LinearOperator<T>,
    DB: LinearOperator<T>,
    DA: LinearOperator<T>,
    S: Fn(&[T]) -> Result<Vec<T>, SolverError>,
{
    validate_finite_scalar("generalized eigenvector sensitivity", eigenvalue)?;
    validate_finite_slice("generalized eigenvector sensitivity", right_eigenvector)?;
    validate_finite_slice("generalized eigenvector sensitivity", left_eigenvector)?;
    let n = right_eigenvector.len();
    if left_eigenvector.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: left_eigenvector.len(),
        });
    }
    for (name, rows, cols) in [
        ("mass", mass.num_rows(), mass.num_cols()),
        (
            "operator derivative",
            operator_derivative.num_rows(),
            operator_derivative.num_cols(),
        ),
    ] {
        if rows != n || cols != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: rows.max(cols),
            });
        }
        let _ = name;
    }
    if let Some(derivative) = mass_derivative
        && (derivative.num_rows() != n || derivative.num_cols() != n)
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: derivative.num_rows().max(derivative.num_cols()),
        });
    }

    let mass_right = mass.try_apply(right_eigenvector)?;
    let denominator = T::vec_dot(left_eigenvector, &mass_right);
    if denominator.norm_sqr() <= T::Real::epsilon() {
        return Err(SolverError::OperationFailed {
            operation: "generalized eigenvector sensitivity",
            reason: "left/right B-inner product is zero or numerically singular".to_string(),
        });
    }
    let operator_right = operator_derivative.try_apply(right_eigenvector)?;
    let mass_derivative_right = if let Some(derivative) = mass_derivative {
        derivative.try_apply(right_eigenvector)?
    } else {
        vec![T::zero(); n]
    };
    let eigenvalue_sensitivity = {
        let mut numerator = T::vec_dot(left_eigenvector, &operator_right);
        numerator -= eigenvalue * T::vec_dot(left_eigenvector, &mass_derivative_right);
        numerator / denominator
    };
    let mass_term = mass.try_apply(right_eigenvector)?;
    let mut rhs = operator_right;
    for ((value, mass_derivative_value), mass_value) in rhs
        .iter_mut()
        .zip(mass_derivative_right.iter())
        .zip(mass_term.iter())
    {
        *value =
            -*value + eigenvalue * *mass_derivative_value + eigenvalue_sensitivity * *mass_value;
    }
    let mut derivative =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solve_singular(&rhs))).map_err(
            |payload| SolverError::OperationFailed {
                operation: "generalized eigenvector sensitivity solve",
                reason: if let Some(message) = payload.downcast_ref::<&'static str>() {
                    (*message).to_string()
                } else if let Some(message) = payload.downcast_ref::<String>() {
                    message.clone()
                } else {
                    "eigenvector sensitivity solve panicked".to_string()
                },
            },
        )??;
    if derivative.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: derivative.len(),
        });
    }
    validate_finite_slice("generalized eigenvector sensitivity solve", &derivative)?;
    let mass_derivative_state = mass.try_apply(&derivative)?;
    let gauge = T::vec_dot(left_eigenvector, &mass_derivative_state) / denominator;
    for (value, eigenvector_value) in derivative.iter_mut().zip(right_eigenvector) {
        *value -= gauge * *eigenvector_value;
    }
    validate_finite_slice("generalized eigenvector sensitivity", &derivative)?;
    Ok(derivative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn generalized_eigenvector_sensitivity_is_gauge_fixed() {
        let mass = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 1.0)]);
        let derivative = CsrMatrix::from_triplets(2, 2, vec![(1, 0, 1.0)]);
        let sensitivity = generalized_eigenvector_sensitivity(
            2.0,
            &[1.0, 0.0],
            &[1.0, 0.0],
            &mass,
            None::<&CsrMatrix<f64>>,
            &derivative,
            |rhs: &[f64]| Ok(vec![0.0, rhs[1]]),
        )
        .expect("reduced eigenvector sensitivity solve should succeed");
        assert!((sensitivity[0]).abs() < 1.0e-12);
        assert!((sensitivity[1] + 1.0).abs() < 1.0e-12);
    }
    use num_complex::Complex64;

    #[test]
    fn generalized_sensitivity_matches_mass_and_operator_derivatives() {
        let mass = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 2.0)]);
        let mass_derivative = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 0.5)]);
        let operator_derivative = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 3.0), (1, 1, 7.0)]);
        let sensitivity = generalized_eigenvalue_sensitivity(
            2.0,
            &[1.0, 0.0],
            &[1.0, 0.0],
            &mass,
            Some(&mass_derivative),
            &operator_derivative,
        )
        .expect("simple generalized eigenpair should be differentiable");
        assert!((sensitivity - 1.0_f64).abs() < 1.0e-12);
    }

    #[test]
    fn complex_sensitivity_preserves_hermitian_inner_product() {
        let mass = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(2.0, 0.0)),
                (1, 1, Complex64::new(1.0, 0.0)),
            ],
        );
        let derivative = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(1.0, 2.0)),
                (1, 1, Complex64::new(4.0, 0.0)),
            ],
        );
        let sensitivity = generalized_eigenvalue_sensitivity(
            Complex64::new(3.0, 0.0),
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &mass,
            None::<&CsrMatrix<Complex64>>,
            &derivative,
        )
        .expect("complex generalized eigenpair should be differentiable");
        assert!((sensitivity - Complex64::new(0.5, 1.0)).norm() < 1.0e-12);
    }

    #[test]
    fn zero_generalized_norm_is_rejected() {
        let mass = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.0)]);
        let derivative = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.0)]);
        let error = generalized_eigenvalue_sensitivity(
            1.0,
            &[1.0],
            &[0.0],
            &mass,
            None::<&CsrMatrix<f64>>,
            &derivative,
        )
        .expect_err("zero B-inner product must be rejected");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }
}
