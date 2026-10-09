//! Validated risk and inequality-aggregation primitives.
//!
//! These helpers are intentionally solver-independent: callers can use the
//! returned weights to assemble objective/constraint gradients for MMA,
//! SQP, trust-region, or a PDE-constrained outer loop. All inputs are checked
//! before sorting or evaluating transcendental functions.

use std::cell::RefCell;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::types::{OptimError, OptimResult, RealScalar};
use crate::unconstrained::{BfgsConfig, bfgs};
use sonium_solvers_core::traits::{
    AccuracyStatus, SolverError, SolverErrorKind, SolverFailure, SolverResultContract, SolverStatus,
};

/// Errors raised by risk/aggregation helpers before an optimizer iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskAggregationError {
    /// The sample or aggregation input contains no values.
    EmptyInput,
    /// A confidence level is not strictly inside `(0, 1)`.
    InvalidConfidence,
    /// A KS aggregation sharpness is not finite and positive.
    InvalidSharpness,
    /// A probability is outside the closed unit interval.
    InvalidProbability,
    /// A value or gradient contains NaN or infinity.
    NonFiniteInput,
    /// Gradient rows do not match the sample count or dimension.
    DimensionMismatch { expected: usize, got: usize },
    /// Aggregated gradient does not match the design-vector dimension.
    DesignDimensionMismatch { expected: usize, got: usize },
    /// The scenario evaluator panicked while producing a sample packet.
    CallbackPanicked,
}

impl SolverFailure for RiskAggregationError {
    fn error_kind(&self) -> SolverErrorKind {
        match self {
            Self::DimensionMismatch { .. } | Self::DesignDimensionMismatch { .. } => {
                SolverErrorKind::DimensionMismatch
            }
            Self::EmptyInput
            | Self::InvalidConfidence
            | Self::InvalidSharpness
            | Self::InvalidProbability
            | Self::NonFiniteInput
            | Self::CallbackPanicked => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        SolverStatus::NumericalError
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }
}

impl fmt::Display for RiskAggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "risk aggregation requires at least one value"),
            Self::InvalidConfidence => write!(f, "confidence must be finite and in (0, 1)"),
            Self::InvalidSharpness => write!(f, "KS sharpness must be finite and positive"),
            Self::InvalidProbability => write!(f, "probability must be finite and in [0, 1]"),
            Self::NonFiniteInput => write!(f, "risk aggregation input contains NaN or infinity"),
            Self::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "risk gradient dimension mismatch: expected {expected}, got {got}"
                )
            }
            Self::DesignDimensionMismatch { expected, got } => write!(
                f,
                "risk objective gradient dimension mismatch: expected {expected}, got {got}"
            ),
            Self::CallbackPanicked => write!(f, "risk scenario evaluator panicked"),
        }
    }
}

impl std::error::Error for RiskAggregationError {}

/// Select the risk functional used by [`RiskObjective`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskAggregation<T: RealScalar> {
    /// Empirical conditional value at risk at the supplied confidence.
    Cvar { confidence: T },
    /// Kreisselmeier–Steinhauser smooth maximum with positive sharpness.
    Ks { sharpness: T },
}

/// Scenario losses and parameter gradients returned by a risk objective
/// evaluator. Rows in `gradients` correspond one-to-one with `losses`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioEvaluation<T: RealScalar> {
    pub losses: Vec<T>,
    pub gradients: Vec<Vec<T>>,
}

impl<T: RealScalar> ScenarioEvaluation<T> {
    pub fn new(losses: Vec<T>, gradients: Vec<Vec<T>>) -> Self {
        Self { losses, gradients }
    }
}

/// Aggregated objective value, gradient, and normalized scenario weights.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskObjectiveEvaluation<T: RealScalar> {
    pub value: T,
    pub gradient: Vec<T>,
    pub weights: Vec<T>,
}

#[derive(Debug, Clone)]
struct CachedRiskEvaluation<T: RealScalar> {
    x: Vec<T>,
    evaluation: RiskObjectiveEvaluation<T>,
}

/// Checked scenario objective adapter for PDE- or simulation-backed
/// optimization. A successful evaluation is cached by design point so an
/// optimizer's separate value and gradient callbacks share one scenario run.
pub struct RiskObjective<T: RealScalar, F> {
    evaluator: F,
    aggregation: RiskAggregation<T>,
    cache: RefCell<Option<CachedRiskEvaluation<T>>>,
    last_error: RefCell<Option<RiskAggregationError>>,
}

impl<T: RealScalar, F> RiskObjective<T, F>
where
    F: Fn(&[T]) -> Result<ScenarioEvaluation<T>, RiskAggregationError>,
{
    pub fn new(evaluator: F, aggregation: RiskAggregation<T>) -> Self {
        Self {
            evaluator,
            aggregation,
            cache: RefCell::new(None),
            last_error: RefCell::new(None),
        }
    }

    pub fn aggregation(&self) -> RiskAggregation<T> {
        self.aggregation
    }

    pub fn evaluate(&self, x: &[T]) -> Result<RiskObjectiveEvaluation<T>, RiskAggregationError> {
        if x.iter().any(|value| !value.is_finite()) {
            return self.fail(RiskAggregationError::NonFiniteInput);
        }
        if let Some(cached) = self.cache.borrow().as_ref()
            && cached.x == x
        {
            return Ok(cached.evaluation.clone());
        }

        let scenarios = match catch_unwind(AssertUnwindSafe(|| (self.evaluator)(x))) {
            Ok(result) => result,
            Err(_) => return self.fail(RiskAggregationError::CallbackPanicked),
        };
        let scenarios = match scenarios {
            Ok(scenarios) => scenarios,
            Err(error) => return self.fail(error),
        };
        let (value, gradient, weights) = match self.aggregation {
            RiskAggregation::Cvar { confidence } => {
                let (result, gradient) = match empirical_cvar_with_gradients(
                    &scenarios.losses,
                    &scenarios.gradients,
                    confidence,
                ) {
                    Ok(value) => value,
                    Err(error) => return self.fail(error),
                };
                (result.value, gradient, result.weights)
            }
            RiskAggregation::Ks { sharpness } => {
                let (result, gradient) = match ks_aggregate_with_gradients(
                    &scenarios.losses,
                    &scenarios.gradients,
                    sharpness,
                ) {
                    Ok(value) => value,
                    Err(error) => return self.fail(error),
                };
                (result.value, gradient, result.weights)
            }
        };
        if gradient.len() != x.len() {
            return self.fail(RiskAggregationError::DesignDimensionMismatch {
                expected: x.len(),
                got: gradient.len(),
            });
        }
        let evaluation = RiskObjectiveEvaluation {
            value,
            gradient,
            weights,
        };
        *self.cache.borrow_mut() = Some(CachedRiskEvaluation {
            x: x.to_vec(),
            evaluation: evaluation.clone(),
        });
        Ok(evaluation)
    }

    pub fn value(&self, x: &[T]) -> Result<T, RiskAggregationError> {
        self.evaluate(x).map(|evaluation| evaluation.value)
    }

    pub fn gradient(&self, x: &[T]) -> Result<Vec<T>, RiskAggregationError> {
        self.evaluate(x).map(|evaluation| evaluation.gradient)
    }

    fn fail<U>(&self, error: RiskAggregationError) -> Result<U, RiskAggregationError> {
        *self.last_error.borrow_mut() = Some(error.clone());
        Err(error)
    }

    fn take_last_error(&self) -> Option<RiskAggregationError> {
        self.last_error.borrow_mut().take()
    }
}

fn risk_error_to_optim(error: RiskAggregationError) -> OptimError {
    match error {
        RiskAggregationError::DimensionMismatch { expected, got }
        | RiskAggregationError::DesignDimensionMismatch { expected, got } => {
            OptimError::DimensionMismatch { expected, got }
        }
        RiskAggregationError::CallbackPanicked => OptimError::CallbackPanicked {
            iteration: 0,
            stage: "risk scenario evaluator".into(),
        },
        other => OptimError::NumericalError {
            iteration: 0,
            message: other.to_string(),
        },
    }
}

/// Run BFGS on a checked scenario objective using CVaR or KS aggregation.
///
/// The adapter performs one checked preflight, shares cached scenario
/// evaluations between objective and gradient callbacks, and maps callback or
/// aggregation failures into the optimizer's canonical error vocabulary.
pub fn bfgs_risk<T, F>(
    evaluator: F,
    x0: &[T],
    config: &BfgsConfig<T>,
    aggregation: RiskAggregation<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> Result<ScenarioEvaluation<T>, RiskAggregationError>,
{
    let objective = RiskObjective::new(evaluator, aggregation);
    objective.evaluate(x0).map_err(risk_error_to_optim)?;
    let nan = T::nan();
    let value = |x: &[T]| objective.value(x).unwrap_or(nan);
    let gradient = |x: &[T]| objective.gradient(x).unwrap_or_else(|_| vec![nan; x.len()]);
    let result = bfgs(&value, &gradient, x0, config);
    if let Some(error) = objective.take_last_error() {
        return Err(risk_error_to_optim(error));
    }
    result
}

/// Empirical CVaR value and normalized scenario weights.
#[derive(Debug, Clone, PartialEq)]
pub struct CvarResult<T: RealScalar> {
    /// Mean loss in the worst `(1 - confidence)` tail.
    pub value: T,
    /// Empirical VaR threshold at the requested confidence.
    pub threshold: T,
    /// Effective number of equally weighted samples in the tail.
    pub tail_mass: T,
    /// Weights aligned with the original `losses` order; they sum to one.
    pub weights: Vec<T>,
}

/// Smooth maximum (Kreisselmeier–Steinhauser) value and scenario weights.
#[derive(Debug, Clone, PartialEq)]
pub struct KsResult<T: RealScalar> {
    /// `max(values)`-approximating aggregate.
    pub value: T,
    /// Softmax weights aligned with the original value order.
    pub weights: Vec<T>,
}

fn validate_normalized_weights<T: RealScalar>(weights: &[T]) -> Result<(), &'static str> {
    if weights.is_empty() {
        return Err("scenario weights are empty");
    }

    let mut sum = T::zero();
    for &weight in weights {
        if !weight.is_finite() {
            return Err("scenario weights contain a non-finite value");
        }
        if weight < T::zero() {
            return Err("scenario weights contain a negative value");
        }
        sum += weight;
    }
    if !sum.is_finite() || sum <= T::zero() {
        return Err("scenario weights have an invalid sum");
    }

    // Summation error grows with the number of scenarios. Keep this tight
    // enough to reject malformed packets while accepting f32/f64 reductions.
    let count = T::from_usize(weights.len()).unwrap_or(T::one());
    let scale = T::from_f64(32.0).unwrap_or(T::one());
    let tolerance = T::machine_epsilon() * count * scale;
    if (sum - T::one()).abs() > tolerance {
        return Err("scenario weights are not normalized");
    }
    Ok(())
}

fn contract_error(operation: &'static str, reason: &'static str) -> SolverError {
    SolverError::OperationFailed {
        operation,
        reason: reason.into(),
    }
}

impl<T: RealScalar> CvarResult<T> {
    fn validation_reason(&self) -> Result<(), &'static str> {
        if !self.value.is_finite() {
            return Err("CVaR value is non-finite");
        }
        if !self.threshold.is_finite() {
            return Err("CVaR threshold is non-finite");
        }
        if !self.tail_mass.is_finite() || self.tail_mass <= T::zero() {
            return Err("CVaR tail mass is invalid");
        }
        let count = T::from_usize(self.weights.len()).unwrap_or(T::one());
        if self.tail_mass > count {
            return Err("CVaR tail mass exceeds scenario count");
        }
        validate_normalized_weights(&self.weights)
    }

    pub fn solver_status(&self) -> SolverStatus {
        if self.validation_reason().is_ok() {
            SolverStatus::Converged
        } else {
            SolverStatus::NumericalError
        }
    }

    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: RealScalar> SolverResultContract for CvarResult<T> {
    fn solver_status(&self) -> SolverStatus {
        CvarResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        CvarResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        self.validation_reason()
            .map_err(|reason| contract_error("CVaR result contract", reason))
    }
}

impl<T: RealScalar> KsResult<T> {
    fn validation_reason(&self) -> Result<(), &'static str> {
        if !self.value.is_finite() {
            return Err("KS value is non-finite");
        }
        validate_normalized_weights(&self.weights)
    }

    pub fn solver_status(&self) -> SolverStatus {
        if self.validation_reason().is_ok() {
            SolverStatus::Converged
        } else {
            SolverStatus::NumericalError
        }
    }

    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: RealScalar> SolverResultContract for KsResult<T> {
    fn solver_status(&self) -> SolverStatus {
        KsResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        KsResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        self.validation_reason()
            .map_err(|reason| contract_error("KS result contract", reason))
    }
}

impl<T: RealScalar> RiskObjectiveEvaluation<T> {
    fn validation_reason(&self) -> Result<(), &'static str> {
        if !self.value.is_finite() {
            return Err("risk objective value is non-finite");
        }
        if self.gradient.iter().any(|value| !value.is_finite()) {
            return Err("risk objective gradient contains a non-finite value");
        }
        validate_normalized_weights(&self.weights)
    }

    pub fn solver_status(&self) -> SolverStatus {
        if self.validation_reason().is_ok() {
            SolverStatus::Converged
        } else {
            SolverStatus::NumericalError
        }
    }

    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: RealScalar> SolverResultContract for RiskObjectiveEvaluation<T> {
    fn solver_status(&self) -> SolverStatus {
        RiskObjectiveEvaluation::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        RiskObjectiveEvaluation::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        self.validation_reason()
            .map_err(|reason| contract_error("risk objective evaluation contract", reason))
    }
}

fn validate_values<T: RealScalar>(values: &[T]) -> Result<(), RiskAggregationError> {
    if values.is_empty() {
        return Err(RiskAggregationError::EmptyInput);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(RiskAggregationError::NonFiniteInput);
    }
    Ok(())
}

fn validate_gradients<T: RealScalar>(
    values: &[T],
    gradients: &[Vec<T>],
) -> Result<usize, RiskAggregationError> {
    if gradients.len() != values.len() {
        return Err(RiskAggregationError::DimensionMismatch {
            expected: values.len(),
            got: gradients.len(),
        });
    }
    let dimension = gradients.first().map_or(0, Vec::len);
    if gradients
        .iter()
        .any(|gradient| gradient.iter().any(|value| !value.is_finite()))
    {
        return Err(RiskAggregationError::NonFiniteInput);
    }
    if gradients.iter().any(|gradient| gradient.len() != dimension) {
        return Err(RiskAggregationError::DimensionMismatch {
            expected: dimension,
            got: gradients
                .iter()
                .find(|gradient| gradient.len() != dimension)
                .map_or(dimension, Vec::len),
        });
    }
    Ok(dimension)
}

fn validate_confidence<T: RealScalar>(confidence: T) -> Result<(), RiskAggregationError> {
    if !confidence.is_finite() || confidence <= T::zero() || confidence >= T::one() {
        Err(RiskAggregationError::InvalidConfidence)
    } else {
        Ok(())
    }
}

/// Compute empirical CVaR with exact fractional weighting at the tail edge.
///
/// `confidence = 0.95` means that the worst five percent of losses are
/// averaged. If the tail cuts through a sample, that sample receives the
/// corresponding fractional weight instead of being rounded away.
pub fn empirical_cvar<T: RealScalar>(
    losses: &[T],
    confidence: T,
) -> Result<CvarResult<T>, RiskAggregationError> {
    validate_values(losses)?;
    validate_confidence(confidence)?;
    let n = losses.len();
    let n_t = T::from_usize(n).ok_or(RiskAggregationError::NonFiniteInput)?;
    let tail_mass = (T::one() - confidence) * n_t;
    if !tail_mass.is_finite() || tail_mass <= T::zero() {
        return Err(RiskAggregationError::InvalidConfidence);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| losses[a].partial_cmp(&losses[b]).unwrap());
    let quantile_rank = (confidence * n_t).ceil();
    let quantile_index = quantile_rank
        .to_usize()
        .unwrap_or(n)
        .saturating_sub(1)
        .min(n - 1);
    let threshold = losses[order[quantile_index]];

    let mut weights = vec![T::zero(); n];
    let mut remaining = tail_mass;
    let mut weighted_sum = T::zero();
    for &index in order.iter().rev() {
        if remaining <= T::zero() {
            break;
        }
        let weight = T::min(T::one(), remaining);
        weights[index] = weight / tail_mass;
        weighted_sum += weight * losses[index];
        remaining -= weight;
    }
    let result = CvarResult {
        value: weighted_sum / tail_mass,
        threshold,
        tail_mass,
        weights,
    };
    result
        .validate_contract()
        .map_err(|_| RiskAggregationError::NonFiniteInput)?;
    Ok(result)
}

/// Compute empirical CVaR and its scenario-weighted gradient.
pub fn empirical_cvar_with_gradients<T: RealScalar>(
    losses: &[T],
    gradients: &[Vec<T>],
    confidence: T,
) -> Result<(CvarResult<T>, Vec<T>), RiskAggregationError> {
    validate_values(losses)?;
    let dimension = validate_gradients(losses, gradients)?;
    let result = empirical_cvar(losses, confidence)?;
    let mut gradient = vec![T::zero(); dimension];
    for (weight, row) in result.weights.iter().zip(gradients) {
        for (target, value) in gradient.iter_mut().zip(row) {
            *target += *weight * *value;
        }
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(RiskAggregationError::NonFiniteInput);
    }
    Ok((result, gradient))
}

/// Compute a numerically stable KS smooth maximum and its softmax weights.
pub fn ks_aggregate<T: RealScalar>(
    values: &[T],
    sharpness: T,
) -> Result<KsResult<T>, RiskAggregationError> {
    validate_values(values)?;
    if !sharpness.is_finite() || sharpness <= T::zero() {
        return Err(RiskAggregationError::InvalidSharpness);
    }
    let max_value = values.iter().copied().fold(T::neg_infinity(), T::max);
    let mut exponentials = Vec::with_capacity(values.len());
    let mut sum = T::zero();
    for &value in values {
        let exponential = (sharpness * (value - max_value)).exp();
        exponentials.push(exponential);
        sum += exponential;
    }
    let log_sum = sum.ln();
    let weights: Vec<T> = exponentials.iter().map(|value| *value / sum).collect();
    let result = KsResult {
        value: max_value + log_sum / sharpness,
        weights,
    };
    result
        .validate_contract()
        .map_err(|_| RiskAggregationError::NonFiniteInput)?;
    Ok(result)
}

/// Compute a KS aggregate and its weighted gradient.
pub fn ks_aggregate_with_gradients<T: RealScalar>(
    values: &[T],
    gradients: &[Vec<T>],
    sharpness: T,
) -> Result<(KsResult<T>, Vec<T>), RiskAggregationError> {
    validate_values(values)?;
    let dimension = validate_gradients(values, gradients)?;
    let result = ks_aggregate(values, sharpness)?;
    let mut gradient = vec![T::zero(); dimension];
    for (weight, row) in result.weights.iter().zip(gradients) {
        for (target, value) in gradient.iter_mut().zip(row) {
            *target += *weight * *value;
        }
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(RiskAggregationError::NonFiniteInput);
    }
    Ok((result, gradient))
}

/// Check an empirical chance constraint `P(loss <= limit) >= 1 - failure_probability`.
pub fn chance_constraint_satisfied<T: RealScalar>(
    losses: &[T],
    limit: T,
    failure_probability: T,
) -> Result<bool, RiskAggregationError> {
    validate_values(losses)?;
    if !limit.is_finite() {
        return Err(RiskAggregationError::NonFiniteInput);
    }
    if !failure_probability.is_finite()
        || failure_probability < T::zero()
        || failure_probability > T::one()
    {
        return Err(RiskAggregationError::InvalidProbability);
    }
    let failures = losses.iter().filter(|&&loss| loss > limit).count();
    let observed = T::from_usize(failures).ok_or(RiskAggregationError::NonFiniteInput)?
        / T::from_usize(losses.len()).ok_or(RiskAggregationError::NonFiniteInput)?;
    Ok(observed <= failure_probability)
}
