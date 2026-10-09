//! Core types for optimization algorithms.
//!
//! Defines the [`RealScalar`] trait, result types, status enums, bound constraints,
//! and general nonlinear constraints shared by all optimizers.

use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::Debug;

use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, SolverError, SolverFailure, SolverResultContract, SolverStatus,
};

// ---------------------------------------------------------------------------
// Scalar trait
// ---------------------------------------------------------------------------

/// Marker trait for real scalar types usable in optimization (f64, f32).
///
/// Unlike [`ComplexField`](crate::ComplexField), this is restricted to
/// real-valued types since optimisation objectives are always real.
///
/// Only `f32`/`f64` implement this trait, so small-constant `from_f64`
/// conversions (2.0, 4.0, step-size floors) used across this crate are
/// infallible in practice. A future `RealScalar` with a narrower range must
/// audit those sites (or centralise them behind a fallible helper) before
/// it can be added.
pub trait RealScalar:
    ComplexField<Real = Self>
    + Float
    + NumAssign
    + FromPrimitive
    + ToPrimitive
    + Send
    + Sync
    + Debug
    + Copy
    + 'static
{
    /// Machine epsilon for this type.
    fn machine_epsilon() -> Self;

    /// `sqrt(machine_epsilon)` -- common tolerance baseline.
    fn sqrt_epsilon() -> Self;
}

impl RealScalar for f64 {
    #[inline]
    fn machine_epsilon() -> Self {
        f64::EPSILON
    }
    #[inline]
    fn sqrt_epsilon() -> Self {
        1.4901161193847656e-8 // f64::EPSILON.sqrt()
    }
}

impl RealScalar for f32 {
    #[inline]
    fn machine_epsilon() -> Self {
        f32::EPSILON
    }
    #[inline]
    fn sqrt_epsilon() -> Self {
        3.4526698e-4 // f32::EPSILON.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Status / Result
// ---------------------------------------------------------------------------

/// Termination status of an optimisation run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimStatus {
    /// Gradient norm below tolerance.
    Converged,
    /// Function value change below tolerance.
    FunctionTolerance,
    /// Step size below tolerance.
    StepTolerance,
    /// Maximum iterations reached without convergence.
    MaxIterationsReached,
    /// Maximum function evaluations reached.
    MaxFunctionEvals,
    /// Line search failed to find acceptable step.
    LineSearchFailed,
    /// Trust region collapsed to zero.
    TrustRegionCollapsed,
    /// Numerical breakdown (NaN, Inf, or singular Hessian).
    NumericalError,
    /// The optimizer's local stopping test fired, but the independently
    /// recomputed original-problem KKT residual did not satisfy its gate.
    KktNotSatisfied,
    /// Feasibility restoration failed to reduce the constraint violation
    /// (the problem is likely infeasible).
    RestorationFailed,
}

/// Result of an optimisation run.
#[derive(Debug, Clone)]
pub struct OptimResult<T: RealScalar> {
    /// Solution vector (minimiser).
    pub x: Vec<T>,
    /// Objective function value at the solution.
    pub fun: T,
    /// Gradient at the solution (if available).
    pub grad: Option<Vec<T>>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of objective function evaluations.
    pub fun_evals: usize,
    /// Number of gradient evaluations.
    pub grad_evals: usize,
    /// Number of Hessian evaluations or Hessian-vector products.
    pub hess_evals: usize,
    /// Final gradient norm (infinity norm).
    pub grad_norm: T,
    /// Whether the optimiser converged.
    pub converged: bool,
    /// Detailed termination status.
    pub status: OptimStatus,
    /// Independently recomputed constrained-optimality report, when a
    /// constrained optimizer supplied its final multipliers.
    pub kkt: Option<KktReport<T>>,
}

impl<T: RealScalar> OptimResult<T> {
    /// Return the canonical solver status for this optimization result.
    ///
    /// Optimization has richer termination reasons than the shared solver
    /// vocabulary. Function/step tolerances are successful convergence;
    /// failed acceptance or KKT gates are stagnation; evaluation/iteration
    /// caps retain the max-iterations meaning. Non-finite public state is
    /// always a numerical error, even if the optimizer-specific status claims
    /// success.
    pub fn solver_status(&self) -> SolverStatus {
        let gradient_finite = match &self.grad {
            Some(gradient) => {
                self.grad_norm.is_finite() && gradient.iter().all(|value| value.is_finite())
            }
            None => true,
        };
        let kkt_finite = match &self.kkt {
            Some(report) => {
                let diagnostics = &report.diagnostics;
                let multipliers = &report.multipliers;
                let ipm_finite = report.ipm.as_ref().is_none_or(|ipm| {
                    ipm.barrier_parameter.is_finite()
                        && ipm.dual_residual.is_finite()
                        && ipm.primal_residual.is_finite()
                        && ipm.complementarity.is_finite()
                });
                report.norm.is_finite()
                    && diagnostics.primal_feasibility.is_finite()
                    && diagnostics.stationarity.is_finite()
                    && diagnostics.dual_feasibility.is_finite()
                    && diagnostics.complementarity.is_finite()
                    && multipliers
                        .constraints
                        .iter()
                        .all(|value| value.is_finite())
                    && multipliers
                        .lower_bounds
                        .iter()
                        .chain(multipliers.upper_bounds.iter())
                        .all(|value| value.is_finite())
                    && ipm_finite
            }
            None => true,
        };

        if !self.fun.is_finite()
            || self.x.iter().any(|value| !value.is_finite())
            || !gradient_finite
            || !kkt_finite
        {
            return SolverStatus::NumericalError;
        }

        match self.status {
            OptimStatus::Converged
            | OptimStatus::FunctionTolerance
            | OptimStatus::StepTolerance => SolverStatus::Converged,
            OptimStatus::MaxIterationsReached | OptimStatus::MaxFunctionEvals => {
                SolverStatus::MaxIterationsReached
            }
            OptimStatus::LineSearchFailed
            | OptimStatus::TrustRegionCollapsed
            | OptimStatus::KktNotSatisfied
            | OptimStatus::RestorationFailed => SolverStatus::Stagnated,
            OptimStatus::NumericalError => SolverStatus::NumericalError,
        }
    }

    /// Alias matching the checked result vocabulary used by other solver
    /// families.
    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    /// Return whether the optimizer's requested accuracy gate was attained.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            SolverStatus::MaxIterationsReached | SolverStatus::Stagnated => {
                AccuracyStatus::NotAttained
            }
            SolverStatus::Breakdown | SolverStatus::NumericalError | SolverStatus::Diverged => {
                AccuracyStatus::NotChecked
            }
        }
    }

    /// Alias matching the checked result vocabulary used by other solver
    /// families.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

/// Reject a result that contains non-finite state before it can be reported as
/// a successful (or otherwise usable) optimisation result.  Optimisers are
/// allowed to return a partial iterate on numerical failure, but the public
/// contract must never expose NaN/Inf as a valid solution or diagnostic.
pub(crate) fn validate_result_finite<T: RealScalar>(
    result: &OptimResult<T>,
) -> Result<(), OptimError> {
    if let Some(index) = result.x.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: result.iterations,
            message: format!("result x contains a non-finite value at index {index}"),
        });
    }
    if !result.fun.is_finite() {
        return Err(OptimError::NumericalError {
            iteration: result.iterations,
            message: "result objective is non-finite".into(),
        });
    }
    if result.grad.is_some() && !result.grad_norm.is_finite() {
        return Err(OptimError::NumericalError {
            iteration: result.iterations,
            message: "result gradient norm is non-finite".into(),
        });
    }
    if let Some(gradient) = result.grad.as_deref()
        && let Some(index) = gradient.iter().position(|value| !value.is_finite())
    {
        return Err(OptimError::NumericalError {
            iteration: result.iterations,
            message: format!("result gradient contains a non-finite value at index {index}"),
        });
    }
    Ok(())
}

/// Validate the shape and finite-value contract of an attached KKT report.
///
/// `solver_status` uses the scalar KKT residuals to classify a result, so a
/// malformed report could otherwise look converged while carrying truncated
/// active-set metadata.  Keep this check independent of the user constraint
/// callbacks: the report is already the optimizer's auditable output.
fn validate_kkt_report<T: RealScalar>(report: &KktReport<T>) -> Result<(), String> {
    let finite = |value: T, name: &str| {
        if value.is_finite() {
            Ok(())
        } else {
            Err(format!("KKT {name} is non-finite"))
        }
    };

    finite(report.norm, "norm")?;
    let diagnostics = &report.diagnostics;
    finite(diagnostics.primal_feasibility, "primal feasibility")?;
    finite(diagnostics.stationarity, "stationarity")?;
    finite(diagnostics.dual_feasibility, "dual feasibility")?;
    finite(diagnostics.complementarity, "complementarity")?;

    for (name, values) in [
        (
            "constraint multiplier",
            report.multipliers.constraints.as_slice(),
        ),
        (
            "lower-bound multiplier",
            report.multipliers.lower_bounds.as_slice(),
        ),
        (
            "upper-bound multiplier",
            report.multipliers.upper_bounds.as_slice(),
        ),
    ] {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(format!("KKT {name} at index {index} is non-finite"));
        }
    }

    let validate_indices = |name: &str, indices: &[usize], upper: usize| {
        if let Some(index) = indices.iter().position(|&value| value >= upper) {
            return Err(format!(
                "KKT {name} index {} is outside length {upper}",
                indices[index]
            ));
        }
        if indices.windows(2).any(|window| window[0] >= window[1]) {
            return Err(format!("KKT {name} indices are not strictly increasing"));
        }
        Ok(())
    };
    validate_indices(
        "active lower-bound",
        &diagnostics.active_lower_bounds,
        report.multipliers.lower_bounds.len(),
    )?;
    validate_indices(
        "active upper-bound",
        &diagnostics.active_upper_bounds,
        report.multipliers.upper_bounds.len(),
    )?;

    if let Some(ipm) = &report.ipm {
        finite(ipm.barrier_parameter, "IPM barrier parameter")?;
        finite(ipm.dual_residual, "IPM dual residual")?;
        finite(ipm.primal_residual, "IPM primal residual")?;
        finite(ipm.complementarity, "IPM complementarity")?;
        if ipm.active_inequalities != ipm.active_inequality_rows.len() {
            return Err("IPM active inequality count does not match row metadata".into());
        }
        if ipm.active_bounds != ipm.active_bound_rows.len() {
            return Err("IPM active-bound count does not match row metadata".into());
        }
        if ipm
            .restoration_successes
            .checked_add(ipm.restoration_failures)
            .is_none_or(|outcomes| outcomes > ipm.restoration_attempts)
        {
            return Err("IPM restoration outcome counts exceed attempts".into());
        }
        validate_indices(
            "IPM active inequality row",
            &ipm.active_inequality_rows,
            usize::MAX,
        )?;
        validate_indices("IPM active bound row", &ipm.active_bound_rows, usize::MAX)?;
    }
    Ok(())
}

impl<T: RealScalar> SolverResultContract for OptimResult<T> {
    fn solver_status(&self) -> SolverStatus {
        OptimResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        OptimResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        validate_result_finite(self).map_err(|error| SolverError::OperationFailed {
            operation: "optimization result contract",
            reason: error.to_string(),
        })?;
        if let Some(report) = &self.kkt {
            validate_kkt_report(report).map_err(|reason| SolverError::OperationFailed {
                operation: "optimization KKT report contract",
                reason,
            })?;
        }
        if !self.grad_norm.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "optimization result contract",
                reason: "gradient norm is non-finite".into(),
            });
        }
        if self.converged != (self.solver_status() == SolverStatus::Converged) {
            return Err(SolverError::OperationFailed {
                operation: "optimization result contract",
                reason: "converged flag and solver status disagree".into(),
            });
        }
        Ok(())
    }
}

/// Lagrange multipliers supplied to [`OptimResult::kkt_diagnostics`].
///
/// The solver-wide constraint convention is `c(x) = 0` for equalities and
/// `c(x) >= 0` for inequalities. Consequently, inequality and bound
/// multipliers are required to be non-negative, and the Lagrangian used by
/// the diagnostics is
/// `f(x) - λ_eq c_eq(x) - λ_ineq c_ineq(x) - λ_lower (x-lb)
///  - λ_upper (ub-x)`.
#[derive(Debug, Clone, Default)]
pub struct KktMultipliers<T: RealScalar> {
    /// One multiplier for every scalar output of every user constraint, in
    /// the same order as the flattened constraint outputs.
    pub constraints: Vec<T>,
    /// One multiplier for each lower bound. Entries for unbounded variables
    /// must be zero.
    pub lower_bounds: Vec<T>,
    /// One multiplier for each upper bound. Entries for unbounded variables
    /// must be zero.
    pub upper_bounds: Vec<T>,
}

/// Independently recomputed constrained-optimality metrics.
#[derive(Debug, Clone)]
pub struct KktDiagnostics<T: RealScalar> {
    /// Maximum equality residual, inequality violation, or bound violation.
    pub primal_feasibility: T,
    /// Infinity norm of the Lagrangian stationarity residual.
    pub stationarity: T,
    /// Largest violation of non-negative inequality/bound multipliers.
    pub dual_feasibility: T,
    /// Largest complementarity product for inequalities and bounds.
    pub complementarity: T,
    /// Variables at their lower bound within a scale-aware tolerance.
    pub active_lower_bounds: Vec<usize>,
    /// Variables at their upper bound within a scale-aware tolerance.
    pub active_upper_bounds: Vec<usize>,
}

/// KKT multipliers and independently recomputed residuals attached to an
/// [`OptimResult`].  `norm` is the maximum of primal feasibility,
/// stationarity, dual feasibility, and complementarity.
#[derive(Debug, Clone)]
pub struct KktReport<T: RealScalar> {
    pub multipliers: KktMultipliers<T>,
    pub diagnostics: KktDiagnostics<T>,
    pub norm: T,
    /// Optional algorithm-state diagnostics supplied by a constrained
    /// optimizer. Present for IPM results; `None` for other methods.
    pub ipm: Option<IpmDiagnostics<T>>,
}

/// Auditable final state from an interior-point run.
#[derive(Debug, Clone)]
pub struct IpmDiagnostics<T: RealScalar> {
    /// Number of outer barrier iterations performed.
    pub iterations: usize,
    /// Barrier parameter at termination.
    pub barrier_parameter: T,
    /// Scaled dual residual used by the IPM stopping test.
    pub dual_residual: T,
    /// Maximum equality and slack/primal residual used by the stopping test.
    pub primal_residual: T,
    /// Scaled complementarity residual used by the stopping test.
    pub complementarity: T,
    /// Number of filter failures that entered feasibility restoration.
    pub restoration_attempts: usize,
    /// Number of restoration phases that produced a new iterate.
    pub restoration_successes: usize,
    /// Number of final inequality/slack rows near the active boundary.
    pub active_inequalities: usize,
    /// Number of final bound rows near the active boundary.
    pub active_bounds: usize,
    /// Indices of all final inequality rows near the active boundary.
    pub active_inequality_rows: Vec<usize>,
    /// Indices of final bound rows near the active boundary, relative to the
    /// bound-row portion of the internal inequality system.
    pub active_bound_rows: Vec<usize>,
    /// Number of trial-step backtracks performed by the filter line search.
    pub line_search_backtracks: usize,
    /// Number of restoration attempts that failed to produce a new iterate.
    pub restoration_failures: usize,
}

/// Constraint values and Jacobian already evaluated at an optimizer result.
/// This is used by composite constrained APIs whose temporary constraints
/// cannot be represented as owned [`Constraint`] closures.
pub(crate) struct EvaluatedKktConstraint<T: RealScalar> {
    pub(crate) kind: ConstraintKind,
    pub(crate) dim: usize,
    pub(crate) values: Vec<T>,
    pub(crate) jacobian: sonium_solvers_core::dense::DenseMatrix<T>,
}

impl<T: RealScalar> OptimResult<T> {
    /// Attach an auditable KKT report to this result.
    pub fn attach_kkt(
        &mut self,
        constraints: &[Constraint<T>],
        bounds: &[Bound<T>],
        multipliers: KktMultipliers<T>,
    ) -> Result<(), OptimError> {
        let diagnostics = self.kkt_diagnostics(constraints, bounds, Some(&multipliers))?;
        let norm = diagnostics
            .primal_feasibility
            .max(diagnostics.stationarity)
            .max(diagnostics.dual_feasibility)
            .max(diagnostics.complementarity);
        self.kkt = Some(KktReport {
            multipliers,
            diagnostics,
            norm,
            ipm: None,
        });
        Ok(())
    }

    /// Attach a KKT report from constraint values/Jacobians evaluated by a
    /// composite constrained method. The calculations intentionally mirror
    /// [`Self::kkt_diagnostics`] but do not re-invoke temporary callbacks.
    pub(crate) fn attach_evaluated_kkt(
        &mut self,
        constraints: &[EvaluatedKktConstraint<T>],
        bounds: &[Bound<T>],
        multipliers: KktMultipliers<T>,
    ) -> Result<(), OptimError> {
        let n = self.x.len();
        let gradient = self.grad.as_deref().ok_or(OptimError::MissingGradient)?;
        if gradient.len() != n {
            return Err(OptimError::DimensionMismatch {
                expected: n,
                got: gradient.len(),
            });
        }
        validate_bounds(bounds, n)?;
        if self.x.iter().any(|value| !value.is_finite())
            || gradient.iter().any(|value| !value.is_finite())
        {
            return Err(OptimError::NumericalError {
                iteration: self.iterations,
                message: "KKT diagnostics received non-finite x or gradient".to_string(),
            });
        }

        let constraint_count: usize = constraints.iter().map(|constraint| constraint.dim).sum();
        if multipliers.constraints.len() != constraint_count {
            return Err(OptimError::DimensionMismatch {
                expected: constraint_count,
                got: multipliers.constraints.len(),
            });
        }
        if multipliers.lower_bounds.len() != bounds.len()
            || multipliers.upper_bounds.len() != bounds.len()
        {
            return Err(OptimError::DimensionMismatch {
                expected: n,
                got: multipliers
                    .lower_bounds
                    .len()
                    .max(multipliers.upper_bounds.len()),
            });
        }
        if multipliers
            .constraints
            .iter()
            .chain(multipliers.lower_bounds.iter())
            .chain(multipliers.upper_bounds.iter())
            .any(|value| !value.is_finite())
        {
            return Err(OptimError::NumericalError {
                iteration: self.iterations,
                message: "KKT diagnostics received non-finite multipliers".to_string(),
            });
        }

        let mut stationarity = gradient.to_vec();
        let mut primal_feasibility = T::zero();
        let mut dual_feasibility = T::zero();
        let mut complementarity = T::zero();
        let mut constraint_offset = 0_usize;

        for (constraint_index, constraint) in constraints.iter().enumerate() {
            if constraint.values.len() != constraint.dim {
                return Err(OptimError::InvalidConstraintDimension {
                    index: constraint_index,
                    expected: constraint.dim,
                    got: constraint.values.len(),
                });
            }
            if constraint.jacobian.nrows() != constraint.dim || constraint.jacobian.ncols() != n {
                return Err(OptimError::InvalidConstraintJacobian {
                    index: constraint_index,
                    expected_rows: constraint.dim,
                    expected_cols: n,
                    got_rows: constraint.jacobian.nrows(),
                    got_cols: constraint.jacobian.ncols(),
                });
            }
            if constraint.values.iter().any(|value| !value.is_finite())
                || (0..constraint.dim)
                    .any(|row| (0..n).any(|column| !constraint.jacobian[[row, column]].is_finite()))
            {
                return Err(OptimError::NumericalError {
                    iteration: self.iterations,
                    message: format!(
                        "constraint {constraint_index} returned non-finite evaluated KKT data"
                    ),
                });
            }

            for row in 0..constraint.dim {
                let value = constraint.values[row];
                let multiplier = multipliers.constraints[constraint_offset + row];
                match constraint.kind {
                    ConstraintKind::Equality => {
                        primal_feasibility = primal_feasibility.max(value.abs());
                    }
                    ConstraintKind::Inequality => {
                        primal_feasibility = primal_feasibility.max((-value).max(T::zero()));
                        dual_feasibility = dual_feasibility.max((-multiplier).max(T::zero()));
                        complementarity = complementarity.max((multiplier * value).abs());
                    }
                }
                for (column, stationarity_entry) in stationarity.iter_mut().enumerate() {
                    *stationarity_entry -= multiplier * constraint.jacobian[[row, column]];
                }
            }
            constraint_offset += constraint.dim;
        }

        let mut active_lower_bounds = Vec::new();
        let mut active_upper_bounds = Vec::new();
        for (index, bound) in bounds.iter().enumerate() {
            if let Some(lower) = bound.lower {
                let slack = self.x[index] - lower;
                primal_feasibility = primal_feasibility.max((-slack).max(T::zero()));
                let multiplier = multipliers.lower_bounds[index];
                dual_feasibility = dual_feasibility.max((-multiplier).max(T::zero()));
                complementarity = complementarity.max((multiplier * slack).abs());
                if slack.abs() <= T::sqrt_epsilon() * T::one().max(lower.abs()) {
                    active_lower_bounds.push(index);
                }
                stationarity[index] -= multiplier;
            }
            if let Some(upper) = bound.upper {
                let slack = upper - self.x[index];
                primal_feasibility = primal_feasibility.max((-slack).max(T::zero()));
                let multiplier = multipliers.upper_bounds[index];
                dual_feasibility = dual_feasibility.max((-multiplier).max(T::zero()));
                complementarity = complementarity.max((multiplier * slack).abs());
                if slack.abs() <= T::sqrt_epsilon() * T::one().max(upper.abs()) {
                    active_upper_bounds.push(index);
                }
                stationarity[index] += multiplier;
            }
        }

        let stationarity = stationarity
            .iter()
            .map(|value| value.abs())
            .fold(T::zero(), T::max);
        let diagnostics = KktDiagnostics {
            primal_feasibility,
            stationarity,
            dual_feasibility,
            complementarity,
            active_lower_bounds,
            active_upper_bounds,
        };
        let norm = diagnostics
            .primal_feasibility
            .max(diagnostics.stationarity)
            .max(diagnostics.dual_feasibility)
            .max(diagnostics.complementarity);
        self.kkt = Some(KktReport {
            multipliers,
            diagnostics,
            norm,
            ipm: None,
        });
        Ok(())
    }

    /// Apply the final success gate for a constrained result.  A local
    /// projected-gradient or barrier test is not sufficient to claim success
    /// unless an original-problem KKT report exists and meets `tolerance`.
    ///
    /// Missing reports are failures too: a constrained optimizer must not
    /// claim success merely because its local stopping test fired.  The
    /// tolerance is checked here as well so an invalid caller/configuration
    /// value cannot accidentally make a non-finite KKT norm pass through the
    /// comparison below.
    pub fn gate_kkt(&mut self, tolerance: T) {
        let kkt_ok = tolerance.is_finite()
            && tolerance >= T::zero()
            && self
                .kkt
                .as_ref()
                .is_some_and(|report| report.norm.is_finite() && report.norm <= tolerance);
        if self.converged && !kkt_ok {
            self.converged = false;
            self.status = OptimStatus::KktNotSatisfied;
        }
    }

    /// Recompute KKT and feasibility metrics for a constrained result.
    ///
    /// This deliberately does not infer multipliers from an optimizer's
    /// internal state. Callers pass the multipliers produced by the relevant
    /// QP/barrier solve, making the reported stationarity and complementarity
    /// auditable and independent of the optimizer's stopping test. Passing
    /// `None` uses zero multipliers, which is useful for feasibility-only
    /// checks and derivative-free results.
    pub fn kkt_diagnostics(
        &self,
        constraints: &[Constraint<T>],
        bounds: &[Bound<T>],
        multipliers: Option<&KktMultipliers<T>>,
    ) -> Result<KktDiagnostics<T>, OptimError> {
        let n = self.x.len();
        let gradient = self.grad.as_deref().ok_or(OptimError::MissingGradient)?;
        if gradient.len() != n {
            return Err(OptimError::DimensionMismatch {
                expected: n,
                got: gradient.len(),
            });
        }
        validate_bounds(bounds, n)?;

        if self.x.iter().any(|value| !value.is_finite())
            || gradient.iter().any(|value| !value.is_finite())
        {
            return Err(OptimError::NumericalError {
                iteration: self.iterations,
                message: "KKT diagnostics received non-finite x or gradient".to_string(),
            });
        }
        let constraint_count: usize = constraints.iter().map(|constraint| constraint.dim).sum();
        let (constraint_multipliers, lower_multipliers, upper_multipliers) =
            if let Some(multipliers) = multipliers {
                if multipliers.constraints.len() != constraint_count {
                    return Err(OptimError::DimensionMismatch {
                        expected: constraint_count,
                        got: multipliers.constraints.len(),
                    });
                }
                if multipliers.lower_bounds.len() != bounds.len()
                    || multipliers.upper_bounds.len() != bounds.len()
                {
                    return Err(OptimError::DimensionMismatch {
                        expected: n,
                        got: multipliers
                            .lower_bounds
                            .len()
                            .max(multipliers.upper_bounds.len()),
                    });
                }
                if multipliers
                    .constraints
                    .iter()
                    .any(|value| !value.is_finite())
                    || multipliers
                        .lower_bounds
                        .iter()
                        .any(|value| !value.is_finite())
                    || multipliers
                        .upper_bounds
                        .iter()
                        .any(|value| !value.is_finite())
                {
                    return Err(OptimError::NumericalError {
                        iteration: self.iterations,
                        message: "KKT diagnostics received non-finite multipliers".to_string(),
                    });
                }
                (
                    multipliers.constraints.as_slice(),
                    multipliers.lower_bounds.as_slice(),
                    multipliers.upper_bounds.as_slice(),
                )
            } else {
                (&[][..], &[][..], &[][..])
            };

        let mut stationarity = gradient.to_vec();
        let mut primal_feasibility = T::zero();
        let mut dual_feasibility = T::zero();
        let mut complementarity = T::zero();
        let mut constraint_offset = 0;

        for (constraint_index, constraint) in constraints.iter().enumerate() {
            let values = (constraint.fun)(&self.x);
            if values.len() != constraint.dim {
                return Err(OptimError::InvalidConstraintDimension {
                    index: constraint_index,
                    expected: constraint.dim,
                    got: values.len(),
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(OptimError::NumericalError {
                    iteration: self.iterations,
                    message: format!("constraint {constraint_index} returned non-finite value"),
                });
            }
            let jacobian = if let Some(jacobian) = constraint.jac.as_ref() {
                jacobian(&self.x)
            } else {
                let mut jacobian = sonium_solvers_core::dense::DenseMatrix::from_elem(
                    constraint.dim,
                    n,
                    T::zero(),
                );
                for column in 0..n {
                    let step = T::sqrt_epsilon() * T::one().max(self.x[column].abs());
                    let mut plus = self.x.clone();
                    let mut minus = self.x.clone();
                    plus[column] += step;
                    minus[column] -= step;
                    let plus_values = (constraint.fun)(&plus);
                    let minus_values = (constraint.fun)(&minus);
                    if plus_values.len() != constraint.dim {
                        return Err(OptimError::InvalidConstraintDimension {
                            index: constraint_index,
                            expected: constraint.dim,
                            got: plus_values.len(),
                        });
                    }
                    if minus_values.len() != constraint.dim {
                        return Err(OptimError::InvalidConstraintDimension {
                            index: constraint_index,
                            expected: constraint.dim,
                            got: minus_values.len(),
                        });
                    }
                    for row in 0..constraint.dim {
                        jacobian[[row, column]] =
                            (plus_values[row] - minus_values[row]) / (T::one() + T::one()) / step;
                    }
                }
                jacobian
            };
            if jacobian.nrows() != constraint.dim || jacobian.ncols() != n {
                return Err(OptimError::InvalidConstraintJacobian {
                    index: constraint_index,
                    expected_rows: constraint.dim,
                    expected_cols: n,
                    got_rows: jacobian.nrows(),
                    got_cols: jacobian.ncols(),
                });
            }
            let jacobian_is_finite = (0..constraint.dim)
                .all(|row| (0..n).all(|column| jacobian[[row, column]].is_finite()));
            if !jacobian_is_finite {
                return Err(OptimError::NumericalError {
                    iteration: self.iterations,
                    message: format!("constraint {constraint_index} returned non-finite Jacobian"),
                });
            }

            for row in 0..constraint.dim {
                let value = values[row];
                let multiplier = constraint_multipliers
                    .get(constraint_offset + row)
                    .copied()
                    .unwrap_or(T::zero());
                match constraint.kind {
                    ConstraintKind::Equality => {
                        primal_feasibility = primal_feasibility.max(value.abs());
                    }
                    ConstraintKind::Inequality => {
                        primal_feasibility = primal_feasibility.max((-value).max(T::zero()));
                        dual_feasibility = dual_feasibility.max((-multiplier).max(T::zero()));
                        complementarity = complementarity.max((multiplier * value).abs());
                    }
                }
                for column in 0..n {
                    stationarity[column] -= multiplier * jacobian[[row, column]];
                }
            }
            constraint_offset += constraint.dim;
        }

        let mut active_lower_bounds = Vec::new();
        let mut active_upper_bounds = Vec::new();
        for (index, bound) in bounds.iter().enumerate() {
            if let Some(lower) = bound.lower {
                let slack = self.x[index] - lower;
                primal_feasibility = primal_feasibility.max((-slack).max(T::zero()));
                let multiplier = lower_multipliers.get(index).copied().unwrap_or(T::zero());
                dual_feasibility = dual_feasibility.max((-multiplier).max(T::zero()));
                complementarity = complementarity.max((multiplier * slack).abs());
                stationarity[index] -= multiplier;
                if slack.abs() <= T::sqrt_epsilon() * T::one().max(lower.abs()) {
                    active_lower_bounds.push(index);
                }
            }
            if let Some(upper) = bound.upper {
                let slack = upper - self.x[index];
                primal_feasibility = primal_feasibility.max((-slack).max(T::zero()));
                let multiplier = upper_multipliers.get(index).copied().unwrap_or(T::zero());
                dual_feasibility = dual_feasibility.max((-multiplier).max(T::zero()));
                complementarity = complementarity.max((multiplier * slack).abs());
                stationarity[index] += multiplier;
                if slack.abs() <= T::sqrt_epsilon() * T::one().max(upper.abs()) {
                    active_upper_bounds.push(index);
                }
            }
        }

        Ok(KktDiagnostics {
            primal_feasibility,
            stationarity: stationarity
                .iter()
                .map(|value| value.abs())
                .fold(T::zero(), T::max),
            dual_feasibility,
            complementarity,
            active_lower_bounds,
            active_upper_bounds,
        })
    }
}

/// Per-iteration snapshot (for convergence history).
#[derive(Debug, Clone)]
pub struct IterationInfo<T: RealScalar> {
    pub iteration: usize,
    pub fun: T,
    pub grad_norm: T,
    pub step_size: T,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from optimisation algorithms.
#[derive(Debug, thiserror::Error)]
pub enum OptimError {
    #[error(
        "Optimisation did not converge in {iterations} iterations (grad_norm = {grad_norm:.2e})"
    )]
    NotConverged { iterations: usize, grad_norm: f64 },

    #[error("Line search failed at iteration {iteration}")]
    LineSearchFailed { iteration: usize },

    #[error("Numerical error at iteration {iteration}: {message}")]
    NumericalError { iteration: usize, message: String },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid bounds: lower ({lower}) > upper ({upper}) at index {index}")]
    InvalidBounds {
        index: usize,
        lower: f64,
        upper: f64,
    },

    #[error("initial point at index {index} ({value}) is outside bounds [{lower:?}, {upper:?}]")]
    IncompatibleStartingPoint {
        index: usize,
        value: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    },

    #[error("Infeasible: constraints cannot be satisfied")]
    Infeasible,

    #[error("KKT diagnostics require a gradient")]
    MissingGradient,

    #[error("constraint {index} returned {got} values, expected {expected}")]
    InvalidConstraintDimension {
        index: usize,
        expected: usize,
        got: usize,
    },

    #[error(
        "constraint {index} Jacobian has shape {got_rows}x{got_cols}, expected {expected_rows}x{expected_cols}"
    )]
    InvalidConstraintJacobian {
        index: usize,
        expected_rows: usize,
        expected_cols: usize,
        got_rows: usize,
        got_cols: usize,
    },

    #[error("optimizer callback panicked at iteration {iteration} while evaluating {stage}")]
    CallbackPanicked { iteration: usize, stage: String },
}

impl OptimError {
    /// Return the canonical solver status represented by this failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NotConverged { .. } => SolverStatus::MaxIterationsReached,
            Self::LineSearchFailed { .. } | Self::Infeasible => SolverStatus::Stagnated,
            Self::NumericalError { .. }
            | Self::DimensionMismatch { .. }
            | Self::InvalidBounds { .. }
            | Self::IncompatibleStartingPoint { .. }
            | Self::MissingGradient
            | Self::InvalidConstraintDimension { .. }
            | Self::InvalidConstraintJacobian { .. }
            | Self::CallbackPanicked { .. } => SolverStatus::NumericalError,
        }
    }

    /// Return whether the requested optimization accuracy was established by
    /// this error path.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        match self {
            Self::NotConverged { .. } | Self::LineSearchFailed { .. } | Self::Infeasible => {
                AccuracyStatus::NotAttained
            }
            Self::NumericalError { .. }
            | Self::DimensionMismatch { .. }
            | Self::InvalidBounds { .. }
            | Self::IncompatibleStartingPoint { .. }
            | Self::MissingGradient
            | Self::InvalidConstraintDimension { .. }
            | Self::InvalidConstraintJacobian { .. }
            | Self::CallbackPanicked { .. } => AccuracyStatus::NotChecked,
        }
    }
}

impl SolverFailure for OptimError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NotConverged { .. } | Self::LineSearchFailed { .. } | Self::Infeasible => {
                SolverErrorKind::Convergence
            }
            Self::DimensionMismatch { .. }
            | Self::InvalidConstraintDimension { .. }
            | Self::InvalidConstraintJacobian { .. } => SolverErrorKind::DimensionMismatch,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        OptimError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        OptimError::attainable_accuracy(self)
    }
}

// ---------------------------------------------------------------------------
// Bound constraints
// ---------------------------------------------------------------------------

/// Per-variable bound constraint.
#[derive(Debug, Clone, Copy)]
pub struct Bound<T: RealScalar> {
    pub lower: Option<T>,
    pub upper: Option<T>,
}

impl<T: RealScalar> Bound<T> {
    pub fn unbounded() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }
    pub fn lower(lb: T) -> Self {
        Self {
            lower: Some(lb),
            upper: None,
        }
    }
    pub fn upper(ub: T) -> Self {
        Self {
            lower: None,
            upper: Some(ub),
        }
    }
    pub fn both(lb: T, ub: T) -> Self {
        Self {
            lower: Some(lb),
            upper: Some(ub),
        }
    }
}

/// Validate a bound vector before an optimizer evaluates user callbacks.
pub(crate) fn validate_bounds<T: RealScalar>(
    bounds: &[Bound<T>],
    dimension: usize,
) -> Result<(), OptimError> {
    if !bounds.is_empty() && bounds.len() != dimension {
        return Err(OptimError::DimensionMismatch {
            expected: dimension,
            got: bounds.len(),
        });
    }
    for (index, bound) in bounds.iter().enumerate() {
        let lower_finite = bound.lower.is_none_or(|value| value.is_finite());
        let upper_finite = bound.upper.is_none_or(|value| value.is_finite());
        if !lower_finite || !upper_finite {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: format!("bound {index} is non-finite"),
            });
        }
        if let (Some(lower), Some(upper)) = (bound.lower, bound.upper)
            && lower > upper
        {
            return Err(OptimError::InvalidBounds {
                index,
                lower: lower.to_f64().unwrap_or(f64::NAN),
                upper: upper.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

/// Box constraints: per-variable lower and upper bounds.
pub type Bounds<T> = Vec<Bound<T>>;

/// Project a point onto the feasible box defined by bounds.
pub fn project_bounds<T: RealScalar>(x: &mut [T], bounds: &[Bound<T>]) {
    for (xi, b) in x.iter_mut().zip(bounds.iter()) {
        if let Some(lb) = b.lower
            && *xi < lb
        {
            *xi = lb;
        }
        if let Some(ub) = b.upper
            && *xi > ub
        {
            *xi = ub;
        }
    }
}

// ---------------------------------------------------------------------------
// General constraints (for constrained optimisation)
// ---------------------------------------------------------------------------

/// Type of constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// `c(x) = 0`
    Equality,
    /// `c(x) >= 0`
    Inequality,
}

/// A nonlinear constraint `c(x)` with optional Jacobian.
#[allow(clippy::type_complexity)]
pub struct Constraint<T: RealScalar> {
    /// The kind of constraint.
    pub kind: ConstraintKind,
    /// Number of constraint outputs (m).
    pub dim: usize,
    /// Evaluate the constraint: `x -> c(x)` as `Array1` of length `dim`.
    pub fun: Box<dyn Fn(&[T]) -> Vec<T> + Send + Sync>,
    /// Evaluate the Jacobian: `x -> J(x)` as `Array2` of shape `(dim, n)`.
    /// If `None`, finite differences will be used.
    pub jac: Option<Box<dyn Fn(&[T]) -> sonium_solvers_core::dense::DenseMatrix<T> + Send + Sync>>,
}

impl<T: RealScalar> std::fmt::Debug for Constraint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("kind", &self.kind)
            .field("dim", &self.dim)
            .field("has_jac", &self.jac.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Line search method selector (shared by gradient-based methods)
// ---------------------------------------------------------------------------

/// Which line search algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineSearchMethod {
    #[default]
    StrongWolfe,
    Backtracking,
    /// Hager–Zhang (2005) approximate-Wolfe search with secant² zoom
    /// (see [`crate::line_search::hager_zhang_line_search`]).
    HagerZhang,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_scalar_f64() {
        assert!(f64::machine_epsilon() > 0.0);
        assert!(f64::sqrt_epsilon() > 0.0);
        let x = sonium_solvers_core::array![3.0_f64, 4.0];
        let y = sonium_solvers_core::array![1.0_f64, 2.0];
        assert!((f64::vec_dot(&x, &y) - 11.0).abs() < 1e-12);
        assert!((f64::vec_norm(&x) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_real_scalar_f32() {
        assert!(f32::machine_epsilon() > 0.0);
        let x = sonium_solvers_core::array![3.0_f32, 4.0];
        assert!((f32::vec_norm(&x) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_project_bounds() {
        let bounds = vec![
            Bound::both(0.0, 1.0),
            Bound::lower(-1.0),
            Bound::upper(5.0),
            Bound::unbounded(),
        ];
        let mut x = sonium_solvers_core::array![-0.5, -2.0, 10.0, 100.0];
        project_bounds(&mut x, &bounds);
        assert!((x[0] - 0.0).abs() < 1e-12);
        assert!((x[1] - (-1.0)).abs() < 1e-12);
        assert!((x[2] - 5.0).abs() < 1e-12);
        assert!((x[3] - 100.0).abs() < 1e-12);
    }

    #[test]
    fn kkt_diagnostics_reports_feasibility_stationarity_and_active_bounds() {
        let result = OptimResult {
            x: vec![0.0],
            fun: 0.0,
            grad: Some(vec![1.0]),
            iterations: 1,
            fun_evals: 1,
            grad_evals: 1,
            hess_evals: 0,
            grad_norm: 1.0,
            converged: true,
            status: OptimStatus::Converged,
            kkt: None,
        };
        let multipliers = KktMultipliers {
            constraints: Vec::new(),
            lower_bounds: vec![1.0],
            upper_bounds: vec![0.0],
        };
        let diagnostics = result
            .kkt_diagnostics(&[], &[Bound::lower(0.0)], Some(&multipliers))
            .expect("KKT diagnostics should accept a valid bound system");

        assert_eq!(diagnostics.primal_feasibility, 0.0);
        assert_eq!(diagnostics.stationarity, 0.0);
        assert_eq!(diagnostics.dual_feasibility, 0.0);
        assert_eq!(diagnostics.complementarity, 0.0);
        assert_eq!(diagnostics.active_lower_bounds, vec![0]);
        assert!(diagnostics.active_upper_bounds.is_empty());
    }

    #[test]
    fn kkt_diagnostics_finite_differences_constraints_without_jacobian() {
        let result = OptimResult {
            x: vec![1.0],
            fun: 0.0,
            grad: Some(vec![0.0]),
            iterations: 0,
            fun_evals: 1,
            grad_evals: 1,
            hess_evals: 0,
            grad_norm: 0.0,
            converged: true,
            status: OptimStatus::Converged,
            kkt: None,
        };
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[0] - 1.0]),
            jac: None,
        }];
        let diagnostics = result
            .kkt_diagnostics(&constraints, &[], None)
            .expect("finite-difference Jacobian should be available");

        assert!(diagnostics.primal_feasibility < 1.0e-12);
        assert_eq!(diagnostics.stationarity, 0.0);
    }
}
