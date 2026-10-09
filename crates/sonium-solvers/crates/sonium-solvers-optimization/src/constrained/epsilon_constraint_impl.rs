//! Epsilon-constraint method for generating Pareto fronts.
//!
//! Minimises one objective while constraining the others to be below
//! epsilon values. By sweeping epsilon, we trace the Pareto front.

use crate::constrained::augmented_lagrangian_impl::{
    AugmentedLagrangianConfig, augmented_lagrangian,
};
use crate::constrained::qp_active_set::{checked_constraint_jacobian, checked_constraint_values};
use crate::types::{
    Constraint, ConstraintKind, EvaluatedKktConstraint, KktMultipliers, OptimError, OptimResult,
    RealScalar,
};
use crate::unconstrained::lbfgs_impl::{LbfgsConfig, lbfgs};
use sonium_solvers_core::traits::{
    AccuracyStatus, SolverError, SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

/// Configuration for the epsilon-constraint method.
#[derive(Debug, Clone)]
pub struct EpsilonConstraintConfig<T: RealScalar> {
    /// Number of Pareto points to generate.
    pub num_points: usize,
    /// Tolerance for subproblem optimisation.
    pub tolerance: T,
    pub max_iterations: usize,
}

impl Default for EpsilonConstraintConfig<f64> {
    fn default() -> Self {
        Self {
            num_points: 20,
            tolerance: 1e-6,
            max_iterations: 500,
        }
    }
}

impl Default for EpsilonConstraintConfig<f32> {
    fn default() -> Self {
        Self {
            num_points: 20,
            tolerance: 1e-4,
            max_iterations: 500,
        }
    }
}

/// Result of multi-objective epsilon-constraint method.
#[derive(Debug, Clone)]
pub struct ParetoResult<T: RealScalar> {
    /// Pareto-optimal solutions.
    pub solutions: Vec<Vec<T>>,
    /// Objective values at each Pareto point: `[f_1(x), f_2(x), ...]` per point.
    pub objective_values: Vec<Vec<T>>,
    /// Per-subproblem convergence info.
    pub subproblem_results: Vec<OptimResult<T>>,
}

impl<T: RealScalar> ParetoResult<T> {
    /// Return whether all Pareto points and objective values are finite.
    pub fn is_finite(&self) -> bool {
        self.solutions
            .iter()
            .chain(self.objective_values.iter())
            .flatten()
            .all(|value| value.is_finite())
            && self
                .subproblem_results
                .iter()
                .all(|result| result.solver_status() != SolverStatus::NumericalError)
    }

    /// Return the aggregate status of the epsilon-constraint sweep.
    pub fn solver_status(&self) -> SolverStatus {
        if !self.is_finite() {
            return SolverStatus::NumericalError;
        }
        if self.subproblem_results.is_empty() {
            return SolverStatus::NumericalError;
        }
        if self
            .subproblem_results
            .iter()
            .all(|result| result.solver_status() == SolverStatus::Converged)
        {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        }
    }

    /// Alias matching the checked result vocabulary used by other families.
    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    /// Aggregate attainable accuracy across all Pareto subproblems.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            SolverStatus::MaxIterationsReached => AccuracyStatus::NotAttained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    /// Alias matching the checked result vocabulary used by other families.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: RealScalar> SolverResultContract for ParetoResult<T> {
    fn solver_status(&self) -> SolverStatus {
        ParetoResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        ParetoResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if self.solutions.len() != self.objective_values.len()
            || self.solutions.len() != self.subproblem_results.len()
        {
            return Err(SolverError::DimensionMismatch {
                expected: self.subproblem_results.len(),
                got: self.solutions.len().max(self.objective_values.len()),
            });
        }
        if self.subproblem_results.is_empty() {
            return Err(SolverError::OperationFailed {
                operation: "Pareto result contract",
                reason: "Pareto result must contain at least one subproblem".into(),
            });
        }
        if self
            .solutions
            .iter()
            .chain(self.objective_values.iter())
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "Pareto result contract",
                reason: "solution or objective value is non-finite".into(),
            });
        }
        for result in &self.subproblem_results {
            result
                .validate_contract()
                .map_err(|error| SolverError::OperationFailed {
                    operation: "Pareto result contract",
                    reason: error.to_string(),
                })?;
        }
        Ok(())
    }
}

fn checked_objective<T, F>(
    objective: &F,
    x: &[T],
    index: usize,
    iteration: usize,
) -> Result<T, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let value = catch_unwind(AssertUnwindSafe(|| objective(x))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration,
            stage: format!("objective {index}"),
        }
    })?;
    if !value.is_finite() {
        return Err(OptimError::NumericalError {
            iteration,
            message: format!("objective {index} returned a non-finite value"),
        });
    }
    Ok(value)
}

fn checked_gradient<T, Fg>(
    gradient: &Fg,
    x: &[T],
    index: usize,
    iteration: usize,
) -> Result<Vec<T>, OptimError>
where
    T: RealScalar,
    Fg: Fn(&[T]) -> Vec<T>,
{
    let values = catch_unwind(AssertUnwindSafe(|| gradient(x))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration,
            stage: format!("objective {index} gradient"),
        }
    })?;
    if values.len() != x.len() {
        return Err(OptimError::DimensionMismatch {
            expected: x.len(),
            got: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration,
            message: format!("objective {index} gradient returned a non-finite value"),
        });
    }
    Ok(values)
}

#[derive(Clone, Copy)]
struct PenaltyScalars<T: RealScalar> {
    penalty: T,
    half: T,
    two: T,
}

fn evaluate_penalised_objective<T, F>(
    objectives: &[F],
    x: &[T],
    epsilon_values: &[T],
    extra_constraints: &[Constraint<T>],
    scalars: PenaltyScalars<T>,
    iteration: usize,
) -> Result<T, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let mut value = checked_objective(&objectives[0], x, 0, iteration)?;
    for (secondary_index, &epsilon) in epsilon_values.iter().enumerate() {
        let objective_index = secondary_index + 1;
        let secondary =
            checked_objective(&objectives[objective_index], x, objective_index, iteration)?;
        let violation = (secondary - epsilon).max(T::zero());
        value += scalars.penalty * violation * violation;
    }
    for (constraint_index, constraint) in extra_constraints.iter().enumerate() {
        let values = checked_constraint_values(constraint, x, constraint_index, iteration)?;
        for constraint_value in values {
            match constraint.kind {
                ConstraintKind::Equality => {
                    value += scalars.half * scalars.penalty * constraint_value * constraint_value;
                }
                ConstraintKind::Inequality if constraint_value < T::zero() => {
                    value += scalars.half * scalars.penalty * constraint_value * constraint_value;
                }
                ConstraintKind::Inequality => {}
            }
        }
    }
    if !value.is_finite() {
        return Err(OptimError::NumericalError {
            iteration,
            message: "epsilon-constraint penalised objective generated a non-finite value".into(),
        });
    }
    Ok(value)
}

fn evaluate_penalised_gradient<T, F, Fg>(
    objectives: &[F],
    objective_gradients: &[Fg],
    x: &[T],
    epsilon_values: &[T],
    extra_constraints: &[Constraint<T>],
    scalars: PenaltyScalars<T>,
    iteration: usize,
) -> Result<Vec<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    let mut gradient = checked_gradient(&objective_gradients[0], x, 0, iteration)?;
    for (secondary_index, &epsilon) in epsilon_values.iter().enumerate() {
        let objective_index = secondary_index + 1;
        let secondary =
            checked_objective(&objectives[objective_index], x, objective_index, iteration)?;
        let violation = secondary - epsilon;
        if violation > T::zero() {
            let secondary_gradient = checked_gradient(
                &objective_gradients[objective_index],
                x,
                objective_index,
                iteration,
            )?;
            T::vec_axpy(
                scalars.two * scalars.penalty * violation,
                &secondary_gradient,
                &mut gradient,
            );
        }
    }
    for (constraint_index, constraint) in extra_constraints.iter().enumerate() {
        let values = checked_constraint_values(constraint, x, constraint_index, iteration)?;
        let jacobian =
            checked_constraint_jacobian(constraint, x, x.len(), constraint_index, iteration)?;
        for row in 0..constraint.dim {
            let active = match constraint.kind {
                ConstraintKind::Equality => true,
                ConstraintKind::Inequality => values[row] < T::zero(),
            };
            if active {
                for column in 0..gradient.len() {
                    gradient[column] += scalars.penalty * jacobian[[row, column]] * values[row];
                }
            }
        }
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration,
            message: "epsilon-constraint penalised gradient generated a non-finite value".into(),
        });
    }
    Ok(gradient)
}

fn record_callback_failure(failure: &Arc<Mutex<Option<OptimError>>>, error: OptimError) {
    if let Ok(mut slot) = failure.lock()
        && slot.is_none()
    {
        *slot = Some(error);
    }
}

fn take_callback_failure(failure: &Arc<Mutex<Option<OptimError>>>) -> Option<OptimError> {
    failure.lock().ok().and_then(|mut slot| slot.take())
}

/// Generate a Pareto front using the epsilon-constraint method.
///
/// - `objectives[0]` is minimised directly.
/// - `objectives[1..]` are constrained: `f_i(x) <= epsilon_i`.
/// - `objective_grads[i]` provides the gradient of `objectives[i]`.
/// - Epsilon values are swept linearly between the individual minima.
pub fn epsilon_constraint<T, F, Fg>(
    objectives: &[F],
    objective_grads: &[Fg],
    x0: &[T],
    extra_constraints: &[Constraint<T>],
    config: &EpsilonConstraintConfig<T>,
) -> Result<ParetoResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T + Send + Sync,
    Fg: Fn(&[T]) -> Vec<T> + Send + Sync,
{
    if objectives.len() < 2 {
        return Err(OptimError::DimensionMismatch {
            expected: 2,
            got: objectives.len(),
        });
    }
    if objectives.len() != objective_grads.len() {
        return Err(OptimError::DimensionMismatch {
            expected: objectives.len(),
            got: objective_grads.len(),
        });
    }
    if x0.is_empty() {
        return Err(OptimError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    for (index, (objective, gradient)) in objectives.iter().zip(objective_grads).enumerate() {
        super::validate_optimizer_start(objective, gradient, x0, extra_constraints).map_err(
            |error| match error {
                OptimError::CallbackPanicked { iteration, .. } => OptimError::CallbackPanicked {
                    iteration,
                    stage: format!("epsilon-constraint objective {index} preflight"),
                },
                other => other,
            },
        )?;
    }
    if config.num_points == 0
        || config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::zero()
    {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "epsilon-constraint configuration must be finite and positive".into(),
        });
    }

    let k = objectives.len();
    let half = T::from_f64(0.5).ok_or_else(|| OptimError::NumericalError {
        iteration: 0,
        message: "epsilon-constraint scalar conversion failed for 0.5".into(),
    })?;
    let two = T::from_f64(2.0).ok_or_else(|| OptimError::NumericalError {
        iteration: 0,
        message: "epsilon-constraint scalar conversion failed for 2.0".into(),
    })?;
    let epsilon_penalty = T::from_f64(1000.0).ok_or_else(|| OptimError::NumericalError {
        iteration: 0,
        message: "epsilon-constraint scalar conversion failed for 1000.0".into(),
    })?;
    let kkt_gate_scale = T::from_f64(100.0).ok_or_else(|| OptimError::NumericalError {
        iteration: 0,
        message: "epsilon-constraint scalar conversion failed for 100.0".into(),
    })?;
    let penalty_scalars = PenaltyScalars {
        penalty: epsilon_penalty,
        half,
        two,
    };

    // Step 1: Find individual minima for each secondary objective to get range
    let mut f_min = vec![T::zero(); k];
    let mut f_max = vec![T::zero(); k];

    // Evaluate at x0 for initial range estimate
    for i in 0..k {
        let fi = checked_objective(&objectives[i], x0, i, 0)?;
        checked_gradient(&objective_grads[i], x0, i, 0)?;
        f_min[i] = fi;
        f_max[i] = fi;
    }
    for (index, constraint) in extra_constraints.iter().enumerate() {
        checked_constraint_values(constraint, x0, index, 0)?;
    }

    // Minimise each secondary objective individually to find the range
    for i in 1..k {
        let al_config = AugmentedLagrangianConfig {
            max_inner_iterations: config.max_iterations,
            grad_tolerance: config.tolerance,
            ..Default::default()
        };
        let result = augmented_lagrangian(
            &objectives[i],
            &objective_grads[i],
            x0,
            extra_constraints,
            &al_config,
        )?;
        let fi_min = checked_objective(&objectives[i], &result.x, i, 0)?;
        let fi_max_at_min = checked_objective(&objectives[i], x0, i, 0)?;
        f_min[i] = fi_min;
        f_max[i] = fi_max_at_min;
    }

    // Also minimise primary objective to find f_max for secondaries
    let al_config = AugmentedLagrangianConfig {
        max_inner_iterations: config.max_iterations,
        grad_tolerance: config.tolerance,
        ..Default::default()
    };
    let result = augmented_lagrangian(
        &objectives[0],
        &objective_grads[0],
        x0,
        extra_constraints,
        &al_config,
    )?;
    for i in 1..k {
        let fi_at_primary_min = checked_objective(&objectives[i], &result.x, i, 0)?;
        if fi_at_primary_min > f_max[i] {
            f_max[i] = fi_at_primary_min;
        }
    }

    // Step 2: Sweep epsilon values
    let mut solutions = Vec::with_capacity(config.num_points);
    let mut objective_values = Vec::with_capacity(config.num_points);
    let mut subproblem_results = Vec::with_capacity(config.num_points);

    for p in 0..config.num_points {
        let t = if config.num_points > 1 {
            let numerator = T::from_usize(p).ok_or_else(|| OptimError::NumericalError {
                iteration: p,
                message: "epsilon-constraint point index is not representable".into(),
            })?;
            let denominator =
                T::from_usize(config.num_points - 1).ok_or_else(|| OptimError::NumericalError {
                    iteration: p,
                    message: "epsilon-constraint point count is not representable".into(),
                })?;
            numerator / denominator
        } else {
            half
        };

        // Build penalised objective: f_0(x) + mu * sum max(0, f_i(x) - eps_i)^2
        // This avoids lifetime issues with boxing references to objectives.
        let epsilon_vals: Vec<T> = (1..k)
            .map(|i| f_min[i] + t * (f_max[i] - f_min[i]))
            .collect();
        let callback_failure = Arc::new(Mutex::new(None));
        let objective_failure = Arc::clone(&callback_failure);
        let penalised_f = |xp: &[T]| -> T {
            match evaluate_penalised_objective(
                objectives,
                xp,
                &epsilon_vals,
                extra_constraints,
                penalty_scalars,
                p,
            ) {
                Ok(value) => value,
                Err(error) => {
                    record_callback_failure(&objective_failure, error);
                    T::zero()
                }
            }
        };
        let gradient_failure = Arc::clone(&callback_failure);
        let penalised_grad = |xp: &[T]| -> Vec<T> {
            match evaluate_penalised_gradient(
                objectives,
                objective_grads,
                xp,
                &epsilon_vals,
                extra_constraints,
                penalty_scalars,
                p,
            ) {
                Ok(gradient) => gradient,
                Err(error) => {
                    record_callback_failure(&gradient_failure, error);
                    vec![T::zero(); xp.len()]
                }
            }
        };

        let inner_config = LbfgsConfig {
            max_iterations: config.max_iterations,
            grad_tolerance: config.tolerance,
            ..Default::default()
        };
        let subproblem = lbfgs(&penalised_f, &penalised_grad, x0, &inner_config);
        if let Some(error) = take_callback_failure(&callback_failure) {
            return Err(error);
        }
        let mut result = subproblem?;
        finalize_epsilon_kkt(
            &mut result,
            objectives,
            objective_grads,
            &epsilon_vals,
            extra_constraints,
            config.tolerance * kkt_gate_scale,
            config.max_iterations,
        )?;
        let mut obj_vals = Vec::with_capacity(k);
        for (index, objective) in objectives.iter().enumerate() {
            obj_vals.push(checked_objective(objective, &result.x, index, p)?);
        }
        solutions.push(result.x.clone());
        objective_values.push(obj_vals);
        subproblem_results.push(result);
    }

    Ok(ParetoResult {
        solutions,
        objective_values,
        subproblem_results,
    })
}

/// Re-express the penalised L-BFGS subproblem result in terms of the original
/// epsilon-constrained problem and attach an independently recomputed KKT
/// report. The temporary epsilon constraints are evaluated as data rather
/// than boxed closures, because `epsilon_constraint` accepts callbacks that
/// may borrow the caller's objective array.
fn finalize_epsilon_kkt<T, F, Fg>(
    result: &mut OptimResult<T>,
    objectives: &[F],
    objective_grads: &[Fg],
    epsilon_vals: &[T],
    extra_constraints: &[Constraint<T>],
    tolerance: T,
    max_iterations: usize,
) -> Result<(), OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T + Send + Sync,
    Fg: Fn(&[T]) -> Vec<T> + Send + Sync,
{
    let repaired = refine_epsilon_solution(
        result,
        objectives,
        objective_grads,
        epsilon_vals,
        extra_constraints,
        tolerance,
        max_iterations,
    )?;

    let n = result.x.len();
    let primary_fun = checked_objective(&objectives[0], &result.x, 0, result.iterations)?;
    let primary_grad = checked_gradient(&objective_grads[0], &result.x, 0, result.iterations)?;
    result.fun = primary_fun;
    result.grad = Some(primary_grad.clone());
    result.fun_evals += 1;
    result.grad_evals += 1;

    let extra_rows: usize = extra_constraints
        .iter()
        .map(|constraint| constraint.dim)
        .sum();
    let row_count = epsilon_vals.len() + extra_rows;
    let mut matrix = sonium_solvers_core::dense::DenseMatrix::zeros(row_count, n);
    let mut rhs = vec![T::zero(); row_count];
    let mut senses = Vec::with_capacity(row_count);
    let mut evaluated = Vec::with_capacity(epsilon_vals.len() + extra_constraints.len());
    let mut row = 0_usize;

    for (secondary_index, &epsilon) in epsilon_vals.iter().enumerate() {
        let objective_index = secondary_index + 1;
        let objective_value = checked_objective(
            &objectives[objective_index],
            &result.x,
            objective_index,
            result.iterations,
        )?;
        let value = epsilon - objective_value;
        let gradient = checked_gradient(
            &objective_grads[objective_index],
            &result.x,
            objective_index,
            result.iterations,
        )?;
        if !value.is_finite() {
            return Err(OptimError::NumericalError {
                iteration: result.iterations,
                message: format!("epsilon-constraint {secondary_index} value is non-finite"),
            });
        }
        for column in 0..n {
            // epsilon - f_i(x) >= 0, so its linearised row is -grad(f_i).
            matrix[[row, column]] = -gradient[column];
        }
        rhs[row] = -value;
        senses.push(super::qp_active_set::QpSense::Inequality);
        evaluated.push(EvaluatedKktConstraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            values: vec![value],
            jacobian: sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                1,
                n,
                gradient.iter().map(|entry| -*entry).collect(),
            ),
        });
        row += 1;
    }

    for (constraint_index, constraint) in extra_constraints.iter().enumerate() {
        let values =
            checked_constraint_values(constraint, &result.x, constraint_index, result.iterations)?;
        let jacobian = checked_constraint_jacobian(
            constraint,
            &result.x,
            n,
            constraint_index,
            result.iterations,
        )?;
        for jacobian_row in 0..constraint.dim {
            for column in 0..n {
                matrix[[row, column]] = jacobian[[jacobian_row, column]];
            }
            rhs[row] = -values[jacobian_row];
            senses.push(match constraint.kind {
                ConstraintKind::Equality => super::qp_active_set::QpSense::Equality,
                ConstraintKind::Inequality => super::qp_active_set::QpSense::Inequality,
            });
            row += 1;
        }
        evaluated.push(EvaluatedKktConstraint {
            kind: constraint.kind,
            dim: constraint.dim,
            values,
            jacobian,
        });
    }

    let hessian = sonium_solvers_core::dense::DenseMatrix::eye(n);
    let qp = super::qp_active_set::solve_qp(&hessian, &primary_grad, &matrix, &rhs, &senses);
    result.attach_evaluated_kkt(
        &evaluated,
        &[],
        KktMultipliers {
            constraints: qp.multipliers,
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
        },
    )?;
    if repaired {
        result.converged = true;
        result.status = crate::types::OptimStatus::Converged;
    }
    result.gate_kkt(tolerance);
    crate::types::validate_result_finite(result)?;
    Ok(())
}

/// Refine the penalised L-BFGS point with sequential linearised QP steps.
///
/// The squared penalty is useful for finding a basin, but it only approaches
/// an active epsilon constraint asymptotically.  A few QP feasibility/objective
/// steps recover the original constrained problem without requiring boxed
/// temporary callbacks (the public [`Constraint`] callbacks may be borrowed).
fn refine_epsilon_solution<T, F, Fg>(
    result: &mut OptimResult<T>,
    objectives: &[F],
    objective_grads: &[Fg],
    epsilon_vals: &[T],
    extra_constraints: &[Constraint<T>],
    tolerance: T,
    max_iterations: usize,
) -> Result<bool, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T + Send + Sync,
    Fg: Fn(&[T]) -> Vec<T> + Send + Sync,
{
    let n = result.x.len();
    let extra_rows: usize = extra_constraints
        .iter()
        .map(|constraint| constraint.dim)
        .sum();
    let row_count = epsilon_vals.len() + extra_rows;
    let mut repaired = false;
    let one = T::one();
    let half = T::from_f64(0.5).ok_or_else(|| OptimError::NumericalError {
        iteration: result.iterations,
        message: "epsilon-constraint scalar conversion failed 0.5".into(),
    })?;
    let sufficient_decrease = T::from_f64(1.0e-4).ok_or_else(|| OptimError::NumericalError {
        iteration: result.iterations,
        message: "epsilon-constraint scalar conversion failed 1e-4".into(),
    })?;

    for _ in 0..max_iterations {
        let iteration = result.iterations;
        let primary_value = checked_objective(&objectives[0], &result.x, 0, iteration)?;
        let primary_gradient = checked_gradient(&objective_grads[0], &result.x, 0, iteration)?;
        let mut matrix = sonium_solvers_core::dense::DenseMatrix::zeros(row_count, n);
        let mut rhs = vec![T::zero(); row_count];
        let mut senses = Vec::with_capacity(row_count);
        let mut row = 0_usize;
        let mut violation = T::zero();

        for (secondary_index, &epsilon) in epsilon_vals.iter().enumerate() {
            let objective_index = secondary_index + 1;
            let objective_value = checked_objective(
                &objectives[objective_index],
                &result.x,
                objective_index,
                iteration,
            )?;
            let value = epsilon - objective_value;
            let gradient = checked_gradient(
                &objective_grads[objective_index],
                &result.x,
                objective_index,
                iteration,
            )?;
            violation = violation.max((-value).max(T::zero()));
            for column in 0..n {
                matrix[[row, column]] = -gradient[column];
            }
            rhs[row] = -value;
            senses.push(super::qp_active_set::QpSense::Inequality);
            row += 1;
        }

        for (constraint_index, constraint) in extra_constraints.iter().enumerate() {
            let values =
                checked_constraint_values(constraint, &result.x, constraint_index, iteration)?;
            let jacobian =
                checked_constraint_jacobian(constraint, &result.x, n, constraint_index, iteration)?;
            for jacobian_row in 0..constraint.dim {
                let value = values[jacobian_row];
                violation = match constraint.kind {
                    ConstraintKind::Equality => violation.max(value.abs()),
                    ConstraintKind::Inequality => violation.max((-value).max(T::zero())),
                };
                for column in 0..n {
                    matrix[[row, column]] = jacobian[[jacobian_row, column]];
                }
                rhs[row] = -value;
                senses.push(match constraint.kind {
                    ConstraintKind::Equality => super::qp_active_set::QpSense::Equality,
                    ConstraintKind::Inequality => super::qp_active_set::QpSense::Inequality,
                });
                row += 1;
            }
        }

        let hessian = sonium_solvers_core::dense::DenseMatrix::eye(n);
        let qp =
            super::qp_active_set::solve_qp(&hessian, &primary_gradient, &matrix, &rhs, &senses);
        if !matches!(qp.status, super::qp_active_set::QpStatus::Optimal)
            || qp.d.iter().any(|value| !value.is_finite())
        {
            break;
        }
        let step_norm = qp.d.iter().map(|value| value.abs()).fold(T::zero(), T::max);
        if violation <= tolerance && step_norm <= tolerance {
            repaired = true;
            break;
        }

        let mut alpha = one;
        let mut accepted = false;
        for _ in 0..24 {
            let mut trial = result.x.clone();
            for (value, step) in trial.iter_mut().zip(&qp.d) {
                *value += alpha * *step;
            }
            let trial_value = checked_objective(&objectives[0], &trial, 0, iteration)?;
            let mut trial_violation = T::zero();
            for (secondary_index, &epsilon) in epsilon_vals.iter().enumerate() {
                let objective_index = secondary_index + 1;
                let objective_value = checked_objective(
                    &objectives[objective_index],
                    &trial,
                    objective_index,
                    iteration,
                )?;
                trial_violation = trial_violation.max((objective_value - epsilon).max(T::zero()));
            }
            for (constraint_index, constraint) in extra_constraints.iter().enumerate() {
                let values =
                    checked_constraint_values(constraint, &trial, constraint_index, iteration)?;
                for value in values {
                    trial_violation = match constraint.kind {
                        ConstraintKind::Equality => trial_violation.max(value.abs()),
                        ConstraintKind::Inequality => trial_violation.max((-value).max(T::zero())),
                    };
                }
            }
            let violation_reduced = trial_violation
                < violation * (one - sufficient_decrease * alpha)
                || (trial_violation <= tolerance && violation > tolerance);
            let feasible_objective_reduced = trial_violation <= tolerance
                && violation <= tolerance
                && trial_value < primary_value;
            if violation_reduced || feasible_objective_reduced {
                result.x = trial;
                result.iterations = result.iterations.saturating_add(1);
                result.fun_evals = result.fun_evals.saturating_add(1);
                result.grad_evals = result.grad_evals.saturating_add(1);
                accepted = true;
                if trial_violation <= tolerance {
                    repaired = true;
                }
                break;
            }
            alpha *= half;
        }
        if !accepted {
            break;
        }
    }

    Ok(repaired)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_epsilon_constraint_simple() {
        // Bi-objective: min f1 = x^2, min f2 = (x-2)^2
        // Pareto front: x in [0, 2]
        let f1 = |x: &[f64]| x[0] * x[0];
        let g1 = |x: &[f64]| sonium_solvers_core::array![2.0 * x[0]];
        let f2 = |x: &[f64]| (x[0] - 2.0).powi(2);
        let g2 = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 2.0)];

        let objectives = [f1, f2];
        let grads = [g1, g2];
        let x0 = sonium_solvers_core::array![1.0];
        let config = EpsilonConstraintConfig {
            num_points: 5,
            ..Default::default()
        };

        let result = epsilon_constraint(&objectives, &grads, &x0, &[], &config).unwrap();
        assert!(!result.solutions.is_empty());
        assert_eq!(result.solutions.len(), result.subproblem_results.len());
        assert!(
            result
                .subproblem_results
                .iter()
                .all(|subproblem| subproblem.kkt.is_some())
        );
        assert!(result.is_finite());
        assert_eq!(result.status(), SolverStatus::MaxIterationsReached);
        assert_eq!(result.accuracy_status(), AccuracyStatus::NotAttained);

        // Verify solutions span the Pareto front (distinct x values)
        if result.solutions.len() >= 2 {
            let x_first = result.solutions[0][0];
            let x_last = result.solutions[result.solutions.len() - 1][0];
            assert!(
                (x_first - x_last).abs() > 0.1,
                "Pareto points should be distinct: first={x_first}, last={x_last}"
            );
        }
    }

    #[test]
    fn epsilon_constraint_refines_penalty_points_to_kkt() {
        // The secondary minimum is on x <= 0, so the epsilon=minimum endpoint
        // has a regular active constraint gradient instead of a degenerate
        // unconstrained scalar minimum.
        let f1 = |x: &[f64]| (x[0] + 1.0).powi(2);
        let g1 = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] + 1.0)];
        let f2 = |x: &[f64]| (x[0] - 2.0).powi(2);
        let g2 = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 2.0)];
        let domain = Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![-x[0]]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![-1.0])
            })),
        };
        let config = EpsilonConstraintConfig {
            num_points: 5,
            max_iterations: 100,
            tolerance: 1.0e-8,
        };
        let result = epsilon_constraint(&[f1, f2], &[g1, g2], &[-0.5], &[domain], &config).unwrap();
        assert_eq!(result.solutions.len(), config.num_points);
        assert!(result.subproblem_results.iter().all(|subproblem| {
            subproblem.converged
                && subproblem.status == crate::types::OptimStatus::Converged
                && subproblem
                    .kkt
                    .as_ref()
                    .is_some_and(|report| report.norm <= config.tolerance * 100.0)
        }));
    }

    #[test]
    fn test_epsilon_constraint_extra_constraint_no_jac_binds() {
        // Bi-objective: min f1 = x^2, min f2 = (x-2)^2, with the extra
        // inequality constraint x >= 0.75 (c(x) = x - 0.75 >= 0) declared
        // WITHOUT a Jacobian. Every Pareto point must respect the bound.
        let f1 = |x: &[f64]| x[0] * x[0];
        let g1 = |x: &[f64]| sonium_solvers_core::array![2.0 * x[0]];
        let f2 = |x: &[f64]| (x[0] - 2.0).powi(2);
        let g2 = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 2.0)];

        let objectives = [f1, f2];
        let grads = [g1, g2];
        let x0 = sonium_solvers_core::array![1.0];
        let config = EpsilonConstraintConfig {
            num_points: 5,
            ..Default::default()
        };

        let cfun = |x: &[f64]| sonium_solvers_core::array![x[0] - 0.75];
        let without_jac = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(cfun),
            jac: None,
        }];
        let with_jac = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(cfun),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
            })),
        }];

        let r_fd = epsilon_constraint(&objectives, &grads, &x0, &without_jac, &config).unwrap();
        let r_analytic = epsilon_constraint(&objectives, &grads, &x0, &with_jac, &config).unwrap();
        assert!(!r_fd.solutions.is_empty());
        assert!(
            r_fd.subproblem_results
                .iter()
                .all(|subproblem| subproblem.kkt.is_some())
        );
        assert!(
            r_analytic
                .subproblem_results
                .iter()
                .all(|subproblem| subproblem.kkt.is_some())
        );

        // The previously-dropped constraint now binds on every Pareto point.
        for (i, sol) in r_fd.solutions.iter().enumerate() {
            assert!(
                sol[0] >= 0.75 - 2e-2,
                "Pareto point {i} violates constraint: x={}",
                sol[0]
            );
        }
        // FD fallback reaches the same points as analytic Jacobians.
        assert_eq!(r_fd.solutions.len(), r_analytic.solutions.len());
        for (s_fd, s_an) in r_fd.solutions.iter().zip(r_analytic.solutions.iter()) {
            assert!(
                (s_fd[0] - s_an[0]).abs() < 2e-2,
                "FD vs analytic mismatch: {} vs {}",
                s_fd[0],
                s_an[0]
            );
        }
    }

    #[test]
    fn epsilon_constraint_rejects_invalid_objective_shape_and_values() {
        let objective = |_x: &[f64]| 0.0;
        let gradient = |_x: &[f64]| vec![0.0];
        let error = epsilon_constraint(
            &[objective],
            &[gradient],
            &[0.0],
            &[],
            &EpsilonConstraintConfig::default(),
        )
        .expect_err("one objective cannot define a Pareto front");
        assert!(matches!(error, OptimError::DimensionMismatch { .. }));

        let non_finite = |_x: &[f64]| f64::NAN;
        let error = epsilon_constraint(
            &[non_finite, non_finite],
            &[gradient, gradient],
            &[0.0],
            &[],
            &EpsilonConstraintConfig::default(),
        )
        .expect_err("non-finite objective must fail at the public boundary");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }

    #[test]
    fn epsilon_constraint_rejects_nonfinite_gradient_at_initial_point() {
        let objective = |_x: &[f64]| 0.0;
        let gradient = |_x: &[f64]| vec![f64::INFINITY];
        let error = epsilon_constraint(
            &[objective, objective],
            &[gradient, gradient],
            &[0.0],
            &[],
            &EpsilonConstraintConfig::default(),
        )
        .expect_err("non-finite gradient must fail at the public boundary");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }

    #[test]
    fn epsilon_constraint_propagates_later_nonfinite_objective() {
        let objective_calls = AtomicUsize::new(0);
        let objective = |x: &[f64]| {
            if objective_calls.fetch_add(1, Ordering::Relaxed) >= 4 {
                f64::NAN
            } else {
                x[0] * x[0]
            }
        };
        let gradient = |x: &[f64]| vec![2.0 * x[0]];
        let objectives = [&objective, &objective];
        let gradients = [&gradient, &gradient];
        let config = EpsilonConstraintConfig {
            num_points: 2,
            max_iterations: 10,
            ..Default::default()
        };

        let error = epsilon_constraint(&objectives, &gradients, &[1.0], &[], &config)
            .expect_err("a later non-finite objective must not become an empty Pareto success");
        assert!(matches!(error, OptimError::NumericalError { .. }));
        assert!(objective_calls.load(Ordering::Relaxed) > 4);
    }

    #[test]
    fn epsilon_constraint_propagates_later_wrong_gradient_shape() {
        let gradient_calls = AtomicUsize::new(0);
        let objective = |x: &[f64]| x[0] * x[0];
        let gradient = |x: &[f64]| {
            if gradient_calls.fetch_add(1, Ordering::Relaxed) >= 4 {
                Vec::new()
            } else {
                vec![2.0 * x[0]]
            }
        };
        let objectives = [&objective, &objective];
        let gradients = [&gradient, &gradient];
        let config = EpsilonConstraintConfig {
            num_points: 2,
            max_iterations: 10,
            ..Default::default()
        };

        let error = epsilon_constraint(&objectives, &gradients, &[1.0], &[], &config)
            .expect_err("a later malformed gradient must not be silently skipped");
        assert!(matches!(error, OptimError::DimensionMismatch { .. }));
        assert!(gradient_calls.load(Ordering::Relaxed) > 4);
    }

    #[test]
    fn epsilon_constraint_propagates_later_constraint_failure() {
        let constraint_calls = Arc::new(AtomicUsize::new(0));
        let objective = |x: &[f64]| x[0] * x[0];
        let gradient = |x: &[f64]| vec![2.0 * x[0]];
        let objectives = [&objective, &objective];
        let gradients = [&gradient, &gradient];
        let calls = Arc::clone(&constraint_calls);
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(move |x: &[f64]| {
                if calls.fetch_add(1, Ordering::Relaxed) >= 3 {
                    vec![f64::INFINITY]
                } else {
                    vec![x[0] + 2.0]
                }
            }),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
            })),
        }];
        let config = EpsilonConstraintConfig {
            num_points: 2,
            max_iterations: 10,
            ..Default::default()
        };

        let error = epsilon_constraint(&objectives, &gradients, &[1.0], &constraints, &config)
            .expect_err("a later invalid constraint must not be silently skipped");
        assert!(matches!(error, OptimError::NumericalError { .. }));
        assert!(constraint_calls.load(Ordering::Relaxed) > 3);
    }
}
