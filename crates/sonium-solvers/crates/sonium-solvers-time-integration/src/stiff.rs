//! TR-BDF2 stiff integrator for first-order systems and index-1 DAEs.
//!
//! Solves `M y' = f(t, y)` with optional mass matrix `M`. The method is the
//! L-stable, second-order TR-BDF2 composite (trapezoidal stage over
//! `gamma h` followed by a variable-step BDF2 stage), with an embedded
//! companion estimator and adaptive step control. A singular `M` selects the
//! semi-explicit index-1 DAE path; the Newton iteration matrices stay
//! nonsingular there by the index-1 property, while higher-index systems
//! surface as an explicit singular-matrix error.
//!
//! Scope: dense Newton algebra for small-to-medium stiff systems (reduced
//! structural/thermal models, Robertson-class chemistry, constrained
//! single-body dynamics). Large sparse stiff systems need a Krylov Newton
//! variant that is not provided here.

// Rust guideline compliant 2026-02-21

use sonium_solvers_core::traits::SolverError;

/// TR-BDF2 stage fraction `gamma = 2 - sqrt(2)`.
pub const TRBDF2_GAMMA: f64 = 0.585_786_437_626_905;

/// Configuration for [`trbdf2_integrate`].
#[derive(Debug, Clone)]
pub struct TrBdf2Config {
    /// Initial (or, with `fixed_step`, uniform) step size in seconds.
    pub dt_initial: f64,
    /// Relative tolerance for the embedded local-error estimate.
    pub rel_tol: f64,
    /// Absolute tolerance for the embedded local-error estimate.
    pub abs_tol: f64,
    /// Smallest admissible step; the run fails cleanly below it.
    pub min_dt: f64,
    /// Largest admissible step (`f64::INFINITY` disables the cap).
    pub max_dt: f64,
    /// Maximum accepted steps before the run is rejected as too costly.
    pub max_steps: usize,
    /// Newton iterations per implicit stage before the step is rejected.
    pub max_newton_iterations: usize,
    /// RMS residual target for each stage Newton solve.
    pub newton_tolerance: f64,
    /// Safety factor applied to the controller step prediction.
    pub safety_factor: f64,
    /// Minimum step-size rescale per attempt.
    pub min_scale: f64,
    /// Maximum step-size growth per accepted step.
    pub max_scale: f64,
    /// Take uniform steps of `dt_initial` (adjusted to land on `t1`) with
    /// no rejection; error estimates are still reported per step.
    pub fixed_step: bool,
}

impl Default for TrBdf2Config {
    fn default() -> Self {
        Self {
            dt_initial: 1.0e-3,
            rel_tol: 1.0e-6,
            abs_tol: 1.0e-9,
            min_dt: 1.0e-12,
            max_dt: f64::INFINITY,
            max_steps: 100_000,
            max_newton_iterations: 12,
            newton_tolerance: 1.0e-11,
            safety_factor: 0.9,
            min_scale: 0.2,
            max_scale: 8.0,
            fixed_step: false,
        }
    }
}

impl TrBdf2Config {
    fn validate(&self) -> Result<(), SolverError> {
        let operation = "TR-BDF2 configuration";
        for (name, value) in [
            ("dt_initial", self.dt_initial),
            ("rel_tol", self.rel_tol),
            ("abs_tol", self.abs_tol),
            ("min_dt", self.min_dt),
            ("newton_tolerance", self.newton_tolerance),
            ("safety_factor", self.safety_factor),
            ("min_scale", self.min_scale),
            ("max_scale", self.max_scale),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!("{name} must be finite and positive"),
                });
            }
        }
        if !(self.max_dt.is_finite() || self.max_dt.is_infinite() && self.max_dt > 0.0)
            || self.max_dt <= 0.0
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "max_dt must be positive or +infinity".to_string(),
            });
        }
        if self.max_steps == 0 {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "max_steps must be positive".to_string(),
            });
        }
        if self.max_newton_iterations == 0 {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "max_newton_iterations must be positive".to_string(),
            });
        }
        if self.min_scale >= 1.0 || self.max_scale <= 1.0 {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "require min_scale < 1 < max_scale".to_string(),
            });
        }
        Ok(())
    }
}

/// One attempted step: the variable-step history contract.
#[derive(Debug, Clone)]
pub struct TrBdf2StepRecord {
    /// Step start time.
    pub t: f64,
    /// Attempted step size.
    pub dt: f64,
    /// Scaled RMS of the embedded local-error estimate.
    pub error_estimate: f64,
    /// Newton iterations consumed by both stages.
    pub newton_iterations: usize,
    /// Whether the attempt was rejected by the controller.
    pub rejected: bool,
}

/// Accepted trajectory plus cost accounting.
#[derive(Debug, Clone)]
pub struct TrBdf2Result {
    /// Accepted time mesh including `t0` and `t1`.
    pub times: Vec<f64>,
    /// Accepted states aligned with `times`.
    pub states: Vec<Vec<f64>>,
    /// Accepted-step count.
    pub steps_accepted: usize,
    /// Rejected-attempt count.
    pub steps_rejected: usize,
    /// Right-hand-side evaluations.
    pub function_evaluations: usize,
    /// Jacobian evaluations.
    pub jacobian_evaluations: usize,
    /// Dense LU factorizations of Newton iteration matrices.
    pub factorizations: usize,
    /// Every attempt (accepted and rejected) in order.
    pub history: Vec<TrBdf2StepRecord>,
}

fn rms_norm(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn apply_mass(mass: Option<&Vec<Vec<f64>>>, vector: &[f64]) -> Vec<f64> {
    match mass {
        None => vector.to_vec(),
        Some(matrix) => matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
            .collect(),
    }
}

/// In-place LU with partial pivoting; returns the row permutation.
fn lu_factor(matrix: &mut [Vec<f64>]) -> Result<Vec<usize>, SolverError> {
    let dimension = matrix.len();
    let mut permutation: Vec<usize> = (0..dimension).collect();
    for column in 0..dimension {
        let mut pivot_row = column;
        let mut pivot_value = matrix[permutation[column]][column].abs();
        for row in (column + 1)..dimension {
            let candidate = matrix[permutation[row]][column].abs();
            if candidate > pivot_value {
                pivot_value = candidate;
                pivot_row = row;
            }
        }
        permutation.swap(column, pivot_row);
        let pivot = matrix[permutation[column]][column];
        if !pivot.is_finite() || pivot == 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "TR-BDF2 Newton factorization",
                reason: "iteration matrix is singular (higher-index DAE or degenerate Jacobian)"
                    .to_string(),
            });
        }
        for row in (column + 1)..dimension {
            let factor = matrix[permutation[row]][column] / pivot;
            matrix[permutation[row]][column] = factor;
            let pivot_segment = matrix[permutation[column]][(column + 1)..dimension].to_vec();
            let target = &mut matrix[permutation[row]][(column + 1)..dimension];
            for (slot, &pivot_value) in target.iter_mut().zip(pivot_segment.iter()) {
                *slot -= factor * pivot_value;
            }
        }
    }
    Ok(permutation)
}

fn lu_solve(factored: &[Vec<f64>], permutation: &[usize], rhs: &[f64]) -> Vec<f64> {
    let dimension = factored.len();
    let mut solution = vec![0.0; dimension];
    for row in 0..dimension {
        let mut accumulator = rhs[permutation[row]];
        for column in 0..row {
            accumulator -= factored[permutation[row]][column] * solution[column];
        }
        solution[row] = accumulator;
    }
    for row in (0..dimension).rev() {
        let mut accumulator = solution[row];
        for column in (row + 1)..dimension {
            accumulator -= factored[permutation[row]][column] * solution[column];
        }
        solution[row] = accumulator / factored[permutation[row]][row];
    }
    solution
}

fn check_vector(vector: &[f64], dimension: usize, what: &'static str) -> Result<(), SolverError> {
    if vector.len() != dimension {
        return Err(SolverError::DimensionMismatch {
            expected: dimension,
            got: vector.len(),
        });
    }
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "TR-BDF2 callback",
            reason: format!("{what} must be finite"),
        });
    }
    Ok(())
}

/// Accepted Newton stages plus their force evaluations and iteration count:
/// `(stage1, stage2, force0, force1, force2, newton_iterations)`.
type AttemptedStages = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize);

fn check_jacobian(matrix: &[Vec<f64>], dimension: usize) -> Result<(), SolverError> {
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return Err(SolverError::DimensionMismatch {
            expected: dimension,
            got: matrix.len(),
        });
    }
    if !matrix
        .iter()
        .all(|row| row.iter().all(|value| value.is_finite()))
    {
        return Err(SolverError::OperationFailed {
            operation: "TR-BDF2 callback",
            reason: "Jacobian must be finite".to_string(),
        });
    }
    Ok(())
}

/// Integrate `M y' = f(t, y)` from `t0` to `t1`.
///
/// `mass` is `None` for the identity or a dense `n x n` matrix, which may
/// be singular for semi-explicit index-1 DAEs. Step control uses the
/// mass-weighted local residual of an embedded third-order companion, so
/// the same controller serves the ODE and DAE paths.
///
/// # Errors
///
/// Returns [`SolverError`] on invalid configuration or inputs, callback
/// dimension/finiteness violations, singular Newton matrices, exhausted
/// Newton iterations at the minimum step, or exceeded step budgets.
pub fn trbdf2_integrate<Rhs, Jacobian>(
    rhs: Rhs,
    jacobian: Jacobian,
    mass: Option<&[Vec<f64>]>,
    initial: &[f64],
    t0: f64,
    t1: f64,
    config: &TrBdf2Config,
) -> Result<TrBdf2Result, SolverError>
where
    Rhs: Fn(f64, &[f64]) -> Vec<f64>,
    Jacobian: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    let operation = "TR-BDF2 integration";
    config.validate()?;
    if !t0.is_finite() || !t1.is_finite() || t1 <= t0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "require finite t0 < t1".to_string(),
        });
    }
    let dimension = initial.len();
    check_vector(initial, dimension.max(1), "initial state")?;
    if dimension == 0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "initial state must be non-empty".to_string(),
        });
    }
    let owned_mass: Option<Vec<Vec<f64>>> = match mass {
        None => None,
        Some(matrix) => {
            check_jacobian(matrix, dimension)?;
            Some(matrix.to_vec())
        }
    };
    let mass_ref = owned_mass.as_ref();

    // Stage constants: rho = h2/h1 with h1 = gamma h, h2 = (1-gamma) h.
    let gamma = TRBDF2_GAMMA;
    let rho = (1.0 - gamma) / gamma;
    let bdf_c1 = (1.0 + rho) * (1.0 + rho) / (1.0 + 2.0 * rho);
    let bdf_c0 = -rho * rho / (1.0 + 2.0 * rho);
    let bdf_gain = (1.0 + rho) / (1.0 + 2.0 * rho);
    // Embedded companion weights from the order conditions
    // b0 + b1 + b2 = 1, b1 gamma + b2 = 1/2, b1 gamma^2 + b2 = 1/3.
    let companion_b1 = 1.0 / (6.0 * gamma * (1.0 - gamma));
    let companion_b2 = 0.5 - companion_b1 * gamma;
    let companion_b0 = 1.0 - companion_b1 - companion_b2;

    let mut times = vec![t0];
    let mut states = vec![initial.to_vec()];
    let mut history = Vec::new();
    let mut function_evaluations = 0usize;
    let mut jacobian_evaluations = 0usize;
    let mut factorizations = 0usize;
    let mut steps_accepted = 0usize;
    let mut steps_rejected = 0usize;

    let mut time = t0;
    let mut state = initial.to_vec();
    let mut consecutive_failures = 0usize;
    let mut step = config.dt_initial.min(config.max_dt).min(t1 - t0);
    if !step.is_finite() || step <= 0.0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "initial step underflows the interval".to_string(),
        });
    }
    if config.fixed_step {
        let steps = ((t1 - t0) / step).ceil().max(1.0) as usize;
        step = (t1 - t0) / steps as f64;
    }

    let evaluate_rhs =
        |time: f64, state: &[f64], count: &mut usize| -> Result<Vec<f64>, SolverError> {
            let values = rhs(time, state);
            *count += 1;
            check_vector(&values, dimension, "right-hand side")?;
            Ok(values)
        };
    let evaluate_jacobian =
        |time: f64, state: &[f64], count: &mut usize| -> Result<Vec<Vec<f64>>, SolverError> {
            let matrix = jacobian(time, state);
            *count += 1;
            check_jacobian(&matrix, dimension)?;
            Ok(matrix)
        };

    while time < t1 {
        if steps_accepted >= config.max_steps {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!("max_steps ({}) exceeded", config.max_steps),
            });
        }
        let remaining = t1 - time;
        if remaining <= 0.0 {
            break;
        }
        // Final landing micro-step: accept the Newton solution without
        // control rather than failing a run that already arrived.
        let landing = remaining < config.min_dt;
        step = step.min(remaining).min(config.max_dt);
        if !landing && step < config.min_dt {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!("step size {step} fell below min_dt {}", config.min_dt),
            });
        }
        let stage1_time = time + gamma * step;
        let stage2_time = time + step;

        // Attempt both implicit stages with exact Newton.
        let attempt = (|| -> Result<AttemptedStages, SolverError> {
            let mut newton_total = 0usize;
            let force0 = evaluate_rhs(time, &state, &mut function_evaluations)?;
            // Stage 1: trapezoid over gamma h.
            let mut stage1 = state.clone();
            let stage1_gain = 0.5 * gamma * step;
            for _ in 0..config.max_newton_iterations {
                let force1 = evaluate_rhs(stage1_time, &stage1, &mut function_evaluations)?;
                let mut residual = apply_mass(mass_ref, &stage1);
                let mass_state = apply_mass(mass_ref, &state);
                for row in 0..dimension {
                    residual[row] -= mass_state[row] + stage1_gain * (force0[row] + force1[row]);
                }
                check_vector(&residual, dimension, "stage-1 residual")?;
                if rms_norm(&residual) <= config.newton_tolerance {
                    break;
                }
                let jacobian1 = evaluate_jacobian(stage1_time, &stage1, &mut jacobian_evaluations)?;
                let mut iteration = vec![vec![0.0; dimension]; dimension];
                for row in 0..dimension {
                    for column in 0..dimension {
                        let mass_entry = mass_ref.map_or_else(
                            || if row == column { 1.0 } else { 0.0 },
                            |matrix| matrix[row][column],
                        );
                        iteration[row][column] = mass_entry - stage1_gain * jacobian1[row][column];
                    }
                }
                let permutation = lu_factor(&mut iteration)?;
                factorizations += 1;
                let correction = lu_solve(&iteration, &permutation, &residual);
                check_vector(&correction, dimension, "stage-1 correction")?;
                for row in 0..dimension {
                    stage1[row] -= correction[row];
                }
                newton_total += 1;
            }
            let force1 = evaluate_rhs(stage1_time, &stage1, &mut function_evaluations)?;
            let mut stage1_residual = apply_mass(mass_ref, &stage1);
            let mass_state = apply_mass(mass_ref, &state);
            for row in 0..dimension {
                stage1_residual[row] -= mass_state[row] + stage1_gain * (force0[row] + force1[row]);
            }
            if rms_norm(&stage1_residual) > config.newton_tolerance {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: "stage-1 Newton failed to converge".to_string(),
                });
            }
            // Stage 2: variable-step BDF2 over h2 = (1-gamma) h.
            let stage2_gain = bdf_gain * (1.0 - gamma) * step;
            let mut stage2 = stage1.clone();
            for _ in 0..config.max_newton_iterations {
                let force2 = evaluate_rhs(stage2_time, &stage2, &mut function_evaluations)?;
                let mut residual = apply_mass(mass_ref, &stage2);
                let mass_stage1 = apply_mass(mass_ref, &stage1);
                for row in 0..dimension {
                    residual[row] -= bdf_c1 * mass_stage1[row]
                        + bdf_c0 * mass_state[row]
                        + stage2_gain * force2[row];
                }
                check_vector(&residual, dimension, "stage-2 residual")?;
                if rms_norm(&residual) <= config.newton_tolerance {
                    break;
                }
                let jacobian2 = evaluate_jacobian(stage2_time, &stage2, &mut jacobian_evaluations)?;
                let mut iteration = vec![vec![0.0; dimension]; dimension];
                for row in 0..dimension {
                    for column in 0..dimension {
                        let mass_entry = mass_ref.map_or_else(
                            || if row == column { 1.0 } else { 0.0 },
                            |matrix| matrix[row][column],
                        );
                        iteration[row][column] = mass_entry - stage2_gain * jacobian2[row][column];
                    }
                }
                let permutation = lu_factor(&mut iteration)?;
                factorizations += 1;
                let correction = lu_solve(&iteration, &permutation, &residual);
                check_vector(&correction, dimension, "stage-2 correction")?;
                for row in 0..dimension {
                    stage2[row] -= correction[row];
                }
                newton_total += 1;
            }
            let force2 = evaluate_rhs(stage2_time, &stage2, &mut function_evaluations)?;
            let mut stage2_residual = apply_mass(mass_ref, &stage2);
            let mass_stage1 = apply_mass(mass_ref, &stage1);
            for row in 0..dimension {
                stage2_residual[row] -= bdf_c1 * mass_stage1[row]
                    + bdf_c0 * mass_state[row]
                    + stage2_gain * force2[row];
            }
            if rms_norm(&stage2_residual) > config.newton_tolerance {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: "stage-2 Newton failed to converge".to_string(),
                });
            }
            Ok((stage1, stage2, force0, force1, force2, newton_total))
        })();

        let (stage1, stage2, force0, force1, force2, newton_used) = match attempt {
            Ok(values) => values,
            Err(error) => {
                if config.fixed_step {
                    return Err(error);
                }
                consecutive_failures += 1;
                if consecutive_failures > 50 {
                    return Err(error);
                }
                steps_rejected += 1;
                history.push(TrBdf2StepRecord {
                    t: time,
                    dt: step,
                    error_estimate: f64::INFINITY,
                    newton_iterations: config.max_newton_iterations,
                    rejected: true,
                });
                step *= config.min_scale;
                continue;
            }
        };

        // Embedded companion residual: M(y2 - y0) - h(b0 f0 + b1 f1 + b2 f2).
        let mass_new = apply_mass(mass_ref, &stage2);
        let mass_old = apply_mass(mass_ref, &state);
        let mut error_squared = 0.0;
        for row in 0..dimension {
            let estimate = mass_new[row]
                - mass_old[row]
                - step
                    * (companion_b0 * force0[row]
                        + companion_b1 * force1[row]
                        + companion_b2 * force2[row]);
            let scale =
                config.abs_tol + config.rel_tol * mass_new[row].abs().max(mass_old[row].abs());
            error_squared += (estimate / scale) * (estimate / scale);
        }
        let error = (error_squared / dimension as f64).sqrt();
        if !error.is_finite() {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "error estimate is non-finite".to_string(),
            });
        }

        if config.fixed_step || landing || error <= 1.0 {
            time = stage2_time;
            state = stage2.clone();
            if t1 - time <= 8.0 * f64::EPSILON * t1.abs().max(1.0) {
                time = t1;
            }
            times.push(time);
            states.push(stage2);
            steps_accepted += 1;
            consecutive_failures = 0;
            history.push(TrBdf2StepRecord {
                t: time,
                dt: step,
                error_estimate: error,
                newton_iterations: newton_used,
                rejected: false,
            });
            let _ = stage1;
            if !config.fixed_step {
                let factor = if error == 0.0 {
                    config.max_scale
                } else {
                    (config.safety_factor * error.powf(-1.0 / 3.0)).clamp(1.0, config.max_scale)
                };
                step *= factor;
            }
        } else {
            steps_rejected += 1;
            history.push(TrBdf2StepRecord {
                t: time,
                dt: step,
                error_estimate: error,
                newton_iterations: newton_used,
                rejected: true,
            });
            let factor =
                (config.safety_factor * error.powf(-1.0 / 3.0)).clamp(config.min_scale, 1.0);
            step *= factor;
        }
    }

    Ok(TrBdf2Result {
        times,
        states,
        steps_accepted,
        steps_rejected,
        function_evaluations,
        jacobian_evaluations,
        factorizations,
        history,
    })
}
