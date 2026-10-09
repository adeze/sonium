//! Multi-step discrete adjoints for transient residual trajectories.
//!
//! A steady adjoint cannot be reused one time step at a time: BDF history
//! terms couple every later multiplier to earlier states. This module owns the
//! reusable backward-sweep layer extracted from `sonium-sim-pcfan`'s
//! `transient_adjoint`: validated CSR history Jacobians, the exact backward
//! equation `A_nᵀ λ_n = φ_{x_n} - Σ_{m>n} H_{m,n}ᵀ λ_m` with a checked
//! transpose-GMRES inner solve, and the design-gradient accumulation
//! `Σ(φ_d - R_dᵀ λ)`.
//!
//! Flow models, backends, and provenance policy stay in the simulation
//! crates, which validate their own trajectory metadata and convert to the
//! [`TransientAdjointStep`] layout before delegating here.

/// Validated square CSR state Jacobian for one transient residual block.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientAdjointCsr {
    pub dimension: usize,
    pub row_ptrs: Vec<usize>,
    pub column_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl TransientAdjointCsr {
    /// Construct and validate a CSR state Jacobian.
    pub fn new(
        dimension: usize,
        row_ptrs: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, TransientAdjointError> {
        let matrix = Self {
            dimension,
            row_ptrs,
            column_indices,
            values,
        };
        validate_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Build a CSR matrix from a dense reference matrix for small regression
    /// fixtures and assembly-level tests.
    pub fn from_dense(dense: &[Vec<f64>]) -> Result<Self, TransientAdjointError> {
        let dimension = dense.len();
        if dense.iter().any(|row| row.len() != dimension) {
            return Err(TransientAdjointError::InvalidMatrix(
                "dense matrix must be square",
            ));
        }
        let mut row_ptrs = Vec::with_capacity(dimension + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_ptrs.push(0);
        for row in dense {
            for (column, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(TransientAdjointError::NonFiniteMatrixValue);
                }
                if value != 0.0 {
                    column_indices.push(column);
                    values.push(value);
                }
            }
            row_ptrs.push(values.len());
        }
        Self::new(dimension, row_ptrs, column_indices, values)
    }
}

/// Numerical controls for the sparse transpose solves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientAdjointConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_state_residual: f64,
}

impl Default for TransientAdjointConfig {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-10,
            max_iterations: 512,
            max_state_residual: 1.0e-6,
        }
    }
}

/// Fail-closed errors for transient-adjoint admission or solution.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TransientAdjointError {
    #[error("invalid transient adjoint matrix: {0}")]
    InvalidMatrix(&'static str),
    #[error("transient adjoint matrix has a non-finite value")]
    NonFiniteMatrixValue,
    #[error("invalid transient adjoint config {0} = {1}")]
    InvalidConfig(&'static str, f64),
    #[error("transient adjoint iteration limit must be positive")]
    InvalidIterationLimit,
    #[error("transient adjoint dimension mismatch: {0}")]
    DimensionMismatch(&'static str),
    #[error("transient adjoint non-finite input: {0}")]
    NonFiniteInput(&'static str),
    #[error("transient state residual {norm} exceeds maximum {maximum}")]
    StateResidualTooLarge { norm: f64, maximum: f64 },
    #[error(
        "transient adjoint did not converge after {iterations} iterations: residual {residual}, tolerance {tolerance}"
    )]
    DidNotConverge {
        iterations: usize,
        residual: f64,
        tolerance: f64,
    },
    #[error("singular transient adjoint Hessenberg system at index {index}")]
    SingularAdjointSystem { index: usize },
}
/// A history derivative `∂R_n/∂x_previous_step` in a transient residual.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientHistoryJacobian {
    pub previous_step: usize,
    pub matrix: TransientAdjointCsr,
}

/// Derivative of a transient residual with respect to a BDF history state
/// that is not represented by an earlier entry in the trajectory.
///
/// `stage` identifies the residual block and `history_slot` identifies the
/// incoming BDF level (for example, slot 0 may be the immediately previous
/// state and slot 1 the state before it).  The matrix has the same square
/// state dimension as the residual at `stage`.
///
/// These blocks are intentionally separate from
/// [`TransientHistoryJacobian`]: an out-of-trajectory history
/// state has no trajectory adjoint slot to receive a contribution, so its
/// derivative must be returned explicitly by the adjoint solve.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientStageInitialHistoryJacobian {
    pub stage: usize,
    pub history_slot: usize,
    pub matrix: TransientAdjointCsr,
    /// Direct objective derivative with respect to this incoming history
    /// state. Use a zero vector when the objective has no such term.
    pub objective_gradient: Vec<f64>,
}

/// Objective derivative with respect to one incoming BDF history state.
///
/// The derivative uses the residual convention `R = 0`, so it is
/// `φ_history - (∂R/∂history)ᵀ λ_stage`. Entries preserve the order and labels
/// of the corresponding [`TransientStageInitialHistoryJacobian`]
/// input.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientInitialHistoryAdjoint {
    pub stage: usize,
    pub history_slot: usize,
    pub adjoint: Vec<f64>,
}

/// Resolved transient adjoint result including derivatives for incoming BDF
/// history states.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientAdjointWithInitialHistory {
    pub report: TransientAdjointReport,
    pub initial_history_adjoints: Vec<TransientInitialHistoryAdjoint>,
}

/// One checked residual/objective block for a transient discrete adjoint.
///
/// The residual convention is `R_n(x_n, x_history, d) = 0`; the objective is
/// the sum of the per-step objectives. All derivatives must come from the
/// same converged assembly. Flow-model, backend, and provenance admission
/// stay with the calling simulation, which converts to this layout.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientAdjointStep {
    pub state_jacobian: TransientAdjointCsr,
    pub state_residual: Vec<f64>,
    pub history_jacobians: Vec<TransientHistoryJacobian>,
    pub residual_design_jacobian: Vec<Vec<f64>>,
    pub objective_state_gradient: Vec<f64>,
    pub objective_design_gradient: Vec<f64>,
}

/// Multi-step discrete-adjoint result.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientAdjointReport {
    /// One multiplier vector per retained transient step, chronological.
    pub adjoints: Vec<Vec<f64>>,
    pub design_gradient: Vec<f64>,
    pub state_residual_norm: f64,
    pub adjoint_residual_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Solve the exact backward discrete adjoint of a checked transient residual
/// sequence.
///
/// `A_n = ∂R_n/∂x_n` and `H_{m,n} = ∂R_m/∂x_n`; the backward equation is
/// `A_nᵀ λ_n = φ_{x_n} - Σ_{m>n} H_{m,n}ᵀ λ_m`. The returned design gradient
/// is `Σ(φ_d - R_dᵀ λ)`. This keeps BDF history coupling explicit.
pub fn solve_transient_adjoint(
    config: &TransientAdjointConfig,
    steps: &[TransientAdjointStep],
) -> Result<TransientAdjointReport, TransientAdjointError> {
    validate_config(config)?;
    validate_steps(steps)?;

    let first = steps
        .first()
        .ok_or(TransientAdjointError::DimensionMismatch(
            "transient adjoint requires at least one step",
        ))?;
    let design_dimension = first.objective_design_gradient.len();
    let mut adjoints = vec![Vec::<f64>::new(); steps.len()];
    let mut design_gradient = vec![0.0; design_dimension];
    let state_residual_norm = steps
        .iter()
        .map(|step| l2_norm(&step.state_residual))
        .fold(0.0_f64, f64::max);
    if state_residual_norm > config.max_state_residual {
        return Err(TransientAdjointError::StateResidualTooLarge {
            norm: state_residual_norm,
            maximum: config.max_state_residual,
        });
    }

    let mut adjoint_residual_norm: f64 = 0.0;
    let mut iterations = 0usize;
    for step_index in (0..steps.len()).rev() {
        let step = &steps[step_index];
        let mut rhs = step.objective_state_gradient.clone();

        for (future_index, future_step) in steps.iter().enumerate().skip(step_index + 1) {
            for history in &future_step.history_jacobians {
                if history.previous_step != step_index {
                    continue;
                }
                let contribution = transpose_matvec(&history.matrix, &adjoints[future_index]);
                for (value, contribution) in rhs.iter_mut().zip(contribution) {
                    *value -= contribution;
                }
            }
        }

        let (adjoint, used_iterations, residual) = solve_transpose_gmres(
            &step.state_jacobian,
            &rhs,
            config.tolerance,
            config.max_iterations,
        )?;
        adjoint_residual_norm = adjoint_residual_norm.max(residual);
        iterations = iterations.saturating_add(used_iterations);
        for design_index in 0..design_dimension {
            let residual_design = step
                .residual_design_jacobian
                .iter()
                .zip(&adjoint)
                .map(|(row, &multiplier)| row[design_index] * multiplier)
                .sum::<f64>();
            design_gradient[design_index] +=
                step.objective_design_gradient[design_index] - residual_design;
        }
        adjoints[step_index] = adjoint;
    }

    if !design_gradient.iter().all(|value| value.is_finite()) {
        return Err(TransientAdjointError::NonFiniteInput(
            "transient design gradient",
        ));
    }
    Ok(TransientAdjointReport {
        adjoints,
        design_gradient,
        state_residual_norm,
        adjoint_residual_norm,
        iterations,
        converged: true,
    })
}

/// Solve the transient adjoint and contract its multipliers with BDF history
/// states entering at stage boundaries.
///
/// The existing [`solve_transient_adjoint`] API remains the
/// zero-initial-history convenience form. This checked form is for a fixed-
/// dimensional trajectory segment whose incoming BDF states are retained
/// separately from the represented trajectory. A caller at a remesh/restart
/// handoff must first express those states in the segment's coordinates. It
/// does not differentiate the remesher, topology choice, or mesh geometry.
pub fn solve_transient_adjoint_with_stage_initial_history(
    config: &TransientAdjointConfig,
    steps: &[TransientAdjointStep],
    initial_history_jacobians: &[TransientStageInitialHistoryJacobian],
) -> Result<TransientAdjointWithInitialHistory, TransientAdjointError> {
    validate_config(config)?;
    validate_steps(steps)?;
    validate_stage_initial_history_jacobians(steps, initial_history_jacobians)?;

    let report = solve_transient_adjoint(config, steps)?;
    let initial_history_adjoints = initial_history_jacobians
        .iter()
        .map(|history| TransientInitialHistoryAdjoint {
            stage: history.stage,
            history_slot: history.history_slot,
            adjoint: history
                .objective_gradient
                .iter()
                .zip(transpose_matvec(
                    &history.matrix,
                    &report.adjoints[history.stage],
                ))
                .map(|(&objective, residual)| objective - residual)
                .collect(),
        })
        .collect();

    Ok(TransientAdjointWithInitialHistory {
        report,
        initial_history_adjoints,
    })
}

fn validate_stage_initial_history_jacobians(
    steps: &[TransientAdjointStep],
    initial_history_jacobians: &[TransientStageInitialHistoryJacobian],
) -> Result<(), TransientAdjointError> {
    for history in initial_history_jacobians {
        let Some(step) = steps.get(history.stage) else {
            return Err(TransientAdjointError::DimensionMismatch(
                "transient initial-history stage index",
            ));
        };

        validate_matrix(&history.matrix)?;
        if history.matrix.dimension != step.state_jacobian.dimension {
            return Err(TransientAdjointError::DimensionMismatch(
                "transient initial-history dimensions",
            ));
        }
        if history.objective_gradient.len() != history.matrix.dimension {
            return Err(TransientAdjointError::DimensionMismatch(
                "transient initial-history objective gradient",
            ));
        }
        if history
            .objective_gradient
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(TransientAdjointError::NonFiniteInput(
                "transient initial-history objective gradient",
            ));
        }
    }

    Ok(())
}

fn validate_config(config: &TransientAdjointConfig) -> Result<(), TransientAdjointError> {
    for (name, value) in [
        ("tolerance", config.tolerance),
        ("max_state_residual", config.max_state_residual),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(TransientAdjointError::InvalidConfig(name, value));
        }
    }
    if config.max_iterations == 0 {
        return Err(TransientAdjointError::InvalidIterationLimit);
    }
    Ok(())
}

fn validate_steps(steps: &[TransientAdjointStep]) -> Result<(), TransientAdjointError> {
    let first = steps
        .first()
        .ok_or(TransientAdjointError::DimensionMismatch(
            "transient adjoint requires at least one step",
        ))?;
    let state_dimension = first.state_jacobian.dimension;
    let design_dimension = first.objective_design_gradient.len();

    for (step_index, step) in steps.iter().enumerate() {
        validate_matrix(&step.state_jacobian)?;
        if step.state_jacobian.dimension != state_dimension
            || step.state_residual.len() != state_dimension
            || step.objective_state_gradient.len() != state_dimension
            || step.residual_design_jacobian.len() != state_dimension
            || step.objective_design_gradient.len() != design_dimension
        {
            return Err(TransientAdjointError::DimensionMismatch(
                "transient adjoint state or design dimensions",
            ));
        }
        if step.state_residual.iter().any(|value| !value.is_finite())
            || step
                .objective_state_gradient
                .iter()
                .any(|value| !value.is_finite())
            || step
                .objective_design_gradient
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(TransientAdjointError::NonFiniteInput(
                "transient adjoint objective or residual",
            ));
        }
        if step
            .residual_design_jacobian
            .iter()
            .any(|row| row.len() != design_dimension || row.iter().any(|value| !value.is_finite()))
        {
            return Err(TransientAdjointError::DimensionMismatch(
                "transient adjoint design residual dimensions",
            ));
        }
        for history in &step.history_jacobians {
            if history.previous_step >= step_index {
                return Err(TransientAdjointError::DimensionMismatch(
                    "transient adjoint history must be causal",
                ));
            }
            validate_matrix(&history.matrix)?;
            if history.matrix.dimension != state_dimension {
                return Err(TransientAdjointError::DimensionMismatch(
                    "transient adjoint history dimensions",
                ));
            }
        }
    }
    Ok(())
}

fn validate_matrix(matrix: &TransientAdjointCsr) -> Result<(), TransientAdjointError> {
    if matrix.row_ptrs.len() != matrix.dimension + 1
        || matrix.column_indices.len() != matrix.values.len()
        || matrix.row_ptrs.first().copied() != Some(0)
        || matrix.row_ptrs.last().copied() != Some(matrix.values.len())
        || matrix
            .row_ptrs
            .windows(2)
            .any(|window| window[0] > window[1])
        || matrix
            .column_indices
            .iter()
            .any(|&column| column >= matrix.dimension)
    {
        return Err(TransientAdjointError::InvalidMatrix(
            "transient adjoint CSR dimensions",
        ));
    }
    if matrix.values.iter().any(|value| !value.is_finite()) {
        return Err(TransientAdjointError::NonFiniteMatrixValue);
    }
    Ok(())
}

fn transpose_matvec(matrix: &TransientAdjointCsr, input: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; matrix.dimension];
    for (row, window) in matrix.row_ptrs.windows(2).enumerate() {
        for entry in window[0]..window[1] {
            output[matrix.column_indices[entry]] += matrix.values[entry] * input[row];
        }
    }
    output
}

fn solve_transpose_gmres(
    matrix: &TransientAdjointCsr,
    rhs: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Vec<f64>, usize, f64), TransientAdjointError> {
    let dimension = matrix.dimension;
    let beta = l2_norm(rhs);
    if beta <= tolerance {
        return Ok((vec![0.0; dimension], 0, 0.0));
    }

    let mut solution = vec![0.0; dimension];
    let mut basis = Vec::<Vec<f64>>::with_capacity(max_iterations);
    let mut hessenberg = vec![vec![0.0; max_iterations]; max_iterations + 1];
    let mut cosines = vec![0.0; max_iterations];
    let mut sines = vec![0.0; max_iterations];
    let mut transformed_rhs = vec![0.0; max_iterations + 1];
    transformed_rhs[0] = beta;
    let mut residual_estimate = beta;
    let mut used_iterations = 0usize;

    for iteration in 0..max_iterations {
        if iteration == 0 {
            basis.push(rhs.iter().map(|value| value / beta).collect());
        }
        let basis_vector =
            basis
                .get(iteration)
                .cloned()
                .ok_or(TransientAdjointError::DidNotConverge {
                    iterations: used_iterations,
                    residual: residual_estimate,
                    tolerance,
                })?;
        let mut work = transpose_matvec(matrix, &basis_vector);
        for basis_index in 0..=iteration {
            hessenberg[basis_index][iteration] = dot(&work, &basis[basis_index]);
            for (work_value, &basis_value) in work.iter_mut().zip(&basis[basis_index]) {
                *work_value -= hessenberg[basis_index][iteration] * basis_value;
            }
        }
        hessenberg[iteration + 1][iteration] = l2_norm(&work);
        if hessenberg[iteration + 1][iteration] > f64::EPSILON {
            basis.push(
                work.iter()
                    .map(|value| value / hessenberg[iteration + 1][iteration])
                    .collect(),
            );
        }

        for prior in 0..iteration {
            let rotated = cosines[prior] * hessenberg[prior][iteration]
                + sines[prior] * hessenberg[prior + 1][iteration];
            hessenberg[prior + 1][iteration] = -sines[prior] * hessenberg[prior][iteration]
                + cosines[prior] * hessenberg[prior + 1][iteration];
            hessenberg[prior][iteration] = rotated;
        }

        let diagonal = (hessenberg[iteration][iteration].powi(2)
            + hessenberg[iteration + 1][iteration].powi(2))
        .sqrt();
        if diagonal <= f64::EPSILON {
            return Err(TransientAdjointError::SingularAdjointSystem { index: iteration });
        }
        cosines[iteration] = hessenberg[iteration][iteration] / diagonal;
        sines[iteration] = hessenberg[iteration + 1][iteration] / diagonal;
        hessenberg[iteration][iteration] = diagonal;
        hessenberg[iteration + 1][iteration] = 0.0;
        let transformed = cosines[iteration] * transformed_rhs[iteration]
            + sines[iteration] * transformed_rhs[iteration + 1];
        transformed_rhs[iteration + 1] = -sines[iteration] * transformed_rhs[iteration]
            + cosines[iteration] * transformed_rhs[iteration + 1];
        transformed_rhs[iteration] = transformed;
        residual_estimate = transformed_rhs[iteration + 1].abs();
        used_iterations = iteration + 1;

        let coefficients = back_substitute(&hessenberg, &transformed_rhs, used_iterations)?;
        solution.fill(0.0);
        for (coefficient, vector) in coefficients.iter().zip(&basis) {
            for (solution_value, &basis_value) in solution.iter_mut().zip(vector) {
                *solution_value += coefficient * basis_value;
            }
        }
        let actual_residual = l2_norm(
            &rhs.iter()
                .zip(transpose_matvec(matrix, &solution))
                .map(|(&right, value)| right - value)
                .collect::<Vec<_>>(),
        );
        if actual_residual <= tolerance {
            return Ok((solution, used_iterations, actual_residual));
        }
    }

    Err(TransientAdjointError::DidNotConverge {
        iterations: used_iterations,
        residual: residual_estimate,
        tolerance,
    })
}

fn back_substitute(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    dimension: usize,
) -> Result<Vec<f64>, TransientAdjointError> {
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let diagonal = matrix[row][row];
        if diagonal.abs() <= f64::EPSILON {
            return Err(TransientAdjointError::SingularAdjointSystem { index: row });
        }
        let upper_sum = ((row + 1)..dimension)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - upper_sum) / diagonal;
    }
    Ok(solution)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(values: &[Vec<f64>]) -> TransientAdjointCsr {
        TransientAdjointCsr::from_dense(values).expect("valid test matrix")
    }

    /// Adjoint design gradient vs central finite differences of an explicit
    /// linear forward model with the same Jacobians.
    ///
    /// Forward: `R_0 = A_0 x_0 + B_0 d`, `R_1 = A_1 x_1 + H x_0 + B_1 d`
    /// (so `∂R/∂d = +B`, matching the solver convention
    /// `g = Σ(φ_d - R_dᵀ λ)`), objective
    /// `Φ = c_0·x_0 + c_1·x_1 + e·d`.
    #[test]
    fn adjoint_gradient_matches_finite_differences() {
        let a0 = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let a1 = vec![vec![4.0, 0.0], vec![0.0, 5.0]];
        let h = vec![vec![0.5, 0.0], vec![0.0, 1.0]];
        let b0 = vec![vec![1.0], vec![2.0]];
        let b1 = vec![vec![2.0], vec![1.0]];
        let c0 = vec![1.0, 2.0];
        let c1 = vec![3.0, 4.0];
        let e = vec![0.1, 0.2];

        // Closed-form forward states (diagonal systems, f = 0).
        let forward = |d: f64| {
            let x0 = vec![-b0[0][0] * d / a0[0][0], -b0[1][0] * d / a0[1][1]];
            let x1 = vec![
                -(b1[0][0] * d + h[0][0] * x0[0]) / a1[0][0],
                -(b1[1][0] * d + h[1][1] * x0[1]) / a1[1][1],
            ];
            (x0, x1)
        };
        let objective = |d: f64| {
            let (x0, x1) = forward(d);
            c0.iter().zip(&x0).map(|(a, b)| a * b).sum::<f64>()
                + c1.iter().zip(&x1).map(|(a, b)| a * b).sum::<f64>()
                + e.iter().sum::<f64>() * d
        };

        let steps = vec![
            TransientAdjointStep {
                state_jacobian: csr(&a0),
                state_residual: vec![0.0, 0.0],
                history_jacobians: vec![],
                residual_design_jacobian: b0.clone(),
                objective_state_gradient: c0.clone(),
                objective_design_gradient: vec![e[0]],
            },
            TransientAdjointStep {
                state_jacobian: csr(&a1),
                state_residual: vec![0.0, 0.0],
                history_jacobians: vec![TransientHistoryJacobian {
                    previous_step: 0,
                    matrix: csr(&h),
                }],
                residual_design_jacobian: b1.clone(),
                objective_state_gradient: c1.clone(),
                objective_design_gradient: vec![e[1]],
            },
        ];
        let report = solve_transient_adjoint(&TransientAdjointConfig::default(), &steps)
            .expect("adjoint solve");
        assert!(report.converged);

        let h_fd = 1.0e-7;
        let fd = (objective(h_fd) - objective(-h_fd)) / (2.0 * h_fd);
        assert!(
            (report.design_gradient[0] - fd).abs() < 1.0e-6,
            "adjoint {} vs finite-difference {fd}",
            report.design_gradient[0]
        );
    }

    #[test]
    fn malformed_trajectories_fail_closed() {
        let config = TransientAdjointConfig::default();
        assert!(matches!(
            solve_transient_adjoint(&config, &[]),
            Err(TransientAdjointError::DimensionMismatch(_))
        ));
        assert!(matches!(
            TransientAdjointCsr::from_dense(&[vec![1.0, 2.0]]),
            Err(TransientAdjointError::InvalidMatrix(_))
        ));
        assert!(matches!(
            TransientAdjointCsr::new(2, vec![0, 1, 1], vec![0], vec![f64::NAN]),
            Err(TransientAdjointError::NonFiniteMatrixValue)
        ));
        // Non-causal history: step 0 references step 0 itself.
        let bad = TransientAdjointStep {
            state_jacobian: csr(&[vec![1.0]]),
            state_residual: vec![0.0],
            history_jacobians: vec![TransientHistoryJacobian {
                previous_step: 0,
                matrix: csr(&[vec![1.0]]),
            }],
            residual_design_jacobian: vec![vec![0.0]],
            objective_state_gradient: vec![0.0],
            objective_design_gradient: vec![0.0],
        };
        assert!(matches!(
            solve_transient_adjoint(&config, &[bad]),
            Err(TransientAdjointError::DimensionMismatch(_))
        ));
    }
}
