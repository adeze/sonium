//! Multi-step discrete adjoints for resolved PC-fan trajectories.
//!
//! The steady resolved adjoint intentionally cannot be reused one time step
//! at a time: BDF history terms couple every later multiplier to earlier
//! states. This module keeps that coupling explicit and provides checked
//! CPU-reference linear algebra for production callers that publish the
//! matching residual blocks.

use crate::model::{PcFanCfdBackend, PcFanFlowModel};
use crate::resolved_adjoint::{
    PcFanResolvedAdjointConfig, PcFanResolvedAdjointError, PcFanResolvedFlowSparseMatrix,
};
use serde::{Deserialize, Serialize};

/// A history derivative `∂R_n/∂x_previous_step` in a transient residual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedTransientHistoryJacobian {
    pub previous_step: usize,
    pub matrix: PcFanResolvedFlowSparseMatrix,
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
/// [`PcFanResolvedTransientHistoryJacobian`]: an out-of-trajectory history
/// state has no trajectory adjoint slot to receive a contribution, so its
/// derivative must be returned explicitly by the adjoint solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedTransientStageInitialHistoryJacobian {
    pub stage: usize,
    pub history_slot: usize,
    pub matrix: PcFanResolvedFlowSparseMatrix,
    /// Direct objective derivative with respect to this incoming history
    /// state. Use a zero vector when the objective has no such term.
    pub objective_gradient: Vec<f64>,
}

/// Objective derivative with respect to one incoming BDF history state.
///
/// The derivative uses the residual convention `R = 0`, so it is
/// `φ_history - (∂R/∂history)ᵀ λ_stage`. Entries preserve the order and labels
/// of the corresponding [`PcFanResolvedTransientStageInitialHistoryJacobian`]
/// input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedTransientInitialHistoryAdjoint {
    pub stage: usize,
    pub history_slot: usize,
    pub adjoint: Vec<f64>,
}

/// Resolved transient adjoint result including derivatives for incoming BDF
/// history states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedTransientAdjointWithInitialHistory {
    pub report: PcFanResolvedTransientAdjointReport,
    pub initial_history_adjoints: Vec<PcFanResolvedTransientInitialHistoryAdjoint>,
}

/// One checked residual/objective block for a resolved transient adjoint.
///
/// The residual convention is `R_n(x_n, x_history, d) = 0`; the objective is
/// the sum of the per-step objectives. All derivatives must come from the
/// same converged production assembly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedTransientStep {
    pub flow_model: PcFanFlowModel,
    pub backend: PcFanCfdBackend,
    pub provenance: String,
    pub state_jacobian: PcFanResolvedFlowSparseMatrix,
    pub state_residual: Vec<f64>,
    pub history_jacobians: Vec<PcFanResolvedTransientHistoryJacobian>,
    pub residual_design_jacobian: Vec<Vec<f64>>,
    pub objective_state_gradient: Vec<f64>,
    pub objective_design_gradient: Vec<f64>,
}

/// Multi-step resolved-flow discrete-adjoint result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedTransientAdjointReport {
    pub flow_model: PcFanFlowModel,
    pub backend: PcFanCfdBackend,
    pub provenance: String,
    /// One multiplier vector per retained transient step, chronological.
    pub adjoints: Vec<Vec<f64>>,
    pub design_gradient: Vec<f64>,
    pub state_residual_norm: f64,
    pub adjoint_residual_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}
use sonium_solvers_time_integration::{
    TransientAdjointConfig, TransientAdjointCsr, TransientAdjointError, TransientAdjointStep,
    TransientHistoryJacobian, TransientStageInitialHistoryJacobian, solve_transient_adjoint,
    solve_transient_adjoint_with_stage_initial_history,
};

/// Map the reusable solver errors onto the identical product variants. Each
/// arm is explicit so a new generic variant fails to compile here instead of
/// landing on a wrong product error.
fn map_error(error: TransientAdjointError) -> PcFanResolvedAdjointError {
    match error {
        TransientAdjointError::InvalidMatrix(message) => {
            PcFanResolvedAdjointError::InvalidMatrix(message)
        }
        TransientAdjointError::NonFiniteMatrixValue => {
            PcFanResolvedAdjointError::NonFiniteMatrixValue
        }
        TransientAdjointError::InvalidConfig(name, value) => {
            PcFanResolvedAdjointError::InvalidConfig(name, value)
        }
        TransientAdjointError::InvalidIterationLimit => {
            PcFanResolvedAdjointError::InvalidIterationLimit
        }
        TransientAdjointError::DimensionMismatch(message) => {
            PcFanResolvedAdjointError::DimensionMismatch(message)
        }
        TransientAdjointError::NonFiniteInput(message) => {
            PcFanResolvedAdjointError::NonFiniteInput(message)
        }
        TransientAdjointError::StateResidualTooLarge { norm, maximum } => {
            PcFanResolvedAdjointError::StateResidualTooLarge { norm, maximum }
        }
        TransientAdjointError::DidNotConverge {
            iterations,
            residual,
            tolerance,
        } => PcFanResolvedAdjointError::DidNotConverge {
            iterations,
            residual,
            tolerance,
        },
        TransientAdjointError::SingularAdjointSystem { index } => {
            PcFanResolvedAdjointError::SingularAdjointSystem { index }
        }
    }
}

fn convert_matrix(
    matrix: &PcFanResolvedFlowSparseMatrix,
) -> Result<TransientAdjointCsr, PcFanResolvedAdjointError> {
    TransientAdjointCsr::new(
        matrix.dimension,
        matrix.row_ptrs.clone(),
        matrix.column_indices.clone(),
        matrix.values.clone(),
    )
    .map_err(map_error)
}

fn convert_history(
    history: &PcFanResolvedTransientHistoryJacobian,
) -> Result<TransientHistoryJacobian, PcFanResolvedAdjointError> {
    Ok(TransientHistoryJacobian {
        previous_step: history.previous_step,
        matrix: convert_matrix(&history.matrix)?,
    })
}

fn convert_step(
    step: &PcFanResolvedTransientStep,
) -> Result<TransientAdjointStep, PcFanResolvedAdjointError> {
    Ok(TransientAdjointStep {
        state_jacobian: convert_matrix(&step.state_jacobian)?,
        state_residual: step.state_residual.clone(),
        history_jacobians: step
            .history_jacobians
            .iter()
            .map(convert_history)
            .collect::<Result<Vec<_>, _>>()?,
        residual_design_jacobian: step.residual_design_jacobian.clone(),
        objective_state_gradient: step.objective_state_gradient.clone(),
        objective_design_gradient: step.objective_design_gradient.clone(),
    })
}

fn convert_config(config: &PcFanResolvedAdjointConfig) -> TransientAdjointConfig {
    TransientAdjointConfig {
        tolerance: config.tolerance,
        max_iterations: config.max_iterations,
        max_state_residual: config.max_state_residual,
    }
}

/// Product trajectory admission: flow-model, backend, and provenance policy.
///
/// Dimension, finiteness, causality, and CSR validity are enforced by the
/// reusable solver; this gate keeps the PC-fan assembly contract (converged
/// CFD Newton tangents with recorded provenance, no reduced/probe markers).
fn validate_product_topology(
    steps: &[PcFanResolvedTransientStep],
) -> Result<(), PcFanResolvedAdjointError> {
    let Some(first) = steps.first() else {
        return Ok(());
    };
    for step in steps {
        if !step.flow_model.is_cfd() {
            return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
                step.flow_model,
            ));
        }
        if step.flow_model != first.flow_model || step.backend != first.backend {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient adjoint backend or flow topology",
            ));
        }
        if step.provenance.trim().is_empty() {
            return Err(PcFanResolvedAdjointError::MissingProvenance);
        }
        let provenance = step.provenance.to_ascii_lowercase();
        if [
            "picard",
            "oseen",
            "derivative-probe",
            "finite-difference",
            "black-box",
        ]
        .iter()
        .any(|marker| provenance.contains(marker))
        {
            return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
        }
    }
    Ok(())
}

/// Solve the exact backward discrete adjoint of a checked transient residual
/// sequence.
///
/// `A_n = ∂R_n/∂x_n` and `H_{m,n} = ∂R_m/∂x_n`; the backward equation is
/// `A_nᵀ λ_n = φ_{x_n} - Σ_{m>n} H_{m,n}ᵀ λ_m`. The returned design gradient
/// is `Σ(φ_d - R_dᵀ λ)`. This keeps BDF history coupling explicit. Product
/// admission first, reusable backward sweep second.
pub fn solve_pc_fan_resolved_transient_adjoint(
    config: &PcFanResolvedAdjointConfig,
    steps: &[PcFanResolvedTransientStep],
) -> Result<PcFanResolvedTransientAdjointReport, PcFanResolvedAdjointError> {
    validate_product_topology(steps)?;
    let generic_steps = steps
        .iter()
        .map(convert_step)
        .collect::<Result<Vec<_>, _>>()?;
    let report =
        solve_transient_adjoint(&convert_config(config), &generic_steps).map_err(map_error)?;
    let first = steps
        .first()
        .ok_or(PcFanResolvedAdjointError::DimensionMismatch(
            "transient adjoint requires at least one step",
        ))?;
    Ok(PcFanResolvedTransientAdjointReport {
        flow_model: first.flow_model,
        backend: first.backend,
        provenance: format!("transient-adjoint:{}", first.provenance),
        adjoints: report.adjoints,
        design_gradient: report.design_gradient,
        state_residual_norm: report.state_residual_norm,
        adjoint_residual_norm: report.adjoint_residual_norm,
        iterations: report.iterations,
        converged: report.converged,
    })
}

/// Solve the transient adjoint and contract its multipliers with BDF history
/// states entering at stage boundaries.
///
/// The existing [`solve_pc_fan_resolved_transient_adjoint`] API remains the
/// zero-initial-history convenience form. This checked form is for a fixed-
/// dimensional trajectory segment whose incoming BDF states are retained
/// separately from the represented trajectory. A caller at a remesh/restart
/// handoff must first express those states in the segment's coordinates. It
/// does not differentiate the remesher, topology choice, or mesh geometry.
pub fn solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history(
    config: &PcFanResolvedAdjointConfig,
    steps: &[PcFanResolvedTransientStep],
    initial_history_jacobians: &[PcFanResolvedTransientStageInitialHistoryJacobian],
) -> Result<PcFanResolvedTransientAdjointWithInitialHistory, PcFanResolvedAdjointError> {
    validate_product_topology(steps)?;
    for history in initial_history_jacobians {
        if history.objective_gradient.iter().any(|v| !v.is_finite()) {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "transient initial-history objective gradient",
            ));
        }
    }
    let generic_steps = steps
        .iter()
        .map(convert_step)
        .collect::<Result<Vec<_>, _>>()?;
    let generic_histories = initial_history_jacobians
        .iter()
        .map(|history| {
            Ok::<_, PcFanResolvedAdjointError>(TransientStageInitialHistoryJacobian {
                stage: history.stage,
                history_slot: history.history_slot,
                matrix: convert_matrix(&history.matrix)?,
                objective_gradient: history.objective_gradient.clone(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = solve_transient_adjoint_with_stage_initial_history(
        &convert_config(config),
        &generic_steps,
        &generic_histories,
    )
    .map_err(map_error)?;
    let first = steps
        .first()
        .ok_or(PcFanResolvedAdjointError::DimensionMismatch(
            "transient adjoint requires at least one step",
        ))?;
    Ok(PcFanResolvedTransientAdjointWithInitialHistory {
        report: PcFanResolvedTransientAdjointReport {
            flow_model: first.flow_model,
            backend: first.backend,
            provenance: format!("transient-adjoint:{}", first.provenance),
            adjoints: result.report.adjoints,
            design_gradient: result.report.design_gradient,
            state_residual_norm: result.report.state_residual_norm,
            adjoint_residual_norm: result.report.adjoint_residual_norm,
            iterations: result.report.iterations,
            converged: result.report.converged,
        },
        initial_history_adjoints: result
            .initial_history_adjoints
            .into_iter()
            .map(|adjoint| PcFanResolvedTransientInitialHistoryAdjoint {
                stage: adjoint.stage,
                history_slot: adjoint.history_slot,
                adjoint: adjoint.adjoint,
            })
            .collect(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(values: &[Vec<f64>]) -> PcFanResolvedFlowSparseMatrix {
        PcFanResolvedFlowSparseMatrix::from_dense(values).unwrap()
    }

    fn step(
        history_jacobians: Vec<PcFanResolvedTransientHistoryJacobian>,
        objective_state_gradient: Vec<f64>,
        residual_design_jacobian: Vec<Vec<f64>>,
        objective_design_gradient: Vec<f64>,
    ) -> PcFanResolvedTransientStep {
        PcFanResolvedTransientStep {
            flow_model: PcFanFlowModel::IncompressibleCfdFull,
            backend: PcFanCfdBackend::Cpu,
            provenance: "fixture:resolved-transient-newton:v1".into(),
            state_jacobian: matrix(&[vec![2.0, 0.0], vec![0.0, 3.0]]),
            state_residual: vec![0.0, 0.0],
            history_jacobians,
            residual_design_jacobian,
            objective_state_gradient,
            objective_design_gradient,
        }
    }

    #[test]
    fn transient_adjoint_propagates_bdf_history_and_design_gradient() {
        let first = step(
            vec![],
            vec![1.0, 2.0],
            vec![vec![1.0], vec![2.0]],
            vec![0.1],
        );
        let second = step(
            vec![PcFanResolvedTransientHistoryJacobian {
                previous_step: 0,
                matrix: matrix(&[vec![0.5, 0.0], vec![0.0, 1.0]]),
            }],
            vec![3.0, 4.0],
            vec![vec![2.0], vec![1.0]],
            vec![0.2],
        );

        let report = solve_pc_fan_resolved_transient_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &[first, second],
        )
        .unwrap();

        assert_eq!(report.adjoints.len(), 2);
        assert!((report.adjoints[1][0] - 1.5).abs() < 1.0e-10);
        assert!((report.adjoints[1][1] - 4.0 / 3.0).abs() < 1.0e-10);
        assert!((report.adjoints[0][0] - 0.125).abs() < 1.0e-10);
        assert!((report.adjoints[0][1] - 2.0 / 9.0).abs() < 1.0e-10);
        let expected_gradient =
            0.1 + 0.2 - (1.0 * 0.125 + 2.0 * (2.0 / 9.0)) - (2.0 * 1.5 + 1.0 * (4.0 / 3.0));
        assert!((report.design_gradient[0] - expected_gradient).abs() < 1.0e-10);
        assert!(report.converged);
        assert!(report.adjoint_residual_norm < 1.0e-10);
    }

    #[test]
    fn transient_adjoint_rejects_noncausal_history_and_reduced_flow() {
        let mut noncausal = step(
            vec![PcFanResolvedTransientHistoryJacobian {
                previous_step: 1,
                matrix: matrix(&[vec![1.0, 0.0], vec![0.0, 1.0]]),
            }],
            vec![1.0, 0.0],
            vec![vec![0.0], vec![0.0]],
            vec![0.0],
        );
        assert!(matches!(
            solve_pc_fan_resolved_transient_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &[noncausal.clone()],
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(_))
        ));

        noncausal.history_jacobians.clear();
        noncausal.flow_model = PcFanFlowModel::ReducedOrder;
        assert!(matches!(
            solve_pc_fan_resolved_transient_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &[noncausal],
            ),
            Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
                PcFanFlowModel::ReducedOrder
            ))
        ));
    }
}

#[cfg(test)]
mod stage_initial_history_tests {
    use super::*;

    fn matrix(values: &[Vec<f64>]) -> PcFanResolvedFlowSparseMatrix {
        PcFanResolvedFlowSparseMatrix::from_dense(values).unwrap()
    }

    fn step(
        state_jacobian: PcFanResolvedFlowSparseMatrix,
        history_jacobians: Vec<PcFanResolvedTransientHistoryJacobian>,
        objective_state_gradient: Vec<f64>,
    ) -> PcFanResolvedTransientStep {
        let dimension = state_jacobian.dimension;
        PcFanResolvedTransientStep {
            flow_model: PcFanFlowModel::IncompressibleCfdFull,
            backend: PcFanCfdBackend::Cpu,
            provenance: "fixture:resolved-transient-stage-initial-history:v1".into(),
            state_jacobian,
            state_residual: vec![0.0; dimension],
            history_jacobians,
            residual_design_jacobian: vec![vec![0.0]; dimension],
            objective_state_gradient,
            objective_design_gradient: vec![0.0],
        }
    }

    #[test]
    fn contracts_single_stage_initial_history_with_direct_objective_term() {
        let steps = [step(
            matrix(&[vec![2.0, 0.0], vec![0.0, 4.0]]),
            Vec::new(),
            vec![2.0, 8.0],
        )];

        let result = solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history(
            &PcFanResolvedAdjointConfig::default(),
            &steps,
            &[PcFanResolvedTransientStageInitialHistoryJacobian {
                stage: 0,
                history_slot: 1,
                matrix: matrix(&[vec![1.0, 2.0], vec![3.0, 4.0]]),
                objective_gradient: vec![0.5, 1.5],
            }],
        )
        .unwrap();

        assert_eq!(result.report.adjoints[0], vec![1.0, 2.0]);
        assert_eq!(result.initial_history_adjoints.len(), 1);
        assert_eq!(result.initial_history_adjoints[0].stage, 0);
        assert_eq!(result.initial_history_adjoints[0].history_slot, 1);
        assert!((result.initial_history_adjoints[0].adjoint[0] + 6.5).abs() < 1.0e-12);
        assert!((result.initial_history_adjoints[0].adjoint[1] + 8.5).abs() < 1.0e-12);
    }

    #[test]
    fn composes_initial_history_after_trajectory_history_recurrence() {
        let steps = [
            step(
                matrix(&[vec![2.0, 0.0], vec![0.0, 3.0]]),
                Vec::new(),
                vec![2.0, 3.0],
            ),
            step(
                matrix(&[vec![4.0, 0.0], vec![0.0, 5.0]]),
                vec![PcFanResolvedTransientHistoryJacobian {
                    previous_step: 0,
                    matrix: matrix(&[vec![0.5, 0.0], vec![0.0, 0.25]]),
                }],
                vec![4.0, 10.0],
            ),
        ];

        let result = solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history(
            &PcFanResolvedAdjointConfig::default(),
            &steps,
            &[
                PcFanResolvedTransientStageInitialHistoryJacobian {
                    stage: 0,
                    history_slot: 0,
                    matrix: matrix(&[vec![1.0, 0.0], vec![0.0, 1.0]]),
                    objective_gradient: vec![0.0, 0.0],
                },
                PcFanResolvedTransientStageInitialHistoryJacobian {
                    stage: 1,
                    history_slot: 1,
                    matrix: matrix(&[vec![1.0, 0.0], vec![0.0, 2.0]]),
                    objective_gradient: vec![0.0, 0.0],
                },
            ],
        )
        .unwrap();

        assert!((result.report.adjoints[0][0] - 0.75).abs() < 1.0e-12);
        assert!((result.report.adjoints[0][1] - (5.0 / 6.0)).abs() < 1.0e-12);
        assert!((result.report.adjoints[1][0] - 1.0).abs() < 1.0e-12);
        assert!((result.report.adjoints[1][1] - 2.0).abs() < 1.0e-12);
        assert!((result.initial_history_adjoints[0].adjoint[0] + 0.75).abs() < 1.0e-12);
        assert!((result.initial_history_adjoints[0].adjoint[1] + (5.0 / 6.0)).abs() < 1.0e-12);
        assert!((result.initial_history_adjoints[1].adjoint[0] + 1.0).abs() < 1.0e-12);
        assert!((result.initial_history_adjoints[1].adjoint[1] + 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_invalid_initial_history_stage_or_dimension() {
        let steps = [step(
            matrix(&[vec![1.0, 0.0], vec![0.0, 1.0]]),
            Vec::new(),
            vec![1.0, 1.0],
        )];

        let out_of_range = [PcFanResolvedTransientStageInitialHistoryJacobian {
            stage: 1,
            history_slot: 0,
            matrix: matrix(&[vec![1.0, 0.0], vec![0.0, 1.0]]),
            objective_gradient: vec![0.0, 0.0],
        }];
        assert!(matches!(
            solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history(
                &PcFanResolvedAdjointConfig::default(),
                &steps,
                &out_of_range,
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient initial-history stage index"
            ))
        ));

        let wrong_dimension = [PcFanResolvedTransientStageInitialHistoryJacobian {
            stage: 0,
            history_slot: 0,
            matrix: matrix(&[vec![1.0]]),
            objective_gradient: vec![0.0],
        }];
        assert!(matches!(
            solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history(
                &PcFanResolvedAdjointConfig::default(),
                &steps,
                &wrong_dimension,
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient initial-history dimensions"
            ))
        ));

        let wrong_objective_dimension = [PcFanResolvedTransientStageInitialHistoryJacobian {
            stage: 0,
            history_slot: 0,
            matrix: matrix(&[vec![1.0, 0.0], vec![0.0, 1.0]]),
            objective_gradient: vec![0.0],
        }];
        assert!(matches!(
            solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history(
                &PcFanResolvedAdjointConfig::default(),
                &steps,
                &wrong_objective_dimension,
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient initial-history objective gradient"
            ))
        ));

        let nonfinite_objective = [PcFanResolvedTransientStageInitialHistoryJacobian {
            stage: 0,
            history_slot: 0,
            matrix: matrix(&[vec![1.0, 0.0], vec![0.0, 1.0]]),
            objective_gradient: vec![f64::NAN, 0.0],
        }];
        assert!(matches!(
            solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history(
                &PcFanResolvedAdjointConfig::default(),
                &steps,
                &nonfinite_objective,
            ),
            Err(PcFanResolvedAdjointError::NonFiniteInput(
                "transient initial-history objective gradient"
            ))
        ));
    }
}
