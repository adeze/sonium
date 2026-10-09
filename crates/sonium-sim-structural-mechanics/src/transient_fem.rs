//! Bounded assembled linear-elastic FEM transient reference.
//!
//! This module promotes the bracket workflow's transient step from the
//! reduced SDOF family to a small, checked 2-D FEM path.  It deliberately
//! uses the shared bracket stiffness/mass assembly and the solver crate's
//! Newmark average-acceleration integrator.  It is a CPU/f64 reference for
//! prescribed load histories, not a general nonlinear or large-scale
//! transient structural solver.

use super::{
    BracketStudy, StructuralExecutionProvenance, StructuralMechanicsError,
    StructuralModelAssumption, StructuralPolynomialOrder,
};
use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::physics::solid::elasticity::ElasticityForm;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::traits::LinearOperator;
use sonium_solvers::{
    CsrMatrix, MatrixClass, NewmarkConfig, NewmarkInitial, Ordering, SparseDirectConfig,
    SparseDirectMethod, factorize_sparse, time_integration::try_newmark_integrate,
};

/// Schema version for the assembled FEM transient result.
pub const TRANSIENT_FEM_SCHEMA_VERSION: u32 = 1;
const MAX_TRANSIENT_FEM_STEPS: usize = 10_000;
const MAX_LOAD_SCALE: f64 = 1.0e6;

/// Checked load-history and time-integration inputs for the bounded FEM path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientFemStudy {
    pub bracket: BracketStudy,
    pub polynomial_order: StructuralPolynomialOrder,
    pub time_step_s: f64,
    /// One scale for the bracket's assembled tip-load vector at each time
    /// sample.  The first sample is the unloaded initial state.
    pub load_scales: Vec<f64>,
    pub equilibrium_tolerance_n: f64,
}

impl Default for TransientFemStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl TransientFemStudy {
    pub fn introductory() -> Self {
        Self {
            bracket: BracketStudy::introductory_bracket(),
            polynomial_order: StructuralPolynomialOrder::Linear,
            time_step_s: 1.0e-5,
            load_scales: vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0],
            equilibrium_tolerance_n: 1.0e-6,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if !self.time_step_s.is_finite() || self.time_step_s <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_fem.time_step_s",
                value: self.time_step_s.to_string(),
            });
        }
        if self.load_scales.len() < 2 || self.load_scales.len() - 1 > MAX_TRANSIENT_FEM_STEPS {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_fem.load_scales",
                value: self.load_scales.len().to_string(),
            });
        }
        if self
            .load_scales
            .iter()
            .any(|scale| !scale.is_finite() || scale.abs() > MAX_LOAD_SCALE)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_fem.load_scales",
                value: "non-finite or outside bounded scale".to_string(),
            });
        }
        if self.load_scales[0].abs() > f64::EPSILON {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_fem.load_scales[0]",
                value: "initial load must be zero".to_string(),
            });
        }
        if !self.equilibrium_tolerance_n.is_finite() || self.equilibrium_tolerance_n <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_fem.equilibrium_tolerance_n",
                value: self.equilibrium_tolerance_n.to_string(),
            });
        }
        Ok(())
    }
}

/// Per-sample diagnostics from the assembled FEM Newmark trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientFemStepResult {
    pub step: usize,
    pub time_s: f64,
    pub load_scale: f64,
    pub max_displacement_m: f64,
    pub max_velocity_m_per_s: f64,
    pub max_acceleration_m_per_s2: f64,
    pub dynamic_equilibrium_residual_n: f64,
    pub total_energy_j: f64,
    pub damping_power_w: f64,
}

/// Result of the bounded assembled 2-D FEM transient reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientFemResult {
    pub schema_version: u32,
    pub study: TransientFemStudy,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub free_dofs: usize,
    pub polynomial_order: StructuralPolynomialOrder,
    pub first_bending_frequency_hz: f64,
    /// Mass-proportional damping coefficient in `C = coefficient * M`.
    pub damping_coefficient_s: f64,
    pub steps: Vec<TransientFemStepResult>,
    pub final_displacement: Vec<[f64; 2]>,
    pub final_velocity: Vec<[f64; 2]>,
    pub final_acceleration: Vec<[f64; 2]>,
    pub max_dynamic_equilibrium_residual_n: f64,
    pub max_total_energy_j: f64,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run the checked assembled 2-D FEM Newmark reference.
pub fn run_transient_fem(
    study: &TransientFemStudy,
) -> Result<TransientFemResult, StructuralMechanicsError> {
    study.validate()?;

    let static_result = super::run_static_bracket_order(&study.bracket, study.polynomial_order)?;
    let modal = super::solve_first_bending_mode_order(
        &study.bracket,
        &static_result,
        study.polynomial_order,
    )?;
    let mesh = study.bracket.mesh();
    super::validate_mesh(&mesh, &study.bracket)?;
    let degree = match study.polynomial_order {
        StructuralPolynomialOrder::Linear => PolynomialDegree::P1,
        StructuralPolynomialOrder::Quadratic => PolynomialDegree::P2,
    };
    let space = VectorH1Space::new(degree, 2);
    let material = study.bracket.material.as_fem();
    let form = ElasticityForm::new_2d(material, study.bracket.plane_model.as_fem());
    let matrices =
        super::assemble_structural_matrices(&study.bracket, &mesh, &space, &form, degree);
    let fixed_dofs = space.boundary_dofs(&mesh, study.bracket.loads.fixed_boundary_marker);
    let (elastic_stiffness, free_to_full) =
        super::reduce_triplets(&matrices.stiffness, &fixed_dofs);
    let (geometric_stiffness, _) = super::reduce_triplets(&matrices.geometric, &fixed_dofs);
    let (mass, _) = super::reduce_triplets(&matrices.mass, &fixed_dofs);
    if elastic_stiffness.num_rows == 0 || mass.num_rows == 0 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "transient FEM has no free displacement DOFs".to_string(),
        });
    }
    let stiffness = combine_csr(&[(&elastic_stiffness, 1.0), (&geometric_stiffness, 1.0)]);

    let full_dofs = space.num_global_dofs(&mesh);
    let mut full_load = vec![0.0; full_dofs];
    super::apply_tip_edge_load_order(&mesh, &study.bracket, &space, degree, &mut full_load);
    let reduced_load: Vec<f64> = free_to_full.iter().map(|&dof| full_load[dof]).collect();

    let damping_coefficient_s = 2.0
        * study.bracket.material.damping_ratio
        * modal.first_bending_frequency_hz
        * std::f64::consts::TAU;
    if !damping_coefficient_s.is_finite() || damping_coefficient_s < 0.0 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "transient_fem.damping_coefficient",
            value: damping_coefficient_s,
        });
    }
    let damping = scale_csr(&mass, damping_coefficient_s);
    let config =
        NewmarkConfig::average_acceleration(study.time_step_s, study.load_scales.len() - 1)
            .with_history();
    let effective = combine_csr(&[
        (&mass, 1.0),
        (&damping, config.gamma * study.time_step_s),
        (
            &stiffness,
            config.beta * study.time_step_s * study.time_step_s,
        ),
    ]);
    let factorization = factorize_sparse(
        &effective,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..SparseDirectConfig::default()
        },
        None,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "transient FEM effective Newmark factorization",
        message: error.to_string(),
    })?;

    let zeros = vec![0.0; stiffness.num_rows];
    let initial = NewmarkInitial {
        displacement: zeros.clone(),
        velocity: zeros.clone(),
        acceleration: zeros,
    };
    let load_scales = study.load_scales.clone();
    let force_load = reduced_load.clone();
    let dt = study.time_step_s;
    let force_fn = move |time_s: f64| {
        let step = ((time_s / dt).round() as usize).min(load_scales.len() - 1);
        force_load
            .iter()
            .map(|&value| value * load_scales[step])
            .collect::<Vec<_>>()
    };
    let integration = try_newmark_integrate(
        Some(&damping),
        &stiffness,
        force_fn,
        &initial,
        |rhs| {
            factorization
                .solve(rhs)
                .unwrap_or_else(|_| vec![f64::NAN; rhs.len()])
        },
        &config,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "transient FEM Newmark integration",
        message: error.to_string(),
    })?;
    if integration.solver_status() != sonium_solvers::traits::SolverStatus::Converged {
        return Err(StructuralMechanicsError::NonConvergence {
            iterations: integration.steps_completed,
            residual: f64::INFINITY,
        });
    }
    let displacement_history =
        integration
            .displacement_history
            .ok_or(StructuralMechanicsError::SolverFailure {
                stage: "transient FEM Newmark history",
                message: "Newmark history was not returned".to_string(),
            })?;
    let velocity_history =
        integration
            .velocity_history
            .ok_or(StructuralMechanicsError::SolverFailure {
                stage: "transient FEM Newmark history",
                message: "Newmark velocity history was not returned".to_string(),
            })?;
    let acceleration_history =
        integration
            .acceleration_history
            .ok_or(StructuralMechanicsError::SolverFailure {
                stage: "transient FEM Newmark history",
                message: "Newmark acceleration history was not returned".to_string(),
            })?;

    let mut steps = Vec::with_capacity(displacement_history.len());
    let mut max_dynamic_equilibrium_residual_n: f64 = 0.0;
    let mut max_total_energy_j: f64 = 0.0;
    for (step, ((displacement, velocity), acceleration)) in displacement_history
        .iter()
        .zip(&velocity_history)
        .zip(&acceleration_history)
        .enumerate()
    {
        let load_scale = study.load_scales[step];
        let force = reduced_load
            .iter()
            .map(|&value| value * load_scale)
            .collect::<Vec<_>>();
        let mut residual = mass.apply(acceleration);
        add_scaled(&mut residual, &damping.apply(velocity), 1.0);
        add_scaled(&mut residual, &stiffness.apply(displacement), 1.0);
        add_scaled(&mut residual, &force, -1.0);
        let dynamic_equilibrium_residual_n = l2_norm(&residual);
        let kinetic_energy = 0.5 * dot(velocity, &mass.apply(velocity));
        let strain_energy = 0.5 * dot(displacement, &stiffness.apply(displacement));
        let total_energy_j = kinetic_energy + strain_energy;
        let damping_power_w = dot(velocity, &damping.apply(velocity));
        let full_displacement = expand_reduced(displacement, &free_to_full, full_dofs);
        let full_velocity = expand_reduced(velocity, &free_to_full, full_dofs);
        let full_acceleration = expand_reduced(acceleration, &free_to_full, full_dofs);
        let max_displacement_m = max_nodal_norm(&full_displacement);
        let max_velocity_m_per_s = max_nodal_norm(&full_velocity);
        let max_acceleration_m_per_s2 = max_nodal_norm(&full_acceleration);
        if !dynamic_equilibrium_residual_n.is_finite()
            || !total_energy_j.is_finite()
            || !damping_power_w.is_finite()
        {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "transient_fem.finite_step_diagnostics",
                value: step as f64,
            });
        }
        max_dynamic_equilibrium_residual_n =
            max_dynamic_equilibrium_residual_n.max(dynamic_equilibrium_residual_n);
        max_total_energy_j = max_total_energy_j.max(total_energy_j);
        steps.push(TransientFemStepResult {
            step,
            time_s: step as f64 * study.time_step_s,
            load_scale,
            max_displacement_m,
            max_velocity_m_per_s,
            max_acceleration_m_per_s2,
            dynamic_equilibrium_residual_n,
            total_energy_j,
            damping_power_w,
        });
    }

    let final_displacement = nodal_vectors(&expand_reduced(
        displacement_history
            .last()
            .ok_or(StructuralMechanicsError::SolverFailure {
                stage: "transient FEM Newmark history",
                message: "Newmark displacement history is empty".to_string(),
            })?,
        &free_to_full,
        full_dofs,
    ));
    let final_velocity = nodal_vectors(&expand_reduced(
        velocity_history
            .last()
            .ok_or(StructuralMechanicsError::SolverFailure {
                stage: "transient FEM Newmark history",
                message: "Newmark velocity history is empty".to_string(),
            })?,
        &free_to_full,
        full_dofs,
    ));
    let final_acceleration = nodal_vectors(&expand_reduced(
        acceleration_history
            .last()
            .ok_or(StructuralMechanicsError::SolverFailure {
                stage: "transient FEM Newmark history",
                message: "Newmark acceleration history is empty".to_string(),
            })?,
        &free_to_full,
        full_dofs,
    ));
    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id = "sonium-structural.cpu-reference.f64.transient-fem-2d.v1".to_string();
    execution.limitations.extend([
        "Prescribed-load linear 2D bracket FEM with mass-proportional damping and Newmark average acceleration.".to_string(),
        "Nonlinear, contact, transient topology, distributed/device, and external-solver parity are not claimed.".to_string(),
    ]);
    Ok(TransientFemResult {
        schema_version: TRANSIENT_FEM_SCHEMA_VERSION,
        study: study.clone(),
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs: full_dofs,
        free_dofs: free_to_full.len(),
        polynomial_order: study.polynomial_order,
        first_bending_frequency_hz: modal.first_bending_frequency_hz,
        damping_coefficient_s,
        steps,
        final_displacement,
        final_velocity,
        final_acceleration,
        max_dynamic_equilibrium_residual_n,
        max_total_energy_j,
        converged: max_dynamic_equilibrium_residual_n <= study.equilibrium_tolerance_n,
        model_assumption: StructuralModelAssumption::TransientFem2D,
        execution,
    })
}

fn scale_csr(matrix: &CsrMatrix<f64>, scale: f64) -> CsrMatrix<f64> {
    combine_csr(&[(matrix, scale)])
}

fn combine_csr(terms: &[(&CsrMatrix<f64>, f64)]) -> CsrMatrix<f64> {
    let mut triplets = Vec::new();
    for (matrix, scale) in terms {
        for row in 0..matrix.num_rows {
            for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                triplets.push((
                    row,
                    matrix.col_indices[index],
                    matrix.values[index] * *scale,
                ));
            }
        }
    }
    CsrMatrix::from_triplets(terms[0].0.num_rows, terms[0].0.num_cols, triplets)
}

fn add_scaled(target: &mut [f64], source: &[f64], scale: f64) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += scale * source;
    }
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn l2_norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn expand_reduced(reduced: &[f64], free_to_full: &[usize], full_dofs: usize) -> Vec<f64> {
    let mut full = vec![0.0; full_dofs];
    for (&value, &full_dof) in reduced.iter().zip(free_to_full) {
        full[full_dof] = value;
    }
    full
}

fn max_nodal_norm(values: &[f64]) -> f64 {
    values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|value| (value[0] * value[0] + value[1] * value[1]).sqrt())
        .fold(0.0, f64::max)
}

fn nodal_vectors(values: &[f64]) -> Vec<[f64; 2]> {
    values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|value| [value[0], value[1]])
        .collect()
}
