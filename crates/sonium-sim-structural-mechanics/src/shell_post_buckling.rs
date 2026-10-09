//! Bounded displacement-controlled corotational shell continuation.
//!
//! This reference uses the production triangular shell element and its
//! nonlinear element assembly. The control equation augments all free-DOF
//! equilibrium equations, so the load factor can be followed through a
//! limit point without a scalar constitutive surrogate. It is intentionally
//! bounded to small P1 shell meshes and dense continuation correctors.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_nonlinear_element_form;
use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::mesh::{ElementType, rectangular_mesh_triangles};
use sonium_fem::physics::solid::shell::{
    CorotationalShellForm, ShellElement, ShellSection, add_shell_point_load,
};

/// Schema version for [`CorotationalShellPostBucklingResult`].
pub const COROTATIONAL_SHELL_POST_BUCKLING_SCHEMA_VERSION: u32 = 1;
const MAX_SHELL_CONTINUATION_STEPS: usize = 128;
const DOFS_PER_NODE: usize = 6;

/// Bounded P1 triangular shell continuation inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorotationalShellPostBucklingStudy {
    pub length_m: f64,
    pub width_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub thickness_m: f64,
    pub density_kg_m3: f64,
    /// Small out-of-plane geometric imperfection at the free end.
    pub imperfection_m: f64,
    /// Magnitude used to scale the prescribed external load direction.
    pub reference_load_n: f64,
    /// Final prescribed transverse displacement at the control node.
    pub control_displacement_m: f64,
    pub continuation_steps: usize,
    pub max_newton_iterations: usize,
    /// Dimensionless force/displacement convergence tolerance.
    pub newton_residual_tolerance: f64,
}

impl Default for CorotationalShellPostBucklingStudy {
    fn default() -> Self {
        Self {
            length_m: 1.0,
            width_m: 0.5,
            nx: 2,
            ny: 1,
            youngs_modulus_pa: 2.0e4,
            poisson_ratio: 0.3,
            thickness_m: 0.04,
            density_kg_m3: 1.0,
            imperfection_m: 0.01,
            reference_load_n: 1.0,
            control_displacement_m: -0.08,
            continuation_steps: 24,
            // The production shell tangent is a finite-difference
            // corotational tangent; the bounded reference fixture needs
            // extra corrector iterations to reach the strict residual gate.
            max_newton_iterations: 128,
            newton_residual_tolerance: 1.0e-8,
        }
    }
}

impl CorotationalShellPostBucklingStudy {
    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("shell_post_buckling.length_m", self.length_m),
            ("shell_post_buckling.width_m", self.width_m),
            (
                "shell_post_buckling.youngs_modulus_pa",
                self.youngs_modulus_pa,
            ),
            ("shell_post_buckling.thickness_m", self.thickness_m),
            ("shell_post_buckling.density_kg_m3", self.density_kg_m3),
            (
                "shell_post_buckling.reference_load_n",
                self.reference_load_n,
            ),
            (
                "shell_post_buckling.newton_residual_tolerance",
                self.newton_residual_tolerance,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !(-1.0 < self.poisson_ratio && self.poisson_ratio < 0.5) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell_post_buckling.poisson_ratio",
                value: self.poisson_ratio.to_string(),
            });
        }
        if !self.imperfection_m.is_finite()
            || !self.control_displacement_m.is_finite()
            || self.control_displacement_m.abs() <= f64::EPSILON
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell_post_buckling.displacement_controls",
                value: format!(
                    "imperfection_m={}, control_displacement_m={}",
                    self.imperfection_m, self.control_displacement_m
                ),
            });
        }
        if self.nx == 0 || self.ny == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell_post_buckling.mesh_resolution",
                value: format!("nx={}, ny={}", self.nx, self.ny),
            });
        }
        if self.continuation_steps == 0 || self.continuation_steps > MAX_SHELL_CONTINUATION_STEPS {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell_post_buckling.continuation_steps",
                value: self.continuation_steps.to_string(),
            });
        }
        if self.max_newton_iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell_post_buckling.max_newton_iterations",
                value: self.max_newton_iterations.to_string(),
            });
        }
        Ok(())
    }
}

/// One converged displacement-controlled continuation point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorotationalShellPostBucklingStep {
    pub step: usize,
    pub target_displacement_m: f64,
    pub load_factor: f64,
    pub applied_load_n: f64,
    /// Conjugate internal force at the displacement-control DOF.
    pub control_reaction_n: f64,
    pub newton_iterations: usize,
    pub equilibrium_residual_n: f64,
    pub control_residual_m: f64,
    pub max_translation_m: f64,
    pub converged: bool,
}

/// Versioned bounded corotational shell continuation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorotationalShellPostBucklingResult {
    pub schema_version: u32,
    pub study: CorotationalShellPostBucklingStudy,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub control_node: usize,
    pub control_dof: usize,
    pub fixed_nodes: usize,
    pub steps: Vec<CorotationalShellPostBucklingStep>,
    /// First continuation step at which the load-factor direction reverses.
    pub turning_point_step: Option<usize>,
    pub peak_load_factor: f64,
    pub final_load_factor: f64,
    pub final_control_reaction_n: f64,
    pub max_equilibrium_residual_n: f64,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run a bounded production-shell displacement-controlled continuation.
pub fn run_corotational_shell_post_buckling(
    study: &CorotationalShellPostBucklingStudy,
) -> Result<CorotationalShellPostBucklingResult, StructuralMechanicsError> {
    study.validate()?;

    let mut mesh =
        rectangular_mesh_triangles(0.0, study.length_m, 0.0, study.width_m, study.nx, study.ny);
    let x_tolerance = 1.0e-10 * study.length_m.max(study.width_m).max(1.0);
    for point in &mut mesh.nodes {
        let x_fraction = point.x / study.length_m;
        point.z = study.imperfection_m * x_fraction * x_fraction;
    }
    mesh.validate_connectivity()
        .map_err(|errors| StructuralMechanicsError::InvalidMesh {
            reason: format!("shell continuation mesh connectivity: {errors:?}"),
        })?;

    let section = ShellSection::new(
        study.youngs_modulus_pa,
        study.poisson_ratio,
        study.thickness_m,
        study.density_kg_m3,
    );
    let mut elements = Vec::with_capacity(mesh.elements.len());
    for (index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != ElementType::Triangle || element.nodes.len() != 3 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!("shell continuation element {index} is not a P1 triangle"),
            });
        }
        if element.nodes.iter().any(|&node| node >= mesh.nodes.len()) {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!("shell continuation element {index} references a missing node"),
            });
        }
        elements.push(ShellElement::from_mesh(
            [element.nodes[0], element.nodes[1], element.nodes[2]],
            &mesh,
            section,
        ));
    }
    if elements.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "shell continuation mesh contains no elements".to_string(),
        });
    }

    let n_dofs = mesh.nodes.len() * DOFS_PER_NODE;
    let fixed_nodes = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(node, point)| (point.x.abs() <= x_tolerance).then_some(node))
        .collect::<Vec<_>>();
    if fixed_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "shell continuation fixed edge selected no nodes".to_string(),
        });
    }
    let control_node = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, point)| (point.x - study.length_m).abs() <= x_tolerance)
        .max_by(|(_, left), (_, right)| left.y.total_cmp(&right.y))
        .map(|(node, _)| node)
        .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
            reason: "shell continuation selected no control node".to_string(),
        })?;
    let control_dof = control_node * DOFS_PER_NODE + 2;

    let mut external_load = vec![0.0; n_dofs];
    add_shell_point_load(
        &mut external_load,
        control_node,
        [
            0.20 * study.reference_load_n,
            0.05 * study.reference_load_n,
            -study.reference_load_n,
            0.0,
            0.0,
            0.0,
        ],
    );
    let fixed_dof = fixed_nodes
        .iter()
        .flat_map(|&node| (0..DOFS_PER_NODE).map(move |component| node * DOFS_PER_NODE + component))
        .collect::<Vec<_>>();
    let fixed_set = fixed_dof
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let free_dofs = (0..n_dofs)
        .filter(|dof| !fixed_set.contains(dof))
        .collect::<Vec<_>>();
    if !free_dofs.contains(&control_dof) || free_dofs.len() < 2 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "shell continuation requires at least two free DOFs including control"
                .to_string(),
        });
    }
    // Keep the equilibrium equation at the controlled DOF as well.  The
    // unknown load factor needs all free-DOF equilibrium equations plus the
    // displacement constraint to form a square augmented corrector system.
    let equilibrium_dofs = free_dofs.clone();

    let form = CorotationalShellForm::new(elements, n_dofs);
    let mut solution = vec![0.0; n_dofs];
    let mut load_factor = 0.0;
    let mut steps = Vec::with_capacity(study.continuation_steps);

    for step in 1..=study.continuation_steps {
        let target = study.control_displacement_m * step as f64 / study.continuation_steps as f64;
        let mut iterations = 0;
        let mut equilibrium_residual = f64::INFINITY;
        let mut control_residual = f64::INFINITY;
        let mut control_reaction = f64::NAN;
        let mut converged = false;

        for iteration in 1..=study.max_newton_iterations {
            iterations = iteration;
            let (internal, tangent_triplets) = assemble_nonlinear_element_form(&form, &solution);
            let mut tangent = vec![0.0; n_dofs * n_dofs];
            for ((&row, &column), &value) in tangent_triplets
                .rows
                .iter()
                .zip(&tangent_triplets.cols)
                .zip(&tangent_triplets.values)
            {
                tangent[row * n_dofs + column] += value;
            }

            equilibrium_residual = equilibrium_dofs
                .iter()
                .map(|&dof| (internal[dof] - load_factor * external_load[dof]).abs())
                .fold(0.0, f64::max);
            control_residual = (solution[control_dof] - target).abs();
            let force_scale = study.reference_load_n.max(1.0);
            let displacement_scale = target.abs().max(1.0);
            if equilibrium_residual <= study.newton_residual_tolerance * force_scale
                && control_residual <= study.newton_residual_tolerance * displacement_scale
            {
                control_reaction = internal[control_dof];
                converged = true;
                break;
            }

            let n_free = free_dofs.len();
            let n_unknowns = n_free + 1;
            let mut jacobian = vec![0.0; n_unknowns * n_unknowns];
            let mut rhs = vec![0.0; n_unknowns];
            for (row, &equilibrium_dof) in equilibrium_dofs.iter().enumerate() {
                for (column, &dof) in free_dofs.iter().enumerate() {
                    jacobian[row * n_unknowns + column] = tangent[equilibrium_dof * n_dofs + dof];
                }
                jacobian[row * n_unknowns + n_free] = -external_load[equilibrium_dof];
                rhs[row] =
                    -(internal[equilibrium_dof] - load_factor * external_load[equilibrium_dof]);
            }
            let control_column = free_dofs
                .iter()
                .position(|&dof| dof == control_dof)
                .expect("control DOF is part of free DOFs");
            jacobian[(n_unknowns - 1) * n_unknowns + control_column] = 1.0;
            rhs[n_unknowns - 1] = target - solution[control_dof];

            let correction = solve_dense_system(jacobian, rhs).map_err(|message| {
                StructuralMechanicsError::SolverFailure {
                    stage: "corotational shell continuation corrector",
                    message: format!("step {step}, Newton iteration {iterations}: {message}"),
                }
            })?;
            for (column, &dof) in free_dofs.iter().enumerate() {
                solution[dof] += correction[column];
            }
            load_factor += correction[n_free];
            if solution.iter().any(|value| !value.is_finite()) || !load_factor.is_finite() {
                return Err(StructuralMechanicsError::ValidationFailure {
                    check: "shell_post_buckling.newton_state_finite",
                    value: f64::NAN,
                });
            }
        }

        if !converged {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::CorotationalShell,
                load_step: step,
                iterations,
                residual: equilibrium_residual.max(control_residual),
            });
        }
        let maximum_translation = solution
            .as_chunks::<DOFS_PER_NODE>()
            .0
            .iter()
            .map(|dofs| (dofs[0] * dofs[0] + dofs[1] * dofs[1] + dofs[2] * dofs[2]).sqrt())
            .fold(0.0, f64::max);
        steps.push(CorotationalShellPostBucklingStep {
            step,
            target_displacement_m: target,
            load_factor,
            applied_load_n: load_factor * study.reference_load_n,
            control_reaction_n: control_reaction,
            newton_iterations: iterations,
            equilibrium_residual_n: equilibrium_residual,
            control_residual_m: control_residual,
            max_translation_m: maximum_translation,
            converged: true,
        });
    }

    let turning_point_step = steps
        .windows(2)
        .scan(0_i8, |direction, pair| {
            let delta = pair[1].load_factor - pair[0].load_factor;
            let current = if delta.abs() <= 1.0e-12 {
                0
            } else if delta.is_sign_positive() {
                1
            } else {
                -1
            };
            let previous = *direction;
            if current != 0 {
                *direction = current;
            }
            Some((previous, current))
        })
        .enumerate()
        .find_map(|(index, (previous, current))| {
            (previous != 0 && current != 0 && previous != current).then_some(index + 2)
        });
    let peak_load_factor = steps
        .iter()
        .map(|step| step.load_factor)
        .fold(f64::NEG_INFINITY, f64::max);
    let final_step = steps.last().expect("continuation always has one step");
    let final_load_factor = final_step.load_factor;
    let final_control_reaction_n = final_step.control_reaction_n;
    let max_equilibrium_residual_n = final_step.equilibrium_residual_n;

    Ok(CorotationalShellPostBucklingResult {
        schema_version: COROTATIONAL_SHELL_POST_BUCKLING_SCHEMA_VERSION,
        study: study.clone(),
        nodes: mesh.nodes.len(),
        elements: form.num_elements(),
        dofs: n_dofs,
        control_node,
        control_dof,
        fixed_nodes: fixed_nodes.len(),
        steps,
        turning_point_step,
        peak_load_factor,
        final_load_factor,
        final_control_reaction_n,
        max_equilibrium_residual_n,
        converged: true,
        model_assumption: StructuralModelAssumption::CorotationalShell,
        execution: StructuralExecutionProvenance::cpu_shell_reference(),
    })
}

fn solve_dense_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>) -> Result<Vec<f64>, String> {
    let dimension = rhs.len();
    if matrix.len() != dimension * dimension {
        return Err(format!(
            "augmented matrix has {} entries for {dimension} RHS entries",
            matrix.len()
        ));
    }
    for pivot in 0..dimension {
        let (pivot_row, pivot_value) = (pivot..dimension)
            .map(|row| (row, matrix[row * dimension + pivot].abs()))
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or_else(|| "empty augmented matrix".to_string())?;
        if !pivot_value.is_finite() || pivot_value <= 1.0e-14 {
            return Err(format!("singular augmented tangent at pivot {pivot}"));
        }
        if pivot_row != pivot {
            for column in pivot..dimension {
                matrix.swap(pivot * dimension + column, pivot_row * dimension + column);
            }
            rhs.swap(pivot, pivot_row);
        }
        let diagonal = matrix[pivot * dimension + pivot];
        for row in pivot + 1..dimension {
            let factor = matrix[row * dimension + pivot] / diagonal;
            matrix[row * dimension + pivot] = 0.0;
            for column in pivot + 1..dimension {
                matrix[row * dimension + column] -= factor * matrix[pivot * dimension + column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = rhs[row];
        for column in row + 1..dimension {
            value -= matrix[row * dimension + column] * solution[column];
        }
        solution[row] = value / matrix[row * dimension + row];
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("augmented tangent produced a non-finite correction".to_string());
    }
    Ok(solution)
}
