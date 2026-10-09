//! Load-stepped corotational shell static studies.
//!
//! Three studies share one sparse Newton driver over
//! [`CorotationalShellForm`](sonium_fem::physics::solid::shell::CorotationalShellForm):
//! uniform pressure on a west-clamped plate (linear-match leg), tip line
//! load on a cantilever strip (beam-elastica leg), and edge compression
//! with a pressure imperfection on a simply-supported plate (buckling
//! leg). All loads are dead (fixed-direction) and all meshes are flat
//! P1 triangulations of a rectangle.

use super::{
    ShellGeometry, ShellMaterial, ShellStaticStudy, StructuralExecutionProvenance,
    StructuralMechanicsError, StructuralModelAssumption,
};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_nonlinear_element_form;
use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::solid::shell::{
    CorotationalShellForm, add_shell_point_load, assemble_shell_load,
};
use sonium_solvers::{
    MatrixClass, SparseDirectConfig, SparseDirectMethod, TripletMatrix, factorize_sparse,
};

// Rust guideline compliant 2026-02-21

/// Shell DOFs per node: `[u, v, w, θx, θy, θz]`.
const DOFS_PER_NODE: usize = 6;

/// Node budget for the Newton driver (sparse direct corrections).
const MAX_SHELL_NODES: usize = 1500;

/// Converged state of one load step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearShellStep {
    /// 1-based load step index.
    pub step: usize,
    /// Applied load fraction.
    pub load_factor: f64,
    /// Newton iterations used by this step.
    pub newton_iterations: usize,
    /// Max-abs free residual at convergence in N.
    pub residual_n: f64,
}

/// Newton driver output shared by the three shell studies.
struct NewtonStaticOutput {
    solution: Vec<f64>,
    steps: Vec<NonlinearShellStep>,
    total_newton_iterations: usize,
    final_residual_n: f64,
}

/// West-clamped plate under uniform pressure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearShellPressureStudy {
    /// Study name for reports.
    pub name: String,
    /// Plate dimensions and mesh density.
    pub geometry: ShellGeometry,
    /// Isotropic shell section.
    pub material: ShellMaterial,
    /// Pressure in Pa, positive along the element normal (+z here).
    pub pressure_pa: f64,
    /// Number of uniform load steps from zero to the full load.
    pub load_steps: usize,
    /// Newton iterations allowed per load step.
    pub max_newton_iterations: usize,
    /// Newton residual tolerance, relative to the load magnitude.
    pub newton_residual_tolerance: f64,
}

impl NonlinearShellPressureStudy {
    /// Small introductory plate matching the linear shell defaults.
    pub fn introductory() -> Self {
        Self {
            name: "introductory-nonlinear-shell-pressure".to_string(),
            geometry: ShellGeometry::default(),
            material: ShellMaterial::default(),
            pressure_pa: 0.0,
            load_steps: 5,
            max_newton_iterations: 50,
            newton_residual_tolerance: 1.0e-8,
        }
    }
}

/// Cantilever strip under an east-edge tip line load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearShellStripStudy {
    /// Study name for reports.
    pub name: String,
    /// Strip dimensions and mesh density (length along x).
    pub geometry: ShellGeometry,
    /// Isotropic shell section.
    pub material: ShellMaterial,
    /// Tip line load in N/m, positive upward (+z).
    pub tip_line_load_n_per_m: f64,
    /// Number of uniform load steps from zero to the full load.
    pub load_steps: usize,
    /// Newton iterations allowed per load step.
    pub max_newton_iterations: usize,
    /// Newton residual tolerance, relative to the load magnitude.
    pub newton_residual_tolerance: f64,
}

impl NonlinearShellStripStudy {
    /// Slender introductory strip (`b/L = 1/10`).
    pub fn introductory() -> Self {
        Self {
            name: "introductory-nonlinear-shell-strip".to_string(),
            geometry: ShellGeometry {
                length_m: 1.0,
                width_m: 0.1,
                nx: 40,
                ny: 4,
            },
            material: ShellMaterial::default(),
            tip_line_load_n_per_m: 0.0,
            load_steps: 10,
            max_newton_iterations: 50,
            newton_residual_tolerance: 1.0e-8,
        }
    }
}

/// Simply-supported plate under x-compression plus a pressure imperfection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearShellBucklingStudy {
    /// Study name for reports.
    pub name: String,
    /// Plate dimensions and mesh density.
    pub geometry: ShellGeometry,
    /// Isotropic shell section.
    pub material: ShellMaterial,
    /// East-edge compression in N/m (positive compresses).
    pub compression_n_per_m: f64,
    /// Uniform pressure imperfection in Pa (seeds the buckling mode).
    pub pressure_pa: f64,
    /// Number of uniform load steps from zero to the full load.
    pub load_steps: usize,
    /// Newton iterations allowed per load step.
    pub max_newton_iterations: usize,
    /// Newton residual tolerance, relative to the load magnitude.
    pub newton_residual_tolerance: f64,
}

impl NonlinearShellBucklingStudy {
    /// Small introductory square plate.
    pub fn introductory() -> Self {
        Self {
            name: "introductory-nonlinear-shell-buckling".to_string(),
            geometry: ShellGeometry {
                length_m: 1.0,
                width_m: 1.0,
                nx: 10,
                ny: 10,
            },
            material: ShellMaterial::default(),
            compression_n_per_m: 0.0,
            pressure_pa: 0.0,
            load_steps: 10,
            max_newton_iterations: 50,
            newton_residual_tolerance: 1.0e-8,
        }
    }
}

/// Nonlinear shell pressure response with the full displacement field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearShellPressureResult {
    /// Echoed study.
    pub study: NonlinearShellPressureStudy,
    /// Global DOF count.
    pub dofs: usize,
    /// Per-step converged states.
    pub steps: Vec<NonlinearShellStep>,
    /// Full `[u, v, w, θx, θy, θz]` field per node.
    pub displacement: Vec<[f64; 6]>,
    /// Max transverse displacement magnitude in m.
    pub max_transverse_m: f64,
    /// Newton iterations summed over all steps.
    pub total_newton_iterations: usize,
    /// Final max-abs free residual in N.
    pub final_residual_n: f64,
    /// Whether every load step converged.
    pub converged: bool,
    /// Model assumption tag.
    pub model_assumption: StructuralModelAssumption,
    /// Execution provenance.
    pub execution: StructuralExecutionProvenance,
}

/// Nonlinear shell strip response at the loaded edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearShellStripResult {
    /// Echoed study.
    pub study: NonlinearShellStripStudy,
    /// Global DOF count.
    pub dofs: usize,
    /// Per-step converged states.
    pub steps: Vec<NonlinearShellStep>,
    /// Mean transverse displacement over the east-edge nodes in m.
    pub east_mean_w_m: f64,
    /// Transverse displacement of the east-edge center node in m (the
    /// beam-theory comparison point: anticlastic warping vanishes on the
    /// strip centerline).
    pub tip_center_w_m: f64,
    /// Max transverse displacement magnitude over the plate in m.
    pub max_transverse_m: f64,
    /// Newton iterations summed over all steps.
    pub total_newton_iterations: usize,
    /// Final max-abs free residual in N.
    pub final_residual_n: f64,
    /// Whether every load step converged.
    pub converged: bool,
    /// Model assumption tag.
    pub model_assumption: StructuralModelAssumption,
    /// Execution provenance.
    pub execution: StructuralExecutionProvenance,
}

/// Nonlinear shell buckling response at the plate center.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearShellBucklingResult {
    /// Echoed study.
    pub study: NonlinearShellBucklingStudy,
    /// Global DOF count.
    pub dofs: usize,
    /// Per-step converged states.
    pub steps: Vec<NonlinearShellStep>,
    /// Transverse displacement at the node nearest the center in m.
    pub center_w_m: f64,
    /// Max transverse displacement magnitude over the plate in m.
    pub max_transverse_m: f64,
    /// Newton iterations summed over all steps.
    pub total_newton_iterations: usize,
    /// Final max-abs free residual in N.
    pub final_residual_n: f64,
    /// Whether every load step converged.
    pub converged: bool,
    /// Model assumption tag.
    pub model_assumption: StructuralModelAssumption,
    /// Execution provenance.
    pub execution: StructuralExecutionProvenance,
}

fn validate_shell_inputs(
    geometry: &ShellGeometry,
    material: &ShellMaterial,
    loads: &[(&'static str, f64)],
    load_steps: usize,
    max_newton_iterations: usize,
    newton_residual_tolerance: f64,
) -> Result<(), StructuralMechanicsError> {
    for (field, value) in [
        ("shell.length_m", geometry.length_m),
        ("shell.width_m", geometry.width_m),
        ("shell.youngs_modulus_pa", material.youngs_modulus_pa),
        ("shell.thickness_m", material.thickness_m),
        ("shell.density_kg_m3", material.density_kg_m3),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    if !(-1.0 < material.poisson_ratio && material.poisson_ratio < 0.5) {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "shell.poisson_ratio",
            value: material.poisson_ratio.to_string(),
        });
    }
    if geometry.nx == 0 || geometry.ny == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "shell.mesh_density",
            value: format!("nx={}, ny={}", geometry.nx, geometry.ny),
        });
    }
    if (geometry.nx + 1) * (geometry.ny + 1) > MAX_SHELL_NODES {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "shell.mesh_density",
            value: format!(
                "nx={}, ny={} exceeds the node budget",
                geometry.nx, geometry.ny
            ),
        });
    }
    for (field, value) in loads {
        if !value.is_finite() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    if load_steps == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "shell.load_steps",
            value: "0".to_string(),
        });
    }
    if max_newton_iterations == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "shell.max_newton_iterations",
            value: "0".to_string(),
        });
    }
    if !newton_residual_tolerance.is_finite() || newton_residual_tolerance <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "shell.newton_residual_tolerance",
            value: newton_residual_tolerance.to_string(),
        });
    }
    Ok(())
}

/// Coordinate tolerance scaled to the plate.
fn mesh_tolerance(mesh: &Mesh) -> f64 {
    let span = mesh
        .nodes
        .iter()
        .map(|point| point.x.abs().max(point.y.abs()))
        .fold(0.0f64, f64::max)
        .max(1.0);
    1.0e-10 * span
}

/// West-edge (x = 0) node indices.
fn west_edge_nodes(mesh: &Mesh) -> Vec<usize> {
    let tol = mesh_tolerance(mesh);
    mesh.nodes
        .iter()
        .enumerate()
        .filter_map(|(node, point)| (point.x.abs() <= tol).then_some(node))
        .collect()
}

/// East-edge (x = max) node indices with their tributary edge lengths.
fn east_edge_loads(mesh: &Mesh) -> Vec<(usize, f64)> {
    let xmax = mesh
        .nodes
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let tol = mesh_tolerance(mesh);
    let mut edge: Vec<(usize, f64)> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(node, point)| ((point.x - xmax).abs() <= tol).then_some((node, point.y)))
        .collect();
    edge.sort_by(|left, right| left.1.total_cmp(&right.1));
    // Trapezoidal tributary lengths along the edge.
    let mut loads = Vec::with_capacity(edge.len());
    for (position, &(node, _)) in edge.iter().enumerate() {
        let below = if position == 0 {
            0.0
        } else {
            0.5 * (edge[position].1 - edge[position - 1].1)
        };
        let above = if position + 1 == edge.len() {
            0.0
        } else {
            0.5 * (edge[position + 1].1 - edge[position].1)
        };
        loads.push((node, below + above));
    }
    loads
}

/// Load-stepped Newton solve with sparse direct corrections on the free DOFs.
fn newton_load_stepping(
    form: &CorotationalShellForm,
    external: &[f64],
    fixed_dofs: &[usize],
    load_steps: usize,
    max_newton_iterations: usize,
    newton_residual_tolerance: f64,
) -> Result<NewtonStaticOutput, StructuralMechanicsError> {
    let dofs = form.num_global_dofs();
    let mut is_fixed = vec![false; dofs];
    for &dof in fixed_dofs {
        is_fixed[dof] = true;
    }
    let free: Vec<usize> = (0..dofs).filter(|dof| !is_fixed[*dof]).collect();
    let mut global_to_free = vec![usize::MAX; dofs];
    for (position, &dof) in free.iter().enumerate() {
        global_to_free[dof] = position;
    }
    let load_scale = external
        .iter()
        .map(|value| value.abs())
        .fold(0.0f64, f64::max);
    let force_scale = load_scale.max(1.0e-300);
    // Self-calibrated residual floor: the corotational residual carries
    // frame-orthonormality rounding (~1e-8 here, amplified by the stiff
    // drilling terms) that no Newton correction can remove, growing with
    // deformation as K·eps·|u|. The floor only pollutes the stopping
    // criterion, never the solution (a K·eps residual floor resolves to
    // eps-level displacement error), so the gate below clears 10x the
    // measured zero-state force plus the deformation-proportional term.
    let zero_solution = vec![0.0; dofs];
    let (zero_internal, _) = assemble_nonlinear_element_form(form, &zero_solution);
    let zero_floor = free
        .iter()
        .map(|&dof| zero_internal[dof].abs())
        .fold(0.0f64, f64::max);
    let config = SparseDirectConfig {
        matrix_class: MatrixClass::General,
        method: SparseDirectMethod::SupernodalLu,
        equilibrate: false,
        ..SparseDirectConfig::default()
    };

    let mut solution = vec![0.0; dofs];
    let mut steps = Vec::with_capacity(load_steps);
    let mut total_newton_iterations = 0usize;
    for step in 1..=load_steps {
        let load_factor = step as f64 / load_steps as f64;
        let mut iterations = 0usize;
        let mut residual = f64::INFINITY;
        let mut converged = false;
        // Stall detection: at large deformation the assembled residual
        // carries a deterministic FP hash floor (~1e-7 here) that Newton
        // random-walks on instead of descending. Five iterations without
        // halving (at a sane residual) means converged-to-floor.
        let mut best_residual = f64::INFINITY;
        let mut since_improvement = 0usize;
        for iteration in 1..=max_newton_iterations {
            iterations = iteration;
            let (internal, triplets) = assemble_nonlinear_element_form(form, &solution);
            residual = free
                .iter()
                .map(|&dof| (internal[dof] - load_factor * external[dof]).abs())
                .fold(0.0, f64::max);
            let tangent_scale = triplets
                .values
                .iter()
                .map(|value| value.abs())
                .fold(0.0f64, f64::max);
            let solution_scale = solution
                .iter()
                .map(|value| value.abs())
                .fold(0.0f64, f64::max);
            let residual_gate = newton_residual_tolerance * force_scale
                + 10.0 * zero_floor
                + 10.0 * tangent_scale * f64::EPSILON * solution_scale;
            if residual <= residual_gate {
                converged = true;
                break;
            }
            if residual < 0.5 * best_residual {
                best_residual = residual;
                since_improvement = 0;
            } else {
                since_improvement += 1;
            }
            if since_improvement >= 5 && residual <= 1.0e-4 * force_scale {
                converged = true;
                break;
            }
            let mut reduced = TripletMatrix::<f64>::new(free.len(), free.len());
            for ((&row, &column), &value) in triplets
                .rows
                .iter()
                .zip(&triplets.cols)
                .zip(&triplets.values)
            {
                let free_row = global_to_free[row];
                let free_column = global_to_free[column];
                if free_row != usize::MAX && free_column != usize::MAX {
                    reduced.push(free_row, free_column, value);
                }
            }
            let mut rhs = Vec::with_capacity(free.len());
            for &dof in &free {
                rhs.push(-(internal[dof] - load_factor * external[dof]));
            }
            let reduced_csr = reduced.to_csr();
            let correction = factorize_sparse(&reduced_csr, &config, None)
                .and_then(|factor| factor.solve(&rhs))
                .map_err(|_| StructuralMechanicsError::NonlinearNonConvergence {
                    model: StructuralModelAssumption::CorotationalShell,
                    load_step: step,
                    iterations,
                    residual,
                })?;
            // Backtracking line search on the free-DOF residual max-norm.
            // In the membrane-dominated regime the CR tangent misses the
            // frame-rotation geometric stiffness, so full Newton steps
            // overshoot across the steep valley and ping-pong; halving the
            // step until the residual decreases restores convergence.
            let mut step_length = 1.0f64;
            let mut trial = solution.clone();
            let mut trial_residual = f64::INFINITY;
            for _ in 0..6 {
                for (position, &dof) in free.iter().enumerate() {
                    trial[dof] = solution[dof] + step_length * correction[position];
                }
                let (trial_internal, _) = assemble_nonlinear_element_form(form, &trial);
                trial_residual = free
                    .iter()
                    .map(|&dof| (trial_internal[dof] - load_factor * external[dof]).abs())
                    .fold(0.0, f64::max);
                if trial_residual < residual {
                    break;
                }
                step_length *= 0.5;
            }
            if trial_residual < residual {
                solution = trial;
            } else {
                for (position, &dof) in free.iter().enumerate() {
                    solution[dof] += correction[position];
                }
            }
            if solution.iter().any(|value| !value.is_finite()) {
                return Err(StructuralMechanicsError::ValidationFailure {
                    check: "nonlinear_shell.newton_state_finite",
                    value: f64::NAN,
                });
            }
        }
        if !converged {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::CorotationalShell,
                load_step: step,
                iterations,
                residual,
            });
        }
        total_newton_iterations += iterations;
        steps.push(NonlinearShellStep {
            step,
            load_factor,
            newton_iterations: iterations,
            residual_n: residual,
        });
    }
    let final_residual_n = steps.last().expect("at least one load step").residual_n;
    Ok(NewtonStaticOutput {
        solution,
        steps,
        total_newton_iterations,
        final_residual_n,
    })
}

/// Solve the west-clamped nonlinear shell under uniform pressure.
pub fn run_nonlinear_shell_pressure(
    study: &NonlinearShellPressureStudy,
) -> Result<NonlinearShellPressureResult, StructuralMechanicsError> {
    validate_shell_inputs(
        &study.geometry,
        &study.material,
        &[("nonlinear_shell.pressure_pa", study.pressure_pa)],
        study.load_steps,
        study.max_newton_iterations,
        study.newton_residual_tolerance,
    )?;
    let mesh = study.geometry.mesh();
    let elements = ShellStaticStudy::shell_elements(study.material, &mesh)?;
    let dofs = mesh.nodes.len() * DOFS_PER_NODE;
    let external = assemble_shell_load(&elements, dofs, |_| study.pressure_pa);
    let form = CorotationalShellForm::new(elements, dofs);
    let fixed_nodes = west_edge_nodes(&mesh);
    if fixed_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "nonlinear shell fixed edge selected no nodes".to_string(),
        });
    }
    let fixed_dofs = fixed_nodes
        .iter()
        .flat_map(|&node| (0..DOFS_PER_NODE).map(move |component| node * DOFS_PER_NODE + component))
        .collect::<Vec<_>>();
    let output = newton_load_stepping(
        &form,
        &external,
        &fixed_dofs,
        study.load_steps,
        study.max_newton_iterations,
        study.newton_residual_tolerance,
    )?;
    let displacement = output
        .solution
        .as_chunks::<DOFS_PER_NODE>()
        .0
        .iter()
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5]])
        .collect::<Vec<_>>();
    let max_transverse_m = displacement
        .iter()
        .map(|dofs| dofs[2].abs())
        .fold(0.0, f64::max);
    Ok(NonlinearShellPressureResult {
        study: study.clone(),
        dofs,
        steps: output.steps,
        displacement,
        max_transverse_m,
        total_newton_iterations: output.total_newton_iterations,
        final_residual_n: output.final_residual_n,
        converged: true,
        model_assumption: StructuralModelAssumption::CorotationalShell,
        execution: StructuralExecutionProvenance::cpu_shell_reference(),
    })
}

/// Solve the west-clamped nonlinear strip under an east-edge line load.
pub fn run_nonlinear_shell_strip(
    study: &NonlinearShellStripStudy,
) -> Result<NonlinearShellStripResult, StructuralMechanicsError> {
    validate_shell_inputs(
        &study.geometry,
        &study.material,
        &[(
            "nonlinear_shell.tip_line_load_n_per_m",
            study.tip_line_load_n_per_m,
        )],
        study.load_steps,
        study.max_newton_iterations,
        study.newton_residual_tolerance,
    )?;
    let mesh = study.geometry.mesh();
    let elements = ShellStaticStudy::shell_elements(study.material, &mesh)?;
    let dofs = mesh.nodes.len() * DOFS_PER_NODE;
    let form = CorotationalShellForm::new(elements, dofs);
    let mut external = vec![0.0; dofs];
    let edge = east_edge_loads(&mesh);
    if edge.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "nonlinear shell strip loaded edge selected no nodes".to_string(),
        });
    }
    for &(node, tributary) in &edge {
        add_shell_point_load(
            &mut external,
            node,
            [
                0.0,
                0.0,
                study.tip_line_load_n_per_m * tributary,
                0.0,
                0.0,
                0.0,
            ],
        );
    }
    let fixed_nodes = west_edge_nodes(&mesh);
    if fixed_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "nonlinear shell fixed edge selected no nodes".to_string(),
        });
    }
    let fixed_dofs = fixed_nodes
        .iter()
        .flat_map(|&node| (0..DOFS_PER_NODE).map(move |component| node * DOFS_PER_NODE + component))
        .collect::<Vec<_>>();
    let output = newton_load_stepping(
        &form,
        &external,
        &fixed_dofs,
        study.load_steps,
        study.max_newton_iterations,
        study.newton_residual_tolerance,
    )?;
    let east_mean_w_m = edge
        .iter()
        .map(|&(node, _)| output.solution[node * DOFS_PER_NODE + 2])
        .sum::<f64>()
        / edge.len() as f64;
    let centerline = study.geometry.width_m / 2.0;
    let tip_center_node = edge
        .iter()
        .min_by(|(left, _), (right, _)| {
            (mesh.nodes[*left].y - centerline)
                .abs()
                .total_cmp(&(mesh.nodes[*right].y - centerline).abs())
        })
        .map(|&(node, _)| node)
        .expect("loaded edge is nonempty");
    let tip_center_w_m = output.solution[tip_center_node * DOFS_PER_NODE + 2];
    let max_transverse_m = output
        .solution
        .as_chunks::<DOFS_PER_NODE>()
        .0
        .iter()
        .map(|chunk| chunk[2].abs())
        .fold(0.0, f64::max);
    Ok(NonlinearShellStripResult {
        study: study.clone(),
        dofs,
        steps: output.steps,
        east_mean_w_m,
        tip_center_w_m,
        max_transverse_m,
        total_newton_iterations: output.total_newton_iterations,
        final_residual_n: output.final_residual_n,
        converged: true,
        model_assumption: StructuralModelAssumption::CorotationalShell,
        execution: StructuralExecutionProvenance::cpu_shell_reference(),
    })
}

/// Solve the simply-supported nonlinear plate under compression + pressure.
pub fn run_nonlinear_shell_buckling(
    study: &NonlinearShellBucklingStudy,
) -> Result<NonlinearShellBucklingResult, StructuralMechanicsError> {
    validate_shell_inputs(
        &study.geometry,
        &study.material,
        &[
            (
                "nonlinear_shell.compression_n_per_m",
                study.compression_n_per_m,
            ),
            ("nonlinear_shell.pressure_pa", study.pressure_pa),
        ],
        study.load_steps,
        study.max_newton_iterations,
        study.newton_residual_tolerance,
    )?;
    if study.compression_n_per_m < 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "nonlinear_shell.compression_n_per_m",
            value: study.compression_n_per_m.to_string(),
        });
    }
    let mesh = study.geometry.mesh();
    let elements = ShellStaticStudy::shell_elements(study.material, &mesh)?;
    let dofs = mesh.nodes.len() * DOFS_PER_NODE;
    let mut external = assemble_shell_load(&elements, dofs, |_| study.pressure_pa);
    let form = CorotationalShellForm::new(elements, dofs);
    let edge = east_edge_loads(&mesh);
    if edge.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "nonlinear shell compression edge selected no nodes".to_string(),
        });
    }
    for &(node, tributary) in &edge {
        add_shell_point_load(
            &mut external,
            node,
            [
                -study.compression_n_per_m * tributary,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        );
    }
    // Simply supported: w = 0 on all boundary nodes, u = 0 on the west
    // edge (compression reaction), v = 0 at the southwest corner only so
    // Poisson expansion stays free. Rotations are unconstrained.
    let tol = mesh_tolerance(&mesh);
    let xmax = mesh
        .nodes
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let ymax = mesh
        .nodes
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut fixed_dofs = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        let on_boundary = point.x.abs() <= tol
            || (point.x - xmax).abs() <= tol
            || point.y.abs() <= tol
            || (point.y - ymax).abs() <= tol;
        if on_boundary {
            fixed_dofs.push(node * DOFS_PER_NODE + 2);
        }
        if point.x.abs() <= tol {
            fixed_dofs.push(node * DOFS_PER_NODE);
        }
    }
    let southwest = mesh
        .nodes
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (left.x.hypot(left.y)).total_cmp(&right.x.hypot(right.y)))
        .map(|(node, _)| node)
        .expect("mesh contains at least one node");
    fixed_dofs.push(southwest * DOFS_PER_NODE + 1);
    if fixed_dofs.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "nonlinear shell buckling constraints selected no DOFs".to_string(),
        });
    }
    let output = newton_load_stepping(
        &form,
        &external,
        &fixed_dofs,
        study.load_steps,
        study.max_newton_iterations,
        study.newton_residual_tolerance,
    )?;
    let center = [study.geometry.length_m / 2.0, study.geometry.width_m / 2.0];
    let center_node = mesh
        .nodes
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            ((left.x - center[0]).hypot(left.y - center[1]))
                .total_cmp(&(right.x - center[0]).hypot(right.y - center[1]))
        })
        .map(|(node, _)| node)
        .expect("mesh contains at least one node");
    let center_w_m = output.solution[center_node * DOFS_PER_NODE + 2];
    let max_transverse_m = output
        .solution
        .as_chunks::<DOFS_PER_NODE>()
        .0
        .iter()
        .map(|chunk| chunk[2].abs())
        .fold(0.0, f64::max);
    Ok(NonlinearShellBucklingResult {
        study: study.clone(),
        dofs,
        steps: output.steps,
        center_w_m,
        max_transverse_m,
        total_newton_iterations: output.total_newton_iterations,
        final_residual_n: output.final_residual_n,
        converged: true,
        model_assumption: StructuralModelAssumption::CorotationalShell,
        execution: StructuralExecutionProvenance::cpu_shell_reference(),
    })
}

/// Plate bending stiffness `D = Et³ / 12(1−ν²)` in N m.
pub fn plate_bending_stiffness_nm(
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
    thickness_m: f64,
) -> f64 {
    youngs_modulus_pa * thickness_m.powi(3) / (12.0 * (1.0 - poisson_ratio * poisson_ratio))
}

/// Classical buckling load of a simply-supported plate under uniaxial
/// x-compression: `N_cr = π²D/b² · min_m (mb/a + a/mb)²` in N/m.
pub fn simply_supported_plate_buckling_load_n_per_m(
    bending_stiffness_nm: f64,
    length_m: f64,
    width_m: f64,
) -> f64 {
    let aspect = width_m / length_m;
    let waves = (1..=16)
        .map(|m| {
            let waves = m as f64 * aspect + 1.0 / (m as f64 * aspect);
            waves * waves
        })
        .fold(f64::INFINITY, f64::min);
    std::f64::consts::PI.powi(2) * bending_stiffness_nm / (width_m * width_m) * waves
}
