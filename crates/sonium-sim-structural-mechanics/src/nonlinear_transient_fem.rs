//! Production-backed nonlinear 3-D transient FEM evidence.
//!
//! The reference integrates a bounded Newmark average-acceleration step with
//! the existing total-Lagrangian Neo-Hookean FEM/Newton path.  Consistent
//! mass and mass-proportional damping are added to the nonlinear residual and
//! tangent at element level, so every time step is an actual nonlinear FEM
//! solve rather than a scalar or post-processed surrogate.

use crate::hyperelastic::total_p1_tetra_energy;
use crate::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::{MassMatrix, assemble_mass};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{LocalMatrix, LocalVector, NonlinearForm};
use sonium_fem::mesh::{Mesh, box_mesh_tetrahedra};
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_with_bcs_and_load};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::time_integration::{
    NewmarkCoefficients, correct_from_displacement_slice, mass_proportional_damping_power,
    predict_displacement_slice, predict_velocity_slice, trapezoidal_force_work_increment,
    trapezoidal_power_increment,
};

/// Schema version for [`NonlinearTransientFem3DResult`].
pub const NONLINEAR_TRANSIENT_FEM_3D_SCHEMA_VERSION: u32 = 1;

const MAX_NONLINEAR_TRANSIENT_FEM_3D_STEPS: usize = 512;
const MAX_NONLINEAR_TRANSIENT_FEM_3D_LOAD_FACTOR: f64 = 1.0e6;

/// Bounded force-history study for the production 3-D hyperelastic transient
/// FEM/Newton path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearTransientFem3DStudy {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub youngs_modulus_pa: f64,
    pub poissons_ratio: f64,
    pub density_kg_m3: f64,
    pub damping_ratio: f64,
    pub reference_frequency_hz: f64,
    pub total_force_n: f64,
    pub time_step_s: f64,
    /// Dimensionless force multipliers. The first sample is the unloaded
    /// initial state; subsequent samples may vary in either direction.
    pub load_factors: Vec<f64>,
    pub max_newton_iterations: usize,
    pub newton_residual_tolerance: f64,
}

impl Default for NonlinearTransientFem3DStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl NonlinearTransientFem3DStudy {
    /// Small finite-strain block fixture suitable for focused CPU tests.
    pub fn introductory() -> Self {
        Self {
            length_m: 0.40,
            width_m: 0.20,
            height_m: 0.40,
            nx: 2,
            ny: 1,
            nz: 2,
            youngs_modulus_pa: 2.0e6,
            poissons_ratio: 0.30,
            density_kg_m3: 1_000.0,
            damping_ratio: 0.02,
            reference_frequency_hz: 5.0,
            total_force_n: 16.0,
            time_step_s: 1.0e-3,
            load_factors: vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0],
            max_newton_iterations: 40,
            newton_residual_tolerance: 1.0e-8,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("nonlinear_transient_fem_3d.length_m", self.length_m),
            ("nonlinear_transient_fem_3d.width_m", self.width_m),
            ("nonlinear_transient_fem_3d.height_m", self.height_m),
            (
                "nonlinear_transient_fem_3d.youngs_modulus_pa",
                self.youngs_modulus_pa,
            ),
            (
                "nonlinear_transient_fem_3d.density_kg_m3",
                self.density_kg_m3,
            ),
            (
                "nonlinear_transient_fem_3d.reference_frequency_hz",
                self.reference_frequency_hz,
            ),
            (
                "nonlinear_transient_fem_3d.total_force_n",
                self.total_force_n,
            ),
            ("nonlinear_transient_fem_3d.time_step_s", self.time_step_s),
            (
                "nonlinear_transient_fem_3d.newton_residual_tolerance",
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
        if !self.poissons_ratio.is_finite() || !(0.0..0.5).contains(&self.poissons_ratio) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient_fem_3d.poissons_ratio",
                value: self.poissons_ratio.to_string(),
            });
        }
        if !self.damping_ratio.is_finite() || !(0.0..=1.0).contains(&self.damping_ratio) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient_fem_3d.damping_ratio",
                value: self.damping_ratio.to_string(),
            });
        }
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient_fem_3d.resolution",
                value: format!("({}, {}, {})", self.nx, self.ny, self.nz),
            });
        }
        if self.load_factors.len() < 2
            || self.load_factors.len() > MAX_NONLINEAR_TRANSIENT_FEM_3D_STEPS
            || self.load_factors.iter().any(|value| {
                !value.is_finite() || value.abs() > MAX_NONLINEAR_TRANSIENT_FEM_3D_LOAD_FACTOR
            })
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient_fem_3d.load_factors",
                value: self.load_factors.len().to_string(),
            });
        }
        if self.load_factors[0].abs() > f64::EPSILON {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient_fem_3d.load_factors[0]",
                value: self.load_factors[0].to_string(),
            });
        }
        if self.max_newton_iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient_fem_3d.max_newton_iterations",
                value: self.max_newton_iterations.to_string(),
            });
        }
        Ok(())
    }
}

/// One converged or rejected implicit-Newmark nonlinear FEM sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearTransientFem3DStepResult {
    pub step: usize,
    pub time_s: f64,
    pub load_factor: f64,
    pub applied_force_n: f64,
    pub newton_iterations: usize,
    pub dynamic_equilibrium_residual_n: f64,
    pub max_displacement_m: f64,
    pub max_velocity_m_per_s: f64,
    pub max_acceleration_m_per_s2: f64,
    pub kinetic_energy_j: f64,
    pub strain_energy_j: f64,
    pub total_energy_j: f64,
    pub damping_power_w: f64,
    pub converged: bool,
}

/// Versioned result for the bounded nonlinear transient FEM path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearTransientFem3DResult {
    pub schema_version: u32,
    pub study: NonlinearTransientFem3DStudy,
    pub mesh_nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub bottom_nodes: usize,
    pub top_nodes: usize,
    pub steps: Vec<NonlinearTransientFem3DStepResult>,
    pub final_displacement_m: f64,
    pub final_velocity_m_per_s: f64,
    pub final_acceleration_m_per_s2: f64,
    pub final_dynamic_equilibrium_residual_n: f64,
    pub max_dynamic_equilibrium_residual_n: f64,
    pub max_total_energy_j: f64,
    pub total_input_work_j: f64,
    pub total_damping_dissipation_j: f64,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run the bounded production-backed nonlinear 3-D transient FEM reference.
pub fn run_nonlinear_transient_fem_3d(
    study: &NonlinearTransientFem3DStudy,
) -> Result<NonlinearTransientFem3DResult, StructuralMechanicsError> {
    study.validate()?;

    let mesh = box_mesh_tetrahedra(
        0.0,
        study.length_m,
        0.0,
        study.width_m,
        0.0,
        study.height_m,
        study.nx,
        study.ny,
        study.nz,
    );
    mesh.validate_connectivity()
        .map_err(|errors| StructuralMechanicsError::InvalidMesh {
            reason: format!("3-D nonlinear transient mesh connectivity: {errors:?}"),
        })?;

    let tolerance = 1.0e-12
        * study
            .length_m
            .max(study.width_m)
            .max(study.height_m)
            .max(1.0);
    let bottom_nodes = nodes_at_z(&mesh, 0.0, tolerance);
    let top_nodes = nodes_at_z(&mesh, study.height_m, tolerance);
    if bottom_nodes.is_empty() || top_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!(
                "3-D nonlinear transient requires bottom and top nodes (bottom={}, top={})",
                bottom_nodes.len(),
                top_nodes.len()
            ),
        });
    }

    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let dofs = space.num_global_dofs(&mesh);
    let material = NeoHookean::from_young(study.youngs_modulus_pa, study.poissons_ratio, 3);
    let scalar_mass = assemble_mass(&mesh, PolynomialDegree::P1);
    let mass = vector_mass_matrix(&scalar_mass, 3, study.density_kg_m3);
    if mass.iter().any(|value| !value.is_finite()) {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "nonlinear_transient_fem_3d.mass_matrix_finite",
            value: f64::NAN,
        });
    }
    let mass_damping_per_mass_s =
        2.0 * study.damping_ratio * std::f64::consts::TAU * study.reference_frequency_hz;

    let dt = study.time_step_s;
    let coefficients = NewmarkCoefficients::average_acceleration(dt).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "3-D nonlinear transient Newmark coefficients",
            message: error.to_string(),
        }
    })?;
    let acceleration_coefficient = coefficients.acceleration_coefficient();
    let velocity_coefficient = coefficients.velocity_coefficient();
    let effective_coefficient = coefficients
        .effective_mass_coefficient(mass_damping_per_mass_s)
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D nonlinear transient effective Newmark coefficient",
            message: error.to_string(),
        })?;

    let mut config = FemNewtonConfig::default();
    config.newton.max_iterations = study.max_newton_iterations;
    config.newton.absolute_tolerance = study.newton_residual_tolerance;
    config.newton.relative_tolerance = study.newton_residual_tolerance;
    config.newton.line_search = sonium_solvers::LineSearch::Backtracking {
        c1: 1.0e-4,
        max_halvings: 20,
    };
    config.inner_solver.max_iterations = study.max_newton_iterations.saturating_mul(20);

    let mut displacement = vec![0.0; dofs];
    let mut velocity = vec![0.0; dofs];
    let mut acceleration = vec![0.0; dofs];
    let zero_force = load_vector(dofs, &top_nodes, study.total_force_n, 0.0);
    let zero_strain_energy = total_p1_tetra_energy(&mesh, &displacement, &material)?;
    let mut steps = vec![step_result(
        0,
        0.0,
        study.load_factors[0],
        0.0,
        0,
        0.0,
        &displacement,
        &velocity,
        &acceleration,
        0.0,
        zero_strain_energy,
        0.0,
        true,
    )];
    let mut max_dynamic_residual: f64 = 0.0;
    let mut max_total_energy = zero_strain_energy;
    let mut total_input_work = 0.0;
    let mut total_damping_dissipation = 0.0;
    let mut previous_damping_power = 0.0;
    let mut previous_force = zero_force;

    for step in 1..study.load_factors.len() {
        let mut predicted_displacement = vec![0.0; dofs];
        let mut predicted_velocity = vec![0.0; dofs];
        predict_displacement_slice(
            &coefficients,
            &displacement,
            &velocity,
            &acceleration,
            &mut predicted_displacement,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D nonlinear transient Newmark predictor",
            message: error.to_string(),
        })?;
        predict_velocity_slice(
            &coefficients,
            &velocity,
            &acceleration,
            &mut predicted_velocity,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D nonlinear transient Newmark predictor",
            message: error.to_string(),
        })?;
        let force = load_vector(
            dofs,
            &top_nodes,
            study.total_force_n,
            study.load_factors[step],
        );
        let mass_predicted = mat_vec(&mass, &predicted_displacement, dofs);
        let mass_velocity = mat_vec(&mass, &predicted_velocity, dofs);
        let external_load = force
            .iter()
            .zip(&mass_predicted)
            .zip(&mass_velocity)
            .map(|((&f, &mp), &mv)| {
                f + acceleration_coefficient * mp
                    + mass_damping_per_mass_s * velocity_coefficient * mp
                    - mass_damping_per_mass_s * mv
            })
            .collect::<Vec<_>>();
        if external_load.iter().any(|value| !value.is_finite()) {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "nonlinear_transient_fem_3d.effective_load_finite",
                value: step as f64,
            });
        }

        let form = EffectiveDynamicHyperelasticForm {
            hyperelastic: HyperelasticForm::new(material),
            density_kg_m3: study.density_kg_m3,
            effective_coefficient,
        };
        let newton = fem_newton_solve_with_bcs_and_load(
            &form,
            &space,
            &mesh,
            &guided_boundary_conditions(&bottom_nodes, &top_nodes),
            &predicted_displacement,
            &external_load,
            &config,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D nonlinear transient FEM Newton solve",
            message: error.to_string(),
        })?;
        if !newton.converged || !newton.residual.is_finite() {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::NonlinearTransientFem3D,
                load_step: step,
                iterations: newton.iterations,
                residual: newton.residual,
            });
        }

        let next_displacement = newton.x;
        let mut next_velocity = vec![0.0; dofs];
        let mut next_acceleration = vec![0.0; dofs];
        correct_from_displacement_slice(
            &coefficients,
            &predicted_displacement,
            &predicted_velocity,
            &next_displacement,
            &mut next_velocity,
            &mut next_acceleration,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D nonlinear transient Newmark corrector",
            message: error.to_string(),
        })?;

        let mass_velocity = mat_vec(&mass, &next_velocity, dofs);
        let kinetic_energy = 0.5 * dot(&next_velocity, &mass_velocity);
        let strain_energy = total_p1_tetra_energy(&mesh, &next_displacement, &material)?;
        let total_energy = kinetic_energy + strain_energy;
        let damping_power = mass_proportional_damping_power(
            mass_damping_per_mass_s,
            &next_velocity,
            &mass_velocity,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D nonlinear transient damping power",
            message: error.to_string(),
        })?;
        total_input_work += trapezoidal_force_work_increment(
            &previous_force,
            &force,
            &displacement,
            &next_displacement,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D nonlinear transient input work",
            message: error.to_string(),
        })?;
        total_damping_dissipation +=
            trapezoidal_power_increment(dt, previous_damping_power, damping_power);
        previous_damping_power = damping_power;
        previous_force = force;

        if !kinetic_energy.is_finite()
            || !strain_energy.is_finite()
            || !total_energy.is_finite()
            || !damping_power.is_finite()
        {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "nonlinear_transient_fem_3d.finite_energy_diagnostics",
                value: step as f64,
            });
        }

        max_dynamic_residual = max_dynamic_residual.max(newton.residual);
        max_total_energy = max_total_energy.max(total_energy);
        steps.push(step_result(
            step,
            step as f64 * dt,
            study.load_factors[step],
            study.total_force_n * study.load_factors[step],
            newton.iterations,
            newton.residual,
            &next_displacement,
            &next_velocity,
            &next_acceleration,
            kinetic_energy,
            strain_energy,
            damping_power,
            true,
        ));
        displacement = next_displacement;
        velocity = next_velocity;
        acceleration = next_acceleration;
    }

    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.nonlinear-transient-fem-3d.v1".to_string();
    execution.limitations.extend([
        "Bounded P1 tetrahedral Neo-Hookean block with mass-proportional damping and prescribed nodal force history; no contact, fracture, topology update, or shell dynamics.".to_string(),
        "CPU/f64 reference path only; no CUDA, Metal, WASM, external-solver, or calibrated measured-loading parity is represented.".to_string(),
    ]);

    Ok(NonlinearTransientFem3DResult {
        schema_version: NONLINEAR_TRANSIENT_FEM_3D_SCHEMA_VERSION,
        study: study.clone(),
        mesh_nodes: mesh.nodes.len(),
        elements: mesh.elements.len(),
        dofs,
        bottom_nodes: bottom_nodes.len(),
        top_nodes: top_nodes.len(),
        final_displacement_m: max_nodal_norm(&displacement),
        final_velocity_m_per_s: max_nodal_norm(&velocity),
        final_acceleration_m_per_s2: max_nodal_norm(&acceleration),
        final_dynamic_equilibrium_residual_n: steps
            .last()
            .map(|step| step.dynamic_equilibrium_residual_n)
            .unwrap_or(0.0),
        max_dynamic_equilibrium_residual_n: max_dynamic_residual,
        max_total_energy_j: max_total_energy,
        total_input_work_j: total_input_work,
        total_damping_dissipation_j: total_damping_dissipation,
        converged: true,
        steps,
        model_assumption: StructuralModelAssumption::NonlinearTransientFem3D,
        execution,
    })
}

struct EffectiveDynamicHyperelasticForm {
    hyperelastic: HyperelasticForm<NeoHookean>,
    density_kg_m3: f64,
    effective_coefficient: f64,
}

impl NonlinearForm<f64> for EffectiveDynamicHyperelasticForm {
    fn element_residual(
        &self,
        test: &sonium_fem::spaces::BasisTabulation,
        local_solution: &[f64],
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        <HyperelasticForm<NeoHookean> as NonlinearForm<f64>>::element_residual(
            &self.hyperelastic,
            test,
            local_solution,
            weight,
            local_vector,
        );
        let weighted_mass = weight * test.det_j * self.density_kg_m3 * self.effective_coefficient;
        for a in 0..test.n_dofs {
            let component = a % test.spatial_dim;
            let phi_a = test.vector_value(a, component);
            let mut value = 0.0;
            for b in (component..test.n_dofs).step_by(test.spatial_dim) {
                value += phi_a * test.vector_value(b, component) * local_solution[b];
            }
            local_vector.add(a, weighted_mass * value);
        }
    }

    fn element_tangent(
        &self,
        test: &sonium_fem::spaces::BasisTabulation,
        trial: &sonium_fem::spaces::BasisTabulation,
        local_solution: &[f64],
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        <HyperelasticForm<NeoHookean> as NonlinearForm<f64>>::element_tangent(
            &self.hyperelastic,
            test,
            trial,
            local_solution,
            weight,
            local_matrix,
        );
        let weighted_mass = weight * test.det_j * self.density_kg_m3 * self.effective_coefficient;
        for a in 0..test.n_dofs {
            for b in 0..trial.n_dofs {
                if a % test.spatial_dim == b % trial.spatial_dim {
                    let component = a % test.spatial_dim;
                    local_matrix.add(
                        a,
                        b,
                        weighted_mass
                            * test.vector_value(a, component)
                            * trial.vector_value(b, component),
                    );
                }
            }
        }
    }
}

fn nodes_at_z(mesh: &Mesh, z: f64, tolerance: f64) -> Vec<usize> {
    mesh.nodes
        .iter()
        .enumerate()
        .filter_map(|(node, point)| ((point.z - z).abs() <= tolerance).then_some(node))
        .collect()
}

fn guided_boundary_conditions(bottom_nodes: &[usize], top_nodes: &[usize]) -> Vec<(usize, f64)> {
    let mut conditions = Vec::with_capacity(bottom_nodes.len() * 3 + top_nodes.len() * 2);
    for &node in bottom_nodes {
        conditions.extend([(3 * node, 0.0), (3 * node + 1, 0.0), (3 * node + 2, 0.0)]);
    }
    for &node in top_nodes {
        conditions.extend([(3 * node, 0.0), (3 * node + 1, 0.0)]);
    }
    conditions
}

fn load_vector(dofs: usize, top_nodes: &[usize], total_force_n: f64, load_factor: f64) -> Vec<f64> {
    let mut load = vec![0.0; dofs];
    let force_per_node = -total_force_n * load_factor / top_nodes.len() as f64;
    for &node in top_nodes {
        load[3 * node + 2] = force_per_node;
    }
    load
}

fn vector_mass_matrix(scalar: &MassMatrix, components: usize, density: f64) -> Vec<f64> {
    let dofs = scalar.dim * components;
    let mut matrix = vec![0.0; dofs * dofs];
    for ((&row, &column), &value) in scalar.rows.iter().zip(&scalar.cols).zip(&scalar.values) {
        for component in 0..components {
            let vector_row = row * components + component;
            let vector_column = column * components + component;
            matrix[vector_row * dofs + vector_column] += density * value;
        }
    }
    matrix
}

fn mat_vec(matrix: &[f64], vector: &[f64], dimension: usize) -> Vec<f64> {
    matrix
        .chunks_exact(dimension)
        .map(|row| row.iter().zip(vector).map(|(&a, &b)| a * b).sum())
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn max_nodal_norm(values: &[f64]) -> f64 {
    values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|node| dot(node, node).sqrt())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_dynamic_tangent_has_quadratic_taylor_remainder() {
        let mesh = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1);
        let space = VectorH1Space::new(PolynomialDegree::P1, 3);
        let material = NeoHookean::from_young(2.0e6, 0.30, 3);
        let form = EffectiveDynamicHyperelasticForm {
            hyperelastic: HyperelasticForm::new(material),
            density_kg_m3: 1_000.0,
            effective_coefficient: 4.0e6,
        };
        let tab = space.tabulate(&mesh, 0, 0.25, 0.25, 0.25);
        let state: Vec<f64> = (0..tab.n_dofs)
            .map(|i| 0.01 * ((i % 5) as f64 - 2.0))
            .collect();
        let direction: Vec<f64> = (0..tab.n_dofs)
            .map(|i| 0.1 * ((i % 7) as f64 - 3.0))
            .collect();

        let report = sonium_fem::validation::check_tangent_consistency(
            |current| {
                let mut residual = LocalVector::zeros(current.len());
                form.element_residual(&tab, current, 1.0, &mut residual);
                residual.values
            },
            |current, tangent_direction| {
                let mut tangent = LocalMatrix::zeros(current.len());
                form.element_tangent(&tab, &tab, current, 1.0, &mut tangent);
                (0..current.len())
                    .map(|row| {
                        (0..current.len())
                            .map(|column| tangent.get(row, column) * tangent_direction[column])
                            .sum()
                    })
                    .collect()
            },
            &state,
            &direction,
            &[1.0e-3, 5.0e-4, 2.5e-4],
        )
        .expect("effective dynamic residual and tangent dimensions must agree");

        assert!(report.observed_order > 1.8, "{report:?}");
    }
}

#[allow(clippy::too_many_arguments)]
fn step_result(
    step: usize,
    time_s: f64,
    load_factor: f64,
    applied_force_n: f64,
    newton_iterations: usize,
    dynamic_equilibrium_residual_n: f64,
    displacement: &[f64],
    velocity: &[f64],
    acceleration: &[f64],
    kinetic_energy_j: f64,
    strain_energy_j: f64,
    damping_power_w: f64,
    converged: bool,
) -> NonlinearTransientFem3DStepResult {
    NonlinearTransientFem3DStepResult {
        step,
        time_s,
        load_factor,
        applied_force_n,
        newton_iterations,
        dynamic_equilibrium_residual_n,
        max_displacement_m: max_nodal_norm(displacement),
        max_velocity_m_per_s: max_nodal_norm(velocity),
        max_acceleration_m_per_s2: max_nodal_norm(acceleration),
        kinetic_energy_j,
        strain_energy_j,
        total_energy_j: kinetic_energy_j + strain_energy_j,
        damping_power_w,
        converged,
    }
}
