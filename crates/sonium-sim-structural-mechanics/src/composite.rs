//! Executable composite laminate membrane studies.
//!
//! The family solves an assembled laminate-A membrane problem under an
//! integrated edge traction.  It reports the ABD provenance and recovers each
//! ply's local stress and failure indices at element centroids, keeping the
//! failure constraint separate from a global von-Mises surrogate.

use super::{
    StructuralMechanicsError, StructuralModelAssumption, dot, free_residual_norm, triplet_matvec,
};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{
    FemSystem, apply_traction_vector, try_apply_homogeneous_dirichlet_dofs,
};
use sonium_fem::mesh::{ElementType, Mesh, rectangular_mesh_triangles};
use sonium_fem::physics::solid::composites::{
    Laminate, LaminateMembraneForm, OrthotropicPly, OrthotropicStrength,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{CgConfig, cg};

/// Fixed and loaded boundary markers from the rectangular mesh generator.
pub const COMPOSITE_FIXED_BOUNDARY_MARKER: i32 = 1;
pub const COMPOSITE_LOADED_BOUNDARY_MARKER: i32 = 2;

/// Ply strength allowables used by Tsai–Hill and maximum-stress checks.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositePlyStrength {
    pub x_tension_pa: f64,
    pub x_compression_pa: f64,
    pub y_tension_pa: f64,
    pub y_compression_pa: f64,
    pub shear_pa: f64,
}

impl Default for CompositePlyStrength {
    fn default() -> Self {
        Self {
            x_tension_pa: 1_500.0e6,
            x_compression_pa: 1_200.0e6,
            y_tension_pa: 40.0e6,
            y_compression_pa: 200.0e6,
            shear_pa: 80.0e6,
        }
    }
}

impl CompositePlyStrength {
    fn as_fem(self) -> OrthotropicStrength {
        OrthotropicStrength {
            x_tension: self.x_tension_pa,
            x_compression: self.x_compression_pa,
            y_tension: self.y_tension_pa,
            y_compression: self.y_compression_pa,
            shear: self.shear_pa,
        }
    }
}

/// One orthotropic ply, including the allowables used for result recovery.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositePly {
    pub e1_pa: f64,
    pub e2_pa: f64,
    pub g12_pa: f64,
    pub poisson_12: f64,
    pub thickness_m: f64,
    pub angle_rad: f64,
    pub strength: CompositePlyStrength,
}

impl CompositePly {
    fn as_fem(self) -> OrthotropicPly {
        OrthotropicPly {
            e1: self.e1_pa,
            e2: self.e2_pa,
            g12: self.g12_pa,
            nu12: self.poisson_12,
            thickness: self.thickness_m,
            angle_rad: self.angle_rad,
        }
    }
}

/// Rectangular mesh for the in-plane laminate membrane study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositeMembraneGeometry {
    pub length_m: f64,
    pub width_m: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Default for CompositeMembraneGeometry {
    fn default() -> Self {
        Self {
            length_m: 1.0,
            width_m: 0.1,
            nx: 20,
            ny: 4,
        }
    }
}

impl CompositeMembraneGeometry {
    fn mesh(self) -> Mesh {
        rectangular_mesh_triangles(0.0, self.length_m, 0.0, self.width_m, self.nx, self.ny)
    }
}

/// A laminate membrane study under a total axial edge force.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeLaminateStudy {
    pub name: String,
    pub geometry: CompositeMembraneGeometry,
    pub plies: Vec<CompositePly>,
    pub total_force_n: f64,
    pub fixed_boundary_marker: i32,
    pub loaded_boundary_marker: i32,
}

impl Default for CompositeLaminateStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl CompositeLaminateStudy {
    /// Unidirectional tension benchmark with a free lateral response.
    pub fn introductory() -> Self {
        Self {
            name: "composite-laminate-unidirectional-tension".to_string(),
            geometry: CompositeMembraneGeometry::default(),
            plies: vec![CompositePly {
                e1_pa: 140.0e9,
                e2_pa: 10.0e9,
                g12_pa: 5.0e9,
                poisson_12: 0.25,
                thickness_m: 1.0e-3,
                angle_rad: 0.0,
                strength: CompositePlyStrength::default(),
            }],
            total_force_n: 1.0e3,
            fixed_boundary_marker: COMPOSITE_FIXED_BOUNDARY_MARKER,
            loaded_boundary_marker: COMPOSITE_LOADED_BOUNDARY_MARKER,
        }
    }

    fn laminate(&self) -> Laminate {
        Laminate {
            plies: self
                .plies
                .iter()
                .copied()
                .map(CompositePly::as_fem)
                .collect(),
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        let geometry = self.geometry;
        for (field, value) in [
            ("composite.length_m", geometry.length_m),
            ("composite.width_m", geometry.width_m),
            ("composite.total_force_n", self.total_force_n),
        ] {
            if !value.is_finite() || value.abs() <= f64::EPSILON {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if geometry.nx < 2 || geometry.ny < 2 || !geometry.ny.is_multiple_of(2) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "composite.mesh_density",
                value: format!(
                    "nx={}, ny={} (ny must be even and both >= 2)",
                    geometry.nx, geometry.ny
                ),
            });
        }
        if self.fixed_boundary_marker == self.loaded_boundary_marker {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "composite.boundary_markers",
                value: "fixed and loaded markers must differ".to_string(),
            });
        }
        if self.plies.is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "composite.plies",
                value: "at least one ply is required".to_string(),
            });
        }
        for (index, ply) in self.plies.iter().enumerate() {
            for (field, value) in [
                ("e1_pa", ply.e1_pa),
                ("e2_pa", ply.e2_pa),
                ("g12_pa", ply.g12_pa),
                ("thickness_m", ply.thickness_m),
                ("x_tension_pa", ply.strength.x_tension_pa),
                ("x_compression_pa", ply.strength.x_compression_pa),
                ("y_tension_pa", ply.strength.y_tension_pa),
                ("y_compression_pa", ply.strength.y_compression_pa),
                ("shear_pa", ply.strength.shear_pa),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    return Err(StructuralMechanicsError::InvalidConfiguration {
                        field: "composite.ply",
                        value: format!("ply {index}: {field}={value}"),
                    });
                }
            }
            if !ply.poisson_12.is_finite() || !(-1.0 < ply.poisson_12 && ply.poisson_12 < 0.5) {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "composite.ply.poisson_12",
                    value: format!("ply {index}: {}", ply.poisson_12),
                });
            }
            if !ply.angle_rad.is_finite() {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "composite.ply.angle_rad",
                    value: format!("ply {index}: {}", ply.angle_rad),
                });
            }
        }
        Ok(())
    }
}

/// Element-centroid stress and failure-index provenance for one ply.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositePlyFailureResult {
    pub ply_index: usize,
    pub angle_rad: f64,
    pub element_index: usize,
    pub global_stress_pa: [f64; 3],
    pub local_stress_pa: [f64; 3],
    pub tsai_hill_index: f64,
    pub max_stress_index: f64,
}

/// Validated assembled response of a composite laminate membrane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeLaminateStaticResult {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub laminate_thickness_m: f64,
    pub membrane_a: [[f64; 3]; 3],
    pub coupling_b: [[f64; 3]; 3],
    pub tip_axial_displacement_m: f64,
    pub analytical_tip_axial_displacement_m: f64,
    pub analytical_relative_error: f64,
    pub applied_force_n: [f64; 2],
    pub reaction_force_n: [f64; 2],
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub load_integration_relative_error: f64,
    pub strain_energy_j: f64,
    pub energy_balance_relative_error: f64,
    pub max_tsai_hill_index: f64,
    pub max_stress_index: f64,
    pub ply_failure: Vec<CompositePlyFailureResult>,
    pub model_assumption: StructuralModelAssumption,
    pub displacement: Vec<[f64; 2]>,
}

/// Solve an assembled composite membrane tension study.
pub fn run_composite_laminate_static(
    study: &CompositeLaminateStudy,
) -> Result<CompositeLaminateStaticResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.geometry.mesh();
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != ElementType::Triangle)
    {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "composite membrane requires triangular elements".to_string(),
        });
    }
    let laminate = study.laminate();
    let abd = laminate.stiffness_abd();
    let a_matrix = abd.a;
    let coupling_norm = abd
        .b
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let a_scale = a_matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    if a_matrix[0][0] <= 0.0
        || !a_matrix
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite())
    {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "composite.membrane_stiffness",
            value: format!("A11={}", a_matrix[0][0]),
        });
    }
    if coupling_norm > 1.0e-8 * a_scale {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "composite.bending_coupling",
            value: format!("B_max={coupling_norm} is not negligible for membrane-only execution"),
        });
    }

    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let dofs = space.num_global_dofs(&mesh);
    let stiffness = assemble_bilinear(
        &LaminateMembraneForm::new(&laminate),
        &space,
        &space,
        &mesh,
        1,
    );
    let unconstrained_stiffness = stiffness.clone();
    let mut system = FemSystem::new(stiffness, vec![0.0; dofs]);
    let traction = study.total_force_n / study.geometry.width_m;
    apply_traction_vector(
        &mut system,
        &mesh,
        &space,
        study.loaded_boundary_marker,
        &|_, _, _| vec![traction, 0.0],
    );
    let external_rhs = system.rhs.clone();
    let fixed_dofs = axial_restraint_dofs(&mesh, study.geometry, study.fixed_boundary_marker)?;
    try_apply_homogeneous_dirichlet_dofs(&mut system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "composite membrane boundary",
            message: error.to_string(),
        }
    })?;
    let solved = cg(
        &system.matrix.to_csr(),
        &system.rhs,
        &CgConfig {
            max_iterations: 10_000,
            tolerance: 1.0e-10,
            print_interval: 0,
        },
    );
    if !solved.converged {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::CompositeLaminate,
            iterations: solved.iterations,
            residual: solved.residual,
        });
    }

    let displacement = solved
        .x
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| [chunk[0], chunk[1]])
        .collect::<Vec<_>>();
    let internal_force = triplet_matvec(&unconstrained_stiffness, &solved.x);
    let applied_force_n = component_force_2(&external_rhs);
    let reaction_force_n = reaction_force_2(&internal_force, &external_rhs, &fixed_dofs);
    let force_scale = norm2(applied_force_n).max(1.0);
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
    let force_balance_residual_n = norm2(add2(applied_force_n, reaction_force_n));
    let expected_force = [study.total_force_n, 0.0];
    let load_integration_relative_error =
        norm2(sub2(applied_force_n, expected_force)) / study.total_force_n.abs();
    let internal_energy = 0.5 * dot(&solved.x, &internal_force);
    let rhs_energy = 0.5 * dot(&solved.x, &external_rhs);
    let energy_balance_relative_error =
        (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0);
    if !load_integration_relative_error.is_finite() || load_integration_relative_error > 1.0e-10 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "composite.load_integration",
            value: load_integration_relative_error,
        });
    }
    if !free_equilibrium_residual_n.is_finite()
        || free_equilibrium_residual_n > 1.0e-6 * force_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "composite.free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !force_balance_residual_n.is_finite() || force_balance_residual_n > 1.0e-6 * force_scale {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "composite.force_balance",
            value: force_balance_residual_n,
        });
    }
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1.0e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "composite.energy_identity",
            value: energy_balance_relative_error,
        });
    }

    let target_strain = solve_3x3(
        a_matrix,
        [study.total_force_n / study.geometry.width_m, 0.0, 0.0],
    )
    .ok_or_else(|| StructuralMechanicsError::SolverFailure {
        stage: "composite analytical benchmark",
        message: "laminate A matrix is singular".to_string(),
    })?;
    let center_tip_node = study.geometry.ny / 2 * (study.geometry.nx + 1) + study.geometry.nx;
    let tip_axial_displacement_m = solved.x[2 * center_tip_node];
    let analytical_tip_axial_displacement_m = target_strain[0] * study.geometry.length_m;
    let analytical_relative_error =
        (tip_axial_displacement_m - analytical_tip_axial_displacement_m).abs()
            / analytical_tip_axial_displacement_m.abs().max(f64::EPSILON);
    let ply_failure = recover_ply_failure(&mesh, &displacement, &study.plies);
    let max_tsai_hill_index = ply_failure
        .iter()
        .map(|failure| failure.tsai_hill_index)
        .fold(0.0, f64::max);
    let max_stress_index = ply_failure
        .iter()
        .map(|failure| failure.max_stress_index)
        .fold(0.0, f64::max);
    Ok(CompositeLaminateStaticResult {
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs,
        solver_iterations: solved.iterations,
        solver_residual: solved.residual,
        converged: solved.converged,
        laminate_thickness_m: laminate.total_thickness(),
        membrane_a: abd.a,
        coupling_b: abd.b,
        tip_axial_displacement_m,
        analytical_tip_axial_displacement_m,
        analytical_relative_error,
        applied_force_n,
        reaction_force_n,
        free_equilibrium_residual_n,
        force_balance_residual_n,
        load_integration_relative_error,
        strain_energy_j: internal_energy,
        energy_balance_relative_error,
        max_tsai_hill_index,
        max_stress_index,
        ply_failure,
        model_assumption: StructuralModelAssumption::CompositeLaminate,
        displacement,
    })
}

fn axial_restraint_dofs(
    mesh: &Mesh,
    geometry: CompositeMembraneGeometry,
    marker: i32,
) -> Result<Vec<usize>, StructuralMechanicsError> {
    let tolerance = 1.0e-10 * geometry.length_m.max(geometry.width_m).max(1.0);
    let mut dofs = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        if (point.x).abs() <= tolerance {
            dofs.push(2 * node); // axial restraint across the left edge
        }
    }
    let bottom_left = mesh
        .nodes
        .iter()
        .position(|point| point.x.abs() <= tolerance && point.y.abs() <= tolerance)
        .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
            reason: format!("boundary marker {marker} has no bottom-left node"),
        })?;
    dofs.push(2 * bottom_left + 1); // one transverse pin removes rigid drift
    Ok(dofs)
}

fn recover_ply_failure(
    mesh: &Mesh,
    displacement: &[[f64; 2]],
    plies: &[CompositePly],
) -> Vec<CompositePlyFailureResult> {
    let mut governing = plies
        .iter()
        .enumerate()
        .map(|(ply_index, ply)| CompositePlyFailureResult {
            ply_index,
            angle_rad: ply.angle_rad,
            element_index: 0,
            global_stress_pa: [0.0; 3],
            local_stress_pa: [0.0; 3],
            tsai_hill_index: f64::NEG_INFINITY,
            max_stress_index: f64::NEG_INFINITY,
        })
        .collect::<Vec<_>>();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let nodes = element.vertices();
        if nodes.len() != 3 {
            continue;
        }
        let gradients = triangle_shape_gradients(mesh, nodes);
        let mut grad = [[0.0; 2]; 2];
        for (local, &node) in nodes.iter().enumerate() {
            for component in 0..2 {
                for direction in 0..2 {
                    grad[component][direction] +=
                        displacement[node][component] * gradients[local][direction];
                }
            }
        }
        let strain = [grad[0][0], grad[1][1], grad[0][1] + grad[1][0]];
        for (ply_index, ply) in plies.iter().enumerate() {
            let q = ply.as_fem().transformed_stiffness();
            let global_stress = std::array::from_fn(|row| {
                (0..3)
                    .map(|column| q[row][column] * strain[column])
                    .sum::<f64>()
            });
            let local_stress = global_to_local_stress(global_stress, ply.angle_rad);
            let strength = ply.strength.as_fem();
            let tsai_hill_index = strength.tsai_hill_index(local_stress);
            let max_stress_index = strength.max_stress_index(local_stress);
            if tsai_hill_index > governing[ply_index].tsai_hill_index {
                governing[ply_index] = CompositePlyFailureResult {
                    ply_index,
                    angle_rad: ply.angle_rad,
                    element_index,
                    global_stress_pa: global_stress,
                    local_stress_pa: local_stress,
                    tsai_hill_index,
                    max_stress_index,
                };
            }
        }
    }
    governing
}

fn triangle_shape_gradients(mesh: &Mesh, nodes: &[usize]) -> [[f64; 2]; 3] {
    let p0 = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let determinant = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    [
        [(p1.y - p2.y) / determinant, (p2.x - p1.x) / determinant],
        [(p2.y - p0.y) / determinant, (p0.x - p2.x) / determinant],
        [(p0.y - p1.y) / determinant, (p1.x - p0.x) / determinant],
    ]
}

fn global_to_local_stress(global: [f64; 3], angle: f64) -> [f64; 3] {
    let m = angle.cos();
    let n = angle.sin();
    [
        m * m * global[0] + n * n * global[1] + 2.0 * m * n * global[2],
        n * n * global[0] + m * m * global[1] - 2.0 * m * n * global[2],
        -m * n * global[0] + m * n * global[1] + (m * m - n * n) * global[2],
    ]
}

fn solve_3x3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let mut augmented = [[0.0; 4]; 3];
    for row in 0..3 {
        augmented[row][..3].copy_from_slice(&matrix[row]);
        augmented[row][3] = rhs[row];
    }
    for pivot in 0..3 {
        let pivot_row = (pivot..3).max_by(|&lhs, &rhs| {
            augmented[lhs][pivot]
                .abs()
                .total_cmp(&augmented[rhs][pivot].abs())
        })?;
        if augmented[pivot_row][pivot].abs() <= f64::EPSILON {
            return None;
        }
        augmented.swap(pivot, pivot_row);
        let scale = augmented[pivot][pivot];
        for value in augmented[pivot].iter_mut().skip(pivot) {
            *value /= scale;
        }
        let pivot_values = augmented[pivot];
        for (row, row_values) in augmented.iter_mut().enumerate() {
            if row == pivot {
                continue;
            }
            let factor = row_values[pivot];
            for (value, pivot_value) in row_values
                .iter_mut()
                .skip(pivot)
                .zip(pivot_values.iter().skip(pivot))
            {
                *value -= factor * pivot_value;
            }
        }
    }
    Some([augmented[0][3], augmented[1][3], augmented[2][3]])
}

fn component_force_2(vector: &[f64]) -> [f64; 2] {
    vector
        .as_chunks::<2>()
        .0
        .iter()
        .fold([0.0; 2], |mut total, chunk| {
            total[0] += chunk[0];
            total[1] += chunk[1];
            total
        })
}

fn reaction_force_2(internal: &[f64], external: &[f64], fixed_dofs: &[usize]) -> [f64; 2] {
    fixed_dofs.iter().fold([0.0; 2], |mut total, &dof| {
        total[dof % 2] += internal[dof] - external[dof];
        total
    })
}

fn add2(lhs: [f64; 2], rhs: [f64; 2]) -> [f64; 2] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1]]
}

fn sub2(lhs: [f64; 2], rhs: [f64; 2]) -> [f64; 2] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1]]
}

fn norm2(value: [f64; 2]) -> f64 {
    value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_study() -> CompositeLaminateStudy {
        let mut study = CompositeLaminateStudy::introductory();
        study.geometry.nx = 8;
        study.geometry.ny = 2;
        study
    }

    #[test]
    fn composite_static_matches_constant_strain_and_reports_failure_indices() {
        let result = run_composite_laminate_static(&small_study()).unwrap();

        assert!(result.converged);
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::CompositeLaminate
        );
        assert!(result.analytical_relative_error < 1.0e-8);
        assert!(result.load_integration_relative_error < 1.0e-10);
        assert!(result.free_equilibrium_residual_n < 1.0e-6);
        assert!(result.force_balance_residual_n < 1.0e-6);
        assert!(result.energy_balance_relative_error < 1.0e-10);
        assert_eq!(result.ply_failure.len(), 1);
        assert!(result.ply_failure[0].global_stress_pa[0] > 0.0);
        assert!(result.max_tsai_hill_index > 0.0);
        assert!(result.max_tsai_hill_index < 1.0);
        assert!(result.max_stress_index < 1.0);
    }

    #[test]
    fn symmetric_and_unsymmetric_laminates_have_distinct_b_coupling() {
        let mut study = small_study();
        let ply = study.plies[0];
        study.plies = vec![
            ply,
            CompositePly {
                angle_rad: std::f64::consts::FRAC_PI_2,
                ..ply
            },
        ];
        assert!(run_composite_laminate_static(&study).is_err());

        study.plies = vec![
            ply,
            CompositePly {
                angle_rad: std::f64::consts::FRAC_PI_2,
                ..ply
            },
            CompositePly {
                angle_rad: std::f64::consts::FRAC_PI_2,
                ..ply
            },
            ply,
        ];
        let result = run_composite_laminate_static(&study).unwrap();
        assert!(
            result
                .coupling_b
                .iter()
                .flat_map(|row| row.iter())
                .all(|value| value.abs() < 1.0e-8)
        );
    }

    #[test]
    fn composite_rejects_invalid_mesh_and_empty_layup() {
        let mut study = small_study();
        study.geometry.ny = 3;
        assert!(matches!(
            run_composite_laminate_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "composite.mesh_density",
                ..
            })
        ));

        let mut study = small_study();
        study.plies.clear();
        assert!(matches!(
            run_composite_laminate_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "composite.plies",
                ..
            })
        ));
    }
}
