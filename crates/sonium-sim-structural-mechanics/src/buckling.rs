//! Linear Euler buckling evidence for the assembled beam family.
//!
//! This is a pre-buckling eigenproblem: the geometric stiffness is the
//! unit-compression tangent of an Euler–Bernoulli beam, and the reported load
//! factor is compared with Euler's closed form.  Post-buckling equilibrium and
//! imperfection sensitivity are intentionally outside this result type.

use super::{StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};
use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection, assemble_beam_stiffness};
use sonium_fem::physics::solid::eigenmodes::{
    build_linear_buckling_problem, critical_loads, euler_column_buckling_load,
};
use sonium_solvers::{LobpcgConfig, TripletMatrix, lobpcg_generalized};

/// Idealized end conditions supported by the linear Euler buckling family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BucklingEndCondition {
    /// Pinned translation with free rotation at both ends: `K = 1`.
    PinnedPinned,
    /// Clamped at the first end and free at the second: `K = 2`.
    Cantilever,
}

impl BucklingEndCondition {
    fn effective_length_factor(self) -> f64 {
        match self {
            Self::PinnedPinned => 1.0,
            Self::Cantilever => 2.0,
        }
    }
}

/// Beam geometry and mesh density for the linear buckling problem.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BucklingGeometry {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub elements: usize,
}

impl Default for BucklingGeometry {
    fn default() -> Self {
        Self {
            length_m: 1.0,
            width_m: 0.02,
            height_m: 0.02,
            elements: 12,
        }
    }
}

/// Linear elastic Euler-column buckling study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucklingStudy {
    pub name: String,
    pub geometry: BucklingGeometry,
    pub youngs_modulus_pa: f64,
    pub end_condition: BucklingEndCondition,
}

impl Default for BucklingStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl BucklingStudy {
    pub fn introductory() -> Self {
        Self {
            name: "linear-euler-buckling-column".to_string(),
            geometry: BucklingGeometry::default(),
            youngs_modulus_pa: 200.0e9,
            end_condition: BucklingEndCondition::PinnedPinned,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("buckling.length_m", self.geometry.length_m),
            ("buckling.width_m", self.geometry.width_m),
            ("buckling.height_m", self.geometry.height_m),
            ("buckling.youngs_modulus_pa", self.youngs_modulus_pa),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.geometry.elements < 2 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "buckling.elements",
                value: self.geometry.elements.to_string(),
            });
        }
        Ok(())
    }

    fn second_moment_m4(&self) -> f64 {
        self.geometry.width_m * self.geometry.height_m.powi(3) / 12.0
    }

    /// Return the closed-form Euler critical load for this study.
    ///
    /// This is an application-level reference quantity, not the result of
    /// the assembled eigenproblem.  Keeping the validation here makes the
    /// reference safe for callers that construct studies from untrusted
    /// configuration data.
    pub fn analytical_euler_load_n(&self) -> Result<f64, StructuralMechanicsError> {
        self.validate()?;
        Ok(euler_column_buckling_load(
            self.youngs_modulus_pa,
            self.second_moment_m4(),
            self.geometry.length_m,
            self.end_condition.effective_length_factor(),
        ))
    }
}

/// Linear buckling eigenvalue, analytical comparison, and mode provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucklingResult {
    pub elements: usize,
    pub dofs: usize,
    pub eigenvalue_n: f64,
    pub analytical_euler_load_n: f64,
    pub analytical_relative_error: f64,
    pub solver_iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
    pub end_condition: BucklingEndCondition,
    /// Reconstructed `[transverse displacement, rotation]` mode per node.
    pub mode_shape: Vec<[f64; 2]>,
    pub model_assumption: StructuralModelAssumption,
}

/// Solve the assembled linearized beam buckling eigenproblem.
pub fn run_linear_buckling(
    study: &BucklingStudy,
) -> Result<BucklingResult, StructuralMechanicsError> {
    study.validate()?;
    let geometry = study.geometry;
    let section = BeamSection::rectangular(
        study.youngs_modulus_pa,
        geometry.width_m,
        geometry.height_m,
        0.0,
    );
    let elements = (0..geometry.elements)
        .map(|element| {
            let x0 = geometry.length_m * element as f64 / geometry.elements as f64;
            let x1 = geometry.length_m * (element + 1) as f64 / geometry.elements as f64;
            BeamElement2D::new([element, element + 1], x0, 0.0, x1, 0.0, section)
        })
        .collect::<Vec<_>>();
    let dofs = 3 * (geometry.elements + 1);
    let elastic = assemble_beam_stiffness(&elements, dofs);
    let geometric = assemble_geometric_stiffness(&elements, dofs);
    let constrained = buckling_constraints(geometry.elements, study.end_condition);
    let problem = build_linear_buckling_problem(&elastic, &geometric, &constrained);
    let config = LobpcgConfig {
        num_eigenvalues: 1,
        max_iterations: 2_000,
        tolerance: 1.0e-9,
        ..LobpcgConfig::default()
    };
    let solved = lobpcg_generalized(&problem.stiffness_like, &problem.mass_like, &config).map_err(
        |error| StructuralMechanicsError::SolverFailure {
            stage: "linear buckling",
            message: error.to_string(),
        },
    )?;
    if !solved.converged {
        return Err(StructuralMechanicsError::ModalNonConvergence {
            iterations: solved.iterations,
            residual: solved.residual_norms.iter().copied().fold(0.0, f64::max),
        });
    }
    let critical = critical_loads(&solved.eigenvalues);
    let eigenvalue_n =
        critical
            .first()
            .copied()
            .ok_or_else(|| StructuralMechanicsError::SolverFailure {
                stage: "linear buckling",
                message: "eigensolver returned no critical loads".to_string(),
            })?;
    if !eigenvalue_n.is_finite() || eigenvalue_n <= 0.0 {
        return Err(StructuralMechanicsError::UnstableModel {
            eigenvalue: eigenvalue_n,
        });
    }
    let analytical_euler_load_n = study.analytical_euler_load_n()?;
    let analytical_relative_error =
        (eigenvalue_n - analytical_euler_load_n).abs() / analytical_euler_load_n;
    let free_dofs = free_dof_map(dofs, &constrained);
    let vector = solved.eigenvectors.column_to_vec(0);
    let mut mode_shape = vec![[0.0; 2]; geometry.elements + 1];
    for (reduced, &full_dof) in free_dofs.iter().enumerate() {
        match full_dof % 3 {
            1 => mode_shape[full_dof / 3][0] = vector[reduced],
            2 => mode_shape[full_dof / 3][1] = vector[reduced],
            _ => {}
        }
    }
    Ok(BucklingResult {
        elements: geometry.elements,
        dofs,
        eigenvalue_n,
        analytical_euler_load_n,
        analytical_relative_error,
        solver_iterations: solved.iterations,
        residual_norm: solved.residual_norms[0],
        converged: solved.converged,
        end_condition: study.end_condition,
        mode_shape,
        model_assumption: StructuralModelAssumption::Beam,
    })
}

fn assemble_geometric_stiffness(elements: &[BeamElement2D], dofs: usize) -> TripletMatrix<f64> {
    let mut matrix = TripletMatrix::new(dofs, dofs);
    for element in elements {
        let length = element.length;
        let scale = 1.0 / (30.0 * length);
        let l = length;
        let local = [
            36.0 * scale,
            3.0 * l * scale,
            -36.0 * scale,
            3.0 * l * scale,
            3.0 * l * scale,
            4.0 * l * l * scale,
            -3.0 * l * scale,
            -l * l * scale,
            -36.0 * scale,
            -3.0 * l * scale,
            36.0 * scale,
            -3.0 * l * scale,
            3.0 * l * scale,
            -l * l * scale,
            -3.0 * l * scale,
            4.0 * l * l * scale,
        ];
        let global = [
            3 * element.nodes[0] + 1,
            3 * element.nodes[0] + 2,
            3 * element.nodes[1] + 1,
            3 * element.nodes[1] + 2,
        ];
        for row in 0..4 {
            for column in 0..4 {
                matrix.push(global[row], global[column], local[row * 4 + column]);
            }
        }
    }
    matrix
}

fn buckling_constraints(elements: usize, end_condition: BucklingEndCondition) -> Vec<usize> {
    let mut constrained = (0..=elements).map(|node| 3 * node).collect::<Vec<_>>();
    match end_condition {
        BucklingEndCondition::PinnedPinned => {
            constrained.extend([1, 3 * elements + 1]);
        }
        BucklingEndCondition::Cantilever => {
            constrained.extend([1, 2]);
        }
    }
    constrained
}

fn free_dof_map(total_dofs: usize, constrained: &[usize]) -> Vec<usize> {
    let mut is_constrained = vec![false; total_dofs];
    for &dof in constrained {
        is_constrained[dof] = true;
    }
    (0..total_dofs)
        .filter(|dof| !is_constrained[*dof])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_study(end_condition: BucklingEndCondition) -> BucklingStudy {
        let mut study = BucklingStudy::introductory();
        study.geometry.elements = 8;
        study.end_condition = end_condition;
        study
    }

    #[test]
    fn pinned_buckling_matches_euler_load_and_has_mode() {
        let result = run_linear_buckling(&small_study(BucklingEndCondition::PinnedPinned)).unwrap();

        assert!(result.converged);
        assert_eq!(result.model_assumption, StructuralModelAssumption::Beam);
        assert!(result.eigenvalue_n > 0.0);
        assert!(result.analytical_relative_error < 0.02);
        assert!(result.mode_shape.iter().any(|value| value[0].abs() > 0.0));
        assert!(result.mode_shape.first().unwrap()[0].abs() < 1.0e-12);
        assert!(result.mode_shape.last().unwrap()[0].abs() < 1.0e-12);
    }

    #[test]
    fn cantilever_buckling_uses_effective_length_factor() {
        let result = run_linear_buckling(&small_study(BucklingEndCondition::Cantilever)).unwrap();
        assert!(result.converged);
        assert!(result.analytical_relative_error < 0.03);
        let reference = BucklingStudy::introductory();
        let pinned_load = euler_column_buckling_load(
            reference.youngs_modulus_pa,
            reference.second_moment_m4(),
            reference.geometry.length_m,
            1.0,
        );
        assert!(result.eigenvalue_n < 0.4 * pinned_load);
    }

    #[test]
    fn buckling_rejects_degenerate_mesh_and_material() {
        let mut study = small_study(BucklingEndCondition::PinnedPinned);
        study.geometry.elements = 1;
        assert!(matches!(
            run_linear_buckling(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "buckling.elements",
                ..
            })
        ));

        let mut study = small_study(BucklingEndCondition::PinnedPinned);
        study.youngs_modulus_pa = 0.0;
        assert!(matches!(
            run_linear_buckling(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "buckling.youngs_modulus_pa",
                ..
            })
        ));
    }
}
