//! Validated orchestration for multiple linear structural load cases.
//!
//! This module deliberately composes the concrete bracket study instead of
//! widening the introductory solver's model assumptions. It provides the
//! first reusable study-family boundary: every load case is solved with the
//! same geometry, material, boundary markers, and validation gates, then
//! reduced to a case result and a conservative response envelope.

use super::{
    BracketStudy, StaticBracketResult, StructuralMechanicsError, apply_tip_edge_load,
    component_force, dot, free_residual_norm, nodal_displacement, nodes_on_marker, reaction_force,
    triplet_matvec,
};
use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet};
use sonium_fem::physics::solid::elasticity::ElasticityForm;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{
    CsrMatrix, MatrixClass, Ordering, SparseDirectConfig, SparseDirectMethod, factorize_sparse,
};
use std::collections::HashSet;

/// One independently solved tip-load case in a [`MultiLoadStudy`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralLoadCase {
    /// Stable user-facing identifier used for critical-case provenance.
    pub name: String,
    /// Downward or upward total force applied on the loaded edge.
    pub vertical_tip_force_n: f64,
    /// Signed axial load applied on the loaded edge and used for the
    /// initial-stress geometric stiffness in modal/harmonic follow-up steps.
    pub axial_prestress_force_n: f64,
}

impl StructuralLoadCase {
    /// Construct a load case from the two load components supported by the
    /// validated introductory bracket model.
    pub fn new(
        name: impl Into<String>,
        vertical_tip_force_n: f64,
        axial_prestress_force_n: f64,
    ) -> Self {
        Self {
            name: name.into(),
            vertical_tip_force_n,
            axial_prestress_force_n,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.name.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "load_case_name",
                value: "empty".to_string(),
            });
        }
        if !self.vertical_tip_force_n.is_finite() || self.vertical_tip_force_n.abs() <= f64::EPSILON
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "load_case.vertical_tip_force_n",
                value: self.vertical_tip_force_n.to_string(),
            });
        }
        if !self.axial_prestress_force_n.is_finite() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "load_case.axial_prestress_force_n",
                value: self.axial_prestress_force_n.to_string(),
            });
        }
        Ok(())
    }

    fn apply_to(&self, base: &BracketStudy) -> BracketStudy {
        let mut study = base.clone();
        study.name = format!("{}::{}", base.name, self.name);
        study.loads.vertical_tip_force_n = self.vertical_tip_force_n;
        study.loads.axial_prestress_force_n = self.axial_prestress_force_n;
        study
    }
}

/// A family of load cases sharing one validated model and mesh definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiLoadStudy {
    pub base: BracketStudy,
    pub load_cases: Vec<StructuralLoadCase>,
}

impl MultiLoadStudy {
    pub fn new(base: BracketStudy, load_cases: Vec<StructuralLoadCase>) -> Self {
        Self { base, load_cases }
    }

    /// Create a two-case smoke family useful for examples and QA.
    pub fn introductory(base: BracketStudy) -> Self {
        let loads = base.loads;
        Self::new(
            base,
            vec![
                StructuralLoadCase::new(
                    "nominal",
                    loads.vertical_tip_force_n,
                    loads.axial_prestress_force_n,
                ),
                StructuralLoadCase::new(
                    "vertical-reversal",
                    -loads.vertical_tip_force_n,
                    loads.axial_prestress_force_n,
                ),
            ],
        )
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.load_cases.is_empty() {
            return Err(StructuralMechanicsError::EmptyLoadCases);
        }
        let mut names = HashSet::with_capacity(self.load_cases.len());
        for load_case in &self.load_cases {
            load_case.validate()?;
            if !names.insert(&load_case.name) {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "load_case_name",
                    value: format!("duplicate: {}", load_case.name),
                });
            }
        }
        Ok(())
    }
}

/// A validated static result associated with its load-case identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadCaseResult {
    pub load_case: StructuralLoadCase,
    pub static_result: StaticBracketResult,
}

/// Conservative response envelope across all solved load cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralEnvelope {
    /// Per-node displacement magnitude envelope in the shared mesh order.
    pub displacement_envelope_m: Vec<f64>,
    pub max_displacement_m: f64,
    pub critical_displacement_case: String,
    pub max_von_mises_pa: f64,
    pub critical_stress_case: String,
    pub max_strain_energy_j: f64,
    pub max_force_balance_residual_n: f64,
    pub max_free_equilibrium_residual_n: f64,
}

/// Results for all load cases and their conservative response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiLoadResult {
    pub study: MultiLoadStudy,
    pub cases: Vec<LoadCaseResult>,
    pub envelope: StructuralEnvelope,
}

/// A load-case result produced by one reused sparse factorization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReusedLoadCaseResult {
    pub load_case: StructuralLoadCase,
    pub displacement: Vec<[f64; 2]>,
    pub max_displacement_m: f64,
    pub average_tip_deflection_m: f64,
    pub strain_energy_j: f64,
    pub solver_residual: f64,
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub converged: bool,
}

/// Multi-load evidence that assembles and factorizes one common FEM operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReusedMultiLoadResult {
    pub study: MultiLoadStudy,
    pub cases: Vec<ReusedLoadCaseResult>,
    pub assembled_matrix_count: usize,
    pub factorization_count: usize,
    pub factor_memory_bytes: usize,
    pub reused_factorization: bool,
}

/// Solve all linear load cases with one shared sparse Cholesky factorization.
///
/// The load vectors and boundary-conditioned right-hand sides vary per case,
/// while geometry/material/constraints are assembled and factorized exactly
/// once. This is the reusable linear-study path needed by multi-load and
/// deterministic uncertainty workflows.
pub fn run_multi_load_study_reused(
    study: &MultiLoadStudy,
) -> Result<ReusedMultiLoadResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.base.mesh();
    super::validate_mesh(&mesh, &study.base)?;
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let dofs = space.num_global_dofs(&mesh);
    let form = ElasticityForm::new_2d(
        study.base.material.as_fem(),
        study.base.plane_model.as_fem(),
    );
    let matrices = super::assemble_structural_matrices(
        &study.base,
        &mesh,
        &space,
        &form,
        PolynomialDegree::P1,
    );
    let unconstrained_stiffness = matrices.stiffness;
    let mut zero_system = FemSystem::new(unconstrained_stiffness.clone(), vec![0.0; dofs]);
    try_apply_homogeneous_dirichlet(
        &mut zero_system,
        &mesh,
        &space,
        study.base.loads.fixed_boundary_marker,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "reused multi-load boundary",
        message: error.to_string(),
    })?;
    let constrained_stiffness: CsrMatrix<f64> = zero_system.matrix.to_csr();
    let factorization = factorize_sparse(
        &constrained_stiffness,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..SparseDirectConfig::default()
        },
        None,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "reused multi-load factorization",
        message: error.to_string(),
    })?;
    let factor_memory_bytes = factorization.factor_memory_bytes();
    let fixed_dofs = space.boundary_dofs(&mesh, study.base.loads.fixed_boundary_marker);
    let tip_nodes = nodes_on_marker(&mesh, study.base.loads.loaded_boundary_marker);
    if tip_nodes.is_empty() {
        return Err(StructuralMechanicsError::MissingLoadedBoundary {
            marker: study.base.loads.loaded_boundary_marker,
        });
    }

    let mut cases = Vec::with_capacity(study.load_cases.len());
    for load_case in &study.load_cases {
        let case_study = load_case.apply_to(&study.base);
        let mut external_rhs = vec![0.0; dofs];
        apply_tip_edge_load(&mesh, &case_study, &mut external_rhs);
        let mut system = FemSystem::new(unconstrained_stiffness.clone(), external_rhs.clone());
        try_apply_homogeneous_dirichlet(
            &mut system,
            &mesh,
            &space,
            case_study.loads.fixed_boundary_marker,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "reused multi-load case boundary",
            message: error.to_string(),
        })?;
        let solution = factorization.solve(&system.rhs).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "reused multi-load solve",
                message: error.to_string(),
            }
        })?;
        let internal_force = triplet_matvec(&unconstrained_stiffness, &solution);
        let displacement = nodal_displacement(&solution, mesh.num_nodes());
        let max_displacement_m = displacement
            .iter()
            .map(|[ux, uy]| (ux * ux + uy * uy).sqrt())
            .fold(0.0, f64::max);
        let average_tip_deflection_m = tip_nodes
            .iter()
            .map(|&node| solution[node * 2 + 1])
            .sum::<f64>()
            / tip_nodes.len() as f64;
        let applied_force_n = component_force(&external_rhs);
        let reaction_force_n = reaction_force(&internal_force, &external_rhs, &fixed_dofs);
        let free_equilibrium_residual_n =
            free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
        let force_balance_residual_n = [
            applied_force_n[0] + reaction_force_n[0],
            applied_force_n[1] + reaction_force_n[1],
        ]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
        let solver_residual = csr_residual(&constrained_stiffness, &solution, &system.rhs);
        let strain_energy_j = 0.5 * dot(&solution, &internal_force);
        let force_scale = dot(&applied_force_n, &applied_force_n).sqrt().max(1.0);
        if solver_residual > 1.0e-8
            || free_equilibrium_residual_n > 1.0e-6 * force_scale
            || force_balance_residual_n > 1.0e-6 * force_scale
        {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "reused_multi_load_balance",
                value: solver_residual
                    .max(free_equilibrium_residual_n / force_scale)
                    .max(force_balance_residual_n / force_scale),
            });
        }
        cases.push(ReusedLoadCaseResult {
            load_case: load_case.clone(),
            displacement,
            max_displacement_m,
            average_tip_deflection_m,
            strain_energy_j,
            solver_residual,
            free_equilibrium_residual_n,
            force_balance_residual_n,
            converged: true,
        });
    }
    Ok(ReusedMultiLoadResult {
        study: study.clone(),
        cases,
        assembled_matrix_count: 1,
        factorization_count: 1,
        factor_memory_bytes,
        reused_factorization: study.load_cases.len() > 1,
    })
}

fn csr_residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    (0..matrix.num_rows)
        .map(|row| {
            let value = (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * solution[matrix.col_indices[index]])
                .sum::<f64>()
                - rhs[row];
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

/// Solve every load case with the same validated bracket model.
pub fn run_multi_load_study(
    study: &MultiLoadStudy,
) -> Result<MultiLoadResult, StructuralMechanicsError> {
    study.validate()?;

    let mut cases = Vec::with_capacity(study.load_cases.len());
    for load_case in &study.load_cases {
        let case_study = load_case.apply_to(&study.base);
        let static_result = super::run_static_bracket(&case_study)?;
        cases.push(LoadCaseResult {
            load_case: load_case.clone(),
            static_result,
        });
    }

    let envelope = build_envelope(&cases)?;
    Ok(MultiLoadResult {
        study: study.clone(),
        cases,
        envelope,
    })
}

fn build_envelope(
    cases: &[LoadCaseResult],
) -> Result<StructuralEnvelope, StructuralMechanicsError> {
    let first = cases
        .first()
        .ok_or(StructuralMechanicsError::EmptyLoadCases)?;
    let mut displacement_envelope_m = vec![0.0_f64; first.static_result.displacement.len()];
    let mut max_displacement_m = 0.0;
    let mut critical_displacement_case = first.load_case.name.clone();
    let mut max_von_mises_pa = 0.0;
    let mut critical_stress_case = first.load_case.name.clone();
    let mut max_strain_energy_j: f64 = 0.0;
    let mut max_force_balance_residual_n: f64 = 0.0;
    let mut max_free_equilibrium_residual_n: f64 = 0.0;

    for case in cases {
        if case.static_result.displacement.len() != displacement_envelope_m.len() {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "load cases produced different mesh sizes".to_string(),
            });
        }
        for (envelope, [ux, uy]) in displacement_envelope_m
            .iter_mut()
            .zip(&case.static_result.displacement)
        {
            *envelope = (*envelope).max((ux * ux + uy * uy).sqrt());
        }

        if case.static_result.max_displacement_m > max_displacement_m {
            max_displacement_m = case.static_result.max_displacement_m;
            critical_displacement_case = case.load_case.name.clone();
        }
        if case.static_result.max_von_mises_pa > max_von_mises_pa {
            max_von_mises_pa = case.static_result.max_von_mises_pa;
            critical_stress_case = case.load_case.name.clone();
        }
        max_strain_energy_j = max_strain_energy_j.max(case.static_result.strain_energy_j);
        max_force_balance_residual_n =
            max_force_balance_residual_n.max(case.static_result.force_balance_residual_n);
        max_free_equilibrium_residual_n =
            max_free_equilibrium_residual_n.max(case.static_result.free_equilibrium_residual_n);
    }

    Ok(StructuralEnvelope {
        displacement_envelope_m,
        max_displacement_m,
        critical_displacement_case,
        max_von_mises_pa,
        critical_stress_case,
        max_strain_energy_j,
        max_force_balance_residual_n,
        max_free_equilibrium_residual_n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_study() -> BracketStudy {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 8;
        study.geometry.ny = 2;
        study
    }

    #[test]
    fn constructors_keep_load_case_identity_and_base_model() {
        let base = small_study();
        let case = StructuralLoadCase::new("service", -120.0, 250.0);
        assert_eq!(case.name, "service");
        assert_eq!(case.vertical_tip_force_n, -120.0);
        assert_eq!(case.axial_prestress_force_n, 250.0);

        let study = MultiLoadStudy::introductory(base.clone());
        assert_eq!(study.base, base);
        assert_eq!(
            study
                .load_cases
                .iter()
                .map(|load_case| load_case.name.as_str())
                .collect::<Vec<_>>(),
            vec!["nominal", "vertical-reversal"]
        );
    }

    #[test]
    fn multi_load_result_tracks_critical_cases_and_node_envelope() {
        let base = small_study();
        let study = MultiLoadStudy::new(
            base.clone(),
            vec![
                StructuralLoadCase::new("light", -60.0, 0.0),
                StructuralLoadCase::new("heavy", -180.0, 0.0),
            ],
        );

        let result = run_multi_load_study(&study).unwrap();

        assert_eq!(result.cases.len(), 2);
        assert_eq!(result.envelope.critical_displacement_case, "heavy");
        assert_eq!(result.envelope.critical_stress_case, "heavy");
        assert!(result.envelope.max_displacement_m > 0.0);
        assert!(result.envelope.max_von_mises_pa > 0.0);
        assert_eq!(
            result.envelope.displacement_envelope_m.len(),
            result.cases[0].static_result.displacement.len()
        );
        assert!(
            result
                .envelope
                .displacement_envelope_m
                .iter()
                .any(|value| *value > 0.0)
        );
        assert_eq!(study.base, base);
    }

    #[test]
    fn reused_multi_load_factorization_preserves_balance_and_case_scaling() {
        let base = small_study();
        let study = MultiLoadStudy::new(
            base,
            vec![
                StructuralLoadCase::new("light", -60.0, 0.0),
                StructuralLoadCase::new("heavy", -180.0, 0.0),
            ],
        );

        let result = run_multi_load_study_reused(&study).unwrap();

        assert_eq!(result.cases.len(), 2);
        assert_eq!(result.assembled_matrix_count, 1);
        assert_eq!(result.factorization_count, 1);
        assert!(result.factor_memory_bytes > 0);
        assert!(result.reused_factorization);
        assert!(result.cases[1].max_displacement_m > result.cases[0].max_displacement_m);
        assert!(
            result
                .cases
                .iter()
                .all(|case| case.converged && case.solver_residual < 1.0e-8)
        );
        assert!(
            result
                .cases
                .iter()
                .all(|case| case.free_equilibrium_residual_n < 1.0e-6)
        );
    }

    #[test]
    fn empty_and_invalid_load_cases_fail_before_solving() {
        let base = small_study();
        assert_eq!(
            run_multi_load_study(&MultiLoadStudy::new(base.clone(), Vec::new())),
            Err(StructuralMechanicsError::EmptyLoadCases)
        );

        let study = MultiLoadStudy::new(base, vec![StructuralLoadCase::new("", -10.0, 0.0)]);
        assert_eq!(
            run_multi_load_study(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "load_case_name",
                value: "empty".to_string(),
            })
        );

        let duplicate = MultiLoadStudy::new(
            study.base,
            vec![
                StructuralLoadCase::new("same", -10.0, 0.0),
                StructuralLoadCase::new("same", -20.0, 0.0),
            ],
        );
        assert_eq!(
            run_multi_load_study(&duplicate),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "load_case_name",
                value: "duplicate: same".to_string(),
            })
        );
    }
}
