//! Reduced-order structural solves with an explicit residual-based enrichment gate.
//!
//! This module deliberately keeps the scope narrow: a linear Euler/Timoshenko
//! beam assembled by the existing FEM beam elements.  Modal vectors from a
//! reference design form the recycled Galerkin space.  A nearby Young's
//! modulus query is solved in that space and, when its exact full-order
//! residual is too large, the full-order displacement is orthogonalised into
//! the space before the query is solved again.  The result therefore reports
//! whether the reduced basis was sufficient instead of presenting a modal
//! estimate as an unqualified FEM answer.

use serde::{Deserialize, Serialize};
use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection, assemble_beam_stiffness};

use crate::{
    Beam2DStudy, StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption,
};

/// Configuration for a modal-subspace structural query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralRomStudy {
    pub name: String,
    pub base: Beam2DStudy,
    /// Number of reference modes used to seed the recycled Galerkin space.
    pub reduced_modes: usize,
    /// Young's modulus multiplier for the nearby query design.
    pub query_youngs_modulus_scale: f64,
    /// Relative full-order residual required for accepting the reduced solve.
    pub residual_tolerance: f64,
}

impl Default for StructuralRomStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl StructuralRomStudy {
    /// Small deterministic beam query suitable for regression and examples.
    pub fn introductory() -> Self {
        let mut base = Beam2DStudy::introductory();
        base.geometry.elements = 4;
        base.modal_modes = 2;
        Self {
            name: "beam-static-modal-rom".to_string(),
            base,
            reduced_modes: 2,
            query_youngs_modulus_scale: 1.05,
            residual_tolerance: 1.0e-8,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.reduced_modes == 0 || self.reduced_modes > self.base.geometry.elements * 3 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "structural_rom.reduced_modes",
                value: self.reduced_modes.to_string(),
            });
        }
        if !self.query_youngs_modulus_scale.is_finite() || self.query_youngs_modulus_scale <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "structural_rom.query_youngs_modulus_scale",
                value: self.query_youngs_modulus_scale.to_string(),
            });
        }
        if !self.residual_tolerance.is_finite() || self.residual_tolerance <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "structural_rom.residual_tolerance",
                value: self.residual_tolerance.to_string(),
            });
        }
        Ok(())
    }
}

/// Auditable result from the modal-subspace structural ROM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralRomResult {
    pub study: StructuralRomStudy,
    pub full_order_dofs: usize,
    pub reference_basis_dimension: usize,
    pub final_basis_dimension: usize,
    pub recycled_basis: bool,
    pub enriched_with_full_order_snapshot: bool,
    pub query_youngs_modulus_scale: f64,
    pub reduced_tip_deflection_m: f64,
    pub full_order_tip_deflection_m: f64,
    pub relative_tip_error: f64,
    pub reduced_relative_residual: f64,
    pub full_order_relative_residual: f64,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run a beam static query through a recycled modal Galerkin space.
pub fn run_structural_rom(
    study: &StructuralRomStudy,
) -> Result<StructuralRomResult, StructuralMechanicsError> {
    study.validate()?;

    let mut reference = study.base.clone();
    reference.modal_modes = study.reduced_modes;
    let modal = crate::run_beam2d_modal(&reference)?;
    let (reference_stiffness, free_dofs) = reduced_stiffness(&reference, 1.0);
    let full_order_dofs = reference_stiffness.len();

    let mut basis = Vec::with_capacity(study.reduced_modes + 1);
    for mode in &modal.modes {
        let vector = mode
            .iter()
            .flat_map(|node| node.iter().copied())
            .skip(3)
            .collect::<Vec<_>>();
        append_orthonormal(&mut basis, vector);
    }
    let reference_basis_dimension = basis.len();
    if reference_basis_dimension == 0 {
        return Err(StructuralMechanicsError::SolverFailure {
            stage: "structural ROM basis",
            message: "reference modal solve returned an empty basis".to_string(),
        });
    }

    let mut query = study.base.clone();
    query.material.youngs_modulus_pa *= study.query_youngs_modulus_scale;
    let full_order = crate::run_beam2d_static(&query)?;
    let (query_stiffness, query_free_dofs) = reduced_stiffness(&query, 1.0);
    debug_assert_eq!(free_dofs, query_free_dofs);
    let rhs = query_rhs(query.geometry.elements, query.vertical_tip_force_n);
    let full_order_vector = full_order
        .displacement
        .iter()
        .flat_map(|node| node.iter().copied())
        .skip(3)
        .collect::<Vec<_>>();
    let full_order_relative_residual =
        relative_residual(&query_stiffness, &full_order_vector, &rhs);

    let mut reduced_vector = galerkin_solve(&query_stiffness, &rhs, &basis)?;
    let mut reduced_relative_residual = relative_residual(&query_stiffness, &reduced_vector, &rhs);
    let mut enriched_with_full_order_snapshot = false;
    if reduced_relative_residual > study.residual_tolerance {
        append_orthonormal(&mut basis, full_order_vector.clone());
        enriched_with_full_order_snapshot = true;
        reduced_vector = galerkin_solve(&query_stiffness, &rhs, &basis)?;
        reduced_relative_residual = relative_residual(&query_stiffness, &reduced_vector, &rhs);
    }

    let reduced_tip_index = query.geometry.elements * 3 - 2;
    let reduced_tip_deflection_m =
        reduced_vector
            .get(reduced_tip_index)
            .copied()
            .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
                reason: "structural ROM tip DOF is outside the reduced free system".to_string(),
            })?;
    let relative_tip_error = (reduced_tip_deflection_m - full_order.tip_deflection_m).abs()
        / full_order.tip_deflection_m.abs().max(1.0e-30);
    let converged = full_order.converged
        && full_order_relative_residual.is_finite()
        && reduced_relative_residual.is_finite()
        && reduced_relative_residual <= study.residual_tolerance;

    Ok(StructuralRomResult {
        study: study.clone(),
        full_order_dofs,
        reference_basis_dimension,
        final_basis_dimension: basis.len(),
        recycled_basis: reference_basis_dimension > 0,
        enriched_with_full_order_snapshot,
        query_youngs_modulus_scale: study.query_youngs_modulus_scale,
        reduced_tip_deflection_m,
        full_order_tip_deflection_m: full_order.tip_deflection_m,
        relative_tip_error,
        reduced_relative_residual,
        full_order_relative_residual,
        converged,
        model_assumption: StructuralModelAssumption::ReducedOrderBeam,
        execution: StructuralExecutionProvenance::cpu_rom_reference(),
    })
}

fn beam_elements(study: &Beam2DStudy, youngs_modulus_scale: f64) -> Vec<BeamElement2D> {
    let geometry = study.geometry;
    let youngs_modulus_pa = study.material.youngs_modulus_pa * youngs_modulus_scale;
    let section = if study.timoshenko {
        BeamSection::timoshenko(
            youngs_modulus_pa,
            geometry.second_moment_m4(),
            geometry.area_m2(),
            study.material.density_kg_m3,
            study.material.poisson_ratio,
            crate::beam::RECTANGULAR_SHEAR_CORRECTION,
        )
    } else {
        BeamSection::rectangular(
            youngs_modulus_pa,
            geometry.width_m,
            geometry.height_m,
            study.material.density_kg_m3,
        )
    };
    let dx = geometry.length_m / geometry.elements as f64;
    (0..geometry.elements)
        .map(|index| {
            BeamElement2D::new(
                [index, index + 1],
                index as f64 * dx,
                0.0,
                (index + 1) as f64 * dx,
                0.0,
                section,
            )
        })
        .collect()
}

fn reduced_stiffness(
    study: &Beam2DStudy,
    youngs_modulus_scale: f64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let dofs = (study.geometry.elements + 1) * 3;
    let triplet = assemble_beam_stiffness(&beam_elements(study, youngs_modulus_scale), dofs);
    let free_dofs = (3..dofs).collect::<Vec<_>>();
    let mut matrix = vec![vec![0.0; free_dofs.len()]; free_dofs.len()];
    for ((&row, &col), &value) in triplet.rows.iter().zip(&triplet.cols).zip(&triplet.values) {
        if row >= 3 && col >= 3 {
            matrix[row - 3][col - 3] += value;
        }
    }
    (matrix, free_dofs)
}

fn query_rhs(elements: usize, force_n: f64) -> Vec<f64> {
    let mut rhs = vec![0.0; elements * 3];
    rhs[elements * 3 - 2] = force_n;
    rhs
}

fn append_orthonormal(basis: &mut Vec<Vec<f64>>, mut vector: Vec<f64>) -> bool {
    for existing in basis.iter() {
        let projection = dot(existing, &vector);
        for (value, &component) in vector.iter_mut().zip(existing) {
            *value -= projection * component;
        }
    }
    let norm = dot(&vector, &vector).sqrt();
    if !norm.is_finite() || norm <= 1.0e-12 {
        return false;
    }
    for value in &mut vector {
        *value /= norm;
    }
    basis.push(vector);
    true
}

fn galerkin_solve(
    stiffness: &[Vec<f64>],
    rhs: &[f64],
    basis: &[Vec<f64>],
) -> Result<Vec<f64>, StructuralMechanicsError> {
    let reduced_size = basis.len();
    let mut reduced_stiffness = vec![vec![0.0; reduced_size]; reduced_size];
    let mut reduced_rhs = vec![0.0; reduced_size];
    let mut applied = Vec::with_capacity(reduced_size);
    for vector in basis {
        applied.push(matvec(stiffness, vector));
        reduced_rhs[applied.len() - 1] = dot(vector, rhs);
    }
    for row in 0..reduced_size {
        for col in 0..reduced_size {
            reduced_stiffness[row][col] = dot(&basis[row], &applied[col]);
        }
    }
    let coefficients = dense_solve(&mut reduced_stiffness, &mut reduced_rhs)?;
    let mut solution = vec![0.0; stiffness.len()];
    for (coefficient, vector) in coefficients.iter().zip(basis) {
        for (value, &component) in solution.iter_mut().zip(vector) {
            *value += coefficient * component;
        }
    }
    Ok(solution)
}

fn dense_solve(
    matrix: &mut [Vec<f64>],
    rhs: &mut [f64],
) -> Result<Vec<f64>, StructuralMechanicsError> {
    let n = rhs.len();
    for pivot in 0..n {
        let (pivot_row, pivot_value) = (pivot..n)
            .map(|row| (row, matrix[row][pivot].abs()))
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .unwrap_or((pivot, 0.0));
        if !pivot_value.is_finite() || pivot_value <= 1.0e-14 {
            return Err(StructuralMechanicsError::SolverFailure {
                stage: "structural ROM reduced solve",
                message: format!("singular reduced stiffness pivot {pivot}"),
            });
        }
        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);
        let diagonal = matrix[pivot][pivot];
        let pivot_tail = matrix[pivot][(pivot + 1)..n].to_vec();
        for row in (pivot + 1)..n {
            let factor = matrix[row][pivot] / diagonal;
            matrix[row][pivot] = 0.0;
            for (offset, value) in matrix[row][(pivot + 1)..n].iter_mut().enumerate() {
                *value -= factor * pivot_tail[offset];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let tail = (row + 1..n)
            .map(|col| matrix[row][col] * solution[col])
            .sum::<f64>();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    Ok(solution)
}

fn matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
        .collect()
}

fn relative_residual(matrix: &[Vec<f64>], solution: &[f64], rhs: &[f64]) -> f64 {
    let residual = matvec(matrix, solution)
        .iter()
        .zip(rhs)
        .map(|(lhs, rhs)| lhs - rhs)
        .collect::<Vec<_>>();
    dot(&residual, &residual).sqrt() / dot(rhs, rhs).sqrt().max(1.0e-30)
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rom_recycles_and_enriches_with_a_residual_gate() {
        let mut study = StructuralRomStudy::introductory();
        study.base.geometry.elements = 4;
        study.reduced_modes = 1;
        study.base.modal_modes = 1;
        let result = run_structural_rom(&study).expect("ROM should converge");

        assert!(result.recycled_basis);
        assert_eq!(result.reference_basis_dimension, 1);
        assert!(result.enriched_with_full_order_snapshot);
        assert!(result.final_basis_dimension >= result.reference_basis_dimension);
        assert!(result.reduced_relative_residual <= study.residual_tolerance);
        assert!(result.full_order_relative_residual < 1.0e-8);
        assert!(result.relative_tip_error < 1.0e-7);
        assert!(result.converged);
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::ReducedOrderBeam
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.structural-rom-beam.v1"
        );
    }

    #[test]
    fn rom_rejects_nonpositive_query_scale() {
        let mut study = StructuralRomStudy::introductory();
        study.query_youngs_modulus_scale = 0.0;
        assert!(matches!(
            run_structural_rom(&study),
            Err(StructuralMechanicsError::InvalidConfiguration { .. })
        ));
    }
}
