//! Executable 2D Euler-Bernoulli/Timoshenko beam study family.

use super::{
    DampingConvention, ModalNormalization, StructuralMaterial, StructuralMechanicsError,
    StructuralModelAssumption,
};
use serde::{Deserialize, Serialize};
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet_dofs};
use sonium_fem::physics::solid::beam::{
    BeamElement2D, BeamSection, assemble_beam_mass, assemble_beam_stiffness,
};
use sonium_solvers::{
    LinearOperator, LobpcgConfig, ShiftInvertConfig, lobpcg_generalized, lu_solve,
    shift_invert_lanczos_generalized,
};

/// Timoshenko shear-correction factor for a rectangular cross-section.
/// Correct only because [`Beam2DGeometry`] is rectangular (`I = bh^3/12`
/// below); any future I/T/box section needs its own coefficient threaded
/// through [`BeamSection::timoshenko`] instead of this constant.
pub const RECTANGULAR_SHEAR_CORRECTION: f64 = 5.0 / 6.0;

/// Beam span and mesh resolution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Beam2DGeometry {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub elements: usize,
}

impl Default for Beam2DGeometry {
    fn default() -> Self {
        Self {
            length_m: 0.18,
            width_m: 0.006,
            height_m: 0.036,
            elements: 12,
        }
    }
}

impl Beam2DGeometry {
    pub fn second_moment_m4(self) -> f64 {
        self.width_m * self.height_m.powi(3) / 12.0
    }

    pub fn area_m2(self) -> f64 {
        self.width_m * self.height_m
    }
}

/// Complete beam study configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Beam2DStudy {
    pub name: String,
    pub geometry: Beam2DGeometry,
    pub material: StructuralMaterial,
    pub vertical_tip_force_n: f64,
    pub timoshenko: bool,
    pub modal_modes: usize,
}

impl Beam2DStudy {
    pub fn introductory() -> Self {
        Self {
            name: "introductory-2d-beam".to_string(),
            geometry: Beam2DGeometry::default(),
            material: StructuralMaterial::default(),
            vertical_tip_force_n: -120.0,
            timoshenko: false,
            modal_modes: 3,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("beam2d.length_m", self.geometry.length_m),
            ("beam2d.width_m", self.geometry.width_m),
            ("beam2d.height_m", self.geometry.height_m),
            ("beam2d.youngs_modulus_pa", self.material.youngs_modulus_pa),
            ("beam2d.poisson_ratio", self.material.poisson_ratio),
            ("beam2d.density_kg_m3", self.material.density_kg_m3),
        ] {
            if !value.is_finite()
                || value <= 0.0
                || (field == "beam2d.poisson_ratio" && !(value > -1.0 && value < 0.5))
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.geometry.elements == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "beam2d.elements",
                value: "0".to_string(),
            });
        }
        // Zero load is legitimate (stiffness sanity, eigen-only, F=0
        // compliance baseline); only non-finite input is rejected.
        if !self.vertical_tip_force_n.is_finite() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "beam2d.vertical_tip_force_n",
                value: self.vertical_tip_force_n.to_string(),
            });
        }
        if self.modal_modes == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "beam2d.modal_modes",
                value: "0".to_string(),
            });
        }
        Ok(())
    }

    fn section(&self) -> BeamSection {
        if self.timoshenko {
            BeamSection::timoshenko(
                self.material.youngs_modulus_pa,
                self.geometry.second_moment_m4(),
                self.geometry.area_m2(),
                self.material.density_kg_m3,
                self.material.poisson_ratio,
                RECTANGULAR_SHEAR_CORRECTION,
            )
        } else {
            BeamSection::rectangular(
                self.material.youngs_modulus_pa,
                self.geometry.width_m,
                self.geometry.height_m,
                self.material.density_kg_m3,
            )
        }
    }

    fn elements(&self) -> Vec<BeamElement2D> {
        let section = self.section();
        let count = self.geometry.elements;
        let dx = self.geometry.length_m / count as f64;
        (0..count)
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
}

/// Static beam response with analytical cantilever comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Beam2DStaticResult {
    pub elements: usize,
    pub dofs: usize,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub tip_deflection_m: f64,
    pub analytical_tip_deflection_m: f64,
    pub analytical_relative_error: f64,
    pub max_displacement_m: f64,
    pub strain_energy_j: f64,
    pub applied_force_n: [f64; 2],
    pub reaction_force_n: [f64; 2],
    pub reaction_moment_nm: f64,
    pub force_balance_residual_n: f64,
    pub free_equilibrium_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub displacement: Vec<[f64; 3]>,
}

/// Generalized beam stiffness/mass eigenproblem result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Beam2DModalResult {
    pub eigenvalues: Vec<f64>,
    pub frequencies_hz: Vec<f64>,
    pub residual_norms: Vec<f64>,
    pub solver_iterations: usize,
    pub converged: bool,
    pub normalization: ModalNormalization,
    pub mass_normalization_error: f64,
    pub orthogonality_error: f64,
    pub damping_ratio: f64,
    pub damping_convention: DampingConvention,
    pub model_assumption: StructuralModelAssumption,
    pub modes: Vec<Vec<[f64; 3]>>,
}

/// Relative solver tolerance for the beam static solve.
///
/// The static path below uses a direct dense solve, which reaches true
/// residuals near 5e-12 even on the finest QA mesh. 1e-10 keeps over two
/// decades of solver margin while holding tip-deflection error near 1e-13,
/// five decades inside the 1e-8 physics gates. Krylov methods are the wrong
/// tool here: beam stiffness entries span ~18 orders of magnitude
/// (E*I/h^3 terms next to unit Dirichlet rows), which stalls both plain and
/// Jacobi-preconditioned CG just above 1e-10 on fine meshes.
pub const BEAM_STATIC_SOLVER_TOLERANCE: f64 = 1e-10;

/// Solve the assembled beam static system.
pub fn run_beam2d_static(
    study: &Beam2DStudy,
) -> Result<Beam2DStaticResult, StructuralMechanicsError> {
    study.validate()?;
    let elements = study.elements();
    let dofs = (study.geometry.elements + 1) * 3;
    let stiffness = assemble_beam_stiffness(&elements, dofs);
    let unconstrained_stiffness = stiffness.clone();
    let mut rhs = vec![0.0; dofs];
    let tip_vertical_dof = dofs - 2;
    rhs[tip_vertical_dof] = study.vertical_tip_force_n;
    let external_rhs = rhs.clone();
    let mut system = FemSystem::new(stiffness, rhs);
    let fixed_dofs = [0, 1, 2];
    try_apply_homogeneous_dirichlet_dofs(&mut system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "beam2d boundary",
            message: error.to_string(),
        }
    })?;
    // Direct dense LU: the beam static systems are tiny (at most 51 DOFs for
    // the 16-element QA mesh) but far too ill-conditioned for Krylov methods
    // at the physics-required tolerance. Reported as a single solver pass so
    // downstream iteration accounting stays positive.
    let csr = system.matrix.to_csr();
    let solution = lu_solve(&csr.to_dense(), &system.rhs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "beam2d static",
            message: error.to_string(),
        }
    })?;
    let residual = {
        let image = csr.apply(&solution);
        let rhs_norm = system
            .rhs
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let residual_norm = system
            .rhs
            .iter()
            .zip(image.iter())
            .map(|(rhs, image)| (rhs - image) * (rhs - image))
            .sum::<f64>()
            .sqrt();
        if rhs_norm > 0.0 {
            residual_norm / rhs_norm
        } else {
            residual_norm
        }
    };
    if !residual.is_finite() || residual > BEAM_STATIC_SOLVER_TOLERANCE {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::Beam,
            iterations: 1,
            residual,
        });
    }

    let displacement = solution
        .as_chunks::<3>()
        .0
        .iter()
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let internal_force = super::triplet_matvec(&unconstrained_stiffness, &solution);
    let reaction_force_n = reaction_force_2(&internal_force, &external_rhs, &fixed_dofs);
    let reaction_moment_nm = reaction_moment(&internal_force, &external_rhs, &fixed_dofs);
    let applied_force_n = component_force_2(&external_rhs);
    let force_balance_residual_n = norm2(add2(applied_force_n, reaction_force_n));
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
    let internal_energy = 0.5 * super::dot(&solution, &internal_force);
    let rhs_energy = 0.5 * super::dot(&solution, &external_rhs);
    let energy_balance_relative_error =
        (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0);
    let analytical_tip_deflection_m = study.vertical_tip_force_n * study.geometry.length_m.powi(3)
        / (3.0 * study.material.youngs_modulus_pa * study.geometry.second_moment_m4());
    let tip_deflection_m = solution[tip_vertical_dof];
    Ok(Beam2DStaticResult {
        elements: study.geometry.elements,
        dofs,
        solver_iterations: 1,
        solver_residual: residual,
        converged: true,
        tip_deflection_m,
        analytical_tip_deflection_m,
        analytical_relative_error: (tip_deflection_m.abs() - analytical_tip_deflection_m.abs())
            .abs()
            / analytical_tip_deflection_m.abs(),
        max_displacement_m: displacement
            .iter()
            .map(|value| (value[0] * value[0] + value[1] * value[1]).sqrt())
            .fold(0.0, f64::max),
        strain_energy_j: internal_energy,
        applied_force_n,
        reaction_force_n,
        reaction_moment_nm,
        force_balance_residual_n,
        free_equilibrium_residual_n,
        energy_balance_relative_error,
        model_assumption: StructuralModelAssumption::Beam,
        displacement,
    })
}

/// Solve the assembled generalized beam stiffness/mass eigenproblem.
pub fn run_beam2d_modal(
    study: &Beam2DStudy,
) -> Result<Beam2DModalResult, StructuralMechanicsError> {
    study.validate()?;
    let elements = study.elements();
    let dofs = (study.geometry.elements + 1) * 3;
    let stiffness = assemble_beam_stiffness(&elements, dofs);
    let mass = assemble_beam_mass(&elements, dofs);
    let (stiffness, free_to_full) = super::reduce_triplets(&stiffness, &[0, 1, 2]);
    let (mass, _) = super::reduce_triplets(&mass, &[0, 1, 2]);
    let solver_config = LobpcgConfig {
        num_eigenvalues: study.modal_modes,
        max_iterations: 1_000,
        tolerance: 1e-9,
        ..LobpcgConfig::default()
    };
    let lobpcg = lobpcg_generalized(&stiffness, &mass, &solver_config).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "beam2d modal",
            message: error.to_string(),
        }
    })?;
    let (eigenvalues, eigenvectors, residual_norms, solver_iterations, converged) =
        if lobpcg.converged {
            (
                lobpcg.eigenvalues,
                lobpcg.eigenvectors,
                lobpcg.residual_norms,
                lobpcg.iterations,
                true,
            )
        } else {
            let n = stiffness.num_rows;
            let targeted = shift_invert_lanczos_generalized(
                &stiffness,
                &mass,
                &ShiftInvertConfig {
                    target: 0.0,
                    num_eigenvalues: study.modal_modes,
                    krylov_dimension: (4 * study.modal_modes + 12).min(n),
                    tolerance: 1.0e-6,
                    breakdown_tolerance: 1.0e-14,
                },
            )
            .map_err(|error| StructuralMechanicsError::SolverFailure {
                stage: "beam2d modal shift-invert fallback",
                message: error.to_string(),
            })?;
            (
                targeted.eigenvalues,
                targeted.eigenvectors,
                targeted.residual_norms,
                targeted.iterations,
                targeted.converged,
            )
        };
    if !converged {
        return Err(StructuralMechanicsError::ModalNonConvergence {
            iterations: solver_iterations,
            residual: residual_norms.iter().copied().fold(0.0, f64::max),
        });
    }
    if eigenvalues
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(StructuralMechanicsError::UnstableModel {
            eigenvalue: eigenvalues.iter().copied().fold(f64::INFINITY, f64::min),
        });
    }
    let vectors = (0..eigenvectors.ncols())
        .map(|column| eigenvectors.column_to_vec(column))
        .collect::<Vec<_>>();
    let mut modes = Vec::with_capacity(vectors.len());
    let mut mass_normalization_error: f64 = 0.0;
    let mut orthogonality_error: f64 = 0.0;
    for vector in &vectors {
        let mass_vector = csr_matvec(&mass, vector);
        mass_normalization_error =
            mass_normalization_error.max((super::dot(vector, &mass_vector) - 1.0).abs());
        let mut full = vec![[0.0; 3]; study.geometry.elements + 1];
        for (reduced, &full_dof) in free_to_full.iter().enumerate() {
            full[full_dof / 3][full_dof % 3] = vector[reduced];
        }
        modes.push(full);
    }
    for (index, lhs) in vectors.iter().enumerate() {
        for rhs in vectors.iter().skip(index + 1) {
            orthogonality_error =
                orthogonality_error.max(super::dot(lhs, &csr_matvec(&mass, rhs)).abs());
        }
    }
    Ok(Beam2DModalResult {
        eigenvalues: eigenvalues.clone(),
        frequencies_hz: eigenvalues
            .iter()
            .map(|value| value.sqrt() / (2.0 * std::f64::consts::PI))
            .collect(),
        residual_norms,
        solver_iterations,
        converged,
        normalization: ModalNormalization::MassNormalized,
        mass_normalization_error,
        orthogonality_error,
        damping_ratio: study.material.damping_ratio,
        damping_convention: DampingConvention::MassProportionalAtFirstMode,
        model_assumption: StructuralModelAssumption::Beam,
        modes,
    })
}

fn csr_matvec(matrix: &sonium_solvers::CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * vector[matrix.col_indices[index]])
                .sum()
        })
        .collect()
}

fn component_force_2(vector: &[f64]) -> [f64; 2] {
    vector
        .as_chunks::<3>()
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
        if dof % 3 < 2 {
            total[dof % 3] += internal[dof] - external[dof];
        }
        total
    })
}

fn reaction_moment(internal: &[f64], external: &[f64], fixed_dofs: &[usize]) -> f64 {
    fixed_dofs
        .iter()
        .filter(|dof| **dof % 3 == 2)
        .map(|dof| internal[*dof] - external[*dof])
        .sum()
}

fn free_residual_norm(
    internal: &[f64],
    external: &[f64],
    dirichlet: &std::collections::HashSet<usize>,
) -> f64 {
    internal
        .iter()
        .zip(external)
        .enumerate()
        .filter(|(dof, _)| !dirichlet.contains(dof))
        .map(|(_, (lhs, rhs))| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn add2(lhs: [f64; 2], rhs: [f64; 2]) -> [f64; 2] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1]]
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

    fn small_study() -> Beam2DStudy {
        let mut study = Beam2DStudy::introductory();
        study.geometry.elements = 4;
        study.modal_modes = 2;
        study
    }

    #[test]
    fn beam_static_matches_cantilever_theory_and_equilibrium() {
        let result = run_beam2d_static(&small_study()).unwrap();

        assert!(result.converged);
        assert_eq!(result.model_assumption, StructuralModelAssumption::Beam);
        assert!(result.tip_deflection_m < 0.0);
        assert!(result.analytical_relative_error < 1e-10);
        assert!(
            result.force_balance_residual_n < 1e-8,
            "force_balance={}, applied={:?}, reaction={:?}",
            result.force_balance_residual_n,
            result.applied_force_n,
            result.reaction_force_n
        );
        assert!(result.free_equilibrium_residual_n < 1e-8);
        assert!(result.energy_balance_relative_error < 1e-10);
    }

    #[test]
    fn beam_modal_result_is_mass_normalized() {
        let result = run_beam2d_modal(&small_study()).unwrap();

        assert!(result.converged);
        assert_eq!(result.frequencies_hz.len(), 2);
        assert!(
            result
                .frequencies_hz
                .iter()
                .all(|frequency| *frequency > 0.0)
        );
        assert!(result.mass_normalization_error < 1e-8);
        assert!(result.orthogonality_error < 1e-8);
        assert_eq!(result.modes[0].len(), 5);
    }

    #[test]
    fn slender_beam_modal_fallback_matches_cantilever_roots() {
        let study = Beam2DStudy {
            name: "slender modal fallback".into(),
            geometry: Beam2DGeometry {
                length_m: 1.0,
                width_m: 0.1,
                height_m: 0.05,
                elements: 40,
            },
            material: StructuralMaterial {
                youngs_modulus_pa: 2.0e7,
                poisson_ratio: 0.3,
                density_kg_m3: 1000.0,
                damping_ratio: 0.0,
            },
            vertical_tip_force_n: -1.0,
            timoshenko: false,
            modal_modes: 3,
        };
        let result = run_beam2d_modal(&study).unwrap();
        let scale = (study.material.youngs_modulus_pa * study.geometry.second_moment_m4()
            / (study.material.density_kg_m3 * study.geometry.area_m2()))
        .sqrt()
            / (2.0 * std::f64::consts::PI * study.geometry.length_m.powi(2));
        let roots: [f64; 3] = [
            1.875_104_068_711_961,
            4.694_091_132_974_175,
            7.854_757_438_237_613,
        ];
        for (&actual, root) in result.frequencies_hz.iter().zip(roots) {
            let exact = root.powi(2) * scale;
            assert!((actual - exact).abs() / exact < 2.0e-4);
        }
        assert!(result.converged);
        assert!(result.mass_normalization_error < 1.0e-8);
    }

    #[test]
    fn beam_rejects_zero_elements() {
        let mut study = small_study();
        study.geometry.elements = 0;
        assert!(matches!(
            run_beam2d_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "beam2d.elements",
                ..
            })
        ));
    }
}
