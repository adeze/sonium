//! Displacement-controlled three-dimensional hyperelastic evidence.
//!
//! This module deliberately uses a small P1 tetrahedral box and a prescribed
//! affine extension.  The benchmark exercises Sonium's actual nonlinear FEM
//! residual/tangent assembly and Newton solve while retaining a homogeneous
//! uniaxial deformation (zero lateral traction) for energy and constitutive
//! checks.  It is not a force-controlled post-buckling or fracture solver.

use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, box_mesh_tetrahedra};
use sonium_fem::physics::solid::nonlinear_solid::{
    HyperelasticForm, HyperelasticMaterial, NeoHookean,
};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_with_bcs};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};

use crate::StructuralMechanicsError;

/// A small 3D displacement-controlled total-Lagrangian hyperelastic study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DStudy {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub youngs_modulus_pa: f64,
    pub poissons_ratio: f64,
    pub prescribed_extension_m: f64,
    pub load_steps: usize,
}

impl Hyperelastic3DStudy {
    /// A compact homogeneous-extension case suitable for unit and QA runs.
    pub fn introductory() -> Self {
        Self {
            length_m: 1.0,
            width_m: 0.25,
            height_m: 0.25,
            nx: 2,
            ny: 1,
            nz: 1,
            youngs_modulus_pa: 2.0e6,
            poissons_ratio: 0.3,
            prescribed_extension_m: 0.05,
            load_steps: 4,
        }
    }
}

/// Evidence returned by [`run_hyperelastic_3d`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DResult {
    pub mesh_nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub load_steps: usize,
    pub newton_iterations: usize,
    pub prescribed_extension_m: f64,
    pub final_residual: f64,
    pub converged: bool,
    pub max_displacement_m: f64,
    pub max_affine_displacement_error_m: f64,
    pub deformation_gradient: [f64; 9],
    pub deformation_gradient_error: f64,
    pub strain_energy_j: f64,
    pub homogeneous_reference_energy_j: f64,
    pub relative_energy_error: f64,
    pub constitutive_gradient_error: f64,
    /// `r(h/2) / r(h)` for the first-order Taylor remainder of `W(F)`.
    /// A smooth twice-differentiable material should approach 1/4.
    pub taylor_remainder_ratio: f64,
}

/// Run the actual FEM Newton solve through a displacement-controlled load path.
pub fn run_hyperelastic_3d(
    study: &Hyperelastic3DStudy,
) -> Result<Hyperelastic3DResult, StructuralMechanicsError> {
    validate_study(study)?;

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
    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let dofs = space.num_global_dofs(&mesh);
    let material = NeoHookean::from_young(study.youngs_modulus_pa, study.poissons_ratio, 3);
    let form = HyperelasticForm::new(material);
    let config = FemNewtonConfig::default();
    let mut solution = vec![0.0; dofs];
    let mut total_iterations = 0usize;
    let mut final_residual = f64::INFINITY;

    for step in 1..=study.load_steps {
        let fraction = step as f64 / study.load_steps as f64;
        let dirichlet_dofs = displacement_boundary_conditions(
            &mesh,
            study.length_m,
            study.prescribed_extension_m * fraction,
        );
        let newton =
            fem_newton_solve_with_bcs(&form, &space, &mesh, &dirichlet_dofs, &solution, &config)
                .map_err(|error| StructuralMechanicsError::SolverFailure {
                    stage: "3D hyperelastic Newton solve",
                    message: error.to_string(),
                })?;

        total_iterations += newton.iterations;
        final_residual = newton.residual;
        if !newton.converged {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: crate::StructuralModelAssumption::HyperelasticSolid3D,
                load_step: step,
                iterations: newton.iterations,
                residual: newton.residual,
            });
        }
        solution = newton.x;
    }

    let axial_stretch = 1.0 + study.prescribed_extension_m / study.length_m;
    let lateral_stretch = transverse_stretch(&material, axial_stretch);
    let expected_f = diagonal_deformation_gradient(axial_stretch, lateral_stretch);
    let strain_energy_j = total_p1_tetra_energy(&mesh, &solution, &material)?;
    let homogeneous_reference_energy_j = reference_energy(&mesh, &material, &expected_f)?;
    let relative_energy_error = (strain_energy_j - homogeneous_reference_energy_j).abs()
        / homogeneous_reference_energy_j.abs().max(1.0e-30);
    let max_affine_displacement_error_m =
        max_affine_displacement_error(&mesh, &solution, axial_stretch - 1.0, lateral_stretch - 1.0);
    let max_displacement_m = solution
        .as_chunks::<3>()
        .0
        .iter()
        .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
        .fold(0.0, f64::max);
    let deformation_gradient = first_element_deformation_gradient(&mesh, &solution)?;
    let deformation_gradient_error = max_matrix_difference(&deformation_gradient, &expected_f);
    let (constitutive_gradient_error, taylor_remainder_ratio) =
        constitutive_checks(&material, &expected_f);

    Ok(Hyperelastic3DResult {
        mesh_nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs,
        load_steps: study.load_steps,
        newton_iterations: total_iterations,
        prescribed_extension_m: study.prescribed_extension_m,
        final_residual,
        converged: true,
        max_displacement_m,
        max_affine_displacement_error_m,
        deformation_gradient,
        deformation_gradient_error,
        strain_energy_j,
        homogeneous_reference_energy_j,
        relative_energy_error,
        constitutive_gradient_error,
        taylor_remainder_ratio,
    })
}

fn validate_study(study: &Hyperelastic3DStudy) -> Result<(), StructuralMechanicsError> {
    for (field, value) in [
        ("length_m", study.length_m),
        ("width_m", study.width_m),
        ("height_m", study.height_m),
        ("youngs_modulus_pa", study.youngs_modulus_pa),
        ("prescribed_extension_m", study.prescribed_extension_m),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    if !study.poissons_ratio.is_finite() || !(0.0..0.5).contains(&study.poissons_ratio) {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "poissons_ratio",
            value: study.poissons_ratio.to_string(),
        });
    }
    if study.nx == 0 || study.ny == 0 || study.nz == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "mesh_resolution",
            value: format!("{}x{}x{}", study.nx, study.ny, study.nz),
        });
    }
    if study.load_steps == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "load_steps",
            value: study.load_steps.to_string(),
        });
    }
    if study.prescribed_extension_m >= study.length_m {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "prescribed_extension_m",
            value: study.prescribed_extension_m.to_string(),
        });
    }
    Ok(())
}

fn displacement_boundary_conditions(
    mesh: &Mesh,
    length_m: f64,
    prescribed_extension_m: f64,
) -> Vec<(usize, f64)> {
    let tolerance = length_m.max(1.0) * 1.0e-12;
    let mut dofs = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        let dof = node * 3;
        if point.x.abs() <= tolerance {
            // Only the normal displacement is fixed on the left face. The
            // tangential anchor at the origin removes rigid translation and
            // rotation while allowing the physical lateral contraction.
            dofs.push((dof, 0.0));
            if point.y.abs() <= tolerance && point.z.abs() <= tolerance {
                dofs.push((dof + 1, 0.0));
                dofs.push((dof + 2, 0.0));
            }
        } else if (point.x - length_m).abs() <= tolerance {
            dofs.push((dof, prescribed_extension_m));
        }
    }
    dofs
}

pub(crate) fn total_p1_tetra_energy(
    mesh: &Mesh,
    solution: &[f64],
    material: &impl HyperelasticMaterial,
) -> Result<f64, StructuralMechanicsError> {
    let mut total = 0.0;
    for element in &mesh.elements {
        if element.nodes.len() < 4 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "hyperelastic P1 tetrahedron has fewer than four vertices".to_string(),
            });
        }
        let reference = tetra_matrix(mesh, &element.nodes, None)?;
        let current = tetra_matrix(mesh, &element.nodes, Some(solution))?;
        let determinant = determinant3(&reference);
        let inverse =
            inverse3(&reference).ok_or_else(|| StructuralMechanicsError::InvalidMesh {
                reason: "hyperelastic tetrahedron has zero reference volume".to_string(),
            })?;
        let deformation_gradient = matrix_product(&current, &inverse);
        if determinant <= 0.0 || determinant3(&deformation_gradient) <= 0.0 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "hyperelastic tetrahedron has invalid orientation".to_string(),
            });
        }
        total += determinant * material.energy(&deformation_gradient) / 6.0;
    }
    Ok(total)
}

fn reference_energy(
    mesh: &Mesh,
    material: &impl HyperelasticMaterial,
    deformation_gradient: &[f64; 9],
) -> Result<f64, StructuralMechanicsError> {
    let reference_energy_density = material.energy(deformation_gradient);
    let mut volume = 0.0;
    for element in &mesh.elements {
        let reference = tetra_matrix(mesh, &element.nodes, None)?;
        volume += determinant3(&reference) / 6.0;
    }
    Ok(volume * reference_energy_density)
}

fn first_element_deformation_gradient(
    mesh: &Mesh,
    solution: &[f64],
) -> Result<[f64; 9], StructuralMechanicsError> {
    let element = mesh
        .elements
        .first()
        .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
            reason: "hyperelastic mesh has no elements".to_string(),
        })?;
    let reference = tetra_matrix(mesh, &element.nodes, None)?;
    let current = tetra_matrix(mesh, &element.nodes, Some(solution))?;
    let inverse = inverse3(&reference).ok_or_else(|| StructuralMechanicsError::InvalidMesh {
        reason: "hyperelastic tetrahedron has zero reference volume".to_string(),
    })?;
    Ok(matrix_product(&current, &inverse))
}

fn tetra_matrix(
    mesh: &Mesh,
    nodes: &[usize],
    solution: Option<&[f64]>,
) -> Result<[f64; 9], StructuralMechanicsError> {
    let mut coordinates = [[0.0; 3]; 4];
    for (local, &node) in nodes.iter().take(4).enumerate() {
        let point = mesh
            .nodes
            .get(node)
            .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
                reason: "hyperelastic element references a missing node".to_string(),
            })?;
        coordinates[local] = [point.x, point.y, point.z];
        if let Some(solution) = solution {
            let offset = node * 3;
            if offset + 2 >= solution.len() {
                return Err(StructuralMechanicsError::InvalidMesh {
                    reason: "hyperelastic solution has insufficient displacement DOFs".to_string(),
                });
            }
            coordinates[local][0] += solution[offset];
            coordinates[local][1] += solution[offset + 1];
            coordinates[local][2] += solution[offset + 2];
        }
    }
    Ok([
        coordinates[1][0] - coordinates[0][0],
        coordinates[2][0] - coordinates[0][0],
        coordinates[3][0] - coordinates[0][0],
        coordinates[1][1] - coordinates[0][1],
        coordinates[2][1] - coordinates[0][1],
        coordinates[3][1] - coordinates[0][1],
        coordinates[1][2] - coordinates[0][2],
        coordinates[2][2] - coordinates[0][2],
        coordinates[3][2] - coordinates[0][2],
    ])
}

fn constitutive_checks(
    material: &impl HyperelasticMaterial,
    deformation_gradient: &[f64; 9],
) -> (f64, f64) {
    let direction = [0.10, 0.03, -0.02, -0.04, 0.07, 0.02, 0.01, -0.03, 0.05];
    let stress = material.stress(deformation_gradient);
    let analytic_directional_derivative: f64 = stress
        .iter()
        .zip(direction.iter())
        .map(|(stress, direction)| stress * direction)
        .sum();
    let h = 1.0e-5;
    let plus = perturb_matrix(deformation_gradient, &direction, h);
    let minus = perturb_matrix(deformation_gradient, &direction, -h);
    let numerical_directional_derivative =
        (material.energy(&plus) - material.energy(&minus)) / (2.0 * h);
    let gradient_error = (numerical_directional_derivative - analytic_directional_derivative).abs()
        / (1.0 + analytic_directional_derivative.abs());

    let h_large = 5.0e-3;
    let h_small = h_large / 2.0;
    let base_energy = material.energy(deformation_gradient);
    let first_order = |h: f64| {
        let perturbed = perturb_matrix(deformation_gradient, &direction, h);
        (material.energy(&perturbed) - base_energy - h * analytic_directional_derivative).abs()
    };
    let large_remainder = first_order(h_large);
    let small_remainder = first_order(h_small);
    let ratio = if large_remainder > 1.0e-30 {
        small_remainder / large_remainder
    } else {
        f64::NAN
    };
    (gradient_error, ratio)
}

fn diagonal_deformation_gradient(axial_stretch: f64, lateral_stretch: f64) -> [f64; 9] {
    [
        axial_stretch,
        0.0,
        0.0,
        0.0,
        lateral_stretch,
        0.0,
        0.0,
        0.0,
        lateral_stretch,
    ]
}

fn transverse_stretch(material: &impl HyperelasticMaterial, axial_stretch: f64) -> f64 {
    // For homogeneous uniaxial extension the lateral traction is P22 = P33 =
    // 0. Solve that scalar constitutive equation without adding a numerical
    // root-finder dependency to the structural example crate.
    let mut lower = 0.5;
    let mut upper = 1.5;
    for _ in 0..80 {
        let middle = 0.5 * (lower + upper);
        let stress = material.stress(&diagonal_deformation_gradient(axial_stretch, middle))[4];
        if stress > 0.0 {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    0.5 * (lower + upper)
}

fn perturb_matrix(base: &[f64; 9], direction: &[f64; 9], scale: f64) -> [f64; 9] {
    std::array::from_fn(|index| base[index] + scale * direction[index])
}

fn max_affine_displacement_error(
    mesh: &Mesh,
    solution: &[f64],
    axial_strain: f64,
    lateral_strain: f64,
) -> f64 {
    mesh.nodes
        .iter()
        .enumerate()
        .map(|(node, point)| {
            let offset = node * 3;
            let expected = [
                axial_strain * point.x,
                lateral_strain * point.y,
                lateral_strain * point.z,
            ];
            (0..3)
                .map(|component| (solution[offset + component] - expected[component]).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

fn max_matrix_difference(left: &[f64; 9], right: &[f64; 9]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn determinant3(matrix: &[f64; 9]) -> f64 {
    matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
}

fn inverse3(matrix: &[f64; 9]) -> Option<[f64; 9]> {
    let determinant = determinant3(matrix);
    if determinant.abs() <= 1.0e-15 {
        return None;
    }
    Some([
        (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / determinant,
        (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / determinant,
        (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / determinant,
        (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / determinant,
        (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / determinant,
        (matrix[2] * matrix[3] - matrix[0] * matrix[5]) / determinant,
        (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / determinant,
        (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / determinant,
        (matrix[0] * matrix[4] - matrix[1] * matrix[3]) / determinant,
    ])
}

fn matrix_product(left: &[f64; 9], right: &[f64; 9]) -> [f64; 9] {
    std::array::from_fn(|index| {
        let row = index / 3;
        let column = index % 3;
        (0..3)
            .map(|inner| left[row * 3 + inner] * right[inner * 3 + column])
            .sum()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn introductory_hyperelastic_case_is_affine_and_converged() {
        let result = run_hyperelastic_3d(&Hyperelastic3DStudy::introductory()).unwrap();
        assert!(result.converged);
        assert_eq!(result.load_steps, 4);
        assert!(result.newton_iterations > 0);
        assert!(result.final_residual < 1.0e-8, "{}", result.final_residual);
        assert!(
            result.max_affine_displacement_error_m < 1.0e-10,
            "affine error={}",
            result.max_affine_displacement_error_m
        );
        assert!(
            result.deformation_gradient_error < 1.0e-10,
            "gradient error={}",
            result.deformation_gradient_error
        );
        assert!(
            result.relative_energy_error < 1.0e-10,
            "energy error={}",
            result.relative_energy_error
        );
        assert!(
            result.constitutive_gradient_error < 1.0e-8,
            "constitutive error={}",
            result.constitutive_gradient_error
        );
        assert!(
            (result.taylor_remainder_ratio - 0.25).abs() < 0.02,
            "Taylor ratio={}",
            result.taylor_remainder_ratio
        );
    }

    #[test]
    fn invalid_hyperelastic_mesh_resolution_is_rejected() {
        let mut study = Hyperelastic3DStudy::introductory();
        study.nx = 0;
        let error = run_hyperelastic_3d(&study).unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidConfiguration {
                field: "mesh_resolution",
                ..
            }
        ));
    }
}
