//! Staggered phase-field damage evidence.
//!
//! The mechanics solve uses the linear elasticity form degraded by a
//! piecewise-constant element damage field. The scalar phase-field solve uses
//! Sonium FEM's fracture operator and an irreversible history field. Both
//! displacement- and force-controlled load paths are available; neither is a
//! claim of arc-length control or a post-failure crack solver.
//!
//! Each load step runs a single staggered pass (mechanics, then damage),
//! so the mechanics response lags the damage field by one step wherever
//! damage is nonuniform. The history driving term carries only the
//! tensile (Miehe spectral) energy: compression alone never nucleates
//! damage, though transverse Poisson tension still can (splitting).

use serde::{Deserialize, Serialize};
use sonium_fem::assembly::{assemble_bilinear_per_element, assemble_linear_per_element};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::mesh::{Mesh, rectangular_mesh_triangles};
use sonium_fem::physics::solid::elasticity::{IsotropicMaterial, PlaneAssumption};
use sonium_fem::physics::solid::fracture::{
    DamageHistoryRhsForm, DegradedElasticityForm, PhaseFieldFractureMaterial,
    assemble_damage_operator_with_history, plane_stress_energy_split,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{
    MatrixClass, Ordering, SparseDirectConfig, SparseDirectMethod, factorize_sparse,
};
use std::collections::HashMap;

use crate::{StructuralMechanicsError, StructuralModelAssumption};

/// Load control used by a phase-field damage result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseFieldLoadingControl {
    /// Prescribe the right-edge extension at each continuation step.
    PrescribedExtension,
    /// Apply a ramped total x-force to the right edge.
    ForceControlled,
}

/// A small 2D staggered phase-field damage study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseFieldDamageStudy {
    pub width_m: f64,
    pub height_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub youngs_modulus_pa: f64,
    pub poissons_ratio: f64,
    pub critical_energy_release_rate_j_per_m2: f64,
    pub length_scale_m: f64,
    pub residual_stiffness: f64,
    pub prescribed_extension_m: f64,
    pub load_steps: usize,
    /// Tip x-coordinate of the seeded edge notch: elements with
    /// centroid x <= `seed_x_m` inside the midline band carry the seed
    /// history. `seed_x_m == width_m` seeds a through-crack.
    pub seed_x_m: f64,
    /// Half-width of the seeded damage band about the plate midline.
    pub seed_half_width_m: f64,
    /// History energy density used to seed the initial damage band.
    pub seed_history_energy_density_j_per_m3: f64,
}

impl PhaseFieldDamageStudy {
    /// A compact damage-path case suitable for structural QA runs.
    pub fn introductory() -> Self {
        Self {
            width_m: 1.0,
            height_m: 0.5,
            nx: 8,
            ny: 4,
            youngs_modulus_pa: 1.0e6,
            poissons_ratio: 0.3,
            critical_energy_release_rate_j_per_m2: 1_000.0,
            length_scale_m: 0.125,
            residual_stiffness: 1.0e-6,
            prescribed_extension_m: 0.01,
            load_steps: 3,
            seed_x_m: 0.25,
            seed_half_width_m: 0.12,
            seed_history_energy_density_j_per_m3: 3_000.0,
        }
    }
}

/// Force-controlled variant of [`PhaseFieldDamageStudy`].
///
/// The base study supplies the mesh, material, seed, and continuation
/// settings. The force is the total x-force applied to the right edge; the
/// two-dimensional model assumes unit out-of-plane thickness.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseFieldForceControlledStudy {
    pub base: PhaseFieldDamageStudy,
    pub total_force_n: f64,
}

impl PhaseFieldForceControlledStudy {
    /// A force-controlled counterpart to the introductory displacement case.
    pub fn introductory() -> Self {
        let base = PhaseFieldDamageStudy::introductory();
        Self {
            total_force_n: base.youngs_modulus_pa * base.height_m * base.prescribed_extension_m
                / base.width_m,
            base,
        }
    }
}

/// Results and quality gates from [`run_phase_field_damage`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseFieldDamageResult {
    pub mesh_nodes: usize,
    pub elements: usize,
    pub mechanics_dofs: usize,
    pub damage_dofs: usize,
    pub load_steps: usize,
    pub mechanics_factorizations: usize,
    pub damage_factorizations: usize,
    pub final_mechanics_residual: f64,
    pub final_free_equilibrium_residual: f64,
    pub final_damage_residual: f64,
    pub max_damage: f64,
    pub mean_damage: f64,
    pub seed_damage: f64,
    pub damage_localization_ratio: f64,
    pub max_bound_violation: f64,
    pub final_elastic_energy_j: f64,
    pub final_crack_surface_energy_j: f64,
    pub loading_control: PhaseFieldLoadingControl,
    /// Total force at each continuation step for force-controlled runs.
    /// This is empty for prescribed-extension runs.
    pub applied_force_history_n: Vec<f64>,
    /// Mean right-edge x-displacement at each continuation step.
    pub tip_displacement_history_m: Vec<f64>,
    pub final_applied_force_n: f64,
    pub final_tip_displacement_m: f64,
    pub history_monotone: bool,
    pub damage_monotone: bool,
    /// Final nodal damage field (mesh node order) for profile assertions.
    pub nodal_damage: Vec<f64>,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
}

/// Run a staggered mechanics/phase-field damage path.
pub fn run_phase_field_damage(
    study: &PhaseFieldDamageStudy,
) -> Result<PhaseFieldDamageResult, StructuralMechanicsError> {
    run_phase_field_damage_with_loading(study, PhaseFieldLoading::PrescribedExtension)
}

/// Run the staggered damage path with a ramped right-edge force.
pub fn run_phase_field_damage_force_controlled(
    study: &PhaseFieldForceControlledStudy,
) -> Result<PhaseFieldDamageResult, StructuralMechanicsError> {
    if !study.total_force_n.is_finite() || study.total_force_n <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "total_force_n",
            value: study.total_force_n.to_string(),
        });
    }
    run_phase_field_damage_with_loading(
        &study.base,
        PhaseFieldLoading::ForceControlled {
            total_force_n: study.total_force_n,
        },
    )
}

#[derive(Debug, Clone, Copy)]
enum PhaseFieldLoading {
    PrescribedExtension,
    ForceControlled { total_force_n: f64 },
}

impl PhaseFieldLoading {
    fn result_control(self) -> PhaseFieldLoadingControl {
        match self {
            Self::PrescribedExtension => PhaseFieldLoadingControl::PrescribedExtension,
            Self::ForceControlled { .. } => PhaseFieldLoadingControl::ForceControlled,
        }
    }
}

fn run_phase_field_damage_with_loading(
    study: &PhaseFieldDamageStudy,
    loading: PhaseFieldLoading,
) -> Result<PhaseFieldDamageResult, StructuralMechanicsError> {
    validate_study(study)?;
    let mesh =
        rectangular_mesh_triangles(0.0, study.width_m, 0.0, study.height_m, study.nx, study.ny);
    let mechanics_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let mechanics_dofs = mechanics_space.num_global_dofs(&mesh);
    let damage_space_dofs = mesh.num_nodes();
    let elastic_material = IsotropicMaterial::new(study.youngs_modulus_pa, study.poissons_ratio);
    let fracture_material = PhaseFieldFractureMaterial {
        critical_energy_release_rate: study.critical_energy_release_rate_j_per_m2,
        length_scale: study.length_scale_m,
        residual_stiffness: study.residual_stiffness,
    };
    let mut damage = vec![0.0; damage_space_dofs];
    let mut damage_factorizations = 0usize;
    let mut history = vec![0.0; mesh.num_elements()];
    let mut mechanics_residual = f64::INFINITY;
    let mut free_equilibrium_residual = f64::INFINITY;
    let mut damage_residual = f64::INFINITY;
    let mut max_bound_violation: f64 = 0.0;
    let mut history_monotone = true;
    let mut damage_monotone = true;
    let mut final_displacement = vec![0.0; mechanics_dofs];
    let mut final_element_energy = vec![0.0; mesh.num_elements()];
    let mut applied_force_history_n = match loading {
        PhaseFieldLoading::PrescribedExtension => Vec::new(),
        PhaseFieldLoading::ForceControlled { .. } => Vec::with_capacity(study.load_steps),
    };
    let mut tip_displacement_history_m = Vec::with_capacity(study.load_steps);

    for step in 1..=study.load_steps {
        let fraction = step as f64 / study.load_steps as f64;
        let element_damage = element_damage_from_nodes(&mesh, &damage)?;
        let mechanics_forms: Vec<_> = element_damage
            .iter()
            .map(|&element_damage| DegradedElasticityForm {
                elastic_material,
                fracture_material,
                damage: element_damage,
                plane: Some(PlaneAssumption::PlaneStress),
            })
            .collect();
        let stiffness = assemble_bilinear_per_element(
            &mechanics_forms,
            &mechanics_space,
            &mechanics_space,
            &mesh,
            2,
        );
        let unconstrained_matrix = stiffness.to_csr();
        let (external_rhs, prescribed) = match loading {
            PhaseFieldLoading::PrescribedExtension => (
                vec![0.0; mechanics_dofs],
                displacement_boundary_conditions(
                    &mesh,
                    study.width_m,
                    study.prescribed_extension_m * fraction,
                ),
            ),
            PhaseFieldLoading::ForceControlled { total_force_n } => (
                right_edge_force_rhs(&mesh, study.width_m, total_force_n * fraction)?,
                support_boundary_conditions(&mesh, study.width_m),
            ),
        };
        let mut mechanics_system = FemSystem::new(stiffness, external_rhs.clone());
        apply_prescribed_dofs(&mut mechanics_system, &prescribed)?;
        let mechanics_matrix = mechanics_system.matrix.to_csr();
        let mechanics_factorization = factorize_sparse(&mechanics_matrix, &spd_config(), None)
            .map_err(|error| StructuralMechanicsError::SolverFailure {
                stage: "phase-field mechanics factorization",
                message: error.to_string(),
            })?;
        let displacement = mechanics_factorization
            .solve(&mechanics_system.rhs)
            .map_err(|error| StructuralMechanicsError::SolverFailure {
                stage: "phase-field mechanics solve",
                message: error.to_string(),
            })?;
        mechanics_residual = max_residual(&mechanics_matrix, &displacement, &mechanics_system.rhs);
        free_equilibrium_residual = max_free_residual(
            &unconstrained_matrix,
            &displacement,
            &external_rhs,
            &prescribed,
        );
        final_displacement = displacement;
        tip_displacement_history_m.push(mean_right_edge_displacement(
            &mesh,
            study.width_m,
            &final_displacement,
        )?);
        if let PhaseFieldLoading::ForceControlled { total_force_n } = loading {
            applied_force_history_n.push(total_force_n * fraction);
        }

        // History sees only the tensile (spectral-split) energy: the
        // total energy below still feeds the degraded elastic energy.
        let current_tensile: Vec<f64> = mesh
            .elements
            .iter()
            .map(|element| {
                element_tensile_energy_density(
                    &mesh,
                    element,
                    &final_displacement,
                    &elastic_material,
                )
            })
            .collect::<Result<_, _>>()?;
        let current_energy: Vec<f64> = mesh
            .elements
            .iter()
            .map(|element| {
                element_strain_energy_density(
                    &mesh,
                    element,
                    &final_displacement,
                    &elastic_material,
                )
            })
            .collect::<Result<_, _>>()?;
        debug_assert_eq!(current_tensile.len(), current_energy.len());
        let current_driving = current_tensile;
        let next_history: Vec<f64> = mesh
            .elements
            .iter()
            .zip(current_driving.iter())
            .enumerate()
            .map(|(element_index, (element, &energy))| {
                let centroid = element_centroid(&mesh, element)?;
                let seeded = centroid[0] <= study.seed_x_m
                    && (centroid[1] - 0.5 * study.height_m).abs() <= study.seed_half_width_m;
                let seed_energy = if seeded {
                    study.seed_history_energy_density_j_per_m3
                } else {
                    0.0
                };
                Ok(f64::max(
                    f64::max(history[element_index], energy),
                    seed_energy,
                ))
            })
            .collect::<Result<_, StructuralMechanicsError>>()?;
        history_monotone &= next_history
            .iter()
            .zip(history.iter())
            .all(|(next, previous)| *next + 1.0e-12 >= *previous);
        history = next_history
            .iter()
            .zip(history.iter())
            .map(|(next, previous)| next.max(*previous))
            .collect();
        final_element_energy = current_energy;

        let damage_forms: Vec<_> = history
            .iter()
            .map(|&history_energy_density| DamageHistoryRhsForm {
                history_energy_density,
            })
            .collect();
        let rhs = assemble_linear_per_element(
            &damage_forms,
            &sonium_fem::spaces::H1Space::new(PolynomialDegree::P1),
            &mesh,
            2,
        );
        // Consistent AT2 operator with the current history: rebuilt (and
        // refactorized) every step because the +2H mass term evolves.
        let damage_operator = assemble_damage_operator_with_history(
            &mesh,
            PolynomialDegree::P1,
            fracture_material,
            &history,
        )
        .to_csr();
        let damage_factorization = factorize_sparse(&damage_operator, &spd_config(), None)
            .map_err(|error| StructuralMechanicsError::SolverFailure {
                stage: "phase-field damage factorization",
                message: error.to_string(),
            })?;
        damage_factorizations += 1;
        let raw_damage = damage_factorization.solve(&rhs).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "phase-field damage solve",
                message: error.to_string(),
            }
        })?;
        damage_residual = max_residual(&damage_operator, &raw_damage, &rhs);
        max_bound_violation = max_bound_violation.max(
            raw_damage
                .iter()
                .map(|value| (*value - 1.0).max(0.0))
                .fold(0.0, f64::max),
        );
        let updated_damage: Vec<f64> = damage
            .iter()
            .zip(raw_damage.iter())
            .map(|(previous, next)| previous.max(next.clamp(0.0, 1.0)))
            .collect();
        damage_monotone &= updated_damage
            .iter()
            .zip(damage.iter())
            .all(|(next, previous)| *next + 1.0e-12 >= *previous);
        damage = updated_damage;
    }

    let element_damage = element_damage_from_nodes(&mesh, &damage)?;
    let final_elastic_energy_j = mesh
        .elements
        .iter()
        .zip(final_element_energy.iter())
        .zip(element_damage.iter())
        .map(|((element, energy_density), damage)| {
            triangle_area(&mesh, element)
                .map(|area| area * energy_density * fracture_material.degradation(*damage))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum();
    let final_crack_surface_energy_j = mesh
        .elements
        .iter()
        .zip(element_damage.iter())
        .map(|(element, &element_damage_value)| {
            let (area, gradients) = triangle_area_and_gradients(&mesh, element)?;
            let grad_x = gradients
                .iter()
                .enumerate()
                .map(|(index, gradient)| {
                    damage_at_node(&damage, element.nodes[index]) * gradient[0]
                })
                .sum::<f64>();
            let grad_y = gradients
                .iter()
                .enumerate()
                .map(|(index, gradient)| {
                    damage_at_node(&damage, element.nodes[index]) * gradient[1]
                })
                .sum::<f64>();
            Ok(area
                * fracture_material
                    .crack_surface_density(element_damage_value, grad_x * grad_x + grad_y * grad_y))
        })
        .collect::<Result<Vec<_>, StructuralMechanicsError>>()?
        .into_iter()
        .sum();
    let mean_damage = damage.iter().sum::<f64>() / damage.len().max(1) as f64;
    let seed_damage = mesh
        .elements
        .iter()
        .zip(element_damage.iter())
        .filter_map(|(element, damage)| {
            let centroid = element_centroid(&mesh, element).ok()?;
            ((centroid[0] <= study.seed_x_m)
                && (centroid[1] - 0.5 * study.height_m).abs() <= study.seed_half_width_m)
                .then_some(*damage)
        })
        .fold(0.0, f64::max);
    let damage_localization_ratio = if mean_damage > 1.0e-14 {
        seed_damage / mean_damage
    } else {
        0.0
    };
    let converged = mechanics_residual < 1.0e-6
        && free_equilibrium_residual < 1.0e-6
        && damage_residual < 1.0e-6
        && history_monotone
        && damage_monotone
        && max_bound_violation < 1.0e-8;

    Ok(PhaseFieldDamageResult {
        mesh_nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        mechanics_dofs,
        damage_dofs: damage_space_dofs,
        load_steps: study.load_steps,
        mechanics_factorizations: study.load_steps,
        damage_factorizations,
        final_mechanics_residual: mechanics_residual,
        final_free_equilibrium_residual: free_equilibrium_residual,
        final_damage_residual: damage_residual,
        max_damage: damage.iter().copied().fold(0.0, f64::max),
        mean_damage,
        seed_damage,
        damage_localization_ratio,
        max_bound_violation,
        final_elastic_energy_j,
        final_crack_surface_energy_j,
        loading_control: loading.result_control(),
        final_applied_force_n: applied_force_history_n.last().copied().unwrap_or(0.0),
        final_tip_displacement_m: tip_displacement_history_m.last().copied().unwrap_or(0.0),
        applied_force_history_n,
        tip_displacement_history_m,
        history_monotone,
        damage_monotone,
        nodal_damage: damage,
        converged,
        model_assumption: StructuralModelAssumption::PhaseFieldDamage2D,
    })
}

fn validate_study(study: &PhaseFieldDamageStudy) -> Result<(), StructuralMechanicsError> {
    for (field, value) in [
        ("width_m", study.width_m),
        ("height_m", study.height_m),
        ("youngs_modulus_pa", study.youngs_modulus_pa),
        (
            "critical_energy_release_rate_j_per_m2",
            study.critical_energy_release_rate_j_per_m2,
        ),
        ("length_scale_m", study.length_scale_m),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    // Extension may be negative (uniaxial compression path); only a zero
    // or non-finite drive is rejected.
    if !study.prescribed_extension_m.is_finite() || study.prescribed_extension_m == 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "prescribed_extension_m",
            value: study.prescribed_extension_m.to_string(),
        });
    }
    // Zero seed energy selects the unseeded homogeneous path.
    if !study.seed_history_energy_density_j_per_m3.is_finite()
        || study.seed_history_energy_density_j_per_m3 < 0.0
    {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "seed_history_energy_density_j_per_m3",
            value: study.seed_history_energy_density_j_per_m3.to_string(),
        });
    }
    if !study.poissons_ratio.is_finite() || !(-1.0..0.5).contains(&study.poissons_ratio) {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "poissons_ratio",
            value: study.poissons_ratio.to_string(),
        });
    }
    if !study.residual_stiffness.is_finite() || !(0.0..=1.0).contains(&study.residual_stiffness) {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "residual_stiffness",
            value: study.residual_stiffness.to_string(),
        });
    }
    if study.nx == 0 || study.ny == 0 || study.load_steps == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "mesh_or_load_resolution",
            value: format!("{}x{}, {} steps", study.nx, study.ny, study.load_steps),
        });
    }
    // seed_x_m == width_m is the through-crack limit (no tip); it is
    // admitted so the transverse profile has a tip-free reference.
    if study.seed_x_m <= 0.0 || study.seed_x_m > study.width_m {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "seed_x_m",
            value: study.seed_x_m.to_string(),
        });
    }
    if !study.seed_half_width_m.is_finite() || study.seed_half_width_m <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "seed_half_width_m",
            value: study.seed_half_width_m.to_string(),
        });
    }
    Ok(())
}

fn spd_config() -> SparseDirectConfig {
    SparseDirectConfig {
        matrix_class: MatrixClass::SymmetricPositiveDefinite,
        method: SparseDirectMethod::SimplicialCholesky,
        ordering: Ordering::Auto,
        ..SparseDirectConfig::default()
    }
}

fn displacement_boundary_conditions(
    mesh: &Mesh,
    width_m: f64,
    extension_m: f64,
) -> Vec<(usize, f64)> {
    let mut prescribed = support_boundary_conditions(mesh, width_m);
    let tolerance = width_m.max(1.0) * 1.0e-12;
    prescribed.extend(mesh.nodes.iter().enumerate().filter_map(|(node, point)| {
        ((point.x - width_m).abs() <= tolerance).then_some((node * 2, extension_m))
    }));
    prescribed
}

fn support_boundary_conditions(mesh: &Mesh, width_m: f64) -> Vec<(usize, f64)> {
    let tolerance = width_m.max(1.0) * 1.0e-12;
    let mut prescribed = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        let dof = node * 2;
        if point.x.abs() <= tolerance {
            prescribed.push((dof, 0.0));
            if point.y.abs() <= tolerance {
                prescribed.push((dof + 1, 0.0));
            }
        }
    }
    prescribed
}

fn right_edge_nodes(
    mesh: &Mesh,
    width_m: f64,
) -> Result<Vec<(usize, f64)>, StructuralMechanicsError> {
    let tolerance = width_m.max(1.0) * 1.0e-12;
    let mut nodes: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(node, point)| {
            ((point.x - width_m).abs() <= tolerance).then_some((node, point.y))
        })
        .collect();
    nodes.sort_by(|left, right| left.1.total_cmp(&right.1));
    if nodes.len() < 2 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "phase-field force control requires at least two right-edge nodes".to_string(),
        });
    }
    if let (Some(first), Some(last)) = (nodes.first(), nodes.last())
        && last.1 - first.1 <= tolerance
    {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "phase-field right edge has zero length".to_string(),
        });
    }
    Ok(nodes)
}

fn right_edge_force_rhs(
    mesh: &Mesh,
    width_m: f64,
    total_force_n: f64,
) -> Result<Vec<f64>, StructuralMechanicsError> {
    let nodes = right_edge_nodes(mesh, width_m)?;
    let (Some(first), Some(last)) = (nodes.first(), nodes.last()) else {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "phase-field force control found no right-edge nodes".to_string(),
        });
    };
    let edge_length = last.1 - first.1;
    if !edge_length.is_finite() || edge_length <= 0.0 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "phase-field force-control right edge has invalid length".to_string(),
        });
    }
    let traction_n_per_m = total_force_n / edge_length;
    let mut rhs = vec![0.0; mesh.num_nodes() * 2];
    for window in nodes.windows(2) {
        let segment = window[1].1 - window[0].1;
        let contribution = 0.5 * traction_n_per_m * segment;
        rhs[window[0].0 * 2] += contribution;
        rhs[window[1].0 * 2] += contribution;
    }
    Ok(rhs)
}

fn mean_right_edge_displacement(
    mesh: &Mesh,
    width_m: f64,
    displacement: &[f64],
) -> Result<f64, StructuralMechanicsError> {
    let nodes = right_edge_nodes(mesh, width_m)?;
    let mut sum = 0.0;
    for (node, _) in &nodes {
        let dof = node * 2;
        sum += displacement.get(dof).copied().ok_or_else(|| {
            StructuralMechanicsError::InvalidMesh {
                reason: "phase-field solution has insufficient displacement DOFs".to_string(),
            }
        })?;
    }
    Ok(sum / nodes.len() as f64)
}

fn apply_prescribed_dofs(
    system: &mut FemSystem<f64>,
    prescribed: &[(usize, f64)],
) -> Result<(), StructuralMechanicsError> {
    let values: HashMap<usize, f64> = prescribed.iter().copied().collect();
    for (&dof, &value) in &values {
        if dof >= system.rhs.len() || dof >= system.matrix.num_cols {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!("prescribed DOF {dof} is outside the mechanics system"),
            });
        }
        if !value.is_finite() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "prescribed_displacement",
                value: value.to_string(),
            });
        }
    }
    for index in 0..system.matrix.values.len() {
        let row = system.matrix.rows[index];
        let column = system.matrix.cols[index];
        if !values.contains_key(&row)
            && let Some(&value) = values.get(&column)
        {
            system.rhs[row] -= system.matrix.values[index] * value;
        }
    }
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut matrix_values = Vec::new();
    for index in 0..system.matrix.values.len() {
        let row = system.matrix.rows[index];
        let column = system.matrix.cols[index];
        if !values.contains_key(&row) && !values.contains_key(&column) {
            rows.push(row);
            cols.push(column);
            matrix_values.push(system.matrix.values[index]);
        }
    }
    for (&dof, &value) in &values {
        rows.push(dof);
        cols.push(dof);
        matrix_values.push(1.0);
        system.rhs[dof] = value;
        system.dirichlet_dofs.insert(dof);
    }
    system.matrix.rows = rows;
    system.matrix.cols = cols;
    system.matrix.values = matrix_values;
    Ok(())
}

fn max_residual(matrix: &sonium_solvers::CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    matrix
        .matvec(solution)
        .iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn max_free_residual(
    matrix: &sonium_solvers::CsrMatrix<f64>,
    solution: &[f64],
    rhs: &[f64],
    prescribed: &[(usize, f64)],
) -> f64 {
    let values = matrix.matvec(solution);
    values
        .iter()
        .zip(rhs.iter())
        .enumerate()
        .filter(|(dof, _)| {
            !prescribed
                .iter()
                .any(|(prescribed_dof, _)| prescribed_dof == dof)
        })
        .map(|(_, (left, right))| (left - right).abs())
        .fold(0.0, f64::max)
}

fn element_damage_from_nodes(
    mesh: &Mesh,
    damage: &[f64],
) -> Result<Vec<f64>, StructuralMechanicsError> {
    mesh.elements
        .iter()
        .map(|element| {
            if element.nodes.len() < 3 {
                return Err(StructuralMechanicsError::InvalidMesh {
                    reason: "phase-field element has fewer than three vertices".to_string(),
                });
            }
            let values = element
                .nodes
                .iter()
                .take(3)
                .map(|&node| {
                    damage
                        .get(node)
                        .copied()
                        .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
                            reason: "phase-field damage references a missing node".to_string(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(values.iter().sum::<f64>() / values.len() as f64)
        })
        .collect()
}

fn damage_at_node(damage: &[f64], node: usize) -> f64 {
    damage[node]
}

fn element_centroid(
    mesh: &Mesh,
    element: &sonium_fem::mesh::Element,
) -> Result<[f64; 2], StructuralMechanicsError> {
    if element.nodes.len() < 3 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "phase-field element has fewer than three vertices".to_string(),
        });
    }
    let points = element
        .nodes
        .iter()
        .take(3)
        .map(|&node| mesh.nodes.get(node))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
            reason: "phase-field element references a missing node".to_string(),
        })?;
    Ok([
        points.iter().map(|point| point.x).sum::<f64>() / 3.0,
        points.iter().map(|point| point.y).sum::<f64>() / 3.0,
    ])
}

fn triangle_area(
    mesh: &Mesh,
    element: &sonium_fem::mesh::Element,
) -> Result<f64, StructuralMechanicsError> {
    Ok(triangle_area_and_gradients(mesh, element)?.0)
}

fn triangle_area_and_gradients(
    mesh: &Mesh,
    element: &sonium_fem::mesh::Element,
) -> Result<(f64, [[f64; 2]; 3]), StructuralMechanicsError> {
    if element.nodes.len() < 3 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "phase-field element has fewer than three vertices".to_string(),
        });
    }
    let points = element
        .nodes
        .iter()
        .take(3)
        .map(|&node| mesh.nodes.get(node))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
            reason: "phase-field element references a missing node".to_string(),
        })?;
    let determinant = (points[1].x - points[0].x) * (points[2].y - points[0].y)
        - (points[2].x - points[0].x) * (points[1].y - points[0].y);
    if determinant.abs() <= 1.0e-15 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "phase-field element has zero area".to_string(),
        });
    }
    let gradients = [
        [
            (points[1].y - points[2].y) / determinant,
            (points[2].x - points[1].x) / determinant,
        ],
        [
            (points[2].y - points[0].y) / determinant,
            (points[0].x - points[2].x) / determinant,
        ],
        [
            (points[0].y - points[1].y) / determinant,
            (points[1].x - points[0].x) / determinant,
        ],
    ];
    Ok((0.5 * determinant.abs(), gradients))
}

/// Element-average infinitesimal strain `(exx, eyy, gamma_xy)`.
fn element_strain_2d(
    mesh: &Mesh,
    element: &sonium_fem::mesh::Element,
    displacement: &[f64],
) -> Result<(f64, f64, f64), StructuralMechanicsError> {
    let (_, gradients) = triangle_area_and_gradients(mesh, element)?;
    let mut exx = 0.0;
    let mut eyy = 0.0;
    let mut gamma_xy = 0.0;
    for (local, &node) in element.nodes.iter().take(3).enumerate() {
        let offset = node * 2;
        if offset + 1 >= displacement.len() {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "phase-field mechanics solution has insufficient DOFs".to_string(),
            });
        }
        let ux = displacement[offset];
        let uy = displacement[offset + 1];
        exx += ux * gradients[local][0];
        eyy += uy * gradients[local][1];
        gamma_xy += ux * gradients[local][1] + uy * gradients[local][0];
    }
    Ok((exx, eyy, gamma_xy))
}

fn element_strain_energy_density(
    mesh: &Mesh,
    element: &sonium_fem::mesh::Element,
    displacement: &[f64],
    material: &IsotropicMaterial,
) -> Result<f64, StructuralMechanicsError> {
    let (exx, eyy, gamma_xy) = element_strain_2d(mesh, element, displacement)?;
    let stiffness = material.voigt_2d_plane_stress();
    let sigma_xx = stiffness[0] * exx + stiffness[1] * eyy;
    let sigma_yy = stiffness[3] * exx + stiffness[4] * eyy;
    let tau_xy = stiffness[8] * gamma_xy;
    Ok(0.5 * (sigma_xx * exx + sigma_yy * eyy + tau_xy * gamma_xy).max(0.0))
}

/// Tensile strain-energy density driving damage (Miehe spectral split).
fn element_tensile_energy_density(
    mesh: &Mesh,
    element: &sonium_fem::mesh::Element,
    displacement: &[f64],
    material: &IsotropicMaterial,
) -> Result<f64, StructuralMechanicsError> {
    let (exx, eyy, gamma_xy) = element_strain_2d(mesh, element, displacement)?;
    let (tensile, _) = plane_stress_energy_split(
        exx,
        eyy,
        gamma_xy,
        material.youngs_modulus,
        material.poisson_ratio,
    );
    Ok(tensile.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn introductory_phase_field_damage_path_is_bounded_and_monotone() {
        let result = run_phase_field_damage(&PhaseFieldDamageStudy::introductory()).unwrap();
        assert!(result.converged, "{result:?}");
        assert_eq!(result.damage_factorizations, result.load_steps);
        assert_eq!(result.mechanics_factorizations, result.load_steps);
        assert!(result.max_damage > 0.0);
        assert!(result.max_damage <= 1.0);
        assert!(
            result.seed_damage >= result.mean_damage,
            "seed={} mean={} ratio={} max={}",
            result.seed_damage,
            result.mean_damage,
            result.damage_localization_ratio,
            result.max_damage
        );
        assert!(result.history_monotone);
        assert!(result.damage_monotone);
        assert!(result.final_crack_surface_energy_j >= 0.0);
    }

    #[test]
    fn invalid_phase_field_length_scale_is_rejected() {
        let mut study = PhaseFieldDamageStudy::introductory();
        study.length_scale_m = 0.0;
        let error = run_phase_field_damage(&study).unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidConfiguration {
                field: "length_scale_m",
                ..
            }
        ));
    }

    #[test]
    fn force_controlled_phase_field_path_integrates_edge_force_and_records_response() {
        let study = PhaseFieldForceControlledStudy::introductory();
        let mesh = rectangular_mesh_triangles(
            0.0,
            study.base.width_m,
            0.0,
            study.base.height_m,
            study.base.nx,
            study.base.ny,
        );
        let rhs = right_edge_force_rhs(&mesh, study.base.width_m, study.total_force_n).unwrap();
        assert!((rhs.iter().step_by(2).sum::<f64>() - study.total_force_n).abs() < 1.0e-12);

        let result = run_phase_field_damage_force_controlled(&study).unwrap();
        assert_eq!(
            result.loading_control,
            PhaseFieldLoadingControl::ForceControlled
        );
        assert_eq!(result.applied_force_history_n.len(), result.load_steps);
        assert_eq!(result.tip_displacement_history_m.len(), result.load_steps);
        assert!(
            result
                .applied_force_history_n
                .windows(2)
                .all(|window| window[1] > window[0])
        );
        assert!(
            result
                .tip_displacement_history_m
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(result.final_applied_force_n > 0.0);
        assert!(result.final_tip_displacement_m > 0.0);
        assert!(result.final_free_equilibrium_residual < 1.0e-6);
        assert!(result.converged, "{result:?}");
    }
}
