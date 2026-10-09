//! Executable thermoelastic structural studies.
//!
//! This family validates the thermal eigenstrain path against uniform free
//! expansion.  A single pin removes rigid translation while preserving the
//! exact affine thermal displacement field, so any recovered mechanical stress
//! is a meaningful discretization diagnostic rather than a support singularity.

use super::{
    StructuralMechanicsError, StructuralModelAssumption, dot, free_residual_norm, triplet_matvec,
};
use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet_dofs};
use sonium_fem::mesh::{Mesh, rectangular_mesh_triangles};
use sonium_fem::physics::coupled::thermoelasticity::{
    ThermoelasticMaterial, assemble_thermal_strain_load, assemble_thermoelastic_stiffness,
};
use sonium_fem::physics::solid::elasticity::{IsotropicMaterial, PlaneAssumption};
use sonium_solvers::{CgConfig, cg};

/// Uniform-temperature thermal load used by the executable family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalLoadModel {
    UniformTemperature,
}

/// Rectangular mesh for a plane-stress thermoelastic study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermoelasticGeometry {
    pub length_m: f64,
    pub width_m: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Default for ThermoelasticGeometry {
    fn default() -> Self {
        Self {
            length_m: 1.0,
            width_m: 0.1,
            nx: 20,
            ny: 4,
        }
    }
}

impl ThermoelasticGeometry {
    fn mesh(self) -> Mesh {
        rectangular_mesh_triangles(0.0, self.length_m, 0.0, self.width_m, self.nx, self.ny)
    }
}

/// Isotropic thermoelastic material and temperature inputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermoelasticMaterialInput {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub thermal_expansion_per_k: f64,
    pub reference_temperature_k: f64,
    pub applied_temperature_k: f64,
}

impl Default for ThermoelasticMaterialInput {
    fn default() -> Self {
        Self {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.0,
            thermal_expansion_per_k: 1.0e-5,
            reference_temperature_k: 20.0,
            applied_temperature_k: 120.0,
        }
    }
}

impl ThermoelasticMaterialInput {
    fn thermoelastic(self) -> ThermoelasticMaterial {
        ThermoelasticMaterial {
            young_modulus: self.youngs_modulus_pa,
            poisson_ratio: self.poisson_ratio,
            thermal_expansion: self.thermal_expansion_per_k,
            reference_temperature: self.reference_temperature_k,
        }
    }

    fn elastic(self) -> IsotropicMaterial {
        IsotropicMaterial::new(self.youngs_modulus_pa, self.poisson_ratio)
    }

    fn delta_temperature(self) -> f64 {
        self.applied_temperature_k - self.reference_temperature_k
    }

    fn thermal_strain(self) -> f64 {
        self.thermal_expansion_per_k * self.delta_temperature()
    }
}

/// Thermoelastic study configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoelasticStudy {
    pub name: String,
    pub geometry: ThermoelasticGeometry,
    pub material: ThermoelasticMaterialInput,
    pub load_model: ThermalLoadModel,
}

impl Default for ThermoelasticStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl ThermoelasticStudy {
    pub fn introductory() -> Self {
        Self {
            name: "thermoelastic-free-expansion".to_string(),
            geometry: ThermoelasticGeometry::default(),
            material: ThermoelasticMaterialInput::default(),
            load_model: ThermalLoadModel::UniformTemperature,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        let geometry = self.geometry;
        for (field, value) in [
            ("thermoelastic.length_m", geometry.length_m),
            ("thermoelastic.width_m", geometry.width_m),
            (
                "thermoelastic.youngs_modulus_pa",
                self.material.youngs_modulus_pa,
            ),
            (
                "thermoelastic.thermal_expansion_per_k",
                self.material.thermal_expansion_per_k,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        for (field, value) in [
            (
                "thermoelastic.reference_temperature_k",
                self.material.reference_temperature_k,
            ),
            (
                "thermoelastic.applied_temperature_k",
                self.material.applied_temperature_k,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.material.poisson_ratio.is_finite()
            || !(-1.0 < self.material.poisson_ratio && self.material.poisson_ratio < 0.5)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "thermoelastic.poisson_ratio",
                value: self.material.poisson_ratio.to_string(),
            });
        }
        if geometry.nx < 2 || geometry.ny < 2 || !geometry.ny.is_multiple_of(2) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "thermoelastic.mesh_density",
                value: format!(
                    "nx={}, ny={} (ny must be even and both >= 2)",
                    geometry.nx, geometry.ny
                ),
            });
        }
        if self.material.applied_temperature_k == self.material.reference_temperature_k {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "thermoelastic.temperature_delta",
                value: "applied and reference temperatures must differ".to_string(),
            });
        }
        Ok(())
    }
}

/// Validated free-expansion response and thermal stress diagnostic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoelasticStaticResult {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub delta_temperature_k: f64,
    pub thermal_strain: f64,
    pub max_displacement_m: f64,
    pub max_affine_displacement_error_m: f64,
    pub expected_corner_displacement_m: [f64; 2],
    pub actual_corner_displacement_m: [f64; 2],
    pub max_mechanical_stress_pa: f64,
    pub reaction_force_n: [f64; 2],
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub strain_energy_j: f64,
    pub energy_balance_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub load_model: ThermalLoadModel,
    pub displacement: Vec<[f64; 2]>,
}

/// Solve uniform-temperature free expansion with an assembled thermal load.
pub fn run_thermoelastic_static(
    study: &ThermoelasticStudy,
) -> Result<ThermoelasticStaticResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.geometry.mesh();
    let material = study.material;
    let thermoelastic = material.thermoelastic();
    let stiffness = assemble_thermoelastic_stiffness(
        &mesh,
        PolynomialDegree::P1,
        2,
        material.elastic(),
        Some(PlaneAssumption::PlaneStress),
    );
    let unconstrained_stiffness = stiffness.clone();
    let thermal_load = assemble_thermal_strain_load(
        &mesh,
        PolynomialDegree::P1,
        2,
        thermoelastic,
        material.applied_temperature_k,
    );
    let external_rhs = thermal_load.clone();
    let mut system = FemSystem::new(stiffness, thermal_load);
    // Pin one node only: this removes rigid translation without suppressing
    // the affine free-expansion solution elsewhere in the domain.
    let fixed_dofs = [0, 1];
    try_apply_homogeneous_dirichlet_dofs(&mut system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "thermoelastic boundary",
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
            model: StructuralModelAssumption::PlaneStress,
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
    let reaction_force_n = [
        internal_force[0] - external_rhs[0],
        internal_force[1] - external_rhs[1],
    ];
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
    let force_balance_residual_n = norm2(reaction_force_n);
    let internal_energy = 0.5 * dot(&solved.x, &internal_force);
    let rhs_energy = 0.5 * dot(&solved.x, &external_rhs);
    let energy_balance_relative_error =
        (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0);

    let expected_strain = material.thermal_strain();
    let mut max_affine_displacement_error_m: f64 = 0.0;
    let mut max_displacement_m: f64 = 0.0;
    let mut max_mechanical_stress_pa: f64 = 0.0;
    for (node, point) in mesh.nodes.iter().enumerate() {
        let expected = [expected_strain * point.x, expected_strain * point.y];
        max_affine_displacement_error_m = max_affine_displacement_error_m.max(norm2([
            displacement[node][0] - expected[0],
            displacement[node][1] - expected[1],
        ]));
        max_displacement_m = max_displacement_m.max(norm2(displacement[node]));
    }
    for element in &mesh.elements {
        let nodes = element.vertices();
        if nodes.len() != 3 {
            continue;
        }
        let gradients = triangle_shape_gradients(&mesh, nodes);
        let mut grad = [[0.0; 2]; 2];
        for (local, &node) in nodes.iter().enumerate() {
            for component in 0..2 {
                for direction in 0..2 {
                    grad[component][direction] +=
                        displacement[node][component] * gradients[local][direction];
                }
            }
        }
        let mechanical_strain = [
            grad[0][0] - expected_strain,
            grad[1][1] - expected_strain,
            grad[0][1] + grad[1][0],
        ];
        let nu = material.poisson_ratio;
        let scale = material.youngs_modulus_pa / (1.0 - nu * nu);
        let stress = [
            scale * (mechanical_strain[0] + nu * mechanical_strain[1]),
            scale * (nu * mechanical_strain[0] + mechanical_strain[1]),
            scale * (1.0 - nu) * 0.5 * mechanical_strain[2],
        ];
        let von_mises = (stress[0] * stress[0] - stress[0] * stress[1]
            + stress[1] * stress[1]
            + 3.0 * stress[2] * stress[2])
            .sqrt();
        max_mechanical_stress_pa = max_mechanical_stress_pa.max(von_mises);
    }

    let corner = mesh.num_nodes() - 1;
    let expected_corner_displacement_m = [
        expected_strain * study.geometry.length_m,
        expected_strain * study.geometry.width_m,
    ];
    let actual_corner_displacement_m = displacement[corner];
    let thermal_load_scale = external_rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    if !free_equilibrium_residual_n.is_finite()
        || free_equilibrium_residual_n > 1.0e-6 * thermal_load_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "thermoelastic.free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !force_balance_residual_n.is_finite()
        || force_balance_residual_n > 1.0e-6 * thermal_load_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "thermoelastic.force_balance",
            value: force_balance_residual_n,
        });
    }
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1.0e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "thermoelastic.energy_identity",
            value: energy_balance_relative_error,
        });
    }

    Ok(ThermoelasticStaticResult {
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs: displacement.len() * 2,
        solver_iterations: solved.iterations,
        solver_residual: solved.residual,
        converged: solved.converged,
        delta_temperature_k: material.delta_temperature(),
        thermal_strain: expected_strain,
        max_displacement_m,
        max_affine_displacement_error_m,
        expected_corner_displacement_m,
        actual_corner_displacement_m,
        max_mechanical_stress_pa,
        reaction_force_n,
        free_equilibrium_residual_n,
        force_balance_residual_n,
        strain_energy_j: internal_energy,
        energy_balance_relative_error,
        model_assumption: StructuralModelAssumption::PlaneStress,
        load_model: study.load_model,
        displacement,
    })
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

    fn small_study() -> ThermoelasticStudy {
        let mut study = ThermoelasticStudy::introductory();
        study.geometry.nx = 8;
        study.geometry.ny = 2;
        study
    }

    #[test]
    fn thermoelastic_free_expansion_matches_affine_solution() {
        let result = run_thermoelastic_static(&small_study()).unwrap();

        assert!(result.converged);
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::PlaneStress
        );
        assert!(result.max_affine_displacement_error_m < 1.0e-10);
        assert!(
            result.max_mechanical_stress_pa
                < 1.0e-9
                    * ThermoelasticMaterialInput::default().youngs_modulus_pa
                    * result.thermal_strain.abs(),
            "max mechanical stress={}",
            result.max_mechanical_stress_pa
        );
        assert!(result.free_equilibrium_residual_n < 1.0e-2);
        assert!(result.force_balance_residual_n < 1.0e-2);
        assert!(result.energy_balance_relative_error < 1.0e-10);
        assert!(result.actual_corner_displacement_m[0] > 0.0);
        assert!(result.actual_corner_displacement_m[1] > 0.0);
    }

    #[test]
    fn thermoelastic_temperature_sign_reverses_displacement() {
        let hot = run_thermoelastic_static(&small_study()).unwrap();
        let mut cold_study = small_study();
        cold_study.material.applied_temperature_k = 0.0;
        let cold = run_thermoelastic_static(&cold_study).unwrap();
        assert!(hot.actual_corner_displacement_m[0] > 0.0);
        assert!(cold.actual_corner_displacement_m[0] < 0.0);
    }

    #[test]
    fn thermoelastic_rejects_zero_temperature_delta_and_invalid_mesh() {
        let mut study = small_study();
        study.material.applied_temperature_k = study.material.reference_temperature_k;
        assert!(matches!(
            run_thermoelastic_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "thermoelastic.temperature_delta",
                ..
            })
        ));

        let mut study = small_study();
        study.geometry.ny = 3;
        assert!(matches!(
            run_thermoelastic_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "thermoelastic.mesh_density",
                ..
            })
        ));
    }
}
