//! Executable Mindlin plate study families.
//!
//! The family intentionally uses a simply-supported Navier benchmark rather
//! than reporting an unvalidated plate scalar.  The assembled P1 triangular
//! Mindlin form uses Sonium FEM's DSG shear projection, and the result keeps
//! the support, load, equilibrium, and energy evidence alongside the center
//! deflection.

use super::{
    StructuralMechanicsError, StructuralModelAssumption, dot, free_residual_norm, triplet_matvec,
};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet_dofs};
use sonium_fem::mesh::{Mesh, rectangular_mesh_triangles};
use sonium_fem::physics::solid::mindlin_plate::{
    MindlinPlateForm, MindlinPlateLoadForm, MindlinPlateShearStabilization, MindlinPlateSpace,
};
use sonium_fem::spaces::FunctionSpace;
use sonium_solvers::{CgConfig, cg};

/// Shear treatment recorded in a plate study and its result provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlateShearTreatment {
    /// Element-average DSG/MITC-style shear projection for P1 triangles.
    Dsg,
    /// Fully compatible `theta - grad(w)` shear field.
    Consistent,
}

impl PlateShearTreatment {
    fn as_fem(self) -> MindlinPlateShearStabilization {
        match self {
            Self::Dsg => MindlinPlateShearStabilization::Dsg,
            Self::Consistent => MindlinPlateShearStabilization::Consistent,
        }
    }
}

/// Rectangular plate dimensions and structured triangular mesh density.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Plate2DGeometry {
    pub length_m: f64,
    pub width_m: f64,
    pub thickness_m: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Default for Plate2DGeometry {
    fn default() -> Self {
        Self {
            length_m: 1.0,
            width_m: 1.0,
            thickness_m: 0.1,
            nx: 20,
            ny: 20,
        }
    }
}

impl Plate2DGeometry {
    pub fn area_m2(self) -> f64 {
        self.length_m * self.width_m
    }

    pub fn mesh(self) -> Mesh {
        rectangular_mesh_triangles(0.0, self.length_m, 0.0, self.width_m, self.nx, self.ny)
    }
}

/// Isotropic linear-elastic plate material.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Plate2DMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
}

impl Default for Plate2DMaterial {
    fn default() -> Self {
        Self {
            youngs_modulus_pa: 1.0e6,
            poisson_ratio: 0.3,
        }
    }
}

/// A simply-supported Mindlin plate study under a sinusoidal transverse load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plate2DStudy {
    pub name: String,
    pub geometry: Plate2DGeometry,
    pub material: Plate2DMaterial,
    /// Pressure amplitude in `q(x,y) = q0 sin(pi*x/a) sin(pi*y/b)`.
    pub pressure_amplitude_pa: f64,
    /// Mindlin shear correction factor.
    pub shear_correction: f64,
    pub shear_treatment: PlateShearTreatment,
}

impl Default for Plate2DStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl Plate2DStudy {
    /// Construct the square-plate Navier benchmark used by the family tests.
    pub fn introductory() -> Self {
        Self {
            name: "mindlin-simply-supported-sinusoidal-plate".to_string(),
            geometry: Plate2DGeometry::default(),
            material: Plate2DMaterial::default(),
            pressure_amplitude_pa: 1.0e6,
            shear_correction: 5.0 / 6.0,
            shear_treatment: PlateShearTreatment::Dsg,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        let geometry = self.geometry;
        for (field, value) in [
            ("plate2d.length_m", geometry.length_m),
            ("plate2d.width_m", geometry.width_m),
            ("plate2d.thickness_m", geometry.thickness_m),
            ("plate2d.youngs_modulus_pa", self.material.youngs_modulus_pa),
            ("plate2d.poisson_ratio", self.material.poisson_ratio),
            ("plate2d.shear_correction", self.shear_correction),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.pressure_amplitude_pa.is_finite()
            || self.pressure_amplitude_pa.abs() <= f64::EPSILON
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "plate2d.pressure_amplitude_pa",
                value: self.pressure_amplitude_pa.to_string(),
            });
        }
        if !(-1.0 < self.material.poisson_ratio && self.material.poisson_ratio < 0.5) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "plate2d.poisson_ratio",
                value: self.material.poisson_ratio.to_string(),
            });
        }
        if geometry.nx < 2
            || geometry.ny < 2
            || !geometry.nx.is_multiple_of(2)
            || !geometry.ny.is_multiple_of(2)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "plate2d.mesh_density",
                value: format!(
                    "nx={}, ny={} (both must be even and >= 2)",
                    geometry.nx, geometry.ny
                ),
            });
        }
        Ok(())
    }

    pub fn analytical_center_deflection_m(&self) -> f64 {
        let a = self.geometry.length_m;
        let b = self.geometry.width_m;
        let pi = std::f64::consts::PI;
        let lambda = (pi / a).powi(2) + (pi / b).powi(2);
        let d_bending = self.material.youngs_modulus_pa * self.geometry.thickness_m.powi(3)
            / (12.0 * (1.0 - self.material.poisson_ratio.powi(2)));
        let shear_modulus =
            self.material.youngs_modulus_pa / (2.0 * (1.0 + self.material.poisson_ratio));
        let shear_rigidity = self.shear_correction * shear_modulus * self.geometry.thickness_m;
        self.pressure_amplitude_pa
            * (1.0 / (d_bending * lambda.powi(2)) + 1.0 / (shear_rigidity * lambda))
    }

    fn pressure(&self, x: f64, y: f64) -> f64 {
        let pi = std::f64::consts::PI;
        self.pressure_amplitude_pa
            * (pi * x / self.geometry.length_m).sin()
            * (pi * y / self.geometry.width_m).sin()
    }
}

/// Validated assembled static response of a Mindlin plate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plate2DStaticResult {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub center_deflection_m: f64,
    pub analytical_center_deflection_m: f64,
    pub analytical_relative_error: f64,
    pub max_deflection_m: f64,
    pub total_applied_load_n: f64,
    pub total_reaction_n: f64,
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub load_integration_relative_error: f64,
    pub strain_energy_j: f64,
    pub energy_balance_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub shear_treatment: PlateShearTreatment,
    /// Full `[w, theta_x, theta_y]` field in Mindlin plate DOF order.
    pub displacement: Vec<[f64; 3]>,
}

/// Solve the simply-supported sinusoidal Mindlin plate benchmark.
pub fn run_plate2d_static(
    study: &Plate2DStudy,
) -> Result<Plate2DStaticResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.geometry.mesh();
    let space = MindlinPlateSpace::new(PolynomialDegree::P1);
    let dofs = space.num_global_dofs(&mesh);
    let form = MindlinPlateForm::with_shear_stabilization(
        study.material.youngs_modulus_pa,
        study.material.poisson_ratio,
        study.geometry.thickness_m,
        study.shear_correction,
        study.shear_treatment.as_fem(),
    );
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 3);
    let unconstrained_stiffness = stiffness.clone();
    let load_form = MindlinPlateLoadForm {
        load_fn: |x, y| study.pressure(x, y),
    };
    let rhs = assemble_linear(&load_form, &space, &mesh, 3);
    let external_rhs = rhs.clone();
    let mut system = FemSystem::new(stiffness, rhs);
    let constrained = simply_supported_dofs(&mesh, study.geometry);
    try_apply_homogeneous_dirichlet_dofs(&mut system, &constrained).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "plate2d boundary",
            message: error.to_string(),
        }
    })?;

    let solved = cg(
        &system.matrix.to_csr(),
        &system.rhs,
        &CgConfig {
            max_iterations: 12_000,
            tolerance: 1.0e-10,
            print_interval: 0,
        },
    );
    if !solved.converged {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::MindlinPlate,
            iterations: solved.iterations,
            residual: solved.residual,
        });
    }

    let displacement = solved
        .x
        .as_chunks::<3>()
        .0
        .iter()
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let internal_force = triplet_matvec(&unconstrained_stiffness, &solved.x);
    let total_applied_load_n = external_rhs.iter().step_by(3).copied().sum::<f64>();
    let total_reaction_n = constrained
        .iter()
        .filter(|dof| **dof % 3 == 0)
        .map(|dof| internal_force[*dof] - external_rhs[*dof])
        .sum::<f64>();
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
    let force_balance_residual_n = (total_applied_load_n + total_reaction_n).abs();
    let expected_load =
        4.0 * study.pressure_amplitude_pa * study.geometry.area_m2() / std::f64::consts::PI.powi(2);
    let load_integration_relative_error =
        (total_applied_load_n - expected_load).abs() / expected_load;
    let internal_energy = 0.5 * dot(&solved.x, &internal_force);
    let rhs_energy = 0.5 * dot(&solved.x, &external_rhs);
    let energy_balance_relative_error =
        (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0);
    let load_scale = total_applied_load_n.abs().max(1.0);
    if !load_integration_relative_error.is_finite() || load_integration_relative_error > 2.0e-3 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "plate2d.load_integration",
            value: load_integration_relative_error,
        });
    }
    if !free_equilibrium_residual_n.is_finite() || free_equilibrium_residual_n > 1.0e-6 * load_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "plate2d.free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !force_balance_residual_n.is_finite() || force_balance_residual_n > 1.0e-6 * load_scale {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "plate2d.force_balance",
            value: force_balance_residual_n,
        });
    }
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1.0e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "plate2d.energy_identity",
            value: energy_balance_relative_error,
        });
    }

    let center_node = (study.geometry.ny / 2) * (study.geometry.nx + 1) + study.geometry.nx / 2;
    let center_deflection_m = solved.x[3 * center_node];
    let analytical_center_deflection_m = study.analytical_center_deflection_m();
    let analytical_relative_error = (center_deflection_m - analytical_center_deflection_m).abs()
        / analytical_center_deflection_m.abs();
    Ok(Plate2DStaticResult {
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs,
        solver_iterations: solved.iterations,
        solver_residual: solved.residual,
        converged: solved.converged,
        center_deflection_m,
        analytical_center_deflection_m,
        analytical_relative_error,
        max_deflection_m: displacement
            .iter()
            .map(|value| value[0].abs())
            .fold(0.0, f64::max),
        total_applied_load_n,
        total_reaction_n,
        free_equilibrium_residual_n,
        force_balance_residual_n,
        load_integration_relative_error,
        strain_energy_j: internal_energy,
        energy_balance_relative_error,
        model_assumption: StructuralModelAssumption::MindlinPlate,
        shear_treatment: study.shear_treatment,
        displacement,
    })
}

fn simply_supported_dofs(mesh: &Mesh, geometry: Plate2DGeometry) -> Vec<usize> {
    let tolerance = 1.0e-10 * geometry.length_m.max(geometry.width_m).max(1.0);
    let mut dofs = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        let left = point.x.abs() <= tolerance;
        let right = (point.x - geometry.length_m).abs() <= tolerance;
        let bottom = point.y.abs() <= tolerance;
        let top = (point.y - geometry.width_m).abs() <= tolerance;
        if left || right || bottom || top {
            dofs.push(3 * node); // w = 0 on every supported edge
        }
        if bottom || top {
            dofs.push(3 * node + 1); // theta_x = 0 on y-normal edges
        }
        if left || right {
            dofs.push(3 * node + 2); // theta_y = 0 on x-normal edges
        }
    }
    dofs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_study() -> Plate2DStudy {
        let mut study = Plate2DStudy::introductory();
        study.geometry.nx = 40;
        study.geometry.ny = 40;
        study
    }

    #[test]
    fn plate_static_matches_navier_and_validates_balance() {
        let result = run_plate2d_static(&small_study()).unwrap();

        assert!(result.converged);
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::MindlinPlate
        );
        assert_eq!(result.nodes, 1681);
        assert_eq!(result.elements, 3200);
        assert!(result.center_deflection_m > 0.0);
        assert!(
            result.analytical_relative_error < 0.02,
            "center={}, exact={}, relative_error={}",
            result.center_deflection_m,
            result.analytical_center_deflection_m,
            result.analytical_relative_error
        );
        assert!(result.load_integration_relative_error < 2.0e-3);
        // The assembled residual is a force norm, so scale the gate with the
        // applied load rather than imposing a mesh/load-dependent absolute N
        // bound (the production validation uses this same normalization).
        assert!(
            result.free_equilibrium_residual_n
                < 1.0e-6 * result.total_applied_load_n.abs().max(1.0)
        );
        assert!(result.force_balance_residual_n < 1.0e-6);
        assert!(result.strain_energy_j > 0.0);
        assert!(result.energy_balance_relative_error < 1.0e-10);
        assert!((result.total_reaction_n + result.total_applied_load_n).abs() < 1.0e-6);
    }

    #[test]
    fn plate_dsg_is_less_locked_than_compatible_form() {
        let mut dsg = small_study();
        dsg.geometry.nx = 10;
        dsg.geometry.ny = 10;
        dsg.geometry.thickness_m = 0.1;
        let stabilized = run_plate2d_static(&dsg).unwrap();

        dsg.shear_treatment = PlateShearTreatment::Consistent;
        let compatible = run_plate2d_static(&dsg).unwrap();
        assert!(stabilized.center_deflection_m > compatible.center_deflection_m);
        assert!(stabilized.analytical_relative_error < compatible.analytical_relative_error);
    }

    #[test]
    fn plate_rejects_invalid_mesh_and_material() {
        let mut study = small_study();
        study.geometry.nx = 11;
        assert!(matches!(
            run_plate2d_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "plate2d.mesh_density",
                ..
            })
        ));

        let mut study = small_study();
        study.material.poisson_ratio = 0.5;
        assert!(matches!(
            run_plate2d_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "plate2d.poisson_ratio",
                ..
            })
        ));
    }
}
