//! Axisymmetric elasticity reference studies.
//!
//! The cylinder case is a genuine meridional FEM solve: a 2D triangular mesh
//! in `(r, z)` is assembled with Sonium FEM's axisymmetric elasticity form.
//! The axial displacement is constrained to zero to represent the plane-strain
//! Lamé solution, while pressure is integrated on the inner and outer radial
//! boundaries.  The result is compared with the closed-form thick-cylinder
//! displacement and stress fields.

use super::{
    StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption, dot,
    free_residual_norm, triplet_matvec,
};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet_dofs};
use sonium_fem::mesh::{Mesh, rectangular_mesh_triangles};
use sonium_fem::physics::solid::axisymmetric::AxisymmetricElasticityForm;
use sonium_fem::physics::solid::elasticity::IsotropicMaterial;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{CgConfig, DiagonalPreconditioner, pcg};

const INNER_BOUNDARY_MARKER: i32 = 1;
const OUTER_BOUNDARY_MARKER: i32 = 2;
const DOFS_PER_NODE: usize = 2;

/// Geometry and meridional discretization of a thick-walled cylinder.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricCylinderGeometry {
    pub inner_radius_m: f64,
    pub outer_radius_m: f64,
    pub length_m: f64,
    pub radial_elements: usize,
    pub axial_elements: usize,
}

impl Default for AxisymmetricCylinderGeometry {
    fn default() -> Self {
        Self {
            inner_radius_m: 0.05,
            outer_radius_m: 0.10,
            length_m: 0.05,
            radial_elements: 24,
            axial_elements: 2,
        }
    }
}

impl AxisymmetricCylinderGeometry {
    fn validate(self) -> Result<(), StructuralMechanicsError> {
        if !self.inner_radius_m.is_finite()
            || !self.outer_radius_m.is_finite()
            || !self.length_m.is_finite()
            || self.inner_radius_m <= 0.0
            || self.outer_radius_m <= self.inner_radius_m
            || self.length_m <= 0.0
            || self.radial_elements == 0
            || self.axial_elements == 0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "axisymmetric_cylinder.geometry",
                value: format!(
                    "inner={}, outer={}, length={}, radial_elements={}, axial_elements={}",
                    self.inner_radius_m,
                    self.outer_radius_m,
                    self.length_m,
                    self.radial_elements,
                    self.axial_elements
                ),
            });
        }
        Ok(())
    }

    fn mesh(self) -> Mesh {
        rectangular_mesh_triangles(
            self.inner_radius_m,
            self.outer_radius_m,
            0.0,
            self.length_m,
            self.radial_elements,
            self.axial_elements,
        )
    }
}

/// Isotropic material for the axisymmetric cylinder reference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricCylinderMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
}

impl Default for AxisymmetricCylinderMaterial {
    fn default() -> Self {
        Self {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.30,
        }
    }
}

impl AxisymmetricCylinderMaterial {
    fn validate(self) -> Result<(), StructuralMechanicsError> {
        if !self.youngs_modulus_pa.is_finite()
            || self.youngs_modulus_pa <= 0.0
            || !self.poisson_ratio.is_finite()
            || !(-1.0..0.5).contains(&self.poisson_ratio)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "axisymmetric_cylinder.material",
                value: format!(
                    "youngs_modulus={}, poisson_ratio={}",
                    self.youngs_modulus_pa, self.poisson_ratio
                ),
            });
        }
        Ok(())
    }

    fn as_fem(self) -> IsotropicMaterial {
        IsotropicMaterial::new(self.youngs_modulus_pa, self.poisson_ratio)
    }

    fn lame(self) -> (f64, f64) {
        let lambda = self.youngs_modulus_pa * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio));
        let mu = self.youngs_modulus_pa / (2.0 * (1.0 + self.poisson_ratio));
        (lambda, mu)
    }
}

/// A validated plane-strain axisymmetric thick-cylinder study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricCylinderStudy {
    pub geometry: AxisymmetricCylinderGeometry,
    pub material: AxisymmetricCylinderMaterial,
    pub internal_pressure_pa: f64,
    pub external_pressure_pa: f64,
}

impl Default for AxisymmetricCylinderStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl AxisymmetricCylinderStudy {
    pub fn introductory() -> Self {
        Self {
            geometry: AxisymmetricCylinderGeometry::default(),
            material: AxisymmetricCylinderMaterial::default(),
            internal_pressure_pa: 1.0e6,
            external_pressure_pa: 0.0,
        }
    }

    fn validate(self) -> Result<(), StructuralMechanicsError> {
        self.geometry.validate()?;
        self.material.validate()?;
        if !self.internal_pressure_pa.is_finite()
            || !self.external_pressure_pa.is_finite()
            || self.internal_pressure_pa < 0.0
            || self.external_pressure_pa < 0.0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "axisymmetric_cylinder.pressure",
                value: format!(
                    "internal={}, external={}",
                    self.internal_pressure_pa, self.external_pressure_pa
                ),
            });
        }
        Ok(())
    }
}

/// Quality and provenance data from the axisymmetric cylinder solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricCylinderResult {
    pub study: AxisymmetricCylinderStudy,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub free_equilibrium_residual_n: f64,
    pub strain_energy_j: f64,
    pub energy_balance_relative_error: f64,
    pub max_radial_displacement_relative_error: f64,
    pub max_radial_stress_relative_error: f64,
    pub max_hoop_stress_relative_error: f64,
    pub inner_radial_displacement_m: f64,
    pub exact_inner_radial_displacement_m: f64,
    pub outer_radial_displacement_m: f64,
    pub exact_outer_radial_displacement_m: f64,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

#[derive(Debug, Clone, Copy)]
struct LameSolution {
    constant_stress_pa: f64,
    inverse_square_stress_pa_m2: f64,
    displacement_a: f64,
    displacement_b_m3: f64,
}

impl LameSolution {
    fn displacement(self, radius_m: f64) -> f64 {
        self.displacement_a * radius_m + self.displacement_b_m3 / radius_m
    }

    fn radial_stress(self, radius_m: f64) -> f64 {
        self.constant_stress_pa - self.inverse_square_stress_pa_m2 / radius_m.powi(2)
    }

    fn hoop_stress(self, radius_m: f64) -> f64 {
        self.constant_stress_pa + self.inverse_square_stress_pa_m2 / radius_m.powi(2)
    }
}

fn lame_solution(study: AxisymmetricCylinderStudy) -> LameSolution {
    let (lambda, mu) = study.material.lame();
    let ri = study.geometry.inner_radius_m;
    let ro = study.geometry.outer_radius_m;
    let pressure_difference = study.internal_pressure_pa - study.external_pressure_pa;
    let inverse_square_stress_pa_m2 = pressure_difference / (ri.powi(-2) - ro.powi(-2));
    let constant_stress_pa = -study.external_pressure_pa + inverse_square_stress_pa_m2 / ro.powi(2);
    LameSolution {
        constant_stress_pa,
        inverse_square_stress_pa_m2,
        displacement_a: constant_stress_pa / (2.0 * (lambda + mu)),
        displacement_b_m3: inverse_square_stress_pa_m2 / (2.0 * mu),
    }
}

fn pressure_rhs(mesh: &Mesh, study: AxisymmetricCylinderStudy) -> Vec<f64> {
    let mut rhs = vec![0.0; mesh.num_nodes() * DOFS_PER_NODE];
    for boundary in &mesh.boundaries {
        let pressure = match boundary.marker {
            INNER_BOUNDARY_MARKER => study.internal_pressure_pa,
            OUTER_BOUNDARY_MARKER => -study.external_pressure_pa,
            _ => continue,
        };
        if pressure.abs() <= f64::EPSILON {
            continue;
        }
        for segment in boundary.nodes.windows(2) {
            let [first, second] = [segment[0], segment[1]];
            let p0 = mesh.nodes[first];
            let p1 = mesh.nodes[second];
            let edge_length = (p1.x - p0.x).hypot(p1.y - p0.y);
            let radius = 0.5 * (p0.x + p1.x).abs();
            let nodal_load = pressure * std::f64::consts::TAU * radius * edge_length * 0.5;
            rhs[first * DOFS_PER_NODE] += nodal_load;
            rhs[second * DOFS_PER_NODE] += nodal_load;
        }
    }
    rhs
}

/// Run the assembled axisymmetric thick-cylinder reference solve.
pub fn run_axisymmetric_cylinder(
    study: &AxisymmetricCylinderStudy,
) -> Result<AxisymmetricCylinderResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.geometry.mesh();
    let space = VectorH1Space::new(PolynomialDegree::P1, DOFS_PER_NODE);
    let stiffness = assemble_bilinear(
        &AxisymmetricElasticityForm::new(study.material.as_fem()),
        &space,
        &space,
        &mesh,
        3,
    );
    let unconstrained_stiffness = stiffness.clone();
    let external_rhs = pressure_rhs(&mesh, *study);
    let mut system = FemSystem::new(stiffness, external_rhs.clone());
    let axial_dofs = (0..mesh.num_nodes())
        .map(|node| node * DOFS_PER_NODE + 1)
        .collect::<Vec<_>>();
    try_apply_homogeneous_dirichlet_dofs(&mut system, &axial_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "axisymmetric cylinder boundary",
            message: error.to_string(),
        }
    })?;
    let matrix = system.matrix.to_csr();
    // Use a dimensionless Jacobi preconditioner. Its uniform scaling leaves
    // PCG unchanged in exact arithmetic, while avoiding tiny preconditioned
    // dot products caused by SI stiffness magnitudes. Physical equations,
    // relative convergence tolerance and final force checks remain unchanged.
    let stiffness_scale = matrix
        .values
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let mut normalized = matrix.clone();
    for value in &mut normalized.values {
        *value /= stiffness_scale;
    }
    let preconditioner = DiagonalPreconditioner::from_csr(&normalized);
    let mut solved = pcg(
        &matrix,
        &preconditioner,
        &system.rhs,
        &CgConfig {
            max_iterations: 20_000,
            tolerance: 1.0e-11,
            print_interval: 0,
        },
    );
    let rhs_norm = system
        .rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let mut verified_residual = f64::INFINITY;
    // Restart on the independently recomputed defect to recover the final
    // digits lost in CG's recurrence. Correction solves need only reduce that
    // defect; the original 1e-11 physical-system tolerance remains the gate.
    for refinement in 0..=3 {
        let applied = matrix.matvec(&solved.x);
        let defect: Vec<_> = system
            .rhs
            .iter()
            .zip(applied)
            .map(|(b, ax)| b - ax)
            .collect();
        let defect_norm = defect.iter().map(|value| value * value).sum::<f64>().sqrt();
        verified_residual = if rhs_norm > 0.0 {
            defect_norm / rhs_norm
        } else {
            defect_norm
        };
        if verified_residual < 1e-11 || !verified_residual.is_finite() || refinement == 3 {
            break;
        }
        let correction = pcg(
            &matrix,
            &preconditioner,
            &defect,
            &CgConfig {
                max_iterations: 20_000,
                tolerance: 1e-3,
                print_interval: 0,
            },
        );
        for (value, change) in solved.x.iter_mut().zip(correction.x) {
            *value += change;
        }
        solved.iterations += correction.iterations;
    }
    if !verified_residual.is_finite() || verified_residual >= 1e-11 {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::AxisymmetricElasticity,
            iterations: solved.iterations,
            residual: verified_residual,
        });
    }

    let internal_force = triplet_matvec(&unconstrained_stiffness, &solved.x);
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
    let strain_energy_j = 0.5 * dot(&solved.x, &internal_force);
    let rhs_energy = 0.5 * dot(&solved.x, &external_rhs);
    let energy_balance_relative_error =
        (strain_energy_j - rhs_energy).abs() / rhs_energy.abs().max(1.0);
    let load_scale = external_rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    if !free_equilibrium_residual_n.is_finite() || free_equilibrium_residual_n > 1.0e-8 * load_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "axisymmetric_cylinder.free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1.0e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "axisymmetric_cylinder.energy_identity",
            value: energy_balance_relative_error,
        });
    }

    let exact = lame_solution(*study);
    let mut max_radial_displacement_relative_error: f64 = 0.0;
    for (node, point) in mesh.nodes.iter().enumerate() {
        let numerical = solved.x[node * DOFS_PER_NODE];
        let reference = exact.displacement(point.x);
        max_radial_displacement_relative_error = max_radial_displacement_relative_error
            .max((numerical - reference).abs() / reference.abs().max(1.0e-30));
    }

    let pressure_scale = study
        .internal_pressure_pa
        .max(study.external_pressure_pa)
        .max(1.0);
    let mut max_radial_stress_relative_error: f64 = 0.0;
    let mut max_hoop_stress_relative_error: f64 = 0.0;
    for element in &mesh.elements {
        let nodes = element.vertices();
        if nodes.len() != 3 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "axisymmetric cylinder mesh contains a non-triangular element".to_string(),
            });
        }
        let gradients = super::triangle_shape_gradients(&mesh, nodes);
        let mut radial_gradient = 0.0;
        let mut radial_displacement = 0.0;
        let mut radius = 0.0;
        for (local, &node) in nodes.iter().enumerate() {
            radial_gradient += solved.x[node * DOFS_PER_NODE] * gradients[local][0];
            radial_displacement += solved.x[node * DOFS_PER_NODE] / 3.0;
            radius += mesh.nodes[node].x / 3.0;
        }
        let hoop_strain = radial_displacement / radius.max(1.0e-30);
        let radial_strain = radial_gradient;
        let (lambda, mu) = study.material.lame();
        let radial_stress = (lambda + 2.0 * mu) * radial_strain + lambda * hoop_strain;
        let hoop_stress = lambda * radial_strain + (lambda + 2.0 * mu) * hoop_strain;
        let reference_radial_stress = exact.radial_stress(radius);
        let reference_hoop_stress = exact.hoop_stress(radius);
        // Use the applied pressure as the stress scale so the zero radial
        // stress at the outer wall does not turn roundoff into a false gate.
        max_radial_stress_relative_error = max_radial_stress_relative_error
            .max((radial_stress - reference_radial_stress).abs() / pressure_scale);
        max_hoop_stress_relative_error = max_hoop_stress_relative_error
            .max((hoop_stress - reference_hoop_stress).abs() / pressure_scale);
    }

    let mid_row = study.geometry.axial_elements / 2;
    let inner_node = mid_row * (study.geometry.radial_elements + 1);
    let outer_node = inner_node + study.geometry.radial_elements;
    Ok(AxisymmetricCylinderResult {
        study: *study,
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs: space.num_global_dofs(&mesh),
        solver_iterations: solved.iterations,
        solver_residual: verified_residual,
        converged: true,
        free_equilibrium_residual_n,
        strain_energy_j,
        energy_balance_relative_error,
        max_radial_displacement_relative_error,
        max_radial_stress_relative_error,
        max_hoop_stress_relative_error,
        inner_radial_displacement_m: solved.x[inner_node * DOFS_PER_NODE],
        exact_inner_radial_displacement_m: exact.displacement(study.geometry.inner_radius_m),
        outer_radial_displacement_m: solved.x[outer_node * DOFS_PER_NODE],
        exact_outer_radial_displacement_m: exact.displacement(study.geometry.outer_radius_m),
        model_assumption: StructuralModelAssumption::AxisymmetricElasticity,
        execution: StructuralExecutionProvenance::cpu_axisymmetric_reference(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refined_axisymmetric_solve_preserves_accuracy_under_stiffness_scaling() {
        for modulus in [20e9, 2000e9] {
            let mut study = AxisymmetricCylinderStudy::introductory();
            study.geometry.radial_elements = 32;
            study.geometry.axial_elements = 32;
            study.material.youngs_modulus_pa = modulus;
            let result =
                run_axisymmetric_cylinder(&study).expect("scaled refined solve should converge");
            assert!(result.solver_residual < 1e-11);
            assert!(result.free_equilibrium_residual_n < 1e-6);
            assert!(result.energy_balance_relative_error < 1e-10);
            assert!(result.max_radial_displacement_relative_error < 0.005);
        }
    }

    #[test]
    fn axisymmetric_cylinder_matches_lame_solution_and_balance() {
        let mut study = AxisymmetricCylinderStudy::introductory();
        study.geometry.radial_elements = 12;
        let result = run_axisymmetric_cylinder(&study).expect("axisymmetric FEM should converge");
        assert!(result.converged);
        assert!(result.free_equilibrium_residual_n < 1.0e-6);
        assert!(result.energy_balance_relative_error < 1.0e-10);
        assert!(
            result.max_radial_displacement_relative_error < 0.02,
            "{result:#?}"
        );
        assert!(
            result.max_radial_stress_relative_error < 0.10,
            "{result:#?}"
        );
        assert!(result.max_hoop_stress_relative_error < 0.10, "{result:#?}");
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::AxisymmetricElasticity
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.axisymmetric-cylinder.v1"
        );
    }

    #[test]
    fn axisymmetric_cylinder_rejects_invalid_geometry_and_pressure() {
        let mut study = AxisymmetricCylinderStudy::introductory();
        study.geometry.inner_radius_m = study.geometry.outer_radius_m;
        assert!(matches!(
            run_axisymmetric_cylinder(&study),
            Err(StructuralMechanicsError::InvalidConfiguration { .. })
        ));
        let mut study = AxisymmetricCylinderStudy::introductory();
        study.internal_pressure_pa = -1.0;
        assert!(matches!(
            run_axisymmetric_cylinder(&study),
            Err(StructuralMechanicsError::InvalidConfiguration { .. })
        ));
        let mut study = AxisymmetricCylinderStudy::introductory();
        study.material.youngs_modulus_pa = 0.0;
        assert!(matches!(
            run_axisymmetric_cylinder(&study),
            Err(StructuralMechanicsError::InvalidConfiguration { .. })
        ));
        let mut study = AxisymmetricCylinderStudy::introductory();
        study.material.poisson_ratio = 0.5;
        assert!(matches!(
            run_axisymmetric_cylinder(&study),
            Err(StructuralMechanicsError::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn pressure_rhs_matches_inner_and_outer_surface_resultant() {
        let mut study = AxisymmetricCylinderStudy::introductory();
        study.external_pressure_pa = 0.25e6;
        let mesh = study.geometry.mesh();
        let rhs = pressure_rhs(&mesh, study);
        let radial_resultant = rhs.iter().step_by(DOFS_PER_NODE).sum::<f64>();
        let expected = std::f64::consts::TAU
            * study.geometry.length_m
            * (study.internal_pressure_pa * study.geometry.inner_radius_m
                - study.external_pressure_pa * study.geometry.outer_radius_m);
        assert!((radial_resultant - expected).abs() < 1.0e-9 * expected.abs().max(1.0));
    }
}
