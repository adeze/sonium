//! Nonlinear axisymmetric magnetostatic bias solves.
//!
//! The unknown is the azimuthal vector potential `A_theta(r,z)`. Constitutive
//! response is represented by monotone `B-H` data and differentiated
//! consistently, so Newton's tangent is the exact derivative of the residual.

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature::QuadratureRule;
use crate::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{LinearSolveConfig, LinearSolverType, NewtonConfig, NewtonError, solve_real};
use std::collections::{BTreeMap, HashSet};
use std::f64::consts::PI;

/// Strictly monotone first-quadrant magnetic constitutive curve.
#[derive(Debug, Clone, PartialEq)]
pub struct BhCurve {
    points: Vec<(f64, f64)>,
}

impl BhCurve {
    pub fn new(points: Vec<(f64, f64)>) -> Result<Self, BhCurveError> {
        if points.len() < 2 {
            return Err(BhCurveError::TooFewPoints);
        }
        if points[0] != (0.0, 0.0) {
            return Err(BhCurveError::MustStartAtOrigin);
        }
        for (index, &(b, h)) in points.iter().enumerate() {
            if !b.is_finite() || !h.is_finite() || b < 0.0 || h < 0.0 {
                return Err(BhCurveError::InvalidPoint { index });
            }
        }
        for index in 1..points.len() {
            if points[index].0 <= points[index - 1].0 {
                return Err(BhCurveError::FluxDensityNotIncreasing { index });
            }
            if points[index].1 <= points[index - 1].1 {
                return Err(BhCurveError::FieldStrengthNotIncreasing { index });
            }
        }
        Ok(Self { points })
    }

    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }

    /// Field magnitude `H(B)` using positive-slope piecewise-linear segments.
    pub fn field_strength(&self, flux_density_t: f64) -> f64 {
        let b = flux_density_t.abs();
        let segment = self.segment(b);
        let (b0, h0) = self.points[segment];
        h0 + self.segment_slope(segment) * (b - b0)
    }

    /// Secant reluctivity `nu = H/B`, with the origin defined by the first slope.
    pub fn reluctivity(&self, flux_density_t: f64) -> f64 {
        let b = flux_density_t.abs();
        if b <= f64::EPSILON {
            self.segment_slope(0)
        } else {
            self.field_strength(b) / b
        }
    }

    /// Differential reluctivity `dH/dB` used in the consistent Newton tangent.
    pub fn differential_reluctivity(&self, flux_density_t: f64) -> f64 {
        self.segment_slope(self.segment(flux_density_t.abs()))
    }

    /// Magnetic energy density `integral_0^B H(b) db` in J/m^3.
    pub fn energy_density(&self, flux_density_t: f64) -> f64 {
        let b = flux_density_t.abs();
        let segment = self.segment(b);
        let mut energy = 0.0;
        for index in 0..segment {
            let (b0, h0) = self.points[index];
            let (b1, h1) = self.points[index + 1];
            energy += 0.5 * (h0 + h1) * (b1 - b0);
        }
        let (b0, h0) = self.points[segment];
        let db = b - b0;
        energy + h0 * db + 0.5 * self.segment_slope(segment) * db * db
    }

    pub fn coenergy_density(&self, flux_density_t: f64) -> f64 {
        let b = flux_density_t.abs();
        b * self.field_strength(b) - self.energy_density(b)
    }

    fn segment(&self, b: f64) -> usize {
        let insertion = self.points.partition_point(|&(point_b, _)| point_b <= b);
        insertion.saturating_sub(1).min(self.points.len() - 2)
    }

    fn segment_slope(&self, segment: usize) -> f64 {
        let (b0, h0) = self.points[segment];
        let (b1, h1) = self.points[segment + 1];
        (h1 - h0) / (b1 - b0)
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BhCurveError {
    #[error("a B-H curve requires at least two points")]
    TooFewPoints,
    #[error("a B-H curve must start at (0, 0)")]
    MustStartAtOrigin,
    #[error("B-H point {index} is non-finite or outside the first quadrant")]
    InvalidPoint { index: usize },
    #[error("flux density must increase strictly at B-H point {index}")]
    FluxDensityNotIncreasing { index: usize },
    #[error("field strength must increase strictly at B-H point {index}")]
    FieldStrengthNotIncreasing { index: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub enum MagneticConstitutiveLaw {
    Linear { permeability_h_per_m: f64 },
    Nonlinear(BhCurve),
}

impl MagneticConstitutiveLaw {
    pub fn linear(permeability_h_per_m: f64) -> Result<Self, NonlinearMagneticError> {
        if !permeability_h_per_m.is_finite() || permeability_h_per_m <= 0.0 {
            return Err(NonlinearMagneticError::InvalidPermeability);
        }
        Ok(Self::Linear {
            permeability_h_per_m,
        })
    }

    pub fn reluctivity(&self, b: f64) -> f64 {
        match self {
            Self::Linear {
                permeability_h_per_m,
            } => 1.0 / permeability_h_per_m,
            Self::Nonlinear(curve) => curve.reluctivity(b),
        }
    }

    pub fn differential_reluctivity(&self, b: f64) -> f64 {
        match self {
            Self::Linear {
                permeability_h_per_m,
            } => 1.0 / permeability_h_per_m,
            Self::Nonlinear(curve) => curve.differential_reluctivity(b),
        }
    }

    pub fn energy_density(&self, b: f64) -> f64 {
        match self {
            Self::Linear {
                permeability_h_per_m,
            } => 0.5 * b * b / permeability_h_per_m,
            Self::Nonlinear(curve) => curve.energy_density(b),
        }
    }

    pub fn coenergy_density(&self, b: f64) -> f64 {
        match self {
            Self::Linear {
                permeability_h_per_m,
            } => 0.5 * b * b / permeability_h_per_m,
            Self::Nonlinear(curve) => curve.coenergy_density(b),
        }
    }

    fn h_and_tangent(&self, b: [f64; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
        let magnitude = b[0].hypot(b[1]);
        let nu = self.reluctivity(magnitude);
        let differential = self.differential_reluctivity(magnitude);
        let h = [nu * b[0], nu * b[1]];
        if magnitude <= 1.0e-14 {
            return (h, [[differential, 0.0], [0.0, differential]]);
        }
        let radial = [b[0] / magnitude, b[1] / magnitude];
        let correction = differential - nu;
        (
            h,
            [
                [
                    nu + correction * radial[0] * radial[0],
                    correction * radial[0] * radial[1],
                ],
                [
                    correction * radial[1] * radial[0],
                    nu + correction * radial[1] * radial[1],
                ],
            ],
        )
    }
}

/// Region-aware nonlinear `A_theta` problem.
pub struct AxisymmetricNonlinearMagneticProblem {
    pub mesh: Mesh,
    pub materials: BTreeMap<i32, MagneticConstitutiveLaw>,
    pub azimuthal_current_density_a_per_m2: BTreeMap<i32, f64>,
    pub quadrature_order: usize,
}

impl AxisymmetricNonlinearMagneticProblem {
    pub fn validate(&self) -> Result<(), NonlinearMagneticError> {
        if self.quadrature_order == 0 {
            return Err(NonlinearMagneticError::InvalidQuadratureOrder);
        }
        for (element_index, element) in self.mesh.elements.iter().enumerate() {
            if !self.materials.contains_key(&element.marker) {
                return Err(NonlinearMagneticError::MissingMaterial {
                    element_index,
                    marker: element.marker,
                });
            }
        }
        for (&marker, &source) in &self.azimuthal_current_density_a_per_m2 {
            if !source.is_finite() {
                return Err(NonlinearMagneticError::InvalidCurrentDensity { marker });
            }
        }
        Ok(())
    }

    pub fn num_dofs(&self) -> usize {
        self.mesh.num_nodes()
    }

    /// Assemble the nonlinear residual and exact material tangent.
    pub fn assemble(
        &self,
        potential: &[f64],
        load_factor: f64,
    ) -> Result<(Vec<f64>, TripletMatrix<f64>), NonlinearMagneticError> {
        self.validate()?;
        if potential.len() != self.num_dofs() {
            return Err(NonlinearMagneticError::DimensionMismatch {
                expected: self.num_dofs(),
                actual: potential.len(),
            });
        }
        if !load_factor.is_finite() || !(0.0..=1.0).contains(&load_factor) {
            return Err(NonlinearMagneticError::InvalidLoadFactor);
        }
        let space = H1Space::new(PolynomialDegree::P1);
        let mut residual = vec![0.0; self.num_dofs()];
        let mut tangent = TripletMatrix::new(self.num_dofs(), self.num_dofs());
        for element_index in 0..self.mesh.num_elements() {
            let element = &self.mesh.elements[element_index];
            let material = &self.materials[&element.marker];
            let source = self
                .azimuthal_current_density_a_per_m2
                .get(&element.marker)
                .copied()
                .unwrap_or(0.0)
                * load_factor;
            let dofs = space.local_to_global(&self.mesh, element_index);
            let quadrature = QuadratureRule::new(element.element_type, self.quadrature_order);
            for point in quadrature.iter() {
                let tab = space.tabulate(
                    &self.mesh,
                    element_index,
                    point.xi(),
                    point.eta(),
                    point.zeta(),
                );
                let radius = tab.physical_coords[0].abs().max(1.0e-12);
                let weight = 2.0 * PI * radius * tab.det_j * point.weight;
                let basis_b: Vec<_> = (0..dofs.len())
                    .map(|local| {
                        [
                            -tab.grad(local, 1),
                            tab.grad(local, 0) + tab.value(local) / radius,
                        ]
                    })
                    .collect();
                let mut b = [0.0; 2];
                let mut a = 0.0;
                for (local, &dof) in dofs.iter().enumerate() {
                    b[0] += potential[dof] * basis_b[local][0];
                    b[1] += potential[dof] * basis_b[local][1];
                    a += potential[dof] * tab.value(local);
                }
                let (h, constitutive_tangent) = material.h_and_tangent(b);
                for (local_i, &global_i) in dofs.iter().enumerate() {
                    let delta_b_i = basis_b[local_i];
                    residual[global_i] += (h[0] * delta_b_i[0] + h[1] * delta_b_i[1]
                        - source * tab.value(local_i))
                        * weight;
                    for (local_j, &global_j) in dofs.iter().enumerate() {
                        let delta_b_j = basis_b[local_j];
                        let db_h = [
                            constitutive_tangent[0][0] * delta_b_j[0]
                                + constitutive_tangent[0][1] * delta_b_j[1],
                            constitutive_tangent[1][0] * delta_b_j[0]
                                + constitutive_tangent[1][1] * delta_b_j[1],
                        ];
                        tangent.push(
                            global_i,
                            global_j,
                            (delta_b_i[0] * db_h[0] + delta_b_i[1] * db_h[1]) * weight,
                        );
                    }
                }
                let _ = a;
            }
        }
        Ok((residual, tangent))
    }

    pub fn quadrature_state(
        &self,
        potential: &[f64],
    ) -> Result<Vec<MagneticBiasQuadraturePoint>, NonlinearMagneticError> {
        self.validate()?;
        if potential.len() != self.num_dofs() {
            return Err(NonlinearMagneticError::DimensionMismatch {
                expected: self.num_dofs(),
                actual: potential.len(),
            });
        }
        let space = H1Space::new(PolynomialDegree::P1);
        let mut state = Vec::new();
        for element_index in 0..self.mesh.num_elements() {
            let element = &self.mesh.elements[element_index];
            let material = &self.materials[&element.marker];
            let dofs = space.local_to_global(&self.mesh, element_index);
            let quadrature = QuadratureRule::new(element.element_type, self.quadrature_order);
            for (quadrature_index, point) in quadrature.iter().enumerate() {
                let tab = space.tabulate(
                    &self.mesh,
                    element_index,
                    point.xi(),
                    point.eta(),
                    point.zeta(),
                );
                let radius = tab.physical_coords[0].abs().max(1.0e-12);
                let mut b = [0.0; 2];
                let mut potential_at_point = 0.0;
                for (local, &dof) in dofs.iter().enumerate() {
                    b[0] -= potential[dof] * tab.grad(local, 1);
                    b[1] += potential[dof] * (tab.grad(local, 0) + tab.value(local) / radius);
                    potential_at_point += potential[dof] * tab.value(local);
                }
                let magnitude = b[0].hypot(b[1]);
                let nu = material.reluctivity(magnitude);
                state.push(MagneticBiasQuadraturePoint {
                    element_index,
                    quadrature_index,
                    marker: element.marker,
                    position_rz_m: [tab.physical_coords[0], tab.physical_coords[1]],
                    integration_weight_m3: 2.0 * PI * radius * tab.det_j * point.weight,
                    vector_potential_atheta: potential_at_point,
                    flux_density_rz_t: b,
                    field_strength_rz_a_per_m: [nu * b[0], nu * b[1]],
                    reluctivity_m_per_h: nu,
                    differential_reluctivity_m_per_h: material.differential_reluctivity(magnitude),
                    energy_density_j_per_m3: material.energy_density(magnitude),
                    coenergy_density_j_per_m3: material.coenergy_density(magnitude),
                });
            }
        }
        Ok(state)
    }

    pub fn energy_report(&self, state: &[MagneticBiasQuadraturePoint]) -> MagneticEnergyReport {
        let mut energy_j = 0.0;
        let mut coenergy_j = 0.0;
        let mut source_work_j = 0.0;
        for point in state {
            energy_j += point.energy_density_j_per_m3 * point.integration_weight_m3;
            coenergy_j += point.coenergy_density_j_per_m3 * point.integration_weight_m3;
            let source = self
                .azimuthal_current_density_a_per_m2
                .get(&point.marker)
                .copied()
                .unwrap_or(0.0);
            source_work_j += source * point.vector_potential_atheta * point.integration_weight_m3;
        }
        MagneticEnergyReport {
            energy_j,
            coenergy_j,
            source_work_j,
            balance_error_j: source_work_j - energy_j - coenergy_j,
        }
    }

    pub fn extract_force_factor(
        &self,
        state: &[MagneticBiasQuadraturePoint],
        coil_marker: i32,
        active_turns: f64,
    ) -> Result<f64, NonlinearMagneticError> {
        if !active_turns.is_finite() || active_turns <= 0.0 {
            return Err(NonlinearMagneticError::InvalidActiveTurns);
        }
        let mut weighted_radial_flux = 0.0;
        let mut weight = 0.0;
        for point in state.iter().filter(|point| point.marker == coil_marker) {
            weighted_radial_flux += point.flux_density_rz_t[0] * point.integration_weight_m3;
            weight += point.integration_weight_m3;
        }
        if weight == 0.0 {
            return Err(NonlinearMagneticError::MissingCoilRegion { coil_marker });
        }
        let mean_radius = state
            .iter()
            .filter(|point| point.marker == coil_marker)
            .map(|point| point.position_rz_m[0] * point.integration_weight_m3)
            .sum::<f64>()
            / weight;
        Ok(2.0 * PI * mean_radius * active_turns * weighted_radial_flux.abs() / weight)
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearBiasSolveConfig {
    pub continuation_steps: usize,
    pub newton: NewtonConfig,
    pub inner_solver: LinearSolveConfig,
}

impl Default for NonlinearBiasSolveConfig {
    fn default() -> Self {
        Self {
            continuation_steps: 8,
            newton: NewtonConfig {
                max_iterations: 40,
                absolute_tolerance: 1.0e-9,
                relative_tolerance: 1.0e-9,
                line_search: sonium_solvers::LineSearch::backtracking(),
                ..NewtonConfig::default()
            },
            inner_solver: LinearSolveConfig {
                solver: LinearSolverType::GmresIlu,
                max_iterations: 2_000,
                // The inner solve is deliberately inexact: asking ILU/GMRES
                // for more accuracy than this makes otherwise adequate
                // Newton updates fail at the finite-precision Krylov
                // breakdown floor.
                tolerance: 1.0e-9,
                ..LinearSolveConfig::default()
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct MagneticBiasQuadraturePoint {
    pub element_index: usize,
    pub quadrature_index: usize,
    pub marker: i32,
    pub position_rz_m: [f64; 2],
    pub integration_weight_m3: f64,
    pub vector_potential_atheta: f64,
    pub flux_density_rz_t: [f64; 2],
    pub field_strength_rz_a_per_m: [f64; 2],
    pub reluctivity_m_per_h: f64,
    pub differential_reluctivity_m_per_h: f64,
    pub energy_density_j_per_m3: f64,
    pub coenergy_density_j_per_m3: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagneticEnergyReport {
    pub energy_j: f64,
    pub coenergy_j: f64,
    pub source_work_j: f64,
    pub balance_error_j: f64,
}

#[derive(Debug, Clone)]
pub struct NonlinearMagneticBiasSolution {
    pub vector_potential_atheta: Vec<f64>,
    pub quadrature_state: Vec<MagneticBiasQuadraturePoint>,
    pub continuation_iterations: Vec<usize>,
    pub final_residual: f64,
    pub energy: MagneticEnergyReport,
}

pub fn solve_nonlinear_axisymmetric_bias(
    problem: &AxisymmetricNonlinearMagneticProblem,
    dirichlet_dofs: &[usize],
    config: &NonlinearBiasSolveConfig,
) -> Result<NonlinearMagneticBiasSolution, NonlinearMagneticError> {
    problem.validate()?;
    if config.continuation_steps == 0 {
        return Err(NonlinearMagneticError::InvalidContinuationSteps);
    }
    let constrained: HashSet<_> = dirichlet_dofs.iter().copied().collect();
    if constrained.iter().any(|&dof| dof >= problem.num_dofs()) {
        return Err(NonlinearMagneticError::InvalidDirichletDof);
    }
    let mut potential = vec![0.0; problem.num_dofs()];
    let mut continuation_iterations = Vec::with_capacity(config.continuation_steps);
    let mut final_residual = f64::INFINITY;
    for step in 1..=config.continuation_steps {
        let load_factor = step as f64 / config.continuation_steps as f64;
        let result = sonium_solvers::newton_raphson(
            &potential,
            |trial| {
                let (mut residual, _) = problem
                    .assemble(trial, load_factor)
                    .expect("validated nonlinear magnetic assembly");
                for &dof in &constrained {
                    residual[dof] = 0.0;
                }
                residual
            },
            |trial, rhs| {
                let (_, mut tangent) = problem
                    .assemble(trial, load_factor)
                    .map_err(|error| error.to_string())?;
                apply_constraints(&mut tangent, &constrained);
                let mut constrained_rhs = rhs.to_vec();
                for &dof in &constrained {
                    constrained_rhs[dof] = 0.0;
                }
                solve_real(&tangent.to_csr(), &constrained_rhs, &config.inner_solver)
                    .map(|solution| solution.x)
                    .map_err(|error| error.to_string())
            },
            &config.newton,
        )?;
        potential = result.x;
        for &dof in &constrained {
            potential[dof] = 0.0;
        }
        continuation_iterations.push(result.iterations);
        final_residual = result.residual;
    }
    let quadrature_state = problem.quadrature_state(&potential)?;
    let energy = problem.energy_report(&quadrature_state);
    Ok(NonlinearMagneticBiasSolution {
        vector_potential_atheta: potential,
        quadrature_state,
        continuation_iterations,
        final_residual,
        energy,
    })
}

fn apply_constraints(matrix: &mut TripletMatrix<f64>, constrained: &HashSet<usize>) {
    for index in 0..matrix.nnz() {
        if constrained.contains(&matrix.rows[index]) || constrained.contains(&matrix.cols[index]) {
            matrix.values[index] = 0.0;
        }
    }
    for &dof in constrained {
        matrix.push(dof, dof, 1.0);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NonlinearMagneticError {
    #[error("magnetic permeability must be finite and positive")]
    InvalidPermeability,
    #[error("quadrature order must be positive")]
    InvalidQuadratureOrder,
    #[error("cell {element_index} has no magnetic material for marker {marker}")]
    MissingMaterial { element_index: usize, marker: i32 },
    #[error("current density for marker {marker} must be finite")]
    InvalidCurrentDensity { marker: i32 },
    #[error("potential has {actual} DOFs; expected {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("continuation load factor must lie in [0, 1]")]
    InvalidLoadFactor,
    #[error("continuation requires at least one load step")]
    InvalidContinuationSteps,
    #[error("Dirichlet DOF is outside the problem")]
    InvalidDirichletDof,
    #[error("active turns must be finite and positive")]
    InvalidActiveTurns,
    #[error("mesh has no quadrature points in coil marker {coil_marker}")]
    MissingCoilRegion { coil_marker: i32 },
    #[error(transparent)]
    Newton(#[from] NewtonError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::rectangular_mesh_triangles;
    use crate::physics::electromagnetics::magnetostatics::MU_0;

    fn steel_curve() -> BhCurve {
        BhCurve::new(vec![
            (0.0, 0.0),
            (0.5, 100.0),
            (1.0, 300.0),
            (1.4, 1_000.0),
            (1.7, 8_000.0),
            (2.0, 80_000.0),
        ])
        .unwrap()
    }

    #[test]
    fn bh_curve_is_monotone_and_energy_coenergy_are_conjugate() {
        let curve = steel_curve();
        for pair in [0.1, 0.75, 1.2, 1.8, 2.2] {
            let h = curve.field_strength(pair);
            assert!(curve.reluctivity(pair) > 0.0);
            assert!(curve.differential_reluctivity(pair) > 0.0);
            assert!(
                (curve.energy_density(pair) + curve.coenergy_density(pair) - pair * h).abs()
                    < 1.0e-9
            );
        }
        assert!(matches!(
            BhCurve::new(vec![(0.0, 0.0), (1.0, 2.0), (0.5, 3.0)]),
            Err(BhCurveError::FluxDensityNotIncreasing { .. })
        ));
    }

    #[test]
    fn consistent_tangent_matches_directional_finite_difference() {
        let mut mesh = rectangular_mesh_triangles(0.01, 0.03, -0.01, 0.01, 2, 2);
        for element in &mut mesh.elements {
            element.marker = 1;
        }
        let problem = AxisymmetricNonlinearMagneticProblem {
            mesh,
            materials: BTreeMap::from([(1, MagneticConstitutiveLaw::Nonlinear(steel_curve()))]),
            azimuthal_current_density_a_per_m2: BTreeMap::new(),
            quadrature_order: 2,
        };
        let potential: Vec<_> = (0..problem.num_dofs())
            .map(|index| 1.0e-3 * index as f64)
            .collect();
        let direction: Vec<_> = (0..problem.num_dofs())
            .map(|index| 0.25 - 0.03 * index as f64)
            .collect();
        let (residual, tangent) = problem.assemble(&potential, 1.0).unwrap();
        let epsilon = 1.0e-7;
        let perturbed: Vec<_> = potential
            .iter()
            .zip(&direction)
            .map(|(a, d)| a + epsilon * d)
            .collect();
        let (next_residual, _) = problem.assemble(&perturbed, 1.0).unwrap();
        let mut tangent_direction = vec![0.0; problem.num_dofs()];
        for index in 0..tangent.nnz() {
            tangent_direction[tangent.rows[index]] +=
                tangent.values[index] * direction[tangent.cols[index]];
        }
        let error = next_residual
            .iter()
            .zip(&residual)
            .zip(&tangent_direction)
            .map(|((next, base), exact)| ((next - base) / epsilon - exact).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = tangent_direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(
            error / scale < 1.0e-5,
            "relative tangent error={}",
            error / scale
        );
    }

    #[test]
    fn continuation_converges_and_stores_bias_quadrature() {
        let mut mesh = rectangular_mesh_triangles(0.0, 0.03, -0.01, 0.01, 4, 4);
        for element in &mut mesh.elements {
            let center = element
                .vertices()
                .iter()
                .map(|node| mesh.nodes[*node].x)
                .sum::<f64>()
                / 3.0;
            element.marker = if center < 0.015 { 1 } else { 2 };
        }
        let boundary_dofs: Vec<_> = mesh
            .boundaries
            .iter()
            .flat_map(|boundary| boundary.nodes.iter().copied())
            .collect();
        let problem = AxisymmetricNonlinearMagneticProblem {
            mesh,
            materials: BTreeMap::from([
                (1, MagneticConstitutiveLaw::Nonlinear(steel_curve())),
                (2, MagneticConstitutiveLaw::linear(MU_0).unwrap()),
            ]),
            azimuthal_current_density_a_per_m2: BTreeMap::from([(2, 2.0e6)]),
            quadrature_order: 2,
        };
        let result = solve_nonlinear_axisymmetric_bias(
            &problem,
            &boundary_dofs,
            &NonlinearBiasSolveConfig {
                continuation_steps: 4,
                ..NonlinearBiasSolveConfig::default()
            },
        )
        .unwrap();
        assert_eq!(result.continuation_iterations.len(), 4);
        assert!(!result.quadrature_state.is_empty());
        assert!(
            result
                .quadrature_state
                .iter()
                .all(|point| point.reluctivity_m_per_h > 0.0)
        );
        let scale = result.energy.source_work_j.abs().max(1.0e-12);
        assert!(result.energy.balance_error_j.abs() / scale < 1.0e-6);
    }

    #[test]
    fn force_factor_preserves_signed_flux_cancellation() {
        let problem = AxisymmetricNonlinearMagneticProblem {
            mesh: Mesh::new(2),
            materials: BTreeMap::new(),
            azimuthal_current_density_a_per_m2: BTreeMap::new(),
            quadrature_order: 2,
        };
        let point = |radial_flux: f64| MagneticBiasQuadraturePoint {
            element_index: 0,
            quadrature_index: 0,
            marker: 7,
            position_rz_m: [0.02, 0.0],
            integration_weight_m3: 1.0,
            vector_potential_atheta: 0.0,
            flux_density_rz_t: [radial_flux, 0.0],
            field_strength_rz_a_per_m: [0.0, 0.0],
            reluctivity_m_per_h: 1.0,
            differential_reluctivity_m_per_h: 1.0,
            energy_density_j_per_m3: 0.0,
            coenergy_density_j_per_m3: 0.0,
        };
        let force_factor = problem
            .extract_force_factor(&[point(1.0), point(-1.0)], 7, 100.0)
            .unwrap();
        assert!(force_factor.abs() < 1.0e-14);
    }
}
