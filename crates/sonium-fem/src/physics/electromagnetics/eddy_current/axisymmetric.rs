//! Axisymmetric harmonic eddy-current perturbation about a magnetic bias.

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::physics::electromagnetics::magnetostatics::MagneticBiasQuadraturePoint;
use crate::quadrature::QuadratureRule;
use crate::solver::{SolverConfig, SolverError, SolverType, solve_csr};
use crate::spaces::{FunctionSpace, H1Space};
use num_complex::Complex64;
use std::collections::{BTreeMap, HashSet};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct DifferentialReluctivityField {
    pub by_element_m_per_h: Vec<f64>,
}

impl DifferentialReluctivityField {
    pub fn from_bias_quadrature(
        num_elements: usize,
        bias: &[MagneticBiasQuadraturePoint],
    ) -> Result<Self, AxisymmetricEddyError> {
        let mut weighted = vec![0.0; num_elements];
        let mut weights = vec![0.0; num_elements];
        for point in bias {
            if point.element_index >= num_elements {
                return Err(AxisymmetricEddyError::BiasElementOutOfRange {
                    element_index: point.element_index,
                });
            }
            weighted[point.element_index] +=
                point.differential_reluctivity_m_per_h * point.integration_weight_m3;
            weights[point.element_index] += point.integration_weight_m3;
        }
        let mut by_element_m_per_h = Vec::with_capacity(num_elements);
        for element_index in 0..num_elements {
            if weights[element_index] <= 0.0 {
                return Err(AxisymmetricEddyError::MissingBiasElement { element_index });
            }
            let value = weighted[element_index] / weights[element_index];
            if !value.is_finite() || value <= 0.0 {
                return Err(AxisymmetricEddyError::InvalidDifferentialReluctivity {
                    element_index,
                });
            }
            by_element_m_per_h.push(value);
        }
        Ok(Self { by_element_m_per_h })
    }
}

pub struct AxisymmetricEddyCurrentProblem {
    pub mesh: Mesh,
    pub differential_reluctivity: DifferentialReluctivityField,
    pub conductivity_s_per_m: BTreeMap<i32, f64>,
    /// Impressed azimuthal current density for one ampere of terminal current.
    pub source_current_density_per_amp_a_per_m2: BTreeMap<i32, f64>,
    pub quadrature_order: usize,
}

impl AxisymmetricEddyCurrentProblem {
    pub fn validate(&self) -> Result<(), AxisymmetricEddyError> {
        if self.differential_reluctivity.by_element_m_per_h.len() != self.mesh.num_elements() {
            return Err(AxisymmetricEddyError::ReluctivityLengthMismatch);
        }
        if self.quadrature_order == 0 {
            return Err(AxisymmetricEddyError::InvalidQuadratureOrder);
        }
        for (&marker, &conductivity) in &self.conductivity_s_per_m {
            if !conductivity.is_finite() || conductivity < 0.0 {
                return Err(AxisymmetricEddyError::InvalidConductivity { marker });
            }
        }
        for (&marker, &source) in &self.source_current_density_per_amp_a_per_m2 {
            if !source.is_finite() {
                return Err(AxisymmetricEddyError::InvalidSource { marker });
            }
        }
        Ok(())
    }

    pub fn assemble(
        &self,
        frequency_hz: f64,
    ) -> Result<(TripletMatrix<Complex64>, Vec<Complex64>), AxisymmetricEddyError> {
        self.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(AxisymmetricEddyError::InvalidFrequency);
        }
        let omega = 2.0 * PI * frequency_hz;
        let space = H1Space::new(PolynomialDegree::P1);
        let mut matrix = TripletMatrix::new(self.mesh.num_nodes(), self.mesh.num_nodes());
        let mut rhs = vec![Complex64::default(); self.mesh.num_nodes()];
        for element_index in 0..self.mesh.num_elements() {
            let element = &self.mesh.elements[element_index];
            let nu = self.differential_reluctivity.by_element_m_per_h[element_index];
            let conductivity = self
                .conductivity_s_per_m
                .get(&element.marker)
                .copied()
                .unwrap_or(0.0);
            let source = self
                .source_current_density_per_amp_a_per_m2
                .get(&element.marker)
                .copied()
                .unwrap_or(0.0);
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
                for (local_i, &global_i) in dofs.iter().enumerate() {
                    rhs[global_i] += source * tab.value(local_i) * weight;
                    for (local_j, &global_j) in dofs.iter().enumerate() {
                        let magnetic = nu
                            * (basis_b[local_i][0] * basis_b[local_j][0]
                                + basis_b[local_i][1] * basis_b[local_j][1]);
                        let conductive =
                            omega * conductivity * tab.value(local_i) * tab.value(local_j);
                        matrix.push(
                            global_i,
                            global_j,
                            Complex64::new(magnetic, conductive) * weight,
                        );
                    }
                }
            }
        }
        Ok((matrix, rhs))
    }

    pub fn solve_unit_current(
        &self,
        frequency_hz: f64,
        dirichlet_dofs: &[usize],
    ) -> Result<AxisymmetricEddyCurrentSolution, AxisymmetricEddyError> {
        let (mut matrix, mut rhs) = self.assemble(frequency_hz)?;
        apply_constraints(&mut matrix, &mut rhs, dirichlet_dofs)?;
        let solver = SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        };
        let solved = solve_csr(&matrix.to_csr(), &rhs, &solver)?;
        let quadrature_state = self.quadrature_state(frequency_hz, &solved.values)?;
        let omega = 2.0 * PI * frequency_hz;
        let flux_linkage_h = self.flux_linkage(&quadrature_state);
        let magnetic_impedance_ohm = Complex64::new(0.0, omega) * flux_linkage_h;
        let eddy_loss_w = quadrature_state
            .iter()
            .map(|point| point.eddy_loss_density_w_per_m3 * point.integration_weight_m3)
            .sum();
        Ok(AxisymmetricEddyCurrentSolution {
            frequency_hz,
            vector_potential_atheta: solved.values,
            quadrature_state,
            flux_linkage_h,
            magnetic_impedance_ohm,
            eddy_loss_w,
            effective_inductance_h: magnetic_impedance_ohm.im / omega,
            eddy_resistance_ohm: magnetic_impedance_ohm.re,
        })
    }

    fn quadrature_state(
        &self,
        frequency_hz: f64,
        potential: &[Complex64],
    ) -> Result<Vec<EddyCurrentQuadraturePoint>, AxisymmetricEddyError> {
        let omega = 2.0 * PI * frequency_hz;
        let space = H1Space::new(PolynomialDegree::P1);
        let mut state = Vec::new();
        for element_index in 0..self.mesh.num_elements() {
            let element = &self.mesh.elements[element_index];
            let conductivity = self
                .conductivity_s_per_m
                .get(&element.marker)
                .copied()
                .unwrap_or(0.0);
            let source = self
                .source_current_density_per_amp_a_per_m2
                .get(&element.marker)
                .copied()
                .unwrap_or(0.0);
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
                let mut a = Complex64::default();
                let mut b = [Complex64::default(); 2];
                for (local, &dof) in dofs.iter().enumerate() {
                    a += potential[dof] * tab.value(local);
                    b[0] -= potential[dof] * tab.grad(local, 1);
                    b[1] += potential[dof] * (tab.grad(local, 0) + tab.value(local) / radius);
                }
                let electric = Complex64::new(0.0, -omega) * a;
                let eddy_current = conductivity * electric;
                state.push(EddyCurrentQuadraturePoint {
                    element_index,
                    quadrature_index,
                    marker: element.marker,
                    position_rz_m: [tab.physical_coords[0], tab.physical_coords[1]],
                    integration_weight_m3: 2.0 * PI * radius * tab.det_j * point.weight,
                    vector_potential_atheta: a,
                    flux_density_rz_t: b,
                    impressed_current_density_a_per_m2: source,
                    eddy_current_density_a_per_m2: eddy_current,
                    eddy_loss_density_w_per_m3: if conductivity > 0.0 {
                        eddy_current.norm_sqr() / conductivity
                    } else {
                        0.0
                    },
                });
            }
        }
        Ok(state)
    }

    fn flux_linkage(&self, state: &[EddyCurrentQuadraturePoint]) -> Complex64 {
        state
            .iter()
            .map(|point| {
                point.impressed_current_density_a_per_m2
                    * point.vector_potential_atheta
                    * point.integration_weight_m3
            })
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct EddyCurrentQuadraturePoint {
    pub element_index: usize,
    pub quadrature_index: usize,
    pub marker: i32,
    pub position_rz_m: [f64; 2],
    pub integration_weight_m3: f64,
    pub vector_potential_atheta: Complex64,
    pub flux_density_rz_t: [Complex64; 2],
    pub impressed_current_density_a_per_m2: f64,
    pub eddy_current_density_a_per_m2: Complex64,
    pub eddy_loss_density_w_per_m3: f64,
}

#[derive(Debug, Clone)]
pub struct AxisymmetricEddyCurrentSolution {
    pub frequency_hz: f64,
    pub vector_potential_atheta: Vec<Complex64>,
    pub quadrature_state: Vec<EddyCurrentQuadraturePoint>,
    pub flux_linkage_h: Complex64,
    pub magnetic_impedance_ohm: Complex64,
    pub effective_inductance_h: f64,
    pub eddy_resistance_ohm: f64,
    pub eddy_loss_w: f64,
}

/// Classical conductor skin depth `sqrt(2/(omega*mu*sigma))`.
pub fn skin_depth_m(
    frequency_hz: f64,
    permeability_h_per_m: f64,
    conductivity_s_per_m: f64,
) -> f64 {
    (1.0 / (PI * frequency_hz * permeability_h_per_m * conductivity_s_per_m)).sqrt()
}

/// Surface impedance of a finite conducting slab backed by a symmetry plane.
pub fn conducting_slab_surface_impedance(
    frequency_hz: f64,
    permeability_h_per_m: f64,
    conductivity_s_per_m: f64,
    half_thickness_m: f64,
) -> Complex64 {
    let delta = skin_depth_m(frequency_hz, permeability_h_per_m, conductivity_s_per_m);
    let propagation = Complex64::new(1.0, 1.0) / delta;
    propagation / conductivity_s_per_m / (propagation * half_thickness_m).tanh()
}

/// Exact internal impedance per unit length of a solid round conductor.
pub fn round_conductor_internal_impedance_per_m(
    frequency_hz: f64,
    permeability_h_per_m: f64,
    conductivity_s_per_m: f64,
    radius_m: f64,
) -> Complex64 {
    let delta = skin_depth_m(frequency_hz, permeability_h_per_m, conductivity_s_per_m);
    let argument = Complex64::new(1.0, -1.0) * radius_m / delta;
    let ratio = argument * complex_bessel_j0(argument) / (2.0 * complex_bessel_j1(argument));
    ratio / (PI * radius_m * radius_m * conductivity_s_per_m)
}

fn complex_bessel_j0(z: Complex64) -> Complex64 {
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = term;
    for k in 1..80 {
        term *= -z * z / (4.0 * (k * k) as f64);
        sum += term;
        if term.norm() < 1.0e-15 * sum.norm().max(1.0) {
            break;
        }
    }
    sum
}

fn complex_bessel_j1(z: Complex64) -> Complex64 {
    let mut term = z / 2.0;
    let mut sum = term;
    for k in 1..80 {
        term *= -z * z / (4.0 * k as f64 * (k + 1) as f64);
        sum += term;
        if term.norm() < 1.0e-15 * sum.norm().max(1.0) {
            break;
        }
    }
    sum
}

fn apply_constraints(
    matrix: &mut TripletMatrix<Complex64>,
    rhs: &mut [Complex64],
    dofs: &[usize],
) -> Result<(), AxisymmetricEddyError> {
    let constrained: HashSet<_> = dofs.iter().copied().collect();
    if constrained.iter().any(|&dof| dof >= rhs.len()) {
        return Err(AxisymmetricEddyError::InvalidDirichletDof);
    }
    for index in 0..matrix.nnz() {
        if constrained.contains(&matrix.rows[index]) || constrained.contains(&matrix.cols[index]) {
            matrix.values[index] = Complex64::default();
        }
    }
    for &dof in &constrained {
        matrix.push(dof, dof, Complex64::new(1.0, 0.0));
        rhs[dof] = Complex64::default();
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum AxisymmetricEddyError {
    #[error("bias quadrature references missing element {element_index}")]
    BiasElementOutOfRange { element_index: usize },
    #[error("bias quadrature has no state for element {element_index}")]
    MissingBiasElement { element_index: usize },
    #[error("element {element_index} has invalid differential reluctivity")]
    InvalidDifferentialReluctivity { element_index: usize },
    #[error("differential reluctivity length does not match the mesh")]
    ReluctivityLengthMismatch,
    #[error("quadrature order must be positive")]
    InvalidQuadratureOrder,
    #[error("conductivity for marker {marker} must be finite and non-negative")]
    InvalidConductivity { marker: i32 },
    #[error("source current density for marker {marker} must be finite")]
    InvalidSource { marker: i32 },
    #[error("frequency must be finite and positive")]
    InvalidFrequency,
    #[error("Dirichlet DOF is outside the problem")]
    InvalidDirichletDof,
    #[error(transparent)]
    Solver(#[from] SolverError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::rectangular_mesh_triangles;
    use crate::physics::electromagnetics::magnetostatics::MU_0;

    fn uniform_problem(conductivity: f64) -> AxisymmetricEddyCurrentProblem {
        let mut mesh = rectangular_mesh_triangles(0.005, 0.015, -0.005, 0.005, 4, 4);
        for element in &mut mesh.elements {
            element.marker = 1;
        }
        AxisymmetricEddyCurrentProblem {
            differential_reluctivity: DifferentialReluctivityField {
                by_element_m_per_h: vec![1.0 / MU_0; mesh.num_elements()],
            },
            mesh,
            conductivity_s_per_m: BTreeMap::from([(1, conductivity)]),
            source_current_density_per_amp_a_per_m2: BTreeMap::from([(1, 1.0e5)]),
            quadrature_order: 2,
        }
    }

    #[test]
    fn nonconducting_limit_has_zero_loss_and_real_flux_linkage() {
        let problem = uniform_problem(0.0);
        let boundary: Vec<_> = problem
            .mesh
            .boundaries
            .iter()
            .flat_map(|face| face.nodes.iter().copied())
            .collect();
        let solution = problem.solve_unit_current(1_000.0, &boundary).unwrap();
        assert!(solution.eddy_loss_w.abs() < 1.0e-20);
        assert!(solution.eddy_resistance_ohm.abs() < 1.0e-12);
        assert!(solution.flux_linkage_h.im.abs() < 1.0e-12);
        assert!(solution.effective_inductance_h > 0.0);
    }

    #[test]
    fn conducting_solution_is_passive_and_power_consistent() {
        let problem = uniform_problem(6.0e6);
        let boundary: Vec<_> = problem
            .mesh
            .boundaries
            .iter()
            .flat_map(|face| face.nodes.iter().copied())
            .collect();
        let solution = problem.solve_unit_current(2_000.0, &boundary).unwrap();
        assert!(solution.eddy_resistance_ohm >= 0.0);
        assert!(solution.effective_inductance_h > 0.0);
        assert!(
            (solution.eddy_resistance_ohm - solution.eddy_loss_w).abs()
                / solution.eddy_loss_w.max(1.0e-16)
                < 1.0e-8
        );
    }

    #[test]
    fn slab_and_round_conductor_recover_dc_and_skin_effect_limits() {
        let sigma = 5.8e7;
        let low = round_conductor_internal_impedance_per_m(1.0, MU_0, sigma, 1.0e-3);
        let dc = 1.0 / (PI * 1.0e-6 * sigma);
        assert!((low.re / dc - 1.0).abs() < 1.0e-5);
        assert!(low.im > 0.0);
        let slab = conducting_slab_surface_impedance(10_000.0, MU_0, sigma, 1.0e-2);
        let asymptotic = Complex64::new(1.0, 1.0) / (sigma * skin_depth_m(10_000.0, MU_0, sigma));
        assert!((slab / asymptotic - Complex64::new(1.0, 0.0)).norm() < 1.0e-6);
        assert!(slab.re > 0.0 && slab.im > 0.0);
    }
}
