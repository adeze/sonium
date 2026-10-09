//! Weighted axisymmetric meridional Navier--Stokes reference.
//!
//! Unlike the legacy body-fitted adapter, this module assembles the
//! meridional equations with the cylindrical `2πr` measure, the geometric
//! radial viscous term `μ u_r/r²`, and the axisymmetric divergence
//! `∂r u_r + u_r/r + ∂z u_z`.  It is deliberately a bounded laminar P1/P1
//! Picard reference; turbulence, centerline regularity, and 3-D rotation
//! remain separate fidelity levels.

use crate::body_fitted_flow_reference::{
    BodyFittedDuctStation, BodyFittedFlowNode, BodyFittedFlowOptions, BodyFittedTurbulenceModel,
};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::{assemble_bilinear, assemble_linear, mixed::assemble_mixed};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use sonium_fem::mesh::{Mesh, duct_profile_mesh};
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space, VectorH1Space};
use sonium_solvers::SparseLuFactorization;
use std::fmt;
use std::sync::Arc;

/// Versioned identity for the weighted axisymmetric meridional reference.
pub const AXISYMMETRIC_MERIDIONAL_FLOW_REFERENCE_SCHEMA_V1: &str =
    "propeller.body-fitted-axisymmetric-meridional-flow-reference.v1";

type FieldFn = Arc<dyn Fn([f64; 3]) -> [f64; 2] + Send + Sync>;
pub type AxisymmetricMeridionalForceFn = Arc<dyn Fn([f64; 3]) -> [f64; 2] + Send + Sync>;

/// Weighted axisymmetric meridional result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisymmetricMeridionalFlowResult {
    /// Contract schema.
    pub schema: String,
    /// Solved nodal field in mesh order.
    pub nodes: Vec<BodyFittedFlowNode>,
    /// Number of generated mesh nodes.
    pub mesh_node_count: usize,
    /// Number of generated triangular elements.
    pub mesh_element_count: usize,
    /// Picard iterations performed.
    pub iterations: usize,
    /// Final relative velocity change.
    pub residual: f64,
    /// Whether the weighted Picard iteration converged.
    pub converged: bool,
    /// Inlet volume flux from `2πr` radial quadrature [m³/s].
    pub inlet_volume_flux_m3_s: f64,
    /// Outlet volume flux from `2πr` radial quadrature [m³/s].
    pub outlet_volume_flux_m3_s: f64,
    /// Relative inlet/outlet weighted flux mismatch.
    pub relative_volume_flux_mismatch: f64,
    /// Maximum nodal meridional speed [m/s].
    pub maximum_speed_m_s: f64,
    /// Minimum nodal pressure [Pa].
    pub minimum_pressure_pa: f64,
    /// Maximum nodal pressure [Pa].
    pub maximum_pressure_pa: f64,
}

impl AxisymmetricMeridionalFlowResult {
    /// Whether the weighted state and all diagnostics are finite and coherent.
    pub fn is_finite(&self) -> bool {
        self.schema == AXISYMMETRIC_MERIDIONAL_FLOW_REFERENCE_SCHEMA_V1
            && !self.nodes.is_empty()
            && self.mesh_node_count == self.nodes.len()
            && self.mesh_element_count > 0
            && self.residual.is_finite()
            && self.residual >= 0.0
            && self.inlet_volume_flux_m3_s.is_finite()
            && self.outlet_volume_flux_m3_s.is_finite()
            && self.relative_volume_flux_mismatch.is_finite()
            && self.relative_volume_flux_mismatch >= 0.0
            && self.maximum_speed_m_s.is_finite()
            && self.maximum_speed_m_s >= 0.0
            && self.minimum_pressure_pa.is_finite()
            && self.maximum_pressure_pa.is_finite()
            && self.minimum_pressure_pa <= self.maximum_pressure_pa
            && self.nodes.iter().all(|node| {
                node.radial_position_m.is_finite()
                    && node.radial_position_m > 0.0
                    && node.axial_position_m.is_finite()
                    && node.radial_velocity_m_s.is_finite()
                    && node.axial_velocity_m_s.is_finite()
                    && node.pressure_pa.is_finite()
            })
    }
}

/// Failure-closed errors from the weighted axisymmetric reference.
#[derive(Debug, Clone, PartialEq)]
pub enum AxisymmetricMeridionalFlowError {
    /// Empty profile.
    EmptyProfile,
    /// Invalid station field.
    InvalidStation {
        /// Zero-based station index.
        index: usize,
        /// Invalid field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Axial positions are not strictly increasing.
    NonMonotonicProfile { index: usize },
    /// Scalar or model option is outside the admitted domain.
    InvalidOption {
        /// Invalid option field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Mesh estimate exceeds the explicit work budget.
    MeshBudgetExceeded { nodes: usize, elements: usize },
    /// The direct solve failed.
    LinearSolve(String),
    /// A required Picard solve did not converge.
    SolverDidNotConverge { iterations: usize, residual: f64 },
    /// The assembled/published state is invalid.
    InvalidResult(&'static str),
}

impl fmt::Display for AxisymmetricMeridionalFlowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProfile => formatter.write_str("axisymmetric profile is empty"),
            Self::InvalidStation {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid axisymmetric station {index} field {field}: {requirement}"
            ),
            Self::NonMonotonicProfile { index } => {
                write!(
                    formatter,
                    "axisymmetric profile is not increasing at {index}"
                )
            }
            Self::InvalidOption { field, requirement } => {
                write!(
                    formatter,
                    "invalid axisymmetric option {field}: {requirement}"
                )
            }
            Self::MeshBudgetExceeded { nodes, elements } => write!(
                formatter,
                "axisymmetric mesh estimate {nodes} nodes and {elements} elements exceeds budget"
            ),
            Self::LinearSolve(error) => {
                write!(formatter, "axisymmetric linear solve failed: {error}")
            }
            Self::SolverDidNotConverge {
                iterations,
                residual,
            } => write!(
                formatter,
                "axisymmetric Picard solve did not converge after {iterations} iterations (residual {residual})"
            ),
            Self::InvalidResult(field) => {
                write!(formatter, "axisymmetric result field {field} is invalid")
            }
        }
    }
}

impl std::error::Error for AxisymmetricMeridionalFlowError {}

/// Solve the weighted axisymmetric meridional equations.
pub fn solve_axisymmetric_meridional_flow(
    profile: &[BodyFittedDuctStation],
    options: BodyFittedFlowOptions,
) -> Result<AxisymmetricMeridionalFlowResult, AxisymmetricMeridionalFlowError> {
    solve_axisymmetric_meridional_flow_with_centrifugal_source(profile, options, None)
}

/// Solve weighted axisymmetric meridional flow with an optional supplied
/// radial centrifugal acceleration.  A supplied source replaces the analytic
/// rotating-wall source, which lets a later outer swirl iteration feed back
/// its transported field without changing the serialized flow options.
pub fn solve_axisymmetric_meridional_flow_with_centrifugal_source(
    profile: &[BodyFittedDuctStation],
    options: BodyFittedFlowOptions,
    centrifugal_source: Option<AxisymmetricMeridionalForceFn>,
) -> Result<AxisymmetricMeridionalFlowResult, AxisymmetricMeridionalFlowError> {
    validate_profile(profile)?;
    validate_options(&options)?;
    let (mesh, node_count, element_count) = build_mesh(profile, &options)?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let n_velocity = velocity_space.num_global_dofs(&mesh);
    let n_pressure = pressure_space.num_global_dofs(&mesh);
    if n_velocity != 2 * mesh.num_nodes() || n_pressure != mesh.num_nodes() {
        return Err(AxisymmetricMeridionalFlowError::InvalidResult("dof_layout"));
    }

    let z_min = profile[0].axial_position_m;
    let z_max = profile[profile.len() - 1].axial_position_m;
    let boundary_tolerance = profile
        .windows(2)
        .map(|pair| pair[1].axial_position_m - pair[0].axial_position_m)
        .fold(f64::INFINITY, f64::min)
        .min(
            profile
                .iter()
                .map(|station| station.outer_radius_m - station.inner_radius_m)
                .fold(f64::INFINITY, f64::min),
        )
        * 1.0e-6;
    let boundary_tolerance = boundary_tolerance.max(1.0e-12);
    let inner_profile = profile.to_vec();
    let boundary_dofs = locate_dofs_geometrical(&mesh, &velocity_space, |radius, axial, _| {
        let (inner, outer) = interpolate_profile(&inner_profile, axial);
        (axial - z_min).abs() <= boundary_tolerance
            || (radius - inner).abs() <= boundary_tolerance
            || (radius - outer).abs() <= boundary_tolerance
    });
    if boundary_dofs.is_empty() {
        return Err(AxisymmetricMeridionalFlowError::InvalidResult(
            "boundary_dofs",
        ));
    }
    let coordinates = velocity_space.dof_coordinates(&mesh).ok_or(
        AxisymmetricMeridionalFlowError::InvalidResult("velocity_coordinates"),
    )?;
    let bc_profile = profile.to_vec();
    let bc_values_velocity = |point: [f64; 3]| {
        let (inner, outer) = interpolate_profile(&bc_profile, point[1]);
        let on_wall = (point[0] - inner).abs() <= boundary_tolerance
            || (point[0] - outer).abs() <= boundary_tolerance;
        if on_wall {
            [0.0, 0.0]
        } else {
            [0.0, options.inlet_axial_velocity_m_s]
        }
    };
    let mut is_bc = vec![false; n_velocity + n_pressure];
    let mut bc_values = vec![0.0; n_velocity + n_pressure];
    for &dof in &boundary_dofs {
        let point = *coordinates
            .get(dof)
            .ok_or(AxisymmetricMeridionalFlowError::InvalidResult(
                "boundary_coordinate",
            ))?;
        is_bc[dof] = true;
        bc_values[dof] = bc_values_velocity(point)[dof % 2];
    }
    let pressure_coordinates = pressure_space.dof_coordinates(&mesh).ok_or(
        AxisymmetricMeridionalFlowError::InvalidResult("pressure_coordinates"),
    )?;
    let centroid = [
        mesh.nodes.iter().map(|point| point.x).sum::<f64>() / mesh.nodes.len() as f64,
        mesh.nodes.iter().map(|point| point.y).sum::<f64>() / mesh.nodes.len() as f64,
    ];
    let pressure_pin = pressure_coordinates
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let left_distance = (left[0] - centroid[0]).hypot(left[1] - centroid[1]);
            let right_distance = (right[0] - centroid[0]).hypot(right[1] - centroid[1]);
            left_distance.total_cmp(&right_distance)
        })
        .map(|(index, _)| index)
        .ok_or(AxisymmetricMeridionalFlowError::InvalidResult(
            "pressure_pin",
        ))?;
    is_bc[n_velocity + pressure_pin] = true;
    bc_values[n_velocity + pressure_pin] = 0.0;

    let profile_for_source = profile.to_vec();
    let rotating_wall_model = options.rotating_wall_model;
    let force_source = centrifugal_source.unwrap_or_else(|| {
        Arc::new(move |point: [f64; 3]| {
            let (inner, outer) = interpolate_profile(&profile_for_source, point[1]);
            [
                rotating_wall_model.centrifugal_acceleration_m_s2(point[0], inner, outer),
                0.0,
            ]
        })
    });

    let mut velocity = vec![0.0; n_velocity];
    let mut pressure = vec![0.0; n_pressure];
    for &dof in &boundary_dofs {
        velocity[dof] = bc_values[dof];
    }
    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 1..=options.max_picard_iterations {
        let nodal_velocity = velocity
            .as_chunks::<2>()
            .0
            .iter()
            .map(|components| [components[0], components[1]])
            .collect::<Vec<_>>();
        let advector_profile = profile.to_vec();
        let advector: FieldFn = Arc::new(move |point| {
            interpolate_structured_field(
                &advector_profile,
                options.axial_elements,
                options.radial_elements,
                &nodal_velocity,
                point[0],
                point[1],
            )
        });
        let oseen = AxisymmetricOseenForm {
            density_kg_m3: options.density_kg_m3,
            dynamic_viscosity_pa_s: options.dynamic_viscosity_pa_s,
            advector,
        };
        let divergence = AxisymmetricDivergenceForm;
        let mixed = assemble_mixed(
            &oseen,
            &divergence,
            &velocity_space,
            &pressure_space,
            &mesh,
            2,
        );
        let pressure_stabilization = AxisymmetricPressureGradientForm {
            coefficient: pressure_stabilization_coefficient(profile, &options),
        };
        let pressure_stabilization = assemble_bilinear(
            &pressure_stabilization,
            &pressure_space,
            &pressure_space,
            &mesh,
            2,
        );
        let force = AxisymmetricBodyForceForm {
            force: Arc::clone(&force_source),
        };
        let rhs_velocity = assemble_linear(&force, &velocity_space, &mesh, 2);
        let mut rhs = vec![0.0; n_velocity + n_pressure];
        rhs[..n_velocity].copy_from_slice(&rhs_velocity);
        let mut monolithic = mixed.to_monolithic();
        for entry in 0..pressure_stabilization.nnz() {
            monolithic.push(
                n_velocity + pressure_stabilization.rows[entry],
                n_velocity + pressure_stabilization.cols[entry],
                -pressure_stabilization.values[entry],
            );
        }
        monolithic.push(n_velocity + pressure_pin, n_velocity + pressure_pin, 1.0);
        let mut matrix = monolithic.to_csr();
        apply_dirichlet_to_csr(&mut matrix, &mut rhs, &is_bc, &bc_values);
        let solution = SparseLuFactorization::factorize(&matrix)
            .and_then(|factorization| factorization.solve(&rhs))
            .map_err(|error| AxisymmetricMeridionalFlowError::LinearSolve(error.to_string()))?;
        if solution.len() != n_velocity + n_pressure
            || solution.iter().any(|value| !value.is_finite())
        {
            return Err(AxisymmetricMeridionalFlowError::InvalidResult(
                "linear_state",
            ));
        }
        let old_norm = velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let mut next_velocity = solution[..n_velocity].to_vec();
        for (next, old) in next_velocity.iter_mut().zip(velocity.iter()) {
            *next = *old + options.picard_relaxation * (*next - *old);
        }
        for &dof in &boundary_dofs {
            next_velocity[dof] = bc_values[dof];
        }
        let delta_norm = next_velocity
            .iter()
            .zip(velocity.iter())
            .map(|(next, old)| (next - old) * (next - old))
            .sum::<f64>()
            .sqrt();
        residual = delta_norm
            / old_norm
                .max(
                    next_velocity
                        .iter()
                        .map(|value| value * value)
                        .sum::<f64>()
                        .sqrt(),
                )
                .max(1.0);
        velocity = next_velocity;
        pressure = solution[n_velocity..].to_vec();
        pressure[pressure_pin] = 0.0;
        iterations = iteration;
        if iteration >= options.min_picard_iterations && residual <= options.picard_tolerance {
            converged = true;
            break;
        }
    }
    if options.require_convergence && !converged {
        return Err(AxisymmetricMeridionalFlowError::SolverDidNotConverge {
            iterations,
            residual,
        });
    }
    let nodes = mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| BodyFittedFlowNode {
            radial_position_m: point.x,
            axial_position_m: point.y,
            radial_velocity_m_s: velocity[2 * index],
            axial_velocity_m_s: velocity[2 * index + 1],
            pressure_pa: pressure[index],
        })
        .collect::<Vec<_>>();
    let inlet_volume_flux_m3_s = weighted_station_flux(&nodes, z_min, boundary_tolerance);
    let outlet_volume_flux_m3_s = weighted_station_flux(&nodes, z_max, boundary_tolerance);
    let maximum_speed_m_s = nodes
        .iter()
        .map(|node| {
            (node.radial_velocity_m_s * node.radial_velocity_m_s
                + node.axial_velocity_m_s * node.axial_velocity_m_s)
                .sqrt()
        })
        .fold(0.0, f64::max);
    let minimum_pressure_pa = nodes
        .iter()
        .map(|node| node.pressure_pa)
        .fold(f64::INFINITY, f64::min);
    let maximum_pressure_pa = nodes
        .iter()
        .map(|node| node.pressure_pa)
        .fold(f64::NEG_INFINITY, f64::max);
    let inlet_area_m2 = std::f64::consts::PI
        * (profile[0].outer_radius_m.powi(2) - profile[0].inner_radius_m.powi(2));
    let flux_scale = inlet_volume_flux_m3_s
        .abs()
        .max(inlet_area_m2 * options.inlet_axial_velocity_m_s.abs())
        .max(1.0e-12);
    let relative_volume_flux_mismatch =
        (outlet_volume_flux_m3_s - inlet_volume_flux_m3_s).abs() / flux_scale;
    let result = AxisymmetricMeridionalFlowResult {
        schema: AXISYMMETRIC_MERIDIONAL_FLOW_REFERENCE_SCHEMA_V1.to_owned(),
        nodes,
        mesh_node_count: node_count,
        mesh_element_count: element_count,
        iterations,
        residual,
        converged,
        inlet_volume_flux_m3_s,
        outlet_volume_flux_m3_s,
        relative_volume_flux_mismatch,
        maximum_speed_m_s,
        minimum_pressure_pa,
        maximum_pressure_pa,
    };
    if !result.is_finite() {
        return Err(AxisymmetricMeridionalFlowError::InvalidResult(
            "finite_state",
        ));
    }
    Ok(result)
}

struct AxisymmetricOseenForm {
    density_kg_m3: f64,
    dynamic_viscosity_pa_s: f64,
    advector: FieldFn,
}

impl BilinearForm<f64> for AxisymmetricOseenForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let radius = test.physical_coords[0];
        if radius <= 0.0 || !radius.is_finite() {
            return;
        }
        let axisymmetric_weight = weight * test.det_j * 2.0 * std::f64::consts::PI * radius;
        let advector = (self.advector)(test.physical_coords);
        let dimension = test.spatial_dim;
        for i in 0..test.n_dofs {
            let test_component = i % dimension;
            let test_value = test.vector_value(i, test_component);
            for j in 0..trial.n_dofs {
                let trial_component = j % dimension;
                if test_component != trial_component {
                    continue;
                }
                let gradient_dot = (0..dimension)
                    .map(|direction| test.grad(i, direction) * trial.grad(j, direction))
                    .sum::<f64>();
                let radial_geometric = if test_component == 0 {
                    test_value * trial.vector_value(j, 0) / (radius * radius)
                } else {
                    0.0
                };
                let advection =
                    test_value * (advector[0] * trial.grad(j, 0) + advector[1] * trial.grad(j, 1));
                local_matrix.add(
                    i,
                    j,
                    (self.dynamic_viscosity_pa_s * (gradient_dot + radial_geometric)
                        + self.density_kg_m3 * advection)
                        * axisymmetric_weight,
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
struct AxisymmetricDivergenceForm;

impl BilinearForm<f64> for AxisymmetricDivergenceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let radius = test.physical_coords[0];
        if radius <= 0.0 || !radius.is_finite() {
            return;
        }
        let axisymmetric_weight = weight * test.det_j * 2.0 * std::f64::consts::PI * radius;
        for i in 0..test.n_dofs {
            let divergence = match i % test.spatial_dim {
                0 => test.grad(i, 0) + test.vector_value(i, 0) / radius,
                1 => test.grad(i, 1),
                _ => 0.0,
            };
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, -trial.value(j) * divergence * axisymmetric_weight);
            }
        }
    }
}

/// Residual pressure-gradient stabilization for the equal-order P1/P1 pair.
///
/// The stabilization removes the checkerboard pressure null modes while
/// retaining the physical pressure gauge pin. It uses the same cylindrical
/// measure as the momentum and divergence blocks.
struct AxisymmetricPressureGradientForm {
    coefficient: f64,
}

impl BilinearForm<f64> for AxisymmetricPressureGradientForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let radius = test.physical_coords[0];
        if radius <= 0.0 || !radius.is_finite() {
            return;
        }
        let axisymmetric_weight = weight * test.det_j * 2.0 * std::f64::consts::PI * radius;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let gradient_dot = (0..test.spatial_dim)
                    .map(|direction| test.grad(i, direction) * trial.grad(j, direction))
                    .sum::<f64>();
                local_matrix.add(i, j, self.coefficient * gradient_dot * axisymmetric_weight);
            }
        }
    }
}

struct AxisymmetricBodyForceForm {
    force: AxisymmetricMeridionalForceFn,
}

impl LinearForm<f64> for AxisymmetricBodyForceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let radius = test.physical_coords[0];
        if radius <= 0.0 || !radius.is_finite() {
            return;
        }
        let force = (self.force)(test.physical_coords);
        let axisymmetric_weight = weight * test.det_j * 2.0 * std::f64::consts::PI * radius;
        for i in 0..test.n_dofs {
            local_vector.add(
                i,
                force[i % test.spatial_dim]
                    * test.vector_value(i, i % test.spatial_dim)
                    * axisymmetric_weight,
            );
        }
    }
}

fn apply_dirichlet_to_csr(
    matrix: &mut sonium_solvers::CsrMatrix<f64>,
    rhs: &mut [f64],
    is_bc: &[bool],
    bc_values: &[f64],
) {
    for row in 0..rhs.len() {
        let start = matrix.row_ptrs[row];
        let end = matrix.row_ptrs[row + 1];
        if is_bc[row] {
            for entry in start..end {
                matrix.values[entry] = if matrix.col_indices[entry] == row {
                    1.0
                } else {
                    0.0
                };
            }
            rhs[row] = bc_values[row];
        } else {
            for entry in start..end {
                let column = matrix.col_indices[entry];
                if is_bc[column] {
                    rhs[row] -= matrix.values[entry] * bc_values[column];
                    matrix.values[entry] = 0.0;
                }
            }
        }
    }
}

fn validate_profile(
    profile: &[BodyFittedDuctStation],
) -> Result<(), AxisymmetricMeridionalFlowError> {
    if profile.is_empty() {
        return Err(AxisymmetricMeridionalFlowError::EmptyProfile);
    }
    for (index, station) in profile.iter().enumerate() {
        if !station.axial_position_m.is_finite() {
            return Err(AxisymmetricMeridionalFlowError::InvalidStation {
                index,
                field: "axial_position_m",
                requirement: "finite",
            });
        }
        if !station.inner_radius_m.is_finite() || station.inner_radius_m <= 0.0 {
            return Err(AxisymmetricMeridionalFlowError::InvalidStation {
                index,
                field: "inner_radius_m",
                requirement: "finite and strictly positive for annular cylindrical weighting",
            });
        }
        if !station.outer_radius_m.is_finite() || station.outer_radius_m <= station.inner_radius_m {
            return Err(AxisymmetricMeridionalFlowError::InvalidStation {
                index,
                field: "outer_radius_m",
                requirement: "finite and greater than inner_radius_m",
            });
        }
        if let Some(previous) = index.checked_sub(1).and_then(|i| profile.get(i))
            && station.axial_position_m <= previous.axial_position_m
        {
            return Err(AxisymmetricMeridionalFlowError::NonMonotonicProfile { index });
        }
    }
    if profile.len() < 2 {
        return Err(AxisymmetricMeridionalFlowError::InvalidStation {
            index: 0,
            field: "profile",
            requirement: "at least two axial stations",
        });
    }
    Ok(())
}

fn validate_options(
    options: &BodyFittedFlowOptions,
) -> Result<(), AxisymmetricMeridionalFlowError> {
    for (field, value) in [
        ("density_kg_m3", options.density_kg_m3),
        ("dynamic_viscosity_pa_s", options.dynamic_viscosity_pa_s),
        ("picard_tolerance", options.picard_tolerance),
        ("linear_solver_tolerance", options.linear_solver_tolerance),
        ("inlet_axial_velocity_m_s", options.inlet_axial_velocity_m_s),
    ] {
        if !value.is_finite() || (field != "inlet_axial_velocity_m_s" && value <= 0.0) {
            return Err(AxisymmetricMeridionalFlowError::InvalidOption {
                field,
                requirement: "finite and positive except inlet velocity",
            });
        }
    }
    if options.turbulence_model != BodyFittedTurbulenceModel::Laminar {
        return Err(AxisymmetricMeridionalFlowError::InvalidOption {
            field: "turbulence_model",
            requirement: "Laminar for this weighted reference",
        });
    }
    if !(0.0 < options.picard_relaxation && options.picard_relaxation <= 1.0) {
        return Err(AxisymmetricMeridionalFlowError::InvalidOption {
            field: "picard_relaxation",
            requirement: "in (0, 1]",
        });
    }
    if options.axial_elements == 0
        || options.radial_elements == 0
        || options.max_picard_iterations == 0
        || options.min_picard_iterations == 0
        || options.min_picard_iterations > options.max_picard_iterations
    {
        return Err(AxisymmetricMeridionalFlowError::InvalidOption {
            field: "mesh_or_solver_budget",
            requirement: "positive and internally ordered",
        });
    }
    if options.maximum_nodes == 0 || options.maximum_elements == 0 {
        return Err(AxisymmetricMeridionalFlowError::InvalidOption {
            field: "mesh_budget",
            requirement: "positive",
        });
    }
    Ok(())
}

fn pressure_stabilization_coefficient(
    profile: &[BodyFittedDuctStation],
    options: &BodyFittedFlowOptions,
) -> f64 {
    let axial_step = profile
        .windows(2)
        .map(|pair| pair[1].axial_position_m - pair[0].axial_position_m)
        .fold(f64::INFINITY, f64::min)
        / options.axial_elements as f64;
    let radial_step = profile
        .iter()
        .map(|station| station.outer_radius_m - station.inner_radius_m)
        .fold(f64::INFINITY, f64::min)
        / options.radial_elements as f64;
    let mesh_length_m = axial_step.min(radial_step);
    (mesh_length_m * mesh_length_m / options.dynamic_viscosity_pa_s).clamp(1.0e-12, 1.0e6)
}

fn build_mesh(
    profile: &[BodyFittedDuctStation],
    options: &BodyFittedFlowOptions,
) -> Result<(Mesh, usize, usize), AxisymmetricMeridionalFlowError> {
    let node_count = (options.axial_elements + 1)
        .checked_mul(options.radial_elements + 1)
        .ok_or(AxisymmetricMeridionalFlowError::MeshBudgetExceeded {
            nodes: usize::MAX,
            elements: usize::MAX,
        })?;
    let element_count = options
        .axial_elements
        .checked_mul(options.radial_elements)
        .and_then(|value| value.checked_mul(2))
        .ok_or(AxisymmetricMeridionalFlowError::MeshBudgetExceeded {
            nodes: node_count,
            elements: usize::MAX,
        })?;
    if node_count > options.maximum_nodes || element_count > options.maximum_elements {
        return Err(AxisymmetricMeridionalFlowError::MeshBudgetExceeded {
            nodes: node_count,
            elements: element_count,
        });
    }
    let z_min = profile[0].axial_position_m;
    let z_max = profile[profile.len() - 1].axial_position_m;
    let inner = profile.to_vec();
    let outer = profile.to_vec();
    let mesh = duct_profile_mesh(
        z_min,
        z_max,
        move |axial| interpolate_profile(&inner, axial).0,
        move |axial| interpolate_profile(&outer, axial).1,
        options.axial_elements,
        options.radial_elements,
    );
    if mesh.num_nodes() != node_count || mesh.num_elements() != element_count {
        return Err(AxisymmetricMeridionalFlowError::InvalidResult(
            "mesh_counts",
        ));
    }
    Ok((mesh, node_count, element_count))
}

fn interpolate_profile(profile: &[BodyFittedDuctStation], axial: f64) -> (f64, f64) {
    if axial <= profile[0].axial_position_m {
        return (profile[0].inner_radius_m, profile[0].outer_radius_m);
    }
    if axial >= profile[profile.len() - 1].axial_position_m {
        let station = profile[profile.len() - 1];
        return (station.inner_radius_m, station.outer_radius_m);
    }
    for pair in profile.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if axial <= right.axial_position_m {
            let fraction =
                (axial - left.axial_position_m) / (right.axial_position_m - left.axial_position_m);
            return (
                left.inner_radius_m + fraction * (right.inner_radius_m - left.inner_radius_m),
                left.outer_radius_m + fraction * (right.outer_radius_m - left.outer_radius_m),
            );
        }
    }
    unreachable!("profile bounds handled above")
}

fn interpolate_structured_field(
    profile: &[BodyFittedDuctStation],
    axial_elements: usize,
    radial_elements: usize,
    values: &[[f64; 2]],
    radius: f64,
    axial: f64,
) -> [f64; 2] {
    let stride = radial_elements + 1;
    let z_min = profile[0].axial_position_m;
    let z_max = profile[profile.len() - 1].axial_position_m;
    let z_fraction = ((axial - z_min) / (z_max - z_min)).clamp(0.0, 1.0);
    let axial_position = z_fraction * axial_elements as f64;
    let axial_index = (axial_position.floor() as usize).min(axial_elements - 1);
    let axial_fraction = axial_position - axial_index as f64;
    let z0 = z_min + axial_index as f64 / axial_elements as f64 * (z_max - z_min);
    let z1 = z_min + (axial_index + 1) as f64 / axial_elements as f64 * (z_max - z_min);
    let (inner0, outer0) = interpolate_profile(profile, z0);
    let (inner1, outer1) = interpolate_profile(profile, z1);
    let inner = inner0 + axial_fraction * (inner1 - inner0);
    let outer = outer0 + axial_fraction * (outer1 - outer0);
    let radial_fraction = ((radius - inner) / (outer - inner)).clamp(0.0, 1.0);
    let radial_position = radial_fraction * radial_elements as f64;
    let radial_index = (radial_position.floor() as usize).min(radial_elements - 1);
    let radial_weight = radial_position - radial_index as f64;
    let index00 = axial_index * stride + radial_index;
    let index01 = index00 + 1;
    let index10 = (axial_index + 1) * stride + radial_index;
    let index11 = index10 + 1;
    std::array::from_fn(|component| {
        let lower = values[index00][component]
            + radial_weight * (values[index01][component] - values[index00][component]);
        let upper = values[index10][component]
            + radial_weight * (values[index11][component] - values[index10][component]);
        lower + axial_fraction * (upper - lower)
    })
}

fn weighted_station_flux(nodes: &[BodyFittedFlowNode], axial: f64, tolerance: f64) -> f64 {
    let mut station = nodes
        .iter()
        .filter(|node| (node.axial_position_m - axial).abs() <= tolerance)
        .collect::<Vec<_>>();
    station.sort_by(|left, right| left.radial_position_m.total_cmp(&right.radial_position_m));
    station
        .windows(2)
        .map(|pair| {
            let radius_delta = pair[1].radial_position_m - pair[0].radial_position_m;
            0.5 * radius_delta
                * (pair[0].radial_position_m * pair[0].axial_velocity_m_s
                    + pair[1].radial_position_m * pair[1].axial_velocity_m_s)
        })
        .sum::<f64>()
        * 2.0
        * std::f64::consts::PI
}
