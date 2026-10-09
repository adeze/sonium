//! Steady axisymmetric azimuthal momentum transport over a body-fitted mesh.
//!
//! This module is intentionally narrower than a full 3-D rotating CFD solver:
//! it solves the axisymmetric `u_theta` equation on the same `(r, z)` mesh as
//! [`crate::body_fitted_flow_reference`], using its meridional velocity as the
//! advecting field.  The cylindrical diffusion and meridional advection terms
//! are assembled with the physical `2*pi*r` measure, so the published field is
//! an actual transported swirl state rather than a post-processed Couette
//! profile.

use crate::body_fitted_flow_reference::{
    BodyFittedDuctStation, BodyFittedFlowError, BodyFittedFlowOptions, BodyFittedRotatingWallModel,
    solve_body_fitted_duct_flow,
};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_per_element};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::{Mesh, duct_profile_mesh};
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_solvers::SparseLuFactorization;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Versioned identity for the coupled meridional/swirl reference.
pub const BODY_FITTED_SWIRL_TRANSPORT_SCHEMA_V1: &str =
    "propeller.body-fitted-axisymmetric-swirl-transport.v1";

/// Controls the steady scalar azimuthal momentum solve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyFittedSwirlOptions {
    /// Prescribed inlet angular velocity [rad/s] for non-wall inlet nodes.
    pub inlet_angular_velocity_rad_s: f64,
    /// Relative residual accepted from the sparse linear solve.
    pub linear_residual_tolerance: f64,
    /// Maximum number of transported scalar DOFs admitted by this reference.
    pub maximum_dofs: usize,
}

impl Default for BodyFittedSwirlOptions {
    fn default() -> Self {
        Self {
            inlet_angular_velocity_rad_s: 0.0,
            linear_residual_tolerance: 1.0e-9,
            maximum_dofs: 10_000,
        }
    }
}

/// One nodal axisymmetric meridional/swirl state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyFittedSwirlNode {
    /// Radial coordinate [m].
    pub radial_position_m: f64,
    /// Axial coordinate [m].
    pub axial_position_m: f64,
    /// Meridional radial velocity [m/s].
    pub radial_velocity_m_s: f64,
    /// Meridional axial velocity [m/s].
    pub axial_velocity_m_s: f64,
    /// Transported tangential velocity [m/s].
    pub tangential_velocity_m_s: f64,
    /// Tangential velocity divided by radius [rad/s].
    pub angular_velocity_rad_s: f64,
}

/// Coupled body-fitted meridional and azimuthal transport result.
#[derive(Debug, PartialEq)]
pub struct BodyFittedSwirlResult {
    /// Contract schema.
    pub schema: String,
    /// Nodal coupled state.
    pub nodes: Vec<BodyFittedSwirlNode>,
    /// The meridional solve used as the transport advector.
    pub meridional: crate::body_fitted_flow_reference::BodyFittedFlowResult,
    /// Number of generated mesh nodes.
    pub mesh_node_count: usize,
    /// Number of generated mesh elements.
    pub mesh_element_count: usize,
    /// Relative sparse-linear residual.
    pub linear_residual: f64,
    /// Maximum absolute tangential velocity [m/s].
    pub maximum_tangential_speed_m_s: f64,
    /// Maximum absolute angular velocity [rad/s].
    pub maximum_angular_velocity_rad_s: f64,
}

impl BodyFittedSwirlResult {
    /// Whether all published coupled fields and diagnostics are finite.
    pub fn is_finite(&self) -> bool {
        self.schema == BODY_FITTED_SWIRL_TRANSPORT_SCHEMA_V1
            && !self.nodes.is_empty()
            && self.mesh_node_count == self.nodes.len()
            && self.mesh_element_count > 0
            && self.meridional.is_finite()
            && self.linear_residual.is_finite()
            && self.linear_residual >= 0.0
            && self.maximum_tangential_speed_m_s.is_finite()
            && self.maximum_tangential_speed_m_s >= 0.0
            && self.maximum_angular_velocity_rad_s.is_finite()
            && self.maximum_angular_velocity_rad_s >= 0.0
            && self.nodes.iter().all(|node| {
                node.radial_position_m.is_finite()
                    && node.radial_position_m > 0.0
                    && node.axial_position_m.is_finite()
                    && node.radial_velocity_m_s.is_finite()
                    && node.axial_velocity_m_s.is_finite()
                    && node.tangential_velocity_m_s.is_finite()
                    && node.angular_velocity_rad_s.is_finite()
            })
    }
}

/// Failure-closed errors from the coupled swirl transport adapter.
#[derive(Debug, Clone, PartialEq)]
pub enum BodyFittedSwirlError {
    /// The preceding meridional solve rejected its input or state.
    Meridional(BodyFittedFlowError),
    /// A scalar option or geometry value is outside the admitted domain.
    InvalidOption {
        /// Invalid field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The body-fitted mesh could not be reconstructed consistently.
    InvalidMesh,
    /// The scalar sparse solve failed.
    LinearSolve(String),
    /// The solved scalar state or residual is invalid.
    InvalidResult(&'static str),
}

impl fmt::Display for BodyFittedSwirlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meridional(error) => write!(formatter, "meridional flow failed: {error}"),
            Self::InvalidOption { field, requirement } => {
                write!(formatter, "invalid swirl option {field}: {requirement}")
            }
            Self::InvalidMesh => formatter.write_str("body-fitted swirl mesh is inconsistent"),
            Self::LinearSolve(error) => write!(formatter, "swirl transport solve failed: {error}"),
            Self::InvalidResult(field) => {
                write!(formatter, "invalid body-fitted swirl result field {field}")
            }
        }
    }
}

impl std::error::Error for BodyFittedSwirlError {}

impl From<BodyFittedFlowError> for BodyFittedSwirlError {
    fn from(error: BodyFittedFlowError) -> Self {
        Self::Meridional(error)
    }
}

#[derive(Debug, Clone)]
struct SwirlElementForm {
    meridional_velocity: Vec<[f64; 2]>,
    density_kg_m3: f64,
    effective_viscosity_pa_s: f64,
}

impl BilinearForm<f64> for SwirlElementForm {
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
        let radial_velocity = self
            .meridional_velocity
            .iter()
            .enumerate()
            .map(|(index, velocity)| velocity[0] * trial.value(index))
            .sum::<f64>();
        let axial_velocity = self
            .meridional_velocity
            .iter()
            .enumerate()
            .map(|(index, velocity)| velocity[1] * trial.value(index))
            .sum::<f64>();
        let axisymmetric_weight = weight * test.det_j * 2.0 * std::f64::consts::PI * radius;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let diffusion = self.effective_viscosity_pa_s
                    * (test.grad(i, 0) * trial.grad(j, 0)
                        + test.grad(i, 1) * trial.grad(j, 1)
                        + test.value(i) * trial.value(j) / (radius * radius));
                let advection = self.density_kg_m3
                    * test.value(i)
                    * (radial_velocity * trial.grad(j, 0)
                        + axial_velocity * trial.grad(j, 1)
                        + radial_velocity * trial.value(j) / radius);
                local_matrix.add(i, j, (diffusion + advection) * axisymmetric_weight);
            }
        }
    }
}

/// Solve the steady axisymmetric azimuthal momentum equation coupled to a
/// body-fitted meridional FEM state.
pub fn solve_body_fitted_swirl_transport(
    profile: &[BodyFittedDuctStation],
    flow_options: BodyFittedFlowOptions,
    swirl_options: BodyFittedSwirlOptions,
) -> Result<BodyFittedSwirlResult, BodyFittedSwirlError> {
    validate_swirl_options(profile, swirl_options)?;
    let meridional = solve_body_fitted_duct_flow(profile, flow_options)?;
    solve_body_fitted_swirl_transport_with_meridional(
        profile,
        flow_options,
        swirl_options,
        meridional,
    )
}

/// Solve only the azimuthal transport stage for a previously computed
/// meridional state on the matching body-fitted mesh.
pub(crate) fn solve_body_fitted_swirl_transport_with_meridional(
    profile: &[BodyFittedDuctStation],
    flow_options: BodyFittedFlowOptions,
    swirl_options: BodyFittedSwirlOptions,
    meridional: crate::body_fitted_flow_reference::BodyFittedFlowResult,
) -> Result<BodyFittedSwirlResult, BodyFittedSwirlError> {
    validate_swirl_options(profile, swirl_options)?;
    let mesh = build_matching_mesh(profile, &flow_options)?;
    if mesh.num_nodes() != meridional.nodes.len()
        || mesh.num_elements() != meridional.mesh_element_count
    {
        return Err(BodyFittedSwirlError::InvalidMesh);
    }

    let scalar_space = H1Space::new(PolynomialDegree::P1);
    if scalar_space.num_global_dofs(&mesh) > swirl_options.maximum_dofs {
        return Err(BodyFittedSwirlError::InvalidOption {
            field: "maximum_dofs",
            requirement: "at least the generated scalar DOF count",
        });
    }
    let effective_viscosity = flow_options
        .turbulence_model
        .effective_dynamic_viscosity_pa_s(
            flow_options.dynamic_viscosity_pa_s,
            flow_options.density_kg_m3,
            meridional.maximum_speed_m_s,
        );
    if !effective_viscosity.is_finite() || effective_viscosity <= 0.0 {
        return Err(BodyFittedSwirlError::InvalidResult("effective_viscosity"));
    }

    let element_forms = mesh
        .elements
        .iter()
        .enumerate()
        .map(|(element_index, element)| {
            let local_dofs = scalar_space.local_to_global(&mesh, element_index);
            if local_dofs.len() != element.nodes.len() {
                return Err(BodyFittedSwirlError::InvalidMesh);
            }
            let local_velocity = local_dofs
                .iter()
                .map(|&dof| {
                    meridional
                        .nodes
                        .get(dof)
                        .map(|node| [node.radial_velocity_m_s, node.axial_velocity_m_s])
                        .ok_or(BodyFittedSwirlError::InvalidMesh)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(SwirlElementForm {
                meridional_velocity: local_velocity,
                density_kg_m3: flow_options.density_kg_m3,
                effective_viscosity_pa_s: effective_viscosity,
            })
        })
        .collect::<Result<Vec<_>, BodyFittedSwirlError>>()?;

    let assembled =
        assemble_bilinear_per_element(&element_forms, &scalar_space, &scalar_space, &mesh, 2);
    let (conditioned, rhs) = apply_swirl_dirichlet_conditions(
        assembled,
        &mesh,
        &scalar_space,
        profile,
        flow_options.rotating_wall_model,
        swirl_options.inlet_angular_velocity_rad_s,
    )?;
    let csr = conditioned.to_csr();
    let factor = SparseLuFactorization::factorize(&csr)
        .map_err(|error| BodyFittedSwirlError::LinearSolve(error.to_string()))?;
    let tangential = factor
        .solve(&rhs)
        .map_err(|error| BodyFittedSwirlError::LinearSolve(error.to_string()))?;
    if tangential.len() != mesh.num_nodes() || tangential.iter().any(|value| !value.is_finite()) {
        return Err(BodyFittedSwirlError::InvalidResult("tangential_velocity"));
    }

    let residual_vector = csr.matvec(&tangential);
    let residual_norm = residual_vector
        .iter()
        .zip(rhs.iter())
        .map(|(actual, expected)| actual - expected)
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    let linear_residual = residual_norm / rhs_norm.max(1.0);
    if !linear_residual.is_finite() || linear_residual > swirl_options.linear_residual_tolerance {
        return Err(BodyFittedSwirlError::InvalidResult("linear_residual"));
    }

    let nodes = mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let tangential_velocity_m_s = tangential[index];
            BodyFittedSwirlNode {
                radial_position_m: point.x,
                axial_position_m: point.y,
                radial_velocity_m_s: meridional.nodes[index].radial_velocity_m_s,
                axial_velocity_m_s: meridional.nodes[index].axial_velocity_m_s,
                tangential_velocity_m_s,
                angular_velocity_rad_s: tangential_velocity_m_s / point.x,
            }
        })
        .collect::<Vec<_>>();
    let maximum_tangential_speed_m_s = nodes
        .iter()
        .map(|node| node.tangential_velocity_m_s.abs())
        .fold(0.0, f64::max);
    let maximum_angular_velocity_rad_s = nodes
        .iter()
        .map(|node| node.angular_velocity_rad_s.abs())
        .fold(0.0, f64::max);
    let result = BodyFittedSwirlResult {
        schema: BODY_FITTED_SWIRL_TRANSPORT_SCHEMA_V1.to_owned(),
        nodes,
        mesh_node_count: mesh.num_nodes(),
        mesh_element_count: mesh.num_elements(),
        linear_residual,
        maximum_tangential_speed_m_s,
        maximum_angular_velocity_rad_s,
        meridional,
    };
    if !result.is_finite() {
        return Err(BodyFittedSwirlError::InvalidResult("finite_state"));
    }
    Ok(result)
}

fn validate_swirl_options(
    profile: &[BodyFittedDuctStation],
    options: BodyFittedSwirlOptions,
) -> Result<(), BodyFittedSwirlError> {
    if profile.is_empty() {
        return Err(BodyFittedSwirlError::InvalidOption {
            field: "profile",
            requirement: "non-empty",
        });
    }
    if profile.iter().any(|station| station.inner_radius_m <= 0.0) {
        return Err(BodyFittedSwirlError::InvalidOption {
            field: "inner_radius_m",
            requirement: "strictly positive for cylindrical swirl transport",
        });
    }
    if !options.inlet_angular_velocity_rad_s.is_finite() {
        return Err(BodyFittedSwirlError::InvalidOption {
            field: "inlet_angular_velocity_rad_s",
            requirement: "finite",
        });
    }
    if !options.linear_residual_tolerance.is_finite() || options.linear_residual_tolerance <= 0.0 {
        return Err(BodyFittedSwirlError::InvalidOption {
            field: "linear_residual_tolerance",
            requirement: "finite positive",
        });
    }
    if options.maximum_dofs == 0 {
        return Err(BodyFittedSwirlError::InvalidOption {
            field: "maximum_dofs",
            requirement: "positive",
        });
    }
    Ok(())
}

fn build_matching_mesh(
    profile: &[BodyFittedDuctStation],
    options: &BodyFittedFlowOptions,
) -> Result<Mesh, BodyFittedSwirlError> {
    let z_min = profile
        .first()
        .ok_or(BodyFittedSwirlError::InvalidMesh)?
        .axial_position_m;
    let z_max = profile
        .last()
        .ok_or(BodyFittedSwirlError::InvalidMesh)?
        .axial_position_m;
    let inner = profile.to_vec();
    let outer = profile.to_vec();
    let mesh = duct_profile_mesh(
        z_min,
        z_max,
        move |z| interpolate_profile(&inner, z).0,
        move |z| interpolate_profile(&outer, z).1,
        options.axial_elements,
        options.radial_elements,
    );
    if mesh.num_nodes() > options.maximum_nodes || mesh.num_elements() > options.maximum_elements {
        return Err(BodyFittedSwirlError::InvalidMesh);
    }
    Ok(mesh)
}

fn interpolate_profile(profile: &[BodyFittedDuctStation], z: f64) -> (f64, f64) {
    if z <= profile[0].axial_position_m {
        return (profile[0].inner_radius_m, profile[0].outer_radius_m);
    }
    if z >= profile[profile.len() - 1].axial_position_m {
        let station = profile[profile.len() - 1];
        return (station.inner_radius_m, station.outer_radius_m);
    }
    for pair in profile.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if z <= right.axial_position_m {
            let fraction =
                (z - left.axial_position_m) / (right.axial_position_m - left.axial_position_m);
            return (
                left.inner_radius_m + fraction * (right.inner_radius_m - left.inner_radius_m),
                left.outer_radius_m + fraction * (right.outer_radius_m - left.outer_radius_m),
            );
        }
    }
    unreachable!("profile interpolation bounds checked")
}

fn apply_swirl_dirichlet_conditions(
    assembled: TripletMatrix<f64>,
    mesh: &Mesh,
    space: &H1Space,
    profile: &[BodyFittedDuctStation],
    rotating_wall_model: BodyFittedRotatingWallModel,
    inlet_angular_velocity_rad_s: f64,
) -> Result<(TripletMatrix<f64>, Vec<f64>), BodyFittedSwirlError> {
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
    let z_min = profile[0].axial_position_m;
    let wall_dofs = locate_dofs_geometrical(mesh, space, |radius, z, _| {
        let (inner, outer) = interpolate_profile(profile, z);
        (radius - inner).abs() <= boundary_tolerance || (radius - outer).abs() <= boundary_tolerance
    });
    let inlet_dofs = locate_dofs_geometrical(mesh, space, |radius, z, _| {
        (z - z_min).abs() <= boundary_tolerance
            && (radius - interpolate_profile(profile, z).0).abs() > boundary_tolerance
            && (radius - interpolate_profile(profile, z).1).abs() > boundary_tolerance
    });
    if wall_dofs.is_empty() || inlet_dofs.is_empty() {
        return Err(BodyFittedSwirlError::InvalidMesh);
    }

    let mut prescribed = HashMap::new();
    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or(BodyFittedSwirlError::InvalidMesh)?;
    let wall_set: HashSet<usize> = wall_dofs.iter().copied().collect();
    for dof in wall_dofs.iter().chain(inlet_dofs.iter()).copied() {
        let point = coordinates
            .get(dof)
            .ok_or(BodyFittedSwirlError::InvalidMesh)?;
        let (inner, _) = interpolate_profile(profile, point[1]);
        let is_inner = (point[0] - inner).abs() <= boundary_tolerance;
        let wall_angular_velocity = match rotating_wall_model {
            BodyFittedRotatingWallModel::None => 0.0,
            BodyFittedRotatingWallModel::AnnularCouette {
                inner_angular_velocity_rad_s,
                outer_angular_velocity_rad_s: _,
            } if is_inner => inner_angular_velocity_rad_s,
            BodyFittedRotatingWallModel::AnnularCouette {
                outer_angular_velocity_rad_s,
                ..
            } => outer_angular_velocity_rad_s,
        };
        let value = if wall_set.contains(&dof) {
            wall_angular_velocity * point[0]
        } else {
            inlet_angular_velocity_rad_s * point[0]
        };
        if !value.is_finite() {
            return Err(BodyFittedSwirlError::InvalidResult("boundary_value"));
        }
        prescribed.insert(dof, value);
    }

    let mut rhs = vec![0.0; assembled.num_rows];
    let prescribed_dofs: HashSet<usize> = prescribed.keys().copied().collect();
    for index in 0..assembled.nnz() {
        let row = assembled.rows[index];
        let col = assembled.cols[index];
        if !prescribed_dofs.contains(&row)
            && let Some(value) = prescribed.get(&col)
        {
            rhs[row] -= assembled.values[index] * value;
        }
    }
    let mut conditioned = TripletMatrix::with_capacity(
        assembled.num_rows,
        assembled.num_cols,
        assembled.nnz() + prescribed.len(),
    );
    for index in 0..assembled.nnz() {
        let row = assembled.rows[index];
        let col = assembled.cols[index];
        if !prescribed_dofs.contains(&row) && !prescribed_dofs.contains(&col) {
            conditioned.push(row, col, assembled.values[index]);
        }
    }
    for (dof, value) in prescribed {
        conditioned.push(dof, dof, 1.0);
        rhs[dof] = value;
    }
    Ok((conditioned, rhs))
}
