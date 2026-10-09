//! Checked body-fitted meridional duct-flow reference.
//!
//! This module adapts the production FEM incompressible Navier--Stokes Picard
//! operator to a caller-supplied `(r, z)` duct/shroud profile.  The generated
//! mesh follows both the inner and outer walls, so this is a real body-fitted
//! laminar flow state rather than the quasi-one-dimensional station model in
//! [`crate::duct_shroud_reference`].
//!
//! The state is deliberately labelled meridional and laminar.  The FEM
//! operator is not an axisymmetric weighted formulation and this contract
//! does not invent turbulence, rotating-wall, compressibility, or FW-H
//! coupling.  Those effects must be added at a higher fidelity boundary with
//! their own validation evidence.

use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::mesh::duct_profile_mesh;
use sonium_fem::physics::fluid::stokes::navier_stokes::{NavierStokesConfig, solve_picard};
use sonium_fem::physics::fluid::stokes::turbulence::TurbulenceModel;
use sonium_fem::spaces::{H1Space, VectorH1Space};
use std::fmt;
use std::sync::Arc;

/// Versioned identity for this body-fitted flow contract.
pub const BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1: &str =
    "propeller.body-fitted-meridional-flow-reference.v1";

/// One axial station defining the inner and outer body-fitted walls.
/// Turbulence closure exposed by the body-fitted propeller adapter.
///
/// The algebraic SST option selects the bounded closure already implemented
/// by the production FEM solver. It is not a transport-equation k-omega SST
/// implementation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum BodyFittedTurbulenceModel {
    /// No eddy viscosity; use the configured molecular viscosity.
    #[default]
    Laminar,
    /// Bounded algebraic k-omega SST eddy-viscosity closure.
    KOmegaSstAlgebraic {
        /// Empirical eddy-viscosity scale factor.
        c1: f64,
        /// Characteristic wall/shear length scale [m].
        characteristic_length_m: f64,
        /// Maximum turbulent-to-molecular viscosity ratio.
        nu_t_max_ratio: f64,
    },
}

/// Optional annular-Couette rotating-wall source for the meridional solve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum BodyFittedRotatingWallModel {
    /// No rotational body force; preserves the legacy meridional solve.
    #[default]
    None,
    /// Use the exact annular Couette swirl profile to add `u_theta² / r`.
    AnnularCouette {
        /// Inner-wall angular velocity [rad/s].
        inner_angular_velocity_rad_s: f64,
        /// Outer-wall angular velocity [rad/s].
        outer_angular_velocity_rad_s: f64,
    },
}

impl BodyFittedRotatingWallModel {
    fn validate(self) -> Result<(), (&'static str, &'static str)> {
        match self {
            Self::None => Ok(()),
            Self::AnnularCouette {
                inner_angular_velocity_rad_s,
                outer_angular_velocity_rad_s,
            } => {
                for (field, value) in [
                    (
                        "rotating_wall.inner_angular_velocity_rad_s",
                        inner_angular_velocity_rad_s,
                    ),
                    (
                        "rotating_wall.outer_angular_velocity_rad_s",
                        outer_angular_velocity_rad_s,
                    ),
                ] {
                    if !value.is_finite() {
                        return Err((field, "finite"));
                    }
                }
                Ok(())
            }
        }
    }

    /// Evaluate the radial centrifugal acceleration at a meridional point.
    pub fn centrifugal_acceleration_m_s2(
        self,
        radius_m: f64,
        inner_radius_m: f64,
        outer_radius_m: f64,
    ) -> f64 {
        let Self::AnnularCouette {
            inner_angular_velocity_rad_s,
            outer_angular_velocity_rad_s,
        } = self
        else {
            return 0.0;
        };
        let denominator = outer_radius_m * outer_radius_m - inner_radius_m * inner_radius_m;
        if !radius_m.is_finite()
            || !inner_radius_m.is_finite()
            || !outer_radius_m.is_finite()
            || radius_m <= 0.0
            || outer_radius_m <= inner_radius_m
            || denominator <= 0.0
        {
            return f64::NAN;
        }
        let inner_wall_speed = inner_angular_velocity_rad_s * inner_radius_m;
        let outer_wall_speed = outer_angular_velocity_rad_s * outer_radius_m;
        let coefficient_a =
            (outer_wall_speed * outer_radius_m - inner_wall_speed * inner_radius_m) / denominator;
        let coefficient_b = inner_radius_m
            * outer_radius_m
            * (inner_wall_speed * outer_radius_m - outer_wall_speed * inner_radius_m)
            / denominator;
        let tangential_speed = coefficient_a * radius_m + coefficient_b / radius_m;
        tangential_speed * tangential_speed / radius_m
    }
}

impl BodyFittedTurbulenceModel {
    /// Return the corresponding production FEM closure.
    fn to_fem(self) -> TurbulenceModel {
        match self {
            Self::Laminar => TurbulenceModel::Laminar,
            Self::KOmegaSstAlgebraic {
                c1,
                characteristic_length_m,
                nu_t_max_ratio,
            } => TurbulenceModel::KOmegaSstAlgebraic {
                c1,
                characteristic_length_m,
                nu_t_max_ratio,
            },
        }
    }

    /// Compute the closure's effective viscosity for a diagnostic velocity.
    pub fn effective_dynamic_viscosity_pa_s(
        self,
        molecular_viscosity_pa_s: f64,
        density_kg_m3: f64,
        maximum_speed_m_s: f64,
    ) -> f64 {
        self.to_fem()
            .effective_mu(molecular_viscosity_pa_s, density_kg_m3, maximum_speed_m_s)
    }

    fn validate(self) -> Result<(), (&'static str, &'static str)> {
        match self {
            Self::Laminar => Ok(()),
            Self::KOmegaSstAlgebraic {
                c1,
                characteristic_length_m,
                nu_t_max_ratio,
            } => {
                for (field, value) in [
                    ("turbulence.c1", c1),
                    (
                        "turbulence.characteristic_length_m",
                        characteristic_length_m,
                    ),
                    ("turbulence.nu_t_max_ratio", nu_t_max_ratio),
                ] {
                    if !value.is_finite() || value <= 0.0 {
                        return Err((field, "finite positive"));
                    }
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyFittedDuctStation {
    /// Axial coordinate `z` [m].
    pub axial_position_m: f64,
    /// Inner radius `r` [m].
    pub inner_radius_m: f64,
    /// Outer radius `r` [m].
    pub outer_radius_m: f64,
}

/// Controls a bounded body-fitted FEM flow solve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyFittedFlowOptions {
    /// Fluid density [kg/m³].
    pub density_kg_m3: f64,
    /// Dynamic viscosity [Pa·s].
    pub dynamic_viscosity_pa_s: f64,
    /// Laminar or bounded algebraic k-omega SST closure.
    pub turbulence_model: BodyFittedTurbulenceModel,
    /// Optional annular-Couette centrifugal source in the meridional solve.
    pub rotating_wall_model: BodyFittedRotatingWallModel,
    /// Prescribed axial inlet velocity [m/s].
    pub inlet_axial_velocity_m_s: f64,
    /// Number of axial mesh cells.
    pub axial_elements: usize,
    /// Number of radial mesh cells.
    pub radial_elements: usize,
    /// Maximum Picard iterations.
    pub max_picard_iterations: usize,
    /// Minimum Picard iterations before convergence is accepted.
    pub min_picard_iterations: usize,
    /// Relative Picard velocity-change tolerance.
    pub picard_tolerance: f64,
    /// Picard under-relaxation factor.
    pub picard_relaxation: f64,
    /// Maximum iterations for each linearized solve.
    pub max_linear_solver_iterations: usize,
    /// Relative tolerance for each linearized solve.
    pub linear_solver_tolerance: f64,
    /// Reject a finite but unconverged flow state when true.
    pub require_convergence: bool,
    /// Maximum generated mesh nodes.
    pub maximum_nodes: usize,
    /// Maximum generated mesh elements.
    pub maximum_elements: usize,
}

impl Default for BodyFittedFlowOptions {
    fn default() -> Self {
        Self {
            density_kg_m3: 1.2,
            dynamic_viscosity_pa_s: 1.8e-5,
            turbulence_model: BodyFittedTurbulenceModel::Laminar,
            rotating_wall_model: BodyFittedRotatingWallModel::None,
            inlet_axial_velocity_m_s: 8.0,
            axial_elements: 4,
            radial_elements: 3,
            max_picard_iterations: 40,
            min_picard_iterations: 2,
            picard_tolerance: 1.0e-7,
            picard_relaxation: 0.7,
            max_linear_solver_iterations: 200,
            linear_solver_tolerance: 1.0e-9,
            require_convergence: true,
            maximum_nodes: 10_000,
            maximum_elements: 20_000,
        }
    }
}

/// One solved nodal meridional state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyFittedFlowNode {
    /// Radial coordinate [m].
    pub radial_position_m: f64,
    /// Axial coordinate [m].
    pub axial_position_m: f64,
    /// Radial velocity component [m/s].
    pub radial_velocity_m_s: f64,
    /// Axial velocity component [m/s].
    pub axial_velocity_m_s: f64,
    /// FEM pressure DOF [Pa] up to the solver's pressure gauge.
    pub pressure_pa: f64,
}

/// Checked body-fitted flow result and conservation diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BodyFittedFlowResult {
    /// Versioned contract identity.
    pub schema: String,
    /// Solved nodal field in mesh order.
    pub nodes: Vec<BodyFittedFlowNode>,
    /// Number of generated mesh nodes.
    pub mesh_node_count: usize,
    /// Number of generated triangular elements.
    pub mesh_element_count: usize,
    /// Picard iterations performed by the FEM operator.
    pub iterations: usize,
    /// Final relative velocity-change residual.
    pub residual: f64,
    /// Whether the production FEM operator reported convergence.
    pub converged: bool,
    /// Mean inlet axial velocity over non-wall inlet nodes [m/s].
    pub inlet_mean_axial_velocity_m_s: f64,
    /// Mean outlet axial velocity over outlet nodes [m/s].
    pub outlet_mean_axial_velocity_m_s: f64,
    /// Relative inlet/outlet mean-velocity mismatch.
    pub relative_mean_velocity_mismatch: f64,
    /// Maximum nodal speed [m/s].
    pub maximum_speed_m_s: f64,
    /// Minimum nodal pressure [Pa].
    pub minimum_pressure_pa: f64,
    /// Maximum nodal pressure [Pa].
    pub maximum_pressure_pa: f64,
}

impl BodyFittedFlowResult {
    /// Whether all published fields and diagnostics are finite and coherent.
    pub fn is_finite(&self) -> bool {
        self.schema == BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1
            && !self.nodes.is_empty()
            && self.mesh_node_count == self.nodes.len()
            && self.mesh_element_count > 0
            && self.residual.is_finite()
            && self.residual >= 0.0
            && self.inlet_mean_axial_velocity_m_s.is_finite()
            && self.outlet_mean_axial_velocity_m_s.is_finite()
            && self.relative_mean_velocity_mismatch.is_finite()
            && self.relative_mean_velocity_mismatch >= 0.0
            && self.maximum_speed_m_s.is_finite()
            && self.maximum_speed_m_s >= 0.0
            && self.minimum_pressure_pa.is_finite()
            && self.maximum_pressure_pa.is_finite()
            && self.minimum_pressure_pa <= self.maximum_pressure_pa
            && self.nodes.iter().all(|node| {
                node.radial_position_m.is_finite()
                    && node.radial_position_m >= 0.0
                    && node.axial_position_m.is_finite()
                    && node.radial_velocity_m_s.is_finite()
                    && node.axial_velocity_m_s.is_finite()
                    && node.pressure_pa.is_finite()
            })
    }
}

/// Failure-closed errors from the body-fitted flow adapter.
#[derive(Debug, Clone, PartialEq)]
pub enum BodyFittedFlowError {
    /// A scalar option is outside its admitted domain.
    InvalidOption {
        /// Option name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No profile stations were supplied.
    EmptyProfile,
    /// A profile station has malformed or nonphysical geometry.
    InvalidStation {
        /// Zero-based station index.
        index: usize,
        /// Invalid field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Axial station positions are not strictly increasing.
    NonMonotonicProfile {
        /// Later station index.
        index: usize,
    },
    /// Estimated mesh size exceeds a caller budget.
    MeshBudgetExceeded {
        /// Estimated nodes.
        nodes: usize,
        /// Estimated elements.
        elements: usize,
    },
    /// FEM returned an unconverged state and convergence was required.
    SolverDidNotConverge {
        /// Picard iterations performed.
        iterations: usize,
        /// Final residual.
        residual: f64,
    },
    /// A generated or solved field was non-finite or inconsistent.
    InvalidSolverResult {
        /// Field that failed validation.
        field: &'static str,
    },
}

impl fmt::Display for BodyFittedFlowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOption { field, requirement } => {
                write!(
                    formatter,
                    "invalid body-fitted flow option {field}: {requirement}"
                )
            }
            Self::EmptyProfile => formatter.write_str("body-fitted flow profile is empty"),
            Self::InvalidStation {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid body-fitted flow station {index} field {field}: {requirement}"
            ),
            Self::NonMonotonicProfile { index } => write!(
                formatter,
                "body-fitted flow profile axial position is not increasing at {index}"
            ),
            Self::MeshBudgetExceeded { nodes, elements } => write!(
                formatter,
                "body-fitted flow mesh estimate {nodes} nodes and {elements} elements exceeds budget"
            ),
            Self::SolverDidNotConverge {
                iterations,
                residual,
            } => write!(
                formatter,
                "body-fitted FEM flow did not converge after {iterations} iterations (residual {residual})"
            ),
            Self::InvalidSolverResult { field } => {
                write!(
                    formatter,
                    "body-fitted FEM flow result field {field} is invalid"
                )
            }
        }
    }
}

impl std::error::Error for BodyFittedFlowError {}

/// Solve a bounded laminar incompressible flow on a body-fitted meridional
/// duct/shroud mesh.
pub fn solve_body_fitted_duct_flow(
    profile: &[BodyFittedDuctStation],
    options: BodyFittedFlowOptions,
) -> Result<BodyFittedFlowResult, BodyFittedFlowError> {
    solve_body_fitted_duct_flow_with_centrifugal_source(profile, options, None)
}

/// Solve body-fitted flow with an optional caller-supplied centrifugal field.
///
/// The source is an acceleration in the positive radial direction, evaluated
/// at FEM quadrature points.  It is intentionally separate from
/// [`BodyFittedFlowOptions`] so a coupled swirl iteration can feed back its
/// transported `u_theta` field without making a closure part of the serialized
/// options contract.  When absent, the configured analytic rotating-wall
/// model remains the source of the centrifugal acceleration.
pub(crate) fn solve_body_fitted_duct_flow_with_centrifugal_source(
    profile: &[BodyFittedDuctStation],
    options: BodyFittedFlowOptions,
    centrifugal_acceleration: Option<Arc<dyn Fn([f64; 3]) -> f64 + Send + Sync>>,
) -> Result<BodyFittedFlowResult, BodyFittedFlowError> {
    validate_profile(profile)?;
    validate_options(&options)?;

    let node_count = (options.axial_elements + 1)
        .checked_mul(options.radial_elements + 1)
        .ok_or(BodyFittedFlowError::MeshBudgetExceeded {
            nodes: usize::MAX,
            elements: usize::MAX,
        })?;
    let element_count = options
        .axial_elements
        .checked_mul(options.radial_elements)
        .and_then(|count| count.checked_mul(2))
        .ok_or(BodyFittedFlowError::MeshBudgetExceeded {
            nodes: node_count,
            elements: usize::MAX,
        })?;
    if node_count > options.maximum_nodes || element_count > options.maximum_elements {
        return Err(BodyFittedFlowError::MeshBudgetExceeded {
            nodes: node_count,
            elements: element_count,
        });
    }

    let z_min = profile[0].axial_position_m;
    let z_max = profile
        .last()
        .ok_or(BodyFittedFlowError::EmptyProfile)?
        .axial_position_m;
    let inner_profile = profile.to_vec();
    let outer_profile = profile.to_vec();
    let mesh = duct_profile_mesh(
        z_min,
        z_max,
        move |z| interpolate_profile(&inner_profile, z).0,
        move |z| interpolate_profile(&outer_profile, z).1,
        options.axial_elements,
        options.radial_elements,
    );

    if mesh.num_nodes() != node_count || mesh.num_elements() != element_count {
        return Err(BodyFittedFlowError::InvalidSolverResult {
            field: "generated_mesh_size",
        });
    }

    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let boundary_tolerance = ((z_max - z_min) / options.axial_elements as f64).min(
        profile
            .iter()
            .map(|station| station.outer_radius_m - station.inner_radius_m)
            .fold(f64::INFINITY, f64::min)
            / options.radial_elements as f64,
    ) * 1.0e-6;
    let boundary_tolerance = boundary_tolerance.max(1.0e-12);
    let boundary_profile = profile.to_vec();
    let boundary_dofs = locate_dofs_geometrical(&mesh, &velocity_space, move |r, z, _| {
        let (inner, outer) = interpolate_profile(&boundary_profile, z);
        (z - z_min).abs() <= boundary_tolerance
            || (r - inner).abs() <= boundary_tolerance
            || (r - outer).abs() <= boundary_tolerance
    });
    if boundary_dofs.is_empty() {
        return Err(BodyFittedFlowError::InvalidSolverResult {
            field: "boundary_dofs",
        });
    }

    let bc_profile = profile.to_vec();
    let bc_fn = move |r: f64, z: f64, _unused: f64| -> Vec<f64> {
        let (inner, outer) = interpolate_profile(&bc_profile, z);
        let on_wall =
            (r - inner).abs() <= boundary_tolerance || (r - outer).abs() <= boundary_tolerance;
        if on_wall {
            vec![0.0, 0.0]
        } else {
            vec![0.0, options.inlet_axial_velocity_m_s]
        }
    };

    let config = NavierStokesConfig {
        mu: options.dynamic_viscosity_pa_s,
        rho: options.density_kg_m3,
        max_picard_iterations: options.max_picard_iterations,
        min_picard_iterations: options.min_picard_iterations,
        picard_tolerance: options.picard_tolerance,
        picard_relaxation: options.picard_relaxation,
        linear_solver_max_iterations: options.max_linear_solver_iterations,
        linear_solver_tolerance: options.linear_solver_tolerance,
        use_newton_refinement: false,
        turbulence: options.turbulence_model.to_fem(),
        ..NavierStokesConfig::default()
    };

    let force_profile = profile.to_vec();
    let rotating_wall_model = options.rotating_wall_model;
    let density = options.density_kg_m3;
    let body_force = move |position: [f64; 3]| {
        let (inner, outer) = interpolate_profile(&force_profile, position[1]);
        let acceleration = centrifugal_acceleration.as_ref().map_or_else(
            || rotating_wall_model.centrifugal_acceleration_m_s2(position[0], inner, outer),
            |source| source(position),
        );
        vec![density * acceleration, 0.0]
    };
    let fem_result = solve_picard(
        &mesh,
        &velocity_space,
        &pressure_space,
        &config,
        &bc_fn,
        &boundary_dofs,
        body_force,
    );
    if options.require_convergence && !fem_result.converged {
        return Err(BodyFittedFlowError::SolverDidNotConverge {
            iterations: fem_result.iterations,
            residual: fem_result.residual,
        });
    }
    if !fem_result.residual.is_finite()
        || fem_result.residual < 0.0
        || fem_result.velocity.len() != 2 * mesh.num_nodes()
        || fem_result.pressure.len() != mesh.num_nodes()
        || fem_result
            .velocity
            .iter()
            .chain(fem_result.pressure.iter())
            .any(|value| !value.is_finite())
    {
        return Err(BodyFittedFlowError::InvalidSolverResult { field: "fem_state" });
    }

    let nodes = mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| BodyFittedFlowNode {
            radial_position_m: point.x,
            axial_position_m: point.y,
            radial_velocity_m_s: fem_result.velocity[2 * index],
            axial_velocity_m_s: fem_result.velocity[2 * index + 1],
            pressure_pa: fem_result.pressure[index],
        })
        .collect::<Vec<_>>();

    let inlet_nodes = nodes
        .iter()
        .filter(|node| (node.axial_position_m - z_min).abs() <= boundary_tolerance)
        .filter(|node| {
            let (inner, outer) = interpolate_profile(profile, node.axial_position_m);
            (node.radial_position_m - inner).abs() > boundary_tolerance
                && (node.radial_position_m - outer).abs() > boundary_tolerance
        })
        .collect::<Vec<_>>();
    let outlet_nodes = nodes
        .iter()
        .filter(|node| (node.axial_position_m - z_max).abs() <= boundary_tolerance)
        .collect::<Vec<_>>();
    if inlet_nodes.is_empty() || outlet_nodes.is_empty() {
        return Err(BodyFittedFlowError::InvalidSolverResult {
            field: "inlet_or_outlet_nodes",
        });
    }

    let mean = |values: &[&BodyFittedFlowNode]| {
        values
            .iter()
            .map(|node| node.axial_velocity_m_s)
            .sum::<f64>()
            / values.len() as f64
    };
    let inlet_mean = mean(&inlet_nodes);
    let outlet_mean = mean(&outlet_nodes);
    let scale = inlet_mean
        .abs()
        .max(options.inlet_axial_velocity_m_s.abs())
        .max(1.0);
    let relative_mismatch = (outlet_mean - inlet_mean).abs() / scale;
    let maximum_speed = nodes
        .iter()
        .map(|node| (node.radial_velocity_m_s.powi(2) + node.axial_velocity_m_s.powi(2)).sqrt())
        .fold(0.0, f64::max);
    let minimum_pressure = nodes
        .iter()
        .map(|node| node.pressure_pa)
        .fold(f64::INFINITY, f64::min);
    let maximum_pressure = nodes
        .iter()
        .map(|node| node.pressure_pa)
        .fold(f64::NEG_INFINITY, f64::max);
    let result = BodyFittedFlowResult {
        schema: BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1.to_owned(),
        mesh_node_count: mesh.num_nodes(),
        mesh_element_count: mesh.num_elements(),
        nodes,
        iterations: fem_result.iterations,
        residual: fem_result.residual,
        converged: fem_result.converged,
        inlet_mean_axial_velocity_m_s: inlet_mean,
        outlet_mean_axial_velocity_m_s: outlet_mean,
        relative_mean_velocity_mismatch: relative_mismatch,
        maximum_speed_m_s: maximum_speed,
        minimum_pressure_pa: minimum_pressure,
        maximum_pressure_pa: maximum_pressure,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(BodyFittedFlowError::InvalidSolverResult {
            field: "published_result",
        })
    }
}

fn validate_profile(profile: &[BodyFittedDuctStation]) -> Result<(), BodyFittedFlowError> {
    if profile.is_empty() {
        return Err(BodyFittedFlowError::EmptyProfile);
    }
    for (index, station) in profile.iter().enumerate() {
        if !station.axial_position_m.is_finite() {
            return Err(BodyFittedFlowError::InvalidStation {
                index,
                field: "axial_position_m",
                requirement: "finite",
            });
        }
        if !station.inner_radius_m.is_finite() || station.inner_radius_m < 0.0 {
            return Err(BodyFittedFlowError::InvalidStation {
                index,
                field: "inner_radius_m",
                requirement: "finite non-negative",
            });
        }
        if !station.outer_radius_m.is_finite() || station.outer_radius_m <= station.inner_radius_m {
            return Err(BodyFittedFlowError::InvalidStation {
                index,
                field: "outer_radius_m",
                requirement: "finite and greater than inner radius",
            });
        }
        if index > 0
            && let Some(previous) = profile.get(index - 1)
            && station.axial_position_m <= previous.axial_position_m
        {
            return Err(BodyFittedFlowError::NonMonotonicProfile { index });
        }
    }
    if profile.len() < 2 {
        return Err(BodyFittedFlowError::InvalidStation {
            index: 0,
            field: "profile",
            requirement: "at least two axial stations",
        });
    }
    Ok(())
}

fn validate_options(options: &BodyFittedFlowOptions) -> Result<(), BodyFittedFlowError> {
    for (field, value) in [
        ("density_kg_m3", options.density_kg_m3),
        ("dynamic_viscosity_pa_s", options.dynamic_viscosity_pa_s),
        ("picard_tolerance", options.picard_tolerance),
        ("picard_relaxation", options.picard_relaxation),
        ("linear_solver_tolerance", options.linear_solver_tolerance),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(BodyFittedFlowError::InvalidOption {
                field,
                requirement: "finite positive",
            });
        }
    }
    if let Err((field, requirement)) = options.turbulence_model.validate() {
        return Err(BodyFittedFlowError::InvalidOption { field, requirement });
    }
    if let Err((field, requirement)) = options.rotating_wall_model.validate() {
        return Err(BodyFittedFlowError::InvalidOption { field, requirement });
    }
    if !options.inlet_axial_velocity_m_s.is_finite() {
        return Err(BodyFittedFlowError::InvalidOption {
            field: "inlet_axial_velocity_m_s",
            requirement: "finite",
        });
    }
    if !(0.0 < options.picard_relaxation && options.picard_relaxation <= 1.0) {
        return Err(BodyFittedFlowError::InvalidOption {
            field: "picard_relaxation",
            requirement: "in (0, 1]",
        });
    }
    if options.axial_elements == 0
        || options.radial_elements == 0
        || options.max_picard_iterations == 0
        || options.min_picard_iterations == 0
        || options.min_picard_iterations > options.max_picard_iterations
        || options.max_linear_solver_iterations == 0
        || options.maximum_nodes == 0
        || options.maximum_elements == 0
    {
        return Err(BodyFittedFlowError::InvalidOption {
            field: "mesh_or_solver_budget",
            requirement: "positive and internally ordered",
        });
    }
    Ok(())
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
    unreachable!("profile bounds handled above")
}
