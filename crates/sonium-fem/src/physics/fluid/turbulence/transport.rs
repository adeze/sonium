//! Checked, stateful k--omega scalar transport.
//!
//! The existing turbulence forms assemble one frozen-coefficient scalar
//! operator.  This module supplies the missing time-level contract around
//! those forms: BDF history, production, sparse solves, positivity checks,
//! and residual diagnostics.  Momentum coupling is intentionally outside this
//! API; a caller must use the returned state and its own coupled Jacobian
//! before admitting a turbulent Newton or adjoint calculation.

use super::{
    KOmegaSstConstants, TurbulenceProductionForm, TurbulenceTransportForm, eddy_viscosity_k_omega,
};
use crate::assembly::prebaked::PrebakedAssembler;
use crate::assembly::{
    assemble_bilinear, assemble_bilinear_per_element, assemble_linear_per_element,
};
use crate::basis::PolynomialDegree;
use crate::forms::MassForm;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::{FunctionSpace, H1Space};
use sonium_solvers::direct::SparseLuFactorization;
use sonium_solvers::traits::LinearOperator;
use thiserror::Error;

/// Stable provenance for the frozen-coefficient scalar transport contract.
pub const K_OMEGA_TRANSPORT_PROVENANCE: &str = "fem:turbulence:k-omega-bdf-transport:v1";

/// Provenance for the exact tangent of the frozen-coefficient scalar BDF
/// residual assembled by [`advance_komega_bdf_step_with_element_fields`].
///
/// Advecting velocity, eddy viscosity, production, and reaction coefficients
/// are frozen for the segregated update; this is not a coupled
/// momentum/k--omega Newton tangent.
pub const K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE: &str =
    "fem:turbulence:k-omega-bdf-frozen-coefficient-tangent:v1";

/// Configuration for one scalar k--omega BDF update.
///
/// The velocity, eddy viscosity, and production values are frozen during the
/// update.  They may be refreshed from the latest mean-flow iterate by the
/// caller for a Picard coupling.  Spatially varying coefficients can be
/// added through the existing per-element assembly API without changing the
/// state or history contract defined here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KOmegaTransportConfig {
    /// BDF time step in seconds.
    pub dt: f64,
    /// BDF order, either 1 or 2.
    pub bdf_order: usize,
    /// Molecular kinematic viscosity.
    pub molecular_viscosity: f64,
    /// Frozen eddy viscosity used in both scalar diffusion operators.
    pub eddy_viscosity: f64,
    /// Frozen advecting velocity in the mesh coordinate frame.
    pub velocity: [f64; 3],
    /// Frozen k-equation production source.
    pub k_production: f64,
    /// Frozen omega-equation production source.
    pub omega_production: f64,
    /// Strict lower bound admitted for both transported fields.
    pub positivity_floor: f64,
    /// SST constants used for reaction and k-production limiting.
    pub constants: KOmegaSstConstants,
}

impl Default for KOmegaTransportConfig {
    fn default() -> Self {
        Self {
            dt: 1.0e-3,
            bdf_order: 1,
            molecular_viscosity: 1.5e-5,
            eddy_viscosity: 0.0,
            velocity: [0.0; 3],
            k_production: 0.0,
            omega_production: 0.0,
            positivity_floor: 1.0e-12,
            constants: KOmegaSstConstants::default(),
        }
    }
}

/// Current and retained previous time levels for the two transported fields.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaTransportState {
    /// Current turbulent kinetic energy, in scalar-space DOF order.
    pub k: Vec<f64>,
    /// Previous-step turbulent kinetic energy.
    pub k_prev: Vec<f64>,
    /// Current specific dissipation rate, in scalar-space DOF order.
    pub omega: Vec<f64>,
    /// Previous-step specific dissipation rate.
    pub omega_prev: Vec<f64>,
    /// Physical time of the current state.
    pub time: f64,
    /// Number of accepted updates represented by this state.
    pub step: usize,
}

/// Frozen element-local coefficients for one scalar k--omega update.
///
/// All vectors must have exactly `mesh.num_elements()` entries. Keeping these
/// values element-local makes the transport API useful for a resolved flow
/// field while retaining a segregated/Picard coupling boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaTransportElementFields {
    /// Advecting velocity for each mesh element.
    pub velocity: Vec<[f64; 3]>,
    /// Eddy viscosity used by both scalar diffusion operators.
    pub eddy_viscosity: Vec<f64>,
    /// k-equation production source for each element.
    pub k_production: Vec<f64>,
    /// omega-equation production source for each element.
    pub omega_production: Vec<f64>,
}

impl KOmegaTransportElementFields {
    /// Build element fields from uniform coefficients.
    pub fn uniform(mesh: &Mesh, config: &KOmegaTransportConfig) -> Self {
        let elements = mesh.num_elements();
        Self {
            velocity: vec![config.velocity; elements],
            eddy_viscosity: vec![config.eddy_viscosity; elements],
            k_production: vec![config.k_production; elements],
            omega_production: vec![config.omega_production; elements],
        }
    }
}

impl KOmegaTransportState {
    /// Construct a spatially constant positive state for a manufactured test
    /// or a controlled solver start.
    pub fn from_constant(
        num_dofs: usize,
        k: f64,
        omega: f64,
        time: f64,
        step: usize,
    ) -> Result<Self, KOmegaTransportError> {
        if num_dofs == 0 {
            return Err(KOmegaTransportError::InvalidState(
                "transport state must contain at least one DOF",
            ));
        }
        if !k.is_finite() || !omega.is_finite() || !time.is_finite() || time < 0.0 {
            return Err(KOmegaTransportError::NonFiniteState);
        }
        if k <= 0.0 || omega <= 0.0 {
            return Err(KOmegaTransportError::NonPositiveState {
                field: if k <= 0.0 { "k" } else { "omega" },
                index: 0,
                value: if k <= 0.0 { k } else { omega },
            });
        }
        Ok(Self {
            k: vec![k; num_dofs],
            k_prev: vec![k; num_dofs],
            omega: vec![omega; num_dofs],
            omega_prev: vec![omega; num_dofs],
            time,
            step,
        })
    }

    /// Compute one frozen eddy-viscosity value per P1 mesh element.
    ///
    /// The element value is obtained from the arithmetic means of the current
    /// P1 k and omega nodal fields. This is the checked bridge a segregated
    /// momentum assembly can use between scalar transport updates.
    pub fn p1_eddy_viscosity_per_element(
        &self,
        mesh: &Mesh,
        limiter: f64,
    ) -> Result<Vec<f64>, KOmegaTransportError> {
        if !limiter.is_finite() || limiter <= 0.0 {
            return Err(KOmegaTransportError::InvalidConfig(
                "eddy-viscosity limiter must be finite and positive",
            ));
        }
        if self.k.len() != mesh.num_nodes() {
            return Err(KOmegaTransportError::DimensionMismatch {
                field: "k",
                expected: mesh.num_nodes(),
                actual: self.k.len(),
            });
        }
        if self.omega.len() != mesh.num_nodes() {
            return Err(KOmegaTransportError::DimensionMismatch {
                field: "omega",
                expected: mesh.num_nodes(),
                actual: self.omega.len(),
            });
        }
        let mut values = Vec::with_capacity(mesh.num_elements());
        for (element_index, element) in mesh.elements.iter().enumerate() {
            if element.nodes.is_empty() {
                return Err(KOmegaTransportError::InvalidElementField {
                    field: "element nodes",
                    element: element_index,
                });
            }
            let mut k = 0.0;
            let mut omega = 0.0;
            for &node in &element.nodes {
                let (&k_node, &omega_node) = match (self.k.get(node), self.omega.get(node)) {
                    (Some(k_node), Some(omega_node)) => (k_node, omega_node),
                    _ => {
                        return Err(KOmegaTransportError::InvalidElementField {
                            field: "element nodes",
                            element: element_index,
                        });
                    }
                };
                if !k_node.is_finite()
                    || !omega_node.is_finite()
                    || k_node <= 0.0
                    || omega_node <= 0.0
                {
                    return Err(KOmegaTransportError::InvalidElementField {
                        field: "k/omega",
                        element: element_index,
                    });
                }
                k += k_node;
                omega += omega_node;
            }
            let count = element.nodes.len() as f64;
            let viscosity = eddy_viscosity_k_omega(k / count, omega / count, limiter);
            if !viscosity.is_finite() || viscosity < 0.0 {
                return Err(KOmegaTransportError::InvalidElementField {
                    field: "eddy_viscosity",
                    element: element_index,
                });
            }
            values.push(viscosity);
        }
        Ok(values)
    }
}

/// Diagnostics returned after an accepted k--omega update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KOmegaTransportStepReport {
    /// Stable operator/state provenance.
    pub provenance: &'static str,
    /// BDF order used by the update.
    pub bdf_order: usize,
    /// L2 norm of the assembled k-equation residual.
    pub k_residual_l2: f64,
    /// L2 norm of the assembled omega-equation residual.
    pub omega_residual_l2: f64,
    /// Minimum accepted k value.
    pub minimum_k: f64,
    /// Minimum accepted omega value.
    pub minimum_omega: f64,
}

/// Jacobians of one frozen-coefficient k--omega BDF residual.
///
/// The new-state Jacobians are the matrices factored by the transport solve.
/// The history matrices differentiate the residual with respect to the two
/// retained history levels. Reaction and production coefficients are frozen
/// because that is the contract of the segregated transport advance.
#[derive(Debug, Clone)]
pub struct KOmegaTransportLinearization {
    pub provenance: &'static str,
    pub k_state_jacobian: sonium_solvers::CsrMatrix<f64>,
    pub omega_state_jacobian: sonium_solvers::CsrMatrix<f64>,
    pub k_history_jacobian: sonium_solvers::CsrMatrix<f64>,
    pub k_previous_history_jacobian: sonium_solvers::CsrMatrix<f64>,
    pub omega_history_jacobian: sonium_solvers::CsrMatrix<f64>,
    pub omega_previous_history_jacobian: sonium_solvers::CsrMatrix<f64>,
}

impl PartialEq for KOmegaTransportLinearization {
    fn eq(&self, other: &Self) -> bool {
        self.provenance == other.provenance
            && csr_equal(&self.k_state_jacobian, &other.k_state_jacobian)
            && csr_equal(&self.omega_state_jacobian, &other.omega_state_jacobian)
            && csr_equal(&self.k_history_jacobian, &other.k_history_jacobian)
            && csr_equal(
                &self.k_previous_history_jacobian,
                &other.k_previous_history_jacobian,
            )
            && csr_equal(&self.omega_history_jacobian, &other.omega_history_jacobian)
            && csr_equal(
                &self.omega_previous_history_jacobian,
                &other.omega_previous_history_jacobian,
            )
    }
}

fn csr_equal(
    left: &sonium_solvers::CsrMatrix<f64>,
    right: &sonium_solvers::CsrMatrix<f64>,
) -> bool {
    left.num_rows == right.num_rows
        && left.num_cols == right.num_cols
        && left.values == right.values
        && left.col_indices == right.col_indices
        && left.row_ptrs == right.row_ptrs
}

/// Result of one accepted scalar transport update.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaTransportStep {
    /// New current/previous k--omega history state.
    pub state: KOmegaTransportState,
    /// Admission and residual diagnostics.
    pub report: KOmegaTransportStepReport,
    /// Residual vector of the accepted k solve, in scalar-space DOF order.
    pub k_residual: Vec<f64>,
    /// Residual vector of the accepted omega solve, in scalar-space DOF order.
    pub omega_residual: Vec<f64>,
    /// Exact tangent of the frozen-coefficient residual used for this step.
    pub linearization: KOmegaTransportLinearization,
}

/// Failure while validating or solving a scalar transport update.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum KOmegaTransportError {
    #[error("invalid k--omega transport configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid k--omega transport state: {0}")]
    InvalidState(&'static str),
    #[error("k--omega transport state contains a non-finite value")]
    NonFiniteState,
    #[error("{field}[{index}] is not positive: {value}")]
    NonPositiveState {
        field: &'static str,
        index: usize,
        value: f64,
    },
    #[error("{field} state has {actual} DOFs, expected {expected}")]
    DimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{field} has {actual} element values, expected {expected}")]
    ElementFieldCount {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{field} has an invalid value in element {element}")]
    InvalidElementField { field: &'static str, element: usize },
    #[error("k--omega transport sparse solve failed: {0}")]
    LinearSolve(String),
    #[error("k--omega transport produced a non-finite {field} value at DOF {index}")]
    NonFiniteSolution { field: &'static str, index: usize },
    #[error(
        "k--omega transport solution violated the {field} positivity floor at DOF {index}: {value}"
    )]
    PositivityViolation {
        field: &'static str,
        index: usize,
        value: f64,
    },
}

/// Advance both scalar transport equations by one checked BDF step.
///
/// The two operators are assembled independently on the supplied scalar H1
/// space and solved with sparse LU.  BDF history enters through the consistent
/// mass matrix, so a caller can transfer `KOmegaTransportState` across a mesh
/// change and then reassemble on the target mesh.
pub fn advance_komega_bdf_step(
    mesh: &Mesh,
    degree: PolynomialDegree,
    config: &KOmegaTransportConfig,
    state: &KOmegaTransportState,
) -> Result<KOmegaTransportStep, KOmegaTransportError> {
    let fields = KOmegaTransportElementFields::uniform(mesh, config);
    advance_komega_bdf_step_with_element_fields(mesh, degree, config, state, &fields)
}

/// Advance both scalar transport equations using element-local coefficients.
pub fn advance_komega_bdf_step_with_element_fields(
    mesh: &Mesh,
    degree: PolynomialDegree,
    config: &KOmegaTransportConfig,
    state: &KOmegaTransportState,
    fields: &KOmegaTransportElementFields,
) -> Result<KOmegaTransportStep, KOmegaTransportError> {
    validate_config(config)?;
    validate_element_fields(mesh, fields)?;

    let space = H1Space::new(degree);
    let num_dofs = space.num_global_dofs(mesh);
    validate_state(state, num_dofs, config.positivity_floor)?;

    let coefficients =
        crate::physics::fluid::stokes::time_stepping::BdfCoefficients::for_order(config.bdf_order);
    let omega_reference = state.omega.iter().sum::<f64>() / state.omega.len() as f64;
    let k_reference = state.k.iter().sum::<f64>() / state.k.len() as f64;
    let k_reaction = config.constants.k_reaction(omega_reference);
    let omega_reaction = config.constants.omega_reaction(omega_reference);
    let k_forms = fields
        .velocity
        .iter()
        .zip(&fields.eddy_viscosity)
        .map(|(&velocity, &eddy_viscosity)| TurbulenceTransportForm {
            molecular_viscosity: config.molecular_viscosity,
            eddy_viscosity,
            sigma: config.constants.sigma_k,
            velocity,
            reaction: k_reaction,
        })
        .collect::<Vec<_>>();
    let omega_forms = fields
        .velocity
        .iter()
        .zip(&fields.eddy_viscosity)
        .map(|(&velocity, &eddy_viscosity)| TurbulenceTransportForm {
            molecular_viscosity: config.molecular_viscosity,
            eddy_viscosity,
            sigma: config.constants.sigma_omega,
            velocity,
            reaction: omega_reaction,
        })
        .collect::<Vec<_>>();
    let transport_quad = quadrature::required_order_for_stiffness(degree.degree());
    let mass_quad = quadrature::required_order_for_mass(degree.degree());
    let k_transport = assemble_bilinear_per_element(&k_forms, &space, &space, mesh, transport_quad);
    let omega_transport =
        assemble_bilinear_per_element(&omega_forms, &space, &space, mesh, transport_quad);
    let k_mass = assemble_bilinear(&MassForm, &space, &space, mesh, mass_quad);
    let omega_mass = assemble_bilinear(&MassForm, &space, &space, mesh, mass_quad);
    let k_assembler = PrebakedAssembler::from_triplets([k_transport, k_mass], &[]);
    let omega_assembler = PrebakedAssembler::from_triplets([omega_transport, omega_mass], &[]);
    let mass = k_assembler.assemble_real(&[0.0, 1.0]);

    let temporal_scale = coefficients.a0 / config.dt;
    let k_operator = k_assembler.assemble_real(&[1.0, temporal_scale]);
    let omega_operator = omega_assembler.assemble_real(&[1.0, temporal_scale]);

    let k_history = history_vector(&state.k, &state.k_prev, &coefficients, config.dt);
    let omega_history = history_vector(&state.omega, &state.omega_prev, &coefficients, config.dt);
    let k_source_forms = fields
        .k_production
        .iter()
        .map(|&production| TurbulenceProductionForm {
            production: production
                .min(10.0 * config.constants.beta_star * k_reference * omega_reference),
        })
        .collect::<Vec<_>>();
    let omega_source_forms = fields
        .omega_production
        .iter()
        .map(|&production| TurbulenceProductionForm { production })
        .collect::<Vec<_>>();
    let k_source = assemble_linear_per_element(&k_source_forms, &space, mesh, mass_quad);
    let omega_source = assemble_linear_per_element(&omega_source_forms, &space, mesh, mass_quad);
    let k_rhs = add_vectors(&mass.apply(&k_history), &k_source);
    let omega_rhs = add_vectors(&mass.apply(&omega_history), &omega_source);

    let k_factor = SparseLuFactorization::factorize(&k_operator)
        .map_err(|error| KOmegaTransportError::LinearSolve(error.to_string()))?;
    let omega_factor = SparseLuFactorization::factorize(&omega_operator)
        .map_err(|error| KOmegaTransportError::LinearSolve(error.to_string()))?;
    let k = k_factor
        .solve(&k_rhs)
        .map_err(|error| KOmegaTransportError::LinearSolve(error.to_string()))?;
    let omega = omega_factor
        .solve(&omega_rhs)
        .map_err(|error| KOmegaTransportError::LinearSolve(error.to_string()))?;

    validate_solution("k", &k, config.positivity_floor)?;
    validate_solution("omega", &omega, config.positivity_floor)?;

    let k_residual_l2 = residual_l2(&k_operator, &k, &k_rhs);
    let omega_residual_l2 = residual_l2(&omega_operator, &omega, &omega_rhs);
    let k_residual = k_operator
        .apply(&k)
        .iter()
        .zip(&k_rhs)
        .map(|(&left, &right)| left - right)
        .collect::<Vec<_>>();
    let omega_residual = omega_operator
        .apply(&omega)
        .iter()
        .zip(&omega_rhs)
        .map(|(&left, &right)| left - right)
        .collect::<Vec<_>>();
    let next_state = KOmegaTransportState {
        k_prev: state.k.clone(),
        omega_prev: state.omega.clone(),
        k: k.clone(),
        omega: omega.clone(),
        time: state.time + config.dt,
        step: state.step.saturating_add(1),
    };

    Ok(KOmegaTransportStep {
        state: next_state,
        report: KOmegaTransportStepReport {
            provenance: K_OMEGA_TRANSPORT_PROVENANCE,
            bdf_order: config.bdf_order,
            k_residual_l2,
            omega_residual_l2,
            minimum_k: k.iter().copied().fold(f64::INFINITY, f64::min),
            minimum_omega: omega.iter().copied().fold(f64::INFINITY, f64::min),
        },
        k_residual,
        omega_residual,
        linearization: KOmegaTransportLinearization {
            provenance: K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE,
            k_state_jacobian: k_operator.clone(),
            omega_state_jacobian: omega_operator.clone(),
            k_history_jacobian: scale_csr(&mass, coefficients.a1 / config.dt),
            k_previous_history_jacobian: scale_csr(&mass, coefficients.a2 / config.dt),
            omega_history_jacobian: scale_csr(&mass, coefficients.a1 / config.dt),
            omega_previous_history_jacobian: scale_csr(&mass, coefficients.a2 / config.dt),
        },
    })
}

fn scale_csr(
    matrix: &sonium_solvers::CsrMatrix<f64>,
    factor: f64,
) -> sonium_solvers::CsrMatrix<f64> {
    sonium_solvers::CsrMatrix {
        num_rows: matrix.num_rows,
        num_cols: matrix.num_cols,
        values: matrix.values.iter().map(|value| factor * value).collect(),
        col_indices: matrix.col_indices.clone(),
        row_ptrs: matrix.row_ptrs.clone(),
    }
}

fn validate_element_fields(
    mesh: &Mesh,
    fields: &KOmegaTransportElementFields,
) -> Result<(), KOmegaTransportError> {
    let expected = mesh.num_elements();
    for (field, actual) in [
        ("velocity", fields.velocity.len()),
        ("eddy_viscosity", fields.eddy_viscosity.len()),
        ("k_production", fields.k_production.len()),
        ("omega_production", fields.omega_production.len()),
    ] {
        if actual != expected {
            return Err(KOmegaTransportError::ElementFieldCount {
                field,
                expected,
                actual,
            });
        }
    }
    for (element, velocity) in fields.velocity.iter().enumerate() {
        if velocity.iter().any(|value| !value.is_finite()) {
            return Err(KOmegaTransportError::InvalidElementField {
                field: "velocity",
                element,
            });
        }
    }
    for (element, &value) in fields.eddy_viscosity.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(KOmegaTransportError::InvalidElementField {
                field: "eddy_viscosity",
                element,
            });
        }
    }
    for (field, values) in [
        ("k_production", &fields.k_production),
        ("omega_production", &fields.omega_production),
    ] {
        for (element, &value) in values.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(KOmegaTransportError::InvalidElementField { field, element });
            }
        }
    }
    Ok(())
}

fn validate_config(config: &KOmegaTransportConfig) -> Result<(), KOmegaTransportError> {
    if !config.dt.is_finite() || config.dt <= 0.0 {
        return Err(KOmegaTransportError::InvalidConfig(
            "dt must be finite and positive",
        ));
    }
    if !matches!(config.bdf_order, 1 | 2) {
        return Err(KOmegaTransportError::InvalidConfig(
            "bdf_order must be 1 or 2",
        ));
    }
    if !config.molecular_viscosity.is_finite() || config.molecular_viscosity < 0.0 {
        return Err(KOmegaTransportError::InvalidConfig(
            "molecular_viscosity must be finite and non-negative",
        ));
    }
    if !config.eddy_viscosity.is_finite() || config.eddy_viscosity < 0.0 {
        return Err(KOmegaTransportError::InvalidConfig(
            "eddy_viscosity must be finite and non-negative",
        ));
    }
    if config.velocity.iter().any(|value| !value.is_finite()) {
        return Err(KOmegaTransportError::InvalidConfig(
            "velocity must contain only finite values",
        ));
    }
    if !config.k_production.is_finite() || config.k_production < 0.0 {
        return Err(KOmegaTransportError::InvalidConfig(
            "k_production must be finite and non-negative",
        ));
    }
    if !config.omega_production.is_finite() || config.omega_production < 0.0 {
        return Err(KOmegaTransportError::InvalidConfig(
            "omega_production must be finite and non-negative",
        ));
    }
    if !config.positivity_floor.is_finite() || config.positivity_floor <= 0.0 {
        return Err(KOmegaTransportError::InvalidConfig(
            "positivity_floor must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_state(
    state: &KOmegaTransportState,
    expected: usize,
    floor: f64,
) -> Result<(), KOmegaTransportError> {
    if !state.time.is_finite() || state.time < 0.0 {
        return Err(KOmegaTransportError::NonFiniteState);
    }
    for (field, values) in [
        ("k", &state.k),
        ("k_prev", &state.k_prev),
        ("omega", &state.omega),
        ("omega_prev", &state.omega_prev),
    ] {
        if values.len() != expected {
            return Err(KOmegaTransportError::DimensionMismatch {
                field,
                expected,
                actual: values.len(),
            });
        }
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(KOmegaTransportError::NonFiniteSolution { field, index });
            }
            if value < floor {
                return Err(KOmegaTransportError::NonPositiveState {
                    field,
                    index,
                    value,
                });
            }
        }
    }
    Ok(())
}

fn validate_solution(
    field: &'static str,
    values: &[f64],
    floor: f64,
) -> Result<(), KOmegaTransportError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(KOmegaTransportError::NonFiniteSolution { field, index });
        }
        if value < floor {
            return Err(KOmegaTransportError::PositivityViolation {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn history_vector(
    current: &[f64],
    previous: &[f64],
    coefficients: &crate::physics::fluid::stokes::time_stepping::BdfCoefficients,
    dt: f64,
) -> Vec<f64> {
    current
        .iter()
        .zip(previous)
        .map(|(&now, &before)| (-coefficients.a1 * now - coefficients.a2 * before) / dt)
        .collect()
}

fn add_vectors(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(&a, &b)| a + b).collect()
}

fn residual_l2(matrix: &sonium_solvers::CsrMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    matrix
        .apply(x)
        .iter()
        .zip(rhs)
        .map(|(&lhs, &b)| {
            let residual = lhs - b;
            residual * residual
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manufactured_config(order: usize, k: f64, omega: f64) -> KOmegaTransportConfig {
        let constants = KOmegaSstConstants::default();
        KOmegaTransportConfig {
            bdf_order: order,
            molecular_viscosity: 1.0e-5,
            k_production: constants.beta_star * omega * k,
            omega_production: constants.beta * omega * omega,
            constants,
            ..KOmegaTransportConfig::default()
        }
    }

    #[test]
    fn bdf1_preserves_constant_manufactured_state() {
        let mesh = crate::mesh::unit_square_triangles(2);
        let degree = PolynomialDegree::P1;
        let dofs = H1Space::new(degree).num_global_dofs(&mesh);
        let state = KOmegaTransportState::from_constant(dofs, 2.0, 3.0, 0.0, 0).unwrap();
        let result =
            advance_komega_bdf_step(&mesh, degree, &manufactured_config(1, 2.0, 3.0), &state)
                .unwrap();

        assert!(
            result
                .state
                .k
                .iter()
                .all(|value| (*value - 2.0).abs() < 1.0e-10)
        );
        assert!(
            result
                .state
                .omega
                .iter()
                .all(|value| (*value - 3.0).abs() < 1.0e-10)
        );
        assert!(result.report.k_residual_l2 < 1.0e-10);
        assert!(result.report.omega_residual_l2 < 1.0e-10);
        assert_eq!(result.state.step, 1);
    }

    #[test]
    fn bdf2_preserves_constant_manufactured_history() {
        let mesh = crate::mesh::unit_square_triangles(2);
        let degree = PolynomialDegree::P1;
        let dofs = H1Space::new(degree).num_global_dofs(&mesh);
        let mut state = KOmegaTransportState::from_constant(dofs, 2.0, 3.0, 0.1, 4).unwrap();
        state.k_prev.fill(2.0);
        state.omega_prev.fill(3.0);
        let result =
            advance_komega_bdf_step(&mesh, degree, &manufactured_config(2, 2.0, 3.0), &state)
                .unwrap();

        assert!(
            result
                .state
                .k
                .iter()
                .all(|value| (*value - 2.0).abs() < 1.0e-10)
        );
        assert!(
            result
                .state
                .omega
                .iter()
                .all(|value| (*value - 3.0).abs() < 1.0e-10)
        );
        assert_eq!(result.state.time, 0.1 + KOmegaTransportConfig::default().dt);
        assert_eq!(result.state.step, 5);
    }

    #[test]
    fn frozen_transport_step_publishes_state_and_history_tangents() {
        let mesh = crate::mesh::unit_square_triangles(1);
        let degree = PolynomialDegree::P1;
        let dofs = H1Space::new(degree).num_global_dofs(&mesh);
        let state = KOmegaTransportState::from_constant(dofs, 2.0, 3.0, 0.0, 0).unwrap();
        let result =
            advance_komega_bdf_step(&mesh, degree, &manufactured_config(2, 2.0, 3.0), &state)
                .unwrap();

        let tangent = &result.linearization;
        assert_eq!(
            tangent.provenance,
            K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE
        );
        assert_eq!(tangent.k_state_jacobian.num_rows, dofs);
        assert_eq!(tangent.omega_state_jacobian.num_rows, dofs);
        assert!(
            tangent
                .k_history_jacobian
                .values
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            tangent
                .omega_previous_history_jacobian
                .values
                .iter()
                .any(|value| value.abs() > 0.0)
        );
    }

    #[test]
    fn rejects_non_positive_history_before_assembly() {
        let mesh = crate::mesh::unit_square_triangles(1);
        let degree = PolynomialDegree::P1;
        let dofs = H1Space::new(degree).num_global_dofs(&mesh);
        let mut state = KOmegaTransportState::from_constant(dofs, 2.0, 3.0, 0.0, 0).unwrap();
        state.k[0] = 0.0;

        let error =
            advance_komega_bdf_step(&mesh, degree, &manufactured_config(1, 2.0, 3.0), &state)
                .unwrap_err();
        assert!(matches!(
            error,
            KOmegaTransportError::NonPositiveState { field: "k", .. }
        ));
    }

    #[test]
    fn element_fields_produce_a_spatially_resolved_update() {
        let mesh = crate::mesh::unit_square_triangles(2);
        let degree = PolynomialDegree::P1;
        let dofs = H1Space::new(degree).num_global_dofs(&mesh);
        let state = KOmegaTransportState::from_constant(dofs, 2.0, 3.0, 0.0, 0).unwrap();
        let config = manufactured_config(1, 2.0, 3.0);
        let mut fields = KOmegaTransportElementFields::uniform(&mesh, &config);
        for (element, production) in fields.k_production.iter_mut().enumerate() {
            *production = if element % 2 == 0 { 0.05 } else { 0.9 };
        }
        for (element, production) in fields.omega_production.iter_mut().enumerate() {
            *production = if element % 2 == 0 { 0.02 } else { 0.4 };
        }

        let result =
            advance_komega_bdf_step_with_element_fields(&mesh, degree, &config, &state, &fields)
                .unwrap();

        assert!(result.state.k.iter().all(|value| value.is_finite()));
        assert!(result.state.omega.iter().all(|value| value.is_finite()));
        assert!(
            result.state.k.iter().copied().fold(f64::MIN, f64::max)
                > result.state.k.iter().copied().fold(f64::MAX, f64::min)
        );
        assert!(result.report.k_residual_l2 < 1.0e-9);
    }

    #[test]
    fn element_fields_reject_wrong_topology_cardinality() {
        let mesh = crate::mesh::unit_square_triangles(1);
        let config = KOmegaTransportConfig::default();
        let state =
            KOmegaTransportState::from_constant(mesh.num_nodes(), 1.0, 1.0, 0.0, 0).unwrap();
        let mut fields = KOmegaTransportElementFields::uniform(&mesh, &config);
        fields.velocity.pop();

        let error = advance_komega_bdf_step_with_element_fields(
            &mesh,
            PolynomialDegree::P1,
            &config,
            &state,
            &fields,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            KOmegaTransportError::ElementFieldCount {
                field: "velocity",
                ..
            }
        ));
    }

    #[test]
    fn p1_state_publishes_element_eddy_viscosity_for_momentum_coupling() {
        let mesh = crate::mesh::unit_square_triangles(2);
        let state =
            KOmegaTransportState::from_constant(mesh.num_nodes(), 2.0, 4.0, 0.0, 0).unwrap();

        let viscosity = state.p1_eddy_viscosity_per_element(&mesh, 10.0).unwrap();

        assert_eq!(viscosity.len(), mesh.num_elements());
        assert!(viscosity.iter().all(|value| (*value - 0.2).abs() < 1.0e-12));
    }
}
