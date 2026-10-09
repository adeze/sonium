//! PC-fan adapter for the FEM transported k--omega state contract.
//!
//! This is deliberately a narrow integration seam.  It consumes the checked
//! P1 history state used by topology-transfer code, advances the two scalar
//! equations on the target mesh, and returns the same history shape.  The
//! the legacy scalar API remains a segregated Picard update, while the
//! rotating-wall transient runner uses the checked nonlinear residual and
//! current-state blocks below for its monolithic Newton step.

use crate::state_transfer::PcFanCfdKOmegaBdfState;
use sonium_fem::assembly::{
    PrebakedAssembler, assemble_bilinear, assemble_bilinear_per_element,
    assemble_linear_per_element,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::MassForm;
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::fluid::turbulence::{
    KOmegaTransportConfig, KOmegaTransportElementFields, KOmegaTransportError,
    KOmegaTransportLinearization, KOmegaTransportState, KOmegaTransportStepReport,
    TurbulenceProductionForm, TurbulenceTransportForm, advance_komega_bdf_step_with_element_fields,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_solvers::CsrMatrix;
use sonium_solvers::traits::LinearOperator;

/// Provenance for the current transported-turbulence block coupling. The
/// blocks cover flow/scalar cross terms and scalar-state corrections for the
/// element-average eddy-viscosity, mean-advection, and strain-production
/// dependencies used by the Newton/BDF runner.
pub const PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE: &str =
    "pcfan:turbulence:segregated-coupled-bdf-tangent:v1";

/// Provenance for the square block matrix assembled from the flow, scalar,
/// and off-diagonal Newton/BDF tangents.
pub const PC_FAN_KOMEGA_MONOLITHIC_BLOCK_PROVENANCE: &str =
    "pcfan:turbulence:segregated-monolithic-block:v1";

/// Failure while advancing the PC-fan transported turbulence state.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanKOmegaTransportError {
    Fem(KOmegaTransportError),
    InvalidCoupling(&'static str),
}

impl std::fmt::Display for PcFanKOmegaTransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fem(error) => {
                write!(
                    formatter,
                    "transported PC-fan state advance failed: {error}"
                )
            }
            Self::InvalidCoupling(reason) => {
                write!(formatter, "invalid transported PC-fan coupling: {reason}")
            }
        }
    }
}

impl std::error::Error for PcFanKOmegaTransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fem(error) => Some(error),
            Self::InvalidCoupling(_) => None,
        }
    }
}

impl From<KOmegaTransportError> for PcFanKOmegaTransportError {
    fn from(error: KOmegaTransportError) -> Self {
        Self::Fem(error)
    }
}
pub use sonium_fem::physics::fluid::turbulence::{
    KOmegaCoupledLinearization as PcFanKOmegaCoupledLinearization,
    KOmegaMonolithicLinearization as PcFanKOmegaMonolithicLinearization,
};
use sonium_fem::physics::fluid::turbulence::{
    KOmegaCouplingError, assemble_komega_coupled_linearization,
    assemble_komega_monolithic_history_linearization, assemble_komega_monolithic_linearization,
    assemble_komega_transient_current_linearization,
    assemble_komega_transient_history_linearization,
    tetrahedron_shape_gradients as fem_tetrahedron_shape_gradients,
};

/// Map the reusable coupling errors onto the identical product variants.
fn map_coupling_error(error: KOmegaCouplingError) -> PcFanKOmegaTransportError {
    match error {
        KOmegaCouplingError::Fem(error) => PcFanKOmegaTransportError::Fem(error),
        KOmegaCouplingError::InvalidCoupling(reason) => {
            PcFanKOmegaTransportError::InvalidCoupling(reason)
        }
    }
}

/// Result of one PC-fan P1 k--omega update.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanKOmegaTransportStep {
    /// Updated history in the state-transfer-compatible representation.
    pub state: PcFanCfdKOmegaBdfState,
    /// FEM transport admission and residual diagnostics.
    pub report: KOmegaTransportStepReport,
    /// Accepted k residual vector, in P1 scalar DOF order.
    pub k_residual: Vec<f64>,
    /// Accepted omega residual vector, in P1 scalar DOF order.
    pub omega_residual: Vec<f64>,
    /// Exact tangent of the frozen-coefficient scalar residual. This remains
    /// a segregated scalar tangent; it is not a coupled momentum tangent.
    pub linearization: KOmegaTransportLinearization,
}

/// Assemble a square block matrix from the retained segregated tangents.
///
/// The flow matrix is ordered `[velocity, pressure]`; the two scalar
/// matrices are ordered by their P1 global DOFs. The coupling blocks are
/// checked against those dimensions before being inserted, preventing a
/// pressure/scalar offset from becoming a silent out-of-bounds assembly.
/// Thin wrapper stamping the PC-fan block provenance over the reusable
/// [`sonium_fem::physics::fluid::turbulence`] composition.
pub fn assemble_pc_fan_cfd_komega_monolithic_linearization(
    flow_jacobian: &CsrMatrix<f64>,
    scalar_linearization: &KOmegaTransportLinearization,
    coupled_linearization: &PcFanKOmegaCoupledLinearization,
) -> Result<PcFanKOmegaMonolithicLinearization, PcFanKOmegaTransportError> {
    assemble_komega_monolithic_linearization(
        flow_jacobian,
        scalar_linearization,
        coupled_linearization,
        PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE,
        PC_FAN_KOMEGA_MONOLITHIC_BLOCK_PROVENANCE,
    )
    .map_err(map_coupling_error)
}

/// Assemble the same-time block of the segregated transient residual.
///
/// The production runner intentionally omits current scalar-state corrections
/// from this retained segregated map. The scalar equations still depend on
/// the current flow through the retained velocity cross blocks. Thin wrapper
/// over the reusable composition.
pub fn assemble_pc_fan_cfd_komega_transient_current_linearization(
    flow_jacobian: &CsrMatrix<f64>,
    scalar_linearization: &KOmegaTransportLinearization,
    coupled_linearization: &PcFanKOmegaCoupledLinearization,
) -> Result<CsrMatrix<f64>, PcFanKOmegaTransportError> {
    assemble_komega_transient_current_linearization(
        flow_jacobian,
        scalar_linearization,
        coupled_linearization,
        PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE,
        PC_FAN_KOMEGA_MONOLITHIC_BLOCK_PROVENANCE,
    )
    .map_err(map_coupling_error)
}

/// Assemble the transported portion of one BDF history block.
///
/// `history_order == 1` is the immediately preceding state and receives the
/// lagged momentum derivatives. `history_order == 2` receives only the older
/// scalar BDF history contribution; the segregated momentum map does not
/// read that older transported state. Thin wrapper over the reusable
/// composition.
pub fn assemble_pc_fan_cfd_komega_transient_history_linearization(
    flow_dimension: usize,
    scalar_linearization: &KOmegaTransportLinearization,
    coupled_linearization: &PcFanKOmegaCoupledLinearization,
    history_order: usize,
) -> Result<CsrMatrix<f64>, PcFanKOmegaTransportError> {
    assemble_komega_transient_history_linearization(
        flow_dimension,
        scalar_linearization,
        coupled_linearization,
        history_order,
        PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE,
    )
    .map_err(map_coupling_error)
}

/// Assemble a transported history block for the genuine monolithic Newton
/// residual. Unlike the segregated map, momentum has no dependence on an old
/// scalar iterate: its scalar dependence is in the current-state block. Thin
/// wrapper over the reusable composition.
pub fn assemble_pc_fan_cfd_komega_monolithic_history_linearization(
    flow_dimension: usize,
    scalar_linearization: &KOmegaTransportLinearization,
    history_order: usize,
) -> Result<CsrMatrix<f64>, PcFanKOmegaTransportError> {
    assemble_komega_monolithic_history_linearization(
        flow_dimension,
        scalar_linearization,
        history_order,
    )
    .map_err(map_coupling_error)
}

/// Assemble the momentum/scalar coupled linearization for one BDF state.
///
/// Thin wrapper stamping the PC-fan coupling provenance over the reusable
/// [`sonium_fem::physics::fluid::turbulence`] assembly.
#[allow(clippy::too_many_arguments)]
pub fn assemble_pc_fan_cfd_komega_coupled_linearization(
    mesh: &Mesh,
    velocity_space: &VectorH1Space,
    scalar_space: &H1Space,
    velocity: &[f64],
    turbulence: &PcFanCfdKOmegaBdfState,
    fields: &KOmegaTransportElementFields,
    density: f64,
    limiter: f64,
) -> Result<PcFanKOmegaCoupledLinearization, PcFanKOmegaTransportError> {
    assemble_komega_coupled_linearization(
        mesh,
        velocity_space,
        scalar_space,
        velocity,
        turbulence,
        fields,
        density,
        limiter,
        PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE,
    )
    .map_err(map_coupling_error)
}

/// P1 shape gradients on one tetrahedral element, shared with the reusable
/// turbulence coupling assembly.
fn tetrahedron_shape_gradients(
    mesh: &Mesh,
    vertices: &[usize],
) -> Result<[[f64; 3]; 4], PcFanKOmegaTransportError> {
    fem_tetrahedron_shape_gradients(mesh, vertices).map_err(map_coupling_error)
}

pub fn assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
    mesh: &Mesh,
    config: &KOmegaTransportConfig,
    history: &PcFanCfdKOmegaBdfState,
    current: &PcFanCfdKOmegaBdfState,
    fields: &KOmegaTransportElementFields,
) -> Result<(Vec<f64>, Vec<f64>, KOmegaTransportLinearization), PcFanKOmegaTransportError> {
    let scalar_space = H1Space::new(PolynomialDegree::P1);
    let dimension = scalar_space.num_global_dofs(mesh);
    if dimension == 0
        || history.k.len() != dimension
        || history.k_prev.len() != dimension
        || history.omega.len() != dimension
        || history.omega_prev.len() != dimension
        || current.k.len() != dimension
        || current.k_prev.len() != dimension
        || current.omega.len() != dimension
        || current.omega_prev.len() != dimension
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "nonlinear scalar state dimension mismatch",
        ));
    }
    if !config.dt.is_finite()
        || config.dt <= 0.0
        || !matches!(config.bdf_order, 1 | 2)
        || !config.molecular_viscosity.is_finite()
        || config.molecular_viscosity < 0.0
        || !config.positivity_floor.is_finite()
        || config.positivity_floor <= 0.0
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "invalid nonlinear scalar transport configuration",
        ));
    }
    if history
        .k
        .iter()
        .chain(&history.k_prev)
        .chain(&history.omega)
        .chain(&history.omega_prev)
        .chain(&current.k)
        .chain(&current.k_prev)
        .chain(&current.omega)
        .chain(&current.omega_prev)
        .any(|value| !value.is_finite() || *value < config.positivity_floor)
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "nonlinear scalar state must be finite and positive",
        ));
    }
    let expected_elements = mesh.num_elements();
    if fields.velocity.len() != expected_elements
        || fields.eddy_viscosity.len() != expected_elements
        || fields.k_production.len() != expected_elements
        || fields.omega_production.len() != expected_elements
        || fields
            .velocity
            .iter()
            .flat_map(|velocity| velocity.iter())
            .chain(&fields.eddy_viscosity)
            .chain(&fields.k_production)
            .chain(&fields.omega_production)
            .any(|value| !value.is_finite())
        || fields.eddy_viscosity.iter().any(|value| *value < 0.0)
        || fields
            .k_production
            .iter()
            .chain(&fields.omega_production)
            .any(|value| *value < 0.0)
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "invalid nonlinear scalar element fields",
        ));
    }

    let coefficients =
        sonium_fem::physics::fluid::stokes::time_stepping::BdfCoefficients::for_order(
            config.bdf_order,
        );
    let k_reference = current.k.iter().sum::<f64>() / dimension as f64;
    let omega_reference = current.omega.iter().sum::<f64>() / dimension as f64;
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
    let stiffness_quad = sonium_fem::quadrature::required_order_for_stiffness(1);
    let mass_quad = sonium_fem::quadrature::required_order_for_mass(1);
    let k_transport =
        assemble_bilinear_per_element(&k_forms, &scalar_space, &scalar_space, mesh, stiffness_quad);
    let omega_transport = assemble_bilinear_per_element(
        &omega_forms,
        &scalar_space,
        &scalar_space,
        mesh,
        stiffness_quad,
    );
    let k_mass =
        assemble_bilinear(&MassForm, &scalar_space, &scalar_space, mesh, mass_quad).to_csr();
    let omega_mass =
        assemble_bilinear(&MassForm, &scalar_space, &scalar_space, mesh, mass_quad).to_csr();
    let k_operator = PrebakedAssembler::from_triplets(
        [
            k_transport,
            assemble_bilinear(&MassForm, &scalar_space, &scalar_space, mesh, mass_quad),
        ],
        &[],
    )
    .assemble_real(&[1.0, coefficients.a0 / config.dt]);
    let omega_operator = PrebakedAssembler::from_triplets(
        [
            omega_transport,
            assemble_bilinear(&MassForm, &scalar_space, &scalar_space, mesh, mass_quad),
        ],
        &[],
    )
    .assemble_real(&[1.0, coefficients.a0 / config.dt]);
    let k_history = bdf_history_vector(&history.k, &history.k_prev, &coefficients, config.dt);
    let omega_history = bdf_history_vector(
        &history.omega,
        &history.omega_prev,
        &coefficients,
        config.dt,
    );
    let k_sources = fields
        .k_production
        .iter()
        .map(|&production| TurbulenceProductionForm {
            production: config.constants.production_limiter(
                production,
                k_reference,
                omega_reference,
            ),
        })
        .collect::<Vec<_>>();
    let omega_sources = fields
        .omega_production
        .iter()
        .map(|&production| TurbulenceProductionForm { production })
        .collect::<Vec<_>>();
    let k_source = assemble_linear_per_element(&k_sources, &scalar_space, mesh, mass_quad);
    let omega_source = assemble_linear_per_element(&omega_sources, &scalar_space, mesh, mass_quad);
    let k_rhs = add_vectors(&k_mass.apply(&k_history), &k_source);
    let omega_rhs = add_vectors(&omega_mass.apply(&omega_history), &omega_source);
    let k_residual = k_operator
        .apply(&current.k)
        .iter()
        .zip(&k_rhs)
        .map(|(&left, &right)| left - right)
        .collect::<Vec<_>>();
    let omega_residual = omega_operator
        .apply(&current.omega)
        .iter()
        .zip(&omega_rhs)
        .map(|(&left, &right)| left - right)
        .collect::<Vec<_>>();
    if k_residual
        .iter()
        .chain(&omega_residual)
        .any(|value| !value.is_finite())
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "nonlinear scalar residual is non-finite",
        ));
    }
    Ok((
        k_residual,
        omega_residual,
        KOmegaTransportLinearization {
            provenance:
                sonium_fem::physics::fluid::turbulence::K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE,
            k_state_jacobian: k_operator,
            omega_state_jacobian: omega_operator,
            k_history_jacobian: scale_csr(&k_mass, coefficients.a1 / config.dt),
            k_previous_history_jacobian: scale_csr(&k_mass, coefficients.a2 / config.dt),
            omega_history_jacobian: scale_csr(&omega_mass, coefficients.a1 / config.dt),
            omega_previous_history_jacobian: scale_csr(&omega_mass, coefficients.a2 / config.dt),
        },
    ))
}

/// Assemble the exact fixed-state design derivative of the transported P1
/// k--omega BDF residual caused by affine tetrahedral ALE motion.
///
/// The derivative holds the nodal flow and turbulence state fixed, matching a
/// discrete-adjoint residual partial. It includes the determinant and inverse
/// Jacobian terms in mass, diffusion, and advection, plus the strain-derived
/// production source and its limiter branch. State-dependent reaction and
/// eddy-viscosity coefficients are therefore frozen here; their state
/// derivatives belong to [`assemble_pc_fan_cfd_komega_coupled_linearization`].
#[allow(clippy::too_many_arguments)]
pub fn assemble_pc_fan_cfd_komega_scalar_mesh_motion_design_residual<const DESIGN_DIM: usize>(
    mesh: &Mesh,
    config: &KOmegaTransportConfig,
    history: &PcFanCfdKOmegaBdfState,
    current: &PcFanCfdKOmegaBdfState,
    velocity: &[f64],
    fields: &KOmegaTransportElementFields,
    nodal_coordinate_jacobian: &[[[f64; 3]; DESIGN_DIM]],
) -> Result<Vec<Vec<f64>>, PcFanKOmegaTransportError> {
    let scalar_space = H1Space::new(PolynomialDegree::P1);
    let dimension = scalar_space.num_global_dofs(mesh);
    if mesh.dimension != 3 || dimension == 0 {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "scalar ALE derivative requires a non-empty 3-D mesh",
        ));
    }
    if history.k.len() != dimension
        || history.k_prev.len() != dimension
        || history.omega.len() != dimension
        || history.omega_prev.len() != dimension
        || current.k.len() != dimension
        || current.k_prev.len() != dimension
        || current.omega.len() != dimension
        || current.omega_prev.len() != dimension
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "scalar ALE derivative state dimensions",
        ));
    }
    if velocity.len() < mesh.num_nodes().saturating_mul(3)
        || fields.velocity.len() != mesh.num_elements()
        || fields.eddy_viscosity.len() != mesh.num_elements()
        || fields.k_production.len() != mesh.num_elements()
        || fields.omega_production.len() != mesh.num_elements()
        || nodal_coordinate_jacobian.len() != mesh.num_nodes()
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "scalar ALE derivative field dimensions",
        ));
    }
    if !config.dt.is_finite()
        || config.dt <= 0.0
        || !matches!(config.bdf_order, 1 | 2)
        || !config.molecular_viscosity.is_finite()
        || config.molecular_viscosity < 0.0
        || !config.positivity_floor.is_finite()
        || config.positivity_floor <= 0.0
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "scalar ALE derivative configuration",
        ));
    }
    if history
        .k
        .iter()
        .chain(&history.k_prev)
        .chain(&history.omega)
        .chain(&history.omega_prev)
        .chain(&current.k)
        .chain(&current.k_prev)
        .chain(&current.omega)
        .chain(&current.omega_prev)
        .chain(velocity)
        .any(|value| !value.is_finite())
        || fields
            .velocity
            .iter()
            .flatten()
            .chain(&fields.eddy_viscosity)
            .chain(&fields.k_production)
            .chain(&fields.omega_production)
            .any(|value| !value.is_finite())
        || nodal_coordinate_jacobian
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "non-finite scalar ALE derivative input",
        ));
    }
    if history
        .k
        .iter()
        .chain(&history.k_prev)
        .chain(&history.omega)
        .chain(&history.omega_prev)
        .chain(&current.k)
        .chain(&current.k_prev)
        .chain(&current.omega)
        .chain(&current.omega_prev)
        .any(|value| *value < config.positivity_floor)
        || fields.eddy_viscosity.iter().any(|value| *value < 0.0)
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "non-positive scalar ALE derivative input",
        ));
    }

    let coefficients =
        sonium_fem::physics::fluid::stokes::time_stepping::BdfCoefficients::for_order(
            config.bdf_order,
        );
    let state_count = dimension as f64;
    let k_reference = current.k.iter().sum::<f64>() / state_count;
    let omega_reference = current.omega.iter().sum::<f64>() / state_count;
    let k_reaction = config.constants.k_reaction(omega_reference);
    let omega_reaction = config.constants.omega_reaction(omega_reference);
    let k_production_limit =
        10.0 * config.constants.beta_star * k_reference.max(0.0) * omega_reference.max(0.0);
    let mut derivative = vec![vec![0.0; DESIGN_DIM]; dimension.saturating_mul(2)];

    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        if vertices.len() != 4 {
            return Err(PcFanKOmegaTransportError::InvalidCoupling(
                "scalar ALE derivative requires affine tetrahedral elements",
            ));
        }
        let scalar_dofs = scalar_space.local_to_global(mesh, element_index);
        if scalar_dofs.len() != 4 {
            return Err(PcFanKOmegaTransportError::InvalidCoupling(
                "scalar ALE derivative requires P1 scalar elements",
            ));
        }
        let (_jacobian, determinant, determinant_abs, inverse, gradients) =
            affine_tetrahedron_geometry(mesh, vertices)?;
        let mut local_k = vec![0.0; DESIGN_DIM.saturating_mul(4)];
        let mut local_omega = vec![0.0; DESIGN_DIM.saturating_mul(4)];

        let mut velocity_gradient = [[0.0; 3]; 3];
        for (local_node, &node) in vertices.iter().enumerate() {
            for component in 0..3 {
                let value = velocity[node * 3 + component];
                for direction in 0..3 {
                    velocity_gradient[component][direction] +=
                        value * gradients[local_node][direction];
                }
            }
        }
        let mut strain = [[0.0; 3]; 3];
        for row in 0..3 {
            for column in 0..3 {
                strain[row][column] =
                    0.5 * (velocity_gradient[row][column] + velocity_gradient[column][row]);
            }
        }
        let element_k = scalar_dofs.iter().map(|&dof| current.k[dof]).sum::<f64>() / 4.0;
        let k_history = scalar_dofs
            .iter()
            .map(|&dof| {
                (-coefficients.a1 * history.k[dof] - coefficients.a2 * history.k_prev[dof])
                    / config.dt
            })
            .collect::<Vec<_>>();
        let omega_history = scalar_dofs
            .iter()
            .map(|&dof| {
                (-coefficients.a1 * history.omega[dof] - coefficients.a2 * history.omega_prev[dof])
                    / config.dt
            })
            .collect::<Vec<_>>();
        let k_source = config.constants.production_limiter(
            fields.k_production[element_index],
            k_reference,
            omega_reference,
        );
        let omega_source = fields.omega_production[element_index];
        let k_source_is_limited = fields.k_production[element_index] >= k_production_limit;
        let diffusivity_k = config.molecular_viscosity
            + fields.eddy_viscosity[element_index] / config.constants.sigma_k;
        let diffusivity_omega = config.molecular_viscosity
            + fields.eddy_viscosity[element_index] / config.constants.sigma_omega;
        let reaction_k = k_reaction;
        let reaction_omega = omega_reaction;

        for design_index in 0..DESIGN_DIM {
            let mut d_jacobian = [[0.0; 3]; 3];
            for row in 0..3 {
                for column in 0..3 {
                    d_jacobian[row][column] = nodal_coordinate_jacobian[vertices[column + 1]]
                        [design_index][row]
                        - nodal_coordinate_jacobian[vertices[0]][design_index][row];
                }
            }
            let d_determinant = determinant
                * (0..3)
                    .map(|index| {
                        (0..3)
                            .map(|inner| inverse[index][inner] * d_jacobian[inner][index])
                            .sum::<f64>()
                    })
                    .sum::<f64>();
            let d_determinant_abs = determinant.signum() * d_determinant;
            let d_inverse_transpose = negative_matmul3(
                &matmul3(&transpose3(&inverse), &transpose3(&d_jacobian)),
                &transpose3(&inverse),
            );
            let mut d_gradients = [[0.0; 3]; 4];
            let reference_gradients = [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ];
            for local_node in 0..4 {
                d_gradients[local_node] =
                    matvec3(&d_inverse_transpose, reference_gradients[local_node]);
            }
            let mut d_velocity_gradient = [[0.0; 3]; 3];
            for (local_node, &node) in vertices.iter().enumerate() {
                for component in 0..3 {
                    let value = velocity[node * 3 + component];
                    for direction in 0..3 {
                        d_velocity_gradient[component][direction] +=
                            value * d_gradients[local_node][direction];
                    }
                }
            }
            let mut d_strain_square = 0.0;
            for row in 0..3 {
                for column in 0..3 {
                    let d_strain =
                        0.5 * (d_velocity_gradient[row][column] + d_velocity_gradient[column][row]);
                    d_strain_square += 2.0 * strain[row][column] * d_strain;
                }
            }
            let d_k_production = 2.0 * fields.eddy_viscosity[element_index] * d_strain_square;
            let d_omega_production = d_k_production / element_k.max(config.positivity_floor);
            let d_k_source = if k_source_is_limited {
                0.0
            } else {
                d_k_production
            };
            let transport_quadrature = QuadratureRule::new(
                element.element_type,
                sonium_fem::quadrature::required_order_for_stiffness(1),
            );
            for point in transport_quadrature.iter() {
                let values = [
                    1.0 - point.xi() - point.eta() - point.zeta(),
                    point.xi(),
                    point.eta(),
                    point.zeta(),
                ];
                let weight = point.weight;
                for test in 0..4 {
                    let mut k_base = 0.0;
                    let mut omega_base = 0.0;
                    let mut k_gradient_derivative = 0.0;
                    let mut omega_gradient_derivative = 0.0;
                    for trial in 0..4 {
                        let k_value = current.k[scalar_dofs[trial]];
                        let omega_value = current.omega[scalar_dofs[trial]];
                        let grad_dot = dot3(gradients[test], gradients[trial]);
                        let d_grad_dot = dot3(d_gradients[test], gradients[trial])
                            + dot3(gradients[test], d_gradients[trial]);
                        let advection = dot3(fields.velocity[element_index], gradients[trial]);
                        let d_advection = dot3(fields.velocity[element_index], d_gradients[trial]);
                        let mass = values[test] * values[trial];
                        k_base += (diffusivity_k * grad_dot
                            + values[test] * advection
                            + reaction_k * mass)
                            * k_value;
                        omega_base += (diffusivity_omega * grad_dot
                            + values[test] * advection
                            + reaction_omega * mass)
                            * omega_value;
                        k_gradient_derivative +=
                            (diffusivity_k * d_grad_dot + values[test] * d_advection) * k_value;
                        omega_gradient_derivative += (diffusivity_omega * d_grad_dot
                            + values[test] * d_advection)
                            * omega_value;
                    }
                    local_k[design_index * 4 + test] += weight
                        * (d_determinant_abs * k_base + determinant_abs * k_gradient_derivative);
                    local_omega[design_index * 4 + test] += weight
                        * (d_determinant_abs * omega_base
                            + determinant_abs * omega_gradient_derivative);
                }
            }

            let mass_quadrature = QuadratureRule::new(
                element.element_type,
                sonium_fem::quadrature::required_order_for_mass(1),
            );
            for point in mass_quadrature.iter() {
                let values = [
                    1.0 - point.xi() - point.eta() - point.zeta(),
                    point.xi(),
                    point.eta(),
                    point.zeta(),
                ];
                let weight = point.weight;
                for test in 0..4 {
                    let k_source_base = -values[test] * k_source;
                    let omega_source_base = -values[test] * omega_source;
                    let k_source_derivative = -values[test] * d_k_source;
                    let omega_source_derivative = -values[test] * d_omega_production;
                    let mut k_base = k_source_base;
                    let mut omega_base = omega_source_base;
                    for trial in 0..4 {
                        let mass = values[test] * values[trial];
                        k_base += mass
                            * (coefficients.a0 / config.dt * current.k[scalar_dofs[trial]]
                                - k_history[trial]);
                        omega_base += mass
                            * (coefficients.a0 / config.dt * current.omega[scalar_dofs[trial]]
                                - omega_history[trial]);
                    }
                    local_k[design_index * 4 + test] += weight
                        * (d_determinant_abs * k_base + determinant_abs * k_source_derivative);
                    local_omega[design_index * 4 + test] += weight
                        * (d_determinant_abs * omega_base
                            + determinant_abs * omega_source_derivative);
                }
            }
        }
        for design_index in 0..DESIGN_DIM {
            for local_node in 0..4 {
                let row = scalar_dofs[local_node];
                derivative[row][design_index] += local_k[design_index * 4 + local_node];
                derivative[dimension + row][design_index] +=
                    local_omega[design_index * 4 + local_node];
            }
        }
    }
    if derivative
        .iter()
        .flatten()
        .any(|value: &f64| !value.is_finite())
    {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "non-finite scalar ALE derivative",
        ));
    }
    Ok(derivative)
}

fn bdf_history_vector(
    current: &[f64],
    previous: &[f64],
    coefficients: &sonium_fem::physics::fluid::stokes::time_stepping::BdfCoefficients,
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

fn scale_csr(matrix: &CsrMatrix<f64>, factor: f64) -> CsrMatrix<f64> {
    CsrMatrix {
        num_rows: matrix.num_rows,
        num_cols: matrix.num_cols,
        values: matrix.values.iter().map(|value| factor * value).collect(),
        col_indices: matrix.col_indices.clone(),
        row_ptrs: matrix.row_ptrs.clone(),
    }
}

/// Apply the lagged k--omega eddy viscosity to one FEM momentum config.
///
/// The current Navier--Stokes assembly accepts one scalar dynamic-viscosity
/// coefficient per segregated step. This checked bridge averages the target
/// P1 element viscosities and marks the value as transported provenance in the
/// FEM config; it does not claim an exact scalar/momentum Newton block.
pub fn apply_pc_fan_cfd_komega_mean_eddy_viscosity(
    momentum_config: &mut sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesConfig,
    mesh: &Mesh,
    state: &PcFanCfdKOmegaBdfState,
    limiter: f64,
) -> Result<f64, PcFanKOmegaTransportError> {
    let fem_state = KOmegaTransportState {
        k: state.k.clone(),
        k_prev: state.k_prev.clone(),
        omega: state.omega.clone(),
        omega_prev: state.omega_prev.clone(),
        time: state.time,
        step: state.step,
    };
    let element_viscosities = fem_state.p1_eddy_viscosity_per_element(mesh, limiter)?;
    let mean = element_viscosities.iter().sum::<f64>() / element_viscosities.len().max(1) as f64;
    if !mean.is_finite() || mean < 0.0 {
        return Err(PcFanKOmegaTransportError::Fem(
            KOmegaTransportError::InvalidConfig("mean eddy viscosity is not finite"),
        ));
    }
    momentum_config.transported_eddy_viscosity = Some(mean);
    momentum_config.transported_eddy_viscosity_per_element = Some(element_viscosities);
    Ok(mean)
}

/// Initialise a positive P1 k--omega history for a resolved PC-fan mesh.
pub fn initialize_pc_fan_cfd_komega_bdf_state(
    mesh: &Mesh,
    time: f64,
    step: usize,
    k: f64,
    omega: f64,
) -> Result<PcFanCfdKOmegaBdfState, PcFanKOmegaTransportError> {
    let num_dofs = H1Space::new(PolynomialDegree::P1).num_global_dofs(mesh);
    let state = KOmegaTransportState::from_constant(num_dofs, k, omega, time, step)?;
    Ok(PcFanCfdKOmegaBdfState {
        k: state.k,
        k_prev: state.k_prev,
        omega: state.omega,
        omega_prev: state.omega_prev,
        time: state.time,
        step: state.step,
    })
}

/// Derive frozen element-local transport coefficients from an accepted P2
/// mean-flow velocity and the current P1 turbulent state.
///
/// The velocity and strain rate are evaluated from tetrahedron vertex values.
/// This gives the scalar transport solve a spatially varying advector and
/// production source while keeping the momentum/scalar coupling segregated.
pub fn derive_pc_fan_cfd_komega_element_fields(
    mesh: &Mesh,
    velocity: &[f64],
    state: &PcFanCfdKOmegaBdfState,
    limiter: f64,
) -> Result<KOmegaTransportElementFields, PcFanKOmegaTransportError> {
    let required_velocity_dofs = mesh.num_nodes().saturating_mul(3);
    if velocity.len() < required_velocity_dofs {
        return Err(PcFanKOmegaTransportError::Fem(
            KOmegaTransportError::DimensionMismatch {
                field: "velocity",
                expected: required_velocity_dofs,
                actual: velocity.len(),
            },
        ));
    }

    let fem_state = KOmegaTransportState {
        k: state.k.clone(),
        k_prev: state.k_prev.clone(),
        omega: state.omega.clone(),
        omega_prev: state.omega_prev.clone(),
        time: state.time,
        step: state.step,
    };
    let eddy_viscosity = fem_state.p1_eddy_viscosity_per_element(mesh, limiter)?;
    let mut fields = KOmegaTransportElementFields {
        velocity: Vec::with_capacity(mesh.num_elements()),
        eddy_viscosity,
        k_production: Vec::with_capacity(mesh.num_elements()),
        omega_production: Vec::with_capacity(mesh.num_elements()),
    };

    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        if vertices.len() != 4 {
            return Err(PcFanKOmegaTransportError::Fem(
                KOmegaTransportError::InvalidConfig(
                    "resolved transported PC-fan coupling requires tetrahedral elements",
                ),
            ));
        }

        let mut element_velocity = [0.0; 3];
        let mut mean_k = 0.0;
        for &node in vertices {
            let velocity_offset = node.saturating_mul(3);
            for component in 0..3 {
                element_velocity[component] += velocity[velocity_offset + component] / 4.0;
            }
            mean_k += fem_state.k[node] / 4.0;
        }

        let gradients = tetrahedron_shape_gradients(mesh, vertices)?;
        let mut velocity_gradient = [[0.0; 3]; 3];
        for (local_node, &node) in vertices.iter().enumerate() {
            let velocity_offset = node * 3;
            for component in 0..3 {
                for direction in 0..3 {
                    velocity_gradient[component][direction] +=
                        velocity[velocity_offset + component] * gradients[local_node][direction];
                }
            }
        }

        let mut strain_square = 0.0;
        for (row, velocity_gradient_row) in velocity_gradient.iter().enumerate() {
            for (column, &velocity_gradient_value) in velocity_gradient_row.iter().enumerate() {
                let strain = 0.5 * (velocity_gradient_value + velocity_gradient[column][row]);
                strain_square += strain * strain;
            }
        }
        let production = 2.0 * fields.eddy_viscosity[element_index] * strain_square;
        let omega_production = production / mean_k.max(1.0e-12);
        if !element_velocity.iter().all(|value| value.is_finite())
            || !production.is_finite()
            || !omega_production.is_finite()
        {
            return Err(PcFanKOmegaTransportError::Fem(
                KOmegaTransportError::InvalidConfig(
                    "derived transported PC-fan element field is non-finite",
                ),
            ));
        }
        fields.velocity.push(element_velocity);
        fields.k_production.push(production);
        fields.omega_production.push(omega_production);
    }

    Ok(fields)
}

type AffineTetrahedronGeometry = ([[f64; 3]; 3], f64, f64, [[f64; 3]; 3], [[f64; 3]; 4]);

fn affine_tetrahedron_geometry(
    mesh: &Mesh,
    vertices: &[usize],
) -> Result<AffineTetrahedronGeometry, PcFanKOmegaTransportError> {
    if vertices.len() != 4 {
        return Err(PcFanKOmegaTransportError::InvalidCoupling(
            "affine tetrahedral geometry requires four vertices",
        ));
    }
    let points = vertices
        .iter()
        .map(|&node| mesh.nodes.get(node).copied())
        .collect::<Option<Vec<_>>>()
        .ok_or(PcFanKOmegaTransportError::Fem(
            KOmegaTransportError::InvalidConfig("element vertex index out of range"),
        ))?;
    let a = vector_between(points[1], points[0]);
    let b = vector_between(points[2], points[0]);
    let c = vector_between(points[3], points[0]);
    let jacobian = [[a[0], b[0], c[0]], [a[1], b[1], c[1]], [a[2], b[2], c[2]]];
    let determinant = dot(a, cross(b, c));
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return Err(PcFanKOmegaTransportError::Fem(
            KOmegaTransportError::InvalidConfig("degenerate tetrahedral element"),
        ));
    }
    let inverse = [
        [
            (b[1] * c[2] - b[2] * c[1]) / determinant,
            (b[2] * c[0] - b[0] * c[2]) / determinant,
            (b[0] * c[1] - b[1] * c[0]) / determinant,
        ],
        [
            (c[1] * a[2] - c[2] * a[1]) / determinant,
            (c[2] * a[0] - c[0] * a[2]) / determinant,
            (c[0] * a[1] - c[1] * a[0]) / determinant,
        ],
        [
            (a[1] * b[2] - a[2] * b[1]) / determinant,
            (a[2] * b[0] - a[0] * b[2]) / determinant,
            (a[0] * b[1] - a[1] * b[0]) / determinant,
        ],
    ];
    let reference_gradients = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let inverse_transpose = transpose3(&inverse);
    let mut gradients = [[0.0; 3]; 4];
    for (gradient, reference) in gradients.iter_mut().zip(reference_gradients) {
        *gradient = matvec3(&inverse_transpose, reference);
    }
    Ok((jacobian, determinant, determinant.abs(), inverse, gradients))
}

fn matmul3(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            for inner in 0..3 {
                result[row][column] += left[row][inner] * right[inner][column];
            }
        }
    }
    result
}

fn negative_matmul3(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = matmul3(left, right);
    for row in &mut result {
        for value in row {
            *value = -*value;
        }
    }
    result
}

fn transpose3(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn vector_between(a: sonium_fem::mesh::Point, b: sonium_fem::mesh::Point) -> [f64; 3] {
    [a.x - b.x, a.y - b.y, a.z - b.z]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn matvec3(matrix: &[[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        dot3(matrix[0], vector),
        dot3(matrix[1], vector),
        dot3(matrix[2], vector),
    ]
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    dot(left, right)
}

pub fn advance_pc_fan_cfd_komega_bdf_step(
    mesh: &Mesh,
    config: &KOmegaTransportConfig,
    state: &PcFanCfdKOmegaBdfState,
) -> Result<PcFanKOmegaTransportStep, PcFanKOmegaTransportError> {
    let fields = KOmegaTransportElementFields::uniform(mesh, config);
    advance_pc_fan_cfd_komega_bdf_step_with_element_fields(mesh, config, state, &fields)
}

/// Advance a transferred PC-fan P1 state using resolved element-local fields.
///
/// This is the production coupling seam: the caller supplies element
/// velocities, eddy viscosities, and production terms from the current mean
/// flow iterate, while this adapter owns target-mesh assembly and BDF history.
pub fn advance_pc_fan_cfd_komega_bdf_step_with_element_fields(
    mesh: &Mesh,
    config: &KOmegaTransportConfig,
    state: &PcFanCfdKOmegaBdfState,
    fields: &KOmegaTransportElementFields,
) -> Result<PcFanKOmegaTransportStep, PcFanKOmegaTransportError> {
    let fem_state = KOmegaTransportState {
        k: state.k.clone(),
        k_prev: state.k_prev.clone(),
        omega: state.omega.clone(),
        omega_prev: state.omega_prev.clone(),
        time: state.time,
        step: state.step,
    };
    let step = advance_komega_bdf_step_with_element_fields(
        mesh,
        PolynomialDegree::P1,
        config,
        &fem_state,
        fields,
    )?;
    let state = PcFanCfdKOmegaBdfState {
        k: step.state.k,
        k_prev: step.state.k_prev,
        omega: step.state.omega,
        omega_prev: step.state.omega_prev,
        time: step.state.time,
        step: step.state.step,
    };
    Ok(PcFanKOmegaTransportStep {
        state,
        report: step.report,
        k_residual: step.k_residual,
        omega_residual: step.omega_residual,
        linearization: step.linearization,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::physics::fluid::turbulence::KOmegaSstConstants;
    use sonium_fem::spaces::{FunctionSpace, H1Space};

    #[test]
    fn resolved_adapter_derives_spatial_transport_fields_from_velocity() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let dofs = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
        let state = initialize_pc_fan_cfd_komega_bdf_state(&mesh, 0.25, 3, 2.0, 4.0)
            .expect("positive scalar state should initialise");
        let velocity = mesh
            .nodes
            .iter()
            .flat_map(|node| [node.x, 0.0, 0.0])
            .collect::<Vec<_>>();
        let fields = derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &state, 10.0)
            .expect("tetrahedral velocity fields should derive");

        assert_eq!(state.k.len(), dofs);
        assert_eq!(fields.velocity.len(), mesh.num_elements());
        assert_eq!(fields.k_production.len(), mesh.num_elements());
        assert!(fields.k_production.iter().all(|value| *value > 0.0));
        assert!(fields.omega_production.iter().all(|value| *value > 0.0));
        assert!(
            fields
                .velocity
                .iter()
                .all(|value| value.iter().all(|component| component.is_finite()))
        );
    }

    #[test]
    fn adapter_advances_transferred_p1_history() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let dofs = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
        let constants = KOmegaSstConstants::default();
        let k = 1.5;
        let omega = 2.5;
        let state = PcFanCfdKOmegaBdfState {
            k: vec![k; dofs],
            k_prev: vec![k; dofs],
            omega: vec![omega; dofs],
            omega_prev: vec![omega; dofs],
            time: 0.0,
            step: 7,
        };
        let config = KOmegaTransportConfig {
            bdf_order: 1,
            molecular_viscosity: 1.0e-5,
            k_production: constants.beta_star * omega * k,
            omega_production: constants.beta * omega * omega,
            constants,
            ..KOmegaTransportConfig::default()
        };

        let result = advance_pc_fan_cfd_komega_bdf_step(&mesh, &config, &state).unwrap();

        assert_eq!(result.state.step, 8);
        assert_eq!(result.state.time, config.dt);
        assert!(
            result
                .state
                .k
                .iter()
                .all(|value| (*value - k).abs() < 1.0e-10)
        );
        assert!(
            result
                .state
                .omega
                .iter()
                .all(|value| (*value - omega).abs() < 1.0e-10)
        );
        assert_eq!(
            result.report.provenance,
            "fem:turbulence:k-omega-bdf-transport:v1"
        );
    }

    #[test]
    fn adapter_publishes_lagged_momentum_viscosity() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let dofs = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
        let state = PcFanCfdKOmegaBdfState {
            k: vec![2.0; dofs],
            k_prev: vec![2.0; dofs],
            omega: vec![4.0; dofs],
            omega_prev: vec![4.0; dofs],
            time: 0.0,
            step: 0,
        };
        let mut momentum =
            sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesConfig::default();

        let mean = apply_pc_fan_cfd_komega_mean_eddy_viscosity(&mut momentum, &mesh, &state, 10.0)
            .unwrap();

        assert!((mean - 0.2).abs() < 1.0e-12);
        assert_eq!(momentum.transported_eddy_viscosity, Some(mean));
        let element_viscosities = momentum
            .transported_eddy_viscosity_per_element
            .as_ref()
            .expect("adapter should publish element viscosities");
        assert_eq!(element_viscosities.len(), mesh.num_elements());
        assert!(
            element_viscosities
                .iter()
                .all(|value| (*value - 0.2).abs() < 1.0e-12)
        );
    }

    fn csr_row_abs_sum(matrix: &sonium_solvers::CsrMatrix<f64>, row: usize) -> f64 {
        matrix.row_ptrs[row..=row + 1]
            .windows(2)
            .flat_map(|window| {
                matrix.values[window[0]..window[1]]
                    .iter()
                    .map(|value| value.abs())
            })
            .sum()
    }

    #[test]
    fn coupled_linearization_has_rectangular_finite_nonzero_blocks() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let scalar_space = H1Space::new(PolynomialDegree::P1);
        let velocity_space = sonium_fem::spaces::VectorH1Space::new(PolynomialDegree::P1, 3);
        let scalar_dofs = scalar_space.num_global_dofs(&mesh);
        let state = PcFanCfdKOmegaBdfState {
            k: (0..scalar_dofs).map(|index| 1.0 + index as f64).collect(),
            k_prev: vec![1.0; scalar_dofs],
            omega: (0..scalar_dofs).map(|index| 2.0 + index as f64).collect(),
            omega_prev: vec![2.0; scalar_dofs],
            time: 0.0,
            step: 0,
        };
        let velocity = mesh
            .nodes
            .iter()
            .flat_map(|node| [node.x, 2.0 * node.y, 3.0 * node.z])
            .collect::<Vec<_>>();
        let fields = derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &state, 10.0)
            .expect("unit-cube tetrahedra should produce coupling fields");

        let linearization = assemble_pc_fan_cfd_komega_coupled_linearization(
            &mesh,
            &velocity_space,
            &scalar_space,
            &velocity,
            &state,
            &fields,
            1.2,
            10.0,
        )
        .expect("coupled blocks should assemble");

        let velocity_dofs = velocity_space.num_global_dofs(&mesh);
        assert_eq!(
            linearization.provenance,
            PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE
        );
        assert_eq!(linearization.momentum_wrt_k.num_rows, velocity_dofs);
        assert_eq!(linearization.momentum_wrt_k.num_cols, scalar_dofs);
        assert_eq!(linearization.k_wrt_velocity.num_rows, scalar_dofs);
        assert_eq!(linearization.k_wrt_velocity.num_cols, velocity_dofs);
        assert_eq!(linearization.k_wrt_k.num_rows, scalar_dofs);
        assert_eq!(linearization.k_wrt_k.num_cols, scalar_dofs);
        assert_eq!(linearization.k_wrt_omega.num_rows, scalar_dofs);
        assert_eq!(linearization.k_wrt_omega.num_cols, scalar_dofs);
        assert_eq!(linearization.omega_wrt_k.num_rows, scalar_dofs);
        assert_eq!(linearization.omega_wrt_k.num_cols, scalar_dofs);
        assert_eq!(linearization.omega_wrt_omega.num_rows, scalar_dofs);
        assert_eq!(linearization.omega_wrt_omega.num_cols, scalar_dofs);
        assert!(
            linearization
                .momentum_wrt_k
                .values
                .iter()
                .chain(&linearization.momentum_wrt_omega.values)
                .chain(&linearization.k_wrt_velocity.values)
                .chain(&linearization.omega_wrt_velocity.values)
                .all(|value| value.is_finite())
        );
        for matrix in [
            &linearization.k_wrt_k,
            &linearization.k_wrt_omega,
            &linearization.omega_wrt_k,
            &linearization.omega_wrt_omega,
        ] {
            assert!(matrix.values.iter().all(|value| value.is_finite()));
            assert!(matrix.values.iter().any(|value| value.abs() > 1.0e-14));
        }
        assert!(
            linearization
                .momentum_wrt_k
                .values
                .iter()
                .any(|value| value.abs() > 1.0e-14)
        );
        assert!(
            linearization
                .k_wrt_velocity
                .values
                .iter()
                .any(|value| value.abs() > 1.0e-14)
        );
        // The vector test basis must contribute all three component rows. A
        // scalar `test.value(row)` would silently zero the y/z basis rows.
        assert!(csr_row_abs_sum(&linearization.momentum_wrt_k, 1) > 1.0e-14);
        assert!(csr_row_abs_sum(&linearization.momentum_wrt_k, 2) > 1.0e-14);
    }

    #[test]
    fn coupled_linearization_rejects_invalid_density_limiter_and_field_layout() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let scalar_space = H1Space::new(PolynomialDegree::P1);
        let velocity_space = sonium_fem::spaces::VectorH1Space::new(PolynomialDegree::P1, 3);
        let state = initialize_pc_fan_cfd_komega_bdf_state(&mesh, 0.0, 0, 1.0, 2.0)
            .expect("positive state should initialise");
        let velocity = vec![0.5; velocity_space.num_global_dofs(&mesh)];
        let fields = derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &state, 10.0)
            .expect("unit-cube tetrahedra should produce coupling fields");

        assert!(matches!(
            assemble_pc_fan_cfd_komega_coupled_linearization(
                &mesh,
                &velocity_space,
                &scalar_space,
                &velocity,
                &state,
                &fields,
                0.0,
                10.0,
            ),
            Err(PcFanKOmegaTransportError::InvalidCoupling(_))
        ));
        assert!(matches!(
            assemble_pc_fan_cfd_komega_coupled_linearization(
                &mesh,
                &velocity_space,
                &scalar_space,
                &velocity,
                &state,
                &fields,
                1.0,
                0.0,
            ),
            Err(PcFanKOmegaTransportError::InvalidCoupling(_))
        ));

        let mut malformed_fields = fields.clone();
        malformed_fields.velocity.pop();
        assert!(matches!(
            assemble_pc_fan_cfd_komega_coupled_linearization(
                &mesh,
                &velocity_space,
                &scalar_space,
                &velocity,
                &state,
                &malformed_fields,
                1.0,
                10.0,
            ),
            Err(PcFanKOmegaTransportError::InvalidCoupling(_))
        ));
    }

    #[test]
    fn monolithic_linearization_pads_pressure_and_preserves_all_blocks() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let scalar_space = H1Space::new(PolynomialDegree::P1);
        let velocity_space = sonium_fem::spaces::VectorH1Space::new(PolynomialDegree::P1, 3);
        let scalar_dofs = scalar_space.num_global_dofs(&mesh);
        let velocity_dofs = velocity_space.num_global_dofs(&mesh);
        let state = PcFanCfdKOmegaBdfState {
            k: (0..scalar_dofs).map(|index| 1.0 + index as f64).collect(),
            k_prev: vec![1.0; scalar_dofs],
            omega: (0..scalar_dofs).map(|index| 2.0 + index as f64).collect(),
            omega_prev: vec![2.0; scalar_dofs],
            time: 0.0,
            step: 0,
        };
        let velocity = mesh
            .nodes
            .iter()
            .flat_map(|node| [node.x, 2.0 * node.y, 3.0 * node.z])
            .collect::<Vec<_>>();
        let fields = derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &state, 10.0)
            .expect("unit-cube tetrahedra should produce coupling fields");
        let config = KOmegaTransportConfig {
            dt: 1.0e-3,
            bdf_order: 1,
            ..KOmegaTransportConfig::default()
        };
        let scalar_step =
            advance_pc_fan_cfd_komega_bdf_step_with_element_fields(&mesh, &config, &state, &fields)
                .expect("scalar tangent should assemble");
        let coupled = assemble_pc_fan_cfd_komega_coupled_linearization(
            &mesh,
            &velocity_space,
            &scalar_space,
            &velocity,
            &state,
            &fields,
            1.2,
            10.0,
        )
        .expect("off-diagonal blocks should assemble");
        let pressure_dofs = scalar_dofs;
        let flow_dimension = velocity_dofs + pressure_dofs;
        let flow_jacobian = CsrMatrix::from_triplets(
            flow_dimension,
            flow_dimension,
            (0..flow_dimension)
                .map(|index| (index, index, 1.0))
                .collect(),
        );

        let monolithic = assemble_pc_fan_cfd_komega_monolithic_linearization(
            &flow_jacobian,
            &scalar_step.linearization,
            &coupled,
        )
        .expect("square transported block should assemble");
        assert_eq!(
            monolithic.provenance,
            PC_FAN_KOMEGA_MONOLITHIC_BLOCK_PROVENANCE
        );
        assert_eq!(monolithic.flow_dimension, flow_dimension);
        assert_eq!(monolithic.k_dimension, scalar_dofs);
        assert_eq!(monolithic.omega_dimension, scalar_dofs);
        assert_eq!(
            monolithic.state_jacobian.num_rows,
            flow_dimension + 2 * scalar_dofs
        );
        assert_eq!(
            monolithic.state_jacobian.num_cols,
            flow_dimension + 2 * scalar_dofs
        );
        let k_offset = flow_dimension;
        let omega_offset = flow_dimension + scalar_dofs;
        let has_nonzero_block =
            |row_start: usize, row_end: usize, col_start: usize, col_end: usize| {
                (row_start..row_end).any(|row| {
                    let start = monolithic.state_jacobian.row_ptrs[row];
                    let end = monolithic.state_jacobian.row_ptrs[row + 1];
                    monolithic.state_jacobian.col_indices[start..end]
                        .iter()
                        .zip(&monolithic.state_jacobian.values[start..end])
                        .any(|(&column, &value)| {
                            (col_start..col_end).contains(&column) && value.abs() > 1.0e-14
                        })
                })
            };
        assert!(has_nonzero_block(
            k_offset,
            omega_offset,
            k_offset,
            omega_offset
        ));
        assert!(has_nonzero_block(
            k_offset,
            omega_offset,
            omega_offset,
            omega_offset + scalar_dofs
        ));
        assert!(has_nonzero_block(
            omega_offset,
            omega_offset + scalar_dofs,
            k_offset,
            omega_offset
        ));
        assert!(has_nonzero_block(
            omega_offset,
            omega_offset + scalar_dofs,
            omega_offset,
            omega_offset + scalar_dofs
        ));
        // Pressure rows have their flow diagonal but no transported scalar
        // cross entries; scalar rows retain their velocity coupling.
        let pressure_row = velocity_dofs;
        let pressure_start = monolithic.state_jacobian.row_ptrs[pressure_row];
        let pressure_end = monolithic.state_jacobian.row_ptrs[pressure_row + 1];
        assert_eq!(pressure_end - pressure_start, 1);
        assert_eq!(
            monolithic.state_jacobian.col_indices[pressure_start],
            pressure_row
        );
        let k_row = flow_dimension;
        let k_start = monolithic.state_jacobian.row_ptrs[k_row];
        let k_end = monolithic.state_jacobian.row_ptrs[k_row + 1];
        assert!(
            monolithic.state_jacobian.col_indices[k_start..k_end]
                .iter()
                .any(|&column| column < velocity_dofs)
        );
    }

    #[test]
    fn nonlinear_scalar_residual_reassembles_at_current_state() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let scalar_space = H1Space::new(PolynomialDegree::P1);
        let scalar_dofs = scalar_space.num_global_dofs(&mesh);
        let state = PcFanCfdKOmegaBdfState {
            k: vec![1.0; scalar_dofs],
            k_prev: vec![1.0; scalar_dofs],
            omega: vec![2.0; scalar_dofs],
            omega_prev: vec![2.0; scalar_dofs],
            time: 0.0,
            step: 0,
        };
        let velocity = mesh
            .nodes
            .iter()
            .flat_map(|node| [node.x, 2.0 * node.y, 3.0 * node.z])
            .collect::<Vec<_>>();
        let fields = derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &state, 10.0)
            .expect("unit-cube fields should assemble");
        let config = KOmegaTransportConfig {
            dt: 1.0e-3,
            bdf_order: 1,
            ..KOmegaTransportConfig::default()
        };
        let (residual, omega_residual, linearization) =
            assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
                &mesh, &config, &state, &state, &fields,
            )
            .expect("nonlinear scalar residual should assemble");
        assert_eq!(residual.len(), scalar_dofs);
        assert_eq!(omega_residual.len(), scalar_dofs);
        assert!(residual.iter().all(|value| value.is_finite()));
        assert!(omega_residual.iter().all(|value| value.is_finite()));
        assert_eq!(linearization.k_state_jacobian.num_rows, scalar_dofs);
        assert_eq!(linearization.omega_state_jacobian.num_rows, scalar_dofs);

        let mut perturbed = state.clone();
        perturbed.k[0] += 1.0e-6;
        let perturbed_fields =
            derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &perturbed, 10.0)
                .expect("perturbed fields should assemble");
        let (perturbed_residual, _, _) =
            assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
                &mesh,
                &config,
                &state,
                &perturbed,
                &perturbed_fields,
            )
            .expect("perturbed nonlinear residual should assemble");
        assert!(
            residual
                .iter()
                .zip(perturbed_residual)
                .any(|(before, after)| (after - before).abs() > 1.0e-12)
        );
    }

    #[test]
    fn nonlinear_scalar_jacobian_matches_directional_residual_change() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let scalar_space = H1Space::new(PolynomialDegree::P1);
        let velocity_space = sonium_fem::spaces::VectorH1Space::new(PolynomialDegree::P1, 3);
        let scalar_dofs = scalar_space.num_global_dofs(&mesh);
        let state = PcFanCfdKOmegaBdfState {
            k: vec![2.0; scalar_dofs],
            k_prev: vec![1.0; scalar_dofs],
            omega: vec![4.0; scalar_dofs],
            omega_prev: vec![2.0; scalar_dofs],
            time: 0.0,
            step: 0,
        };
        let velocity = mesh
            .nodes
            .iter()
            .flat_map(|node| [node.x, 2.0 * node.y, 3.0 * node.z])
            .collect::<Vec<_>>();
        let fields = derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &state, 10.0)
            .expect("unit-cube fields should assemble");
        let config = KOmegaTransportConfig {
            dt: 1.0e-3,
            bdf_order: 1,
            ..KOmegaTransportConfig::default()
        };
        let (residual, omega_residual, scalar_linearization) =
            assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
                &mesh, &config, &state, &state, &fields,
            )
            .expect("nonlinear scalar residual should assemble");
        let coupled_linearization = assemble_pc_fan_cfd_komega_coupled_linearization(
            &mesh,
            &velocity_space,
            &scalar_space,
            &velocity,
            &state,
            &fields,
            1.2,
            10.0,
        )
        .expect("coupled scalar blocks should assemble");

        let epsilon = 1.0e-7;
        let delta_k = (0..scalar_dofs)
            .map(|index| 0.15 + 0.02 * index as f64)
            .collect::<Vec<_>>();
        let delta_omega = (0..scalar_dofs)
            .map(|index| -0.18 + 0.015 * index as f64)
            .collect::<Vec<_>>();
        let mut perturbed = state.clone();
        perturbed
            .k
            .iter_mut()
            .zip(&delta_k)
            .for_each(|(value, direction)| *value += epsilon * direction);
        perturbed
            .omega
            .iter_mut()
            .zip(&delta_omega)
            .for_each(|(value, direction)| *value += epsilon * direction);
        let perturbed_fields =
            derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &perturbed, 10.0)
                .expect("perturbed fields should assemble");
        let (perturbed_residual, perturbed_omega_residual, _) =
            assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
                &mesh,
                &config,
                &state,
                &perturbed,
                &perturbed_fields,
            )
            .expect("perturbed nonlinear scalar residual should assemble");

        let k_base_direction = scalar_linearization.k_state_jacobian.apply(&delta_k);
        let k_state_direction = coupled_linearization.k_wrt_k.apply(&delta_k);
        let k_omega_direction = coupled_linearization.k_wrt_omega.apply(&delta_omega);
        let omega_base_direction = scalar_linearization
            .omega_state_jacobian
            .apply(&delta_omega);
        let omega_k_direction = coupled_linearization.omega_wrt_k.apply(&delta_k);
        let omega_state_direction = coupled_linearization.omega_wrt_omega.apply(&delta_omega);
        let k_direction = k_base_direction
            .iter()
            .zip(&k_state_direction)
            .zip(&k_omega_direction)
            .map(|((base, k_state), omega_state)| base + k_state + omega_state)
            .collect::<Vec<_>>();
        let omega_direction = omega_base_direction
            .iter()
            .zip(&omega_k_direction)
            .zip(&omega_state_direction)
            .map(|((base, k_state), omega_state)| base + k_state + omega_state)
            .collect::<Vec<_>>();
        let k_max_error = residual
            .iter()
            .zip(&perturbed_residual)
            .zip(&k_direction)
            .map(|((before, after), direction)| ((after - before) / epsilon - direction).abs())
            .fold(0.0_f64, f64::max);
        let omega_max_error = omega_residual
            .iter()
            .zip(&perturbed_omega_residual)
            .zip(&omega_direction)
            .map(|((before, after), direction)| ((after - before) / epsilon - direction).abs())
            .fold(0.0_f64, f64::max);
        let max_error = k_max_error.max(omega_max_error);
        assert!(
            max_error < 1.0e-4,
            "directional Jacobian error: {max_error} (k={k_max_error}, omega={omega_max_error})"
        );
    }

    #[test]
    fn scalar_mesh_motion_design_residual_matches_reassembly() {
        let mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let scalar_space = H1Space::new(PolynomialDegree::P1);
        let scalar_dofs = scalar_space.num_global_dofs(&mesh);
        let current = PcFanCfdKOmegaBdfState {
            k: (0..scalar_dofs)
                .map(|index| 1.4 + 0.05 * index as f64)
                .collect(),
            k_prev: (0..scalar_dofs)
                .map(|index| 1.1 + 0.03 * index as f64)
                .collect(),
            omega: (0..scalar_dofs)
                .map(|index| 3.2 + 0.07 * index as f64)
                .collect(),
            omega_prev: (0..scalar_dofs)
                .map(|index| 2.7 + 0.04 * index as f64)
                .collect(),
            time: 0.0,
            step: 1,
        };
        let velocity = mesh
            .nodes
            .iter()
            .flat_map(|node| [node.x, 0.4 * node.y, 0.7 * node.z])
            .collect::<Vec<_>>();
        let fields = derive_pc_fan_cfd_komega_element_fields(&mesh, &velocity, &current, 10.0)
            .expect("unit-cube fields assemble");
        let config = KOmegaTransportConfig {
            dt: 0.25,
            bdf_order: 1,
            ..KOmegaTransportConfig::default()
        };
        let nodal_motion = mesh
            .nodes
            .iter()
            .map(|node| {
                [[
                    0.02 + 0.01 * node.x,
                    -0.015 + 0.008 * node.y,
                    0.012 + 0.006 * node.z,
                ]]
            })
            .collect::<Vec<[[f64; 3]; 1]>>();
        let analytic = assemble_pc_fan_cfd_komega_scalar_mesh_motion_design_residual::<1>(
            &mesh,
            &config,
            &current,
            &current,
            &velocity,
            &fields,
            &nodal_motion,
        )
        .expect("scalar mesh-motion derivative assemble");

        let epsilon = 1.0e-6;
        let mut perturbed_mesh = mesh.clone();
        for (point, motion) in perturbed_mesh.nodes.iter_mut().zip(&nodal_motion) {
            point.x += epsilon * motion[0][0];
            point.y += epsilon * motion[0][1];
            point.z += epsilon * motion[0][2];
        }
        let perturbed_fields =
            derive_pc_fan_cfd_komega_element_fields(&perturbed_mesh, &velocity, &current, 10.0)
                .expect("perturbed unit-cube fields assemble");
        let (base_k, base_omega, _) =
            assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
                &mesh, &config, &current, &current, &fields,
            )
            .expect("base scalar residual assemble");
        let (perturbed_k, perturbed_omega, _) =
            assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
                &perturbed_mesh,
                &config,
                &current,
                &current,
                &perturbed_fields,
            )
            .expect("perturbed scalar residual assemble");

        for (row, (base, perturbed)) in base_k.iter().zip(&perturbed_k).enumerate() {
            let finite_difference = (perturbed - base) / epsilon;
            let error = (finite_difference - analytic[row][0]).abs();
            let tolerance = 5.0e-5 * (1.0 + finite_difference.abs().max(analytic[row][0].abs()));
            assert!(
                error < tolerance,
                "k mesh-motion derivative row {row}: finite difference={finite_difference}, analytic={}, error={error}, tolerance={tolerance}",
                analytic[row][0]
            );
        }
        let offset = scalar_dofs;
        for (row, (base, perturbed)) in base_omega.iter().zip(&perturbed_omega).enumerate() {
            let finite_difference = (perturbed - base) / epsilon;
            let error = (finite_difference - analytic[offset + row][0]).abs();
            let tolerance =
                5.0e-5 * (1.0 + finite_difference.abs().max(analytic[offset + row][0].abs()));
            assert!(
                error < tolerance,
                "omega mesh-motion derivative row {row}: finite difference={finite_difference}, analytic={}, error={error}, tolerance={tolerance}",
                analytic[offset + row][0]
            );
        }
    }
}
