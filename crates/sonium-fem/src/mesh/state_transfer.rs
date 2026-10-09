//! Checked CFD state transfer across topology-changing tetrahedral meshes.
//!
//! This module owns the reusable transfer layer extracted from
//! `sonium-sim-pcfan`'s `state_transfer`: P1 nodal states with conservative
//! correction, Taylor--Hood P2/P1 states with exact-transpose adjoints,
//! BDF history handoffs and their reverse chains, and P1 k--omega history
//! transfer with physical-domain (`k >= 0`, `omega > 0`) enforcement.
//!
//! The low-level interpolation, conservative correction, and H1 adjoint maps
//! live in [`super::interpolation`]; this module binds them to CFD field
//! layouts (`BdfState`, interleaved P2 velocity, P1 scalars) and records
//! per-transfer diagnostics. Product geometry, operating points, and fan
//! objectives stay in the simulation crates, which keep their own report
//! shapes and provenance strings as thin wrappers over these functions.

use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, InterpolationError, Mesh};
use crate::physics::fluid::stokes::time_stepping::BdfState;
use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

/// Failure while validating or transferring a resolved CFD state.
#[derive(Debug, Clone, PartialEq)]
pub enum CfdStateTransferError {
    /// The source or target mesh is not a supported oriented P1 tetrahedral
    /// volume mesh.
    InvalidMesh {
        /// Mesh role that failed validation.
        role: &'static str,
        /// Human-readable validation detail.
        reason: String,
    },
    /// The velocity array does not have one vector per source mesh node.
    SourceVelocityCount { expected: usize, actual: usize },
    /// The pressure array does not have one value per source mesh node.
    SourcePressureCount { expected: usize, actual: usize },
    /// A source velocity component is not finite.
    NonFiniteSourceVelocity { node: usize, component: usize },
    /// A source pressure value is not finite.
    NonFiniteSourcePressure { node: usize },
    /// The source P2 velocity state does not match the scalar/vector H1 DOF
    /// layout used by the transient solver.
    SourceVelocityDofCount { expected: usize, actual: usize },
    /// The source P1 pressure state does not match the H1 DOF layout used by
    /// the transient solver.
    SourcePressureDofCount { expected: usize, actual: usize },
    /// The target P2 velocity adjoint does not match the target vector H1
    /// layout used by the transient solver.
    TargetVelocityDofCount { expected: usize, actual: usize },
    /// The target P1 pressure adjoint does not match the target H1 layout used
    /// by the transient solver.
    TargetPressureDofCount { expected: usize, actual: usize },
    /// A source higher-order velocity DOF is not finite.
    NonFiniteSourceVelocityDof { dof: usize },
    /// A source higher-order pressure DOF is not finite.
    NonFiniteSourcePressureDof { dof: usize },
    /// A target adjoint higher-order velocity DOF is not finite.
    NonFiniteTargetVelocityDof { dof: usize },
    /// A target adjoint pressure DOF is not finite.
    NonFiniteTargetPressureDof { dof: usize },
    /// The source BDF time metadata is not finite.
    NonFiniteSourceTime,
    /// The FEM checked transfer rejected the operation.
    Interpolation(InterpolationError),
}

impl std::fmt::Display for CfdStateTransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMesh { role, reason } => write!(f, "invalid {role} mesh: {reason}"),
            Self::SourceVelocityCount { expected, actual } => {
                write!(f, "source velocity has {actual} nodes, expected {expected}")
            }
            Self::SourcePressureCount { expected, actual } => {
                write!(f, "source pressure has {actual} nodes, expected {expected}")
            }
            Self::NonFiniteSourceVelocity { node, component } => write!(
                f,
                "source velocity component {component} at node {node} is not finite"
            ),
            Self::NonFiniteSourcePressure { node } => {
                write!(f, "source pressure at node {node} is not finite")
            }
            Self::SourceVelocityDofCount { expected, actual } => write!(
                f,
                "source P2 velocity has {actual} DOFs, expected {expected}"
            ),
            Self::SourcePressureDofCount { expected, actual } => write!(
                f,
                "source P1 pressure has {actual} DOFs, expected {expected}"
            ),
            Self::TargetVelocityDofCount { expected, actual } => write!(
                f,
                "target P2 velocity adjoint has {actual} DOFs, expected {expected}"
            ),
            Self::TargetPressureDofCount { expected, actual } => write!(
                f,
                "target P1 pressure adjoint has {actual} DOFs, expected {expected}"
            ),
            Self::NonFiniteSourceVelocityDof { dof } => {
                write!(f, "source P2 velocity DOF {dof} is not finite")
            }
            Self::NonFiniteSourcePressureDof { dof } => {
                write!(f, "source P1 pressure DOF {dof} is not finite")
            }
            Self::NonFiniteTargetVelocityDof { dof } => {
                write!(f, "target P2 velocity adjoint DOF {dof} is not finite")
            }
            Self::NonFiniteTargetPressureDof { dof } => {
                write!(f, "target P1 pressure adjoint DOF {dof} is not finite")
            }
            Self::NonFiniteSourceTime => write!(f, "source BDF time is not finite"),
            Self::Interpolation(error) => write!(f, "state interpolation failed: {error}"),
        }
    }
}

impl std::error::Error for CfdStateTransferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Interpolation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<InterpolationError> for CfdStateTransferError {
    fn from(error: InterpolationError) -> Self {
        Self::Interpolation(error)
    }
}

/// Diagnostics proving the scope and conservation correction of a transfer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfdStateTransferReport {
    /// Number of nodes in the source mesh.
    pub source_node_count: usize,
    /// Number of nodes in the target mesh.
    pub target_node_count: usize,
    /// Number of elements in the source mesh.
    pub source_element_count: usize,
    /// Number of elements in the target mesh.
    pub target_element_count: usize,
    /// Source domain volume.
    pub source_volume_m3: f64,
    /// Target domain volume.
    pub target_volume_m3: f64,
    /// Largest conservative correction applied to any velocity component.
    pub max_velocity_correction_m_s: f64,
    /// Largest conservative correction applied to pressure.
    pub max_pressure_correction_pa: f64,
}

/// A fully transferred resolved-CFD state.
#[derive(Debug, Clone, PartialEq)]
pub struct CfdStateTransfer {
    /// Target-node velocity in product Cartesian coordinates.
    pub velocity_m_s: Vec<[f64; 3]>,
    /// Target-node pressure.
    pub pressure_pa: Vec<f64>,
    /// Validation and conservation diagnostics for the transfer.
    pub report: CfdStateTransferReport,
}

/// Diagnostics for the transient Taylor--Hood P2/P1 state transfer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct P2P1StateTransferReport {
    /// Number of nodes in the source mesh.
    pub source_node_count: usize,
    /// Number of nodes in the target mesh.
    pub target_node_count: usize,
    /// Number of elements in the source mesh.
    pub source_element_count: usize,
    /// Number of elements in the target mesh.
    pub target_element_count: usize,
    /// Source P2 velocity DOF count, including all three interleaved
    /// components.
    pub source_velocity_dof_count: usize,
    /// Target P2 velocity DOF count, including all three interleaved
    /// components.
    pub target_velocity_dof_count: usize,
    /// Source P1 pressure DOF count.
    pub source_pressure_dof_count: usize,
    /// Target P1 pressure DOF count.
    pub target_pressure_dof_count: usize,
}

/// A fully transferred transient Taylor--Hood state.
#[derive(Debug, Clone, PartialEq)]
pub struct P2P1StateTransfer {
    /// Target P2 velocity DOFs in `VectorH1Space` interleaved ordering:
    /// `[node_or_edge_0_x, node_or_edge_0_y, node_or_edge_0_z, ...]`.
    pub velocity_dofs: Vec<f64>,
    /// Target P1 pressure DOFs.
    pub pressure_dofs: Vec<f64>,
    /// Validation diagnostics for the transfer.
    pub report: P2P1StateTransferReport,
}

/// Result of applying the exact transpose of a P2/P1 state transfer.
///
/// The values are ordered on the source mesh. This is a state-adjoint
/// transfer only; it does not differentiate the discrete topology or the
/// remesher that selected the mesh pair.
#[derive(Debug, Clone, PartialEq)]
pub struct P2P1StateAdjointTransfer {
    /// Source P2 velocity dual values in interleaved vector-H1 ordering.
    pub velocity_dofs: Vec<f64>,
    /// Source P1 pressure dual values.
    pub pressure_dofs: Vec<f64>,
    /// Validation diagnostics for the mesh pair and DOF layouts.
    pub report: P2P1StateTransferReport,
}

/// Diagnostics for a complete topology-changing BDF handoff chain.
#[derive(Debug, Clone, PartialEq)]
pub struct P2P1StateTransferChainReport {
    /// Number of source-to-target remesh transitions.
    pub stage_count: usize,
    /// Per-transition transfer diagnostics in chronological order.
    pub stages: Vec<P2P1StateTransferReport>,
    /// Target matrices must be assembled again after every transition.
    pub requires_operator_reassembly: bool,
    /// Stable provenance for the checked handoff contract.
    pub provenance: &'static str,
}

/// P2/P1 BDF states after each topology-changing remesh transition.
#[derive(Debug, Clone)]
pub struct P2P1StateTransferChain {
    /// One transferred state per transition, in mesh-chain order.
    pub states: Vec<BdfState>,
    /// Chain-level and per-stage transfer diagnostics.
    pub report: P2P1StateTransferChainReport,
}

/// Diagnostics for a reverse multi-stage P2/P1 state-adjoint transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct P2P1StateAdjointTransferChainReport {
    /// Number of source-to-target remesh transitions traversed in reverse.
    pub stage_count: usize,
    /// Per-transition diagnostics, in reverse traversal order.
    pub stages: Vec<P2P1StateTransferReport>,
    /// Stable provenance for the exact transpose contract.
    pub provenance: &'static str,
}

/// P2/P1 BDF adjoints in reverse mesh-chain order.
#[derive(Debug, Clone)]
pub struct P2P1StateAdjointTransferChain {
    /// One adjoint state per transition, first at the penultimate mesh and
    /// last at the original source mesh.
    pub states: Vec<BdfState>,
    /// Chain-level transpose diagnostics.
    pub report: P2P1StateAdjointTransferChainReport,
}

/// Error while transferring transported k--omega history across a remesh.
///
/// This is intentionally separate from [`CfdStateTransferError`]: the
/// latter covers the velocity/pressure restart contract, while this error
/// also enforces the physical admissibility of the two transported scalar
/// fields (`k >= 0` and `omega > 0`).
#[derive(Debug, Clone, PartialEq)]
pub enum KOmegaStateTransferError {
    /// The velocity/pressure mesh-pair validation or interpolation failed.
    StateTransfer(CfdStateTransferError),
    /// A scalar field does not match the P1 H1 layout of its source mesh.
    FieldDofCount {
        /// `"k"` or `"omega"`.
        field: &'static str,
        /// Expected source P1 scalar DOF count.
        expected: usize,
        /// Supplied source scalar DOF count.
        actual: usize,
    },
    /// A scalar source or transferred value is not finite.
    NonFiniteField {
        /// `"k"` or `"omega"`.
        field: &'static str,
        /// Zero-based scalar DOF index.
        dof: usize,
    },
    /// A target scalar adjoint value is not finite.
    NonFiniteTargetField {
        /// `"k"` or `"omega"`.
        field: &'static str,
        /// Zero-based scalar DOF index.
        dof: usize,
    },
    /// A scalar value violates the k--omega physical domain.
    InvalidFieldValue {
        /// `"k"` or `"omega"`.
        field: &'static str,
        /// Zero-based scalar DOF index.
        dof: usize,
        /// Invalid value.
        value: f64,
    },
    /// The BDF history timestamp is not finite.
    NonFiniteSourceTime,
}

impl std::fmt::Display for KOmegaStateTransferError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateTransfer(error) => write!(formatter, "state transfer failed: {error}"),
            Self::FieldDofCount {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} scalar state has {actual} DOFs, expected {expected}"
            ),
            Self::NonFiniteField { field, dof } => {
                write!(formatter, "{field} scalar DOF {dof} is not finite")
            }
            Self::NonFiniteTargetField { field, dof } => {
                write!(formatter, "{field} scalar adjoint DOF {dof} is not finite")
            }
            Self::InvalidFieldValue { field, dof, value } => write!(
                formatter,
                "{field} scalar DOF {dof} has physically invalid value {value}"
            ),
            Self::NonFiniteSourceTime => write!(formatter, "source BDF time is not finite"),
        }
    }
}

impl std::error::Error for KOmegaStateTransferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StateTransfer(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CfdStateTransferError> for KOmegaStateTransferError {
    fn from(error: CfdStateTransferError) -> Self {
        Self::StateTransfer(error)
    }
}

/// P1 k--omega fields at the current and previous BDF time levels.
///
/// This is a history handoff record, not a transported turbulence solver. A
/// production k--omega implementation must assemble and solve its scalar
/// transport equations and their coupled Jacobian after receiving this
/// record.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaBdfState {
    /// Current turbulent kinetic energy, in P1 scalar DOF order.
    pub k: Vec<f64>,
    /// Previous-step turbulent kinetic energy, in P1 scalar DOF order.
    pub k_prev: Vec<f64>,
    /// Current specific dissipation rate, in P1 scalar DOF order.
    pub omega: Vec<f64>,
    /// Previous-step specific dissipation rate, in P1 scalar DOF order.
    pub omega_prev: Vec<f64>,
    /// BDF state timestamp shared with the mean-flow state.
    pub time: f64,
    /// BDF step number shared with the mean-flow state.
    pub step: usize,
}

/// Diagnostics for one P1 k--omega history transfer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KOmegaBdfStateTransferReport {
    /// Number of source P1 scalar DOFs per transported field.
    pub source_scalar_dof_count: usize,
    /// Number of target P1 scalar DOFs per transported field.
    pub target_scalar_dof_count: usize,
    /// Minimum k value over both transferred history levels.
    pub minimum_k: f64,
    /// Minimum omega value over both transferred history levels.
    pub minimum_omega: f64,
}

/// Complete transferred P1 k--omega history state for one target mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaBdfStateTransfer {
    /// Target current and previous k/omega fields.
    pub state: KOmegaBdfState,
    /// Validation diagnostics for the transfer.
    pub report: KOmegaBdfStateTransferReport,
}

/// Result of applying the exact transpose of a P1 k--omega history transfer.
///
/// Adjoint values are signed dual values, so unlike a forward restart state
/// they are checked for finiteness but not for `k`/`omega` positivity.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaBdfStateAdjointTransfer {
    /// Source-mesh current and previous k--omega adjoints.
    pub state: KOmegaBdfState,
    /// Source and target scalar DOF counts.
    pub source_scalar_dof_count: usize,
    pub target_scalar_dof_count: usize,
}

/// Diagnostics for a multi-stage transported-history handoff.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaBdfStateTransferChainReport {
    /// Number of source-to-target remesh transitions.
    pub stage_count: usize,
    /// Per-transition diagnostics in chronological order.
    pub stages: Vec<KOmegaBdfStateTransferReport>,
    /// Target mean-flow and scalar operators must be reassembled.
    pub requires_operator_reassembly: bool,
    /// Stable provenance for the checked handoff contract.
    pub provenance: &'static str,
}

/// Transferred k--omega history states in mesh-chain order.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaBdfStateTransferChain {
    /// One state per topology-changing transition.
    pub states: Vec<KOmegaBdfState>,
    /// Chain-level handoff diagnostics.
    pub report: KOmegaBdfStateTransferChainReport,
}

/// Diagnostics for a reverse multi-stage k--omega state-adjoint transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaBdfStateAdjointTransferChainReport {
    /// Number of source-to-target remesh transitions traversed in reverse.
    pub stage_count: usize,
    /// Per-transition source/target scalar DOF counts in reverse order.
    pub stages: Vec<(usize, usize)>,
    /// Stable provenance for the exact transpose contract.
    pub provenance: &'static str,
}

/// P1 k--omega BDF adjoints in reverse mesh-chain order.
#[derive(Debug, Clone, PartialEq)]
pub struct KOmegaBdfStateAdjointTransferChain {
    /// One adjoint state per transition, ending at the original source mesh.
    pub states: Vec<KOmegaBdfState>,
    /// Chain-level transpose diagnostics.
    pub report: KOmegaBdfStateAdjointTransferChainReport,
}

/// Transfer a resolved P1 CFD state from one oriented tetrahedral mesh to
/// another.
///
/// The meshes may have different node counts and element connectivity.  FEM
/// point-in-element interpolation handles the topology change, while the
/// conservative scalar transfer applies a constant correction so the nodal
/// lumped integral is preserved.  The source and target must cover the same
/// positive-volume domain.  This function returns only a complete state; any
/// invalid topology, uncovered target node, malformed source data, or failed
/// conservation check returns an error.
pub fn transfer_p1_cfd_state(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_velocity_m_s: &[[f64; 3]],
    source_pressure_pa: &[f64],
) -> Result<CfdStateTransfer, CfdStateTransferError> {
    let source_orientation = validate_cfd_transfer_mesh(source_mesh, "source")?;
    let target_orientation = validate_cfd_transfer_mesh(target_mesh, "target")?;
    if source_orientation != target_orientation {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "target",
            reason: "source and target tetrahedral orientation conventions differ".into(),
        });
    }

    if source_velocity_m_s.len() != source_mesh.num_nodes() {
        return Err(CfdStateTransferError::SourceVelocityCount {
            expected: source_mesh.num_nodes(),
            actual: source_velocity_m_s.len(),
        });
    }
    if source_pressure_pa.len() != source_mesh.num_nodes() {
        return Err(CfdStateTransferError::SourcePressureCount {
            expected: source_mesh.num_nodes(),
            actual: source_pressure_pa.len(),
        });
    }
    for (node, velocity) in source_velocity_m_s.iter().enumerate() {
        if let Some(component) = velocity.iter().position(|value| !value.is_finite()) {
            return Err(CfdStateTransferError::NonFiniteSourceVelocity { node, component });
        }
    }
    if let Some(node) = source_pressure_pa
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(CfdStateTransferError::NonFiniteSourcePressure { node });
    }

    // Build the locator once so all fields use identical shared-boundary
    // ownership and fail at the same uncovered target node.
    let interpolation_data =
        crate::mesh::create_interpolation_data_checked(target_mesh, source_mesh)?;

    let mut velocity_components = Vec::with_capacity(3);
    let mut max_velocity_correction_m_s: f64 = 0.0;
    for component in 0..3 {
        let source_component: Vec<f64> = source_velocity_m_s
            .iter()
            .map(|velocity| velocity[component])
            .collect();
        let ordinary = crate::mesh::interpolate_field_checked(
            &interpolation_data,
            source_mesh,
            target_mesh,
            &source_component,
        )?;
        let conservative = crate::mesh::interpolate_field_conservative_checked(
            &interpolation_data,
            source_mesh,
            target_mesh,
            &source_component,
        )?;
        max_velocity_correction_m_s = max_velocity_correction_m_s.max(
            ordinary
                .iter()
                .zip(&conservative)
                .map(|(left, right)| (right - left).abs())
                .fold(0.0, f64::max),
        );
        velocity_components.push(conservative);
    }

    let ordinary_pressure = crate::mesh::interpolate_field_checked(
        &interpolation_data,
        source_mesh,
        target_mesh,
        source_pressure_pa,
    )?;
    let pressure = crate::mesh::interpolate_field_conservative_checked(
        &interpolation_data,
        source_mesh,
        target_mesh,
        source_pressure_pa,
    )?;
    let max_pressure_correction_pa = ordinary_pressure
        .iter()
        .zip(&pressure)
        .map(|(left, right)| (right - left).abs())
        .fold(0.0, f64::max);

    let mut velocity_m_s = vec![[0.0; 3]; target_mesh.num_nodes()];
    for (node, velocity) in velocity_m_s.iter_mut().enumerate() {
        for component in 0..3 {
            velocity[component] = velocity_components[component][node];
        }
    }

    Ok(CfdStateTransfer {
        velocity_m_s,
        pressure_pa: pressure,
        report: CfdStateTransferReport {
            source_node_count: source_mesh.num_nodes(),
            target_node_count: target_mesh.num_nodes(),
            source_element_count: source_mesh.num_elements(),
            target_element_count: target_mesh.num_elements(),
            source_volume_m3: mesh_volume(source_mesh),
            target_volume_m3: mesh_volume(target_mesh),
            max_velocity_correction_m_s,
            max_pressure_correction_pa,
        },
    })
}

/// Transfer the transient solver's P2-velocity/P1-pressure state between
/// topology-changing affine tetrahedral meshes.
///
/// The source and target meshes use vertex-only tetrahedral geometry; the FEM
/// H1 spaces construct the complete P2 edge DOF layout.  Each interleaved
/// velocity component is transferred as an independent scalar P2 H1 field,
/// then interleaved again for `VectorH1Space`.  Pressure is transferred as a
/// scalar P1 H1 field.  Unlike the P1 nodal restart contract above, this path
/// does not apply a constant nodal correction: a correction would alter valid
/// higher-order DOFs and would not represent a generally conservative H1
/// projection.  A complete state is returned only after every component has
/// passed checked interpolation.
pub fn transfer_p2p1_state(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_velocity_dofs: &[f64],
    source_pressure_dofs: &[f64],
) -> Result<P2P1StateTransfer, CfdStateTransferError> {
    let source_orientation = validate_cfd_transfer_mesh(source_mesh, "source")?;
    let target_orientation = validate_cfd_transfer_mesh(target_mesh, "target")?;
    if source_orientation != target_orientation {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "target",
            reason: "source and target tetrahedral orientation conventions differ".into(),
        });
    }
    let source_volume = mesh_volume(source_mesh);
    let target_volume = mesh_volume(target_mesh);
    let volume_tolerance = 1.0e-10 * source_volume.max(target_volume);
    if (source_volume - target_volume).abs() > volume_tolerance {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "target",
            reason: format!(
                "source and target domain volumes differ ({source_volume:.16e} vs {target_volume:.16e})"
            ),
        });
    }

    let source_scalar_velocity_space = H1Space::new(PolynomialDegree::P2);
    let target_scalar_velocity_space = H1Space::new(PolynomialDegree::P2);
    let source_pressure_space = H1Space::new(PolynomialDegree::P1);
    let target_pressure_space = H1Space::new(PolynomialDegree::P1);
    let source_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let target_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);

    let source_velocity_dof_count = source_velocity_space.num_global_dofs(source_mesh);
    if source_velocity_dofs.len() != source_velocity_dof_count {
        return Err(CfdStateTransferError::SourceVelocityDofCount {
            expected: source_velocity_dof_count,
            actual: source_velocity_dofs.len(),
        });
    }
    let source_pressure_dof_count = source_pressure_space.num_global_dofs(source_mesh);
    if source_pressure_dofs.len() != source_pressure_dof_count {
        return Err(CfdStateTransferError::SourcePressureDofCount {
            expected: source_pressure_dof_count,
            actual: source_pressure_dofs.len(),
        });
    }
    if let Some(dof) = source_velocity_dofs
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(CfdStateTransferError::NonFiniteSourceVelocityDof { dof });
    }
    if let Some(dof) = source_pressure_dofs
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(CfdStateTransferError::NonFiniteSourcePressureDof { dof });
    }

    let mut transferred_components = Vec::with_capacity(3);
    for component in 0..3 {
        let source_component = source_velocity_dofs
            .as_chunks::<3>()
            .0
            .iter()
            .map(|dofs| dofs[component])
            .collect::<Vec<_>>();
        transferred_components.push(crate::mesh::transfer_h1_state_checked(
            source_mesh,
            target_mesh,
            &source_scalar_velocity_space,
            &target_scalar_velocity_space,
            &source_component,
        )?);
    }
    let pressure_dofs = crate::mesh::transfer_h1_state_checked(
        source_mesh,
        target_mesh,
        &source_pressure_space,
        &target_pressure_space,
        source_pressure_dofs,
    )?;

    let target_scalar_velocity_dof_count =
        target_scalar_velocity_space.num_global_dofs(target_mesh);
    let target_velocity_dof_count = target_velocity_space.num_global_dofs(target_mesh);
    let mut velocity_dofs = Vec::with_capacity(target_velocity_dof_count);
    for scalar_dof in 0..target_scalar_velocity_dof_count {
        for component in &transferred_components {
            velocity_dofs.push(component[scalar_dof]);
        }
    }

    Ok(P2P1StateTransfer {
        velocity_dofs,
        pressure_dofs,
        report: P2P1StateTransferReport {
            source_node_count: source_mesh.num_nodes(),
            target_node_count: target_mesh.num_nodes(),
            source_element_count: source_mesh.num_elements(),
            target_element_count: target_mesh.num_elements(),
            source_velocity_dof_count,
            target_velocity_dof_count,
            source_pressure_dof_count,
            target_pressure_dof_count: target_pressure_space.num_global_dofs(target_mesh),
        },
    })
}

/// Transfer both velocity levels of a transient BDF state across a changed
/// tetrahedral topology.
///
/// `BdfState` stores the current velocity and the previous velocity required
/// by BDF2, plus the current pressure and time metadata.  The two velocity
/// levels are transferred independently with the checked P2/P1 field contract
/// above; the target solver must reassemble its BDF matrices and history
/// Jacobians for the target topology.  No operator, residual, or turbulence
/// history is copied across the mesh change.
/// Apply the exact transpose of a P2/P1 state transfer.
///
/// The returned values live on `source_mesh` and are suitable for propagating
/// a state adjoint through one topology-changing restart. The operation is
/// the transpose of the checked interpolation map, not a derivative of the
/// remesher or of the discrete topology selection.
pub fn transfer_p2p1_state_adjoint(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    target_velocity_adjoint_dofs: &[f64],
    target_pressure_adjoint_dofs: &[f64],
) -> Result<P2P1StateAdjointTransfer, CfdStateTransferError> {
    let source_orientation = validate_cfd_transfer_mesh(source_mesh, "source")?;
    let target_orientation = validate_cfd_transfer_mesh(target_mesh, "target")?;
    if source_orientation != target_orientation {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "target",
            reason: "source and target tetrahedral orientation conventions differ".into(),
        });
    }
    let source_volume = mesh_volume(source_mesh);
    let target_volume = mesh_volume(target_mesh);
    let volume_tolerance = 1.0e-10 * source_volume.max(target_volume);
    if (source_volume - target_volume).abs() > volume_tolerance {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "target",
            reason: format!(
                "source and target domain volumes differ ({source_volume:.16e} vs {target_volume:.16e})"
            ),
        });
    }

    let source_scalar_velocity_space = H1Space::new(PolynomialDegree::P2);
    let target_scalar_velocity_space = H1Space::new(PolynomialDegree::P2);
    let source_pressure_space = H1Space::new(PolynomialDegree::P1);
    let target_pressure_space = H1Space::new(PolynomialDegree::P1);
    let source_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let target_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);

    let source_velocity_dof_count = source_velocity_space.num_global_dofs(source_mesh);
    let target_velocity_dof_count = target_velocity_space.num_global_dofs(target_mesh);
    if target_velocity_adjoint_dofs.len() != target_velocity_dof_count {
        return Err(CfdStateTransferError::TargetVelocityDofCount {
            expected: target_velocity_dof_count,
            actual: target_velocity_adjoint_dofs.len(),
        });
    }
    let source_pressure_dof_count = source_pressure_space.num_global_dofs(source_mesh);
    let target_pressure_dof_count = target_pressure_space.num_global_dofs(target_mesh);
    if target_pressure_adjoint_dofs.len() != target_pressure_dof_count {
        return Err(CfdStateTransferError::TargetPressureDofCount {
            expected: target_pressure_dof_count,
            actual: target_pressure_adjoint_dofs.len(),
        });
    }
    if let Some(dof) = target_velocity_adjoint_dofs
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(CfdStateTransferError::NonFiniteTargetVelocityDof { dof });
    }
    if let Some(dof) = target_pressure_adjoint_dofs
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(CfdStateTransferError::NonFiniteTargetPressureDof { dof });
    }

    let mut source_components = Vec::with_capacity(3);
    for component in 0..3 {
        let component_values: Vec<f64> = (0..target_scalar_velocity_space
            .num_global_dofs(target_mesh))
            .map(|scalar_dof| target_velocity_adjoint_dofs[3 * scalar_dof + component])
            .collect();
        source_components.push(
            crate::mesh::transfer_h1_state_adjoint_checked(
                source_mesh,
                target_mesh,
                &source_scalar_velocity_space,
                &target_scalar_velocity_space,
                &component_values,
            )
            .map_err(CfdStateTransferError::from)?,
        );
    }
    let source_pressure_dofs = crate::mesh::transfer_h1_state_adjoint_checked(
        source_mesh,
        target_mesh,
        &source_pressure_space,
        &target_pressure_space,
        target_pressure_adjoint_dofs,
    )
    .map_err(CfdStateTransferError::from)?;

    let mut source_velocity_dofs = vec![0.0; source_velocity_dof_count];
    for scalar_dof in 0..source_scalar_velocity_space.num_global_dofs(source_mesh) {
        source_velocity_dofs[3 * scalar_dof] = source_components[0][scalar_dof];
        source_velocity_dofs[3 * scalar_dof + 1] = source_components[1][scalar_dof];
        source_velocity_dofs[3 * scalar_dof + 2] = source_components[2][scalar_dof];
    }

    Ok(P2P1StateAdjointTransfer {
        velocity_dofs: source_velocity_dofs,
        pressure_dofs: source_pressure_dofs,
        report: P2P1StateTransferReport {
            source_node_count: source_mesh.num_nodes(),
            target_node_count: target_mesh.num_nodes(),
            source_element_count: source_mesh.num_elements(),
            target_element_count: target_mesh.num_elements(),
            source_velocity_dof_count,
            target_velocity_dof_count,
            source_pressure_dof_count,
            target_pressure_dof_count,
        },
    })
}

/// Apply the exact transpose of the P2/P1 BDF state transfer.
pub fn transfer_p2p1_bdf_state_adjoint(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    target_adjoint: &BdfState,
) -> Result<BdfState, CfdStateTransferError> {
    let target_pressure_space = H1Space::new(PolynomialDegree::P1);
    let target_pressure_dof_count = target_pressure_space.num_global_dofs(target_mesh);
    let zero_pressure_adjoint = vec![0.0; target_pressure_dof_count];
    let current = transfer_p2p1_state_adjoint(
        source_mesh,
        target_mesh,
        &target_adjoint.velocity,
        &target_adjoint.pressure,
    )?;
    let previous_velocity = transfer_p2p1_state_adjoint(
        source_mesh,
        target_mesh,
        &target_adjoint.velocity_prev,
        &zero_pressure_adjoint,
    )?
    .velocity_dofs;

    Ok(BdfState {
        velocity: current.velocity_dofs,
        velocity_prev: previous_velocity,
        pressure: current.pressure_dofs,
        time: target_adjoint.time,
        step: target_adjoint.step,
    })
}

/// Apply BDF state-adjoint transfer through a mesh chain in reverse order.
pub fn transfer_p2p1_bdf_state_adjoint_chain(
    mesh_chain: &[&Mesh],
    terminal_adjoint: &BdfState,
    provenance: &'static str,
) -> Result<P2P1StateAdjointTransferChain, CfdStateTransferError> {
    if mesh_chain.len() < 2 {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "mesh chain",
            reason: "a topology-changing chain requires at least one transition".into(),
        });
    }

    let mut states = Vec::with_capacity(mesh_chain.len() - 1);
    let mut stages = Vec::with_capacity(mesh_chain.len() - 1);
    let mut state = terminal_adjoint.clone();
    for index in (1..mesh_chain.len()).rev() {
        let transferred = transfer_p2p1_state_adjoint(
            mesh_chain[index - 1],
            mesh_chain[index],
            &state.velocity,
            &state.pressure,
        )?;
        stages.push(transferred.report);
        let target_pressure_space = H1Space::new(PolynomialDegree::P1);
        let zero_pressure_adjoint =
            vec![0.0; target_pressure_space.num_global_dofs(mesh_chain[index])];
        let previous_velocity = transfer_p2p1_state_adjoint(
            mesh_chain[index - 1],
            mesh_chain[index],
            &state.velocity_prev,
            &zero_pressure_adjoint,
        )?;
        state = BdfState {
            velocity: transferred.velocity_dofs,
            velocity_prev: previous_velocity.velocity_dofs,
            pressure: transferred.pressure_dofs,
            time: state.time,
            step: state.step,
        };
        states.push(state.clone());
    }

    Ok(P2P1StateAdjointTransferChain {
        states,
        report: P2P1StateAdjointTransferChainReport {
            stage_count: mesh_chain.len() - 1,
            stages,
            provenance,
        },
    })
}

/// Transfer both velocity levels and the pressure of a transient BDF state
/// across changed tetrahedral topology.
pub fn transfer_p2p1_bdf_state(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_state: &BdfState,
) -> Result<BdfState, CfdStateTransferError> {
    if !source_state.time.is_finite() {
        return Err(CfdStateTransferError::NonFiniteSourceTime);
    }

    let current = transfer_p2p1_state(
        source_mesh,
        target_mesh,
        &source_state.velocity,
        &source_state.pressure,
    )?;
    let previous_velocity = transfer_p2p1_state(
        source_mesh,
        target_mesh,
        &source_state.velocity_prev,
        &source_state.pressure,
    )?
    .velocity_dofs;

    Ok(BdfState {
        velocity: current.velocity_dofs,
        velocity_prev: previous_velocity,
        pressure: current.pressure_dofs,
        time: source_state.time,
        step: source_state.step,
    })
}

/// Transfer a BDF state through an ordered topology-changing mesh chain.
///
/// `mesh_chain[0]` owns `source_state`; every following mesh is a new target
/// topology. Each stage transfers both retained velocity levels and the P1
/// pressure state. No BDF matrix, sparsity pattern, turbulence history, or
/// adjoint artifact is copied across a transition: callers must assemble the
/// target operators before advancing the next step.
pub fn transfer_p2p1_bdf_state_chain(
    mesh_chain: &[&Mesh],
    source_state: &BdfState,
    provenance: &'static str,
) -> Result<P2P1StateTransferChain, CfdStateTransferError> {
    if mesh_chain.len() < 2 {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "mesh chain",
            reason: "a topology-changing chain requires at least one transition".into(),
        });
    }

    let mut states = Vec::with_capacity(mesh_chain.len() - 1);
    let mut stages = Vec::with_capacity(mesh_chain.len() - 1);
    let mut state = source_state.clone();
    for transition in mesh_chain.windows(2) {
        let transferred = transfer_p2p1_state(
            transition[0],
            transition[1],
            &state.velocity,
            &state.pressure,
        )?;
        let previous_velocity = transfer_p2p1_state(
            transition[0],
            transition[1],
            &state.velocity_prev,
            &state.pressure,
        )?
        .velocity_dofs;
        stages.push(transferred.report);
        state = BdfState {
            velocity: transferred.velocity_dofs,
            velocity_prev: previous_velocity,
            pressure: transferred.pressure_dofs,
            time: state.time,
            step: state.step,
        };
        states.push(state.clone());
    }

    Ok(P2P1StateTransferChain {
        states,
        report: P2P1StateTransferChainReport {
            stage_count: mesh_chain.len() - 1,
            stages,
            requires_operator_reassembly: true,
            provenance,
        },
    })
}

/// Transfer a P1 k--omega BDF history state between compatible tetrahedral
/// topologies.
///
/// Both current and previous levels are transferred with the checked FEM H1
/// inverse map. The scalar fields are not conservatively corrected: a P1
/// interpolation of admissible nodal values preserves the physical scalar
/// domain, whereas a mass correction could create a negative k or omega.
/// Operators, boundary treatment, and turbulence-model coupling remain the
/// responsibility of the target solve.
/// Apply the exact transpose of a P1 k--omega BDF history transfer.
///
/// This propagates signed adjoint values from the target mesh to the source
/// mesh. Positivity is a forward-state invariant and is therefore not
/// imposed on adjoint values.
pub fn transfer_komega_bdf_state_adjoint(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    target_adjoint: &KOmegaBdfState,
) -> Result<KOmegaBdfStateAdjointTransfer, KOmegaStateTransferError> {
    validate_komega_mesh_pair(source_mesh, target_mesh)?;
    let source_space = H1Space::new(PolynomialDegree::P1);
    let target_space = H1Space::new(PolynomialDegree::P1);
    let source_dof_count = source_space.num_global_dofs(source_mesh);
    let target_dof_count = target_space.num_global_dofs(target_mesh);

    let transfer_field = |field: &'static str, values: &[f64]| {
        validate_komega_adjoint_field(field, values, target_dof_count)?;
        crate::mesh::transfer_h1_state_adjoint_checked(
            source_mesh,
            target_mesh,
            &source_space,
            &target_space,
            values,
        )
        .map_err(CfdStateTransferError::from)
        .map_err(KOmegaStateTransferError::from)
    };

    Ok(KOmegaBdfStateAdjointTransfer {
        state: KOmegaBdfState {
            k: transfer_field("k", &target_adjoint.k)?,
            k_prev: transfer_field("k", &target_adjoint.k_prev)?,
            omega: transfer_field("omega", &target_adjoint.omega)?,
            omega_prev: transfer_field("omega", &target_adjoint.omega_prev)?,
            time: target_adjoint.time,
            step: target_adjoint.step,
        },
        source_scalar_dof_count: source_dof_count,
        target_scalar_dof_count: target_dof_count,
    })
}

/// Apply k--omega BDF state-adjoint transfer through a mesh chain in reverse.
pub fn transfer_komega_bdf_state_adjoint_chain(
    mesh_chain: &[&Mesh],
    terminal_adjoint: &KOmegaBdfState,
    provenance: &'static str,
) -> Result<KOmegaBdfStateAdjointTransferChain, KOmegaStateTransferError> {
    if mesh_chain.len() < 2 {
        return Err(KOmegaStateTransferError::StateTransfer(
            CfdStateTransferError::InvalidMesh {
                role: "mesh chain",
                reason: "a topology-changing chain requires at least one transition".into(),
            },
        ));
    }

    let mut states = Vec::with_capacity(mesh_chain.len() - 1);
    let mut stages = Vec::with_capacity(mesh_chain.len() - 1);
    let mut state = terminal_adjoint.clone();
    for index in (1..mesh_chain.len()).rev() {
        let transferred =
            transfer_komega_bdf_state_adjoint(mesh_chain[index - 1], mesh_chain[index], &state)?;
        stages.push((
            transferred.source_scalar_dof_count,
            transferred.target_scalar_dof_count,
        ));
        state = transferred.state;
        states.push(state.clone());
    }

    Ok(KOmegaBdfStateAdjointTransferChain {
        states,
        report: KOmegaBdfStateAdjointTransferChainReport {
            stage_count: mesh_chain.len() - 1,
            stages,
            provenance,
        },
    })
}

/// Transfer P1 k--omega BDF history across compatible tetrahedral topologies.
pub fn transfer_komega_bdf_state(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_state: &KOmegaBdfState,
) -> Result<KOmegaBdfStateTransfer, KOmegaStateTransferError> {
    if !source_state.time.is_finite() {
        return Err(KOmegaStateTransferError::NonFiniteSourceTime);
    }

    validate_komega_mesh_pair(source_mesh, target_mesh)?;

    let source_space = H1Space::new(PolynomialDegree::P1);
    let target_space = H1Space::new(PolynomialDegree::P1);
    let source_dof_count = source_space.num_global_dofs(source_mesh);
    let target_dof_count = target_space.num_global_dofs(target_mesh);

    let transfer_field = |field: &'static str, values: &[f64]| {
        validate_komega_field(field, values, source_dof_count)?;
        let transferred = crate::mesh::transfer_h1_state_checked(
            source_mesh,
            target_mesh,
            &source_space,
            &target_space,
            values,
        )
        .map_err(CfdStateTransferError::from)
        .map_err(KOmegaStateTransferError::from)?;
        validate_komega_field(field, &transferred, target_dof_count)?;
        Ok::<Vec<f64>, KOmegaStateTransferError>(transferred)
    };

    let k = transfer_field("k", &source_state.k)?;
    let k_prev = transfer_field("k", &source_state.k_prev)?;
    let omega = transfer_field("omega", &source_state.omega)?;
    let omega_prev = transfer_field("omega", &source_state.omega_prev)?;
    let minimum_k = k
        .iter()
        .chain(&k_prev)
        .copied()
        .fold(f64::INFINITY, f64::min);
    let minimum_omega = omega
        .iter()
        .chain(&omega_prev)
        .copied()
        .fold(f64::INFINITY, f64::min);

    Ok(KOmegaBdfStateTransfer {
        state: KOmegaBdfState {
            k,
            k_prev,
            omega,
            omega_prev,
            time: source_state.time,
            step: source_state.step,
        },
        report: KOmegaBdfStateTransferReport {
            source_scalar_dof_count: source_dof_count,
            target_scalar_dof_count: target_dof_count,
            minimum_k,
            minimum_omega,
        },
    })
}

/// Transfer P1 k--omega history through an ordered topology-changing mesh
/// chain. Each stage transfers all four scalar BDF levels and requires fresh
/// target mean-flow and turbulence operator assembly.
pub fn transfer_komega_bdf_state_chain(
    mesh_chain: &[&Mesh],
    source_state: &KOmegaBdfState,
    provenance: &'static str,
) -> Result<KOmegaBdfStateTransferChain, KOmegaStateTransferError> {
    if mesh_chain.len() < 2 {
        return Err(KOmegaStateTransferError::StateTransfer(
            CfdStateTransferError::InvalidMesh {
                role: "mesh chain",
                reason: "a topology-changing chain requires at least one transition".into(),
            },
        ));
    }

    let mut states = Vec::with_capacity(mesh_chain.len() - 1);
    let mut stages = Vec::with_capacity(mesh_chain.len() - 1);
    let mut state = source_state.clone();
    for transition in mesh_chain.windows(2) {
        let transferred = transfer_komega_bdf_state(transition[0], transition[1], &state)?;
        stages.push(transferred.report);
        state = transferred.state;
        states.push(state.clone());
    }

    Ok(KOmegaBdfStateTransferChain {
        states,
        report: KOmegaBdfStateTransferChainReport {
            stage_count: mesh_chain.len() - 1,
            stages,
            requires_operator_reassembly: true,
            provenance,
        },
    })
}

fn validate_komega_mesh_pair(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
) -> Result<(), KOmegaStateTransferError> {
    let source_orientation = validate_cfd_transfer_mesh(source_mesh, "source")?;
    let target_orientation = validate_cfd_transfer_mesh(target_mesh, "target")?;
    if source_orientation != target_orientation {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "target",
            reason: "source and target tetrahedral orientation conventions differ".into(),
        }
        .into());
    }

    let source_volume = mesh_volume(source_mesh);
    let target_volume = mesh_volume(target_mesh);
    let volume_tolerance = 1.0e-10 * source_volume.max(target_volume);
    if (source_volume - target_volume).abs() > volume_tolerance {
        return Err(CfdStateTransferError::InvalidMesh {
            role: "target",
            reason: format!(
                "source and target domain volumes differ ({source_volume:.16e} vs {target_volume:.16e})"
            ),
        }
        .into());
    }
    Ok(())
}

fn validate_komega_adjoint_field(
    field: &'static str,
    values: &[f64],
    expected_dof_count: usize,
) -> Result<(), KOmegaStateTransferError> {
    if values.len() != expected_dof_count {
        return Err(KOmegaStateTransferError::FieldDofCount {
            field,
            expected: expected_dof_count,
            actual: values.len(),
        });
    }
    if let Some(dof) = values.iter().position(|value| !value.is_finite()) {
        return Err(KOmegaStateTransferError::NonFiniteTargetField { field, dof });
    }
    Ok(())
}

fn validate_komega_field(
    field: &'static str,
    values: &[f64],
    expected_dof_count: usize,
) -> Result<(), KOmegaStateTransferError> {
    if values.len() != expected_dof_count {
        return Err(KOmegaStateTransferError::FieldDofCount {
            field,
            expected: expected_dof_count,
            actual: values.len(),
        });
    }
    for (dof, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(KOmegaStateTransferError::NonFiniteField { field, dof });
        }
        let admissible = match field {
            "k" => value >= 0.0,
            "omega" => value > 0.0,
            _ => false,
        };
        if !admissible {
            return Err(KOmegaStateTransferError::InvalidFieldValue { field, dof, value });
        }
    }
    Ok(())
}

fn validate_cfd_transfer_mesh(
    mesh: &Mesh,
    role: &'static str,
) -> Result<i8, CfdStateTransferError> {
    if mesh.dimension != 3 {
        return Err(CfdStateTransferError::InvalidMesh {
            role,
            reason: format!("expected a 3-D mesh, got dimension {}", mesh.dimension),
        });
    }
    if mesh.nodes.is_empty() || mesh.elements.is_empty() {
        return Err(CfdStateTransferError::InvalidMesh {
            role,
            reason: "mesh must contain nodes and elements".into(),
        });
    }
    if let Err(errors) = mesh.validate_connectivity() {
        return Err(CfdStateTransferError::InvalidMesh {
            role,
            reason: errors
                .first()
                .map(ToString::to_string)
                .unwrap_or_else(|| "connectivity validation failed".into()),
        });
    }

    let mut orientation = None;
    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != ElementType::Tetrahedron
            || element.nodes.len() != ElementType::Tetrahedron.num_vertices()
        {
            return Err(CfdStateTransferError::InvalidMesh {
                role,
                reason: format!(
                    "element {element_index} must be an affine P1 tetrahedron, got {:?} with {} nodes",
                    element.element_type,
                    element.nodes.len()
                ),
            });
        }
        let signed_volume = signed_tetrahedron_volume(mesh, element_index);
        if !signed_volume.is_finite() || signed_volume == 0.0 {
            return Err(CfdStateTransferError::InvalidMesh {
                role,
                reason: format!(
                    "element {element_index} has non-finite or zero signed volume {signed_volume}"
                ),
            });
        }
        let element_orientation = if signed_volume.is_sign_positive() {
            1
        } else {
            -1
        };
        if let Some(expected_orientation) = orientation {
            if expected_orientation != element_orientation {
                return Err(CfdStateTransferError::InvalidMesh {
                    role,
                    reason: format!(
                        "element {element_index} has inconsistent tetrahedral orientation"
                    ),
                });
            }
        } else {
            orientation = Some(element_orientation);
        }
    }
    Ok(orientation.expect("non-empty mesh has at least one element"))
}

fn signed_tetrahedron_volume(mesh: &Mesh, element_index: usize) -> f64 {
    let element = &mesh.elements[element_index];
    let a = &mesh.nodes[element.nodes[0]];
    let b = &mesh.nodes[element.nodes[1]];
    let c = &mesh.nodes[element.nodes[2]];
    let d = &mesh.nodes[element.nodes[3]];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    (ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]))
        / 6.0
}

fn mesh_volume(mesh: &Mesh) -> f64 {
    mesh.elements
        .iter()
        .enumerate()
        .map(|(element, _)| mesh.element_measure(element))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generators::unit_cube_tetrahedra;

    fn affine_p2_velocity(mesh: &Mesh) -> Vec<f64> {
        let space = H1Space::new(PolynomialDegree::P2);
        space
            .dof_coordinates(mesh)
            .expect("P2 DOF coordinates")
            .iter()
            .flat_map(|point| {
                [
                    point[0] + 2.0 * point[1],
                    -point[2] + 0.5,
                    point[0] - point[1] + point[2],
                ]
            })
            .collect()
    }

    fn affine_p1_pressure(mesh: &Mesh) -> Vec<f64> {
        let space = H1Space::new(PolynomialDegree::P1);
        space
            .dof_coordinates(mesh)
            .expect("P1 DOF coordinates")
            .iter()
            .map(|point| 2.0 * point[0] - point[1] + 3.0 * point[2])
            .collect()
    }

    #[test]
    fn constant_p1_vector_state_is_preserved() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let velocity = vec![[1.25, -0.5, 3.0]; source.num_nodes()];
        let pressure = vec![101_325.0; source.num_nodes()];
        let transferred =
            transfer_p1_cfd_state(&source, &target, &velocity, &pressure).expect("transfer");
        assert_eq!(transferred.velocity_m_s.len(), target.num_nodes());
        for value in &transferred.velocity_m_s {
            assert!((value[0] - 1.25).abs() < 1.0e-12);
            assert!((value[1] + 0.5).abs() < 1.0e-12);
            assert!((value[2] - 3.0).abs() < 1.0e-12);
        }
        for value in &transferred.pressure_pa {
            assert!((value - 101_325.0).abs() < 1.0e-9);
        }
        assert!(transferred.report.max_velocity_correction_m_s < 1.0e-12);
        assert!(transferred.report.max_pressure_correction_pa < 1.0e-9);
    }

    #[test]
    fn p2p1_forward_adjoint_dot_product_identity() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let source_velocity = affine_p2_velocity(&source);
        let source_pressure = affine_p1_pressure(&source);
        let forward = transfer_p2p1_state(&source, &target, &source_velocity, &source_pressure)
            .expect("forward transfer");
        // Deterministic target-space test vectors.
        let target_space_p = H1Space::new(PolynomialDegree::P1);
        let n_target_p = target_space_p.num_global_dofs(&target);
        let y_p: Vec<f64> = (0..n_target_p)
            .map(|i| ((i * 37 + 11) % 97) as f64 / 97.0 - 0.5)
            .collect();
        let zero_v =
            vec![0.0; VectorH1Space::new(PolynomialDegree::P2, 3).num_global_dofs(&target)];
        let adjoint =
            transfer_p2p1_state_adjoint(&source, &target, &zero_v, &y_p).expect("adjoint transfer");
        let lhs: f64 = forward
            .pressure_dofs
            .iter()
            .zip(&y_p)
            .map(|(a, b)| a * b)
            .sum();
        // Adjoint pressure lives on the source mesh with the source ordering.
        let rhs: f64 = source_pressure
            .iter()
            .zip(&adjoint.pressure_dofs)
            .map(|(a, b)| a * b)
            .sum();
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        assert!(
            (lhs - rhs).abs() / scale < 1.0e-10,
            "adjoint mismatch {lhs:e} vs {rhs:e}"
        );
    }

    #[test]
    fn bdf_chain_threads_state_and_provenance() {
        let mesh_a = unit_cube_tetrahedra(1);
        let mesh_b = unit_cube_tetrahedra(2);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let state = BdfState {
            velocity: affine_p2_velocity(&mesh_a),
            velocity_prev: affine_p2_velocity(&mesh_a),
            pressure: affine_p1_pressure(&mesh_a),
            time: 0.01,
            step: 3,
        };
        assert_eq!(
            state.velocity.len(),
            velocity_space.num_global_dofs(&mesh_a)
        );
        assert_eq!(
            state.pressure.len(),
            pressure_space.num_global_dofs(&mesh_a)
        );
        let chain = transfer_p2p1_bdf_state_chain(
            &[&mesh_a, &mesh_b],
            &state,
            "test:transfer-provenance:v1",
        )
        .expect("chain transfer");
        assert_eq!(chain.states.len(), 1);
        assert_eq!(chain.report.stage_count, 1);
        assert!(chain.report.requires_operator_reassembly);
        assert_eq!(chain.report.provenance, "test:transfer-provenance:v1");
        assert_eq!(chain.states[0].time, 0.01);
        assert_eq!(chain.states[0].step, 3);
    }

    #[test]
    fn komega_domain_checks_hold_and_adjoint_stays_signed() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let n_source = space.num_global_dofs(&source);
        let state = KOmegaBdfState {
            k: vec![0.5; n_source],
            k_prev: vec![0.0; n_source],
            omega: vec![2.0; n_source],
            omega_prev: vec![3.0; n_source],
            time: 0.02,
            step: 5,
        };
        let transferred = transfer_komega_bdf_state(&source, &target, &state).expect("transfer");
        assert!(transferred.report.minimum_k >= 0.0);
        assert!(transferred.report.minimum_omega > 0.0);
        let mut bad = state.clone();
        bad.k[0] = -1.0;
        assert!(matches!(
            transfer_komega_bdf_state(&source, &target, &bad),
            Err(KOmegaStateTransferError::InvalidFieldValue { .. })
        ));
        // Adjoints are signed duals: negativity must not fail.
        let n_target = space.num_global_dofs(&target);
        let adjoint_in = KOmegaBdfState {
            k: vec![-0.5; n_target],
            k_prev: vec![0.0; n_target],
            omega: vec![-2.0; n_target],
            omega_prev: vec![0.0; n_target],
            time: 0.02,
            step: 5,
        };
        let adjoint = transfer_komega_bdf_state_adjoint(&source, &target, &adjoint_in)
            .expect("signed adjoint transfers");
        assert_eq!(adjoint.state.k.len(), n_source);
    }
}
