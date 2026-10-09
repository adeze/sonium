//! FEM consumer for a canonical MeshLang state-transfer plan.
//!
//! MeshLang owns the entity-level change contract while FEM owns the
//! numerical interpolation kernels.  This module is the checked bridge
//! between those layers: it maps plan actions to the existing nodal and
//! conservative transfer routines and keeps application-defined actions in
//! a caller-owned callback.

use super::{
    InterpolationData, InterpolationError, Mesh, PartitionGhostLayout, RefinementTransferDirection,
    create_interpolation_data_checked, interpolate_field_checked,
    interpolate_field_conservative_checked, transfer_h1_state_with_refinement_history_checked,
    transfer_hcurl_mixed_triangle_state_checked, transfer_hcurl_order1_state_checked,
    transfer_hcurl_order1_state_with_refinement_chain_checked,
    transfer_hcurl_order2_triangle_state_checked,
    transfer_hcurl_order2_triangle_state_with_refinement_chain_checked,
    transfer_hdiv_rt0_state_checked, transfer_hdiv_rt0_state_with_refinement_chain_checked,
    transfer_hdiv_rt1_state_checked, transfer_hdiv_rt1_state_with_refinement_chain_checked,
    validate_refinement_transfer_chain_checked,
};
use crate::basis::PolynomialDegree;
use crate::spaces::{FunctionSpace, H1Space, HcurlSpace, HdivOrder, HdivSpace};
use meshlang_ast::{
    EntityId, IrStateTransferAction, IrStateTransferPlan, IrStateTransferPlanChain,
};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Failure while applying a MeshLang plan to FEM nodal data.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MeshlangTransferError {
    /// The plan was not in canonical entity order.
    #[error("invalid MeshLang state-transfer plan: {0}")]
    InvalidPlan(String),
    /// A plan action needs a source entity that was not mapped by the caller.
    #[error("source entity {0:?} has no mapped node")]
    MissingSourceEntity(EntityId),
    /// A numerical transfer needs a target entity that was not mapped.
    #[error("target entity {0:?} has no mapped node")]
    MissingTargetEntity(EntityId),
    /// A caller-supplied source node is outside the source mesh.
    #[error("source entity {entity:?} maps to invalid node {node}")]
    InvalidSourceNode { entity: EntityId, node: usize },
    /// A caller-supplied target node is outside the target mesh.
    #[error("target entity {entity:?} maps to invalid node {node}")]
    InvalidTargetNode { entity: EntityId, node: usize },
    /// An intermediate chain target map does not cover every nodal value.
    #[error("target entity map does not cover mesh node {node}")]
    IncompleteTargetNodeMap { node: usize },
    /// The nodal source vector has the wrong length.
    #[error("source values have length {actual}, expected {expected}")]
    SourceValueCount { expected: usize, actual: usize },
    /// A source value copied by identity is not finite.
    #[error("source value for entity {entity:?} is not finite")]
    NonFiniteSource { entity: EntityId },
    /// A callback or numerical kernel produced a non-finite value.
    #[error("transferred value for entity {entity:?} is not finite")]
    NonFiniteResult { entity: EntityId },
    /// Existing FEM interpolation rejected the mesh or transfer metadata.
    #[error("FEM interpolation failed: {0}")]
    Interpolation(#[from] InterpolationError),
    /// A partition layout references a node outside its mesh.
    #[error("{side} partition layout references node {node}, but mesh has {nodes} nodes")]
    InvalidPartitionNode {
        side: &'static str,
        node: usize,
        nodes: usize,
    },
    /// A partition layout repeats a node or lists it as both owned and ghost.
    #[error("{side} partition layout repeats node {node}")]
    DuplicatePartitionNode { side: &'static str, node: usize },
    /// A caller-owned owner/ghost exchange failed before its result was committed.
    #[error("partition owner/ghost exchange failed: {0}")]
    PartitionExchange(String),
    /// An exchange returned an entity that was not requested as a ghost.
    #[error("partition exchange returned unknown ghost entity {0:?}")]
    UnknownPartitionGhostEntity(EntityId),
    /// An exchange omitted one of the requested ghost entities.
    #[error("partition exchange omitted ghost entity {0:?}")]
    MissingPartitionGhostEntity(EntityId),
    /// An exchange returned a non-finite ghost value.
    #[error("partition exchange returned non-finite value for ghost entity {entity:?}")]
    NonFinitePartitionGhost { entity: EntityId },
    /// An application-defined action was rejected by its callback.
    #[error("application-defined transfer for entity {entity:?} failed: {error}")]
    ApplicationDefined { entity: EntityId, error: String },
    /// The canonical action is not meaningful for the selected FEM field.
    #[error("state-transfer action {action:?} is unsupported for entity {entity:?}")]
    UnsupportedAction {
        entity: EntityId,
        action: IrStateTransferAction,
    },
    /// The RT1 bridge was called with a space configured at another order.
    #[error(
        "RT1 H(div) state transfer requires RT1 source and target spaces (source={source_order:?}, target={target_order:?})"
    )]
    UnsupportedHdivOrder {
        source_order: HdivOrder,
        target_order: HdivOrder,
    },
    /// Higher-order H1 bridges remain explicit about the polynomial degree
    /// they transfer. P2 and P3 use checked interpolation kernels; mixed-
    /// degree calls fail before numerical work begins.
    #[error(
        "MeshLang H1 transfer requires {expected_degree:?} source and target spaces (source={source_degree:?}, target={target_degree:?})"
    )]
    UnsupportedH1Degree {
        expected_degree: PolynomialDegree,
        source_degree: PolynomialDegree,
        target_degree: PolynomialDegree,
    },
    /// H1 space topology must expose one global DOF per active mesh-node
    /// range so the canonical entity-to-DOF map remains unambiguous. A
    /// coarsened mesh may retain orphaned nodes until explicit compaction.
    #[error("{side} H1 space exposes {actual} global DOFs, expected {expected} active mesh nodes")]
    InvalidH1DofLayout {
        side: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A checked H1 space rejected the supplied mesh before any callbacks.
    #[error("{side} H1 space validation failed: {error}")]
    InvalidH1Space { side: &'static str, error: String },
}

fn validate_finite_mesh(mesh: &Mesh, role: &'static str) -> Result<(), MeshlangTransferError> {
    if mesh
        .nodes
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
    {
        return Err(MeshlangTransferError::Interpolation(
            InterpolationError::InvalidMesh {
                mesh: role,
                reason: "mesh coordinates must be finite".into(),
            },
        ));
    }
    Ok(())
}

/// Apply an entity-level MeshLang plan to a nodal FEM field.
///
/// `source_entities` and `target_entities` are explicit because canonical
/// entity IDs are not required to be mesh node indices.  Interpolation and
/// conservative actions use the FEM kernels once per plan, while preserve,
/// initialize, and application-defined actions retain their plan semantics.
/// All plan and mesh checks happen before the returned map is exposed.
/// `initialize_or_application` is called only for `Initialize` and
/// `ApplicationDefined`; preserve and numerical actions never invoke it.
pub fn apply_nodal_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    interpolation_data: &InterpolationData,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if source_values.len() != source_mesh.num_nodes() {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_mesh.num_nodes(),
            actual: source_values.len(),
        });
    }

    for (&entity, &node) in source_entities {
        if node >= source_mesh.num_nodes() {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node });
        }
    }
    for (&entity, &node) in target_entities {
        if node >= target_mesh.num_nodes() {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node });
        }
    }

    // Validate every source value before any callback can run.  This keeps
    // preserve/application-defined actions consistent with FEM's checked
    // interpolation routines.
    for entry in &plan.entries {
        if matches!(
            entry.action,
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined
        ) {
            if let Some(&node) = source_entities.get(&entry.entity) {
                let value = source_values[node];
                if !value.is_finite() {
                    return Err(MeshlangTransferError::NonFiniteSource {
                        entity: entry.entity,
                    });
                }
            } else if entry.action == IrStateTransferAction::Preserve {
                return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
            }
        }
        if matches!(
            entry.action,
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
        ) && !target_entities.contains_key(&entry.entity)
        {
            return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
        }
    }

    let has_interpolation = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate);
    let has_conservative = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Conservative);
    let interpolated = has_interpolation.then(|| {
        interpolate_field_checked(interpolation_data, source_mesh, target_mesh, source_values)
    });
    let conservative = has_conservative.then(|| {
        interpolate_field_conservative_checked(
            interpolation_data,
            source_mesh,
            target_mesh,
            source_values,
        )
    });
    let interpolated = interpolated.transpose()?;
    let conservative = conservative.transpose()?;

    // Resolve all non-callback values before invoking user code.  A callback
    // failure therefore cannot leave a partially numerical transfer hidden in
    // the result map.
    let mut resolved = Vec::with_capacity(plan.entries.len());
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let node = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                Some(source_values[node])
            }
            IrStateTransferAction::Interpolate => {
                let node = target_entities[&entry.entity];
                Some(interpolated.as_ref().expect("validated interpolation path")[node])
            }
            IrStateTransferAction::Conservative => {
                let node = target_entities[&entry.entity];
                Some(conservative.as_ref().expect("validated conservative path")[node])
            }
            IrStateTransferAction::Initialize => None,
            IrStateTransferAction::ApplicationDefined => source_entities
                .get(&entry.entity)
                .map(|&node| source_values[node]),
            IrStateTransferAction::Drop => continue,
        };
        resolved.push((entry.entity, entry.action, value));
    }

    let mut output = BTreeMap::new();
    for (entity, action, value) in resolved {
        let value = match action {
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(entity, action, value).map_err(|error| {
                    MeshlangTransferError::ApplicationDefined {
                        entity,
                        error: error.to_string(),
                    }
                })?
            }
            IrStateTransferAction::Preserve
            | IrStateTransferAction::Interpolate
            | IrStateTransferAction::Conservative => value.expect("numeric action has a value"),
            IrStateTransferAction::Drop => unreachable!("drop entries are removed above"),
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult { entity });
        }
        output.insert(entity, value);
    }
    Ok(output)
}

/// Apply a canonical MeshLang plan to a cubic (P3/Q3) scalar H1 field.
///
/// The P3 shape-function evaluator is used for every local degree of freedom;
/// callers still own the geometric quality contract for curved/isoparametric
/// source meshes. Space validation and complete DOF layout admission happen
/// before any plan callback.
pub fn apply_h1_p3_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    validate_h1_degree_spaces(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        PolynomialDegree::P3,
    )?;
    let interpolation_data = create_interpolation_data_checked(target_mesh, source_mesh)?;
    apply_nodal_state_transfer(
        plan,
        source_mesh,
        target_mesh,
        &interpolation_data,
        source_values,
        source_entities,
        target_entities,
        initialize_or_application,
    )
}

/// Apply a canonical MeshLang plan across one checked adaptive H1 lineage step.
///
/// FEM validates the parent/child history before interpolating the nodal state;
/// MeshLang retains ownership of entity actions and callback semantics.  H1
/// adaptive transfer currently has no scalar conservative kernel, so a
/// `Conservative` plan entry is rejected before any numerical work or callback.
/// Apply canonical MeshLang plan cubic H1 field across one checked adaptive
/// refinement or coarsening step. The parent/child topology remains owned by
/// the mesher history while FEM validates the P3 interpolation contract.
#[expect(
    clippy::too_many_arguments,
    reason = "checked state-transfer entry point threading plan, meshes, callbacks and options; arity is the stable checked-call contract"
)]
pub fn apply_h1_p3_state_transfer_with_refinement_history<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    validate_h1_degree_spaces(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        PolynomialDegree::P3,
    )?;
    apply_nodal_state_transfer_with_refinement_history(
        plan,
        source_mesh,
        target_mesh,
        source_values,
        history,
        direction,
        source_entities,
        target_entities,
        initialize_or_application,
    )
}

/// Apply canonical MeshLang plan chain across checked adaptive P3 H1 stages.
/// Meshes and histories are supplied in coarse-to-fine order; coarsening
/// executes the same lineage in reverse and retains orphaned node values.
pub fn apply_h1_p3_state_transfer_with_refinement_chain<E, F>(
    plan_chain: &IrStateTransferPlanChain,
    meshes: &[&Mesh],
    spaces: &[&H1Space],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    entity_maps: &[BTreeMap<EntityId, usize>],
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(usize, EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    if spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "adaptive P3 H1 chain requires one space per mesh".into(),
        ));
    }
    for stage in 0..histories.len() {
        validate_h1_degree_spaces(
            meshes[stage],
            meshes[stage + 1],
            spaces[stage],
            spaces[stage + 1],
            PolynomialDegree::P3,
        )?;
    }
    apply_nodal_state_transfer_with_refinement_chain(
        plan_chain,
        meshes,
        source_values,
        histories,
        direction,
        entity_maps,
        initialize_or_application,
    )
}

fn validate_h1_degree_spaces(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    expected_degree: PolynomialDegree,
) -> Result<(), MeshlangTransferError> {
    if source_space.degree != expected_degree || target_space.degree != expected_degree {
        return Err(MeshlangTransferError::UnsupportedH1Degree {
            expected_degree,
            source_degree: source_space.degree,
            target_degree: target_space.degree,
        });
    }

    source_space
        .validate(source_mesh)
        .map_err(|error| MeshlangTransferError::InvalidH1Space {
            side: "source",
            error: error.to_string(),
        })?;
    target_space
        .validate(target_mesh)
        .map_err(|error| MeshlangTransferError::InvalidH1Space {
            side: "target",
            error: error.to_string(),
        })?;

    for (side, mesh, space) in [
        ("source", source_mesh, source_space),
        ("target", target_mesh, target_space),
    ] {
        let expected = h1_mesh_dof_range(mesh);
        let actual = space.num_global_dofs(mesh);
        if actual != expected {
            return Err(MeshlangTransferError::InvalidH1DofLayout {
                side,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn h1_mesh_dof_range(mesh: &Mesh) -> usize {
    mesh.elements
        .iter()
        .flat_map(|element| element.nodes.iter().copied())
        .max()
        .map_or(0, |node| node + 1)
}

fn h1_mesh_geometry_degree(mesh: &Mesh) -> Option<PolynomialDegree> {
    let mut elements = mesh.elements.iter();
    let first = elements.next()?;
    let degree = match first.nodes.len() {
        n if n == first.element_type.num_vertices() => PolynomialDegree::P1,
        n if n == first.element_type.num_nodes_p2() => PolynomialDegree::P2,
        n if n == first.element_type.num_nodes_p3() => PolynomialDegree::P3,
        _ => return None,
    };

    elements
        .all(|element| match element.nodes.len() {
            n if n == element.element_type.num_vertices() => degree == PolynomialDegree::P1,
            n if n == element.element_type.num_nodes_p2() => degree == PolynomialDegree::P2,
            n if n == element.element_type.num_nodes_p3() => degree == PolynomialDegree::P3,
            _ => false,
        })
        .then_some(degree)
}

fn validate_h1_mixed_spaces(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
) -> Result<(), MeshlangTransferError> {
    source_space
        .validate(source_mesh)
        .map_err(|error| MeshlangTransferError::InvalidH1Space {
            side: "source",
            error: error.to_string(),
        })?;
    target_space
        .validate(target_mesh)
        .map_err(|error| MeshlangTransferError::InvalidH1Space {
            side: "target",
            error: error.to_string(),
        })?;

    for (side, mesh, space) in [
        ("source", source_mesh, source_space),
        ("target", target_mesh, target_space),
    ] {
        let geometry_degree =
            h1_mesh_geometry_degree(mesh).ok_or_else(|| MeshlangTransferError::InvalidH1Space {
                side,
                error: "mesh geometry order is empty, malformed, or mixed".to_string(),
            })?;
        if geometry_degree != space.degree {
            return Err(MeshlangTransferError::InvalidH1Space {
                side,
                error: format!(
                    "H1 degree {:?} does not match explicit mesh geometry degree {:?}",
                    space.degree, geometry_degree
                ),
            });
        }
        let expected = h1_mesh_dof_range(mesh);
        let actual = space.num_global_dofs(mesh);
        if actual != expected {
            return Err(MeshlangTransferError::InvalidH1DofLayout {
                side,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

/// Apply a canonical MeshLang plan between independently ordered scalar H1
/// spaces. Each side must use matching, uniform P1/P2/P3 geometry and expose
/// one global DOF per mesh node; the checked nodal interpolator then evaluates
/// the source field at every target DOF coordinate.
pub fn apply_h1_mixed_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    validate_h1_mixed_spaces(source_mesh, target_mesh, source_space, target_space)?;
    let interpolation_data = create_interpolation_data_checked(target_mesh, source_mesh)?;
    apply_nodal_state_transfer(
        plan,
        source_mesh,
        target_mesh,
        &interpolation_data,
        source_values,
        source_entities,
        target_entities,
        initialize_or_application,
    )
}

/// Adapt a refinement history recorded at either polynomial order to the
/// endpoint mesh used by a mixed-order transfer.
///
/// The generic lineage validator intentionally compares complete node lists:
/// that is the right contract when both endpoints use the same H1 order. A
/// mixed-order hand-off has different high-order trace/interior nodes at the
/// parent endpoint, however, while element IDs, corner topology, level, and
/// parent identity remain the stable lineage contract. Validate those stable
/// fields here, then replace only the parent snapshot's order-specific node
/// list with the endpoint representation before passing it to the strict
/// checked transfer kernel.
fn normalize_mixed_h1_refinement_history(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
) -> Result<crate::mesh::refinement::RefinementHistory, MeshlangTransferError> {
    let mut normalized = history.clone();
    for parent in &mut normalized.parents {
        let endpoint = match direction {
            RefinementTransferDirection::Refine => source_mesh
                .elements
                .get(parent.original_index)
                .ok_or_else(|| {
                    MeshlangTransferError::Interpolation(
                        InterpolationError::RefinementHistoryMismatch {
                            parent_id: parent.element.id,
                            reason: "parent original index outside source mesh".into(),
                        },
                    )
                })?,
            RefinementTransferDirection::Coarsen => target_mesh
                .elements
                .iter()
                .find(|element| element.id == parent.element.id)
                .ok_or_else(|| {
                    MeshlangTransferError::Interpolation(
                        InterpolationError::RefinementHistoryMismatch {
                            parent_id: parent.element.id,
                            reason: "parent snapshot absent from target mesh".into(),
                        },
                    )
                })?,
        };

        if !supported_h1_element_layout(&parent.element)
            || !supported_h1_element_layout(endpoint)
            || !mixed_h1_element_identity(&parent.element, endpoint)
        {
            return Err(MeshlangTransferError::Interpolation(
                InterpolationError::RefinementHistoryMismatch {
                    parent_id: parent.element.id,
                    reason: "mixed-order parent snapshot disagrees on stable topology".into(),
                },
            ));
        }

        parent.element.nodes = endpoint.nodes.clone();
    }
    Ok(normalized)
}

fn supported_h1_element_layout(element: &crate::mesh::Element) -> bool {
    let node_count = element.nodes.len();
    node_count == element.element_type.num_vertices()
        || node_count == element.element_type.num_nodes_p2()
        || node_count == element.element_type.num_nodes_p3()
}

fn mixed_h1_element_identity(left: &crate::mesh::Element, right: &crate::mesh::Element) -> bool {
    if left.id != right.id
        || left.element_type != right.element_type
        || left.parent_id != right.parent_id
        || left.level != right.level
        || left.marker != right.marker
    {
        return false;
    }
    let vertex_count = left.element_type.num_vertices();
    left.nodes.len() >= vertex_count
        && right.nodes.len() >= vertex_count
        && left.nodes[..vertex_count] == right.nodes[..vertex_count]
}

/// Apply a canonical MeshLang plan across one adaptive H1 lineage step while
/// allowing the source and target scalar spaces to have different polynomial
/// orders. The lineage validator is topology/identity based, so a P1 parent
/// may refine into P2 children (or the reverse coarsening direction) as long
/// as element IDs, parent links, and levels remain unchanged. The checked
/// nodal interpolation kernel evaluates the source geometry at every target
/// node and covers the mixed P1/P2/P3 scalar hand-off without invoking
/// callbacks for numerical actions.
#[expect(
    clippy::too_many_arguments,
    reason = "checked state-transfer entry point threading plan, meshes, callbacks and options; arity is the stable checked-call contract"
)]
pub fn apply_h1_mixed_state_transfer_with_refinement_history<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    validate_h1_mixed_spaces(source_mesh, target_mesh, source_space, target_space)?;
    let normalized_history =
        normalize_mixed_h1_refinement_history(source_mesh, target_mesh, history, direction)?;
    apply_nodal_state_transfer_with_refinement_history(
        plan,
        source_mesh,
        target_mesh,
        source_values,
        &normalized_history,
        direction,
        source_entities,
        target_entities,
        initialize_or_application,
    )
}

/// Apply a canonical MeshLang plan chain across mixed-order adaptive H1
/// stages. Each adjacent pair may use a different supported scalar order
/// (P1, P2, or P3); endpoint-specific history layouts are normalized before
/// the transactional generic chain executor validates lineage and runs any
/// callbacks.
pub fn apply_h1_mixed_state_transfer_with_refinement_chain<E, F>(
    plan_chain: &IrStateTransferPlanChain,
    meshes: &[&Mesh],
    spaces: &[&H1Space],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    entity_maps: &[BTreeMap<EntityId, usize>],
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(usize, EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "mixed adaptive H1 chain requires one space per mesh and at least two meshes".into(),
        ));
    }
    if histories.len() + 1 != meshes.len()
        || entity_maps.len() != meshes.len()
        || plan_chain.stage_count() != histories.len()
    {
        return Err(MeshlangTransferError::InvalidPlan(
            "mixed adaptive H1 chain mesh, history, map, and plan counts disagree".into(),
        ));
    }
    plan_chain
        .validate()
        .map_err(|error| MeshlangTransferError::InvalidPlan(error.to_string()))?;

    let mut normalized_histories = Vec::with_capacity(histories.len());
    for (stage, history) in histories.iter().enumerate() {
        let (source_index, target_index) = match direction {
            RefinementTransferDirection::Refine => (stage, stage + 1),
            RefinementTransferDirection::Coarsen => (stage + 1, stage),
        };
        validate_h1_mixed_spaces(
            meshes[source_index],
            meshes[target_index],
            spaces[source_index],
            spaces[target_index],
        )?;
        normalized_histories.push(normalize_mixed_h1_refinement_history(
            meshes[source_index],
            meshes[target_index],
            history,
            direction,
        )?);
    }

    apply_nodal_state_transfer_with_refinement_chain(
        plan_chain,
        meshes,
        source_values,
        &normalized_histories,
        direction,
        entity_maps,
        initialize_or_application,
    )
}

/// Apply a canonical MeshLang plan to a quadratic (P2/Q2) scalar H1 field.
///
/// This explicit higher-order entry point validates both H1 spaces and their
/// complete global DOF layout before constructing checked interpolation
/// metadata. The interpolation kernel evaluates all quadratic edge, face, and
/// interior shape functions, so higher-order DOFs cannot be silently dropped.
pub fn apply_h1_p2_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    validate_h1_degree_spaces(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        PolynomialDegree::P2,
    )?;
    let interpolation_data = create_interpolation_data_checked(target_mesh, source_mesh)?;
    apply_nodal_state_transfer(
        plan,
        source_mesh,
        target_mesh,
        &interpolation_data,
        source_values,
        source_entities,
        target_entities,
        initialize_or_application,
    )
}

/// Apply a canonical MeshLang plan to a quadratic H1 field across one checked
/// adaptive refinement or coarsening step.
#[expect(
    clippy::too_many_arguments,
    reason = "checked state-transfer entry point threading plan, meshes, callbacks and options; arity is the stable checked-call contract"
)]
pub fn apply_h1_p2_state_transfer_with_refinement_history<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    validate_h1_degree_spaces(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        PolynomialDegree::P2,
    )?;
    apply_nodal_state_transfer_with_refinement_history(
        plan,
        source_mesh,
        target_mesh,
        source_values,
        history,
        direction,
        source_entities,
        target_entities,
        initialize_or_application,
    )
}

/// Apply a canonical MeshLang plan chain across checked adaptive P2 H1
/// stages. Meshes and spaces remain in coarse-to-fine lineage order; the
/// coarsening direction executes the same histories in reverse. The generic
/// nodal chain retains orphaned node values until the caller explicitly
/// compacts the final mesh.
pub fn apply_h1_p2_state_transfer_with_refinement_chain<E, F>(
    plan_chain: &IrStateTransferPlanChain,
    meshes: &[&Mesh],
    spaces: &[&H1Space],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    entity_maps: &[BTreeMap<EntityId, usize>],
    initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(usize, EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    if spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "adaptive P2 H1 chain requires one space per mesh".into(),
        ));
    }
    for stage in 0..histories.len() {
        validate_h1_degree_spaces(
            meshes[stage],
            meshes[stage + 1],
            spaces[stage],
            spaces[stage + 1],
            PolynomialDegree::P2,
        )?;
    }
    apply_nodal_state_transfer_with_refinement_chain(
        plan_chain,
        meshes,
        source_values,
        histories,
        direction,
        entity_maps,
        initialize_or_application,
    )
}

pub fn apply_nodal_state_transfer_with_refinement_history<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if source_values.len() != source_mesh.num_nodes() {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_mesh.num_nodes(),
            actual: source_values.len(),
        });
    }
    for (&entity, &node) in source_entities {
        if node >= source_mesh.num_nodes() {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node });
        }
    }
    for (&entity, &node) in target_entities {
        if node >= target_mesh.num_nodes() {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node });
        }
    }

    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&node) = source_entities.get(&entry.entity) {
                    if !source_values[node].is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                } else if entry.action == IrStateTransferAction::Preserve {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let transferred = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            transfer_h1_state_with_refinement_history_checked(
                source_mesh,
                target_mesh,
                source_values,
                history,
                direction,
            )
        })
        .transpose()?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let node = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[node]
            }
            IrStateTransferAction::Interpolate => {
                let node = target_entities[&entry.entity];
                transferred
                    .as_ref()
                    .expect("validated adaptive H1 interpolation path")[node]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&node| source_values[node]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H1 action"),
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Owner/ghost portion of a checked nodal state-transfer result.
///
/// The transfer itself still runs against the caller-owned global nodal
/// vector. This boundary deterministically filters the result through a
/// [`PartitionGhostLayout`], making the values that a distributed caller must
/// retain locally explicit without pretending to perform MPI synchronization.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionedNodalState {
    /// Values whose canonical node owner is this partition.
    pub owned: BTreeMap<EntityId, f64>,
    /// Values needed locally but owned by another partition.
    pub ghost: BTreeMap<EntityId, f64>,
}

fn validate_partition_nodes(
    layout: &PartitionGhostLayout,
    mesh: &Mesh,
    side: &'static str,
) -> Result<(BTreeSet<usize>, BTreeSet<usize>), MeshlangTransferError> {
    let mut seen = BTreeSet::new();
    for &node in layout.owned_nodes.iter().chain(&layout.ghost_nodes) {
        if node >= mesh.num_nodes() {
            return Err(MeshlangTransferError::InvalidPartitionNode {
                side,
                node,
                nodes: mesh.num_nodes(),
            });
        }
        if !seen.insert(node) {
            return Err(MeshlangTransferError::DuplicatePartitionNode { side, node });
        }
    }
    Ok((
        layout.owned_nodes.iter().copied().collect(),
        layout.ghost_nodes.iter().copied().collect(),
    ))
}

/// Apply one adaptive MeshLang nodal transfer and split its result by
/// deterministic partition ownership.
///
/// `source_entities` and `target_entities` may describe the complete global
/// entity map. Entries not present in `target_layout` are omitted from the
/// returned local state, while malformed layout indices are rejected before
/// the numerical kernel or application callback runs.
#[expect(
    clippy::too_many_arguments,
    reason = "checked state-transfer entry point threading plan, meshes, callbacks and options; arity is the stable checked-call contract"
)]
pub fn apply_partitioned_nodal_state_transfer_with_refinement_history<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    source_layout: &PartitionGhostLayout,
    target_layout: &PartitionGhostLayout,
    initialize_or_application: F,
) -> Result<PartitionedNodalState, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    // Validate both sides before invoking the existing transactional bridge.
    let _ = validate_partition_nodes(source_layout, source_mesh, "source")?;
    let (owned_nodes, ghost_nodes) =
        validate_partition_nodes(target_layout, target_mesh, "target")?;
    let transferred = apply_nodal_state_transfer_with_refinement_history(
        plan,
        source_mesh,
        target_mesh,
        source_values,
        history,
        direction,
        source_entities,
        target_entities,
        initialize_or_application,
    )?;

    let mut owned = BTreeMap::new();
    let mut ghost = BTreeMap::new();
    for (entity, value) in transferred {
        let Some(&node) = target_entities.get(&entity) else {
            return Err(MeshlangTransferError::MissingTargetEntity(entity));
        };
        if owned_nodes.contains(&node) {
            owned.insert(entity, value);
        } else if ghost_nodes.contains(&node) {
            ghost.insert(entity, value);
        }
    }
    Ok(PartitionedNodalState { owned, ghost })
}

/// Complete a caller-owned owner/ghost exchange for a partitioned transfer.
///
/// The callback receives an immutable snapshot of locally owned values and
/// must return exactly the ghost keys requested by the prior transfer. This
/// keeps MPI/device transport outside FEM while making the hand-off checked:
/// no partial ghost map is published, unknown or missing keys fail closed, and
/// non-finite exchanged values are rejected.
pub fn synchronize_partitioned_nodal_state<E, F>(
    state: PartitionedNodalState,
    exchange: F,
) -> Result<PartitionedNodalState, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnOnce(&BTreeMap<EntityId, f64>) -> Result<BTreeMap<EntityId, f64>, E>,
{
    for (&entity, &value) in state.owned.iter().chain(state.ghost.iter()) {
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFinitePartitionGhost { entity });
        }
    }

    let exchanged = exchange(&state.owned)
        .map_err(|error| MeshlangTransferError::PartitionExchange(error.to_string()))?;
    for (&entity, &value) in &exchanged {
        if !state.ghost.contains_key(&entity) {
            return Err(MeshlangTransferError::UnknownPartitionGhostEntity(entity));
        }
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFinitePartitionGhost { entity });
        }
    }
    for &entity in state.ghost.keys() {
        if !exchanged.contains_key(&entity) {
            return Err(MeshlangTransferError::MissingPartitionGhostEntity(entity));
        }
    }
    Ok(PartitionedNodalState {
        owned: state.owned,
        ghost: exchanged,
    })
}

fn materialize_chain_nodal_values(
    values: &BTreeMap<EntityId, f64>,
    mesh: &Mesh,
    entities: &BTreeMap<EntityId, usize>,
) -> Result<Vec<f64>, MeshlangTransferError> {
    let mut nodal = vec![0.0; mesh.num_nodes()];
    let mut covered = vec![false; mesh.num_nodes()];
    for (&entity, &node) in entities {
        if node >= mesh.num_nodes() {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node });
        }
        let value = values
            .get(&entity)
            .copied()
            .ok_or(MeshlangTransferError::MissingTargetEntity(entity))?;
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult { entity });
        }
        nodal[node] = value;
        covered[node] = true;
    }
    if let Some(node) = covered.iter().position(|covered| !covered) {
        return Err(MeshlangTransferError::IncompleteTargetNodeMap { node });
    }
    Ok(nodal)
}

/// Apply a canonical MeshLang plan chain across checked adaptive H1 stages.
///
/// `meshes` and `histories` retain the coarse-to-fine lineage order. Plans
/// are supplied in execution order, so callers performing coarsening provide
/// the inverse actions in reverse stage order. Every lineage and plan is
/// validated before the first numerical transfer or callback. Intermediate
/// maps are materialized into complete nodal vectors and are never returned;
/// only the final keyed target map is published.
pub fn apply_nodal_state_transfer_with_refinement_chain<E, F>(
    plan_chain: &IrStateTransferPlanChain,
    meshes: &[&Mesh],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    entity_maps: &[BTreeMap<EntityId, usize>],
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(usize, EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan_chain
        .validate()
        .map_err(|error| MeshlangTransferError::InvalidPlan(error.to_string()))?;
    if meshes.len() < 2
        || histories.len() + 1 != meshes.len()
        || plan_chain.stage_count() != histories.len()
        || entity_maps.len() != meshes.len()
    {
        return Err(MeshlangTransferError::InvalidPlan(
            "adaptive state-transfer chain requires one plan/history per adjacent mesh pair and one entity map per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;

    let source_mesh_index = match direction {
        RefinementTransferDirection::Refine => 0,
        RefinementTransferDirection::Coarsen => meshes.len() - 1,
    };
    if source_values.len() != meshes[source_mesh_index].num_nodes() {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: meshes[source_mesh_index].num_nodes(),
            actual: source_values.len(),
        });
    }

    let mut current_values = source_values.to_vec();
    let mut final_map = None;
    for execution_stage in 0..histories.len() {
        let lineage_stage = match direction {
            RefinementTransferDirection::Refine => execution_stage,
            RefinementTransferDirection::Coarsen => histories.len() - 1 - execution_stage,
        };
        let (source_index, target_index) = match direction {
            RefinementTransferDirection::Refine => (lineage_stage, lineage_stage + 1),
            RefinementTransferDirection::Coarsen => (lineage_stage + 1, lineage_stage),
        };
        let stage_map = apply_nodal_state_transfer_with_refinement_history(
            &plan_chain.stages[execution_stage],
            meshes[source_index],
            meshes[target_index],
            &current_values,
            &histories[lineage_stage],
            direction,
            &entity_maps[source_index],
            &entity_maps[target_index],
            |entity, action, previous| {
                initialize_or_application(execution_stage, entity, action, previous)
            },
        )?;
        if execution_stage + 1 == histories.len() {
            final_map = Some(stage_map);
        } else {
            current_values = materialize_chain_nodal_values(
                &stage_map,
                meshes[target_index],
                &entity_maps[target_index],
            )?;
        }
    }
    final_map.ok_or_else(|| {
        MeshlangTransferError::InvalidPlan(
            "adaptive state-transfer chain produced no final stage".into(),
        )
    })
}

/// Apply a canonical plan to an order-1 H(curl) edge state.
///
/// The checked interpolation kernel reconstructs target edge moments from
/// the source finite-element field. Entity maps therefore point at global
/// edge DOFs rather than mesh vertex indices. `Conservative` is rejected
/// explicitly: scalar lumped-mass conservation is not a valid substitute for
/// the tangential-moment transfer contract of H(curl).
pub fn apply_hcurl_order1_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    validate_finite_mesh(source_mesh, "source")?;
    validate_finite_mesh(target_mesh, "target")?;

    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &node) in source_entities {
        if node >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node });
        }
    }
    for (&entity, &node) in target_entities {
        if node >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node });
        }
    }

    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&node) = source_entities.get(&entry.entity) {
                    if !source_values[node].is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                } else if entry.action == IrStateTransferAction::Preserve {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let has_interpolation = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate);
    let interpolated = has_interpolation
        .then(|| {
            transfer_hcurl_order1_state_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                source_values,
            )
        })
        .transpose()?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let node = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[node]
            }
            IrStateTransferAction::Interpolate => {
                let node = target_entities[&entry.entity];
                interpolated
                    .as_ref()
                    .expect("validated H(curl) interpolation path")[node]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&node| source_values[node]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply canonical plan to a mixed-order triangular H(curl) state.
///
/// Source and target spaces may independently use first- or second-order
/// Nédélec edge moments. The checked interpolation kernel reconstructs target
/// moments from the physical tangential trace and projects order-2 interior
/// bubbles. Entity actions retain the same validation, callback, and finite
/// result guarantees as the fixed-order bridges.
pub fn apply_hcurl_mixed_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    validate_finite_mesh(source_mesh, "source")?;
    validate_finite_mesh(target_mesh, "target")?;

    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }

    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    if !source_values[dof].is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                } else if entry.action == IrStateTransferAction::Preserve {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let interpolated = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            transfer_hcurl_mixed_triangle_state_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                source_values,
            )
        })
        .transpose()?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate => {
                let dof = target_entities[&entry.entity];
                interpolated
                    .as_ref()
                    .expect("validated mixed-order H(curl) interpolation path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply a canonical plan to an order-2 H(curl) state on affine or quadratic
/// isoparametric triangles.
///
/// Entity maps point at the two canonical edge moments and the element-local
/// interior bubble DOFs exposed by [`HcurlSpace::local_to_global`].  The
/// The checked interpolation kernel owns supported-geometry admission;
/// unsupported element families and malformed curved layouts are rejected
/// before any MeshLang callback runs.
pub fn apply_hcurl_order2_triangle_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    validate_finite_mesh(source_mesh, "source")?;
    validate_finite_mesh(target_mesh, "target")?;

    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }

    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    if !source_values[dof].is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                } else if entry.action == IrStateTransferAction::Preserve {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let interpolated = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            transfer_hcurl_order2_triangle_state_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                source_values,
            )
        })
        .transpose()?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate => {
                let dof = target_entities[&entry.entity];
                interpolated
                    .as_ref()
                    .expect("validated order-2 H(curl) interpolation path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply a canonical plan to a complex order-2 H(curl) state on affine
/// triangles. Real and imaginary edge/bubble moments use the checked order-2
/// Piola transfer independently; callbacks run exactly once for
/// initialize/application-defined entries.
/// Apply a canonical MeshLang plan across a checked order-2 H(curl)
/// refinement chain. The interpolation kernel owns mesh-history validation;
/// this bridge owns entity actions and callback isolation.
pub fn apply_hcurl_order2_triangle_state_transfer_with_refinement_chain<E, F>(
    plan: &IrStateTransferPlan,
    meshes: &[&Mesh],
    spaces: &[&HcurlSpace],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "order-2 H(curl) refinement chain requires at least two meshes and one space per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)
        .map_err(MeshlangTransferError::Interpolation)?;
    let (source_index, target_index) = match direction {
        RefinementTransferDirection::Refine => (0, meshes.len() - 1),
        RefinementTransferDirection::Coarsen => (meshes.len() - 1, 0),
    };
    let source_dofs = spaces[source_index].num_global_dofs(meshes[source_index]);
    let target_dofs = spaces[target_index].num_global_dofs(meshes[target_index]);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    if !source_values[dof].is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                } else if entry.action == IrStateTransferAction::Preserve {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }
    let transferred = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
                meshes,
                spaces,
                source_values,
                histories,
                direction,
            )
        })
        .transpose()
        .map_err(MeshlangTransferError::Interpolation)?;
    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate => {
                let dof = target_entities[&entry.entity];
                transferred
                    .as_ref()
                    .expect("validated order-2 H(curl) chain interpolation path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply a canonical MeshLang plan across a checked multi-stage order-2
/// H(curl) refinement chain for a Complex64 state. Real and imaginary
/// components use independent checked chain transfers; callbacks are invoked
/// only once while resolving the plan.
/// Apply a canonical MeshLang plan across a checked multi-stage order-1
/// H(curl) refinement chain. The interpolation kernel owns mesh-history
/// validation; this bridge owns entity actions and callback isolation.
pub fn apply_hcurl_order1_state_transfer_with_refinement_chain<E, F>(
    plan: &IrStateTransferPlan,
    meshes: &[&Mesh],
    spaces: &[&HcurlSpace],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "order-1 H(curl) refinement chain requires at least two meshes and one space per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)
        .map_err(MeshlangTransferError::Interpolation)?;

    let (source_index, target_index) = match direction {
        RefinementTransferDirection::Refine => (0, meshes.len() - 1),
        RefinementTransferDirection::Coarsen => (meshes.len() - 1, 0),
    };
    let source_dofs = spaces[source_index].num_global_dofs(meshes[source_index]);
    let target_dofs = spaces[target_index].num_global_dofs(meshes[target_index]);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity)
                    && !source_values[dof].is_finite()
                {
                    return Err(MeshlangTransferError::NonFiniteSource {
                        entity: entry.entity,
                    });
                }
                if entry.action == IrStateTransferAction::Preserve
                    && !source_entities.contains_key(&entry.entity)
                {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let transferred = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            transfer_hcurl_order1_state_with_refinement_chain_checked(
                meshes,
                spaces,
                source_values,
                histories,
                direction,
            )
        })
        .transpose()
        .map_err(MeshlangTransferError::Interpolation)?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => source_values[source_entities[&entry.entity]],
            IrStateTransferAction::Interpolate => transferred
                .as_ref()
                .expect("validated order-1 H(curl) chain interpolation path")
                [target_entities[&entry.entity]],
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply a canonical MeshLang plan across a checked multi-stage order-1
/// H(curl) refinement chain to a complex-valued state. Real and imaginary
/// components use independent checked transfers; callbacks run exactly once.
pub fn apply_complex_hcurl_order1_state_transfer_with_refinement_chain<E, F>(
    plan: &IrStateTransferPlan,
    meshes: &[&Mesh],
    spaces: &[&HcurlSpace],
    source_values: &[Complex64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "order-1 H(curl) refinement chain requires at least two meshes and one space per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)
        .map_err(MeshlangTransferError::Interpolation)?;

    let (source_index, target_index) = match direction {
        RefinementTransferDirection::Refine => (0, meshes.len() - 1),
        RefinementTransferDirection::Coarsen => (meshes.len() - 1, 0),
    };
    let source_dofs = spaces[source_index].num_global_dofs(meshes[source_index]);
    let target_dofs = spaces[target_index].num_global_dofs(meshes[target_index]);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    let value = source_values[dof];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                }
                if entry.action == IrStateTransferAction::Preserve
                    && !source_entities.contains_key(&entry.entity)
                {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let transferred = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
            let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
            let real = transfer_hcurl_order1_state_with_refinement_chain_checked(
                meshes,
                spaces,
                &real_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            let imaginary = transfer_hcurl_order1_state_with_refinement_chain_checked(
                meshes,
                spaces,
                &imaginary_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            Ok::<_, MeshlangTransferError>(
                real.into_iter()
                    .zip(imaginary)
                    .map(|(real, imaginary)| Complex64::new(real, imaginary))
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => source_values[source_entities[&entry.entity]],
            IrStateTransferAction::Interpolate => transferred
                .as_ref()
                .expect("validated complex order-1 H(curl) chain interpolation path")
                [target_entities[&entry.entity]],
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply canonical MeshLang plan across a checked multi-stage RT0 H(div)
/// refinement chain. The interpolation kernel owns mesh-history validation;
/// this bridge owns entity actions and callback isolation.
pub fn apply_hdiv_rt0_state_transfer_with_refinement_chain<E, F>(
    plan: &IrStateTransferPlan,
    meshes: &[&Mesh],
    spaces: &[&HdivSpace],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "RT0 H(div) refinement chain requires at least two meshes and one space per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)
        .map_err(MeshlangTransferError::Interpolation)?;

    let (source_index, target_index) = match direction {
        RefinementTransferDirection::Refine => (0, meshes.len() - 1),
        RefinementTransferDirection::Coarsen => (meshes.len() - 1, 0),
    };
    let source_dofs = spaces[source_index].num_global_dofs(meshes[source_index]);
    let target_dofs = spaces[target_index].num_global_dofs(meshes[target_index]);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity)
                    && !source_values[dof].is_finite()
                {
                    return Err(MeshlangTransferError::NonFiniteSource {
                        entity: entry.entity,
                    });
                }
                if entry.action == IrStateTransferAction::Preserve
                    && !source_entities.contains_key(&entry.entity)
                {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let transferred = plan
        .entries
        .iter()
        .any(|entry| {
            matches!(
                entry.action,
                IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
            )
        })
        .then(|| {
            transfer_hdiv_rt0_state_with_refinement_chain_checked(
                meshes,
                spaces,
                source_values,
                histories,
                direction,
            )
        })
        .transpose()
        .map_err(MeshlangTransferError::Interpolation)?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => source_values[source_entities[&entry.entity]],
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => transferred
                .as_ref()
                .expect("validated RT0 chain interpolation path")[target_entities[&entry.entity]],
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply a canonical MeshLang plan across a checked multi-stage RT0 H(div)
/// refinement chain to a complex-valued state. Real and imaginary components
/// use independent checked transfers; callbacks run exactly once.
pub fn apply_complex_hdiv_rt0_state_transfer_with_refinement_chain<E, F>(
    plan: &IrStateTransferPlan,
    meshes: &[&Mesh],
    spaces: &[&HdivSpace],
    source_values: &[Complex64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "RT0 H(div) refinement chain requires at least two meshes and one space per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)
        .map_err(MeshlangTransferError::Interpolation)?;

    let (source_index, target_index) = match direction {
        RefinementTransferDirection::Refine => (0, meshes.len() - 1),
        RefinementTransferDirection::Coarsen => (meshes.len() - 1, 0),
    };
    let source_dofs = spaces[source_index].num_global_dofs(meshes[source_index]);
    let target_dofs = spaces[target_index].num_global_dofs(meshes[target_index]);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    let value = source_values[dof];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                }
                if entry.action == IrStateTransferAction::Preserve
                    && !source_entities.contains_key(&entry.entity)
                {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let transferred = plan
        .entries
        .iter()
        .any(|entry| {
            matches!(
                entry.action,
                IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
            )
        })
        .then(|| {
            let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
            let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
            let real = transfer_hdiv_rt0_state_with_refinement_chain_checked(
                meshes,
                spaces,
                &real_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            let imaginary = transfer_hdiv_rt0_state_with_refinement_chain_checked(
                meshes,
                spaces,
                &imaginary_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            Ok::<_, MeshlangTransferError>(
                real.into_iter()
                    .zip(imaginary)
                    .map(|(real, imaginary)| Complex64::new(real, imaginary))
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => source_values[source_entities[&entry.entity]],
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => transferred
                .as_ref()
                .expect("validated complex RT0 chain interpolation path")
                [target_entities[&entry.entity]],
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

pub fn apply_hdiv_rt1_state_transfer_with_refinement_chain<E, F>(
    plan: &IrStateTransferPlan,
    meshes: &[&Mesh],
    spaces: &[&HdivSpace],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "RT1 H(div) refinement chain requires at least two meshes and one space per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)
        .map_err(MeshlangTransferError::Interpolation)?;
    let (source_index, target_index) = match direction {
        RefinementTransferDirection::Refine => (0, meshes.len() - 1),
        RefinementTransferDirection::Coarsen => (meshes.len() - 1, 0),
    };
    let source_dofs = spaces[source_index].num_global_dofs(meshes[source_index]);
    let target_dofs = spaces[target_index].num_global_dofs(meshes[target_index]);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &node) in source_entities {
        if node >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node });
        }
    }
    for (&entity, &node) in target_entities {
        if node >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if !source_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }
    if let Some(entity) = source_entities.iter().find_map(|(&entity, &node)| {
        (node < source_values.len() && !source_values[node].is_finite()).then_some(entity)
    }) {
        return Err(MeshlangTransferError::NonFiniteSource { entity });
    }
    let transferred = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            transfer_hdiv_rt1_state_with_refinement_chain_checked(
                meshes,
                spaces,
                source_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)
        })
        .transpose()?;
    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => source_values[source_entities[&entry.entity]],
            IrStateTransferAction::Interpolate => transferred
                .as_ref()
                .expect("validated RT1 H(div) chain interpolation path")
                [target_entities[&entry.entity]],
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&node| source_values[node]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated RT1 action"),
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

pub fn apply_complex_hdiv_rt1_state_transfer_with_refinement_chain<E, F>(
    plan: &IrStateTransferPlan,
    meshes: &[&Mesh],
    spaces: &[&HdivSpace],
    source_values: &[Complex64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "complex RT1 H(div) refinement chain requires at least two meshes and one space per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)
        .map_err(MeshlangTransferError::Interpolation)?;
    let (source_index, target_index) = match direction {
        RefinementTransferDirection::Refine => (0, meshes.len() - 1),
        RefinementTransferDirection::Coarsen => (meshes.len() - 1, 0),
    };
    let source_dofs = spaces[source_index].num_global_dofs(meshes[source_index]);
    let target_dofs = spaces[target_index].num_global_dofs(meshes[target_index]);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &node) in source_entities {
        if node >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node });
        }
    }
    for (&entity, &node) in target_entities {
        if node >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if !source_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }
    if let Some(entity) = source_entities.iter().find_map(|(&entity, &node)| {
        (node < source_values.len()
            && (!source_values[node].re.is_finite() || !source_values[node].im.is_finite()))
        .then_some(entity)
    }) {
        return Err(MeshlangTransferError::NonFiniteSource { entity });
    }
    let transferred = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
            let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
            let real = transfer_hdiv_rt1_state_with_refinement_chain_checked(
                meshes,
                spaces,
                &real_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            let imaginary = transfer_hdiv_rt1_state_with_refinement_chain_checked(
                meshes,
                spaces,
                &imaginary_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            Ok::<_, MeshlangTransferError>(
                real.into_iter()
                    .zip(imaginary)
                    .map(|(real, imaginary)| Complex64::new(real, imaginary))
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;
    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => source_values[source_entities[&entry.entity]],
            IrStateTransferAction::Interpolate => transferred
                .as_ref()
                .expect("validated complex RT1 H(div) chain interpolation path")
                [target_entities[&entry.entity]],
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&node| source_values[node]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated RT1 action"),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

pub fn apply_complex_hcurl_order2_triangle_state_transfer_with_refinement_chain<E, F>(
    plan: &IrStateTransferPlan,
    meshes: &[&Mesh],
    spaces: &[&HcurlSpace],
    source_values: &[Complex64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if meshes.len() < 2 || spaces.len() != meshes.len() {
        return Err(MeshlangTransferError::InvalidPlan(
            "complex order-2 H(curl) refinement chain requires at least two meshes and one space per mesh"
                .into(),
        ));
    }
    validate_refinement_transfer_chain_checked(meshes, histories, direction)
        .map_err(MeshlangTransferError::Interpolation)?;
    let (source_index, target_index) = match direction {
        RefinementTransferDirection::Refine => (0, meshes.len() - 1),
        RefinementTransferDirection::Coarsen => (meshes.len() - 1, 0),
    };
    let source_dofs = spaces[source_index].num_global_dofs(meshes[source_index]);
    let target_dofs = spaces[target_index].num_global_dofs(meshes[target_index]);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    let value = source_values[dof];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                } else if entry.action == IrStateTransferAction::Preserve {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }
    let transferred = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
            let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
            let real = transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
                meshes,
                spaces,
                &real_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            let imaginary = transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
                meshes,
                spaces,
                &imaginary_values,
                histories,
                direction,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            Ok::<_, MeshlangTransferError>(
                real.into_iter()
                    .zip(imaginary)
                    .map(|(real, imaginary)| Complex64::new(real, imaginary))
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;
    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate => {
                let dof = target_entities[&entry.entity];
                transferred
                    .as_ref()
                    .expect("validated complex order-2 H(curl) chain interpolation path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

pub fn apply_complex_hcurl_order2_triangle_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[Complex64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    let value = source_values[dof];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                } else if entry.action == IrStateTransferAction::Preserve {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }
    let transferred = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
        .then(|| {
            let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
            let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
            let real = transfer_hcurl_order2_triangle_state_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                &real_values,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            let imaginary = transfer_hcurl_order2_triangle_state_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                &imaginary_values,
            )
            .map_err(MeshlangTransferError::Interpolation)?;
            Ok::<_, MeshlangTransferError>(
                real.into_iter()
                    .zip(imaginary)
                    .map(|(real, imaginary)| Complex64::new(real, imaginary))
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;
    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate => {
                let dof = target_entities[&entry.entity];
                transferred
                    .as_ref()
                    .expect("validated complex order-2 H(curl) interpolation path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply a canonical plan to a complex order-1 H(curl) edge state.
///
/// Real and imaginary edge moments use the same checked Piola transfer
/// independently. Scalar conservative actions remain rejected because the
/// H(curl) contract is tangential-moment preservation, not a lumped mass map.
pub fn apply_complex_hcurl_order1_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[Complex64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                if !source_values[dof].re.is_finite() || !source_values[dof].im.is_finite() {
                    return Err(MeshlangTransferError::NonFiniteSource {
                        entity: entry.entity,
                    });
                }
            }
            IrStateTransferAction::Interpolate => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Conservative => {
                return Err(MeshlangTransferError::UnsupportedAction {
                    entity: entry.entity,
                    action: entry.action,
                });
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let interpolated = if plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate)
    {
        let real = source_values
            .iter()
            .map(|value| value.re)
            .collect::<Vec<_>>();
        let imaginary = source_values
            .iter()
            .map(|value| value.im)
            .collect::<Vec<_>>();
        let real = transfer_hcurl_order1_state_checked(
            source_mesh,
            target_mesh,
            source_space,
            target_space,
            &real,
        )?;
        let imaginary = transfer_hcurl_order1_state_checked(
            source_mesh,
            target_mesh,
            source_space,
            target_space,
            &imaginary,
        )?;
        Some(
            real.into_iter()
                .zip(imaginary)
                .map(|(real, imaginary)| Complex64::new(real, imaginary))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate => {
                let dof = target_entities[&entry.entity];
                interpolated
                    .as_ref()
                    .expect("validated complex H(curl) interpolation path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
            IrStateTransferAction::Conservative => unreachable!("validated H(curl) action"),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply a canonical plan to an RT0 H(div) normal-flux state.
///
/// RT0 coefficients are oriented normal-flux moments, so both
/// `Interpolate` and `Conservative` actions use the checked RT0 transfer
/// kernel. This preserves the distinction from H(curl), where a scalar
/// conservative action is rejected because it cannot represent a tangential
/// moment. Entity maps point canonical MeshLang IDs at global RT0 DOFs;
/// canonical IDs are deliberately not assumed to be mesh-local indices.
pub fn apply_hdiv_rt0_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;

    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }

    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity)
                    && !source_values[dof].is_finite()
                {
                    return Err(MeshlangTransferError::NonFiniteSource {
                        entity: entry.entity,
                    });
                }
                if entry.action == IrStateTransferAction::Preserve
                    && !source_entities.contains_key(&entry.entity)
                {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let needs_transfer = plan.entries.iter().any(|entry| {
        matches!(
            entry.action,
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
        )
    });
    let transferred = needs_transfer
        .then(|| {
            transfer_hdiv_rt0_state_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                source_values,
            )
        })
        .transpose()
        .map_err(MeshlangTransferError::Interpolation)?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                let dof = target_entities[&entry.entity];
                transferred.as_ref().expect("validated RT0 transfer path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Complex-valued counterpart of [`apply_nodal_state_transfer`].
///
/// Harmonic FEM fields use the same entity-level plan but must retain both
/// real and imaginary components through interpolation and conservative
/// correction.  The callback is invoked only for initialize and
/// application-defined actions, exactly as in the real-valued bridge.
/// Apply a canonical plan to a complex-valued RT0 H(div) normal-flux state.
///
/// The checked RT0 kernel is real-valued, so this bridge transfers the real
/// and imaginary components independently and recombines them only after both
/// component transfers succeed. This keeps the orientation and Piola/moment
/// semantics identical to the real-valued bridge while retaining phase.
/// Apply a canonical plan to a supported simplex RT1 H(div) normal-flux state.
///
/// The checked RT1 transfer kernel recovers edge/face moments and cell moments
/// on affine or quadratic isoparametric triangles and affine tetrahedra. `Interpolate` and `Conservative`
/// therefore share the same checked transfer, while
/// preserve/initialize/application-defined/drop retain canonical plan
/// semantics. Entity maps refer to global RT1 DOFs and are
/// validated independently of their canonical MeshLang IDs.
pub fn apply_hdiv_rt1_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<f64>) -> Result<f64, E>,
{
    if source_space.order() != HdivOrder::Rt1 || target_space.order() != HdivOrder::Rt1 {
        return Err(MeshlangTransferError::UnsupportedHdivOrder {
            source_order: source_space.order(),
            target_order: target_space.order(),
        });
    }
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    validate_finite_mesh(source_mesh, "source")?;
    validate_finite_mesh(target_mesh, "target")?;

    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }

    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity)
                    && !source_values[dof].is_finite()
                {
                    return Err(MeshlangTransferError::NonFiniteSource {
                        entity: entry.entity,
                    });
                }
                if entry.action == IrStateTransferAction::Preserve
                    && !source_entities.contains_key(&entry.entity)
                {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let needs_transfer = plan.entries.iter().any(|entry| {
        matches!(
            entry.action,
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
        )
    });
    let transferred = needs_transfer
        .then(|| {
            transfer_hdiv_rt1_state_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                source_values,
            )
        })
        .transpose()
        .map_err(MeshlangTransferError::Interpolation)?;

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                let dof = target_entities[&entry.entity];
                transferred.as_ref().expect("validated RT1 transfer path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
        };
        if !value.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

/// Apply a canonical plan to a complex-valued RT1 H(div) state.
///
/// Real and imaginary components are transferred independently through the
/// checked RT1 kernel, preserving orientation, edge/face moments, and cell
/// moments without introducing a separate complex interpolation algorithm.
pub fn apply_complex_hdiv_rt1_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[Complex64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    if source_space.order() != HdivOrder::Rt1 || target_space.order() != HdivOrder::Rt1 {
        return Err(MeshlangTransferError::UnsupportedHdivOrder {
            source_order: source_space.order(),
            target_order: target_space.order(),
        });
    }
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    validate_finite_mesh(source_mesh, "source")?;
    validate_finite_mesh(target_mesh, "target")?;

    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }
    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    let value = source_values[dof];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                }
                if entry.action == IrStateTransferAction::Preserve
                    && !source_entities.contains_key(&entry.entity)
                {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let needs_transfer = plan.entries.iter().any(|entry| {
        matches!(
            entry.action,
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
        )
    });
    let transferred = if needs_transfer {
        let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
        let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
        let real = transfer_hdiv_rt1_state_checked(
            source_mesh,
            target_mesh,
            source_space,
            target_space,
            &real_values,
        )
        .map_err(MeshlangTransferError::Interpolation)?;
        let imaginary = transfer_hdiv_rt1_state_checked(
            source_mesh,
            target_mesh,
            source_space,
            target_space,
            &imaginary_values,
        )
        .map_err(MeshlangTransferError::Interpolation)?;
        Some(
            real.into_iter()
                .zip(imaginary)
                .map(|(real, imaginary)| Complex64::new(real, imaginary))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                let dof = target_entities[&entry.entity];
                transferred
                    .as_ref()
                    .expect("validated complex RT1 transfer path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

pub fn apply_complex_hdiv_rt0_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[Complex64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;

    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if source_values.len() != source_dofs {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }

    for (&entity, &dof) in source_entities {
        if dof >= source_dofs {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node: dof });
        }
    }
    for (&entity, &dof) in target_entities {
        if dof >= target_dofs {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node: dof });
        }
    }

    for entry in &plan.entries {
        match entry.action {
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined => {
                if let Some(&dof) = source_entities.get(&entry.entity) {
                    let value = source_values[dof];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(MeshlangTransferError::NonFiniteSource {
                            entity: entry.entity,
                        });
                    }
                } else if entry.action == IrStateTransferAction::Preserve {
                    return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
                }
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                if !target_entities.contains_key(&entry.entity) {
                    return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
                }
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::Drop => {}
        }
    }

    let needs_transfer = plan.entries.iter().any(|entry| {
        matches!(
            entry.action,
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
        )
    });
    let transferred = if needs_transfer {
        let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
        let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
        let real = transfer_hdiv_rt0_state_checked(
            source_mesh,
            target_mesh,
            source_space,
            target_space,
            &real_values,
        )
        .map_err(MeshlangTransferError::Interpolation)?;
        let imaginary = transfer_hdiv_rt0_state_checked(
            source_mesh,
            target_mesh,
            source_space,
            target_space,
            &imaginary_values,
        )
        .map_err(MeshlangTransferError::Interpolation)?;
        Some(
            real.into_iter()
                .zip(imaginary)
                .map(|(real, imaginary)| Complex64::new(real, imaginary))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let mut output = BTreeMap::new();
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let dof = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                source_values[dof]
            }
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                let dof = target_entities[&entry.entity];
                transferred
                    .as_ref()
                    .expect("validated complex RT0 transfer path")[dof]
            }
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(
                    entry.entity,
                    entry.action,
                    source_entities
                        .get(&entry.entity)
                        .map(|&dof| source_values[dof]),
                )
                .map_err(|error| MeshlangTransferError::ApplicationDefined {
                    entity: entry.entity,
                    error: error.to_string(),
                })?
            }
            IrStateTransferAction::Drop => continue,
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult {
                entity: entry.entity,
            });
        }
        output.insert(entry.entity, value);
    }
    Ok(output)
}

pub fn apply_complex_nodal_state_transfer<E, F>(
    plan: &IrStateTransferPlan,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    interpolation_data: &InterpolationData,
    source_values: &[Complex64],
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    mut initialize_or_application: F,
) -> Result<BTreeMap<EntityId, Complex64>, MeshlangTransferError>
where
    E: fmt::Display,
    F: FnMut(EntityId, IrStateTransferAction, Option<Complex64>) -> Result<Complex64, E>,
{
    plan.validate()
        .map_err(|message| MeshlangTransferError::InvalidPlan(message.into()))?;
    if source_values.len() != source_mesh.num_nodes() {
        return Err(MeshlangTransferError::SourceValueCount {
            expected: source_mesh.num_nodes(),
            actual: source_values.len(),
        });
    }
    for (&entity, &node) in source_entities {
        if node >= source_mesh.num_nodes() {
            return Err(MeshlangTransferError::InvalidSourceNode { entity, node });
        }
    }
    for (&entity, &node) in target_entities {
        if node >= target_mesh.num_nodes() {
            return Err(MeshlangTransferError::InvalidTargetNode { entity, node });
        }
    }
    for entry in &plan.entries {
        if matches!(
            entry.action,
            IrStateTransferAction::Preserve | IrStateTransferAction::ApplicationDefined
        ) {
            if let Some(&node) = source_entities.get(&entry.entity) {
                let value = source_values[node];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(MeshlangTransferError::NonFiniteSource {
                        entity: entry.entity,
                    });
                }
            } else if entry.action == IrStateTransferAction::Preserve {
                return Err(MeshlangTransferError::MissingSourceEntity(entry.entity));
            }
        }
        if matches!(
            entry.action,
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
        ) && !target_entities.contains_key(&entry.entity)
        {
            return Err(MeshlangTransferError::MissingTargetEntity(entry.entity));
        }
    }

    let has_interpolation = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Interpolate);
    let has_conservative = plan
        .entries
        .iter()
        .any(|entry| entry.action == IrStateTransferAction::Conservative);
    let interpolated = has_interpolation.then(|| {
        super::interpolate_complex_field_checked(
            interpolation_data,
            source_mesh,
            target_mesh,
            source_values,
        )
    });
    let conservative = has_conservative.then(|| {
        super::interpolate_complex_field_conservative_checked(
            interpolation_data,
            source_mesh,
            target_mesh,
            source_values,
        )
    });
    let interpolated = interpolated.transpose()?;
    let conservative = conservative.transpose()?;

    let mut resolved = Vec::with_capacity(plan.entries.len());
    for entry in &plan.entries {
        let value = match entry.action {
            IrStateTransferAction::Preserve => {
                let node = source_entities
                    .get(&entry.entity)
                    .copied()
                    .ok_or(MeshlangTransferError::MissingSourceEntity(entry.entity))?;
                Some(source_values[node])
            }
            IrStateTransferAction::Interpolate => {
                let node = target_entities[&entry.entity];
                Some(interpolated.as_ref().expect("validated interpolation path")[node])
            }
            IrStateTransferAction::Conservative => {
                let node = target_entities[&entry.entity];
                Some(conservative.as_ref().expect("validated conservative path")[node])
            }
            IrStateTransferAction::Initialize => None,
            IrStateTransferAction::ApplicationDefined => source_entities
                .get(&entry.entity)
                .map(|&node| source_values[node]),
            IrStateTransferAction::Drop => continue,
        };
        resolved.push((entry.entity, entry.action, value));
    }

    let mut output = BTreeMap::new();
    for (entity, action, value) in resolved {
        let value = match action {
            IrStateTransferAction::Initialize | IrStateTransferAction::ApplicationDefined => {
                initialize_or_application(entity, action, value).map_err(|error| {
                    MeshlangTransferError::ApplicationDefined {
                        entity,
                        error: error.to_string(),
                    }
                })?
            }
            IrStateTransferAction::Preserve
            | IrStateTransferAction::Interpolate
            | IrStateTransferAction::Conservative => value.expect("numeric action has a value"),
            IrStateTransferAction::Drop => unreachable!("drop entries are removed above"),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MeshlangTransferError::NonFiniteResult { entity });
        }
        output.insert(entity, value);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{MeshTopology, create_interpolation_data_checked, unit_square_triangles};
    use crate::spaces::{FunctionSpace, HcurlSpace};
    use meshlang_ast::{EntityDimension, IrStateTransferEntry, StateTransferResponsibility};

    fn entity(index: u64) -> EntityId {
        EntityId::new(EntityDimension::Point, index)
    }

    fn plan(entries: Vec<IrStateTransferEntry>) -> IrStateTransferPlan {
        IrStateTransferPlan {
            responsibility: StateTransferResponsibility::ApplicationDefined,
            entries,
        }
    }

    #[test]
    fn bridges_preserve_interpolate_conservative_initialize_application_and_drop() {
        let source_mesh = unit_square_triangles(1);
        let target_mesh = source_mesh.clone();
        let data = create_interpolation_data_checked(&target_mesh, &source_mesh).unwrap();
        let source_values = vec![1.0, 2.0, 3.0, 4.0];
        let source_entities = (0..source_values.len())
            .map(|node| (entity(node as u64), node))
            .collect();
        let target_entities = (0..source_values.len())
            .map(|node| (entity(node as u64), node))
            .collect();
        let plan = plan(vec![
            IrStateTransferEntry {
                entity: entity(0),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: entity(1),
                action: IrStateTransferAction::Interpolate,
            },
            IrStateTransferEntry {
                entity: entity(2),
                action: IrStateTransferAction::Conservative,
            },
            IrStateTransferEntry {
                entity: entity(3),
                action: IrStateTransferAction::Initialize,
            },
            IrStateTransferEntry {
                entity: entity(4),
                action: IrStateTransferAction::ApplicationDefined,
            },
            IrStateTransferEntry {
                entity: entity(5),
                action: IrStateTransferAction::Drop,
            },
        ]);

        let transferred = apply_nodal_state_transfer::<&'static str, _>(
            &plan,
            &source_mesh,
            &target_mesh,
            &data,
            &source_values,
            &source_entities,
            &target_entities,
            |_, action, source| match action {
                IrStateTransferAction::Initialize => Ok(42.0),
                IrStateTransferAction::ApplicationDefined => Ok(source.unwrap_or(7.0) + 1.0),
                _ => Err("unexpected callback action"),
            },
        )
        .unwrap();

        assert_eq!(transferred.get(&entity(0)), Some(&1.0));
        assert_eq!(transferred.get(&entity(1)), Some(&2.0));
        assert_eq!(transferred.get(&entity(2)), Some(&3.0));
        assert_eq!(transferred.get(&entity(3)), Some(&42.0));
        assert_eq!(transferred.get(&entity(4)), Some(&8.0));
        assert!(!transferred.contains_key(&entity(5)));
    }

    #[test]
    fn rejects_missing_preserved_entity_before_interpolation_or_callback() {
        let mesh = unit_square_triangles(1);
        let data = create_interpolation_data_checked(&mesh, &mesh).unwrap();
        let plan = plan(vec![IrStateTransferEntry {
            entity: entity(9),
            action: IrStateTransferAction::Preserve,
        }]);
        let mut callback_called = false;
        let error = apply_nodal_state_transfer::<&'static str, _>(
            &plan,
            &mesh,
            &mesh,
            &data,
            &[1.0; 4],
            &BTreeMap::new(),
            &BTreeMap::new(),
            |_, _, _| {
                callback_called = true;
                Ok(0.0)
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            MeshlangTransferError::MissingSourceEntity(found) if found == entity(9)
        ));
        assert!(!callback_called);
    }

    #[test]
    fn rejects_invalid_target_mapping_and_non_finite_callback_results() {
        let mesh = unit_square_triangles(1);
        let data = create_interpolation_data_checked(&mesh, &mesh).unwrap();
        let interpolate = plan(vec![IrStateTransferEntry {
            entity: entity(1),
            action: IrStateTransferAction::Interpolate,
        }]);
        let mut targets = BTreeMap::new();
        targets.insert(entity(1), mesh.num_nodes());
        let error = apply_nodal_state_transfer::<&'static str, _>(
            &interpolate,
            &mesh,
            &mesh,
            &data,
            &[1.0; 4],
            &BTreeMap::new(),
            &targets,
            |_, _, _| Ok(0.0),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            MeshlangTransferError::InvalidTargetNode { .. }
        ));

        let initialize = plan(vec![IrStateTransferEntry {
            entity: entity(2),
            action: IrStateTransferAction::Initialize,
        }]);
        let error = apply_nodal_state_transfer::<&'static str, _>(
            &initialize,
            &mesh,
            &mesh,
            &data,
            &[1.0; 4],
            &BTreeMap::new(),
            &BTreeMap::new(),
            |_, _, _| Ok(f64::NAN),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            MeshlangTransferError::NonFiniteResult { entity: found } if found == entity(2)
        ));
    }

    #[test]
    fn complex_bridge_preserves_both_components_and_interpolates() {
        let mesh = unit_square_triangles(1);
        let data = create_interpolation_data_checked(&mesh, &mesh).unwrap();
        let values = vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(4.0, -4.0),
        ];
        let entities = (0..values.len())
            .map(|node| (entity(node as u64), node))
            .collect::<BTreeMap<_, _>>();
        let plan = plan(vec![
            IrStateTransferEntry {
                entity: entity(0),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: entity(1),
                action: IrStateTransferAction::Interpolate,
            },
        ]);
        let transferred = apply_complex_nodal_state_transfer::<&'static str, _>(
            &plan,
            &mesh,
            &mesh,
            &data,
            &values,
            &entities,
            &entities,
            |_, _, _| Err("unexpected callback"),
        )
        .unwrap();
        assert_eq!(transferred[&entity(0)], values[0]);
        assert_eq!(transferred[&entity(1)], values[1]);
    }

    #[test]
    fn complex_order2_hcurl_bridge_matches_real_and_imaginary_transfers() {
        let mesh = unit_square_triangles(1);
        let source_space = HcurlSpace::order2();
        let target_space = HcurlSpace::order2();
        let dofs = source_space.num_global_dofs(&mesh);
        let values = (0..dofs)
            .map(|index| Complex64::new(index as f64 * 0.25, -(index as f64) * 0.5))
            .collect::<Vec<_>>();
        let entities = (0..dofs)
            .map(|dof| (entity(dof as u64), dof))
            .collect::<BTreeMap<_, _>>();
        let plan = IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Interpolate,
            entries: (0..dofs)
                .map(|dof| IrStateTransferEntry {
                    entity: entity(dof as u64),
                    action: IrStateTransferAction::Interpolate,
                })
                .collect(),
        };
        let transferred = apply_complex_hcurl_order2_triangle_state_transfer::<&'static str, _>(
            &plan,
            &mesh,
            &mesh,
            &source_space,
            &target_space,
            &values,
            &entities,
            &entities,
            |_, _, _| Err("unexpected callback"),
        )
        .expect("complex order-2 H(curl) bridge should succeed");
        for (dof, value) in values.iter().enumerate() {
            let actual = transferred[&entity(dof as u64)];
            assert!((actual.re - value.re).abs() < 1.0e-10);
            assert!((actual.im - value.im).abs() < 1.0e-10);
        }
    }

    #[test]
    fn order2_hcurl_bridge_consumes_refinement_chain_plan() {
        let level0 = unit_square_triangles(1);
        let mut level1 = level0.clone();
        let (_, history0) =
            crate::mesh::refinement::refine_elements_with_history_checked(&mut level1, &[0])
                .expect("first order-2 refinement history");
        let level1_child = level1
            .elements
            .iter()
            .position(|element| element.parent_id == Some(history0.parents[0].element.id))
            .expect("refined child");
        let mut level2 = level1.clone();
        let (_, history1) = crate::mesh::refinement::refine_elements_with_history_checked(
            &mut level2,
            &[level1_child],
        )
        .expect("second order-2 refinement history");
        let meshes = [&level0, &level1, &level2];
        let histories = [history0, history1];
        let space = HcurlSpace::order2();
        let spaces = [&space, &space, &space];
        let source_values = (0..space.num_global_dofs(&level0))
            .map(|index| 0.125 + 0.031 * index as f64 - 0.0007 * (index * index) as f64)
            .collect::<Vec<_>>();
        let source_entities = (0..source_values.len())
            .map(|dof| (entity(dof as u64), dof))
            .collect::<BTreeMap<_, _>>();
        let target_dofs = space.num_global_dofs(&level2);
        let target_entities = (0..target_dofs)
            .map(|dof| (entity(dof as u64), dof))
            .collect::<BTreeMap<_, _>>();
        let plan = IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Interpolate,
            entries: (0..target_dofs)
                .map(|dof| IrStateTransferEntry {
                    entity: entity(dof as u64),
                    action: IrStateTransferAction::Interpolate,
                })
                .collect(),
        };
        let refined =
            apply_hcurl_order2_triangle_state_transfer_with_refinement_chain::<&'static str, _>(
                &plan,
                &meshes,
                &spaces,
                &source_values,
                &histories,
                RefinementTransferDirection::Refine,
                &source_entities,
                &target_entities,
                |_, _, _| Err("unexpected chain callback"),
            )
            .expect("MeshLang order-2 chain bridge should succeed");
        assert_eq!(refined.len(), target_dofs);
        assert!(refined.values().all(|value| value.is_finite()));

        let complex_source = source_values
            .iter()
            .map(|value| Complex64::new(*value, -*value * 0.5))
            .collect::<Vec<_>>();
        let complex_refined =
            apply_complex_hcurl_order2_triangle_state_transfer_with_refinement_chain::<
                &'static str,
                _,
            >(
                &plan,
                &meshes,
                &spaces,
                &complex_source,
                &histories,
                RefinementTransferDirection::Refine,
                &source_entities,
                &target_entities,
                |_, _, _| Err("unexpected complex chain callback"),
            )
            .expect("complex MeshLang order-2 chain bridge should succeed");
        assert_eq!(complex_refined.len(), target_dofs);
        for dof in 0..target_dofs {
            let value = complex_refined[&entity(dof as u64)];
            assert!(value.re.is_finite() && value.im.is_finite());
        }
    }

    fn edge_entity(index: usize) -> EntityId {
        EntityId::new(EntityDimension::Curve, index as u64)
    }

    fn edge_moment(mesh: &Mesh, edge: (usize, usize)) -> f64 {
        let p0 = mesh.nodes[edge.0];
        let p1 = mesh.nodes[edge.1];
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x))
    }

    #[test]
    fn hcurl_bridge_transfers_interpolated_edge_state() {
        let source_mesh = unit_square_triangles(1);
        let target_mesh = unit_square_triangles(2);
        let source_space = HcurlSpace::new();
        let target_space = HcurlSpace::new();
        let source_topology = MeshTopology::build(&source_mesh);
        let target_topology = MeshTopology::build(&target_mesh);
        let source_values: Vec<_> = source_topology
            .edges
            .iter()
            .map(|&edge| edge_moment(&source_mesh, edge))
            .collect();
        let source_entities = source_topology
            .edges
            .iter()
            .enumerate()
            .map(|(index, _)| (edge_entity(index), index))
            .collect::<BTreeMap<_, _>>();
        let target_entities = target_topology
            .edges
            .iter()
            .enumerate()
            .map(|(index, _)| (edge_entity(index), index))
            .collect::<BTreeMap<_, _>>();
        let plan = IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Interpolate,
            entries: target_topology
                .edges
                .iter()
                .enumerate()
                .map(|(index, _)| IrStateTransferEntry {
                    entity: edge_entity(index),
                    action: IrStateTransferAction::Interpolate,
                })
                .collect(),
        };

        let expected = transfer_hcurl_order1_state_checked(
            &source_mesh,
            &target_mesh,
            &source_space,
            &target_space,
            &source_values,
        )
        .expect("checked H(curl) transfer should succeed");
        let transferred = apply_hcurl_order1_state_transfer::<&'static str, _>(
            &plan,
            &source_mesh,
            &target_mesh,
            &source_space,
            &target_space,
            &source_values,
            &source_entities,
            &target_entities,
            |_, _, _| Err("callback must not run for interpolation"),
        )
        .expect("MeshLang H(curl) bridge should succeed");
        for (index, value) in expected.iter().enumerate() {
            assert!((transferred[&edge_entity(index)] - value).abs() < 1.0e-10);
        }
    }

    #[test]
    fn hcurl_bridge_rejects_scalar_conservative_action() {
        let mesh = unit_square_triangles(1);
        let space = HcurlSpace::new();
        let topology = MeshTopology::build(&mesh);
        let plan = IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Conservative,
            entries: vec![IrStateTransferEntry {
                entity: edge_entity(0),
                action: IrStateTransferAction::Conservative,
            }],
        };
        let source_entities = BTreeMap::from([(edge_entity(0), 0)]);
        let target_entities = BTreeMap::from([(edge_entity(0), 0)]);
        let error = apply_hcurl_order1_state_transfer::<&'static str, _>(
            &plan,
            &mesh,
            &mesh,
            &space,
            &space,
            &vec![0.0; topology.edges.len()],
            &source_entities,
            &target_entities,
            |_, _, _| Ok(0.0),
        )
        .expect_err("conservative scalar action must not reach H(curl)");
        assert!(matches!(
            error,
            MeshlangTransferError::UnsupportedAction {
                entity,
                action: IrStateTransferAction::Conservative
            } if entity == edge_entity(0)
        ));
    }
}
