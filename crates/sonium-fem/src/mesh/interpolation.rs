//! Function interpolation between meshes
//!
//! Transfers solution data from one mesh to another via point-in-element
//! lookup and barycentric interpolation.
//!
//! # Algorithm
//!
//! 1. Build a spatial index (bounding box tree) over source mesh elements.
//! 2. For each target mesh node, find the containing source element.
//! 3. Compute barycentric coordinates for the target point in the source element.
//! 4. Evaluate the source function at the target point using shape functions.

use crate::basis::{
    PolynomialDegree, evaluate_shape, p1_triangle, p2_tetrahedron, p2_tetrahedron_grad,
    p2_triangle, p2_triangle_grad, q1_hexahedron, q1_hexahedron_grad, q1_quadrilateral,
    q1_quadrilateral_grad, q2_hexahedron, q2_hexahedron_grad, q2_quadrilateral,
    q2_quadrilateral_grad, try_evaluate_shape,
};
use crate::boundary::periodic::{MpcError, MpcSystem};
use crate::mesh::{ElementType, Mesh, Point};
use crate::quadrature::{gauss_legendre_1d, gauss_tetrahedron, gauss_triangle};
use crate::spaces::{FunctionSpace, H1Space, HcurlSpace, HdivOrder, HdivSpace};
use num_complex::Complex64;
use std::collections::HashMap;

/// Failure while constructing or applying checked mesh interpolation.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationError {
    /// A source or target mesh has invalid node/element topology.
    InvalidMesh {
        /// Mesh role that failed validation.
        mesh: &'static str,
        /// Explanation of the invalid topology or coordinate.
        reason: String,
    },
    /// The source value array does not match the source nodal layout.
    SourceValueCount { expected: usize, actual: usize },
    /// The target/dual value array does not match the target nodal layout.
    TargetValueCount { expected: usize, actual: usize },
    /// A source scalar is not finite.
    NonFiniteSource { node: usize },
    /// A target/dual scalar is not finite.
    NonFiniteTarget { node: usize },
    /// The source element-value array does not match the source element layout.
    SourceElementValueCount { expected: usize, actual: usize },
    /// A source element design value is not finite.
    NonFiniteSourceElement { element: usize },
    /// A target node lies outside the source mesh.
    TargetOutside { node: usize },
    /// A target element centroid lies outside the source mesh.
    TargetElementOutside { element: usize },
    /// Public interpolation metadata is inconsistent with the source mesh.
    InvalidData { node: usize, reason: String },
    /// Interpolation generated a non-finite result.
    NonFiniteResult { node: usize },
    /// The requested physical point contains a NaN or infinity.
    NonFinitePoint,
    /// Conservative transfer requires source and target domains to have the
    /// same positive measure.
    ConservationMismatch {
        source_measure: f64,
        target_measure: f64,
    },
    /// An adjoint inner-product weight array has the wrong length.
    AdjointWeightCount {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
    /// An adjoint inner-product weight is not finite and strictly positive.
    InvalidAdjointWeight { role: &'static str, node: usize },
    /// A refinement history is empty or inconsistent with a mesh lineage.
    RefinementHistoryMismatch { parent_id: usize, reason: String },
    /// A reduced adaptive state cannot be recovered or satisfy the target
    /// multipoint-constraint system.
    MpcStateError { role: &'static str, error: MpcError },
    /// The number of MPC systems does not match the mesh/history chain.
    MpcChainLengthMismatch { expected: usize, actual: usize },
    /// A target mesh element cannot be explained by the requested lineage.
    RefinementLineageMismatch {
        element: usize,
        direction: &'static str,
        reason: String,
    },
    /// A multi-stage refinement history chain is malformed.
    RefinementChainMismatch { stage: usize, reason: String },
}

impl std::fmt::Display for InterpolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMesh { mesh, reason } => write!(f, "invalid {mesh} mesh: {reason}"),
            Self::SourceValueCount { expected, actual } => {
                write!(f, "source values have length {actual}, expected {expected}")
            }
            Self::TargetValueCount { expected, actual } => {
                write!(f, "target values have length {actual}, expected {expected}")
            }
            Self::NonFiniteSource { node } => {
                write!(f, "source value at node {node} is not finite")
            }
            Self::NonFiniteTarget { node } => {
                write!(f, "target value at node {node} is not finite")
            }
            Self::SourceElementValueCount { expected, actual } => write!(
                f,
                "source element values have length {actual}, expected {expected}"
            ),
            Self::NonFiniteSourceElement { element } => {
                write!(
                    f,
                    "source element design value at element {element} is not finite"
                )
            }
            Self::TargetOutside { node } => {
                write!(f, "target node {node} lies outside source mesh")
            }
            Self::TargetElementOutside { element } => {
                write!(f, "target element {element} lies outside source mesh")
            }
            Self::InvalidData { node, reason } => {
                write!(
                    f,
                    "invalid interpolation data for target node {node}: {reason}"
                )
            }
            Self::NonFiniteResult { node } => {
                write!(f, "interpolated value at target node {node} is not finite")
            }
            Self::NonFinitePoint => f.write_str("interpolation point must be finite"),
            Self::ConservationMismatch {
                source_measure,
                target_measure,
            } => write!(
                f,
                "conservative transfer requires matching domain measures, source={source_measure}, target={target_measure}"
            ),
            Self::AdjointWeightCount {
                role,
                expected,
                actual,
            } => write!(
                f,
                "{role} adjoint weights have length {actual}, expected {expected}"
            ),
            Self::InvalidAdjointWeight { role, node } => write!(
                f,
                "{role} adjoint weight at node {node} must be finite and positive"
            ),
            Self::MpcStateError { role, error } => {
                write!(f, "{role} MPC state is invalid: {error}")
            }
            Self::MpcChainLengthMismatch { expected, actual } => write!(
                f,
                "MPC chain has length {actual}, expected {expected} for the mesh chain"
            ),
            Self::RefinementHistoryMismatch { parent_id, reason } => write!(
                f,
                "refinement history parent {parent_id} is inconsistent: {reason}"
            ),
            Self::RefinementLineageMismatch {
                element,
                direction,
                reason,
            } => write!(
                f,
                "{direction} refinement lineage element {element} is inconsistent: {reason}"
            ),
            Self::RefinementChainMismatch { stage, reason } => write!(
                f,
                "refinement history chain stage {stage} is inconsistent: {reason}"
            ),
        }
    }
}

impl std::error::Error for InterpolationError {}

const LOCATOR_LEAF_SIZE: usize = 8;
const LOCATOR_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone)]
struct LocatorElement {
    minimum: [f64; 3],
    maximum: [f64; 3],
    center: [f64; 3],
}

#[derive(Debug, Clone)]
enum LocatorNode {
    Leaf {
        minimum: [f64; 3],
        maximum: [f64; 3],
        elements: Vec<usize>,
    },
    Branch {
        minimum: [f64; 3],
        maximum: [f64; 3],
        left: usize,
        right: usize,
    },
}

/// Deterministic source-element locator used by all checked transfer paths.
///
/// A leaf retains element indices and point queries visit every matching leaf
/// so shared-boundary ownership remains the lowest source element index, just
/// as in the historical linear scan. This keeps the acceleration from
/// changing transfer semantics.
#[derive(Debug, Clone)]
struct ElementLocator {
    elements: Vec<LocatorElement>,
    nodes: Vec<LocatorNode>,
    root: Option<usize>,
}

impl ElementLocator {
    fn new(mesh: &Mesh) -> Self {
        let elements = (0..mesh.num_elements())
            .map(|element| {
                let (minimum, maximum) = element_bounding_box(mesh, element);
                LocatorElement {
                    minimum,
                    maximum,
                    center: [
                        minimum[0] + 0.5 * (maximum[0] - minimum[0]),
                        minimum[1] + 0.5 * (maximum[1] - minimum[1]),
                        minimum[2] + 0.5 * (maximum[2] - minimum[2]),
                    ],
                }
            })
            .collect::<Vec<_>>();
        let mut locator = Self {
            elements,
            nodes: Vec::new(),
            root: None,
        };
        if !locator.elements.is_empty() {
            let indices = (0..locator.elements.len()).collect();
            locator.root = Some(locator.build_node(indices));
        }
        locator
    }

    fn build_node(&mut self, mut indices: Vec<usize>) -> usize {
        let (minimum, maximum) = self.bounds(&indices);
        if indices.len() <= LOCATOR_LEAF_SIZE {
            let node = self.nodes.len();
            self.nodes.push(LocatorNode::Leaf {
                minimum,
                maximum,
                elements: indices,
            });
            return node;
        }

        let axis = (0..3)
            .max_by(|&left, &right| {
                (maximum[left] - minimum[left]).total_cmp(&(maximum[right] - minimum[right]))
            })
            .unwrap_or(0);
        indices.sort_by(|&left, &right| {
            self.elements[left].center[axis].total_cmp(&self.elements[right].center[axis])
        });
        let right_indices = indices.split_off(indices.len() / 2);
        let left = self.build_node(indices);
        let right = self.build_node(right_indices);
        let node = self.nodes.len();
        self.nodes.push(LocatorNode::Branch {
            minimum,
            maximum,
            left,
            right,
        });
        node
    }

    fn bounds(&self, indices: &[usize]) -> ([f64; 3], [f64; 3]) {
        let mut minimum = [f64::INFINITY; 3];
        let mut maximum = [f64::NEG_INFINITY; 3];
        for &index in indices {
            let element = &self.elements[index];
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(element.minimum[axis]);
                maximum[axis] = maximum[axis].max(element.maximum[axis]);
            }
        }
        (minimum, maximum)
    }

    fn containing_element(&self, mesh: &Mesh, point: [f64; 3]) -> Option<usize> {
        if point.iter().any(|coordinate| !coordinate.is_finite()) {
            return None;
        }
        let root = self.root?;
        let mut stack = vec![root];
        let mut result = None;
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            let (minimum, maximum) = match node {
                LocatorNode::Leaf {
                    minimum, maximum, ..
                }
                | LocatorNode::Branch {
                    minimum, maximum, ..
                } => (minimum, maximum),
            };
            if (0..3).any(|axis| {
                point[axis] < minimum[axis] - LOCATOR_TOLERANCE
                    || point[axis] > maximum[axis] + LOCATOR_TOLERANCE
            }) {
                continue;
            }
            match node {
                LocatorNode::Leaf { elements, .. } => {
                    for &element in elements {
                        let record = &self.elements[element];
                        if (0..3).any(|axis| {
                            point[axis] < record.minimum[axis] - LOCATOR_TOLERANCE
                                || point[axis] > record.maximum[axis] + LOCATOR_TOLERANCE
                        }) {
                            continue;
                        }
                        if point_in_element(mesh, element, point).is_some()
                            && result.is_none_or(|current| element < current)
                        {
                            result = Some(element);
                        }
                    }
                }
                LocatorNode::Branch { left, right, .. } => {
                    stack.push(*left);
                    stack.push(*right);
                }
            }
        }
        result
    }
}

/// Pre-computed interpolation data between a source and target mesh.
///
/// Stores the source element index and barycentric coordinates for each
/// target mesh node. Once built, this can be reused for multiple field
/// transfers between the same mesh pair.
#[derive(Debug, Clone)]
pub struct InterpolationData {
    /// For each target node: `Some((source_elem_idx, bary_coords))` or `None` if not found.
    pub target_to_source: Vec<Option<(usize, Vec<f64>)>>,
}

/// Direction of a checked transfer across an adaptive refinement operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementTransferDirection {
    /// Transfer values from the pre-refinement parent mesh to its children.
    Refine,
    /// Transfer values from a refined child mesh to restored parent elements.
    Coarsen,
}

impl RefinementTransferDirection {
    fn label(self) -> &'static str {
        match self {
            Self::Refine => "refine",
            Self::Coarsen => "coarsen",
        }
    }
}

/// Evidence returned by the checked adaptive-lineage validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementTransferReport {
    /// Number of parent snapshots covered by the history.
    pub parent_count: usize,
    /// Number of direct child elements covered by the history.
    pub child_count: usize,
    /// Number of unchanged elements retained across the transfer.
    pub retained_element_count: usize,
    /// Direction validated by the report.
    pub direction: RefinementTransferDirection,
}

/// Evidence returned by a checked multi-stage adaptive transfer validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementTransferChainReport {
    /// Direction in which the chain is traversed.
    pub direction: RefinementTransferDirection,
    /// Per-stage lineage reports, in execution order.
    pub stages: Vec<RefinementTransferReport>,
}

impl InterpolationData {
    /// Validate interpolation metadata against the source mesh and target
    /// node count before applying a state transfer.
    pub fn validate(
        &self,
        source_mesh: &Mesh,
        target_mesh: &Mesh,
    ) -> Result<(), InterpolationError> {
        validate_interpolation_mesh(source_mesh, "source")?;
        validate_interpolation_mesh(target_mesh, "target")?;
        if self.target_to_source.len() != target_mesh.num_nodes() {
            return Err(InterpolationError::InvalidData {
                node: self.target_to_source.len().min(target_mesh.num_nodes()),
                reason: format!(
                    "metadata has {} target entries, expected {}",
                    self.target_to_source.len(),
                    target_mesh.num_nodes()
                ),
            });
        }
        for (target_node, entry) in self.target_to_source.iter().enumerate() {
            let Some((element, barycentric)) = entry else {
                return Err(InterpolationError::TargetOutside { node: target_node });
            };
            let Some(source_element) = source_mesh.elements.get(*element) else {
                return Err(InterpolationError::InvalidData {
                    node: target_node,
                    reason: format!("source element {element} does not exist"),
                });
            };
            if source_element.nodes.len() < source_element.element_type.num_vertices() {
                return Err(InterpolationError::InvalidData {
                    node: target_node,
                    reason: "source element has too few node indices".into(),
                });
            }
            let vertices = source_element.vertices();
            if barycentric.len() != vertices.len()
                || barycentric.iter().any(|value| !value.is_finite())
                || barycentric.iter().any(|value| *value < -1e-8)
                || (barycentric.iter().sum::<f64>() - 1.0).abs() > 1e-7
            {
                return Err(InterpolationError::InvalidData {
                    node: target_node,
                    reason: "barycentric weights are malformed".into(),
                });
            }
            if let Some(&node) = vertices
                .iter()
                .find(|&&node| node >= source_mesh.nodes.len())
            {
                return Err(InterpolationError::InvalidData {
                    node: target_node,
                    reason: format!("source element references node {node} out of bounds"),
                });
            }
            let target = target_mesh.nodes[target_node];
            if !mapped_element_point_matches(
                source_mesh,
                source_element,
                barycentric,
                [target.x, target.y, target.z],
            ) {
                return Err(InterpolationError::InvalidData {
                    node: target_node,
                    reason: "barycentric weights do not map to target node".into(),
                });
            }
        }
        Ok(())
    }
}

/// Build interpolation metadata while rejecting malformed meshes and target
/// nodes outside the source domain.
pub fn create_interpolation_data_checked(
    target_mesh: &Mesh,
    source_mesh: &Mesh,
) -> Result<InterpolationData, InterpolationError> {
    validate_interpolation_mesh(source_mesh, "source")?;
    validate_interpolation_mesh(target_mesh, "target")?;
    let data = create_interpolation_data(target_mesh, source_mesh);
    data.validate(source_mesh, target_mesh)?;
    Ok(data)
}

/// Checked scalar state transfer between meshes.
///
/// P1/Q1 and P2/Q2 nodal layouts are supported. For a quadratic source
/// element the complete local shape-function vector is evaluated, so edge,
/// face, and interior degrees of freedom are transferred instead of being
/// silently discarded. The interpolation metadata remains barycentric/Q1
/// data for backwards compatibility.
pub fn interpolate_field_checked(
    data: &InterpolationData,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    if source_values.len() != source_mesh.num_nodes() {
        return Err(InterpolationError::SourceValueCount {
            expected: source_mesh.num_nodes(),
            actual: source_values.len(),
        });
    }
    if let Some(node) = source_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteSource { node });
    }
    data.validate(source_mesh, target_mesh)?;
    let mut output = Vec::with_capacity(target_mesh.num_nodes());
    for (target_node, (element, barycentric)) in data.target_to_source.iter().flatten().enumerate()
    {
        let source_element = &source_mesh.elements[*element];
        let weights = element_shape_weights(barycentric, source_element);
        let value = source_element
            .nodes
            .iter()
            .zip(weights)
            .map(|(&node, weight)| source_values[node] * weight)
            .sum::<f64>();
        if !value.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node: target_node });
        }
        output.push(value);
    }
    Ok(output)
}

/// Transfer a scalar H1 state between affine or explicit-P2/P3 meshes, including mixed field
/// orders. The source state is interpreted in the source space's complete
/// global DOF numbering, not as a mesh-node-only P1 vector.
///
/// Source and target geometry may use affine vertex-only or explicit P2/P3
/// isoparametric connectivity. P1, P2, and P3 field spaces are admitted
/// independently on compatible geometry through a checked inverse map.
pub fn transfer_h1_state_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "source and target H1 meshes have different dimensions".into(),
        });
    }
    // Reject unsupported/non-conforming geometry before asking H1Space to
    // construct topology. In particular, malformed explicit P3 fixtures
    // must return a checked error instead of reaching topology indexing.
    validate_h1_transfer_geometry(source_mesh, "source")?;
    validate_h1_transfer_geometry(target_mesh, "target")?;

    source_space
        .validate(source_mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: "source",
            reason: format!("H1 space validation failed: {error}"),
        })?;
    target_space
        .validate(target_mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: "target",
            reason: format!("H1 space validation failed: {error}"),
        })?;
    let source_dofs = source_space.num_global_dofs(source_mesh);
    if source_values.len() != source_dofs {
        return Err(InterpolationError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    if let Some(node) = source_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteSource { node });
    }
    let target_coordinates = target_space.dof_coordinates(target_mesh).ok_or_else(|| {
        InterpolationError::InvalidData {
            node: 0,
            reason: "target H1 space did not provide global DOF coordinates".into(),
        }
    })?;
    let source_locator = ElementLocator::new(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if target_coordinates.len() != target_dofs {
        return Err(InterpolationError::InvalidData {
            node: target_coordinates.len().min(target_dofs),
            reason: format!(
                "target H1 coordinates have length {}, expected {target_dofs}",
                target_coordinates.len()
            ),
        });
    }

    let mut target_values = Vec::with_capacity(target_dofs);
    for (target_dof, point) in target_coordinates.into_iter().enumerate() {
        if point.iter().any(|value| !value.is_finite()) {
            return Err(InterpolationError::NonFinitePoint);
        }
        let element = source_locator
            .containing_element(source_mesh, point)
            .ok_or(InterpolationError::TargetOutside { node: target_dof })?;
        let barycentric = point_in_element(source_mesh, element, point).ok_or(
            InterpolationError::InvalidData {
                node: target_dof,
                reason: "source locator returned an invalid affine element".into(),
            },
        )?;
        let (xi, eta, zeta) = match barycentric.as_slice() {
            [_, first, second] => (*first, *second, 0.0),
            [_, first, second, third] => (*first, *second, *third),
            _ => {
                return Err(InterpolationError::InvalidData {
                    node: target_dof,
                    reason: "source element returned malformed barycentric coordinates".into(),
                });
            }
        };
        let tabulation = source_space
            .try_tabulate(source_mesh, element, xi, eta, zeta)
            .map_err(|error| InterpolationError::InvalidData {
                node: target_dof,
                reason: format!("source H1 tabulation failed: {error}"),
            })?;
        let local_dofs = source_space.local_to_global(source_mesh, element);
        if local_dofs.len() != tabulation.n_dofs {
            return Err(InterpolationError::InvalidData {
                node: target_dof,
                reason: "source H1 local DOF and tabulation lengths differ".into(),
            });
        }
        let value = local_dofs
            .iter()
            .enumerate()
            .map(|(local, &global)| source_values[global] * tabulation.value(local))
            .sum::<f64>();
        if !value.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node: target_dof });
        }
        target_values.push(value);
    }
    Ok(target_values)
}

/// Transfer an element-wise design field across a checked remesh.
///
/// Design variables such as density, thickness, or material-region values are
/// piecewise constant per source element rather than nodal state. Each target
/// element is assigned the source value containing its geometric centroid.
/// This gives deterministic inheritance for first-order refinement (every
/// child receives its recorded parent value) without inventing values at new
/// nodes. The source and target domains must be compatible; a target centroid
/// outside the source mesh is reported instead of silently clamping it.
/// Apply the exact Euclidean transpose of checked H1 state transfer.
///
/// [`transfer_h1_state_checked`] constructs a linear map by evaluating the
/// source H1 basis at every target global DOF coordinate. This function
/// applies that map's transpose to target-space dual/state values. It is
/// intentionally an algebraic transpose: callers that use mass-weighted
/// inner products must apply the corresponding diagonal weights around this
/// operation. No derivative with respect to the discrete mesh topology is
/// implied.
pub fn transfer_h1_state_adjoint_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    target_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "source and target H1 meshes have different dimensions".into(),
        });
    }

    validate_h1_transfer_geometry(source_mesh, "source")?;
    validate_h1_transfer_geometry(target_mesh, "target")?;
    source_space
        .validate(source_mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: "source",
            reason: format!("H1 space validation failed: {error}"),
        })?;
    target_space
        .validate(target_mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: "target",
            reason: format!("H1 space validation failed: {error}"),
        })?;

    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    if target_values.len() != target_dofs {
        return Err(InterpolationError::TargetValueCount {
            expected: target_dofs,
            actual: target_values.len(),
        });
    }
    if let Some(node) = target_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteTarget { node });
    }

    let target_coordinates = target_space.dof_coordinates(target_mesh).ok_or_else(|| {
        InterpolationError::InvalidData {
            node: 0,
            reason: "target H1 space did not provide global DOF coordinates".into(),
        }
    })?;
    if target_coordinates.len() != target_dofs {
        return Err(InterpolationError::InvalidData {
            node: target_coordinates.len().min(target_dofs),
            reason: format!(
                "target H1 coordinates have length {}, expected {target_dofs}",
                target_coordinates.len()
            ),
        });
    }

    let source_locator = ElementLocator::new(source_mesh);
    let mut source_values = vec![0.0; source_dofs];
    for (target_dof, point) in target_coordinates.into_iter().enumerate() {
        if point.iter().any(|value| !value.is_finite()) {
            return Err(InterpolationError::NonFinitePoint);
        }
        let element = source_locator
            .containing_element(source_mesh, point)
            .ok_or(InterpolationError::TargetOutside { node: target_dof })?;
        let barycentric = point_in_element(source_mesh, element, point).ok_or(
            InterpolationError::InvalidData {
                node: target_dof,
                reason: "source locator returned invalid affine element".into(),
            },
        )?;
        let (xi, eta, zeta) = match barycentric.as_slice() {
            [_, first, second] => (*first, *second, 0.0),
            [_, first, second, third] => (*first, *second, *third),
            _ => {
                return Err(InterpolationError::InvalidData {
                    node: target_dof,
                    reason: "source element returned malformed barycentric coordinates".into(),
                });
            }
        };
        let tabulation = source_space
            .try_tabulate(source_mesh, element, xi, eta, zeta)
            .map_err(|error| InterpolationError::InvalidData {
                node: target_dof,
                reason: format!("source H1 tabulation failed: {error}"),
            })?;
        let local_dofs = source_space.local_to_global(source_mesh, element);
        if local_dofs.len() != tabulation.n_dofs {
            return Err(InterpolationError::InvalidData {
                node: target_dof,
                reason: "source H1 local DOF and tabulation lengths differ".into(),
            });
        }

        for (local, &global) in local_dofs.iter().enumerate() {
            source_values[global] += target_values[target_dof] * tabulation.value(local);
        }
    }

    if let Some(node) = source_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteResult { node });
    }
    Ok(source_values)
}

pub fn transfer_design_field_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    validate_interpolation_mesh(source_mesh, "source")?;
    validate_interpolation_mesh(target_mesh, "target")?;
    if source_values.len() != source_mesh.num_elements() {
        return Err(InterpolationError::SourceElementValueCount {
            expected: source_mesh.num_elements(),
            actual: source_values.len(),
        });
    }
    if let Some(element) = source_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteSourceElement { element });
    }

    let source_locator = ElementLocator::new(source_mesh);
    let mut output = Vec::with_capacity(target_mesh.num_elements());
    for (target_element, element) in target_mesh.elements.iter().enumerate() {
        let centroid = element_centroid(target_mesh, element)?;
        let source_element = source_locator
            .containing_element(source_mesh, centroid)
            .ok_or(InterpolationError::TargetElementOutside {
                element: target_element,
            })?;
        let value = source_values[source_element];
        if !value.is_finite() {
            return Err(InterpolationError::NonFiniteResult {
                node: target_element,
            });
        }
        output.push(value);
    }
    Ok(output)
}

/// Validate that two meshes are the direct parent/child pair described by a
/// checked mesher refinement history.
///
/// Geometric interpolation alone cannot distinguish an adaptive refinement
/// from an unrelated remesh with coincident coordinates.  This validator uses
/// the stable element IDs, `parent_id`, refinement level, element family, and
/// parent snapshots recorded by `RefinementHistory`.  It accepts unchanged
/// neighbouring elements, but rejects missing sibling families, stale
/// histories, foreign parent IDs, and arbitrary topology substitutions.
pub fn validate_refinement_transfer_lineage_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
) -> Result<RefinementTransferReport, InterpolationError> {
    validate_interpolation_mesh(source_mesh, "source")?;
    validate_interpolation_mesh(target_mesh, "target")?;
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "adaptive transfer meshes have different dimensions".into(),
        });
    }
    if history.parents.is_empty() {
        return Err(InterpolationError::RefinementHistoryMismatch {
            parent_id: 0,
            reason: "history contains no parent snapshots".into(),
        });
    }

    let source_by_id = element_id_map(source_mesh, "source")?;
    let target_by_id = element_id_map(target_mesh, "target")?;
    let mut parents_by_id = HashMap::with_capacity(history.parents.len());
    for parent in &history.parents {
        let parent_id = parent.element.id;
        if parents_by_id.insert(parent_id, parent).is_some() {
            return Err(InterpolationError::RefinementHistoryMismatch {
                parent_id,
                reason: "history contains duplicate parent IDs".into(),
            });
        }
        if direction == RefinementTransferDirection::Refine
            && parent.original_index >= source_mesh.elements.len()
        {
            return Err(InterpolationError::RefinementHistoryMismatch {
                parent_id,
                reason: "parent original index is outside the source mesh".into(),
            });
        }
    }

    let mut child_count = 0;
    for (&parent_id, parent) in &parents_by_id {
        match direction {
            RefinementTransferDirection::Refine => {
                let source_parent = source_by_id.get(&parent_id).ok_or_else(|| {
                    InterpolationError::RefinementHistoryMismatch {
                        parent_id,
                        reason: "parent snapshot is absent from the source mesh".into(),
                    }
                })?;
                if !same_element_identity(source_parent, &parent.element) {
                    return Err(InterpolationError::RefinementHistoryMismatch {
                        parent_id,
                        reason: "source parent does not match the history snapshot".into(),
                    });
                }
                let children = direct_children(target_mesh, parent_id, parent.element.level);
                validate_child_family(parent_id, &parent.element, &children, direction)?;
                child_count += children.len();
            }
            RefinementTransferDirection::Coarsen => {
                let target_parent = target_by_id.get(&parent_id).ok_or_else(|| {
                    InterpolationError::RefinementHistoryMismatch {
                        parent_id,
                        reason: "restored parent is absent from the target mesh".into(),
                    }
                })?;
                if !same_element_identity(target_parent, &parent.element) {
                    return Err(InterpolationError::RefinementHistoryMismatch {
                        parent_id,
                        reason: "target parent does not match the history snapshot".into(),
                    });
                }
                let children = direct_children(source_mesh, parent_id, parent.element.level);
                validate_child_family(parent_id, &parent.element, &children, direction)?;
                child_count += children.len();
            }
        }
    }

    let mut retained_element_count = 0;
    for (element_index, target_element) in target_mesh.elements.iter().enumerate() {
        match direction {
            RefinementTransferDirection::Refine => {
                if let Some(parent_id) = target_element.parent_id {
                    if let Some(parent) = parents_by_id.get(&parent_id) {
                        if target_element.level != parent.element.level + 1
                            || target_element.element_type != parent.element.element_type
                        {
                            return Err(lineage_error(
                                element_index,
                                direction,
                                "target child family has the wrong level or element type",
                            ));
                        }
                    } else {
                        let source_element =
                            source_by_id.get(&target_element.id).ok_or_else(|| {
                                lineage_error(
                                    element_index,
                                    direction,
                                    "target child refers to a parent outside the supplied history",
                                )
                            })?;
                        if !same_element_identity(source_element, target_element) {
                            return Err(lineage_error(
                                element_index,
                                direction,
                                "retained element with an older parent lineage was modified",
                            ));
                        }
                        retained_element_count += 1;
                    }
                } else {
                    let source_element = source_by_id.get(&target_element.id).ok_or_else(|| {
                        lineage_error(
                            element_index,
                            direction,
                            "target element is neither a recorded child nor a retained source element",
                        )
                    })?;
                    if parents_by_id.contains_key(&target_element.id)
                        || !same_element_identity(source_element, target_element)
                    {
                        return Err(lineage_error(
                            element_index,
                            direction,
                            "recorded parent was retained or an unchanged element was modified",
                        ));
                    }
                    retained_element_count += 1;
                }
            }
            RefinementTransferDirection::Coarsen => {
                if let Some(parent) = parents_by_id.get(&target_element.id) {
                    if !same_element_identity(&parent.element, target_element) {
                        return Err(lineage_error(
                            element_index,
                            direction,
                            "restored parent does not match its history snapshot",
                        ));
                    }
                } else {
                    let source_element = source_by_id.get(&target_element.id).ok_or_else(|| {
                        lineage_error(
                            element_index,
                            direction,
                            "target element is neither a restored parent nor a retained source element",
                        )
                    })?;
                    if source_element
                        .parent_id
                        .is_some_and(|parent_id| parents_by_id.contains_key(&parent_id))
                        || !same_element_identity(source_element, target_element)
                    {
                        return Err(lineage_error(
                            element_index,
                            direction,
                            "a child was retained or an unchanged element was modified",
                        ));
                    }
                    retained_element_count += 1;
                }
            }
        }
    }

    Ok(RefinementTransferReport {
        parent_count: parents_by_id.len(),
        child_count,
        retained_element_count,
        direction,
    })
}

/// Validate a sequence of adjacent adaptive refinement operations.
///
/// The meshes are ordered from coarse to fine.  Refinement walks the supplied
/// histories from left to right; coarsening walks them from right to left.
/// Each history is checked against its own adjacent mesh pair, so descendants
/// cannot be mistaken for direct children of an earlier stage.
pub fn validate_refinement_transfer_chain_checked(
    meshes: &[&Mesh],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
) -> Result<RefinementTransferChainReport, InterpolationError> {
    if meshes.len() < 2 {
        return Err(InterpolationError::RefinementChainMismatch {
            stage: 0,
            reason: "a chain requires at least one adjacent mesh pair".into(),
        });
    }
    if histories.len() + 1 != meshes.len() {
        return Err(InterpolationError::RefinementChainMismatch {
            stage: histories.len().min(meshes.len().saturating_sub(1)),
            reason: format!(
                "{} histories require {} meshes, received {}",
                histories.len(),
                histories.len() + 1,
                meshes.len()
            ),
        });
    }

    let mut stages = Vec::with_capacity(histories.len());
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                let report = validate_refinement_transfer_lineage_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    &histories[stage],
                    direction,
                )
                .map_err(|error| InterpolationError::RefinementChainMismatch {
                    stage,
                    reason: error.to_string(),
                })?;
                stages.push(report);
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                let report = validate_refinement_transfer_lineage_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    &histories[stage],
                    direction,
                )
                .map_err(|error| InterpolationError::RefinementChainMismatch {
                    stage,
                    reason: error.to_string(),
                })?;
                stages.push(report);
            }
        }
    }

    Ok(RefinementTransferChainReport { direction, stages })
}

/// Transfer a scalar H1 state through a checked multi-stage lineage chain.
pub fn transfer_h1_state_with_refinement_chain_checked(
    meshes: &[&Mesh],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;
    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_h1_state_with_refinement_history_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_h1_state_with_refinement_history_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
    }
    Ok(values)
}

fn validate_refinement_chain_space_count(
    meshes: &[&Mesh],
    space_count: usize,
    space_name: &'static str,
) -> Result<(), InterpolationError> {
    if space_count != meshes.len() {
        return Err(InterpolationError::RefinementChainMismatch {
            stage: 0,
            reason: format!(
                "{space_name} chain has {space_count} spaces, expected {}",
                meshes.len()
            ),
        });
    }
    Ok(())
}

/// Transfer an order-1 H(curl) state through a checked multi-stage lineage
/// chain, preserving topology-owned edge moments at every stage.
pub fn transfer_hcurl_order1_state_with_refinement_chain_checked(
    meshes: &[&Mesh],
    spaces: &[&HcurlSpace],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_chain_space_count(meshes, spaces.len(), "H(curl)")?;
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;
    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_hcurl_order1_state_with_refinement_history_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    spaces[stage],
                    spaces[stage + 1],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_hcurl_order1_state_with_refinement_history_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    spaces[stage + 1],
                    spaces[stage],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
    }
    Ok(values)
}

/// Transfer an order-2 H(curl) state through a checked multi-stage lineage
/// chain on affine triangular meshes.
///
/// Each stage validates its explicit refinement history before evaluating the
/// order-2 edge moments and interior-bubble projection. Coarsening walks the
/// same histories in reverse, so callers cannot apply a descendant history to
/// the wrong mesh pair.
pub fn transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
    meshes: &[&Mesh],
    spaces: &[&HcurlSpace],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_chain_space_count(meshes, spaces.len(), "order-2 H(curl)")?;
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;

    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_hcurl_order2_triangle_state_with_refinement_history_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    spaces[stage],
                    spaces[stage + 1],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_hcurl_order2_triangle_state_with_refinement_history_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    spaces[stage + 1],
                    spaces[stage],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
    }
    Ok(values)
}

/// Transfer an order-2 H(curl) state across one checked adaptive refinement
/// or coarsening operation on affine triangular meshes.
pub fn transfer_hcurl_order2_triangle_state_with_refinement_history_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_transfer_lineage_checked(source_mesh, target_mesh, history, direction)?;
    transfer_hcurl_order2_triangle_state_checked(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        source_values,
    )
}

/// Transfer an RT0 H(div) state through a checked multi-stage lineage chain.
pub fn transfer_hdiv_rt0_state_with_refinement_chain_checked(
    meshes: &[&Mesh],
    spaces: &[&HdivSpace],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_chain_space_count(meshes, spaces.len(), "RT0 H(div)")?;
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;
    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_hdiv_rt0_state_with_refinement_history_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    spaces[stage],
                    spaces[stage + 1],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_hdiv_rt0_state_with_refinement_history_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    spaces[stage + 1],
                    spaces[stage],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
    }
    Ok(values)
}

/// Transfer an RT1 H(div) state through a checked multi-stage lineage chain.
pub fn transfer_hdiv_rt1_state_with_refinement_chain_checked(
    meshes: &[&Mesh],
    spaces: &[&HdivSpace],
    source_values: &[f64],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_chain_space_count(meshes, spaces.len(), "RT1 H(div)")?;
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;
    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_hdiv_rt1_state_with_refinement_history_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    spaces[stage],
                    spaces[stage + 1],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_hdiv_rt1_state_with_refinement_history_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    spaces[stage + 1],
                    spaces[stage],
                    &values,
                    &histories[stage],
                    direction,
                )?;
            }
        }
    }
    Ok(values)
}

/// Transfer a scalar H1 nodal state across a checked adaptive refinement or
/// coarsening operation.
pub fn transfer_h1_state_with_refinement_history_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_transfer_lineage_checked(source_mesh, target_mesh, history, direction)?;
    let data = create_interpolation_data_checked(target_mesh, source_mesh)?;
    interpolate_field_checked(&data, source_mesh, target_mesh, source_values)
}

/// Transfer an order-1 H(curl) state across a checked adaptive refinement or
/// coarsening operation, preserving the topology-owned edge moments.
pub fn transfer_hcurl_order1_state_with_refinement_history_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_transfer_lineage_checked(source_mesh, target_mesh, history, direction)?;
    transfer_hcurl_order1_state_checked(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        source_values,
    )
}

/// Transfer an RT0 H(div) state across a checked adaptive refinement or
/// coarsening operation, preserving oriented normal-flux moments.
pub fn transfer_hdiv_rt0_state_with_refinement_history_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_transfer_lineage_checked(source_mesh, target_mesh, history, direction)?;
    transfer_hdiv_rt0_state_checked(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        source_values,
    )
}

/// Transfer an RT1 H(div) state across a checked adaptive refinement or
/// coarsening operation, preserving edge/face and cell moments.
pub fn transfer_hdiv_rt1_state_with_refinement_history_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
) -> Result<Vec<f64>, InterpolationError> {
    validate_refinement_transfer_lineage_checked(source_mesh, target_mesh, history, direction)?;
    transfer_hdiv_rt1_state_checked(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        source_values,
    )
}

fn transfer_real_state_with_mpc_checked(
    source_mpc: &MpcSystem,
    target_mpc: &MpcSystem,
    source_values: &[f64],
    constraint_tolerance: f64,
    transfer: impl FnOnce(&[f64]) -> Result<Vec<f64>, InterpolationError>,
) -> Result<Vec<f64>, InterpolationError> {
    let source_full = source_mpc.recover_real(source_values).map_err(|error| {
        InterpolationError::MpcStateError {
            role: "source",
            error,
        }
    })?;
    let target_full = transfer(&source_full)?;
    target_mpc
        .restrict_real_checked(&target_full, constraint_tolerance)
        .map_err(|error| InterpolationError::MpcStateError {
            role: "target",
            error,
        })
}

/// Transfer a reduced H1 state across checked adaptive lineage while applying
/// source and target real MPC systems around the full-space transfer.
pub fn transfer_h1_state_with_refinement_history_and_mpc_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[f64],
    source_mpc: &MpcSystem,
    target_mpc: &MpcSystem,
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    constraint_tolerance: f64,
) -> Result<Vec<f64>, InterpolationError> {
    transfer_real_state_with_mpc_checked(
        source_mpc,
        target_mpc,
        source_values,
        constraint_tolerance,
        |full_values| {
            transfer_h1_state_with_refinement_history_checked(
                source_mesh,
                target_mesh,
                full_values,
                history,
                direction,
            )
        },
    )
}

/// Transfer a reduced order-1 H(curl) state across checked adaptive lineage
/// while applying source and target real MPC systems.
#[expect(
    clippy::too_many_arguments,
    reason = "checked state-transfer entry point threading source/target meshes, lineage and MPC systems; arity is the stable checked-call contract"
)]
pub fn transfer_hcurl_order1_state_with_refinement_history_and_mpc_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
    source_mpc: &MpcSystem,
    target_mpc: &MpcSystem,
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    constraint_tolerance: f64,
) -> Result<Vec<f64>, InterpolationError> {
    transfer_real_state_with_mpc_checked(
        source_mpc,
        target_mpc,
        source_values,
        constraint_tolerance,
        |full_values| {
            transfer_hcurl_order1_state_with_refinement_history_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                full_values,
                history,
                direction,
            )
        },
    )
}

/// Transfer a reduced RT0 H(div) state across checked adaptive lineage while
/// applying source and target real MPC systems.
#[expect(
    clippy::too_many_arguments,
    reason = "checked state-transfer entry point threading source/target meshes, lineage and MPC systems; arity is the stable checked-call contract"
)]
pub fn transfer_hdiv_rt0_state_with_refinement_history_and_mpc_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
    source_mpc: &MpcSystem,
    target_mpc: &MpcSystem,
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    constraint_tolerance: f64,
) -> Result<Vec<f64>, InterpolationError> {
    transfer_real_state_with_mpc_checked(
        source_mpc,
        target_mpc,
        source_values,
        constraint_tolerance,
        |full_values| {
            transfer_hdiv_rt0_state_with_refinement_history_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                full_values,
                history,
                direction,
            )
        },
    )
}

/// Transfer a reduced RT1 H(div) state across checked adaptive lineage while
/// applying source and target real MPC systems.
#[expect(
    clippy::too_many_arguments,
    reason = "checked state-transfer entry point threading source/target meshes, lineage and MPC systems; arity is the stable checked-call contract"
)]
pub fn transfer_hdiv_rt1_state_with_refinement_history_and_mpc_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
    source_mpc: &MpcSystem,
    target_mpc: &MpcSystem,
    history: &crate::mesh::refinement::RefinementHistory,
    direction: RefinementTransferDirection,
    constraint_tolerance: f64,
) -> Result<Vec<f64>, InterpolationError> {
    transfer_real_state_with_mpc_checked(
        source_mpc,
        target_mpc,
        source_values,
        constraint_tolerance,
        |full_values| {
            transfer_hdiv_rt1_state_with_refinement_history_checked(
                source_mesh,
                target_mesh,
                source_space,
                target_space,
                full_values,
                history,
                direction,
            )
        },
    )
}

fn validate_mpc_chain_length(
    mesh_count: usize,
    mpc_count: usize,
) -> Result<(), InterpolationError> {
    if mesh_count != mpc_count {
        return Err(InterpolationError::MpcChainLengthMismatch {
            expected: mesh_count,
            actual: mpc_count,
        });
    }
    Ok(())
}

/// Transfer a reduced H1 state through a checked adaptive lineage chain with
/// one real MPC system per mesh stage.
pub fn transfer_h1_state_with_refinement_chain_and_mpc_checked(
    meshes: &[&Mesh],
    source_values: &[f64],
    mpcs: &[&MpcSystem],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    constraint_tolerance: f64,
) -> Result<Vec<f64>, InterpolationError> {
    validate_mpc_chain_length(meshes.len(), mpcs.len())?;
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;
    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_h1_state_with_refinement_history_and_mpc_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    &values,
                    mpcs[stage],
                    mpcs[stage + 1],
                    &histories[stage],
                    direction,
                    constraint_tolerance,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_h1_state_with_refinement_history_and_mpc_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    &values,
                    mpcs[stage + 1],
                    mpcs[stage],
                    &histories[stage],
                    direction,
                    constraint_tolerance,
                )?;
            }
        }
    }
    Ok(values)
}

/// Transfer a reduced order-1 H(curl) state through a checked adaptive
/// lineage chain with one real MPC system per mesh stage.
pub fn transfer_hcurl_order1_state_with_refinement_chain_and_mpc_checked(
    meshes: &[&Mesh],
    spaces: &[&HcurlSpace],
    source_values: &[f64],
    mpcs: &[&MpcSystem],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    constraint_tolerance: f64,
) -> Result<Vec<f64>, InterpolationError> {
    validate_mpc_chain_length(meshes.len(), mpcs.len())?;
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;
    if spaces.len() != meshes.len() {
        return Err(InterpolationError::InvalidData {
            node: spaces.len(),
            reason: format!(
                "H(curl) space chain has length {}, expected {}",
                spaces.len(),
                meshes.len()
            ),
        });
    }
    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_hcurl_order1_state_with_refinement_history_and_mpc_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    spaces[stage],
                    spaces[stage + 1],
                    &values,
                    mpcs[stage],
                    mpcs[stage + 1],
                    &histories[stage],
                    direction,
                    constraint_tolerance,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_hcurl_order1_state_with_refinement_history_and_mpc_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    spaces[stage + 1],
                    spaces[stage],
                    &values,
                    mpcs[stage + 1],
                    mpcs[stage],
                    &histories[stage],
                    direction,
                    constraint_tolerance,
                )?;
            }
        }
    }
    Ok(values)
}

/// Transfer a reduced RT0 H(div) state through a checked adaptive lineage
/// chain with one real MPC system per mesh stage.
pub fn transfer_hdiv_rt0_state_with_refinement_chain_and_mpc_checked(
    meshes: &[&Mesh],
    spaces: &[&HdivSpace],
    source_values: &[f64],
    mpcs: &[&MpcSystem],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    constraint_tolerance: f64,
) -> Result<Vec<f64>, InterpolationError> {
    validate_mpc_chain_length(meshes.len(), mpcs.len())?;
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;
    if spaces.len() != meshes.len() {
        return Err(InterpolationError::InvalidData {
            node: spaces.len(),
            reason: format!(
                "RT0 space chain has length {}, expected {}",
                spaces.len(),
                meshes.len()
            ),
        });
    }
    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_hdiv_rt0_state_with_refinement_history_and_mpc_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    spaces[stage],
                    spaces[stage + 1],
                    &values,
                    mpcs[stage],
                    mpcs[stage + 1],
                    &histories[stage],
                    direction,
                    constraint_tolerance,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_hdiv_rt0_state_with_refinement_history_and_mpc_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    spaces[stage + 1],
                    spaces[stage],
                    &values,
                    mpcs[stage + 1],
                    mpcs[stage],
                    &histories[stage],
                    direction,
                    constraint_tolerance,
                )?;
            }
        }
    }
    Ok(values)
}

/// Transfer a reduced RT1 H(div) state through a checked adaptive lineage
/// chain with one real MPC system per mesh stage.
pub fn transfer_hdiv_rt1_state_with_refinement_chain_and_mpc_checked(
    meshes: &[&Mesh],
    spaces: &[&HdivSpace],
    source_values: &[f64],
    mpcs: &[&MpcSystem],
    histories: &[crate::mesh::refinement::RefinementHistory],
    direction: RefinementTransferDirection,
    constraint_tolerance: f64,
) -> Result<Vec<f64>, InterpolationError> {
    validate_mpc_chain_length(meshes.len(), mpcs.len())?;
    validate_refinement_transfer_chain_checked(meshes, histories, direction)?;
    if spaces.len() != meshes.len() {
        return Err(InterpolationError::InvalidData {
            node: spaces.len(),
            reason: format!(
                "RT1 space chain has length {}, expected {}",
                spaces.len(),
                meshes.len()
            ),
        });
    }
    let mut values = source_values.to_vec();
    match direction {
        RefinementTransferDirection::Refine => {
            for stage in 0..histories.len() {
                values = transfer_hdiv_rt1_state_with_refinement_history_and_mpc_checked(
                    meshes[stage],
                    meshes[stage + 1],
                    spaces[stage],
                    spaces[stage + 1],
                    &values,
                    mpcs[stage],
                    mpcs[stage + 1],
                    &histories[stage],
                    direction,
                    constraint_tolerance,
                )?;
            }
        }
        RefinementTransferDirection::Coarsen => {
            for stage in (0..histories.len()).rev() {
                values = transfer_hdiv_rt1_state_with_refinement_history_and_mpc_checked(
                    meshes[stage + 1],
                    meshes[stage],
                    spaces[stage + 1],
                    spaces[stage],
                    &values,
                    mpcs[stage + 1],
                    mpcs[stage],
                    &histories[stage],
                    direction,
                    constraint_tolerance,
                )?;
            }
        }
    }
    Ok(values)
}

fn element_id_map<'a>(
    mesh: &'a Mesh,
    role: &'static str,
) -> Result<HashMap<usize, &'a crate::mesh::Element>, InterpolationError> {
    let mut by_id = HashMap::with_capacity(mesh.elements.len());
    for element in &mesh.elements {
        if by_id.insert(element.id, element).is_some() {
            return Err(InterpolationError::InvalidMesh {
                mesh: role,
                reason: format!("element ID {} occurs more than once", element.id),
            });
        }
    }
    Ok(by_id)
}

fn direct_children(
    mesh: &Mesh,
    parent_id: usize,
    parent_level: usize,
) -> Vec<&crate::mesh::Element> {
    mesh.elements
        .iter()
        .filter(|element| element.parent_id == Some(parent_id) && element.level == parent_level + 1)
        .collect()
}

fn validate_child_family(
    parent_id: usize,
    parent: &crate::mesh::Element,
    children: &[&crate::mesh::Element],
    direction: RefinementTransferDirection,
) -> Result<(), InterpolationError> {
    let expected = match parent.element_type {
        crate::mesh::ElementType::Triangle | crate::mesh::ElementType::Quadrilateral => 4,
        crate::mesh::ElementType::Tetrahedron | crate::mesh::ElementType::Hexahedron => 8,
    };
    if children.len() != expected {
        return Err(InterpolationError::RefinementHistoryMismatch {
            parent_id,
            reason: format!(
                "{} direction has {} direct children, expected {}",
                direction.label(),
                children.len(),
                expected
            ),
        });
    }
    if children
        .iter()
        .any(|child| child.element_type != parent.element_type || child.level != parent.level + 1)
    {
        return Err(InterpolationError::RefinementHistoryMismatch {
            parent_id,
            reason: "direct children have inconsistent type or level".into(),
        });
    }
    Ok(())
}

fn same_element_identity(left: &crate::mesh::Element, right: &crate::mesh::Element) -> bool {
    left.id == right.id
        && left.element_type == right.element_type
        && left.nodes == right.nodes
        && left.parent_id == right.parent_id
        && left.level == right.level
        && left.marker == right.marker
}

fn lineage_error(
    element: usize,
    direction: RefinementTransferDirection,
    reason: &str,
) -> InterpolationError {
    InterpolationError::RefinementLineageMismatch {
        element,
        direction: direction.label(),
        reason: reason.into(),
    }
}

/// Checked complex state transfer for harmonic FEM fields.
///
/// As with [`interpolate_field_checked`], quadratic source elements use all
/// local P2/Q2 shape functions while preserving the historical P1 metadata
/// representation.
pub fn interpolate_complex_field_checked(
    data: &InterpolationData,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[Complex64],
) -> Result<Vec<Complex64>, InterpolationError> {
    if source_values.len() != source_mesh.num_nodes() {
        return Err(InterpolationError::SourceValueCount {
            expected: source_mesh.num_nodes(),
            actual: source_values.len(),
        });
    }
    if let Some(node) = source_values
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(InterpolationError::NonFiniteSource { node });
    }
    data.validate(source_mesh, target_mesh)?;
    let mut output = Vec::with_capacity(target_mesh.num_nodes());
    for (target_node, (element, barycentric)) in data.target_to_source.iter().flatten().enumerate()
    {
        let source_element = &source_mesh.elements[*element];
        let weights = element_shape_weights(barycentric, source_element);
        let value = source_element
            .nodes
            .iter()
            .zip(weights)
            .map(|(&node, weight)| source_values[node] * weight)
            .sum::<Complex64>();
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node: target_node });
        }
        output.push(value);
    }
    Ok(output)
}

/// Transfer a lowest-order H(curl) edge state from a source mesh to a
/// refinement-compatible target mesh.
///
/// The state entries are global edge line moments in the canonical topology
/// orientation. Each target edge moment is reconstructed by integrating the
/// source finite-element field with Gauss quadrature. This is a checked
/// adaptive-transfer primitive: it admits P1/P2 triangles/tetrahedra and
/// order-1 H(curl); mixed element families and order-2 mode transfer fail
/// closed until their orientation maps are specified.
pub fn transfer_hcurl_order1_state_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    validate_hcurl_transfer_mesh(source_mesh, source_space, "source")?;
    validate_hcurl_transfer_mesh(target_mesh, target_space, "target")?;
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "source and target H(curl) meshes have different dimensions".into(),
        });
    }

    let source_dofs = source_space.num_global_dofs(source_mesh);
    if source_values.len() != source_dofs {
        return Err(InterpolationError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    if let Some((dof, _)) = source_values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(InterpolationError::NonFiniteSource { node: dof });
    }

    let source_locator = ElementLocator::new(source_mesh);
    let target_topology = crate::mesh::topology::MeshTopology::build(target_mesh);
    let mut target_values = vec![0.0; target_space.num_global_dofs(target_mesh)];
    let quadrature = gauss_legendre_1d(8);

    for (edge_index, &(start, end)) in target_topology.edges.iter().enumerate() {
        let (p0, pm, p1) = edge_geometry_points(target_mesh, (start, end)).ok_or(
            InterpolationError::InvalidData {
                node: edge_index,
                reason: "target edge is not owned by a supported simplex element".into(),
            },
        )?;
        let mut moment: f64 = 0.0;
        for quadrature_point in &quadrature {
            let coordinate = quadrature_point.xi();
            let weight = quadrature_point.weight;
            let point = edge_curve_point(p0, pm, p1, coordinate);
            let source_element =
                source_element_for_point(&source_locator, source_mesh, point, edge_index)?;
            let barycentric = point_in_element(source_mesh, source_element, point).ok_or(
                InterpolationError::TargetElementOutside {
                    element: edge_index,
                },
            )?;
            let (xi, eta, zeta) = match barycentric.as_slice() {
                [_, first, second] => (*first, *second, 0.0),
                [_, first, second, third] => (*first, *second, *third),
                _ => {
                    return Err(InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "source simplex returned malformed barycentric coordinates".into(),
                    });
                }
            };
            let tabulation = source_space
                .try_tabulate(source_mesh, source_element, xi, eta, zeta)
                .map_err(|error| InterpolationError::InvalidData {
                    node: edge_index,
                    reason: format!("source H(curl) tabulation failed: {error}"),
                })?;
            let local_dofs = source_space.local_to_global(source_mesh, source_element);
            let mut field = [0.0; 3];
            for (local, &global) in local_dofs.iter().enumerate() {
                let value = source_values
                    .get(global)
                    .ok_or(InterpolationError::InvalidData {
                        node: edge_index,
                        reason: format!("source local DOF {global} is outside the state vector"),
                    })?;
                for (component, slot) in field[..tabulation.spatial_dim].iter_mut().enumerate() {
                    *slot += *value * tabulation.vector_value(local, component);
                }
            }
            let tangent = edge_curve_tangent(p0, pm, p1, coordinate);
            moment +=
                weight * (field[0] * tangent[0] + field[1] * tangent[1] + field[2] * tangent[2]);
        }
        if !moment.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node: edge_index });
        }
        target_values[edge_index] = moment;
    }

    Ok(target_values)
}

/// Transfer a second-order Nédélec field across supported triangular meshes.
///
/// The order-2 triangle has two canonical moments on every global edge and
/// two element-interior bubble DOFs.  Edge moments are reconstructed first
/// from the physical tangential trace; the interior coefficients are then
/// obtained from a checked two-by-two physical L2 projection after removing
/// the already reconstructed edge contribution.  This keeps the transfer
/// independent of element ownership on shared edges and exactly reproduces
/// any field in the target order-2 space.
///
/// Affine P1 and explicit P2 isoparametric triangle geometry are admitted.
/// Quadrilateral elements, tetrahedra, and unsupported mixed-order layouts
/// remain outside this field-specific transfer contract.  Callers that need those
/// layouts must select a field-specific transfer policy instead of silently
/// treating scalar nodal interpolation as an H(curl) transfer.
pub fn transfer_hcurl_order2_triangle_state_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    validate_hcurl_order2_triangle_transfer_mesh(source_mesh, source_space, "source")?;
    validate_hcurl_order2_triangle_transfer_mesh(target_mesh, target_space, "target")?;
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "source and target H(curl) meshes have different dimensions".into(),
        });
    }

    let source_dofs = source_space.num_global_dofs(source_mesh);
    if source_values.len() != source_dofs {
        return Err(InterpolationError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    if let Some((node, _)) = source_values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(InterpolationError::NonFiniteSource { node });
    }

    let source_locator = ElementLocator::new(source_mesh);
    let target_topology = crate::mesh::topology::MeshTopology::build(target_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    let mut target_values = vec![0.0; target_dofs];

    // The edge coordinate is the standard Gauss-Legendre coordinate
    // s in [-1, 1].  For the canonical second edge moment the trace weight
    // is -s = lambda_start - lambda_end.
    let edge_quadrature = gauss_legendre_1d(8);
    for (edge_index, &(start, end)) in target_topology.edges.iter().enumerate() {
        let (p0, pm, p1) = edge_geometry_points(target_mesh, (start, end)).ok_or(
            InterpolationError::InvalidData {
                node: edge_index,
                reason: "target edge is not owned by a supported triangle".into(),
            },
        )?;
        let mut moment0 = 0.0;
        let mut moment1 = 0.0;
        for quadrature_point in &edge_quadrature {
            let coordinate = quadrature_point.xi();
            let point = edge_curve_point(p0, pm, p1, coordinate);
            let field = evaluate_hcurl_order2_field_at_point(
                &source_locator,
                source_mesh,
                source_space,
                source_values,
                point,
                edge_index,
            )?;
            let tangent = edge_curve_tangent(p0, pm, p1, coordinate);
            let tangential = field[0] * tangent[0] + field[1] * tangent[1];
            moment0 += quadrature_point.weight * tangential;
            moment1 += quadrature_point.weight * (-coordinate) * tangential;
        }
        if !moment0.is_finite() || !moment1.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node: edge_index });
        }
        let first = edge_index * 2;
        target_values[first] = moment0;
        // Integral((-s)^2, s=-1..1, with the edge Jacobian) = 1/3.
        target_values[first + 1] = 3.0 * moment1;
    }

    // Interior bubbles are local to each triangle.  Project the residual
    // against the two target bubble functions and solve the resulting SPD
    // 2x2 Gram system.  Order-2 basis values are physical H(curl) values, so
    // the tabulation determinant supplies the physical integration measure.
    let element_quadrature = gauss_triangle(8);
    for (element_index, _element) in target_mesh.elements.iter().enumerate() {
        let local_dofs = target_space.local_to_global(target_mesh, element_index);
        let mut gram = [[0.0_f64; 2]; 2];
        let mut rhs = [0.0_f64; 2];
        for quadrature_point in &element_quadrature {
            let xi = quadrature_point.xi();
            let eta = quadrature_point.eta();
            let target_tabulation = target_space
                .try_tabulate(target_mesh, element_index, xi, eta, 0.0)
                .map_err(|error| InterpolationError::InvalidData {
                    node: element_index,
                    reason: format!("target H(curl) tabulation failed: {error}"),
                })?;
            let field = evaluate_hcurl_order2_field_at_point(
                &source_locator,
                source_mesh,
                source_space,
                source_values,
                target_tabulation.physical_coords,
                element_index,
            )?;
            let mut residual = field;
            for (local, &global) in local_dofs.iter().take(6).enumerate() {
                let value =
                    target_values
                        .get(global)
                        .copied()
                        .ok_or(InterpolationError::InvalidData {
                            node: element_index,
                            reason: format!("target local DOF {global} outside state vector"),
                        })?;
                residual[0] -= value * target_tabulation.vector_value(local, 0);
                residual[1] -= value * target_tabulation.vector_value(local, 1);
            }
            let weight = quadrature_point.weight * target_tabulation.det_j;
            for a in 0..2 {
                let bubble_a = 6 + a;
                let ax = target_tabulation.vector_value(bubble_a, 0);
                let ay = target_tabulation.vector_value(bubble_a, 1);
                rhs[a] += weight * (residual[0] * ax + residual[1] * ay);
                for (b, slot) in gram[a].iter_mut().enumerate() {
                    let bubble_b = 6 + b;
                    *slot += weight
                        * (ax * target_tabulation.vector_value(bubble_b, 0)
                            + ay * target_tabulation.vector_value(bubble_b, 1));
                }
            }
        }
        let determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
        if !determinant.is_finite() || determinant.abs() <= 1.0e-14 {
            return Err(InterpolationError::InvalidData {
                node: element_index,
                reason: "order-2 H(curl) interior bubble Gram matrix is singular".into(),
            });
        }
        let first = (gram[1][1] * rhs[0] - gram[0][1] * rhs[1]) / determinant;
        let second = (gram[0][0] * rhs[1] - gram[0][1] * rhs[0]) / determinant;
        let interior = &local_dofs[6..8];
        if !first.is_finite() || !second.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node: interior[0] });
        }
        target_values[interior[0]] = first;
        target_values[interior[1]] = second;
    }

    Ok(target_values)
}

/// Transfer a triangular H(curl) state between order-1 and order-2 spaces.
///
/// The source and target may use different Nédélec orders, but both must be
/// supported triangular spaces on 2-D P1/P2 geometry. Target edge moments are
/// always reconstructed from the physical tangential trace; order-2 targets
/// additionally receive the checked two-bubble L2 projection used by the
/// native order-2 transfer. This keeps mixed-order MeshLang consumers on the
/// same checked geometry and finite-value path as same-order transfers.
pub fn transfer_hcurl_mixed_triangle_state_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HcurlSpace,
    target_space: &HcurlSpace,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    validate_hcurl_mixed_triangle_transfer_mesh(source_mesh, source_space, "source")?;
    validate_hcurl_mixed_triangle_transfer_mesh(target_mesh, target_space, "target")?;
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "source and target H(curl) meshes have different dimensions".into(),
        });
    }

    let source_dofs = source_space.num_global_dofs(source_mesh);
    if source_values.len() != source_dofs {
        return Err(InterpolationError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    if let Some((dof, _)) = source_values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(InterpolationError::NonFiniteSource { node: dof });
    }

    let source_locator = ElementLocator::new(source_mesh);
    let target_topology = crate::mesh::topology::MeshTopology::build(target_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    let mut target_values = vec![0.0; target_dofs];
    let edge_quadrature = gauss_legendre_1d(8);

    for (edge_index, &(start, end)) in target_topology.edges.iter().enumerate() {
        let (p0, pm, p1) = edge_geometry_points(target_mesh, (start, end)).ok_or(
            InterpolationError::InvalidData {
                node: edge_index,
                reason: "target edge is not owned by a supported triangle".into(),
            },
        )?;
        let mut moment0 = 0.0;
        let mut moment1 = 0.0;
        for quadrature_point in &edge_quadrature {
            let coordinate = quadrature_point.xi();
            let point = edge_curve_point(p0, pm, p1, coordinate);
            let field = evaluate_hcurl_field_at_point(
                &source_locator,
                source_mesh,
                source_space,
                source_values,
                point,
                edge_index,
            )?;
            let tangent = edge_curve_tangent(p0, pm, p1, coordinate);
            let tangential = field[0] * tangent[0] + field[1] * tangent[1];
            moment0 += quadrature_point.weight * tangential;
            if target_space.order == 2 {
                moment1 += quadrature_point.weight * (-coordinate) * tangential;
            }
        }
        if !moment0.is_finite() || !moment1.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node: edge_index });
        }
        if target_space.order == 1 {
            target_values[edge_index] = moment0;
        } else {
            let first = edge_index * 2;
            target_values[first] = moment0;
            target_values[first + 1] = 3.0 * moment1;
        }
    }

    if target_space.order == 2 {
        let element_quadrature = gauss_triangle(8);
        for (element_index, _element) in target_mesh.elements.iter().enumerate() {
            let local_dofs = target_space.local_to_global(target_mesh, element_index);
            if local_dofs.len() != 8 {
                return Err(InterpolationError::InvalidData {
                    node: element_index,
                    reason: "target mixed-order H(curl) triangle must expose 8 DOFs".into(),
                });
            }
            let mut gram = [[0.0_f64; 2]; 2];
            let mut rhs = [0.0_f64; 2];
            for quadrature_point in &element_quadrature {
                let target_tabulation = target_space
                    .try_tabulate(
                        target_mesh,
                        element_index,
                        quadrature_point.xi(),
                        quadrature_point.eta(),
                        0.0,
                    )
                    .map_err(|error| InterpolationError::InvalidData {
                        node: element_index,
                        reason: format!("target mixed-order H(curl) tabulation failed: {error}"),
                    })?;
                let field = evaluate_hcurl_field_at_point(
                    &source_locator,
                    source_mesh,
                    source_space,
                    source_values,
                    target_tabulation.physical_coords,
                    element_index,
                )?;
                let mut residual = field;
                for (local, &global) in local_dofs.iter().take(6).enumerate() {
                    let value =
                        target_values
                            .get(global)
                            .ok_or(InterpolationError::InvalidData {
                                node: element_index,
                                reason: format!(
                                    "target mixed-order H(curl) DOF {global} outside state vector"
                                ),
                            })?;
                    residual[0] -= value * target_tabulation.vector_value(local, 0);
                    residual[1] -= value * target_tabulation.vector_value(local, 1);
                }
                let weight = quadrature_point.weight * target_tabulation.det_j;
                for a in 0..2 {
                    let bubble_a = 6 + a;
                    let ax = target_tabulation.vector_value(bubble_a, 0);
                    let ay = target_tabulation.vector_value(bubble_a, 1);
                    rhs[a] += weight * (residual[0] * ax + residual[1] * ay);
                    for (b, slot) in gram[a].iter_mut().enumerate() {
                        let bubble_b = 6 + b;
                        *slot += weight
                            * (ax * target_tabulation.vector_value(bubble_b, 0)
                                + ay * target_tabulation.vector_value(bubble_b, 1));
                    }
                }
            }
            let determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
            if !determinant.is_finite() || determinant.abs() <= 1.0e-14 {
                return Err(InterpolationError::InvalidData {
                    node: element_index,
                    reason: "mixed-order H(curl) interior bubble Gram matrix is singular".into(),
                });
            }
            let first = (gram[1][1] * rhs[0] - gram[0][1] * rhs[1]) / determinant;
            let second = (gram[0][0] * rhs[1] - gram[0][1] * rhs[0]) / determinant;
            if !first.is_finite() || !second.is_finite() {
                return Err(InterpolationError::NonFiniteResult {
                    node: local_dofs[6],
                });
            }
            target_values[local_dofs[6]] = first;
            target_values[local_dofs[7]] = second;
        }
    }

    Ok(target_values)
}

fn evaluate_hcurl_field_at_point(
    source_locator: &ElementLocator,
    source_mesh: &Mesh,
    source_space: &HcurlSpace,
    source_values: &[f64],
    point: [f64; 3],
    entity: usize,
) -> Result<[f64; 2], InterpolationError> {
    let source_element = source_element_for_point(source_locator, source_mesh, point, entity)?;
    let barycentric = if source_mesh.elements[source_element].nodes.len() == 6 {
        point_in_triangle_p2(source_mesh, source_element, point)
    } else {
        point_in_element(source_mesh, source_element, point)
    }
    .ok_or(InterpolationError::TargetElementOutside { element: entity })?;
    let (xi, eta) = match barycentric.as_slice() {
        [_, first, second] => (*first, *second),
        _ => {
            return Err(InterpolationError::InvalidData {
                node: entity,
                reason: "source triangle returned malformed barycentric coordinates".into(),
            });
        }
    };
    let tabulation = source_space
        .try_tabulate(source_mesh, source_element, xi, eta, 0.0)
        .map_err(|error| InterpolationError::InvalidData {
            node: entity,
            reason: format!("source mixed-order H(curl) tabulation failed: {error}"),
        })?;
    let local_dofs = source_space.local_to_global(source_mesh, source_element);
    let mut field = [0.0; 2];
    for (local, &global) in local_dofs.iter().enumerate() {
        let value = source_values
            .get(global)
            .ok_or(InterpolationError::InvalidData {
                node: entity,
                reason: format!("source mixed-order H(curl) DOF {global} outside state vector"),
            })?;
        field[0] += value * tabulation.vector_value(local, 0);
        field[1] += value * tabulation.vector_value(local, 1);
    }
    if !field[0].is_finite() || !field[1].is_finite() {
        return Err(InterpolationError::NonFiniteResult { node: entity });
    }
    Ok(field)
}

fn validate_hcurl_mixed_triangle_transfer_mesh(
    mesh: &Mesh,
    space: &HcurlSpace,
    role: &'static str,
) -> Result<(), InterpolationError> {
    validate_interpolation_mesh(mesh, role)?;
    if mesh.dimension != 2 {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "mixed-order triangular H(curl) transfer requires 2D mesh".into(),
        });
    }
    if !matches!(space.order, 1 | 2) {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: format!(
                "mixed-order H(curl) transfer does not support order {}",
                space.order
            ),
        });
    }
    if mesh.elements.iter().any(|element| {
        element.element_type != ElementType::Triangle || element.try_polynomial_order().is_err()
    }) {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "mixed-order H(curl) transfer requires P1/P2 triangles".into(),
        });
    }
    space
        .validate(mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: role,
            reason: error.to_string(),
        })
}

fn evaluate_hcurl_order2_field_at_point(
    source_locator: &ElementLocator,
    source_mesh: &Mesh,
    source_space: &HcurlSpace,
    source_values: &[f64],
    point: [f64; 3],
    entity: usize,
) -> Result<[f64; 2], InterpolationError> {
    let source_element = source_element_for_point(source_locator, source_mesh, point, entity)?;
    let barycentric = if source_mesh.elements[source_element].nodes.len() == 6 {
        point_in_triangle_p2(source_mesh, source_element, point)
    } else {
        point_in_element(source_mesh, source_element, point)
    }
    .ok_or(InterpolationError::TargetElementOutside { element: entity })?;
    let (xi, eta) = match barycentric.as_slice() {
        [_, first, second] => (*first, *second),
        _ => {
            return Err(InterpolationError::InvalidData {
                node: entity,
                reason: "source triangle returned malformed barycentric coordinates".into(),
            });
        }
    };
    let tabulation = source_space
        .try_tabulate(source_mesh, source_element, xi, eta, 0.0)
        .map_err(|error| InterpolationError::InvalidData {
            node: entity,
            reason: format!("source H(curl) tabulation failed: {error}"),
        })?;
    let local_dofs = source_space.local_to_global(source_mesh, source_element);
    let mut field = [0.0; 2];
    for (local, &global) in local_dofs.iter().enumerate() {
        let value = source_values
            .get(global)
            .copied()
            .ok_or(InterpolationError::InvalidData {
                node: entity,
                reason: format!("source local DOF {global} outside state vector"),
            })?;
        field[0] += value * tabulation.vector_value(local, 0);
        field[1] += value * tabulation.vector_value(local, 1);
    }
    Ok(field)
}

/// Transfer an RT0 H(div) state between compatible affine simplex meshes.
///
/// RT0 coefficients are oriented normal-flux moments: edge moments in 2-D
/// and triangular-face moments in 3-D.  The source field is reconstructed
/// with the production contravariant-Piola tabulator, then each target
/// entity is integrated against its outward physical normal.  The target
/// basis moment is used as the normalization factor, so the result follows
/// the topology-owned global orientation used by [`HdivSpace`].
///
/// This checked transfer intentionally admits only P1 triangles/tetrahedra.
/// Curved/isoparametric maps, quadrilateral/hexahedral RT0, and RT1 modes are
/// rejected until their mode and geometry transfer contracts are specified.
/// Source ownership is resolved per quadrature point, so a target entity may
/// cross conforming source-element boundaries.
pub fn transfer_hdiv_rt0_state_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    validate_hdiv_rt0_transfer_mesh(source_mesh, source_space, "source")?;
    validate_hdiv_rt0_transfer_mesh(target_mesh, target_space, "target")?;
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "source and target H(div) meshes have different dimensions".into(),
        });
    }
    let source_dofs = source_space.num_global_dofs(source_mesh);
    if source_values.len() != source_dofs {
        return Err(InterpolationError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    if let Some(node) = source_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteSource { node });
    }

    let source_locator = ElementLocator::new(source_mesh);
    let topology = crate::mesh::topology::MeshTopology::build(target_mesh);
    let mut output = vec![0.0; target_space.num_global_dofs(target_mesh)];
    match target_mesh.dimension {
        2 => {
            let quadrature = gauss_legendre_1d(8);
            for (edge_index, &edge) in topology.edges.iter().enumerate() {
                let (target_element, local_edge) = target_edge_owner(target_mesh, edge)?.ok_or(
                    InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target edge has no owning triangle".into(),
                    },
                )?;
                let vertices = target_mesh.elements[target_element].vertices();
                let &(a, b) = target_mesh.elements[target_element]
                    .element_type
                    .local_edges()
                    .get(local_edge)
                    .ok_or(InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target local edge index is invalid".into(),
                    })?;
                let p0 = target_mesh.nodes[vertices[a]];
                let p1 = target_mesh.nodes[vertices[b]];
                let normal = outward_edge_normal_jacobian(
                    p0,
                    p1,
                    simplex_centroid(target_mesh, target_element),
                )
                .ok_or(InterpolationError::InvalidData {
                    node: edge_index,
                    reason: "target edge is degenerate".into(),
                })?;
                let local_dofs = target_space.local_to_global(target_mesh, target_element);
                let local_dof = local_dofs.iter().position(|&dof| dof == edge_index).ok_or(
                    InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target RT0 edge is absent from its owner map".into(),
                    },
                )?;
                let flux = integrate_source_edge_flux(
                    source_mesh,
                    &source_locator,
                    source_space,
                    source_values,
                    p0,
                    p1,
                    normal,
                    &quadrature,
                    edge_index,
                )?;
                let normalization = integrate_target_edge_basis_flux(
                    target_mesh,
                    target_space,
                    target_element,
                    local_edge,
                    local_dof,
                    normal,
                    &quadrature,
                    edge_index,
                )?;
                output[edge_index] = checked_moment_ratio(flux, normalization, edge_index)?;
            }
        }
        3 => {
            let quadrature = gauss_triangle(4);
            for (face_index, slot) in output[..topology.num_faces()].iter_mut().enumerate() {
                let (target_element, local_face) = target_face_owner(target_mesh, face_index)?
                    .ok_or(InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target face has no owning tetrahedron".into(),
                    })?;
                let vertices = target_mesh.elements[target_element].vertices();
                let local_vertices = target_mesh.elements[target_element]
                    .element_type
                    .local_faces()
                    .get(local_face)
                    .ok_or(InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target local face index is invalid".into(),
                    })?;
                let p0 = target_mesh.nodes[vertices[local_vertices[0]]];
                let p1 = target_mesh.nodes[vertices[local_vertices[1]]];
                let p2 = target_mesh.nodes[vertices[local_vertices[2]]];
                let surface = outward_face_surface_jacobian(
                    p0,
                    p1,
                    p2,
                    simplex_centroid(target_mesh, target_element),
                )
                .ok_or(InterpolationError::InvalidData {
                    node: face_index,
                    reason: "target face is degenerate".into(),
                })?;
                let local_dofs = target_space.local_to_global(target_mesh, target_element);
                let local_dof = local_dofs.iter().position(|&dof| dof == face_index).ok_or(
                    InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target RT0 face is absent from its owner map".into(),
                    },
                )?;
                let flux = integrate_source_face_flux(
                    source_mesh,
                    &source_locator,
                    source_space,
                    source_values,
                    [p0, p1, p2],
                    surface,
                    &quadrature,
                    face_index,
                )?;
                let normalization = integrate_target_face_basis_flux(
                    target_mesh,
                    target_space,
                    target_element,
                    local_face,
                    local_dof,
                    surface,
                    &quadrature,
                    face_index,
                )?;
                *slot = checked_moment_ratio(flux, normalization, face_index)?;
            }
        }
        dimension => {
            return Err(InterpolationError::InvalidMesh {
                mesh: "target",
                reason: format!("H(div) RT0 transfer dimension {dimension} unsupported"),
            });
        }
    }
    Ok(output)
}

/* Legacy malformed duplicate retained below for now while the checked RT1
 * implementation is repaired; it is deliberately excluded from parsing. */
/* pub fn transfer_hdiv_rt0_state_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    validate_hdiv_rt0_transfer_mesh(source_mesh, source_space, "source")?;
    validate_hdiv_rt0_transfer_mesh(target_mesh, target_space, "target")?;
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "source and target H(div) meshes have different dimensions".into(),
        });
    }

    let source_dofs = source_space.num_global_dofs(source_mesh);
    if source_values.len() != source_dofs {
        return Err(InterpolationError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    if let Some(dof) = source_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteSource { node: dof });
    }

    let target_topology = crate::mesh::topology::MeshTopology::build(target_mesh);
    let source_locator = ElementLocator::new(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    let mut target_values = vec![0.0; target_dofs];
    match target_mesh.dimension {
        2 => {
            let quadrature = gauss_legendre_1d(8);
            for (edge_index, &edge) in target_topology.edges.iter().enumerate() {
                let (target_element, local_edge) = target_edge_owner(target_mesh, edge)?.ok_or(
                    InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target edge has no owning triangle".into(),
                    },
                )?;
                let local_dofs = target_space.local_to_global(target_mesh, target_element);
                let local_dof = local_dofs.iter().position(|&dof| dof == edge_index).ok_or(
                    InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target RT0 edge is absent from its owner map".into(),
                    },
                )?;
                let vertices = target_mesh.elements[target_element].vertices();
                let &(a, b) = target_mesh.elements[target_element]
                    .element_type
                    .local_edges()
                    .get(local_edge)
                    .ok_or(InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target local edge index is invalid".into(),
                    })?;
                let p0 = target_mesh.nodes[vertices[a]];
                let p1 = target_mesh.nodes[vertices[b]];
                let centroid = simplex_centroid(target_mesh, target_element);
                let normal_jacobian = outward_edge_normal_jacobian(p0, p1, centroid).ok_or(
                    InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target edge is degenerate".into(),
                    },
                )?;
                let source_element =
                    source_element_for_edge(source_mesh, p0, p1, &quadrature, edge_index)?;
                let flux = integrate_source_edge_flux(
                    source_mesh,
                    source_space,
                    source_values,
                    source_element,
                    p0,
                    p1,
                    normal_jacobian,
                    &quadrature,
                    edge_index,
                )?;
                let normalization = integrate_target_edge_basis_flux(
                    target_mesh,
                    target_space,
                    target_element,
                    local_edge,
                    local_dof,
                    normal_jacobian,
                    &quadrature,
                    edge_index,
                )?;
                target_values[edge_index] = checked_moment_ratio(flux, normalization, edge_index)?;
            }
        }
        3 => {
            let quadrature = gauss_triangle(4);
            for face_index in 0..target_topology.num_faces() {
                let (target_element, local_face) = target_face_owner(target_mesh, face_index)?
                    .ok_or(InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target face has no owning tetrahedron".into(),
                    })?;
                let local_dofs = target_space.local_to_global(target_mesh, target_element);
                let local_dof = local_dofs.iter().position(|&dof| dof == face_index).ok_or(
                    InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target RT0 face is absent from its owner map".into(),
                    },
                )?;
                let vertices = target_mesh.elements[target_element].vertices();
                let local_face_vertices = target_mesh.elements[target_element]
                    .element_type
                    .local_faces()
                    .get(local_face)
                    .ok_or(InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target local face index is invalid".into(),
                    })?;
                let p0 = target_mesh.nodes[vertices[local_face_vertices[0]]];
                let p1 = target_mesh.nodes[vertices[local_face_vertices[1]]];
                let p2 = target_mesh.nodes[vertices[local_face_vertices[2]]];
                let centroid = simplex_centroid(target_mesh, target_element);
                let surface_jacobian = outward_face_surface_jacobian(p0, p1, p2, centroid).ok_or(
                    InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target face is degenerate".into(),
                    },
                )?;
                let source_element =
                    source_element_for_face(source_mesh, p0, p1, p2, &quadrature, face_index)?;
                let flux = integrate_source_face_flux(
                    source_mesh,
                    source_space,
                    source_values,
                    source_element,
                    [p0, p1, p2],
                    surface_jacobian,
                    &quadrature,
                    face_index,
                )?;
                let normalization = integrate_target_face_basis_flux(
                    target_mesh,
                    target_space,
                    target_element,
                    local_face,
                    local_dof,
                    surface_jacobian,
                    &quadrature,
                    face_index,
                )?;
                target_values[face_index] = checked_moment_ratio(flux, normalization, face_index)?;
            }
        }
        dimension => {
            return Err(InterpolationError::InvalidMesh {
                mesh: "target",
                reason: format!("H(div) RT0 transfer dimension {dimension} unsupported"),
            });
        }
    }
    Ok(target_values)
}

*/

/// Transfer an RT1 H(div) state across supported simplex meshes.
///
/// The bounded contract admits P1/P2 triangles and affine P1 tetrahedra.  The
/// source field is reconstructed with the production Piola tabulation; each
/// target RT1 coefficient is then recovered from its edge/face or cell
/// moments using a small checked dense solve.  Quadrilateral/hexahedral,
/// mixed-order and adaptive coarsening transfers are intentionally rejected
/// until their geometry and ownership contracts are specified.
/// Source ownership for edge, face, and cell quadrature is resolved per point,
/// so target entities may cross conforming source-element boundaries.
pub fn transfer_hdiv_rt1_state_checked(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    validate_hdiv_rt1_transfer_mesh(source_mesh, source_space, "source")?;
    validate_hdiv_rt1_transfer_mesh(target_mesh, target_space, "target")?;
    if source_mesh.dimension != target_mesh.dimension {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "source and target H(div) meshes have different dimensions".into(),
        });
    }

    let source_dofs = source_space.num_global_dofs(source_mesh);
    if source_values.len() != source_dofs {
        return Err(InterpolationError::SourceValueCount {
            expected: source_dofs,
            actual: source_values.len(),
        });
    }
    if let Some(dof) = source_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteSource { node: dof });
    }

    let source_locator = ElementLocator::new(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    let mut target_values = vec![0.0; target_dofs];
    match target_mesh.dimension {
        2 => {
            let topology = crate::mesh::topology::MeshTopology::build(target_mesh);
            let (edge_dofs, cell_dofs) = target_space
                .rt1_global_dof_layout(target_mesh)
                .ok_or_else(|| InterpolationError::InvalidMesh {
                    mesh: "target",
                    reason: "target RT1 2-D DOF layout is unavailable".into(),
                })?;
            if edge_dofs + cell_dofs != target_dofs {
                return Err(InterpolationError::InvalidData {
                    node: 0,
                    reason: "target RT1 2-D DOF layout does not cover the state vector".into(),
                });
            }
            let quadrature = gauss_legendre_1d(8);
            for (edge_index, &edge) in topology.edges.iter().enumerate() {
                let (target_element, local_edge) = target_edge_owner(target_mesh, edge)?.ok_or(
                    InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target RT1 edge has no owning triangle".into(),
                    },
                )?;
                let vertices = target_mesh.elements[target_element].vertices();
                let &(a, b) = target_mesh.elements[target_element]
                    .element_type
                    .local_edges()
                    .get(local_edge)
                    .ok_or(InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target RT1 local edge is unavailable".into(),
                    })?;
                let oriented_edge = (vertices[a], vertices[b]);
                let (edge_p0, edge_pm, edge_p1) = edge_geometry_points(target_mesh, oriented_edge)
                    .ok_or(InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target RT1 edge geometry is unavailable".into(),
                    })?;
                let centroid = simplex_centroid(target_mesh, target_element);
                let curved_edge = (edge_pm.x - 0.5 * (edge_p0.x + edge_p1.x)).abs()
                    + (edge_pm.y - 0.5 * (edge_p0.y + edge_p1.y)).abs()
                    + (edge_pm.z - 0.5 * (edge_p0.z + edge_p1.z)).abs()
                    > 1.0e-12;
                let local_dofs = target_space.local_to_global(target_mesh, target_element);
                let mut matrix = [[0.0; 2]; 2];
                let mut rhs = [0.0; 2];
                for point in &quadrature {
                    let coordinate = point.xi();
                    let physical = if curved_edge {
                        edge_curve_point(edge_p0, edge_pm, edge_p1, coordinate)
                    } else {
                        edge_point(edge_p0, edge_p1, coordinate)
                    };
                    let normal_jacobian = if curved_edge {
                        curved_edge_normal_jacobian(edge_p0, edge_pm, edge_p1, coordinate, centroid)
                    } else {
                        outward_edge_normal_jacobian(edge_p0, edge_p1, centroid)
                    }
                    .ok_or(InterpolationError::InvalidData {
                        node: edge_index,
                        reason: "target RT1 edge is degenerate".into(),
                    })?;
                    let source_element = source_element_for_point(
                        &source_locator,
                        source_mesh,
                        physical,
                        edge_index,
                    )?;
                    let source_field = hdiv_field_at_point(
                        source_mesh,
                        source_space,
                        source_values,
                        source_element,
                        physical,
                        edge_index,
                    )?;
                    let flux = dot3(source_field, normal_jacobian);
                    for mode in 0..2 {
                        let weight = point.weight * coordinate.powi(mode as i32);
                        rhs[mode] += weight * flux;
                        for (basis_mode, slot) in matrix[mode].iter_mut().enumerate() {
                            let global_dof = 2 * edge_index + basis_mode;
                            let local_dof = local_dofs
                                .iter()
                                .position(|&dof| dof == global_dof)
                                .ok_or(InterpolationError::InvalidData {
                                    node: edge_index,
                                    reason: format!(
                                    "target RT1 edge DOF {global_dof} is absent from its owner map"
                                ),
                                })?;
                            let reference =
                                target_edge_reference_coordinates(local_edge, coordinate);
                            let basis = hdiv_basis_field_at_reference(
                                target_mesh,
                                target_space,
                                target_element,
                                local_dof,
                                reference,
                                edge_index,
                            )?;
                            *slot += weight * dot3(basis, normal_jacobian);
                        }
                    }
                }
                let coefficients = solve_small_system(matrix, rhs, edge_index, "edge")?;
                for mode in 0..2 {
                    target_values[2 * edge_index + mode] = coefficients[mode];
                }
            }
            let cell_quadrature = gauss_triangle(4);
            for element in 0..target_mesh.num_elements() {
                let local_dofs = target_space.local_to_global(target_mesh, element);
                let coefficients = transfer_rt1_triangle_cell(
                    source_mesh,
                    &source_locator,
                    target_mesh,
                    source_space,
                    target_space,
                    source_values,
                    &target_values,
                    element,
                    &cell_quadrature,
                )?;
                for (mode, &coef) in coefficients.iter().enumerate() {
                    let expected = edge_dofs + 2 * element + mode;
                    if local_dofs.get(6 + mode).copied() != Some(expected)
                        || expected >= edge_dofs + cell_dofs
                    {
                        return Err(InterpolationError::InvalidData {
                            node: element,
                            reason: "target RT1 triangle cell DOF layout is inconsistent".into(),
                        });
                    }
                    target_values[expected] = coef;
                }
            }
        }
        3 => {
            let topology = crate::mesh::topology::MeshTopology::build(target_mesh);
            let layout = target_space.rt1_dof_layout_3d(target_mesh).ok_or_else(|| {
                InterpolationError::InvalidMesh {
                    mesh: "target",
                    reason: "target RT1 3-D DOF layout is unavailable".into(),
                }
            })?;
            let face_quadrature = gauss_triangle(4);
            for face_index in 0..topology.num_faces() {
                let (target_element, local_face) = target_face_owner(target_mesh, face_index)?
                    .ok_or(InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target RT1 face has no owning tetrahedron".into(),
                    })?;
                let face_modes =
                    layout
                        .face_dofs(face_index)
                        .ok_or(InterpolationError::InvalidData {
                            node: face_index,
                            reason: "target RT1 face range is unavailable".into(),
                        })?;
                if face_modes.len() != 3 {
                    return Err(InterpolationError::InvalidData {
                        node: face_index,
                        reason: "target RT1 transfer requires triangular faces".into(),
                    });
                }
                let face_corners = rt1_face_reference_corners(local_face);
                let physical_face = face_corners.map(|reference| {
                    let point = simplex_reference_point(target_mesh, target_element, reference);
                    Point::new_3d(point[0], point[1], point[2])
                });
                let centroid = simplex_centroid(target_mesh, target_element);
                let surface_jacobian = outward_face_surface_jacobian(
                    physical_face[0],
                    physical_face[1],
                    physical_face[2],
                    centroid,
                )
                .ok_or(InterpolationError::InvalidData {
                    node: face_index,
                    reason: "target RT1 face is degenerate".into(),
                })?;
                let local_dofs = target_space.local_to_global(target_mesh, target_element);
                let mut matrix = [[0.0; 3]; 3];
                let mut rhs = [0.0; 3];
                for point in &face_quadrature {
                    let a = point.xi();
                    let b = point.eta();
                    let reference = rt1_face_reference_point(local_face, a, b);
                    let physical = simplex_reference_point(target_mesh, target_element, reference);
                    let source_element = source_element_for_point(
                        &source_locator,
                        source_mesh,
                        physical,
                        face_index,
                    )?;
                    let source_field = hdiv_field_at_point(
                        source_mesh,
                        source_space,
                        source_values,
                        source_element,
                        physical,
                        face_index,
                    )?;
                    for mode in 0..3 {
                        let weight = point.weight
                            * face_monomial(local_face, mode, reference)
                            * dot3(source_field, surface_jacobian);
                        rhs[mode] += weight;
                        for (basis_mode, slot) in matrix[mode].iter_mut().enumerate() {
                            let global_dof = face_modes.start + basis_mode;
                            let local_dof = local_dofs
                                .iter()
                                .position(|&dof| dof == global_dof)
                                .ok_or(InterpolationError::InvalidData {
                                    node: face_index,
                                    reason: format!(
                                    "target RT1 face DOF {global_dof} is absent from its owner map"
                                ),
                                })?;
                            let basis = hdiv_basis_field_at_reference(
                                target_mesh,
                                target_space,
                                target_element,
                                local_dof,
                                reference,
                                face_index,
                            )?;
                            *slot += point.weight
                                * face_monomial(local_face, mode, reference)
                                * dot3(basis, surface_jacobian);
                        }
                    }
                }
                let coefficients = solve_small_system(matrix, rhs, face_index, "face")?;
                for (mode, value) in coefficients.into_iter().enumerate() {
                    target_values[face_modes.start + mode] = value;
                }
            }
            for element in 0..target_mesh.num_elements() {
                let cell_modes =
                    layout
                        .cell_dofs(element)
                        .ok_or(InterpolationError::InvalidData {
                            node: element,
                            reason: "target RT1 cell range is unavailable".into(),
                        })?;
                let local_dofs = target_space.local_to_global(target_mesh, element);
                if cell_modes.len() != 3 || local_dofs.len() != 15 {
                    return Err(InterpolationError::InvalidData {
                        node: element,
                        reason: "target RT1 tetrahedron cell layout is inconsistent".into(),
                    });
                }
                let cell_quadrature = gauss_tetrahedron(4);
                let coefficients = transfer_rt1_tetrahedron_cell(
                    source_mesh,
                    &source_locator,
                    target_mesh,
                    source_space,
                    target_space,
                    source_values,
                    element,
                    &cell_quadrature,
                )?;
                for (mode, &coef) in coefficients.iter().enumerate() {
                    let expected = cell_modes.start + mode;
                    if local_dofs.get(12 + mode).copied() != Some(expected) {
                        return Err(InterpolationError::InvalidData {
                            node: element,
                            reason: "target RT1 tetrahedron cell DOF layout is inconsistent".into(),
                        });
                    }
                    target_values[expected] = coef;
                }
            }
        }
        dimension => {
            return Err(InterpolationError::InvalidMesh {
                mesh: "target",
                reason: format!("H(div) RT1 transfer dimension {dimension} unsupported"),
            });
        }
    }
    if let Some(dof) = target_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteResult { node: dof });
    }
    Ok(target_values)
}

fn validate_hdiv_rt1_transfer_mesh(
    mesh: &Mesh,
    space: &HdivSpace,
    role: &'static str,
) -> Result<(), InterpolationError> {
    if mesh
        .nodes
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
    {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "mesh coordinates must be finite".into(),
        });
    }
    validate_interpolation_mesh(mesh, role)?;
    if !matches!(mesh.dimension, 2 | 3) {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: format!(
                "H(div) RT1 transfer dimension {} unsupported",
                mesh.dimension
            ),
        });
    }
    if space.order() != HdivOrder::Rt1 {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "only RT1 state transfer is implemented".into(),
        });
    }
    space
        .validate(mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: role,
            reason: error.to_string(),
        })?;
    let valid_geometry = mesh.elements.iter().all(|element| {
        let expected = if mesh.dimension == 2 {
            ElementType::Triangle
        } else {
            ElementType::Tetrahedron
        };
        let order_supported = if mesh.dimension == 2 {
            matches!(element.try_polynomial_order(), Ok(1) | Ok(2))
        } else {
            element.try_polynomial_order() == Ok(1)
        };
        element.element_type == expected && order_supported
    });
    if !valid_geometry {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "RT1 transfer admits P1/P2 triangles or affine P1 tetrahedra".into(),
        });
    }
    Ok(())
}

fn target_edge_reference_coordinates(local_edge: usize, coordinate: f64) -> (f64, f64, f64) {
    let (a, b) = match local_edge {
        0 => (0, 1),
        1 => (1, 2),
        2 => (2, 0),
        _ => unreachable!("triangle has three local edges"),
    };
    let mut barycentric = [0.0; 3];
    barycentric[a] = 0.5 * (1.0 - coordinate);
    barycentric[b] = 0.5 * (1.0 + coordinate);
    (barycentric[1], barycentric[2], 0.0)
}

fn rt1_face_reference_point(face: usize, a: f64, b: f64) -> (f64, f64, f64) {
    match face {
        0 => (a, b, 0.0),
        1 => (a, 0.0, b),
        2 => (0.0, a, b),
        3 => (a, b, 1.0 - a - b),
        _ => unreachable!("tetrahedron has four local faces"),
    }
}

fn rt1_face_reference_corners(face: usize) -> [(f64, f64, f64); 3] {
    [
        rt1_face_reference_point(face, 0.0, 0.0),
        rt1_face_reference_point(face, 1.0, 0.0),
        rt1_face_reference_point(face, 0.0, 1.0),
    ]
}

fn face_monomial(face: usize, mode: usize, reference: (f64, f64, f64)) -> f64 {
    let point = [reference.0, reference.1, reference.2];
    match mode {
        0 => 1.0,
        1 => match face {
            0 | 1 => point[0],
            2 | 3 => point[1],
            _ => unreachable!("tetrahedron has four local faces"),
        },
        2 => match face {
            0 => point[1],
            1..=3 => point[2],
            _ => unreachable!("tetrahedron has four local faces"),
        },
        _ => unreachable!("RT1 triangular face has three modes"),
    }
}

fn hdiv_basis_field_at_reference(
    mesh: &Mesh,
    space: &HdivSpace,
    element: usize,
    local_dof: usize,
    reference: (f64, f64, f64),
    entity: usize,
) -> Result<[f64; 3], InterpolationError> {
    let tabulation = space
        .try_tabulate(mesh, element, reference.0, reference.1, reference.2)
        .map_err(|error| InterpolationError::InvalidData {
            node: entity,
            reason: format!("target RT1 tabulation failed: {error}"),
        })?;
    if local_dof >= tabulation.n_dofs {
        return Err(InterpolationError::InvalidData {
            node: entity,
            reason: format!("target RT1 local DOF {local_dof} is outside tabulation"),
        });
    }
    let mut field = [0.0; 3];
    for (component, slot) in field[..tabulation.spatial_dim].iter_mut().enumerate() {
        *slot = tabulation.vector_value(local_dof, component);
    }
    if field[..tabulation.spatial_dim]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(InterpolationError::NonFiniteResult { node: entity });
    }
    Ok(field)
}

fn transfer_rt1_triangle_cell(
    source_mesh: &Mesh,
    source_locator: &ElementLocator,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
    target_values: &[f64],
    target_element: usize,
    quadrature: &[crate::quadrature::QuadraturePoint],
) -> Result<[f64; 2], InterpolationError> {
    let mut matrix = [[0.0; 2]; 2];
    let mut rhs = [0.0; 2];
    let local_dofs = target_space.local_to_global(target_mesh, target_element);
    if local_dofs.len() != 8 {
        return Err(InterpolationError::InvalidData {
            node: target_element,
            reason: "target RT1 triangle cell layout is inconsistent".into(),
        });
    }
    for point in quadrature {
        let reference = (point.xi(), point.eta(), 0.0);
        let target_tabulation = target_space
            .try_tabulate(target_mesh, target_element, reference.0, reference.1, 0.0)
            .map_err(|error| InterpolationError::InvalidData {
                node: target_element,
                reason: format!("target RT1 cell tabulation failed: {error}"),
            })?;
        let physical = target_tabulation.physical_coords;
        let source_element =
            source_element_for_point(source_locator, source_mesh, physical, target_element)?;
        let source_field = hdiv_field_at_point(
            source_mesh,
            source_space,
            source_values,
            source_element,
            physical,
            target_element,
        )?;
        for component in 0..2 {
            let weight = point.weight * target_tabulation.det_j.abs();
            let mut residual = source_field[component];
            for (local, &global) in local_dofs[..6].iter().enumerate() {
                let value = target_values
                    .get(global)
                    .ok_or(InterpolationError::InvalidData {
                        node: target_element,
                        reason: format!("target RT1 edge DOF {global} is outside state vector"),
                    })?;
                residual -= *value * target_tabulation.vector_value(local, component);
            }
            rhs[component] += weight * residual;
            for (mode, slot) in matrix[component].iter_mut().enumerate() {
                *slot += weight * target_tabulation.vector_value(6 + mode, component);
            }
        }
    }
    solve_small_system(matrix, rhs, target_element, "cell")
}

fn transfer_rt1_tetrahedron_cell(
    source_mesh: &Mesh,
    source_locator: &ElementLocator,
    target_mesh: &Mesh,
    source_space: &HdivSpace,
    target_space: &HdivSpace,
    source_values: &[f64],
    target_element: usize,
    quadrature: &[crate::quadrature::QuadraturePoint],
) -> Result<[f64; 3], InterpolationError> {
    let vertices = target_mesh.elements[target_element].vertices();
    let p0 = target_mesh.nodes[vertices[0]];
    let p1 = target_mesh.nodes[vertices[1]];
    let p2 = target_mesh.nodes[vertices[2]];
    let p3 = target_mesh.nodes[vertices[3]];
    let jacobian = tetrahedron_jacobian(p0, p1, p2, p3).ok_or(InterpolationError::InvalidData {
        node: target_element,
        reason: "target RT1 tetrahedron has a degenerate Jacobian".into(),
    })?;
    let mut matrix = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for point in quadrature {
        let reference = (point.xi(), point.eta(), point.zeta());
        let physical = simplex_reference_point(target_mesh, target_element, reference);
        let source_element =
            source_element_for_point(source_locator, source_mesh, physical, target_element)?;
        let source_field = hdiv_field_at_point(
            source_mesh,
            source_space,
            source_values,
            source_element,
            physical,
            target_element,
        )?;
        let target_tabulation = target_space
            .try_tabulate(
                target_mesh,
                target_element,
                reference.0,
                reference.1,
                reference.2,
            )
            .map_err(|error| InterpolationError::InvalidData {
                node: target_element,
                reason: format!("target RT1 cell tabulation failed: {error}"),
            })?;
        for component in 0..3 {
            let weight = point.weight * jacobian;
            rhs[component] += weight * source_field[component];
            for (mode, slot) in matrix[component].iter_mut().enumerate() {
                *slot += weight * target_tabulation.vector_value(12 + mode, component);
            }
        }
    }
    solve_small_system(matrix, rhs, target_element, "cell")
}

/// Locate the source element containing one quadrature point.
///
/// A target entity may cross source-element boundaries.  Selecting one source
/// element for the whole entity silently rejected valid conforming transfers;
/// resolving ownership per point admits refinement/coarsening interfaces while
/// preserving the existing fail-closed outside-domain behavior.
fn source_element_for_point(
    locator: &ElementLocator,
    source_mesh: &Mesh,
    point: [f64; 3],
    entity: usize,
) -> Result<usize, InterpolationError> {
    if point.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(InterpolationError::NonFinitePoint);
    }
    locator
        .containing_element(source_mesh, point)
        .ok_or(InterpolationError::TargetElementOutside { element: entity })
}

fn simplex_reference_point(mesh: &Mesh, element: usize, reference: (f64, f64, f64)) -> [f64; 3] {
    let vertices = mesh.elements[element].vertices();
    let p0 = mesh.nodes[vertices[0]];
    let p1 = mesh.nodes[vertices[1]];
    let p2 = mesh.nodes[vertices[2]];
    let mut point = [
        p0.x + reference.0 * (p1.x - p0.x) + reference.1 * (p2.x - p0.x),
        p0.y + reference.0 * (p1.y - p0.y) + reference.1 * (p2.y - p0.y),
        p0.z + reference.0 * (p1.z - p0.z) + reference.1 * (p2.z - p0.z),
    ];
    if vertices.len() == 4 {
        let p3 = mesh.nodes[vertices[3]];
        point[0] += reference.2 * (p3.x - p0.x);
        point[1] += reference.2 * (p3.y - p0.y);
        point[2] += reference.2 * (p3.z - p0.z);
    }
    point
}

fn tetrahedron_jacobian(p0: Point, p1: Point, p2: Point, p3: Point) -> Option<f64> {
    let a = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let b = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
    let c = [p3.x - p0.x, p3.y - p0.y, p3.z - p0.z];
    let determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    if determinant.is_finite() && determinant.abs() > 1.0e-14 {
        Some(determinant.abs())
    } else {
        None
    }
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn solve_small_system<const N: usize>(
    mut matrix: [[f64; N]; N],
    mut rhs: [f64; N],
    entity: usize,
    kind: &str,
) -> Result<[f64; N], InterpolationError> {
    for pivot in 0..N {
        let mut row = pivot;
        for candidate in (pivot + 1)..N {
            if matrix[candidate][pivot].abs() > matrix[row][pivot].abs() {
                row = candidate;
            }
        }
        if !matrix[row][pivot].is_finite() || matrix[row][pivot].abs() <= 1.0e-12 {
            return Err(InterpolationError::InvalidData {
                node: entity,
                reason: format!("RT1 {kind} moment matrix is singular or non-finite"),
            });
        }
        if row != pivot {
            matrix.swap(row, pivot);
            rhs.swap(row, pivot);
        }
        let diagonal = matrix[pivot][pivot];
        for slot in &mut matrix[pivot][pivot..N] {
            *slot /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for other in 0..N {
            if other == pivot {
                continue;
            }
            let factor = matrix[other][pivot];
            let pivot_row = matrix[pivot];
            for (off, slot) in matrix[other][pivot..N].iter_mut().enumerate() {
                *slot -= factor * pivot_row[pivot + off];
            }
            rhs[other] -= factor * rhs[pivot];
        }
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteResult { node: entity });
    }
    Ok(rhs)
}

fn validate_hdiv_rt0_transfer_mesh(
    mesh: &Mesh,
    space: &HdivSpace,
    role: &'static str,
) -> Result<(), InterpolationError> {
    validate_interpolation_mesh(mesh, role)?;
    if !matches!(mesh.dimension, 2 | 3) {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: format!(
                "H(div) RT0 transfer dimension {} unsupported",
                mesh.dimension
            ),
        });
    }
    if space.order() != HdivOrder::Rt0 {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "only RT0 state transfer is implemented".into(),
        });
    }
    space
        .validate(mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: role,
            reason: error.to_string(),
        })?;
    if mesh
        .nodes
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
    {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "mesh coordinates must be finite".into(),
        });
    }
    let expected = if mesh.dimension == 2 {
        ElementType::Triangle
    } else {
        ElementType::Tetrahedron
    };
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != expected || element.try_polynomial_order() != Ok(1))
    {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "RT0 transfer admits only affine P1 triangles or tetrahedra".into(),
        });
    }
    Ok(())
}

fn target_edge_owner(
    mesh: &Mesh,
    edge: (usize, usize),
) -> Result<Option<(usize, usize)>, InterpolationError> {
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        for (local_edge, &(a, b)) in element.element_type.local_edges().iter().enumerate() {
            let candidate = (vertices[a], vertices[b]);
            if same_unordered_edge(candidate, edge) {
                return Ok(Some((element_index, local_edge)));
            }
        }
    }
    Ok(None)
}

fn target_face_owner(
    mesh: &Mesh,
    face_index: usize,
) -> Result<Option<(usize, usize)>, InterpolationError> {
    let topology = crate::mesh::topology::MeshTopology::build(mesh);
    let face_nodes = topology
        .faces
        .get(face_index)
        .ok_or(InterpolationError::InvalidData {
            node: face_index,
            reason: "target face index is out of bounds".into(),
        })?
        .nodes();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        for (local_face, local_vertices) in element.element_type.local_faces().iter().enumerate() {
            let mut candidate: Vec<_> = local_vertices
                .iter()
                .map(|&index| vertices[index])
                .collect();
            candidate.sort_unstable();
            let mut expected = face_nodes.clone();
            expected.sort_unstable();
            if candidate == expected {
                return Ok(Some((element_index, local_face)));
            }
        }
    }
    Ok(None)
}

fn same_unordered_edge(left: (usize, usize), right: (usize, usize)) -> bool {
    (left.0 == right.0 && left.1 == right.1) || (left.0 == right.1 && left.1 == right.0)
}

fn simplex_centroid(mesh: &Mesh, element_index: usize) -> Point {
    let element = &mesh.elements[element_index];
    let vertices = element.vertices();
    let scale = 1.0 / vertices.len() as f64;
    let mut centroid = Point::new_3d(0.0, 0.0, 0.0);
    for &node in vertices {
        let point = mesh.nodes[node];
        centroid.x += point.x * scale;
        centroid.y += point.y * scale;
        centroid.z += point.z * scale;
    }
    centroid
}

fn outward_edge_normal_jacobian(p0: Point, p1: Point, centroid: Point) -> Option<[f64; 3]> {
    let tangent = [p1.x - p0.x, p1.y - p0.y];
    let norm = tangent[0].hypot(tangent[1]);
    if !norm.is_finite() || norm <= 1.0e-14 {
        return None;
    }
    let midpoint = [(p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5];
    let to_centroid = [centroid.x - midpoint[0], centroid.y - midpoint[1]];
    let mut normal = [tangent[1] * 0.5, -tangent[0] * 0.5, 0.0];
    if normal[0] * to_centroid[0] + normal[1] * to_centroid[1] > 0.0 {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
    }
    Some(normal)
}

fn outward_face_surface_jacobian(
    p0: Point,
    p1: Point,
    p2: Point,
    centroid: Point,
) -> Option<[f64; 3]> {
    let ab = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let ac = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
    let mut surface = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let norm = surface[0].hypot(surface[1]).hypot(surface[2]);
    if !norm.is_finite() || norm <= 1.0e-14 {
        return None;
    }
    let face_centroid = [
        (p0.x + p1.x + p2.x) / 3.0,
        (p0.y + p1.y + p2.y) / 3.0,
        (p0.z + p1.z + p2.z) / 3.0,
    ];
    let to_centroid = [
        centroid.x - face_centroid[0],
        centroid.y - face_centroid[1],
        centroid.z - face_centroid[2],
    ];
    if surface[0] * to_centroid[0] + surface[1] * to_centroid[1] + surface[2] * to_centroid[2] > 0.0
    {
        surface.iter_mut().for_each(|value| *value = -*value);
    }
    Some(surface)
}

fn edge_point(p0: Point, p1: Point, coordinate: f64) -> [f64; 3] {
    let left = 0.5 * (1.0 - coordinate);
    let right = 0.5 * (1.0 + coordinate);
    [
        left * p0.x + right * p1.x,
        left * p0.y + right * p1.y,
        left * p0.z + right * p1.z,
    ]
}

fn face_point(face: [Point; 3], xi: f64, eta: f64) -> [f64; 3] {
    [
        face[0].x + xi * (face[1].x - face[0].x) + eta * (face[2].x - face[0].x),
        face[0].y + xi * (face[1].y - face[0].y) + eta * (face[2].y - face[0].y),
        face[0].z + xi * (face[1].z - face[0].z) + eta * (face[2].z - face[0].z),
    ]
}

fn hdiv_reference_coordinates(
    barycentric: &[f64],
    entity: usize,
) -> Result<(f64, f64, f64), InterpolationError> {
    match barycentric {
        [_, xi, eta] => Ok((*xi, *eta, 0.0)),
        [_, xi, eta, zeta] => Ok((*xi, *eta, *zeta)),
        _ => Err(InterpolationError::InvalidData {
            node: entity,
            reason: "source simplex returned malformed barycentric coordinates".into(),
        }),
    }
}

fn hdiv_field_at_point(
    mesh: &Mesh,
    space: &HdivSpace,
    values: &[f64],
    element: usize,
    point: [f64; 3],
    entity: usize,
) -> Result<[f64; 3], InterpolationError> {
    let (xi, eta, zeta) = if mesh.elements[element].element_type == ElementType::Triangle
        && mesh.elements[element].nodes.len() == ElementType::Triangle.num_nodes_p2()
    {
        let barycentric = point_in_triangle_p2(mesh, element, point)
            .ok_or(InterpolationError::TargetElementOutside { element: entity })?;
        hdiv_reference_coordinates(&barycentric, entity)?
    } else {
        let barycentric = point_in_element(mesh, element, point)
            .ok_or(InterpolationError::TargetElementOutside { element: entity })?;
        hdiv_reference_coordinates(&barycentric, entity)?
    };
    let tabulation = space
        .try_tabulate(mesh, element, xi, eta, zeta)
        .map_err(|error| InterpolationError::InvalidData {
            node: entity,
            reason: format!("source RT0 tabulation failed: {error}"),
        })?;
    let local_dofs = space.local_to_global(mesh, element);
    let mut field = [0.0; 3];
    for (local, &global) in local_dofs.iter().enumerate() {
        let value = values.get(global).ok_or(InterpolationError::InvalidData {
            node: entity,
            reason: format!("source local DOF {global} outside state vector"),
        })?;
        for (component, slot) in field[..tabulation.spatial_dim].iter_mut().enumerate() {
            *slot += *value * tabulation.vector_value(local, component);
        }
    }
    if field[..tabulation.spatial_dim]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(InterpolationError::NonFiniteResult { node: entity });
    }
    Ok(field)
}

fn integrate_source_edge_flux(
    mesh: &Mesh,
    locator: &ElementLocator,
    space: &HdivSpace,
    values: &[f64],
    p0: Point,
    p1: Point,
    normal_jacobian: [f64; 3],
    quadrature: &[crate::quadrature::QuadraturePoint],
    entity: usize,
) -> Result<f64, InterpolationError> {
    let mut flux = 0.0;
    for point in quadrature {
        let physical = edge_point(p0, p1, point.xi());
        let element = source_element_for_point(locator, mesh, physical, entity)?;
        let field = hdiv_field_at_point(mesh, space, values, element, physical, entity)?;
        flux += point.weight
            * (field[0] * normal_jacobian[0]
                + field[1] * normal_jacobian[1]
                + field[2] * normal_jacobian[2]);
    }
    Ok(flux)
}

fn integrate_target_edge_basis_flux(
    mesh: &Mesh,
    space: &HdivSpace,
    element: usize,
    local_edge: usize,
    local_dof: usize,
    normal_jacobian: [f64; 3],
    quadrature: &[crate::quadrature::QuadraturePoint],
    entity: usize,
) -> Result<f64, InterpolationError> {
    let &(a, b) = mesh.elements[element]
        .element_type
        .local_edges()
        .get(local_edge)
        .ok_or(InterpolationError::InvalidData {
            node: entity,
            reason: "target local edge index is invalid".into(),
        })?;
    let mut flux = 0.0;
    for point in quadrature {
        let coordinate = point.xi();
        let lambda_a = 0.5 * (1.0 - coordinate);
        let lambda_b = 0.5 * (1.0 + coordinate);
        let mut barycentric = [0.0; 3];
        barycentric[a] = lambda_a;
        barycentric[b] = lambda_b;
        let tabulation = space
            .try_tabulate(mesh, element, barycentric[1], barycentric[2], 0.0)
            .map_err(|error| InterpolationError::InvalidData {
                node: entity,
                reason: format!("target RT0 tabulation failed: {error}"),
            })?;
        let value = [
            tabulation.vector_value(local_dof, 0),
            tabulation.vector_value(local_dof, 1),
            0.0,
        ];
        flux += point.weight
            * (value[0] * normal_jacobian[0]
                + value[1] * normal_jacobian[1]
                + value[2] * normal_jacobian[2]);
    }
    Ok(flux)
}

fn integrate_source_face_flux(
    mesh: &Mesh,
    locator: &ElementLocator,
    space: &HdivSpace,
    values: &[f64],
    face: [Point; 3],
    surface_jacobian: [f64; 3],
    quadrature: &[crate::quadrature::QuadraturePoint],
    entity: usize,
) -> Result<f64, InterpolationError> {
    let mut flux = 0.0;
    for point in quadrature {
        let physical = face_point(face, point.xi(), point.eta());
        let element = source_element_for_point(locator, mesh, physical, entity)?;
        let field = hdiv_field_at_point(mesh, space, values, element, physical, entity)?;
        flux += point.weight
            * (field[0] * surface_jacobian[0]
                + field[1] * surface_jacobian[1]
                + field[2] * surface_jacobian[2]);
    }
    Ok(flux)
}

fn integrate_target_face_basis_flux(
    mesh: &Mesh,
    space: &HdivSpace,
    element: usize,
    local_face: usize,
    local_dof: usize,
    surface_jacobian: [f64; 3],
    quadrature: &[crate::quadrature::QuadraturePoint],
    entity: usize,
) -> Result<f64, InterpolationError> {
    let local_face_vertices = mesh.elements[element]
        .element_type
        .local_faces()
        .get(local_face)
        .ok_or(InterpolationError::InvalidData {
            node: entity,
            reason: "target local face index is invalid".into(),
        })?;
    let mut flux = 0.0;
    for point in quadrature {
        let xi = point.xi();
        let eta = point.eta();
        let mut barycentric = [0.0; 4];
        barycentric[local_face_vertices[0]] = 1.0 - xi - eta;
        barycentric[local_face_vertices[1]] = xi;
        barycentric[local_face_vertices[2]] = eta;
        let tabulation = space
            .try_tabulate(
                mesh,
                element,
                barycentric[1],
                barycentric[2],
                barycentric[3],
            )
            .map_err(|error| InterpolationError::InvalidData {
                node: entity,
                reason: format!("target RT0 tabulation failed: {error}"),
            })?;
        let value = [
            tabulation.vector_value(local_dof, 0),
            tabulation.vector_value(local_dof, 1),
            tabulation.vector_value(local_dof, 2),
        ];
        flux += point.weight
            * (value[0] * surface_jacobian[0]
                + value[1] * surface_jacobian[1]
                + value[2] * surface_jacobian[2]);
    }
    Ok(flux)
}

fn checked_moment_ratio(
    flux: f64,
    normalization: f64,
    entity: usize,
) -> Result<f64, InterpolationError> {
    if !flux.is_finite() || !normalization.is_finite() || normalization.abs() <= 1.0e-12 {
        return Err(InterpolationError::InvalidData {
            node: entity,
            reason: format!(
                "RT0 normal-flux normalization is invalid: flux={flux}, normalization={normalization}"
            ),
        });
    }
    let value = flux / normalization;
    if !value.is_finite() {
        return Err(InterpolationError::NonFiniteResult { node: entity });
    }
    Ok(value)
}

fn edge_geometry_points(mesh: &Mesh, edge: (usize, usize)) -> Option<(Point, Point, Point)> {
    for element in &mesh.elements {
        let vertices = element.vertices();
        for (local_edge, &(a, b)) in element.element_type.local_edges().iter().enumerate() {
            let endpoints = (vertices[a], vertices[b]);
            if !((endpoints.0 == edge.0 && endpoints.1 == edge.1)
                || (endpoints.0 == edge.1 && endpoints.1 == edge.0))
            {
                continue;
            }
            let p0 = mesh.nodes[edge.0];
            let p1 = mesh.nodes[edge.1];
            let midpoint = if matches!(element.try_polynomial_order(), Ok(2)) {
                mesh.nodes[element.nodes[element.element_type.num_vertices() + local_edge]]
            } else {
                p0.midpoint(&p1)
            };
            return Some((p0, midpoint, p1));
        }
    }
    None
}

fn edge_curve_point(p0: Point, pm: Point, p1: Point, coordinate: f64) -> [f64; 3] {
    let shape = [
        0.5 * coordinate * (coordinate - 1.0),
        1.0 - coordinate * coordinate,
        0.5 * coordinate * (coordinate + 1.0),
    ];
    [
        shape[0] * p0.x + shape[1] * pm.x + shape[2] * p1.x,
        shape[0] * p0.y + shape[1] * pm.y + shape[2] * p1.y,
        shape[0] * p0.z + shape[1] * pm.z + shape[2] * p1.z,
    ]
}

fn edge_curve_tangent(p0: Point, pm: Point, p1: Point, coordinate: f64) -> [f64; 3] {
    let derivative = [coordinate - 0.5, -2.0 * coordinate, coordinate + 0.5];
    [
        derivative[0] * p0.x + derivative[1] * pm.x + derivative[2] * p1.x,
        derivative[0] * p0.y + derivative[1] * pm.y + derivative[2] * p1.y,
        derivative[0] * p0.z + derivative[1] * pm.z + derivative[2] * p1.z,
    ]
}

fn curved_edge_normal_jacobian(
    p0: Point,
    pm: Point,
    p1: Point,
    coordinate: f64,
    centroid: Point,
) -> Option<[f64; 3]> {
    let point = edge_curve_point(p0, pm, p1, coordinate);
    let tangent = edge_curve_tangent(p0, pm, p1, coordinate);
    let mut normal = [tangent[1], -tangent[0], 0.0];
    let norm = normal[0].hypot(normal[1]);
    if !norm.is_finite() || norm <= 1.0e-14 {
        return None;
    }
    let to_centroid = [centroid.x - point[0], centroid.y - point[1]];
    if normal[0] * to_centroid[0] + normal[1] * to_centroid[1] > 0.0 {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
    }
    Some(normal)
}

fn validate_hcurl_transfer_mesh(
    mesh: &Mesh,
    space: &HcurlSpace,
    role: &'static str,
) -> Result<(), InterpolationError> {
    validate_interpolation_mesh(mesh, role)?;
    if !matches!(mesh.dimension, 2 | 3) {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: format!(
                "H(curl) transfer dimension {} is unsupported",
                mesh.dimension
            ),
        });
    }
    if space.order != 1 {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: format!("order-{} H(curl) transfer is not implemented", space.order),
        });
    }
    if mesh.elements.iter().any(|element| {
        !matches!(
            element.element_type,
            ElementType::Triangle | ElementType::Tetrahedron
        ) || element.try_polynomial_order().is_err()
    }) {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "H(curl) transfer requires P1/P2 triangles or tetrahedra".into(),
        });
    }
    space
        .validate(mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: role,
            reason: error.to_string(),
        })
}

fn validate_hcurl_order2_triangle_transfer_mesh(
    mesh: &Mesh,
    space: &HcurlSpace,
    role: &'static str,
) -> Result<(), InterpolationError> {
    validate_interpolation_mesh(mesh, role)?;
    if mesh.dimension != 2 {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "order-2 triangle H(curl) transfer requires a 2D mesh".into(),
        });
    }
    if space.order != 2 {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: format!(
                "order-2 triangle H(curl) transfer requires an order-2 space, got order-{}",
                space.order
            ),
        });
    }
    if mesh.elements.iter().any(|element| {
        element.element_type != ElementType::Triangle || !matches!(element.nodes.len(), 3 | 6)
    }) {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "order-2 triangle H(curl) transfer requires affine P1 or explicit P2 triangles"
                .into(),
        });
    }
    space
        .validate(mesh)
        .map_err(|error| InterpolationError::InvalidMesh {
            mesh: role,
            reason: error.to_string(),
        })
}

/// Checked conservative scalar state transfer.
///
/// The nodal field is first interpolated, then corrected by a constant so its
/// lumped-nodal integral matches the source integral exactly. This contract is
/// valid for source/target meshes covering the same positive-measure domain;
/// mismatched domains and degenerate cells are rejected.
pub fn interpolate_field_conservative_checked(
    data: &InterpolationData,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    let mut output = interpolate_field_checked(data, source_mesh, target_mesh, source_values)?;
    let source_weights = lumped_nodal_weights(source_mesh, "source")?;
    let target_weights = lumped_nodal_weights(target_mesh, "target")?;
    let source_measure: f64 = source_weights.iter().sum();
    let target_measure: f64 = target_weights.iter().sum();
    if !measures_match(source_measure, target_measure) {
        return Err(InterpolationError::ConservationMismatch {
            source_measure,
            target_measure,
        });
    }
    let source_integral = source_values
        .iter()
        .zip(&source_weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>();
    let target_integral = output
        .iter()
        .zip(&target_weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>();
    let correction = (source_integral - target_integral) / target_measure;
    if !correction.is_finite() {
        return Err(InterpolationError::NonFiniteResult { node: 0 });
    }
    for (node, value) in output.iter_mut().enumerate() {
        *value += correction;
        if !value.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node });
        }
    }
    Ok(output)
}

/// Checked conservative complex state transfer for harmonic fields.
pub fn interpolate_complex_field_conservative_checked(
    data: &InterpolationData,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_values: &[Complex64],
) -> Result<Vec<Complex64>, InterpolationError> {
    let mut output =
        interpolate_complex_field_checked(data, source_mesh, target_mesh, source_values)?;
    let source_weights = lumped_nodal_weights(source_mesh, "source")?;
    let target_weights = lumped_nodal_weights(target_mesh, "target")?;
    let source_measure: f64 = source_weights.iter().sum();
    let target_measure: f64 = target_weights.iter().sum();
    if !measures_match(source_measure, target_measure) {
        return Err(InterpolationError::ConservationMismatch {
            source_measure,
            target_measure,
        });
    }
    let source_integral = source_values
        .iter()
        .zip(&source_weights)
        .map(|(value, weight)| *value * *weight)
        .sum::<Complex64>();
    let target_integral = output
        .iter()
        .zip(&target_weights)
        .map(|(value, weight)| *value * *weight)
        .sum::<Complex64>();
    let correction = (source_integral - target_integral) / target_measure;
    if !correction.re.is_finite() || !correction.im.is_finite() {
        return Err(InterpolationError::NonFiniteResult { node: 0 });
    }
    for (node, value) in output.iter_mut().enumerate() {
        *value += correction;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node });
        }
    }
    Ok(output)
}

/// Apply the mass-weighted adjoint of checked nodal interpolation.
///
/// If `I` is the source-to-target interpolation operator, this returns
/// `M_source⁻¹ Iᵀ M_target target_values`, where the caller supplies the
/// positive diagonal inner-product weights in `source_weights` and
/// `target_weights`.  Supplying the weights explicitly keeps the operation
/// valid for higher-order spaces whose simple lumped weights can be zero or
/// signed.  The returned vector therefore satisfies, up to roundoff,
///
/// `target_weights · (I source_values * target_values)`
/// `= source_weights · (source_values * adjoint target_values)`.
pub fn interpolate_field_adjoint_checked(
    data: &InterpolationData,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    target_values: &[f64],
    source_weights: &[f64],
    target_weights: &[f64],
) -> Result<Vec<f64>, InterpolationError> {
    if let Some(node) = target_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteTarget { node });
    }
    validate_adjoint_inputs(
        data,
        source_mesh,
        target_mesh,
        target_values,
        source_weights,
        target_weights,
    )?;

    let mut weighted_transpose = vec![0.0; source_mesh.num_nodes()];
    for (target_node, entry) in data.target_to_source.iter().enumerate() {
        let (element, barycentric) = entry.as_ref().expect("validated interpolation metadata");
        let weights = element_shape_weights(barycentric, &source_mesh.elements[*element]);
        let target_value = target_values[target_node] * target_weights[target_node];
        for (&source_node, weight) in source_mesh.elements[*element].nodes.iter().zip(weights) {
            weighted_transpose[source_node] += weight * target_value;
        }
    }
    let mut output = Vec::with_capacity(source_mesh.num_nodes());
    for (node, (value, weight)) in weighted_transpose.iter().zip(source_weights).enumerate() {
        let value = value / weight;
        if !value.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node });
        }
        output.push(value);
    }
    Ok(output)
}

/// Apply the mass-weighted Hermitian adjoint of checked complex interpolation.
///
/// The interpolation weights and supplied inner-product weights are real, so
/// the Hermitian adjoint uses the same transpose accumulation while preserving
/// the complex target dual values.
pub fn interpolate_complex_field_adjoint_checked(
    data: &InterpolationData,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    target_values: &[Complex64],
    source_weights: &[f64],
    target_weights: &[f64],
) -> Result<Vec<Complex64>, InterpolationError> {
    if let Some(node) = target_values
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(InterpolationError::NonFiniteTarget { node });
    }
    validate_adjoint_inputs(
        data,
        source_mesh,
        target_mesh,
        target_values,
        source_weights,
        target_weights,
    )?;

    let mut weighted_transpose = vec![Complex64::new(0.0, 0.0); source_mesh.num_nodes()];
    for (target_node, entry) in data.target_to_source.iter().enumerate() {
        let (element, barycentric) = entry.as_ref().expect("validated interpolation metadata");
        let weights = element_shape_weights(barycentric, &source_mesh.elements[*element]);
        let target_value = target_values[target_node] * target_weights[target_node];
        for (&source_node, weight) in source_mesh.elements[*element].nodes.iter().zip(weights) {
            weighted_transpose[source_node] += target_value * weight;
        }
    }
    let mut output = Vec::with_capacity(source_mesh.num_nodes());
    for (node, (value, weight)) in weighted_transpose.iter().zip(source_weights).enumerate() {
        let value = *value / *weight;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(InterpolationError::NonFiniteResult { node });
        }
        output.push(value);
    }
    Ok(output)
}

fn validate_adjoint_inputs<T>(
    data: &InterpolationData,
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    target_values: &[T],
    source_weights: &[f64],
    target_weights: &[f64],
) -> Result<(), InterpolationError> {
    validate_interpolation_mesh(source_mesh, "source")?;
    validate_interpolation_mesh(target_mesh, "target")?;
    data.validate(source_mesh, target_mesh)?;
    if target_values.len() != target_mesh.num_nodes() {
        return Err(InterpolationError::TargetValueCount {
            expected: target_mesh.num_nodes(),
            actual: target_values.len(),
        });
    }
    if source_weights.len() != source_mesh.num_nodes() {
        return Err(InterpolationError::AdjointWeightCount {
            role: "source",
            expected: source_mesh.num_nodes(),
            actual: source_weights.len(),
        });
    }
    if target_weights.len() != target_mesh.num_nodes() {
        return Err(InterpolationError::AdjointWeightCount {
            role: "target",
            expected: target_mesh.num_nodes(),
            actual: target_weights.len(),
        });
    }
    for (node, &weight) in source_weights.iter().enumerate() {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(InterpolationError::InvalidAdjointWeight {
                role: "source",
                node,
            });
        }
    }
    for (node, &weight) in target_weights.iter().enumerate() {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(InterpolationError::InvalidAdjointWeight {
                role: "target",
                node,
            });
        }
    }
    Ok(())
}

fn lumped_nodal_weights(mesh: &Mesh, role: &'static str) -> Result<Vec<f64>, InterpolationError> {
    let mut weights = vec![0.0; mesh.num_nodes()];
    for (element, cell) in mesh.elements.iter().enumerate() {
        let measure = mesh.element_measure(element);
        if !measure.is_finite() || measure <= 0.0 {
            return Err(InterpolationError::InvalidMesh {
                mesh: role,
                reason: format!("element {element} has non-positive measure {measure}"),
            });
        }
        let local_weights = element_integral_weights(cell, measure);
        for (&node, &weight) in cell.nodes.iter().zip(&local_weights) {
            weights[node] += weight;
        }
    }
    Ok(weights)
}

/// Integrals of the local Lagrange shape functions for an affine element.
///
/// P1/Q1 cells retain the historical equal vertex weights. For quadratic
/// cells these are the exact basis-function integrals on the reference cell,
/// mapped by the physical element measure. They therefore include edge,
/// face, and interior state values in conservative transfers.
fn element_integral_weights(element: &crate::mesh::Element, measure: f64) -> Vec<f64> {
    if !element.is_p2() {
        let weight = measure / element.element_type.num_vertices() as f64;
        return vec![weight; element.nodes.len()];
    }

    let reference = match element.element_type {
        ElementType::Triangle => vec![0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ElementType::Quadrilateral => vec![
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            4.0 / 9.0,
        ],
        ElementType::Tetrahedron => vec![
            -1.0 / 20.0,
            -1.0 / 20.0,
            -1.0 / 20.0,
            -1.0 / 20.0,
            1.0 / 5.0,
            1.0 / 5.0,
            1.0 / 5.0,
            1.0 / 5.0,
            1.0 / 5.0,
            1.0 / 5.0,
        ],
        ElementType::Hexahedron => {
            let mut values = vec![1.0 / 216.0; 8];
            values.extend(std::iter::repeat_n(1.0 / 54.0, 12));
            values.extend(std::iter::repeat_n(2.0 / 27.0, 6));
            values.push(8.0 / 27.0);
            values
        }
    };
    reference
        .into_iter()
        .map(|weight| measure * weight)
        .collect()
}

fn measures_match(source: f64, target: f64) -> bool {
    source.is_finite()
        && target.is_finite()
        && (source - target).abs() <= 1e-10 * source.abs().max(target.abs()).max(1.0)
}

fn validate_interpolation_mesh(mesh: &Mesh, role: &'static str) -> Result<(), InterpolationError> {
    if mesh.nodes.is_empty() || mesh.elements.is_empty() {
        return Err(InterpolationError::InvalidMesh {
            mesh: role,
            reason: "nodes and elements must be non-empty".into(),
        });
    }
    if let Err(errors) = mesh.validate_connectivity() {
        let reason = errors
            .into_iter()
            .next()
            .map(|error| error.to_string())
            .unwrap_or_else(|| "mesh connectivity validation failed".into());
        return Err(InterpolationError::InvalidMesh { mesh: role, reason });
    }
    Ok(())
}

fn validate_h1_transfer_geometry(
    mesh: &Mesh,
    role: &'static str,
) -> Result<(), InterpolationError> {
    validate_interpolation_mesh(mesh, role)?;
    for (element, cell) in mesh.elements.iter().enumerate() {
        let affine = cell.nodes.len() == cell.element_type.num_vertices();
        let explicit_p2 = cell.nodes.len() == cell.element_type.num_nodes_p2();
        let explicit_p3 = cell.nodes.len() == cell.element_type.num_nodes_p3();
        if !affine && !explicit_p2 && !explicit_p3 {
            return Err(InterpolationError::InvalidMesh {
                mesh: role,
                reason: format!(
                    "element {element} requires affine, explicit-P2, or explicit-P3 geometry for checked H1 transfer"
                ),
            });
        }
    }
    Ok(())
}

/// Create interpolation data from source mesh to target mesh.
///
/// For each target mesh node, finds the containing source element and
/// computes barycentric coordinates through a deterministic BVH source
/// locator. Shared-boundary ownership remains the lowest source element index.
pub fn create_interpolation_data(target_mesh: &Mesh, source_mesh: &Mesh) -> InterpolationData {
    let mut target_to_source = Vec::with_capacity(target_mesh.num_nodes());

    // Precompute bounding boxes for source elements
    let locator = ElementLocator::new(source_mesh);

    for target_node in &target_mesh.nodes {
        let pt = [target_node.x, target_node.y, target_node.z];
        let found = locator
            .containing_element(source_mesh, pt)
            .and_then(|elem_idx| {
                point_in_element(source_mesh, elem_idx, pt).map(|bary| (elem_idx, bary))
            });

        target_to_source.push(found);
    }

    InterpolationData { target_to_source }
}

/// Interpolate a scalar field from source mesh to target mesh.
///
/// `source_values` has one value per source mesh node (P1/Q1 or P2/Q2
/// Lagrange).
/// Returns one value per target mesh node.
/// Nodes not found in any source element get `default_value`.
pub fn interpolate_field(
    data: &InterpolationData,
    source_mesh: &Mesh,
    source_values: &[f64],
    default_value: f64,
) -> Vec<f64> {
    data.target_to_source
        .iter()
        .map(|entry| match entry {
            Some((elem_idx, bary)) => {
                let element = &source_mesh.elements[*elem_idx];
                let weights = element_shape_weights(bary, element);
                element
                    .nodes
                    .iter()
                    .zip(weights)
                    .map(|(&node, weight)| weight * source_values[node])
                    .sum()
            }
            None => default_value,
        })
        .collect()
}

/// Interpolate a nodal P1 scalar field at one physical point.
///
/// Returns `None` when the point is outside the mesh. Points on a shared face
/// or boundary are accepted by the same barycentric tolerance used by mesh-to-
/// mesh interpolation.
pub fn interpolate_scalar_at_point(
    mesh: &Mesh,
    nodal_values: &[f64],
    point: [f64; 3],
) -> Option<f64> {
    interpolate_scalar_at_point_checked(mesh, nodal_values, point)
        .ok()
        .flatten()
}

/// Checked scalar interpolation at one physical point.
///
/// This is the fallible boundary for callers that need to distinguish an
/// invalid mesh/value buffer from a valid point that is simply outside the
/// domain. The historical [`interpolate_scalar_at_point`] wrapper maps both
/// cases to `None` for compatibility.
pub fn interpolate_scalar_at_point_checked(
    mesh: &Mesh,
    nodal_values: &[f64],
    point: [f64; 3],
) -> Result<Option<f64>, InterpolationError> {
    validate_interpolation_mesh(mesh, "source")?;
    if nodal_values.len() != mesh.num_nodes() {
        return Err(InterpolationError::SourceValueCount {
            expected: mesh.num_nodes(),
            actual: nodal_values.len(),
        });
    }
    if let Some(node) = nodal_values.iter().position(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFiniteSource { node });
    }
    if point.iter().any(|value| !value.is_finite()) {
        return Err(InterpolationError::NonFinitePoint);
    }

    let locator = ElementLocator::new(mesh);
    let Some(element_index) = locator.containing_element(mesh, point) else {
        return Ok(None);
    };
    let barycentric = point_in_element(mesh, element_index, point)
        .ok_or(InterpolationError::TargetElementOutside { element: 0 })?;
    let element = &mesh.elements[element_index];
    let weights = element_shape_weights(&barycentric, element);
    let value = element
        .nodes
        .iter()
        .zip(weights)
        .map(|(&node_index, weight)| weight * nodal_values[node_index])
        .sum();
    Ok(Some(value))
}

/// Evaluate the local nodal shape functions represented by interpolation
/// metadata. The metadata stores P1/Q1 weights because those are sufficient
/// to locate a point and are the long-standing public representation. For a
/// quadratic source element, recover the reference coordinates and evaluate
/// the matching P2/Q2 basis so all local degrees of freedom participate.
fn element_shape_weights(barycentric: &[f64], element: &crate::mesh::Element) -> Vec<f64> {
    if element.nodes.len() == element.element_type.num_nodes_p3() {
        let (xi, eta, zeta) = match element.element_type {
            ElementType::Triangle => (
                barycentric.get(1).copied().unwrap_or_default(),
                barycentric.get(2).copied().unwrap_or_default(),
                0.0,
            ),
            ElementType::Tetrahedron => (
                barycentric.get(1).copied().unwrap_or_default(),
                barycentric.get(2).copied().unwrap_or_default(),
                barycentric.get(3).copied().unwrap_or_default(),
            ),
            ElementType::Quadrilateral => (
                -barycentric[0] + barycentric[1] + barycentric[2] - barycentric[3],
                -barycentric[0] - barycentric[1] + barycentric[2] + barycentric[3],
                0.0,
            ),
            ElementType::Hexahedron => (
                -barycentric[0] + barycentric[1] + barycentric[2] - barycentric[3] - barycentric[4]
                    + barycentric[5]
                    + barycentric[6]
                    - barycentric[7],
                -barycentric[0] - barycentric[1] + barycentric[2] + barycentric[3]
                    - barycentric[4]
                    - barycentric[5]
                    + barycentric[6]
                    + barycentric[7],
                -barycentric[0] - barycentric[1] - barycentric[2] - barycentric[3]
                    + barycentric[4]
                    + barycentric[5]
                    + barycentric[6]
                    + barycentric[7],
            ),
        };
        return evaluate_shape(element.element_type, PolynomialDegree::P3, xi, eta, zeta).values;
    }
    if !element.is_p2() {
        return barycentric.to_vec();
    }

    match element.element_type {
        ElementType::Triangle => p2_triangle(barycentric[1], barycentric[2]).to_vec(),
        ElementType::Tetrahedron => {
            p2_tetrahedron(barycentric[1], barycentric[2], barycentric[3]).to_vec()
        }
        ElementType::Quadrilateral => {
            let [xi, eta] = q1_reference_coordinates(barycentric);
            q2_quadrilateral(xi, eta).to_vec()
        }
        ElementType::Hexahedron => {
            let [xi, eta, zeta] = q1_reference_coordinates_3d(barycentric);
            q2_hexahedron(xi, eta, zeta).to_vec()
        }
    }
}

/// Recover tensor-product reference coordinates from Q1 nodal weights.
fn q1_reference_coordinates(weights: &[f64]) -> [f64; 2] {
    [
        -weights[0] + weights[1] + weights[2] - weights[3],
        -weights[0] - weights[1] + weights[2] + weights[3],
    ]
}

/// Recover tensor-product reference coordinates from Q1 hexahedral weights.
fn q1_reference_coordinates_3d(weights: &[f64]) -> [f64; 3] {
    [
        -weights[0] + weights[1] + weights[2] - weights[3] - weights[4] + weights[5] + weights[6]
            - weights[7],
        -weights[0] - weights[1] + weights[2] + weights[3] - weights[4] - weights[5]
            + weights[6]
            + weights[7],
        -weights[0] - weights[1] - weights[2] - weights[3]
            + weights[4]
            + weights[5]
            + weights[6]
            + weights[7],
    ]
}

/// Compute the bounding box of an element.
fn element_centroid(
    mesh: &Mesh,
    element: &crate::mesh::Element,
) -> Result<[f64; 3], InterpolationError> {
    let vertices = element.vertices();
    if vertices.is_empty() {
        return Err(InterpolationError::InvalidMesh {
            mesh: "target",
            reason: "element has no vertices".into(),
        });
    }
    let mut centroid = [0.0; 3];
    for &node in vertices {
        let Some(point) = mesh.nodes.get(node) else {
            return Err(InterpolationError::InvalidMesh {
                mesh: "target",
                reason: format!("element references node {node} out of bounds"),
            });
        };
        centroid[0] += point.x;
        centroid[1] += point.y;
        centroid[2] += point.z;
    }
    let scale = 1.0 / vertices.len() as f64;
    centroid
        .iter_mut()
        .for_each(|component| *component *= scale);
    if centroid.iter().any(|component| !component.is_finite()) {
        return Err(InterpolationError::NonFinitePoint);
    }
    Ok(centroid)
}

#[cfg(test)]
fn locate_element_containing_point(mesh: &Mesh, point: [f64; 3]) -> Option<usize> {
    (0..mesh.num_elements()).find(|&element| {
        let (minimum, maximum) = element_bounding_box(mesh, element);
        let tolerance = 1e-10;
        if (0..3).any(|axis| {
            point[axis] < minimum[axis] - tolerance || point[axis] > maximum[axis] + tolerance
        }) {
            return false;
        }
        point_in_element(mesh, element, point).is_some()
    })
}

fn element_bounding_box(mesh: &Mesh, elem_idx: usize) -> ([f64; 3], [f64; 3]) {
    let nodes = &mesh.elements[elem_idx].nodes;
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for &vi in nodes {
        let p = &mesh.nodes[vi];
        min[0] = min[0].min(p.x);
        min[1] = min[1].min(p.y);
        min[2] = min[2].min(p.z);
        max[0] = max[0].max(p.x);
        max[1] = max[1].max(p.y);
        max[2] = max[2].max(p.z);
    }
    (min, max)
}

/// Check if a point lies inside an element and return barycentric coordinates.
///
/// Returns `None` if the point is outside (any barycentric coordinate < -tol).
fn point_in_element(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let elem = &mesh.elements[elem_idx];
    match elem.element_type {
        ElementType::Triangle if elem.nodes.len() == elem.element_type.num_nodes_p3() => {
            point_in_isoparametric_element(mesh, elem_idx, pt, PolynomialDegree::P3)
        }
        ElementType::Quadrilateral if elem.nodes.len() == elem.element_type.num_nodes_p3() => {
            point_in_isoparametric_element(mesh, elem_idx, pt, PolynomialDegree::P3)
        }
        ElementType::Tetrahedron if elem.nodes.len() == elem.element_type.num_nodes_p3() => {
            point_in_isoparametric_element(mesh, elem_idx, pt, PolynomialDegree::P3)
        }
        ElementType::Hexahedron if elem.nodes.len() == elem.element_type.num_nodes_p3() => {
            point_in_isoparametric_element(mesh, elem_idx, pt, PolynomialDegree::P3)
        }
        ElementType::Triangle if elem.is_p2() => point_in_triangle_p2(mesh, elem_idx, pt),
        ElementType::Quadrilateral if elem.is_p2() => point_in_quadrilateral_p2(mesh, elem_idx, pt),
        ElementType::Tetrahedron if elem.is_p2() => point_in_tetrahedron_p2(mesh, elem_idx, pt),
        ElementType::Hexahedron if elem.is_p2() => point_in_hexahedron_p2(mesh, elem_idx, pt),
        ElementType::Triangle => point_in_triangle(mesh, elem_idx, pt),
        ElementType::Quadrilateral => point_in_quadrilateral(mesh, elem_idx, pt),
        ElementType::Tetrahedron => point_in_tetrahedron(mesh, elem_idx, pt),
        ElementType::Hexahedron => point_in_hexahedron(mesh, elem_idx, pt),
    }
}

/// Locate a point in an explicit P3 isoparametric element by Newton iteration.
///
/// The existing P1/P2 helpers intentionally keep their element-specific
/// closed forms. P3 geometry needs the checked reference-shape evaluator so
/// that curved edge, face, and cell nodes participate in both the map and its
/// Jacobian. The returned values are the P1 reference weights, matching the
/// convention used by the other point-location helpers.
fn point_in_isoparametric_element(
    mesh: &Mesh,
    elem_idx: usize,
    point: [f64; 3],
    geometry_degree: PolynomialDegree,
) -> Option<Vec<f64>> {
    let element = mesh.elements.get(elem_idx)?;
    if element.nodes.len() != element.element_type.num_nodes_p3()
        || geometry_degree != PolynomialDegree::P3
    {
        return None;
    }

    let mut reference = match element.element_type {
        ElementType::Triangle => {
            let barycentric =
                point_in_triangle(mesh, elem_idx, point).unwrap_or_else(|| vec![1.0 / 3.0; 3]);
            [barycentric[1], barycentric[2], 0.0]
        }
        ElementType::Tetrahedron => {
            let barycentric =
                point_in_tetrahedron(mesh, elem_idx, point).unwrap_or_else(|| vec![0.25; 4]);
            [barycentric[1], barycentric[2], barycentric[3]]
        }
        ElementType::Quadrilateral | ElementType::Hexahedron => [0.0; 3],
    };

    let physical_dimension = element.element_type.dimension();
    let mut converged = false;
    for _ in 0..40 {
        let shape = try_evaluate_shape(
            element.element_type,
            geometry_degree,
            reference[0],
            reference[1],
            reference[2],
        )
        .ok()?;
        if shape.values.len() != element.nodes.len()
            || shape.gradients.len() != shape.values.len() * shape.grad_dim
        {
            return None;
        }

        let mut mapped = [0.0; 3];
        let mut jacobian = [[0.0; 3]; 3];
        for (local, &node_index) in element.nodes.iter().enumerate() {
            let node = mesh.nodes.get(node_index)?;
            let coordinates = [node.x, node.y, node.z];
            for physical in 0..physical_dimension {
                mapped[physical] += shape.values[local] * coordinates[physical];
                for (parametric, slot) in jacobian[physical][..physical_dimension]
                    .iter_mut()
                    .enumerate()
                {
                    *slot += shape.gradients[local * shape.grad_dim + parametric]
                        * coordinates[physical];
                }
            }
        }

        let delta = if physical_dimension == 2 {
            let delta = solve_2x2(
                [
                    [jacobian[0][0], jacobian[0][1]],
                    [jacobian[1][0], jacobian[1][1]],
                ],
                [mapped[0] - point[0], mapped[1] - point[1]],
            )?;
            [delta[0], delta[1], 0.0]
        } else {
            solve_3x3(
                jacobian,
                [
                    mapped[0] - point[0],
                    mapped[1] - point[1],
                    mapped[2] - point[2],
                ],
            )?
        };

        for axis in 0..physical_dimension {
            reference[axis] -= delta[axis];
        }
        if delta[..physical_dimension]
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
            < 1.0e-12
        {
            converged = true;
            break;
        }
    }

    if !converged {
        return None;
    }

    let reference_weights = evaluate_shape(
        element.element_type,
        PolynomialDegree::P1,
        reference[0],
        reference[1],
        reference[2],
    )
    .values;
    if reference_weights.iter().any(|value| *value < -1.0e-8)
        || reference_weights.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some(reference_weights)
}

fn point_in_triangle_p2(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let element = &mesh.elements[elem_idx];
    let mut reference = point_in_triangle(mesh, elem_idx, pt).unwrap_or_else(|| vec![1.0 / 3.0; 3]);
    let mut reference_coordinates = [reference[1], reference[2]];

    for _ in 0..24 {
        let values = p2_triangle(reference_coordinates[0], reference_coordinates[1]);
        let gradients = p2_triangle_grad(reference_coordinates[0], reference_coordinates[1]);
        let mut mapped = [0.0_f64; 2];
        let mut jacobian = [[0.0_f64; 2]; 2];
        for (local, &node_index) in element.nodes.iter().enumerate() {
            let node = &mesh.nodes[node_index];
            for physical in 0..2 {
                let coordinate = if physical == 0 { node.x } else { node.y };
                mapped[physical] += values[local] * coordinate;
                for parametric in 0..2 {
                    jacobian[physical][parametric] += gradients[local][parametric] * coordinate;
                }
            }
        }
        let delta = solve_2x2(jacobian, [mapped[0] - pt[0], mapped[1] - pt[1]])?;
        reference_coordinates[0] -= delta[0];
        reference_coordinates[1] -= delta[1];
        if delta[0].abs().max(delta[1].abs()) < 1e-12 {
            break;
        }
    }

    let values = p1_triangle(reference_coordinates[0], reference_coordinates[1]);
    let tolerance = 1e-8;
    if values.iter().any(|&value| value < -tolerance) {
        return None;
    }
    reference = values.to_vec();
    if mapped_element_point_matches(mesh, element, &reference, pt) {
        Some(reference)
    } else {
        None
    }
}

fn point_in_tetrahedron_p2(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let element = &mesh.elements[elem_idx];
    let mut reference = point_in_tetrahedron(mesh, elem_idx, pt).unwrap_or_else(|| vec![0.25; 4]);
    let mut reference_coordinates = [reference[1], reference[2], reference[3]];

    for _ in 0..30 {
        let values = p2_tetrahedron(
            reference_coordinates[0],
            reference_coordinates[1],
            reference_coordinates[2],
        );
        let gradients = p2_tetrahedron_grad(
            reference_coordinates[0],
            reference_coordinates[1],
            reference_coordinates[2],
        );
        let mut mapped = [0.0_f64; 3];
        let mut jacobian = [[0.0_f64; 3]; 3];
        for (local, &node_index) in element.nodes.iter().enumerate() {
            let node = &mesh.nodes[node_index];
            let coordinates = [node.x, node.y, node.z];
            for physical in 0..3 {
                mapped[physical] += values[local] * coordinates[physical];
                for parametric in 0..3 {
                    jacobian[physical][parametric] +=
                        gradients[local][parametric] * coordinates[physical];
                }
            }
        }
        let delta = solve_3x3(
            jacobian,
            [mapped[0] - pt[0], mapped[1] - pt[1], mapped[2] - pt[2]],
        )?;
        for axis in 0..3 {
            reference_coordinates[axis] -= delta[axis];
        }
        if delta
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
            < 1e-12
        {
            break;
        }
    }

    let values = vec![
        1.0 - reference_coordinates[0] - reference_coordinates[1] - reference_coordinates[2],
        reference_coordinates[0],
        reference_coordinates[1],
        reference_coordinates[2],
    ];
    if values.iter().any(|&value| value < -1e-8) {
        return None;
    }
    reference = values;
    mapped_element_point_matches(mesh, element, &reference, pt).then_some(reference)
}

fn point_in_quadrilateral(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let vertices = mesh.elements[elem_idx].vertices();
    let mut reference = [0.0_f64; 2];

    for _ in 0..20 {
        let values = q1_quadrilateral(reference[0], reference[1]);
        let gradients = q1_quadrilateral_grad(reference[0], reference[1]);
        let mut mapped = [0.0_f64; 3];
        let mut jacobian = [[0.0_f64; 2]; 3];
        for (local, &node_index) in vertices.iter().enumerate() {
            let node = &mesh.nodes[node_index];
            let coordinates = [node.x, node.y, node.z];
            for physical in 0..3 {
                mapped[physical] += values[local] * coordinates[physical];
                for parametric in 0..2 {
                    jacobian[physical][parametric] +=
                        gradients[local][parametric] * coordinates[physical];
                }
            }
        }

        let residual = [mapped[0] - pt[0], mapped[1] - pt[1], mapped[2] - pt[2]];
        let mut normal = [[0.0_f64; 2]; 2];
        let mut projected = [0.0_f64; 2];
        for physical in 0..3 {
            for row in 0..2 {
                projected[row] += jacobian[physical][row] * residual[physical];
                for col in 0..2 {
                    normal[row][col] += jacobian[physical][row] * jacobian[physical][col];
                }
            }
        }
        let delta = solve_2x2(normal, projected)?;
        reference[0] -= delta[0];
        reference[1] -= delta[1];
        if delta[0].abs().max(delta[1].abs()) < 1e-12 {
            break;
        }
    }

    let tolerance = 1e-8;
    if reference
        .iter()
        .any(|&coordinate| coordinate < -1.0 - tolerance || coordinate > 1.0 + tolerance)
    {
        return None;
    }
    let values = q1_quadrilateral(reference[0], reference[1]);
    mapped_point_matches(mesh, vertices, &values, pt).then(|| values.to_vec())
}

fn point_in_quadrilateral_p2(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let element = &mesh.elements[elem_idx];
    let mut reference = [0.0_f64; 2];

    for _ in 0..30 {
        let values = q2_quadrilateral(reference[0], reference[1]);
        let gradients = q2_quadrilateral_grad(reference[0], reference[1]);
        let mut mapped = [0.0_f64; 3];
        let mut jacobian = [[0.0_f64; 2]; 3];
        for (local, &node_index) in element.nodes.iter().enumerate() {
            let node = &mesh.nodes[node_index];
            let coordinates = [node.x, node.y, node.z];
            for physical in 0..3 {
                mapped[physical] += values[local] * coordinates[physical];
                for parametric in 0..2 {
                    jacobian[physical][parametric] +=
                        gradients[local][parametric] * coordinates[physical];
                }
            }
        }
        let residual = [mapped[0] - pt[0], mapped[1] - pt[1], mapped[2] - pt[2]];
        let mut normal = [[0.0_f64; 2]; 2];
        let mut projected = [0.0_f64; 2];
        for physical in 0..3 {
            for row in 0..2 {
                projected[row] += jacobian[physical][row] * residual[physical];
                for col in 0..2 {
                    normal[row][col] += jacobian[physical][row] * jacobian[physical][col];
                }
            }
        }
        let delta = solve_2x2(normal, projected)?;
        reference[0] -= delta[0];
        reference[1] -= delta[1];
        if delta[0].abs().max(delta[1].abs()) < 1e-12 {
            break;
        }
    }

    if reference.iter().any(|&coordinate| {
        !coordinate.is_finite() || !(-1.0 - 1e-8..=1.0 + 1e-8).contains(&coordinate)
    }) {
        return None;
    }
    let barycentric = q1_quadrilateral(reference[0], reference[1]).to_vec();
    mapped_element_point_matches(mesh, element, &barycentric, pt).then_some(barycentric)
}

fn point_in_hexahedron(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let vertices = mesh.elements[elem_idx].vertices();
    let mut reference = [0.0_f64; 3];

    for _ in 0..24 {
        let values = q1_hexahedron(reference[0], reference[1], reference[2]);
        let gradients = q1_hexahedron_grad(reference[0], reference[1], reference[2]);
        let mut mapped = [0.0_f64; 3];
        let mut jacobian = [[0.0_f64; 3]; 3];
        for (local, &node_index) in vertices.iter().enumerate() {
            let node = &mesh.nodes[node_index];
            let coordinates = [node.x, node.y, node.z];
            for physical in 0..3 {
                mapped[physical] += values[local] * coordinates[physical];
                for parametric in 0..3 {
                    jacobian[physical][parametric] +=
                        gradients[local][parametric] * coordinates[physical];
                }
            }
        }

        let residual = [mapped[0] - pt[0], mapped[1] - pt[1], mapped[2] - pt[2]];
        let delta = solve_3x3(jacobian, residual)?;
        for axis in 0..3 {
            reference[axis] -= delta[axis];
        }
        if delta
            .iter()
            .fold(0.0_f64, |max, value| max.max(value.abs()))
            < 1e-12
        {
            break;
        }
    }

    let tolerance = 1e-8;
    if reference
        .iter()
        .any(|&coordinate| coordinate < -1.0 - tolerance || coordinate > 1.0 + tolerance)
    {
        return None;
    }
    let values = q1_hexahedron(reference[0], reference[1], reference[2]);
    mapped_point_matches(mesh, vertices, &values, pt).then(|| values.to_vec())
}

fn point_in_hexahedron_p2(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let element = &mesh.elements[elem_idx];
    let mut reference = [0.0_f64; 3];

    for _ in 0..36 {
        let values = q2_hexahedron(reference[0], reference[1], reference[2]);
        let gradients = q2_hexahedron_grad(reference[0], reference[1], reference[2]);
        let mut mapped = [0.0_f64; 3];
        let mut jacobian = [[0.0_f64; 3]; 3];
        for (local, &node_index) in element.nodes.iter().enumerate() {
            let node = &mesh.nodes[node_index];
            let coordinates = [node.x, node.y, node.z];
            for physical in 0..3 {
                mapped[physical] += values[local] * coordinates[physical];
                for parametric in 0..3 {
                    jacobian[physical][parametric] +=
                        gradients[local][parametric] * coordinates[physical];
                }
            }
        }
        let delta = solve_3x3(
            jacobian,
            [mapped[0] - pt[0], mapped[1] - pt[1], mapped[2] - pt[2]],
        )?;
        for axis in 0..3 {
            reference[axis] -= delta[axis];
        }
        if delta
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
            < 1e-12
        {
            break;
        }
    }

    if reference.iter().any(|&coordinate| {
        !coordinate.is_finite() || !(-1.0 - 1e-8..=1.0 + 1e-8).contains(&coordinate)
    }) {
        return None;
    }
    let barycentric = q1_hexahedron(reference[0], reference[1], reference[2]).to_vec();
    mapped_element_point_matches(mesh, element, &barycentric, pt).then_some(barycentric)
}

fn mapped_point_matches(mesh: &Mesh, vertices: &[usize], weights: &[f64], point: [f64; 3]) -> bool {
    let mut mapped = [0.0_f64; 3];
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for (&node_index, &weight) in vertices.iter().zip(weights) {
        let node = &mesh.nodes[node_index];
        let coordinates = [node.x, node.y, node.z];
        for axis in 0..3 {
            mapped[axis] += weight * coordinates[axis];
            minimum[axis] = minimum[axis].min(coordinates[axis]);
            maximum[axis] = maximum[axis].max(coordinates[axis]);
        }
    }
    let scale = (0..3)
        .map(|axis| maximum[axis] - minimum[axis])
        .fold(0.0_f64, f64::max)
        .max(1.0);
    mapped
        .iter()
        .zip(point)
        .all(|(&actual, expected)| (actual - expected).abs() <= 1e-9 * scale)
}

fn mapped_element_point_matches(
    mesh: &Mesh,
    element: &crate::mesh::Element,
    barycentric: &[f64],
    point: [f64; 3],
) -> bool {
    let weights = element_shape_weights(barycentric, element);
    let mut mapped = [0.0_f64; 3];
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for (&node_index, &weight) in element.nodes.iter().zip(&weights) {
        let node = &mesh.nodes[node_index];
        let coordinates = [node.x, node.y, node.z];
        for axis in 0..3 {
            mapped[axis] += weight * coordinates[axis];
            minimum[axis] = minimum[axis].min(coordinates[axis]);
            maximum[axis] = maximum[axis].max(coordinates[axis]);
        }
    }
    let scale = (0..3)
        .map(|axis| maximum[axis] - minimum[axis])
        .fold(0.0_f64, f64::max)
        .max(1.0);
    mapped
        .iter()
        .zip(point)
        .all(|(&actual, expected)| (actual - expected).abs() <= 1e-9 * scale)
}

fn solve_2x2(matrix: [[f64; 2]; 2], rhs: [f64; 2]) -> Option<[f64; 2]> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if !determinant.is_finite() || determinant.abs() < 1e-24 {
        return None;
    }
    Some([
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ])
}

fn solve_3x3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let mut augmented = [[0.0_f64; 4]; 3];
    for row in 0..3 {
        augmented[row][..3].copy_from_slice(&matrix[row]);
        augmented[row][3] = rhs[row];
    }
    for pivot_column in 0..3 {
        let pivot_row = (pivot_column..3).max_by(|&left, &right| {
            augmented[left][pivot_column]
                .abs()
                .total_cmp(&augmented[right][pivot_column].abs())
        })?;
        if !augmented[pivot_row][pivot_column].is_finite()
            || augmented[pivot_row][pivot_column].abs() < 1e-18
        {
            return None;
        }
        augmented.swap(pivot_column, pivot_row);
        let pivot = augmented[pivot_column][pivot_column];
        for value in &mut augmented[pivot_column][pivot_column..] {
            *value /= pivot;
        }
        let pivot_values = augmented[pivot_column];
        for (row, row_values) in augmented.iter_mut().enumerate() {
            if row == pivot_column {
                continue;
            }
            let factor = row_values[pivot_column];
            for (value, &pivot_value) in row_values[pivot_column..]
                .iter_mut()
                .zip(&pivot_values[pivot_column..])
            {
                *value -= factor * pivot_value;
            }
        }
    }
    let solution = [augmented[0][3], augmented[1][3], augmented[2][3]];
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

/// Barycentric coordinates for a point in a triangle (2D).
fn point_in_triangle(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let verts = mesh.elements[elem_idx].vertices();
    let p0 = &mesh.nodes[verts[0]];
    let p1 = &mesh.nodes[verts[1]];
    let p2 = &mesh.nodes[verts[2]];

    let v0 = [p1.x - p0.x, p1.y - p0.y];
    let v1 = [p2.x - p0.x, p2.y - p0.y];
    let v2 = [pt[0] - p0.x, pt[1] - p0.y];

    let d00 = v0[0] * v0[0] + v0[1] * v0[1];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1];

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 {
        return None; // Degenerate triangle
    }

    let l1 = (d11 * d20 - d01 * d21) / denom;
    let l2 = (d00 * d21 - d01 * d20) / denom;
    let l0 = 1.0 - l1 - l2;

    let tol = -1e-8;
    if l0 >= tol && l1 >= tol && l2 >= tol {
        Some(vec![l0, l1, l2])
    } else {
        None
    }
}

/// Barycentric coordinates for a point in a tetrahedron (3D).
fn point_in_tetrahedron(mesh: &Mesh, elem_idx: usize, pt: [f64; 3]) -> Option<Vec<f64>> {
    let verts = mesh.elements[elem_idx].vertices();
    let p0 = &mesh.nodes[verts[0]];
    let p1 = &mesh.nodes[verts[1]];
    let p2 = &mesh.nodes[verts[2]];
    let p3 = &mesh.nodes[verts[3]];

    // Solve T * [l1, l2, l3]^T = pt - p0 where T = [p1-p0, p2-p0, p3-p0]
    let t = [
        [p1.x - p0.x, p2.x - p0.x, p3.x - p0.x],
        [p1.y - p0.y, p2.y - p0.y, p3.y - p0.y],
        [p1.z - p0.z, p2.z - p0.z, p3.z - p0.z],
    ];
    let rhs = [pt[0] - p0.x, pt[1] - p0.y, pt[2] - p0.z];

    // Cramer's rule for 3×3
    let det = t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1])
        - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0])
        + t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0]);

    if det.abs() < 1e-30 {
        return None;
    }

    let inv_det = 1.0 / det;
    let l1 = inv_det
        * (rhs[0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1])
            - t[0][1] * (rhs[1] * t[2][2] - t[1][2] * rhs[2])
            + t[0][2] * (rhs[1] * t[2][1] - t[1][1] * rhs[2]));
    let l2 = inv_det
        * (t[0][0] * (rhs[1] * t[2][2] - t[1][2] * rhs[2])
            - rhs[0] * (t[1][0] * t[2][2] - t[1][2] * t[2][0])
            + t[0][2] * (t[1][0] * rhs[2] - rhs[1] * t[2][0]));
    let l3 = inv_det
        * (t[0][0] * (t[1][1] * rhs[2] - rhs[1] * t[2][1])
            - t[0][1] * (t[1][0] * rhs[2] - rhs[1] * t[2][0])
            + rhs[0] * (t[1][0] * t[2][1] - t[1][1] * t[2][0]));
    let l0 = 1.0 - l1 - l2 - l3;

    let tol = -1e-8;
    if l0 >= tol && l1 >= tol && l2 >= tol && l3 >= tol {
        Some(vec![l0, l1, l2, l3])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{
        Mesh,
        refinement::{refine_elements_high_order_checked, uniform_refine_checked},
        unit_cube_hexahedra, unit_cube_tetrahedra, unit_square_quads, unit_square_triangles,
    };

    #[test]
    fn test_interpolation_same_mesh_for_all_supported_elements() {
        let cases: [(&str, fn(usize) -> Mesh); 4] = [
            ("triangle", unit_square_triangles),
            ("quadrilateral", unit_square_quads),
            ("tetrahedron", unit_cube_tetrahedra),
            ("hexahedron", unit_cube_hexahedra),
        ];

        for (element, make_mesh) in cases {
            let mesh = make_mesh(2);
            let data = create_interpolation_data(&mesh, &mesh);
            let source_values: Vec<f64> = mesh
                .nodes
                .iter()
                .map(|node| 1.0 + node.x + 2.0 * node.y + 3.0 * node.z)
                .collect();
            let transferred = interpolate_field(&data, &mesh, &source_values, f64::NAN);

            for (node_index, entry) in data.target_to_source.iter().enumerate() {
                let (_, weights) = entry
                    .as_ref()
                    .unwrap_or_else(|| panic!("{element} node {node_index} was not located"));
                assert!(
                    (weights.iter().sum::<f64>() - 1.0).abs() < 1e-10,
                    "{element} node {node_index} violates partition of unity"
                );
                assert!(
                    (transferred[node_index] - source_values[node_index]).abs() < 1e-10,
                    "{element} same-mesh transfer changed node {node_index}"
                );
            }
        }
    }

    #[test]
    fn element_locator_matches_linear_ownership_on_nodes_and_centroids() {
        let mesh = unit_square_triangles(16);
        let locator = ElementLocator::new(&mesh);

        for node in &mesh.nodes {
            let point = [node.x, node.y, node.z];
            assert_eq!(
                locator.containing_element(&mesh, point),
                locate_element_containing_point(&mesh, point),
                "locator changed shared-boundary ownership at {point:?}"
            );
        }
        for (element, cell) in mesh.elements.iter().enumerate() {
            let point = element_centroid(&mesh, cell).expect("valid element centroid");
            assert_eq!(locator.containing_element(&mesh, point), Some(element));
        }
        assert_eq!(locator.containing_element(&mesh, [-0.1, 0.5, 0.0]), None);
        assert_eq!(
            locator.containing_element(&mesh, [f64::NAN, 0.5, 0.0]),
            None
        );
    }

    #[test]
    fn element_locator_preserves_checked_scalar_transfer_contract() {
        let source = unit_square_triangles(16);
        let target = unit_square_triangles(32);
        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");
        let source_values: Vec<f64> = source
            .nodes
            .iter()
            .map(|node| 1.25 - 0.75 * node.x + 2.0 * node.y)
            .collect();
        let target_values = interpolate_field_checked(&data, &source, &target, &source_values)
            .expect("BVH-backed scalar transfer should succeed");
        for (node, value) in target.nodes.iter().zip(target_values) {
            let expected = 1.25 - 0.75 * node.x + 2.0 * node.y;
            assert!((value - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_interpolation_rejects_outside_points_for_all_supported_elements() {
        let meshes = [
            unit_square_triangles(1),
            unit_square_quads(1),
            unit_cube_tetrahedra(1),
            unit_cube_hexahedra(1),
        ];
        for mesh in meshes {
            let values = vec![1.0; mesh.num_nodes()];
            assert!(interpolate_scalar_at_point(&mesh, &values, [1.5, 1.5, 1.5]).is_none());
        }
    }

    #[test]
    fn checked_point_interpolation_rejects_malformed_inputs() {
        let mesh = unit_square_triangles(1);
        let values = vec![1.0; mesh.num_nodes()];
        assert!(matches!(
            interpolate_scalar_at_point_checked(
                &mesh,
                &values[..values.len() - 1],
                [0.0, 0.0, 0.0]
            ),
            Err(InterpolationError::SourceValueCount { .. })
        ));
        assert!(matches!(
            interpolate_scalar_at_point_checked(&mesh, &values, [f64::NAN, 0.0, 0.0]),
            Err(InterpolationError::NonFinitePoint)
        ));
        assert!(matches!(
            interpolate_scalar_at_point_checked(&mesh, &[f64::INFINITY; 4], [0.0, 0.0, 0.0]),
            Err(InterpolationError::NonFiniteSource { node: 0 })
        ));
    }

    #[test]
    fn test_reference_inversion_on_distorted_quad_and_hex() {
        let mapped_point = |mesh: &Mesh, vertices: &[usize], weights: &[f64]| {
            let mut point = [0.0; 3];
            for (&node_index, &weight) in vertices.iter().zip(weights) {
                let node = &mesh.nodes[node_index];
                point[0] += weight * node.x;
                point[1] += weight * node.y;
                point[2] += weight * node.z;
            }
            point
        };

        let mut quad = unit_square_quads(1);
        let quad_vertices = quad.elements[0].vertices().to_vec();
        let quad_coordinates = [[0.0, 0.0], [1.2, -0.1], [1.05, 1.2], [-0.15, 0.85]];
        for (&node_index, coordinates) in quad_vertices.iter().zip(quad_coordinates) {
            quad.nodes[node_index].x = coordinates[0];
            quad.nodes[node_index].y = coordinates[1];
        }
        let expected_quad = q1_quadrilateral(0.27, -0.34);
        let quad_point = mapped_point(&quad, &quad_vertices, &expected_quad);
        let actual_quad = point_in_element(&quad, 0, quad_point).expect("distorted quad inversion");
        for (actual, expected) in actual_quad.iter().zip(expected_quad) {
            assert!((actual - expected).abs() < 1e-10);
        }

        let mut hex = unit_cube_hexahedra(1);
        let hex_vertices = hex.elements[0].vertices().to_vec();
        let displaced = [
            (5, [1.05, -0.05, 0.95]),
            (6, [1.2, 0.85, 1.1]),
            (7, [-0.1, 1.05, 0.9]),
        ];
        for (local_index, coordinates) in displaced {
            let node = &mut hex.nodes[hex_vertices[local_index]];
            node.x = coordinates[0];
            node.y = coordinates[1];
            node.z = coordinates[2];
        }
        let expected_hex = q1_hexahedron(-0.21, 0.32, 0.17);
        let hex_point = mapped_point(&hex, &hex_vertices, &expected_hex);
        let actual_hex = point_in_element(&hex, 0, hex_point).expect("distorted hex inversion");
        for (actual, expected) in actual_hex.iter().zip(expected_hex) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_interpolate_linear_field_exact_on_quadrilateral_mesh() {
        let mesh = unit_square_quads(2);
        let values: Vec<f64> = mesh
            .nodes
            .iter()
            .map(|node| 2.0 * node.x - 3.0 * node.y + 0.5)
            .collect();
        let point = [0.37, 0.61, 0.0];
        let actual = interpolate_scalar_at_point(&mesh, &values, point)
            .expect("interior quadrilateral point should be located");
        let expected = 2.0 * point[0] - 3.0 * point[1] + 0.5;
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn test_interpolate_linear_field_exact_on_hexahedral_mesh() {
        let mesh = unit_cube_hexahedra(2);
        let values: Vec<f64> = mesh
            .nodes
            .iter()
            .map(|node| node.x + 2.0 * node.y - 0.5 * node.z)
            .collect();
        let point = [0.37, 0.61, 0.42];
        let actual = interpolate_scalar_at_point(&mesh, &values, point)
            .expect("interior hexahedral point should be located");
        let expected = point[0] + 2.0 * point[1] - 0.5 * point[2];
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn test_interpolate_linear_field_exact() {
        let source = unit_square_triangles(4);
        let target = unit_square_triangles(8); // finer mesh

        // Source field: f(x,y) = 2x + 3y (linear — should be exact with P1)
        let source_values: Vec<f64> = source.nodes.iter().map(|n| 2.0 * n.x + 3.0 * n.y).collect();

        let data = create_interpolation_data(&target, &source);
        let target_values = interpolate_field(&data, &source, &source_values, 0.0);

        for (i, node) in target.nodes.iter().enumerate() {
            let expected = 2.0 * node.x + 3.0 * node.y;
            assert!(
                (target_values[i] - expected).abs() < 1e-8,
                "Node {i} ({:.3}, {:.3}): got {:.6}, expected {:.6}",
                node.x,
                node.y,
                target_values[i],
                expected
            );
        }
    }

    #[test]
    fn checked_interpolation_transfers_linear_field_between_refinements() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(2);
        let source_values: Vec<f64> = source.nodes.iter().map(|node| node.x + node.y).collect();

        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");
        let target_values = interpolate_field_checked(&data, &source, &target, &source_values)
            .expect("linear field transfer should succeed");

        for (node, value) in target.nodes.iter().zip(target_values) {
            assert!((value - (node.x + node.y)).abs() < 1e-12);
        }
    }

    #[test]
    fn remesh_refinement_transfer_preserves_linear_state() {
        let source = unit_square_triangles(1);
        let mut target = source.clone();
        let refinement = uniform_refine_checked(&mut target).expect("checked remesh refinement");
        assert_eq!(refinement.removed_elements.len(), source.num_elements());
        assert!(target.num_nodes() > source.num_nodes());

        let source_values: Vec<f64> = source
            .nodes
            .iter()
            .map(|node| 1.25 - 0.75 * node.x + 2.0 * node.y)
            .collect();
        let data = create_interpolation_data_checked(&target, &source)
            .expect("refined mesh stays inside source domain");
        let transferred = interpolate_field_checked(&data, &source, &target, &source_values)
            .expect("refined state transfer should succeed");
        for (node, value) in target.nodes.iter().zip(transferred) {
            let expected = 1.25 - 0.75 * node.x + 2.0 * node.y;
            assert!((value - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn checked_interpolation_rejects_target_outside_source_domain() {
        let source = unit_square_triangles(1);
        let mut target = unit_square_triangles(1);
        target.nodes[0].x = 2.0;

        let error = create_interpolation_data_checked(&target, &source)
            .expect_err("target outside source domain must be rejected");
        assert_eq!(error, InterpolationError::TargetOutside { node: 0 });
    }

    #[test]
    fn checked_interpolation_rejects_invalid_source_values() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(1);
        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");

        assert_eq!(
            interpolate_field_checked(&data, &source, &target, &[]),
            Err(InterpolationError::SourceValueCount {
                expected: source.num_nodes(),
                actual: 0,
            })
        );

        let mut values = vec![0.0; source.num_nodes()];
        values[1] = f64::NAN;
        assert_eq!(
            interpolate_field_checked(&data, &source, &target, &values),
            Err(InterpolationError::NonFiniteSource { node: 1 })
        );
    }

    #[test]
    fn checked_design_field_transfer_inherits_refinement_parent_values() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(2);
        let source_values = [2.5, 7.5];
        let transferred = transfer_design_field_checked(&source, &target, &source_values)
            .expect("element design field should transfer across refinement");
        assert_eq!(transferred.len(), target.num_elements());
        for (target_element, (&value, element)) in
            transferred.iter().zip(&target.elements).enumerate()
        {
            let centroid = element_centroid(&target, element).unwrap();
            let source_element = locate_element_containing_point(&source, centroid).unwrap();
            assert_eq!(
                value, source_values[source_element],
                "target element {target_element}"
            );
        }
    }

    #[test]
    fn checked_design_field_transfer_rejects_bad_buffers_and_outside_elements() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(2);
        assert!(matches!(
            transfer_design_field_checked(&source, &target, &[1.0]),
            Err(InterpolationError::SourceElementValueCount { .. })
        ));
        let non_finite = vec![f64::NAN; source.num_elements()];
        assert!(matches!(
            transfer_design_field_checked(&source, &target, &non_finite),
            Err(InterpolationError::NonFiniteSourceElement { .. })
        ));

        let mut outside = target.clone();
        for point in &mut outside.nodes {
            point.x += 2.0;
            point.y += 2.0;
        }
        assert!(matches!(
            transfer_design_field_checked(&source, &outside, &[1.0, 2.0]),
            Err(InterpolationError::TargetElementOutside { .. })
        ));
    }

    #[test]
    fn checked_complex_interpolation_preserves_linear_field() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(2);
        let source_values: Vec<Complex64> = source
            .nodes
            .iter()
            .map(|node| Complex64::new(node.x, node.y))
            .collect();
        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");

        let target_values =
            interpolate_complex_field_checked(&data, &source, &target, &source_values)
                .expect("complex field transfer should succeed");
        for (node, value) in target.nodes.iter().zip(target_values) {
            assert!((value.re - node.x).abs() < 1e-12);
            assert!((value.im - node.y).abs() < 1e-12);
        }
    }

    #[test]
    fn conservative_interpolation_preserves_lumped_scalar_integral() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(3);
        let source_values: Vec<f64> = source
            .nodes
            .iter()
            .map(|node| 1.0 + 2.0 * node.x - 0.5 * node.y)
            .collect();
        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");
        let transferred =
            interpolate_field_conservative_checked(&data, &source, &target, &source_values)
                .expect("same-domain transfer should conserve the lumped integral");
        let source_weights = lumped_nodal_weights(&source, "source").unwrap();
        let target_weights = lumped_nodal_weights(&target, "target").unwrap();
        let source_integral = source_values
            .iter()
            .zip(source_weights)
            .map(|(value, weight)| value * weight)
            .sum::<f64>();
        let target_integral = transferred
            .iter()
            .zip(target_weights)
            .map(|(value, weight)| value * weight)
            .sum::<f64>();
        assert!((source_integral - target_integral).abs() < 1e-12);
    }

    #[test]
    fn conservative_complex_interpolation_preserves_lumped_integral() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(2);
        let source_values: Vec<Complex64> = source
            .nodes
            .iter()
            .map(|node| Complex64::new(1.0 + node.x, -0.25 * node.y))
            .collect();
        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");
        let transferred =
            interpolate_complex_field_conservative_checked(&data, &source, &target, &source_values)
                .expect("same-domain complex transfer should conserve the lumped integral");
        let source_weights = lumped_nodal_weights(&source, "source").unwrap();
        let target_weights = lumped_nodal_weights(&target, "target").unwrap();
        let source_integral = source_values
            .iter()
            .zip(source_weights)
            .map(|(value, weight)| *value * weight)
            .sum::<Complex64>();
        let target_integral = transferred
            .iter()
            .zip(target_weights)
            .map(|(value, weight)| *value * weight)
            .sum::<Complex64>();
        assert!((source_integral - target_integral).norm() < 1e-12);
    }

    #[test]
    fn adjoint_transfer_preserves_weighted_scalar_pairing() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(2);
        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");
        let source_values: Vec<f64> = source
            .nodes
            .iter()
            .map(|node| 0.75 + 0.5 * node.x - 1.25 * node.y)
            .collect();
        let target_dual: Vec<f64> = target
            .nodes
            .iter()
            .map(|node| -0.25 + 1.5 * node.x + 0.25 * node.y)
            .collect();
        let source_weights = vec![0.7, 1.1, 0.9, 1.3];
        let target_weights = vec![0.8, 1.0, 1.2, 0.9, 1.1, 0.75, 1.25, 0.95, 1.05];
        let target_values = interpolate_field_checked(&data, &source, &target, &source_values)
            .expect("primal transfer should succeed");
        let adjoint = interpolate_field_adjoint_checked(
            &data,
            &source,
            &target,
            &target_dual,
            &source_weights,
            &target_weights,
        )
        .expect("adjoint transfer should succeed");
        let target_pairing: f64 = target_values
            .iter()
            .zip(&target_dual)
            .zip(&target_weights)
            .map(|((value, dual), weight)| value * dual * weight)
            .sum();
        let source_pairing: f64 = source_values
            .iter()
            .zip(&adjoint)
            .zip(&source_weights)
            .map(|((value, dual), weight)| value * dual * weight)
            .sum();
        assert!((target_pairing - source_pairing).abs() < 1e-12);
    }

    #[test]
    fn complex_adjoint_transfer_preserves_hermitian_pairing() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(2);
        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");
        let source_values: Vec<Complex64> = source
            .nodes
            .iter()
            .map(|node| Complex64::new(0.75 + node.x, -0.5 + node.y))
            .collect();
        let target_dual: Vec<Complex64> = target
            .nodes
            .iter()
            .map(|node| Complex64::new(-0.25 + node.x, 0.75 - node.y))
            .collect();
        let source_weights = vec![0.7, 1.1, 0.9, 1.3];
        let target_weights = vec![0.8, 1.0, 1.2, 0.9, 1.1, 0.75, 1.25, 0.95, 1.05];
        let target_values =
            interpolate_complex_field_checked(&data, &source, &target, &source_values)
                .expect("complex primal transfer should succeed");
        let adjoint = interpolate_complex_field_adjoint_checked(
            &data,
            &source,
            &target,
            &target_dual,
            &source_weights,
            &target_weights,
        )
        .expect("complex adjoint transfer should succeed");
        let target_pairing: Complex64 = target_values
            .iter()
            .zip(&target_dual)
            .zip(&target_weights)
            .map(|((value, dual), weight)| value.conj() * dual * *weight)
            .sum();
        let source_pairing: Complex64 = source_values
            .iter()
            .zip(&adjoint)
            .zip(&source_weights)
            .map(|((value, dual), weight)| value.conj() * dual * *weight)
            .sum();
        assert!((target_pairing - source_pairing).norm() < 1e-12);
    }

    #[test]
    fn adjoint_transfer_rejects_nonpositive_inner_product_weights() {
        let source = unit_square_triangles(1);
        let target = unit_square_triangles(1);
        let data = create_interpolation_data_checked(&target, &source).expect("valid mesh pair");
        let source_weights = vec![1.0, 1.0, 1.0, 0.0];
        let target_weights = vec![1.0; target.num_nodes()];
        let values = vec![0.0; target.num_nodes()];
        assert_eq!(
            interpolate_field_adjoint_checked(
                &data,
                &source,
                &target,
                &values,
                &source_weights,
                &target_weights,
            ),
            Err(InterpolationError::InvalidAdjointWeight {
                role: "source",
                node: 3,
            })
        );
    }

    #[test]
    fn conservative_interpolation_rejects_mismatched_domain_measure() {
        let source = unit_square_triangles(1);
        let mut target = unit_square_triangles(1);
        for node in &mut target.nodes {
            node.x *= 0.5;
            node.y *= 0.5;
        }
        let data = create_interpolation_data_checked(&target, &source).expect("target is inside");
        assert!(matches!(
            interpolate_field_conservative_checked(
                &data,
                &source,
                &target,
                &vec![1.0; source.num_nodes()]
            ),
            Err(InterpolationError::ConservationMismatch { .. })
        ));
    }

    #[test]
    fn checked_interpolation_rejects_malformed_public_metadata() {
        let mesh = unit_square_triangles(1);
        let data = InterpolationData {
            target_to_source: vec![Some((0, vec![0.5, 0.5])); mesh.num_nodes()],
        };

        let error = data
            .validate(&mesh, &mesh)
            .expect_err("triangle metadata needs three barycentric weights");
        assert_eq!(
            error,
            InterpolationError::InvalidData {
                node: 0,
                reason: "barycentric weights are malformed".into(),
            }
        );
    }

    #[test]
    fn checked_interpolation_rejects_extrapolating_or_mislocated_metadata() {
        let mesh = unit_square_triangles(1);
        let extrapolating = InterpolationData {
            target_to_source: vec![Some((0, vec![-0.1, 1.1, 0.0])); mesh.num_nodes()],
        };
        assert!(matches!(
            extrapolating.validate(&mesh, &mesh),
            Err(InterpolationError::InvalidData { node: 0, .. })
        ));

        let source_vertex = mesh.elements[0].vertices()[0];
        let target_node = mesh
            .nodes
            .iter()
            .position(|point| {
                let source = mesh.nodes[source_vertex];
                (point.x - source.x).abs() > 1e-12
                    || (point.y - source.y).abs() > 1e-12
                    || (point.z - source.z).abs() > 1e-12
            })
            .expect("unit square has a node distinct from the first vertex");
        let mislocated = InterpolationData {
            target_to_source: vec![Some((0, vec![1.0, 0.0, 0.0])); mesh.num_nodes()],
        };
        assert!(matches!(
            mislocated.validate(&mesh, &mesh),
            Err(InterpolationError::InvalidData { node, .. }) if node == target_node
        ));
    }

    #[test]
    fn checked_interpolation_rejects_short_element_connectivity() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].nodes.truncate(2);
        let target = unit_square_triangles(1);
        let data = InterpolationData {
            target_to_source: Vec::new(),
        };

        assert!(matches!(
            data.validate(&mesh, &mesh),
            Err(InterpolationError::InvalidMesh { mesh: "source", .. })
        ));
        assert!(matches!(
            create_interpolation_data_checked(&target, &mesh),
            Err(InterpolationError::InvalidMesh { mesh: "source", .. })
        ));
    }

    #[test]
    fn checked_interpolation_rejects_invalid_high_order_node_indices() {
        let mut source = unit_square_triangles(1);
        source.elements[0].nodes = vec![0, 1, 2, 0, 1, usize::MAX];
        let target = unit_square_triangles(1);

        assert!(matches!(
            create_interpolation_data_checked(&target, &source),
            Err(InterpolationError::InvalidMesh { mesh: "source", .. })
        ));
    }

    #[test]
    fn test_interpolate_scalar_at_point_is_exact_for_linear_triangular_field() {
        let mesh = unit_square_triangles(4);
        let values: Vec<f64> = mesh
            .nodes
            .iter()
            .map(|node| 2.0 * node.x - 3.0 * node.y + 0.75)
            .collect();

        let point = [0.37, 0.61, 0.0];
        let interpolated = interpolate_scalar_at_point(&mesh, &values, point)
            .expect("interior point must be located in a triangle");
        let expected = 2.0 * point[0] - 3.0 * point[1] + 0.75;

        assert!(
            (interpolated - expected).abs() < 1e-12,
            "P1 interpolation at {point:?}: got {interpolated:.16e}, expected {expected:.16e}"
        );
    }

    #[test]
    fn test_interpolate_scalar_at_point_rejects_points_outside_mesh() {
        let mesh = unit_square_triangles(2);
        let values = vec![1.0; mesh.num_nodes()];

        assert!(interpolate_scalar_at_point(&mesh, &values, [1.5, 0.5, 0.0]).is_none());
    }

    #[test]
    fn checked_interpolation_transfers_p2_triangle_state_after_refinement() {
        let mut source = Mesh::new(2);
        let v0 = source.add_node((0.0, 0.0).into());
        let v1 = source.add_node((1.0, 0.0).into());
        let v2 = source.add_node((0.0, 1.0).into());
        let m01 = source.add_node((0.5, 0.0).into());
        let m12 = source.add_node((0.5, 0.5).into());
        let m20 = source.add_node((0.0, 0.5).into());
        source.add_element(ElementType::Triangle, vec![v0, v1, v2, m01, m12, m20]);

        let mut target = source.clone();
        refine_elements_high_order_checked(&mut target, &[0]).expect("P2 refinement must work");

        let field = |point: &crate::mesh::Point| {
            0.4 + 1.2 * point.x - 0.7 * point.y + 2.0 * point.x * point.x + 0.5 * point.x * point.y
                - 1.1 * point.y * point.y
        };
        let source_values: Vec<_> = source.nodes.iter().map(field).collect();
        let data = create_interpolation_data_checked(&target, &source).expect("compatible meshes");
        let values = interpolate_field_checked(&data, &source, &target, &source_values)
            .expect("P2 transfer must succeed");

        for (node, value) in target.nodes.iter().zip(values) {
            assert!((value - field(node)).abs() < 2e-11, "at {node:?}: {value}");
        }
        let conservative =
            interpolate_field_conservative_checked(&data, &source, &target, &source_values)
                .expect("P2 conservative transfer must succeed");
        for (node, value) in target.nodes.iter().zip(conservative) {
            assert!((value - field(node)).abs() < 2e-10, "at {node:?}: {value}");
        }
    }

    #[test]
    fn checked_interpolation_locates_curved_p2_triangle_geometry() {
        let mut mesh = Mesh::new(2);
        let v0 = mesh.add_node((0.0, 0.0).into());
        let v1 = mesh.add_node((1.0, 0.0).into());
        let v2 = mesh.add_node((0.0, 1.0).into());
        let m01 = mesh.add_node((0.5, -0.05).into());
        let m12 = mesh.add_node((0.53, 0.48).into());
        let m20 = mesh.add_node((-0.04, 0.5).into());
        mesh.add_element(ElementType::Triangle, vec![v0, v1, v2, m01, m12, m20]);
        let source_values = vec![0.2, -0.4, 1.1, 3.0, -2.0, 0.7];

        let data = create_interpolation_data_checked(&mesh, &mesh)
            .expect("curved P2 geometry must be locatable");
        let values = interpolate_field_checked(&data, &mesh, &mesh, &source_values)
            .expect("curved P2 state transfer must succeed");
        for (actual, expected) in values.iter().zip(&source_values) {
            assert!(
                (actual - expected).abs() < 2e-9,
                "got {actual}, expected {expected}"
            );
        }
        let legacy = interpolate_field(&data, &mesh, &source_values, f64::NAN);
        assert!(
            legacy
                .iter()
                .zip(&source_values)
                .all(|(actual, expected)| (actual - expected).abs() < 2e-9)
        );
    }

    #[test]
    fn checked_interpolation_transfers_p2_tetrahedron_state_after_refinement() {
        let mut source = Mesh::new(3);
        let v0 = source.add_node((0.0, 0.0, 0.0).into());
        let v1 = source.add_node((1.0, 0.0, 0.0).into());
        let v2 = source.add_node((0.0, 1.0, 0.0).into());
        let v3 = source.add_node((0.0, 0.0, 1.0).into());
        let m01 = source.add_node((0.5, 0.0, 0.0).into());
        let m02 = source.add_node((0.0, 0.5, 0.0).into());
        let m03 = source.add_node((0.0, 0.0, 0.5).into());
        let m12 = source.add_node((0.5, 0.5, 0.0).into());
        let m13 = source.add_node((0.5, 0.0, 0.5).into());
        let m23 = source.add_node((0.0, 0.5, 0.5).into());
        source.add_element(
            ElementType::Tetrahedron,
            vec![v0, v1, v2, v3, m01, m02, m03, m12, m13, m23],
        );

        let mut target = source.clone();
        refine_elements_high_order_checked(&mut target, &[0]).expect("P2 refinement must work");

        let field = |point: &crate::mesh::Point| {
            0.2 + 0.3 * point.x - 0.8 * point.y + 1.1 * point.z + 1.5 * point.x * point.x
                - 0.4 * point.x * point.y
                + 0.6 * point.y * point.z
                - 0.9 * point.z * point.z
        };
        let source_values: Vec<_> = source.nodes.iter().map(field).collect();
        let data = create_interpolation_data_checked(&target, &source).expect("compatible meshes");
        let values = interpolate_field_checked(&data, &source, &target, &source_values)
            .expect("P2 transfer must succeed");

        for (node, value) in target.nodes.iter().zip(values) {
            assert!((value - field(node)).abs() < 5e-11, "at {node:?}: {value}");
        }
        let conservative =
            interpolate_field_conservative_checked(&data, &source, &target, &source_values)
                .expect("P2 conservative transfer must succeed");
        for (node, value) in target.nodes.iter().zip(conservative) {
            assert!((value - field(node)).abs() < 5e-10, "at {node:?}: {value}");
        }
    }

    #[test]
    fn checked_interpolation_preserves_q2_quadrilateral_and_hex_state() {
        let cases = [unit_square_quads(1).to_p2(), unit_cube_hexahedra(1).to_p2()];
        for source in cases {
            let target = source.clone();
            let field = |point: &crate::mesh::Point| {
                0.1 + 0.2 * point.x - 0.3 * point.y + 0.4 * point.z + 0.7 * point.x * point.x
                    - 0.5 * point.x * point.y
                    + 0.6 * point.y * point.z
                    - 0.8 * point.z * point.z
            };
            let source_values: Vec<_> = source.nodes.iter().map(field).collect();
            let data = create_interpolation_data_checked(&target, &source)
                .expect("Q2 meshes must validate");
            let values = interpolate_field_checked(&data, &source, &target, &source_values)
                .expect("Q2 transfer must succeed");
            for (node, value) in target.nodes.iter().zip(values) {
                assert!((value - field(node)).abs() < 2e-11, "at {node:?}: {value}");
            }
        }
    }

    #[test]
    fn test_point_in_triangle() {
        let mesh = unit_square_triangles(1);
        // First triangle has vertices at (0,0), (1,0), (0,1) (approx)
        let inside = super::point_in_triangle(&mesh, 0, [0.2, 0.2, 0.0]);
        assert!(inside.is_some(), "Point should be inside triangle");
        let bary = inside.unwrap();
        assert!(
            (bary.iter().sum::<f64>() - 1.0).abs() < 1e-10,
            "Bary coords sum to 1"
        );
    }
}
