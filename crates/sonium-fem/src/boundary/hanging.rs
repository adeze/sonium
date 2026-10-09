//! Checked scalar hanging-node constraints for non-conforming 2-D P1 meshes.
//!
//! A P1 hanging node on a coarse edge is constrained by linear interpolation
//! of the two coarse-edge endpoint values.  This module deliberately keeps
//! geometric detection separate from mesh refinement: callers can build the
//! constraints after importing or adapting a mesh and then use the existing
//! [`MpcSystem`](super::periodic::MpcSystem) condensation/recovery path.

use super::periodic::{MpcError, MpcSystem, MpcTerm, MultiPointConstraint};
use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, Mesh, types::MeshConnectivityError};
use crate::spaces::{FunctionSpace, H1Space, SpaceValidationError, VectorH1Space};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

/// A scalar P1 hanging-node interpolation constraint.
///
/// The relation is
/// `u[slave] = (1 - parameter) * u[edge[0]] + parameter * u[edge[1]]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HangingNodeConstraint {
    /// Global scalar DOF/node constrained by the edge interpolation.
    pub slave: usize,
    /// Ordered coarse-edge endpoint node IDs.
    pub edge: [usize; 2],
    /// Affine coordinate of the slave along `edge`, in `(0, 1)`.
    pub parameter: f64,
}

impl HangingNodeConstraint {
    /// Convert this geometric relation to the complex-valued MPC contract.
    pub fn to_mpc(self) -> MultiPointConstraint {
        let one_minus_parameter = 1.0 - self.parameter;
        MultiPointConstraint {
            slave: self.slave,
            masters: vec![
                MpcTerm {
                    dof: self.edge[0],
                    coefficient: Complex64::new(one_minus_parameter, 0.0),
                },
                MpcTerm {
                    dof: self.edge[1],
                    coefficient: Complex64::new(self.parameter, 0.0),
                },
            ],
        }
    }
}

/// Errors returned while admitting a mesh to hanging-node detection.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HangingNodeError {
    /// The geometric tolerance must be finite and strictly positive.
    #[error("hanging-node tolerance {0} must be finite and positive")]
    InvalidTolerance(f64),
    /// This packet intentionally covers scalar 2-D triangle meshes only.
    #[error("hanging-node detection requires a 2-D mesh, got dimension {0}")]
    UnsupportedDimension(usize),
    /// The mesh failed the shared connectivity admission gate.
    #[error("invalid mesh connectivity: {errors:?}")]
    InvalidMesh { errors: Vec<MeshConnectivityError> },
    /// Non-triangular or higher-order elements need a different trace rule.
    #[error(
        "element {element_index} ({element_type:?}) is not a P1 triangle; node count is {actual}"
    )]
    UnsupportedElement {
        element_index: usize,
        element_type: ElementType,
        actual: usize,
    },
    /// A triangle with zero area cannot define a reliable edge trace.
    #[error("element {element_index} has degenerate geometry")]
    DegenerateElement { element_index: usize },
    /// A coarse edge with zero length cannot define an interpolation parameter.
    #[error("edge {edge:?} has degenerate geometry")]
    DegenerateEdge { edge: [usize; 2] },
    /// A coarse triangular face zero area cannot define interpolation coordinates.
    #[error("face {face:?} degenerate geometry")]
    DegenerateFace { face: [usize; 3] },
    /// Equal-length candidate coarse edges would produce different relations.
    #[error("node {slave} lies on geometrically ambiguous coarse edges")]
    AmbiguousConstraint { slave: usize },
    /// The detected constraints failed the common MPC admission gate.
    #[error("hanging-node MPC construction failed: {0}")]
    Mpc(#[source] MpcError),
    /// The H1 space failed its checked topology/DOF admission contract.
    #[error("invalid H1 space for hanging-node constraints: {0}")]
    InvalidH1Space(#[source] SpaceValidationError),
    /// This bounded high-order consumer accepts affine vertex-only geometry.
    #[error(
        "element {element_index} has {actual} geometry nodes; affine high-order hanging-node constraints require {expected} vertex nodes"
    )]
    UnsupportedGeometry {
        element_index: usize,
        actual: usize,
        expected: usize,
    },
    /// This bounded consumer currently covers 2-D triangle/quad and 3-D tetrahedral H1 traces.
    #[error("element {element_index} ({element_type:?}) is unsupported for H1 hanging constraints")]
    UnsupportedH1Element {
        element_index: usize,
        element_type: ElementType,
    },
    /// The H1 space did not expose coordinates for every global DOF.
    #[error("H1 space did not provide coordinates for all global DOFs")]
    MissingDofCoordinates,
    /// A global H1 DOF coordinate cannot participate in geometric admission.
    #[error("global H1 DOF {dof} has non-finite coordinates")]
    NonFiniteDofCoordinate { dof: usize },
    /// A supported element did not expose the expected edge trace DOFs.
    #[error(
        "element {element_index} edge {edge:?} exposes {actual} trace DOFs; expected {expected}"
    )]
    InvalidTraceDofs {
        element_index: usize,
        edge: [usize; 2],
        actual: usize,
        expected: usize,
    },
    #[error(
        "element {element_index} face {face:?} exposes {actual} trace DOFs; expected {expected}"
    )]
    InvalidFaceTraceDofs {
        element_index: usize,
        face: [usize; 3],
        actual: usize,
        expected: usize,
    },
    #[error("cannot construct finite Lagrange interpolation on H1 hanging face trace")]
    InvalidFaceInterpolation,
    /// A quadrilateral coarse face has zero area or a singular affine frame.
    #[error("quadrilateral face {face:?} has degenerate geometry")]
    DegenerateQuadrilateralFace { face: [usize; 4] },
    /// A supported hexahedral face did not expose the expected tensor-product trace.
    #[error(
        "element {element_index} face {face:?} exposes {actual} quadrilateral trace DOFs; expected {expected}"
    )]
    InvalidQuadrilateralFaceTraceDofs {
        element_index: usize,
        face: [usize; 4],
        actual: usize,
        expected: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    edge: [usize; 2],
    parameter: f64,
    length: f64,
}

/// One scalar H1 high-order hanging-DOF relation on an affine coarse edge.
///
/// `masters` contains the Lagrange edge trace DOFs in geometric order. The
/// relation is `u[slave] = sum(masters[i].coefficient * u[masters[i].dof])`.
#[derive(Debug, Clone, PartialEq)]
pub struct H1HangingNodeConstraint {
    /// Global H1 DOF constrained by the coarse-edge trace.
    pub slave: usize,
    /// Ordered coarse-edge endpoint node IDs.
    pub edge: [usize; 2],
    /// Affine coordinate of the slave along `edge`, in `(0, 1)`.
    pub parameter: f64,
    /// Coarse-edge Lagrange trace terms.
    pub masters: Vec<MpcTerm>,
}

impl H1HangingNodeConstraint {
    /// Convert the checked relation to the generic MPC contract.
    pub fn to_mpc(&self) -> MultiPointConstraint {
        MultiPointConstraint {
            slave: self.slave,
            masters: self.masters.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct H1HangingFaceConstraint {
    pub slave: usize,
    pub face: [usize; 3],
    pub reference: [f64; 2],
    pub masters: Vec<MpcTerm>,
}

impl H1HangingFaceConstraint {
    pub fn to_mpc(&self) -> MultiPointConstraint {
        MultiPointConstraint {
            slave: self.slave,
            masters: self.masters.clone(),
        }
    }
}

/// One scalar H1 hanging-DOF relation on an affine quadrilateral face.
///
/// `reference` is the tensor-product face coordinate in `[0, 1]^2`, and
/// `masters` contains the coarse Q1/Q2/Q3 face trace in geometric order.
#[derive(Debug, Clone, PartialEq)]
pub struct H1HangingQuadFaceConstraint {
    pub slave: usize,
    pub face: [usize; 4],
    pub reference: [f64; 2],
    pub masters: Vec<MpcTerm>,
}

impl H1HangingQuadFaceConstraint {
    pub fn to_mpc(&self) -> MultiPointConstraint {
        MultiPointConstraint {
            slave: self.slave,
            masters: self.masters.clone(),
        }
    }
}

/// Detect deterministic P1 hanging-node interpolation constraints in 2-D.
///
/// The mesh must contain only valid P1 triangles.  A node is admitted as a
/// hanging node when it is a used element vertex strictly inside a longer
/// element edge, within `tolerance` in physical distance.  When nested
/// refinement creates more than one candidate edge for a slave, the longest
/// containing edge is selected; equal-length alternatives fail closed.
pub fn hanging_node_constraints_2d(
    mesh: &Mesh,
    tolerance: f64,
) -> Result<Vec<HangingNodeConstraint>, HangingNodeError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(HangingNodeError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 2 {
        return Err(HangingNodeError::UnsupportedDimension(mesh.dimension));
    }
    mesh.validate_connectivity()
        .map_err(|errors| HangingNodeError::InvalidMesh { errors })?;

    let mut used_vertices = BTreeSet::new();
    let mut edges = BTreeSet::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != ElementType::Triangle || element.nodes.len() != 3 {
            return Err(HangingNodeError::UnsupportedElement {
                element_index,
                element_type: element.element_type,
                actual: element.nodes.len(),
            });
        }

        let vertices = element.vertices();
        let [a, b, c] = [vertices[0], vertices[1], vertices[2]];
        used_vertices.extend([a, b, c]);
        let pa = mesh.nodes[a];
        let pb = mesh.nodes[b];
        let pc = mesh.nodes[c];
        let ab = [pb.x - pa.x, pb.y - pa.y];
        let ac = [pc.x - pa.x, pc.y - pa.y];
        let twice_area = ab[0] * ac[1] - ab[1] * ac[0];
        let max_edge = [pa.distance(&pb), pb.distance(&pc), pc.distance(&pa)]
            .into_iter()
            .fold(0.0_f64, f64::max);
        if !twice_area.is_finite()
            || max_edge <= tolerance
            || twice_area.abs() <= tolerance * max_edge
        {
            return Err(HangingNodeError::DegenerateElement { element_index });
        }

        for (first, second) in element.edges() {
            edges.insert([first.min(second), first.max(second)]);
        }
    }

    let mut candidates: BTreeMap<usize, Vec<Candidate>> = BTreeMap::new();
    for edge in edges {
        let first = mesh.nodes[edge[0]];
        let second = mesh.nodes[edge[1]];
        let dx = second.x - first.x;
        let dy = second.y - first.y;
        let length_squared = dx * dx + dy * dy;
        let length = length_squared.sqrt();
        if !length.is_finite() || length <= tolerance {
            return Err(HangingNodeError::DegenerateEdge { edge });
        }
        // With a tolerance at least half the edge length there is no
        // unambiguous strict interior point to admit on this edge.
        if tolerance * 2.0 >= length {
            continue;
        }
        let parameter_margin = tolerance / length;
        for &node in &used_vertices {
            if node == edge[0] || node == edge[1] {
                continue;
            }
            let point = mesh.nodes[node];
            let px = point.x - first.x;
            let py = point.y - first.y;
            let parameter = (dx * px + dy * py) / length_squared;
            let distance = (dx * py - dy * px).abs() / length;
            if distance <= tolerance
                && parameter > parameter_margin
                && parameter < 1.0 - parameter_margin
            {
                candidates.entry(node).or_default().push(Candidate {
                    edge,
                    parameter,
                    length,
                });
            }
        }
    }

    let mut constraints = Vec::with_capacity(candidates.len());
    for (slave, mut node_candidates) in candidates {
        node_candidates.sort_by(|left, right| {
            right
                .length
                .total_cmp(&left.length)
                .then_with(|| left.edge.cmp(&right.edge))
        });
        let best = node_candidates[0];
        if node_candidates
            .iter()
            .skip(1)
            .any(|candidate| (best.length - candidate.length).abs() <= tolerance)
        {
            return Err(HangingNodeError::AmbiguousConstraint { slave });
        }
        if !best.parameter.is_finite() || !(0.0..1.0).contains(&best.parameter) {
            return Err(HangingNodeError::AmbiguousConstraint { slave });
        }
        constraints.push(HangingNodeConstraint {
            slave,
            edge: best.edge,
            parameter: best.parameter,
        });
    }
    Ok(constraints)
}

/// Build a full scalar MPC system for all P1 nodes in a 2-D mesh.
pub fn hanging_node_mpc_system_2d(
    mesh: &Mesh,
    tolerance: f64,
) -> Result<MpcSystem, HangingNodeError> {
    let constraints = hanging_node_constraints_2d(mesh, tolerance)?;
    let mpcs: Vec<_> = constraints
        .into_iter()
        .map(HangingNodeConstraint::to_mpc)
        .collect();
    MpcSystem::new(mesh.nodes.len(), &mpcs).map_err(HangingNodeError::Mpc)
}

#[derive(Debug)]
struct H1EdgeRecord {
    edge: [usize; 2],
    length: f64,
    trace: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Copy)]
struct H1Candidate {
    edge_index: usize,
    parameter: f64,
    length: f64,
}

/// Build checked scalar H1 P1/P2/P3 hanging-node constraints on affine 2-D
/// triangle and quadrilateral meshes.
///
/// The mesh stores only vertex geometry in this bounded consumer. High-order
/// edge DOFs are owned by the supplied [`H1Space`] and are located from its
/// global DOF-coordinate contract. Every fine-edge vertex or edge DOF that is
/// strictly inside a longer coarse edge receives the exact Lagrange
/// interpolation relation from that coarse edge. Explicit curved geometry,
/// 3-D elements, malformed spaces, and ambiguous geometric ownership fail
/// closed rather than producing an unsafe constraint system.
pub fn h1_hanging_node_constraints_2d(
    mesh: &Mesh,
    space: &H1Space,
    tolerance: f64,
) -> Result<Vec<H1HangingNodeConstraint>, HangingNodeError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(HangingNodeError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 2 {
        return Err(HangingNodeError::UnsupportedDimension(mesh.dimension));
    }
    mesh.validate_connectivity()
        .map_err(|errors| HangingNodeError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(HangingNodeError::InvalidH1Space)?;

    for (element_index, element) in mesh.elements.iter().enumerate() {
        if !matches!(
            element.element_type,
            ElementType::Triangle | ElementType::Quadrilateral
        ) {
            return Err(HangingNodeError::UnsupportedH1Element {
                element_index,
                element_type: element.element_type,
            });
        }
        let expected = element.element_type.num_vertices();
        if element.nodes.len() != expected {
            return Err(HangingNodeError::UnsupportedGeometry {
                element_index,
                actual: element.nodes.len(),
                expected,
            });
        }
    }

    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or(HangingNodeError::MissingDofCoordinates)?;
    if coordinates.len() != space.num_global_dofs(mesh) {
        return Err(HangingNodeError::MissingDofCoordinates);
    }
    for (dof, point) in coordinates.iter().enumerate() {
        if !point.iter().all(|value| value.is_finite()) {
            return Err(HangingNodeError::NonFiniteDofCoordinate { dof });
        }
    }

    let expected_trace = match space.degree {
        PolynomialDegree::P1 => 2,
        PolynomialDegree::P2 => 3,
        PolynomialDegree::P3 => 4,
    };
    let mut owners = BTreeMap::<[usize; 2], usize>::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        for &(local_first, local_second) in element.element_type.local_edges() {
            let first = vertices[local_first];
            let second = vertices[local_second];
            let edge = [first.min(second), first.max(second)];
            owners.entry(edge).or_insert(element_index);
        }
    }

    let mut edges = Vec::with_capacity(owners.len());
    for (edge, &owner_element) in &owners {
        let first = mesh.nodes[edge[0]];
        let second = mesh.nodes[edge[1]];
        let dx = second.x - first.x;
        let dy = second.y - first.y;
        let length_squared = dx * dx + dy * dy;
        let length = length_squared.sqrt();
        if !length.is_finite() || length <= tolerance {
            return Err(HangingNodeError::DegenerateEdge { edge: *edge });
        }

        let local_dofs = space.local_to_global(mesh, owner_element);
        let mut trace = local_dofs
            .into_iter()
            .filter_map(|dof| {
                let (parameter, distance) = affine_segment_parameter(
                    coordinates[dof],
                    [first.x, first.y],
                    [second.x, second.y],
                    length_squared,
                    length,
                )?;
                if distance <= tolerance
                    && parameter >= -tolerance / length
                    && parameter <= 1.0 + tolerance / length
                {
                    Some((dof, parameter.clamp(0.0, 1.0)))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        trace.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        trace.dedup_by(|left, right| left.0 == right.0);
        if trace.len() != expected_trace {
            return Err(HangingNodeError::InvalidTraceDofs {
                element_index: owner_element,
                edge: *edge,
                actual: trace.len(),
                expected: expected_trace,
            });
        }
        edges.push(H1EdgeRecord {
            edge: *edge,
            length,
            trace,
        });
    }

    let mut candidates = BTreeMap::<usize, Vec<H1Candidate>>::new();
    for (edge_index, edge) in edges.iter().enumerate() {
        let first = mesh.nodes[edge.edge[0]];
        let second = mesh.nodes[edge.edge[1]];
        let dx = second.x - first.x;
        let dy = second.y - first.y;
        let length_squared = dx * dx + dy * dy;
        let parameter_margin = tolerance / edge.length;
        for (dof, &point) in coordinates.iter().enumerate() {
            if edge.trace.iter().any(|&(trace_dof, _)| trace_dof == dof) {
                continue;
            }
            // A DOF already owned by a longer edge is a coarse master, even
            // when its coordinate lies on a shorter fine edge.
            if edges.iter().any(|owner| {
                owner.length > edge.length + tolerance
                    && owner.trace.iter().any(|&(trace_dof, _)| trace_dof == dof)
            }) {
                continue;
            }
            let Some((parameter, distance)) = affine_segment_parameter(
                point,
                [first.x, first.y],
                [second.x, second.y],
                length_squared,
                edge.length,
            ) else {
                continue;
            };
            if distance <= tolerance
                && parameter > parameter_margin
                && parameter < 1.0 - parameter_margin
            {
                candidates.entry(dof).or_default().push(H1Candidate {
                    edge_index,
                    parameter,
                    length: edge.length,
                });
            }
        }
    }

    let mut constraints = Vec::with_capacity(candidates.len());
    for (slave, mut node_candidates) in candidates {
        node_candidates.sort_by(|left, right| {
            right
                .length
                .total_cmp(&left.length)
                .then_with(|| left.edge_index.cmp(&right.edge_index))
        });
        let best = node_candidates[0];
        if node_candidates.iter().skip(1).any(|candidate| {
            (best.length - candidate.length).abs() <= tolerance
                && candidate.edge_index != best.edge_index
        }) {
            return Err(HangingNodeError::AmbiguousConstraint { slave });
        }
        let edge = &edges[best.edge_index];
        let masters = lagrange_edge_terms(&edge.trace, best.parameter)?;
        constraints.push(H1HangingNodeConstraint {
            slave,
            edge: edge.edge,
            parameter: best.parameter,
            masters,
        });
    }
    Ok(constraints)
}

/// Build the generic scalar MPC system for checked H1 P1/P2/P3 constraints.
pub fn h1_hanging_node_mpc_system_2d(
    mesh: &Mesh,
    space: &H1Space,
    tolerance: f64,
) -> Result<MpcSystem, HangingNodeError> {
    let constraints = h1_hanging_node_constraints_2d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .map(H1HangingNodeConstraint::to_mpc)
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(HangingNodeError::Mpc)
}

#[derive(Debug)]
struct H1FaceRecord {
    face: [usize; 3],
    area: f64,
    trace: Vec<(usize, [f64; 2])>,
}

#[derive(Debug, Clone, Copy)]
struct H1FaceCandidate {
    face_index: usize,
    reference: [f64; 2],
    area: f64,
}

/// Build checked scalar H1 P1/P2/P3 hanging-node constraints on affine 3-D
/// tetrahedral meshes with triangular non-conforming faces.
///
/// The face trace uses the actual global H1 DOF coordinates and a small
/// Vandermonde solve in barycentric coordinates. This covers fine vertices,
/// edge DOFs, and P3 face-interior DOFs on a coarse triangular face. Explicit
/// curved geometry, hexahedral faces, and non-tetrahedral elements remain
/// intentionally fail-closed until their own trace ownership contract exists.
pub fn h1_hanging_face_constraints_3d(
    mesh: &Mesh,
    space: &H1Space,
    tolerance: f64,
) -> Result<Vec<H1HangingFaceConstraint>, HangingNodeError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(HangingNodeError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 3 {
        return Err(HangingNodeError::UnsupportedDimension(mesh.dimension));
    }
    mesh.validate_connectivity()
        .map_err(|errors| HangingNodeError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(HangingNodeError::InvalidH1Space)?;

    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != ElementType::Tetrahedron {
            return Err(HangingNodeError::UnsupportedH1Element {
                element_index,
                element_type: element.element_type,
            });
        }
        if element.nodes.len() != element.element_type.num_vertices() {
            return Err(HangingNodeError::UnsupportedGeometry {
                element_index,
                actual: element.nodes.len(),
                expected: element.element_type.num_vertices(),
            });
        }
    }

    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or(HangingNodeError::MissingDofCoordinates)?;
    if coordinates.len() != space.num_global_dofs(mesh) {
        return Err(HangingNodeError::MissingDofCoordinates);
    }
    for (dof, point) in coordinates.iter().enumerate() {
        if !point.iter().all(|value| value.is_finite()) {
            return Err(HangingNodeError::NonFiniteDofCoordinate { dof });
        }
    }

    let expected_trace = match space.degree {
        PolynomialDegree::P1 => 3,
        PolynomialDegree::P2 => 6,
        PolynomialDegree::P3 => 10,
    };
    let mut owners = BTreeMap::<[usize; 3], (usize, [usize; 3])>::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        for local_face in element.element_type.local_faces() {
            let face = [
                vertices[local_face[0]],
                vertices[local_face[1]],
                vertices[local_face[2]],
            ];
            let mut key = face;
            key.sort_unstable();
            owners.entry(key).or_insert((element_index, face));
        }
    }

    let mut faces = Vec::with_capacity(owners.len());
    for (&key, &(owner_element, face)) in &owners {
        let first = mesh.nodes[face[0]];
        let second = mesh.nodes[face[1]];
        let third = mesh.nodes[face[2]];
        let first_xyz = [first.x, first.y, first.z];
        let second_xyz = [second.x, second.y, second.z];
        let third_xyz = [third.x, third.y, third.z];
        let area = triangle_area(first_xyz, second_xyz, third_xyz);
        if !area.is_finite() || area <= tolerance {
            return Err(HangingNodeError::DegenerateFace { face: key });
        }
        let local_dofs = space.local_to_global(mesh, owner_element);
        let mut trace = local_dofs
            .into_iter()
            .filter_map(|dof| {
                let (reference, distance) =
                    triangle_reference(coordinates[dof], first_xyz, second_xyz, third_xyz)?;
                let lambda0 = 1.0 - reference[0] - reference[1];
                let margin = tolerance / area.max(tolerance);
                if distance <= tolerance
                    && reference[0] >= -margin
                    && reference[1] >= -margin
                    && lambda0 >= -margin
                    && reference[0] <= 1.0 + margin
                    && reference[1] <= 1.0 + margin
                {
                    Some((dof, reference))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        trace.sort_by(|left, right| {
            left.1[0]
                .total_cmp(&right.1[0])
                .then_with(|| left.1[1].total_cmp(&right.1[1]))
                .then_with(|| left.0.cmp(&right.0))
        });
        trace.dedup_by(|left, right| left.0 == right.0);
        if trace.len() != expected_trace {
            return Err(HangingNodeError::InvalidFaceTraceDofs {
                element_index: owner_element,
                face,
                actual: trace.len(),
                expected: expected_trace,
            });
        }
        faces.push(H1FaceRecord { face, area, trace });
    }

    let mut candidates = BTreeMap::<usize, Vec<H1FaceCandidate>>::new();
    for (face_index, face) in faces.iter().enumerate() {
        let first = mesh.nodes[face.face[0]];
        let second = mesh.nodes[face.face[1]];
        let third = mesh.nodes[face.face[2]];
        let first_xyz = [first.x, first.y, first.z];
        let second_xyz = [second.x, second.y, second.z];
        let third_xyz = [third.x, third.y, third.z];
        for (dof, &point) in coordinates.iter().enumerate() {
            if face.trace.iter().any(|&(trace_dof, _)| trace_dof == dof) {
                continue;
            }
            // A DOF already owned by a larger face is a coarse master, even
            // when its coordinate lies on a shorter fine face.
            if faces.iter().any(|owner| {
                owner.area > face.area + tolerance
                    && owner.trace.iter().any(|&(trace_dof, _)| trace_dof == dof)
            }) {
                continue;
            }
            let Some((reference, distance)) =
                triangle_reference(point, first_xyz, second_xyz, third_xyz)
            else {
                continue;
            };
            let lambda0 = 1.0 - reference[0] - reference[1];
            let margin = tolerance / face.area.max(tolerance);
            if distance <= tolerance
                && reference[0] > -margin
                && reference[1] > -margin
                && lambda0 > -margin
                && reference[0] < 1.0 + margin
                && reference[1] < 1.0 + margin
            {
                candidates.entry(dof).or_default().push(H1FaceCandidate {
                    face_index,
                    reference,
                    area: face.area,
                });
            }
        }
    }

    // A hanging point on a coarse edge belongs geometrically to two coarse
    // faces. Prefer the face carrying the larger nonconforming trace before
    // using area as the tie-breaker; equal-area side faces with fewer fine
    // trace DOFs must not make a valid face patch ambiguous.
    let mut face_candidate_counts = vec![0usize; faces.len()];
    for face_candidates in candidates.values() {
        for candidate in face_candidates {
            face_candidate_counts[candidate.face_index] += 1;
        }
    }

    let mut constraints = Vec::with_capacity(candidates.len());
    for (slave, mut face_candidates) in candidates {
        face_candidates.sort_by(|left, right| {
            face_candidate_counts[right.face_index]
                .cmp(&face_candidate_counts[left.face_index])
                .then_with(|| right.area.total_cmp(&left.area))
                .then_with(|| left.face_index.cmp(&right.face_index))
        });
        let best = face_candidates[0];
        let best_count = face_candidate_counts[best.face_index];
        if face_candidates.iter().skip(1).any(|candidate| {
            face_candidate_counts[candidate.face_index] == best_count
                && (best.area - candidate.area).abs() <= tolerance
                && candidate.face_index != best.face_index
        }) {
            return Err(HangingNodeError::AmbiguousConstraint { slave });
        }
        let face = &faces[best.face_index];
        let masters = triangle_lagrange_terms(&face.trace, best.reference, space.degree)?;
        constraints.push(H1HangingFaceConstraint {
            slave,
            face: face.face,
            reference: best.reference,
            masters,
        });
    }
    Ok(constraints)
}

/// Build the generic scalar MPC system for 3-D triangular H1 face traces.
pub fn h1_hanging_face_mpc_system_3d(
    mesh: &Mesh,
    space: &H1Space,
    tolerance: f64,
) -> Result<MpcSystem, HangingNodeError> {
    let constraints = h1_hanging_face_constraints_3d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .map(H1HangingFaceConstraint::to_mpc)
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(HangingNodeError::Mpc)
}

/// Lift scalar H1 hanging-node constraints to an interleaved vector H1 space.
///
/// `VectorH1Space` owns the global numbering `[scalar_dof * components + c]`.
/// Keeping the lift here makes the ownership rule explicit and prevents vector
/// consumers from independently reconstructing component offsets.
pub(crate) fn lift_h1_mpc_constraints(
    scalar_mpcs: &[MultiPointConstraint],
    scalar_dofs: usize,
    vector_dofs: usize,
    components: usize,
) -> Result<MpcSystem, HangingNodeError> {
    if components == 0 || vector_dofs != scalar_dofs.saturating_mul(components) {
        return Err(HangingNodeError::Mpc(MpcError::InvalidComponents(
            components,
        )));
    }

    let mut vector_mpcs = Vec::with_capacity(scalar_mpcs.len() * components);
    for scalar_mpc in scalar_mpcs {
        for component in 0..components {
            let slave = scalar_mpc
                .slave
                .checked_mul(components)
                .and_then(|dof| dof.checked_add(component))
                .ok_or(HangingNodeError::Mpc(MpcError::DofOutOfRange {
                    dof: usize::MAX,
                    num_dofs: vector_dofs,
                }))?;
            let masters = scalar_mpc
                .masters
                .iter()
                .map(|term| {
                    term.dof
                        .checked_mul(components)
                        .and_then(|dof| dof.checked_add(component))
                        .map(|dof| MpcTerm {
                            dof,
                            coefficient: term.coefficient,
                        })
                        .ok_or(HangingNodeError::Mpc(MpcError::DofOutOfRange {
                            dof: usize::MAX,
                            num_dofs: vector_dofs,
                        }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            vector_mpcs.push(MultiPointConstraint { slave, masters });
        }
    }

    MpcSystem::new(vector_dofs, &vector_mpcs).map_err(HangingNodeError::Mpc)
}

/// Build a checked vector-H1 MPC system for 2-D hanging edges.
pub fn vector_h1_hanging_node_mpc_system_2d(
    mesh: &Mesh,
    space: &VectorH1Space,
    tolerance: f64,
) -> Result<MpcSystem, HangingNodeError> {
    let scalar_space = H1Space::new(space.degree());
    let scalar_constraints = h1_hanging_node_constraints_2d(mesh, &scalar_space, tolerance)?;
    let scalar_mpcs = scalar_constraints
        .iter()
        .map(H1HangingNodeConstraint::to_mpc)
        .collect::<Vec<_>>();
    lift_h1_mpc_constraints(
        &scalar_mpcs,
        scalar_space.num_global_dofs(mesh),
        space.num_global_dofs(mesh),
        space.num_components(),
    )
}

/// Build a checked vector-H1 MPC system for 3-D tetrahedral hanging faces.
pub fn vector_h1_hanging_face_mpc_system_3d(
    mesh: &Mesh,
    space: &VectorH1Space,
    tolerance: f64,
) -> Result<MpcSystem, HangingNodeError> {
    let scalar_space = H1Space::new(space.degree());
    let scalar_constraints = h1_hanging_face_constraints_3d(mesh, &scalar_space, tolerance)?;
    let scalar_mpcs = scalar_constraints
        .iter()
        .map(H1HangingFaceConstraint::to_mpc)
        .collect::<Vec<_>>();
    lift_h1_mpc_constraints(
        &scalar_mpcs,
        scalar_space.num_global_dofs(mesh),
        space.num_global_dofs(mesh),
        space.num_components(),
    )
}

fn triangle_area(first: [f64; 3], second: [f64; 3], third: [f64; 3]) -> f64 {
    let ab = [
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    ];
    let ac = [
        third[0] - first[0],
        third[1] - first[1],
        third[2] - first[2],
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * cross.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn triangle_reference(
    point: [f64; 3],
    first: [f64; 3],
    second: [f64; 3],
    third: [f64; 3],
) -> Option<([f64; 2], f64)> {
    let v0 = [
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    ];
    let v1 = [
        third[0] - first[0],
        third[1] - first[1],
        third[2] - first[2],
    ];
    let v2 = [
        point[0] - first[0],
        point[1] - first[1],
        point[2] - first[2],
    ];
    let dot = |left: [f64; 3], right: [f64; 3]| {
        left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
    };
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denominator = d00 * d11 - d01 * d01;
    if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
        return None;
    }
    let reference = [
        (d11 * d20 - d01 * d21) / denominator,
        (d00 * d21 - d01 * d20) / denominator,
    ];
    let normal = [
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0],
    ];
    let normal_length = dot(normal, normal).sqrt();
    let distance = dot(v2, normal).abs() / normal_length;
    if reference.iter().all(|value| value.is_finite()) && distance.is_finite() {
        Some((reference, distance))
    } else {
        None
    }
}

fn triangle_lagrange_terms(
    trace: &[(usize, [f64; 2])],
    reference: [f64; 2],
    degree: PolynomialDegree,
) -> Result<Vec<MpcTerm>, HangingNodeError> {
    let order = match degree {
        PolynomialDegree::P1 => 1,
        PolynomialDegree::P2 => 2,
        PolynomialDegree::P3 => 3,
    };
    let expected = (order + 1) * (order + 2) / 2;
    if trace.len() != expected {
        return Err(HangingNodeError::InvalidFaceInterpolation);
    }
    let exponents = (0..=order)
        .flat_map(|total| (0..=total).map(move |first| (first, total - first)))
        .collect::<Vec<_>>();
    let monomials = |point: [f64; 2]| {
        exponents
            .iter()
            .map(|&(first, second)| point[0].powi(first as i32) * point[1].powi(second as i32))
            .collect::<Vec<_>>()
    };
    let mut matrix = vec![vec![0.0; expected]; expected];
    for (row, &(_, point)) in trace.iter().enumerate() {
        for (column, value) in monomials(point).into_iter().enumerate() {
            matrix[column][row] = value;
        }
    }
    let coefficients = solve_dense_system(matrix, monomials(reference))
        .ok_or(HangingNodeError::InvalidFaceInterpolation)?;
    Ok(trace
        .iter()
        .zip(coefficients)
        .map(|(&(dof, _), coefficient)| MpcTerm {
            dof,
            coefficient: Complex64::new(coefficient, 0.0),
        })
        .collect())
}

fn solve_dense_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let size = rhs.len();
    if matrix.len() != size || matrix.iter().any(|row| row.len() != size) {
        return None;
    }
    for pivot in 0..size {
        let (pivot_row, pivot_value) = (pivot..size)
            .map(|row| (row, matrix[row][pivot].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))?;
        if !pivot_value.is_finite() || pivot_value <= 1.0e-14 {
            return None;
        }
        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);
        let diagonal = matrix[pivot][pivot];
        for slot in &mut matrix[pivot][pivot..size] {
            *slot /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            #[expect(
                clippy::needless_range_loop,
                reason = "simultaneous read of the pivot row and in-place update of another row; the borrow checker cannot express disjoint row borrows, and snapshotting would heap-clone per elimination step"
            )]
            for column in pivot..size {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    if rhs.iter().all(|value| value.is_finite()) {
        Some(rhs)
    } else {
        None
    }
}

fn affine_segment_parameter(
    point: [f64; 3],
    first: [f64; 2],
    second: [f64; 2],
    length_squared: f64,
    length: f64,
) -> Option<(f64, f64)> {
    if !length_squared.is_finite() || length_squared <= 0.0 || !length.is_finite() {
        return None;
    }
    let px = point[0] - first[0];
    let py = point[1] - first[1];
    let dx = second[0] - first[0];
    let dy = second[1] - first[1];
    let parameter = (dx * px + dy * py) / length_squared;
    let distance = (dx * py - dy * px).abs() / length;
    if parameter.is_finite() && distance.is_finite() {
        Some((parameter, distance))
    } else {
        None
    }
}

fn lagrange_edge_terms(
    trace: &[(usize, f64)],
    parameter: f64,
) -> Result<Vec<MpcTerm>, HangingNodeError> {
    let mut terms = Vec::with_capacity(trace.len());
    for (index, &(dof, node_parameter)) in trace.iter().enumerate() {
        let mut coefficient = 1.0;
        for (other_index, &(_, other_parameter)) in trace.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let denominator = node_parameter - other_parameter;
            if denominator == 0.0 || !denominator.is_finite() {
                return Err(HangingNodeError::InvalidTraceDofs {
                    element_index: usize::MAX,
                    edge: [usize::MAX; 2],
                    actual: trace.len(),
                    expected: trace.len(),
                });
            }
            coefficient *= (parameter - other_parameter) / denominator;
        }
        if !coefficient.is_finite() {
            return Err(HangingNodeError::InvalidTraceDofs {
                element_index: usize::MAX,
                edge: [usize::MAX; 2],
                actual: trace.len(),
                expected: trace.len(),
            });
        }
        terms.push(MpcTerm {
            dof,
            coefficient: Complex64::new(coefficient, 0.0),
        });
    }
    Ok(terms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Point;

    fn one_hanging_node_mesh() -> Mesh {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(0.5, 0.0),
            Point::new_2d(0.5, -1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
        mesh.add_element(ElementType::Triangle, vec![0, 3, 4]);
        mesh.add_element(ElementType::Triangle, vec![3, 1, 4]);
        mesh
    }

    fn two_hanging_node_mesh() -> Mesh {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(0.25, 0.0),
            Point::new_2d(0.75, 0.0),
            Point::new_2d(0.5, -1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
        mesh.add_element(ElementType::Triangle, vec![0, 3, 5]);
        mesh.add_element(ElementType::Triangle, vec![3, 4, 5]);
        mesh.add_element(ElementType::Triangle, vec![4, 1, 5]);
        mesh
    }

    fn conforming_square_mesh() -> Mesh {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(1.0, 1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
        mesh.add_element(ElementType::Triangle, vec![1, 3, 2]);
        mesh
    }

    #[test]
    fn detects_midpoint_and_builds_interpolation_mpc() {
        let mesh = one_hanging_node_mesh();
        let constraints = hanging_node_constraints_2d(&mesh, 1.0e-10).unwrap();
        assert_eq!(
            constraints,
            vec![HangingNodeConstraint {
                slave: 3,
                edge: [0, 1],
                parameter: 0.5,
            }]
        );
        let mpc = constraints[0].to_mpc();
        assert_eq!(mpc.slave, 3);
        assert_eq!(mpc.masters[0].dof, 0);
        assert_eq!(mpc.masters[1].dof, 1);
        assert!((mpc.masters[0].coefficient.re - 0.5).abs() < 1.0e-14);
        assert!((mpc.masters[1].coefficient.re - 0.5).abs() < 1.0e-14);
    }

    #[test]
    fn admits_multiple_hanging_nodes_in_deterministic_slave_order() {
        let mesh = two_hanging_node_mesh();
        let constraints = hanging_node_constraints_2d(&mesh, 1.0e-10).unwrap();
        assert_eq!(
            constraints
                .iter()
                .map(|item| item.slave)
                .collect::<Vec<_>>(),
            [3, 4]
        );
        assert!((constraints[0].parameter - 0.25).abs() < 1.0e-14);
        assert!((constraints[1].parameter - 0.75).abs() < 1.0e-14);
    }

    #[test]
    fn mpc_system_recovers_hanging_values_and_leaves_conforming_mesh_unchanged() {
        let mesh = two_hanging_node_mesh();
        let system = hanging_node_mpc_system_2d(&mesh, 1.0e-10).unwrap();
        assert_eq!(system.num_full_dofs(), 6);
        assert_eq!(system.num_reduced_dofs(), 4);
        let full = system
            .recover(&[2.0.into(), 6.0.into(), 9.0.into(), 4.0.into()])
            .unwrap();
        assert!((full[3].re - 3.0).abs() < 1.0e-14);
        assert!((full[4].re - 5.0).abs() < 1.0e-14);

        let conforming = conforming_square_mesh();
        let no_constraints = hanging_node_constraints_2d(&conforming, 1.0e-10).unwrap();
        assert!(no_constraints.is_empty());
    }

    #[test]
    fn rejects_invalid_tolerance_dimension_order_and_malformed_mesh() {
        let mesh = one_hanging_node_mesh();
        assert!(matches!(
            hanging_node_constraints_2d(&mesh, 0.0),
            Err(HangingNodeError::InvalidTolerance(0.0))
        ));
        assert!(matches!(
            hanging_node_constraints_2d(&mesh, f64::NAN),
            Err(HangingNodeError::InvalidTolerance(value)) if value.is_nan()
        ));

        let mut three_d = Mesh::new(3);
        three_d.add_node(Point::new_3d(0.0, 0.0, 0.0));
        assert!(matches!(
            hanging_node_constraints_2d(&three_d, 1.0e-10),
            Err(HangingNodeError::UnsupportedDimension(3))
        ));

        let mut malformed = Mesh::new(2);
        malformed.add_node(Point::new_2d(0.0, 0.0));
        malformed.add_node(Point::new_2d(1.0, 0.0));
        malformed.add_node(Point::new_2d(0.0, 1.0));
        malformed.add_element(ElementType::Triangle, vec![0, 1, 9]);
        assert!(matches!(
            hanging_node_constraints_2d(&malformed, 1.0e-10),
            Err(HangingNodeError::InvalidMesh { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_order_and_degenerate_geometry() {
        let mut p2 = Mesh::new(2);
        for index in 0..6 {
            p2.add_node(Point::new_2d(index as f64, 0.0));
        }
        p2.add_element(ElementType::Triangle, vec![0, 1, 2, 3, 4, 5]);
        assert!(matches!(
            hanging_node_constraints_2d(&p2, 1.0e-10),
            Err(HangingNodeError::UnsupportedElement {
                element_index: 0,
                element_type: ElementType::Triangle,
                actual: 6,
            })
        ));

        let mut degenerate = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(2.0, 0.0),
        ] {
            degenerate.add_node(point);
        }
        degenerate.add_element(ElementType::Triangle, vec![0, 1, 2]);
        assert!(matches!(
            hanging_node_constraints_2d(&degenerate, 1.0e-10),
            Err(HangingNodeError::DegenerateElement { element_index: 0 })
        ));
    }

    #[test]
    fn accepts_small_geometric_noise_only_inside_requested_tolerance() {
        let mut mesh = one_hanging_node_mesh();
        mesh.nodes[3].y = 5.0e-8;
        assert!(
            hanging_node_constraints_2d(&mesh, 1.0e-9)
                .unwrap()
                .is_empty()
        );
        assert_eq!(hanging_node_constraints_2d(&mesh, 1.0e-7).unwrap().len(), 1);
    }

    fn polynomial_value(point: [f64; 3], degree: PolynomialDegree) -> f64 {
        let quadratic = if degree == PolynomialDegree::P1 {
            0.0
        } else {
            0.5 * point[0] * point[0]
        };
        let cubic = if degree == PolynomialDegree::P3 {
            0.25 * point[0] * point[0] * point[0]
        } else {
            0.0
        };
        1.25 + 2.0 * point[0] - 0.75 * point[1] + quadratic + cubic
    }

    fn assert_high_order_constraints(
        mesh: &Mesh,
        edge: [usize; 2],
        degree: PolynomialDegree,
        expected: usize,
    ) {
        let space = H1Space::new(degree);
        let constraints = h1_hanging_node_constraints_2d(mesh, &space, 1.0e-10).unwrap();
        assert_eq!(constraints.len(), expected, "constraints: {constraints:?}");
        let expected_trace = match degree {
            PolynomialDegree::P1 => 2,
            PolynomialDegree::P2 => 3,
            PolynomialDegree::P3 => 4,
        };
        for constraint in &constraints {
            assert_eq!(constraint.edge, edge);
            assert_eq!(constraint.masters.len(), expected_trace);
            let coefficient_sum: f64 = constraint
                .masters
                .iter()
                .map(|term| term.coefficient.re)
                .sum();
            assert!((coefficient_sum - 1.0).abs() < 1.0e-12);
        }

        let mpc = h1_hanging_node_mpc_system_2d(mesh, &space, 1.0e-10).unwrap();
        let coordinates = space.dof_coordinates(mesh).unwrap();
        let reduced = mpc
            .reduced_full_dofs()
            .iter()
            .map(|&dof| Complex64::new(polynomial_value(coordinates[dof], degree), 0.0))
            .collect::<Vec<_>>();
        let recovered = mpc.recover(&reduced).unwrap();
        for (dof, value) in recovered.iter().enumerate() {
            assert!(
                (value.re - polynomial_value(coordinates[dof], degree)).abs() < 1.0e-10,
                "recovered DOF {dof} does not preserve the affine-edge polynomial"
            );
            assert!(value.im.abs() < 1.0e-14);
        }
    }

    #[test]
    fn h1_p1_hanging_constraints_match_legacy_contract() {
        let mesh = one_hanging_node_mesh();
        assert_high_order_constraints(&mesh, [0, 1], PolynomialDegree::P1, 1);
    }

    #[test]
    fn h1_p2_hanging_constraints_cover_fine_vertex_and_edge_dofs() {
        let mesh = one_hanging_node_mesh();
        assert_high_order_constraints(&mesh, [0, 1], PolynomialDegree::P2, 3);
    }

    #[test]
    fn h1_p3_hanging_constraints_cover_fine_vertex_and_edge_dofs() {
        let mesh = one_hanging_node_mesh();
        assert_high_order_constraints(&mesh, [0, 1], PolynomialDegree::P3, 5);
    }

    fn one_hanging_quad_mesh() -> Mesh {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(1.0, 1.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(0.5, 0.0),
            Point::new_2d(0.5, -1.0),
            Point::new_2d(0.0, -1.0),
            Point::new_2d(1.0, -1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Quadrilateral, vec![0, 1, 2, 3]);
        mesh.add_element(ElementType::Quadrilateral, vec![0, 4, 5, 6]);
        mesh.add_element(ElementType::Quadrilateral, vec![4, 1, 7, 5]);
        mesh
    }

    #[test]
    fn h1_p2_p3_quadrilateral_hanging_traces_are_constrained() {
        let mesh = one_hanging_quad_mesh();
        assert_high_order_constraints(&mesh, [0, 1], PolynomialDegree::P2, 3);
        assert_high_order_constraints(&mesh, [0, 1], PolynomialDegree::P3, 5);
    }

    fn one_hanging_tetra_face_mesh() -> Mesh {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
            Point::new_3d(0.5, 0.0, 0.0),
            Point::new_3d(0.5, 0.5, 0.0),
            Point::new_3d(0.0, 0.5, 0.0),
            Point::new_3d(0.1, 0.1, -1.0),
            Point::new_3d(0.6, 0.1, -1.0),
            Point::new_3d(0.1, 0.6, -1.0),
            Point::new_3d(0.3, 0.3, -1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        mesh.add_element(ElementType::Tetrahedron, vec![0, 4, 6, 7]);
        mesh.add_element(ElementType::Tetrahedron, vec![4, 1, 5, 8]);
        mesh.add_element(ElementType::Tetrahedron, vec![6, 5, 2, 9]);
        mesh.add_element(ElementType::Tetrahedron, vec![4, 5, 6, 10]);
        mesh
    }

    fn face_polynomial(point: [f64; 3], degree: PolynomialDegree) -> f64 {
        let quadratic = if degree == PolynomialDegree::P1 {
            0.0
        } else {
            0.25 * (point[0] * point[0] + point[1] * point[1])
        };
        let cubic = if degree == PolynomialDegree::P3 {
            0.125 * point[0] * point[1] * point[2]
        } else {
            0.0
        };
        0.5 + 1.5 * point[0] - 0.75 * point[1] + 0.25 * point[2] + quadratic + cubic
    }

    fn assert_face_constraints(degree: PolynomialDegree, expected: usize) {
        let mesh = one_hanging_tetra_face_mesh();
        let space = H1Space::new(degree);
        let constraints = h1_hanging_face_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
        assert_eq!(constraints.len(), expected);
        let expected_trace = match degree {
            PolynomialDegree::P1 => 3,
            PolynomialDegree::P2 => 6,
            PolynomialDegree::P3 => 10,
        };
        for constraint in &constraints {
            let mut face = constraint.face;
            face.sort_unstable();
            assert_eq!(face, [0, 1, 2]);
            assert_eq!(constraint.masters.len(), expected_trace);
            let sum: f64 = constraint
                .masters
                .iter()
                .map(|term| term.coefficient.re)
                .sum();
            assert!((sum - 1.0).abs() < 1.0e-10);
        }

        let mpc = h1_hanging_face_mpc_system_3d(&mesh, &space, 1.0e-10).unwrap();
        let coordinates = space.dof_coordinates(&mesh).unwrap();
        let reduced = mpc
            .reduced_full_dofs()
            .iter()
            .map(|&dof| Complex64::new(face_polynomial(coordinates[dof], degree), 0.0))
            .collect::<Vec<_>>();
        let recovered = mpc.recover(&reduced).unwrap();
        for (dof, value) in recovered.iter().enumerate() {
            assert!(
                (value.re - face_polynomial(coordinates[dof], degree)).abs() < 1.0e-9,
                "face recovery mismatch at DOF {dof}"
            );
            assert!(value.im.abs() < 1.0e-14);
        }
    }

    #[test]
    fn h1_p1_p2_p3_tetrahedral_face_hanging_traces_are_constrained() {
        assert_face_constraints(PolynomialDegree::P1, 3);
        assert_face_constraints(PolynomialDegree::P2, 12);
        assert_face_constraints(PolynomialDegree::P3, 25);
    }

    #[test]
    fn h1_face_constraints_reject_non_tetrahedral_and_explicit_geometry() {
        let mut hexa = Mesh::new(3);
        for index in 0..8 {
            hexa.add_node(Point::new_3d(
                (index & 1) as f64,
                ((index >> 1) & 1) as f64,
                ((index >> 2) & 1) as f64,
            ));
        }
        hexa.add_element(ElementType::Hexahedron, (0..8).collect());
        let p2 = H1Space::new(PolynomialDegree::P2);
        assert!(matches!(
            h1_hanging_face_constraints_3d(&hexa, &p2, 1.0e-10),
            Err(HangingNodeError::UnsupportedH1Element {
                element_index: 0,
                element_type: ElementType::Hexahedron,
            })
        ));

        let mut explicit = one_hanging_tetra_face_mesh();
        explicit.elements[0].nodes.extend([11, 12, 13, 14, 15, 16]);
        for index in 11..17 {
            explicit.add_node(Point::new_3d(0.0, 0.0, index as f64));
        }
        assert!(matches!(
            h1_hanging_face_constraints_3d(&explicit, &p2, 1.0e-10),
            Err(HangingNodeError::InvalidH1Space(_))
                | Err(HangingNodeError::UnsupportedGeometry { .. })
        ));
    }

    #[test]
    fn h1_hanging_constraints_reject_explicit_geometry_and_invalid_inputs() {
        let mut explicit = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(0.5, 0.0),
            Point::new_2d(0.5, 0.5),
            Point::new_2d(0.0, 0.5),
        ] {
            explicit.add_node(point);
        }
        explicit.add_element(ElementType::Triangle, vec![0, 1, 2, 3, 4, 5]);
        let p2 = H1Space::new(PolynomialDegree::P2);
        assert!(matches!(
            h1_hanging_node_constraints_2d(&explicit, &p2, 1.0e-10),
            Err(HangingNodeError::UnsupportedGeometry {
                element_index: 0,
                actual: 6,
                expected: 3,
            })
        ));

        let mesh = one_hanging_node_mesh();
        assert!(matches!(
            h1_hanging_node_constraints_2d(&mesh, &p2, 0.0),
            Err(HangingNodeError::InvalidTolerance(0.0))
        ));
        let mut three_d = Mesh::new(3);
        three_d.add_node(Point::new_3d(0.0, 0.0, 0.0));
        assert!(matches!(
            h1_hanging_node_constraints_2d(&three_d, &p2, 1.0e-10),
            Err(HangingNodeError::UnsupportedDimension(3))
        ));
    }
}
