//! Mesh types for 2D and 3D finite element analysis
//!
//! Supports triangular, quadrilateral, tetrahedral, and hexahedral elements.

use super::parallelism::{ParallelPolicy, TOPOLOGY_ITEMS_PER_THREAD};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

/// A point in 2D or 3D space
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point {
    /// Create a 2D point (z = 0)
    pub fn new_2d(x: f64, y: f64) -> Self {
        Self { x, y, z: 0.0 }
    }

    /// Create a 3D point
    pub fn new_3d(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another point
    pub fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Midpoint between two points
    pub fn midpoint(&self, other: &Point) -> Point {
        Point {
            x: 0.5 * (self.x + other.x),
            y: 0.5 * (self.y + other.y),
            z: 0.5 * (self.z + other.z),
        }
    }
}

impl From<(f64, f64)> for Point {
    fn from(p: (f64, f64)) -> Self {
        Point::new_2d(p.0, p.1)
    }
}

impl From<(f64, f64, f64)> for Point {
    fn from(p: (f64, f64, f64)) -> Self {
        Point::new_3d(p.0, p.1, p.2)
    }
}

/// Element type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementType {
    /// 2D triangle (3 nodes for P1, 6 for P2)
    Triangle,
    /// 2D quadrilateral (4 nodes for Q1, 9 for Q2)
    Quadrilateral,
    /// 3D tetrahedron (4 nodes for P1, 10 for P2)
    Tetrahedron,
    /// 3D hexahedron (8 nodes for Q1, 27 for Q2)
    Hexahedron,
}

impl ElementType {
    /// Number of vertices (corners) for this element type
    pub fn num_vertices(&self) -> usize {
        match self {
            ElementType::Triangle => 3,
            ElementType::Quadrilateral => 4,
            ElementType::Tetrahedron => 4,
            ElementType::Hexahedron => 8,
        }
    }

    /// Number of nodes for P1/Q1 elements
    pub fn num_nodes_p1(&self) -> usize {
        self.num_vertices()
    }

    /// Number of nodes for P2/Q2 elements
    pub fn num_nodes_p2(&self) -> usize {
        match self {
            ElementType::Triangle => 6,      // 3 vertices + 3 edge midpoints
            ElementType::Quadrilateral => 9, // 4 vertices + 4 edge midpoints + 1 center
            ElementType::Tetrahedron => 10,  // 4 vertices + 6 edge midpoints
            ElementType::Hexahedron => 27,   // 8 vertices + 12 edge + 6 face + 1 center
        }
    }

    /// Number of nodes in the cubic Lagrange layout for this element type.
    ///
    /// The canonical ordering is vertices, two nodes per edge, face-interior
    /// nodes, then cell-interior nodes.
    pub fn num_nodes_p3(&self) -> usize {
        match self {
            ElementType::Triangle => 10,
            ElementType::Quadrilateral => 16,
            ElementType::Tetrahedron => 20,
            ElementType::Hexahedron => 64,
        }
    }

    /// Spatial dimension of this element
    pub fn dimension(&self) -> usize {
        match self {
            ElementType::Triangle | ElementType::Quadrilateral => 2,
            ElementType::Tetrahedron | ElementType::Hexahedron => 3,
        }
    }

    /// Number of edges for this element type
    pub fn num_edges(&self) -> usize {
        match self {
            ElementType::Triangle => 3,
            ElementType::Quadrilateral => 4,
            ElementType::Tetrahedron => 6,
            ElementType::Hexahedron => 12,
        }
    }

    /// Number of faces for this element type
    pub fn num_faces(&self) -> usize {
        match self {
            ElementType::Triangle => 3,
            ElementType::Quadrilateral => 4,
            ElementType::Tetrahedron => 4,
            ElementType::Hexahedron => 6,
        }
    }

    /// Local edge endpoints (as indices into the vertex slice) in the
    /// canonical FEM order used by sonium. The order matches
    /// [`Element::edges`] and the P2 node layout in [`p2`](super::p2).
    pub fn local_edges(&self) -> &'static [(usize, usize)] {
        match self {
            ElementType::Triangle => &[(0, 1), (1, 2), (2, 0)],
            ElementType::Quadrilateral => &[(0, 1), (1, 2), (2, 3), (3, 0)],
            ElementType::Tetrahedron => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            ElementType::Hexahedron => &[
                // bottom face
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                // top face
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                // vertical edges
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ],
        }
    }

    /// Local face node lists (vertex indices) in canonical FEM order.
    ///
    /// Each face is listed as a CCW vertex traversal (outward normal), which is
    /// what Q2 face-centre nodes are taken from. Returns an empty slice for 2D
    /// element types (their "faces" are edges, already exposed via
    /// [`local_edges`](Self::local_edges)).
    pub fn local_faces(&self) -> &'static [&'static [usize]] {
        match self {
            ElementType::Triangle | ElementType::Quadrilateral => &[],
            // Tet faces: each opposite to the named vertex, CCW from outside.
            ElementType::Tetrahedron => &[
                &[0, 2, 1], // opposite v3 (bottom)
                &[0, 1, 3], // opposite v2
                &[0, 3, 2], // opposite v1
                &[1, 2, 3], // opposite v0
            ],
            // Hex faces: CCW outward-normal traversal.
            ElementType::Hexahedron => &[
                &[0, 3, 2, 1], // bottom (z−)
                &[4, 5, 6, 7], // top    (z+)
                &[0, 1, 5, 4], // front  (y−)
                &[2, 3, 7, 6], // back   (y+)
                &[0, 4, 7, 3], // left   (x−)
                &[1, 2, 6, 5], // right  (x+)
            ],
        }
    }

    /// True if this element type has an interior (volume/face) bubble node in
    /// its P2/Q2 layout. Only the Lagrange Q2 quadrilateral and Hex27 have one.
    pub fn p2_has_interior(&self) -> bool {
        matches!(self, ElementType::Quadrilateral | ElementType::Hexahedron)
    }

    /// True if this element type has face-centre nodes at order 2. Only Hex27
    /// does; Q2 quadrilaterals have only the interior node, not separate face
    /// nodes.
    pub fn p2_has_face_centres(&self) -> bool {
        matches!(self, ElementType::Hexahedron)
    }
}

/// Polynomial degree for basis functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolynomialDegree {
    P1, // Linear
    P2, // Quadratic
    P3, // Cubic
}

impl PolynomialDegree {
    /// Get the degree as a number
    pub fn as_usize(&self) -> usize {
        match self {
            PolynomialDegree::P1 => 1,
            PolynomialDegree::P2 => 2,
            PolynomialDegree::P3 => 3,
        }
    }
}

/// A finite element with node indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    /// Element type
    pub element_type: ElementType,
    /// Node indices (vertices first, then edge/face/interior nodes for higher order)
    pub nodes: Vec<usize>,
    /// Element ID for refinement tracking
    pub id: usize,
    /// Parent element ID (for refined elements)
    pub parent_id: Option<usize>,
    /// Refinement level
    pub level: usize,
    /// Cell/material marker. Zero denotes an unclassified cell.
    #[serde(default)]
    pub marker: i32,
}

/// Error returned when an element's node count does not describe a supported
/// polynomial layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ElementOrderError {
    #[error(
        "element {element_type:?} has {actual} nodes; expected P1 ({p1}), P2 ({p2}), or P3 ({p3})"
    )]
    InvalidNodeCount {
        element_type: ElementType,
        actual: usize,
        p1: usize,
        p2: usize,
        p3: usize,
    },
}

impl Element {
    /// Create a new element
    pub fn new(element_type: ElementType, nodes: Vec<usize>, id: usize) -> Self {
        Self {
            element_type,
            nodes,
            id,
            parent_id: None,
            level: 0,
            marker: 0,
        }
    }

    /// Assign a cell/material marker while constructing an element.
    pub fn with_marker(mut self, marker: i32) -> Self {
        self.marker = marker;
        self
    }

    /// Get vertex nodes (first N nodes are always vertices)
    pub fn vertices(&self) -> &[usize] {
        &self.nodes[..self.element_type.num_vertices()]
    }

    /// Polynomial order of this element, inferred from `nodes.len()`.
    ///
    /// Returns 1 for P1/Q1 (vertex-only) and 2 for P2/Q2 (vertices + edge
    /// midpoints + optional face/interior bubbles). Panics if the node count
    /// matches neither layout; prefer [`Self::try_polynomial_order`] for
    /// externally supplied elements.
    pub fn polynomial_order(&self) -> usize {
        self.try_polynomial_order()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Checked polynomial-order inference for externally supplied elements.
    pub fn try_polynomial_order(&self) -> Result<usize, ElementOrderError> {
        let n = self.nodes.len();
        if n == self.element_type.num_nodes_p1() {
            Ok(1)
        } else if n == self.element_type.num_nodes_p2() {
            Ok(2)
        } else if n == self.element_type.num_nodes_p3() {
            Ok(3)
        } else {
            Err(ElementOrderError::InvalidNodeCount {
                element_type: self.element_type,
                actual: n,
                p1: self.element_type.num_nodes_p1(),
                p2: self.element_type.num_nodes_p2(),
                p3: self.element_type.num_nodes_p3(),
            })
        }
    }

    /// True iff this element carries P2/Q2 nodes.
    ///
    /// Never panics: malformed node counts report `false` instead of aborting
    /// (see [`Self::try_polynomial_order`]).
    pub fn is_p2(&self) -> bool {
        matches!(self.try_polynomial_order(), Ok(2))
    }

    /// Get edges as pairs of node indices
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let verts = self.vertices();
        match self.element_type {
            ElementType::Triangle => vec![
                (verts[0], verts[1]),
                (verts[1], verts[2]),
                (verts[2], verts[0]),
            ],
            ElementType::Quadrilateral => vec![
                (verts[0], verts[1]),
                (verts[1], verts[2]),
                (verts[2], verts[3]),
                (verts[3], verts[0]),
            ],
            ElementType::Tetrahedron => vec![
                (verts[0], verts[1]),
                (verts[0], verts[2]),
                (verts[0], verts[3]),
                (verts[1], verts[2]),
                (verts[1], verts[3]),
                (verts[2], verts[3]),
            ],
            ElementType::Hexahedron => vec![
                // Bottom face
                (verts[0], verts[1]),
                (verts[1], verts[2]),
                (verts[2], verts[3]),
                (verts[3], verts[0]),
                // Top face
                (verts[4], verts[5]),
                (verts[5], verts[6]),
                (verts[6], verts[7]),
                (verts[7], verts[4]),
                // Vertical edges
                (verts[0], verts[4]),
                (verts[1], verts[5]),
                (verts[2], verts[6]),
                (verts[3], verts[7]),
            ],
        }
    }

    /// Get faces as lists of node indices (for 3D elements only).
    /// Returns empty vec for 2D elements.
    pub fn faces(&self) -> Vec<Vec<usize>> {
        let verts = self.vertices();
        match self.element_type {
            ElementType::Triangle | ElementType::Quadrilateral => {
                vec![]
            }
            ElementType::Tetrahedron => vec![
                vec![verts[0], verts[1], verts[2]], // opposite v3
                vec![verts[0], verts[1], verts[3]], // opposite v2
                vec![verts[0], verts[2], verts[3]], // opposite v1
                vec![verts[1], verts[2], verts[3]], // opposite v0
            ],
            ElementType::Hexahedron => vec![
                vec![verts[0], verts[1], verts[2], verts[3]], // bottom (z-)
                vec![verts[4], verts[5], verts[6], verts[7]], // top    (z+)
                vec![verts[0], verts[1], verts[5], verts[4]], // front  (y-)
                vec![verts[2], verts[3], verts[7], verts[6]], // back   (y+)
                vec![verts[0], verts[3], verts[7], verts[4]], // left   (x-)
                vec![verts[1], verts[2], verts[6], verts[5]], // right  (x+)
            ],
        }
    }

    /// Get the boundary entities with their complete polynomial trace nodes.
    ///
    /// P1 elements return the same corner-only entities as [`Self::edges`] or
    /// [`Self::faces`]. P2/Q2 elements append edge mid-side nodes in the
    /// canonical local order and, for Hex27 faces, the face-centre node. This
    /// is the authoritative layout used by boundary validation and refinement
    /// propagation; topological neighbor keys continue to use corner nodes.
    pub fn boundary_entities(&self) -> Vec<Vec<usize>> {
        let vertices = self.vertices();
        let order = self.try_polynomial_order().unwrap_or(1);
        let edge_node = |a: usize, b: usize| {
            self.element_type
                .local_edges()
                .iter()
                .position(|&(left, right)| (left == a && right == b) || (left == b && right == a))
                .and_then(|edge| self.nodes.get(self.element_type.num_vertices() + edge))
                .copied()
        };

        if self.element_type.dimension() == 2 {
            return self
                .element_type
                .local_edges()
                .iter()
                .map(|&(a, b)| {
                    let mut entity = vec![vertices[a], vertices[b]];
                    if let (2, Some(midpoint)) = (order, edge_node(a, b)) {
                        entity.push(midpoint);
                    }
                    entity
                })
                .collect();
        }

        self.element_type
            .local_faces()
            .iter()
            .enumerate()
            .map(|(face_index, face)| {
                let mut entity: Vec<usize> = face.iter().map(|&vertex| vertices[vertex]).collect();
                if order == 2 {
                    for pair in face.windows(2) {
                        if let Some(midpoint) = edge_node(pair[0], pair[1]) {
                            entity.push(midpoint);
                        }
                    }
                    if let Some(midpoint) = edge_node(face[face.len() - 1], face[0]) {
                        entity.push(midpoint);
                    }
                    if self.element_type == ElementType::Hexahedron {
                        let offset = self.element_type.num_vertices()
                            + self.element_type.num_edges()
                            + face_index;
                        if let Some(center) = self.nodes.get(offset).copied() {
                            entity.push(center);
                        }
                    }
                }
                entity
            })
            .collect()
    }
}

/// Boundary condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Dirichlet: u = g (essential BC)
    Dirichlet,
    /// Neumann: ∂u/∂n = h (natural BC)
    Neumann,
    /// Robin: ∂u/∂n + α*u = g (mixed BC)
    Robin,
    /// Perfectly Matched Layer (absorbing BC)
    PML,
    /// Interior (not a boundary)
    Interior,
}

/// A boundary face or edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryFace {
    /// Node indices defining this boundary face/edge
    pub nodes: Vec<usize>,
    /// Boundary type
    pub boundary_type: BoundaryType,
    /// Boundary marker (for identifying different boundaries)
    pub marker: i32,
    /// Owning element index
    pub element_idx: usize,
    /// Local face/edge index within the element
    pub local_idx: usize,
}

/// A named physical group, identified by topological dimension and integer tag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalGroup {
    /// Topological dimension: 0 point, 1 curve, 2 surface, 3 volume.
    pub dimension: usize,
    /// Positive physical tag used in mesh formats and marker fields.
    pub tag: i32,
    /// Human-readable, dimension-local group name.
    pub name: String,
}

/// Dimension-aware name/tag registry for mesh regions and boundaries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalGroupRegistry {
    groups: Vec<PhysicalGroup>,
}

impl PhysicalGroupRegistry {
    /// Register a group. Tags and names must each be unique within a dimension.
    pub fn register(
        &mut self,
        dimension: usize,
        tag: i32,
        name: impl Into<String>,
    ) -> Result<(), PhysicalGroupError> {
        let name = name.into();
        if dimension > 3 {
            return Err(PhysicalGroupError::InvalidDimension(dimension));
        }
        if tag <= 0 {
            return Err(PhysicalGroupError::InvalidTag(tag));
        }
        if name.trim().is_empty() || name.contains('\0') || name.contains(['\n', '\r']) {
            return Err(PhysicalGroupError::InvalidName);
        }
        if let Some(existing) = self.by_tag(dimension, tag) {
            return if existing.name == name {
                Ok(())
            } else {
                Err(PhysicalGroupError::DuplicateTag { dimension, tag })
            };
        }
        if self.by_name(dimension, &name).is_some() {
            return Err(PhysicalGroupError::DuplicateName { dimension, name });
        }
        self.groups.push(PhysicalGroup {
            dimension,
            tag,
            name,
        });
        self.groups
            .sort_by_key(|group| (group.dimension, group.tag));
        Ok(())
    }

    /// Look up a physical group by dimension and tag.
    pub fn by_tag(&self, dimension: usize, tag: i32) -> Option<&PhysicalGroup> {
        self.groups
            .iter()
            .find(|group| group.dimension == dimension && group.tag == tag)
    }

    /// Look up a physical group by dimension and name.
    pub fn by_name(&self, dimension: usize, name: &str) -> Option<&PhysicalGroup> {
        self.groups
            .iter()
            .find(|group| group.dimension == dimension && group.name == name)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &PhysicalGroup> {
        self.groups.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Revalidate invariants after deserializing externally supplied metadata.
    pub fn validate(&self) -> Result<(), PhysicalGroupError> {
        let mut validated = Self::default();
        for group in &self.groups {
            validated.register(group.dimension, group.tag, group.name.clone())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PhysicalGroupError {
    #[error("physical-group dimension {0} is outside 0..=3")]
    InvalidDimension(usize),
    #[error("physical-group tag {0} must be positive")]
    InvalidTag(i32),
    #[error("physical-group name must be non-empty and single-line")]
    InvalidName,
    #[error("physical-group tag {tag} is already registered in dimension {dimension}")]
    DuplicateTag { dimension: usize, tag: i32 },
    #[error("physical-group name {name:?} is already registered in dimension {dimension}")]
    DuplicateName { dimension: usize, name: String },
    #[error("physical-group dimension {dimension} exceeds mesh dimension {mesh_dimension}")]
    DimensionExceedsMesh {
        dimension: usize,
        mesh_dimension: usize,
    },
    #[error("element index {index} is outside mesh element range 0..{element_count}")]
    InvalidElementIndex { index: usize, element_count: usize },
}

/// Structural errors found in externally supplied mesh connectivity.
///
/// Mesh vectors remain public for serialization and interoperability, so a
/// consumer needs a non-panicking gate before asking topology, quality, or
/// assembly code to dereference element and boundary indices.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MeshConnectivityError {
    #[error("mesh dimension {dimension} must be 2 or 3")]
    InvalidMeshDimension { dimension: usize },
    #[error("node {node_index} has non-finite coordinates")]
    NonFiniteNode { node_index: usize },
    #[error(
        "element {element_index} has dimension {element_dimension}, expected mesh dimension {mesh_dimension}"
    )]
    ElementDimensionMismatch {
        element_index: usize,
        element_dimension: usize,
        mesh_dimension: usize,
    },
    #[error("element {element_index} ({element_type:?}) has unsupported node count {actual}")]
    ElementNodeCount {
        element_index: usize,
        element_type: ElementType,
        actual: usize,
    },
    #[error("face has unsupported node count {actual}; expected 3 or 4")]
    InvalidFaceNodeCount { actual: usize },
    #[error("element {element_index} node {node_position} references missing node {node_index}")]
    InvalidElementNode {
        element_index: usize,
        node_position: usize,
        node_index: usize,
    },
    #[error(
        "element {element_index} repeats node {node_index} at positions {first_position} and {duplicate_position}"
    )]
    DuplicateElementNode {
        element_index: usize,
        first_position: usize,
        duplicate_position: usize,
        node_index: usize,
    },
    #[error("boundary {boundary_index} references missing owning element {element_index}")]
    InvalidBoundaryElement {
        boundary_index: usize,
        element_index: usize,
    },
    #[error("boundary {boundary_index} node {node_position} references missing node {node_index}")]
    InvalidBoundaryNode {
        boundary_index: usize,
        node_position: usize,
        node_index: usize,
    },
    #[error(
        "boundary {boundary_index} local face/edge {local_index} is outside element {element_index}"
    )]
    InvalidBoundaryLocalIndex {
        boundary_index: usize,
        element_index: usize,
        local_index: usize,
    },
    #[error(
        "boundary {boundary_index} nodes do not match owning element {element_index} local face/edge {local_index}"
    )]
    BoundaryNodeSetMismatch {
        boundary_index: usize,
        element_index: usize,
        local_index: usize,
    },
}

/// One enumerated boundary face: (sorted key, traversal-ordered face,
/// element index, local face index).
type FaceEntry = (Vec<usize>, Vec<usize>, usize, usize);

/// A finite element mesh
#[derive(Debug, Clone, Serialize)]
pub struct Mesh {
    /// Mesh dimension (2 or 3)
    pub dimension: usize,
    /// Node coordinates
    pub nodes: Vec<Point>,
    /// Elements
    pub elements: Vec<Element>,
    /// Boundary faces/edges
    pub boundaries: Vec<BoundaryFace>,
    /// Named physical groups for cell regions and boundary entities.
    #[serde(default)]
    pub physical_groups: PhysicalGroupRegistry,
    /// Next element ID for refinement
    pub(crate) next_element_id: usize,
    /// Node to element connectivity (cached)
    #[serde(skip)]
    pub node_to_elements: Option<Vec<Vec<usize>>>,
    #[serde(skip)]
    node_to_elements_fingerprint: Option<u64>,
    /// Edge/face topology (cached, built on demand for H(curl)/H(div) spaces)
    #[serde(skip)]
    pub topology: Option<super::topology::MeshTopology>,
    #[serde(skip)]
    topology_fingerprint: Option<u64>,
    /// Element-to-element neighbor table (cached). Two elements are neighbors
    /// iff they share a full face (3D) or edge (2D). Built lazily by
    /// `Mesh::element_neighbors`; invalidated whenever nodes or elements
    /// change.
    #[serde(skip)]
    pub element_neighbors: Option<Vec<Vec<usize>>>,
    #[serde(skip)]
    element_neighbors_fingerprint: Option<u64>,
}

/// Wire representation used to validate meshes as they cross a serde
/// boundary. Derived deserialization would otherwise permit malformed node
/// indexes, element dimensions, or boundary ownership to reach geometry and
/// FEM kernels before the caller remembered to invoke `validate_connectivity`.
#[derive(Debug, Deserialize)]
struct MeshWire {
    dimension: usize,
    nodes: Vec<Point>,
    elements: Vec<Element>,
    boundaries: Vec<BoundaryFace>,
    #[serde(default)]
    physical_groups: PhysicalGroupRegistry,
    #[serde(default)]
    next_element_id: usize,
}

impl<'de> Deserialize<'de> for Mesh {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MeshWire::deserialize(deserializer)?;
        let next_element_id = if wire.next_element_id == 0 {
            wire.elements
                .iter()
                .map(|element| element.id.saturating_add(1))
                .max()
                .unwrap_or(0)
        } else {
            wire.next_element_id
        };
        let mesh = Self {
            dimension: wire.dimension,
            nodes: wire.nodes,
            elements: wire.elements,
            boundaries: wire.boundaries,
            physical_groups: wire.physical_groups,
            next_element_id,
            node_to_elements: None,
            node_to_elements_fingerprint: None,
            topology: None,
            topology_fingerprint: None,
            element_neighbors: None,
            element_neighbors_fingerprint: None,
        };
        mesh.validate_connectivity().map_err(|errors| {
            <D::Error as serde::de::Error>::custom(format_mesh_errors(&errors))
        })?;
        mesh.validate_physical_groups()
            .map_err(|error| <D::Error as serde::de::Error>::custom(error.to_string()))?;
        Ok(mesh)
    }
}

fn format_mesh_errors(errors: &[MeshConnectivityError]) -> String {
    errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

impl Mesh {
    /// Create a new empty mesh
    pub fn new(dimension: usize) -> Self {
        assert!(dimension == 2 || dimension == 3, "Dimension must be 2 or 3");
        Self {
            dimension,
            nodes: Vec::new(),
            elements: Vec::new(),
            boundaries: Vec::new(),
            physical_groups: PhysicalGroupRegistry::default(),
            next_element_id: 0,
            node_to_elements: None,
            node_to_elements_fingerprint: None,
            topology: None,
            topology_fingerprint: None,
            element_neighbors: None,
            element_neighbors_fingerprint: None,
        }
    }

    /// Invalidate every cached derived structure. Call this whenever node
    /// coordinates, the element list, or boundary tags change in a way that
    /// the lazy caches cannot observe directly. The `add_node` / `add_element`
    /// fast paths already call this for you; external mutators that bypass
    /// those methods (e.g. swapping `mesh.elements` wholesale) must invoke it
    /// themselves.
    pub fn invalidate_caches(&mut self) {
        self.node_to_elements = None;
        self.node_to_elements_fingerprint = None;
        self.topology = None;
        self.topology_fingerprint = None;
        self.element_neighbors = None;
        self.element_neighbors_fingerprint = None;
    }

    /// Fingerprint the index-bearing mesh state used by all derived caches.
    ///
    /// The mesh's node and element vectors remain public for serialization and
    /// interoperability.  Consequently callers can mutate them without going
    /// through the mutation helpers above.  Cache accessors compare this
    /// fingerprint before reusing a cache, so those mutations cannot leave a
    /// stale connectivity or topology result silently in use.  Coordinates
    /// are intentionally excluded: none of these caches depends on them.
    fn cache_fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.nodes.len().hash(&mut hasher);
        self.elements.len().hash(&mut hasher);
        for element in &self.elements {
            let kind = match element.element_type {
                ElementType::Triangle => 0_u8,
                ElementType::Quadrilateral => 1,
                ElementType::Tetrahedron => 2,
                ElementType::Hexahedron => 3,
            };
            kind.hash(&mut hasher);
            element.nodes.hash(&mut hasher);
        }
        self.boundaries.len().hash(&mut hasher);
        for boundary in &self.boundaries {
            boundary.nodes.hash(&mut hasher);
            boundary.boundary_type.hash(&mut hasher);
            boundary.marker.hash(&mut hasher);
            boundary.element_idx.hash(&mut hasher);
            boundary.local_idx.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Add a node and return its index
    pub fn add_node(&mut self, point: Point) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(point);
        self.invalidate_caches();
        idx
    }

    /// Add an element and return its index
    pub fn add_element(&mut self, element_type: ElementType, nodes: Vec<usize>) -> usize {
        self.add_element_with_marker(element_type, nodes, 0)
    }

    /// Add an element with a cell/material marker and return its index.
    pub fn add_element_with_marker(
        &mut self,
        element_type: ElementType,
        nodes: Vec<usize>,
        marker: i32,
    ) -> usize {
        let idx = self.elements.len();
        let id = self.next_element_id;
        self.next_element_id += 1;
        self.elements
            .push(Element::new(element_type, nodes, id).with_marker(marker));
        self.invalidate_caches();
        idx
    }

    /// Change the marker of an existing cell.
    pub fn set_cell_marker(
        &mut self,
        element_idx: usize,
        marker: i32,
    ) -> Result<(), PhysicalGroupError> {
        let element_count = self.elements.len();
        let element =
            self.elements
                .get_mut(element_idx)
                .ok_or(PhysicalGroupError::InvalidElementIndex {
                    index: element_idx,
                    element_count,
                })?;
        element.marker = marker;
        self.invalidate_caches();
        Ok(())
    }

    /// Iterate over element indices carrying `marker`.
    pub fn cells_with_marker(&self, marker: i32) -> impl Iterator<Item = usize> + '_ {
        self.elements
            .iter()
            .enumerate()
            .filter_map(move |(index, element)| (element.marker == marker).then_some(index))
    }

    /// Register a named group whose dimension is valid for this mesh.
    pub fn register_physical_group(
        &mut self,
        dimension: usize,
        tag: i32,
        name: impl Into<String>,
    ) -> Result<(), PhysicalGroupError> {
        if dimension > self.dimension {
            return Err(PhysicalGroupError::DimensionExceedsMesh {
                dimension,
                mesh_dimension: self.dimension,
            });
        }
        self.physical_groups.register(dimension, tag, name)
    }

    /// Validate named-group invariants and compatibility with this mesh.
    pub fn validate_physical_groups(&self) -> Result<(), PhysicalGroupError> {
        self.physical_groups.validate()?;
        if let Some(group) = self
            .physical_groups
            .iter()
            .find(|group| group.dimension > self.dimension)
        {
            return Err(PhysicalGroupError::DimensionExceedsMesh {
                dimension: group.dimension,
                mesh_dimension: self.dimension,
            });
        }
        Ok(())
    }

    /// Validate all index, element-order, boundary-incidence, and coordinate
    /// invariants that topology and assembly code rely on.
    ///
    /// The mesh fields intentionally remain public for serde and external
    /// format adapters. Call this gate after deserialization or direct vector
    /// mutation; it reports every discovered issue and never panics while
    /// inspecting malformed input.
    pub fn validate_connectivity(&self) -> Result<(), Vec<MeshConnectivityError>> {
        let mut errors = Vec::new();
        if self.dimension != 2 && self.dimension != 3 {
            errors.push(MeshConnectivityError::InvalidMeshDimension {
                dimension: self.dimension,
            });
        }
        for (node_index, point) in self.nodes.iter().enumerate() {
            if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
                errors.push(MeshConnectivityError::NonFiniteNode { node_index });
            }
        }
        for (element_index, element) in self.elements.iter().enumerate() {
            let element_dimension = element.element_type.dimension();
            if element_dimension != self.dimension {
                errors.push(MeshConnectivityError::ElementDimensionMismatch {
                    element_index,
                    element_dimension,
                    mesh_dimension: self.dimension,
                });
            }
            let expected_p1 = element.element_type.num_vertices();
            let expected_p2 = element.element_type.num_nodes_p2();
            let expected_p3 = element.element_type.num_nodes_p3();
            if element.nodes.len() != expected_p1
                && element.nodes.len() != expected_p2
                && element.nodes.len() != expected_p3
            {
                errors.push(MeshConnectivityError::ElementNodeCount {
                    element_index,
                    element_type: element.element_type,
                    actual: element.nodes.len(),
                });
            }
            for (node_position, &node_index) in element.nodes.iter().enumerate() {
                if node_index >= self.nodes.len() {
                    errors.push(MeshConnectivityError::InvalidElementNode {
                        element_index,
                        node_position,
                        node_index,
                    });
                }
                if let Some(first_position) = element.nodes[..node_position]
                    .iter()
                    .position(|&previous| previous == node_index)
                {
                    errors.push(MeshConnectivityError::DuplicateElementNode {
                        element_index,
                        first_position,
                        duplicate_position: node_position,
                        node_index,
                    });
                }
            }
        }
        for (boundary_index, boundary) in self.boundaries.iter().enumerate() {
            for (node_position, &node_index) in boundary.nodes.iter().enumerate() {
                if node_index >= self.nodes.len() {
                    errors.push(MeshConnectivityError::InvalidBoundaryNode {
                        boundary_index,
                        node_position,
                        node_index,
                    });
                }
            }
            let Some(element) = self.elements.get(boundary.element_idx) else {
                errors.push(MeshConnectivityError::InvalidBoundaryElement {
                    boundary_index,
                    element_index: boundary.element_idx,
                });
                continue;
            };
            if element.element_type.dimension() != self.dimension
                || (element.nodes.len() != element.element_type.num_vertices()
                    && element.nodes.len() != element.element_type.num_nodes_p2())
            {
                continue;
            }
            let local_count = if self.dimension == 3 {
                element.element_type.num_faces()
            } else {
                element.element_type.num_edges()
            };
            if boundary.local_idx >= local_count {
                errors.push(MeshConnectivityError::InvalidBoundaryLocalIndex {
                    boundary_index,
                    element_index: boundary.element_idx,
                    local_index: boundary.local_idx,
                });
                continue;
            }
            let mut expected = element
                .boundary_entities()
                .into_iter()
                .nth(boundary.local_idx)
                .expect("validated local boundary index");
            let mut actual = boundary.nodes.clone();
            expected.sort_unstable();
            actual.sort_unstable();
            if expected != actual {
                errors.push(MeshConnectivityError::BoundaryNodeSetMismatch {
                    boundary_index,
                    element_index: boundary.element_idx,
                    local_index: boundary.local_idx,
                });
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Element-to-element neighbor table. Two elements are neighbors iff they
    /// share a full face (3D) or edge (2D). Built lazily on first access and
    /// cached on `self.element_neighbors`. Re-uses the same canonical key the
    /// partition graph builder uses, so the two views are identical.
    pub fn element_neighbors(&mut self) -> &[Vec<usize>] {
        let fingerprint = self.cache_fingerprint();
        if self.element_neighbors_fingerprint != Some(fingerprint) {
            self.element_neighbors = None;
        }
        if self.element_neighbors.is_none() {
            let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); self.elements.len()];
            let mut entity_owners: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
            for (elem_idx, elem) in self.elements.iter().enumerate() {
                let entities: Vec<Vec<usize>> = if elem.element_type.dimension() == 3 {
                    elem.faces()
                } else {
                    elem.edges().into_iter().map(|(a, b)| vec![a, b]).collect()
                };
                for mut key in entities {
                    key.sort_unstable();
                    let owners = entity_owners.entry(key).or_default();
                    // Keep the graph complete even for a non-manifold entity
                    // shared by three or more elements. The manifold checker
                    // reports that geometry separately, but downstream
                    // partition/ghost construction still needs every pair of
                    // incident elements represented.
                    for &other in owners.iter() {
                        adjacency[elem_idx].insert(other);
                        adjacency[other].insert(elem_idx);
                    }
                    owners.push(elem_idx);
                }
            }
            let table: Vec<Vec<usize>> = adjacency
                .into_iter()
                .map(|s| {
                    let mut v: Vec<usize> = s.into_iter().collect();
                    v.sort_unstable();
                    v
                })
                .collect();
            self.element_neighbors = Some(table);
            self.element_neighbors_fingerprint = Some(fingerprint);
        }
        self.element_neighbors.as_deref().unwrap()
    }

    /// Get next available element ID and increment counter
    pub fn next_element_id(&mut self) -> usize {
        let id = self.next_element_id;
        self.next_element_id += 1;
        id
    }

    /// Get or build the mesh topology (edge/face connectivity).
    ///
    /// Built lazily on first access and cached. Invalidated when nodes or elements change.
    ///
    /// Panics on malformed public vectors; prefer [`Self::try_topology`] for
    /// deserialized or externally mutated meshes.
    pub fn topology(&mut self) -> &super::topology::MeshTopology {
        let fingerprint = self.cache_fingerprint();
        if self.topology_fingerprint != Some(fingerprint) {
            self.topology = None;
        }
        if self.topology.is_none() {
            self.topology = Some(super::topology::MeshTopology::build(self));
            self.topology_fingerprint = Some(fingerprint);
        }
        self.topology.as_ref().unwrap()
    }

    /// Fallible topology access for deserialized or externally mutated meshes.
    ///
    /// Unlike [`Self::topology`], this entry point validates connectivity before
    /// any edge/face indexing and returns every structural diagnostic without
    /// panicking.
    pub fn try_topology(
        &mut self,
    ) -> Result<&super::topology::MeshTopology, Vec<MeshConnectivityError>> {
        self.validate_connectivity()?;
        let fingerprint = self.cache_fingerprint();
        if self.topology_fingerprint != Some(fingerprint) {
            self.topology = None;
        }
        if self.topology.is_none() {
            let topology = super::topology::MeshTopology::try_build(self)?;
            self.topology = Some(topology);
            self.topology_fingerprint = Some(fingerprint);
        }
        Ok(self.topology.as_ref().expect("topology inserted above"))
    }

    /// Add a boundary face/edge
    pub fn add_boundary(
        &mut self,
        nodes: Vec<usize>,
        boundary_type: BoundaryType,
        marker: i32,
        element_idx: usize,
        local_idx: usize,
    ) {
        self.boundaries.push(BoundaryFace {
            nodes,
            boundary_type,
            marker,
            element_idx,
            local_idx,
        });
        self.invalidate_caches();
    }

    /// Number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of elements
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Get node coordinates
    pub fn node(&self, idx: usize) -> &Point {
        &self.nodes[idx]
    }

    /// Get element
    pub fn element(&self, idx: usize) -> &Element {
        &self.elements[idx]
    }

    /// Build node-to-element connectivity
    pub fn build_connectivity(&mut self) {
        let mut node_to_elems: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (elem_idx, elem) in self.elements.iter().enumerate() {
            for &node_idx in &elem.nodes {
                node_to_elems[node_idx].push(elem_idx);
            }
        }
        self.node_to_elements = Some(node_to_elems);
        self.node_to_elements_fingerprint = Some(self.cache_fingerprint());
    }

    /// Get elements containing a node
    pub fn elements_containing_node(&self, node_idx: usize) -> Option<&[usize]> {
        if self.node_to_elements_fingerprint != Some(self.cache_fingerprint()) {
            return None;
        }
        self.node_to_elements
            .as_ref()
            .and_then(|c| c.get(node_idx).map(Vec::as_slice))
    }

    /// Find boundary edges/faces automatically.
    ///
    /// Boundary faces appear in exactly one element. The stored
    /// `BoundaryFace.nodes` preserves the **traversal order** of the parent
    /// element's local face, not a sorted order — downstream code (face
    /// normals, P2 promotion, edge enumeration) depends on this.
    ///
    /// On invalid connectivity the boundaries are left cleared and the event
    /// is logged; use [`Self::try_detect_boundaries`] when the caller must
    /// distinguish "no boundaries" from "invalid mesh".
    pub fn detect_boundaries(&mut self) {
        if let Err(errors) = self.try_detect_boundaries() {
            log::warn!(
                "detect_boundaries: invalid mesh ({} error(s)); boundaries left cleared",
                errors.len()
            );
        }
    }

    /// Fallible boundary detection for deserialized or externally mutated meshes.
    ///
    /// Returns the number of boundary faces found. On invalid connectivity the
    /// mesh boundaries are left cleared and every structural diagnostic is
    /// returned, distinguishing "no boundaries" (`Ok(0)`) from "invalid mesh"
    /// (`Err`); [`Self::detect_boundaries`] keeps the legacy unit signature
    /// and logs in that case.
    pub fn try_detect_boundaries(&mut self) -> Result<usize, Vec<MeshConnectivityError>> {
        self.invalidate_caches();
        self.boundaries.clear();
        // Refuse malformed public vectors rather than letting
        // `element_faces` dereference an invalid node or assume an unsupported
        // element arity and panic.
        self.validate_connectivity()?;

        // Per-element face enumeration is independent; the machine-adaptive
        // gate decides whether pool dispatch pays off on this machine.
        let policy = ParallelPolicy::from_env();
        let parallel = policy.decide(self.elements.len(), TOPOLOGY_ITEMS_PER_THREAD);
        let per_elem = self.boundary_batches(parallel);

        // Merge histograms: one shared HashMap cannot be updated from
        // multiple threads without locking, so the parallel path counts
        // per-shard histograms concurrently and merges them in shard order.
        // Both paths produce identical counts and first-occurrence data.
        let face_count = Self::merge_histograms(per_elem, &policy, parallel);

        let mut boundary_candidates = face_count
            .into_iter()
            .filter_map(|(_, (face_nodes, elem_idx, local_idx, count))| {
                (count == 1).then_some((face_nodes, elem_idx, local_idx))
            })
            .collect::<Vec<_>>();
        // HashMap iteration order is intentionally randomized. Keep boundary
        // numbering stable across runs/platforms so marker arrays, serialized
        // meshes, and downstream QA artifacts remain reproducible.
        boundary_candidates.sort_by(
            |(left_nodes, left_elem, left_local), (right_nodes, right_elem, right_local)| {
                left_nodes
                    .cmp(right_nodes)
                    .then(left_elem.cmp(right_elem))
                    .then(left_local.cmp(right_local))
            },
        );
        let count = boundary_candidates.len();
        self.boundaries.extend(boundary_candidates.into_iter().map(
            |(nodes, element_idx, local_idx)| {
                BoundaryFace {
                    nodes,
                    boundary_type: BoundaryType::Neumann, // Default
                    marker: 0,
                    element_idx,
                    local_idx,
                }
            },
        ));
        Ok(count)
    }

    /// Per-element face batches in element order, each entry holding the
    /// sorted key, the traversal-ordered face, and its provenance.
    fn boundary_batches(&self, parallel: bool) -> Vec<Vec<FaceEntry>> {
        fn batch(mesh: &Mesh, elem_idx: usize, elem: &Element) -> Vec<FaceEntry> {
            mesh.element_faces(elem)
                .into_iter()
                .enumerate()
                .map(|(local_idx, face)| {
                    let mut sorted = face.clone();
                    sorted.sort_unstable();
                    (sorted, face, elem_idx, local_idx)
                })
                .collect()
        }
        if parallel {
            self.elements
                .par_iter()
                .enumerate()
                .map(|(elem_idx, elem)| batch(self, elem_idx, elem))
                .collect()
        } else {
            self.elements
                .iter()
                .enumerate()
                .map(|(elem_idx, elem)| batch(self, elem_idx, elem))
                .collect()
        }
    }

    /// Merge per-element face batches into a sorted-key histogram.
    ///
    /// `parallel` selects the sharded merge: contiguous batch ranges are
    /// counted into per-shard maps concurrently, then merged in shard order
    /// so first-occurrence data and counts are identical to the serial
    /// merge. Exposed for parity tests; prefer [`Self::try_detect_boundaries`].
    fn merge_histograms(
        per_elem: Vec<Vec<FaceEntry>>,
        policy: &ParallelPolicy,
        parallel: bool,
    ) -> HashMap<Vec<usize>, (Vec<usize>, usize, usize, usize)> {
        type Histogram = HashMap<Vec<usize>, (Vec<usize>, usize, usize, usize)>;
        fn bump(
            face_count: &mut Histogram,
            sorted: Vec<usize>,
            face: Vec<usize>,
            elem_idx: usize,
            local_idx: usize,
        ) {
            face_count
                .entry(sorted)
                .and_modify(|entry| entry.3 += 1)
                .or_insert((face, elem_idx, local_idx, 1));
        }
        if !parallel {
            let mut face_count: Histogram = HashMap::new();
            for batch in per_elem {
                for (sorted, face, elem_idx, local_idx) in batch {
                    bump(&mut face_count, sorted, face, elem_idx, local_idx);
                }
            }
            return face_count;
        }
        let shards = policy
            .shard_count(per_elem.len(), TOPOLOGY_ITEMS_PER_THREAD)
            .max(2);
        let chunk = per_elem.len().div_ceil(shards);
        let mut per_elem = per_elem;
        let shard_maps: Vec<Histogram> = per_elem
            .par_chunks_mut(chunk)
            .map(|batches| {
                let mut local: Histogram = HashMap::new();
                for batch in batches.iter_mut() {
                    for (sorted, face, elem_idx, local_idx) in batch.drain(..) {
                        bump(&mut local, sorted, face, elem_idx, local_idx);
                    }
                }
                local
            })
            .collect();
        let mut face_count: Histogram = HashMap::new();
        for mut local in shard_maps {
            for (sorted, (face, elem_idx, local_idx, count)) in local.drain() {
                match face_count.entry(sorted) {
                    std::collections::hash_map::Entry::Occupied(mut slot) => {
                        slot.get_mut().3 += count;
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert((face, elem_idx, local_idx, count));
                    }
                }
            }
        }
        face_count
    }

    /// Get faces/edges of an element (for boundary detection)
    fn element_faces(&self, elem: &Element) -> Vec<Vec<usize>> {
        let verts = elem.vertices();
        match elem.element_type {
            ElementType::Triangle => vec![
                vec![verts[0], verts[1]],
                vec![verts[1], verts[2]],
                vec![verts[2], verts[0]],
            ],
            ElementType::Quadrilateral => vec![
                vec![verts[0], verts[1]],
                vec![verts[1], verts[2]],
                vec![verts[2], verts[3]],
                vec![verts[3], verts[0]],
            ],
            ElementType::Tetrahedron => vec![
                vec![verts[0], verts[1], verts[2]],
                vec![verts[0], verts[1], verts[3]],
                vec![verts[0], verts[2], verts[3]],
                vec![verts[1], verts[2], verts[3]],
            ],
            ElementType::Hexahedron => vec![
                vec![verts[0], verts[1], verts[2], verts[3]], // Bottom
                vec![verts[4], verts[5], verts[6], verts[7]], // Top
                vec![verts[0], verts[1], verts[5], verts[4]], // Front
                vec![verts[2], verts[3], verts[7], verts[6]], // Back
                vec![verts[0], verts[3], verts[7], verts[4]], // Left
                vec![verts[1], verts[2], verts[6], verts[5]], // Right
            ],
        }
    }

    /// Set boundary condition on faces matching a predicate
    pub fn set_boundary_condition<F>(
        &mut self,
        boundary_type: BoundaryType,
        marker: i32,
        predicate: F,
    ) where
        F: Fn(&[Point]) -> bool,
    {
        for boundary in &mut self.boundaries {
            let face_points: Vec<Point> = boundary.nodes.iter().map(|&i| self.nodes[i]).collect();
            if predicate(&face_points) {
                boundary.boundary_type = boundary_type;
                boundary.marker = marker;
            }
        }
        self.invalidate_caches();
    }

    /// Get all boundary nodes with a specific type
    pub fn boundary_nodes(&self, boundary_type: BoundaryType) -> HashSet<usize> {
        let mut nodes = HashSet::new();
        for boundary in &self.boundaries {
            if boundary.boundary_type == boundary_type {
                nodes.extend(boundary.nodes.iter().cloned());
            }
        }
        nodes
    }

    /// Compute element centroid
    pub fn element_centroid(&self, elem_idx: usize) -> Point {
        let elem = &self.elements[elem_idx];
        let verts = elem.vertices();
        let n = verts.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &v in verts {
            cx += self.nodes[v].x;
            cy += self.nodes[v].y;
            cz += self.nodes[v].z;
        }
        Point::new_3d(cx / n, cy / n, cz / n)
    }

    /// Compute element area (2D) or volume (3D)
    pub fn element_measure(&self, elem_idx: usize) -> f64 {
        let elem = &self.elements[elem_idx];
        let verts: Vec<&Point> = elem.vertices().iter().map(|&i| &self.nodes[i]).collect();

        match elem.element_type {
            ElementType::Triangle => {
                // Area = 0.5 * |det([x2-x1, y2-y1; x3-x1, y3-y1])|
                let v1 = (verts[1].x - verts[0].x, verts[1].y - verts[0].y);
                let v2 = (verts[2].x - verts[0].x, verts[2].y - verts[0].y);
                0.5 * (v1.0 * v2.1 - v1.1 * v2.0).abs()
            }
            ElementType::Quadrilateral => {
                // Split into two triangles
                let a1 = {
                    let v1 = (verts[1].x - verts[0].x, verts[1].y - verts[0].y);
                    let v2 = (verts[2].x - verts[0].x, verts[2].y - verts[0].y);
                    0.5 * (v1.0 * v2.1 - v1.1 * v2.0).abs()
                };
                let a2 = {
                    let v1 = (verts[2].x - verts[0].x, verts[2].y - verts[0].y);
                    let v2 = (verts[3].x - verts[0].x, verts[3].y - verts[0].y);
                    0.5 * (v1.0 * v2.1 - v1.1 * v2.0).abs()
                };
                a1 + a2
            }
            ElementType::Tetrahedron => {
                // Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
                let v1 = (
                    verts[1].x - verts[0].x,
                    verts[1].y - verts[0].y,
                    verts[1].z - verts[0].z,
                );
                let v2 = (
                    verts[2].x - verts[0].x,
                    verts[2].y - verts[0].y,
                    verts[2].z - verts[0].z,
                );
                let v3 = (
                    verts[3].x - verts[0].x,
                    verts[3].y - verts[0].y,
                    verts[3].z - verts[0].z,
                );
                let det = v1.0 * (v2.1 * v3.2 - v2.2 * v3.1) - v1.1 * (v2.0 * v3.2 - v2.2 * v3.0)
                    + v1.2 * (v2.0 * v3.1 - v2.1 * v3.0);
                det.abs() / 6.0
            }
            ElementType::Hexahedron => {
                // Exact volume via the 6-tet decomposition rooted at vertex 0.
                // Vertices follow the canonical sonium order (bottom 0..3
                // CCW from below, top 4..7 above the matching bottom corners).
                // The six tets share edge (0-6) — the space diagonal — and
                // tile the cube without gaps or overlap, so the sum of their
                // signed volumes is the hex volume regardless of distortion.
                let tets: [[usize; 4]; 6] = [
                    [0, 1, 2, 6],
                    [0, 2, 3, 6],
                    [0, 3, 7, 6],
                    [0, 7, 4, 6],
                    [0, 4, 5, 6],
                    [0, 5, 1, 6],
                ];
                let mut volume = 0.0;
                for tet in tets {
                    let p0 = verts[tet[0]];
                    let p1 = verts[tet[1]];
                    let p2 = verts[tet[2]];
                    let p3 = verts[tet[3]];
                    let v1 = (p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
                    let v2 = (p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
                    let v3 = (p3.x - p0.x, p3.y - p0.y, p3.z - p0.z);
                    let det = v1.0 * (v2.1 * v3.2 - v2.2 * v3.1)
                        - v1.1 * (v2.0 * v3.2 - v2.2 * v3.0)
                        + v1.2 * (v2.0 * v3.1 - v2.1 * v3.0);
                    volume += det.abs() / 6.0;
                }
                volume
            }
        }
    }
}

// VTU export: implement sonium_xem's MeshView for our Mesh so binaries can
// pass `&mesh` directly to `sonium_xem::io::vtu::VtuWriter::new`.
impl sonium_xem::io::vtu::MeshView for Mesh {
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn num_cells(&self) -> usize {
        self.elements.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.nodes.iter().map(|p| [p.x, p.y, p.z]))
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (sonium_xem::io::vtu::CellKind, Vec<u64>)> + '_> {
        use sonium_xem::io::vtu::CellKind;
        Box::new(self.elements.iter().map(|e| {
            let kind = match e.element_type {
                ElementType::Triangle => CellKind::Triangle,
                ElementType::Quadrilateral => CellKind::Quad,
                ElementType::Tetrahedron => CellKind::Tetra,
                ElementType::Hexahedron => CellKind::Hexahedron,
            };
            let verts: Vec<u64> = e.vertices().iter().map(|&i| i as u64).collect();
            (kind, verts)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point::new_2d(0.0, 0.0);
        let p2 = Point::new_2d(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_midpoint() {
        let p1 = Point::new_2d(0.0, 0.0);
        let p2 = Point::new_2d(2.0, 4.0);
        let mid = p1.midpoint(&p2);
        assert!((mid.x - 1.0).abs() < 1e-10);
        assert!((mid.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_element_type_properties() {
        assert_eq!(ElementType::Triangle.num_vertices(), 3);
        assert_eq!(ElementType::Triangle.num_nodes_p2(), 6);
        assert_eq!(ElementType::Tetrahedron.dimension(), 3);
    }

    #[test]
    fn test_mesh_triangle_area() {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, 1.0));
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);

        let area = mesh.element_measure(0);
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn connectivity_gate_accepts_valid_element_and_boundary() {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, 1.0));
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
        mesh.add_boundary(vec![0, 1], BoundaryType::Neumann, 7, 0, 0);

        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn connectivity_gate_reports_malformed_external_mesh_without_panicking() {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, f64::NAN));
        mesh.add_element(ElementType::Triangle, vec![0, 1, 99]);
        mesh.add_boundary(vec![0, 99], BoundaryType::Neumann, 0, 0, 9);

        let errors = mesh
            .validate_connectivity()
            .expect_err("malformed mesh must fail the connectivity gate");
        assert!(errors.iter().any(|error| matches!(
            error,
            MeshConnectivityError::NonFiniteNode { node_index: 2 }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MeshConnectivityError::InvalidElementNode {
                element_index: 0,
                node_position: 2,
                node_index: 99
            }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MeshConnectivityError::InvalidBoundaryNode {
                boundary_index: 0,
                node_position: 1,
                node_index: 99
            }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MeshConnectivityError::InvalidBoundaryLocalIndex {
                boundary_index: 0,
                element_index: 0,
                local_index: 9
            }
        )));
    }

    #[test]
    fn connectivity_gate_rejects_repeated_element_nodes_before_topology_building() {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, 1.0));
        mesh.add_element(ElementType::Triangle, vec![0, 1, 1]);

        let errors = mesh
            .validate_connectivity()
            .expect_err("repeated element vertices must be rejected");
        assert!(errors.iter().any(|error| matches!(
            error,
            MeshConnectivityError::DuplicateElementNode {
                element_index: 0,
                first_position: 1,
                duplicate_position: 2,
                node_index: 1,
            }
        )));
        assert!(mesh.try_topology().is_err());
    }

    #[test]
    fn cell_markers_and_dimension_local_physical_names_are_first_class() {
        let mut mesh = Mesh::new(3);
        mesh.add_element_with_marker(ElementType::Tetrahedron, vec![0, 1, 2, 3], 11);
        mesh.add_element_with_marker(ElementType::Tetrahedron, vec![1, 2, 3, 4], 12);
        mesh.register_physical_group(3, 11, "steel").unwrap();
        mesh.register_physical_group(2, 11, "steel").unwrap();

        assert_eq!(mesh.cells_with_marker(11).collect::<Vec<_>>(), vec![0]);
        assert_eq!(mesh.physical_groups.by_name(3, "steel").unwrap().tag, 11);
        assert_eq!(mesh.physical_groups.by_name(2, "steel").unwrap().tag, 11);
        assert!(mesh.register_physical_group(3, 12, "steel").is_err());
        assert!(mesh.register_physical_group(4, 13, "invalid").is_err());
        mesh.validate_physical_groups().unwrap();
    }

    #[test]
    fn mesh_metadata_serialization_is_backward_compatible() {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, 1.0));
        mesh.add_element_with_marker(ElementType::Triangle, vec![0, 1, 2], 7);
        mesh.register_physical_group(2, 7, "fluid").unwrap();

        let encoded = serde_json::to_string(&mesh).unwrap();
        let decoded: Mesh = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.elements[0].marker, 7);
        assert_eq!(decoded.physical_groups.by_tag(2, 7).unwrap().name, "fluid");

        let mut legacy = serde_json::to_value(&mesh).unwrap();
        legacy.as_object_mut().unwrap().remove("physical_groups");
        legacy["elements"][0]
            .as_object_mut()
            .unwrap()
            .remove("marker");
        let decoded_legacy: Mesh = serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded_legacy.elements[0].marker, 0);
        assert!(decoded_legacy.physical_groups.is_empty());
    }

    #[test]
    fn mesh_serde_admission_rejects_malformed_connectivity() {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, 1.0));
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);

        let mut malformed = serde_json::to_value(&mesh).expect("serialize valid mesh");
        malformed["elements"][0]["nodes"][2] = serde_json::json!(99);
        let error = serde_json::from_value::<Mesh>(malformed)
            .expect_err("deserialization must validate element node indexes");
        assert!(error.to_string().contains("node"));

        let mut duplicate_node = serde_json::to_value(&mesh).expect("serialize valid mesh");
        duplicate_node["elements"][0]["nodes"][2] = serde_json::json!(1);
        let error = serde_json::from_value::<Mesh>(duplicate_node)
            .expect_err("deserialization must reject repeated element nodes");
        assert!(error.to_string().contains("repeats node"));

        let mut bad_dimension = serde_json::to_value(&mesh).expect("serialize valid mesh");
        bad_dimension["dimension"] = serde_json::json!(4);
        let error = serde_json::from_value::<Mesh>(bad_dimension)
            .expect_err("deserialization must validate mesh dimensions");
        assert!(error.to_string().contains("dimension"));
    }

    #[test]
    fn direct_element_mutation_cannot_reuse_stale_connectivity_caches() {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        mesh.build_connectivity();
        assert_eq!(mesh.elements_containing_node(0), Some(&[0][..]));

        // Bypass the convenience method deliberately: public mesh vectors are
        // part of the serialization/interoperability surface.
        mesh.elements[0].nodes.swap(0, 1);
        assert_eq!(mesh.elements_containing_node(0), None);
        mesh.build_connectivity();
        assert_eq!(mesh.elements_containing_node(0), Some(&[0][..]));
    }

    #[test]
    fn direct_element_append_refreshes_neighbor_and_topology_caches() {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
            Point::new_3d(0.0, 0.0, -1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        let _ = mesh.element_neighbors().to_vec();
        let _ = mesh.topology().element_faces.len();

        let second = Element::new(ElementType::Tetrahedron, vec![0, 2, 1, 4], 1);
        mesh.elements.push(second);
        assert_eq!(mesh.element_neighbors().len(), 2);
        assert_eq!(mesh.topology().element_faces[1].len(), 4);
    }

    #[test]
    fn boundary_detection_rejects_malformed_public_vectors_without_panicking() {
        let mut mesh = Mesh::new(3);
        mesh.nodes.push(Point::new_3d(0.0, 0.0, 0.0));
        mesh.nodes.push(Point::new_3d(1.0, 0.0, 0.0));
        mesh.nodes.push(Point::new_3d(0.0, 1.0, 0.0));
        mesh.elements
            .push(Element::new(ElementType::Tetrahedron, vec![0, 1, 99, 0], 0));

        mesh.detect_boundaries();

        assert!(mesh.boundaries.is_empty());
        assert!(mesh.validate_connectivity().is_err());
    }

    #[test]
    fn try_detect_boundaries_distinguishes_invalid_mesh_from_no_boundaries() {
        let mut mesh = Mesh::new(3);
        mesh.nodes.push(Point::new_3d(0.0, 0.0, 0.0));
        mesh.nodes.push(Point::new_3d(1.0, 0.0, 0.0));
        mesh.nodes.push(Point::new_3d(0.0, 1.0, 0.0));
        mesh.elements
            .push(Element::new(ElementType::Tetrahedron, vec![0, 1, 99, 0], 0));
        assert!(mesh.try_detect_boundaries().is_err());

        let mut tet = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ] {
            tet.add_node(point);
        }
        tet.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        assert_eq!(tet.try_detect_boundaries(), Ok(4));
    }

    #[test]
    fn direct_boundary_marker_mutation_refreshes_topology_cache() {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        mesh.detect_boundaries();
        assert!(!mesh.topology().boundary_faces(0).is_empty());
        let before = mesh.cache_fingerprint();

        for boundary in &mut mesh.boundaries {
            boundary.marker = 42;
        }
        assert_ne!(before, mesh.cache_fingerprint());

        assert!(mesh.topology().boundary_faces(0).is_empty());
        assert!(!mesh.topology().boundary_faces(42).is_empty());
    }

    #[test]
    fn sharded_histogram_merge_matches_serial_merge() {
        let mesh = crate::mesh::generators::rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 24, 24);
        assert!(mesh.validate_connectivity().is_ok());
        let batches = mesh.boundary_batches(false);
        let policy = ParallelPolicy::with_threads(4);
        let serial = Mesh::merge_histograms(batches.clone(), &policy, false);
        let sharded = Mesh::merge_histograms(batches, &policy, true);
        assert_eq!(serial.len(), sharded.len());
        for (key, value) in &serial {
            assert_eq!(sharded.get(key), Some(value), "key {key:?} differs");
        }
    }

    #[test]
    fn checked_element_order_rejects_malformed_external_connectivity() {
        let element = Element::new(ElementType::Triangle, vec![0, 1], 0);
        assert_eq!(
            element.try_polynomial_order(),
            Err(ElementOrderError::InvalidNodeCount {
                element_type: ElementType::Triangle,
                actual: 2,
                p1: 3,
                p2: 6,
                p3: 10,
            })
        );
    }
}
