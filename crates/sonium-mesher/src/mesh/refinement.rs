//! Mesh refinement (h-refinement)
//!
//! Implements bisection-based refinement for triangles and tetrahedra.

use super::validate::check_orientation;
use super::{BoundaryFace, Element, ElementType, Mesh, Point};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Invalid input supplied to adaptive mesh refinement.
#[derive(Debug, Clone, PartialEq)]
pub enum RefinementError {
    /// The mesh connectivity or coordinates are invalid for refinement.
    InvalidMesh {
        element: usize,
        reason: &'static str,
    },
    /// A mesh node contains a non-finite coordinate.
    InvalidNode { node: usize, reason: &'static str },
    /// The estimator supplied a different number of errors than mesh elements.
    ErrorCountMismatch { elements: usize, errors: usize },
    /// The refinement threshold was negative or non-finite.
    InvalidThreshold(f64),
    /// An element error was negative or non-finite.
    InvalidElementError { index: usize, value: f64 },
    /// A coarsening history cannot be applied to the current mesh.
    InvalidHistory {
        parent_id: usize,
        reason: &'static str,
    },
    /// No parent families were supplied for coarsening.
    EmptyHistory,
}

impl fmt::Display for RefinementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMesh { element, reason } => {
                write!(f, "invalid mesh element {element} for refinement: {reason}")
            }
            Self::InvalidNode { node, reason } => {
                write!(f, "invalid mesh node {node} for refinement: {reason}")
            }
            Self::ErrorCountMismatch { elements, errors } => write!(
                f,
                "element error count {errors} does not match mesh element count {elements}"
            ),
            Self::InvalidThreshold(value) => {
                write!(
                    f,
                    "refinement threshold must be finite and non-negative, got {value}"
                )
            }
            Self::InvalidElementError { index, value } => write!(
                f,
                "element error at index {index} must be finite and non-negative, got {value}"
            ),
            Self::InvalidHistory { parent_id, reason } => {
                write!(
                    f,
                    "invalid coarsening history for parent {parent_id}: {reason}"
                )
            }
            Self::EmptyHistory => write!(f, "coarsening history must contain at least one parent"),
        }
    }
}

impl std::error::Error for RefinementError {}

/// Edge represented by sorted node indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge(pub usize, pub usize);

impl Edge {
    pub fn new(a: usize, b: usize) -> Self {
        if a < b { Edge(a, b) } else { Edge(b, a) }
    }
}

/// Refinement result
pub struct RefinementResult {
    /// Indices of new elements created
    pub new_elements: Vec<usize>,
    /// Indices of elements removed
    pub removed_elements: Vec<usize>,
    /// Indices of new nodes created
    pub new_nodes: Vec<usize>,
}

/// A snapshot of one parent element and its pre-refinement index.
///
/// The index is retained solely to restore boundary records. Element IDs are
/// the identity used when matching children, so histories remain valid even
/// when unrelated element ordering changes during refinement.
#[derive(Debug, Clone)]
pub struct ParentRefinement {
    /// Index of the parent in the mesh immediately before refinement.
    pub original_index: usize,
    /// Parent element to restore during coarsening.
    pub element: Element,
}

/// Reversible provenance for checked first-, second-, or third-order
/// refinement.
///
/// A history is intentionally explicit rather than inferred from child
/// geometry: midpoint/centre nodes are not enough to distinguish a parent
/// corner from an interior node in every element family. The history can be
/// persisted by an adaptation driver and passed to
/// [`coarsen_elements_checked`] after the corresponding refined mesh has been
/// accepted.
#[derive(Debug, Clone)]
pub struct RefinementHistory {
    /// Parent snapshots in canonical element-index order.
    pub parents: Vec<ParentRefinement>,
    /// Boundary records belonging to those parents before refinement.
    pub boundaries: Vec<BoundaryFace>,
}

/// Summary of a successful checked coarsening operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoarseningResult {
    /// Element indices occupied by the restored parent elements.
    pub restored_elements: Vec<usize>,
    /// Element indices removed from the refined mesh before compaction.
    pub removed_elements: Vec<usize>,
}

/// Helper struct to manage midpoint creation
struct MidpointManager {
    edge_midpoints: HashMap<Edge, usize>,
    face_centers: HashMap<Vec<usize>, usize>,
    body_centers: HashMap<Vec<usize>, usize>,
    /// New Q2 points on a parent Hex27 face. The key is the sparse
    /// quadratic Lagrange stencil over that face's nine trace nodes, rather
    /// than a local face orientation, so neighbouring hexes reuse the same
    /// node even when their local face orderings differ.
    quadratic_face_points: HashMap<Vec<(usize, i16)>, usize>,
    new_nodes: Vec<usize>,
}

impl MidpointManager {
    fn new() -> Self {
        Self {
            edge_midpoints: HashMap::new(),
            face_centers: HashMap::new(),
            body_centers: HashMap::new(),
            quadratic_face_points: HashMap::new(),
            new_nodes: Vec::new(),
        }
    }

    fn get_midpoint(&mut self, mesh: &mut Mesh, a: usize, b: usize) -> usize {
        let edge = Edge::new(a, b);
        if let Some(&mid_idx) = self.edge_midpoints.get(&edge) {
            mid_idx
        } else {
            let mid = mesh.nodes[a].midpoint(&mesh.nodes[b]);
            let idx = mesh.add_node(mid);
            self.new_nodes.push(idx);
            self.edge_midpoints.insert(edge, idx);
            idx
        }
    }

    /// Insert a point on a quadratic parent trace, keyed by the two existing
    /// trace nodes that bracket it. Unlike [`Self::get_midpoint`], `point` is
    /// evaluated from the parent isoparametric map, so curved Q2 edges retain
    /// their geometry after h-refinement.
    fn get_mapped_edge_point(
        &mut self,
        mesh: &mut Mesh,
        a: usize,
        b: usize,
        point: Point,
    ) -> usize {
        let edge = Edge::new(a, b);
        if let Some(&index) = self.edge_midpoints.get(&edge) {
            return index;
        }
        let index = mesh.add_node(point);
        self.new_nodes.push(index);
        self.edge_midpoints.insert(edge, index);
        index
    }

    /// Reuse a new node that lies on one or more parent Hex27 faces.
    ///
    /// A point on an edge or vertex supplies one key for every incident
    /// parent face. Registering all aliases makes it share correctly with
    /// cells reached from either face without relying on physical-coordinate
    /// comparisons.
    fn get_mapped_face_point(
        &mut self,
        mesh: &mut Mesh,
        keys: Vec<Vec<(usize, i16)>>,
        point: Point,
    ) -> usize {
        if let Some(index) = keys
            .iter()
            .find_map(|key| self.quadratic_face_points.get(key).copied())
        {
            for key in keys {
                if let Some(previous) = self.quadratic_face_points.insert(key, index) {
                    debug_assert_eq!(previous, index);
                }
            }
            return index;
        }

        let index = mesh.add_node(point);
        self.new_nodes.push(index);
        for key in keys {
            if let Some(previous) = self.quadratic_face_points.insert(key, index) {
                debug_assert_eq!(previous, index);
            }
        }
        index
    }

    /// Bilinear centre of a quad face given by four node indices (in any
    /// order — keyed by their sorted form to dedupe across neighbour hexes).
    fn get_face_center(&mut self, mesh: &mut Mesh, nodes: [usize; 4]) -> usize {
        let mut key = nodes.to_vec();
        key.sort_unstable();
        if let Some(&idx) = self.face_centers.get(&key) {
            return idx;
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &n in &nodes {
            let p = mesh.nodes[n];
            cx += p.x;
            cy += p.y;
            cz += p.z;
        }
        let centre = Point::new_3d(cx * 0.25, cy * 0.25, cz * 0.25);
        let idx = mesh.add_node(centre);
        self.new_nodes.push(idx);
        self.face_centers.insert(key, idx);
        idx
    }

    /// Centroid of a hex (mean of the 8 corner nodes). Cached by the sorted
    /// 8-tuple so two adjacent hex parents could in principle reuse the same
    /// body node — in practice each hex has a unique interior centre.
    fn get_body_center(&mut self, mesh: &mut Mesh, nodes: [usize; 8]) -> usize {
        let mut key = nodes.to_vec();
        key.sort_unstable();
        if let Some(&idx) = self.body_centers.get(&key) {
            return idx;
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &n in &nodes {
            let p = mesh.nodes[n];
            cx += p.x;
            cy += p.y;
            cz += p.z;
        }
        let inv = 1.0 / 8.0;
        let centre = Point::new_3d(cx * inv, cy * inv, cz * inv);
        let idx = mesh.add_node(centre);
        self.new_nodes.push(idx);
        self.body_centers.insert(key, idx);
        idx
    }
}

fn orient_tetrahedron_positive(mesh: &Mesh, nodes: &mut [usize; 4]) {
    let [a, b, c, d] = nodes.map(|node| mesh.nodes[node]);
    let determinant = (b.x - a.x) * ((c.y - a.y) * (d.z - a.z) - (c.z - a.z) * (d.y - a.y))
        - (b.y - a.y) * ((c.x - a.x) * (d.z - a.z) - (c.z - a.z) * (d.x - a.x))
        + (b.z - a.z) * ((c.x - a.x) * (d.y - a.y) - (c.y - a.y) * (d.x - a.x));
    if determinant < 0.0 {
        nodes.swap(0, 1);
    }
}

/// Refine selected elements using bisection
///
/// For triangles: Uses longest-edge bisection (red-green refinement)
/// For tetrahedra: Uses edge bisection with closure
pub fn refine_elements(mesh: &mut Mesh, elements_to_refine: &[usize]) -> RefinementResult {
    let mut removed_elements = Vec::new();
    let mut refined_parent_ids = HashMap::new();
    let mut child_elements = Vec::new();
    let mut midpoint_mgr = MidpointManager::new();
    let original_boundaries = mesh.boundaries.clone();

    for &elem_idx in elements_to_refine {
        if elem_idx >= mesh.elements.len() || removed_elements.contains(&elem_idx) {
            continue;
        }

        let elem = mesh.elements[elem_idx].clone();
        let child_start = child_elements.len();

        match elem.element_type {
            ElementType::Triangle => {
                // Red refinement: divide into 4 triangles
                let v = elem.vertices();
                let m01 = midpoint_mgr.get_midpoint(mesh, v[0], v[1]);
                let m12 = midpoint_mgr.get_midpoint(mesh, v[1], v[2]);
                let m20 = midpoint_mgr.get_midpoint(mesh, v[2], v[0]);

                // 4 child triangles
                let children = vec![
                    vec![v[0], m01, m20],
                    vec![m01, v[1], m12],
                    vec![m20, m12, v[2]],
                    vec![m01, m12, m20],
                ];

                for nodes in children {
                    let mut child_elem =
                        Element::new(ElementType::Triangle, nodes, mesh.next_element_id());
                    child_elem.parent_id = Some(elem.id);
                    child_elem.level = elem.level + 1;
                    child_elem.marker = elem.marker;
                    child_elements.push(child_elem);
                }
            }
            ElementType::Tetrahedron => {
                // Bey 8-subtet refinement: the four corner tets (one per
                // parent vertex) are fixed; the inner octahedron is
                // triangulated by one of three possible diagonals. Picking
                // the **shortest** diagonal minimises slivers — this is the
                // standard quality fix over a fixed-diagonal split.
                let v = elem.vertices();
                let m01 = midpoint_mgr.get_midpoint(mesh, v[0], v[1]);
                let m02 = midpoint_mgr.get_midpoint(mesh, v[0], v[2]);
                let m03 = midpoint_mgr.get_midpoint(mesh, v[0], v[3]);
                let m12 = midpoint_mgr.get_midpoint(mesh, v[1], v[2]);
                let m13 = midpoint_mgr.get_midpoint(mesh, v[1], v[3]);
                let m23 = midpoint_mgr.get_midpoint(mesh, v[2], v[3]);

                let len_sq = |a: usize, b: usize| {
                    let pa = mesh.nodes[a];
                    let pb = mesh.nodes[b];
                    let dx = pa.x - pb.x;
                    let dy = pa.y - pb.y;
                    let dz = pa.z - pb.z;
                    dx * dx + dy * dy + dz * dz
                };

                let d01_23 = len_sq(m01, m23);
                let d02_13 = len_sq(m02, m13);
                let d03_12 = len_sq(m03, m12);

                let inner: [[usize; 4]; 4] = if d02_13 <= d01_23 && d02_13 <= d03_12 {
                    // Diagonal m02–m13 — four inner tets group by parent
                    // vertex (v0, v1, v3, v2 respectively).
                    [
                        [m01, m02, m03, m13],
                        [m01, m02, m12, m13],
                        [m02, m03, m13, m23],
                        [m02, m12, m13, m23],
                    ]
                } else if d01_23 <= d03_12 {
                    // Diagonal m01–m23
                    [
                        [m01, m23, m02, m03],
                        [m01, m23, m12, m13],
                        [m01, m23, m02, m12],
                        [m01, m23, m03, m13],
                    ]
                } else {
                    // Diagonal m03–m12
                    [
                        [m03, m12, m01, m02],
                        [m03, m12, m01, m13],
                        [m03, m12, m02, m23],
                        [m03, m12, m13, m23],
                    ]
                };

                let corner: [[usize; 4]; 4] = [
                    [v[0], m01, m02, m03],
                    [m01, v[1], m12, m13],
                    [m02, m12, v[2], m23],
                    [m03, m13, m23, v[3]],
                ];

                for tet in corner.iter().chain(inner.iter()) {
                    let mut oriented = *tet;
                    orient_tetrahedron_positive(mesh, &mut oriented);
                    let nodes = oriented.to_vec();
                    let mut child_elem =
                        Element::new(ElementType::Tetrahedron, nodes, mesh.next_element_id());
                    child_elem.parent_id = Some(elem.id);
                    child_elem.level = elem.level + 1;
                    child_elem.marker = elem.marker;
                    child_elements.push(child_elem);
                }
            }
            ElementType::Quadrilateral => {
                // Divide into 4 quadrilaterals
                let v = elem.vertices();
                let m01 = midpoint_mgr.get_midpoint(mesh, v[0], v[1]);
                let m12 = midpoint_mgr.get_midpoint(mesh, v[1], v[2]);
                let m23 = midpoint_mgr.get_midpoint(mesh, v[2], v[3]);
                let m30 = midpoint_mgr.get_midpoint(mesh, v[3], v[0]);

                // Center point
                let cx = (mesh.nodes[v[0]].x
                    + mesh.nodes[v[1]].x
                    + mesh.nodes[v[2]].x
                    + mesh.nodes[v[3]].x)
                    / 4.0;
                let cy = (mesh.nodes[v[0]].y
                    + mesh.nodes[v[1]].y
                    + mesh.nodes[v[2]].y
                    + mesh.nodes[v[3]].y)
                    / 4.0;
                let center = mesh.add_node(Point::new_2d(cx, cy));
                midpoint_mgr.new_nodes.push(center);

                let children = vec![
                    vec![v[0], m01, center, m30],
                    vec![m01, v[1], m12, center],
                    vec![center, m12, v[2], m23],
                    vec![m30, center, m23, v[3]],
                ];

                for nodes in children {
                    let mut child_elem =
                        Element::new(ElementType::Quadrilateral, nodes, mesh.next_element_id());
                    child_elem.parent_id = Some(elem.id);
                    child_elem.level = elem.level + 1;
                    child_elem.marker = elem.marker;
                    child_elements.push(child_elem);
                }
            }
            ElementType::Hexahedron => {
                // Uniform 8-way subdivision of the parent hex. Sonium hex
                // vertex order has the bottom face (z-) at corners 0..3 going
                // CCW from below and the top face (z+) at corners 4..7 with
                // corner (i+4) above corner i. The 8 child hexes each occupy
                // one octant of the parent, anchored at a parent corner; the
                // remaining 7 corners come from edge midpoints (12), face
                // centres (6) and the body centre (1) — 27 nodes total, the
                // same skeleton as the Q2 layout but used here for h-refinement
                // (P1 children) rather than degree elevation.
                let v = elem.vertices();
                let v: [usize; 8] = [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]];

                // Edge midpoints, one per parent edge.
                let m01 = midpoint_mgr.get_midpoint(mesh, v[0], v[1]);
                let m12 = midpoint_mgr.get_midpoint(mesh, v[1], v[2]);
                let m23 = midpoint_mgr.get_midpoint(mesh, v[2], v[3]);
                let m30 = midpoint_mgr.get_midpoint(mesh, v[3], v[0]);
                let m45 = midpoint_mgr.get_midpoint(mesh, v[4], v[5]);
                let m56 = midpoint_mgr.get_midpoint(mesh, v[5], v[6]);
                let m67 = midpoint_mgr.get_midpoint(mesh, v[6], v[7]);
                let m74 = midpoint_mgr.get_midpoint(mesh, v[7], v[4]);
                let m04 = midpoint_mgr.get_midpoint(mesh, v[0], v[4]);
                let m15 = midpoint_mgr.get_midpoint(mesh, v[1], v[5]);
                let m26 = midpoint_mgr.get_midpoint(mesh, v[2], v[6]);
                let m37 = midpoint_mgr.get_midpoint(mesh, v[3], v[7]);

                // Face centres, keyed by sorted vertex tuple so neighbours
                // sharing a face would dedupe (with the current single-parent
                // refinement we just create them fresh).
                let fb = midpoint_mgr.get_face_center(mesh, [v[0], v[1], v[2], v[3]]); // bottom
                let ft = midpoint_mgr.get_face_center(mesh, [v[4], v[5], v[6], v[7]]); // top
                let ff = midpoint_mgr.get_face_center(mesh, [v[0], v[1], v[5], v[4]]); // front
                let fk = midpoint_mgr.get_face_center(mesh, [v[2], v[3], v[7], v[6]]); // back
                let fl = midpoint_mgr.get_face_center(mesh, [v[0], v[3], v[7], v[4]]); // left
                let fr = midpoint_mgr.get_face_center(mesh, [v[1], v[2], v[6], v[5]]); // right

                let bc = midpoint_mgr.get_body_center(mesh, v);

                // Each child is listed in sonium hex order:
                // [bottom CCW from below, top CCW from above each bottom corner].
                let children: [[usize; 8]; 8] = [
                    // Octant at v0 (-,-,-)
                    [v[0], m01, fb, m30, m04, ff, bc, fl],
                    // Octant at v1 (+,-,-)
                    [m01, v[1], m12, fb, ff, m15, fr, bc],
                    // Octant at v2 (+,+,-)
                    [fb, m12, v[2], m23, bc, fr, m26, fk],
                    // Octant at v3 (-,+,-)
                    [m30, fb, m23, v[3], fl, bc, fk, m37],
                    // Octant at v4 (-,-,+)
                    [m04, ff, bc, fl, v[4], m45, ft, m74],
                    // Octant at v5 (+,-,+)
                    [ff, m15, fr, bc, m45, v[5], m56, ft],
                    // Octant at v6 (+,+,+)
                    [bc, fr, m26, fk, ft, m56, v[6], m67],
                    // Octant at v7 (-,+,+)
                    [fl, bc, fk, m37, m74, ft, m67, v[7]],
                ];

                for child_nodes in children {
                    let mut child_elem = Element::new(
                        ElementType::Hexahedron,
                        child_nodes.to_vec(),
                        mesh.next_element_id(),
                    );
                    child_elem.parent_id = Some(elem.id);
                    child_elem.level = elem.level + 1;
                    child_elem.marker = elem.marker;
                    child_elements.push(child_elem);
                }
            }
        }

        if child_elements.len() > child_start {
            removed_elements.push(elem_idx);
            refined_parent_ids.insert(elem_idx, elem.id);
        }
    }

    let removed: HashSet<usize> = removed_elements.iter().copied().collect();
    let mut old_to_new_element = HashMap::new();
    let mut retained =
        Vec::with_capacity(mesh.elements.len() - removed.len() + child_elements.len());
    for (idx, elem) in mesh.elements.drain(..).enumerate() {
        if !removed.contains(&idx) {
            old_to_new_element.insert(idx, retained.len());
            retained.push(elem);
        }
    }

    let first_child_idx = retained.len();
    let new_elements: Vec<usize> =
        (first_child_idx..first_child_idx + child_elements.len()).collect();
    retained.extend(child_elements);
    mesh.elements = retained;
    mesh.boundaries = propagate_boundaries(
        mesh,
        original_boundaries,
        &old_to_new_element,
        &refined_parent_ids,
    );
    mesh.invalidate_caches();

    RefinementResult {
        new_elements,
        removed_elements,
        new_nodes: midpoint_mgr.new_nodes,
    }
}

fn propagate_boundaries(
    mesh: &Mesh,
    original_boundaries: Vec<BoundaryFace>,
    old_to_new_element: &HashMap<usize, usize>,
    refined_parent_ids: &HashMap<usize, usize>,
) -> Vec<BoundaryFace> {
    let mut propagated = Vec::with_capacity(original_boundaries.len());

    for boundary in original_boundaries {
        if let Some(&new_element_idx) = old_to_new_element.get(&boundary.element_idx) {
            let mut remapped = boundary;
            remapped.element_idx = new_element_idx;
            propagated.push(remapped);
            continue;
        }

        let Some(&parent_id) = refined_parent_ids.get(&boundary.element_idx) else {
            continue;
        };

        for (elem_idx, elem) in mesh.elements.iter().enumerate() {
            if elem.parent_id != Some(parent_id) {
                continue;
            }

            for (local_idx, nodes) in boundary_entities(elem).into_iter().enumerate() {
                if entity_lies_on_boundary(mesh, &nodes, &boundary.nodes) {
                    let mut sorted_nodes = nodes;
                    // P1 boundary vectors historically use sorted corner
                    // sets. Preserve canonical trace ordering for P2/Q2 so
                    // high-order consumers retain edge/face DOF orientation.
                    if elem.try_polynomial_order().unwrap_or(1) == 1 {
                        sorted_nodes.sort_unstable();
                    }
                    propagated.push(BoundaryFace {
                        nodes: sorted_nodes,
                        boundary_type: boundary.boundary_type,
                        marker: boundary.marker,
                        element_idx: elem_idx,
                        local_idx,
                    });
                }
            }
        }
    }

    propagated
}

fn boundary_entities(elem: &Element) -> Vec<Vec<usize>> {
    elem.boundary_entities()
}

fn entity_lies_on_boundary(mesh: &Mesh, entity_nodes: &[usize], boundary_nodes: &[usize]) -> bool {
    match boundary_nodes.len() {
        2 => entity_nodes
            .iter()
            .all(|&node_idx| point_on_segment(mesh.nodes[node_idx], boundary_nodes, mesh)),
        3.. => entity_nodes
            .iter()
            .all(|&node_idx| point_on_plane(mesh.nodes[node_idx], boundary_nodes, mesh)),
        _ => false,
    }
}

fn point_on_segment(point: Point, boundary_nodes: &[usize], mesh: &Mesh) -> bool {
    let a = mesh.nodes[boundary_nodes[0]];
    let b = mesh.nodes[boundary_nodes[1]];
    let ab = (b.x - a.x, b.y - a.y, b.z - a.z);
    let ap = (point.x - a.x, point.y - a.y, point.z - a.z);
    let cross = (
        ab.1 * ap.2 - ab.2 * ap.1,
        ab.2 * ap.0 - ab.0 * ap.2,
        ab.0 * ap.1 - ab.1 * ap.0,
    );
    let ab_len_sq = ab.0 * ab.0 + ab.1 * ab.1 + ab.2 * ab.2;
    let tol = 1e-12 * ab_len_sq.max(1.0);
    let cross_len_sq = cross.0 * cross.0 + cross.1 * cross.1 + cross.2 * cross.2;
    if cross_len_sq > tol {
        return false;
    }

    let dot = ap.0 * ab.0 + ap.1 * ab.1 + ap.2 * ab.2;
    dot >= -tol && dot <= ab_len_sq + tol
}

fn point_on_plane(point: Point, boundary_nodes: &[usize], mesh: &Mesh) -> bool {
    let a = mesh.nodes[boundary_nodes[0]];
    let b = mesh.nodes[boundary_nodes[1]];
    let c = mesh.nodes[boundary_nodes[2]];
    let ab = (b.x - a.x, b.y - a.y, b.z - a.z);
    let ac = (c.x - a.x, c.y - a.y, c.z - a.z);
    let ap = (point.x - a.x, point.y - a.y, point.z - a.z);
    let normal = (
        ab.1 * ac.2 - ab.2 * ac.1,
        ab.2 * ac.0 - ab.0 * ac.2,
        ab.0 * ac.1 - ab.1 * ac.0,
    );
    let normal_len_sq = normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2;
    let distance = normal.0 * ap.0 + normal.1 * ap.1 + normal.2 * ap.2;

    distance.abs() <= 1e-12 * normal_len_sq.sqrt().max(1.0)
}

/// Uniform refinement of the entire mesh
pub fn uniform_refine(mesh: &mut Mesh) -> RefinementResult {
    let elements_to_refine: Vec<usize> = (0..mesh.num_elements()).collect();
    refine_elements(mesh, &elements_to_refine)
}

/// Uniform refinement with mesh and selection validation before mutation.
pub fn uniform_refine_checked(mesh: &mut Mesh) -> Result<RefinementResult, RefinementError> {
    validate_refinement_mesh(mesh)?;
    let elements_to_refine: Vec<usize> = (0..mesh.num_elements()).collect();
    let mut candidate = mesh.clone();
    let result = refine_elements(&mut candidate, &elements_to_refine);
    validate_refined_mesh(&candidate)?;
    *mesh = candidate;
    Ok(result)
}

/// Refine selected elements after validating the mesh and selection.
pub fn refine_elements_checked(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<RefinementResult, RefinementError> {
    validate_refinement_mesh(mesh)?;
    let mut seen = HashSet::with_capacity(elements_to_refine.len());
    for &element in elements_to_refine {
        if element >= mesh.elements.len() {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "selected element index is outside the mesh",
            });
        }
        if !seen.insert(element) {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "selected element index is duplicated",
            });
        }
    }
    // Canonicalize the caller's selection before midpoint allocation and
    // child emission. The unchecked compatibility API retains historical
    // ordering, but production checked refinement must not let a hash/set or
    // parallel estimator iteration order change node or element numbering.
    let mut canonical_selection = elements_to_refine.to_vec();
    canonical_selection.sort_unstable();
    let mut candidate = mesh.clone();
    let result = refine_elements(&mut candidate, &canonical_selection);
    validate_refined_mesh(&candidate)?;
    *mesh = candidate;
    Ok(result)
}

/// Refine selected first-order elements and retain enough provenance for a
/// later checked coarsening operation.
///
/// The ordinary checked API remains the compatibility entry point and does
/// not allocate history. Adaptation loops that need reversible refinement
/// should use this function and keep the returned history alongside the
/// refined state.
pub fn refine_elements_with_history_checked(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<(RefinementResult, RefinementHistory), RefinementError> {
    validate_refinement_mesh(mesh)?;
    let mut seen = HashSet::with_capacity(elements_to_refine.len());
    for &element in elements_to_refine {
        if element >= mesh.elements.len() {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "selected element index is outside the mesh",
            });
        }
        if !seen.insert(element) {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "selected element index is duplicated",
            });
        }
    }

    let mut selection = elements_to_refine.to_vec();
    selection.sort_unstable();
    let history = RefinementHistory {
        parents: selection
            .iter()
            .map(|&original_index| ParentRefinement {
                original_index,
                element: mesh.elements[original_index].clone(),
            })
            .collect(),
        boundaries: mesh
            .boundaries
            .iter()
            .filter(|boundary| selection.binary_search(&boundary.element_idx).is_ok())
            .cloned()
            .collect(),
    };

    let mut candidate = mesh.clone();
    let result = refine_elements(&mut candidate, &selection);
    validate_refined_mesh(&candidate)?;
    *mesh = candidate;
    Ok((result, history))
}

/// Uniform first-order refinement with reversible provenance.
pub fn uniform_refine_with_history_checked(
    mesh: &mut Mesh,
) -> Result<(RefinementResult, RefinementHistory), RefinementError> {
    let selection: Vec<usize> = (0..mesh.num_elements()).collect();
    refine_elements_with_history_checked(mesh, &selection)
}

/// Coarsen complete sibling families created by
/// [`refine_elements_with_history_checked`].
///
/// Coarsening is transactional. A parent is restored only when every direct
/// child is present, no child has descendants, and all connectivity/orientation
/// checks pass. Midpoint and centre nodes are deliberately retained: removing
/// them would renumber external state vectors and is therefore a separate,
/// explicit garbage-collection operation.
pub fn coarsen_elements_checked(
    mesh: &mut Mesh,
    history: &RefinementHistory,
    parent_ids: &[usize],
) -> Result<CoarseningResult, RefinementError> {
    if history.parents.is_empty() || parent_ids.is_empty() {
        return Err(RefinementError::EmptyHistory);
    }
    if mesh.validate_connectivity().is_err() {
        return Err(RefinementError::InvalidMesh {
            element: 0,
            reason: "mesh connectivity validation failed",
        });
    }
    if let Some(&element) = check_orientation(mesh).first() {
        return Err(RefinementError::InvalidMesh {
            element,
            reason: "mesh contains an inverted element",
        });
    }

    let mut requested = parent_ids.to_vec();
    requested.sort_unstable();
    if requested.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(RefinementError::InvalidHistory {
            parent_id: requested[0],
            reason: "parent ID was supplied more than once",
        });
    }

    let mut selected = Vec::with_capacity(requested.len());
    for parent_id in requested {
        let Some(parent) = history
            .parents
            .iter()
            .find(|parent| parent.element.id == parent_id)
        else {
            return Err(RefinementError::InvalidHistory {
                parent_id,
                reason: "parent is absent from the refinement history",
            });
        };
        let parent_order = parent.element.try_polynomial_order().ok();
        if !matches!(parent_order, Some(1) | Some(2) | Some(3)) {
            return Err(RefinementError::InvalidHistory {
                parent_id,
                reason: "only first-, second-, or third-order parent families can be coarsened",
            });
        }
        if mesh.elements.iter().any(|element| element.id == parent_id) {
            return Err(RefinementError::InvalidHistory {
                parent_id,
                reason: "parent is already present in the mesh",
            });
        }

        let expected_children = match parent.element.element_type {
            ElementType::Triangle | ElementType::Quadrilateral => 4,
            ElementType::Tetrahedron | ElementType::Hexahedron => 8,
        };
        let children: Vec<(usize, usize)> = mesh
            .elements
            .iter()
            .enumerate()
            .filter(|(_, element)| {
                element.parent_id == Some(parent_id) && element.level == parent.element.level + 1
            })
            .map(|(index, element)| (index, element.id))
            .collect();
        if children.len() != expected_children {
            return Err(RefinementError::InvalidHistory {
                parent_id,
                reason: "sibling family is incomplete or has already been coarsened",
            });
        }
        if children
            .iter()
            .any(|(index, _)| mesh.elements[*index].try_polynomial_order().ok() != parent_order)
        {
            return Err(RefinementError::InvalidHistory {
                parent_id,
                reason: "sibling family polynomial order differs from parent",
            });
        }
        let child_ids: HashSet<usize> = children.iter().map(|(_, id)| *id).collect();
        if mesh.elements.iter().any(|element| {
            element
                .parent_id
                .is_some_and(|ancestor| child_ids.contains(&ancestor))
        }) {
            return Err(RefinementError::InvalidHistory {
                parent_id,
                reason: "a direct child has descendants; coarsening would discard refinement",
            });
        }
        selected.push((parent.clone(), children));
    }

    let removed_set: HashSet<usize> = selected
        .iter()
        .flat_map(|(_, children)| children.iter().map(|(index, _)| *index))
        .collect();
    let mut old_to_new = HashMap::with_capacity(mesh.elements.len() - removed_set.len());
    let mut retained = Vec::with_capacity(mesh.elements.len() - removed_set.len() + selected.len());
    let mut removed_elements = Vec::with_capacity(removed_set.len());
    for (index, element) in mesh.elements.iter().cloned().enumerate() {
        if removed_set.contains(&index) {
            removed_elements.push(index);
        } else {
            old_to_new.insert(index, retained.len());
            retained.push(element);
        }
    }

    selected.sort_by_key(|(parent, _)| parent.element.id);
    let mut restored_elements = Vec::with_capacity(selected.len());
    let mut parent_new_index = HashMap::with_capacity(selected.len());
    for (parent, _) in &selected {
        let index = retained.len();
        parent_new_index.insert(parent.element.id, index);
        restored_elements.push(index);
        retained.push(parent.element.clone());
    }

    let mut boundaries = Vec::with_capacity(mesh.boundaries.len());
    for boundary in &mesh.boundaries {
        if removed_set.contains(&boundary.element_idx) {
            continue;
        }
        let Some(&element_idx) = old_to_new.get(&boundary.element_idx) else {
            return Err(RefinementError::InvalidHistory {
                parent_id: 0,
                reason: "boundary references an element removed during coarsening",
            });
        };
        let mut remapped = boundary.clone();
        remapped.element_idx = element_idx;
        boundaries.push(remapped);
    }
    for (parent, _) in &selected {
        let Some(&element_idx) = parent_new_index.get(&parent.element.id) else {
            unreachable!("every selected parent receives a new element index");
        };
        for boundary in history
            .boundaries
            .iter()
            .filter(|boundary| boundary.element_idx == parent.original_index)
        {
            let mut restored = boundary.clone();
            restored.element_idx = element_idx;
            restored.local_idx = parent
                .element
                .boundary_entities()
                .iter()
                .position(|nodes| {
                    let mut lhs = nodes.clone();
                    let mut rhs = restored.nodes.clone();
                    lhs.sort_unstable();
                    rhs.sort_unstable();
                    lhs == rhs
                })
                .unwrap_or(restored.local_idx);
            boundaries.push(restored);
        }
    }

    let mut candidate = mesh.clone();
    candidate.elements = retained;
    candidate.boundaries = boundaries;
    candidate.invalidate_caches();
    if candidate.validate_connectivity().is_err() {
        return Err(RefinementError::InvalidHistory {
            parent_id: 0,
            reason: "coarsening produced invalid connectivity or boundary ownership",
        });
    }
    if let Some(&element) = check_orientation(&candidate).first() {
        return Err(RefinementError::InvalidMesh {
            element,
            reason: "coarsening produced an inverted element",
        });
    }
    *mesh = candidate;
    Ok(CoarseningResult {
        restored_elements,
        removed_elements,
    })
}

/// Refine selected P2 cells or P3 triangles while preserving their
/// high-order trace layout.
///
/// The historical [`refine_elements_checked`] API deliberately rejects
/// higher-order cells because its P1 children would discard mid-side DOFs.
/// This entry point performs red refinement on P2 triangles, the standard
/// eight-subtet split on P2 tetrahedra, and tensor-product refinement on Q2
/// quadrilaterals and Hex27 elements. P3 red refinement supports triangles
/// and quadrilaterals. Every child retains the parent's high-order
/// isoparametric geometry.
pub fn refine_elements_high_order_checked(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<RefinementResult, RefinementError> {
    validate_high_order_refinement_mesh(mesh)?;
    if elements_to_refine.is_empty() {
        return Ok(RefinementResult {
            new_elements: Vec::new(),
            removed_elements: Vec::new(),
            new_nodes: Vec::new(),
        });
    }
    let mut seen = HashSet::with_capacity(elements_to_refine.len());
    for &element in elements_to_refine {
        if element >= mesh.elements.len() {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "selected element index is outside the mesh",
            });
        }
        if !seen.insert(element) {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "selected element index is duplicated",
            });
        }
        let cell = &mesh.elements[element];
        if !matches!(cell.try_polynomial_order().ok(), Some(2) | Some(3))
            || !matches!(
                cell.element_type,
                ElementType::Triangle
                    | ElementType::Quadrilateral
                    | ElementType::Tetrahedron
                    | ElementType::Hexahedron
            )
        {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "high-order refinement supports P2 cells or P3 elements",
            });
        }
    }

    let selected_order = elements_to_refine
        .first()
        .and_then(|&index| mesh.elements[index].try_polynomial_order().ok())
        .expect("validated high-order selection is non-empty");
    if elements_to_refine
        .iter()
        .any(|&index| mesh.elements[index].try_polynomial_order().ok() != Some(selected_order))
    {
        return Err(RefinementError::InvalidMesh {
            element: elements_to_refine[0],
            reason: "one high-order refinement step cannot mix polynomial orders",
        });
    }
    if selected_order == 3
        && elements_to_refine.iter().any(|&index| {
            !matches!(
                mesh.elements[index].element_type,
                ElementType::Triangle
                    | ElementType::Quadrilateral
                    | ElementType::Tetrahedron
                    | ElementType::Hexahedron
            )
        })
    {
        return Err(RefinementError::InvalidMesh {
            element: elements_to_refine[0],
            reason: "P3 refinement currently supports triangles, quadrilaterals, tetrahedra, and hexahedra",
        });
    }
    if selected_order == 3
        && elements_to_refine.iter().any(|&index| {
            mesh.elements[index].element_type != mesh.elements[elements_to_refine[0]].element_type
        })
    {
        return Err(RefinementError::InvalidMesh {
            element: elements_to_refine[0],
            reason: "one P3 refinement step cannot mix element types",
        });
    }

    let mut selection = elements_to_refine.to_vec();
    selection.sort_unstable();
    let mut candidate = mesh.clone();
    let result = if selected_order == 2 {
        refine_p2_elements(&mut candidate, &selection)?
    } else if candidate.elements[selection[0]].element_type == ElementType::Triangle {
        refine_p3_triangle_elements(&mut candidate, &selection)?
    } else if candidate.elements[selection[0]].element_type == ElementType::Tetrahedron {
        refine_p3_tetrahedron_elements(&mut candidate, &selection)?
    } else if candidate.elements[selection[0]].element_type == ElementType::Quadrilateral {
        refine_p3_quadrilateral_elements(&mut candidate, &selection)?
    } else {
        refine_p3_hexahedron_elements(&mut candidate, &selection)?
    };
    validate_high_order_refined_mesh(&candidate)?;
    *mesh = candidate;
    Ok(result)
}

/// Refine selected quadratic cells and retain provenance for checked
/// high-order state transfer and reversible coarsening.
pub fn refine_elements_high_order_with_history_checked(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<(RefinementResult, RefinementHistory), RefinementError> {
    validate_high_order_refinement_mesh(mesh)?;
    if elements_to_refine.is_empty() {
        return Ok((
            RefinementResult {
                new_elements: Vec::new(),
                removed_elements: Vec::new(),
                new_nodes: Vec::new(),
            },
            RefinementHistory {
                parents: Vec::new(),
                boundaries: Vec::new(),
            },
        ));
    }
    let mut seen = HashSet::with_capacity(elements_to_refine.len());
    for &element in elements_to_refine {
        if element >= mesh.elements.len() {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "selected element index is outside mesh",
            });
        }
        if !seen.insert(element) {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "selected element index is duplicated",
            });
        }
        let cell = &mesh.elements[element];
        if !matches!(cell.try_polynomial_order().ok(), Some(2) | Some(3))
            || !matches!(
                cell.element_type,
                ElementType::Triangle
                    | ElementType::Quadrilateral
                    | ElementType::Tetrahedron
                    | ElementType::Hexahedron
            )
        {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "high-order refinement supports P2 cells or P3 elements",
            });
        }
    }

    let selected_order = elements_to_refine
        .first()
        .and_then(|&index| mesh.elements[index].try_polynomial_order().ok())
        .expect("validated high-order selection is non-empty");
    if elements_to_refine
        .iter()
        .any(|&index| mesh.elements[index].try_polynomial_order().ok() != Some(selected_order))
    {
        return Err(RefinementError::InvalidMesh {
            element: elements_to_refine[0],
            reason: "one high-order refinement step cannot mix polynomial orders",
        });
    }
    if selected_order == 3
        && elements_to_refine.iter().any(|&index| {
            !matches!(
                mesh.elements[index].element_type,
                ElementType::Triangle
                    | ElementType::Quadrilateral
                    | ElementType::Tetrahedron
                    | ElementType::Hexahedron
            )
        })
    {
        return Err(RefinementError::InvalidMesh {
            element: elements_to_refine[0],
            reason: "P3 refinement currently supports triangles, quadrilaterals, tetrahedra, and hexahedra",
        });
    }
    if selected_order == 3
        && elements_to_refine.iter().any(|&index| {
            mesh.elements[index].element_type != mesh.elements[elements_to_refine[0]].element_type
        })
    {
        return Err(RefinementError::InvalidMesh {
            element: elements_to_refine[0],
            reason: "one P3 refinement step cannot mix element types",
        });
    }

    let mut selection = elements_to_refine.to_vec();
    selection.sort_unstable();
    let history = RefinementHistory {
        parents: selection
            .iter()
            .map(|&original_index| ParentRefinement {
                original_index,
                element: mesh.elements[original_index].clone(),
            })
            .collect(),
        boundaries: mesh
            .boundaries
            .iter()
            .filter(|boundary| selection.binary_search(&boundary.element_idx).is_ok())
            .cloned()
            .collect(),
    };

    let result = refine_elements_high_order_checked(mesh, &selection)?;
    Ok((result, history))
}

fn refine_p2_elements(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<RefinementResult, RefinementError> {
    let mut removed_elements = Vec::with_capacity(elements_to_refine.len());
    let mut refined_parent_ids = HashMap::with_capacity(elements_to_refine.len());
    let mut child_elements = Vec::new();
    let mut midpoint_mgr = MidpointManager::new();
    let original_boundaries = mesh.boundaries.clone();

    // Reuse the parent's existing mid-side nodes. Without this seed, a
    // refined child edge that lies on a parent edge would create a duplicate
    // node at the same location and split the quadratic trace incorrectly.
    for &element_index in elements_to_refine {
        let element = &mesh.elements[element_index];
        for (edge_index, &(a, b)) in element.element_type.local_edges().iter().enumerate() {
            let vertices = element.vertices();
            let midpoint = element.nodes[element.element_type.num_vertices() + edge_index];
            midpoint_mgr
                .edge_midpoints
                .entry(Edge::new(vertices[a], vertices[b]))
                .or_insert(midpoint);
        }
    }

    for &element_index in elements_to_refine {
        let element = mesh.elements[element_index].clone();
        let child_start = child_elements.len();
        match element.element_type {
            ElementType::Triangle => {
                let vertices = [element.nodes[0], element.nodes[1], element.nodes[2]];
                let parent_midpoints = [element.nodes[3], element.nodes[4], element.nodes[5]];
                let child_vertices = [
                    (
                        [vertices[0], parent_midpoints[0], parent_midpoints[2]],
                        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]],
                    ),
                    (
                        [parent_midpoints[0], vertices[1], parent_midpoints[1]],
                        [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
                    ),
                    (
                        [parent_midpoints[2], parent_midpoints[1], vertices[2]],
                        [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
                    ),
                    (
                        [
                            parent_midpoints[0],
                            parent_midpoints[1],
                            parent_midpoints[2],
                        ],
                        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
                    ),
                ];
                for (child, reference_vertices) in child_vertices {
                    let nodes = p2_triangle_child_nodes(
                        mesh,
                        &mut midpoint_mgr,
                        &element,
                        child,
                        reference_vertices,
                    );
                    child_elements.push(child_element(
                        &element,
                        ElementType::Triangle,
                        nodes,
                        mesh,
                    ));
                }
            }
            ElementType::Quadrilateral => {
                for nodes in q2_quadrilateral_children(mesh, &mut midpoint_mgr, &element) {
                    child_elements.push(child_element(
                        &element,
                        ElementType::Quadrilateral,
                        nodes,
                        mesh,
                    ));
                }
            }
            ElementType::Tetrahedron => {
                let vertices = [
                    element.nodes[0],
                    element.nodes[1],
                    element.nodes[2],
                    element.nodes[3],
                ];
                let parent_reference = [
                    ([1.0, 0.0, 0.0, 0.0], vertices[0]),
                    ([0.0, 1.0, 0.0, 0.0], vertices[1]),
                    ([0.0, 0.0, 1.0, 0.0], vertices[2]),
                    ([0.0, 0.0, 0.0, 1.0], vertices[3]),
                    ([0.5, 0.5, 0.0, 0.0], element.nodes[4]),
                    ([0.5, 0.0, 0.5, 0.0], element.nodes[5]),
                    ([0.5, 0.0, 0.0, 0.5], element.nodes[6]),
                    ([0.0, 0.5, 0.5, 0.0], element.nodes[7]),
                    ([0.0, 0.5, 0.0, 0.5], element.nodes[8]),
                    ([0.0, 0.0, 0.5, 0.5], element.nodes[9]),
                ];
                for child in p1_tetra_children(mesh, &mut midpoint_mgr, vertices) {
                    let reference_vertices = child.map(|node| {
                        parent_reference
                            .iter()
                            .find_map(|&(reference, candidate)| {
                                (candidate == node).then_some(reference)
                            })
                            .expect("P2 tetra child vertex must belong to parent geometry")
                    });
                    let nodes = p2_tetra_child_nodes(
                        mesh,
                        &mut midpoint_mgr,
                        &element,
                        child,
                        reference_vertices,
                    );
                    child_elements.push(child_element(
                        &element,
                        ElementType::Tetrahedron,
                        nodes,
                        mesh,
                    ));
                }
            }
            ElementType::Hexahedron => {
                for nodes in q2_hexahedron_children(mesh, &mut midpoint_mgr, &element) {
                    child_elements.push(child_element(
                        &element,
                        ElementType::Hexahedron,
                        nodes,
                        mesh,
                    ));
                }
            }
        }
        if child_elements.len() > child_start {
            removed_elements.push(element_index);
            refined_parent_ids.insert(element_index, element.id);
        }
    }

    let removed: HashSet<usize> = removed_elements.iter().copied().collect();
    let mut old_to_new_element = HashMap::with_capacity(mesh.elements.len() - removed.len());
    let mut retained =
        Vec::with_capacity(mesh.elements.len() - removed.len() + child_elements.len());
    for (index, element) in mesh.elements.drain(..).enumerate() {
        if !removed.contains(&index) {
            old_to_new_element.insert(index, retained.len());
            retained.push(element);
        }
    }
    let first_child_index = retained.len();
    let new_elements = (first_child_index..first_child_index + child_elements.len()).collect();
    retained.extend(child_elements);
    mesh.elements = retained;
    mesh.boundaries = propagate_boundaries(
        mesh,
        original_boundaries,
        &old_to_new_element,
        &refined_parent_ids,
    );
    mesh.invalidate_caches();

    Ok(RefinementResult {
        new_elements,
        removed_elements,
        new_nodes: midpoint_mgr.new_nodes,
    })
}

/// Red-refine selected cubic triangles while preserving their P3 trace and
/// interior nodes. The parent reference lattice uses denominator six: this
/// makes every child P3 node an integer lattice point and gives the edge
/// manager a deterministic key for shared nodes.
fn refine_p3_triangle_elements(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<RefinementResult, RefinementError> {
    let mut point_manager = P3PointManager::new();
    let original_boundaries = mesh.boundaries.clone();
    let mut child_elements = Vec::with_capacity(elements_to_refine.len() * 4);

    for &element_index in elements_to_refine {
        let parent = mesh.elements[element_index].clone();
        point_manager.seed_parent(mesh, &parent);
        let lattice_vertices = [[6, 0, 0], [0, 6, 0], [0, 0, 6]];
        let child_lattice_vertices = [[
            [lattice_vertices[0], [3, 3, 0], [3, 0, 3]],
            [[3, 3, 0], lattice_vertices[1], [0, 3, 3]],
            [lattice_vertices[2], [3, 0, 3], [0, 3, 3]],
            [[3, 3, 0], [0, 3, 3], [3, 0, 3]],
        ]];

        for child_vertices in child_lattice_vertices[0] {
            let nodes = p3_child_nodes(mesh, &mut point_manager, &parent, child_vertices);
            let mut child = Element::new(ElementType::Triangle, nodes, mesh.next_element_id());
            child.parent_id = Some(parent.id);
            child.level = parent.level + 1;
            child.marker = parent.marker;
            child_elements.push(child);
        }
    }

    let removed_set: HashSet<usize> = elements_to_refine.iter().copied().collect();
    let mut old_to_new = HashMap::with_capacity(mesh.elements.len() - removed_set.len());
    let mut retained =
        Vec::with_capacity(mesh.elements.len() - removed_set.len() + child_elements.len());
    let mut removed_elements = Vec::with_capacity(removed_set.len());
    for (index, element) in mesh.elements.iter().cloned().enumerate() {
        if removed_set.contains(&index) {
            removed_elements.push(index);
        } else {
            old_to_new.insert(index, retained.len());
            retained.push(element);
        }
    }
    let new_elements = (retained.len()..retained.len() + child_elements.len()).collect();
    retained.extend(child_elements);

    let mut boundaries = Vec::with_capacity(original_boundaries.len());
    for boundary in original_boundaries {
        if removed_set.contains(&boundary.element_idx) {
            continue;
        }
        let Some(&element_idx) = old_to_new.get(&boundary.element_idx) else {
            return Err(RefinementError::InvalidHistory {
                parent_id: 0,
                reason: "boundary references an element removed during P3 refinement",
            });
        };
        let mut remapped = boundary;
        remapped.element_idx = element_idx;
        boundaries.push(remapped);
    }

    mesh.elements = retained;
    mesh.boundaries = boundaries;
    mesh.invalidate_caches();
    Ok(RefinementResult {
        new_elements,
        removed_elements,
        new_nodes: point_manager.new_nodes,
    })
}

/// Red-refine selected cubic tetrahedra while preserving their P3 trace and
/// face-interior nodes. Child nodes use canonical global vertex/edge/face
/// keys for points on the parent boundary, and a parent-tetrahedron key for
/// interior points, so adjacent parents share every trace node exactly once.
fn refine_p3_tetrahedron_elements(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<RefinementResult, RefinementError> {
    let mut point_manager = P3TetraPointManager::new();
    let original_boundaries = mesh.boundaries.clone();
    let mut child_elements = Vec::with_capacity(elements_to_refine.len() * 8);

    for &element_index in elements_to_refine {
        let parent = mesh.elements[element_index].clone();
        point_manager.seed_parent(mesh, &parent);
        for child_labels in p3_tetra_child_labels(mesh, &parent) {
            let nodes = p3_tetra_child_nodes(mesh, &mut point_manager, &parent, child_labels);
            let mut child = Element::new(ElementType::Tetrahedron, nodes, mesh.next_element_id());
            child.parent_id = Some(parent.id);
            child.level = parent.level + 1;
            child.marker = parent.marker;
            child_elements.push(child);
        }
    }

    let removed_set: HashSet<usize> = elements_to_refine.iter().copied().collect();
    let mut old_to_new = HashMap::with_capacity(mesh.elements.len() - removed_set.len());
    let mut retained =
        Vec::with_capacity(mesh.elements.len() - removed_set.len() + child_elements.len());
    let mut removed_elements = Vec::with_capacity(removed_set.len());
    for (index, element) in mesh.elements.iter().cloned().enumerate() {
        if removed_set.contains(&index) {
            removed_elements.push(index);
        } else {
            old_to_new.insert(index, retained.len());
            retained.push(element);
        }
    }
    let first_child_index = retained.len();
    retained.extend(child_elements);

    let mut boundaries = Vec::with_capacity(original_boundaries.len());
    for boundary in original_boundaries {
        if removed_set.contains(&boundary.element_idx) {
            continue;
        }
        let Some(&element_idx) = old_to_new.get(&boundary.element_idx) else {
            return Err(RefinementError::InvalidHistory {
                parent_id: 0,
                reason: "boundary references an element removed during P3 tetrahedral refinement",
            });
        };
        let mut remapped = boundary;
        remapped.element_idx = element_idx;
        boundaries.push(remapped);
    }

    mesh.elements = retained;
    mesh.boundaries = boundaries;
    mesh.invalidate_caches();
    Ok(RefinementResult {
        new_elements: (first_child_index..mesh.elements.len()).collect(),
        removed_elements,
        new_nodes: point_manager.new_nodes,
    })
}

const P3_TETRA_LATTICES: [[usize; 4]; 20] = [
    [3, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, 3],
    [2, 1, 0, 0],
    [1, 2, 0, 0],
    [2, 0, 1, 0],
    [1, 0, 2, 0],
    [2, 0, 0, 1],
    [1, 0, 0, 2],
    [0, 2, 1, 0],
    [0, 1, 2, 0],
    [0, 2, 0, 1],
    [0, 1, 0, 2],
    [0, 0, 2, 1],
    [0, 0, 1, 2],
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
];

fn p3_tetra_child_labels(mesh: &Mesh, parent: &Element) -> Vec<[usize; 4]> {
    let diagonal_02_13 = distance_squared(
        p3_tetra_label_point(mesh, parent, 5),
        p3_tetra_label_point(mesh, parent, 8),
    );
    let diagonal_01_23 = distance_squared(
        p3_tetra_label_point(mesh, parent, 4),
        p3_tetra_label_point(mesh, parent, 9),
    );
    let diagonal_03_12 = distance_squared(
        p3_tetra_label_point(mesh, parent, 6),
        p3_tetra_label_point(mesh, parent, 7),
    );
    let inner = if diagonal_02_13 <= diagonal_01_23 && diagonal_02_13 <= diagonal_03_12 {
        [[4, 5, 6, 8], [4, 5, 7, 8], [5, 6, 8, 9], [5, 7, 8, 9]]
    } else if diagonal_01_23 <= diagonal_03_12 {
        [[4, 9, 5, 6], [4, 9, 7, 8], [4, 9, 5, 7], [4, 9, 6, 8]]
    } else {
        [[6, 7, 4, 5], [6, 7, 4, 8], [6, 7, 5, 9], [6, 7, 8, 9]]
    };
    let raw = [
        [0, 4, 5, 6],
        [4, 1, 7, 8],
        [5, 7, 2, 9],
        [6, 8, 9, 3],
        inner[0],
        inner[1],
        inner[2],
        inner[3],
    ];
    raw.map(|labels| orient_p3_tetra_labels(mesh, parent, labels))
        .to_vec()
}

fn p3_tetra_child_nodes(
    mesh: &mut Mesh,
    manager: &mut P3TetraPointManager,
    parent: &Element,
    child_labels: [usize; 4],
) -> Vec<usize> {
    P3_TETRA_LATTICES
        .into_iter()
        .map(|alpha| {
            let mut key = [0usize; 4];
            for (local, &weight) in alpha.iter().enumerate() {
                let vertex = tetra_label_barycentric(child_labels[local]);
                for component in 0..4 {
                    key[component] += weight * vertex[component];
                }
            }
            manager.node_for(mesh, parent, key)
        })
        .collect()
}

fn orient_p3_tetra_labels(mesh: &Mesh, parent: &Element, mut labels: [usize; 4]) -> [usize; 4] {
    let points = labels.map(|label| p3_tetra_label_point(mesh, parent, label));
    if signed_tetra_volume(points[0], points[1], points[2], points[3]) < 0.0 {
        labels.swap(2, 3);
    }
    labels
}

fn p3_tetra_label_point(mesh: &Mesh, parent: &Element, label: usize) -> Point {
    let half = tetra_label_barycentric(label);
    p3_tetra_parent_point(mesh, parent, half.map(|value| value * 3))
}

fn tetra_label_barycentric(label: usize) -> [usize; 4] {
    match label {
        0 => [2, 0, 0, 0],
        1 => [0, 2, 0, 0],
        2 => [0, 0, 2, 0],
        3 => [0, 0, 0, 2],
        4 => [1, 1, 0, 0],
        5 => [1, 0, 1, 0],
        6 => [1, 0, 0, 1],
        7 => [0, 1, 1, 0],
        8 => [0, 1, 0, 1],
        9 => [0, 0, 1, 1],
        _ => unreachable!("P3 tetra refinement label must be in 0..10"),
    }
}

fn p3_tetra_parent_point(mesh: &Mesh, parent: &Element, key: [usize; 4]) -> Point {
    let barycentric = key.map(|value| value as f64 / 6.0);
    let mut point = Point::new_3d(0.0, 0.0, 0.0);
    for (&node, &alpha) in parent.nodes.iter().zip(P3_TETRA_LATTICES.iter()) {
        let mut basis = 1.0;
        for (component, &degree) in alpha.iter().enumerate() {
            for m in 0..degree {
                basis *= (3.0 * barycentric[component] - m as f64) / (degree - m) as f64;
            }
        }
        point.x += basis * mesh.nodes[node].x;
        point.y += basis * mesh.nodes[node].y;
        point.z += basis * mesh.nodes[node].z;
    }
    point
}

fn distance_squared(a: Point, b: Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

fn signed_tetra_volume(a: Point, b: Point, c: Point, d: Point) -> f64 {
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum P3TetraPointKey {
    Vertex(usize),
    Edge {
        vertices: [usize; 2],
        weights: [u8; 2],
    },
    Face {
        vertices: [usize; 3],
        weights: [u8; 3],
    },
    Cell {
        vertices: [usize; 4],
        weights: [u8; 4],
    },
}

fn p3_tetra_point_key(parent: &Element, key: [usize; 4]) -> P3TetraPointKey {
    let mut entries = parent.nodes[..4]
        .iter()
        .copied()
        .zip(key)
        .filter(|(_, weight)| *weight != 0)
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(vertex, _)| *vertex);

    match entries.as_slice() {
        [(vertex, _)] => P3TetraPointKey::Vertex(*vertex),
        [(vertex0, weight0), (vertex1, weight1)] => P3TetraPointKey::Edge {
            vertices: [*vertex0, *vertex1],
            weights: [*weight0 as u8, *weight1 as u8],
        },
        [(vertex0, weight0), (vertex1, weight1), (vertex2, weight2)] => P3TetraPointKey::Face {
            vertices: [*vertex0, *vertex1, *vertex2],
            weights: [*weight0 as u8, *weight1 as u8, *weight2 as u8],
        },
        [
            (vertex0, weight0),
            (vertex1, weight1),
            (vertex2, weight2),
            (vertex3, weight3),
        ] => P3TetraPointKey::Cell {
            vertices: [*vertex0, *vertex1, *vertex2, *vertex3],
            weights: [
                *weight0 as u8,
                *weight1 as u8,
                *weight2 as u8,
                *weight3 as u8,
            ],
        },
        [] => unreachable!("P3 tetra point key must have non-zero barycentric weight"),
        _ => unreachable!("P3 tetra point key has an invalid support"),
    }
}

struct P3TetraPointManager {
    points: HashMap<P3TetraPointKey, usize>,
    new_nodes: Vec<usize>,
}

impl P3TetraPointManager {
    fn new() -> Self {
        Self {
            points: HashMap::new(),
            new_nodes: Vec::new(),
        }
    }

    fn seed_parent(&mut self, _mesh: &Mesh, parent: &Element) {
        for (node, alpha) in parent.nodes.iter().copied().zip(P3_TETRA_LATTICES) {
            self.points
                .entry(p3_tetra_point_key(parent, alpha.map(|value| value * 2)))
                .or_insert(node);
        }
    }

    fn node_for(&mut self, mesh: &mut Mesh, parent: &Element, key: [usize; 4]) -> usize {
        let point_key = p3_tetra_point_key(parent, key);
        if let Some(&node) = self.points.get(&point_key) {
            return node;
        }
        let node = mesh.add_node(p3_tetra_parent_point(mesh, parent, key));
        self.points.insert(point_key, node);
        self.new_nodes.push(node);
        node
    }
}

/// Tensor-product red refinement for selected cubic quadrilaterals. The
/// parent reference lattice uses seven points per axis (`0..=6`), so each
/// child Q3 layout is an exact 4x4 window and all shared trace points have a
/// stable integer key.
fn refine_p3_quadrilateral_elements(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<RefinementResult, RefinementError> {
    let mut point_manager = P3QuadPointManager::new();
    let original_boundaries = mesh.boundaries.clone();
    let mut child_elements = Vec::with_capacity(elements_to_refine.len() * 4);

    for &element_index in elements_to_refine {
        let parent = mesh.elements[element_index].clone();
        point_manager.seed_parent(&parent);
        for [base_i, base_j] in [[0usize, 0usize], [3, 0], [0, 3], [3, 3]] {
            let nodes =
                p3_quadrilateral_child_nodes(mesh, &mut point_manager, &parent, base_i, base_j);
            let mut child = Element::new(ElementType::Quadrilateral, nodes, mesh.next_element_id());
            child.parent_id = Some(parent.id);
            child.level = parent.level + 1;
            child.marker = parent.marker;
            child_elements.push(child);
        }
    }

    let removed_set: HashSet<usize> = elements_to_refine.iter().copied().collect();
    let mut old_to_new = HashMap::with_capacity(mesh.elements.len() - removed_set.len());
    let mut retained =
        Vec::with_capacity(mesh.elements.len() - removed_set.len() + child_elements.len());
    let mut removed_elements = Vec::with_capacity(removed_set.len());
    for (index, element) in mesh.elements.iter().cloned().enumerate() {
        if removed_set.contains(&index) {
            removed_elements.push(index);
        } else {
            old_to_new.insert(index, retained.len());
            retained.push(element);
        }
    }
    let new_elements = (retained.len()..retained.len() + child_elements.len()).collect();
    retained.extend(child_elements);

    let mut boundaries = Vec::with_capacity(original_boundaries.len());
    for boundary in original_boundaries {
        if removed_set.contains(&boundary.element_idx) {
            continue;
        }
        let Some(&element_idx) = old_to_new.get(&boundary.element_idx) else {
            return Err(RefinementError::InvalidHistory {
                parent_id: 0,
                reason: "boundary references an element removed during P3 refinement",
            });
        };
        let mut remapped = boundary;
        remapped.element_idx = element_idx;
        boundaries.push(remapped);
    }

    mesh.elements = retained;
    mesh.boundaries = boundaries;
    mesh.invalidate_caches();
    Ok(RefinementResult {
        new_elements,
        removed_elements,
        new_nodes: point_manager.new_nodes,
    })
}

struct P3PointManager {
    edge_points: HashMap<(usize, usize, u8), usize>,
    interior_points: HashMap<(usize, [usize; 3]), usize>,
    new_nodes: Vec<usize>,
}

struct P3QuadPointManager {
    edge_points: HashMap<(usize, usize, u8), usize>,
    interior_points: HashMap<(usize, [usize; 2]), usize>,
    new_nodes: Vec<usize>,
}

impl P3QuadPointManager {
    fn new() -> Self {
        Self {
            edge_points: HashMap::new(),
            interior_points: HashMap::new(),
            new_nodes: Vec::new(),
        }
    }

    fn seed_parent(&mut self, parent: &Element) {
        let vertices = parent.vertices();
        for (edge_index, &(a, b)) in ElementType::Quadrilateral.local_edges().iter().enumerate() {
            let first = parent.nodes[4 + edge_index * 2];
            let second = parent.nodes[5 + edge_index * 2];
            self.insert_edge(vertices[a], vertices[b], 2, first);
            self.insert_edge(vertices[a], vertices[b], 4, second);
        }
        for (lattice, node) in [
            ([2, 2], parent.nodes[12]),
            ([4, 2], parent.nodes[13]),
            ([4, 4], parent.nodes[14]),
            ([2, 4], parent.nodes[15]),
        ] {
            self.interior_points.insert((parent.id, lattice), node);
        }
    }

    fn insert_edge(&mut self, a: usize, b: usize, numerator: u8, node: usize) {
        let key = if a < b {
            (a, b, numerator)
        } else {
            (b, a, 6 - numerator)
        };
        self.edge_points.insert(key, node);
    }

    fn node_for(&mut self, mesh: &mut Mesh, parent: &Element, lattice: [usize; 2]) -> usize {
        let vertices = parent.vertices();
        for (expected, node) in [
            ([0, 0], vertices[0]),
            ([6, 0], vertices[1]),
            ([6, 6], vertices[2]),
            ([0, 6], vertices[3]),
        ] {
            if lattice == expected {
                return node;
            }
        }

        let edge = if lattice[1] == 0 {
            Some((vertices[0], vertices[1], lattice[0]))
        } else if lattice[0] == 6 {
            Some((vertices[1], vertices[2], lattice[1]))
        } else if lattice[1] == 6 {
            Some((vertices[2], vertices[3], 6 - lattice[0]))
        } else if lattice[0] == 0 {
            Some((vertices[3], vertices[0], 6 - lattice[1]))
        } else {
            None
        };
        if let Some((a, b, numerator)) = edge {
            let key = if a < b {
                (a, b, numerator as u8)
            } else {
                (b, a, 6 - numerator as u8)
            };
            if let Some(&node) = self.edge_points.get(&key) {
                return node;
            }
            let point = q3_quadrilateral_point(mesh, parent, lattice);
            let node = mesh.add_node(point);
            self.edge_points.insert(key, node);
            self.new_nodes.push(node);
            return node;
        }

        if let Some(&node) = self.interior_points.get(&(parent.id, lattice)) {
            return node;
        }
        let point = q3_quadrilateral_point(mesh, parent, lattice);
        let node = mesh.add_node(point);
        self.interior_points.insert((parent.id, lattice), node);
        self.new_nodes.push(node);
        node
    }
}

fn p3_quadrilateral_child_nodes(
    mesh: &mut Mesh,
    manager: &mut P3QuadPointManager,
    parent: &Element,
    base_i: usize,
    base_j: usize,
) -> Vec<usize> {
    const NODE_INDICES: [(usize, usize); 16] = [
        (0, 0),
        (3, 0),
        (3, 3),
        (0, 3),
        (1, 0),
        (2, 0),
        (3, 1),
        (3, 2),
        (2, 3),
        (1, 3),
        (0, 2),
        (0, 1),
        (1, 1),
        (2, 1),
        (2, 2),
        (1, 2),
    ];
    NODE_INDICES
        .into_iter()
        .map(|(i, j)| manager.node_for(mesh, parent, [base_i + i, base_j + j]))
        .collect()
}

fn q3_lagrange_values(value: f64) -> [f64; 4] {
    let nodes = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    let mut values = [0.0; 4];
    for i in 0..4 {
        let mut numerator = 1.0;
        let mut denominator = 1.0;
        for j in 0..4 {
            if i != j {
                numerator *= value - nodes[j];
                denominator *= nodes[i] - nodes[j];
            }
        }
        values[i] = numerator / denominator;
    }
    values
}

fn q3_quadrilateral_point(mesh: &Mesh, parent: &Element, lattice: [usize; 2]) -> Point {
    const NODE_INDICES: [(usize, usize); 16] = [
        (0, 0),
        (3, 0),
        (3, 3),
        (0, 3),
        (1, 0),
        (2, 0),
        (3, 1),
        (3, 2),
        (2, 3),
        (1, 3),
        (0, 2),
        (0, 1),
        (1, 1),
        (2, 1),
        (2, 2),
        (1, 2),
    ];
    let lx = q3_lagrange_values(-1.0 + lattice[0] as f64 / 3.0);
    let ly = q3_lagrange_values(-1.0 + lattice[1] as f64 / 3.0);
    let mut point = Point::new_3d(0.0, 0.0, 0.0);
    for (&node, &(i, j)) in parent.nodes.iter().zip(NODE_INDICES.iter()) {
        let weight = lx[i] * ly[j];
        let coordinate = mesh.nodes[node];
        point.x += weight * coordinate.x;
        point.y += weight * coordinate.y;
        point.z += weight * coordinate.z;
    }
    point
}

impl P3PointManager {
    fn new() -> Self {
        Self {
            edge_points: HashMap::new(),
            interior_points: HashMap::new(),
            new_nodes: Vec::new(),
        }
    }

    fn seed_parent(&mut self, _mesh: &Mesh, parent: &Element) {
        let vertices = parent.vertices();
        for (edge_index, &(a, b)) in ElementType::Triangle.local_edges().iter().enumerate() {
            let first = parent.nodes[3 + edge_index * 2];
            let second = parent.nodes[4 + edge_index * 2];
            self.insert_edge(vertices[a], vertices[b], 2, first);
            self.insert_edge(vertices[a], vertices[b], 4, second);
        }
        self.interior_points
            .insert((parent.id, [2, 2, 2]), parent.nodes[9]);
    }

    fn insert_edge(&mut self, a: usize, b: usize, numerator: u8, node: usize) {
        let key = if a < b {
            (a, b, numerator)
        } else {
            (b, a, 6 - numerator)
        };
        self.edge_points.insert(key, node);
    }

    fn node_for(&mut self, mesh: &mut Mesh, parent: &Element, lattice: [usize; 3]) -> usize {
        let vertices = parent.vertices();
        if lattice == [6, 0, 0] {
            return vertices[0];
        }
        if lattice == [0, 6, 0] {
            return vertices[1];
        }
        if lattice == [0, 0, 6] {
            return vertices[2];
        }

        let edge = if lattice[2] == 0 {
            Some((vertices[0], vertices[1], lattice[1]))
        } else if lattice[0] == 0 {
            Some((vertices[1], vertices[2], lattice[2]))
        } else if lattice[1] == 0 {
            Some((vertices[2], vertices[0], lattice[0]))
        } else {
            None
        };
        if let Some((a, b, numerator)) = edge {
            let key = if a < b {
                (a, b, numerator as u8)
            } else {
                (b, a, 6 - numerator as u8)
            };
            if let Some(&node) = self.edge_points.get(&key) {
                return node;
            }
            let point = p3_triangle_point(mesh, parent, lattice);
            let node = mesh.add_node(point);
            self.edge_points.insert(key, node);
            self.new_nodes.push(node);
            return node;
        }

        if let Some(&node) = self.interior_points.get(&(parent.id, lattice)) {
            return node;
        }
        let point = p3_triangle_point(mesh, parent, lattice);
        let node = mesh.add_node(point);
        self.interior_points.insert((parent.id, lattice), node);
        self.new_nodes.push(node);
        node
    }
}

fn p3_child_nodes(
    mesh: &mut Mesh,
    manager: &mut P3PointManager,
    parent: &Element,
    vertices: [[usize; 3]; 3],
) -> Vec<usize> {
    let mix = |a: [usize; 3], b: [usize; 3], numerator: usize| {
        [
            ((3 - numerator) * a[0] + numerator * b[0]) / 3,
            ((3 - numerator) * a[1] + numerator * b[1]) / 3,
            ((3 - numerator) * a[2] + numerator * b[2]) / 3,
        ]
    };
    let average = [
        (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3,
        (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3,
        (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3,
    ];
    let lattice = [
        vertices[0],
        vertices[1],
        vertices[2],
        mix(vertices[0], vertices[1], 1),
        mix(vertices[0], vertices[1], 2),
        mix(vertices[1], vertices[2], 1),
        mix(vertices[1], vertices[2], 2),
        mix(vertices[2], vertices[0], 1),
        mix(vertices[2], vertices[0], 2),
        average,
    ];
    lattice
        .into_iter()
        .map(|point| manager.node_for(mesh, parent, point))
        .collect()
}

fn p3_triangle_point(mesh: &Mesh, parent: &Element, lattice: [usize; 3]) -> Point {
    const EXPONENTS: [[usize; 3]; 10] = [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 1, 1],
    ];
    const COEFFICIENTS: [f64; 10] = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 6.0];
    let barycentric = [
        lattice[0] as f64 / 6.0,
        lattice[1] as f64 / 6.0,
        lattice[2] as f64 / 6.0,
    ];
    let mut point = Point::new_3d(0.0, 0.0, 0.0);
    for (node, (&[a, b, c], coefficient)) in
        parent.nodes.iter().zip(EXPONENTS.iter().zip(COEFFICIENTS))
    {
        let weight = coefficient
            * barycentric[0].powi(a as i32)
            * barycentric[1].powi(b as i32)
            * barycentric[2].powi(c as i32);
        let coordinate = mesh.nodes[*node];
        point.x += weight * coordinate.x;
        point.y += weight * coordinate.y;
        point.z += weight * coordinate.z;
    }
    point
}

fn child_element(
    parent: &Element,
    element_type: ElementType,
    nodes: Vec<usize>,
    mesh: &mut Mesh,
) -> Element {
    let mut child = Element::new(element_type, nodes, mesh.next_element_id());
    child.parent_id = Some(parent.id);
    child.level = parent.level + 1;
    child.marker = parent.marker;
    child
}

/// Build a quadratic child triangle while preserving a curved parent P2 map.
/// New child edge nodes are evaluated at the midpoint in parent reference
/// coordinates rather than averaging physical endpoints.
fn p2_triangle_child_nodes(
    mesh: &mut Mesh,
    midpoint_mgr: &mut MidpointManager,
    parent: &Element,
    vertices: [usize; 3],
    reference_vertices: [[f64; 3]; 3],
) -> Vec<usize> {
    debug_assert_eq!(parent.element_type, ElementType::Triangle);
    debug_assert!(parent.nodes.len() >= ElementType::Triangle.num_nodes_p2());

    let mut nodes = vertices.to_vec();
    for &(a, b) in ElementType::Triangle.local_edges() {
        let reference = [
            0.5 * (reference_vertices[a][0] + reference_vertices[b][0]),
            0.5 * (reference_vertices[a][1] + reference_vertices[b][1]),
            0.5 * (reference_vertices[a][2] + reference_vertices[b][2]),
        ];
        let [l0, l1, l2] = reference;
        let weights = [
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ];
        let mut point = Point::new_3d(0.0, 0.0, 0.0);
        for (&node, weight) in parent.nodes.iter().take(6).zip(weights) {
            let source = mesh.nodes[node];
            point.x += weight * source.x;
            point.y += weight * source.y;
            point.z += weight * source.z;
        }
        nodes.push(midpoint_mgr.get_mapped_edge_point(mesh, vertices[a], vertices[b], point));
    }
    nodes
}

/// Build a quadratic child tetrahedron from parent-reference coordinates so
/// curved P2 tetrahedral traces survive adaptive refinement.
fn p2_tetra_child_nodes(
    mesh: &mut Mesh,
    midpoint_mgr: &mut MidpointManager,
    parent: &Element,
    vertices: [usize; 4],
    reference_vertices: [[f64; 4]; 4],
) -> Vec<usize> {
    debug_assert_eq!(parent.element_type, ElementType::Tetrahedron);
    debug_assert!(parent.nodes.len() >= ElementType::Tetrahedron.num_nodes_p2());

    let mut nodes = vertices.to_vec();
    for &(a, b) in ElementType::Tetrahedron.local_edges() {
        let reference = [
            0.5 * (reference_vertices[a][0] + reference_vertices[b][0]),
            0.5 * (reference_vertices[a][1] + reference_vertices[b][1]),
            0.5 * (reference_vertices[a][2] + reference_vertices[b][2]),
            0.5 * (reference_vertices[a][3] + reference_vertices[b][3]),
        ];
        let [l0, l1, l2, l3] = reference;
        let weights = [
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l0 * l1,
            4.0 * l0 * l2,
            4.0 * l0 * l3,
            4.0 * l1 * l2,
            4.0 * l1 * l3,
            4.0 * l2 * l3,
        ];
        let mut point = Point::new_3d(0.0, 0.0, 0.0);
        for (&node, weight) in parent.nodes.iter().take(10).zip(weights) {
            let source = mesh.nodes[node];
            point.x += weight * source.x;
            point.y += weight * source.y;
            point.z += weight * source.z;
        }
        nodes.push(midpoint_mgr.get_mapped_edge_point(mesh, vertices[a], vertices[b], point));
    }
    nodes
}

/// Return the physical point of a Q2 quadrilateral at a parent-reference
/// coordinate. Quad node order is vertices, edge midpoints, then centre.
fn q2_quadrilateral_point(mesh: &Mesh, parent: &Element, xi: f64, eta: f64) -> Point {
    debug_assert_eq!(parent.element_type, ElementType::Quadrilateral);
    debug_assert_eq!(
        parent.nodes.len(),
        ElementType::Quadrilateral.num_nodes_p2()
    );

    let l_x = [0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)];
    let l_y = [
        0.5 * eta * (eta - 1.0),
        1.0 - eta * eta,
        0.5 * eta * (eta + 1.0),
    ];
    let weights = [
        l_x[0] * l_y[0],
        l_x[2] * l_y[0],
        l_x[2] * l_y[2],
        l_x[0] * l_y[2],
        l_x[1] * l_y[0],
        l_x[2] * l_y[1],
        l_x[1] * l_y[2],
        l_x[0] * l_y[1],
        l_x[1] * l_y[1],
    ];
    let mut point = Point::new_3d(0.0, 0.0, 0.0);
    for (&node, &weight) in parent.nodes.iter().zip(weights.iter()) {
        point.x += weight * mesh.nodes[node].x;
        point.y += weight * mesh.nodes[node].y;
        point.z += weight * mesh.nodes[node].z;
    }
    point
}

/// Subdivide a Q2 quadrilateral into four Q2 children. The complete 5×5
/// reference lattice is evaluated through the parent map, preserving curved
/// geometry. Newly created parent-trace points are keyed by their bracketing
/// existing Q2 edge nodes so adjacent refined cells share the same trace DOF.
fn q2_quadrilateral_children(
    mesh: &mut Mesh,
    midpoint_mgr: &mut MidpointManager,
    parent: &Element,
) -> Vec<Vec<usize>> {
    let mut grid = [[usize::MAX; 5]; 5];
    // Parent Q2 nodes occupy the even lattice sites.
    grid[0][0] = parent.nodes[0];
    grid[0][4] = parent.nodes[1];
    grid[4][4] = parent.nodes[2];
    grid[4][0] = parent.nodes[3];
    grid[0][2] = parent.nodes[4];
    grid[2][4] = parent.nodes[5];
    grid[4][2] = parent.nodes[6];
    grid[2][0] = parent.nodes[7];
    grid[2][2] = parent.nodes[8];

    for j in 0..5 {
        for i in 0..5 {
            if grid[j][i] != usize::MAX {
                continue;
            }
            let point =
                q2_quadrilateral_point(mesh, parent, -1.0 + 0.5 * i as f64, -1.0 + 0.5 * j as f64);
            let trace_segment = if (j == 0 || j == 4) && i % 2 == 1 {
                Some((grid[j][i - 1], grid[j][i + 1]))
            } else if (i == 0 || i == 4) && j % 2 == 1 {
                Some((grid[j - 1][i], grid[j + 1][i]))
            } else {
                None
            };
            grid[j][i] = if let Some((a, b)) = trace_segment {
                midpoint_mgr.get_mapped_edge_point(mesh, a, b, point)
            } else {
                let node = mesh.add_node(point);
                midpoint_mgr.new_nodes.push(node);
                node
            };
        }
    }

    let mut children = Vec::with_capacity(4);
    for j in [0, 2] {
        for i in [0, 2] {
            children.push(vec![
                grid[j][i],
                grid[j][i + 2],
                grid[j + 2][i + 2],
                grid[j + 2][i],
                grid[j][i + 1],
                grid[j + 1][i + 2],
                grid[j + 2][i + 1],
                grid[j + 1][i],
                grid[j + 1][i + 1],
            ]);
        }
    }
    children
}

/// Cubic hexahedron node lattice in the canonical FEM order. Coordinates are
/// `0..=3` in the parent Q3 reference cube; the adaptive denominator-six
/// lattice is obtained by multiplying them by two.
const P3_HEX_NODE_LATTICE: [[usize; 3]; 64] = [
    [0, 0, 0],
    [3, 0, 0],
    [3, 3, 0],
    [0, 3, 0],
    [0, 0, 3],
    [3, 0, 3],
    [3, 3, 3],
    [0, 3, 3],
    [1, 0, 0],
    [2, 0, 0],
    [3, 1, 0],
    [3, 2, 0],
    [2, 3, 0],
    [1, 3, 0],
    [0, 2, 0],
    [0, 1, 0],
    [1, 0, 3],
    [2, 0, 3],
    [3, 1, 3],
    [3, 2, 3],
    [2, 3, 3],
    [1, 3, 3],
    [0, 2, 3],
    [0, 1, 3],
    [0, 0, 1],
    [0, 0, 2],
    [3, 0, 1],
    [3, 0, 2],
    [3, 3, 1],
    [3, 3, 2],
    [0, 3, 1],
    [0, 3, 2],
    [1, 1, 0],
    [2, 1, 0],
    [2, 2, 0],
    [1, 2, 0],
    [1, 1, 3],
    [2, 1, 3],
    [2, 2, 3],
    [1, 2, 3],
    [1, 0, 1],
    [2, 0, 1],
    [2, 0, 2],
    [1, 0, 2],
    [1, 3, 1],
    [2, 3, 1],
    [2, 3, 2],
    [1, 3, 2],
    [0, 1, 1],
    [0, 2, 1],
    [0, 2, 2],
    [0, 1, 2],
    [3, 1, 1],
    [3, 2, 1],
    [3, 2, 2],
    [3, 1, 2],
    [1, 1, 1],
    [2, 1, 1],
    [2, 2, 1],
    [1, 2, 1],
    [1, 1, 2],
    [2, 1, 2],
    [2, 2, 2],
    [1, 2, 2],
];

const P3_HEX_CORNER_LATTICE: [[usize; 3]; 8] = [
    [0, 0, 0],
    [6, 0, 0],
    [6, 6, 0],
    [0, 6, 0],
    [0, 0, 6],
    [6, 0, 6],
    [6, 6, 6],
    [0, 6, 6],
];

const P3_HEX_FACE_CORNERS: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum P3HexPointKey {
    Vertex(usize),
    Edge {
        vertices: [usize; 2],
        weights: [u8; 2],
    },
    Face {
        vertices: [usize; 4],
        weights: [u16; 4],
    },
    Cell {
        vertices: [usize; 8],
        lattice: [u8; 3],
    },
}

fn p3_hex_corner_index(point: [usize; 3]) -> usize {
    P3_HEX_CORNER_LATTICE
        .iter()
        .position(|corner| *corner == point)
        .expect("P3 hexahedron boundary point must have a corner")
}

fn p3_hex_point_key(parent: &Element, point: [usize; 3]) -> P3HexPointKey {
    let boundary_axes = point
        .iter()
        .filter(|&&value| value == 0 || value == 6)
        .count();
    match boundary_axes {
        3 => P3HexPointKey::Vertex(parent.nodes[p3_hex_corner_index(point)]),
        2 => {
            let free_axis = (0..3)
                .find(|&axis| point[axis] != 0 && point[axis] != 6)
                .expect("P3 hexahedron edge point must have one free axis");
            let mut low = point;
            let mut high = point;
            low[free_axis] = 0;
            high[free_axis] = 6;
            let mut entries = [
                (
                    parent.nodes[p3_hex_corner_index(low)],
                    (6 - point[free_axis]) as u8,
                ),
                (
                    parent.nodes[p3_hex_corner_index(high)],
                    point[free_axis] as u8,
                ),
            ];
            entries.sort_unstable_by_key(|(vertex, _)| *vertex);
            P3HexPointKey::Edge {
                vertices: [entries[0].0, entries[1].0],
                weights: [entries[0].1, entries[1].1],
            }
        }
        1 => {
            let corners = P3_HEX_FACE_CORNERS
                .iter()
                .find(|corners| {
                    (0..3).any(|axis| {
                        corners
                            .iter()
                            .all(|&corner| P3_HEX_CORNER_LATTICE[corner][axis] == point[axis])
                    })
                })
                .expect("P3 hexahedron face point must lie on a face");
            let fixed_axis = (0..3)
                .find(|&axis| {
                    corners
                        .iter()
                        .all(|&corner| P3_HEX_CORNER_LATTICE[corner][axis] == point[axis])
                })
                .expect("P3 hexahedron face fixes one axis");
            let u_axis = (0..3)
                .find(|&axis| {
                    P3_HEX_CORNER_LATTICE[corners[1]][axis]
                        != P3_HEX_CORNER_LATTICE[corners[0]][axis]
                })
                .expect("P3 hexahedron face has a first tangent axis");
            let v_axis = (0..3)
                .find(|&axis| {
                    P3_HEX_CORNER_LATTICE[corners[3]][axis]
                        != P3_HEX_CORNER_LATTICE[corners[0]][axis]
                })
                .expect("P3 hexahedron face has a second tangent axis");
            debug_assert_ne!(fixed_axis, u_axis);
            debug_assert_ne!(fixed_axis, v_axis);
            let u = point[u_axis] as u16;
            let v = point[v_axis] as u16;
            let weights = [(6 - u) * (6 - v), u * (6 - v), u * v, (6 - u) * v];
            let mut entries = [
                (parent.nodes[corners[0]], weights[0]),
                (parent.nodes[corners[1]], weights[1]),
                (parent.nodes[corners[2]], weights[2]),
                (parent.nodes[corners[3]], weights[3]),
            ];
            entries.sort_unstable_by_key(|(vertex, _)| *vertex);
            P3HexPointKey::Face {
                vertices: entries.map(|(vertex, _)| vertex),
                weights: entries.map(|(_, weight)| weight),
            }
        }
        _ => {
            let mut vertices = [0usize; 8];
            vertices.copy_from_slice(&parent.nodes[..8]);
            vertices.sort_unstable();
            P3HexPointKey::Cell {
                vertices,
                lattice: point.map(|value| value as u8),
            }
        }
    }
}

struct P3HexPointManager {
    points: HashMap<P3HexPointKey, usize>,
    new_nodes: Vec<usize>,
}

impl P3HexPointManager {
    fn new() -> Self {
        Self {
            points: HashMap::new(),
            new_nodes: Vec::new(),
        }
    }

    fn seed_parent(&mut self, parent: &Element) {
        for (node, lattice) in parent.nodes.iter().copied().zip(P3_HEX_NODE_LATTICE) {
            self.points
                .entry(p3_hex_point_key(parent, lattice.map(|value| value * 2)))
                .or_insert(node);
        }
    }

    fn node_for(&mut self, mesh: &mut Mesh, parent: &Element, point: [usize; 3]) -> usize {
        let key = p3_hex_point_key(parent, point);
        if let Some(&node) = self.points.get(&key) {
            return node;
        }
        let node = mesh.add_node(p3_hexahedron_point(mesh, parent, point));
        self.points.insert(key, node);
        self.new_nodes.push(node);
        node
    }
}

fn p3_hexahedron_point(mesh: &Mesh, parent: &Element, lattice: [usize; 3]) -> Point {
    debug_assert_eq!(parent.element_type, ElementType::Hexahedron);
    debug_assert_eq!(parent.nodes.len(), ElementType::Hexahedron.num_nodes_p3());
    let values = lattice.map(|value| q3_lagrange_values(-1.0 + value as f64 / 3.0));
    let mut point = Point::new_3d(0.0, 0.0, 0.0);
    for (&node, &[i, j, k]) in parent.nodes.iter().zip(P3_HEX_NODE_LATTICE.iter()) {
        let weight = values[0][i] * values[1][j] * values[2][k];
        point.x += weight * mesh.nodes[node].x;
        point.y += weight * mesh.nodes[node].y;
        point.z += weight * mesh.nodes[node].z;
    }
    point
}

fn p3_hexahedron_child_nodes(
    mesh: &mut Mesh,
    manager: &mut P3HexPointManager,
    parent: &Element,
    base: [usize; 3],
) -> Vec<usize> {
    P3_HEX_NODE_LATTICE
        .map(|[i, j, k]| manager.node_for(mesh, parent, [base[0] + i, base[1] + j, base[2] + k]))
        .to_vec()
}

fn refine_p3_hexahedron_elements(
    mesh: &mut Mesh,
    elements_to_refine: &[usize],
) -> Result<RefinementResult, RefinementError> {
    let mut point_manager = P3HexPointManager::new();
    let original_boundaries = mesh.boundaries.clone();
    let mut child_elements = Vec::with_capacity(elements_to_refine.len() * 8);

    for &element_index in elements_to_refine {
        let parent = mesh.elements[element_index].clone();
        point_manager.seed_parent(&parent);
        for i in [0usize, 3] {
            for j in [0usize, 3] {
                for k in [0usize, 3] {
                    let nodes =
                        p3_hexahedron_child_nodes(mesh, &mut point_manager, &parent, [i, j, k]);
                    let mut child =
                        Element::new(ElementType::Hexahedron, nodes, mesh.next_element_id());
                    child.parent_id = Some(parent.id);
                    child.level = parent.level + 1;
                    child.marker = parent.marker;
                    child_elements.push(child);
                }
            }
        }
    }

    let removed_set: HashSet<usize> = elements_to_refine.iter().copied().collect();
    let mut old_to_new = HashMap::with_capacity(mesh.elements.len() - removed_set.len());
    let mut retained =
        Vec::with_capacity(mesh.elements.len() - removed_set.len() + child_elements.len());
    let mut removed_elements = Vec::with_capacity(removed_set.len());
    for (index, element) in mesh.elements.iter().cloned().enumerate() {
        if removed_set.contains(&index) {
            removed_elements.push(index);
        } else {
            old_to_new.insert(index, retained.len());
            retained.push(element);
        }
    }
    let first_child_index = retained.len();
    retained.extend(child_elements);

    let mut boundaries = Vec::with_capacity(original_boundaries.len());
    for boundary in original_boundaries {
        if removed_set.contains(&boundary.element_idx) {
            continue;
        }
        let Some(&element_idx) = old_to_new.get(&boundary.element_idx) else {
            return Err(RefinementError::InvalidHistory {
                parent_id: 0,
                reason: "boundary references an element removed during P3 hexahedral refinement",
            });
        };
        let mut remapped = boundary;
        remapped.element_idx = element_idx;
        boundaries.push(remapped);
    }

    mesh.elements = retained;
    mesh.boundaries = boundaries;
    mesh.invalidate_caches();
    Ok(RefinementResult {
        new_elements: (first_child_index..mesh.elements.len()).collect(),
        removed_elements,
        new_nodes: point_manager.new_nodes,
    })
}

/// Hex27 reference lattice locations in the element-node order used by
/// [`ElementType::Hexahedron`]. Each coordinate is `0`, `2`, or `4` on the
/// parent 5×5×5 refinement lattice.
const HEX27_PARENT_LATTICE: [[usize; 3]; 27] = [
    [0, 0, 0],
    [4, 0, 0],
    [4, 4, 0],
    [0, 4, 0],
    [0, 0, 4],
    [4, 0, 4],
    [4, 4, 4],
    [0, 4, 4], // vertices
    [2, 0, 0],
    [4, 2, 0],
    [2, 4, 0],
    [0, 2, 0], // bottom edges
    [2, 0, 4],
    [4, 2, 4],
    [2, 4, 4],
    [0, 2, 4], // top edges
    [0, 0, 2],
    [4, 0, 2],
    [4, 4, 2],
    [0, 4, 2], // vertical edges
    [2, 2, 0],
    [2, 2, 4],
    [2, 0, 2],
    [2, 4, 2],
    [0, 2, 2],
    [4, 2, 2],
    [2, 2, 2], // face centres and volume centre
];

fn q2_lagrange_values(coordinate: f64) -> [f64; 3] {
    [
        0.5 * coordinate * (coordinate - 1.0),
        1.0 - coordinate * coordinate,
        0.5 * coordinate * (coordinate + 1.0),
    ]
}

/// Evaluate a Hex27 parent map at a point in its reference cube.
fn q2_hexahedron_point(mesh: &Mesh, parent: &Element, xi: f64, eta: f64, zeta: f64) -> Point {
    debug_assert_eq!(parent.element_type, ElementType::Hexahedron);
    debug_assert_eq!(parent.nodes.len(), ElementType::Hexahedron.num_nodes_p2());

    let lx = q2_lagrange_values(xi);
    let ly = q2_lagrange_values(eta);
    let lz = q2_lagrange_values(zeta);
    let mut point = Point::new_3d(0.0, 0.0, 0.0);
    for (&node, &[i, j, k]) in parent.nodes.iter().zip(HEX27_PARENT_LATTICE.iter()) {
        let weight = lx[i / 2] * ly[j / 2] * lz[k / 2];
        point.x += weight * mesh.nodes[node].x;
        point.y += weight * mesh.nodes[node].y;
        point.z += weight * mesh.nodes[node].z;
    }
    point
}

fn lattice_offset(point: usize, from: usize, to: usize) -> usize {
    if to >= from {
        point - from
    } else {
        from - point
    }
}

/// Canonical sparse Q2 interpolation stencils for every parent face that
/// contains a 5×5×5 lattice point. Sorting by global trace-node ID makes the
/// resulting keys invariant under neighbouring local-face rotations.
fn q2_hexahedron_face_point_keys(parent: &Element, point: [usize; 3]) -> Vec<Vec<(usize, i16)>> {
    let face_entities = parent.boundary_entities();
    let mut keys = Vec::new();

    for (face_index, face) in ElementType::Hexahedron.local_faces().iter().enumerate() {
        let corners = [
            HEX27_PARENT_LATTICE[face[0]],
            HEX27_PARENT_LATTICE[face[1]],
            HEX27_PARENT_LATTICE[face[2]],
            HEX27_PARENT_LATTICE[face[3]],
        ];
        let fixed_axis = (0..3)
            .find(|&axis| {
                corners
                    .iter()
                    .all(|corner| corner[axis] == corners[0][axis])
            })
            .expect("a hexahedron face fixes exactly one reference axis");
        if point[fixed_axis] != corners[0][fixed_axis] {
            continue;
        }

        let u_axis = (0..3)
            .find(|&axis| corners[1][axis] != corners[0][axis])
            .expect("a hexahedron face has a first tangential axis");
        let v_axis = (0..3)
            .find(|&axis| corners[3][axis] != corners[0][axis])
            .expect("a hexahedron face has a second tangential axis");
        let u = lattice_offset(point[u_axis], corners[0][u_axis], corners[1][u_axis]);
        let v = lattice_offset(point[v_axis], corners[0][v_axis], corners[3][v_axis]);
        let lu = q2_lagrange_integer_weights(u);
        let lv = q2_lagrange_integer_weights(v);
        let weights = [
            lu[0] * lv[0],
            lu[2] * lv[0],
            lu[2] * lv[2],
            lu[0] * lv[2],
            lu[1] * lv[0],
            lu[2] * lv[1],
            lu[1] * lv[2],
            lu[0] * lv[1],
            lu[1] * lv[1],
        ];
        let mut key: Vec<_> = face_entities[face_index]
            .iter()
            .copied()
            .zip(weights)
            .filter(|(_, weight)| *weight != 0)
            .collect();
        key.sort_unstable();
        keys.push(key);
    }
    keys
}

/// Quadratic Lagrange weights at the five parent refinement-lattice sites,
/// scaled by eight. Integer stencils make face-node keys exact.
fn q2_lagrange_integer_weights(index: usize) -> [i16; 3] {
    match index {
        0 => [8, 0, 0],
        1 => [3, 6, -1],
        2 => [0, 8, 0],
        3 => [-1, 6, 3],
        4 => [0, 0, 8],
        _ => unreachable!("Q2 lattice index must be in 0..=4"),
    }
}

/// Subdivide a Hex27 parent into eight Hex27 children using a complete 5×5×5
/// parent reference lattice. Points are evaluated through the Q2 map rather
/// than by linear interpolation, preserving curved faces and volume geometry.
fn q2_hexahedron_children(
    mesh: &mut Mesh,
    midpoint_mgr: &mut MidpointManager,
    parent: &Element,
) -> Vec<Vec<usize>> {
    debug_assert_eq!(parent.element_type, ElementType::Hexahedron);
    let mut grid = [[[usize::MAX; 5]; 5]; 5];
    for (&node, &[i, j, k]) in parent.nodes.iter().zip(HEX27_PARENT_LATTICE.iter()) {
        grid[k][j][i] = node;
    }

    for (k, plane) in grid.iter_mut().enumerate() {
        for (j, row) in plane.iter_mut().enumerate() {
            for (i, node) in row.iter_mut().enumerate() {
                if *node != usize::MAX {
                    continue;
                }
                let point = q2_hexahedron_point(
                    mesh,
                    parent,
                    -1.0 + 0.5 * i as f64,
                    -1.0 + 0.5 * j as f64,
                    -1.0 + 0.5 * k as f64,
                );
                let keys = q2_hexahedron_face_point_keys(parent, [i, j, k]);
                *node = if keys.is_empty() {
                    let node = mesh.add_node(point);
                    midpoint_mgr.new_nodes.push(node);
                    node
                } else {
                    midpoint_mgr.get_mapped_face_point(mesh, keys, point)
                };
            }
        }
    }

    let mut children = Vec::with_capacity(8);
    for k in [0, 2] {
        for j in [0, 2] {
            for i in [0, 2] {
                let local = HEX27_PARENT_LATTICE.map(|[x, y, z]| [i + x / 2, j + y / 2, k + z / 2]);
                children.push(local.into_iter().map(|[x, y, z]| grid[z][y][x]).collect());
            }
        }
    }
    children
}

fn p1_tetra_children(
    mesh: &Mesh,
    midpoint_mgr: &mut MidpointManager,
    vertices: [usize; 4],
) -> [[usize; 4]; 8] {
    let [v0, v1, v2, v3] = vertices;
    let m01 = midpoint_mgr
        .edge_midpoints
        .get(&Edge::new(v0, v1))
        .copied()
        .unwrap_or_else(|| unreachable!("seeded P2 edge midpoint is missing"));
    let m02 = midpoint_mgr
        .edge_midpoints
        .get(&Edge::new(v0, v2))
        .copied()
        .unwrap_or_else(|| unreachable!("seeded P2 edge midpoint is missing"));
    let m03 = midpoint_mgr
        .edge_midpoints
        .get(&Edge::new(v0, v3))
        .copied()
        .unwrap_or_else(|| unreachable!("seeded P2 edge midpoint is missing"));
    let m12 = midpoint_mgr
        .edge_midpoints
        .get(&Edge::new(v1, v2))
        .copied()
        .unwrap_or_else(|| unreachable!("seeded P2 edge midpoint is missing"));
    let m13 = midpoint_mgr
        .edge_midpoints
        .get(&Edge::new(v1, v3))
        .copied()
        .unwrap_or_else(|| unreachable!("seeded P2 edge midpoint is missing"));
    let m23 = midpoint_mgr
        .edge_midpoints
        .get(&Edge::new(v2, v3))
        .copied()
        .unwrap_or_else(|| unreachable!("seeded P2 edge midpoint is missing"));

    let len_sq = |a: usize, b: usize| {
        let pa = mesh.nodes[a];
        let pb = mesh.nodes[b];
        let dx = pa.x - pb.x;
        let dy = pa.y - pb.y;
        let dz = pa.z - pb.z;
        dx * dx + dy * dy + dz * dz
    };
    let d01_23 = len_sq(m01, m23);
    let d02_13 = len_sq(m02, m13);
    let d03_12 = len_sq(m03, m12);
    let inner = if d02_13 <= d01_23 && d02_13 <= d03_12 {
        [
            [m01, m02, m03, m13],
            [m01, m02, m12, m13],
            [m02, m03, m13, m23],
            [m02, m12, m13, m23],
        ]
    } else if d01_23 <= d03_12 {
        [
            [m01, m23, m02, m03],
            [m01, m23, m12, m13],
            [m01, m23, m02, m12],
            [m01, m23, m03, m13],
        ]
    } else {
        [
            [m03, m12, m01, m02],
            [m03, m12, m01, m13],
            [m03, m12, m02, m23],
            [m03, m12, m13, m23],
        ]
    };
    let children = [
        [v0, m01, m02, m03],
        [m01, v1, m12, m13],
        [m02, m12, v2, m23],
        [m03, m13, m23, v3],
        inner[0],
        inner[1],
        inner[2],
        inner[3],
    ];
    children.map(|child| orient_tetra_vertices(mesh, child))
}

fn orient_tetra_vertices(mesh: &Mesh, mut vertices: [usize; 4]) -> [usize; 4] {
    let a = mesh.nodes[vertices[0]];
    let b = mesh.nodes[vertices[1]];
    let c = mesh.nodes[vertices[2]];
    let d = mesh.nodes[vertices[3]];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    let determinant = ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
        - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);
    if determinant < 0.0 {
        vertices.swap(2, 3);
    }
    vertices
}

fn validate_high_order_refinement_mesh(mesh: &Mesh) -> Result<(), RefinementError> {
    for (node, point) in mesh.nodes.iter().enumerate() {
        if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
            return Err(RefinementError::InvalidNode {
                node,
                reason: "mesh contains a non-finite node coordinate",
            });
        }
    }
    for (element, cell) in mesh.elements.iter().enumerate() {
        if cell.try_polynomial_order().is_err()
            || cell.nodes.iter().any(|&node| node >= mesh.nodes.len())
        {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "element connectivity does not describe a valid P1/P2 layout",
            });
        }
    }
    mesh.validate_connectivity()
        .map_err(|_| RefinementError::InvalidMesh {
            element: 0,
            reason: "mesh connectivity validation failed",
        })
}

fn validate_high_order_refined_mesh(mesh: &Mesh) -> Result<(), RefinementError> {
    validate_high_order_refinement_mesh(mesh)?;
    if let Some(&element) = check_orientation(mesh).first() {
        return Err(RefinementError::InvalidMesh {
            element,
            reason: "high-order refinement produced an inverted element",
        });
    }
    Ok(())
}

/// Refine elements where error exceeds threshold
pub fn adaptive_refine(
    mesh: &mut Mesh,
    element_errors: &[f64],
    threshold: f64,
) -> Result<RefinementResult, RefinementError> {
    let element_count = mesh.num_elements();
    if element_errors.len() != element_count {
        return Err(RefinementError::ErrorCountMismatch {
            elements: element_count,
            errors: element_errors.len(),
        });
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(RefinementError::InvalidThreshold(threshold));
    }
    if let Some((index, value)) = element_errors
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0)
    {
        return Err(RefinementError::InvalidElementError { index, value });
    }

    let elements_to_refine: Vec<usize> = element_errors
        .iter()
        .enumerate()
        .filter(|(_, e)| **e > threshold)
        .map(|(i, _)| i)
        .collect();

    refine_elements_checked(mesh, &elements_to_refine)
}

fn validate_refinement_mesh(mesh: &Mesh) -> Result<(), RefinementError> {
    if let Some(node) = mesh
        .nodes
        .iter()
        .position(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
    {
        return Err(RefinementError::InvalidNode {
            node,
            reason: "mesh contains a non-finite node coordinate",
        });
    }
    for (element, cell) in mesh.elements.iter().enumerate() {
        if cell.nodes.len() < cell.element_type.num_vertices() {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "element has too few node indices",
            });
        }
        let order = cell
            .try_polynomial_order()
            .map_err(|_| RefinementError::InvalidMesh {
                element,
                reason: "element node count does not describe a supported P1/P2 layout",
            })?;
        if order != 1 {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "checked refinement of higher-order elements is unsupported",
            });
        }
        if cell.nodes.iter().any(|&node| node >= mesh.nodes.len()) {
            return Err(RefinementError::InvalidMesh {
                element,
                reason: "element references a node outside the mesh",
            });
        }
    }
    if mesh.validate_connectivity().is_err() {
        return Err(RefinementError::InvalidMesh {
            element: 0,
            reason: "mesh connectivity validation failed",
        });
    }
    Ok(())
}

/// Validate the result before a checked refinement operation commits it.
/// Keeping this separate from the input check makes the checked APIs
/// transactional: a failed postcondition leaves the caller's mesh untouched.
fn validate_refined_mesh(mesh: &Mesh) -> Result<(), RefinementError> {
    validate_refinement_mesh(mesh)?;
    if let Some(&element) = check_orientation(mesh).first() {
        return Err(RefinementError::InvalidMesh {
            element,
            reason: "refinement produced an inverted element",
        });
    }
    Ok(())
}

/// Mark elements for refinement based on Doerfler marking strategy
///
/// Mark enough elements to capture a fraction theta of the total error
pub fn doerfler_marking(element_errors: &[f64], theta: f64) -> Result<Vec<usize>, String> {
    if !theta.is_finite() || !(0.0..=1.0).contains(&theta) {
        return Err("Doerfler theta must be finite and within [0, 1]".into());
    }
    if element_errors
        .iter()
        .any(|error| !error.is_finite() || *error < 0.0)
    {
        return Err("element errors must be finite and non-negative".into());
    }
    let total_error_sq: f64 = element_errors.iter().map(|e| e * e).sum();
    let target = theta * total_error_sq;

    // Sort elements by error (descending)
    let mut indexed_errors: Vec<(usize, f64)> = element_errors
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();
    indexed_errors.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut marked = Vec::new();
    let mut accumulated = 0.0;

    for (idx, error) in indexed_errors {
        marked.push(idx);
        accumulated += error * error;
        if accumulated >= target {
            break;
        }
    }

    Ok(marked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generators::{
        box_mesh_hexahedra, rectangular_mesh_quads, unit_cube_hexahedra, unit_cube_tetrahedra,
        unit_square_quads, unit_square_triangles,
    };
    use crate::mesh::promote_to_p2;
    use crate::mesh::types::BoundaryType;

    #[test]
    fn test_triangle_refinement() {
        let mut mesh = unit_square_triangles(1);
        assert_eq!(mesh.num_elements(), 2);

        let result = uniform_refine(&mut mesh);

        // Each triangle becomes 4
        assert_eq!(result.new_elements.len(), 8);
        assert_eq!(result.removed_elements.len(), 2);
        assert_eq!(mesh.num_elements(), 8);
        assert!(mesh.elements.iter().all(|elem| elem.parent_id.is_some()));
    }

    #[test]
    fn checked_p2_triangle_refinement_preserves_trace_nodes_and_markers() {
        let mut p1 = unit_square_triangles(1);
        p1.detect_boundaries();
        p1.elements[0].marker = 7;
        let mut mesh = promote_to_p2(&p1);
        assert!(mesh.validate_connectivity().is_ok());

        let result = refine_elements_high_order_checked(&mut mesh, &[0]).unwrap();

        assert_eq!(result.new_elements.len(), 4);
        assert_eq!(result.removed_elements, vec![0]);
        assert!(
            mesh.elements
                .iter()
                .filter(|element| element.parent_id == Some(p1.elements[0].id))
                .all(|element| { element.try_polynomial_order() == Ok(2) && element.marker == 7 })
        );
        assert!(
            mesh.boundaries
                .iter()
                .all(|boundary| boundary.nodes.len() == 3)
        );
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_curved_p2_triangle_refinement_evaluates_parent_geometry() {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(0.5, 0.2),
            Point::new_2d(0.5, 0.5),
            Point::new_2d(0.0, 0.5),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Triangle, (0..6).collect());

        let result = refine_elements_high_order_checked(&mut mesh, &[0]).unwrap();
        assert_eq!(result.new_elements.len(), 4);
        assert!(result.new_nodes.iter().any(|&node| {
            (mesh.nodes[node].x - 0.25).abs() < 1.0e-12
                && (mesh.nodes[node].y - 0.15).abs() < 1.0e-12
        }));
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_curved_p2_triangle_refinement_reuses_shared_mapped_edges() {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(1.0, 1.0),
            Point::new_2d(0.5, 0.0),
            Point::new_2d(0.55, 0.5),
            Point::new_2d(0.0, 0.5),
            Point::new_2d(1.0, 0.5),
            Point::new_2d(0.5, 1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2, 4, 5, 6]);
        mesh.add_element(ElementType::Triangle, vec![1, 3, 2, 7, 8, 5]);

        let result = refine_elements_high_order_checked(&mut mesh, &[0, 1]).unwrap();
        assert_eq!(result.new_elements.len(), 8);
        // Nine mapped edges per child family, minus the two subdivisions of
        // the shared parent edge reused by the adjacent triangle.
        assert_eq!(result.new_nodes.len(), 16);
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_curved_p2_tetrahedron_refinement_evaluates_parent_geometry() {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
            Point::new_3d(0.5, 0.2, 0.0),
            Point::new_3d(0.0, 0.5, 0.0),
            Point::new_3d(0.0, 0.0, 0.5),
            Point::new_3d(0.5, 0.5, 0.0),
            Point::new_3d(0.5, 0.0, 0.5),
            Point::new_3d(0.0, 0.5, 0.5),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, (0..10).collect());

        let result = refine_elements_high_order_checked(&mut mesh, &[0]).unwrap();
        assert_eq!(result.new_elements.len(), 8);
        assert!(result.new_nodes.iter().any(|&node| {
            (mesh.nodes[node].x - 0.25).abs() < 1.0e-12
                && (mesh.nodes[node].y - 0.15).abs() < 1.0e-12
                && mesh.nodes[node].z.abs() < 1.0e-12
        }));
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_p2_tetrahedron_refinement_preserves_quadratic_layout() {
        let mut p1 = unit_cube_tetrahedra(1);
        p1.detect_boundaries();
        let mut mesh = promote_to_p2(&p1);
        assert!(mesh.validate_connectivity().is_ok());

        let result = refine_elements_high_order_checked(&mut mesh, &[0]).unwrap();

        assert_eq!(result.new_elements.len(), 8);
        assert!(
            mesh.elements
                .iter()
                .filter(|element| element.parent_id == Some(p1.elements[0].id))
                .all(|element| element.try_polynomial_order() == Ok(2))
        );
        assert!(
            mesh.boundaries
                .iter()
                .all(|boundary| boundary.nodes.len() == 6)
        );
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_q2_quadrilateral_refinement_preserves_curved_geometry_and_markers() {
        let mut p1 = unit_square_quads(1);
        p1.detect_boundaries();
        p1.elements[0].marker = 17;
        let mut mesh = promote_to_p2(&p1);
        // Curve the parent bottom edge. A linear midpoint subdivision would
        // place the new quarter-point at y=0.1; the Q2 map requires y=0.15.
        mesh.nodes[4].y = 0.2;

        let result = refine_elements_high_order_checked(&mut mesh, &[0]).unwrap();

        assert_eq!(result.new_elements.len(), 4);
        assert_eq!(result.removed_elements, vec![0]);
        assert_eq!(result.new_nodes.len(), 16);
        assert!(mesh.elements.iter().all(|element| {
            element.try_polynomial_order() == Ok(2)
                && element.marker == 17
                && element.parent_id == Some(p1.elements[0].id)
        }));
        assert!(
            mesh.boundaries
                .iter()
                .all(|boundary| boundary.nodes.len() == 3)
        );
        assert!(result.new_nodes.iter().any(|&node| {
            (mesh.nodes[node].x - 0.25).abs() < 1e-12 && (mesh.nodes[node].y - 0.15).abs() < 1e-12
        }));
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_q2_quadrilateral_refinement_reuses_shared_trace_nodes() {
        let mut p1 = rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, 2, 1);
        p1.detect_boundaries();
        let mut mesh = promote_to_p2(&p1);

        let result = refine_elements_high_order_checked(&mut mesh, &[1, 0]).unwrap();

        // Two Q2 parents contribute 25 lattice nodes each and share five
        // nodes on their common Q2 trace. The two newly created quarter-trace
        // nodes must therefore be reused rather than duplicated.
        assert_eq!(result.new_elements.len(), 8);
        assert_eq!(mesh.num_nodes(), 45);
        assert_eq!(result.new_nodes.len(), 30);
        assert_eq!(
            mesh.nodes
                .iter()
                .filter(|point| (point.x - 0.5).abs() < 1e-12)
                .count(),
            5
        );
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_hex27_refinement_preserves_curved_geometry_and_boundary_markers() {
        let mut p1 = unit_cube_hexahedra(1);
        p1.detect_boundaries();
        p1.elements[0].marker = 23;
        let mut mesh = promote_to_p2(&p1);
        // Curve the parent bottom face through its Q2 face-centre control
        // point. Its (-0.5, -0.5, -1) child-lattice point has a 9/16
        // contribution from this control point.
        mesh.nodes[mesh.elements[0].nodes[20]].z = -0.2;

        let result = refine_elements_high_order_checked(&mut mesh, &[0]).unwrap();
        assert_eq!(result.new_elements.len(), 8);
        assert_eq!(result.removed_elements, vec![0]);
        assert_eq!(result.new_nodes.len(), 98);
        assert_eq!(mesh.num_nodes(), 125);
        assert!(mesh.elements.iter().all(|element| {
            element.try_polynomial_order() == Ok(2)
                && element.marker == 23
                && element.parent_id == Some(p1.elements[0].id)
        }));
        assert!(
            mesh.boundaries
                .iter()
                .all(|boundary| boundary.nodes.len() == 9)
        );
        assert!(result.new_nodes.iter().any(|&node| {
            let point = mesh.nodes[node];
            (point.x - 0.25).abs() < 1e-12
                && (point.y - 0.25).abs() < 1e-12
                && (point.z + 0.1125).abs() < 1e-12
        }));
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_p2_refinement_history_round_trips_through_coarsening() {
        let mut mesh = promote_to_p2(&unit_square_triangles(1));
        let original = mesh.clone();
        let (result, history) =
            refine_elements_high_order_with_history_checked(&mut mesh, &[0]).unwrap();
        assert_eq!(result.new_elements.len(), 4);
        assert_eq!(history.parents.len(), 1);
        let parent_id = history.parents[0].element.id;
        let coarsening = coarsen_elements_checked(&mut mesh, &history, &[parent_id]).unwrap();
        assert_eq!(coarsening.removed_elements.len(), 4);
        assert_eq!(mesh.elements.len(), original.elements.len());
        assert_eq!(mesh.boundaries.len(), original.boundaries.len());
        let mut actual_elements = mesh
            .elements
            .iter()
            .map(|element| (element.id, element.element_type, element.nodes.clone()))
            .collect::<Vec<_>>();
        let mut original_elements = original
            .elements
            .iter()
            .map(|element| (element.id, element.element_type, element.nodes.clone()))
            .collect::<Vec<_>>();
        actual_elements.sort_by_key(|element| element.0);
        original_elements.sort_by_key(|element| element.0);
        assert_eq!(actual_elements, original_elements);
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_p3_triangle_refinement_history_round_trips_through_coarsening() {
        let mut mesh = Mesh::new(2);
        let points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0 / 3.0, 0.0),
            (2.0 / 3.0, 0.0),
            (2.0 / 3.0, 1.0 / 3.0),
            (1.0 / 3.0, 2.0 / 3.0),
            (0.0, 2.0 / 3.0),
            (0.0, 1.0 / 3.0),
            (1.0 / 3.0, 1.0 / 3.0),
        ];
        for (x, y) in points {
            mesh.add_node(Point::new_2d(x, y));
        }
        mesh.add_element(ElementType::Triangle, (0..10).collect());
        let original = mesh.clone();

        let (result, history) =
            refine_elements_high_order_with_history_checked(&mut mesh, &[0]).unwrap();
        assert_eq!(result.new_elements.len(), 4);
        assert_eq!(result.new_nodes.len(), 18);
        assert!(
            mesh.elements
                .iter()
                .all(|element| element.try_polynomial_order() == Ok(3))
        );
        let parent_id = history.parents[0].element.id;
        let coarsening = coarsen_elements_checked(&mut mesh, &history, &[parent_id]).unwrap();
        assert_eq!(coarsening.removed_elements.len(), 4);
        assert_eq!(mesh.elements.len(), original.elements.len());
        let mut actual = mesh
            .elements
            .iter()
            .map(|element| (element.id, element.element_type, element.nodes.clone()))
            .collect::<Vec<_>>();
        let mut expected = original
            .elements
            .iter()
            .map(|element| (element.id, element.element_type, element.nodes.clone()))
            .collect::<Vec<_>>();
        actual.sort_by_key(|element| element.0);
        expected.sort_by_key(|element| element.0);
        assert_eq!(actual, expected);
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_p3_quadrilateral_refinement_history_round_trips_through_coarsening() {
        let mut mesh = Mesh::new(2);
        let points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0 / 3.0, 0.0),
            (2.0 / 3.0, 0.0),
            (1.0, 1.0 / 3.0),
            (1.0, 2.0 / 3.0),
            (2.0 / 3.0, 1.0),
            (1.0 / 3.0, 1.0),
            (0.0, 2.0 / 3.0),
            (0.0, 1.0 / 3.0),
            (1.0 / 3.0, 1.0 / 3.0),
            (2.0 / 3.0, 1.0 / 3.0),
            (2.0 / 3.0, 2.0 / 3.0),
            (1.0 / 3.0, 2.0 / 3.0),
        ];
        for (x, y) in points {
            mesh.add_node(Point::new_2d(x, y));
        }
        mesh.add_element(ElementType::Quadrilateral, (0..16).collect());
        let original = mesh.clone();

        let (result, history) =
            refine_elements_high_order_with_history_checked(&mut mesh, &[0]).unwrap();
        assert_eq!(result.new_elements.len(), 4);
        assert_eq!(result.new_nodes.len(), 33);
        assert!(
            mesh.elements
                .iter()
                .all(|element| element.try_polynomial_order() == Ok(3))
        );
        let parent_id = history.parents[0].element.id;
        let coarsening = coarsen_elements_checked(&mut mesh, &history, &[parent_id]).unwrap();
        assert_eq!(coarsening.removed_elements.len(), 4);
        assert_eq!(mesh.elements.len(), original.elements.len());
        assert_eq!(mesh.elements[0].nodes, original.elements[0].nodes);
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_hex27_refinement_reuses_shared_face_nodes() {
        let mut p1 = box_mesh_hexahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1);
        p1.detect_boundaries();
        let mut mesh = promote_to_p2(&p1);

        let result = refine_elements_high_order_checked(&mut mesh, &[1, 0]).unwrap();
        assert_eq!(result.new_elements.len(), 16);
        // Two 5³ lattices share a complete 5×5 Q2 parent trace.
        assert_eq!(mesh.num_nodes(), 225);
        assert_eq!(result.new_nodes.len(), 180);
        assert_eq!(
            mesh.nodes
                .iter()
                .filter(|point| (point.x - 0.5).abs() < 1e-12)
                .count(),
            25
        );
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn test_doerfler_marking() {
        let errors = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let marked = doerfler_marking(&errors, 0.5).unwrap();

        // Should mark element 3 (error 0.8) first, then 1 (0.5), etc.
        assert!(!marked.is_empty());
        assert!(marked.contains(&3)); // Highest error
    }

    #[test]
    fn test_adaptive_refine() {
        let mut mesh = unit_square_triangles(1);
        let errors = vec![0.5, 0.1]; // Only refine first element

        let result = adaptive_refine(&mut mesh, &errors, 0.3).expect("valid refinement input");

        assert_eq!(result.removed_elements.len(), 1);
        assert_eq!(result.new_elements.len(), 4);
    }

    #[test]
    fn adaptive_refine_rejects_invalid_error_contracts() {
        let mut mesh = unit_square_triangles(1);

        assert!(matches!(
            adaptive_refine(&mut mesh, &[0.1], 0.05),
            Err(RefinementError::ErrorCountMismatch {
                elements: 2,
                errors: 1,
            })
        ));
        assert!(matches!(
            adaptive_refine(&mut mesh, &[0.1, 0.2], f64::NAN),
            Err(RefinementError::InvalidThreshold(value)) if value.is_nan()
        ));
        assert!(matches!(
            adaptive_refine(&mut mesh, &[0.1, f64::INFINITY], 0.05),
            Err(RefinementError::InvalidElementError { index: 1, value })
                if value.is_infinite()
        ));
        assert!(matches!(
            adaptive_refine(&mut mesh, &[0.1, -0.2], 0.05),
            Err(RefinementError::InvalidElementError {
                index: 1,
                value: -0.2,
            })
        ));
        assert_eq!(mesh.num_elements(), 2);
    }

    #[test]
    fn checked_refinement_rejects_malformed_mesh_without_mutation() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].nodes.truncate(2);
        let original_nodes = mesh.nodes.clone();
        let original_elements: Vec<_> = mesh
            .elements
            .iter()
            .map(|element| (element.element_type, element.nodes.clone(), element.id))
            .collect();

        assert!(matches!(
            adaptive_refine(&mut mesh, &[1.0, 0.0], 0.5),
            Err(RefinementError::InvalidMesh {
                element: 0,
                reason: "element has too few node indices",
            })
        ));
        assert_eq!(mesh.nodes, original_nodes);
        assert_eq!(
            mesh.elements
                .iter()
                .map(|element| (element.element_type, element.nodes.clone(), element.id))
                .collect::<Vec<_>>(),
            original_elements
        );
    }

    #[test]
    fn checked_refinement_rejects_invalid_selection_without_mutation() {
        let mut mesh = unit_square_triangles(1);
        let original_nodes = mesh.nodes.clone();
        let original_elements: Vec<_> = mesh
            .elements
            .iter()
            .map(|element| (element.element_type, element.nodes.clone(), element.id))
            .collect();
        let invalid_element = mesh.num_elements();

        assert!(matches!(
            refine_elements_checked(&mut mesh, &[invalid_element]),
            Err(RefinementError::InvalidMesh {
                reason: "selected element index is outside the mesh",
                ..
            })
        ));
        assert_eq!(mesh.nodes, original_nodes);
        assert_eq!(
            mesh.elements
                .iter()
                .map(|element| (element.element_type, element.nodes.clone(), element.id))
                .collect::<Vec<_>>(),
            original_elements
        );
    }

    #[test]
    fn checked_refinement_rejects_invalid_boundary_without_mutation() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        mesh.boundaries[0].local_idx = 99;
        let original = mesh.clone();

        let error = match refine_elements_checked(&mut mesh, &[0]) {
            Ok(_) => panic!("invalid boundary must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            RefinementError::InvalidMesh {
                reason: "mesh connectivity validation failed",
                ..
            }
        ));
        assert_eq!(mesh.nodes, original.nodes);
        assert_eq!(
            mesh.elements
                .iter()
                .map(|element| (element.element_type, element.nodes.clone(), element.id))
                .collect::<Vec<_>>(),
            original
                .elements
                .iter()
                .map(|element| (element.element_type, element.nodes.clone(), element.id))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            mesh.boundaries[0].local_idx,
            original.boundaries[0].local_idx
        );
    }

    #[test]
    fn checked_refinement_rejects_higher_order_without_mutation() {
        let source = unit_square_triangles(1);
        let mut mesh = source.to_p2();
        let original = mesh.clone();

        assert!(matches!(
            refine_elements_checked(&mut mesh, &[0]),
            Err(RefinementError::InvalidMesh {
                element: 0,
                reason: "checked refinement of higher-order elements is unsupported",
            })
        ));
        assert_eq!(mesh.nodes, original.nodes);
        assert_eq!(mesh.elements.len(), original.elements.len());
        for (actual, expected) in mesh.elements.iter().zip(original.elements.iter()) {
            assert_eq!(actual.element_type, expected.element_type);
            assert_eq!(actual.nodes, expected.nodes);
            assert_eq!(actual.id, expected.id);
            assert_eq!(actual.parent_id, expected.parent_id);
            assert_eq!(actual.level, expected.level);
            assert_eq!(actual.marker, expected.marker);
        }
    }

    #[test]
    fn checked_refinement_commits_only_valid_oriented_connectivity() {
        let mut mesh = unit_square_triangles(1);
        refine_elements_checked(&mut mesh, &[0]).expect("valid refinement");

        assert!(mesh.validate_connectivity().is_ok());
        assert!(check_orientation(&mesh).is_empty());
    }

    #[test]
    fn checked_refinement_canonicalizes_selection_order() {
        let mut forward = unit_square_triangles(1);
        let mut reverse = forward.clone();

        let forward_result = refine_elements_checked(&mut forward, &[0, 1])
            .expect("forward selection should refine");
        let reverse_result = refine_elements_checked(&mut reverse, &[1, 0])
            .expect("reverse selection should refine");

        assert_eq!(forward_result.new_elements, reverse_result.new_elements);
        assert_eq!(
            forward_result.removed_elements,
            reverse_result.removed_elements
        );
        assert_eq!(forward_result.new_nodes, reverse_result.new_nodes);
        assert_eq!(forward.nodes, reverse.nodes);
        let signature = |mesh: &Mesh| {
            mesh.elements
                .iter()
                .map(|element| {
                    (
                        element.element_type,
                        element.nodes.clone(),
                        element.id,
                        element.parent_id,
                        element.level,
                        element.marker,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(signature(&forward), signature(&reverse));
    }

    #[test]
    fn refine_selected_triangle_replaces_parent_without_area_duplication() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].marker = 42;
        let parent_id = mesh.elements[0].id;
        let initial_area: f64 = (0..mesh.num_elements())
            .map(|elem_idx| mesh.element_measure(elem_idx))
            .sum();

        let result = refine_elements(&mut mesh, &[0]);
        let refined_area: f64 = (0..mesh.num_elements())
            .map(|elem_idx| mesh.element_measure(elem_idx))
            .sum();

        assert_eq!(result.removed_elements, vec![0]);
        assert_eq!(result.new_elements.len(), 4);
        assert_eq!(mesh.num_elements(), 5);
        assert!(mesh.elements.iter().all(|elem| elem.id != parent_id));
        assert_eq!(mesh.cells_with_marker(42).count(), 4);
        assert!((refined_area - initial_area).abs() < 1e-12);
    }

    #[test]
    fn refine_selected_tetrahedron_replaces_parent_without_volume_duplication() {
        let mut mesh = Mesh::new(3);
        mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        let parent_id = mesh.elements[0].id;
        let initial_volume = mesh.element_measure(0);

        let result = refine_elements(&mut mesh, &[0]);
        let refined_volume: f64 = (0..mesh.num_elements())
            .map(|elem_idx| mesh.element_measure(elem_idx))
            .sum();

        assert_eq!(result.removed_elements, vec![0]);
        assert_eq!(result.new_elements.len(), 8);
        assert_eq!(mesh.num_elements(), 8);
        assert!(mesh.elements.iter().all(|elem| elem.id != parent_id));
        assert!(
            check_orientation(&mesh).is_empty(),
            "tetrahedral refinement must orient every child positively"
        );
        assert!((refined_volume - initial_volume).abs() < 1e-12);
    }

    #[test]
    fn refine_selected_hexahedron_replaces_parent_and_preserves_volume() {
        use crate::mesh::generators::unit_cube_hexahedra;
        let mut mesh = unit_cube_hexahedra(1);
        assert_eq!(mesh.num_elements(), 1);
        let parent_id = mesh.elements[0].id;
        let initial_volume = mesh.element_measure(0);

        let result = refine_elements(&mut mesh, &[0]);

        // 8 child hexes, parent gone.
        assert_eq!(result.removed_elements, vec![0]);
        assert_eq!(result.new_elements.len(), 8);
        assert_eq!(mesh.num_elements(), 8);
        assert!(mesh.elements.iter().all(|elem| elem.id != parent_id));
        assert!(
            mesh.elements
                .iter()
                .all(|elem| elem.element_type == ElementType::Hexahedron)
        );

        let refined_volume: f64 = (0..mesh.num_elements())
            .map(|elem_idx| mesh.element_measure(elem_idx))
            .sum();
        assert!(
            (refined_volume - initial_volume).abs() < 1e-10,
            "refined hex volume {refined_volume} drifted from parent {initial_volume}",
        );

        // Every child should have one parent corner — i.e. exactly one of its
        // eight nodes must be in {0,1,2,3,4,5,6,7} (the parent corners).
        for elem in &mesh.elements {
            let parent_corners = elem.vertices().iter().filter(|&&n| n < 8).count();
            assert_eq!(
                parent_corners,
                1,
                "child hex {:?} has {} parent corners, expected 1",
                elem.vertices(),
                parent_corners
            );
        }
    }

    #[test]
    fn refine_selected_triangle_splits_boundary_markers_to_children() {
        let mut mesh = unit_square_triangles(1);
        let initial_boundaries = mesh.boundaries.len();

        refine_elements(&mut mesh, &[0]);

        assert_eq!(initial_boundaries, 4);
        assert_eq!(mesh.boundaries.len(), 6);
        assert!(
            mesh.boundaries
                .iter()
                .all(|boundary| boundary.element_idx < mesh.num_elements())
        );

        let marker_count = |marker| {
            mesh.boundaries
                .iter()
                .filter(|boundary| boundary.marker == marker)
                .count()
        };
        assert_eq!(marker_count(1), 1);
        assert_eq!(marker_count(2), 2);
        assert_eq!(marker_count(3), 2);
        assert_eq!(marker_count(4), 1);

        assert!(mesh.boundaries.iter().any(|boundary| {
            boundary.marker == 2
                && boundary.boundary_type == BoundaryType::Dirichlet
                && mesh.elements[boundary.element_idx].parent_id.is_some()
        }));
        assert!(mesh.boundaries.iter().any(|boundary| {
            boundary.marker == 3
                && boundary.boundary_type == BoundaryType::Neumann
                && mesh.elements[boundary.element_idx].parent_id.is_some()
        }));
    }

    #[test]
    fn doerfler_marking_rejects_non_finite_errors() {
        assert!(doerfler_marking(&[1.0, f64::NAN], 0.5).is_err());
        assert!(doerfler_marking(&[1.0], 1.5).is_err());
    }

    #[test]
    fn checked_coarsening_restores_triangle_family_and_boundaries() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let original = mesh.clone();
        let original_area: f64 = (0..mesh.num_elements())
            .map(|index| mesh.element_measure(index))
            .sum();
        let (refinement, history) = refine_elements_with_history_checked(&mut mesh, &[0]).unwrap();
        assert_eq!(refinement.new_elements.len(), 4);
        let parent_id = history.parents[0].element.id;
        let refined_area: f64 = (0..mesh.num_elements())
            .map(|index| mesh.element_measure(index))
            .sum();
        assert!((refined_area - original_area).abs() < 1.0e-12);

        let result = coarsen_elements_checked(&mut mesh, &history, &[parent_id]).unwrap();
        assert_eq!(result.restored_elements.len(), 1);
        assert_eq!(mesh.num_elements(), original.num_elements());
        assert_eq!(mesh.boundaries.len(), original.boundaries.len());
        let coarsened_area: f64 = (0..mesh.num_elements())
            .map(|index| mesh.element_measure(index))
            .sum();
        assert!((coarsened_area - original_area).abs() < 1.0e-12);
        assert!(mesh.validate_connectivity().is_ok());
        assert!(check_orientation(&mesh).is_empty());
        assert_eq!(
            mesh.cells_with_marker(original.elements[0].marker).count(),
            original
                .cells_with_marker(original.elements[0].marker)
                .count()
        );
    }

    #[test]
    fn checked_coarsening_rejects_incomplete_family_without_mutation() {
        let mut mesh = unit_square_triangles(1);
        // Keep the fixture focused on sibling admission; removing one child
        // would otherwise leave a stale boundary index before the coarsening
        // history can report its intended diagnostic.
        mesh.boundaries.clear();
        let (_, history) = refine_elements_with_history_checked(&mut mesh, &[0]).unwrap();
        let parent_id = history.parents[0].element.id;
        let child_index = mesh
            .elements
            .iter()
            .position(|element| element.parent_id == Some(parent_id))
            .unwrap();
        mesh.elements.remove(child_index);
        let before = mesh.clone();
        assert!(matches!(
            coarsen_elements_checked(&mut mesh, &history, &[parent_id]),
            Err(RefinementError::InvalidHistory {
                reason: "sibling family is incomplete or has already been coarsened",
                ..
            })
        ));
        let element_signature = |elements: &[Element]| {
            elements
                .iter()
                .map(|element| {
                    (
                        element.element_type,
                        element.nodes.clone(),
                        element.id,
                        element.parent_id,
                        element.level,
                        element.marker,
                    )
                })
                .collect::<Vec<_>>()
        };
        let boundary_signature = |boundaries: &[BoundaryFace]| {
            boundaries
                .iter()
                .map(|boundary| {
                    (
                        boundary.nodes.clone(),
                        boundary.boundary_type,
                        boundary.marker,
                        boundary.element_idx,
                        boundary.local_idx,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            element_signature(&mesh.elements),
            element_signature(&before.elements)
        );
        assert_eq!(
            boundary_signature(&mesh.boundaries),
            boundary_signature(&before.boundaries)
        );
    }

    #[test]
    fn checked_tetra_refinement_round_trips_boundary_markers_and_topology() {
        let mut mesh = Mesh::new(3);
        mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        mesh.detect_boundaries();

        // Give each parent face its own marker so propagation cannot pass
        // merely because all boundary records share the default marker.
        for (index, boundary) in mesh.boundaries.iter_mut().enumerate() {
            boundary.marker = 10 + index as i32;
        }
        let original_boundaries = mesh.boundaries.clone();
        let original_topology_markers: Vec<_> = original_boundaries
            .iter()
            .map(|boundary| boundary.marker)
            .collect();

        let (refinement, history) =
            refine_elements_with_history_checked(&mut mesh, &[0]).expect("valid tetra refinement");

        assert_eq!(refinement.new_elements.len(), 8);
        assert_eq!(mesh.boundaries.len(), 16);
        assert!(mesh.validate_connectivity().is_ok());
        assert!(check_orientation(&mesh).is_empty());

        let topology = mesh.try_topology().expect("refined topology should build");
        for marker in &original_topology_markers {
            assert_eq!(
                topology.boundary_faces(*marker).len(),
                4,
                "each tetra face should refine into four marked boundary faces"
            );
        }

        let parent_id = history.parents[0].element.id;
        let coarsening = coarsen_elements_checked(&mut mesh, &history, &[parent_id])
            .expect("refinement history should restore the tetrahedron");

        assert_eq!(coarsening.restored_elements.len(), 1);
        assert_eq!(coarsening.removed_elements.len(), 8);
        assert_eq!(mesh.num_elements(), 1);
        let boundary_signature = |boundaries: &[BoundaryFace]| {
            boundaries
                .iter()
                .map(|boundary| {
                    (
                        boundary.nodes.clone(),
                        boundary.boundary_type,
                        boundary.marker,
                        boundary.element_idx,
                        boundary.local_idx,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            boundary_signature(&mesh.boundaries),
            boundary_signature(&original_boundaries)
        );
        assert!(mesh.validate_connectivity().is_ok());
        let topology = mesh
            .try_topology()
            .expect("coarsened topology should build");
        for marker in original_topology_markers {
            assert_eq!(topology.boundary_faces(marker).len(), 1);
        }
    }

    #[test]
    fn checked_p3_hexahedral_refinement_round_trips_cubic_geometry_and_history() {
        let mut mesh = Mesh::new(3);
        for [i, j, k] in P3_HEX_NODE_LATTICE {
            mesh.add_node(Point::new_3d(
                i as f64 / 3.0,
                j as f64 / 3.0,
                k as f64 / 3.0,
            ));
        }
        mesh.add_element(ElementType::Hexahedron, (0..64).collect());
        mesh.detect_boundaries();

        let (refinement, history) =
            refine_elements_high_order_with_history_checked(&mut mesh, &[0])
                .expect("checked P3 hexahedral refinement");
        assert_eq!(refinement.new_elements.len(), 8);
        assert_eq!(refinement.new_nodes.len(), 279);
        assert_eq!(mesh.num_nodes(), 343);
        assert!(mesh.elements.iter().all(|element| {
            element.element_type == ElementType::Hexahedron
                && element.nodes.len() == ElementType::Hexahedron.num_nodes_p3()
                && element.parent_id == Some(history.parents[0].element.id)
        }));
        assert!(mesh.validate_connectivity().is_ok());
        assert!(check_orientation(&mesh).is_empty());
        assert!(mesh.nodes.iter().all(|point| {
            point.x >= -1.0e-12
                && point.y >= -1.0e-12
                && point.z >= -1.0e-12
                && point.x <= 1.0 + 1.0e-12
                && point.y <= 1.0 + 1.0e-12
                && point.z <= 1.0 + 1.0e-12
        }));

        let parent_id = history.parents[0].element.id;
        let coarsening = coarsen_elements_checked(&mut mesh, &history, &[parent_id])
            .expect("P3 hexahedral history should coarsen");
        assert_eq!(coarsening.restored_elements.len(), 1);
        assert_eq!(coarsening.removed_elements.len(), 8);
        assert_eq!(mesh.num_elements(), 1);
        assert!(mesh.validate_connectivity().is_ok());
    }

    #[test]
    fn checked_p3_hexahedral_refinement_reuses_shared_face_nodes() {
        let mut mesh = Mesh::new(3);
        let mut node_ids = std::collections::HashMap::<(i32, i32, i32), usize>::new();
        let mut add_box = |min: (i32, i32, i32), max: (i32, i32, i32)| {
            P3_HEX_NODE_LATTICE
                .iter()
                .map(|[i, j, k]| {
                    let scaled = (
                        min.0 * 3 + *i as i32 * (max.0 - min.0),
                        min.1 * 3 + *j as i32 * (max.1 - min.1),
                        min.2 * 3 + *k as i32 * (max.2 - min.2),
                    );
                    if let Some(&node) = node_ids.get(&scaled) {
                        node
                    } else {
                        let node = mesh.add_node(Point::new_3d(
                            scaled.0 as f64 / 3.0,
                            scaled.1 as f64 / 3.0,
                            scaled.2 as f64 / 3.0,
                        ));
                        node_ids.insert(scaled, node);
                        node
                    }
                })
                .collect::<Vec<_>>()
        };

        let first = add_box((0, 0, 0), (1, 1, 1));
        let second = add_box((1, 0, 0), (2, 1, 1));
        mesh.add_element(ElementType::Hexahedron, first);
        mesh.add_element(ElementType::Hexahedron, second);
        mesh.detect_boundaries();

        let result = refine_elements_high_order_checked(&mut mesh, &[0, 1])
            .expect("checked adjacent P3 hexahedral refinement");
        assert_eq!(result.new_elements.len(), 16);
        assert_eq!(result.new_nodes.len(), 525);
        assert_eq!(mesh.num_nodes(), 637);
        assert!(mesh.elements.iter().all(|element| {
            element.element_type == ElementType::Hexahedron
                && element.nodes.len() == ElementType::Hexahedron.num_nodes_p3()
                && element.parent_id.is_some()
        }));
        assert!(mesh.validate_connectivity().is_ok());
        assert!(check_orientation(&mesh).is_empty());
    }

    #[test]
    fn checked_p3_tetrahedral_refinement_reuses_shared_face_nodes() {
        let mut mesh = Mesh::new(3);
        let mut node_ids = std::collections::HashMap::<(i32, i32, i32), usize>::new();

        let mut add_p3_tetra = |vertices: [(i32, i32, i32); 4]| {
            P3_TETRA_LATTICES
                .iter()
                .map(|alpha| {
                    let mut scaled = (0_i32, 0_i32, 0_i32);
                    for (weight, vertex) in alpha.iter().zip(vertices) {
                        scaled.0 += *weight as i32 * vertex.0;
                        scaled.1 += *weight as i32 * vertex.1;
                        scaled.2 += *weight as i32 * vertex.2;
                    }
                    if let Some(&node) = node_ids.get(&scaled) {
                        node
                    } else {
                        let node = mesh.add_node(Point::new_3d(
                            scaled.0 as f64 / 3.0,
                            scaled.1 as f64 / 3.0,
                            scaled.2 as f64 / 3.0,
                        ));
                        node_ids.insert(scaled, node);
                        node
                    }
                })
                .collect::<Vec<_>>()
        };

        let shared_a = (0, 0, 0);
        let shared_b = (1, 0, 0);
        let shared_c = (0, 1, 0);
        let first = add_p3_tetra([shared_a, shared_b, shared_c, (0, 0, 1)]);
        // Reverse the shared face order while keeping the second tetrahedron
        // positively oriented. This exercises face-key canonicalization.
        let second = add_p3_tetra([shared_a, shared_c, shared_b, (0, 0, -1)]);
        mesh.add_element(ElementType::Tetrahedron, first);
        mesh.add_element(ElementType::Tetrahedron, second);
        mesh.detect_boundaries();

        let result = refine_elements_high_order_checked(&mut mesh, &[0, 1])
            .expect("checked adjacent P3 tetrahedral refinement");

        assert_eq!(result.new_elements.len(), 16);
        assert_eq!(result.new_nodes.len(), 110);
        assert_eq!(mesh.num_nodes(), 140);
        assert!(
            mesh.elements
                .iter()
                .all(|element| element.element_type == ElementType::Tetrahedron
                    && element.nodes.len() == ElementType::Tetrahedron.num_nodes_p3()
                    && element.parent_id.is_some())
        );
        assert!(mesh.validate_connectivity().is_ok());
        assert!(check_orientation(&mesh).is_empty());
    }

    #[test]
    fn checked_p3_tetrahedral_refinement_round_trips_cubic_geometry_and_history() {
        let mut mesh = Mesh::new(3);
        for alpha in P3_TETRA_LATTICES {
            mesh.add_node(Point::new_3d(
                alpha[1] as f64 / 3.0,
                alpha[2] as f64 / 3.0,
                alpha[3] as f64 / 3.0,
            ));
        }
        mesh.add_element(ElementType::Tetrahedron, (0..20).collect());
        mesh.detect_boundaries();

        let (refinement, history) =
            refine_elements_high_order_with_history_checked(&mut mesh, &[0])
                .expect("checked P3 tetrahedral refinement");
        assert_eq!(refinement.new_elements.len(), 8);
        assert_eq!(refinement.new_nodes.len(), 64);
        assert_eq!(mesh.num_nodes(), 84);
        assert!(mesh.elements.iter().all(|element| {
            element.element_type == ElementType::Tetrahedron
                && element.nodes.len() == ElementType::Tetrahedron.num_nodes_p3()
                && element.parent_id == Some(history.parents[0].element.id)
        }));
        assert!(mesh.validate_connectivity().is_ok());
        assert!(check_orientation(&mesh).is_empty());
        assert!(mesh.nodes.iter().all(|point| {
            point.x >= -1.0e-12
                && point.y >= -1.0e-12
                && point.z >= -1.0e-12
                && point.x + point.y + point.z <= 1.0 + 1.0e-12
        }));

        let parent_id = history.parents[0].element.id;
        let coarsening = coarsen_elements_checked(&mut mesh, &history, &[parent_id])
            .expect("P3 tetrahedral history should coarsen");
        assert_eq!(coarsening.restored_elements.len(), 1);
        assert_eq!(coarsening.removed_elements.len(), 8);
        assert_eq!(mesh.num_elements(), 1);
        assert!(mesh.validate_connectivity().is_ok());
    }
}
