//! H1-conforming function spaces (Lagrange nodal elements)

use crate::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use crate::mesh::{ElementType, Mesh};
use std::sync::{Arc, RwLock};

use super::traits::{
    BasisTabulation, FieldType, FunctionSpace, SpaceValidationError, validate_function_space,
    validate_mesh_connectivity,
};

type TopologyCacheKey = (usize, usize, usize);
type TopologyCache =
    Arc<RwLock<Option<(TopologyCacheKey, Arc<crate::mesh::topology::MeshTopology>)>>>;

/// Scalar H1-conforming function space using Lagrange nodal basis functions.
///
/// DOFs are located at mesh nodes. For P1 elements, one DOF per vertex.
/// For P2, additional DOFs at edge midpoints, etc.
#[derive(Debug, Clone)]
pub struct H1Space {
    pub degree: PolynomialDegree,
    topology_cache: TopologyCache,
}

impl H1Space {
    pub fn new(degree: PolynomialDegree) -> Self {
        Self {
            degree,
            topology_cache: Arc::new(RwLock::new(None)),
        }
    }

    fn topology(&self, mesh: &Mesh) -> Arc<crate::mesh::topology::MeshTopology> {
        let key = (
            mesh as *const Mesh as usize,
            mesh.num_nodes(),
            mesh.num_elements(),
        );
        if let Some((_, topology)) = self
            .topology_cache
            .read()
            .expect("H1 topology cache poisoned")
            .as_ref()
            .filter(|(cached_key, _)| *cached_key == key)
        {
            return Arc::clone(topology);
        }

        let topology = Arc::new(crate::mesh::topology::MeshTopology::build(mesh));
        *self
            .topology_cache
            .write()
            .expect("H1 topology cache poisoned") = Some((key, Arc::clone(&topology)));
        topology
    }

    fn assert_supported(&self, _elem_type: ElementType) {
        // Every currently exposed element family supports P1/P2/P3 H1 bases.
    }
}

impl H1Space {
    /// Return the explicit isoparametric geometry order carried by the mesh.
    ///
    /// A vertex-only mesh derives higher-order solution DOFs from topology;
    /// an element whose connectivity contains the full P2/P3 node layout
    /// carries its geometry explicitly.  The function intentionally inspects
    /// only the first element because `validate` separately rejects mixed
    /// geometry orders before any assembly is admitted.
    fn explicit_geometry_degree(mesh: &Mesh) -> Option<PolynomialDegree> {
        mesh.elements.first().and_then(|element| {
            let nodes = element.nodes.len();
            if nodes == element.element_type.num_vertices() {
                Some(PolynomialDegree::P1)
            } else if nodes == element.element_type.num_nodes_p2() {
                Some(PolynomialDegree::P2)
            } else if nodes == element.element_type.num_nodes_p3() {
                Some(PolynomialDegree::P3)
            } else {
                None
            }
        })
    }

    fn has_explicit_p2_geometry(mesh: &Mesh) -> bool {
        Self::explicit_geometry_degree(mesh) == Some(PolynomialDegree::P2)
    }

    fn has_explicit_p3_geometry(mesh: &Mesh) -> bool {
        Self::explicit_geometry_degree(mesh) == Some(PolynomialDegree::P3)
    }

    fn geometry_is_uniform(mesh: &Mesh) -> bool {
        mesh.elements
            .first()
            .map(|first| {
                let first_len = first.nodes.len();
                mesh.elements.iter().all(|element| {
                    element.nodes.len() == first_len
                        && (first_len == element.element_type.num_vertices()
                            || first_len == element.element_type.num_nodes_p2()
                            || first_len == element.element_type.num_nodes_p3())
                })
            })
            .unwrap_or(true)
    }

    fn element_geometry_degree(&self, element: &crate::mesh::Element) -> PolynomialDegree {
        match element.nodes.len() {
            n if n == element.element_type.num_vertices() => PolynomialDegree::P1,
            n if n == element.element_type.num_nodes_p2() => PolynomialDegree::P2,
            n if n == element.element_type.num_nodes_p3() => PolynomialDegree::P3,
            _ => panic!(
                "invalid H1 geometry connectivity: {} nodes for {:?}",
                element.nodes.len(),
                element.element_type
            ),
        }
    }

    fn vertex_dof_count(mesh: &Mesh) -> usize {
        mesh.elements
            .iter()
            .flat_map(|element| element.nodes[..element.element_type.num_vertices()].iter())
            .copied()
            .max()
            .map_or(0, |node| node + 1)
    }
}

/// Recover the complete P3 trace from a corner-only boundary entity.
///
/// `Mesh::detect_boundaries` intentionally stores topological corner nodes so
/// that neighboring elements can be matched independently of polynomial
/// order.  Explicit P3 H1 geometry uses the mesh node IDs as its DOFs, so the
/// boundary query must add the two cubic nodes on every boundary edge and the
/// face-interior nodes on every boundary face.  The entity is identified by
/// its corner node set, which also works for imported boundaries that already
/// carry some or all higher-order nodes.
fn explicit_p3_boundary_trace_nodes(
    element: &crate::mesh::Element,
    boundary_nodes: &[usize],
) -> Vec<usize> {
    let element_type = element.element_type;
    if element.nodes.len() < element_type.num_nodes_p3() {
        return Vec::new();
    }

    let boundary_nodes: std::collections::HashSet<usize> = boundary_nodes.iter().copied().collect();
    let contains_corners = |corners: &[usize]| {
        corners
            .iter()
            .all(|&local_vertex| boundary_nodes.contains(&element.nodes[local_vertex]))
    };
    let vertex_count = element_type.num_vertices();
    let edge_offset = vertex_count;
    let face_offset = edge_offset + 2 * element_type.num_edges();

    let mut trace = Vec::new();
    match element_type {
        ElementType::Triangle | ElementType::Quadrilateral => {
            let Some((edge_idx, &(left, right))) = element_type
                .local_edges()
                .iter()
                .enumerate()
                .find(|(_, edge)| contains_corners(&[edge.0, edge.1]))
            else {
                return trace;
            };
            trace.extend([element.nodes[left], element.nodes[right]]);
            trace.extend([
                element.nodes[edge_offset + 2 * edge_idx],
                element.nodes[edge_offset + 2 * edge_idx + 1],
            ]);
        }
        ElementType::Tetrahedron | ElementType::Hexahedron => {
            let Some((face_idx, face)) = element_type
                .local_faces()
                .iter()
                .enumerate()
                .find(|(_, face)| contains_corners(face))
            else {
                return trace;
            };
            trace.extend(face.iter().map(|&local_vertex| element.nodes[local_vertex]));
            for (edge_idx, &(left, right)) in element_type.local_edges().iter().enumerate() {
                if face.contains(&left) && face.contains(&right) {
                    trace.extend([
                        element.nodes[edge_offset + 2 * edge_idx],
                        element.nodes[edge_offset + 2 * edge_idx + 1],
                    ]);
                }
            }
            let face_nodes = match element_type {
                ElementType::Tetrahedron => 1,
                ElementType::Hexahedron => 4,
                ElementType::Triangle | ElementType::Quadrilateral => unreachable!(),
            };
            let first_face_node = face_offset + face_idx * face_nodes;
            trace.extend(
                (0..face_nodes).map(|local_slot| element.nodes[first_face_node + local_slot]),
            );
        }
    }
    trace
}

/// Map a Q3 face-interior node from an element-local face orientation to the
/// canonical orientation used by the mesh topology.
#[allow(dead_code)]
fn p3_face_interior_slot_oriented(
    local_face_idx: usize,
    element_vertices: &[usize],
    canonical_face: &[usize],
    local_slot: usize,
) -> usize {
    const REFERENCE_VERTICES: [[f64; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    const FACE_POINTS: [[[f64; 3]; 4]; 6] = [
        [
            [-1.0 / 3.0, -1.0 / 3.0, -1.0],
            [1.0 / 3.0, -1.0 / 3.0, -1.0],
            [1.0 / 3.0, 1.0 / 3.0, -1.0],
            [-1.0 / 3.0, 1.0 / 3.0, -1.0],
        ],
        [
            [-1.0 / 3.0, -1.0 / 3.0, 1.0],
            [1.0 / 3.0, -1.0 / 3.0, 1.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0],
            [-1.0 / 3.0, 1.0 / 3.0, 1.0],
        ],
        [
            [-1.0 / 3.0, -1.0, -1.0 / 3.0],
            [1.0 / 3.0, -1.0, -1.0 / 3.0],
            [1.0 / 3.0, -1.0, 1.0 / 3.0],
            [-1.0 / 3.0, -1.0, 1.0 / 3.0],
        ],
        [
            [-1.0 / 3.0, 1.0, -1.0 / 3.0],
            [1.0 / 3.0, 1.0, -1.0 / 3.0],
            [1.0 / 3.0, 1.0, 1.0 / 3.0],
            [-1.0 / 3.0, 1.0, 1.0 / 3.0],
        ],
        [
            [-1.0, -1.0 / 3.0, -1.0 / 3.0],
            [-1.0, 1.0 / 3.0, -1.0 / 3.0],
            [-1.0, 1.0 / 3.0, 1.0 / 3.0],
            [-1.0, -1.0 / 3.0, 1.0 / 3.0],
        ],
        [
            [1.0, -1.0 / 3.0, -1.0 / 3.0],
            [1.0, 1.0 / 3.0, -1.0 / 3.0],
            [1.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0, -1.0 / 3.0, 1.0 / 3.0],
        ],
    ];
    let canonical_reference: [[f64; 3]; 4] = canonical_face
        .iter()
        .map(|&node| {
            let local_vertex = element_vertices
                .iter()
                .position(|&candidate| candidate == node)
                .expect("face topology must preserve all face vertices");
            REFERENCE_VERTICES[local_vertex]
        })
        .collect::<Vec<_>>()
        .try_into()
        .expect("hexahedral face must have four vertices");
    let center = [
        canonical_reference
            .iter()
            .map(|point| point[0])
            .sum::<f64>()
            / 4.0,
        canonical_reference
            .iter()
            .map(|point| point[1])
            .sum::<f64>()
            / 4.0,
        canonical_reference
            .iter()
            .map(|point| point[2])
            .sum::<f64>()
            / 4.0,
    ];
    let u_axis = [
        0.5 * (canonical_reference[1][0] - canonical_reference[0][0]),
        0.5 * (canonical_reference[1][1] - canonical_reference[0][1]),
        0.5 * (canonical_reference[1][2] - canonical_reference[0][2]),
    ];
    let v_axis = [
        0.5 * (canonical_reference[3][0] - canonical_reference[0][0]),
        0.5 * (canonical_reference[3][1] - canonical_reference[0][1]),
        0.5 * (canonical_reference[3][2] - canonical_reference[0][2]),
    ];
    let point = FACE_POINTS[local_face_idx][local_slot];
    let centered = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    let u = centered[0] * u_axis[0] + centered[1] * u_axis[1] + centered[2] * u_axis[2];
    let v = centered[0] * v_axis[0] + centered[1] * v_axis[1] + centered[2] * v_axis[2];
    match (u.is_sign_negative(), v.is_sign_negative()) {
        (true, true) => 0,
        (false, true) => 1,
        (false, false) => 2,
        (true, false) => 3,
    }
}

const Q3_HEX_FACE_NODE_INDICES: [[(usize, usize, usize); 4]; 6] = [
    [(1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0)],
    [(1, 1, 3), (2, 1, 3), (2, 2, 3), (1, 2, 3)],
    [(1, 0, 1), (2, 0, 1), (2, 0, 2), (1, 0, 2)],
    [(1, 3, 1), (2, 3, 1), (2, 3, 2), (1, 3, 2)],
    [(0, 1, 1), (0, 2, 1), (0, 2, 2), (0, 1, 2)],
    [(3, 1, 1), (3, 2, 1), (3, 2, 2), (3, 1, 2)],
];

fn q3_hex_reference_point(indices: (usize, usize, usize)) -> [f64; 3] {
    [
        2.0 * indices.0 as f64 / 3.0 - 1.0,
        2.0 * indices.1 as f64 / 3.0 - 1.0,
        2.0 * indices.2 as f64 / 3.0 - 1.0,
    ]
}

fn q3_hex_face_point_key(
    element: &crate::mesh::Element,
    local_face: usize,
    local_slot: usize,
) -> Vec<(usize, usize)> {
    let (i, j, k) = Q3_HEX_FACE_NODE_INDICES[local_face][local_slot];
    let weights = [
        (3 - i) * (3 - j) * (3 - k),
        i * (3 - j) * (3 - k),
        i * j * (3 - k),
        (3 - i) * j * (3 - k),
        (3 - i) * (3 - j) * k,
        i * (3 - j) * k,
        i * j * k,
        (3 - i) * j * k,
    ];
    let mut key: Vec<_> = element
        .vertices()
        .iter()
        .copied()
        .zip(weights)
        .filter(|(_, weight)| *weight != 0)
        .collect();
    key.sort_unstable_by_key(|(node, _)| *node);
    key
}

fn p3_hex_face_interior_slot(
    mesh: &Mesh,
    topology: &crate::mesh::topology::MeshTopology,
    elem_idx: usize,
    local_face: usize,
    face_idx: usize,
    local_slot: usize,
) -> usize {
    let local_key = q3_hex_face_point_key(&mesh.elements[elem_idx], local_face, local_slot);
    let (canonical_elem, canonical_face) = topology
        .element_faces
        .iter()
        .enumerate()
        .find_map(|(candidate_elem, faces)| {
            faces
                .iter()
                .position(|&candidate_face| candidate_face == face_idx)
                .map(|candidate_face| (candidate_elem, candidate_face))
        })
        .expect("every topology face must belong to an element");
    for candidate_slot in 0..4 {
        if local_key
            == q3_hex_face_point_key(
                &mesh.elements[canonical_elem],
                canonical_face,
                candidate_slot,
            )
        {
            return candidate_slot;
        }
    }
    panic!("face topology has unsupported Q3 orientation");
}

fn p3_hex_face_point(
    mesh: &Mesh,
    topology: &crate::mesh::topology::MeshTopology,
    face_idx: usize,
    local_slot: usize,
) -> [f64; 3] {
    let (elem_idx, local_face) = topology
        .element_faces
        .iter()
        .enumerate()
        .find_map(|(candidate_elem, faces)| {
            faces
                .iter()
                .position(|&candidate_face| candidate_face == face_idx)
                .map(|candidate_face| (candidate_elem, candidate_face))
        })
        .expect("every topology face must belong to an element");
    let reference = q3_hex_reference_point(Q3_HEX_FACE_NODE_INDICES[local_face][local_slot]);
    let shape = evaluate_shape(
        ElementType::Hexahedron,
        PolynomialDegree::P1,
        reference[0],
        reference[1],
        reference[2],
    );
    let vertices = mesh.elements[elem_idx].vertices();
    let mut point = [0.0; 3];
    for (weight, &node) in shape.values.iter().zip(vertices.iter()) {
        let coordinates = mesh.nodes[node];
        point[0] += weight * coordinates.x;
        point[1] += weight * coordinates.y;
        point[2] += weight * coordinates.z;
    }
    point
}

impl FunctionSpace for H1Space {
    fn field_type(&self) -> FieldType {
        FieldType::Scalar
    }

    fn polynomial_degree(&self) -> usize {
        self.degree.degree()
    }

    fn num_components(&self) -> usize {
        1
    }

    fn validate(&self, mesh: &Mesh) -> Result<(), SpaceValidationError> {
        // Reject malformed connectivity before classifying geometry order.
        // This keeps a short node list from being misclassified as mixed
        // geometry while still keeping mixed geometry away from DOF mapping.
        validate_mesh_connectivity(mesh)?;
        if !Self::geometry_is_uniform(mesh) {
            return Err(SpaceValidationError::InvalidMesh {
                element: 0,
                reason: "mixed affine/P2/P3 geometry is not supported by H1Space",
            });
        }

        // Admit either vertex-only affine geometry or explicit P2
        // isoparametric geometry before checking the common DOF contract.
        // Reject malformed topology only after the geometry-order gate so
        // mixed explicit/affine meshes cannot panic in local DOF mapping.
        validate_function_space(self, mesh)?;

        for (element, mesh_element) in mesh.elements.iter().enumerate() {
            let valid_geometry = mesh_element.nodes.len()
                == mesh_element.element_type.num_vertices()
                || mesh_element.nodes.len() == mesh_element.element_type.num_nodes_p2()
                || mesh_element.nodes.len() == mesh_element.element_type.num_nodes_p3();
            if !valid_geometry {
                return Err(SpaceValidationError::InvalidMesh {
                    element,
                    reason: "element geometry must be P1, P2, or P3 connectivity",
                });
            }
            if mesh_element.nodes.len() == mesh_element.element_type.num_nodes_p2()
                && self.degree == PolynomialDegree::P3
            {
                return Err(SpaceValidationError::InvalidMesh {
                    element,
                    reason: "P3 H1 on explicit P2 geometry is not supported",
                });
            }
            if mesh_element.nodes.len() == mesh_element.element_type.num_nodes_p3()
                && self.degree == PolynomialDegree::P2
            {
                return Err(SpaceValidationError::InvalidMesh {
                    element,
                    reason: "P2 H1 on explicit P3 geometry is not supported",
                });
            }
        }
        Ok(())
    }

    fn num_global_dofs(&self, mesh: &Mesh) -> usize {
        if let Some(elem) = mesh.elements.first() {
            self.assert_supported(elem.element_type);
        }

        match self.degree {
            PolynomialDegree::P1 => {
                if Self::has_explicit_p2_geometry(mesh) || Self::has_explicit_p3_geometry(mesh) {
                    Self::vertex_dof_count(mesh)
                } else {
                    mesh.num_nodes()
                }
            }
            PolynomialDegree::P2 => {
                if Self::has_explicit_p2_geometry(mesh) {
                    return mesh
                        .elements
                        .iter()
                        .flat_map(|element| element.nodes.iter())
                        .copied()
                        .max()
                        .map_or(0, |node| node + 1);
                }
                // P2/Q2 global DOF layout (range-based, see [`local_to_global`]):
                //   [ vertex nodes | edge midpoints | face centres (Hex27) | cell-interior bubbles ]
                //
                let topo = self.topology(mesh);
                let mut n = mesh.num_nodes() + topo.num_edges();
                let has_face_centres = mesh
                    .elements
                    .iter()
                    .any(|e| e.element_type.p2_has_face_centres());
                if has_face_centres {
                    n += topo.num_faces();
                }
                let n_interior = mesh
                    .elements
                    .iter()
                    .filter(|e| e.element_type.p2_has_interior())
                    .count();
                n + n_interior
            }
            PolynomialDegree::P3 => {
                if Self::has_explicit_p3_geometry(mesh) {
                    return mesh
                        .elements
                        .iter()
                        .flat_map(|element| element.nodes.iter())
                        .copied()
                        .max()
                        .map_or(0, |node| node + 1);
                }
                let topology = self.topology(mesh);
                let edge_offset = mesh.num_nodes();
                let mut count = edge_offset + 2 * topology.num_edges();
                let face_dofs_per_face = mesh
                    .elements
                    .first()
                    .map(|element| match element.element_type {
                        ElementType::Tetrahedron => 1,
                        ElementType::Hexahedron => 4,
                        ElementType::Triangle | ElementType::Quadrilateral => 0,
                    })
                    .unwrap_or(0);
                count += face_dofs_per_face * topology.num_faces();
                count += mesh
                    .elements
                    .iter()
                    .map(|element| match element.element_type {
                        ElementType::Triangle => 1,
                        ElementType::Quadrilateral => 4,
                        ElementType::Tetrahedron => 0,
                        ElementType::Hexahedron => 8,
                    })
                    .sum::<usize>();
                count
            }
        }
    }

    fn num_local_dofs(&self, elem_type: ElementType) -> usize {
        self.assert_supported(elem_type);
        match self.degree {
            PolynomialDegree::P1 => elem_type.num_vertices(),
            PolynomialDegree::P2 => elem_type.num_nodes_p2(),
            PolynomialDegree::P3 => crate::basis::num_nodes(elem_type, PolynomialDegree::P3),
        }
    }

    fn local_to_global(&self, mesh: &Mesh, elem_idx: usize) -> Vec<usize> {
        let elem = &mesh.elements[elem_idx];
        self.assert_supported(elem.element_type);
        match self.degree {
            PolynomialDegree::P1 => elem.nodes[..self.num_local_dofs(elem.element_type)].to_vec(),
            PolynomialDegree::P2 => {
                if Self::has_explicit_p2_geometry(mesh) {
                    return elem.nodes[..elem.element_type.num_nodes_p2()].to_vec();
                }
                // P2/Q2 DOF layout (must match num_global_dofs):
                //   [ vertex | edge | face (Hex27 only) | cell-interior bubble (Q2 quad/hex) ]
                let topo = self.topology(mesh);
                let n_nodes = mesh.num_nodes();
                let n_edges = topo.num_edges();
                let has_face_centres = mesh
                    .elements
                    .iter()
                    .any(|e| e.element_type.p2_has_face_centres());
                let face_offset = n_nodes + n_edges;
                let n_face_dofs = if has_face_centres {
                    topo.num_faces()
                } else {
                    0
                };
                let interior_offset = face_offset + n_face_dofs;

                let n_vertices = elem.element_type.num_vertices();
                let edge_indices = topo.element_edge_indices(elem_idx);

                let local_count = elem.element_type.num_nodes_p2();
                let mut dofs = Vec::with_capacity(local_count);

                // Vertex DOFs first (same as P1)
                for &v in &elem.nodes[..n_vertices] {
                    dofs.push(v);
                }
                // Edge midpoint DOFs
                for &edge_idx in edge_indices {
                    dofs.push(n_nodes + edge_idx);
                }
                // Face-centre DOFs (Hex27 only)
                if elem.element_type.p2_has_face_centres() {
                    let face_indices = topo.element_face_indices(elem_idx);
                    for &face_idx in face_indices {
                        dofs.push(face_offset + face_idx);
                    }
                }
                // Cell-interior bubble DOF (Q2 quad/hex). Global index is the
                // prefix count of interior-bearing elements before `elem_idx`.
                if elem.element_type.p2_has_interior() {
                    let interior_local_idx = mesh.elements[..elem_idx]
                        .iter()
                        .filter(|e| e.element_type.p2_has_interior())
                        .count();
                    dofs.push(interior_offset + interior_local_idx);
                }

                debug_assert_eq!(dofs.len(), local_count);
                dofs
            }
            PolynomialDegree::P3 => {
                if Self::has_explicit_p3_geometry(mesh) {
                    return elem.nodes[..elem.element_type.num_nodes_p3()].to_vec();
                }
                // P3: vertices + 2 per edge + face-interior DOFs (shared via global face index)
                // DOF layout: [vertices | edge DOFs (2 per) | face-interior DOFs (1 per global face)]
                let topo = self.topology(mesh);
                let n_nodes = mesh.num_nodes();
                let n_edges = topo.num_edges();
                let n_vertices = elem.element_type.num_vertices();
                let edge_indices = topo.element_edge_indices(elem_idx);
                let edge_signs = topo.element_edge_signs(elem_idx);
                let face_indices = topo.element_face_indices(elem_idx);

                let face_dofs_per_face = match elem.element_type {
                    ElementType::Tetrahedron => 1,
                    ElementType::Hexahedron => 4,
                    ElementType::Triangle | ElementType::Quadrilateral => 0,
                };
                let n_local_dofs =
                    n_vertices + 2 * edge_indices.len() + face_dofs_per_face * face_indices.len();
                let mut dofs = Vec::with_capacity(n_local_dofs);

                for &v in &elem.nodes[..n_vertices] {
                    dofs.push(v);
                }
                for (&edge_idx, &edge_sign) in edge_indices.iter().zip(edge_signs) {
                    let first = n_nodes + edge_idx * 2;
                    let second = first + 1;
                    if edge_sign >= 0 {
                        dofs.push(first);
                        dofs.push(second);
                    } else {
                        dofs.push(second);
                        dofs.push(first);
                    }
                }
                // In 3-D tetrahedra, P3 has one shared interior DOF per
                // triangular face. In 2-D, the remaining P3 nodes are
                // element-interior and therefore must not be shared.
                let interior_offset = n_nodes + 2 * n_edges;
                match elem.element_type {
                    ElementType::Tetrahedron => {
                        for &face_idx in face_indices {
                            dofs.push(interior_offset + face_idx);
                        }
                    }
                    ElementType::Triangle | ElementType::Quadrilateral => {
                        let previous_interior = mesh.elements[..elem_idx]
                            .iter()
                            .map(|element| match element.element_type {
                                ElementType::Triangle => 1,
                                ElementType::Quadrilateral => 4,
                                _ => 0,
                            })
                            .sum::<usize>();
                        let local_interior_count = match elem.element_type {
                            ElementType::Triangle => 1,
                            ElementType::Quadrilateral => 4,
                            _ => 0,
                        };
                        for local in 0..local_interior_count {
                            dofs.push(interior_offset + previous_interior + local);
                        }
                    }
                    ElementType::Hexahedron => {
                        let face_offset = interior_offset;
                        for (local_face_idx, &face_idx) in face_indices.iter().enumerate() {
                            for local_slot in 0..4 {
                                let canonical_slot = p3_hex_face_interior_slot(
                                    mesh,
                                    &topo,
                                    elem_idx,
                                    local_face_idx,
                                    face_idx,
                                    local_slot,
                                );
                                dofs.push(face_offset + face_idx * 4 + canonical_slot);
                            }
                        }

                        let cell_offset = face_offset + 4 * topo.num_faces();
                        let previous_hex_cells = mesh.elements[..elem_idx]
                            .iter()
                            .filter(|element| element.element_type == ElementType::Hexahedron)
                            .count();
                        for local in 0..8 {
                            dofs.push(cell_offset + previous_hex_cells * 8 + local);
                        }
                    }
                }
                dofs
            }
        }
    }

    fn tabulate(
        &self,
        mesh: &Mesh,
        elem_idx: usize,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> BasisTabulation {
        let elem = &mesh.elements[elem_idx];
        let elem_type = elem.element_type;
        self.assert_supported(elem_type);
        let spatial_dim = elem_type.dimension();

        // Evaluate reference-domain shape functions at the requested polynomial degree
        let shape = evaluate_shape(elem_type, self.degree, xi, eta, zeta);
        let n_dofs = shape.values.len();

        // Select the isoparametric geometry basis from the mesh connectivity.
        let geometry_degree = self.element_geometry_degree(elem);
        let geom_shape = evaluate_shape(elem_type, geometry_degree, xi, eta, zeta);
        let geometry_nodes = match geometry_degree {
            PolynomialDegree::P1 => &elem.nodes[..elem_type.num_vertices()],
            PolynomialDegree::P2 => &elem.nodes[..elem_type.num_nodes_p2()],
            PolynomialDegree::P3 => &elem.nodes[..elem_type.num_nodes_p3()],
        };

        // Compute the geometry Jacobian and transform solution gradients
        let (det_j, phys_grads, hessians, phys_coords) = match spatial_dim {
            2 => {
                let coords: Vec<[f64; 2]> = geometry_nodes
                    .iter()
                    .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y])
                    .collect();
                let jac = Jacobian::from_2d(&geom_shape.gradients, geom_shape.grad_dim, &coords);

                let mut phys_grads = Vec::with_capacity(n_dofs * 2);
                for g in shape.gradients.chunks(shape.grad_dim) {
                    let mut pg = [0.0; 2];
                    jac.transform_gradient_into(g, &mut pg);
                    phys_grads.extend_from_slice(&pg);
                }

                // Physical coordinates via isoparametric geometry interpolation
                let mut px = 0.0;
                let mut py = 0.0;
                for (i, &v) in geometry_nodes.iter().enumerate() {
                    px += geom_shape.values[i] * mesh.nodes[v].x;
                    py += geom_shape.values[i] * mesh.nodes[v].y;
                }

                let hessians = reference_hessians(elem_type, self.degree, xi, eta, zeta)
                    .map(|h| {
                        if geometry_degree == PolynomialDegree::P2 {
                            let geometry_coordinates: Vec<f64> = coords
                                .iter()
                                .flat_map(|coordinate| coordinate.iter().copied())
                                .collect();
                            let geometry_reference_hessians =
                                reference_hessians(elem_type, geometry_degree, xi, eta, zeta)
                                    .unwrap_or_default();
                            let geometry_hessians = combine_geometry_hessians(
                                &geometry_reference_hessians,
                                &geometry_coordinates,
                                2,
                            );
                            transform_curved_hessians(
                                &h,
                                &shape.gradients,
                                &geometry_hessians,
                                &jac.inverse,
                                2,
                            )
                        } else if geometry_degree == PolynomialDegree::P3 {
                            // P3 geometry is admitted for first-derivative
                            // assembly.  The higher-order geometric Hessian
                            // contract is not yet available for this public
                            // tabulation path, so do not return an affine
                            // Hessian that would be physically incorrect on a
                            // curved P3 map.
                            Vec::new()
                        } else {
                            transform_hessians(&h, &jac.inverse, 2)
                        }
                    })
                    .unwrap_or_default();

                (jac.det.abs(), phys_grads, hessians, [px, py, 0.0])
            }
            3 => {
                let coords: Vec<[f64; 3]> = geometry_nodes
                    .iter()
                    .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y, mesh.nodes[v].z])
                    .collect();
                let jac = Jacobian::from_3d(&geom_shape.gradients, geom_shape.grad_dim, &coords);

                let mut phys_grads = Vec::with_capacity(n_dofs * 3);
                for g in shape.gradients.chunks(shape.grad_dim) {
                    let mut pg = [0.0; 3];
                    jac.transform_gradient_into(g, &mut pg);
                    phys_grads.extend_from_slice(&pg);
                }

                let mut px = 0.0;
                let mut py = 0.0;
                let mut pz = 0.0;
                for (i, &v) in geometry_nodes.iter().enumerate() {
                    px += geom_shape.values[i] * mesh.nodes[v].x;
                    py += geom_shape.values[i] * mesh.nodes[v].y;
                    pz += geom_shape.values[i] * mesh.nodes[v].z;
                }

                let hessians = reference_hessians(elem_type, self.degree, xi, eta, zeta)
                    .map(|h| {
                        if geometry_degree == PolynomialDegree::P2 {
                            let geometry_coordinates: Vec<f64> = coords
                                .iter()
                                .flat_map(|coordinate| coordinate.iter().copied())
                                .collect();
                            let geometry_reference_hessians =
                                reference_hessians(elem_type, geometry_degree, xi, eta, zeta)
                                    .unwrap_or_default();
                            let geometry_hessians = combine_geometry_hessians(
                                &geometry_reference_hessians,
                                &geometry_coordinates,
                                3,
                            );
                            transform_curved_hessians(
                                &h,
                                &shape.gradients,
                                &geometry_hessians,
                                &jac.inverse,
                                3,
                            )
                        } else if geometry_degree == PolynomialDegree::P3 {
                            // See the 2-D branch above: curved P3 geometry
                            // currently guarantees values and gradients, but
                            // deliberately does not fabricate Hessians.
                            Vec::new()
                        } else {
                            transform_hessians(&h, &jac.inverse, 3)
                        }
                    })
                    .unwrap_or_default();

                (jac.det.abs(), phys_grads, hessians, [px, py, pz])
            }
            _ => unreachable!("spatial dimension must be 2 or 3"),
        };

        BasisTabulation {
            n_dofs,
            spatial_dim,
            values: shape.values,
            derivatives: phys_grads,
            hessians,
            det_j,
            physical_coords: phys_coords,
        }
    }

    fn boundary_dofs(&self, mesh: &Mesh, boundary_marker: i32) -> Vec<usize> {
        let mut dofs = std::collections::HashSet::new();
        // Vertex DOFs (all orders)
        for face in &mesh.boundaries {
            if face.marker == boundary_marker {
                if self.degree == PolynomialDegree::P1 {
                    let element = &mesh.elements[face.element_idx];
                    let vertices = &element.nodes[..element.element_type.num_vertices()];
                    for &node in &face.nodes {
                        if vertices.contains(&node) {
                            dofs.insert(node);
                        }
                    }
                } else {
                    for &node in &face.nodes {
                        dofs.insert(node);
                    }
                    if self.degree == PolynomialDegree::P3 && Self::has_explicit_p3_geometry(mesh) {
                        let element = &mesh.elements[face.element_idx];
                        dofs.extend(explicit_p3_boundary_trace_nodes(element, &face.nodes));
                    }
                }
            }
        }

        // Edge midpoint and face-centre DOFs for P2+
        if matches!(self.degree, PolynomialDegree::P2 | PolynomialDegree::P3)
            && !Self::has_explicit_p2_geometry(mesh)
            && !Self::has_explicit_p3_geometry(mesh)
        {
            let topo = self.topology(mesh);
            let n_nodes = mesh.num_nodes();
            let n_edges = topo.num_edges();
            if let Some(boundary_edges) = topo.boundary_edges.get(&boundary_marker) {
                for &edge_idx in boundary_edges {
                    match self.degree {
                        PolynomialDegree::P2 => {
                            dofs.insert(n_nodes + edge_idx);
                        }
                        PolynomialDegree::P3 => {
                            dofs.insert(n_nodes + edge_idx * 2);
                            dofs.insert(n_nodes + edge_idx * 2 + 1);
                        }
                        _ => {}
                    }
                }
            }

            // Q2 hex face-centre DOFs that lie on the boundary.
            if matches!(self.degree, PolynomialDegree::P2) {
                let has_face_centres = mesh
                    .elements
                    .iter()
                    .any(|e| e.element_type.p2_has_face_centres());
                if has_face_centres {
                    let face_offset = n_nodes + n_edges;
                    if let Some(boundary_faces) = topo.boundary_faces.get(&boundary_marker) {
                        for &face_idx in boundary_faces {
                            dofs.insert(face_offset + face_idx);
                        }
                    }
                }
            }
        }

        if matches!(self.degree, PolynomialDegree::P3) && !Self::has_explicit_p3_geometry(mesh) {
            let topo = self.topology(mesh);
            let face_offset = mesh.num_nodes() + 2 * topo.num_edges();
            let face_dofs_per_face = mesh
                .elements
                .first()
                .map(|element| match element.element_type {
                    ElementType::Tetrahedron => 1,
                    ElementType::Hexahedron => 4,
                    ElementType::Triangle | ElementType::Quadrilateral => 0,
                })
                .unwrap_or(0);
            if let Some(boundary_faces) = topo.boundary_faces.get(&boundary_marker) {
                for &face_idx in boundary_faces {
                    for local in 0..face_dofs_per_face {
                        dofs.insert(face_offset + face_idx * face_dofs_per_face + local);
                    }
                }
            }
        }
        let mut result: Vec<usize> = dofs.into_iter().collect();
        result.sort_unstable();
        result
    }

    fn dof_coordinate(&self, mesh: &Mesh, dof: usize) -> Option<[f64; 3]> {
        if dof < mesh.num_nodes() {
            let point = mesh.nodes[dof];
            return Some([point.x, point.y, point.z]);
        }
        if self.degree == PolynomialDegree::P3 {
            let topology = self.topology(mesh);
            let edge_offset = mesh.num_nodes();
            let edge_end = edge_offset + 2 * topology.num_edges();
            if dof < edge_end {
                let edge_slot = dof - edge_offset;
                let (a, b) = topology.edges[edge_slot / 2];
                let fraction = if edge_slot.is_multiple_of(2) {
                    1.0 / 3.0
                } else {
                    2.0 / 3.0
                };
                let pa = mesh.nodes[a];
                let pb = mesh.nodes[b];
                return Some([
                    (1.0 - fraction) * pa.x + fraction * pb.x,
                    (1.0 - fraction) * pa.y + fraction * pb.y,
                    (1.0 - fraction) * pa.z + fraction * pb.z,
                ]);
            }
            let face_offset = edge_end;
            let face_dofs_per_face = mesh
                .elements
                .first()
                .map(|element| match element.element_type {
                    ElementType::Tetrahedron => 1,
                    ElementType::Hexahedron => 4,
                    ElementType::Triangle | ElementType::Quadrilateral => 0,
                })
                .unwrap_or(0);
            let face_dof_count = face_dofs_per_face * topology.num_faces();
            if dof < face_offset + face_dof_count {
                let face_slot = dof - face_offset;
                let face_idx = face_slot / face_dofs_per_face;
                let local_slot = face_slot % face_dofs_per_face;
                let nodes = topology.faces[face_idx].nodes();
                if face_dofs_per_face == 1 {
                    let count = nodes.len() as f64;
                    return Some([
                        nodes.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / count,
                        nodes.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / count,
                        nodes.iter().map(|&node| mesh.nodes[node].z).sum::<f64>() / count,
                    ]);
                }
                if face_dofs_per_face == 4 {
                    return Some(p3_hex_face_point(mesh, &topology, face_idx, local_slot));
                }

                const FACE_COORDINATES: [(f64, f64); 4] = [
                    (1.0 / 3.0, 1.0 / 3.0),
                    (2.0 / 3.0, 1.0 / 3.0),
                    (2.0 / 3.0, 2.0 / 3.0),
                    (1.0 / 3.0, 2.0 / 3.0),
                ];
                let (u, v) = FACE_COORDINATES[local_slot];
                let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
                let mut point = [0.0; 3];
                for (weight, &node) in weights.iter().zip(&nodes) {
                    let coordinates = mesh.nodes[node];
                    point[0] += weight * coordinates.x;
                    point[1] += weight * coordinates.y;
                    point[2] += weight * coordinates.z;
                }
                return Some(point);
            }

            if face_dofs_per_face == 4 {
                let cell_offset = face_offset + face_dof_count;
                if dof >= cell_offset {
                    let cell_slot = dof - cell_offset;
                    let cell_idx = cell_slot / 8;
                    let local_slot = cell_slot % 8;
                    let element = mesh
                        .elements
                        .iter()
                        .filter(|element| element.element_type == ElementType::Hexahedron)
                        .nth(cell_idx)?;
                    const CELL_COORDINATES: [(f64, f64, f64); 8] = [
                        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                        (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                        (2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0),
                        (1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0),
                        (1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0),
                        (2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0),
                        (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
                        (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
                    ];
                    let (u, v, w) = CELL_COORDINATES[local_slot];
                    let vertices = element.vertices();
                    let weights = [
                        (1.0 - u) * (1.0 - v) * (1.0 - w),
                        u * (1.0 - v) * (1.0 - w),
                        u * v * (1.0 - w),
                        (1.0 - u) * v * (1.0 - w),
                        (1.0 - u) * (1.0 - v) * w,
                        u * (1.0 - v) * w,
                        u * v * w,
                        (1.0 - u) * v * w,
                    ];
                    let mut point = [0.0; 3];
                    for (weight, &node) in weights.iter().zip(vertices.iter()) {
                        let coordinates = mesh.nodes[node];
                        point[0] += weight * coordinates.x;
                        point[1] += weight * coordinates.y;
                        point[2] += weight * coordinates.z;
                    }
                    return Some(point);
                }
            }
            return None;
        }
        if self.degree != PolynomialDegree::P2 {
            return None;
        }

        let topology = self.topology(mesh);
        let edge_offset = mesh.num_nodes();
        if dof < edge_offset + topology.num_edges() {
            let (a, b) = topology.edges[dof - edge_offset];
            let pa = mesh.nodes[a];
            let pb = mesh.nodes[b];
            return Some([
                0.5 * (pa.x + pb.x),
                0.5 * (pa.y + pb.y),
                0.5 * (pa.z + pb.z),
            ]);
        }

        let face_offset = edge_offset + topology.num_edges();
        let has_face_centres = mesh
            .elements
            .iter()
            .any(|element| element.element_type.p2_has_face_centres());
        if has_face_centres && dof < face_offset + topology.num_faces() {
            let nodes = topology.faces[dof - face_offset].nodes();
            let count = nodes.len() as f64;
            return Some([
                nodes.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / count,
                nodes.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / count,
                nodes.iter().map(|&node| mesh.nodes[node].z).sum::<f64>() / count,
            ]);
        }

        let interior_offset = if has_face_centres {
            face_offset + topology.num_faces()
        } else {
            face_offset
        };
        let interior_index = dof.checked_sub(interior_offset)?;
        let element = mesh
            .elements
            .iter()
            .filter(|element| element.element_type.p2_has_interior())
            .nth(interior_index)?;
        let vertices = element.vertices();
        let count = vertices.len() as f64;
        Some([
            vertices.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / count,
            vertices.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / count,
            vertices.iter().map(|&node| mesh.nodes[node].z).sum::<f64>() / count,
        ])
    }

    fn dof_coordinates(&self, mesh: &Mesh) -> Option<Vec<[f64; 3]>> {
        let mut coordinates: Vec<[f64; 3]> = mesh
            .nodes
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect();
        if self.degree == PolynomialDegree::P1 {
            if Self::has_explicit_p2_geometry(mesh) || Self::has_explicit_p3_geometry(mesh) {
                coordinates.truncate(Self::vertex_dof_count(mesh));
            }
            return Some(coordinates);
        }
        if (self.degree == PolynomialDegree::P2 && Self::has_explicit_p2_geometry(mesh))
            || (self.degree == PolynomialDegree::P3 && Self::has_explicit_p3_geometry(mesh))
        {
            return Some(coordinates);
        }
        if self.degree == PolynomialDegree::P3 {
            let topology = self.topology(mesh);
            coordinates.extend(topology.edges.iter().flat_map(|&(a, b)| {
                let pa = mesh.nodes[a];
                let pb = mesh.nodes[b];
                [
                    [
                        (2.0 * pa.x + pb.x) / 3.0,
                        (2.0 * pa.y + pb.y) / 3.0,
                        (2.0 * pa.z + pb.z) / 3.0,
                    ],
                    [
                        (pa.x + 2.0 * pb.x) / 3.0,
                        (pa.y + 2.0 * pb.y) / 3.0,
                        (pa.z + 2.0 * pb.z) / 3.0,
                    ],
                ]
            }));
            let face_dofs_per_face = mesh
                .elements
                .first()
                .map(|element| match element.element_type {
                    ElementType::Tetrahedron => 1,
                    ElementType::Hexahedron => 4,
                    ElementType::Triangle | ElementType::Quadrilateral => 0,
                })
                .unwrap_or(0);
            for (face_idx, face) in topology.faces.iter().enumerate() {
                let nodes = face.nodes();
                if face_dofs_per_face == 1 {
                    let count = nodes.len() as f64;
                    coordinates.push([
                        nodes.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / count,
                        nodes.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / count,
                        nodes.iter().map(|&node| mesh.nodes[node].z).sum::<f64>() / count,
                    ]);
                } else if face_dofs_per_face == 4 {
                    for local_slot in 0..4 {
                        coordinates.push(p3_hex_face_point(mesh, &topology, face_idx, local_slot));
                    }
                } else {
                    const FACE_COORDINATES: [(f64, f64); 4] = [
                        (1.0 / 3.0, 1.0 / 3.0),
                        (2.0 / 3.0, 1.0 / 3.0),
                        (2.0 / 3.0, 2.0 / 3.0),
                        (1.0 / 3.0, 2.0 / 3.0),
                    ];
                    for &(u, v) in &FACE_COORDINATES[..face_dofs_per_face] {
                        let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
                        let mut point = [0.0; 3];
                        for (weight, &node) in weights.iter().zip(&nodes) {
                            let coordinates = mesh.nodes[node];
                            point[0] += weight * coordinates.x;
                            point[1] += weight * coordinates.y;
                            point[2] += weight * coordinates.z;
                        }
                        coordinates.push(point);
                    }
                }
            }
            if face_dofs_per_face == 4 {
                const CELL_COORDINATES: [(f64, f64, f64); 8] = [
                    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                    (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                    (2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0),
                    (1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0),
                    (1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0),
                    (2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0),
                    (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
                    (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
                ];
                for element in mesh
                    .elements
                    .iter()
                    .filter(|element| element.element_type == ElementType::Hexahedron)
                {
                    let vertices = element.vertices();
                    for &(u, v, w) in &CELL_COORDINATES {
                        let weights = [
                            (1.0 - u) * (1.0 - v) * (1.0 - w),
                            u * (1.0 - v) * (1.0 - w),
                            u * v * (1.0 - w),
                            (1.0 - u) * v * (1.0 - w),
                            (1.0 - u) * (1.0 - v) * w,
                            u * (1.0 - v) * w,
                            u * v * w,
                            (1.0 - u) * v * w,
                        ];
                        let mut point = [0.0; 3];
                        for (weight, &node) in weights.iter().zip(vertices.iter()) {
                            let coordinates = mesh.nodes[node];
                            point[0] += weight * coordinates.x;
                            point[1] += weight * coordinates.y;
                            point[2] += weight * coordinates.z;
                        }
                        coordinates.push(point);
                    }
                }
            }
            for element in &mesh.elements {
                let vertices = element.vertices();
                match element.element_type {
                    ElementType::Triangle => {
                        let count = vertices.len() as f64;
                        coordinates.push([
                            vertices.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / count,
                            vertices.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / count,
                            vertices.iter().map(|&node| mesh.nodes[node].z).sum::<f64>() / count,
                        ]);
                    }
                    ElementType::Quadrilateral => {
                        for &(u, v) in &[
                            (1.0 / 3.0, 1.0 / 3.0),
                            (2.0 / 3.0, 1.0 / 3.0),
                            (1.0 / 3.0, 2.0 / 3.0),
                            (2.0 / 3.0, 2.0 / 3.0),
                        ] {
                            let weights =
                                [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
                            let mut point = [0.0; 3];
                            for (weight, &node) in weights.iter().zip(vertices.iter()) {
                                let coordinates = mesh.nodes[node];
                                point[0] += weight * coordinates.x;
                                point[1] += weight * coordinates.y;
                                point[2] += weight * coordinates.z;
                            }
                            coordinates.push(point);
                        }
                    }
                    ElementType::Tetrahedron | ElementType::Hexahedron => {}
                }
            }

            debug_assert_eq!(coordinates.len(), self.num_global_dofs(mesh));
            return Some(coordinates);
        }
        if self.degree != PolynomialDegree::P2 {
            return None;
        }

        let topology = self.topology(mesh);
        coordinates.extend(topology.edges.iter().map(|&(a, b)| {
            let pa = mesh.nodes[a];
            let pb = mesh.nodes[b];
            [
                0.5 * (pa.x + pb.x),
                0.5 * (pa.y + pb.y),
                0.5 * (pa.z + pb.z),
            ]
        }));
        if mesh
            .elements
            .iter()
            .any(|element| element.element_type.p2_has_face_centres())
        {
            coordinates.extend(topology.faces.iter().map(|face| {
                let nodes = face.nodes();
                let count = nodes.len() as f64;
                [
                    nodes.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / count,
                    nodes.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / count,
                    nodes.iter().map(|&node| mesh.nodes[node].z).sum::<f64>() / count,
                ]
            }));
        }
        coordinates.extend(
            mesh.elements
                .iter()
                .filter(|element| element.element_type.p2_has_interior())
                .map(|element| {
                    let vertices = element.vertices();
                    let count = vertices.len() as f64;
                    [
                        vertices.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / count,
                        vertices.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / count,
                        vertices.iter().map(|&node| mesh.nodes[node].z).sum::<f64>() / count,
                    ]
                }),
        );
        debug_assert_eq!(coordinates.len(), self.num_global_dofs(mesh));
        Some(coordinates)
    }
}

/// Vector-valued H1-conforming function space.
///
/// Each mesh node gets `num_components` DOFs (one per spatial dimension).
/// Used for displacement fields in elasticity.
///
/// DOF ordering is interleaved: `[node0_x, node0_y, node0_z, node1_x, ...]`
#[derive(Debug, Clone)]
pub struct VectorH1Space {
    scalar_space: H1Space,
    components: usize,
}

impl VectorH1Space {
    /// Create a vector H1 space with the given polynomial degree and number of components.
    ///
    /// `components` is typically the spatial dimension (2 for 2D, 3 for 3D).
    pub fn new(degree: PolynomialDegree, components: usize) -> Self {
        Self {
            scalar_space: H1Space::new(degree),
            components,
        }
    }

    /// Polynomial degree of the underlying scalar Lagrange space.
    pub fn degree(&self) -> PolynomialDegree {
        self.scalar_space.degree
    }
}

impl FunctionSpace for VectorH1Space {
    fn field_type(&self) -> FieldType {
        FieldType::Vector
    }

    fn polynomial_degree(&self) -> usize {
        self.scalar_space.degree.degree()
    }

    fn num_components(&self) -> usize {
        self.components
    }

    fn num_global_dofs(&self, mesh: &Mesh) -> usize {
        self.scalar_space.num_global_dofs(mesh) * self.components
    }

    fn num_local_dofs(&self, elem_type: ElementType) -> usize {
        self.scalar_space.num_local_dofs(elem_type) * self.components
    }

    fn local_to_global(&self, mesh: &Mesh, elem_idx: usize) -> Vec<usize> {
        let scalar_dofs = self.scalar_space.local_to_global(mesh, elem_idx);
        let mut dofs = Vec::with_capacity(scalar_dofs.len() * self.components);
        // Interleaved ordering: [node0_x, node0_y, node0_z, node1_x, ...]
        for &node in &scalar_dofs {
            for c in 0..self.components {
                dofs.push(node * self.components + c);
            }
        }
        dofs
    }

    fn tabulate(
        &self,
        mesh: &Mesh,
        elem_idx: usize,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> BasisTabulation {
        let scalar_tab = self.scalar_space.tabulate(mesh, elem_idx, xi, eta, zeta);
        let n_scalar = scalar_tab.n_dofs;
        let dim = scalar_tab.spatial_dim;
        let n_vector_dofs = n_scalar * self.components;

        // Expand scalar basis to vector basis functions.
        // For each scalar basis phi_i, create `components` vector basis functions:
        //   e.g. (phi_i, 0, 0), (0, phi_i, 0), (0, 0, phi_i)
        //
        // Values layout: [phi_0, 0, 0,  0, phi_0, 0,  0, 0, phi_0,  phi_1, 0, 0, ...]
        let mut values = vec![0.0; n_vector_dofs * dim];
        for i in 0..n_scalar {
            for c in 0..self.components {
                let vec_dof = i * self.components + c;
                // Vector DOF `vec_dof` has value phi_i in component c, 0 elsewhere
                values[vec_dof * dim + c] = scalar_tab.values[i];
            }
        }

        // Derivatives: for elasticity we need gradients of vector basis functions.
        // grad(e_c * phi_i) has component [c][d] = d(phi_i)/dx_d, others zero.
        // We store as strain-compatible: [n_vector_dofs * dim] where each block of
        // `dim` values is the gradient of the c-th component of the vector DOF.
        //
        // For vector DOF (i, c): derivatives at [vec_dof * dim + d] = dphi_i/dx_d if
        // this is the c-th component, but we need the full gradient tensor.
        // Store as: [n_vector_dofs * dim] = gradient of the "active" component.
        let mut derivatives = vec![0.0; n_vector_dofs * dim];
        for i in 0..n_scalar {
            for c in 0..self.components {
                let vec_dof = i * self.components + c;
                for d in 0..dim {
                    derivatives[vec_dof * dim + d] = scalar_tab.grad(i, d);
                }
            }
        }

        BasisTabulation {
            n_dofs: n_vector_dofs,
            spatial_dim: dim,
            values,
            derivatives,
            hessians: Vec::new(),
            det_j: scalar_tab.det_j,
            physical_coords: scalar_tab.physical_coords,
        }
    }

    fn boundary_dofs(&self, mesh: &Mesh, boundary_marker: i32) -> Vec<usize> {
        let scalar_dofs = self.scalar_space.boundary_dofs(mesh, boundary_marker);
        let mut dofs = Vec::with_capacity(scalar_dofs.len() * self.components);
        for &node in &scalar_dofs {
            for c in 0..self.components {
                dofs.push(node * self.components + c);
            }
        }
        dofs
    }

    fn dof_coordinate(&self, mesh: &Mesh, dof: usize) -> Option<[f64; 3]> {
        if dof >= self.num_global_dofs(mesh) {
            return None;
        }
        self.scalar_space
            .dof_coordinate(mesh, dof / self.components)
    }

    fn dof_coordinates(&self, mesh: &Mesh) -> Option<Vec<[f64; 3]>> {
        Some(
            self.scalar_space
                .dof_coordinates(mesh)?
                .into_iter()
                .flat_map(|coordinate| std::iter::repeat_n(coordinate, self.components))
                .collect(),
        )
    }
}

fn reference_hessians(
    elem_type: ElementType,
    degree: PolynomialDegree,
    xi: f64,
    eta: f64,
    zeta: f64,
) -> Option<Vec<f64>> {
    if degree != PolynomialDegree::P2
        && !(elem_type == ElementType::Tetrahedron && degree == PolynomialDegree::P3)
    {
        return None;
    }

    match elem_type {
        ElementType::Triangle => {
            // P2 triangle with barycentric basis ordered as:
            // v0, v1, v2, e01, e12, e20. Each block is row-major 2x2.
            Some(vec![
                4.0, 4.0, 4.0, 4.0, // v0
                4.0, 0.0, 0.0, 0.0, // v1
                0.0, 0.0, 0.0, 4.0, // v2
                -8.0, -4.0, -4.0, 0.0, // e01
                0.0, 4.0, 4.0, 0.0, // e12
                0.0, -4.0, -4.0, -8.0, // e20
            ])
        }
        ElementType::Tetrahedron if degree == PolynomialDegree::P3 => Some(
            crate::basis::p3_tetrahedron_hessian(xi, eta, zeta)
                .into_iter()
                .flat_map(|matrix| matrix.into_iter().flatten())
                .collect(),
        ),
        ElementType::Tetrahedron => {
            let dl = [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
            let mut h = vec![0.0; 10 * 9];
            for a in 0..4 {
                for r in 0..3 {
                    for c in 0..3 {
                        h[a * 9 + r * 3 + c] = 4.0 * dl[a][r] * dl[a][c];
                    }
                }
            }
            for (k, (a, b)) in edges.iter().enumerate() {
                let idx = 4 + k;
                for r in 0..3 {
                    for c in 0..3 {
                        h[idx * 9 + r * 3 + c] =
                            4.0 * (dl[*a][r] * dl[*b][c] + dl[*b][r] * dl[*a][c]);
                    }
                }
            }
            Some(h)
        }
        ElementType::Quadrilateral => Some(q2_quadrilateral_reference_hessians(xi, eta)),
        ElementType::Hexahedron => Some(q2_hexahedron_reference_hessians(xi, eta, zeta)),
    }
}

fn q2_1d_values(s: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    (
        [
            0.5 * s * (s - 1.0),
            (1.0 - s) * (1.0 + s),
            0.5 * s * (s + 1.0),
        ],
        [s - 0.5, -2.0 * s, s + 0.5],
        [1.0, -2.0, 1.0],
    )
}

fn q2_quadrilateral_reference_hessians(xi: f64, eta: f64) -> Vec<f64> {
    let (lx, dx, ddx) = q2_1d_values(xi);
    let (ly, dy, ddy) = q2_1d_values(eta);
    let node_indices = [
        (0, 0),
        (2, 0),
        (2, 2),
        (0, 2),
        (1, 0),
        (2, 1),
        (1, 2),
        (0, 1),
        (1, 1),
    ];

    let mut h = Vec::with_capacity(9 * 4);
    for (i, j) in node_indices {
        let dxy = dx[i] * dy[j];
        h.extend_from_slice(&[ddx[i] * ly[j], dxy, dxy, lx[i] * ddy[j]]);
    }
    h
}

fn q2_hexahedron_reference_hessians(xi: f64, eta: f64, zeta: f64) -> Vec<f64> {
    let (lx, dx, ddx) = q2_1d_values(xi);
    let (ly, dy, ddy) = q2_1d_values(eta);
    let (lz, dz, ddz) = q2_1d_values(zeta);
    let node_indices = [
        (0, 0, 0),
        (2, 0, 0),
        (2, 2, 0),
        (0, 2, 0),
        (0, 0, 2),
        (2, 0, 2),
        (2, 2, 2),
        (0, 2, 2),
        (1, 0, 0),
        (2, 1, 0),
        (1, 2, 0),
        (0, 1, 0),
        (1, 0, 2),
        (2, 1, 2),
        (1, 2, 2),
        (0, 1, 2),
        (0, 0, 1),
        (2, 0, 1),
        (2, 2, 1),
        (0, 2, 1),
        (1, 1, 0),
        (1, 1, 2),
        (1, 0, 1),
        (1, 2, 1),
        (0, 1, 1),
        (2, 1, 1),
        (1, 1, 1),
    ];

    let mut h = Vec::with_capacity(27 * 9);
    for (i, j, k) in node_indices {
        let dxy = dx[i] * dy[j] * lz[k];
        let dxz = dx[i] * ly[j] * dz[k];
        let dyz = lx[i] * dy[j] * dz[k];
        h.extend_from_slice(&[
            ddx[i] * ly[j] * lz[k],
            dxy,
            dxz,
            dxy,
            lx[i] * ddy[j] * lz[k],
            dyz,
            dxz,
            dyz,
            lx[i] * ly[j] * ddz[k],
        ]);
    }
    h
}

fn combine_geometry_hessians(
    reference: &[f64],
    geometry_coordinates: &[f64],
    dim: usize,
) -> Vec<f64> {
    let node_hessian_size = dim * dim;
    let mut geometry = vec![0.0; dim * node_hessian_size];
    for (node, hessian) in reference.chunks(node_hessian_size).enumerate() {
        for coordinate in 0..dim {
            let value = geometry_coordinates[node * dim + coordinate];
            for entry in 0..node_hessian_size {
                geometry[coordinate * node_hessian_size + entry] += value * hessian[entry];
            }
        }
    }
    geometry
}

fn transform_curved_hessians(
    reference: &[f64],
    reference_gradients: &[f64],
    geometry_hessians: &[f64],
    jac_inverse: &[f64],
    dim: usize,
) -> Vec<f64> {
    let node_hessian_size = dim * dim;
    let basis_count = reference.len() / node_hessian_size;
    let mut out = vec![0.0; reference.len()];
    for basis in 0..basis_count {
        let gradients = &reference_gradients[basis * dim..(basis + 1) * dim];
        for a in 0..dim {
            for b in 0..dim {
                let mut value = 0.0;
                for r in 0..dim {
                    for s in 0..dim {
                        value += jac_inverse[r * dim + a]
                            * reference[basis * node_hessian_size + r * dim + s]
                            * jac_inverse[s * dim + b];
                        for coordinate in 0..dim {
                            for q in 0..dim {
                                value -= jac_inverse[s * dim + b]
                                    * jac_inverse[r * dim + coordinate]
                                    * geometry_hessians
                                        [coordinate * node_hessian_size + q * dim + s]
                                    * jac_inverse[q * dim + a]
                                    * gradients[r];
                            }
                        }
                    }
                }
                out[basis * node_hessian_size + a * dim + b] = value;
            }
        }
    }
    out
}

fn transform_hessians(reference: &[f64], jac_inverse: &[f64], dim: usize) -> Vec<f64> {
    let n = reference.len() / (dim * dim);
    let mut out = vec![0.0; reference.len()];
    for basis in 0..n {
        for a in 0..dim {
            for b in 0..dim {
                let mut value = 0.0;
                for r in 0..dim {
                    for s in 0..dim {
                        // H_phys = J^{-T} H_ref J^{-1}
                        value += jac_inverse[r * dim + a]
                            * reference[basis * dim * dim + r * dim + s]
                            * jac_inverse[s * dim + b];
                    }
                }
                out[basis * dim * dim + a * dim + b] = value;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    fn explicit_p3_triangle_mesh(curved: bool) -> Mesh {
        // P3 triangle ordering: vertices, two nodes per edge, then the
        // interior node.  The coordinates are nodal coordinates for the
        // reference triangle, with one interior node optionally displaced
        // to make the isoparametric map genuinely curved.
        let mut mesh = Mesh::new(2);
        let mut points = vec![
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
        if curved {
            points[9].1 += 0.08;
        }
        for (x, y) in points {
            mesh.add_node(crate::mesh::Point::new_2d(x, y));
        }
        mesh.add_element(ElementType::Triangle, (0..10).collect());
        mesh
    }

    #[test]
    fn explicit_p3_geometry_is_valid_numbered_and_tabulated() {
        let mesh = explicit_p3_triangle_mesh(true);
        let space = H1Space::new(PolynomialDegree::P3);

        assert!(space.validate(&mesh).is_ok());
        assert_eq!(space.num_global_dofs(&mesh), 10);
        assert_eq!(space.local_to_global(&mesh, 0), (0..10).collect::<Vec<_>>());
        assert_eq!(space.dof_coordinates(&mesh).unwrap().len(), 10);

        let tabulation = space.tabulate(&mesh, 0, 0.21, 0.27, 0.0);
        assert_eq!(tabulation.n_dofs, 10);
        assert!((tabulation.values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(tabulation.det_j.is_finite() && tabulation.det_j > 0.0);
        assert!(tabulation.derivatives.iter().all(|value| value.is_finite()));
        assert!(tabulation.hessians.is_empty());
        // The displaced bubble must affect the physical map; this catches a
        // regression that silently falls back to vertex-only affine geometry.
        assert!(tabulation.physical_coords[1] > 0.27);
    }

    #[test]
    fn explicit_p3_geometry_drives_finite_symmetric_assembly() {
        let mesh = explicit_p3_triangle_mesh(true);
        let stiffness = crate::assembly::assemble_stiffness(&mesh, PolynomialDegree::P3);
        let mass = crate::assembly::assemble_mass(&mesh, PolynomialDegree::P3);

        assert_eq!(stiffness.dim, 10);
        assert_eq!(mass.dim, 10);
        assert!(stiffness.values.iter().all(|value| value.is_finite()));
        assert!(mass.values.iter().all(|value| value.is_finite()));

        let stiffness_dense =
            stiffness_to_dense(&stiffness.rows, &stiffness.cols, &stiffness.values, 10);
        let mass_dense = stiffness_to_dense(&mass.rows, &mass.cols, &mass.values, 10);
        for matrix in [stiffness_dense, mass_dense] {
            for row in 0..10 {
                for column in 0..10 {
                    assert!((matrix[row][column] - matrix[column][row]).abs() < 1e-10);
                }
            }
        }
    }

    fn stiffness_to_dense(
        rows: &[usize],
        cols: &[usize],
        values: &[f64],
        dimension: usize,
    ) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; dimension]; dimension];
        for ((&row, &column), &value) in rows.iter().zip(cols).zip(values) {
            dense[row][column] += value;
        }
        dense
    }

    #[test]
    fn test_h1_space_p3_triangle_dofs_and_partition_of_unity() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P3);
        let topology = crate::mesh::topology::MeshTopology::build(&mesh);
        let expected = mesh.num_nodes() + 2 * topology.num_edges() + mesh.num_elements();

        assert!(space.validate(&mesh).is_ok());
        assert_eq!(space.num_global_dofs(&mesh), expected);
        assert_eq!(space.num_local_dofs(ElementType::Triangle), 10);

        let mut seen = std::collections::HashSet::new();
        for element in 0..mesh.num_elements() {
            let dofs = space.local_to_global(&mesh, element);
            assert_eq!(dofs.len(), 10);
            let mut local = dofs.clone();
            local.sort_unstable();
            local.dedup();
            assert_eq!(local.len(), dofs.len());
            seen.extend(dofs);
            let tabulation = space.tabulate(&mesh, element, 0.21, 0.27, 0.0);
            assert!((tabulation.values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert_eq!(seen.len(), expected);
    }

    #[test]
    fn test_h1_space_p3_quadrilateral_dofs_and_partition_of_unity() {
        let mesh = unit_square_quads(2);
        let space = H1Space::new(PolynomialDegree::P3);
        let topology = crate::mesh::topology::MeshTopology::build(&mesh);
        let expected = mesh.num_nodes() + 2 * topology.num_edges() + 4 * mesh.num_elements();

        assert!(space.validate(&mesh).is_ok());
        assert_eq!(space.num_global_dofs(&mesh), expected);
        assert_eq!(space.num_local_dofs(ElementType::Quadrilateral), 16);

        let mut seen = std::collections::HashSet::new();
        for element in 0..mesh.num_elements() {
            let dofs = space.local_to_global(&mesh, element);
            assert_eq!(dofs.len(), 16);
            for &dof in &dofs {
                assert!(dof < expected);
                seen.insert(dof);
            }
            let tabulation = space.tabulate(&mesh, element, 0.13, -0.23, 0.0);
            assert!((tabulation.values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert_eq!(seen.len(), expected);
    }

    use super::*;
    use crate::mesh::{unit_cube_hexahedra, unit_square_quads, unit_square_triangles};

    #[test]
    fn test_h1_space_p1_dof_count() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);
        assert_eq!(space.num_global_dofs(&mesh), mesh.num_nodes());
        assert_eq!(space.num_local_dofs(ElementType::Triangle), 3);
    }

    #[test]
    fn test_h1_space_tabulation_partition_of_unity() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);

        // Tabulate at centroid of first element
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);

        // Partition of unity: sum of basis values == 1
        let sum: f64 = tab.values.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "Partition of unity failed: sum = {sum}"
        );
    }

    #[test]
    fn test_h1_space_local_to_global() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);

        let dofs = space.local_to_global(&mesh, 0);
        assert_eq!(dofs.len(), 3);
        // DOFs should be valid node indices
        for &d in &dofs {
            assert!(d < mesh.num_nodes());
        }
    }

    #[test]
    fn test_vector_h1_space_dof_count() {
        let mesh = unit_square_triangles(4);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);

        assert_eq!(space.num_global_dofs(&mesh), mesh.num_nodes() * 2);
        assert_eq!(space.num_local_dofs(ElementType::Triangle), 6);
    }

    #[test]
    fn test_vector_h1_local_to_global_interleaved() {
        let mesh = unit_square_triangles(4);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);

        let dofs = space.local_to_global(&mesh, 0);
        assert_eq!(dofs.len(), 6); // 3 nodes * 2 components

        // Check interleaved ordering
        let scalar_dofs = H1Space::new(PolynomialDegree::P1).local_to_global(&mesh, 0);
        for (i, &node) in scalar_dofs.iter().enumerate() {
            assert_eq!(dofs[i * 2], node * 2);
            assert_eq!(dofs[i * 2 + 1], node * 2 + 1);
        }
    }

    #[test]
    fn test_h1_space_p2_dof_count() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P2);
        let topo = crate::mesh::topology::MeshTopology::build(&mesh);
        let expected = mesh.num_nodes() + topo.num_edges();
        assert_eq!(space.num_global_dofs(&mesh), expected);
        assert_eq!(space.num_local_dofs(ElementType::Triangle), 6);
    }

    #[test]
    fn test_h1_space_p2_local_to_global() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P2);
        let n_nodes = mesh.num_nodes();
        let n_global = space.num_global_dofs(&mesh);

        for elem_idx in 0..mesh.num_elements() {
            let dofs = space.local_to_global(&mesh, elem_idx);
            // Three vertex DOFs followed by three edge-midpoint DOFs.
            assert_eq!(dofs.len(), 6);
            // First three are vertex DOFs (< num_nodes).
            for &d in &dofs[..3] {
                assert!(d < n_nodes, "vertex DOF {d} >= num_nodes {n_nodes}");
            }
            // Last 3 are edge midpoint DOFs (>= num_nodes, < total)
            for &d in &dofs[3..] {
                assert!(d >= n_nodes, "edge DOF {d} < num_nodes {n_nodes}");
                assert!(d < n_global, "edge DOF {d} >= n_global {n_global}");
            }
        }
    }

    #[test]
    fn test_h1_space_p2_partition_of_unity() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P2);

        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        assert_eq!(tab.n_dofs, 6);

        let sum: f64 = tab.values.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "P2 partition of unity failed: sum = {sum}"
        );
    }

    #[test]
    fn test_vector_h1_p2_dof_count() {
        let mesh = unit_square_triangles(4);
        let space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let scalar_space = H1Space::new(PolynomialDegree::P2);
        assert_eq!(
            space.num_global_dofs(&mesh),
            scalar_space.num_global_dofs(&mesh) * 2
        );
        assert_eq!(space.num_local_dofs(ElementType::Triangle), 12); // 6 * 2
    }

    #[test]
    fn test_h1_space_q2_quad_dof_count() {
        let mesh = unit_square_quads(3);
        let space = H1Space::new(PolynomialDegree::P2);
        let topo = crate::mesh::topology::MeshTopology::build(&mesh);
        // Q2 quad: vertex + edge + 1 interior per element (no face DOFs in 2D).
        let expected = mesh.num_nodes() + topo.num_edges() + mesh.num_elements();
        assert_eq!(space.num_global_dofs(&mesh), expected);
        assert_eq!(space.num_local_dofs(ElementType::Quadrilateral), 9);
    }

    #[test]
    fn test_h1_space_q2_quad_local_to_global_unique() {
        let mesh = unit_square_quads(3);
        let space = H1Space::new(PolynomialDegree::P2);
        let n_global = space.num_global_dofs(&mesh);

        for elem_idx in 0..mesh.num_elements() {
            let dofs = space.local_to_global(&mesh, elem_idx);
            assert_eq!(dofs.len(), 9);
            let mut sorted = dofs.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), dofs.len(), "duplicate DOFs in element");
            for &d in &dofs {
                assert!(d < n_global, "DOF {d} >= n_global {n_global}");
            }
        }
    }

    #[test]
    fn test_h1_space_q2_quad_partition_of_unity() {
        let mesh = unit_square_quads(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let tab = space.tabulate(&mesh, 0, 0.123, -0.456, 0.0);
        assert_eq!(tab.n_dofs, 9);
        let sum: f64 = tab.values.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "Q2 quad partition of unity: sum = {sum}"
        );
    }

    #[test]
    fn test_h1_space_q2_quad_has_hessians() {
        let mesh = unit_square_quads(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let tab = space.tabulate(&mesh, 0, 0.123, -0.456, 0.0);

        assert!(tab.has_hessians());
        assert_eq!(
            tab.hessians.len(),
            tab.n_dofs * tab.spatial_dim * tab.spatial_dim
        );
        for row in 0..tab.spatial_dim {
            for col in 0..tab.spatial_dim {
                let sum: f64 = (0..tab.n_dofs).map(|i| tab.hessian(i, row, col)).sum();
                assert!(
                    sum.abs() < 1e-12,
                    "Q2 quad Hessian partition component ({row},{col}) sums to {sum}"
                );
            }
        }
    }

    #[test]
    fn test_h1_space_q2_hex_dof_count() {
        let mesh = unit_cube_hexahedra(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let topo = crate::mesh::topology::MeshTopology::build(&mesh);
        // Q2 hex (Hex27): vertex + edge + face + 1 interior per element.
        let expected = mesh.num_nodes() + topo.num_edges() + topo.num_faces() + mesh.num_elements();
        assert_eq!(space.num_global_dofs(&mesh), expected);
        assert_eq!(space.num_local_dofs(ElementType::Hexahedron), 27);
    }

    #[test]
    fn test_h1_space_q2_hex_local_to_global_unique() {
        let mesh = unit_cube_hexahedra(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let n_global = space.num_global_dofs(&mesh);

        for elem_idx in 0..mesh.num_elements() {
            let dofs = space.local_to_global(&mesh, elem_idx);
            assert_eq!(dofs.len(), 27);
            let mut sorted = dofs.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), dofs.len(), "duplicate DOFs in element");
            for &d in &dofs {
                assert!(d < n_global, "DOF {d} >= n_global {n_global}");
            }
        }
    }

    #[test]
    fn test_h1_space_q2_hex_partition_of_unity() {
        let mesh = unit_cube_hexahedra(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let tab = space.tabulate(&mesh, 0, 0.1, -0.2, 0.3);
        assert_eq!(tab.n_dofs, 27);
        let sum: f64 = tab.values.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "Q2 hex partition of unity: sum = {sum}"
        );
    }

    #[test]
    fn test_h1_space_q2_hex_has_hessians() {
        let mesh = unit_cube_hexahedra(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let tab = space.tabulate(&mesh, 0, 0.1, -0.2, 0.3);

        assert!(tab.has_hessians());
        assert_eq!(
            tab.hessians.len(),
            tab.n_dofs * tab.spatial_dim * tab.spatial_dim
        );
        for row in 0..tab.spatial_dim {
            for col in 0..tab.spatial_dim {
                let sum: f64 = (0..tab.n_dofs).map(|i| tab.hessian(i, row, col)).sum();
                assert!(
                    sum.abs() < 1e-12,
                    "Q2 hex Hessian partition component ({row},{col}) sums to {sum}"
                );
            }
        }
    }

    /// Across the full mesh, every global Q2 DOF should be referenced by at least
    /// one element's `local_to_global`, and the union must equal `num_global_dofs`.
    /// This catches face-/interior-DOF indexing bugs that single-element tests miss.
    #[test]
    fn test_h1_space_q2_hex_dofs_cover_global_range() {
        let mesh = unit_cube_hexahedra(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let n_global = space.num_global_dofs(&mesh);

        let mut seen = std::collections::HashSet::with_capacity(n_global);
        for elem_idx in 0..mesh.num_elements() {
            for &d in &space.local_to_global(&mesh, elem_idx) {
                seen.insert(d);
            }
        }
        assert_eq!(
            seen.len(),
            n_global,
            "Q2 hex global DOFs covered: {} / {}",
            seen.len(),
            n_global
        );
        // All DOFs must be in 0..n_global.
        for &d in &seen {
            assert!(d < n_global);
        }
    }

    /// Same coverage check for Q2 quad meshes (vertex + edge + cell-interior).
    #[test]
    fn test_h1_space_q2_quad_dofs_cover_global_range() {
        let mesh = unit_square_quads(3);
        let space = H1Space::new(PolynomialDegree::P2);
        let n_global = space.num_global_dofs(&mesh);

        let mut seen = std::collections::HashSet::with_capacity(n_global);
        for elem_idx in 0..mesh.num_elements() {
            for &d in &space.local_to_global(&mesh, elem_idx) {
                seen.insert(d);
            }
        }
        assert_eq!(
            seen.len(),
            n_global,
            "Q2 quad global DOFs covered: {} / {}",
            seen.len(),
            n_global
        );
    }

    /// Boundary DOFs for Q2 hex must include face-centre DOFs on boundary faces,
    /// not just vertex and edge-midpoint DOFs.
    #[test]
    fn test_h1_space_q2_hex_boundary_dofs_include_face_centres() {
        let mesh = unit_cube_hexahedra(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let topo = crate::mesh::topology::MeshTopology::build(&mesh);
        let n_nodes = mesh.num_nodes();
        let n_edges = topo.num_edges();
        let face_offset = n_nodes + n_edges;

        // Find any boundary marker that has boundary faces.
        let Some((&marker, boundary_faces)) =
            topo.boundary_faces.iter().find(|(_, f)| !f.is_empty())
        else {
            panic!("hex test mesh has no boundary faces");
        };
        let expected_face_dofs: std::collections::HashSet<usize> =
            boundary_faces.iter().map(|&f| face_offset + f).collect();
        assert!(!expected_face_dofs.is_empty());

        let bdofs: std::collections::HashSet<usize> =
            space.boundary_dofs(&mesh, marker).into_iter().collect();

        for d in &expected_face_dofs {
            assert!(
                bdofs.contains(d),
                "boundary face-centre DOF {d} (marker {marker}) missing from boundary_dofs"
            );
        }
    }
}
