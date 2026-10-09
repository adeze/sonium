//! Global continuous quadratic topology for Tri3 boundary panels.

use std::collections::HashMap;
use std::fmt;

use crate::core::types::{ElementType, Mesh};

/// Errors reported while constructing the continuous quadratic topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuousQuadraticDofMapError {
    /// Element type not covered by the continuous quadratic topology.
    UnsupportedElementType {
        /// Element index in the mesh.
        element: usize,
        /// Element type encountered.
        element_type: ElementType,
    },
    /// A panel did not contain the connectivity entries required by its type.
    InvalidConnectivity {
        /// Element index in the mesh.
        element: usize,
        /// Number of entries encountered.
        actual: usize,
        /// Number of entries required by the panel type.
        expected: usize,
    },
    /// A connectivity entry referred outside the node table.
    InvalidNodeIndex {
        /// Element index in the mesh.
        element: usize,
        /// Local connectivity slot.
        local: usize,
        /// Invalid global node index.
        node: usize,
    },
    /// A panel repeated a vertex, which makes edge ownership ambiguous.
    RepeatedNode {
        /// Element index in the mesh.
        element: usize,
        /// Repeated global node index.
        node: usize,
    },
}

impl fmt::Display for ContinuousQuadraticDofMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedElementType {
                element,
                element_type,
            } => write!(
                f,
                "continuous quadratic topology does not support element {element} of type {element_type:?}"
            ),
            Self::InvalidConnectivity {
                element,
                actual,
                expected,
            } => write!(
                f,
                "continuous quadratic element {element} has {actual} connectivity entries, expected {expected}"
            ),
            Self::InvalidNodeIndex {
                element,
                local,
                node,
            } => write!(
                f,
                "continuous quadratic element {element} connectivity[{local}]={node} is outside the mesh node table"
            ),
            Self::RepeatedNode { element, node } => write!(
                f,
                "continuous quadratic element {element} repeats node {node}"
            ),
        }
    }
}

impl std::error::Error for ContinuousQuadraticDofMapError {}

/// Global quadratic scalar DOF map for a conforming Tri3/Quad4 surface mesh.
///
/// The local ordering is the same as [`crate::core::mesh::basis_values`]:
/// three vertices followed by edge midpoints `(0,1)`, `(1,2)`, `(2,0)`.
/// Edge keys are orientation independent, so adjacent triangles share the
/// same midpoint DOF even when their local winding is opposite. Evaluation
/// elements are retained in `element_dofs` as empty entries.
///
/// For Quad4 panels the local ordering is the tensor-product Q2 ordering
/// used by [`crate::core::mesh::basis_values`]: four vertices, four edge
/// midpoints `(0,1)`, `(1,2)`, `(2,3)`, `(3,0)`, and one panel-interior
/// center. The center is intentionally not shared with another panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousQuadraticDofMap {
    element_dofs: Vec<Vec<usize>>,
    vertex_dofs: Vec<Option<usize>>,
    edge_dofs: HashMap<(usize, usize), usize>,
    interior_dofs: Vec<Option<usize>>,
    dof_owners: Vec<(usize, usize)>,
}

impl ContinuousQuadraticDofMap {
    /// Construct the map from the mesh's shared Tri3 vertex connectivity.
    pub fn from_mesh(mesh: &Mesh) -> Result<Self, ContinuousQuadraticDofMapError> {
        let mut map = Self {
            element_dofs: vec![Vec::new(); mesh.elements.len()],
            vertex_dofs: vec![None; mesh.nodes.len()],
            edge_dofs: HashMap::new(),
            interior_dofs: vec![None; mesh.elements.len()],
            dof_owners: Vec::new(),
        };

        for (element_index, element) in mesh.elements.iter().enumerate() {
            if element.property.is_evaluation() {
                continue;
            }
            let expected_vertices = match element.element_type {
                ElementType::Tri3 => 3,
                ElementType::Quad4 => 4,
            };
            if element.connectivity.len() != expected_vertices {
                return Err(ContinuousQuadraticDofMapError::InvalidConnectivity {
                    element: element_index,
                    actual: element.connectivity.len(),
                    expected: expected_vertices,
                });
            }

            for (local, &node) in element.connectivity.iter().enumerate() {
                if node >= mesh.nodes.len() {
                    return Err(ContinuousQuadraticDofMapError::InvalidNodeIndex {
                        element: element_index,
                        local,
                        node,
                    });
                }
                if element.connectivity[..local].contains(&node) {
                    return Err(ContinuousQuadraticDofMapError::RepeatedNode {
                        element: element_index,
                        node,
                    });
                }
            }

            let vertices = element.connectivity.clone();
            let edges: Vec<(usize, usize)> = match element.element_type {
                ElementType::Tri3 => vec![
                    (vertices[0], vertices[1]),
                    (vertices[1], vertices[2]),
                    (vertices[2], vertices[0]),
                ],
                ElementType::Quad4 => vec![
                    (vertices[0], vertices[1]),
                    (vertices[1], vertices[2]),
                    (vertices[2], vertices[3]),
                    (vertices[3], vertices[0]),
                ],
            };
            let mut local_dofs = Vec::with_capacity(match element.element_type {
                ElementType::Tri3 => 6,
                ElementType::Quad4 => 9,
            });

            for (local, &vertex) in vertices.iter().enumerate() {
                let dof = if let Some(&dof) = map.vertex_dofs[vertex].as_ref() {
                    dof
                } else {
                    let dof = map.dof_owners.len();
                    map.vertex_dofs[vertex] = Some(dof);
                    map.dof_owners.push((element_index, local));
                    dof
                };
                local_dofs.push(dof);
            }

            for (local_edge, &(a, b)) in edges.iter().enumerate() {
                let key = if a < b { (a, b) } else { (b, a) };
                let dof = if let Some(&dof) = map.edge_dofs.get(&key) {
                    dof
                } else {
                    let dof = map.dof_owners.len();
                    map.edge_dofs.insert(key, dof);
                    // Quadratic local basis indices 3.. are edge midpoints
                    // for Tri3 and 4..7 for Quad4.
                    let local_basis = match element.element_type {
                        ElementType::Tri3 => local_edge + 3,
                        ElementType::Quad4 => local_edge + 4,
                    };
                    map.dof_owners.push((element_index, local_basis));
                    dof
                };
                local_dofs.push(dof);
            }
            if element.element_type == ElementType::Quad4 {
                // The Q2 tensor-product center is interior to this panel and
                // therefore has no neighboring panel to share it with.
                let dof = map.dof_owners.len();
                map.interior_dofs[element_index] = Some(dof);
                map.dof_owners.push((element_index, 8));
                local_dofs.push(dof);
            }
            map.element_dofs[element_index] = local_dofs;
        }

        Ok(map)
    }

    /// Number of globally unique quadratic DOFs.
    pub fn num_dofs(&self) -> usize {
        self.dof_owners.len()
    }

    /// Global DOFs for one mesh element in quadratic local basis order.
    pub fn element_dofs(&self, element: usize) -> Option<&[usize]> {
        self.element_dofs.get(element).map(Vec::as_slice)
    }

    /// Return the first panel/local basis slot that owns a global DOF.
    ///
    /// The owner is used by collocation assembly to select one deterministic
    /// interior point per unique row; it is not an ownership restriction on
    /// the field interpolation.
    pub fn dof_owner(&self, dof: usize) -> Option<(usize, usize)> {
        self.dof_owners.get(dof).copied()
    }

    /// Return the shared DOF for a mesh vertex, if it participates in the map.
    pub fn vertex_dof(&self, node: usize) -> Option<usize> {
        self.vertex_dofs.get(node).copied().flatten()
    }

    /// Return the shared DOF for an edge, independent of its orientation.
    pub fn edge_dof(&self, a: usize, b: usize) -> Option<usize> {
        let key = if a < b { (a, b) } else { (b, a) };
        self.edge_dofs.get(&key).copied()
    }

    /// Return the panel-interior Q2 DOF for a Quad4 element, if present.
    pub fn interior_dof(&self, element: usize) -> Option<usize> {
        self.interior_dofs.get(element).copied().flatten()
    }
}

impl Mesh {
    /// Build globally continuous quadratic DOFs for active Tri3/Quad4 panels.
    pub fn continuous_quadratic_dof_map(
        &self,
    ) -> Result<ContinuousQuadraticDofMap, ContinuousQuadraticDofMapError> {
        ContinuousQuadraticDofMap::from_mesh(self)
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;
    use crate::core::types::{BoundaryCondition, Element, ElementProperty};

    fn triangle(element: Vec<usize>) -> Element {
        Element {
            connectivity: element,
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0, 0.0, 1.0]; 3],
            center: [0.0, 0.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
            group: 0,
            dof_addresses: Vec::new(),
        }
    }

    fn two_triangle_mesh() -> Mesh {
        Mesh {
            nodes: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            elements: vec![triangle(vec![0, 1, 2]), triangle(vec![1, 3, 2])],
            external_node_numbers: Vec::new(),
            external_element_numbers: Vec::new(),
            num_boundary_nodes: 4,
            num_evaluation_nodes: 0,
            num_boundary_elements: 2,
            num_evaluation_elements: 0,
            symmetry_planes: [0; 3],
            symmetry_coordinates: [0.0; 3],
            num_reflections: 0,
            node_to_elements: Vec::new(),
        }
    }

    #[test]
    fn shared_vertices_and_edges_use_one_global_dof() {
        let map = two_triangle_mesh()
            .continuous_quadratic_dof_map()
            .expect("valid Tri3 topology");

        // Four vertices plus five unique edges.
        assert_eq!(map.num_dofs(), 9);
        assert_eq!(map.edge_dof(1, 2), map.edge_dof(2, 1));

        let first = map.element_dofs(0).expect("first element map");
        let second = map.element_dofs(1).expect("second element map");
        assert_eq!(first.len(), 6);
        assert_eq!(second.len(), 6);
        // First triangle's local edge (1,2) and second triangle's local edge
        // (2,1) are the same orientation-independent midpoint DOF.
        assert_eq!(first[4], second[5]);
        assert_eq!(first[1], second[0]);
    }

    #[test]
    fn continuous_quadratic_map_rejects_invalid_quad_connectivity() {
        let mut mesh = two_triangle_mesh();
        mesh.elements[0].element_type = ElementType::Quad4;
        mesh.elements[0].connectivity.pop();
        let error = mesh.continuous_quadratic_dof_map().unwrap_err();
        assert!(matches!(
            error,
            ContinuousQuadraticDofMapError::InvalidConnectivity {
                element: 0,
                actual: 2,
                expected: 4,
            }
        ));
    }
}
