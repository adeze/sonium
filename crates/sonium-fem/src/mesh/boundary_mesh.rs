//! Boundary submesh construction and DOF mapping
//!
//! Creates a lower-dimensional mesh from boundary facets of a parent mesh,
//! enabling FunctionSpace construction and form assembly on the boundary.
//!
//! # Usage
//!
//! ```ignore
//! let (boundary, map) = create_boundary_submesh(&mesh, boundary_marker);
//! let boundary_space = H1Space::new(PolynomialDegree::P1);
//! let boundary_mass = assemble_bilinear(&MassForm, &boundary_space, &boundary_space, &boundary.mesh, 2);
//! // Map boundary DOFs back to parent mesh DOFs:
//! let parent_dof = map.parent_node(boundary_dof);
//! ```

use crate::mesh::{Element, ElementType, Mesh, Point};
use std::collections::HashMap;

/// A boundary submesh with mapping back to the parent mesh.
pub struct BoundarySubmesh {
    /// The lower-dimensional mesh (edges for 2D parent, triangles for 3D).
    pub mesh: Mesh,
    /// Maps submesh node indices to parent mesh node indices.
    pub node_to_parent: Vec<usize>,
    /// Maps parent node indices to submesh node indices.
    pub parent_to_node: HashMap<usize, usize>,
    /// Maps submesh element indices to parent boundary face indices.
    pub element_to_boundary: Vec<usize>,
}

impl BoundarySubmesh {
    /// Map a submesh node index to the parent mesh node index.
    pub fn parent_node(&self, sub_node: usize) -> usize {
        self.node_to_parent[sub_node]
    }

    /// Map a parent mesh node index to the submesh node index (if it exists).
    pub fn submesh_node(&self, parent_node: usize) -> Option<usize> {
        self.parent_to_node.get(&parent_node).copied()
    }

    /// Number of nodes in the boundary submesh.
    pub fn num_nodes(&self) -> usize {
        self.node_to_parent.len()
    }

    /// Number of elements in the boundary submesh.
    pub fn num_elements(&self) -> usize {
        self.mesh.num_elements()
    }

    /// Scatter a boundary-assembled vector to the parent mesh DOF ordering.
    ///
    /// `boundary_values` has one entry per boundary submesh node.
    /// Returns a vector of length `parent_n_dofs` with boundary contributions
    /// placed at the corresponding parent DOF indices.
    pub fn scatter_to_parent(&self, boundary_values: &[f64], parent_n_dofs: usize) -> Vec<f64> {
        let mut parent = vec![0.0; parent_n_dofs];
        for (sub_idx, &val) in boundary_values.iter().enumerate() {
            let parent_idx = self.node_to_parent[sub_idx];
            if parent_idx < parent_n_dofs {
                parent[parent_idx] += val;
            }
        }
        parent
    }

    /// Scatter a boundary-assembled triplet matrix to parent DOF ordering.
    ///
    /// Remaps `(row, col)` indices from boundary submesh to parent mesh.
    pub fn scatter_matrix_to_parent(
        &self,
        boundary_matrix: &crate::assembly::TripletMatrix<f64>,
        parent_n_dofs: usize,
    ) -> crate::assembly::TripletMatrix<f64> {
        let mut parent = crate::assembly::TripletMatrix::new(parent_n_dofs, parent_n_dofs);
        for k in 0..boundary_matrix.nnz() {
            let parent_row = self.node_to_parent[boundary_matrix.rows[k]];
            let parent_col = self.node_to_parent[boundary_matrix.cols[k]];
            parent.push(parent_row, parent_col, boundary_matrix.values[k]);
        }
        parent
    }
}

/// Create a boundary submesh from facets with a specific boundary marker.
///
/// Extracts boundary faces/edges with the given marker and builds a
/// lower-dimensional mesh. For a 2D parent mesh, boundary elements are
/// line segments; for 3D, they are triangles or quads.
///
/// Returns the submesh and the node mapping. Use [`BoundarySubmesh::scatter_to_parent`]
/// to map assembled quantities back to the parent mesh DOF ordering.
pub fn create_boundary_submesh(mesh: &Mesh, marker: i32) -> BoundarySubmesh {
    let mut node_to_parent = Vec::new();
    let mut parent_to_node = HashMap::new();
    let mut elements = Vec::new();
    let mut element_to_boundary = Vec::new();

    for (bface_idx, bface) in mesh.boundaries.iter().enumerate() {
        if bface.marker != marker {
            continue;
        }

        // Map parent nodes to submesh nodes (deduplicate)
        let mut local_nodes = Vec::with_capacity(bface.nodes.len());
        for &parent_node in &bface.nodes {
            let sub_node = *parent_to_node.entry(parent_node).or_insert_with(|| {
                let idx = node_to_parent.len();
                node_to_parent.push(parent_node);
                idx
            });
            local_nodes.push(sub_node);
        }

        // Determine element type for the boundary face
        let elem_type = match bface.nodes.len() {
            2 => ElementType::Triangle, // Line segment stored as degenerate triangle
            3 => ElementType::Triangle,
            4 => ElementType::Quadrilateral,
            _ => continue,
        };

        // For 2-node edges, we need a proper 1D element. Since our ElementType
        // doesn't have a Line type, we store them as triangles with the third
        // node duplicated. Alternatively, we use Triangle as the container.
        // For proper FEM on the boundary, we only need the node list.
        let elem_id = elements.len();
        elements.push(Element::new(elem_type, local_nodes, elem_id).with_marker(bface.marker));
        element_to_boundary.push(bface_idx);
    }

    // Build the submesh Mesh struct
    let nodes: Vec<Point> = node_to_parent
        .iter()
        .map(|&parent_idx| mesh.nodes[parent_idx])
        .collect();

    let dimension = 2; // boundary of 2D or 3D is at least 2D for our Mesh
    let mut sub_mesh = Mesh::new(dimension);
    sub_mesh.physical_groups = mesh.physical_groups.clone();
    sub_mesh.nodes = nodes;
    sub_mesh.elements = elements;

    BoundarySubmesh {
        mesh: sub_mesh,
        node_to_parent,
        parent_to_node,
        element_to_boundary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_create_boundary_submesh() {
        let mut mesh = unit_square_triangles(4);
        mesh.detect_boundaries();

        // Marker 0 is the full perimeter on a unit square
        let bsub = create_boundary_submesh(&mesh, 0);

        // Should have boundary nodes (all 4 edges of unit square)
        assert!(bsub.num_nodes() > 0, "Should have boundary nodes");
        assert!(bsub.num_elements() > 0, "Should have boundary elements");

        // All submesh nodes should map to valid parent nodes
        for &parent in &bsub.node_to_parent {
            assert!(parent < mesh.num_nodes());
        }
    }

    #[test]
    fn boundary_submesh_preserves_marker_identity_and_names() {
        let mut mesh = unit_square_triangles(1);
        mesh.boundaries[0].marker = 7;
        mesh.register_physical_group(1, 7, "loaded edge").unwrap();

        let boundary = create_boundary_submesh(&mesh, 7);

        assert!(!boundary.mesh.elements.is_empty());
        assert!(
            boundary
                .mesh
                .elements
                .iter()
                .all(|element| element.marker == 7)
        );
        assert_eq!(
            boundary.mesh.physical_groups.by_tag(1, 7).unwrap().name,
            "loaded edge"
        );
    }

    #[test]
    fn test_scatter_to_parent() {
        let mut mesh = unit_square_triangles(4);
        mesh.detect_boundaries();

        let bsub = create_boundary_submesh(&mesh, 0);
        let n_parent = mesh.num_nodes();

        // Set all boundary values to 1.0
        let bvals = vec![1.0; bsub.num_nodes()];
        let parent_vals = bsub.scatter_to_parent(&bvals, n_parent);

        // Boundary nodes should have value 1.0, interior nodes should have 0.0
        for &parent_idx in &bsub.node_to_parent {
            assert_eq!(parent_vals[parent_idx], 1.0);
        }
    }

    #[test]
    fn test_node_mapping_roundtrip() {
        let mut mesh = unit_square_triangles(4);
        mesh.detect_boundaries();

        let bsub = create_boundary_submesh(&mesh, 0);

        for (sub_idx, &parent_idx) in bsub.node_to_parent.iter().enumerate() {
            let rt = bsub.submesh_node(parent_idx).unwrap();
            assert_eq!(rt, sub_idx);
        }
    }
}
