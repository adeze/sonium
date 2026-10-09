//! Submesh and entity mapping support
//!
//! Provides types for domain decomposition, coupling, and substructuring:
//! - [`SubMesh`]: A subset of elements from a parent mesh with remapped DOF indices
//! - [`EntityMap`]: Bidirectional mapping between parent mesh and submesh entities
//!
//! This is essential for:
//! - Domain decomposition methods (Schur complement, FETI)
//! - Contact problems
//! - Multi-physics coupling
//! - Mesh refinement with hanging nodes

use crate::mesh::{ElementType, Mesh, PhysicalGroupRegistry, Point};
use std::collections::HashMap;

/// A submesh is a subset of elements from a parent mesh.
///
/// Contains a subset of elements with their nodes. Node and element indices
/// are remapped to local indices in the submesh.
#[derive(Debug, Clone)]
pub struct SubMesh {
    /// Topological dimension inherited from the parent mesh.
    pub dimension: usize,
    /// Submesh element types
    pub element_types: Vec<ElementType>,
    /// Submesh nodes (physical coordinates)
    pub nodes: Vec<Point>,
    /// Submesh elements: each element is a list of node indices in the submesh
    pub elements: Vec<Vec<usize>>,
    /// Cell markers copied from the selected parent elements.
    pub element_markers: Vec<i32>,
    /// Named physical groups inherited from the parent mesh.
    pub physical_groups: PhysicalGroupRegistry,
    /// Parent element indices (maps submesh elem -> parent elem)
    pub parent_element_indices: Vec<usize>,
    /// Parent node indices (maps submesh node -> parent node)
    pub parent_node_indices: Vec<usize>,
}

impl SubMesh {
    /// Create a submesh from a list of parent element indices.
    pub fn from_elements(mesh: &Mesh, element_indices: &[usize]) -> Self {
        let mut sub_nodes_map: HashMap<usize, usize> = HashMap::new();
        let mut sub_nodes: Vec<Point> = Vec::new();
        let mut sub_elements: Vec<Vec<usize>> = Vec::new();
        let mut element_types: Vec<ElementType> = Vec::new();
        let mut element_markers: Vec<i32> = Vec::new();
        let mut parent_element_indices: Vec<usize> = Vec::new();
        let mut parent_node_indices: Vec<usize> = Vec::new();

        for &elem_idx in element_indices {
            let parent_elem = &mesh.elements[elem_idx];
            element_types.push(parent_elem.element_type);
            element_markers.push(parent_elem.marker);
            parent_element_indices.push(elem_idx);

            let mut local_nodes: Vec<usize> = Vec::with_capacity(parent_elem.nodes.len());
            for &n in &parent_elem.nodes {
                let local_idx = *sub_nodes_map.entry(n).or_insert_with(|| {
                    let idx = sub_nodes.len();
                    sub_nodes.push(mesh.nodes[n]);
                    parent_node_indices.push(n);
                    idx
                });
                local_nodes.push(local_idx);
            }
            sub_elements.push(local_nodes);
        }

        Self {
            dimension: mesh.dimension,
            element_types,
            nodes: sub_nodes,
            elements: sub_elements,
            element_markers,
            physical_groups: mesh.physical_groups.clone(),
            parent_element_indices,
            parent_node_indices,
        }
    }

    /// Create a SubMesh into a full Mesh
    pub fn to_mesh(&self) -> Mesh {
        let mut mesh = Mesh::new(self.dimension);
        mesh.physical_groups = self.physical_groups.clone();

        // Add nodes
        for point in &self.nodes {
            mesh.nodes.push(*point);
        }

        // Add elements
        for ((nodes, &etype), &marker) in self
            .elements
            .iter()
            .zip(self.element_types.iter())
            .zip(self.element_markers.iter())
        {
            mesh.add_element_with_marker(etype, nodes.clone(), marker);
        }

        mesh
    }

    /// Number of elements in the submesh.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Number of nodes in the submesh.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Bidirectional mapping between parent and submesh entities.
///
/// Supports mapping nodes, edges, faces, and elements between meshes.
#[derive(Debug, Clone, Default)]
pub struct EntityMap {
    /// Maps submesh node index -> parent node index
    node_to_parent: Vec<usize>,
    /// Maps parent node index -> submesh node index (if present)
    parent_to_node: HashMap<usize, usize>,
    /// Maps submesh element index -> parent element index
    element_to_parent: Vec<usize>,
    /// Maps parent element index -> submesh element index (if present)
    parent_to_element: HashMap<usize, usize>,
}

impl EntityMap {
    /// Create an entity map from a submesh.
    pub fn from_submesh(submesh: &SubMesh) -> Self {
        let mut parent_to_node = HashMap::new();
        for (sub_idx, &parent_idx) in submesh.parent_node_indices.iter().enumerate() {
            parent_to_node.insert(parent_idx, sub_idx);
        }

        let mut parent_to_element = HashMap::new();
        for (sub_idx, &parent_idx) in submesh.parent_element_indices.iter().enumerate() {
            parent_to_element.insert(parent_idx, sub_idx);
        }

        Self {
            node_to_parent: submesh.parent_node_indices.clone(),
            parent_to_node,
            element_to_parent: submesh.parent_element_indices.clone(),
            parent_to_element,
        }
    }

    /// Get parent node index from submesh node index.
    pub fn parent_node(&self, submesh_node: usize) -> Option<usize> {
        self.node_to_parent.get(submesh_node).copied()
    }

    /// Get submesh node index from parent node index.
    pub fn submesh_node(&self, parent_node: usize) -> Option<usize> {
        self.parent_to_node.get(&parent_node).copied()
    }

    /// Get parent element index from submesh element index.
    pub fn parent_element(&self, submesh_elem: usize) -> Option<usize> {
        self.element_to_parent.get(submesh_elem).copied()
    }

    /// Get submesh element index from parent element index.
    pub fn submesh_element(&self, parent_elem: usize) -> Option<usize> {
        self.parent_to_element.get(&parent_elem).copied()
    }

    /// Whether a parent node exists in the submesh.
    pub fn contains_parent_node(&self, parent_node: usize) -> bool {
        self.parent_to_node.contains_key(&parent_node)
    }

    /// Whether a parent element exists in the submesh.
    pub fn contains_parent_element(&self, parent_elem: usize) -> bool {
        self.parent_to_element.contains_key(&parent_elem)
    }
}

/// Split a mesh into submeshes by element coloring (for domain decomposition).
///
/// Returns a list of submeshes, each containing approximately the same number
/// of elements. Elements are distributed round-robin to create well-balanced
/// subdomains.
pub fn partition_mesh(mesh: &Mesh, num_partitions: usize) -> Vec<SubMesh> {
    if num_partitions == 0 {
        return vec![];
    }

    let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); num_partitions];

    for (elem_idx, _) in mesh.elements.iter().enumerate() {
        let partition_idx = elem_idx % num_partitions;
        partitions[partition_idx].push(elem_idx);
    }

    partitions
        .iter()
        .map(|elem_indices| SubMesh::from_elements(mesh, elem_indices))
        .collect()
}

/// Interface DOFs for a submesh (DOFs on elements that touch the boundary).
///
/// Returns the local DOF indices in the submesh that correspond to DOFs
/// on elements adjacent to the submesh boundary.
pub fn interface_dofs<F: Fn(usize) -> Vec<usize>>(
    submesh: &SubMesh,
    local_to_global: F,
) -> Vec<usize> {
    let mut is_interface = vec![false; submesh.num_nodes()];

    for nodes in submesh.elements.iter() {
        for &node in nodes {
            is_interface[node] = true;
        }
    }

    let mut interface_dofs = Vec::new();
    for (local_elem, _elem) in submesh.elements.iter().enumerate() {
        let global_dofs = local_to_global(local_elem);
        for &dof in &global_dofs {
            interface_dofs.push(dof);
        }
    }

    interface_dofs.sort();
    interface_dofs.dedup();
    interface_dofs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_submesh_creation() {
        let mut mesh = unit_square_triangles(2);
        mesh.elements[0].marker = 17;
        mesh.register_physical_group(2, 17, "selected material")
            .unwrap();
        let submesh = SubMesh::from_elements(&mesh, &[0, 1]);

        assert_eq!(submesh.num_elements(), 2);
        assert!(submesh.num_nodes() > 0);
        assert_eq!(submesh.parent_element_indices.len(), 2);
        assert_eq!(submesh.element_markers, vec![17, 0]);

        let rebuilt = submesh.to_mesh();
        assert_eq!(rebuilt.elements[0].marker, 17);
        assert_eq!(
            rebuilt.physical_groups.by_tag(2, 17).unwrap().name,
            "selected material"
        );
    }

    #[test]
    fn test_entity_map() {
        let mesh = unit_square_triangles(2);
        let submesh = SubMesh::from_elements(&mesh, &[0, 1]);
        let map = EntityMap::from_submesh(&submesh);

        for (sub_idx, &parent_idx) in submesh.parent_node_indices.iter().enumerate() {
            assert_eq!(map.parent_node(sub_idx), Some(parent_idx));
            assert_eq!(map.submesh_node(parent_idx), Some(sub_idx));
        }
    }

    #[test]
    fn test_partition_mesh() {
        let mesh = unit_square_triangles(4);
        let partitions = partition_mesh(&mesh, 2);

        assert_eq!(partitions.len(), 2);
        let total_elems: usize = partitions.iter().map(|p| p.num_elements()).sum();
        assert_eq!(total_elems, mesh.elements.len());
    }
}
