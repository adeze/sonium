//! Discontinuous Galerkin (DG) function space
//!
//! Element-local DOFs with no inter-element continuity. Each element has its
//! own set of DOFs, enabling jump and average operators on interior facets.
//!
//! DG spaces are used for:
//! - Interior penalty methods (biharmonic, advection)
//! - Mixed formulations (pressure space in Stokes, scalar in mixed Poisson)
//! - Shock capturing and limiters in hyperbolic PDEs

use crate::basis::{PolynomialDegree, evaluate_shape};
use crate::mesh::{ElementType, Mesh};
use crate::spaces::{BasisTabulation, FieldType, FunctionSpace};

/// Discontinuous Galerkin function space (P0 through P3-DG).
///
/// Each element has its own DOFs, not shared with neighbors.
/// For P0: 1 DOF per element (piecewise constant).
/// For P1/P2/P3: the element's nodal Lagrange DOFs per element, discontinuous.
#[derive(Debug, Clone)]
pub struct DGSpace {
    degree: PolynomialDegree,
    piecewise_constant: bool,
}

impl DGSpace {
    pub fn new(degree: PolynomialDegree) -> Self {
        Self {
            degree,
            piecewise_constant: false,
        }
    }

    /// P0: piecewise constant (1 DOF per element).
    pub fn p0() -> Self {
        Self {
            degree: PolynomialDegree::P1,
            piecewise_constant: true,
        }
    }

    fn assert_supported(&self, _elem_type: ElementType) {
        if self.piecewise_constant {
            return;
        }
        match self.degree {
            PolynomialDegree::P1 | PolynomialDegree::P2 | PolynomialDegree::P3 => {}
        }
    }

    fn local_dof_count(&self, elem_type: ElementType) -> usize {
        self.assert_supported(elem_type);
        if self.piecewise_constant {
            1
        } else {
            match self.degree {
                PolynomialDegree::P1 => elem_type.num_nodes_p1(),
                PolynomialDegree::P2 => elem_type.num_nodes_p2(),
                PolynomialDegree::P3 => elem_type.num_nodes_p3(),
            }
        }
    }

    fn element_dof_base(&self, mesh: &Mesh, elem_idx: usize) -> usize {
        mesh.elements[..elem_idx]
            .iter()
            .map(|elem| self.local_dof_count(elem.element_type))
            .sum()
    }

    /// Return the local DG basis indices whose trace has support on a boundary
    /// face.  DG DOFs are element-local, so the owning element is part of the
    /// global numbering and no neighbouring-element deduplication is allowed.
    fn boundary_local_dofs(&self, mesh: &Mesh, face: &crate::mesh::BoundaryFace) -> Vec<usize> {
        let elem = &mesh.elements[face.element_idx];
        let local_count = self.local_dof_count(elem.element_type);
        if self.piecewise_constant {
            return vec![0];
        }

        let mut local_dofs = std::collections::HashSet::new();
        let face_nodes: std::collections::HashSet<usize> = face.nodes.iter().copied().collect();
        let n_vertices = elem.element_type.num_vertices();

        // Vertex basis functions are ordered first for every supported shape.
        for (local, &node) in elem.vertices().iter().enumerate() {
            if face_nodes.contains(&node) {
                local_dofs.insert(local);
            }
        }

        if matches!(self.degree, PolynomialDegree::P2 | PolynomialDegree::P3) {
            // Boundary edge midpoint basis functions follow the vertex basis.
            for (edge_idx, &(a, b)) in elem.element_type.local_edges().iter().enumerate() {
                let va = elem.vertices()[a];
                let vb = elem.vertices()[b];
                if face_nodes.contains(&va) && face_nodes.contains(&vb) {
                    let edge_base = match self.degree {
                        PolynomialDegree::P2 => n_vertices + edge_idx,
                        PolynomialDegree::P3 => n_vertices + 2 * edge_idx,
                        PolynomialDegree::P1 => unreachable!(),
                    };
                    local_dofs.insert(edge_base);
                    if self.degree == PolynomialDegree::P3 {
                        local_dofs.insert(edge_base + 1);
                    }
                }
            }

            // Hex27 has one additional basis function at each face centre.
            if elem.element_type == ElementType::Hexahedron
                && face.local_idx < elem.element_type.local_faces().len()
            {
                let face_base = match self.degree {
                    PolynomialDegree::P2 => n_vertices + elem.element_type.num_edges(),
                    PolynomialDegree::P3 => {
                        n_vertices + 2 * elem.element_type.num_edges() + 4 * face.local_idx
                    }
                    PolynomialDegree::P1 => unreachable!(),
                };
                let face_count = if self.degree == PolynomialDegree::P3 {
                    4
                } else {
                    1
                };
                for offset in 0..face_count {
                    local_dofs.insert(face_base + offset);
                }
            }
            // Cubic tetrahedra have one interior trace node per triangular
            // face, after the vertex and edge nodes.
            if self.degree == PolynomialDegree::P3
                && elem.element_type == ElementType::Tetrahedron
                && face.local_idx < elem.element_type.local_faces().len()
            {
                local_dofs.insert(n_vertices + 2 * elem.element_type.num_edges() + face.local_idx);
            }
        }

        let mut local_dofs: Vec<_> = local_dofs
            .into_iter()
            .filter(|&local| local < local_count)
            .collect();
        local_dofs.sort_unstable();
        local_dofs
    }
}

impl FunctionSpace for DGSpace {
    fn num_components(&self) -> usize {
        1 // Scalar DG space
    }

    fn num_local_dofs(&self, elem_type: ElementType) -> usize {
        self.local_dof_count(elem_type)
    }

    fn num_global_dofs(&self, mesh: &Mesh) -> usize {
        mesh.elements
            .iter()
            .map(|elem| self.local_dof_count(elem.element_type))
            .sum()
    }

    fn local_to_global(&self, mesh: &Mesh, elem_idx: usize) -> Vec<usize> {
        let elem_type = mesh.elements[elem_idx].element_type;
        let local_count = self.local_dof_count(elem_type);
        let base = self.element_dof_base(mesh, elem_idx);
        (base..base + local_count).collect()
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
        let vertices = elem.vertices();
        let dim = elem_type.dimension();

        // Geometry is always mapped with the affine P1 vertex basis.  The
        // solution basis may be P2 (or P0), and must not be used to assemble a
        // Jacobian from a vertex-only coordinate list.
        let geom_shape = evaluate_shape(elem_type, PolynomialDegree::P1, xi, eta, zeta);
        let (values, gradients, grad_dim) = if self.piecewise_constant {
            (vec![1.0], vec![0.0; dim], dim)
        } else {
            let shape = evaluate_shape(elem_type, self.degree, xi, eta, zeta);
            (shape.values, shape.gradients, shape.grad_dim)
        };
        let n_dofs = values.len();

        // Compute Jacobian for coordinate mapping
        let jac = if dim == 2 {
            let coords: Vec<[f64; 2]> = vertices
                .iter()
                .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y])
                .collect();
            crate::basis::Jacobian::from_2d(&geom_shape.gradients, geom_shape.grad_dim, &coords)
        } else {
            let coords: Vec<[f64; 3]> = vertices
                .iter()
                .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y, mesh.nodes[v].z])
                .collect();
            crate::basis::Jacobian::from_3d(&geom_shape.gradients, geom_shape.grad_dim, &coords)
        };

        // Physical gradients via inverse Jacobian: grad_phys = J^{-T} grad_ref
        let mut phys_grads = vec![0.0; n_dofs * dim];
        for (gradient, physical_gradient) in
            gradients.chunks(grad_dim).zip(phys_grads.chunks_mut(dim))
        {
            jac.transform_gradient_into(gradient, physical_gradient);
        }

        // Physical coordinates at quadrature point
        let mut phys = [0.0; 3];
        for (vi, &v) in vertices.iter().enumerate() {
            let node = &mesh.nodes[v];
            phys[0] += geom_shape.values[vi] * node.x;
            phys[1] += geom_shape.values[vi] * node.y;
            phys[2] += geom_shape.values[vi] * node.z;
        }

        BasisTabulation {
            n_dofs,
            spatial_dim: dim,
            values,
            derivatives: phys_grads,
            hessians: Vec::new(),
            det_j: jac.det.abs(),
            physical_coords: phys,
        }
    }

    fn boundary_dofs(&self, mesh: &Mesh, boundary_marker: i32) -> Vec<usize> {
        let mut dofs = Vec::new();
        for face in mesh
            .boundaries
            .iter()
            .filter(|face| face.marker == boundary_marker)
        {
            let base = self.element_dof_base(mesh, face.element_idx);
            dofs.extend(
                self.boundary_local_dofs(mesh, face)
                    .into_iter()
                    .map(|local| base + local),
            );
        }
        dofs.sort_unstable();
        dofs.dedup();
        dofs
    }

    fn field_type(&self) -> FieldType {
        FieldType::Scalar
    }
}

// ---------------------------------------------------------------------------
// Facet integration utilities
// ---------------------------------------------------------------------------

/// Interior facet information for DG assembly.
///
/// Each interior facet connects two elements and provides the local DOFs,
/// normals, and quadrature data needed for jump/average operators.
#[derive(Debug, Clone)]
pub struct InteriorFacet {
    /// Index of the "plus" element (arbitrary assignment).
    pub elem_plus: usize,
    /// Index of the "minus" element.
    pub elem_minus: usize,
    /// Local face index within elem_plus.
    pub local_face_plus: usize,
    /// Local face index within elem_minus.
    pub local_face_minus: usize,
    /// Shared node indices (the facet nodes).
    pub shared_nodes: Vec<usize>,
    /// Facet area (2D: edge length, 3D: triangle area).
    pub area: f64,
    /// Outward normal from elem_plus toward elem_minus.
    pub normal: [f64; 3],
}

/// Build interior facet list from mesh topology.
///
/// Returns all interior facets (shared between exactly two elements)
/// with their connectivity and geometric data.
pub fn build_interior_facets(mesh: &Mesh) -> Vec<InteriorFacet> {
    use std::collections::HashMap;

    // Map sorted face node tuples → (first_element, local_face_idx)
    let mut face_to_elem: HashMap<Vec<usize>, (usize, usize)> = HashMap::new();
    let mut facets = Vec::new();

    for (elem_idx, elem) in mesh.elements.iter().enumerate() {
        let elem_faces = if mesh.dimension == 2 {
            elem.edges().into_iter().map(|(a, b)| vec![a, b]).collect()
        } else {
            elem.faces()
        };
        for (local_face, face_nodes) in elem_faces.iter().enumerate() {
            let mut key = face_nodes.clone();
            key.sort();

            if let Some(&(other_elem, other_local)) = face_to_elem.get(&key) {
                // Interior facet: shared between this element and other_elem
                let area = facet_area(mesh, face_nodes);
                let normal = facet_normal(mesh, face_nodes);
                facets.push(InteriorFacet {
                    elem_plus: other_elem,
                    elem_minus: elem_idx,
                    local_face_plus: other_local,
                    local_face_minus: local_face,
                    shared_nodes: face_nodes.clone(),
                    area,
                    normal,
                });
            } else {
                face_to_elem.insert(key, (elem_idx, local_face));
            }
        }
    }
    facets
}

/// Compute facet area from node coordinates.
fn facet_area(mesh: &Mesh, nodes: &[usize]) -> f64 {
    match nodes.len() {
        2 => {
            // Edge
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;
            let dz = p1.z - p0.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        }
        3 => {
            // Triangle
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let p2 = &mesh.nodes[nodes[2]];
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let cx = v1[1] * v2[2] - v1[2] * v2[1];
            let cy = v1[2] * v2[0] - v1[0] * v2[2];
            let cz = v1[0] * v2[1] - v1[1] * v2[0];
            0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
        }
        _ => 0.0,
    }
}

/// Compute outward normal for a facet (2D: rotate edge tangent, 3D: cross product).
fn facet_normal(mesh: &Mesh, nodes: &[usize]) -> [f64; 3] {
    match nodes.len() {
        2 => {
            // 2D: rotate tangent 90° clockwise
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-15 {
                return [0.0, 0.0, 0.0];
            }
            [dy / len, -dx / len, 0.0]
        }
        3 => {
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let p2 = &mesh.nodes[nodes[2]];
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let cx = v1[1] * v2[2] - v1[2] * v2[1];
            let cy = v1[2] * v2[0] - v1[0] * v2[2];
            let cz = v1[0] * v2[1] - v1[1] * v2[0];
            let len = (cx * cx + cy * cy + cz * cz).sqrt();
            if len < 1e-15 {
                return [0.0, 0.0, 0.0];
            }
            [cx / len, cy / len, cz / len]
        }
        _ => [0.0, 0.0, 0.0],
    }
}

/// Compute the mesh size `h` for an element (diameter of circumscribed sphere approximation).
pub fn element_diameter(mesh: &Mesh, elem_idx: usize) -> f64 {
    let elem = &mesh.elements[elem_idx];
    let vertices = elem.vertices();
    let mut max_dist = 0.0_f64;
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let pi = &mesh.nodes[vertices[i]];
            let pj = &mesh.nodes[vertices[j]];
            let d = ((pi.x - pj.x).powi(2) + (pi.y - pj.y).powi(2) + (pi.z - pj.z).powi(2)).sqrt();
            max_dist = max_dist.max(d);
        }
    }
    max_dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_dg_space_dof_count() {
        let mesh = unit_square_triangles(2);
        let space = DGSpace::new(PolynomialDegree::P1);
        let n_dofs = space.num_global_dofs(&mesh);
        // Each triangle has 3 DOFs, n_elements triangles
        assert_eq!(n_dofs, mesh.num_elements() * 3);
    }

    #[test]
    fn test_dg_space_p0_is_one_dof_per_element() {
        let mesh = unit_square_triangles(2);
        let space = DGSpace::p0();

        assert_eq!(space.num_global_dofs(&mesh), mesh.num_elements());
        assert!(space.validate(&mesh).is_ok());
        for elem_idx in 0..mesh.num_elements() {
            assert_eq!(space.local_to_global(&mesh, elem_idx), vec![elem_idx]);
            let tab = space.tabulate(&mesh, elem_idx, 0.2, 0.3, 0.0);
            assert_eq!(tab.n_dofs, 1);
            assert_eq!(tab.values, vec![1.0]);
            assert!(
                tab.derivatives
                    .iter()
                    .all(|gradient| gradient.abs() < 1e-14)
            );
        }
    }

    #[test]
    fn test_dg_space_p2_has_matching_numbering_and_tabulation() {
        let mesh = unit_square_triangles(2);
        let space = DGSpace::new(PolynomialDegree::P2);

        assert_eq!(space.num_global_dofs(&mesh), mesh.num_elements() * 6);
        assert!(space.validate(&mesh).is_ok());
        let tab = space.tabulate(&mesh, 0, 0.2, 0.3, 0.0);
        assert_eq!(tab.n_dofs, 6);
        assert_eq!(tab.values.len(), 6);
        assert_eq!(tab.derivatives.len(), 12);
        assert!((tab.values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dg_space_p3_has_complete_local_contract() {
        let mesh = unit_square_triangles(2);
        let space = DGSpace::new(PolynomialDegree::P3);

        assert_eq!(space.num_global_dofs(&mesh), mesh.num_elements() * 10);
        assert!(space.validate(&mesh).is_ok());

        let tab = space.tabulate(&mesh, 0, 0.2, 0.3, 0.0);
        assert_eq!(tab.n_dofs, 10);
        assert_eq!(tab.values.len(), 10);
        assert_eq!(tab.derivatives.len(), 20);
        assert!((tab.values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            tab.values
                .iter()
                .chain(&tab.derivatives)
                .all(|v| v.is_finite())
        );

        let dofs = space.local_to_global(&mesh, 0);
        assert_eq!(dofs.len(), 10);
        assert!(dofs.windows(2).all(|window| window[0] + 1 == window[1]));
    }

    #[test]
    fn test_dg_p3_boundary_dofs_exclude_cell_bubble() {
        let mesh = unit_square_triangles(1);
        let space = DGSpace::new(PolynomialDegree::P3);
        let marker = mesh
            .boundaries
            .first()
            .expect("unit-square mesh has boundary faces")
            .marker;
        let boundary = space.boundary_dofs(&mesh, marker);

        // A cubic triangle has three vertices, two nodes on each edge, and
        // one cell-interior node. Boundary traces contain the first nine
        // local DOFs, while the cell bubble must remain unconstrained.
        for face in mesh.boundaries.iter().filter(|face| face.marker == marker) {
            let base = space.local_to_global(&mesh, face.element_idx)[0];
            assert!(!boundary.contains(&(base + 9)));
        }
        assert!(!boundary.is_empty());
    }

    #[test]
    fn test_dg_space_local_to_global_no_sharing() {
        let mesh = unit_square_triangles(2);
        let space = DGSpace::new(PolynomialDegree::P1);
        // Adjacent elements should NOT share DOFs
        let dofs0 = space.local_to_global(&mesh, 0);
        let dofs1 = space.local_to_global(&mesh, 1);
        for d in &dofs0 {
            assert!(!dofs1.contains(d), "DG DOFs should not be shared");
        }
    }

    #[test]
    fn test_dg_boundary_dofs_include_face_traces_not_cell_bubbles() {
        let mesh = unit_square_triangles(1);
        let space = DGSpace::new(PolynomialDegree::P2);
        let boundary_dofs: std::collections::HashSet<_> =
            space.boundary_dofs(&mesh, 0).into_iter().collect();

        for face in mesh.boundaries.iter().filter(|face| face.marker == 0) {
            let base = space
                .local_to_global(&mesh, face.element_idx)
                .first()
                .copied()
                .expect("boundary face must belong to an element");
            let elem = &mesh.elements[face.element_idx];
            let edge_idx = elem
                .element_type
                .local_edges()
                .iter()
                .position(|&(a, b)| {
                    let edge = [elem.vertices()[a], elem.vertices()[b]];
                    edge.iter().all(|node| face.nodes.contains(node))
                })
                .expect("boundary face must be an element edge");
            assert!(
                boundary_dofs.contains(&(base + elem.element_type.num_vertices() + edge_idx)),
                "boundary edge midpoint is missing from DG trace DOFs"
            );
        }

        // Each triangle has one interior edge. Its midpoint basis function is
        // a cell-interior trace DOF and must not be constrained on the outer
        // boundary.
        for elem_idx in 0..mesh.num_elements() {
            let elem = &mesh.elements[elem_idx];
            let base = space.local_to_global(&mesh, elem_idx)[0];
            for (edge_idx, &(a, b)) in elem.element_type.local_edges().iter().enumerate() {
                let is_boundary = mesh.boundaries.iter().any(|face| {
                    face.marker == 0
                        && face.element_idx == elem_idx
                        && face.nodes.contains(&elem.vertices()[a])
                        && face.nodes.contains(&elem.vertices()[b])
                });
                if !is_boundary {
                    assert!(
                        !boundary_dofs
                            .contains(&(base + elem.element_type.num_vertices() + edge_idx))
                    );
                }
            }
        }
    }

    #[test]
    fn test_interior_facets() {
        let mesh = unit_square_triangles(2);
        let facets = build_interior_facets(&mesh);
        // A 2×2 unit square has 8 triangles and several interior edges
        assert!(!facets.is_empty(), "Should find interior facets");
        for f in &facets {
            assert_ne!(f.elem_plus, f.elem_minus);
            assert!(f.area > 0.0, "Facet area should be positive");
            let norm = (f.normal[0].powi(2) + f.normal[1].powi(2) + f.normal[2].powi(2)).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Normal should be unit: {norm}");
        }
    }

    #[test]
    fn test_element_diameter() {
        let mesh = unit_square_triangles(2);
        for i in 0..mesh.num_elements() {
            let h = element_diameter(&mesh, i);
            assert!(h > 0.0 && h < 1.0, "h = {h} for unit square subdivision");
        }
    }
}
