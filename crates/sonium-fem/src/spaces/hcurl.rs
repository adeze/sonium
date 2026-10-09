//! H(curl)-conforming function space (Nédélec edge elements)
//!
//! DOFs are associated with mesh edges. The covariant Piola mapping ensures
//! tangential continuity across element boundaries:
//!
//! ```text
//! N_phys = J^{-T} N_ref          (covariant Piola mapping)
//! curl_phys = (1/|J|) J curl_ref  (curl Piola mapping)
//! ```

use crate::basis::nedelec;
use crate::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use crate::mesh::{ElementType, Mesh};

use super::contract::{
    PiolaError, covariant_piola_2d, covariant_piola_3d, curl_piola_2d, curl_piola_3d,
};
use super::traits::{
    BasisTabulation, FieldType, FunctionSpace, SpaceValidationError, TabulationError,
    validate_function_space,
};

/// H(curl)-conforming function space using Nédélec edge elements.
///
/// - **Order 1** (default): Lowest-order, 1 DOF per edge (3/tri, 4/quad,
///   6/tet, 12/hex).
/// - **Order 2**: Second-order simplex elements, 2 DOFs per edge + interior
///   DOFs (8/tri, 20/tet).
///
/// Per-point tabulation needs no global mesh topology (edge signs are
/// computed locally; see [`local_edge_signs`]). `local_to_global` and
/// `num_global_dofs` still build the global edge/face tables, so hoist
/// those calls out of quadrature loops.
#[derive(Debug, Clone)]
pub struct HcurlSpace {
    /// Nedelec element order (1 or 2).
    pub order: usize,
}

/// Edge orientation signs for one element, computed locally from global
/// node ids: `+1` when the element-local edge runs low→high, `-1`
/// otherwise.
///
/// This replicates the `(min, max, sign)` rule of the global topology
/// build exactly, but in O(edges) with no table: `try_tabulate` runs once
/// per quadrature point, and rebuilding the whole-mesh edge/face tables
/// there made multi-element H(curl) tabulation unusably slow (a 48-tet
/// exact-sequence check never finished). Callers that need *global* edge
/// indices must still go through the topology.
fn local_edge_signs(edges: &[(usize, usize)]) -> Vec<i8> {
    edges
        .iter()
        .map(|&(a, b)| if a < b { 1 } else { -1 })
        .collect()
}

impl HcurlSpace {
    /// Lowest-order Nédélec (order 1).
    pub fn new() -> Self {
        Self { order: 1 }
    }

    /// Second-order Nédélec (order 2): 8 DOFs per triangle, 20 per tet.
    pub fn order2() -> Self {
        Self { order: 2 }
    }
}

impl Default for HcurlSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionSpace for HcurlSpace {
    fn field_type(&self) -> FieldType {
        FieldType::Vector
    }

    fn validate(&self, mesh: &Mesh) -> Result<(), SpaceValidationError> {
        if !matches!(self.order, 1 | 2) {
            return Err(SpaceValidationError::UnsupportedOrder {
                requested: self.order,
                supported: "1 and 2",
            });
        }

        for (element, mesh_element) in mesh.elements.iter().enumerate() {
            let expected_vertices = mesh_element.element_type.num_vertices();
            if mesh_element.nodes.len() < expected_vertices {
                return Err(SpaceValidationError::InvalidMesh {
                    element,
                    reason: "element has too few node indices",
                });
            }
            if mesh_element
                .nodes
                .iter()
                .any(|&node| node >= mesh.nodes.len())
            {
                return Err(SpaceValidationError::InvalidMesh {
                    element,
                    reason: "element references a node outside the mesh",
                });
            }
            if mesh_element.try_polynomial_order().is_err() {
                return Err(SpaceValidationError::InvalidMesh {
                    element,
                    reason: "element node count is neither a supported P1 nor P2 layout",
                });
            }
            let supported = matches!(
                mesh_element.element_type,
                ElementType::Triangle | ElementType::Tetrahedron
            ) || (self.order == 1
                && matches!(
                    mesh_element.element_type,
                    ElementType::Quadrilateral | ElementType::Hexahedron
                ));
            if !supported {
                return Err(SpaceValidationError::UnsupportedElement {
                    element,
                    element_type: mesh_element.element_type,
                    order: self.order,
                });
            }
        }

        validate_function_space(self, mesh)
    }

    fn num_components(&self) -> usize {
        3
    }

    fn num_global_dofs(&self, mesh: &Mesh) -> usize {
        let topo = crate::mesh::topology::MeshTopology::build(mesh);
        let n_edges = topo.num_edges();
        match self.order {
            1 => n_edges,
            2 => {
                // Two moments per edge, two element-interior DOFs per
                // triangle, and two shared face DOFs per tetrahedral face.
                let n_triangle_interior = mesh
                    .elements
                    .iter()
                    .filter(|element| element.element_type == ElementType::Triangle)
                    .count()
                    * 2;
                n_edges * 2 + n_triangle_interior + topo.num_faces() * 2
            }
            _ => n_edges, // fallback to order 1
        }
    }

    fn num_local_dofs(&self, elem_type: ElementType) -> usize {
        match self.order {
            1 => elem_type.num_edges(),
            2 => match elem_type {
                ElementType::Triangle => 8,     // 3 edges × 2 + 2 interior
                ElementType::Tetrahedron => 20, // 6 edges × 2 + 8 face-interior (2 per face)
                _ => elem_type.num_edges(),
            },
            _ => elem_type.num_edges(),
        }
    }

    fn local_to_global(&self, mesh: &Mesh, elem_idx: usize) -> Vec<usize> {
        let topo = crate::mesh::topology::MeshTopology::build(mesh);
        let edge_indices = topo.element_edge_indices(elem_idx);
        match self.order {
            1 => edge_indices.to_vec(),
            2 => {
                let n_edges_global = topo.num_edges();
                let mut dofs =
                    Vec::with_capacity(self.num_local_dofs(mesh.elements[elem_idx].element_type));
                // 2 DOFs per edge: global_edge_idx * 2, global_edge_idx * 2 + 1
                for &edge_idx in edge_indices {
                    dofs.push(edge_idx * 2);
                    dofs.push(edge_idx * 2 + 1);
                }
                let triangle_interior_offset = n_edges_global * 2 + topo.num_faces() * 2;
                match mesh.elements[elem_idx].element_type {
                    ElementType::Triangle => {
                        // Triangle bubbles are element-local. Index them by
                        // the number of preceding triangles, not the raw
                        // element index, so mixed-dimensional meshes cannot
                        // collide with another element's interior DOFs.
                        let triangle_index = mesh.elements[..elem_idx]
                            .iter()
                            .filter(|element| element.element_type == ElementType::Triangle)
                            .count();
                        for k in 0..2 {
                            dofs.push(triangle_interior_offset + triangle_index * 2 + k);
                        }
                    }
                    ElementType::Tetrahedron => {
                        // Tetrahedral face bubbles are shared globally.
                        let face_offset = n_edges_global * 2
                            + mesh
                                .elements
                                .iter()
                                .filter(|element| element.element_type == ElementType::Triangle)
                                .count()
                                * 2;
                        for &face_idx in topo.element_face_indices(elem_idx) {
                            dofs.push(face_offset + face_idx * 2);
                            dofs.push(face_offset + face_idx * 2 + 1);
                        }
                    }
                    _ => {}
                }
                dofs
            }
            _ => edge_indices.to_vec(),
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
        self.try_tabulate(mesh, elem_idx, xi, eta, zeta)
            .unwrap_or_else(|error| panic!("HcurlSpace::tabulate rejected input: {error}"))
    }

    fn try_tabulate(
        &self,
        mesh: &Mesh,
        elem_idx: usize,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> Result<BasisTabulation, TabulationError> {
        if mesh
            .nodes
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
        {
            return Err(TabulationError::Piola(PiolaError::NonFiniteJacobian));
        }
        // NOTE: no `self.validate(mesh)` here by design. Validation walks
        // every element and (via `num_global_dofs`/`local_to_global`)
        // rebuilds the global edge/face tables, so running it per
        // quadrature point turned multi-element tabulation into an
        // O(points × elements) rebuild storm that never finished. Callers
        // must validate the space against the mesh once up front (the
        // `unreachable!` arms below enforce the same contract); only the
        // O(nodes) finiteness pre-check above stays per call.
        // Cheap per-element admission checks. Full `validate(mesh)` stays out
        // of this hot path by design (see NOTE above), but `try_tabulate`
        // must still reject malformed connectivity with the same typed
        // errors instead of panicking inside `edges()`/`vertices()`.
        let elem = mesh.elements.get(elem_idx).ok_or(TabulationError::Space(
            SpaceValidationError::InvalidMesh {
                element: elem_idx,
                reason: "element index out of bounds",
            },
        ))?;
        let elem_type = elem.element_type;
        if elem.nodes.len() < elem_type.num_vertices() {
            return Err(TabulationError::Space(SpaceValidationError::InvalidMesh {
                element: elem_idx,
                reason: "element has too few node indices",
            }));
        }
        if elem.nodes.iter().any(|&node| node >= mesh.nodes.len()) {
            return Err(TabulationError::Space(SpaceValidationError::InvalidMesh {
                element: elem_idx,
                reason: "element references node outside mesh",
            }));
        }
        let spatial_dim = elem_type.dimension();
        // Local edge signs (no global topology build — see `local_edge_signs`).
        let signs = local_edge_signs(&elem.edges());
        let geometry_degree = match elem.try_polynomial_order() {
            Ok(2) => PolynomialDegree::P2,
            _ => PolynomialDegree::P1,
        };
        let geom_shape = evaluate_shape(elem_type, geometry_degree, xi, eta, zeta);
        let geometry_nodes = elem.nodes.as_slice();
        let vertices = elem.vertices();

        match spatial_dim {
            2 => {
                let coords: Vec<[f64; 2]> = geometry_nodes
                    .iter()
                    .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y])
                    .collect();
                let jac = Jacobian::from_2d(&geom_shape.gradients, geom_shape.grad_dim, &coords);
                let jacobian = [
                    [jac.matrix[0][0], jac.matrix[0][1]],
                    [jac.matrix[1][0], jac.matrix[1][1]],
                ];

                // Dispatch to appropriate Nedelec order
                let (ref_vals_vec, ref_curls_vec): (Vec<[f64; 2]>, Vec<f64>) =
                    match (self.order, elem_type) {
                        (2, ElementType::Triangle) => {
                            let vertex_ids = [vertices[0], vertices[1], vertices[2]];
                            let (v, c) = nedelec::nedelec2_triangle_oriented(xi, eta, vertex_ids);
                            (v.to_vec(), c.to_vec())
                        }
                        (1, ElementType::Triangle) => {
                            let (v, c) = nedelec::nedelec1_triangle(xi, eta);
                            (v.to_vec(), c.to_vec())
                        }
                        (1, ElementType::Quadrilateral) => {
                            let (v, c) = nedelec::nedelec1_quadrilateral(xi, eta);
                            (v.to_vec(), c.to_vec())
                        }
                        _ => unreachable!("H(curl) element must be validated before tabulation"),
                    };
                let n_dofs = ref_vals_vec.len();

                // Covariant Piola: N_phys = J^{-T} N_ref
                let mut values = vec![0.0; n_dofs * 2];
                for (e, ref_val) in ref_vals_vec.iter().enumerate() {
                    // Edge DOFs get sign from topology; interior DOFs get +1
                    let s = if self.order == 1 && e < signs.len() {
                        signs[e] as f64
                    } else {
                        1.0
                    };
                    for i in 0..2 {
                        let mapped = covariant_piola_2d(jacobian, *ref_val)?;
                        values[e * 2 + i] = s * mapped[i];
                    }
                }

                // 2D curl is scalar: curl_phys = (1/|J|) * curl_ref
                let det = jac.det.abs();
                let derivatives: Vec<f64> = ref_curls_vec
                    .iter()
                    .enumerate()
                    .map(|(e, &c)| {
                        let s = if self.order == 1 && e < signs.len() {
                            signs[e] as f64
                        } else {
                            1.0
                        };
                        Ok::<f64, TabulationError>(s * curl_piola_2d(jacobian, c)?)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let (px, py) = phys_coords_2d(&geom_shape.values, geometry_nodes, &mesh.nodes);

                Ok(BasisTabulation {
                    n_dofs,
                    spatial_dim: 2,
                    values,
                    derivatives,
                    hessians: Vec::new(),
                    det_j: det,
                    physical_coords: [px, py, 0.0],
                })
            }
            3 => {
                let coords: Vec<[f64; 3]> = geometry_nodes
                    .iter()
                    .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y, mesh.nodes[v].z])
                    .collect();
                let jac = Jacobian::from_3d(&geom_shape.gradients, geom_shape.grad_dim, &coords);
                let jacobian = [
                    [jac.matrix[0][0], jac.matrix[0][1], jac.matrix[0][2]],
                    [jac.matrix[1][0], jac.matrix[1][1], jac.matrix[1][2]],
                    [jac.matrix[2][0], jac.matrix[2][1], jac.matrix[2][2]],
                ];
                let (ref_vals, ref_curls) = match (self.order, elem_type) {
                    (2, ElementType::Tetrahedron) => {
                        let vertex_ids = [vertices[0], vertices[1], vertices[2], vertices[3]];
                        let (values, curls) =
                            nedelec::nedelec2_tetrahedron_oriented(xi, eta, zeta, vertex_ids);
                        (values.to_vec(), curls.to_vec())
                    }
                    (1, ElementType::Tetrahedron) => {
                        let (values, curls) = nedelec::nedelec1_tetrahedron(xi, eta, zeta);
                        (values.to_vec(), curls.to_vec())
                    }
                    (1, ElementType::Hexahedron) => {
                        let (values, curls) = nedelec::nedelec1_hexahedron(xi, eta, zeta);
                        (values.to_vec(), curls.to_vec())
                    }
                    _ => unreachable!("H(curl) element must be validated before tabulation"),
                };
                let n_dofs = ref_vals.len();

                // Covariant Piola: N_phys = J^{-T} N_ref
                let mut values = vec![0.0; n_dofs * 3];
                for (e, ref_val) in ref_vals.iter().enumerate() {
                    let s = if self.order == 1 {
                        signs[e] as f64
                    } else {
                        1.0
                    };
                    for i in 0..3 {
                        let mapped = covariant_piola_3d(jacobian, *ref_val)?;
                        values[e * 3 + i] = s * mapped[i];
                    }
                }

                // Curl Piola: curl_phys = (1/|J|) J curl_ref
                let det = jac.det.abs();
                let mut derivatives = vec![0.0; n_dofs * 3];
                for (e, ref_curl) in ref_curls.iter().enumerate() {
                    let s = if self.order == 1 {
                        signs[e] as f64
                    } else {
                        1.0
                    };
                    for i in 0..3 {
                        let mapped = curl_piola_3d(jacobian, *ref_curl)?;
                        derivatives[e * 3 + i] = s * mapped[i];
                    }
                }

                let (px, py, pz) = phys_coords_3d(&geom_shape.values, geometry_nodes, &mesh.nodes);

                Ok(BasisTabulation {
                    n_dofs,
                    spatial_dim: 3,
                    values,
                    derivatives,
                    hessians: Vec::new(),
                    det_j: det,
                    physical_coords: [px, py, pz],
                })
            }
            _ => unreachable!(),
        }
    }

    fn boundary_dofs(&self, mesh: &Mesh, boundary_marker: i32) -> Vec<usize> {
        let topo = crate::mesh::topology::MeshTopology::build(mesh);
        if self.order == 1 {
            return topo
                .boundary_edges
                .get(&boundary_marker)
                .cloned()
                .unwrap_or_default();
        }

        let mut dofs = Vec::new();
        for edge_idx in topo
            .boundary_edges
            .get(&boundary_marker)
            .into_iter()
            .flatten()
        {
            dofs.push(edge_idx * 2);
            dofs.push(edge_idx * 2 + 1);
        }

        let face_offset = topo.num_edges() * 2
            + mesh
                .elements
                .iter()
                .filter(|element| element.element_type == ElementType::Triangle)
                .count()
                * 2;
        for face_idx in topo
            .boundary_faces
            .get(&boundary_marker)
            .into_iter()
            .flatten()
        {
            dofs.push(face_offset + face_idx * 2);
            dofs.push(face_offset + face_idx * 2 + 1);
        }
        dofs.sort_unstable();
        dofs.dedup();
        dofs
    }
}

fn phys_coords_2d(
    shape_vals: &[f64],
    vertices: &[usize],
    nodes: &[crate::mesh::Point],
) -> (f64, f64) {
    let mut px = 0.0;
    let mut py = 0.0;
    for (i, &v) in vertices.iter().enumerate() {
        px += shape_vals[i] * nodes[v].x;
        py += shape_vals[i] * nodes[v].y;
    }
    (px, py)
}

fn phys_coords_3d(
    shape_vals: &[f64],
    vertices: &[usize],
    nodes: &[crate::mesh::Point],
) -> (f64, f64, f64) {
    let mut px = 0.0;
    let mut py = 0.0;
    let mut pz = 0.0;
    for (i, &v) in vertices.iter().enumerate() {
        px += shape_vals[i] * nodes[v].x;
        py += shape_vals[i] * nodes[v].y;
        pz += shape_vals[i] * nodes[v].z;
    }
    (px, py, pz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::contract::PiolaError;
    use crate::spaces::traits::TabulationError;

    #[test]
    fn test_hcurl_space_dof_count() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();

        let n_dofs = space.num_global_dofs(&mesh);
        let topo = crate::mesh::topology::MeshTopology::build(&mesh);
        assert_eq!(n_dofs, topo.num_edges());
    }

    #[test]
    fn test_hcurl_local_to_global() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();

        let dofs = space.local_to_global(&mesh, 0);
        assert_eq!(dofs.len(), 3);

        let n_edges = space.num_global_dofs(&mesh);
        for &d in &dofs {
            assert!(d < n_edges, "DOF {d} >= num_edges {n_edges}");
        }
    }

    #[test]
    fn test_hcurl_shared_edge_orientation_is_antisymmetric() {
        let mesh = unit_square_triangles(1);
        let space = HcurlSpace::new();
        let topology = crate::mesh::topology::MeshTopology::build(&mesh);

        let first_edges = topology.element_edge_indices(0);
        let second_edges = topology.element_edge_indices(1);
        let (first_local, &shared_edge) = first_edges
            .iter()
            .enumerate()
            .find(|(_, edge)| second_edges.contains(edge))
            .expect("two triangles must share an edge");
        let second_local = second_edges
            .iter()
            .position(|edge| *edge == shared_edge)
            .expect("shared edge must have a local index");
        assert_eq!(
            space.local_to_global(&mesh, 0)[first_local],
            space.local_to_global(&mesh, 1)[second_local]
        );
        assert_eq!(
            topology.element_edge_signs(0)[first_local],
            -topology.element_edge_signs(1)[second_local],
            "shared H(curl) edge must use opposite local orientations"
        );
    }

    #[test]
    fn test_local_edge_signs_match_global_topology() {
        // `try_tabulate` uses `local_edge_signs` instead of a per-call
        // topology rebuild; pin the two to each other on tri and tet
        // meshes so a divergence can never silently flip orientations.
        let tri_mesh = unit_square_triangles(1);
        let tri_topo = crate::mesh::topology::MeshTopology::build(&tri_mesh);
        for (elem_idx, elem) in tri_mesh.elements.iter().enumerate() {
            assert_eq!(
                local_edge_signs(&elem.edges()),
                tri_topo.element_edge_signs(elem_idx),
                "triangle element {elem_idx}"
            );
        }
        let tet_mesh = crate::mesh::unit_cube_tetrahedra(1);
        let tet_topo = crate::mesh::topology::MeshTopology::build(&tet_mesh);
        for (elem_idx, elem) in tet_mesh.elements.iter().enumerate() {
            assert_eq!(
                local_edge_signs(&elem.edges()),
                tet_topo.element_edge_signs(elem_idx),
                "tetrahedron element {elem_idx}"
            );
        }
    }

    #[test]
    fn test_hcurl_tabulate_2d() {
        let mesh = unit_square_triangles(3);
        let space = HcurlSpace::new();

        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);

        assert_eq!(tab.n_dofs, 3);
        assert_eq!(tab.spatial_dim, 2);
        assert_eq!(tab.values.len(), 6);
        assert_eq!(tab.derivatives.len(), 3); // scalar curls in 2D
        assert!(tab.det_j > 0.0);
    }

    #[test]
    fn test_hcurl_tabulate_3d() {
        let mesh = crate::mesh::unit_cube_tetrahedra(2);
        let space = HcurlSpace::new();

        let tab = space.tabulate(&mesh, 0, 0.25, 0.25, 0.25);

        assert_eq!(tab.n_dofs, 6);
        assert_eq!(tab.spatial_dim, 3);
        assert_eq!(tab.values.len(), 18);
        assert_eq!(tab.derivatives.len(), 18); // vector curls in 3D
        assert!(tab.det_j > 0.0);
    }

    #[test]
    fn test_hcurl_order2_triangle_numbering_and_tabulation() {
        let mesh = unit_square_triangles(1);
        let space = HcurlSpace::order2();
        let topology = crate::mesh::topology::MeshTopology::build(&mesh);

        assert_eq!(
            space.num_global_dofs(&mesh),
            topology.num_edges() * 2 + mesh.num_elements() * 2
        );
        for elem_idx in 0..mesh.num_elements() {
            let dofs = space.local_to_global(&mesh, elem_idx);
            assert_eq!(dofs.len(), 8);
            assert!(dofs.iter().all(|&dof| dof < space.num_global_dofs(&mesh)));
        }

        let tab = space.tabulate(&mesh, 0, 0.25, 0.25, 0.0);
        assert_eq!(tab.n_dofs, 8);
        assert_eq!(tab.values.len(), 16);
        assert_eq!(tab.derivatives.len(), 8);
    }

    #[test]
    fn test_hcurl_order2_tetrahedron_numbering_and_tabulation() {
        let mesh = crate::mesh::unit_cube_tetrahedra(1);
        let space = HcurlSpace::order2();
        let topology = crate::mesh::topology::MeshTopology::build(&mesh);
        let expected = topology.num_edges() * 2 + topology.num_faces() * 2;
        assert_eq!(space.num_global_dofs(&mesh), expected);

        for elem_idx in 0..mesh.num_elements() {
            let dofs = space.local_to_global(&mesh, elem_idx);
            assert_eq!(dofs.len(), 20);
            let mut unique = dofs.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(
                unique.len(),
                20,
                "local order-2 tetra DOFs must be distinct"
            );
            assert!(dofs.iter().all(|&dof| dof < expected));
        }

        let mut found_shared_face = false;
        'pair: for first_elem in 0..mesh.num_elements() {
            let first_faces = topology.element_face_indices(first_elem);
            for second_elem in (first_elem + 1)..mesh.num_elements() {
                let second_faces = topology.element_face_indices(second_elem);
                if let Some((first_local, &shared_face)) = first_faces
                    .iter()
                    .enumerate()
                    .find(|(_, face)| second_faces.contains(face))
                {
                    let second_local = second_faces
                        .iter()
                        .position(|face| *face == shared_face)
                        .expect("shared face must have a local index");
                    let first_dofs = space.local_to_global(&mesh, first_elem);
                    let second_dofs = space.local_to_global(&mesh, second_elem);
                    assert_eq!(
                        &first_dofs[12 + 2 * first_local..12 + 2 * first_local + 2],
                        &second_dofs[12 + 2 * second_local..12 + 2 * second_local + 2],
                        "shared tetrahedral faces must share both order-2 face DOFs"
                    );
                    found_shared_face = true;
                    break 'pair;
                }
            }
        }
        assert!(
            found_shared_face,
            "the tetrahedral test mesh must contain a shared face"
        );

        let tab = space.tabulate(&mesh, 0, 0.2, 0.2, 0.2);
        assert_eq!(tab.n_dofs, 20);
        assert_eq!(tab.values.len(), 60);
        assert_eq!(tab.derivatives.len(), 60);
    }

    #[test]
    fn checked_tabulation_rejects_singular_and_nonfinite_jacobians() {
        let space = HcurlSpace::new();
        let mut singular = unit_square_triangles(1);
        singular.nodes[3].x = 2.0;
        singular.nodes[3].y = 0.0;
        assert_eq!(
            space
                .try_tabulate(&singular, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0)
                .unwrap_err(),
            TabulationError::Piola(PiolaError::SingularJacobian)
        );

        let mut nonfinite = unit_square_triangles(1);
        nonfinite.nodes[0].x = f64::NAN;
        assert_eq!(
            space
                .try_tabulate(&nonfinite, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0)
                .unwrap_err(),
            TabulationError::Piola(PiolaError::NonFiniteJacobian)
        );
    }

    #[test]
    fn checked_tabulation_rejects_singular_three_dimensional_jacobian() {
        let mut mesh = crate::mesh::unit_cube_tetrahedra(1);
        for node in &mut mesh.nodes {
            node.z = 0.0;
        }
        assert_eq!(
            HcurlSpace::new()
                .try_tabulate(&mesh, 0, 0.25, 0.25, 0.25)
                .unwrap_err(),
            TabulationError::Piola(PiolaError::SingularJacobian)
        );
    }
}
