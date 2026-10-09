//! H(div)-conforming function space (Raviart-Thomas face elements)
//!
//! DOFs are associated with mesh faces. The contravariant Piola mapping ensures
//! normal continuity across element boundaries:
//!
//! ```text
//! N_phys = (1/|J|) J N_ref     (contravariant Piola mapping)
//! div_phys = (1/|J|) div_ref    (divergence Piola mapping)
//! ```
//!
//! This space is used for mixed formulations like Darcy flow, Stokes flow,
//! and electromagnetic formulations requiring div-conforming vector fields.

use crate::basis::raviart_thomas;
use crate::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use crate::mesh::{ElementType, Mesh};

use super::contract::{
    PiolaError, contravariant_piola_2d, contravariant_piola_3d, divergence_piola_2d,
    divergence_piola_3d,
};
use super::traits::{
    BasisTabulation, FieldType, FunctionSpace, SpaceValidationError, TabulationError,
    validate_function_space,
};

/// Public polynomial order supported by [`HdivSpace`]. RT1 includes affine
/// tetrahedral face and cell moments; curved mixed variants remain separate.
///
/// `Rt0` is the backwards-compatible lowest-order space on triangles,
/// quadrilaterals, tetrahedra, and hexahedra. `Rt1` is implemented on affine
/// triangles and quadrilaterals with two oriented edge moments per edge, and
/// on tetrahedra with three face moments per face. Cell moments complete the
/// corresponding local spaces; curved and higher-order mixed variants remain
/// outside this bounded contract.
/// RT1 tetrahedra use fifteen local DOFs: three face moments on each of the
/// four faces and three Cartesian cell moments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HdivOrder {
    /// Lowest-order Raviart--Thomas space.
    Rt0,
    /// First-order Raviart--Thomas space on affine triangles, quadrilaterals,
    /// tetrahedra, and hexahedra.
    Rt1,
}

impl HdivOrder {
    /// Polynomial degree of the reference vector basis.
    pub const fn polynomial_degree(self) -> usize {
        match self {
            Self::Rt0 => 1,
            Self::Rt1 => 2,
        }
    }
}

/// H(div)-conforming Raviart--Thomas function space.
///
/// `HdivSpace::new()` retains the legacy RT0 contract.  Use
/// `HdivSpace::with_order(HdivOrder::Rt1)` for the high-order 2-D
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdivSpace {
    order: HdivOrder,
}

impl HdivSpace {
    /// Construct the backwards-compatible RT0 space.
    pub fn new() -> Self {
        Self {
            order: HdivOrder::Rt0,
        }
    }

    /// Construct an H(div) space at an explicit Raviart--Thomas order.
    pub const fn with_order(order: HdivOrder) -> Self {
        Self { order }
    }

    /// Return the configured Raviart--Thomas order.
    pub const fn order(self) -> HdivOrder {
        self.order
    }

    /// Whether this space has a complete implementation for an element type.
    pub const fn supports_element(self, elem_type: ElementType) -> bool {
        match self.order {
            HdivOrder::Rt0 => true,
            HdivOrder::Rt1 => {
                matches!(
                    elem_type,
                    ElementType::Triangle
                        | ElementType::Quadrilateral
                        | ElementType::Tetrahedron
                        | ElementType::Hexahedron
                )
            }
        }
    }
}

impl Default for HdivSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionSpace for HdivSpace {
    fn field_type(&self) -> FieldType {
        FieldType::Vector
    }

    fn num_components(&self) -> usize {
        3
    }

    fn num_global_dofs(&self, mesh: &Mesh) -> usize {
        let topo = crate::mesh::topology::MeshTopology::build(mesh);
        match self.order {
            HdivOrder::Rt0 => {
                if mesh.dimension == 2 {
                    topo.num_edges()
                } else {
                    topo.num_faces()
                }
            }
            HdivOrder::Rt1 => {
                if mesh.dimension == 2 {
                    2 * topo.num_edges()
                        + mesh
                            .elements
                            .iter()
                            .map(|element| match element.element_type {
                                ElementType::Triangle => 2,
                                ElementType::Quadrilateral => 4,
                                _ => 0,
                            })
                            .sum::<usize>()
                } else if mesh.dimension == 3 {
                    let face_dofs = topo.faces.iter().map(face_mode_count).sum::<usize>();
                    let cell_dofs = mesh
                        .elements
                        .iter()
                        .map(|element| match element.element_type {
                            ElementType::Tetrahedron => 3,
                            ElementType::Hexahedron => 12,
                            _ => 0,
                        })
                        .sum::<usize>();
                    face_dofs + cell_dofs
                } else {
                    0
                }
            }
        }
    }

    fn num_local_dofs(&self, elem_type: ElementType) -> usize {
        match (self.order, elem_type) {
            (HdivOrder::Rt0, ElementType::Triangle) => 3,
            (HdivOrder::Rt0, ElementType::Quadrilateral) => 4,
            (HdivOrder::Rt0, ElementType::Tetrahedron) => 4,
            (HdivOrder::Rt0, ElementType::Hexahedron) => 6,
            (HdivOrder::Rt1, ElementType::Triangle) => 8,
            (HdivOrder::Rt1, ElementType::Quadrilateral) => 12,
            (HdivOrder::Rt1, ElementType::Tetrahedron) => 15,
            (HdivOrder::Rt1, ElementType::Hexahedron) => 36,
        }
    }

    fn local_to_global(&self, mesh: &Mesh, elem_idx: usize) -> Vec<usize> {
        let topo = crate::mesh::topology::MeshTopology::build(mesh);
        if !self.supports_element(mesh.elements[elem_idx].element_type) {
            return Vec::new();
        }
        match self.order {
            HdivOrder::Rt0 => {
                if mesh.dimension == 2 {
                    topo.element_edge_indices(elem_idx).to_vec()
                } else {
                    topo.element_face_indices(elem_idx).to_vec()
                }
            }
            HdivOrder::Rt1 => {
                if mesh.dimension == 2 {
                    let edge_dofs = 2 * topo.num_edges();
                    let cell_offset = mesh
                        .elements
                        .iter()
                        .take(elem_idx)
                        .map(|element| match element.element_type {
                            ElementType::Triangle => 2,
                            ElementType::Quadrilateral => 4,
                            _ => 0,
                        })
                        .sum::<usize>();
                    let local_cell_dofs = self.num_local_dofs(mesh.elements[elem_idx].element_type)
                        - 2 * topo.element_edge_indices(elem_idx).len();
                    let mut dofs = Vec::with_capacity(
                        2 * topo.element_edge_indices(elem_idx).len() + local_cell_dofs,
                    );
                    for &edge in topo.element_edge_indices(elem_idx) {
                        dofs.push(2 * edge);
                        dofs.push(2 * edge + 1);
                    }
                    dofs.extend(
                        (0..local_cell_dofs).map(|local_dof| edge_dofs + cell_offset + local_dof),
                    );
                    dofs
                } else {
                    let face_offsets = face_dof_offsets(&topo);
                    let face_dofs = face_offsets.last().copied().unwrap_or(0);
                    let cell_offset = mesh
                        .elements
                        .iter()
                        .take(elem_idx)
                        .map(|element| cell_mode_count(element.element_type))
                        .sum::<usize>();
                    let faces = topo.element_face_indices(elem_idx);
                    let local_face_modes = faces
                        .iter()
                        .map(|&face| face_mode_count(&topo.faces[face]))
                        .sum::<usize>();
                    let local_cell_modes = cell_mode_count(mesh.elements[elem_idx].element_type);
                    let mut dofs = Vec::with_capacity(local_face_modes + local_cell_modes);
                    for &face in faces {
                        let start = face_offsets[face];
                        dofs.extend(start..start + face_mode_count(&topo.faces[face]));
                    }
                    dofs.extend(
                        (0..local_cell_modes).map(|local_dof| face_dofs + cell_offset + local_dof),
                    );
                    dofs
                }
            }
        }
    }

    fn validate(&self, mesh: &Mesh) -> Result<(), SpaceValidationError> {
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
            if !self.supports_element(mesh_element.element_type) {
                return Err(SpaceValidationError::UnsupportedElement {
                    element,
                    element_type: mesh_element.element_type,
                    order: self.order.polynomial_degree(),
                });
            }
        }
        validate_function_space(self, mesh)
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
            .unwrap_or_else(|error| panic!("HdivSpace::tabulate rejected input: {error}"))
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
        self.validate(mesh)?;
        let elem = &mesh.elements[elem_idx];
        let elem_type = elem.element_type;
        let spatial_dim = elem_type.dimension();
        let topo = crate::mesh::topology::MeshTopology::build(mesh);
        // In 2D, H(div) DOFs live on edges; in 3D, on faces.
        let signs = if spatial_dim == 2 {
            topo.element_edge_signs(elem_idx)
        } else {
            topo.element_face_signs(elem_idx)
        };
        // Use the full isoparametric geometry for explicit P2/Q2 elements;
        // retain the legacy affine vertex-only path otherwise.
        let geometry_degree = if elem.try_polynomial_order().ok() == Some(2) {
            PolynomialDegree::P2
        } else {
            PolynomialDegree::P1
        };
        let geom_shape = evaluate_shape(elem_type, geometry_degree, xi, eta, zeta);
        let geometry_nodes = &elem.nodes;

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
                let (ref_vals, ref_divs): (Vec<[f64; 2]>, Vec<f64>) = match (self.order, elem_type)
                {
                    (HdivOrder::Rt0, ElementType::Triangle) => {
                        let (values, divs) = raviart_thomas::rt0_triangle(xi, eta);
                        (values.to_vec(), divs.to_vec())
                    }
                    (HdivOrder::Rt0, ElementType::Quadrilateral) => {
                        let (values, divs) = raviart_thomas::rt0_quadrilateral(xi, eta);
                        (values.to_vec(), divs.to_vec())
                    }
                    (HdivOrder::Rt1, ElementType::Triangle) => {
                        let (values, divs) = raviart_thomas::rt1_triangle(xi, eta);
                        (values.to_vec(), divs.to_vec())
                    }
                    (HdivOrder::Rt1, ElementType::Quadrilateral) => {
                        let (values, divs) = raviart_thomas::rt1_quadrilateral(xi, eta);
                        (values.to_vec(), divs.to_vec())
                    }
                    _ => unreachable!("unsupported 2D H(div) order/element pair"),
                };
                let n_dofs = ref_vals.len();

                let det = jac.det.abs();

                let mut values = vec![0.0; n_dofs * 2];
                for (local_dof, ref_val) in ref_vals.iter().enumerate() {
                    let orientation = match self.order {
                        HdivOrder::Rt0 => signs[local_dof] as f64,
                        HdivOrder::Rt1 => {
                            // Reversing an edge reverses both its outward
                            // normal and centered coordinate.  The resulting
                            // orientation factor is s^(q+1): mode 0 gets s,
                            // while the odd mode gets +1.
                            if local_dof < 2 * signs.len() {
                                let edge = local_dof / 2;
                                let mode = local_dof % 2;
                                let s = signs[edge] as f64;
                                if mode == 0 { s } else { 1.0 }
                            } else {
                                1.0
                            }
                        }
                    };
                    let mapped = contravariant_piola_2d(jacobian, *ref_val)?;
                    for i in 0..2 {
                        values[local_dof * 2 + i] = orientation * mapped[i];
                    }
                }

                let derivatives: Vec<f64> = ref_divs
                    .iter()
                    .enumerate()
                    .map(|(local_dof, &d)| {
                        let orientation = match self.order {
                            HdivOrder::Rt0 => signs[local_dof] as f64,
                            HdivOrder::Rt1 => {
                                if local_dof < 2 * signs.len() {
                                    let edge = local_dof / 2;
                                    let mode = local_dof % 2;
                                    let s = signs[edge] as f64;
                                    if mode == 0 { s } else { 1.0 }
                                } else {
                                    1.0
                                }
                            }
                        };
                        Ok::<f64, TabulationError>(orientation * divergence_piola_2d(jacobian, d)?)
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
                let (ref_vals, ref_divs): (Vec<[f64; 3]>, Vec<f64>) = match (self.order, elem_type)
                {
                    (HdivOrder::Rt0, ElementType::Tetrahedron) => {
                        let (values, divs) = raviart_thomas::rt0_tetrahedron(xi, eta, zeta);
                        (values.to_vec(), divs.to_vec())
                    }
                    (HdivOrder::Rt0, ElementType::Hexahedron) => {
                        let (values, divs) = raviart_thomas::rt0_hexahedron(xi, eta, zeta);
                        (values.to_vec(), divs.to_vec())
                    }
                    (HdivOrder::Rt1, ElementType::Tetrahedron) => {
                        let (values, divs) = raviart_thomas::rt1_tetrahedron(xi, eta, zeta);
                        (values.to_vec(), divs.to_vec())
                    }
                    (HdivOrder::Rt1, ElementType::Hexahedron) => {
                        let face_axes = hexahedron_face_axes(mesh, &topo, elem_idx);
                        let (values, divs) =
                            raviart_thomas::rt1_hexahedron_oriented(xi, eta, zeta, face_axes);
                        (values.to_vec(), divs.to_vec())
                    }
                    _ => unreachable!("unsupported 3D H(div) order/element pair"),
                };
                let n_dofs = ref_vals.len();
                let det = jac.det.abs();
                let mut values = vec![0.0; n_dofs * 3];
                for (local_dof, ref_val) in ref_vals.iter().enumerate() {
                    let orientation = match self.order {
                        HdivOrder::Rt0 => signs[local_dof] as f64,
                        HdivOrder::Rt1 => {
                            if elem_type == ElementType::Tetrahedron && local_dof < 12 {
                                signs[local_dof / 3] as f64
                            } else if elem_type == ElementType::Hexahedron && local_dof < 24 {
                                signs[local_dof / 4] as f64
                            } else {
                                1.0
                            }
                        }
                    };
                    let mapped = contravariant_piola_3d(jacobian, *ref_val)?;
                    for i in 0..3 {
                        values[local_dof * 3 + i] = orientation * mapped[i];
                    }
                }
                let derivatives: Vec<f64> = ref_divs
                    .iter()
                    .enumerate()
                    .map(|(local_dof, &d)| {
                        let orientation = match self.order {
                            HdivOrder::Rt0 => signs[local_dof] as f64,
                            HdivOrder::Rt1 => {
                                if elem_type == ElementType::Tetrahedron && local_dof < 12 {
                                    signs[local_dof / 3] as f64
                                } else if elem_type == ElementType::Hexahedron && local_dof < 24 {
                                    signs[local_dof / 4] as f64
                                } else {
                                    1.0
                                }
                            }
                        };
                        Ok::<f64, TabulationError>(orientation * divergence_piola_3d(jacobian, d)?)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
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
        match self.order {
            HdivOrder::Rt0 => {
                if mesh.dimension == 2 {
                    topo.boundary_edges
                        .get(&boundary_marker)
                        .cloned()
                        .unwrap_or_default()
                } else {
                    topo.boundary_faces
                        .get(&boundary_marker)
                        .cloned()
                        .unwrap_or_default()
                }
            }
            HdivOrder::Rt1 => {
                if mesh.dimension == 2 {
                    topo.boundary_edges
                        .get(&boundary_marker)
                        .into_iter()
                        .flat_map(|edges| edges.iter().flat_map(|&edge| [2 * edge, 2 * edge + 1]))
                        .collect()
                } else {
                    let offsets = face_dof_offsets(&topo);
                    topo.boundary_faces
                        .get(&boundary_marker)
                        .into_iter()
                        .flat_map(|faces| {
                            faces.iter().flat_map(|&face| {
                                let start = offsets[face];
                                start..start + face_mode_count(&topo.faces[face])
                            })
                        })
                        .collect()
                }
            }
        }
    }

    fn polynomial_degree(&self) -> usize {
        self.order.polynomial_degree()
    }
}

impl HdivSpace {
    /// Return the edge and cell portions of a validated RT1 2-D mesh.
    ///
    /// The numbering is `2 * num_edges + cell_dofs`: the first block stores
    /// edge moments and the second block stores cell moments in mesh element
    /// order. Triangles contribute two cell moments and quadrilaterals four.
    pub fn rt1_global_dof_layout(&self, mesh: &Mesh) -> Option<(usize, usize)> {
        if self.order != HdivOrder::Rt1 || self.validate(mesh).is_err() {
            return None;
        }
        let topology = crate::mesh::topology::MeshTopology::build(mesh);
        let cell_dofs = mesh
            .elements
            .iter()
            .map(|element| match element.element_type {
                ElementType::Triangle => 2,
                ElementType::Quadrilateral => 4,
                _ => 0,
            })
            .sum();
        if mesh.dimension == 2 {
            Some((2 * topology.num_edges(), cell_dofs))
        } else if mesh.dimension == 3 {
            let face_dofs = topology.faces.iter().map(face_mode_count).sum::<usize>();
            let cell_dofs = mesh
                .elements
                .iter()
                .map(|element| cell_mode_count(element.element_type))
                .sum::<usize>();
            Some((face_dofs, cell_dofs))
        } else {
            None
        }
    }
}

fn face_mode_count(face: &crate::mesh::Face) -> usize {
    match face {
        crate::mesh::Face::Triangle(_) => 3,
        crate::mesh::Face::Quad(_) => 4,
    }
}

fn cell_mode_count(element_type: ElementType) -> usize {
    match element_type {
        ElementType::Triangle => 2,
        ElementType::Quadrilateral => 4,
        ElementType::Tetrahedron => 3,
        ElementType::Hexahedron => 12,
    }
}

fn face_dof_offsets(topo: &crate::mesh::MeshTopology) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(topo.faces.len() + 1);
    offsets.push(0);
    for face in &topo.faces {
        offsets.push(offsets.last().copied().unwrap_or(0) + face_mode_count(face));
    }
    offsets
}

/// Map each local hexahedral face's tangential coordinates to the canonical
/// coordinates chosen by the first element that introduced the global face.
fn hexahedron_face_axes(
    mesh: &Mesh,
    topo: &crate::mesh::MeshTopology,
    elem_idx: usize,
) -> [[i8; 4]; 6] {
    let vertices = mesh.elements[elem_idx].vertices();
    let local_faces = ElementType::Hexahedron.local_faces();
    let reference_vertices = [
        [-1_i8, -1_i8, -1_i8],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ];
    let canonical_face_coordinates = [[-1_i8, -1_i8], [1, -1], [1, 1], [-1, 1]];
    let mut result = [[1_i8, 0, 0, 1]; 6];

    for (local_face, &face_vertices) in local_faces.iter().enumerate() {
        let global_face = topo.element_face_indices(elem_idx)[local_face];
        let canonical_nodes = topo.face(global_face).nodes();
        let mut sums = [0_i16; 4];
        for &local_vertex in face_vertices {
            let node = vertices[local_vertex];
            let Some(canonical_corner) = canonical_nodes
                .iter()
                .position(|&candidate| candidate == node)
            else {
                continue;
            };
            let [u, v] = canonical_face_coordinates[canonical_corner];
            let reference = reference_vertices[local_vertex];
            let [t0, t1] = match local_face {
                0 | 1 => [reference[0], reference[1]],
                2 | 3 => [reference[0], reference[2]],
                4 | 5 => [reference[1], reference[2]],
                _ => unreachable!(),
            };
            sums[0] += i16::from(u) * i16::from(t0);
            sums[1] += i16::from(u) * i16::from(t1);
            sums[2] += i16::from(v) * i16::from(t0);
            sums[3] += i16::from(v) * i16::from(t1);
        }
        result[local_face] = sums.map(|sum| (sum / 4) as i8);
    }
    result
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
    fn test_hdiv_space_dof_count_2d() {
        let mesh = unit_square_triangles(2);
        let space = HdivSpace::new();

        let n_dofs = space.num_global_dofs(&mesh);
        let topo = crate::mesh::topology::MeshTopology::build(&mesh);
        // In 2D, H(div) DOFs live on edges, not faces
        assert_eq!(n_dofs, topo.num_edges());
        assert!(n_dofs > 0);
    }

    #[test]
    fn test_hdiv_space_dof_count_3d() {
        let mesh = crate::mesh::unit_cube_tetrahedra(2);
        let space = HdivSpace::new();

        let n_dofs = space.num_global_dofs(&mesh);
        let topo = crate::mesh::topology::MeshTopology::build(&mesh);
        assert_eq!(n_dofs, topo.num_faces());
    }

    #[test]
    fn test_hdiv_local_to_global() {
        let mesh = unit_square_triangles(2);
        let space = HdivSpace::new();

        let dofs = space.local_to_global(&mesh, 0);
        assert_eq!(dofs.len(), 3);

        let n_faces = space.num_global_dofs(&mesh);
        for &d in &dofs {
            assert!(d < n_faces, "DOF {d} >= num_faces {n_faces}");
        }
    }

    #[test]
    fn test_hdiv_shared_edge_orientation_is_antisymmetric() {
        let mesh = unit_square_triangles(1);
        let space = HdivSpace::new();
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
            "shared H(div) edge must use opposite local orientations"
        );
    }

    #[test]
    fn test_hdiv_rt0_shared_edge_has_one_global_normal_trace() {
        let mesh = unit_square_triangles(1);
        let space = HdivSpace::new();
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

        let reference_edge_midpoint = |local_edge| match local_edge {
            0 => [0.5, 0.0],
            1 => [0.5, 0.5],
            2 => [0.0, 0.5],
            _ => unreachable!("triangle has exactly three local edges"),
        };
        let first_point = reference_edge_midpoint(first_local);
        let second_point = reference_edge_midpoint(second_local);
        let first = space.tabulate(&mesh, 0, first_point[0], first_point[1], 0.0);
        let second = space.tabulate(&mesh, 1, second_point[0], second_point[1], 0.0);

        let (a, b) = topology.edges[shared_edge];
        let dx = mesh.nodes[b].x - mesh.nodes[a].x;
        let dy = mesh.nodes[b].y - mesh.nodes[a].y;
        let edge_length = (dx * dx + dy * dy).sqrt();
        // The mesh topology's canonical orientation is a → b. Use its right
        // normal, matching the reference triangle's outward local-face trace.
        let canonical_normal = [dy / edge_length, -dx / edge_length];
        let normal_trace = |tab: &BasisTabulation, local| {
            let offset = local * tab.spatial_dim;
            tab.values[offset] * canonical_normal[0] + tab.values[offset + 1] * canonical_normal[1]
        };

        let first_trace = normal_trace(&first, first_local);
        let second_trace = normal_trace(&second, second_local);
        assert!(
            (first_trace - 1.0 / edge_length).abs() < 1e-12,
            "first local trace should realize the canonical shared-edge flux"
        );
        assert!(
            (second_trace - 1.0 / edge_length).abs() < 1e-12,
            "second local trace should realize the canonical shared-edge flux"
        );
        assert!((first_trace - second_trace).abs() < 1e-12);
        assert_eq!(
            space.local_to_global(&mesh, 0)[first_local],
            space.local_to_global(&mesh, 1)[second_local]
        );
    }

    #[test]
    fn test_hdiv_tabulate_2d() {
        let mesh = unit_square_triangles(3);
        let space = HdivSpace::new();

        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);

        assert_eq!(tab.n_dofs, 3);
        assert_eq!(tab.spatial_dim, 2);
        assert_eq!(tab.values.len(), 6);
        assert_eq!(tab.derivatives.len(), 3);
        assert!(tab.det_j > 0.0);
    }

    #[test]
    fn test_hdiv_tabulate_3d() {
        let mesh = crate::mesh::unit_cube_tetrahedra(2);
        let space = HdivSpace::new();

        let tab = space.tabulate(&mesh, 0, 0.25, 0.25, 0.25);

        assert_eq!(tab.n_dofs, 4);
        assert_eq!(tab.spatial_dim, 3);
        assert_eq!(tab.values.len(), 12);
        assert_eq!(tab.derivatives.len(), 4);
        assert!(tab.det_j > 0.0);
    }

    #[test]
    fn test_hdiv_tabulate_quadrilateral_rt0() {
        let mesh = crate::mesh::unit_square_quads(1);
        let space = HdivSpace::new();

        let tab = space.tabulate(&mesh, 0, 0.0, 0.0, 0.0);

        assert_eq!(tab.n_dofs, 4);
        assert_eq!(space.num_local_dofs(ElementType::Quadrilateral), tab.n_dofs);
        assert_eq!(tab.spatial_dim, 2);
        assert_eq!(tab.values.len(), 8);
        assert_eq!(tab.derivatives.len(), 4);
        assert!(tab.det_j > 0.0);
        // The physical unit square has area one; each mapped RT0 basis has
        // unit-magnitude integrated divergence. Its sign follows the global
        // facet orientation selected by the topology.
        assert!(
            tab.derivatives
                .iter()
                .all(|div| (div.abs() - 1.0).abs() < 1e-12)
        );
    }

    #[test]
    fn test_hdiv_tabulate_hexahedron_rt0() {
        let mesh = crate::mesh::unit_cube_hexahedra(1);
        let space = HdivSpace::new();

        let tab = space.tabulate(&mesh, 0, 0.0, 0.0, 0.0);

        assert_eq!(tab.n_dofs, 6);
        assert_eq!(space.num_local_dofs(ElementType::Hexahedron), tab.n_dofs);
        assert_eq!(tab.spatial_dim, 3);
        assert_eq!(tab.values.len(), 18);
        assert_eq!(tab.derivatives.len(), 6);
        assert!(tab.det_j > 0.0);
        // The physical unit cube has volume one; each mapped RT0 basis has
        // unit integrated divergence and therefore unit total face flux.
        assert!(tab.derivatives.iter().all(|div| (*div - 1.0).abs() < 1e-12));
    }

    #[test]
    fn checked_tabulation_rejects_singular_and_nonfinite_jacobians() {
        let space = HdivSpace::new();
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
            HdivSpace::new()
                .try_tabulate(&mesh, 0, 0.25, 0.25, 0.25)
                .unwrap_err(),
            TabulationError::Piola(PiolaError::SingularJacobian)
        );
    }
}
impl HdivSpace {
    /// Return global RT1 face-mode offsets for a validated 3-D mesh.
    ///
    /// The offsets account for mixed triangular (three modes) and
    /// quadrilateral (four modes) faces. They are suitable for consumers that
    /// build checked periodic or hanging-face constraints from global face
    /// metadata rather than assuming `mode_count * face_index`.
    pub fn rt1_face_dof_offsets_3d(&self, mesh: &Mesh) -> Option<Vec<usize>> {
        if self.order != HdivOrder::Rt1 || mesh.dimension != 3 || self.validate(mesh).is_err() {
            return None;
        }
        let topology = crate::mesh::topology::MeshTopology::build(mesh);
        let mut offsets = Vec::with_capacity(topology.faces.len() + 1);
        offsets.push(0);
        for face in &topology.faces {
            let next = offsets.last().copied().unwrap_or(0) + face_mode_count(face);
            offsets.push(next);
        }
        Some(offsets)
    }
}

/// Topology-owned global numbering for an RT1 space on a validated 3-D mesh.
///
/// Face modes are stored first, in mesh-topology face order. Cell modes follow
/// in element order. The layout is deliberately explicit so assembly and
/// periodic/MPC consumers do not have to reconstruct numbering with a
/// uniform face-mode assumption (triangular faces have three modes and
/// quadrilateral faces have four).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdivRt1DofLayout3d {
    /// Prefix offsets for face modes; face `i` occupies
    /// `face_offsets[i]..face_offsets[i + 1]`.
    pub face_offsets: Vec<usize>,
    /// Prefix offsets for cell modes; element `i` occupies
    /// `cell_offsets[i]..cell_offsets[i + 1]`.
    pub cell_offsets: Vec<usize>,
    /// Elements incident to each topology face. A valid conforming mesh has
    /// one or two owners per face.
    pub face_owners: Vec<Vec<usize>>,
    /// Checked local-to-global RT1 maps for every element.
    pub element_dofs: Vec<Vec<usize>>,
}

impl HdivRt1DofLayout3d {
    /// Total number of global RT1 DOFs represented by this layout.
    pub fn num_dofs(&self) -> usize {
        self.cell_offsets.last().copied().unwrap_or(0)
    }

    /// Return the global range occupied by a face's modes.
    pub fn face_dofs(&self, face: usize) -> Option<std::ops::Range<usize>> {
        let start = *self.face_offsets.get(face)?;
        let end = *self.face_offsets.get(face + 1)?;
        Some(start..end)
    }

    /// Return the global range occupied by an element's cell modes.
    pub fn cell_dofs(&self, element: usize) -> Option<std::ops::Range<usize>> {
        let start = *self.cell_offsets.get(element)?;
        let end = *self.cell_offsets.get(element + 1)?;
        Some(start..end)
    }
}

impl HdivSpace {
    /// Build checked topology-owned RT1 numbering for a 3-D mesh.
    ///
    /// The method validates that every face is owned by one or two elements,
    /// every element's local map agrees with the face/cell prefix ranges, and
    /// the resulting numbering matches [`FunctionSpace::num_global_dofs`].
    /// `None` is returned for unsupported order/dimension, malformed meshes,
    /// non-manifold ownership, or an inconsistent topology map.
    pub fn rt1_dof_layout_3d(&self, mesh: &Mesh) -> Option<HdivRt1DofLayout3d> {
        if self.order != HdivOrder::Rt1 || mesh.dimension != 3 || self.validate(mesh).is_err() {
            return None;
        }

        let topology = crate::mesh::topology::MeshTopology::build(mesh);
        let face_offsets = face_dof_offsets(&topology);
        let mut cell_offsets = Vec::with_capacity(mesh.elements.len() + 1);
        cell_offsets.push(face_offsets.last().copied().unwrap_or(0));
        for element in &mesh.elements {
            let next =
                cell_offsets.last().copied().unwrap_or(0) + cell_mode_count(element.element_type);
            cell_offsets.push(next);
        }

        let total_dofs = cell_offsets.last().copied().unwrap_or(0);
        if total_dofs != self.num_global_dofs(mesh)
            || face_offsets.len() != topology.faces.len() + 1
        {
            return None;
        }

        let mut face_owners = vec![Vec::new(); topology.faces.len()];
        for element in 0..mesh.elements.len() {
            for &face in topology.element_face_indices(element) {
                let owners = face_owners.get_mut(face)?;
                if !owners.contains(&element) {
                    owners.push(element);
                }
            }
        }
        if face_owners
            .iter()
            .any(|owners| owners.is_empty() || owners.len() > 2)
        {
            return None;
        }

        let mut element_dofs = Vec::with_capacity(mesh.elements.len());
        for element in 0..mesh.elements.len() {
            let dofs = <Self as FunctionSpace>::local_to_global(self, mesh, element);
            if dofs.len() != self.num_local_dofs(mesh.elements[element].element_type)
                || dofs.iter().any(|&dof| dof >= total_dofs)
            {
                return None;
            }
            for (index, &dof) in dofs.iter().enumerate() {
                if dofs[..index].contains(&dof) {
                    return None;
                }
            }

            let mut local_offset = 0;
            for &face in topology.element_face_indices(element) {
                let range = face_offsets[face]..face_offsets[face + 1];
                let count = range.len();
                if dofs.get(local_offset..local_offset + count)?
                    != (range.clone()).collect::<Vec<_>>().as_slice()
                {
                    return None;
                }
                local_offset += count;
            }
            let cell_range = cell_offsets[element]..cell_offsets[element + 1];
            if dofs.get(local_offset..)? != (cell_range.clone()).collect::<Vec<_>>().as_slice() {
                return None;
            }
            element_dofs.push(dofs);
        }

        Some(HdivRt1DofLayout3d {
            face_offsets,
            cell_offsets,
            face_owners,
            element_dofs,
        })
    }
}
