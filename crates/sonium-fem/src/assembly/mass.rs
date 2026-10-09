//! Mass matrix assembly
//!
//! Assembles the element mass matrix M where M_ij = ∫ φ_i · φ_j dΩ

use crate::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use crate::mesh::{ElementType, Mesh};
use crate::parallelism::{ELEMENTS_PER_THREAD, use_parallel};
use crate::quadrature::{QuadratureRule, for_mass};

use rayon::prelude::*;

/// Assembled mass matrix in triplet format
#[derive(Debug, Clone)]
pub struct MassMatrix {
    /// Row indices
    pub rows: Vec<usize>,
    /// Column indices
    pub cols: Vec<usize>,
    /// Values (real for standard mass)
    pub values: Vec<f64>,
    /// Matrix dimension
    pub dim: usize,
}

impl MassMatrix {
    pub fn new(dim: usize) -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
            dim,
        }
    }

    pub fn add(&mut self, i: usize, j: usize, value: f64) {
        self.rows.push(i);
        self.cols.push(j);
        self.values.push(value);
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Compute element mass matrix for a 2D triangle (P1/P2)
fn element_mass_triangle(
    mesh: &Mesh,
    elem_idx: usize,
    quad: &QuadratureRule,
    degree: PolynomialDegree,
    output: &mut Vec<(usize, usize, f64)>,
) {
    let elem = &mesh.elements[elem_idx];
    let nodes = match degree {
        PolynomialDegree::P1 => &elem.nodes[..3],
        PolynomialDegree::P2 => &elem.nodes[..6],
        PolynomialDegree::P3 => &elem.nodes[..10],
    };
    let n_nodes = nodes.len();

    let coords: Vec<[f64; 2]> = nodes
        .iter()
        .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y])
        .collect();

    let mut m_local = [0.0; 100];

    for qp in quad.iter() {
        let shape = evaluate_shape(ElementType::Triangle, degree, qp.xi(), qp.eta(), 0.0);
        let jac = Jacobian::from_2d(&shape.gradients, shape.grad_dim, &coords);
        let w_det = jac.det.abs() * qp.weight;

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                m_local[i * n_nodes + j] += shape.values[i] * shape.values[j] * w_det;
            }
        }
    }

    for i in 0..n_nodes {
        let gi = nodes[i];
        for j in 0..n_nodes {
            let gj = nodes[j];
            let val = m_local[i * n_nodes + j];
            if val.abs() > 1e-15 {
                output.push((gi, gj, val));
            }
        }
    }
}

/// Compute element mass matrix for a 2D quadrilateral (Q1/Q2)
fn element_mass_quad(
    mesh: &Mesh,
    elem_idx: usize,
    quad: &QuadratureRule,
    degree: PolynomialDegree,
    output: &mut Vec<(usize, usize, f64)>,
) {
    let elem = &mesh.elements[elem_idx];
    let nodes = match degree {
        PolynomialDegree::P1 => &elem.nodes[..4],
        PolynomialDegree::P2 => &elem.nodes[..9],
        PolynomialDegree::P3 => &elem.nodes[..16],
    };
    let n_nodes = nodes.len();

    let coords: Vec<[f64; 2]> = nodes
        .iter()
        .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y])
        .collect();

    let mut m_local = [0.0; 256];

    for qp in quad.iter() {
        let shape = evaluate_shape(ElementType::Quadrilateral, degree, qp.xi(), qp.eta(), 0.0);
        let jac = Jacobian::from_2d(&shape.gradients, shape.grad_dim, &coords);
        let w_det = jac.det.abs() * qp.weight;

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                m_local[i * n_nodes + j] += shape.values[i] * shape.values[j] * w_det;
            }
        }
    }

    for i in 0..n_nodes {
        let gi = nodes[i];
        for j in 0..n_nodes {
            let gj = nodes[j];
            let val = m_local[i * n_nodes + j];
            if val.abs() > 1e-15 {
                output.push((gi, gj, val));
            }
        }
    }
}

/// Compute element mass matrix for a 3D tetrahedron (P1/P2)
fn element_mass_tet(
    mesh: &Mesh,
    elem_idx: usize,
    quad: &QuadratureRule,
    degree: PolynomialDegree,
    output: &mut Vec<(usize, usize, f64)>,
) {
    let elem = &mesh.elements[elem_idx];
    let nodes = match degree {
        PolynomialDegree::P1 => &elem.nodes[..4],
        PolynomialDegree::P2 => &elem.nodes[..10],
        PolynomialDegree::P3 => &elem.nodes[..20],
    };
    let n_nodes = nodes.len();

    let coords: Vec<[f64; 3]> = nodes
        .iter()
        .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y, mesh.nodes[v].z])
        .collect();

    let mut m_local = [0.0; 400];

    for qp in quad.iter() {
        let shape = evaluate_shape(
            ElementType::Tetrahedron,
            degree,
            qp.xi(),
            qp.eta(),
            qp.zeta(),
        );
        let jac = Jacobian::from_3d(&shape.gradients, shape.grad_dim, &coords);
        let w_det = jac.det.abs() * qp.weight;

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                m_local[i * n_nodes + j] += shape.values[i] * shape.values[j] * w_det;
            }
        }
    }

    for i in 0..n_nodes {
        let gi = nodes[i];
        for j in 0..n_nodes {
            let gj = nodes[j];
            let val = m_local[i * n_nodes + j];
            if val.abs() > 1e-15 {
                output.push((gi, gj, val));
            }
        }
    }
}

/// Compute element mass matrix for a 3D hexahedron (Q1/Q2)
fn element_mass_hex(
    mesh: &Mesh,
    elem_idx: usize,
    quad: &QuadratureRule,
    degree: PolynomialDegree,
    output: &mut Vec<(usize, usize, f64)>,
) {
    let elem = &mesh.elements[elem_idx];
    let nodes = match degree {
        PolynomialDegree::P1 => &elem.nodes[..8],
        PolynomialDegree::P2 => &elem.nodes[..27],
        PolynomialDegree::P3 => &elem.nodes[..64],
    };
    let n_nodes = nodes.len();

    let coords: Vec<[f64; 3]> = nodes
        .iter()
        .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y, mesh.nodes[v].z])
        .collect();

    let mut m_local = [0.0; 4096];

    for qp in quad.iter() {
        let shape = evaluate_shape(
            ElementType::Hexahedron,
            degree,
            qp.xi(),
            qp.eta(),
            qp.zeta(),
        );
        let jac = Jacobian::from_3d(&shape.gradients, shape.grad_dim, &coords);
        let w_det = jac.det.abs() * qp.weight;

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                m_local[i * n_nodes + j] += shape.values[i] * shape.values[j] * w_det;
            }
        }
    }

    for i in 0..n_nodes {
        let gi = nodes[i];
        for j in 0..n_nodes {
            let gj = nodes[j];
            let val = m_local[i * n_nodes + j];
            if val.abs() > 1e-15 {
                output.push((gi, gj, val));
            }
        }
    }
}

/// Compute element mass contributions (writes triplets into output)
fn compute_element_mass(
    mesh: &Mesh,
    elem_idx: usize,
    degree: PolynomialDegree,
    output: &mut Vec<(usize, usize, f64)>,
) {
    let elem_type = mesh.elements[elem_idx].element_type;
    let quad = for_mass(elem_type, degree.degree());

    match elem_type {
        ElementType::Triangle => element_mass_triangle(mesh, elem_idx, &quad, degree, output),
        ElementType::Quadrilateral => element_mass_quad(mesh, elem_idx, &quad, degree, output),
        ElementType::Tetrahedron => element_mass_tet(mesh, elem_idx, &quad, degree, output),
        ElementType::Hexahedron => element_mass_hex(mesh, elem_idx, &quad, degree, output),
    }
}

/// Assemble global mass matrix from mesh
pub fn assemble_mass(mesh: &Mesh, degree: PolynomialDegree) -> MassMatrix {
    // Match the H1 topology-based numbering for high-order spaces on affine
    // P1 meshes.  The legacy kernels use `Element::nodes` directly and thus
    // cannot represent the additional P2/P3 DOFs unless a high-order mesh
    // connectivity is supplied; the generic assembler handles both the
    // topology and the affine geometry contract explicitly.
    if degree != PolynomialDegree::P1 {
        use crate::forms::MassForm;
        use crate::spaces::H1Space;

        let space = H1Space::new(degree);
        let triplets = super::generic::assemble_bilinear_checked(
            &MassForm,
            &space,
            &space,
            mesh,
            match degree {
                PolynomialDegree::P2 => 7,
                PolynomialDegree::P3 => 8,
                PolynomialDegree::P1 => unreachable!(),
            },
        )
        .unwrap_or_else(|error| panic!("invalid high-order mass assembly: {error}"));

        let mut matrix = MassMatrix::new(triplets.num_rows);
        matrix.rows.reserve(triplets.rows.len());
        matrix.cols.reserve(triplets.cols.len());
        matrix.values.reserve(triplets.values.len());
        for ((&row, &column), &value) in triplets
            .rows
            .iter()
            .zip(&triplets.cols)
            .zip(&triplets.values)
        {
            matrix.add(row, column, value);
        }
        return matrix;
    }

    let n_dofs = mesh.num_nodes(); // Assuming P1
    let n_elems = mesh.num_elements();

    // Fold per-element triplets (rayon above the machine-adaptive gate,
    // serial for small meshes).
    let all_triplets: Vec<(usize, usize, f64)> = if use_parallel(n_elems, ELEMENTS_PER_THREAD) {
        (0..n_elems)
            .into_par_iter()
            .fold(
                || Vec::with_capacity(1024),
                |mut acc, elem_idx| {
                    compute_element_mass(mesh, elem_idx, degree, &mut acc);
                    acc
                },
            )
            .reduce(Vec::new, |mut a, mut b| {
                if a.capacity() < b.capacity() {
                    b.append(&mut a);
                    b
                } else {
                    a.append(&mut b);
                    a
                }
            })
    } else {
        let mut acc = Vec::with_capacity(1024);
        for elem_idx in 0..n_elems {
            compute_element_mass(mesh, elem_idx, degree, &mut acc);
        }
        acc
    };

    let mut matrix = MassMatrix::new(n_dofs);
    matrix.rows.reserve(all_triplets.len());
    matrix.cols.reserve(all_triplets.len());
    matrix.values.reserve(all_triplets.len());

    for (i, j, v) in all_triplets {
        matrix.add(i, j, v);
    }

    matrix
}

/// Compute lumped mass matrix (row-sum lumping)
pub fn assemble_lumped_mass(mesh: &Mesh, degree: PolynomialDegree) -> Vec<f64> {
    let mass = assemble_mass(mesh, degree);
    let mut lumped = vec![0.0; mass.dim];

    for k in 0..mass.nnz() {
        lumped[mass.rows[k]] += mass.values[k];
    }

    lumped
}

/// Assemble boundary mass matrix for a specific boundary marker
///
/// Assembles M_gamma = ∫_Γ φ_i φ_j dΓ for boundaries with the given marker
pub fn assemble_boundary_mass(mesh: &Mesh, _degree: PolynomialDegree, marker: i32) -> MassMatrix {
    let n_dofs = mesh.num_nodes();
    let mut matrix = MassMatrix::new(n_dofs);

    for boundary in &mesh.boundaries {
        if boundary.marker != marker {
            continue;
        }

        let nodes = &boundary.nodes;

        if nodes.len() == 3 {
            // Triangle face
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let p2 = &mesh.nodes[nodes[2]];

            // Compute area
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];

            // Cross product
            let cx = v1[1] * v2[2] - v1[2] * v2[1];
            let cy = v1[2] * v2[0] - v1[0] * v2[2];
            let cz = v1[0] * v2[1] - v1[1] * v2[0];

            let area = 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();

            // P1 triangle mass matrix
            // M_loc = (Area / 12) * [2 1 1; 1 2 1; 1 1 2]
            let factor = area / 12.0;

            for i in 0..3 {
                for j in 0..3 {
                    let val = if i == j { 2.0 * factor } else { factor };
                    matrix.add(nodes[i], nodes[j], val);
                }
            }
        } else if nodes.len() == 2 {
            // Edge (2D mesh)
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];

            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;
            let dz = p1.z - p0.z;
            let len = (dx * dx + dy * dy + dz * dz).sqrt();

            // P1 edge mass matrix
            // M_loc = (Length / 6) * [2 1; 1 2]
            let factor = len / 6.0;

            for i in 0..2 {
                for j in 0..2 {
                    let val = if i == j { 2.0 * factor } else { factor };
                    matrix.add(nodes[i], nodes[j], val);
                }
            }
        } else if nodes.len() == 4 {
            // Quad face - split into two triangles
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let p2 = &mesh.nodes[nodes[2]];
            let p3 = &mesh.nodes[nodes[3]];

            // Triangle 1: 0-1-2
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let cx1 = v1[1] * v2[2] - v1[2] * v2[1];
            let cy1 = v1[2] * v2[0] - v1[0] * v2[2];
            let cz1 = v1[0] * v2[1] - v1[1] * v2[0];
            let area1 = 0.5 * (cx1 * cx1 + cy1 * cy1 + cz1 * cz1).sqrt();
            let factor1 = area1 / 12.0;

            let t1 = [nodes[0], nodes[1], nodes[2]];
            for i in 0..3 {
                for j in 0..3 {
                    let val = if i == j { 2.0 * factor1 } else { factor1 };
                    matrix.add(t1[i], t1[j], val);
                }
            }

            // Triangle 2: 0-2-3
            let v3 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let v4 = [p3.x - p0.x, p3.y - p0.y, p3.z - p0.z];
            let cx2 = v3[1] * v4[2] - v3[2] * v4[1];
            let cy2 = v3[2] * v4[0] - v3[0] * v4[2];
            let cz2 = v3[0] * v4[1] - v3[1] * v4[0];
            let area2 = 0.5 * (cx2 * cx2 + cy2 * cy2 + cz2 * cz2).sqrt();
            let factor2 = area2 / 12.0;

            let t2 = [nodes[0], nodes[2], nodes[3]];
            for i in 0..3 {
                for j in 0..3 {
                    let val = if i == j { 2.0 * factor2 } else { factor2 };
                    matrix.add(t2[i], t2[j], val);
                }
            }
        }
    }

    matrix
}

/// Assemble a P1 face mass matrix over an explicit set of mesh faces.
///
/// Unlike [`assemble_boundary_mass`], this does not require the faces to be
/// stored in `mesh.boundaries`. It is used for artificial subdomain interfaces
/// where the transmission condition lives on interior faces.
pub fn assemble_face_mass(mesh: &Mesh, faces: &[Vec<usize>]) -> MassMatrix {
    let mut matrix = MassMatrix::new(mesh.num_nodes());

    for nodes in faces {
        add_face_mass(mesh, nodes, &mut matrix);
    }

    matrix
}

/// Assemble the P1 surface Laplacian matrix over explicit mesh faces.
///
/// This supplies the tangential-derivative term used by the HABC
/// approximation on artificial Schwarz interfaces.
pub fn assemble_face_surface_stiffness(mesh: &Mesh, faces: &[Vec<usize>]) -> MassMatrix {
    let mut matrix = MassMatrix::new(mesh.num_nodes());

    for nodes in faces {
        add_face_surface_stiffness(mesh, nodes, &mut matrix);
    }

    matrix
}

fn add_face_mass(mesh: &Mesh, nodes: &[usize], matrix: &mut MassMatrix) {
    match nodes.len() {
        2 => add_edge_mass(mesh, [nodes[0], nodes[1]], matrix),
        3 => add_triangle_face_mass(mesh, [nodes[0], nodes[1], nodes[2]], matrix),
        4 => {
            add_triangle_face_mass(mesh, [nodes[0], nodes[1], nodes[2]], matrix);
            add_triangle_face_mass(mesh, [nodes[0], nodes[2], nodes[3]], matrix);
        }
        _ => {}
    }
}

fn add_edge_mass(mesh: &Mesh, nodes: [usize; 2], matrix: &mut MassMatrix) {
    let p0 = &mesh.nodes[nodes[0]];
    let p1 = &mesh.nodes[nodes[1]];

    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let dz = p1.z - p0.z;
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    let factor = len / 6.0;

    for i in 0..2 {
        for j in 0..2 {
            let val = if i == j { 2.0 * factor } else { factor };
            matrix.add(nodes[i], nodes[j], val);
        }
    }
}

fn add_triangle_face_mass(mesh: &Mesh, nodes: [usize; 3], matrix: &mut MassMatrix) {
    let area = triangle_area(mesh, nodes);
    let factor = area / 12.0;

    for i in 0..3 {
        for j in 0..3 {
            let val = if i == j { 2.0 * factor } else { factor };
            matrix.add(nodes[i], nodes[j], val);
        }
    }
}

fn add_face_surface_stiffness(mesh: &Mesh, nodes: &[usize], matrix: &mut MassMatrix) {
    match nodes.len() {
        2 => add_edge_surface_stiffness(mesh, [nodes[0], nodes[1]], matrix),
        3 => add_triangle_surface_stiffness(mesh, [nodes[0], nodes[1], nodes[2]], matrix),
        4 => {
            add_triangle_surface_stiffness(mesh, [nodes[0], nodes[1], nodes[2]], matrix);
            add_triangle_surface_stiffness(mesh, [nodes[0], nodes[2], nodes[3]], matrix);
        }
        _ => {}
    }
}

fn add_edge_surface_stiffness(mesh: &Mesh, nodes: [usize; 2], matrix: &mut MassMatrix) {
    let p0 = &mesh.nodes[nodes[0]];
    let p1 = &mesh.nodes[nodes[1]];

    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let dz = p1.z - p0.z;
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    if len <= 0.0 {
        return;
    }

    let factor = 1.0 / len;
    matrix.add(nodes[0], nodes[0], factor);
    matrix.add(nodes[0], nodes[1], -factor);
    matrix.add(nodes[1], nodes[0], -factor);
    matrix.add(nodes[1], nodes[1], factor);
}

fn add_triangle_surface_stiffness(mesh: &Mesh, nodes: [usize; 3], matrix: &mut MassMatrix) {
    let p0 = &mesh.nodes[nodes[0]];
    let p1 = &mesh.nodes[nodes[1]];
    let p2 = &mesh.nodes[nodes[2]];

    let e01 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let e02 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
    let normal = cross(e01, e02);
    let normal_len = dot(normal, normal).sqrt();
    if normal_len <= 0.0 {
        return;
    }

    let area = 0.5 * normal_len;
    let unit_normal = [
        normal[0] / normal_len,
        normal[1] / normal_len,
        normal[2] / normal_len,
    ];

    let p10 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let p21 = [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z];
    let p02 = [p0.x - p2.x, p0.y - p2.y, p0.z - p2.z];

    let gradients = [
        scale(cross(unit_normal, p21), 1.0 / (2.0 * area)),
        scale(cross(unit_normal, p02), 1.0 / (2.0 * area)),
        scale(cross(unit_normal, p10), 1.0 / (2.0 * area)),
    ];

    for i in 0..3 {
        for j in 0..3 {
            matrix.add(nodes[i], nodes[j], area * dot(gradients[i], gradients[j]));
        }
    }
}

fn triangle_area(mesh: &Mesh, nodes: [usize; 3]) -> f64 {
    let p0 = &mesh.nodes[nodes[0]];
    let p1 = &mesh.nodes[nodes[1]];
    let p2 = &mesh.nodes[nodes[2]];

    let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
    let c = cross(v1, v2);
    0.5 * dot(c, c).sqrt()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Assemble an r-weighted boundary mass matrix for axisymmetric problems.
///
/// For a 2D mesh in the (r, z) meridional plane (x = r, y = z), the
/// axisymmetric surface integral over a boundary edge is:
///
/// ```text
/// M_ij = ∫ r(s) φ_i(s) φ_j(s) ds
/// ```
///
/// where `r(s) = r₀(1-s) + r₁s` is the linear interpolation of the radial
/// coordinate along the edge.
///
/// Exact integration for P1 elements gives:
/// - `M_00 = |e|/12 * (3r₀ + r₁)`
/// - `M_01 = M_10 = |e|/12 * (r₀ + r₁)`
/// - `M_11 = |e|/12 * (r₀ + 3r₁)`
///
/// This is needed to compute acoustic power through duct cross-sections
/// in axisymmetric simulations (power ∝ ∫ r |p|² dr).
pub fn assemble_axi_boundary_mass(
    mesh: &Mesh,
    _degree: PolynomialDegree,
    marker: i32,
) -> MassMatrix {
    let n_dofs = mesh.num_nodes();
    let mut matrix = MassMatrix::new(n_dofs);

    for boundary in &mesh.boundaries {
        if boundary.marker != marker {
            continue;
        }

        let nodes = &boundary.nodes;

        if nodes.len() == 2 {
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];

            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;
            let len = (dx * dx + dy * dy).sqrt();

            // Radial coordinates (x = r in the meridional plane)
            let r0 = p0.x;
            let r1 = p1.x;

            // Exact r-weighted P1 edge mass matrix entries
            let f = len / 12.0;
            matrix.add(nodes[0], nodes[0], f * (3.0 * r0 + r1));
            matrix.add(nodes[0], nodes[1], f * (r0 + r1));
            matrix.add(nodes[1], nodes[0], f * (r0 + r1));
            matrix.add(nodes[1], nodes[1], f * (r0 + 3.0 * r1));
        }
        // Triangle and quad faces are not expected in 2D axisymmetric meshes
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, Mesh, Point, unit_square_triangles};

    #[test]
    fn test_mass_assembly_triangle() {
        let mesh = unit_square_triangles(2);
        let mass = assemble_mass(&mesh, PolynomialDegree::P1);

        assert_eq!(mass.dim, mesh.num_nodes());
        assert!(mass.nnz() > 0);
    }

    #[test]
    fn test_mass_positivity() {
        let mesh = unit_square_triangles(2);
        let mass = assemble_mass(&mesh, PolynomialDegree::P1);

        // All diagonal entries should be positive
        let n = mass.dim;
        let mut diag = vec![0.0; n];

        for k in 0..mass.nnz() {
            if mass.rows[k] == mass.cols[k] {
                diag[mass.rows[k]] += mass.values[k];
            }
        }

        for (i, &d) in diag.iter().enumerate() {
            assert!(d > 0.0, "Diagonal {} is not positive: {}", i, d);
        }
    }

    #[test]
    fn test_mass_integrates_to_area() {
        // For unit square, sum of all mass entries should equal 1 (area)
        let mesh = unit_square_triangles(4);
        let mass = assemble_mass(&mesh, PolynomialDegree::P1);

        let total: f64 = mass.values.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Total mass {} should be 1.0",
            total
        );
    }

    #[test]
    fn test_lumped_mass() {
        let mesh = unit_square_triangles(4);
        let lumped = assemble_lumped_mass(&mesh, PolynomialDegree::P1);

        assert_eq!(lumped.len(), mesh.num_nodes());

        // Sum should still be area
        let total: f64 = lumped.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Total lumped mass {} should be 1.0",
            total
        );
    }

    #[test]
    fn test_boundary_mass() {
        let mesh = unit_square_triangles(2);
        // Unit square: boundaries are length 4
        // assemble_boundary_mass assumes some markers are set.
        // We need to set them first? mesh.detect_boundaries sets everything to 0.
        // So assembling marker 0 should get full perimeter.

        let mut mesh = mesh;
        mesh.detect_boundaries();

        let b_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, 0);

        // Sum of boundary mass should be perimeter length (4.0)
        let total: f64 = b_mass.values.iter().sum();
        assert!((total - 4.0).abs() < 1e-10, "Boundary mass sum: {}", total);
    }

    #[test]
    fn test_explicit_face_mass_integrates_to_area() {
        let mut mesh = Mesh::new(3);
        let n0 = mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        let n1 = mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        let n2 = mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        let n3 = mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tetrahedron, vec![n0, n1, n2, n3]);

        let mass = assemble_face_mass(&mesh, &[vec![n0, n1, n2]]);
        let total: f64 = mass.values.iter().sum();
        assert!((total - 0.5).abs() < 1e-12, "Face mass sum: {total}");
    }

    #[test]
    fn test_explicit_face_surface_stiffness_has_zero_row_sum() {
        let mut mesh = Mesh::new(3);
        let n0 = mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        let n1 = mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        let n2 = mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        let n3 = mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tetrahedron, vec![n0, n1, n2, n3]);

        let stiffness = assemble_face_surface_stiffness(&mesh, &[vec![n0, n1, n2]]);
        let mut row_sums = vec![0.0; mesh.num_nodes()];
        for idx in 0..stiffness.nnz() {
            row_sums[stiffness.rows[idx]] += stiffness.values[idx];
        }

        assert!(row_sums[..3].iter().all(|sum| sum.abs() < 1e-12));
    }
}
