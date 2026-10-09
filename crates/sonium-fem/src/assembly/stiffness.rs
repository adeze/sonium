//! Stiffness matrix assembly
//!
//! Assembles the element stiffness matrix K where K_ij = ∫ ∇φ_i · ∇φ_j dΩ

use crate::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use crate::mesh::{ElementType, Mesh};
use crate::parallelism::{ELEMENTS_PER_THREAD, use_parallel};
use crate::quadrature::{QuadratureRule, for_stiffness};

use rayon::prelude::*;

/// Assembled stiffness matrix in triplet format
#[derive(Debug, Clone)]
pub struct StiffnessMatrix {
    /// Row indices
    pub rows: Vec<usize>,
    /// Column indices
    pub cols: Vec<usize>,
    /// Values (real for standard stiffness)
    pub values: Vec<f64>,
    /// Matrix dimension
    pub dim: usize,
}

impl StiffnessMatrix {
    pub fn new(dim: usize) -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
            dim,
        }
    }

    /// Add a triplet (i, j, value)
    pub fn add(&mut self, i: usize, j: usize, value: f64) {
        self.rows.push(i);
        self.cols.push(j);
        self.values.push(value);
    }

    /// Number of non-zeros
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Compute element stiffness matrix for a 2D triangle (P1/P2)
fn element_stiffness_triangle(
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

    let mut k_local = [0.0; 100];

    for qp in quad.iter() {
        let shape = evaluate_shape(ElementType::Triangle, degree, qp.xi(), qp.eta(), 0.0);
        let jac = Jacobian::from_2d(&shape.gradients, shape.grad_dim, &coords);
        let w_det = jac.det.abs() * qp.weight;

        let mut grads_phys = [[0.0; 2]; 10];
        for (i, g_chunk) in shape
            .gradients
            .chunks(shape.grad_dim)
            .enumerate()
            .take(n_nodes)
        {
            let mut gp = [0.0; 2];
            jac.transform_gradient_into(g_chunk, &mut gp);
            grads_phys[i][0] = gp[0];
            grads_phys[i][1] = gp[1];
        }

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let dot = grads_phys[i][0] * grads_phys[j][0] + grads_phys[i][1] * grads_phys[j][1];
                k_local[i * n_nodes + j] += dot * w_det;
            }
        }
    }

    for i in 0..n_nodes {
        let gi = nodes[i];
        for j in 0..n_nodes {
            let gj = nodes[j];
            let val = k_local[i * n_nodes + j];
            if val.abs() > 1e-15 {
                output.push((gi, gj, val));
            }
        }
    }
}

/// Compute element stiffness matrix for a 2D quadrilateral (Q1/Q2)
fn element_stiffness_quad(
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

    let mut k_local = [0.0; 256];

    for qp in quad.iter() {
        let shape = evaluate_shape(ElementType::Quadrilateral, degree, qp.xi(), qp.eta(), 0.0);
        let jac = Jacobian::from_2d(&shape.gradients, shape.grad_dim, &coords);
        let w_det = jac.det.abs() * qp.weight;

        let mut grads_phys = [[0.0; 2]; 16];
        for (i, g_chunk) in shape
            .gradients
            .chunks(shape.grad_dim)
            .enumerate()
            .take(n_nodes)
        {
            let mut gp = [0.0; 2];
            jac.transform_gradient_into(g_chunk, &mut gp);
            grads_phys[i][0] = gp[0];
            grads_phys[i][1] = gp[1];
        }

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let dot = grads_phys[i][0] * grads_phys[j][0] + grads_phys[i][1] * grads_phys[j][1];
                k_local[i * n_nodes + j] += dot * w_det;
            }
        }
    }

    for i in 0..n_nodes {
        let gi = nodes[i];
        for j in 0..n_nodes {
            let gj = nodes[j];
            let val = k_local[i * n_nodes + j];
            if val.abs() > 1e-15 {
                output.push((gi, gj, val));
            }
        }
    }
}

/// Compute element stiffness matrix for a 3D tetrahedron (P1/P2)
fn element_stiffness_tet(
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

    let mut k_local = [0.0; 400];

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

        let mut grads_phys = [[0.0; 3]; 20];
        for (i, g_chunk) in shape
            .gradients
            .chunks(shape.grad_dim)
            .enumerate()
            .take(n_nodes)
        {
            let mut gp = [0.0; 3];
            jac.transform_gradient_into(g_chunk, &mut gp);
            grads_phys[i][0] = gp[0];
            grads_phys[i][1] = gp[1];
            grads_phys[i][2] = gp[2];
        }

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let dot = grads_phys[i][0] * grads_phys[j][0]
                    + grads_phys[i][1] * grads_phys[j][1]
                    + grads_phys[i][2] * grads_phys[j][2];
                k_local[i * n_nodes + j] += dot * w_det;
            }
        }
    }

    for i in 0..n_nodes {
        let gi = nodes[i];
        for j in 0..n_nodes {
            let gj = nodes[j];
            let val = k_local[i * n_nodes + j];
            if val.abs() > 1e-15 {
                output.push((gi, gj, val));
            }
        }
    }
}

/// Compute element stiffness matrix for a 3D hexahedron (Q1/Q2)
fn element_stiffness_hex(
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

    let mut k_local = [0.0; 4096];

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

        let mut grads_phys = [[0.0; 3]; 64];
        for (i, g_chunk) in shape
            .gradients
            .chunks(shape.grad_dim)
            .enumerate()
            .take(n_nodes)
        {
            let mut gp = [0.0; 3];
            jac.transform_gradient_into(g_chunk, &mut gp);
            grads_phys[i][0] = gp[0];
            grads_phys[i][1] = gp[1];
            grads_phys[i][2] = gp[2];
        }

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let dot = grads_phys[i][0] * grads_phys[j][0]
                    + grads_phys[i][1] * grads_phys[j][1]
                    + grads_phys[i][2] * grads_phys[j][2];
                k_local[i * n_nodes + j] += dot * w_det;
            }
        }
    }

    for i in 0..n_nodes {
        let gi = nodes[i];
        for j in 0..n_nodes {
            let gj = nodes[j];
            let val = k_local[i * n_nodes + j];
            if val.abs() > 1e-15 {
                output.push((gi, gj, val));
            }
        }
    }
}

/// Compute element stiffness contributions (writes triplets into output)
fn compute_element_stiffness(
    mesh: &Mesh,
    elem_idx: usize,
    degree: PolynomialDegree,
    output: &mut Vec<(usize, usize, f64)>,
) {
    let elem_type = mesh.elements[elem_idx].element_type;
    let quad = for_stiffness(elem_type, degree.degree());

    match elem_type {
        ElementType::Triangle => element_stiffness_triangle(mesh, elem_idx, &quad, degree, output),
        ElementType::Quadrilateral => element_stiffness_quad(mesh, elem_idx, &quad, degree, output),
        ElementType::Tetrahedron => element_stiffness_tet(mesh, elem_idx, &quad, degree, output),
        ElementType::Hexahedron => element_stiffness_hex(mesh, elem_idx, &quad, degree, output),
    }
}

/// Assemble global stiffness matrix from mesh
pub fn assemble_stiffness(mesh: &Mesh, degree: PolynomialDegree) -> StiffnessMatrix {
    // The legacy element kernels below use the mesh's node list as the
    // element DOF list.  That is correct for P1, but a P2/P3 H1 space on an
    // affine P1 mesh owns additional edge/face DOFs that are derived from
    // topology rather than stored in `Element::nodes`.  Route those degrees
    // through the generic production assembler so the legacy Helmholtz API
    // has the same global numbering as `H1Space` and cannot index a missing
    // high-order mesh node.
    if degree != PolynomialDegree::P1 {
        use crate::forms::LaplacianForm;
        use crate::spaces::H1Space;

        let space = H1Space::new(degree);
        let triplets = super::generic::assemble_bilinear_checked(
            &LaplacianForm,
            &space,
            &space,
            mesh,
            match degree {
                PolynomialDegree::P2 => 7,
                PolynomialDegree::P3 => 8,
                PolynomialDegree::P1 => unreachable!(),
            },
        )
        .unwrap_or_else(|error| panic!("invalid high-order stiffness assembly: {error}"));

        let mut matrix = StiffnessMatrix::new(triplets.num_rows);
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
                    compute_element_stiffness(mesh, elem_idx, degree, &mut acc);
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
            compute_element_stiffness(mesh, elem_idx, degree, &mut acc);
        }
        acc
    };

    let mut matrix = StiffnessMatrix::new(n_dofs);
    matrix.rows.reserve(all_triplets.len());
    matrix.cols.reserve(all_triplets.len());
    matrix.values.reserve(all_triplets.len());

    for (i, j, v) in all_triplets {
        matrix.add(i, j, v);
    }

    matrix
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_stiffness_assembly_triangle() {
        let mesh = unit_square_triangles(2);
        let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);

        assert_eq!(stiffness.dim, mesh.num_nodes());
        assert!(stiffness.nnz() > 0);
    }

    #[test]
    fn test_stiffness_symmetry() {
        let mesh = unit_square_triangles(2);
        let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);

        // Accumulate into dense matrix for testing
        let n = stiffness.dim;
        let mut dense = vec![vec![0.0; n]; n];

        for k in 0..stiffness.nnz() {
            dense[stiffness.rows[k]][stiffness.cols[k]] += stiffness.values[k];
        }

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i][j] - dense[j][i]).abs();
                assert!(
                    diff < 1e-10,
                    "Asymmetric at ({}, {}): diff = {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }
}
