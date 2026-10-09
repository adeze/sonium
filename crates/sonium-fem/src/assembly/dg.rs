//! Discontinuous Galerkin assembly with interior facet integrals
//!
//! Combines cell integrals (from [`BilinearForm`]) and facet integrals
//! (from [`FacetBilinearForm`]) into a single global matrix suitable
//! for interior penalty and other DG methods.
//!
//! # Assembly structure
//!
//! ```text
//! A = Σ_K ∫_K (cell integrand) dx   +   Σ_F ∫_F (facet integrand) dS
//! ```
//!
//! The cell integrals are assembled via the standard element loop. The facet
//! integrals are assembled over interior facets, contributing to 2×2 blocks
//! coupling the DOFs of the two adjacent elements.

use crate::forms::{BilinearForm, FacetBilinearForm, LocalMatrix};
use crate::mesh::Mesh;
use crate::spaces::FunctionSpace;
use crate::spaces::dg::{InteriorFacet, build_interior_facets, element_diameter};

use super::triplet::TripletMatrix;

/// Assemble a DG system with cell + interior facet integrals.
///
/// - `cell_form`: [`BilinearForm`] for the cell (volume) integral Σ_K ∫_K ... dx
/// - `facet_form`: [`FacetBilinearForm`] for the interior facet integral Σ_F ∫_F ... dS
/// - `space`: a DG function space (element-local DOFs)
/// - `mesh`: the computational mesh
/// - `quad_order`: quadrature order for cell integrals
///
/// Interior facets use midpoint quadrature (1-point). For higher accuracy
/// on non-affine geometries, increase to multi-point facet quadrature.
pub fn assemble_dg<T, CF, FF, S>(
    cell_form: &CF,
    facet_form: &FF,
    space: &S,
    mesh: &Mesh,
    quad_order: usize,
) -> TripletMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    CF: BilinearForm<T>,
    FF: FacetBilinearForm<T>,
    S: FunctionSpace,
{
    // 1. Cell integrals (standard element loop)
    let mut global = super::assemble_bilinear(cell_form, space, space, mesh, quad_order);

    // 2. Interior facet integrals
    let facets = build_interior_facets(mesh);

    for facet in &facets {
        assemble_single_facet(facet_form, space, mesh, facet, &mut global);
    }

    global
}

/// Assemble the facet integral contribution for a single interior facet.
fn assemble_single_facet<T, FF, S>(
    form: &FF,
    space: &S,
    mesh: &Mesh,
    facet: &InteriorFacet,
    global: &mut TripletMatrix<T>,
) where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    FF: FacetBilinearForm<T>,
    S: FunctionSpace,
{
    let dofs_plus = space.local_to_global(mesh, facet.elem_plus);
    let dofs_minus = space.local_to_global(mesh, facet.elem_minus);
    let n_plus = dofs_plus.len();
    let n_minus = dofs_minus.len();
    let n_max = n_plus.max(n_minus);

    // Compute h (average diameter of adjacent elements)
    let h =
        0.5 * (element_diameter(mesh, facet.elem_plus) + element_diameter(mesh, facet.elem_minus));

    // 2×2 block local matrices
    let mut block = [
        LocalMatrix::zeros(n_max), // (plus, plus)
        LocalMatrix::zeros(n_max), // (plus, minus)
        LocalMatrix::zeros(n_max), // (minus, plus)
        LocalMatrix::zeros(n_max), // (minus, minus)
    ];

    // Midpoint quadrature on the facet: evaluate basis at the facet centroid
    // For edges (2D): midpoint of the edge
    // For faces (3D): centroid of the triangle
    let (xi_p, eta_p, zeta_p, xi_m, eta_m, zeta_m) = facet_centroid_reference_coords(mesh, facet);

    let tab_plus = space.tabulate(mesh, facet.elem_plus, xi_p, eta_p, zeta_p);
    let tab_minus = space.tabulate(mesh, facet.elem_minus, xi_m, eta_m, zeta_m);

    // Single-point facet quadrature: weight = facet area
    let weight = facet.area;

    form.integrate_facet(
        &tab_plus,
        &tab_plus,
        &tab_minus,
        &tab_minus,
        facet.normal,
        h,
        weight,
        &mut block,
    );

    // Scatter blocks to global matrix
    // Block (plus, plus)
    for (i, &gi) in dofs_plus.iter().enumerate() {
        for (j, &gj) in dofs_plus.iter().enumerate() {
            let val = block[0].get(i, j);
            global.push(gi, gj, val);
        }
    }
    // Block (plus, minus)
    for (i, &gi) in dofs_plus.iter().enumerate() {
        for (j, &gj) in dofs_minus.iter().enumerate() {
            let val = block[1].get(i, j);
            global.push(gi, gj, val);
        }
    }
    // Block (minus, plus)
    for (i, &gi) in dofs_minus.iter().enumerate() {
        for (j, &gj) in dofs_plus.iter().enumerate() {
            let val = block[2].get(i, j);
            global.push(gi, gj, val);
        }
    }
    // Block (minus, minus)
    for (i, &gi) in dofs_minus.iter().enumerate() {
        for (j, &gj) in dofs_minus.iter().enumerate() {
            let val = block[3].get(i, j);
            global.push(gi, gj, val);
        }
    }
}

/// Compute reference-domain coordinates of the facet centroid for both elements.
///
/// For a 2D mesh with triangular elements, the facet is an edge. Given the
/// shared edge nodes, compute the reference coordinates of the edge midpoint
/// in both the plus and minus elements.
fn facet_centroid_reference_coords(
    mesh: &Mesh,
    facet: &InteriorFacet,
) -> (f64, f64, f64, f64, f64, f64) {
    // Physical centroid of the facet
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    let n = facet.shared_nodes.len() as f64;
    for &node in &facet.shared_nodes {
        let p = &mesh.nodes[node];
        cx += p.x / n;
        cy += p.y / n;
        cz += p.z / n;
    }

    // Map physical centroid to reference coords in each element
    let (xi_p, eta_p, zeta_p) = physical_to_reference(mesh, facet.elem_plus, cx, cy, cz);
    let (xi_m, eta_m, zeta_m) = physical_to_reference(mesh, facet.elem_minus, cx, cy, cz);

    (xi_p, eta_p, zeta_p, xi_m, eta_m, zeta_m)
}

/// Map a physical point to reference coordinates in an element.
///
/// Uses the inverse of the affine P1 map for simplices.
fn physical_to_reference(
    mesh: &Mesh,
    elem_idx: usize,
    px: f64,
    py: f64,
    pz: f64,
) -> (f64, f64, f64) {
    let elem = &mesh.elements[elem_idx];
    let verts = elem.vertices();

    match elem.element_type {
        crate::mesh::ElementType::Triangle => {
            let p0 = &mesh.nodes[verts[0]];
            let p1 = &mesh.nodes[verts[1]];
            let p2 = &mesh.nodes[verts[2]];

            // Solve: [p1-p0, p2-p0] [xi, eta]^T = p - p0
            let a = p1.x - p0.x;
            let b = p2.x - p0.x;
            let c = p1.y - p0.y;
            let d = p2.y - p0.y;
            let det = a * d - b * c;

            if det.abs() < 1e-30 {
                return (1.0 / 3.0, 1.0 / 3.0, 0.0); // degenerate, return centroid
            }

            let rx = px - p0.x;
            let ry = py - p0.y;
            let xi = (d * rx - b * ry) / det;
            let eta = (-c * rx + a * ry) / det;
            (xi, eta, 0.0)
        }
        crate::mesh::ElementType::Tetrahedron => {
            let p0 = &mesh.nodes[verts[0]];
            let p1 = &mesh.nodes[verts[1]];
            let p2 = &mesh.nodes[verts[2]];
            let p3 = &mesh.nodes[verts[3]];

            let t = [
                [p1.x - p0.x, p2.x - p0.x, p3.x - p0.x],
                [p1.y - p0.y, p2.y - p0.y, p3.y - p0.y],
                [p1.z - p0.z, p2.z - p0.z, p3.z - p0.z],
            ];
            let rhs = [px - p0.x, py - p0.y, pz - p0.z];

            let det = t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1])
                - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0])
                + t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0]);

            if det.abs() < 1e-30 {
                return (0.25, 0.25, 0.25);
            }

            let inv = 1.0 / det;
            let xi = inv
                * (rhs[0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1])
                    - t[0][1] * (rhs[1] * t[2][2] - t[1][2] * rhs[2])
                    + t[0][2] * (rhs[1] * t[2][1] - t[1][1] * rhs[2]));
            let eta = inv
                * (t[0][0] * (rhs[1] * t[2][2] - t[1][2] * rhs[2])
                    - rhs[0] * (t[1][0] * t[2][2] - t[1][2] * t[2][0])
                    + t[0][2] * (t[1][0] * rhs[2] - rhs[1] * t[2][0]));
            let zeta = inv
                * (t[0][0] * (t[1][1] * rhs[2] - rhs[1] * t[2][1])
                    - t[0][1] * (t[1][0] * rhs[2] - rhs[1] * t[2][0])
                    + rhs[0] * (t[1][0] * t[2][1] - t[1][1] * t[2][0]));
            (xi, eta, zeta)
        }
        _ => (0.5, 0.5, 0.0), // fallback for quads/hexes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms::{LaplacianForm, LocalMatrix};
    use crate::spaces::BasisTabulation;

    /// Trivial facet form: penalty α/h × [[u]][[v]] (no averaging)
    struct PenaltyForm {
        alpha: f64,
    }

    impl FacetBilinearForm<f64> for PenaltyForm {
        fn integrate_facet(
            &self,
            test_plus: &BasisTabulation,
            _trial_plus: &BasisTabulation,
            test_minus: &BasisTabulation,
            _trial_minus: &BasisTabulation,
            _normal: [f64; 3],
            h: f64,
            weight: f64,
            block: &mut [LocalMatrix<f64>; 4],
        ) {
            let penalty = self.alpha / h * weight;
            let np = test_plus.n_dofs;
            let nm = test_minus.n_dofs;

            // [[u]][[v]] = (u⁺ - u⁻)(v⁺ - v⁻) produces 4 blocks:
            // (+,+) += penalty φ⁺_i φ⁺_j
            // (+,-) -= penalty φ⁺_i φ⁻_j
            // (-,+) -= penalty φ⁻_i φ⁺_j
            // (-,-) += penalty φ⁻_i φ⁻_j
            for i in 0..np {
                for j in 0..np {
                    block[0].add(i, j, penalty * test_plus.value(i) * test_plus.value(j));
                }
                for j in 0..nm {
                    block[1].add(i, j, -penalty * test_plus.value(i) * test_minus.value(j));
                }
            }
            for i in 0..nm {
                for j in 0..np {
                    block[2].add(i, j, -penalty * test_minus.value(i) * test_plus.value(j));
                }
                for j in 0..nm {
                    block[3].add(i, j, penalty * test_minus.value(i) * test_minus.value(j));
                }
            }
        }
    }

    #[test]
    fn test_assemble_dg_cell_only() {
        use crate::basis::PolynomialDegree;
        use crate::mesh::unit_square_triangles;
        use crate::spaces::dg::DGSpace;

        let mesh = unit_square_triangles(2);
        let space = DGSpace::new(PolynomialDegree::P1);
        let n_dofs = space.num_global_dofs(&mesh);

        // Zero penalty → only cell integrals
        let result = assemble_dg(
            &LaplacianForm,
            &PenaltyForm { alpha: 0.0 },
            &space,
            &mesh,
            1,
        );
        assert_eq!(result.num_rows, n_dofs);
        assert!(result.nnz() > 0);
    }

    #[test]
    fn test_assemble_dg_with_penalty() {
        use crate::basis::PolynomialDegree;
        use crate::mesh::unit_square_triangles;
        use crate::spaces::dg::DGSpace;

        let mesh = unit_square_triangles(2);
        let space = DGSpace::new(PolynomialDegree::P1);

        let cell_only = assemble_dg(
            &LaplacianForm,
            &PenaltyForm { alpha: 0.0 },
            &space,
            &mesh,
            1,
        );
        let with_pen = assemble_dg(
            &LaplacianForm,
            &PenaltyForm { alpha: 8.0 },
            &space,
            &mesh,
            1,
        );

        let cell_csr = cell_only.to_csr();
        let pen_csr = with_pen.to_csr();

        let cell_active = cell_csr.values.iter().filter(|&&v| v.abs() > 1e-12).count();
        let pen_active = pen_csr.values.iter().filter(|&&v| v.abs() > 1e-12).count();

        // Penalty adds more non-zero entries (off-diagonal coupling between elements)
        assert!(
            pen_active > cell_active,
            "Penalty should add active facet entries: {} vs {}",
            pen_active,
            cell_active
        );
    }

    #[test]
    fn test_physical_to_reference_centroid() {
        use crate::mesh::unit_square_triangles;

        let mesh = unit_square_triangles(2);
        let verts = mesh.elements[0].vertices();
        let p0 = &mesh.nodes[verts[0]];
        let p1 = &mesh.nodes[verts[1]];
        let p2 = &mesh.nodes[verts[2]];

        // Physical centroid
        let cx = (p0.x + p1.x + p2.x) / 3.0;
        let cy = (p0.y + p1.y + p2.y) / 3.0;

        let (xi, eta, _) = physical_to_reference(&mesh, 0, cx, cy, 0.0);

        // Reference centroid of a triangle is (1/3, 1/3)
        assert!((xi - 1.0 / 3.0).abs() < 1e-10, "xi = {xi}");
        assert!((eta - 1.0 / 3.0).abs() < 1e-10, "eta = {eta}");
    }
}
