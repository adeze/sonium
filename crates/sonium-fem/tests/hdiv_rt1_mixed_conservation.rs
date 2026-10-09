//! Independent RT1 H(div)-scalar mixed-conservation reference.
//!
//! The mixed divergence block represents
//! `-integral p div(v) dOmega`.  A coefficient vector of ones in the
//! continuous P1/Q1 scalar space is therefore an independently integrated
//! negative divergence for every globally oriented RT1 face/edge and cell
//! DOF.  This exercises the higher-order local-to-global map, orientation
//! signs, Piola divergence, and mixed assembly together.

use sonium_fem::assembly::assemble_mixed_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{HdivDivergenceForm, MassForm};
use sonium_fem::mesh::{
    Mesh, unit_cube_hexahedra, unit_cube_tetrahedra, unit_square_quads, unit_square_triangles,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, HdivOrder, HdivSpace};

fn max_conservation_error(mesh: &Mesh) -> f64 {
    let hdiv = HdivSpace::with_order(HdivOrder::Rt1);
    let scalar = H1Space::new(PolynomialDegree::P1);
    let mixed = assemble_mixed_checked(&MassForm, &HdivDivergenceForm, &hdiv, &scalar, mesh, 6)
        .expect("RT1/P1 or RT1/Q1 mixed assembly should validate");

    let mut expected = vec![0.0; hdiv.num_global_dofs(mesh)];
    for element_idx in 0..mesh.num_elements() {
        let element_type = mesh.elements[element_idx].element_type;
        let quadrature = QuadratureRule::new(element_type, 6);
        let local_to_global = hdiv.local_to_global(mesh, element_idx);
        for point in quadrature.iter() {
            let tabulation =
                hdiv.tabulate(mesh, element_idx, point.xi(), point.eta(), point.zeta());
            for (local, &global) in local_to_global.iter().enumerate() {
                expected[global] -= point.weight * tabulation.det_j * tabulation.divergence(local);
            }
        }
    }

    // The scalar coefficient vector is one at every global DOF.  Summing
    // each mixed row therefore applies the block to the constant field.
    let mut actual = vec![0.0; expected.len()];
    for ((&row, &_column), &value) in mixed
        .block_01
        .rows
        .iter()
        .zip(&mixed.block_01.cols)
        .zip(&mixed.block_01.values)
    {
        actual[row] += value;
    }

    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

pub fn rt1_mixed_conservation_reference() {
    for (label, mesh) in [
        ("triangles", unit_square_triangles(2)),
        ("quadrilaterals", unit_square_quads(2)),
        ("tetrahedra", unit_cube_tetrahedra(1)),
        ("hexahedra", unit_cube_hexahedra(1)),
    ] {
        let error = max_conservation_error(&mesh);
        assert!(
            error < 2.0e-11,
            "RT1/P1 mixed conservation error for {label}: {error:.3e}"
        );
    }
}

#[test]
fn rt1_mixed_conservation_survives_triangle_quad_tetrahedron_and_hexahedron_refinement() {
    rt1_mixed_conservation_reference();
}
