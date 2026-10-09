//! Mixed H(div)-scalar conservation contract.
//!
//! The production mixed assembler must preserve the RT0 divergence theorem:
//! applying the assembled `-∫ div(v) p` block to the coefficient vector of
//! the constant scalar field must equal the independently integrated RT0
//! divergence for every globally oriented face DOF.

use sonium_fem::assembly::assemble_mixed_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{HdivDivergenceForm, MassForm};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, HdivSpace};

fn max_conservation_error(refinement: usize) -> f64 {
    let mesh = unit_square_triangles(refinement);
    let hdiv = HdivSpace::new();
    let scalar = H1Space::new(PolynomialDegree::P1);
    let mixed = assemble_mixed_checked(&MassForm, &HdivDivergenceForm, &hdiv, &scalar, &mesh, 4)
        .expect("RT0/P1 mixed assembly should validate");

    let mut independently_integrated = vec![0.0; hdiv.num_global_dofs(&mesh)];
    for element_idx in 0..mesh.num_elements() {
        let element = mesh.elements[element_idx].element_type;
        let quadrature = QuadratureRule::new(element, 4);
        let local_to_global = hdiv.local_to_global(&mesh, element_idx);
        for point in quadrature.iter() {
            let tabulation =
                hdiv.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            for (local, &global) in local_to_global.iter().enumerate() {
                independently_integrated[global] -=
                    point.weight * tabulation.det_j * tabulation.divergence(local);
            }
        }
    }

    // P1 is a partition of unity, so the row sum of the assembled coupling
    // is exactly the same integral when the scalar coefficients are all one.
    let mut assembled_constant = vec![0.0; independently_integrated.len()];
    for ((&row, &column), &value) in mixed
        .block_01
        .rows
        .iter()
        .zip(&mixed.block_01.cols)
        .zip(&mixed.block_01.values)
    {
        assert!(column < scalar.num_global_dofs(&mesh));
        assembled_constant[row] += value;
    }

    assembled_constant
        .iter()
        .zip(independently_integrated)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

#[test]
fn rt0_scalar_coupling_preserves_divergence_conservation() {
    for refinement in [1, 2, 4] {
        let error = max_conservation_error(refinement);
        assert!(
            error < 2.0e-12,
            "mixed RT0/P1 conservation error at refinement {refinement}: {error:e}"
        );
    }
}
