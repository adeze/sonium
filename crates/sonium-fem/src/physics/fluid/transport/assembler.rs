//! Transport assembler helpers.

use std::collections::HashMap;

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::parallelism::{POINTWISE_VALUES_PER_THREAD, collect_indexed};
use crate::quadrature;
use crate::spaces::H1Space;
use sonium_solvers::CsrMatrix;

use super::TransportMaterial;
use super::{
    TransportAdvectionForm, TransportCapacityForm, TransportDiffusionForm, TransportReactionForm,
    TransportSupgForm,
};

/// Transport pre-baked assembler with five components:
/// 0 = diffusion, 1 = advection, 2 = reaction, 3 = capacity, 4 = SUPG.
pub type TransportAssembler = PrebakedAssembler<5>;

/// Build a scalar transport assembler.
///
/// This uses unitless defaults (`diffusivity = 1`, zero advection/reaction,
/// unit capacity), suitable for tests that only validate topology.
pub fn build_transport_assembler(mesh: &Mesh, degree: PolynomialDegree) -> TransportAssembler {
    build_transport_assembler_with_material(
        mesh,
        degree,
        &TransportMaterial::with_capacity(1.0, [0.0, 0.0, 0.0], 0.0, 1.0),
    )
}

/// Build a transport assembler from explicit transport material data.
pub fn build_transport_assembler_with_material(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: &TransportMaterial,
) -> TransportAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = quadrature::required_order_for_mass(degree.degree());

    let diffusion = crate::assembly::assemble_bilinear(
        &TransportDiffusionForm::new(material),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );

    let advection = crate::assembly::assemble_bilinear(
        &TransportAdvectionForm::new(material),
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let reaction = crate::assembly::assemble_bilinear(
        &TransportReactionForm::new(material),
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let capacity = crate::assembly::assemble_bilinear(
        &TransportCapacityForm::new(material),
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let supg = crate::assembly::assemble_bilinear(
        &TransportSupgForm::new(*material),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );

    PrebakedAssembler::from_triplets([diffusion, advection, reaction, capacity, supg], &[])
}

/// Build a transport assembler with additional boundary mass components.
pub fn build_transport_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
    material: &TransportMaterial,
) -> TransportAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = quadrature::required_order_for_mass(degree.degree());

    let diffusion = crate::assembly::assemble_bilinear(
        &TransportDiffusionForm::new(material),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );

    let advection = crate::assembly::assemble_bilinear(
        &TransportAdvectionForm::new(material),
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let reaction = crate::assembly::assemble_bilinear(
        &TransportReactionForm::new(material),
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let capacity = crate::assembly::assemble_bilinear(
        &TransportCapacityForm::new(material),
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let supg = crate::assembly::assemble_bilinear(
        &TransportSupgForm::new(*material),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );

    let mut boundary_triplets: Vec<(usize, TripletMatrix<f64>)> = Vec::new();
    for &tag in boundary_tags {
        let b_mass = crate::assembly::assemble_boundary_mass_generic(mesh, tag);
        if b_mass.nnz() > 0 {
            boundary_triplets.push((tag as usize, b_mass));
        }
    }

    PrebakedAssembler::from_triplets(
        [diffusion, advection, reaction, capacity, supg],
        &boundary_triplets,
    )
}

/// Build a transport assembler with boundary mass contributions and default material.
pub fn build_transport_assembler_with_boundaries_default(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
) -> TransportAssembler {
    build_transport_assembler_with_boundaries(
        mesh,
        degree,
        boundary_tags,
        &TransportMaterial::with_capacity(1.0, [0.0, 0.0, 0.0], 0.0, 1.0),
    )
}

#[inline]
fn assemble_with_real_boundaries(
    values_len: usize,
    row_ptrs: &[usize],
    col_indices: &[usize],
    components: &[Vec<f64>; 5],
    boundary_components: &std::collections::HashMap<usize, Vec<f64>>,
    coeffs: [f64; 5],
    boundary_coeffs: &HashMap<usize, f64>,
) -> CsrMatrix<f64> {
    let values: Vec<f64> = collect_indexed(values_len, POINTWISE_VALUES_PER_THREAD, |i| {
        let mut value = coeffs
            .iter()
            .zip(components.iter())
            .map(|(coeff, component)| coeff * component[i])
            .sum::<f64>();
        for (tag, boundary_coeff) in boundary_coeffs {
            if let Some(boundary) = boundary_components.get(tag) {
                value += boundary_coeff * boundary[i];
            }
        }
        value
    });

    CsrMatrix::from_raw_parts(
        row_ptrs.len() - 1,
        row_ptrs.len() - 1,
        row_ptrs.to_vec(),
        col_indices.to_vec(),
        values,
    )
}

/// Extension trait for simple transport matrix assembly patterns.
pub trait TransportAssemblerExt {
    /// Assemble
    /// `A = D*K_diff + A_adv*K_adv + R*K_reac + M*K_mass + S*K_supg`.
    fn assemble_transport_operator(
        &self,
        diffusion_scale: f64,
        advection_scale: f64,
        reaction_scale: f64,
        capacity_scale: f64,
        supg_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;
}

impl TransportAssemblerExt for TransportAssembler {
    fn assemble_transport_operator(
        &self,
        diffusion_scale: f64,
        advection_scale: f64,
        reaction_scale: f64,
        capacity_scale: f64,
        supg_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        assemble_with_real_boundaries(
            self.nnz(),
            &self.row_ptrs,
            &self.col_indices,
            &self.components,
            &self.boundary_components,
            [
                diffusion_scale,
                advection_scale,
                reaction_scale,
                capacity_scale,
                supg_scale,
            ],
            boundary_coeffs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_build_transport_assembler_smoke() {
        let mesh = unit_square_triangles(4);
        let assembler = build_transport_assembler(&mesh, PolynomialDegree::P1);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());
    }
}
