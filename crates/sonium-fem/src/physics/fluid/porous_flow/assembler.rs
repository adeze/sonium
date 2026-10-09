//! Porous-flow assembler helpers.

use std::collections::HashMap;

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;
use sonium_solvers::CsrMatrix;

use super::{
    DarcyMaterial, DarcyPressureForm, DarcyStorageForm, RichardsFlowForm, RichardsMaterial,
    RichardsStorageForm,
};

/// Porous-flow pre-baked assembler:
/// 0 = Darcy diffusion, 1 = Darcy storage, 2 = Richards diffusion, 3 = Richards storage.
pub type PorousFlowAssembler = PrebakedAssembler<4>;

/// Build porous-flow components without boundary terms.
pub fn build_porous_flow_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    darcy: &DarcyMaterial,
    richards: &RichardsMaterial,
    saturation: f64,
) -> PorousFlowAssembler {
    build_porous_flow_assembler_with_boundaries(mesh, degree, &[], darcy, richards, saturation)
}

/// Build porous-flow components with optional boundary mass contributions.
pub fn build_porous_flow_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
    darcy: &DarcyMaterial,
    richards: &RichardsMaterial,
    saturation: f64,
) -> PorousFlowAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = quadrature::required_order_for_mass(degree.degree());

    let darcy_diff = crate::assembly::assemble_bilinear(
        &DarcyPressureForm::new(darcy),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let darcy_storage = crate::assembly::assemble_bilinear(
        &DarcyStorageForm::new(darcy),
        &space,
        &space,
        mesh,
        quad_mass,
    );
    let richards_diff = crate::assembly::assemble_bilinear(
        &RichardsFlowForm::new(richards, saturation),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let richards_storage = crate::assembly::assemble_bilinear(
        &RichardsStorageForm::new(richards),
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let mut boundary_triplets: Vec<(usize, TripletMatrix<f64>)> = Vec::new();
    for &tag in boundary_tags {
        let b_mass = crate::assembly::assemble_boundary_mass_generic(mesh, tag);
        if b_mass.nnz() > 0 {
            boundary_triplets.push((tag as usize, b_mass));
        }
    }

    PrebakedAssembler::from_triplets(
        [darcy_diff, darcy_storage, richards_diff, richards_storage],
        &boundary_triplets,
    )
}

/// Assembly helpers for porous-flow operators.
pub trait PorousFlowAssemblerExt {
    /// Assemble Darcy operator `K_darcy + theta * M_storage`.
    fn assemble_darcy_operator(
        &self,
        storage_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;

    /// Assemble Richards operator `K_richards + theta * M_richards`.
    fn assemble_richards_operator(
        &self,
        storage_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;

    /// Assemble a general linear combination of all components.
    fn assemble_porous_combination(
        &self,
        coeffs: [f64; 4],
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;
}

impl PorousFlowAssemblerExt for PorousFlowAssembler {
    fn assemble_darcy_operator(
        &self,
        storage_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        self.assemble_porous_combination([1.0, storage_scale, 0.0, 0.0], boundary_coeffs)
    }

    fn assemble_richards_operator(
        &self,
        storage_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        self.assemble_porous_combination([0.0, 0.0, 1.0, storage_scale], boundary_coeffs)
    }

    fn assemble_porous_combination(
        &self,
        coeffs: [f64; 4],
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        let mut system = self.assemble_real(&coeffs);
        for (&tag, &coeff) in boundary_coeffs {
            if let Some(boundary_values) = self.boundary_components.get(&tag) {
                for (a, &b) in system.values.iter_mut().zip(boundary_values.iter()) {
                    *a += coeff * b;
                }
            }
        }
        system
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn build_porous_flow_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let darcy = DarcyMaterial::isotropic(1.0e-12, 1.0e-3, 1.0e-6);
        let richards = RichardsMaterial::new(
            [
                [1.0e-12, 0.0, 0.0],
                [0.0, 1.0e-12, 0.0],
                [0.0, 0.0, 1.0e-12],
            ],
            1.0e-3,
            0.1,
            2.0,
            1.0e-5,
        );

        let assembler =
            build_porous_flow_assembler(&mesh, PolynomialDegree::P1, &darcy, &richards, 0.8);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());

        let a = assembler.assemble_darcy_operator(0.5, &HashMap::new());
        assert_eq!(a.num_rows, mesh.num_nodes());
    }
}
