//! Compressible-flow assembler helpers.

use std::collections::HashMap;

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;
use sonium_solvers::CsrMatrix;

use super::{CompressibilityMassForm, CompressibleDiffusionForm, IdealGas};

/// Components:
/// 0 = compressibility mass, 1 = diffusion proxy.
pub type CompressibleFlowAssembler = PrebakedAssembler<2>;

/// Build linearized compressible-flow components using explicit coefficients.
pub fn build_compressible_flow_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    density: f64,
    sound_speed: f64,
    diffusivity: f64,
) -> CompressibleFlowAssembler {
    build_compressible_flow_assembler_with_boundaries(
        mesh,
        degree,
        &[],
        density,
        sound_speed,
        diffusivity,
    )
}

/// Build linearized compressible-flow components using ideal-gas reference state.
pub fn build_linearized_ideal_gas_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    gas: IdealGas,
    reference_pressure: f64,
    reference_density: f64,
    diffusivity: f64,
) -> CompressibleFlowAssembler {
    let c = gas.speed_of_sound(reference_pressure, reference_density);
    build_compressible_flow_assembler(mesh, degree, reference_density, c, diffusivity)
}

/// Build linearized compressible-flow components with boundary mass terms.
pub fn build_compressible_flow_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
    density: f64,
    sound_speed: f64,
    diffusivity: f64,
) -> CompressibleFlowAssembler {
    let space = H1Space::new(degree);
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());

    let compressibility = crate::assembly::assemble_bilinear(
        &CompressibilityMassForm {
            density,
            sound_speed,
        },
        &space,
        &space,
        mesh,
        quad_mass,
    );
    let diffusion = crate::assembly::assemble_bilinear(
        &CompressibleDiffusionForm { diffusivity },
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

    PrebakedAssembler::from_triplets([compressibility, diffusion], &boundary_triplets)
}

/// Assembly helpers for linearized operators.
pub trait CompressibleFlowAssemblerExt {
    /// Assemble `A = a0 * M_c + a1 * K_diff`.
    fn assemble_linearized_operator(
        &self,
        compressibility_scale: f64,
        diffusion_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;
}

impl CompressibleFlowAssemblerExt for CompressibleFlowAssembler {
    fn assemble_linearized_operator(
        &self,
        compressibility_scale: f64,
        diffusion_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        let mut system = self.assemble_real(&[compressibility_scale, diffusion_scale]);
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
    fn build_compressible_flow_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let assembler =
            build_compressible_flow_assembler(&mesh, PolynomialDegree::P1, 1.2, 340.0, 1.0e-5);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());

        let a = assembler.assemble_linearized_operator(1.0, 0.1, &HashMap::new());
        assert_eq!(a.num_rows, mesh.num_nodes());
    }
}
