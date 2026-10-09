//! Turbulence transport assembler helpers.

use std::collections::HashMap;

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::forms::MassForm;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;
use sonium_solvers::CsrMatrix;

use super::{TurbulenceProductionForm, TurbulenceTransportForm};

/// Components:
/// 0 = transport operator, 1 = mass.
pub type TurbulenceAssembler = PrebakedAssembler<2>;

/// Build turbulence transport components without boundary contributions.
pub fn build_turbulence_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    transport_form: &TurbulenceTransportForm,
) -> TurbulenceAssembler {
    build_turbulence_assembler_with_boundaries(mesh, degree, &[], transport_form)
}

/// Build turbulence transport components with optional boundary mass terms.
pub fn build_turbulence_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
    transport_form: &TurbulenceTransportForm,
) -> TurbulenceAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = quadrature::required_order_for_mass(degree.degree());

    let transport =
        crate::assembly::assemble_bilinear(transport_form, &space, &space, mesh, quad_stiffness);
    let mass = crate::assembly::assemble_bilinear(&MassForm, &space, &space, mesh, quad_mass);

    let mut boundary_triplets: Vec<(usize, TripletMatrix<f64>)> = Vec::new();
    for &tag in boundary_tags {
        let b_mass = crate::assembly::assemble_boundary_mass_generic(mesh, tag);
        if b_mass.nnz() > 0 {
            boundary_triplets.push((tag as usize, b_mass));
        }
    }

    PrebakedAssembler::from_triplets([transport, mass], &boundary_triplets)
}

/// Assemble a turbulence production RHS vector.
pub fn assemble_turbulence_production_rhs(
    mesh: &Mesh,
    degree: PolynomialDegree,
    production: f64,
) -> Vec<f64> {
    let space = H1Space::new(degree);
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    crate::assembly::assemble_linear(
        &TurbulenceProductionForm { production },
        &space,
        mesh,
        quad_mass,
    )
}

/// Assembly helpers for turbulence transport matrices.
pub trait TurbulenceAssemblerExt {
    /// Assemble `A = a0 * K_transport + a1 * M`.
    fn assemble_turbulence_operator(
        &self,
        transport_scale: f64,
        mass_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;
}

impl TurbulenceAssemblerExt for TurbulenceAssembler {
    fn assemble_turbulence_operator(
        &self,
        transport_scale: f64,
        mass_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        let mut system = self.assemble_real(&[transport_scale, mass_scale]);
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
    fn build_turbulence_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let form = TurbulenceTransportForm {
            molecular_viscosity: 1.0e-5,
            eddy_viscosity: 1.0e-3,
            sigma: 1.0,
            velocity: [1.0, 0.0, 0.0],
            reaction: 0.1,
        };
        let assembler = build_turbulence_assembler(&mesh, PolynomialDegree::P1, &form);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());

        let rhs = assemble_turbulence_production_rhs(&mesh, PolynomialDegree::P1, 2.0);
        assert_eq!(rhs.len(), mesh.num_nodes());
    }
}
