//! Electrochemistry assembler helpers.

use std::collections::HashMap;

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;
use sonium_solvers::CsrMatrix;

use super::{ElectroMigrationForm, IonicChargeSourceForm, IonicSpecies, NernstPlanckDiffusionForm};

/// Components:
/// 0 = Nernst-Planck diffusion, 1 = electro-migration.
pub type ElectrochemistryAssembler = PrebakedAssembler<2>;

/// Build electrochemistry components without boundary contributions.
pub fn build_electrochemistry_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    species: IonicSpecies,
    reference_concentration: f64,
) -> ElectrochemistryAssembler {
    build_electrochemistry_assembler_with_boundaries(
        mesh,
        degree,
        &[],
        species,
        reference_concentration,
    )
}

/// Build electrochemistry components with optional boundary mass terms.
pub fn build_electrochemistry_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
    species: IonicSpecies,
    reference_concentration: f64,
) -> ElectrochemistryAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());

    let diffusion = crate::assembly::assemble_bilinear(
        &NernstPlanckDiffusionForm {
            diffusivity: species.diffusivity,
        },
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let migration = crate::assembly::assemble_bilinear(
        &ElectroMigrationForm {
            species,
            reference_concentration,
        },
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

    PrebakedAssembler::from_triplets([diffusion, migration], &boundary_triplets)
}

/// Assemble ionic charge RHS vector for a Poisson equation.
pub fn assemble_ionic_charge_rhs(
    mesh: &Mesh,
    degree: PolynomialDegree,
    valence: f64,
    concentration: f64,
) -> Vec<f64> {
    let space = H1Space::new(degree);
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    crate::assembly::assemble_linear(
        &IonicChargeSourceForm {
            valence,
            concentration,
        },
        &space,
        mesh,
        quad_mass,
    )
}

/// Extension methods for electrochemistry matrix assembly.
pub trait ElectrochemistryAssemblerExt {
    /// Assemble `A = a0 * K_diff + a1 * K_mig`.
    fn assemble_pnp_operator(
        &self,
        diffusion_scale: f64,
        migration_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;
}

impl ElectrochemistryAssemblerExt for ElectrochemistryAssembler {
    fn assemble_pnp_operator(
        &self,
        diffusion_scale: f64,
        migration_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        let mut system = self.assemble_real(&[diffusion_scale, migration_scale]);
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
    fn build_electrochemistry_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let species = IonicSpecies::from_diffusivity(1.0e-9, 1.0, 298.15);
        let assembler =
            build_electrochemistry_assembler(&mesh, PolynomialDegree::P1, species, 1000.0);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());

        let rhs = assemble_ionic_charge_rhs(&mesh, PolynomialDegree::P1, 1.0, 1000.0);
        assert_eq!(rhs.len(), mesh.num_nodes());
    }
}
