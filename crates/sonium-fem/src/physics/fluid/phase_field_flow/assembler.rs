//! Linearized mixed Cahn-Hilliard assembly.

use crate::assembly::TripletMatrix;
use crate::assembly::mixed::MixedSystem;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::{FunctionSpace, H1Space};

use super::{
    ChemicalPotentialDiffusionForm, ChemicalPotentialTangentForm, PhaseAdvectionForm,
    PhaseFieldFluid, PhaseMassForm,
};

fn append_scaled(target: &mut TripletMatrix<f64>, source: &TripletMatrix<f64>, scale: f64) {
    for index in 0..source.nnz() {
        target.push(
            source.rows[index],
            source.cols[index],
            scale * source.values[index],
        );
    }
}

pub fn assemble_linearized_cahn_hilliard<U>(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: PhaseFieldFluid,
    time_step: f64,
    linearization_phase: f64,
    velocity: U,
) -> MixedSystem<f64>
where
    U: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    assert!(time_step > 0.0);
    let space = H1Space::new(degree);
    let mass_order = quadrature::required_order_for_mass(degree.degree());
    let stiffness_order = quadrature::required_order_for_stiffness(degree.degree());
    let mass = crate::assembly::assemble_bilinear(
        &PhaseMassForm { coefficient: 1.0 },
        &space,
        &space,
        mesh,
        mass_order,
    );
    let mut block_00 = TripletMatrix::new(mass.num_rows, mass.num_cols);
    append_scaled(&mut block_00, &mass, 1.0 / time_step);
    let advection = crate::assembly::assemble_bilinear(
        &PhaseAdvectionForm { velocity },
        &space,
        &space,
        mesh,
        stiffness_order,
    );
    append_scaled(&mut block_00, &advection, 1.0);

    let block_01 = crate::assembly::assemble_bilinear(
        &ChemicalPotentialDiffusionForm {
            mobility: material.mobility,
        },
        &space,
        &space,
        mesh,
        stiffness_order,
    );
    let beta = material.double_well_coefficient();
    let block_10 = crate::assembly::assemble_bilinear(
        &ChemicalPotentialTangentForm {
            bulk_second_derivative: beta * (3.0 * linearization_phase.powi(2) - 1.0),
            gradient_energy_coefficient: material.gradient_energy_coefficient(),
        },
        &space,
        &space,
        mesh,
        stiffness_order,
    );
    let mut block_11 = TripletMatrix::new(mass.num_rows, mass.num_cols);
    append_scaled(&mut block_11, &mass, -1.0);
    let dofs = space.num_global_dofs(mesh);
    MixedSystem {
        block_00,
        block_01,
        block_10,
        block_11,
        n_dofs_0: dofs,
        n_dofs_1: dofs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn mixed_phase_chemical_system_has_expected_dimensions_and_mass_null_flux() {
        let mesh = unit_square_triangles(2);
        let material = PhaseFieldFluid {
            density_minus: 1000.0,
            density_plus: 1.0,
            viscosity_minus: 1e-3,
            viscosity_plus: 1e-5,
            surface_tension: 0.07,
            interface_thickness: 0.02,
            mobility: 1e-5,
        };
        let system = assemble_linearized_cahn_hilliard(
            &mesh,
            PolynomialDegree::P1,
            material,
            0.01,
            0.0,
            |_, _, _| [0.0; 3],
        );
        assert_eq!(system.n_dofs_0, mesh.num_nodes());
        assert_eq!(system.to_monolithic().num_rows, 2 * mesh.num_nodes());
        let diffusion = system.block_01.to_csr();
        for row in 0..diffusion.num_rows {
            let sum: f64 = diffusion.row_entries(row).map(|(_, value)| value).sum();
            assert!(
                sum.abs() < 1e-16,
                "constant chemical potential has zero phase flux"
            );
        }
    }
}
