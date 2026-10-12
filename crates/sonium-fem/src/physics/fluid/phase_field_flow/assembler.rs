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

/// Assemble a mixed phase/chemical-potential linearization.
///
/// The advection rule integrates polynomial velocities up to the phase-space
/// degree. For higher-degree or nonpolynomial velocity coefficients, callers
/// can assemble [`PhaseAdvectionForm`] with their chosen quadrature rule.
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
    // A same-degree FE velocity gives value * velocity * gradient degree
    // 3p-1. The chemical tangent also includes a mass term of degree 2p.
    let advection_order = (3 * degree.degree() - 1).max(mass_order);
    let tangent_order = stiffness_order.max(mass_order);
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
        advection_order,
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
        tangent_order,
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
    use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};

    #[test]
    fn polynomial_phase_blocks_match_independent_cube_integrals() {
        let mesh = unit_cube_tetrahedra(1);
        let material = PhaseFieldFluid {
            density_minus: 1.3,
            density_plus: 1.3,
            viscosity_minus: 0.3,
            viscosity_plus: 0.3,
            surface_tension: 0.188_561_808_316_412_7,
            interface_thickness: 0.5,
            mobility: 0.2,
        };
        // P=sum(x_d^p), velocity=(x^p,y^p,z^p). Independent unit-cube
        // integrals of P^2, |grad(P)|^2 and P*velocity.grad(P).
        for (degree, p, mass, stiffness, advection) in [
            (PolynomialDegree::P1, 1, 5.0 / 2.0, 3.0, 5.0 / 2.0),
            (PolynomialDegree::P2, 2, 19.0 / 15.0, 4.0, 2.0),
            (PolynomialDegree::P3, 3, 45.0 / 56.0, 27.0 / 5.0, 7.0 / 4.0),
        ] {
            let space = H1Space::new(degree);
            let values: Vec<_> = space
                .dof_coordinates(&mesh)
                .unwrap()
                .iter()
                .map(|point| point.iter().map(|v| v.powi(p)).sum::<f64>())
                .collect();
            let system =
                assemble_linearized_cahn_hilliard(&mesh, degree, material, 0.2, 0.0, |x, y, z| {
                    [x.powi(p), y.powi(p), z.powi(p)]
                });
            let pair = |matrix: &TripletMatrix<f64>| {
                values
                    .iter()
                    .zip(matrix.to_csr().matvec(&values))
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            };
            for (actual, expected) in [
                (pair(&system.block_00), mass / 0.2 + advection),
                (pair(&system.block_01), 0.2 * stiffness),
                (pair(&system.block_10), -0.4 * mass + 0.1 * stiffness),
                (pair(&system.block_11), -mass),
            ] {
                assert!(
                    (actual - expected).abs() < 3e-12 * expected.abs().max(1.0),
                    "{degree:?}: actual={actual:e}, expected={expected:e}"
                );
            }
        }
    }

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
