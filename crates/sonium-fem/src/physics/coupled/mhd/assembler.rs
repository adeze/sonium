//! Assembly of the velocity/electric-potential MHD coupling blocks.

use crate::assembly::TripletMatrix;
use crate::assembly::mixed::assemble_bilinear_two_spaces;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

use super::{
    LorentzDampingForm, LorentzPotentialCouplingForm, MhdMaterial, MhdPotentialForm,
    MotionalEmfCouplingForm,
};

pub struct MhdElectromagneticBlocks {
    pub velocity_damping: TripletMatrix<f64>,
    pub velocity_from_potential: TripletMatrix<f64>,
    pub potential_from_velocity: TripletMatrix<f64>,
    pub potential_conduction: TripletMatrix<f64>,
    pub velocity_dofs: usize,
    pub potential_dofs: usize,
}

impl MhdElectromagneticBlocks {
    pub fn to_monolithic(&self) -> TripletMatrix<f64> {
        let total = self.velocity_dofs + self.potential_dofs;
        let mut matrix = TripletMatrix::new(total, total);
        for index in 0..self.velocity_damping.nnz() {
            matrix.push(
                self.velocity_damping.rows[index],
                self.velocity_damping.cols[index],
                self.velocity_damping.values[index],
            );
        }
        for index in 0..self.velocity_from_potential.nnz() {
            matrix.push(
                self.velocity_from_potential.rows[index],
                self.velocity_from_potential.cols[index] + self.velocity_dofs,
                self.velocity_from_potential.values[index],
            );
        }
        for index in 0..self.potential_from_velocity.nnz() {
            matrix.push(
                self.potential_from_velocity.rows[index] + self.velocity_dofs,
                self.potential_from_velocity.cols[index],
                self.potential_from_velocity.values[index],
            );
        }
        for index in 0..self.potential_conduction.nnz() {
            matrix.push(
                self.potential_conduction.rows[index] + self.velocity_dofs,
                self.potential_conduction.cols[index] + self.velocity_dofs,
                self.potential_conduction.values[index],
            );
        }
        matrix
    }
}

pub fn assemble_mhd_electromagnetic_blocks(
    mesh: &Mesh,
    degree: PolynomialDegree,
    velocity_components: usize,
    material: MhdMaterial,
) -> MhdElectromagneticBlocks {
    assert!((1..=3).contains(&velocity_components));
    let velocity_space = VectorH1Space::new(degree, velocity_components);
    let potential_space = H1Space::new(degree);
    // Magnetic damping contains value*value (degree 2p), and both
    // cross-space EMF/Lorentz terms contain value*gradient (degree 2p-1).
    // A stiffness-only rule (degree 2p-2) underintegrates these blocks.
    let order = quadrature::required_order_for_stiffness(degree.degree())
        .max(quadrature::required_order_for_mass(degree.degree()));
    let sigma = material.electrical_conductivity;
    let field = material.imposed_magnetic_flux_density;

    let velocity_damping = crate::assembly::assemble_bilinear(
        &LorentzDampingForm {
            electrical_conductivity: sigma,
            magnetic_flux_density: field,
        },
        &velocity_space,
        &velocity_space,
        mesh,
        order,
    );
    let velocity_from_potential = assemble_bilinear_two_spaces(
        &LorentzPotentialCouplingForm {
            electrical_conductivity: sigma,
            magnetic_flux_density: field,
        },
        &velocity_space,
        &potential_space,
        mesh,
        order,
    );
    let potential_from_velocity = assemble_bilinear_two_spaces(
        &MotionalEmfCouplingForm {
            electrical_conductivity: sigma,
            magnetic_flux_density: field,
        },
        &potential_space,
        &velocity_space,
        mesh,
        order,
    );
    let potential_conduction = crate::assembly::assemble_bilinear(
        &MhdPotentialForm::from(material),
        &potential_space,
        &potential_space,
        mesh,
        order,
    );

    MhdElectromagneticBlocks {
        velocity_damping,
        velocity_from_potential,
        potential_from_velocity,
        potential_conduction,
        velocity_dofs: velocity_space.num_global_dofs(mesh),
        potential_dofs: potential_space.num_global_dofs(mesh),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};

    #[test]
    fn polynomial_mhd_blocks_match_independent_cube_integrals() {
        let mesh = unit_cube_tetrahedra(1);
        let material = MhdMaterial {
            density: 1.3,
            dynamic_viscosity: 0.3,
            electrical_conductivity: 2.0,
            imposed_magnetic_flux_density: [0.4, 0.7, 1.1],
        };
        // u=(P,0,0), phi=P, P=sum(x_d^p). These constants integrate
        // P^2, P*d_y(P), and |grad(P)|^2 over the unit cube independently.
        for (degree, p, mass, mixed, stiffness) in [
            (PolynomialDegree::P2, 2, 19.0 / 15.0, 7.0 / 6.0, 4.0),
            (PolynomialDegree::P3, 3, 45.0 / 56.0, 1.0, 27.0 / 5.0),
        ] {
            let velocity = VectorH1Space::new(degree, 3);
            let potential = H1Space::new(degree);
            let polynomial = |point: [f64; 3]| point.iter().map(|v| v.powi(p)).sum::<f64>();
            let u: Vec<_> = velocity
                .dof_coordinates(&mesh)
                .unwrap()
                .iter()
                .enumerate()
                .map(|(d, point)| if d % 3 == 0 { polynomial(*point) } else { 0.0 })
                .collect();
            let phi: Vec<_> = potential
                .dof_coordinates(&mesh)
                .unwrap()
                .iter()
                .map(|point| polynomial(*point))
                .collect();
            let blocks = assemble_mhd_electromagnetic_blocks(&mesh, degree, 3, material);
            let pair = |left: &[f64], matrix: &TripletMatrix<f64>, right: &[f64]| {
                left.iter()
                    .zip(matrix.to_csr().matvec(right))
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            };
            let sigma = 2.0;
            for (actual, expected) in [
                (
                    pair(&u, &blocks.velocity_damping, &u),
                    sigma * (0.7_f64.powi(2) + 1.1_f64.powi(2)) * mass,
                ),
                (
                    pair(&u, &blocks.velocity_from_potential, &phi),
                    sigma * (1.1 - 0.7) * mixed,
                ),
                (
                    pair(&phi, &blocks.potential_from_velocity, &u),
                    sigma * (1.1 - 0.7) * mixed,
                ),
                (
                    pair(&phi, &blocks.potential_conduction, &phi),
                    sigma * stiffness,
                ),
            ] {
                assert!(
                    (actual - expected).abs() < 2e-12 * expected.abs().max(1.0),
                    "{degree:?}: actual={actual:e}, expected={expected:e}"
                );
            }
        }
    }

    #[test]
    fn mhd_blocks_have_vector_scalar_dimensions() {
        let mesh = unit_square_triangles(2);
        let material = MhdMaterial {
            density: 1000.0,
            dynamic_viscosity: 1e-3,
            electrical_conductivity: 1e6,
            imposed_magnetic_flux_density: [0.0, 1.0, 0.0],
        };
        let blocks = assemble_mhd_electromagnetic_blocks(&mesh, PolynomialDegree::P1, 2, material);
        assert_eq!(blocks.velocity_dofs, 2 * mesh.num_nodes());
        assert_eq!(blocks.potential_dofs, mesh.num_nodes());
        assert_eq!(blocks.to_monolithic().num_rows, 3 * mesh.num_nodes());
    }
}
