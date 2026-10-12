//! Multi-angle discrete-ordinates assembly.

use std::f64::consts::PI;

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::{FunctionSpace, H1Space};

use super::{
    AngularQuadrature, RadiativeMaterial, ScatteringCouplingForm, StreamingExtinctionForm,
};

pub struct DiscreteOrdinatesSystem {
    /// Angular block matrix `[outgoing][incoming]`.
    pub blocks: Vec<Vec<TripletMatrix<f64>>>,
    pub spatial_dofs: usize,
}

impl DiscreteOrdinatesSystem {
    pub fn angular_dofs(&self) -> usize {
        self.blocks.len()
    }

    pub fn to_monolithic(&self) -> TripletMatrix<f64> {
        let total = self.angular_dofs() * self.spatial_dofs;
        let mut matrix = TripletMatrix::new(total, total);
        for (outgoing, row) in self.blocks.iter().enumerate() {
            for (incoming, block) in row.iter().enumerate() {
                for index in 0..block.nnz() {
                    matrix.push(
                        outgoing * self.spatial_dofs + block.rows[index],
                        incoming * self.spatial_dofs + block.cols[index],
                        block.values[index],
                    );
                }
            }
        }
        matrix
    }
}

pub fn assemble_discrete_ordinates(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: RadiativeMaterial,
    angular: &AngularQuadrature,
) -> DiscreteOrdinatesSystem {
    angular.validate().expect("valid angular quadrature");
    let space = H1Space::new(degree);
    // The combined streaming/extinction form includes a mass term of degree
    // 2*p, which requires a higher order than the gradient-based stiffness.
    let order = quadrature::required_order_for_stiffness(degree.degree())
        .max(quadrature::required_order_for_mass(degree.degree()));
    let blocks = angular
        .directions
        .iter()
        .enumerate()
        .map(|(outgoing, direction)| {
            angular
                .directions
                .iter()
                .enumerate()
                .map(|(incoming, incoming_direction)| {
                    let phase = material.phase_function(*incoming_direction, *direction);
                    let scattering_coefficient =
                        -material.scattering_coefficient * angular.weights[incoming] * phase
                            / (4.0 * PI);
                    if outgoing == incoming {
                        let mut diagonal = crate::assembly::assemble_bilinear(
                            &StreamingExtinctionForm {
                                direction: *direction,
                                extinction_coefficient: material.extinction_coefficient(),
                            },
                            &space,
                            &space,
                            mesh,
                            order,
                        );
                        let scattering = crate::assembly::assemble_bilinear(
                            &ScatteringCouplingForm {
                                coefficient: scattering_coefficient,
                            },
                            &space,
                            &space,
                            mesh,
                            quadrature::required_order_for_mass(degree.degree()),
                        );
                        for index in 0..scattering.nnz() {
                            diagonal.push(
                                scattering.rows[index],
                                scattering.cols[index],
                                scattering.values[index],
                            );
                        }
                        diagonal
                    } else {
                        crate::assembly::assemble_bilinear(
                            &ScatteringCouplingForm {
                                coefficient: scattering_coefficient,
                            },
                            &space,
                            &space,
                            mesh,
                            quadrature::required_order_for_mass(degree.degree()),
                        )
                    }
                })
                .collect()
        })
        .collect();
    DiscreteOrdinatesSystem {
        blocks,
        spatial_dofs: space.num_global_dofs(mesh),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn three_dimensional_polynomial_transport_has_exact_streaming_and_extinction_energy() {
        let mesh = crate::mesh::unit_cube_tetrahedra(1);
        let material = RadiativeMaterial {
            absorption_coefficient: 2.0,
            scattering_coefficient: 0.0,
            asymmetry_parameter: 0.0,
            refractive_index: 1.0,
            density: 1.0,
            specific_heat: 1.0,
        };
        let angular = AngularQuadrature::s2_3d();
        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let space = H1Space::new(degree);
            let coordinates = space.dof_coordinates(&mesh).expect("H1 coordinates");
            let values: Vec<f64> = coordinates
                .iter()
                .map(|p| match degree {
                    PolynomialDegree::P1 => p.iter().sum(),
                    _ => p.iter().map(|x| x * x).sum(),
                })
                .collect();
            // Unit-cube integrals for phi=x+y+z or phi=x^2+y^2+z^2:
            // integral phi^2 = 5/2 or 19/15, and integral phi*d_x(phi)
            // = 3/2 or 7/6 (the y/z derivatives give the same integral).
            let (mass, streaming) = if degree == PolynomialDegree::P1 {
                (2.5, 1.5)
            } else {
                (19.0 / 15.0, 7.0 / 6.0)
            };
            let system = assemble_discrete_ordinates(&mesh, degree, material, &angular);
            for (a, direction) in angular.directions.iter().enumerate() {
                let action = system.blocks[a][a].to_csr().matvec(&values);
                let actual: f64 = values.iter().zip(action).map(|(u, au)| u * au).sum();
                let expected = 2.0 * mass + streaming * direction.iter().sum::<f64>();
                assert!(
                    (actual - expected).abs() < 1e-11,
                    "{degree:?} angle {a}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn s2_system_has_eight_angular_blocks_and_expected_dimensions() {
        let mesh = unit_square_triangles(1);
        let material = RadiativeMaterial {
            absorption_coefficient: 1.0,
            scattering_coefficient: 2.0,
            asymmetry_parameter: 0.0,
            refractive_index: 1.0,
            density: 1.0,
            specific_heat: 1.0,
        };
        let system = assemble_discrete_ordinates(
            &mesh,
            PolynomialDegree::P1,
            material,
            &AngularQuadrature::s2_3d(),
        );
        assert_eq!(system.angular_dofs(), 8);
        assert_eq!(system.to_monolithic().num_rows, 8 * mesh.num_nodes());
    }

    #[test]
    fn isotropic_scattering_conserves_angular_energy_for_constant_intensity() {
        use crate::forms::MassForm;
        use crate::spaces::H1Space;

        let mesh = unit_square_triangles(1);
        let material = RadiativeMaterial {
            absorption_coefficient: 1.0,
            scattering_coefficient: 2.0,
            asymmetry_parameter: 0.0,
            refractive_index: 1.0,
            density: 1.0,
            specific_heat: 1.0,
        };
        let angular = AngularQuadrature::s2_3d();
        let system = assemble_discrete_ordinates(&mesh, PolynomialDegree::P1, material, &angular);
        let space = H1Space::new(PolynomialDegree::P1);
        let absorption_mass =
            crate::assembly::assemble_bilinear(&MassForm, &space, &space, &mesh, 2).to_csr();
        for outgoing in 0..system.angular_dofs() {
            for row in 0..system.spatial_dofs {
                let angular_row_sum = system.blocks[outgoing]
                    .iter()
                    .map(|block| {
                        block
                            .to_csr()
                            .row_entries(row)
                            .map(|(_, value)| value)
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                let expected = absorption_mass
                    .row_entries(row)
                    .map(|(_, value)| value)
                    .sum::<f64>();
                assert!((angular_row_sum - expected).abs() < 1e-12);
            }
        }
    }
}
