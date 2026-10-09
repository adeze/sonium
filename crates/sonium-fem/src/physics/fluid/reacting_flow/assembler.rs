//! Species transport and reaction-Jacobian assembly.

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;

use super::{ReactionJacobianForm, ReactionMechanism, SpeciesDiffusionForm};

pub struct ReactingFlowOperators {
    pub species_diffusion: Vec<TripletMatrix<f64>>,
    /// Dense species-block layout, row-major `[equation][unknown]`.
    pub reaction_jacobian: Vec<Vec<TripletMatrix<f64>>>,
}

pub fn assemble_reacting_flow_operators(
    mesh: &Mesh,
    degree: PolynomialDegree,
    density: f64,
    mechanism: &ReactionMechanism,
    concentrations: &[f64],
    temperature: f64,
) -> ReactingFlowOperators {
    assert_eq!(mechanism.species.len(), concentrations.len());
    let space = H1Space::new(degree);
    let stiffness_order = quadrature::required_order_for_stiffness(degree.degree());
    let mass_order = quadrature::required_order_for_mass(degree.degree());
    let species_diffusion = mechanism
        .species
        .iter()
        .map(|species| {
            crate::assembly::assemble_bilinear(
                &SpeciesDiffusionForm {
                    density_diffusivity: density * species.diffusivity,
                },
                &space,
                &space,
                mesh,
                stiffness_order,
            )
        })
        .collect();
    let reaction_jacobian = (0..mechanism.species.len())
        .map(|equation| {
            (0..mechanism.species.len())
                .map(|unknown| {
                    let derivative = mechanism
                        .reactions
                        .iter()
                        .map(|reaction| {
                            reaction.stoichiometry[equation]
                                * reaction.progress_rate_derivative(
                                    unknown,
                                    concentrations,
                                    temperature,
                                )
                        })
                        .sum();
                    crate::assembly::assemble_bilinear(
                        &ReactionJacobianForm {
                            source_derivative: derivative,
                        },
                        &space,
                        &space,
                        mesh,
                        mass_order,
                    )
                })
                .collect()
        })
        .collect();
    ReactingFlowOperators {
        species_diffusion,
        reaction_jacobian,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;
    use crate::physics::fluid::reacting_flow::{ArrheniusReaction, ChemicalSpecies};

    #[test]
    fn mechanism_builds_full_nonlinear_species_jacobian() {
        let mechanism = ReactionMechanism {
            species: ["F", "O", "P"]
                .map(|name| ChemicalSpecies {
                    name: name.into(),
                    molar_mass: 0.02,
                    diffusivity: 1e-5,
                    specific_heat: 1000.0,
                })
                .to_vec(),
            reactions: vec![ArrheniusReaction {
                stoichiometry: vec![-1.0, -1.0, 1.0],
                orders: vec![1.0, 1.0, 0.0],
                pre_exponential_factor: 1e3,
                activation_energy: 10_000.0,
                heat_release: 1e6,
            }],
        };
        let mesh = unit_square_triangles(1);
        let operators = assemble_reacting_flow_operators(
            &mesh,
            PolynomialDegree::P1,
            1.0,
            &mechanism,
            &[1.0, 2.0, 0.0],
            1000.0,
        );
        assert_eq!(operators.species_diffusion.len(), 3);
        assert_eq!(operators.reaction_jacobian.len(), 3);
        assert!(
            operators.reaction_jacobian[0][0]
                .values
                .iter()
                .any(|value| *value != 0.0)
        );
    }
}
