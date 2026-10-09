//! Monolithic two-field thermoelectric assembly.

use crate::assembly::assemble_bilinear;
use crate::assembly::mixed::MixedSystem;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::{FunctionSpace, H1Space};

use super::{
    ElectricalConductionForm, PeltierPotentialCouplingForm, SeebeckCurrentCouplingForm,
    ThermoelectricHeatForm, ThermoelectricMaterial,
};

/// Assemble the linearized `(V, T)` block operator at `linearization_temperature`.
pub fn assemble_thermoelectric_system(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: ThermoelectricMaterial,
    linearization_temperature: f64,
) -> MixedSystem<f64> {
    let space = H1Space::new(degree);
    let order = quadrature::required_order_for_stiffness(degree.degree());
    let alpha = material.seebeck(linearization_temperature);
    let peltier = material.peltier(linearization_temperature);
    let sigma = material.electrical_conductivity;

    let block_00 = assemble_bilinear(
        &ElectricalConductionForm {
            electrical_conductivity: sigma,
        },
        &space,
        &space,
        mesh,
        order,
    );
    let block_01 = assemble_bilinear(
        &SeebeckCurrentCouplingForm {
            coefficient: sigma * alpha,
        },
        &space,
        &space,
        mesh,
        order,
    );
    let block_10 = assemble_bilinear(
        &PeltierPotentialCouplingForm {
            coefficient: peltier * sigma,
        },
        &space,
        &space,
        mesh,
        order,
    );
    let block_11 = assemble_bilinear(
        &ThermoelectricHeatForm {
            coefficient: material.thermal_conductivity + peltier * sigma * alpha,
        },
        &space,
        &space,
        mesh,
        order,
    );

    MixedSystem {
        block_00,
        block_01,
        block_10,
        block_11,
        n_dofs_0: space.num_global_dofs(mesh),
        n_dofs_1: space.num_global_dofs(mesh),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    fn material(alpha: f64) -> ThermoelectricMaterial {
        ThermoelectricMaterial {
            electrical_conductivity: 1.0e5,
            thermal_conductivity: 1.5,
            seebeck_coefficient: alpha,
            seebeck_temperature_coefficient: 0.0,
            reference_temperature: 300.0,
            density: 7500.0,
            specific_heat: 200.0,
        }
    }

    #[test]
    fn monolithic_system_has_two_fields_and_decouples_at_zero_alpha() {
        let mesh = unit_square_triangles(2);
        let system =
            assemble_thermoelectric_system(&mesh, PolynomialDegree::P1, material(0.0), 300.0);
        assert_eq!(system.n_dofs_0, mesh.num_nodes());
        assert_eq!(system.n_dofs_1, mesh.num_nodes());
        assert!(system.block_01.values.iter().all(|value| *value == 0.0));
        assert!(system.block_10.values.iter().all(|value| *value == 0.0));
        assert_eq!(system.to_monolithic().num_rows, 2 * mesh.num_nodes());
    }

    #[test]
    fn kelvin_relation_makes_peltier_block_temperature_times_seebeck_block() {
        let mesh = unit_square_triangles(2);
        let temperature = 350.0;
        let system = assemble_thermoelectric_system(
            &mesh,
            PolynomialDegree::P1,
            material(200e-6),
            temperature,
        );
        assert_eq!(system.block_01.rows, system.block_10.rows);
        assert_eq!(system.block_01.cols, system.block_10.cols);
        for (seebeck, peltier) in system.block_01.values.iter().zip(&system.block_10.values) {
            assert!((peltier - temperature * seebeck).abs() < 1e-10);
        }
    }
}
