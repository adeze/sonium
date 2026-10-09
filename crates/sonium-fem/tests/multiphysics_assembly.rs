//! Public assembly smoke and conservation tests for the ten added domains.

use sonium_fem::{
    assembly::region::RegionMap,
    basis::PolynomialDegree,
    mesh::unit_square_triangles,
    physics::{
        coupled::battery::{BatteryMaterial, BatteryRegionKind, assemble_battery_macro_operators},
        coupled::conjugate_heat_transfer::{
            ConjugateHeatTransferAssemblerExt, ConjugateRegion,
            build_conjugate_heat_transfer_assembler,
        },
        coupled::electrothermal::{ElectrothermalMaterial, assemble_dc_current_operator},
        coupled::fsi::{
            FsiInterfaceDofPair, assemble_fsi_interface_constraint, transfer_interface_tractions,
        },
        coupled::mhd::{MhdMaterial, assemble_mhd_electromagnetic_blocks},
        fluid::lubrication::{Lubricant, assemble_reynolds_pressure},
        fluid::phase_field_flow::{PhaseFieldFluid, assemble_linearized_cahn_hilliard},
        fluid::reacting_flow::{
            ArrheniusReaction, ChemicalSpecies, ReactionMechanism, assemble_reacting_flow_operators,
        },
        thermal::radiation_transport::{
            AngularQuadrature, RadiativeMaterial, assemble_discrete_ordinates,
        },
        thermal::thermoelectric::{ThermoelectricMaterial, assemble_thermoelectric_system},
    },
};

fn assert_constant_is_diffusion_null(matrix: &sonium_solvers::CsrMatrix<f64>, tolerance: f64) {
    for row in 0..matrix.num_rows {
        let row_sum: f64 = matrix.row_entries(row).map(|(_, value)| value).sum();
        assert!(row_sum.abs() <= tolerance, "row {row} sum {row_sum}");
    }
}

#[test]
fn lubrication_public_assembler_builds_conservative_reynolds_operator() {
    let mesh = unit_square_triangles(2);
    let lubricant = Lubricant {
        dynamic_viscosity: 0.1,
        density: 850.0,
        ambient_pressure: 0.0,
        minimum_film_thickness: 1e-9,
    };
    let operator =
        assemble_reynolds_pressure(&mesh, PolynomialDegree::P1, lubricant, |_, _, _| 1e-3).to_csr();
    assert_constant_is_diffusion_null(&operator, 1e-20);
}

#[test]
fn thermoelectric_public_assembler_builds_kelvin_consistent_blocks() {
    let mesh = unit_square_triangles(2);
    let temperature = 350.0;
    let system = assemble_thermoelectric_system(
        &mesh,
        PolynomialDegree::P1,
        ThermoelectricMaterial {
            electrical_conductivity: 1e5,
            thermal_conductivity: 1.5,
            seebeck_coefficient: 200e-6,
            seebeck_temperature_coefficient: 0.0,
            reference_temperature: 300.0,
            density: 7500.0,
            specific_heat: 200.0,
        },
        temperature,
    );
    assert_eq!(system.to_monolithic().num_rows, 2 * mesh.num_nodes());
    for (seebeck, peltier) in system.block_01.values.iter().zip(&system.block_10.values) {
        assert!((peltier - temperature * seebeck).abs() < 1e-10);
    }
}

#[test]
fn conjugate_heat_transfer_public_assembler_routes_fluid_and_solid_regions() {
    let mut mesh = unit_square_triangles(1);
    mesh.elements[0].marker = 1;
    mesh.elements[1].marker = 2;
    let mut regions = RegionMap::new();
    regions.insert(1, ConjugateRegion::solid(10.0, 1000.0, 500.0));
    regions.insert(2, ConjugateRegion::fluid(1.0, 1.0, 1000.0, [2.0, 0.0, 0.0]));
    let assembler = build_conjugate_heat_transfer_assembler(&mesh, PolynomialDegree::P1, &regions)
        .expect("both CHT regions mapped");
    let conduction = assembler.assemble_cht_operator(0.0, 0.0);
    assert_constant_is_diffusion_null(&conduction, 1e-12);
}

#[test]
fn electrothermal_public_assembler_builds_temperature_dependent_charge_operator() {
    let mesh = unit_square_triangles(2);
    let material = ElectrothermalMaterial {
        reference_electrical_conductivity: 1e6,
        resistivity_temperature_coefficient: 0.004,
        reference_temperature: 293.15,
        relative_permeability: 1.0,
        thermal_conductivity: 100.0,
        density: 8000.0,
        specific_heat: 400.0,
    };
    let cold = assemble_dc_current_operator(&mesh, PolynomialDegree::P1, material, |_, _, _| 300.0)
        .to_csr();
    let hot = assemble_dc_current_operator(&mesh, PolynomialDegree::P1, material, |_, _, _| 400.0)
        .to_csr();
    assert_constant_is_diffusion_null(&cold, 1e-8);
    let cold_norm = cold.values.iter().map(|value| value.abs()).sum::<f64>();
    let hot_norm = hot.values.iter().map(|value| value.abs()).sum::<f64>();
    assert!(hot_norm < cold_norm);
}

#[test]
fn battery_public_assembler_builds_dfn_macro_operators() {
    let mesh = unit_square_triangles(2);
    let material = BatteryMaterial {
        region: BatteryRegionKind::NegativeElectrode,
        porosity: 0.3,
        bruggeman_exponent: 1.5,
        electrolyte_diffusivity: 1e-10,
        electrolyte_conductivity: 1.0,
        solid_conductivity: 100.0,
        transference_number: 0.4,
        particle_radius: 5e-6,
        solid_diffusivity: 1e-14,
        maximum_solid_concentration: 30_000.0,
        specific_surface_area: 3e5,
        exchange_current_density: 2.0,
        charge_transfer_coefficient: 0.5,
        density: 2000.0,
        specific_heat: 1000.0,
        thermal_conductivity: 2.0,
    };
    let operators = assemble_battery_macro_operators(&mesh, PolynomialDegree::P1, material);
    assert_constant_is_diffusion_null(&operators.electrolyte_diffusion.to_csr(), 1e-20);
    assert_constant_is_diffusion_null(&operators.electrolyte_conduction.to_csr(), 1e-12);
    assert_constant_is_diffusion_null(&operators.solid_conduction.to_csr(), 1e-10);
    assert!(operators.electrolyte_capacity.nnz() > 0);
}

#[test]
fn mhd_public_assembler_builds_velocity_potential_coupling() {
    let mesh = unit_square_triangles(2);
    let blocks = assemble_mhd_electromagnetic_blocks(
        &mesh,
        PolynomialDegree::P1,
        2,
        MhdMaterial {
            density: 1000.0,
            dynamic_viscosity: 1e-3,
            electrical_conductivity: 1e6,
            imposed_magnetic_flux_density: [0.0, 1.0, 0.0],
        },
    );
    assert_eq!(blocks.velocity_dofs, 2 * mesh.num_nodes());
    assert_eq!(blocks.potential_dofs, mesh.num_nodes());
    assert_eq!(blocks.to_monolithic().num_rows, 3 * mesh.num_nodes());
    assert!(blocks.velocity_damping.nnz() > 0);
    assert!(blocks.velocity_from_potential.nnz() > 0);
    assert!(blocks.potential_from_velocity.nnz() > 0);
}

#[test]
fn fsi_public_constraint_has_exact_action_reaction_transfer() {
    let pairs = [
        FsiInterfaceDofPair {
            fluid_dof: 0,
            solid_dof: 1,
            weight: 2.0,
        },
        FsiInterfaceDofPair {
            fluid_dof: 1,
            solid_dof: 0,
            weight: 3.0,
        },
    ];
    let constraint = assemble_fsi_interface_constraint(2, 2, &pairs);
    assert_eq!(constraint.matrix.num_rows, 6);
    let transfer = transfer_interface_tractions(2, 2, &pairs, &[5.0, -7.0]);
    assert_eq!(transfer.net_force, 0.0);
}

#[test]
fn phase_field_public_assembler_builds_mixed_cahn_hilliard_system() {
    let mesh = unit_square_triangles(2);
    let system = assemble_linearized_cahn_hilliard(
        &mesh,
        PolynomialDegree::P1,
        PhaseFieldFluid {
            density_minus: 1000.0,
            density_plus: 1.0,
            viscosity_minus: 1e-3,
            viscosity_plus: 1e-5,
            surface_tension: 0.07,
            interface_thickness: 0.02,
            mobility: 1e-5,
        },
        0.01,
        0.0,
        |_, _, _| [0.0; 3],
    );
    assert_eq!(system.to_monolithic().num_rows, 2 * mesh.num_nodes());
    assert_constant_is_diffusion_null(&system.block_01.to_csr(), 1e-16);
}

#[test]
fn reacting_flow_public_assembler_builds_full_species_jacobian() {
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
    assert!(operators.reaction_jacobian.iter().all(|row| row.len() == 3));
}

#[test]
fn radiation_public_assembler_builds_coupled_s2_transport_system() {
    let mesh = unit_square_triangles(1);
    let system = assemble_discrete_ordinates(
        &mesh,
        PolynomialDegree::P1,
        RadiativeMaterial {
            absorption_coefficient: 1.0,
            scattering_coefficient: 2.0,
            asymmetry_parameter: 0.0,
            refractive_index: 1.0,
            density: 1.0,
            specific_heat: 1.0,
        },
        &AngularQuadrature::s2_3d(),
    );
    assert_eq!(system.angular_dofs(), 8);
    assert_eq!(system.to_monolithic().num_rows, 8 * mesh.num_nodes());
}
