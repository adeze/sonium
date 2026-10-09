use std::collections::HashMap;

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::electromagnetics::maxwell::EmMaterial;
use sonium_fem::physics::solid::elasticity::{IsotropicMaterial, PlaneAssumption};
use sonium_fem::physics::thermal::heat_conduction::ThermalMaterial;
use sonium_fem::physics::{
    coupled::electrochemistry, coupled::thermoelasticity, electromagnetics::semiconductor,
    fluid::compressible_flow, fluid::porous_flow, fluid::shallow_water, fluid::transport,
    fluid::turbulence, solid::composites, solid::eigenmodes, solid::fracture,
};

fn assert_square_nonempty(label: &str, rows: usize, cols: usize, nnz: usize) {
    assert!(rows > 0, "{label} assembled zero rows");
    assert_eq!(rows, cols, "{label} assembled a non-square operator");
    assert!(nnz > 0, "{label} assembled an empty sparsity pattern");
}

fn test_porous_flow_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let darcy = porous_flow::DarcyMaterial::isotropic(1.0e-12, 1.0e-3, 1.0e-6);
    let richards = porous_flow::RichardsMaterial::new(
        [
            [1.0e-12, 0.0, 0.0],
            [0.0, 1.0e-12, 0.0],
            [0.0, 0.0, 1.0e-12],
        ],
        1.0e-3,
        0.1,
        2.0,
        1.0e-5,
    );
    let assembler = porous_flow::build_porous_flow_assembler(
        &mesh,
        PolynomialDegree::P1,
        &darcy,
        &richards,
        0.8,
    );
    let matrix = porous_flow::PorousFlowAssemblerExt::assemble_darcy_operator(
        &assembler,
        0.2,
        &HashMap::new(),
    );
    assert_square_nonempty(
        "porous-flow",
        matrix.num_rows,
        matrix.num_cols,
        matrix.nnz(),
    );
}

fn test_transport_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let material = transport::TransportMaterial::with_capacity(0.02, [0.5, 0.0, 0.0], 0.1, 1.2);
    let assembler =
        transport::build_transport_assembler_with_material(&mesh, PolynomialDegree::P1, &material);
    let matrix = transport::TransportAssemblerExt::assemble_transport_operator(
        &assembler,
        1.0,
        1.0,
        0.5,
        0.25,
        0.1,
        &HashMap::new(),
    );
    assert_square_nonempty("transport", matrix.num_rows, matrix.num_cols, matrix.nnz());
}

fn test_thermoelastic_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let assembler =
        thermoelasticity::build_thermoelastic_assembler(&mesh, PolynomialDegree::P1, 0.75);
    let damping =
        thermoelasticity::ThermoelasticAssemblerExt::assemble_damping_operator(&assembler, 1.0);
    assert_square_nonempty(
        "thermoelastic damping",
        damping.num_rows,
        damping.num_cols,
        damping.nnz(),
    );

    let blocks = thermoelasticity::build_thermoelastic_coupled_blocks(
        &mesh,
        PolynomialDegree::P1,
        2,
        IsotropicMaterial::new(210.0e9, 0.3),
        Some(PlaneAssumption::PlaneStress),
        ThermalMaterial::new(45.0, 7800.0, 500.0),
        0.5,
    );
    assert!(blocks.mechanical_stiffness.nnz() > 0);
    assert!(blocks.heat_conduction.nnz() > 0);
    assert!(blocks.heat_capacity.nnz() > 0);
    assert!(blocks.damping_coupling.nnz() > 0);
}

fn test_composites_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let laminate = composites::Laminate {
        plies: vec![
            composites::OrthotropicPly {
                e1: 140.0,
                e2: 10.0,
                g12: 5.0,
                nu12: 0.25,
                thickness: 0.001,
                angle_rad: 0.0,
            },
            composites::OrthotropicPly {
                e1: 140.0,
                e2: 10.0,
                g12: 5.0,
                nu12: 0.25,
                thickness: 0.001,
                angle_rad: std::f64::consts::FRAC_PI_2,
            },
        ],
    };
    let assembler = composites::build_composites_assembler(&mesh, PolynomialDegree::P1, &laminate);
    let matrix = composites::CompositesAssemblerExt::assemble_membrane_operator(
        &assembler,
        1.0,
        &HashMap::new(),
    );
    assert_square_nonempty("composites", matrix.num_rows, matrix.num_cols, matrix.nnz());

    let bending =
        composites::build_composite_bending_assembler(&mesh, PolynomialDegree::P1, &laminate);
    assert!(bending.nnz() > 0);
}

fn test_fracture_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let material = fracture::PhaseFieldFractureMaterial {
        critical_energy_release_rate: 2.0,
        length_scale: 0.1,
        residual_stiffness: 1.0e-6,
    };
    let assembler = fracture::build_fracture_assembler(&mesh, PolynomialDegree::P1, material);
    let matrix = fracture::FractureAssemblerExt::assemble_damage_operator(&assembler, 1.0);
    assert_square_nonempty("fracture", matrix.num_rows, matrix.num_cols, matrix.nnz());

    let rhs = fracture::assemble_damage_history_rhs(&mesh, PolynomialDegree::P1, 1.0);
    assert_eq!(rhs.len(), mesh.num_nodes());
}

fn test_turbulence_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let form = turbulence::TurbulenceTransportForm {
        molecular_viscosity: 1.0e-5,
        eddy_viscosity: 1.0e-3,
        sigma: 1.0,
        velocity: [1.0, 0.0, 0.0],
        reaction: 0.1,
    };
    let assembler = turbulence::build_turbulence_assembler(&mesh, PolynomialDegree::P1, &form);
    let matrix = turbulence::TurbulenceAssemblerExt::assemble_turbulence_operator(
        &assembler,
        1.0,
        0.5,
        &HashMap::new(),
    );
    assert_square_nonempty("turbulence", matrix.num_rows, matrix.num_cols, matrix.nnz());
}

fn test_compressible_flow_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let assembler = compressible_flow::build_linearized_ideal_gas_assembler(
        &mesh,
        PolynomialDegree::P1,
        compressible_flow::IdealGas::AIR,
        101_325.0,
        1.225,
        1.0e-5,
    );
    let matrix = compressible_flow::CompressibleFlowAssemblerExt::assemble_linearized_operator(
        &assembler,
        1.0,
        0.1,
        &HashMap::new(),
    );
    assert_square_nonempty(
        "compressible-flow",
        matrix.num_rows,
        matrix.num_cols,
        matrix.nnz(),
    );
}

fn test_electrochemistry_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let species = electrochemistry::IonicSpecies::from_diffusivity(1.0e-9, 1.0, 298.15);
    let assembler = electrochemistry::build_electrochemistry_assembler(
        &mesh,
        PolynomialDegree::P1,
        species,
        1000.0,
    );
    let matrix = electrochemistry::ElectrochemistryAssemblerExt::assemble_pnp_operator(
        &assembler,
        1.0,
        0.1,
        &HashMap::new(),
    );
    assert_square_nonempty(
        "electrochemistry",
        matrix.num_rows,
        matrix.num_cols,
        matrix.nnz(),
    );
}

fn test_semiconductor_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let carrier = semiconductor::CarrierMaterial {
        mobility: 0.14,
        lifetime: 1.0e-6,
        temperature: 300.0,
        kind: semiconductor::CarrierKind::Electron,
    };
    let assembler = semiconductor::build_semiconductor_assembler(
        &mesh,
        PolynomialDegree::P1,
        carrier,
        [1.0, 0.0, 0.0],
    );
    let matrix = semiconductor::SemiconductorAssemblerExt::assemble_carrier_operator(
        &assembler,
        1.0,
        0.25,
        0.1,
        &HashMap::new(),
    );
    assert_square_nonempty(
        "semiconductor",
        matrix.num_rows,
        matrix.num_cols,
        matrix.nnz(),
    );
}

fn test_shallow_water_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let material = shallow_water::ShallowWaterMaterial {
        gravity: 9.81,
        mean_depth: 1.5,
        manning_n: 0.02,
    };
    let assembler =
        shallow_water::build_shallow_water_assembler(&mesh, PolynomialDegree::P1, material, 1.0);
    let matrix = shallow_water::ShallowWaterAssemblerExt::assemble_shallow_water_operator(
        &assembler,
        1.0,
        0.5,
        0.1,
        &HashMap::new(),
    );
    assert_square_nonempty(
        "shallow-water",
        matrix.num_rows,
        matrix.num_cols,
        matrix.nnz(),
    );
}

fn test_eigenphysics_assembly() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 4, 1);
    let structural = eigenmodes::build_structural_modal_problem(
        &mesh,
        IsotropicMaterial::new(1000.0, 0.25),
        &eigenmodes::StructuralModalConfig::plane_stress(2.0, vec![1]),
    );
    assert_square_nonempty(
        "structural modal eigenphysics",
        structural.stiffness_like.num_rows,
        structural.stiffness_like.num_cols,
        structural.stiffness_like.nnz(),
    );

    let acoustic = eigenmodes::build_acoustic_eigenmode_problem(
        &mesh,
        &eigenmodes::AcousticEigenmodeConfig {
            degree: PolynomialDegree::P1,
            sound_speed: 343.0,
            pressure_release_boundary_markers: vec![1, 2],
        },
    );
    assert_square_nonempty(
        "acoustic eigenphysics",
        acoustic.stiffness_like.num_rows,
        acoustic.stiffness_like.num_cols,
        acoustic.stiffness_like.nnz(),
    );

    let thermal = eigenmodes::build_thermal_stability_problem(
        &mesh,
        ThermalMaterial::new(2.0, 4.0, 5.0),
        &eigenmodes::ThermalStabilityConfig::default(),
    );
    assert_square_nonempty(
        "thermal stability eigenphysics",
        thermal.stiffness_like.num_rows,
        thermal.stiffness_like.num_cols,
        thermal.stiffness_like.nnz(),
    );

    let em = eigenmodes::build_electromagnetic_eigenmode_problem(
        &mesh,
        EmMaterial::vacuum(),
        &eigenmodes::ElectromagneticEigenmodeConfig::default(),
    );
    assert_square_nonempty(
        "EM eigenphysics",
        em.stiffness_like.num_rows,
        em.stiffness_like.num_cols,
        em.stiffness_like.nnz(),
    );
}

fn main() {
    println!("=== QA: FEM physics assembly integration ===");
    let result = std::panic::catch_unwind(|| {
        test_porous_flow_assembly();
        test_transport_assembly();
        test_thermoelastic_assembly();
        test_composites_assembly();
        test_fracture_assembly();
        test_turbulence_assembly();
        test_compressible_flow_assembly();
        test_electrochemistry_assembly();
        test_semiconductor_assembly();
        test_shallow_water_assembly();
        test_eigenphysics_assembly();
    });

    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(payload) => {
            if let Some(msg) = payload.downcast_ref::<&str>() {
                eprintln!("FAILED: {msg}");
            } else if let Some(msg) = payload.downcast_ref::<String>() {
                eprintln!("FAILED: {msg}");
            } else {
                eprintln!("FAILED: unknown panic");
            }
            std::process::exit(1);
        }
    }
}
