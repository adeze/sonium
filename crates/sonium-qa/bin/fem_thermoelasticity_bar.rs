use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::coupled::thermoelasticity::{
    ThermalStrainLoadForm, ThermoelasticLayer, ThermoelasticMaterial, bonded_layer_free_curvature,
    build_thermoelastic_coupled_blocks,
};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::physics::thermal::heat_conduction::ThermalMaterial;
use sonium_fem::spaces::VectorH1Space;
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres_preconditioned};

fn test_thermoelasticity_bar_free_expansion() {
    let l = 1.0;
    let h = 0.1;
    let nx = 40;
    let ny = 2;
    let e_mod = 200.0;
    let nu = 0.0;
    let alpha = 1.0e-5;
    let t_ref = 20.0;
    let t_hot = 120.0;
    let delta_t = t_hot - t_ref;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);

    let elastic_material = IsotropicMaterial::new(e_mod, nu);
    let thermo_material = ThermoelasticMaterial {
        young_modulus: e_mod,
        poisson_ratio: nu,
        thermal_expansion: alpha,
        reference_temperature: t_ref,
    };

    let form = ElasticityForm::new_2d(elastic_material, PlaneAssumption::PlaneStress);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let thermal_form = ThermalStrainLoadForm {
        material: thermo_material,
        temperature: t_hot,
    };
    let thermal_load = sonium_fem::assembly::assemble_linear(&thermal_form, &space, &mesh, 1);

    let mut system = FemSystem::new(stiffness, thermal_load);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    let csr = system.matrix.to_csr();
    let precond = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(
        &csr,
        &precond,
        &system.rhs,
        &GmresConfig {
            max_iterations: 5000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "GMRES+ILU should converge for thermoelasticity"
    );

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let x = node.x;
        let u_exact_x = alpha * delta_t * x;
        let u_fem_x = result.x[i * 2];
        max_err = max_err.max((u_fem_x - u_exact_x).abs());
    }

    let tip_exact = alpha * delta_t * l;
    let tip_fem = result.x[(mesh.num_nodes() - 1) * 2];
    println!(
        "Thermoelasticity bar: max_err={:.4e}, tip_exact={:.6e}, tip_fem={:.6e}",
        max_err, tip_exact, tip_fem
    );
    assert!(
        max_err < 1e-5,
        "Thermoelasticity error too large: {:.4e}",
        max_err
    );
}

fn test_constrained_bar_and_bimetal_curvature() {
    let material = ThermoelasticMaterial {
        young_modulus: 200.0e9,
        poisson_ratio: 0.3,
        thermal_expansion: 12.0e-6,
        reference_temperature: 293.15,
    };
    let delta_t = 50.0;
    let stress_1d = material.constrained_thermal_stress_1d(293.15 + delta_t);
    let layers = [
        ThermoelasticLayer {
            young_modulus: 70.0e9,
            thickness: 1.0e-3,
            thermal_expansion: 23.0e-6,
        },
        ThermoelasticLayer {
            young_modulus: 200.0e9,
            thickness: 1.0e-3,
            thermal_expansion: 12.0e-6,
        },
    ];
    let curvature = bonded_layer_free_curvature(&layers, delta_t);

    println!(
        "Thermoelastic coefficients: constrained stress={:.4e}, bimetal curvature={:.4e}",
        stress_1d, curvature
    );
    assert!((stress_1d - 120.0e6).abs() < 1.0e-6);
    assert!(curvature.abs() > 0.1);
}

fn test_coupled_block_builder() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.1, 4, 1);
    let blocks = build_thermoelastic_coupled_blocks(
        &mesh,
        PolynomialDegree::P1,
        2,
        IsotropicMaterial::new(10.0, 0.25),
        Some(PlaneAssumption::PlaneStress),
        ThermalMaterial::new(1.0, 2.0, 3.0),
        0.5,
    );

    println!(
        "Thermoelastic blocks: K={} Kt={} Mt={} C={}",
        blocks.mechanical_stiffness.nnz(),
        blocks.heat_conduction.nnz(),
        blocks.heat_capacity.nnz(),
        blocks.damping_coupling.nnz()
    );
    assert!(blocks.mechanical_stiffness.nnz() > 0);
    assert!(blocks.heat_conduction.nnz() > 0);
    assert!(blocks.heat_capacity.nnz() > 0);
    assert!(blocks.damping_coupling.nnz() > 0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: thermoelasticity bar ===");
        test_thermoelasticity_bar_free_expansion();
        test_constrained_bar_and_bimetal_curvature();
        test_coupled_block_builder();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
