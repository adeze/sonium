use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::electromagnetics::semiconductor::{
    CarrierDiffusionForm, CarrierKind, CarrierMaterial, CarrierRecombinationForm, DopingProfile,
    bernoulli_function, pn_junction_built_in_potential, scharfetter_gummel_flux, thermal_voltage,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn exact_carrier_diffusion(x: f64, l: f64, n0: f64, diffusivity: f64, lifetime: f64) -> f64 {
    let ld = (diffusivity * lifetime).sqrt();
    if ld < 1e-14 {
        return n0 * (1.0 - x / l);
    }
    let ratio = l / ld;
    if ratio > 50.0 {
        // Avoid overflow: approximate as exponential decay
        return n0 * ((-x / ld).exp() - (-l / ld).exp()) / (1.0 - (-l / ld).exp());
    }
    n0 * ((l - x) / ld).sinh() / (ratio.sinh())
}

fn test_semiconductor_carrier_diffusion() {
    let l = 1.0e-4;
    let h = 0.1 * l;
    let nx = 80;
    let ny = 2;
    let n0 = 1.0e20;
    let mobility = 0.14;
    let lifetime = 1.0e-6;
    let temperature = 300.0;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = CarrierMaterial {
        mobility,
        lifetime,
        temperature,
        kind: CarrierKind::Electron,
    };
    let diffusivity = material.diffusivity();

    let diff_form = CarrierDiffusionForm::new(&material);
    let reac_form = CarrierRecombinationForm { lifetime };

    let diff_matrix = assemble_bilinear(&diff_form, &space, &space, &mesh, 2);
    let reac_matrix = assemble_bilinear(&reac_form, &space, &space, &mesh, 2);

    // Combine: D*K_diff + K_reac
    let mut system = FemSystem::new(diff_matrix, vec![0.0; n_dofs]);
    for i in 0..reac_matrix.nnz() {
        system.matrix.push(
            reac_matrix.rows[i],
            reac_matrix.cols[i],
            reac_matrix.values[i],
        );
    }

    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| n0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 0.0, None);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES did not converge");

    let mut max_err: f64 = 0.0;
    let mut max_val: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact = exact_carrier_diffusion(node.x, l, n0, diffusivity, lifetime);
        let err = (result.x[i] - exact).abs();
        max_err = max_err.max(err);
        max_val = max_val.max(exact);
    }

    println!(
        "Semiconductor carrier diffusion: max_err={:.4e}, max_val={:.4e}, rel_err={:.4e}",
        max_err,
        max_val,
        max_err / max_val.max(1e-30)
    );
    assert!(
        max_err / max_val.max(1e-30) < 0.01,
        "Carrier diffusion relative error too large: {:.4e}",
        max_err / max_val
    );
}

fn test_pn_junction_and_scharfetter_gummel_flux() {
    let temperature = 300.0;
    let vt = thermal_voltage(temperature);
    let built_in = pn_junction_built_in_potential(1.0e22, 5.0e21, 1.0e16, temperature);
    let zero_bias_flux = scharfetter_gummel_flux(0.003, 1.0e20, 1.0e20, 0.0, 1.0e-6);
    let drift_flux = scharfetter_gummel_flux(0.003, 1.0e20, 0.5e20, 2.0, 1.0e-6);
    let doping = DopingProfile {
        donors: 5.0e21,
        acceptors: 1.0e21,
    };

    println!(
        "Semiconductor helpers: Vt={:.4e}, Vbi={:.4e}, B(0)={:.4e}, flux={:.4e}, rho={:.4e}",
        vt,
        built_in,
        bernoulli_function(0.0),
        drift_flux,
        doping.net_charge_density()
    );
    assert!(vt > 0.02 && vt < 0.03);
    assert!(built_in > 0.5);
    assert!(zero_bias_flux.abs() < 1.0e-6);
    assert!(drift_flux.abs() > 0.0);
    assert!(doping.net_charge_density() > 0.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: semiconductor carrier diffusion ===");
        test_semiconductor_carrier_diffusion();
        test_pn_junction_and_scharfetter_gummel_flux();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
