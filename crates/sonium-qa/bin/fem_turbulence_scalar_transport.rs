use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::BilinearForm;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::turbulence::{
    KOmegaEquation, KOmegaReactionForm, KOmegaSstConstants, SpalartAllmarasConstants,
    TurbulenceProductionForm, TurbulenceTransportForm, eddy_viscosity_k_omega,
    logarithmic_wall_velocity,
};
use sonium_fem::spaces::{BasisTabulation, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn exact_diffusion_production_1d(x: f64, l: f64, diffusivity: f64, production: f64) -> f64 {
    (production / (2.0 * diffusivity)) * x * (l - x)
}

fn test_turbulence_scalar_transport() {
    let l = 1.0;
    let h = 0.1;
    let nx = 40;
    let ny = 2;
    let molecular_viscosity = 1.0;
    let eddy_viscosity = 0.0;
    let sigma = 1.0;
    let velocity = [0.0, 0.0, 0.0];
    let reaction = 0.0;
    let production = 2.0;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);

    let transport_form = TurbulenceTransportForm {
        molecular_viscosity,
        eddy_viscosity,
        sigma,
        velocity,
        reaction,
    };
    let production_form = TurbulenceProductionForm { production };

    let matrix = assemble_bilinear(&transport_form, &space, &space, &mesh, 2);
    let rhs = assemble_linear(&production_form, &space, &mesh, 2);

    let mut system = FemSystem::new(matrix, rhs);
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
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

    let diffusivity = molecular_viscosity + eddy_viscosity / sigma;
    let mut max_err: f64 = 0.0;
    let mut max_val: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact = exact_diffusion_production_1d(node.x, l, diffusivity, production);
        let err = (result.x[i] - exact).abs();
        max_err = max_err.max(err);
        max_val = max_val.max(exact);
    }

    println!(
        "Turbulence scalar transport: max_err={:.4e}, max_val={:.4e}, rel_err={:.4e}",
        max_err,
        max_val,
        max_err / max_val.max(1e-12)
    );
    assert!(
        max_err / max_val.max(1e-12) < 0.01,
        "Turbulence scalar transport error too large: {:.4e}",
        max_err / max_val
    );
}

fn test_turbulence_with_advection() {
    // Advection-diffusion with production: D*phi'' - u*phi' + P = 0
    // With phi(0)=0, phi(L)=0, the exact solution is more complex.
    // For small Pe, approximate as parabolic.
    let l = 1.0;
    let h = 0.1;
    let nx = 80;
    let ny = 2;
    let molecular_viscosity = 1.0;
    let eddy_viscosity = 0.0;
    let sigma = 1.0;
    let velocity = [0.5, 0.0, 0.0];
    let reaction = 0.0;
    let production = 2.0;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);

    let transport_form = TurbulenceTransportForm {
        molecular_viscosity,
        eddy_viscosity,
        sigma,
        velocity,
        reaction,
    };
    let production_form = TurbulenceProductionForm { production };

    let matrix = assemble_bilinear(&transport_form, &space, &space, &mesh, 2);
    let rhs = assemble_linear(&production_form, &space, &mesh, 2);

    let mut system = FemSystem::new(matrix, rhs);
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
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

    // For small Pe, compare with pure diffusion parabolic solution
    let diffusivity = molecular_viscosity + eddy_viscosity / sigma;
    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let approx = exact_diffusion_production_1d(node.x, l, diffusivity, production);
        let err = (result.x[i] - approx).abs();
        max_err = max_err.max(err);
    }

    println!("Turbulence with advection: max_err={:.4e}", max_err);
    // Loose tolerance since advection shifts the peak
    assert!(
        max_err < 0.02,
        "Turbulence with advection error too large: {:.4e}",
        max_err
    );
}

fn test_rans_model_closures_and_wall_law() {
    let sa = SpalartAllmarasConstants::default();
    let nu_tilde = 0.01;
    let production = sa.production(nu_tilde, 100.0);
    let destruction = sa.destruction(nu_tilde, 0.5);
    let turbulent_viscosity = sa.turbulent_viscosity(nu_tilde, 1.0e-5);
    let sst = KOmegaSstConstants::default();
    let limited = sst.production_limiter(1000.0, 0.5, 20.0);
    let nu_t = eddy_viscosity_k_omega(0.5, 20.0, 1.0e-6);
    let u_plus = logarithmic_wall_velocity(0.1, 0.01, 1.0e-5, 0.41, 5.2) / 0.1;

    let tab = BasisTabulation {
        n_dofs: 1,
        spatial_dim: 1,
        values: vec![1.0],
        derivatives: vec![0.0],
        hessians: vec![],
        det_j: 1.0,
        physical_coords: [0.0; 3],
    };
    let mut reaction = sonium_fem::forms::LocalMatrix::zeros(1);
    KOmegaReactionForm {
        constants: sst,
        omega: 20.0,
        equation: KOmegaEquation::Omega,
    }
    .integrate_point(&tab, &tab, 1.0, &mut reaction);

    println!(
        "RANS closures: SA P/D={:.4e}/{:.4e}, nu_t={:.4e}/{:.4e}, SST limit={:.4e}, u+={:.4e}",
        production, destruction, turbulent_viscosity, nu_t, limited, u_plus
    );
    assert!(production > 0.0);
    assert!(destruction > 0.0);
    assert!(turbulent_viscosity > 0.0);
    assert!(limited > 0.0 && limited < 1000.0);
    assert!(reaction.get(0, 0) > 0.0);
    assert!(u_plus > 10.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: turbulence scalar transport ===");
        test_turbulence_scalar_transport();
        test_turbulence_with_advection();
        test_rans_model_closures_and_wall_law();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
