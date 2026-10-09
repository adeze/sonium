use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::transport::{
    TransportCapacityForm, TransportDiffusionForm, TransportMaterial,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn gaussian_1d(x: f64, sigma: f64) -> f64 {
    let s2 = sigma * sigma;
    (-x * x / (2.0 * s2)).exp() / (s2 * std::f64::consts::PI * 2.0).sqrt()
}

fn exact_diffusion_1d(x: f64, t: f64, sigma0: f64, diffusivity: f64) -> f64 {
    if t <= 0.0 {
        return gaussian_1d(x, sigma0);
    }
    let sigma_t = (sigma0 * sigma0 + 2.0 * diffusivity * t).sqrt();
    gaussian_1d(x, sigma_t)
}

fn solve_diffusion_step(
    _mesh: &sonium_fem::mesh::Mesh,
    _space: &H1Space,
    mass_csr: &sonium_solvers::sparse::CsrMatrix<f64>,
    system_matrix: &sonium_solvers::sparse::CsrMatrix<f64>,
    u: &[f64],
) -> Vec<f64> {
    let rhs = mass_csr.matvec(u);

    let result = gmres(
        system_matrix,
        &rhs,
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
    assert!(result.converged, "GMRES did not converge in diffusion step");
    result.x
}

fn test_gaussian_diffusion_1d() {
    let l = 5.0;
    let h = 0.1;
    let nx = 200;
    let ny = 2;
    let diffusivity = 1.0;
    let sigma0 = 0.5;
    let dt = 0.001;
    let n_steps = 10;
    let t_final = dt * n_steps as f64;

    let mesh = rectangular_mesh_triangles(-l, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = TransportMaterial::isotropic(diffusivity, [0.0; 3], 0.0);
    let mass_form = TransportCapacityForm::new(&material);
    let diff_form = TransportDiffusionForm::new(&material);

    let mass = assemble_bilinear(&mass_form, &space, &space, &mesh, 2);
    let stiffness = assemble_bilinear(&diff_form, &space, &space, &mesh, 2);

    // System matrix for backward Euler: M + dt*K
    let mut system = FemSystem::new(mass.clone(), vec![0.0; n_dofs]);
    for i in 0..stiffness.nnz() {
        let row = stiffness.rows[i];
        let col = stiffness.cols[i];
        let val = stiffness.values[i] * dt;
        // Find if (row, col) already exists in system.matrix
        let mut found = false;
        for j in 0..system.matrix.nnz() {
            if system.matrix.rows[j] == row && system.matrix.cols[j] == col {
                system.matrix.values[j] += val;
                found = true;
                break;
            }
        }
        if !found {
            system.matrix.push(row, col, val);
        }
    }

    // Dirichlet BCs: u = 0 at left and right boundaries
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 0.0, None);

    let mass_csr = mass.to_csr();
    let system_csr = system.matrix.to_csr();

    // Initial condition: Gaussian
    let mut u = vec![0.0; n_dofs];
    for i in 0..n_dofs {
        u[i] = gaussian_1d(mesh.nodes[i].x, sigma0);
    }

    // Time stepping
    for _ in 0..n_steps {
        u = solve_diffusion_step(&mesh, &space, &mass_csr, &system_csr, &u);
    }

    // Compare with exact solution in the center region
    let mut max_err: f64 = 0.0;
    let mut max_val: f64 = 0.0;
    for i in 0..n_dofs {
        let x = mesh.nodes[i].x;
        if x.abs() < 3.0 {
            let exact = exact_diffusion_1d(x, t_final, sigma0, diffusivity);
            let err = (u[i] - exact).abs();
            max_err = max_err.max(err);
            max_val = max_val.max(exact);
        }
    }

    println!(
        "Gaussian diffusion: t={:.3}, max_err={:.4e}, max_val={:.4e}, rel_err={:.4e}",
        t_final,
        max_err,
        max_val,
        max_err / max_val.max(1e-12)
    );
    assert!(
        max_err < 5e-3,
        "Gaussian diffusion error too large: {:.4e}",
        max_err
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: transport Gaussian diffusion ===");
        test_gaussian_diffusion_1d();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
