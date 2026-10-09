use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::physics::thermal::heat_conduction::{ConductionForm, ThermalMaterial};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn test_thermal_conduction_1d() {
    // 1D rod simulated in 2D: L=1.0m, x=0 at T=100, x=L at T=200.
    // Analytical solution: T(x) = 100 + 100*x

    let l = 1.0;
    let w = 0.1;
    let conductivity = 50.0; // W/(m*K)

    let nx = 10;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, ny);

    let degree = PolynomialDegree::P1;
    let space = H1Space::new(degree);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = ThermalMaterial::new(conductivity, 450.0, 7800.0);
    let form = ConductionForm::new(&material);

    // Assemble stiffness matrix (K_ij = integral k * grad phi_i . grad phi_j)
    let stiffness = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 1);

    let rhs = vec![0.0; n_dofs];
    let mut system = FemSystem::new(stiffness, rhs);

    // Apply Dirichlet BCs
    // Left edge (x=0, marker 1): T=100
    // Right edge (x=L, marker 2): T=200
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 100.0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 200.0, None);

    // Solve
    let csr = system.matrix.to_csr();

    let config = GmresConfig {
        max_iterations: 1000,
        restart: 50,
        tolerance: 1e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let result = gmres(&csr, &system.rhs, &config);
    assert!(result.converged, "Gmres should converge for thermal");

    // Check solution at midpoint x=0.5, T=150
    let mut found_midpoint = false;
    for i in 0..mesh.num_nodes() {
        if (mesh.nodes[i].x - 0.5).abs() < 1e-10 {
            let t = result.x[i];
            assert!(
                (t - 150.0).abs() < 1e-10,
                "T at x=0.5 is {}, expected 150.0",
                t
            );
            found_midpoint = true;
        }
    }
    assert!(found_midpoint, "Midpoint x=0.5 not found in mesh");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_thermal_conduction_1d();
    });
    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
