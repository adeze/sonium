//! QA: FEM elasticity — 1D bar under axial tension.
//!
//! Analytical: u(x) = F*x/(E*A)

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_qa::analytical::elasticity_1d_bar;
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres_preconditioned};

fn test_elasticity_1d_bar() {
    let l = 1.0;
    let h = 0.05;
    let e_mod = 210e9;
    let nu = 0.3;
    let f_load = 1e6;

    let nx = 20;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);

    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = IsotropicMaterial::new(e_mod, nu);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let area = h; // unit thickness
    let mut rhs = vec![0.0; n_dofs];

    // Apply traction at right boundary (x = L)
    let tol = 1e-10;
    let tip_nodes: Vec<usize> = (0..mesh.num_nodes())
        .filter(|&i| (mesh.nodes[i].x - l).abs() < tol)
        .collect();

    let force_per_node = f_load / tip_nodes.len() as f64;
    for &node in &tip_nodes {
        let u_dof = node * 2; // x-displacement DOF
        rhs[u_dof] = force_per_node;
    }

    let mut system = FemSystem::new(stiffness, rhs);
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
    assert!(result.converged, "GMRES+ILU should converge for elasticity");

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let x = node.x;
        let u_exact = elasticity_1d_bar(x, f_load, e_mod, area);
        let u_fem = result.x[i * 2];
        max_err = max_err.max((u_fem - u_exact).abs());
    }

    let tip_exact = elasticity_1d_bar(l, f_load, e_mod, area);
    let tip_fem = result.x[tip_nodes[0] * 2];
    println!(
        "Elasticity 1D bar: max_err = {:.4e}, tip_exact = {:.6e}, tip_fem = {:.6e}",
        max_err, tip_exact, tip_fem
    );
    assert!(max_err < 1e-6, "Max error too large: {:.4e}", max_err);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elasticity_1d_bar();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
