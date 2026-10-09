//! QA: FEM elasticity — cantilever beam against Euler-Bernoulli beam theory.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres, gmres_preconditioned};
use sonium_xem::io::binary_output_dir;

fn write_displacement_vtu(name: &str, mesh: &sonium_fem::mesh::Mesh, x: &[f64]) {
    let n = mesh.num_nodes();
    let ux: Vec<f64> = (0..n).map(|i| x[i * 2]).collect();
    let uy: Vec<f64> = (0..n).map(|i| x[i * 2 + 1]).collect();
    let uz = vec![0.0; n];
    let mag: Vec<f64> = (0..n)
        .map(|i| (ux[i] * ux[i] + uy[i] * uy[i]).sqrt())
        .collect();
    let mut w = VtuWriter::new(mesh);
    w.add_point_vector("displacement", &ux, &uy, &uz);
    w.add_point_scalar("displacement_magnitude", &mag);
    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let _ = w.write(dir.join(format!("{name}.vtu")));
}

fn test_elasticity_cantilever_2d() {
    // L/h = 10 — Euler-Bernoulli is reasonable, not too thin to cause locking
    let l = 1.0;
    let h = 0.1;
    let e_mod = 2.1e11;
    let nu = 0.3;
    let p_load = 1e6;

    // Dense mesh to overcome P1 triangle shear locking in bending
    let nx = 100;
    let ny = 10;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);

    let degree = PolynomialDegree::P1;
    let space = VectorH1Space::new(degree, 2);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = IsotropicMaterial::new(e_mod, nu);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);

    let stiffness = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 1);

    let mut rhs = vec![0.0; n_dofs];
    let tol = 1e-10;
    let tip_nodes: Vec<usize> = (0..mesh.num_nodes())
        .filter(|&i| (mesh.nodes[i].x - l).abs() < tol)
        .collect();

    let load_per_node = -p_load / tip_nodes.len() as f64;
    for &node in &tip_nodes {
        let v_dof = node * 2 + 1;
        rhs[v_dof] = load_per_node;
    }

    let mut system = FemSystem::new(stiffness, rhs);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    let csr = system.matrix.to_csr();

    let config = GmresConfig {
        max_iterations: 5000,
        restart: 200,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let precond = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(&csr, &precond, &system.rhs, &config);
    assert!(result.converged, "GMRES+ILU should converge for elasticity");

    write_displacement_vtu("cantilever_2d", &mesh, &result.x);

    let i_inertia = 1.0 * h.powi(3) / 12.0;
    let exact_delta = p_load * l.powi(3) / (3.0 * e_mod * i_inertia);

    let mut max_deflection = 0.0;
    for &node in &tip_nodes {
        let v_dof = node * 2 + 1;
        let d = result.x[v_dof].abs();
        if d > max_deflection {
            max_deflection = d;
        }
    }

    println!(
        "FEM tip deflection: {:.6e}, Beam theory: {:.6e}",
        max_deflection, exact_delta
    );

    let rel_diff = (max_deflection - exact_delta).abs() / exact_delta;
    println!("Relative error: {:.4e}", rel_diff);
    assert!(
        rel_diff < 0.03,
        "Deflection error {:.4e} should be < 3%",
        rel_diff
    );
}

fn test_elasticity_plane_stress() {
    let e_mod = 70e9;
    let nu = 0.33;
    let l = 1.0;
    let h = 0.1;
    let nx = 10;
    let ny = 4;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);

    let degree = PolynomialDegree::P1;
    let space = VectorH1Space::new(degree, 2);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = IsotropicMaterial::new(e_mod, nu);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);

    let stiffness = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 1);

    let rhs = vec![0.0; n_dofs];
    let mut system = FemSystem::new(stiffness, rhs);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    let csr = system.matrix.to_csr();

    let config = GmresConfig {
        max_iterations: 1000,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let result = gmres(&csr, &system.rhs, &config);
    assert!(
        result.converged,
        "GMRES should converge for plane stress (zero RHS)"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elasticity_cantilever_2d();
        test_elasticity_plane_stress();
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
