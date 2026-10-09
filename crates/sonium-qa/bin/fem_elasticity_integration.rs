use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres_preconditioned};
use sonium_xem::io::binary_output_dir;

fn test_elasticity_cantilever_2d() {
    // 2D Cantilever beam: L=1.0m, H=0.1m, fixed at x=0, vertical load at tip.
    // We use a relatively thick beam to avoid shear locking with P1 elements.

    let l = 1.0;
    let h = 0.2;
    let e_mod = 2.1e11; // steel
    let nu = 0.3;
    let p_load = 1e6; // 1 MN/m at the tip

    let nx = 40;
    let ny = 8;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);

    let degree = PolynomialDegree::P1;
    let space = VectorH1Space::new(degree, 2);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = IsotropicMaterial::new(e_mod, nu);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);

    // Assemble stiffness matrix
    let stiffness = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 1);

    // RHS: vertical point load at the tip (right edge midpoint)
    let mut rhs = vec![0.0; n_dofs];
    // Find all nodes at x=L
    let tol = 1e-10;
    let tip_nodes: Vec<usize> = (0..mesh.num_nodes())
        .filter(|&i| (mesh.nodes[i].x - l).abs() < tol)
        .collect();

    // Apply load distributed over the tip nodes
    let load_per_node = -p_load / tip_nodes.len() as f64;
    for &node in &tip_nodes {
        let v_dof = node * 2 + 1; // y-component
        rhs[v_dof] = load_per_node;
    }

    let mut system = FemSystem::new(stiffness, rhs);

    // Apply fixed BC at x=0 (Marker 1 from rectangular_mesh_triangles)
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    // Solve
    let csr = system.matrix.to_csr();

    let config = GmresConfig {
        max_iterations: 2000,
        restart: 100,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let preconditioner = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(&csr, &preconditioner, &system.rhs, &config);
    assert!(result.converged, "Gmres should converge for elasticity");

    {
        let n = mesh.num_nodes();
        let ux: Vec<f64> = (0..n).map(|i| result.x[i * 2]).collect();
        let uy: Vec<f64> = (0..n).map(|i| result.x[i * 2 + 1]).collect();
        let uz = vec![0.0; n];
        let mag: Vec<f64> = (0..n)
            .map(|i| (ux[i] * ux[i] + uy[i] * uy[i]).sqrt())
            .collect();
        let mut w = VtuWriter::new(&mesh);
        w.add_point_vector("displacement", &ux, &uy, &uz);
        w.add_point_scalar("displacement_magnitude", &mag);
        let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
        let _ = w.write(dir.join("cantilever.vtu"));
    }

    // Check tip deflection
    // Beam theory (approx): delta = P * L^3 / (3 * E * I)
    // I = 1.0 * H^3 / 12 (for 1m depth in plane strain)
    let i_inertia = 1.0 * h.powi(3) / 12.0;
    let exact_delta = p_load * l.powi(3) / (3.0 * e_mod * i_inertia);

    // Max deflection should be at the tip nodes
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

    // This plane-stress model and the unit-thickness Euler-Bernoulli
    // reference use the same constitutive assumption.
    let rel_diff = (max_deflection - exact_delta).abs() / exact_delta;
    assert!(
        rel_diff < 0.03,
        "Deflection error {} should be < 3%",
        rel_diff
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elasticity_cantilever_2d();
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
