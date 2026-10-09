//! Stokes flow — Taylor-Couette (concentric cylinder flow) with Taylor-Hood P2-P1
//!
//! Solves steady Stokes on an annular domain:
//!   Inner cylinder (r = R1): rotates at angular velocity omega -> u = omega * R1 * e_theta
//!   Outer cylinder (r = R2): stationary -> u = 0
//!
//! Analytical solution:
//!   u_theta(r) = A*r + B/r
//!   where A = -omega * R1^2 / (R2^2 - R1^2)
//!         B =  omega * R1^2 * R2^2 / (R2^2 - R1^2)
//!   u_r = 0

use sonium_fem::assembly;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::assembly::mixed::assemble_mixed;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::forms::MassForm;
use sonium_fem::mesh::generators::annular_mesh_triangles;
use sonium_fem::mesh::topology::MeshTopology;
use sonium_fem::physics::fluid::stokes::forms::{StokesDivergenceForm, StokesViscousForm};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

use sonium_solvers::{
    AmgConfig, BlockPreconditionerConfig, FgmresConfig, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, fgmres_preconditioned,
};
use sonium_xem::io::binary_output_dir;
use std::collections::HashMap;

fn test_stokes_taylor_couette() {
    env_logger::init();
    println!("=== Stokes Taylor-Couette Flow (Taylor-Hood P2-P1) ===\n");

    let mu = 1.0;
    let r1 = 1.0; // inner radius
    let r2 = 2.0; // outer radius
    let omega = 1.0; // angular velocity of inner cylinder
    let n_radial = 8;
    let n_angular = 32;

    // Analytical coefficients
    let a_coeff = -omega * r1 * r1 / (r2 * r2 - r1 * r1);
    let b_coeff = omega * r1 * r1 * r2 * r2 / (r2 * r2 - r1 * r1);

    // --- 1. Mesh ---
    let mesh = annular_mesh_triangles(0.0, 0.0, r1, r2, n_radial, n_angular);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // --- 2. Function spaces ---
    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let n_total = n_v + n_p;
    println!("Velocity P2: {} DOFs", n_v);
    println!("Pressure P1: {} DOFs", n_p);
    println!("Total:       {} DOFs\n", n_total);

    // --- 3. Assemble ---
    println!("Assembling Stokes system...");
    let viscous = StokesViscousForm::new(mu);
    let div_form = StokesDivergenceForm;
    let system = assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 4);
    let mono = system.to_monolithic();

    let mp = assembly::assemble_bilinear(&MassForm, &p_space, &p_space, &mesh, 2);
    let mut schur = TripletMatrix::<f64>::with_capacity(mp.num_rows, mp.num_cols, mp.nnz());
    for i in 0..mp.nnz() {
        schur.push(mp.rows[i], mp.cols[i], mp.values[i] / mu);
    }

    // --- 4. DOF coordinates ---
    let topo = MeshTopology::build(&mesh);
    let n_nodes = mesh.num_nodes();
    let n_edges = topo.num_edges();
    let n_scalar = n_nodes + n_edges;

    let mut dof_x = vec![0.0; n_scalar];
    let mut dof_y = vec![0.0; n_scalar];
    for i in 0..n_nodes {
        dof_x[i] = mesh.nodes[i].x;
        dof_y[i] = mesh.nodes[i].y;
    }
    for (ei, &(a, b)) in topo.edges.iter().enumerate() {
        dof_x[n_nodes + ei] = 0.5 * (mesh.nodes[a].x + mesh.nodes[b].x);
        dof_y[n_nodes + ei] = 0.5 * (mesh.nodes[a].y + mesh.nodes[b].y);
    }

    // --- 5. Boundary conditions ---
    println!("Applying boundary conditions...");
    let mut fem = FemSystem::new(mono, vec![0.0; n_total]);
    let tol_inner = r1 * 0.02;
    let tol_outer = r2 * 0.02;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    for sd in 0..n_scalar {
        let x = dof_x[sd];
        let y = dof_y[sd];
        let r = (x * x + y * y).sqrt();

        if (r - r1).abs() < tol_inner {
            // Inner cylinder: u = omega * R1 * e_theta = omega * R1 * (-sin(theta), cos(theta))
            // = omega * (-y, x) (since r=R1, sin(theta) = y/R1, cos(theta) = x/R1)
            let ux = -omega * y;
            let uy = omega * x;
            bc_map.insert(sd * 2, ux);
            bc_map.insert(sd * 2 + 1, uy);
        } else if (r - r2).abs() < tol_outer {
            // Outer cylinder: u = 0
            bc_map.insert(sd * 2, 0.0);
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }

    // Pin pressure at one point on outer boundary
    for node in 0..n_nodes {
        let x = mesh.nodes[node].x;
        let y = mesh.nodes[node].y;
        let r = (x * x + y * y).sqrt();
        if (r - r2).abs() < tol_outer {
            bc_map.insert(n_v + node, 0.0);
            break;
        }
    }

    println!("  {} constrained DOFs", bc_map.len());

    // Apply BCs
    apply_dirichlet(&mut fem, &bc_map);

    // --- 6. Solve ---
    println!("\nSolving with FGMRES + block preconditioner...");
    let a_csr = fem.matrix.to_csr();
    let schur_csr = schur.to_csr();

    let sp_config = SaddlePointConfig {
        block1_size: n_v,
        block1_config: BlockPreconditionerConfig::Amg(AmgConfig::for_fem()),
        block2_config: BlockPreconditionerConfig::Amg(AmgConfig::for_fem()),
        schur_strategy: SchurStrategy::External,
        upper_triangular: true,
    };
    let precond =
        SaddlePointPreconditioner::<f64>::from_matrix_with_schur(&a_csr, &schur_csr, sp_config);

    let rhs = fem.rhs.clone();
    let config = FgmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
    };
    let result = fgmres_preconditioned(&a_csr, &precond, &rhs, None, &config);

    println!(
        "  {} (iters: {}, residual: {:.2e})",
        if result.converged {
            "Converged"
        } else {
            "NOT converged"
        },
        result.iterations,
        result.residual,
    );

    // --- 7. Compare with analytical ---
    let u_sol: Vec<f64> = result.x.iter().take(n_v).copied().collect();
    let p_sol: Vec<f64> = result.x.iter().skip(n_v).copied().collect();

    let mut vel_err = 0.0;
    let mut vel_norm = 0.0;
    let mut count = 0;

    println!("\n--- Comparison along radial line (theta ~ 0) ---");
    println!(
        "  {:>8} {:>12} {:>12} {:>12}",
        "r", "u_theta_h", "u_theta_ex", "error"
    );

    // Sample along positive x-axis (theta ≈ 0)
    for sd in 0..n_scalar {
        let x = dof_x[sd];
        let y = dof_y[sd];
        let r = (x * x + y * y).sqrt();

        if r < r1 - 0.1 || r > r2 + 0.1 {
            continue;
        }

        // Analytical: u_theta(r) = A*r + B/r
        // In Cartesian: u_x = -u_theta * sin(theta), u_y = u_theta * cos(theta)
        // = -u_theta * y/r, u_theta * x/r
        let u_theta_exact = a_coeff * r + b_coeff / r;
        let ux_exact = -u_theta_exact * y / r;
        let uy_exact = u_theta_exact * x / r;

        let ux_h = u_sol[sd * 2];
        let uy_h = u_sol[sd * 2 + 1];

        vel_err += (ux_h - ux_exact).powi(2) + (uy_h - uy_exact).powi(2);
        vel_norm += ux_exact.powi(2) + uy_exact.powi(2);
        count += 1;

        // Print points near positive x-axis
        let theta = y.atan2(x);
        if theta.abs() < 0.15 && r > r1 + 0.01 && r < r2 - 0.01 {
            let u_theta_h = -ux_h * y / r + uy_h * x / r;
            println!(
                "  {:>8.4} {:>12.6} {:>12.6} {:>12.2e}",
                r,
                u_theta_h,
                u_theta_exact,
                (u_theta_h - u_theta_exact).abs()
            );
        }
    }

    let vel_rel = (vel_err / vel_norm.max(1e-30)).sqrt();
    println!(
        "\nOverall velocity relative L2 error: {:.6e} ({} DOFs sampled)",
        vel_rel, count
    );

    assert!(
        vel_rel < 0.02,
        "Taylor-Couette velocity relative L2 error too large: {vel_rel:.6e} (threshold: 2%)"
    );

    // VTK output
    let mut node_ux = vec![0.0; n_nodes];
    let mut node_uy = vec![0.0; n_nodes];
    let mut node_p = vec![0.0; n_nodes];
    let mut node_utheta = vec![0.0; n_nodes];
    for i in 0..n_nodes {
        node_ux[i] = u_sol[i * 2];
        node_uy[i] = u_sol[i * 2 + 1];
        node_p[i] = p_sol[i];
        let x = mesh.nodes[i].x;
        let y = mesh.nodes[i].y;
        let r = (x * x + y * y).sqrt();
        if r > 1e-10 {
            node_utheta[i] = -node_ux[i] * y / r + node_uy[i] * x / r;
        }
    }

    let mut writer = sonium_fem::io::vtk::VtuWriter::new(&mesh);
    writer.add_point_scalar("ux", &node_ux);
    writer.add_point_scalar("uy", &node_uy);
    writer.add_point_scalar("u_theta", &node_utheta);
    writer.add_point_scalar("pressure", &node_p);

    let output_path = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
        .join("stokes_taylor_couette.vtu");
    writer
        .write(&output_path)
        .expect("write Taylor-Couette VTU");
    println!("\nWrote {}", output_path.display());

    println!("\n=== Done ===");
}

fn apply_dirichlet(system: &mut FemSystem<f64>, bc_map: &HashMap<usize, f64>) {
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row)
            && let Some(&val) = bc_map.get(&col)
        {
            system.rhs[row] -= system.matrix.values[i] * val;
        }
    }
    let mut nr = Vec::with_capacity(system.matrix.nnz());
    let mut nc = Vec::with_capacity(system.matrix.nnz());
    let mut nv = Vec::with_capacity(system.matrix.nnz());
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row) && !bc_map.contains_key(&col) {
            nr.push(row);
            nc.push(col);
            nv.push(system.matrix.values[i]);
        }
    }
    for (&dof, &val) in bc_map {
        nr.push(dof);
        nc.push(dof);
        nv.push(1.0);
        system.rhs[dof] = val;
    }
    system.matrix.rows = nr;
    system.matrix.cols = nc;
    system.matrix.values = nv;
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_stokes_taylor_couette();
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
