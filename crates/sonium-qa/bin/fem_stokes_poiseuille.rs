//! Stokes flow — 2D Poiseuille (channel flow) with Taylor-Hood P2-P1
//!
//! Solves the steady-state Stokes equations on [0,1]^2:
//!
//! ```text
//!   -mu * laplacian(u) + grad(p) = 0
//!                        div(u)  = 0
//! ```
//!
//! Boundary conditions:
//!   - No-slip walls (top y=1, bottom y=0): u = 0
//!   - Inlet (left x=0): u_x = 4*y*(1-y), u_y = 0
//!   - Outlet (right x=1): p = 0 (natural BC via Dirichlet on pressure)
//!
//! Analytical solution:
//!   u_x(y) = 4*y*(1-y)  [parabolic profile]
//!   u_y    = 0
//!   p(x)   = 8*mu*(1-x) [linear pressure drop]

use sonium_fem::assembly;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::assembly::mixed::assemble_mixed;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::forms::MassForm;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::mesh::topology::MeshTopology;
use sonium_fem::physics::fluid::stokes::forms::{StokesDivergenceForm, StokesViscousForm};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

use sonium_solvers::{
    AmgConfig, BlockPreconditionerConfig, FgmresConfig, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, fgmres_preconditioned,
};
use sonium_xem::io::binary_output_dir;

fn test_stokes_poiseuille() {
    env_logger::init();
    println!("=== Stokes Poiseuille Flow (Taylor-Hood P2-P1) ===\n");

    let mu = 1.0; // viscosity
    let n = 16; // mesh resolution

    // --- 1. Mesh ---
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, n, n);
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
    println!("Velocity P2: {} DOFs ({} scalar H1 DOFs x 2)", n_v, n_v / 2);
    println!("Pressure P1: {} DOFs", n_p);
    println!("Total:       {} DOFs\n", n_total);

    // --- 3. Assemble block system ---
    println!("Assembling Stokes system...");
    let viscous = StokesViscousForm::new(mu);
    let div_form = StokesDivergenceForm;
    let system = assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 4);
    let mono = system.to_monolithic();
    println!(
        "  Monolithic: {}x{}, {} nnz",
        mono.num_rows,
        mono.num_cols,
        mono.nnz()
    );

    // Pressure mass matrix for Schur complement approximation: S ~ (1/mu) * M_p
    let mp_triplet = assembly::assemble_bilinear(&MassForm, &p_space, &p_space, &mesh, 2);

    // Scale by 1/mu for Schur approximation
    let mut schur_triplet = TripletMatrix::<f64>::with_capacity(
        mp_triplet.num_rows,
        mp_triplet.num_cols,
        mp_triplet.nnz(),
    );
    for i in 0..mp_triplet.nnz() {
        schur_triplet.push(
            mp_triplet.rows[i],
            mp_triplet.cols[i],
            mp_triplet.values[i] / mu,
        );
    }

    // --- 4. Build DOF coordinate map for P2 velocity ---
    let topo = MeshTopology::build(&mesh);
    let n_nodes = mesh.num_nodes();
    let n_edges = topo.num_edges();
    let n_scalar_dofs = n_nodes + n_edges; // P2 scalar DOFs

    // Physical coordinates for all scalar P2 DOFs
    let mut dof_x = vec![0.0; n_scalar_dofs];
    let mut dof_y = vec![0.0; n_scalar_dofs];

    // Vertex DOFs: coordinates from mesh nodes
    for i in 0..n_nodes {
        dof_x[i] = mesh.nodes[i].x;
        dof_y[i] = mesh.nodes[i].y;
    }

    // Edge midpoint DOFs: average of edge endpoints
    for (edge_idx, &(a, b)) in topo.edges.iter().enumerate() {
        dof_x[n_nodes + edge_idx] = 0.5 * (mesh.nodes[a].x + mesh.nodes[b].x);
        dof_y[n_nodes + edge_idx] = 0.5 * (mesh.nodes[a].y + mesh.nodes[b].y);
    }

    // --- 5. Apply boundary conditions ---
    println!("Applying boundary conditions...");
    let mut fem = FemSystem::new(mono, vec![0.0; n_total]);

    let tol = 1e-10;
    let mut bc_dofs = Vec::new();
    let mut bc_vals = Vec::new();

    // Inlet (x=0, marker 1): u_x = 4*y*(1-y), u_y = 0
    // No-slip walls (y=0 marker 3, y=1 marker 4): u = 0
    // For all velocity DOFs: vector DOF = scalar_dof * 2 + component
    for scalar_dof in 0..n_scalar_dofs {
        let x = dof_x[scalar_dof];
        let y = dof_y[scalar_dof];

        // Bottom wall (y=0) or Top wall (y=1)
        if y.abs() < tol || (y - 1.0).abs() < tol {
            bc_dofs.push(scalar_dof * 2); // u_x = 0
            bc_vals.push(0.0);
            bc_dofs.push(scalar_dof * 2 + 1); // u_y = 0
            bc_vals.push(0.0);
        }
        // Inlet (x=0)
        else if x.abs() < tol {
            bc_dofs.push(scalar_dof * 2); // u_x = 4*y*(1-y)
            bc_vals.push(4.0 * y * (1.0 - y));
            bc_dofs.push(scalar_dof * 2 + 1); // u_y = 0
            bc_vals.push(0.0);
        }
    }

    // Outlet pressure: p = 0 at x=1 (one node suffices to pin pressure)
    // Pressure DOFs are offset by n_v in the monolithic system
    for node in 0..n_nodes {
        let x = mesh.nodes[node].x;
        if (x - 1.0).abs() < tol {
            bc_dofs.push(n_v + node); // pressure DOF
            bc_vals.push(0.0);
        }
    }

    println!("  {} constrained DOFs", bc_dofs.len());

    // Apply BCs manually to monolithic system
    apply_dirichlet_manual(&mut fem, &bc_dofs, &bc_vals);

    // --- 6. Solve ---
    println!("\nSolving with FGMRES + block preconditioner...");
    let a_csr = fem.matrix.to_csr();
    let schur_csr = schur_triplet.to_csr();

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
    let fgmres_config = FgmresConfig {
        max_iterations: 200,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
    };

    let result = fgmres_preconditioned(&a_csr, &precond, &rhs, None, &fgmres_config);

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

    // --- 7. Extract and compare with analytical ---
    let u_sol: Vec<f64> = result.x.iter().take(n_v).copied().collect();
    let p_sol: Vec<f64> = result.x.iter().skip(n_v).copied().collect();

    // Analytical: u_x(y) = 4*y*(1-y), u_y = 0, p(x) = 8*mu*(1-x)
    let mut vel_l2_err = 0.0;
    let mut vel_l2_norm = 0.0;
    let mut pres_l2_err = 0.0;
    let mut pres_l2_norm = 0.0;

    // Sample at interior velocity DOFs (skip boundary)
    for scalar_dof in 0..n_scalar_dofs {
        let x = dof_x[scalar_dof];
        let y = dof_y[scalar_dof];

        let ux_exact = 4.0 * y * (1.0 - y);
        let uy_exact: f64 = 0.0;

        let ux_h = u_sol[scalar_dof * 2];
        let uy_h = u_sol[scalar_dof * 2 + 1];

        vel_l2_err += (ux_h - ux_exact).powi(2) + (uy_h - uy_exact).powi(2);
        vel_l2_norm += ux_exact.powi(2) + uy_exact.powi(2);

        // Pressure error at vertex DOFs only (P1)
        if scalar_dof < n_nodes {
            let p_exact = 8.0 * mu * (1.0 - x);
            let p_h = p_sol[scalar_dof];
            pres_l2_err += (p_h - p_exact).powi(2);
            pres_l2_norm += p_exact.powi(2);
        }
    }

    let vel_rel = (vel_l2_err / vel_l2_norm).sqrt();
    let pres_rel = (pres_l2_err / pres_l2_norm.max(1e-30)).sqrt();

    println!("\n--- Results ---");
    println!("Velocity relative L2 error: {:.6e}", vel_rel);
    println!("Pressure relative L2 error: {:.6e}", pres_rel);

    assert!(
        vel_rel < 1e-4,
        "Poiseuille velocity relative L2 error too large: {vel_rel:.6e} (threshold: 1e-4)"
    );
    assert!(
        pres_rel < 1e-2,
        "Poiseuille pressure relative L2 error too large: {pres_rel:.6e} (threshold: 1e-2)"
    );

    // VTK output (velocity at vertices only)
    let mut writer = sonium_fem::io::vtk::VtuWriter::new(&mesh);
    {
        let mut node_ux = vec![0.0; n_nodes];
        let mut node_uy = vec![0.0; n_nodes];
        let mut node_p = vec![0.0; n_nodes];

        for i in 0..n_nodes {
            node_ux[i] = u_sol[i * 2];
            node_uy[i] = u_sol[i * 2 + 1];
            node_p[i] = p_sol[i];
        }

        writer.add_point_scalar("ux", &node_ux);
        writer.add_point_scalar("uy", &node_uy);
        writer.add_point_scalar("pressure", &node_p);

        let output_path = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
            .join("stokes_poiseuille.vtu");
        writer
            .write(&output_path)
            .expect("write Stokes Poiseuille VTU");
        println!("\nWrote {}", output_path.display());
    }

    println!("\n=== Done ===");
}

/// Apply Dirichlet BCs directly to a monolithic FemSystem.
///
/// For each (dof, value) pair:
/// 1. Correct RHS for column elimination
/// 2. Zero out constrained rows/columns
/// 3. Set diagonal to 1, RHS to prescribed value
fn apply_dirichlet_manual(system: &mut FemSystem<f64>, dofs: &[usize], values: &[f64]) {
    use std::collections::HashMap;

    let bc_map: HashMap<usize, f64> = dofs.iter().copied().zip(values.iter().copied()).collect();

    // First pass: correct RHS
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row)
            && let Some(&val) = bc_map.get(&col)
        {
            system.rhs[row] -= system.matrix.values[i] * val;
        }
    }

    // Second pass: filter + add identity
    let mut new_rows = Vec::with_capacity(system.matrix.nnz());
    let mut new_cols = Vec::with_capacity(system.matrix.nnz());
    let mut new_vals = Vec::with_capacity(system.matrix.nnz());

    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row) && !bc_map.contains_key(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_vals.push(system.matrix.values[i]);
        }
    }

    for (&dof, &val) in &bc_map {
        new_rows.push(dof);
        new_cols.push(dof);
        new_vals.push(1.0);
        system.rhs[dof] = val;
        system.dirichlet_dofs.insert(dof);
    }

    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_vals;
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_stokes_poiseuille();
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
