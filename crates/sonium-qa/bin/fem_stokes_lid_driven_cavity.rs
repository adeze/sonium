//! Stokes flow — 2D Lid-Driven Cavity with Taylor-Hood P2-P1
//!
//! Solves steady Stokes on [0,1]^2:
//!   - Top wall (y=1): u = (1, 0)  (lid moves right)
//!   - Other walls:    u = (0, 0)  (no-slip)
//!   - Pressure pinned at one point (bottom-left corner)
//!
//! For Stokes (Re → 0), the primary vortex center is at approximately
//! (x, y) ≈ (0.5, 0.76) and the flow is symmetric about x = 0.5.

use sonium_fem::assembly;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::assembly::mixed::assemble_mixed;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::forms::MassForm;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::mesh::topology::MeshTopology;
use sonium_fem::physics::fluid::stokes::forms::{StokesDivergenceForm, StokesViscousForm};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

use sonium_solvers::{
    AmgConfig, BlockPreconditionerConfig, FgmresConfig, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, fgmres_preconditioned,
};
use sonium_xem::io::binary_output_dir;
use std::collections::HashMap;

fn test_stokes_lid_driven_cavity() {
    env_logger::init();
    println!("=== Stokes Lid-Driven Cavity (Taylor-Hood P2-P1) ===\n");

    let mu = 1.0;
    let n = 16;

    // --- 1. Mesh ---
    let mesh = unit_square_triangles(n);
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

    // Pressure mass for Schur complement
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
    let tol = 1e-10;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    for sd in 0..n_scalar {
        let x = dof_x[sd];
        let y = dof_y[sd];
        let on_boundary =
            x.abs() < tol || (x - 1.0).abs() < tol || y.abs() < tol || (y - 1.0).abs() < tol;

        if on_boundary {
            if (y - 1.0).abs() < tol && x > tol && x < 1.0 - tol {
                // Lid: u_x = 1, u_y = 0 (exclude corners for compatibility)
                bc_map.insert(sd * 2, 1.0);
            } else {
                // No-slip walls: u_x = 0
                bc_map.insert(sd * 2, 0.0);
            }
            // u_y = 0 on all walls
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }

    // Pin pressure at bottom-left corner (node 0) to remove nullspace
    bc_map.insert(n_v, 0.0);

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

    assert!(
        result.converged,
        "Lid-driven cavity FGMRES did not converge (iters: {}, residual: {:.2e})",
        result.iterations, result.residual,
    );

    // --- 7. Post-process ---
    let u_sol: Vec<f64> = result.x.iter().take(n_v).copied().collect();
    let p_sol: Vec<f64> = result.x.iter().skip(n_v).copied().collect();

    // Check max velocity is bounded
    let max_vel = u_sol
        .chunks(2)
        .map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt())
        .fold(0.0_f64, f64::max);
    println!("Max velocity magnitude: {:.6}", max_vel);
    assert!(
        max_vel > 0.0,
        "Lid-driven cavity max velocity is zero — solution is trivial"
    );
    assert!(
        max_vel < 2.0,
        "Lid-driven cavity max velocity too large: {max_vel:.6} (threshold: 2.0 for unit lid velocity)"
    );

    // Find vortex center y-coordinate: zero-crossing of u_x on vertical centerline x=0.5
    let h = 1.0 / (n as f64);
    let mut centerline_pts: Vec<(f64, f64)> = Vec::new();
    for sd in 0..n_scalar {
        if (dof_x[sd] - 0.5).abs() < h / 2.0 + tol {
            centerline_pts.push((dof_y[sd], u_sol[sd * 2]));
        }
    }
    centerline_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    centerline_pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);

    // Find zero-crossing where u_x goes from negative to positive
    let mut vortex_y_approx = 0.5;
    for w in centerline_pts.windows(2) {
        if w[0].1 < 0.0 && w[1].1 > 0.0 {
            // Linear interpolation
            let t = -w[0].1 / (w[1].1 - w[0].1);
            vortex_y_approx = w[0].0 + t * (w[1].0 - w[0].0);
            break;
        }
    }

    println!("\n--- Results ---");
    println!(
        "Vortex center y (u_x zero-crossing at x=0.5): {:.4}",
        vortex_y_approx
    );
    println!("  (Stokes reference: ~0.76)");

    println!("\nVelocity u_x along x=0.5 (vertical centerline):");
    let mut centerline: Vec<(f64, f64)> = Vec::new();
    for sd in 0..n_scalar {
        if (dof_x[sd] - 0.5).abs() < 1.0 / (2.0 * n as f64) + tol {
            centerline.push((dof_y[sd], u_sol[sd * 2]));
        }
    }
    centerline.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for &(y, ux) in &centerline {
        println!("  y={:.4}  u_x={:.6}", y, ux);
    }

    // Check x-symmetry: u_x(0.5-dx, y) ≈ u_x(0.5+dx, y)
    let mut max_asym = 0.0_f64;
    for sd in 0..n_nodes {
        let x = dof_x[sd];
        let y = dof_y[sd];
        if x < 0.5 && y > tol && y < 1.0 - tol {
            // Find symmetric point
            let mirror_x = 1.0 - x;
            for sd2 in 0..n_nodes {
                if (dof_x[sd2] - mirror_x).abs() < tol && (dof_y[sd2] - y).abs() < tol {
                    let diff = (u_sol[sd * 2] - u_sol[sd2 * 2]).abs();
                    max_asym = max_asym.max(diff);
                }
            }
        }
    }
    println!(
        "\nSymmetry check: max |u_x(x,y) - u_x(1-x,y)| = {:.2e}",
        max_asym
    );

    // VTK output
    let mut node_ux = vec![0.0; n_nodes];
    let mut node_uy = vec![0.0; n_nodes];
    let mut node_p = vec![0.0; n_nodes];
    for i in 0..n_nodes {
        node_ux[i] = u_sol[i * 2];
        node_uy[i] = u_sol[i * 2 + 1];
        node_p[i] = p_sol[i];
    }

    let mut writer = sonium_fem::io::vtk::VtuWriter::new(&mesh);
    writer.add_point_scalar("ux", &node_ux);
    writer.add_point_scalar("uy", &node_uy);
    writer.add_point_scalar("pressure", &node_p);
    let speed: Vec<f64> = (0..n_nodes)
        .map(|i| (node_ux[i] * node_ux[i] + node_uy[i] * node_uy[i]).sqrt())
        .collect();
    writer.add_point_scalar("speed", &speed);

    let output_path =
        binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME")).join("stokes_cavity.vtu");
    writer.write(&output_path).expect("write Stokes cavity VTU");
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
        test_stokes_lid_driven_cavity();
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
