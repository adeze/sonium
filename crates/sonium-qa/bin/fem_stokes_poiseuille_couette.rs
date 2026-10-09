//! QA: Stokes flow — 2D Poiseuille-Couette flow.
//!
//! Analytical: u_x(y) = U*y/H + (-dp/dx)/(2μ)*y*(H-y)

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

fn solve_poiseuille_couette(n: usize, u_wall: f64, dpdx: f64) -> (f64, bool) {
    let mu = 1.0;
    let h = 1.0;
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, h, n, n);

    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let n_total = n_v + n_p;

    let viscous = StokesViscousForm::new(mu);
    let div_form = StokesDivergenceForm;
    let system = assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 4);
    let mono = system.to_monolithic();

    let mp = assembly::assemble_bilinear(&MassForm, &p_space, &p_space, &mesh, 2);
    let mut schur = TripletMatrix::<f64>::with_capacity(mp.num_rows, mp.num_cols, mp.nnz());
    for i in 0..mp.nnz() {
        schur.push(mp.rows[i], mp.cols[i], mp.values[i] / mu);
    }

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

    let mut fem = FemSystem::new(mono, vec![0.0; n_total]);
    let tol = 1e-10;
    let mut bc_dofs = Vec::new();
    let mut bc_vals = Vec::new();

    for sd in 0..n_scalar {
        let x = dof_x[sd];
        let y = dof_y[sd];

        if y.abs() < tol {
            // Bottom wall
            bc_dofs.push(sd * 2);
            bc_vals.push(0.0);
            bc_dofs.push(sd * 2 + 1);
            bc_vals.push(0.0);
        } else if (y - h).abs() < tol {
            // Top wall moving
            bc_dofs.push(sd * 2);
            bc_vals.push(u_wall);
            bc_dofs.push(sd * 2 + 1);
            bc_vals.push(0.0);
        } else if x.abs() < tol {
            // Inlet: exact profile
            let ux_exact = u_wall * y / h + (-dpdx) * y * (h - y) / (2.0 * mu);
            bc_dofs.push(sd * 2);
            bc_vals.push(ux_exact);
            bc_dofs.push(sd * 2 + 1);
            bc_vals.push(0.0);
        }
    }

    // Pin pressure at one corner
    for node in 0..n_nodes {
        let x = mesh.nodes[node].x;
        let y = mesh.nodes[node].y;
        if (x - 1.0).abs() < tol && y.abs() < tol {
            bc_dofs.push(n_v + node);
            bc_vals.push(0.0);
            break;
        }
    }

    apply_dirichlet_manual(&mut fem, &bc_dofs, &bc_vals);

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

    let result = fgmres_preconditioned(
        &a_csr,
        &precond,
        &fem.rhs,
        None,
        &FgmresConfig {
            max_iterations: 500,
            restart: 50,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        },
    );

    let u_sol: Vec<f64> = result.x.iter().take(n_v).copied().collect();

    let mut vel_err = 0.0;
    let mut vel_norm = 0.0;
    for sd in 0..n_scalar {
        let y = dof_y[sd];
        let ux_exact = u_wall * y / h + (-dpdx) * y * (h - y) / (2.0 * mu);
        let ux_h = u_sol[sd * 2];
        let uy_h = u_sol[sd * 2 + 1];
        vel_err += (ux_h - ux_exact).powi(2) + uy_h.powi(2);
        vel_norm += ux_exact.powi(2);
    }

    let rel_err = (vel_err / vel_norm.max(1e-30)).sqrt();
    (rel_err, result.converged)
}

fn apply_dirichlet_manual(system: &mut FemSystem<f64>, dofs: &[usize], values: &[f64]) {
    use std::collections::HashMap;
    let bc_map: HashMap<usize, f64> = dofs.iter().copied().zip(values.iter().copied()).collect();

    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row)
            && let Some(&val) = bc_map.get(&col)
        {
            system.rhs[row] -= system.matrix.values[i] * val;
        }
    }

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

fn test_stokes_poiseuille_couette() {
    let u_wall = 1.0;
    let dpdx = -8.0; // gives parabolic contribution with max 1.0
    let (err, converged) = solve_poiseuille_couette(8, u_wall, dpdx);
    assert!(converged, "Poiseuille-Couette solver did not converge");
    println!("Poiseuille-Couette velocity relative L2 error: {:.6e}", err);
    assert!(
        err < 1e-4,
        "Poiseuille-Couette velocity error too large: {err:.6e}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_stokes_poiseuille_couette();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
