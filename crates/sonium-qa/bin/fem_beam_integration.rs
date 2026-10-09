use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection, assemble_beam_stiffness};
use sonium_solvers::{GmresConfig, gmres};

fn test_beam_cantilever_deflection() {
    // Cantilever beam: L=1.0m, fixed at x=0, vertical load P at x=L.
    // Analytical solution: delta = P * L^3 / (3 * E * I)

    let l_total = 1.0;
    let p_load = 1000.0;
    let e_mod = 200e9; // steel
    let width = 0.01;
    let height = 0.02;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    let n_elements = 10;
    let dx = l_total / n_elements as f64;

    let elements: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            let x1 = i as f64 * dx;
            let x2 = (i + 1) as f64 * dx;
            BeamElement2D::new([i, i + 1], x1, 0.0, x2, 0.0, section)
        })
        .collect();

    let n_nodes = n_elements + 1;
    let n_dofs = n_nodes * 3;

    let triplet = assemble_beam_stiffness(&elements, n_dofs);
    let mut csr = triplet.to_csr();

    // Apply Boundary Conditions (fixed at x=0): DOFs 0, 1, 2
    for dof in 0..3 {
        // Zero out row
        let range = csr.row_range(dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        // Add 1.0 to diagonal
        let diag_pos = csr.row_entries(dof).position(|(j, _)| j == dof);
        if let Some(idx) = diag_pos {
            let global_idx = csr.row_ptrs[dof] + idx;
            csr.values[global_idx] = 1.0;
        }
    }

    // RHS: point load at the tip (last node, vertical DOF)
    let mut rhs = vec![0.0; n_dofs];
    let last_node_v_dof = (n_nodes - 1) * 3 + 1;
    rhs[last_node_v_dof] = -p_load; // downward

    // Solving
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

    let result = gmres(&csr, &rhs, &config);
    assert!(result.converged, "Gmres should converge");

    // Vertical displacement at tip
    let delta = result.x[last_node_v_dof].abs();
    let exact_delta = p_load * l_total.powi(3) / (3.0 * e_mod * i_inertia);

    let error = (delta - exact_delta).abs() / exact_delta;
    println!(
        "FEM delta: {:.6e}, Exact: {:.6e}, Error: {:.2}%",
        delta,
        exact_delta,
        error * 100.0
    );

    // With 10 elements, error should be very small (< 0.1%)
    assert!(error < 0.001, "Error {} should be < 0.001", error);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_beam_cantilever_deflection();
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
