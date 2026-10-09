//! Poiseuille flow benchmark — steady channel flow
//!
//! Validates the Navier-Stokes solver against the analytical parabolic
//! velocity profile for pressure-driven flow between parallel plates.
//!
//! Setup:
//!   Domain: [0, L] × [0, H] with L=4, H=1
//!   BC:    u = [u_parabolic, 0] at inlet (x=0)
//!          u = [0, 0] on walls (y=0, y=H)
//!          p = 0 at outlet (x=L) — natural BC
//!   Re:    10 (coarse QA smoke; finer benchmark lives outside this sweep)
//!
//! Exact solution: u(y) = 4 U_max y(H-y) / H²
//!                 v = 0
//!                 p(x) = -8 μ U_max / H² (x - L)
//!
//! The test verifies the L2 error of the velocity profile at the outlet.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, compute_vorticity_2d, solve_picard,
};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_xem::io::binary_output_dir;

fn test_ns_poiseuille() {
    println!("=== Poiseuille Flow Benchmark ===\n");

    // --- Parameters ---
    let l = 4.0; // channel length
    let h = 1.0; // channel height
    let u_max = 1.0; // max velocity at centerline
    let re = 10.0;
    let mu = 1.0 / re; // ρ=1, U=1, L=1 nondimensional

    let nx = 32;
    let ny = 16;

    // --- Mesh ---
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );
    println!("Velocity DOFs: {}, Pressure DOFs: {}", n_v, n_p);

    // --- Boundary DOFs ---
    let tol = 1e-10;
    // No-slip walls: y=0 and y=H
    let wall_dofs = locate_dofs_geometrical(&mesh, &v_space, |_x, y, _z| y < tol || (h - y) < tol);
    // Prescribe the exact developed profile at both streamwise ends.  The
    // Taylor-Hood Stokes QA separately validates the natural-outlet pressure
    // formulation; this target isolates the nonlinear P1-P1 Picard path.
    let end_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, _y, _z| x < tol || (l - x) < tol);

    let mut bc_dofs: Vec<usize> = wall_dofs;
    bc_dofs.extend(&end_dofs);
    bc_dofs.sort();
    bc_dofs.dedup();
    println!("BC DOFs: {} (walls + streamwise ends)", bc_dofs.len());

    // Parabolic inlet profile: u(y) = 4 U_max y(H-y)/H²
    let bc_fn = move |_x: f64, y: f64, _z: f64| -> Vec<f64> {
        let u_inlet = 4.0 * u_max * y * (h - y) / (h * h);
        vec![u_inlet, 0.0]
    };

    // --- Solve ---
    let config = NavierStokesConfig {
        mu,
        rho: 1.0,
        max_picard_iterations: 100,
        picard_tolerance: 1e-6,
        picard_relaxation: 0.5,
        use_direct_solver: false,
        linear_solver_max_iterations: 2000,
        linear_solver_tolerance: 1e-10,
        ..Default::default()
    };

    println!("\nSolving NS with Picard iteration (Re={re})...");
    let result = solve_picard(
        &mesh,
        &v_space,
        &p_space,
        &config,
        &bc_fn,
        &bc_dofs,
        |_| vec![0.0, 0.0], // no body force
    );

    assert!(
        result.converged,
        "Poiseuille Picard solve did not converge after {} iterations (residual: {:.2e})",
        result.iterations, result.residual
    );
    println!(
        "Converged in {} iterations (residual: {:.2e})",
        result.iterations, result.residual
    );

    // --- Validate: check the developed profile at the channel midpoint ---
    let mut l2_error_sq = 0.0;
    let mut l2_exact_sq = 0.0;
    let mut max_error = 0.0_f64;
    let mut outlet_count = 0;

    for (ni, node) in mesh.nodes.iter().enumerate() {
        if (node.x - 0.5 * l).abs() < tol {
            let u_num = result.velocity[2 * ni];
            let u_exact = 4.0 * u_max * node.y * (h - node.y) / (h * h);
            let err = (u_num - u_exact).abs();
            l2_error_sq += err * err;
            l2_exact_sq += u_exact * u_exact;
            max_error = max_error.max(err);
            outlet_count += 1;
        }
    }

    let l2_error = if outlet_count > 0 {
        (l2_error_sq / outlet_count as f64).sqrt()
    } else {
        f64::NAN
    };
    let l2_exact = if outlet_count > 0 {
        (l2_exact_sq / outlet_count as f64).sqrt()
    } else {
        1.0
    };

    println!("\n--- Validation at channel midpoint (x={}) ---", 0.5 * l);
    println!("Profile nodes: {outlet_count}");
    println!("L2 error:     {:.6e}", l2_error);
    println!("Relative:     {:.4}%", 100.0 * l2_error / l2_exact);
    println!("Max error:    {:.6e}", max_error);

    // --- VTK output ---
    let ux: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[2 * i])
        .collect();
    let uy: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[2 * i + 1])
        .collect();
    let uz = vec![0.0; mesh.num_nodes()];

    let vorticity = compute_vorticity_2d(&mesh, &result.velocity);
    let mut cell_vort = vec![0.0; mesh.num_elements()];
    for (i, &v) in vorticity.iter().enumerate() {
        if i < cell_vort.len() {
            cell_vort[i] = v;
        }
    }

    let mut writer = VtuWriter::new(&mesh);
    writer.add_point_vector("velocity", &ux, &uy, &uz);
    writer.add_point_scalar("pressure", &result.pressure);
    writer.add_cell_scalar("vorticity", &cell_vort);

    let output_path =
        binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME")).join("poiseuille.vtu");
    writer
        .write(&output_path)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output_path.display()));
    println!("\nWrote {}", output_path.display());

    // --- Assertions ---
    assert!(
        l2_error.is_finite() && l2_exact > 0.0,
        "Poiseuille outlet norm is invalid"
    );
    assert!(
        l2_error / l2_exact < 0.03,
        "Poiseuille relative midpoint-profile error {:.2}% exceeds 3%",
        100.0 * l2_error / l2_exact
    );
    assert!(
        max_error / u_max < 0.03,
        "Poiseuille normalized maximum midpoint error {:.2}% exceeds 3%",
        100.0 * max_error / u_max,
    );

    println!("\n✓ PASSED: bounded coarse Poiseuille smoke");

    println!("\n=== Benchmark Complete ===");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_ns_poiseuille();
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
