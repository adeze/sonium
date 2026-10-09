//! Backward-facing step benchmark — steady flow
//!
//! A classic test for NS solvers: flow enters a narrow channel and
//! encounters a sudden expansion. A recirculation zone forms behind
//! the step. The reattachment length x_r is the primary diagnostic.
//!
//! Setup:
//!   Channel: [0, 30] × [0, 1], step at x=1, step height = 0.5
//!   Inlet: parabolic u(y) at x=0, y ∈ [0.5, 1.0]
//!   Walls: no-slip on all solid surfaces
//!   Outlet: do-nothing (natural BC)
//!   Re = 100 (based on step height and max inlet velocity)
//!
//! Reference: Armaly et al. (1983) reattachment length x_r/h ≈ 6 at Re=100

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::backward_facing_step_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, compute_vorticity_2d, solve_picard,
};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

fn main() {
    println!("=== Backward-Facing Step (Re=100) ===\n");

    let x_max = 30.0;
    let channel_h = 1.0;
    let step_x = 1.0;
    let step_h = 0.5;
    let _inlet_h = channel_h - step_h; // height of inlet channel
    let u_max = 1.0;
    let re = 100.0; // based on step height and max velocity
    let mu = u_max * step_h / re;

    let nx = 150;
    let ny = 20;

    let mesh = backward_facing_step_triangles(x_max, channel_h, step_x, step_h, nx, ny);
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );
    println!(
        "DOFs: {} velocity + {} pressure\n",
        v_space.num_global_dofs(&mesh),
        p_space.num_global_dofs(&mesh)
    );

    // --- BCs ---
    let tol = x_max / nx as f64 * 0.6;

    // Inlet: x=0, y ∈ [step_h, channel_h] — parabolic
    let inlet_dofs =
        locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| x < tol && y > step_h - tol);
    // Walls: all solid boundaries (bottom, top, step face)
    let wall_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        y < tol || (channel_h - y) < tol || (x < step_x + tol && y < step_h + tol) // step block
    });

    let mut bc_dofs: Vec<usize> = inlet_dofs;
    bc_dofs.extend(&wall_dofs);
    bc_dofs.sort();
    bc_dofs.dedup();

    let sh = step_h;
    let ch = channel_h;
    let bc_fn = move |x: f64, y: f64, _z: f64| -> Vec<f64> {
        if x < tol && y > sh - tol {
            // Parabolic inlet in the upper portion [step_h, channel_h]
            let y_local = y - sh;
            let h_inlet = ch - sh;
            let u = 4.0 * u_max * y_local * (h_inlet - y_local) / (h_inlet * h_inlet);
            vec![u, 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };

    let config = NavierStokesConfig {
        mu,
        rho: 1.0,
        max_picard_iterations: 40,
        picard_tolerance: 1e-6,
        ..Default::default()
    };

    println!("Solving (Re={re})...");
    let result = solve_picard(&mesh, &v_space, &p_space, &config, &bc_fn, &bc_dofs, |_| {
        vec![0.0, 0.0]
    });

    println!(
        "  {} in {} iterations (residual: {:.2e})",
        if result.converged {
            "Converged"
        } else {
            "Not converged"
        },
        result.iterations,
        result.residual
    );

    // --- Detect reattachment point ---
    // Reattachment: where u changes sign on the bottom wall (y ≈ 0, x > step_x)
    let wall_tol = channel_h / ny as f64;
    let mut reattachment_x = 0.0_f64;
    let mut prev_u = -1.0;
    let mut wall_points: Vec<(f64, f64)> = Vec::new();

    for (ni, node) in mesh.nodes.iter().enumerate() {
        if node.y < wall_tol && node.x > step_x {
            let u = result.velocity[2 * ni];
            wall_points.push((node.x, u));
            if prev_u < 0.0 && u >= 0.0 {
                reattachment_x = node.x;
            }
            prev_u = u;
        }
    }

    let x_r = reattachment_x - step_x;
    let x_r_normalised = x_r / step_h;

    println!("\n--- Results ---");
    println!("Reattachment point: x = {:.2}", reattachment_x);
    println!("x_r / h = {:.2} (expected ~6.0 for Re=100)", x_r_normalised);

    // VTK output
    let ux: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[2 * i])
        .collect();
    let uy: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[2 * i + 1])
        .collect();
    let uz = vec![0.0; mesh.num_nodes()];
    let vorticity = compute_vorticity_2d(&mesh, &result.velocity);

    let mut writer = VtuWriter::new(&mesh);
    writer.add_point_vector("velocity", &ux, &uy, &uz);
    writer.add_point_scalar("pressure", &result.pressure);
    writer.add_cell_scalar("vorticity", &vorticity);

    let vel_mag: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| (ux[i].powi(2) + uy[i].powi(2)).sqrt())
        .collect();
    writer.add_point_scalar("velocity_magnitude", &vel_mag);

    match writer.write("backward_step.vtu") {
        Ok(_) => println!("\nWrote backward_step.vtu"),
        Err(e) => println!("\nVTK error: {e}"),
    }

    println!("\n=== Benchmark Complete ===");
}
