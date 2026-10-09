//! Von Kármán vortex street — unsteady flow past cylinder
//!
//! At Re > ~47 the wake behind a cylinder becomes unstable and
//! periodic vortex shedding occurs. This benchmark validates the
//! BDF2 time stepper by detecting the shedding frequency and computing
//! the Strouhal number (expected St ≈ 0.16–0.20 for Re = 100–200).
//!
//! Setup: same as DFG 2D-2 benchmark but with higher Re and unsteady.
//!   Channel: [0, 2.2] × [0, 0.41]
//!   Cylinder: (0.2, 0.2), r = 0.05
//!   Re = 100 (based on diameter and mean inlet velocity)
//!   BDF2 time integration, Δt = 0.005

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::{VtuWriter, write_pvd};
use sonium_fem::mesh::generators::channel_with_cylinder_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, compute_drag_lift, compute_vorticity_2d, solve_bdf_step,
};
use sonium_fem::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState, TimeMonitor};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

fn main() {
    println!("=== Von Kármán Vortex Street (Re=100, BDF2) ===\n");

    // --- Parameters ---
    let channel_l = 2.2;
    let channel_h = 0.41;
    let cyl_x = 0.2;
    let cyl_y = 0.2;
    let cyl_r = 0.05;
    let u_mean = 1.0;
    let u_max = 1.5 * u_mean;
    let nu = 2.0 * cyl_r * u_mean / 100.0; // Re = U*D/nu = 100
    let dt = 0.005;
    let t_end = 8.0;
    let n_steps = (t_end / dt) as usize;

    println!("Re = {:.0}", u_mean * 2.0 * cyl_r / nu);
    println!("Δt = {dt}, T_end = {t_end}, steps = {n_steps}");

    // --- Mesh ---
    let nx = 88;
    let ny = 16;
    let mesh = channel_with_cylinder_triangles(
        0.0, channel_l, 0.0, channel_h, nx, ny, cyl_x, cyl_y, cyl_r,
    );
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );
    println!("DOFs: {} velocity + {} pressure\n", n_v, n_p);

    // --- BCs ---
    let tol = channel_l / nx as f64 * 0.6;
    let h = channel_h;

    let inlet_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, _y, _z| x < tol);
    let wall_dofs = locate_dofs_geometrical(&mesh, &v_space, |_x, y, _z| y < tol || (h - y) < tol);
    let cyl_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        ((x - cyl_x).powi(2) + (y - cyl_y).powi(2)).sqrt() < cyl_r * 1.5
    });

    let mut bc_dofs: Vec<usize> = inlet_dofs;
    bc_dofs.extend(&wall_dofs);
    bc_dofs.extend(&cyl_dofs);
    bc_dofs.sort();
    bc_dofs.dedup();

    let bc_fn = move |x: f64, y: f64, _z: f64| -> Vec<f64> {
        if x < tol {
            vec![4.0 * u_max * y * (h - y) / (h * h), 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };

    // --- NS config ---
    let config = NavierStokesConfig {
        mu: nu,
        rho: 1.0,
        max_picard_iterations: 1, // semi-implicit: one Oseen solve per time step
        picard_tolerance: 1e-6,
        ..Default::default()
    };
    let bdf = BdfConfig::bdf2(dt, n_steps).with_output(100);

    // --- Initial condition: impulsive start ---
    let mut state = BdfState::from_rest(n_v, n_p);
    // Set inlet BC on initial condition
    for &dof in &bc_dofs {
        let ni = dof / 2;
        let comp = dof % 2;
        if ni < mesh.num_nodes() {
            let p = &mesh.nodes[ni];
            let bc = bc_fn(p.x, p.y, p.z);
            if comp < bc.len() {
                state.velocity[dof] = bc[comp];
                state.velocity_prev[dof] = bc[comp];
            }
        }
    }

    let mut monitor = TimeMonitor::new();
    let mut vtu_files: Vec<(f64, String)> = Vec::new();

    // --- Time loop ---
    println!("Time stepping...");
    for step in 0..n_steps {
        let success = solve_bdf_step(
            &mesh, &v_space, &p_space, &config, &bdf, &mut state, &bc_fn, &bc_dofs,
        );

        if !success {
            println!("  Step {step}: solver failed, aborting");
            break;
        }

        // Monitor drag/lift
        let (drag, lift) = compute_drag_lift(&mesh, &state.velocity, &state.pressure, config.mu, 4);
        let d = 2.0 * cyl_r;
        let c_d = 2.0 * drag / (u_mean * u_mean * d);
        let c_l = 2.0 * lift / (u_mean * u_mean * d);
        let ke: f64 = state.velocity.iter().map(|v| 0.5 * v * v).sum::<f64>() / n_v as f64;
        monitor.record(state.time, c_d, c_l, ke);

        // Progress output
        if (step + 1) % 100 == 0 || step == 0 {
            println!(
                "  t = {:.3}, C_D = {:.4}, C_L = {:.4}, KE = {:.6}",
                state.time, c_d, c_l, ke
            );
        }

        // VTK output
        if bdf.output_interval > 0 && (step + 1) % bdf.output_interval == 0 {
            let filename = format!("vk_{:05}.vtu", step + 1);
            let ux: Vec<f64> = (0..mesh.num_nodes())
                .map(|i| state.velocity[2 * i])
                .collect();
            let uy: Vec<f64> = (0..mesh.num_nodes())
                .map(|i| state.velocity[2 * i + 1])
                .collect();
            let uz = vec![0.0; mesh.num_nodes()];
            let vorticity = compute_vorticity_2d(&mesh, &state.velocity);

            let mut writer = VtuWriter::new(&mesh);
            writer.add_point_vector("velocity", &ux, &uy, &uz);
            writer.add_point_scalar("pressure", &state.pressure);
            writer.add_cell_scalar("vorticity", &vorticity);
            let _ = writer.write(&filename);
            vtu_files.push((state.time, filename));
        }
    }

    // --- Results ---
    println!("\n--- Results ---");
    println!("Final time: {:.3}", state.time);
    println!("Steps completed: {}", state.step);

    if let Some(st) = monitor.strouhal_number(2.0 * cyl_r, u_mean) {
        println!("Strouhal number: {st:.4} (expected ~0.16-0.20 for Re=100)");
    } else {
        println!("Strouhal number: could not detect (flow may not have reached periodic state)");
    }

    // Write PVD collection
    if !vtu_files.is_empty() {
        let pvd_entries: Vec<(f64, &str)> =
            vtu_files.iter().map(|(t, f)| (*t, f.as_str())).collect();
        match write_pvd("von_karman.pvd", &pvd_entries) {
            Ok(_) => println!("\nWrote von_karman.pvd ({} timesteps)", vtu_files.len()),
            Err(e) => println!("\nPVD error: {e}"),
        }
    }

    println!("\n=== Benchmark Complete ===");
}
