//! Flow past cylinder validation — DFG 2D-1 setup
//!
//! Standard benchmark from Schäfer & Turek (1996) for incompressible NS
//! flow around a circular cylinder in a channel.
//!
//! Setup (DFG benchmark 2D-1):
//!   Channel: [0, 2.2] × [0, 0.41]
//!   Cylinder: center (0.2, 0.2), radius 0.05
//!   Inlet: parabolic u(y) = 4 U_m y(H-y)/H², v=0, U_m = 0.3 m/s
//!   Walls: no-slip u=0
//!   Outlet: do-nothing (natural BC)
//!   Re = 20 (steady, DFG 2D-1) or Re = 100 (unsteady, DFG 2D-2)
//!
//! Reference values (DFG 2D-1, steady, Re=20):
//!   C_D ≈ 5.5795  (drag coefficient)
//!   C_L ≈ 0.0106  (lift coefficient)
//!   Δp  ≈ 0.1174  (pressure difference front-back)

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::channel_with_cylinder_triangles;
use sonium_fem::mesh::interpolation::interpolate_scalar_at_point;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, compute_drag_lift, compute_drag_lift_variational, compute_vorticity_2d,
    solve_picard,
};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_xem::io::binary_output_dir;

#[derive(Clone, Copy, Debug)]
struct DfgMetrics {
    c_d: f64,
    c_l: f64,
    delta_p: f64,
}

fn run_ns_flow_past_cylinder(nx: usize, ny: usize, write_vtk: bool) -> DfgMetrics {
    println!("=== Flow Past Cylinder — DFG 2D-1 Validation ===\n");

    // --- Parameters (DFG 2D-1: Re=20, steady) ---
    let channel_l = 2.2;
    let channel_h = 0.41;
    let cyl_x = 0.2;
    let cyl_y = 0.2;
    let cyl_r = 0.05;
    let u_mean = 0.2; // mean inlet velocity
    let u_max = 1.5 * u_mean; // max parabolic velocity
    let re_target = 20.0;
    let nu = u_mean * 2.0 * cyl_r / re_target; // kinematic viscosity
    let re = u_mean * 2.0 * cyl_r / nu; // Re based on diameter and mean velocity
    let mu = nu; // ρ = 1

    println!(
        "Channel: [{}, {}] × [{}, {}]",
        0.0, channel_l, 0.0, channel_h
    );
    println!("Cylinder: center ({cyl_x}, {cyl_y}), radius {cyl_r}");
    println!("Re = {re:.1} (based on diameter and mean velocity)\n");

    // --- Mesh ---
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
    println!(
        "DOFs: {} velocity + {} pressure = {}\n",
        n_v,
        n_p,
        n_v + n_p
    );

    // --- Boundary conditions ---
    let tol = channel_l / nx as f64 * 0.6;

    // Inlet (marker 1): parabolic profile
    let inlet_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, _y, _z| x < tol);
    // Walls (marker 3): no-slip
    let wall_dofs = locate_dofs_geometrical(&mesh, &v_space, |_x, y, _z| {
        y < tol || (channel_h - y) < tol
    });
    // Cylinder (marker 4): no-slip
    let cyl_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        let dist = ((x - cyl_x).powi(2) + (y - cyl_y).powi(2)).sqrt();
        (dist - cyl_r).abs() < 1.0e-10
    });

    let mut bc_dofs: Vec<usize> = inlet_dofs;
    bc_dofs.extend(&wall_dofs);
    bc_dofs.extend(&cyl_dofs);
    bc_dofs.sort();
    bc_dofs.dedup();
    println!("BC DOFs: {}", bc_dofs.len());

    let h = channel_h;
    let bc_fn = move |x: f64, y: f64, _z: f64| -> Vec<f64> {
        if x < tol {
            // Parabolic inlet
            let u = 4.0 * u_max * y * (h - y) / (h * h);
            vec![u, 0.0]
        } else {
            vec![0.0, 0.0] // no-slip
        }
    };

    // --- Solve ---
    let config = NavierStokesConfig {
        mu,
        rho: 1.0,
        max_picard_iterations: 100,
        picard_tolerance: 1e-6,
        picard_relaxation: 1.0,
        use_direct_solver: false,
        linear_solver_max_iterations: 5000,
        linear_solver_tolerance: 1e-9,
        ..Default::default()
    };

    println!("\nSolving (Re={re:.1})...");
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
    assert!(
        result.converged,
        "DFG cylinder Picard solve did not converge after {} iterations (residual {:.2e})",
        result.iterations, result.residual
    );

    // --- Post-processing ---

    // Drag and lift (on cylinder boundary, marker 4)
    let (drag, lift) = compute_drag_lift_variational(
        &mesh,
        &result.velocity,
        &result.pressure,
        mu,
        1.0,
        4,
        |_| vec![0.0, 0.0],
    );
    let (pressure_drag, pressure_lift) =
        compute_drag_lift(&mesh, &result.velocity, &result.pressure, 0.0, 4);
    // Drag/lift coefficients: C_D = 2F_D/(ρ U² D), C_L = 2F_L/(ρ U² D)
    let d = 2.0 * cyl_r;
    let c_d = 2.0 * drag / (u_mean * u_mean * d);
    let c_l = 2.0 * lift / (u_mean * u_mean * d);

    // Pressure difference (front - back of cylinder)
    let p_front = interpolate_scalar_at_point(&mesh, &result.pressure, [cyl_x - cyl_r, cyl_y, 0.0])
        .expect("DFG front pressure probe must lie on the mesh boundary");
    let p_back = interpolate_scalar_at_point(&mesh, &result.pressure, [cyl_x + cyl_r, cyl_y, 0.0])
        .expect("DFG back pressure probe must lie on the mesh boundary");
    let delta_p = p_front - p_back;

    println!("\n--- Results ---");
    println!("C_D = {c_d:.4}  (ref DFG 2D-1: 5.5795)");
    println!("C_L = {c_l:.4}  (ref DFG 2D-1: 0.0106)");
    println!(
        "      pressure-only boundary stress: C_D={:.4}, C_L={:.4}",
        2.0 * pressure_drag / (u_mean * u_mean * d),
        2.0 * pressure_lift / (u_mean * u_mean * d)
    );
    println!("Δp  = {delta_p:.4}  (ref DFG 2D-1: 0.1174)");

    assert!(
        c_d.is_finite() && c_l.is_finite() && delta_p.is_finite(),
        "Coarse DFG smoke quantities must be finite: C_D={c_d:.4}, C_L={c_l:.4}, delta_p={delta_p:.4}",
    );

    // Velocity statistics
    let u_max_num = (0..mesh.num_nodes())
        .map(|i| (result.velocity[2 * i].powi(2) + result.velocity[2 * i + 1].powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    println!("|u|_max = {u_max_num:.4}");
    assert!(
        u_max_num.is_finite(),
        "Velocity magnitude should be finite, got {u_max_num}"
    );

    // --- VTK output ---
    let ux: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[2 * i])
        .collect();
    let uy: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[2 * i + 1])
        .collect();
    let uz = vec![0.0; mesh.num_nodes()];
    let vorticity = compute_vorticity_2d(&mesh, &result.velocity);

    if write_vtk {
        let mut writer = VtuWriter::new(&mesh);
        writer.add_point_vector("velocity", &ux, &uy, &uz);
        writer.add_point_scalar("pressure", &result.pressure);
        writer.add_cell_scalar("vorticity", &vorticity);

        let vel_mag: Vec<f64> = (0..mesh.num_nodes())
            .map(|i| (ux[i].powi(2) + uy[i].powi(2)).sqrt())
            .collect();
        writer.add_point_scalar("velocity_magnitude", &vel_mag);
        let path = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
            .join(format!("cylinder_flow_{nx}x{ny}.vtu"));
        writer.write(&path).expect("write DFG cylinder VTU");
        println!("\nWrote {}", path.display());
    }

    println!("\n=== DFG Validation Complete ===");
    DfgMetrics { c_d, c_l, delta_p }
}

fn test_ns_flow_past_cylinder() {
    let coarse_metrics = run_ns_flow_past_cylinder(66, 12, false);
    let fine_metrics = run_ns_flow_past_cylinder(99, 18, true);
    let c_d_reference = 5.5795;
    let c_l_reference = 0.0106;
    let delta_p_reference = 0.1174;
    let drag_error = |metrics: DfgMetrics| (metrics.c_d - c_d_reference).abs() / c_d_reference;
    let pressure_error =
        |metrics: DfgMetrics| (metrics.delta_p - delta_p_reference).abs() / delta_p_reference;

    println!("DFG refinement: coarse={coarse_metrics:?}, fine={fine_metrics:?}");
    assert!(
        drag_error(fine_metrics) < drag_error(coarse_metrics),
        "DFG drag did not improve under refinement"
    );
    assert!(
        pressure_error(fine_metrics) < pressure_error(coarse_metrics),
        "DFG pressure drop did not improve under refinement"
    );
    assert!(
        drag_error(fine_metrics) < 0.07,
        "Fine-grid DFG drag error {:.2}% exceeds 7%",
        100.0 * drag_error(fine_metrics)
    );
    assert!(
        pressure_error(fine_metrics) < 0.044,
        "Fine-grid DFG pressure error {:.2}% exceeds 4.4%",
        100.0 * pressure_error(fine_metrics)
    );
    assert!(
        (fine_metrics.c_l - c_l_reference).abs() < 0.01,
        "Fine-grid DFG lift differs from reference by more than 0.01"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_ns_flow_past_cylinder();
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
