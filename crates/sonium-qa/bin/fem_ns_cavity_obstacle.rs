//! Native lid-driven cavity with centered square obstacle smoke checks.
//!
//! Extension of the standard cavity benchmark with a solid square block
//! in the center. This creates more complex recirculation patterns and
//! secondary vortices around the obstacle.
//!
//! This is a smoke case: bounded fields, converged Picard updates, and a
//! recirculation signature. It does not establish field, mass, or force
//! accuracy against an independent reference or a mesh-refinement limit.
//!
//! Setup:
//!   Domain: [0,1]² with square obstacle [0.3,0.7] × [0.3,0.7] removed
//!   Lid: u=[1,0] at y=1
//!   Walls + obstacle: no-slip u=0
//!   Re = 100, 400

// Rust guideline compliant 2026-02-21

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::Mesh;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    LinearSolverCpuPreconditionerKind, NavierStokesConfig, compute_vorticity_2d, solve_picard,
};
use sonium_fem::spaces::{H1Space, VectorH1Space};
use std::collections::HashMap;

/// Create a unit square mesh with a centered square hole.
fn cavity_with_obstacle(n: usize, obs_min: f64, obs_max: f64) -> Mesh {
    let full = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, n, n);
    let mut mesh = Mesh::new(2);
    let mut node_map = HashMap::new();

    for elem in &full.elements {
        let verts = elem.vertices();
        let cx: f64 = verts.iter().map(|&v| full.nodes[v].x).sum::<f64>() / verts.len() as f64;
        let cy: f64 = verts.iter().map(|&v| full.nodes[v].y).sum::<f64>() / verts.len() as f64;

        let in_obstacle = cx > obs_min && cx < obs_max && cy > obs_min && cy < obs_max;
        if !in_obstacle {
            let remapped_vertices = verts
                .iter()
                .map(|&old_idx| {
                    *node_map
                        .entry(old_idx)
                        .or_insert_with(|| mesh.add_node(full.nodes[old_idx]))
                })
                .collect();
            mesh.add_element(elem.element_type, remapped_vertices);
        }
    }

    mesh.detect_boundaries();
    mesh
}

fn run_cavity_obstacle(re: f64, n: usize) {
    println!("--- Re = {re}, mesh {n}×{n} ---");

    let obs_min = 0.3;
    let obs_max = 0.7;
    let mesh = cavity_with_obstacle(n, obs_min, obs_max);
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    println!(
        "  {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    let tol = 1e-10;
    let obs_tol = 1.0 / n as f64;

    // All boundaries (outer walls + obstacle surface)
    let bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        // Outer walls
        x < tol || (1.0 - x) < tol || y < tol || (1.0 - y) < tol ||
        // Obstacle surface (near the obstacle edges)
        (x > obs_min - obs_tol && x < obs_max + obs_tol &&
         y > obs_min - obs_tol && y < obs_max + obs_tol &&
         (x < obs_min + obs_tol || x > obs_max - obs_tol ||
          y < obs_min + obs_tol || y > obs_max - obs_tol))
    });

    let bc_fn = move |_x: f64, y: f64, _z: f64| -> Vec<f64> {
        if (1.0 - y) < tol {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };

    let config = NavierStokesConfig {
        mu: 1.0 / re,
        rho: 1.0,
        max_picard_iterations: if re <= 100.0 { 20 } else { 25 },
        picard_tolerance: 1e-4,
        picard_relaxation: if re <= 100.0 { 1.0 } else { 0.7 },
        use_direct_solver: false,
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
        linear_solver_max_iterations: 500,
        linear_solver_tolerance: 1e-5,
        ..Default::default()
    };

    let result = solve_picard(&mesh, &v_space, &p_space, &config, &bc_fn, &bc_dofs, |_| {
        vec![0.0, 0.0]
    });

    println!(
        "  {} in {} iterations",
        if result.converged {
            "Converged"
        } else {
            "Not converged"
        },
        result.iterations
    );
    assert!(
        result.converged,
        "Cavity-obstacle Picard solve failed at Re={re}: iterations={}, residual={:.2e}",
        result.iterations, result.residual
    );

    // --- Assertions ---
    assert!(
        result.iterations > 0 && result.residual.is_finite(),
        "Picard iteration should perform finite nonlinear updates at Re={re} (residual: {:.2e}, iterations: {})",
        result.residual,
        result.iterations,
    );

    // Velocity must be bounded (no blow-up). Lid speed is 1.0, so max should be reasonable.
    let vel_max = (0..mesh.num_nodes())
        .map(|i| (result.velocity[2 * i].powi(2) + result.velocity[2 * i + 1].powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    assert!(
        vel_max < 10.0,
        "Max velocity magnitude {vel_max:.4} exceeds bound (possible blow-up) at Re={re}",
    );
    assert!(
        vel_max > 0.0,
        "Max velocity is zero -- solver produced trivial solution at Re={re}",
    );

    // Vorticity for secondary vortex detection
    let vorticity = compute_vorticity_2d(&mesh, &result.velocity);
    let _max_vort = vorticity.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let min_vort = vorticity.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_vort_pos = vorticity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Vorticity range: [{min_vort:.2}, {max_vort_pos:.2}]");

    // Count recirculation zones (sign changes in vorticity along mid-height)
    let h_tol = 1.0 / n as f64;
    let mut mid_vorticity: Vec<(f64, f64)> = Vec::new();
    for (ei, elem) in mesh.elements.iter().enumerate() {
        let verts = elem.vertices();
        let cy: f64 = verts.iter().map(|&v| mesh.nodes[v].y).sum::<f64>() / verts.len() as f64;
        let cx: f64 = verts.iter().map(|&v| mesh.nodes[v].x).sum::<f64>() / verts.len() as f64;
        if (cy - 0.5).abs() < h_tol && ei < vorticity.len() {
            mid_vorticity.push((cx, vorticity[ei]));
        }
    }
    mid_vorticity.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut sign_changes = 0;
    for i in 1..mid_vorticity.len() {
        if mid_vorticity[i].1 * mid_vorticity[i - 1].1 < 0.0 {
            sign_changes += 1;
        }
    }
    println!("  Vorticity sign changes at y=0.5: {sign_changes} (indicates multiple vortices)");
    assert!(
        sign_changes > 0,
        "Cavity-obstacle flow at Re={re} has no resolved recirculation signature"
    );

    // VTK output
    let ux: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[2 * i])
        .collect();
    let uy: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[2 * i + 1])
        .collect();
    let uz = vec![0.0; mesh.num_nodes()];

    let mut writer = VtuWriter::new(&mesh);
    writer.add_point_vector("velocity", &ux, &uy, &uz);
    writer.add_point_scalar("pressure", &result.pressure);
    writer.add_cell_scalar("vorticity", &vorticity);

    let filename = format!("cavity_obstacle_re{}.vtu", re as u32);
    match writer.write(&filename) {
        Ok(_) => println!("  Wrote {filename}"),
        Err(e) => println!("  VTK error: {e}"),
    }
    let case_id = format!("fem.ns-cavity-obstacle-smoke-re{}.v1", re as u32);
    let record = sonium_qa::QaResult::passed(&case_id)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .seed(format!("unit-square:obstacle-0.3-0.7:Re{re}:mesh{n}:P1-P1"))
        .reference("smoke invariants only; no independent field, mass, or force accuracy reference")
        .reference_checksum("contract:cavity-obstacle-smoke:v1")
        .formulation("native Picard FEM smoke; no numerical accuracy or mesh convergence claim")
        .metric("numerical_solve_executed", 1.0)
        .minimum("numerical_solve_executed", 1.0)
        .metric("accuracy_reference_executed", 0.0)
        .maximum("accuracy_reference_executed", 0.0)
        .metric("solver_converged", 1.0)
        .minimum("solver_converged", 1.0)
        .metric("maximum_speed", vel_max)
        .maximum("maximum_speed", 10.0)
        .metric("recirculation_sign_changes", sign_changes as f64)
        .minimum("recirculation_sign_changes", 1.0);
    println!(
        "QA_RESULT: {}",
        record.to_json_line().expect("valid smoke record")
    );
    println!();
}

fn test_ns_cavity_obstacle() {
    println!("=== Lid-Driven Cavity with Square Obstacle ===\n");
    run_cavity_obstacle(100.0, 16);
    run_cavity_obstacle(400.0, 20);
    println!("=== Native Smoke Checks Complete ===");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_ns_cavity_obstacle();
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
