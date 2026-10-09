//! 3D channel-flow smoke test (rectangular duct)
//!
//! Exercises the 3D Navier-Stokes Picard path with an inf-sup-stable
//! Taylor-Hood (P2/P1) pair on a small rectangular duct.  Equal-order
//! residual stabilisation is deliberately disabled: its PSPG pressure block
//! is neither needed nor consistent with this validation target.
//!
//! Setup:
//!   Domain: [0, 4] × [0, 1] × [0, 1]
//!   Inlet: parabolic u(y,z), v=w=0
//!   Walls: no-slip on y=0,1 and z=0,1
//!   Outlet: natural BC
//!   Re = 10

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::box_mesh_tetrahedra;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    LinearSolverCpuPreconditionerKind, NavierStokesConfig, solve_picard,
};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_xem::io::binary_output_dir;

fn test_ns_3d_channel() {
    println!("=== 3D Channel Flow (Re=10) ===\n");

    let nx = 8;
    let cross_section_subdivisions = 4;
    let mesh = box_mesh_tetrahedra(
        0.0,
        4.0,
        0.0,
        1.0,
        0.0,
        1.0,
        nx,
        cross_section_subdivisions,
        cross_section_subdivisions,
    );
    let v_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    println!(
        "Mesh: {} nodes, {} tets",
        mesh.num_nodes(),
        mesh.num_elements()
    );
    println!("DOFs: {} vel + {} pres = {}\n", n_v, n_p, n_v + n_p);

    let tol = 1e-10;
    let x_tol = 4.0 / nx as f64 * 0.6;

    // Exact developed profile at both streamwise ends.  The natural-outlet
    // formulation is covered by the Taylor-Hood Stokes QA; this target
    // isolates the 3D nonlinear Taylor-Hood path.
    let ends = locate_dofs_geometrical(&mesh, &v_space, |x, _y, _z| {
        x < x_tol || (x - 4.0).abs() < tol
    });
    // Walls: y=0,1 and z=0,1
    let walls = locate_dofs_geometrical(&mesh, &v_space, |_x, y, z| {
        y < tol || (1.0 - y) < tol || z < tol || (1.0 - z) < tol
    });

    let mut bc_dofs = ends;
    bc_dofs.extend(&walls);
    bc_dofs.sort();
    bc_dofs.dedup();

    let bc_fn = |x: f64, y: f64, z: f64| -> Vec<f64> {
        if x < x_tol || (x - 4.0).abs() < tol {
            vec![16.0 * y * (1.0 - y) * z * (1.0 - z), 0.0, 0.0]
        } else {
            vec![0.0, 0.0, 0.0]
        }
    };

    let mu = 0.1;
    let config = NavierStokesConfig {
        mu, // Re = U*L/ν ≈ 1*1/0.1 = 10
        rho: 1.0,
        max_picard_iterations: 20,
        picard_tolerance: 1e-6,
        use_direct_solver: false,
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::IlutSchur,
        linear_solver_max_iterations: 3000,
        linear_solver_tolerance: 1e-10,
        use_residual_stabilisation: false,
        ..Default::default()
    };

    println!("Solving...");
    let result = solve_picard(
        &mesh,
        &v_space,
        &p_space,
        &config,
        &bc_fn,
        &bc_dofs,
        move |position| {
            let y = position[1];
            let z = position[2];
            // For u=16 y(1-y)z(1-z), convection vanishes and
            // -mu*Laplacian(u)=32*mu*[y(1-y)+z(1-z)].
            vec![32.0 * mu * (y * (1.0 - y) + z * (1.0 - z)), 0.0, 0.0]
        },
    );

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
        "3D channel Picard solve did not converge after {} iterations (residual {:.2e})",
        result.iterations, result.residual
    );

    assert!(
        result.velocity.iter().all(|value| value.is_finite())
            && result.pressure.iter().all(|value| value.is_finite()),
        "3D channel solution contains NaN or infinity"
    );

    // Check the developed velocity profile on the interior mid-plane.
    let out_tol = tol;
    let mut max_err = 0.0_f64;
    let mut max_transverse_velocity = 0.0_f64;
    let mut midpoint_count = 0;
    for (ni, node) in mesh.nodes.iter().enumerate() {
        if (node.x - 2.0).abs() < out_tol {
            let u_num = result.velocity[3 * ni];
            let u_exact = 16.0 * node.y * (1.0 - node.y) * node.z * (1.0 - node.z);
            max_err = max_err.max((u_num - u_exact).abs());
            max_transverse_velocity = max_transverse_velocity
                .max(result.velocity[3 * ni + 1].abs())
                .max(result.velocity[3 * ni + 2].abs());
            midpoint_count += 1;
        }
    }

    println!("\n--- Results ---");
    println!("Midpoint-plane nodes: {midpoint_count}");
    println!("Max streamwise error: {max_err:.4}");
    println!("Max transverse speed: {max_transverse_velocity:.3e}");

    // --- Assertions ---
    assert!(
        result.iterations > 0 && result.residual.is_finite(),
        "Picard iteration did not produce a finite residual"
    );
    assert!(
        midpoint_count > 0,
        "3D channel mesh has no nodes on the x=2 midpoint plane"
    );
    assert!(
        max_err < 0.02,
        "Max velocity error at the 3D channel midpoint {max_err:.4} exceeds 0.02",
    );
    assert!(
        max_transverse_velocity < 0.01,
        "Spurious transverse velocity {max_transverse_velocity:.3e} exceeds 0.01",
    );

    // Pressure drop
    let p_inlet: f64 = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.x < out_tol)
        .map(|(i, _)| result.pressure[i])
        .sum::<f64>();
    let p_outlet: f64 = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| (n.x - 4.0).abs() < out_tol)
        .map(|(i, _)| result.pressure[i])
        .sum::<f64>();
    let inlet_count = mesh.nodes.iter().filter(|n| n.x < out_tol).count();
    let outlet_count2 = mesh
        .nodes
        .iter()
        .filter(|n| (n.x - 4.0).abs() < out_tol)
        .count();
    assert!(
        inlet_count > 0 && outlet_count2 > 0,
        "3D channel mesh is missing inlet or outlet pressure samples"
    );
    let dp = p_inlet / inlet_count as f64 - p_outlet / outlet_count2 as f64;
    println!("Pressure drop: {dp:.6}");
    assert!(
        dp.abs() < 0.03,
        "Manufactured zero-pressure-gradient flow has spurious pressure drop {dp:.6}"
    );

    // VTK
    let ux: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[3 * i])
        .collect();
    let uy: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[3 * i + 1])
        .collect();
    let uz: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[3 * i + 2])
        .collect();

    let mut writer = VtuWriter::new(&mesh);
    writer.add_point_vector("velocity", &ux, &uy, &uz);
    writer.add_point_scalar("pressure", &result.pressure);

    let output_path =
        binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME")).join("channel_3d.vtu");
    writer.write(&output_path).expect("write 3D channel VTU");
    println!("\nWrote {}", output_path.display());

    println!("\n=== Benchmark Complete ===");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_ns_3d_channel();
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
