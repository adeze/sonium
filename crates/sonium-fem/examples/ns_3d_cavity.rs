//! 3D Lid-driven cavity — incompressible Navier-Stokes
//!
//! Extension of the 2D cavity benchmark to three dimensions.
//! A unit cube [0,1]³ with no-slip walls on 5 faces and a moving
//! lid on top (z=1) with velocity u=[1,0,0].
//!
//! This validates the 3D VectorH1 + Picard solver and demonstrates
//! that the same solve_picard() function works in both 2D and 3D.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::unit_cube_tetrahedra;
use sonium_fem::physics::fluid::stokes::navier_stokes::{NavierStokesConfig, solve_picard};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

fn main() {
    println!("=== 3D Lid-Driven Cavity (Re=100) ===\n");

    let n = 8; // mesh resolution (n³ subdivisions → 6n³ tetrahedra)
    let re = 100.0;
    let mu = 1.0 / re;

    let mesh = unit_cube_tetrahedra(n);
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 3); // 3D vector
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    println!(
        "Mesh: {} nodes, {} tetrahedra",
        mesh.num_nodes(),
        mesh.num_elements()
    );
    println!(
        "DOFs: {} velocity + {} pressure = {}\n",
        n_v,
        n_p,
        n_v + n_p
    );

    // All boundary DOFs
    let tol = 1e-10;
    let all_boundary = locate_dofs_geometrical(&mesh, &v_space, |x, y, z| {
        x < tol || (1.0 - x) < tol || y < tol || (1.0 - y) < tol || z < tol || (1.0 - z) < tol
    });

    // BC: u=[1,0,0] on lid (z=1), u=[0,0,0] on all other walls
    let bc_fn = |_x: f64, _y: f64, z: f64| -> Vec<f64> {
        if (1.0 - z) < tol {
            vec![1.0, 0.0, 0.0] // lid velocity
        } else {
            vec![0.0, 0.0, 0.0] // no-slip
        }
    };

    let config = NavierStokesConfig {
        mu,
        rho: 1.0,
        max_picard_iterations: 30,
        picard_tolerance: 1e-5,
        ..Default::default()
    };

    println!("Solving (Re={re})...");
    let result = solve_picard(
        &mesh,
        &v_space,
        &p_space,
        &config,
        &bc_fn,
        &all_boundary,
        |_| vec![0.0, 0.0, 0.0], // no body force
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

    // Statistics
    let vel_mag_max = (0..mesh.num_nodes())
        .map(|i| {
            let ux = result.velocity[3 * i];
            let uy = result.velocity[3 * i + 1];
            let uz = result.velocity[3 * i + 2];
            (ux * ux + uy * uy + uz * uz).sqrt()
        })
        .fold(0.0_f64, f64::max);

    let p_max = result
        .pressure
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let p_min = result
        .pressure
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    println!("\n--- Results ---");
    println!("|u|_max = {vel_mag_max:.6}");
    println!("p range: [{p_min:.6}, {p_max:.6}]");

    // Extract u_x along vertical centerline (x=0.5, y=0.5)
    let center_tol = 1.0 / n as f64;
    let mut centerline: Vec<(f64, f64)> = Vec::new();
    for (ni, node) in mesh.nodes.iter().enumerate() {
        if (node.x - 0.5).abs() < center_tol && (node.y - 0.5).abs() < center_tol {
            centerline.push((node.z, result.velocity[3 * ni]));
        }
    }
    centerline.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    if !centerline.is_empty() {
        println!("\nCenterline u_x (x=0.5, y=0.5):");
        for &(z, ux) in &centerline {
            println!("  z = {z:.3}, u_x = {ux:.6}");
        }
    }

    // VTK output
    let ux: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[3 * i])
        .collect();
    let uy: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[3 * i + 1])
        .collect();
    let uz: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| result.velocity[3 * i + 2])
        .collect();
    let vel_mag: Vec<f64> = (0..mesh.num_nodes())
        .map(|i| (ux[i].powi(2) + uy[i].powi(2) + uz[i].powi(2)).sqrt())
        .collect();

    let mut writer = VtuWriter::new(&mesh);
    writer.add_point_vector("velocity", &ux, &uy, &uz);
    writer.add_point_scalar("pressure", &result.pressure);
    writer.add_point_scalar("velocity_magnitude", &vel_mag);

    match writer.write("cavity_3d.vtu") {
        Ok(_) => println!("\nWrote cavity_3d.vtu"),
        Err(e) => println!("\nVTK error: {e}"),
    }

    println!("\n=== Benchmark Complete ===");
}
