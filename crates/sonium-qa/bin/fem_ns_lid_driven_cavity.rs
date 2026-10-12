//! Lid-driven cavity smoke check — incompressible Navier-Stokes
//!
//! Standard benchmark for NS solvers. A square cavity [0,1]² with:
//!   - No-slip walls on bottom, left, right (u=0)
//!   - Moving lid on top (u=[1,0])
//!
//! Solved at Re = 100 on a tiny QA mesh and compared qualitatively to reference data
//! from Ghia, Ghia & Shin (1982), J. Comp. Phys. 48, pp. 387–411.
//!
//! The reference data gives u-velocity along the vertical centerline
//! (x=0.5) and v-velocity along the horizontal centerline (y=0.5).

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, compute_vorticity_2d, solve_picard,
};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::analytical::{ghia_lid_cavity_re100_u, interp_sorted_samples};
use sonium_xem::io::binary_output_dir;

fn interpolate_centerline(samples: &[(f64, f64)], y: f64) -> Option<f64> {
    interp_sorted_samples(samples, y)
}

fn run_cavity(re: f64, n: usize) -> f64 {
    println!("--- Re = {re}, mesh {n}×{n} ---");

    let mesh = unit_square_triangles(n);
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    println!(
        "  DOFs: {} velocity + {} pressure = {}",
        n_v,
        p_space.num_global_dofs(&mesh),
        n_v + p_space.num_global_dofs(&mesh)
    );

    let mu = 1.0 / re;
    let tol = 1e-10;

    // All boundary DOFs: walls + lid
    let all_boundary_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        x < tol || (1.0 - x) < tol || y < tol || (1.0 - y) < tol
    });

    // BC: u=[1,0] on lid (y=1), u=[0,0] on walls
    let bc_fn = |_x: f64, y: f64, _z: f64| -> Vec<f64> {
        if (1.0 - y) < tol {
            vec![1.0, 0.0] // lid velocity
        } else {
            vec![0.0, 0.0] // no-slip walls
        }
    };

    let config = NavierStokesConfig {
        mu,
        rho: 1.0,
        max_picard_iterations: 50,
        picard_tolerance: 1e-6,
        picard_relaxation: 0.7,
        use_direct_solver: false,
        linear_solver_max_iterations: 3000,
        linear_solver_tolerance: 1e-10,
        ..Default::default()
    };

    let result = solve_picard(
        &mesh,
        &v_space,
        &p_space,
        &config,
        &bc_fn,
        &all_boundary_dofs,
        |_| vec![0.0, 0.0],
    );

    assert!(
        result.converged,
        "Lid-driven cavity did not converge: iterations={}, residual={:.2e}",
        result.iterations, result.residual
    );
    println!("  Converged in {} iterations", result.iterations);

    // Extract u-velocity along vertical centerline (x ≈ 0.5)
    let h = 1.0 / n as f64;
    let center_tol = h * 0.6;
    let mut centerline: Vec<(f64, f64)> = Vec::new();
    for (ni, node) in mesh.nodes.iter().enumerate() {
        if (node.x - 0.5).abs() < center_tol {
            centerline.push((node.y, result.velocity[2 * ni]));
        }
    }
    centerline.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Compare with Ghia reference data (Re=100 only)
    let reference_error = if (re - 100.0).abs() < 1.0 {
        println!("\n  Comparison with Ghia et al. (Re=100):");
        println!(
            "  {:>8} {:>10} {:>10} {:>10}",
            "y", "u_num", "u_ref", "error"
        );
        let mut max_err = 0.0_f64;
        for &(y_ref, u_ref) in &ghia_lid_cavity_re100_u() {
            let u_num = interpolate_centerline(&centerline, y_ref)
                .unwrap_or_else(|| panic!("centerline does not bracket y={y_ref}"));
            let err = (u_num - u_ref).abs();
            max_err = max_err.max(err);
            println!(
                "  {:>8.4} {:>10.4} {:>10.4} {:>10.4}",
                y_ref, u_num, u_ref, err
            );
        }
        println!("  Max error vs Ghia: {:.4}", max_err);

        max_err
    } else {
        panic!("no lid-cavity reference configured for Re={re}");
    };

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

    let path = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
        .join(format!("cavity_re{}_n{n}.vtu", re as u32));
    writer.write(&path).expect("write lid-cavity VTU");
    println!("  Wrote {}", path.display());
    println!();
    reference_error
}

fn test_ns_lid_driven_cavity() {
    println!("=== Lid-Driven Cavity Smoke Check ===\n");

    let coarse_error = run_cavity(100.0, 16);
    let fine_error = run_cavity(100.0, 40);
    println!("Ghia max-error refinement: n=16 {coarse_error:.4} -> n=40 {fine_error:.4}");
    assert!(
        fine_error < 0.6 * coarse_error,
        "Lid-cavity error did not decrease by at least 40% under refinement"
    );
    assert!(
        fine_error < 0.03,
        "Fine-grid Ghia centerline max error {fine_error:.4} exceeds 3% of lid speed"
    );

    println!("=== Smoke Check Complete ===");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_ns_lid_driven_cavity();
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
