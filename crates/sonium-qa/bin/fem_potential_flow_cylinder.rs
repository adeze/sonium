//! QA: FEM potential flow around a cylinder.
//!
//! Validates Laplace equation (∇²φ = 0) with mixed boundary conditions
//! on an annular domain. Uses existing `ConductionForm` (diffusion operator
//! is mathematically identical to the Laplace operator).
//!
//! Exact solution: φ = U∞ (r + a²/r) cos θ
//! Tangential velocity on cylinder: v_θ = −2 U∞ sin θ

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::annular_mesh_triangles;
use sonium_fem::physics::thermal::heat_conduction::ConductionForm;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

/// Exact potential: φ = U∞ (r + a²/r) cos θ
fn exact_potential(x: f64, y: f64, u_inf: f64, a: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    let cos_theta = x / r;
    u_inf * (r + a * a / r) * cos_theta
}

/// Exact tangential velocity on cylinder surface: v_θ = −2 U∞ sin θ
fn exact_tangential_velocity(theta: f64, u_inf: f64) -> f64 {
    -2.0 * u_inf * theta.sin()
}

fn max_tangential_velocity_error(
    mesh: &sonium_fem::mesh::Mesh,
    potential: &[f64],
    a: f64,
    u_inf: f64,
    n_angular: usize,
) -> f64 {
    let mut max_error = 0.0_f64;
    let dtheta = 2.0 * std::f64::consts::PI / n_angular as f64;
    for node in &mesh.nodes {
        let r = (node.x * node.x + node.y * node.y).sqrt();
        if (r - a).abs() > 0.01 * a {
            continue;
        }
        let theta = node.y.atan2(node.x);
        let exact_velocity = exact_tangential_velocity(theta, u_inf);
        let mut phi_plus = None;
        let mut phi_minus = None;
        for (index, other) in mesh.nodes.iter().enumerate() {
            let other_radius = (other.x * other.x + other.y * other.y).sqrt();
            if (other_radius - a).abs() > 0.01 * a {
                continue;
            }
            let other_theta = other.y.atan2(other.x);
            let wrapped_difference = |target: f64| {
                (other_theta - target + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                    - std::f64::consts::PI
            };
            if wrapped_difference(theta + dtheta).abs() < 0.1 * dtheta {
                phi_plus = Some(potential[index]);
            }
            if wrapped_difference(theta - dtheta).abs() < 0.1 * dtheta {
                phi_minus = Some(potential[index]);
            }
        }
        let (Some(phi_plus), Some(phi_minus)) = (phi_plus, phi_minus) else {
            panic!("missing angular neighbours at theta={theta}");
        };
        let numerical_velocity = (phi_plus - phi_minus) / (2.0 * dtheta * a);
        max_error = max_error.max((numerical_velocity - exact_velocity).abs());
    }
    max_error
}

fn test_potential_flow_cylinder() {
    let a = 0.5; // cylinder radius
    let b = 3.0; // far-field radius
    let u_inf = 1.0;
    let n_radial = 12;
    let n_angular = 48;

    let mesh = annular_mesh_triangles(0.0, 0.0, a, b, n_radial, n_angular);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    // Assemble Laplace stiffness matrix (ConductionForm with k=1.0)
    let form = ConductionForm::from_conductivity(1.0);
    let matrix = assemble_bilinear(&form, &space, &space, &mesh, 3);
    let rhs = vec![0.0; n_dofs];

    let mut system = FemSystem::new(matrix, rhs);

    // Apply Dirichlet on outer boundary (marker 2): φ = U∞ * x * (1 + a²/b²)
    let factor = 1.0 + a * a / (b * b);
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        2, // outer boundary marker
        &|x, _y, _z| u_inf * x * factor,
        None,
    );

    // Inner boundary (marker 1) is natural Neumann = 0 (no penetration)
    // No action needed — homogeneous Neumann is the default for Laplace.

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );

    assert!(
        result.converged,
        "Solver should converge: residual={:.2e}",
        result.residual
    );

    // Compare φ at sample interior points
    let sample_points = [(0.8, 0.0), (1.0, 0.5), (1.5, 0.0), (2.0, 1.0), (2.5, -0.5)];

    let mut max_phi_error: f64 = 0.0;
    for &(sx, sy) in &sample_points {
        // Find nearest node
        let mut nearest = (0, f64::MAX);
        for (i, node) in mesh.nodes.iter().enumerate() {
            let dist = (node.x - sx).hypot(node.y - sy);
            if dist < nearest.1 {
                nearest = (i, dist);
            }
        }
        let nearest_node = &mesh.nodes[nearest.0];
        let fem_phi = result.x[nearest.0];
        // Compare fields at the same physical coordinate.  Comparing the
        // nearest-node FEM value to the requested off-node sample location
        // folds interpolation distance into the PDE error (and dominated the
        // old 12.5% maximum).
        let exact_phi = exact_potential(nearest_node.x, nearest_node.y, u_inf, a);
        let err = (fem_phi - exact_phi).abs();
        max_phi_error = max_phi_error.max(err);
    }

    let max_vel_error = max_tangential_velocity_error(&mesh, &result.x, a, u_inf, n_angular);

    println!(
        "Potential flow cylinder: max_phi_error={:.2e}, max_vel_error={:.2e}",
        max_phi_error, max_vel_error
    );

    assert!(
        max_phi_error < 0.02,
        "Potential error too large: {:.2e}",
        max_phi_error
    );
    assert!(
        max_vel_error < 0.06,
        "Velocity error too large: {:.2e}",
        max_vel_error
    );
}

fn test_potential_flow_convergence() {
    let a = 0.5;
    let b = 3.0;
    let u_inf = 1.0;

    let resolutions = [(8, 32), (12, 48), (16, 64), (24, 96)];
    let mut max_errors = Vec::new();
    let mut velocity_errors = Vec::new();

    for &(nr, na) in &resolutions {
        let mesh = annular_mesh_triangles(0.0, 0.0, a, b, nr, na);
        let space = H1Space::new(PolynomialDegree::P1);
        let n_dofs = space.num_global_dofs(&mesh);

        let form = ConductionForm::from_conductivity(1.0);
        let matrix = assemble_bilinear(&form, &space, &space, &mesh, 3);
        let rhs = vec![0.0; n_dofs];

        let mut system = FemSystem::new(matrix, rhs);

        let factor = 1.0 + a * a / (b * b);
        apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            2,
            &|x, _y, _z| u_inf * x * factor,
            None,
        );

        let csr = system.matrix.to_csr();
        let result = gmres(
            &csr,
            &system.rhs,
            &GmresConfig {
                max_iterations: 2000,
                restart: 200,
                tolerance: 1e-12,
                print_interval: 0,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                exact_residual_check_interval: 0,
            },
        );
        assert!(result.converged);

        // Compute max error over all interior nodes (not on boundaries)
        let mut max_err: f64 = 0.0;
        for (i, node) in mesh.nodes.iter().enumerate() {
            let r = (node.x * node.x + node.y * node.y).sqrt();
            // Skip boundary nodes (they have exact Dirichlet values)
            if (r - a).abs() < 0.01 * a || (r - b).abs() < 0.01 * b {
                continue;
            }
            let fem_phi = result.x[i];
            let exact_phi = exact_potential(node.x, node.y, u_inf, a);
            let err = (fem_phi - exact_phi).abs();
            max_err = max_err.max(err);
        }
        max_errors.push(max_err);
        let velocity_error = max_tangential_velocity_error(&mesh, &result.x, a, u_inf, na);
        velocity_errors.push(velocity_error);
        println!(
            "  nr={nr} na={na}: potential={max_err:.2e}, tangential velocity={velocity_error:.2e}"
        );
    }

    // Check monotonic improvement
    println!("Potential convergence errors: {max_errors:?}");
    println!("Velocity convergence errors: {velocity_errors:?}");
    assert!(
        max_errors[2] < max_errors[1] && max_errors[1] < max_errors[0],
        "Error should decrease with mesh refinement"
    );
    for (index, pair) in max_errors.windows(2).enumerate() {
        let refinement_ratio = resolutions[index + 1].0 as f64 / resolutions[index].0 as f64;
        let order = (pair[0] / pair[1]).ln() / refinement_ratio.ln();
        assert!(
            order > 1.0,
            "Potential-flow convergence order {order:.3} is below 1"
        );
    }
    assert!(
        max_errors.last().is_some_and(|error| *error < 0.006),
        "Finest potential-flow interior error exceeds 0.6%: {max_errors:?}"
    );
    assert!(
        velocity_errors.windows(2).all(|pair| pair[1] < pair[0]),
        "Tangential-velocity error did not decrease: {velocity_errors:?}"
    );
    assert!(
        velocity_errors.last().is_some_and(|error| *error < 0.03),
        "Finest tangential-velocity error exceeds 3%: {velocity_errors:?}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_potential_flow_cylinder();
        test_potential_flow_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
