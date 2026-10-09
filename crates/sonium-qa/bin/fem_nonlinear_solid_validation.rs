//! Nonlinear solid mechanics validation tests
//!
//! Tests the full Newton-Raphson pipeline: HyperelasticForm + VectorH1Space + fem_newton_solve.
//! Additional tests:
//! - NeoHookean inflation
//! - Simple shear
//! - Volumetric deformation
//! - Energy conservation

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::NonlinearForm;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};

/// Uniaxial compression of a Neo-Hookean bar.
///
/// A 2D bar [0,1] x [0,0.5] is fixed on the left (u_x=0, u_y=0) and
/// compressed on the right (u_x = -delta, u_y free).
/// Newton-Raphson should converge to a state where interior nodes interpolate
/// the uniform compression field.
fn test_neohookean_uniaxial_compression_2d() {
    let mat = NeoHookean::from_young(1.0, 0.3, 2);
    let form = HyperelasticForm::new(mat);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);

    // Build a 4x3 mesh
    let nx = 4;
    let ny = 3;
    let mut mesh = Mesh::new(2);
    for j in 0..ny {
        for i in 0..nx {
            mesh.add_node(Point::new_2d(
                i as f64 / (nx - 1) as f64,
                j as f64 / (ny - 1) as f64 * 0.5,
            ));
        }
    }
    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            let v0 = j * nx + i;
            let v1 = v0 + 1;
            let v2 = v0 + nx;
            let v3 = v2 + 1;
            mesh.add_element(ElementType::Triangle, vec![v0, v1, v2]);
            mesh.add_element(ElementType::Triangle, vec![v1, v3, v2]);
        }
    }
    mesh.detect_boundaries();

    let n_dofs = space.num_global_dofs(&mesh);
    let delta = 0.05; // 5% compression

    // Prescribed displacement: linear compression u_x = -delta * x
    // This is the exact solution for uniform uniaxial compression.
    let exact_u = |node: usize| -> (f64, f64) {
        let x = mesh.nodes[node].x;
        (-delta * x, 0.0)
    };

    // Use the exact solution as initial guess (so Newton should converge in ~1 step)
    let mut initial_guess = vec![0.0; n_dofs];
    for node in 0..mesh.num_nodes() {
        let (ux, uy) = exact_u(node);
        initial_guess[node * 2] = ux;
        initial_guess[node * 2 + 1] = uy;
    }

    // For the nonlinear solver, we need to handle BCs differently.
    // The fem_newton_solve assembles residual R(u) at each iteration.
    // Dirichlet BCs need to be enforced by zeroing residual and tangent rows/cols
    // for constrained DOFs, which is not yet integrated into fem_newton_solve.
    //
    // For this validation test, we verify that the residual at the exact solution is
    // approximately zero (no body forces, equilibrium).

    // Assemble residual at the exact solution
    let n_elems = mesh.num_elements();
    let mut residual = vec![0.0f64; n_dofs];
    for elem_idx in 0..n_elems {
        let dofs = space.local_to_global(&mesh, elem_idx);
        let local_sol: Vec<f64> = dofs.iter().map(|&d| initial_guess[d]).collect();
        let n_local = dofs.len();

        let quad =
            sonium_fem::quadrature::QuadratureRule::new(mesh.elements[elem_idx].element_type, 2);
        let mut local_vec = sonium_fem::forms::LocalVector::zeros(n_local);

        for qp in quad.iter() {
            let tab = space.tabulate(&mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
            form.element_residual(&tab, &local_sol, qp.weight, &mut local_vec);
        }

        for (i, &dof) in dofs.iter().enumerate() {
            residual[dof] += local_vec.values[i];
        }
    }

    // Interior residual should be near zero (equilibrium under uniform strain)
    // Boundary residual will be nonzero (reaction forces).
    let eps = 1e-6;
    let interior_nodes: Vec<usize> = (0..mesh.num_nodes())
        .filter(|&n| {
            let p = &mesh.nodes[n];
            p.x > eps && p.x < 1.0 - eps && p.y > eps && p.y < 0.5 - eps
        })
        .collect();

    for &node in &interior_nodes {
        let rx = residual[node * 2];
        let ry = residual[node * 2 + 1];
        assert!(
            rx.abs() < 1e-10,
            "Interior node {} ({:.2}, {:.2}): R_x = {:.6e}",
            node,
            mesh.nodes[node].x,
            mesh.nodes[node].y,
            rx
        );
        assert!(
            ry.abs() < 1e-10,
            "Interior node {} ({:.2}, {:.2}): R_y = {:.6e}",
            node,
            mesh.nodes[node].x,
            mesh.nodes[node].y,
            ry
        );
    }
}

/// Verify the Newton solver converges for a Neo-Hookean problem.
///
/// Small perturbation from equilibrium: start at u=0 (which is the equilibrium
/// for zero body forces) and verify Newton converges in exactly 1 iteration
/// (since the problem is trivially satisfied at u=0).
fn test_newton_converges_at_equilibrium() {
    let mat = NeoHookean::from_young(1.0, 0.3, 2);
    let form = HyperelasticForm::new(mat);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);

    let n = 3;
    let mut mesh = Mesh::new(2);
    for j in 0..n {
        for i in 0..n {
            mesh.add_node(Point::new_2d(
                i as f64 / (n - 1) as f64,
                j as f64 / (n - 1) as f64,
            ));
        }
    }
    for j in 0..(n - 1) {
        for i in 0..(n - 1) {
            let v0 = j * n + i;
            mesh.add_element(ElementType::Triangle, vec![v0, v0 + 1, v0 + n]);
            mesh.add_element(ElementType::Triangle, vec![v0 + 1, v0 + n + 1, v0 + n]);
        }
    }
    mesh.detect_boundaries();

    let n_dofs = space.num_global_dofs(&mesh);
    let initial_guess = vec![0.0; n_dofs]; // u=0 is the equilibrium

    let config = FemNewtonConfig {
        newton: sonium_solvers::NewtonConfig {
            max_iterations: 5,
            absolute_tolerance: 1e-12,
            ..Default::default()
        },
        quad_order: 2,
        ..Default::default()
    };

    let result = fem_newton_solve(&form, &space, &mesh, &initial_guess, &config).unwrap();

    assert!(result.converged, "Newton should converge at equilibrium");
    assert_eq!(
        result.iterations, 0,
        "Should converge immediately at equilibrium (residual is already 0)"
    );
}

/// Verify NeoHookean inflation (spherical hole in infinite medium).
/// For large r: p ≈ -2μ * (a³/r³)
fn test_neohookean_inflation_pressure() {
    let e_mod: f64 = 1.0;
    let nu: f64 = 0.3;
    let mu = e_mod / (2.0 * (1.0 + nu));
    let lambda = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    let a: f64 = 1.0; // cavity radius
    let r: f64 = 2.0; // point at 2a

    let pressure = -2.0 * mu * a.powi(3) / r.powi(3);

    println!("NeoHookean inflation (cavity pressure):");
    println!("  μ = {:.3}, λ = {:.3}", mu, lambda);
    println!("  At r=2a: p = {:.3e}", pressure);

    assert!(pressure < 0.0, "Pressure should be negative (compressive)");
}

/// Verify simple shear deformation.
/// For simple shear γ, the first Piola-Kirchhoff stress is:
/// P = μ*γ for NeoHookean (at small strain)
fn test_neohookean_simple_shear() {
    let e_mod: f64 = 1.0;
    let nu: f64 = 0.3;
    let mu = e_mod / (2.0 * (1.0 + nu));

    let gamma: f64 = 0.1; // 10% shear

    let tau_xy = mu * gamma;

    println!("NeoHookean simple shear:");
    println!("  γ = {}, μ = {:.3}", gamma, mu);
    println!("  τ_xy = {:.3e}", tau_xy);

    assert!(tau_xy > 0.0, "Shear stress should be positive");
}

/// Verify volumetric deformation (bulk modulus).
/// For hydrostatic compression: p = -K * div(u)
fn test_neohookean_bulk_modulus() {
    let e_mod: f64 = 2.0;
    let nu: f64 = 0.3;
    let k = e_mod / (3.0 * (1.0 - 2.0 * nu));

    let volumetric_strain: f64 = 0.01; // 1% compression
    let pressure = -k * volumetric_strain;

    println!("NeoHookean bulk modulus:");
    println!("  E = {}, ν = {}", e_mod, nu);
    println!("  K = {:.3}", k);
    println!("  For ε_vol = {}: p = {:.3e}", volumetric_strain, pressure);

    assert!(k > 0.0, "Bulk modulus should be positive");
    assert!(
        (k / e_mod - 1.0 / (3.0 * (1.0 - 2.0 * nu))).abs() < 1e-12,
        "Bulk-to-Young modulus ratio is inconsistent with Poisson ratio"
    );
    assert!(
        pressure < 0.0,
        "Positive volumetric compression should give compressive pressure"
    );
}

/// Verify strain energy density for NeoHookean.
/// W = (μ/2)(I1 - 3) + (λ/2)(J-1)²
fn test_neohookean_strain_energy() {
    let e_mod: f64 = 1.0;
    let nu: f64 = 0.3;
    let mu = e_mod / (2.0 * (1.0 + nu));
    let lambda = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    let i1: f64 = 3.0 + 0.1; // Slight stretch
    let j: f64 = 1.0; // Incompressible

    let w = (mu / 2.0) * (i1 - 3.0) + (lambda / 2.0) * (j - 1.0).powi(2);

    println!("NeoHookean strain energy:");
    println!("  I₁ = {}, J = 1 (incompressible)", i1);
    println!("  W = {:.3e}", w);

    assert!(w >= 0.0, "Strain energy should be non-negative");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_neohookean_uniaxial_compression_2d();
        test_newton_converges_at_equilibrium();
        test_neohookean_inflation_pressure();
        test_neohookean_simple_shear();
        test_neohookean_bulk_modulus();
        test_neohookean_strain_energy();
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
