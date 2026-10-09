//! QA: FEM elasticity — Saint-Venant torsion of a rectangular bar.
//!
//! Uses the Prandtl stress-function formulation: -ΔΦ = 2Gθ, Φ=0 on boundary.
//! Exact torque for a x b rectangle (a >= b):
//! J ≈ a*b³ * [16/3 - 3.36*(b/a)*(1 - (b/a)⁴/12)]

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::torsion::{
    TorsionPoissonForm, prandtl_stress_rhs, prandtl_torque, torsion_constant_from_prandtl,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres_preconditioned};

fn test_elasticity_torsion_rectangle() {
    let a = 2.0; // half-width
    let b = 1.0; // half-height
    let g = 80e9; // shear modulus
    let theta = 0.01; // twist rate [rad/m]

    let nx = 40;
    let ny = 20;
    let mesh = rectangular_mesh_triangles(-a, a, -b, b, nx, ny);

    let space = H1Space::new(PolynomialDegree::P1);
    let _n_dofs = space.num_global_dofs(&mesh);

    let form = TorsionPoissonForm::new();
    let rhs_form = prandtl_stress_rhs(g, theta);

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let rhs = assemble_linear(&rhs_form, &space, &mesh, 1);

    let mut system = FemSystem::new(stiffness, rhs);

    // All four sides are Dirichlet Φ = 0
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 2);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 3);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 4);

    let csr = system.matrix.to_csr();
    let precond = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(
        &csr,
        &precond,
        &system.rhs,
        &GmresConfig {
            max_iterations: 5000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES+ILU should converge for torsion");

    // Integrate Φ over the domain to get torque: T = 2 ∫ Φ dA
    let phi_integral = integrate_scalar_over_mesh(&mesh, &space, &result.x);
    let _torque_fem = prandtl_torque(phi_integral);
    let j_fem = torsion_constant_from_prandtl(phi_integral, g, theta);

    // Exact torsion constant for rectangle (approximate formula)
    let ratio = b / a;
    let j_exact = a * b.powi(3) * (16.0 / 3.0 - 3.36 * ratio * (1.0 - ratio.powi(4) / 12.0));

    let rel_err = (j_fem - j_exact).abs() / j_exact.abs();
    println!(
        "Torsion rectangle: J_fem = {:.6e}, J_exact = {:.6e}, rel_err = {:.4e}",
        j_fem, j_exact, rel_err
    );
    assert!(
        rel_err < 0.01,
        "Torsion constant error too large: {:.4e}",
        rel_err
    );
}

fn integrate_scalar_over_mesh(
    mesh: &sonium_fem::mesh::Mesh,
    _space: &H1Space,
    values: &[f64],
) -> f64 {
    let mut integral = 0.0;
    for elem in mesh.elements.iter() {
        let nodes = &elem.nodes;
        if nodes.len() != 3 {
            continue;
        }
        let p0 = &mesh.nodes[nodes[0]];
        let p1 = &mesh.nodes[nodes[1]];
        let p2 = &mesh.nodes[nodes[2]];
        let area = 0.5 * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)).abs();
        let avg_phi = (values[nodes[0]] + values[nodes[1]] + values[nodes[2]]) / 3.0;
        integral += avg_phi * area;
    }
    integral
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elasticity_torsion_rectangle();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
