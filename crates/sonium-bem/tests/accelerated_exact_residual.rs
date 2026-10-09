//! Independent dense-TBEM residual certification for an accelerated solve.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{build_cluster_tree, build_mlfmm_system_stabilized};
use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::{DenseOperator, GmresConfig, MlfmmOperator, solve_gmres};
use sonium_bem::core::{BoundaryCondition, PhysicsParams};
use sonium_solvers::traits::LinearOperator;
use std::f64::consts::PI;

fn relative_residual<O: LinearOperator<Complex64>>(
    operator: &O,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    operator
        .apply(solution)
        .iter()
        .zip(rhs)
        .map(|(&actual, &expected)| (actual - expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE)
}

#[test]
fn mlfmm_solution_is_certified_by_independent_dense_tbem_action() {
    let radius = 0.2;
    let ka = 0.5;
    let speed_of_sound = 343.0;
    let frequency = ka * speed_of_sound / (2.0 * PI * radius);
    let physics = PhysicsParams::new(frequency, speed_of_sound, 1.21, false);
    let mesh = generate_icosphere_mesh(radius, 2);
    let mut elements = mesh.elements.clone();
    for (index, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![index];
    }

    let sign = PhysicsParams::optimal_formulation_sign(ka);
    let beta_scale = PhysicsParams::optimal_beta_scale(ka);
    let exact = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, sign, beta_scale);
    let tree = build_cluster_tree(&elements, 8, &physics);
    let rhs = exact.rhs;
    let exact_operator = DenseOperator::new(exact.matrix);
    let mlfmm = MlfmmOperator::new(build_mlfmm_system_stabilized(
        &elements,
        &mesh.nodes,
        tree,
        &physics,
        sign,
        beta_scale,
    ));
    let config = GmresConfig {
        max_iterations: 300,
        restart: 50,
        tolerance: 1.0e-9,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let solved = solve_gmres(&mlfmm, &rhs, &config);
    assert!(solved.converged, "MLFMM solve did not converge: {solved:?}");

    let accelerated_residual = relative_residual(&mlfmm, &solved.x, &rhs);
    let exact_residual = relative_residual(&exact_operator, &solved.x, &rhs);
    eprintln!(
        "MLFMM residuals: accelerated={accelerated_residual:.6e}, independent_dense_tbem={exact_residual:.6e}"
    );
    assert!(accelerated_residual <= 2.0e-8);
    assert!(
        exact_residual <= 2.0e-8,
        "MLFMM density fails independent dense-TBEM residual: {exact_residual:.3e}"
    );
}
