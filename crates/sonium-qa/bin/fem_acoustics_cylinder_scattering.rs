//! QA: FEM acoustics — 2D rigid cylinder scattering with first-order ABC.
//!
//! Scattered-field formulation:
//!   (∇² + k²) p_s = 0
//!   ∂p_s/∂n = -∂p_inc/∂n on rigid boundary (marker 1)
//!   ∂p_s/∂n + ik p_s = 0 on outer boundary (marker 2)  (Sommerfeld ABC)
//!
//! Incident plane wave: p_inc = e^{ikx}
//! Exact total field from sonium-wave.

use num_complex::Complex64;
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_neumann_complex, apply_robin_complex};
use sonium_fem::mesh::generators::annular_mesh_triangles;
use sonium_fem::physics::acoustics::pressure::PmlHelmholtzForm;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve_complex};
use sonium_wave::analytical::cylinder_scattering_2d;

fn test_acoustics_cylinder_scattering() {
    let a = 0.5; // cylinder radius
    let b = 10.0; // outer radius
    let k = 2.0; // wavenumber
    let n_terms = 20;

    let n_radial = 40;
    let n_angular = 64;
    let mesh = annular_mesh_triangles(0.0, 0.0, a, b, n_radial, n_angular);

    let space = H1Space::new(PolynomialDegree::P1);
    let _n_dofs = space.num_global_dofs(&mesh);

    let form = PmlHelmholtzForm::from_real(k, vec![]);
    let matrix = assemble_bilinear(&form, &space, &space, &mesh, 2);

    let mut system = FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); _n_dofs]);

    // Inner boundary (marker 1): Neumann for scattered field
    apply_neumann_complex(&mut system, &mesh, 1, &|x, y, _z| {
        let theta = y.atan2(x);
        let cos_t = theta.cos();
        let phase = k * a * cos_t;
        let exp_val = Complex64::new(phase.cos(), phase.sin());
        Complex64::new(0.0, k * cos_t) * exp_val
    });

    // Outer boundary (marker 2): first-order absorbing boundary condition
    let alpha = Complex64::new(0.0, -k);
    apply_robin_complex(&mut system, &mesh, 2, &|_, _, _| alpha, &|_, _, _| {
        Complex64::new(0.0, 0.0)
    });

    let csr = system.matrix.to_csr();

    let solve_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 2000,
        tolerance: 1e-10,
        restart: 100,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };

    let result =
        solve_complex(&csr, &system.rhs, &solve_config).expect("Complex solver should converge");
    assert!(result.converged, "Complex GMRES did not converge");

    let exact = cylinder_scattering_2d(
        k,
        a,
        n_terms,
        vec![1.0, 2.0, 4.0, 6.0],
        vec![
            0.0,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_2,
            3.0 * std::f64::consts::FRAC_PI_4,
            std::f64::consts::PI,
        ],
    );
    let _ = exact; // silence unused warning if we switch to per-node evaluation

    let mut max_err: f64 = 0.0;
    let mut max_norm: f64 = 0.0;

    for (i, node) in mesh.nodes.iter().enumerate() {
        let r = (node.x * node.x + node.y * node.y).sqrt();
        if r > b * 0.6 || r < a * 1.1 {
            continue;
        }

        let theta = node.y.atan2(node.x);
        let p_inc = Complex64::new((k * node.x).cos(), (k * node.x).sin());
        let p_total_fem = p_inc + result.x[i];

        let p_exact = exact_cylinder_at_point(k, a, n_terms, r, theta);

        max_err = max_err.max((p_total_fem - p_exact).norm());
        max_norm = max_norm.max(p_exact.norm());
    }

    let rel_err = max_err / max_norm.max(1e-30);
    println!(
        "Acoustics cylinder scattering: max_err = {:.4e}, rel_err = {:.4e}",
        max_err, rel_err
    );
    assert!(rel_err < 0.03, "Relative error too large: {:.4e}", rel_err);
}

fn exact_cylinder_at_point(k: f64, a: f64, n_terms: usize, r: f64, theta: f64) -> Complex64 {
    let sol = sonium_wave::analytical::cylinder_scattering_2d(k, a, n_terms, vec![r], vec![theta]);
    sol.pressure[0]
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_acoustics_cylinder_scattering();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
