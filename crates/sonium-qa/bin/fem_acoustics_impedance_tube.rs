//! QA: FEM acoustics — 1D impedance tube with reflection coefficient.
//!
//! A rigid-walled duct [0,L]×[0,h] with Dirichlet p=1 at x=0 and a Robin
//! impedance boundary at x=L. The exact 1D solution is a standing wave
//! with reflection coefficient R = (Z - Z₀)/(Z + Z₀).
//!
//! For Z = 2·Z₀: R = 1/3, Robin coefficient α = ik/2.

use num_complex::Complex64;
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{DirichletBC, apply_dirichlet};
use sonium_fem::boundary::{RobinBC, apply_robin};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::solver::{SolverConfig, SolverType, solve};

fn exact_impedance_tube(x: f64, k: f64, l: f64, r: f64) -> Complex64 {
    let phase_in = Complex64::new(0.0, -k * x);
    let phase_ref = Complex64::new(0.0, k * (x - 2.0 * l));
    let phase_denom = Complex64::new(0.0, -2.0 * k * l);
    let numer = phase_in.exp() + r * phase_ref.exp();
    let denom = 1.0 + r * phase_denom.exp();
    numer / denom
}

fn test_acoustics_impedance_tube() {
    let l = 2.0;
    let h = 0.2;
    let k = 1.0;
    let z_ratio = 2.0;
    let r = (z_ratio - 1.0) / (z_ratio + 1.0);

    let nx = 40;
    let ny = 4;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);

    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::new(0.0, 0.0)
        });

    let dirichlet_bcs = vec![DirichletBC::new(1, |_, _, _| Complex64::new(1.0, 0.0))];
    apply_dirichlet(&mut problem, &mesh, &dirichlet_bcs);

    let alpha = Complex64::new(0.0, k / z_ratio);
    let robin_bcs = vec![RobinBC::new(
        2,
        move |_, _, _| alpha,
        |_, _, _| Complex64::new(0.0, 0.0),
    )];
    apply_robin(&mut problem, &mesh, &robin_bcs, 2);

    let config = SolverConfig {
        solver_type: SolverType::Direct,
        ..Default::default()
    };

    let solution = solve(&problem, &config).expect("Direct solver should succeed");
    assert!(solution.converged, "Solver did not converge");

    let mut max_err: f64 = 0.0;
    let mut max_norm: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let p_exact = exact_impedance_tube(node.x, k, l, r);
        let p_fem = solution.values[i];
        max_err = max_err.max((p_fem - p_exact).norm());
        max_norm = max_norm.max(p_exact.norm());
    }

    let rel_err = max_err / max_norm.max(1e-30);
    println!(
        "Acoustics impedance tube: max |p| err = {:.4e}, rel err = {:.4e}",
        max_err, rel_err
    );
    assert!(
        rel_err < 0.001,
        "Relative pressure error too large: {:.4e}",
        rel_err
    );
}

fn test_acoustics_impedance_tube_convergence() {
    let l = 2.0;
    let h = 0.2;
    let k = 1.0;
    let z_ratio = 2.0;
    let r = (z_ratio - 1.0) / (z_ratio + 1.0);
    let alpha = Complex64::new(0.0, k / z_ratio);

    let errors: Vec<f64> = [20, 40, 80]
        .iter()
        .map(|&nx| {
            let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, 4);
            let k_complex = Complex64::new(k, 0.0);
            let mut problem =
                HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
                    Complex64::new(0.0, 0.0)
                });

            let dirichlet_bcs = vec![DirichletBC::new(1, |_, _, _| Complex64::new(1.0, 0.0))];
            apply_dirichlet(&mut problem, &mesh, &dirichlet_bcs);

            let robin_bcs = vec![RobinBC::new(
                2,
                move |_, _, _| alpha,
                |_, _, _| Complex64::new(0.0, 0.0),
            )];
            apply_robin(&mut problem, &mesh, &robin_bcs, 2);

            let config = SolverConfig {
                solver_type: SolverType::Direct,
                ..Default::default()
            };

            let solution = solve(&problem, &config).expect("Direct solver should succeed");
            assert!(solution.converged);

            let mut max_err: f64 = 0.0;
            let mut max_norm: f64 = 0.0;
            for (i, node) in mesh.nodes.iter().enumerate() {
                let p_exact = exact_impedance_tube(node.x, k, l, r);
                let p_fem = solution.values[i];
                max_err = max_err.max((p_fem - p_exact).norm());
                max_norm = max_norm.max(p_exact.norm());
            }
            max_err / max_norm.max(1e-30)
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (20.0_f64 / 40.0).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (40.0_f64 / 80.0).log(std::f64::consts::LN_10);

    println!(
        "Impedance tube convergence: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );
    let all_small = errors.iter().all(|&e| e < 1e-4);
    assert!(
        all_small || (rate_1 > 1.0 && rate_2 > 1.0),
        "Convergence rate too low"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_acoustics_impedance_tube();
        test_acoustics_impedance_tube_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
