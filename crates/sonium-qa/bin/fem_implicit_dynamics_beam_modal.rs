//! QA: implicit dynamics beam modal transient verification.
//!
//! Releases a cantilever Euler-Bernoulli beam from its first-mode shape
//! and integrates the free transient with the production Newmark/HHT
//! integrator: the tip response must track the modal cosine, and the
//! trapezoidal amplitude must hold while HHT damps mildly.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::beam::{
    BeamElement2D, BeamSection, assemble_beam_mass, assemble_beam_stiffness,
};
use sonium_fem::solver::implicit_dynamics::{
    ImplicitDynamicsConfig, ImplicitDynamicsIntegrator, ImplicitDynamicsSystem,
};
use sonium_solvers::{
    CsrMatrix, LinearOperator, LinearSolveConfig, LinearSolverType, TripletMatrix, solve_real,
};
use std::f64::consts::PI;

fn extract_submatrix(csr: &CsrMatrix<f64>, free_dofs: &[usize]) -> CsrMatrix<f64> {
    let n = free_dofs.len();
    let mut triplet = TripletMatrix::new(n, n);
    for (new_i, &old_i) in free_dofs.iter().enumerate() {
        for (old_j, val) in csr.row_entries(old_i) {
            if let Some(new_j) = free_dofs.iter().position(|&d| d == old_j) {
                triplet.push(new_i, new_j, val);
            }
        }
    }
    triplet.to_csr()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// First eigenpair of `K phi = lambda M phi` by inverse iteration.
fn first_mode(k: &CsrMatrix<f64>, m: &CsrMatrix<f64>) -> (f64, Vec<f64>) {
    let n = k.num_rows;
    let solver_cfg = LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        max_iterations: 1000,
        tolerance: 1e-8,
        restart: 300,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };
    let n_nodes = n / 3;
    let mut x = vec![0.0; n];
    for j in 0..n_nodes {
        x[3 * j + 1] = (PI * (j + 1) as f64 / (n_nodes + 1) as f64).sin();
    }
    let mut lambda = 0.0;
    for _ in 0..60 {
        let mx = m.apply(&x);
        let result = solve_real(k, &mx, &solver_cfg).expect("mode solve failed");
        assert!(result.converged, "mode eigensolve failed");
        let y = result.x;
        lambda = dot(&y, &k.apply(&y)) / dot(&y, &m.apply(&y));
        let norm = dot(&y, &y).sqrt();
        let next: Vec<f64> = y.iter().map(|v| v / norm).collect();
        let diff: f64 = x.iter().zip(&next).map(|(&a, &b)| (a - b).abs()).sum();
        x = next;
        if diff < 1e-10 {
            break;
        }
    }
    // M-normalize so the release amplitude reads directly.
    let scale = 1.0 / dot(&x, &m.apply(&x)).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
    (lambda, x)
}

fn cantilever_system() -> (CsrMatrix<f64>, CsrMatrix<f64>, f64, f64) {
    let (length, e_mod, width, height, rho) = (1.0, 200e9, 0.01, 0.02, 7800.0);
    let section = BeamSection::rectangular(e_mod, width, height, rho);
    let n_elements = 40;
    let dx = length / n_elements as f64;
    let elements: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            BeamElement2D::new(
                [i, i + 1],
                i as f64 * dx,
                0.0,
                (i + 1) as f64 * dx,
                0.0,
                section,
            )
        })
        .collect();
    let n_dofs = (n_elements + 1) * 3;
    let k_full = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let m_full = assemble_beam_mass(&elements, n_dofs).to_csr();
    let free_dofs: Vec<usize> = (3..n_dofs).collect();
    let beta1 = 1.87510406871196;
    let scale = (e_mod * section.moment_of_inertia / (rho * width * height)).sqrt();
    let omega1_exact = beta1 * beta1 * scale;
    (extract_submatrix(&k_full, &free_dofs), extract_submatrix(&m_full, &free_dofs), omega1_exact, length)
}

fn run_transient(
    k: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
    shape: &[f64],
    omega: f64,
    config: ImplicitDynamicsConfig,
    dt: f64,
    periods: usize,
) -> Vec<f64> {
    let system = ImplicitDynamicsSystem {
        mass: m.clone(),
        stiffness: k.clone(),
        damping: None,
    };
    let mut integrator = ImplicitDynamicsIntegrator::new(system, config, dt).unwrap();
    let force = vec![0.0; shape.len()];
    integrator.initialize(shape, &vec![0.0; shape.len()], &force).unwrap();
    let steps = (periods as f64 * 2.0 * PI / omega / dt).round() as usize;
    // Tip transverse DOF is index 1 of the last free node.
    let tip = shape.len() - 2;
    let mut tip_trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        tip_trace.push(integrator.step(&force).unwrap().displacement[tip]);
    }
    tip_trace
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        let (k, m, omega1_exact, _) = cantilever_system();
        let (lambda1, shape) = first_mode(&k, &m);
        let omega1 = lambda1.sqrt();
        let freq_err = (omega1 - omega1_exact).abs() / omega1_exact;
        println!("beam modal: FEM w1={omega1:.6e} exact={omega1_exact:.6e} err={freq_err:.3e}");
        assert!(freq_err < 1e-4, "first frequency error {freq_err:.3e}");

        let tip = shape.len() - 2;
        let tip0 = shape[tip];
        let dt = 2.0 * PI / omega1 / 100.0;
        let trace = run_transient(
            &k,
            &m,
            &shape,
            omega1,
            ImplicitDynamicsConfig::default(),
            dt,
            2,
        );
        // Quarter-period stations of the modal cosine.
        for &fraction in &[0.25, 0.5, 0.75, 1.0, 1.5, 2.0] {
            let index = (fraction * 100.0) as usize - 1;
            let t = (index + 1) as f64 * dt;
            let exact = tip0 * (omega1 * t).cos();
            let scale = tip0.abs().max(1e-30);
            let error = (trace[index] - exact).abs() / scale;
            println!("transient t={t:.4e}: FEM={:.6e} exact={exact:.6e}", trace[index]);
            assert!(error < 1e-2, "transient error {error:.3e} at t={t}");
        }
        // Trapezoidal amplitude holds over two periods.
        let end_exact = tip0 * (omega1 * 200.0 * dt).cos();
        let end_err = (trace[199] - end_exact).abs() / tip0.abs();
        assert!(end_err < 1e-2, "amplitude error {end_err:.3e}");

        // HHT damps mildly at this resolution without phase damage.
        let hht = run_transient(
            &k,
            &m,
            &shape,
            omega1,
            ImplicitDynamicsConfig::hht(-0.05),
            dt,
            2,
        );
        let ratio = hht[199] / trace[199];
        println!("HHT/trap amplitude ratio after two periods: {ratio:.6}");
        assert!(
            ratio < 1.0 && ratio > 0.9,
            "HHT ratio {ratio} outside (0.9, 1)"
        );
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
