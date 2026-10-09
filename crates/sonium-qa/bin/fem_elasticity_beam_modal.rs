//! QA: FEM structural dynamics — cantilever beam natural frequencies (modal analysis).
//!
//! Validates assembled beam stiffness and consistent mass matrices against
//! Euler-Bernoulli analytical frequencies using inverse iteration.
//!
//! Analytical (Euler-Bernoulli cantilever):
//!   cos(βL) · cosh(βL) = −1
//!   ωₙ = (βₙ)² · √(EI / (ρA)) / L²
//!
//! First four roots βₙL:
//!   β₁L = 1.875104, β₂L = 4.694091, β₃L = 7.854757, β₄L = 10.995541

use sonium_fem::physics::solid::beam::{
    BeamElement2D, BeamSection, assemble_beam_mass, assemble_beam_stiffness,
};
use sonium_solvers::{
    CsrMatrix, LinearOperator, LinearSolveConfig, LinearSolverType, TripletMatrix, solve_real,
};

/// Extract a sub-matrix corresponding to free DOFs.
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

fn normalize(v: &mut [f64]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Inverse iteration for the first `num_modes` eigenpairs of K φ = λ M φ.
/// Returns eigenvalues (λ = ω²) in ascending order.
fn inverse_iteration(
    k: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
    num_modes: usize,
    max_iters: usize,
) -> Vec<f64> {
    let n = k.num_rows;
    let n_nodes = n / 3;
    let solver_cfg = LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        max_iterations: 1000,
        tolerance: 1e-8,
        restart: 300,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };

    let mut eigenvalues = Vec::with_capacity(num_modes);
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(num_modes);

    for mode in 0..num_modes {
        // Initialise only the w DOF (transverse displacement) with sine shape
        let mut x = vec![0.0; n];
        for j in 0..n_nodes {
            let w_val = ((mode + 1) as f64 * std::f64::consts::PI * (j + 1) as f64
                / (n_nodes + 1) as f64)
                .sin();
            x[3 * j + 1] = w_val;
        }

        // M-orthogonalise against previous eigenvectors
        for prev in &eigenvectors {
            let mp = m.apply(prev);
            let alpha = dot(&x, &mp);
            let beta = dot(prev, &mp);
            if beta.abs() > 1e-15 {
                for i in 0..n {
                    x[i] -= alpha / beta * prev[i];
                }
            }
        }
        normalize(&mut x);

        let mut lambda = 0.0;
        for _ in 0..max_iters {
            let mx = m.apply(&x);
            let result = solve_real(k, &mx, &solver_cfg).expect("solve failed");
            assert!(result.converged, "Linear solve failed in inverse iteration");
            let y = result.x;

            // Rayleigh quotient λ = y^T K y / y^T M y
            let ky = k.apply(&y);
            let my = m.apply(&y);
            let new_lambda = dot(&y, &ky) / dot(&y, &my);

            // Euclidean normalise
            let mut y_norm = y.clone();
            normalize(&mut y_norm);

            // M-orthogonalise against previous modes
            for prev in &eigenvectors {
                let mp = m.apply(prev);
                let alpha = dot(&y_norm, &mp);
                for i in 0..n {
                    y_norm[i] -= alpha * prev[i];
                }
            }
            normalize(&mut y_norm);

            let diff: f64 = x.iter().zip(&y_norm).map(|(&a, &b)| (a - b).abs()).sum();

            x = y_norm;
            lambda = new_lambda;

            if diff < 1e-10 {
                break;
            }
        }

        // M-normalise the converged eigenvector so that Gram-Schmidt works
        let mp = m.apply(&x);
        let m_norm = dot(&x, &mp);
        if m_norm > 0.0 {
            let scale = 1.0 / m_norm.sqrt();
            for val in x.iter_mut() {
                *val *= scale;
            }
        }
        eigenvalues.push(lambda);
        eigenvectors.push(x);
    }

    eigenvalues
}

fn test_beam_modal_cantilever() {
    let l = 1.0;
    let e_mod = 200e9;
    let width = 0.01;
    let height = 0.02;
    let rho = 7800.0;
    let section = BeamSection::rectangular(e_mod, width, height, rho);
    let i_inertia = section.moment_of_inertia;
    let area = width * height;

    let n_elements = 40;
    let dx = l / n_elements as f64;

    let elements: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            let x1 = i as f64 * dx;
            let x2 = (i + 1) as f64 * dx;
            BeamElement2D::new([i, i + 1], x1, 0.0, x2, 0.0, section)
        })
        .collect();

    let n_nodes = n_elements + 1;
    let n_dofs = n_nodes * 3; // [u, w, theta] per node

    let k_csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let m_csr = assemble_beam_mass(&elements, n_dofs).to_csr();

    let constrained = [0, 1, 2];
    let free_dofs: Vec<usize> = (0..n_dofs).filter(|d| !constrained.contains(d)).collect();

    let k_free = extract_submatrix(&k_csr, &free_dofs);
    let m_free = extract_submatrix(&m_csr, &free_dofs);

    let eigenvalues = inverse_iteration(&k_free, &m_free, 4, 50);

    // Exact eigenvalues (squared natural frequencies)
    let beta = [
        1.87510406871196,
        4.69409113297417,
        7.85475743823761,
        10.9955407348755,
    ];
    let scale = (e_mod * i_inertia / (rho * area)).sqrt() / (l * l);

    println!("Beam modal (cantilever, {} elems):", n_elements);
    let mut max_err = 0.0f64;
    for (i, (&beta_n, &omega_fem_sq)) in beta.iter().zip(eigenvalues.iter()).enumerate() {
        let omega_exact = (beta_n * beta_n * scale).powi(2);
        let err = (omega_fem_sq - omega_exact).abs() / omega_exact;
        max_err = max_err.max(err);
        println!(
            "  Mode {}: FEM ω² = {:.6e}, exact ω² = {:.6e}, rel_err = {:.4e}",
            i + 1,
            omega_fem_sq,
            omega_exact,
            err
        );
    }

    assert!(
        max_err < 1e-4,
        "Maximum eigenvalue error too large: {:.4e}",
        max_err
    );
}

fn test_beam_modal_convergence() {
    let l = 1.0;
    let e_mod = 200e9;
    let width = 0.01;
    let height = 0.02;
    let rho = 7800.0;
    let section = BeamSection::rectangular(e_mod, width, height, rho);
    let i_inertia = section.moment_of_inertia;
    let area = width * height;

    let beta1 = 1.87510406871196;
    let omega1_exact =
        (beta1 * beta1 * (e_mod * i_inertia / (rho * area)).sqrt() / (l * l)).powi(2);

    let mesh_sizes = [10, 20, 40, 80];
    let mut errors = Vec::new();

    for &n_elements in &mesh_sizes {
        let dx = l / n_elements as f64;
        let elements: Vec<BeamElement2D> = (0..n_elements)
            .map(|i| {
                let x1 = i as f64 * dx;
                let x2 = (i + 1) as f64 * dx;
                BeamElement2D::new([i, i + 1], x1, 0.0, x2, 0.0, section)
            })
            .collect();

        let n_nodes = n_elements + 1;
        let n_dofs = n_nodes * 3;

        let k_csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
        let m_csr = assemble_beam_mass(&elements, n_dofs).to_csr();

        let constrained = [0, 1, 2];
        let free_dofs: Vec<usize> = (0..n_dofs).filter(|d| !constrained.contains(d)).collect();

        let k_free = extract_submatrix(&k_csr, &free_dofs);
        let m_free = extract_submatrix(&m_csr, &free_dofs);

        let eigenvalues = inverse_iteration(&k_free, &m_free, 1, 50);
        let omega1_fem = eigenvalues[0];
        let err = (omega1_fem - omega1_exact).abs() / omega1_exact;
        errors.push(err);
        println!("  Convergence n={}: ω²_err = {:.4e}", n_elements, err);
    }

    let mut rates = Vec::new();
    for i in 1..errors.len() {
        let rate = (errors[i - 1] / errors[i]).log2();
        rates.push(rate);
        println!(
            "  Rate {}->{}: {:.2}",
            mesh_sizes[i - 1],
            mesh_sizes[i],
            rate
        );
    }

    let min_rate = rates.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        min_rate > 1.0,
        "Convergence rate too low: {:.2} (expected > 1.0)",
        min_rate
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_beam_modal_cantilever();
        test_beam_modal_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
