use sonium_fem::physics::solid::beam::{
    BeamElement2D, BeamSection, assemble_beam_mass, assemble_beam_stiffness,
};
use sonium_solvers::{
    CsrMatrix, DiagonalPreconditioner, LobpcgConfig, TripletMatrix,
    lobpcg_generalized_preconditioned,
};

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

fn main() {
    let l = 1.0;
    let e_mod = 200e9;
    let width = 0.01;
    let height = 0.02;
    let rho = 7800.0;
    let section = BeamSection::rectangular(e_mod, width, height, rho);

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
    let n_dofs = n_nodes * 3;

    let k_csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let m_csr = assemble_beam_mass(&elements, n_dofs).to_csr();

    let constrained = [0, 1, 2];
    let free_dofs: Vec<usize> = (0..n_dofs).filter(|d| !constrained.contains(d)).collect();

    let k_free = extract_submatrix(&k_csr, &free_dofs);
    let m_free = extract_submatrix(&m_csr, &free_dofs);

    let precond = DiagonalPreconditioner::from_csr(&k_free);

    let area = width * height;
    let second_moment = width * height.powi(3) / 12.0;
    let bending_scale = e_mod * second_moment / (rho * area * l.powi(4));
    let beta_l: [f64; 4] = [1.875_104_07, 4.694_091_13, 7.854_757_44, 10.995_540_73];
    let expected: Vec<f64> = beta_l
        .iter()
        .map(|beta| beta.powi(4) * bending_scale)
        .collect();

    let mut failed = false;
    for tol in [1e-6, 1e-8] {
        let config = LobpcgConfig {
            num_eigenvalues: 4,
            max_iterations: 2000,
            tolerance: tol,
            ..Default::default()
        };
        let result = lobpcg_generalized_preconditioned(&k_free, &m_free, &precond, &config)
            .unwrap_or_else(|err| {
                eprintln!("FAILED: LOBPCG projected eigensolver did not converge: {err}");
                std::process::exit(1);
            });
        println!(
            "tol={:.0e}: converged={}, iters={}, eigenvalues={:?}, residuals={:?}",
            tol, result.converged, result.iterations, result.eigenvalues, result.residual_norms
        );
        let residuals_valid = result.residual_norms.len() == config.num_eigenvalues
            && result
                .residual_norms
                .iter()
                .all(|residual| residual.is_finite() && *residual <= 10.0 * tol);
        let max_eigenvalue_error = result
            .eigenvalues
            .iter()
            .zip(&expected)
            .map(|(actual, reference)| (actual - reference).abs() / reference)
            .fold(0.0_f64, f64::max);
        println!(
            "  max Euler-Bernoulli eigenvalue error: {:.3e}",
            max_eigenvalue_error
        );
        if !result.converged || !residuals_valid || max_eigenvalue_error > 5e-3 {
            eprintln!(
                "FAILED: LOBPCG tolerance {:.0e}, residual, or analytical eigenvalue gate failed",
                tol,
            );
            failed = true;
        }
    }

    if failed {
        std::process::exit(1);
    }
    println!("SUCCESS");
}
