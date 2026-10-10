//! QA: modal FRF and random-vibration PSD on beam eigenmodes.
//!
//! Extracts the first three cantilever modes with production LOBPCG,
//! then gates the tip receptance (static compliance plus the resonant
//! peak) and the white-noise RMS (numeric PSD integration vs the modal
//! variance sum) against Euler-Bernoulli closed forms.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::beam::{
    BeamElement2D, BeamSection, assemble_beam_mass, assemble_beam_stiffness,
};
use sonium_fem::physics::solid::eigenmodes::{EigenPhysicsKind, EigenPhysicsProblem};
use sonium_fem::solver::vibration_fatigue::{modal_receptance, psd_response_rms};
use sonium_solvers::{CsrMatrix, LobpcgConfig, TripletMatrix};
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

fn main() {
    let result = std::panic::catch_unwind(|| {
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
        let stiffness = extract_submatrix(&k_full, &free_dofs);
        let mass = extract_submatrix(&m_full, &free_dofs);

        // Diagonal mass scaling: rotational DOFs make the raw pair too
        // ill-scaled for unpreconditioned LOBPCG. Eigenvalues are
        // invariant; shapes unscale with the same diagonal.
        let n = stiffness.num_rows;
        let mut diag = vec![0.0; n];
        for i in 0..n {
            for (j, value) in mass.row_entries(i) {
                if i == j {
                    diag[i] = value;
                }
            }
            assert!(diag[i] > 0.0, "mass diagonal must be positive");
        }
        let scale = |csr: &CsrMatrix<f64>| {
            let mut triplet = TripletMatrix::new(n, n);
            for i in 0..n {
                for (j, value) in csr.row_entries(i) {
                    triplet.push(i, j, value / (diag[i] * diag[j]).sqrt());
                }
            }
            triplet.to_csr()
        };
        let problem = EigenPhysicsProblem {
            kind: EigenPhysicsKind::StructuralModal,
            stiffness_like: scale(&stiffness),
            mass_like: scale(&mass),
            angular_frequency_scale: 1.0,
        };
        let config = LobpcgConfig {
            num_eigenvalues: 3,
            max_iterations: 2000,
            tolerance: 1e-8,
            block_padding: 3,
        };
        let solved = problem.solve(&config).expect("beam modes must converge");
        assert!(
            solved.converged,
            "LOBPCG must converge: {:?} in {}",
            solved.residual_norms, solved.iterations
        );
        let omegas: Vec<f64> = problem.eigenvalues_to_angular_frequencies(&solved.eigenvalues);
        let beta_l = [1.87510406871196, 4.69409113297417, 7.85475743823761];
        let scale = (e_mod * section.moment_of_inertia / (rho * width * height)).sqrt();
        for (r, omega) in omegas.iter().enumerate() {
            let exact = beta_l[r] * beta_l[r] * scale;
            let error = (omega - exact).abs() / exact;
            println!("mode {}: FEM w={omega:.6e} exact={exact:.6e} err={error:.2e}", r + 1);
            assert!(error < 1e-3, "mode {} frequency error {error:.2e}", r + 1);
        }
        // Tip transverse shapes (mass-normalized LOBPCG columns,
        // unscaled back to physical DOFs).
        let tip = free_dofs.len() - 2;
        let pairs: Vec<(f64, f64)> = (0..3)
            .map(|r| {
                let column = solved.eigenvectors.column_to_vec(r);
                let physical = column[tip] / diag[tip].sqrt();
                (physical, physical)
            })
            .collect();
        let zetas = vec![0.01; 3];

        // Static compliance from the modal sum vs L^3/(3EI).
        let h0 = modal_receptance(&[0.0], &omegas, &pairs, &zetas).unwrap()[0];
        let compliance = length.powi(3) / (3.0 * e_mod * section.moment_of_inertia);
        let static_err = (h0.re - compliance).abs() / compliance;
        println!("static compliance: modal={:.6e} exact={compliance:.6e}", h0.re);
        assert!(static_err < 0.02, "static error {static_err:.2e}");

        // Resonant peak of mode 1 vs the SDOF equivalent.
        let peak = modal_receptance(&[omegas[0]], &omegas, &pairs, &zetas).unwrap()[0];
        let expected = pairs[0].0 * pairs[0].1 / (2.0 * zetas[0] * omegas[0].powi(2));
        let peak_err = (peak.norm() - expected).abs() / expected;
        println!("resonant peak: modal={:.6e} sdof={expected:.6e}", peak.norm());
        assert!(peak_err < 0.01, "peak error {peak_err:.2e}");

        // White tip-force PSD: numeric integration vs the modal sum.
        let f3 = omegas[2] / (2.0 * PI);
        let n = 200_001;
        let freqs: Vec<f64> = (0..n).map(|i| 2.0 * f3 * i as f64 / (n - 1) as f64).collect();
        let omegas_grid: Vec<f64> = freqs.iter().map(|f| 2.0 * PI * f).collect();
        let frf = modal_receptance(&omegas_grid, &omegas, &pairs, &zetas).unwrap();
        let numeric = psd_response_rms(&freqs, &frf, &vec![1.0; n]).unwrap();
        let modal_sum: f64 = (0..3)
            .map(|r| pairs[r].0.powi(4) / (8.0 * zetas[r] * omegas[r].powi(3)))
            .sum::<f64>()
            .sqrt();
        let psd_err = (numeric - modal_sum).abs() / modal_sum;
        println!("white-noise RMS: numeric={numeric:.6e} modal={modal_sum:.6e}");
        assert!(psd_err < 0.03, "PSD error {psd_err:.2e}");
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
