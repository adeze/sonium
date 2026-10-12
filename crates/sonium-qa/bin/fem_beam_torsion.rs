//! QA: 3-D beam pure torsion vs the twist formula.
//!
//! A circular shaft under end torque must reproduce `phi = T L / GJ`, and an
//! I-beam smoke solve exercises the thin-walled torsion constant. The C3
//! comparison case lives in `fem_opencourant_beam_torsion.rs`.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::beam::{BeamElement3D, Section3D, assemble_beam3d_stiffness};
use sonium_solvers::{CsrMatrix, GmresConfig, gmres};
fn apply_dirichlet_csr(
    csr: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    prescribed_dofs: &[usize],
    prescribed_values: &[f64],
) {
    for dof in prescribed_dofs {
        let range = csr.row_range(*dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        let diag_pos = csr.row_entries(*dof).position(|(j, _)| j == *dof);
        if let Some(idx) = diag_pos {
            let global_idx = csr.row_ptrs[*dof] + idx;
            csr.values[global_idx] = 1.0;
        }
    }
    for (dof, &val) in prescribed_dofs.iter().zip(prescribed_values.iter()) {
        rhs[*dof] = val;
    }
}

/// Twist at the free end of a clamped shaft under end torque.
fn shaft_twist(section: Section3D, length: f64, torque: f64, n_elements: usize) -> f64 {
    let dx = length / n_elements as f64;
    let elements: Vec<BeamElement3D> = (0..n_elements)
        .map(|i| {
            BeamElement3D::new(
                [i, i + 1],
                [i as f64 * dx, 0.0, 0.0],
                [(i + 1) as f64 * dx, 0.0, 0.0],
                section,
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    let n_dofs = (n_elements + 1) * 6;
    let mut csr = assemble_beam3d_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];
    rhs[n_elements * 6 + 3] = torque;
    apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1, 2, 3, 4, 5], &[0.0; 6]);
    let result = gmres(
        &csr,
        &rhs,
        &GmresConfig {
            max_iterations: 5000,
            restart: 200,
            tolerance: 8e-11,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "torsion solve must converge");
    result.x[n_elements * 6 + 3]
}

fn test_pure_torsion() {
    let section = Section3D::circular(70.0e9, 0.33, 0.025, 2700.0, false);
    let length = 2.0;
    let torque = 500.0;
    let twist = shaft_twist(section, length, torque, 6);
    let exact = torque * length / section.gj();
    let error = (twist - exact).abs() / exact;
    println!("torsion: FEM={twist:.12e} exact={exact:.12e} rel={error:.2e}");
    assert!(error < 1.0e-9, "twist error {error:.3e}");
    println!("GOLDEN beam_torsion_twist fem={twist:.17e} exact={exact:.17e}");
}

fn test_ibeam_torsion_smoke() {
    let section = Section3D::i_beam(210.0e9, 0.3, 0.1, 0.2, 0.01, 7800.0, false);
    assert!(section.torsion_constant > 0.0);
    let twist = shaft_twist(section, 3.0, 1000.0, 4);
    let exact = 1000.0 * 3.0 / section.gj();
    let error = (twist - exact).abs() / exact;
    println!("ibeam torsion: FEM={twist:.12e} exact={exact:.12e} rel={error:.2e}");
    assert!(error < 1.0e-9, "ibeam twist error {error:.3e}");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_pure_torsion();
        test_ibeam_torsion_smoke();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
