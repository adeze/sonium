//! QA: FEM beam — simply supported with central point load.
//!
//! Analytical (Euler-Bernoulli): w(x) = F*x*(3L² - 4x²)/(48*E*I) for x ≤ L/2

use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection, assemble_beam_stiffness};
use sonium_qa::analytical::beam_euler_ss_central;
use sonium_solvers::{CsrMatrix, GmresConfig, gmres};

fn apply_dirichlet_csr(
    csr: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    prescribed_dofs: &[usize],
    prescribed_values: &[f64],
) {
    for dof in prescribed_dofs.iter() {
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

fn test_beam_ss_central() {
    let l = 1.0;
    let f = 1000.0;
    let e_mod = 200e9;
    let width = 0.01;
    let height = 0.02;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    // Use an even number of elements so the center is a node
    let n_elements = 20;
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

    let mut csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];

    // Point load at center node
    let center_node = n_nodes / 2;
    rhs[center_node * 3 + 1] = -f;

    // Simply supported: v=0 at both ends, u=0 at left end
    apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1], &[0.0, 0.0]);
    apply_dirichlet_csr(&mut csr, &mut rhs, &[(n_nodes - 1) * 3 + 1], &[0.0]);

    let result = gmres(
        &csr,
        &rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 2e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );

    assert!(
        result.converged,
        "Solver should converge: residual={:.2e}",
        result.residual
    );

    let mid_delta = result.x[center_node * 3 + 1].abs();
    let exact_mid = beam_euler_ss_central(l / 2.0, l, e_mod, i_inertia, f);

    let error = (mid_delta - exact_mid).abs() / exact_mid;
    println!(
        "Beam SS central: FEM mid={:.6e}, Exact={:.6e}, Error={:.2}%",
        mid_delta,
        exact_mid,
        error * 100.0
    );
    assert!(
        error < 1.0e-9,
        "Mid deflection error too large: {:.4e}",
        error
    );
}

fn test_beam_ss_central_convergence() {
    let l = 1.0;
    let f = 1000.0;
    let e_mod = 200e9;
    let width = 0.01;
    let height = 0.02;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    let errors: Vec<f64> = [6, 10, 20]
        .iter()
        .map(|&n_elements| {
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

            let mut csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
            let mut rhs = vec![0.0; n_dofs];

            let center_node = n_nodes / 2;
            rhs[center_node * 3 + 1] = -f;

            apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1], &[0.0, 0.0]);
            apply_dirichlet_csr(&mut csr, &mut rhs, &[(n_nodes - 1) * 3 + 1], &[0.0]);

            let result = gmres(
                &csr,
                &rhs,
                &GmresConfig {
                    max_iterations: 1000,
                    restart: 200,
                    tolerance: 2e-12,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    min_restart: None,
                    max_restart: None,
                    exact_residual_check_interval: 0,
                },
            );
            assert!(result.converged);

            let mid_delta = result.x[center_node * 3 + 1].abs();
            let exact_mid = beam_euler_ss_central(l / 2.0, l, e_mod, i_inertia, f);
            (mid_delta - exact_mid).abs() / exact_mid
        })
        .collect();

    // Fine meshes are exact to machine precision; coarse mesh may have ~5% error
    assert!(
        errors.iter().all(|error| *error < 1.0e-9),
        "Error too large: {:.2e}, {:.2e}, {:.2e}",
        errors[0],
        errors[1],
        errors[2]
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_beam_ss_central();
        test_beam_ss_central_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
