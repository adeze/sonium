//! QA: FEM beam — cantilever with uniformly distributed load.
//!
//! Analytical (Euler-Bernoulli): w(x) = q*x²*(6L² - 4L*x + x²)/(24*E*I)

use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection, assemble_beam_stiffness};
use sonium_qa::analytical::beam_euler_cantilever_uniform;
use sonium_solvers::{CsrMatrix, GmresConfig, gmres};

fn apply_dirichlet_csr(
    csr: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    prescribed_dofs: &[usize],
    prescribed_values: &[f64],
) {
    for dof in prescribed_dofs.iter().take(3) {
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

fn test_beam_cantilever_uniform() {
    let l = 1.0;
    let q = 1000.0;
    let e_mod = 200e9;
    let width = 0.01;
    let height = 0.02;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

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

    // Equivalent nodal loads for uniform distributed load on Euler-Bernoulli beam
    // Vertical: q*L/2 at ends, q*L at interior (sum of two half-elements)
    // Moment: -q*L^2/12 at left end, +q*L^2/12 at right end, 0 at interior
    for node in 0..n_nodes {
        let v_dof = node * 3 + 1;
        let t_dof = node * 3 + 2;
        if node == 0 {
            rhs[v_dof] = -q * dx / 2.0;
            rhs[t_dof] = -q * dx * dx / 12.0;
        } else if node == n_nodes - 1 {
            rhs[v_dof] = -q * dx / 2.0;
            rhs[t_dof] = q * dx * dx / 12.0;
        } else {
            rhs[v_dof] = -q * dx;
            rhs[t_dof] = 0.0;
        }
    }

    // Clamp at left end
    apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1, 2], &[0.0, 0.0, 0.0]);

    let result = gmres(
        &csr,
        &rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 8e-11,
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

    let tip_node = n_nodes - 1;
    let tip_delta = result.x[tip_node * 3 + 1].abs();
    let exact_tip = beam_euler_cantilever_uniform(l, l, e_mod, i_inertia, q);

    let error = (tip_delta - exact_tip).abs() / exact_tip;
    println!(
        "Beam cantilever uniform: FEM tip={:.6e}, Exact={:.6e}, Error={:.2}%",
        tip_delta,
        exact_tip,
        error * 100.0
    );
    assert!(
        error < 1.0e-9,
        "Tip deflection error too large: {:.4e}",
        error
    );
}

fn test_beam_cantilever_uniform_convergence() {
    let l = 1.0;
    let q = 1000.0;
    let e_mod = 200e9;
    let width = 0.01;
    let height = 0.02;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    let errors: Vec<f64> = [5, 10, 20]
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

            for node in 0..n_nodes {
                let v_dof = node * 3 + 1;
                let t_dof = node * 3 + 2;
                if node == 0 {
                    rhs[v_dof] = -q * dx / 2.0;
                    rhs[t_dof] = -q * dx * dx / 12.0;
                } else if node == n_nodes - 1 {
                    rhs[v_dof] = -q * dx / 2.0;
                    rhs[t_dof] = q * dx * dx / 12.0;
                } else {
                    rhs[v_dof] = -q * dx;
                    rhs[t_dof] = 0.0;
                }
            }

            apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1, 2], &[0.0, 0.0, 0.0]);

            let result = gmres(
                &csr,
                &rhs,
                &GmresConfig {
                    max_iterations: 1000,
                    tolerance: 8e-11,
                    convergence_check_interval: 1,
                    min_restart: None,
                    max_restart: None,
                    ..Default::default()
                },
            );
            assert!(result.converged);

            let tip_delta = result.x[(n_nodes - 1) * 3 + 1].abs();
            let exact_tip = beam_euler_cantilever_uniform(l, l, e_mod, i_inertia, q);
            (tip_delta - exact_tip).abs() / exact_tip
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (5.0_f64 / 10.0).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (10.0_f64 / 20.0).log(std::f64::consts::LN_10);

    println!(
        "Beam cantilever uniform convergence: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );
    // Solution is exact to machine precision for all meshes with correct equivalent loads
    assert!(
        errors[0] < 1e-9 && errors[1] < 1e-9 && errors[2] < 1e-9,
        "Error too large"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_beam_cantilever_uniform();
        test_beam_cantilever_uniform_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
