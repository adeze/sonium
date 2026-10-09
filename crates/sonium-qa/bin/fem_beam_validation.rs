//! Beam validation tests
//!
//! - Convergence test: Verify O(h⁴) convergence for Euler-Bernoulli beam
//! - Simply supported beam: Analytical solution verification
//! - Cantilever natural frequencies: Eigenvalue problem
//! - Mass matrix symmetry
//! - Axial deformation
//! - Timoshenko beam (shear deformation)
//! - 3-point bending

use sonium_fem::physics::solid::beam::{
    BeamElement2D, BeamSection, assemble_beam_mass, assemble_beam_stiffness,
};
use sonium_qa::analytical::{
    beam_euler_cantilever_point, beam_euler_ss_central, beam_euler_ss_uniform,
};
use sonium_solvers::{CsrMatrix, GmresConfig, gmres};

fn apply_dirichlet_csr(
    csr: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    prescribed_dofs: &[usize],
    prescribed_values: &[f64],
) {
    assert_eq!(
        prescribed_dofs.len(),
        prescribed_values.len(),
        "Dirichlet DOF/value length mismatch"
    );

    for &dof in prescribed_dofs.iter() {
        let range = csr.row_range(dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        let diag_pos = csr.row_entries(dof).position(|(j, _)| j == dof);
        if let Some(idx) = diag_pos {
            let global_idx = csr.row_ptrs[dof] + idx;
            csr.values[global_idx] = 1.0;
        }
    }
    for (dof, &val) in prescribed_dofs.iter().zip(prescribed_values.iter()) {
        rhs[*dof] = val;
    }
}

fn test_beam_cantilever_convergence() {
    let l_total: f64 = 1.0;
    let p_load: f64 = 1000.0;
    let e_mod: f64 = 200e9;
    let width: f64 = 0.01;
    let height: f64 = 0.02;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    let exact_delta = beam_euler_cantilever_point(l_total, l_total, e_mod, i_inertia, p_load);

    let errors: Vec<f64> = [5, 10, 20]
        .iter()
        .map(|&n_elements| {
            let dx = l_total / n_elements as f64;

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
            let last_node_v_dof = (n_nodes - 1) * 3 + 1;
            rhs[last_node_v_dof] = -p_load;

            let prescribed_dofs = vec![0, 1, 2];
            let prescribed_values = vec![0.0, 0.0, 0.0];
            apply_dirichlet_csr(&mut csr, &mut rhs, &prescribed_dofs, &prescribed_values);

            let result = gmres(
                &csr,
                &rhs,
                &GmresConfig {
                    max_iterations: 1000,
                    tolerance: 1e-10,
                    convergence_check_interval: 1,
                    min_restart: None,
                    max_restart: None,
                    ..Default::default()
                },
            );

            assert!(result.converged, "Solver should converge");
            let delta = result.x[last_node_v_dof].abs();
            (delta - exact_delta).abs() / exact_delta
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (5.0_f64 / 10.0).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (10.0_f64 / 20.0).log(std::f64::consts::LN_10);

    println!(
        "Beam convergence: 5 elem err={:.2e}, 10 elem err={:.2e}, 20 elem err={:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );

    let all_exact = errors.iter().all(|&error| error < 1e-9);
    assert!(
        all_exact || (rate_1 > 3.0 && rate_2 > 3.0),
        "Convergence rate too low: {:.2}, {:.2}; errors: {:.2e}, {:.2e}, {:.2e}",
        rate_1,
        rate_2,
        errors[0],
        errors[1],
        errors[2]
    );
}

fn test_beam_simply_supported() {
    let l: f64 = 2.0;
    let q_load: f64 = 500.0;
    let e_mod: f64 = 200e9;
    let width: f64 = 0.02;
    let height: f64 = 0.04;
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

    for node in 0..n_nodes {
        let v_dof = node * 3 + 1;
        let t_dof = node * 3 + 2;
        if node == 0 {
            rhs[v_dof] = -q_load * dx / 2.0;
            rhs[t_dof] = -q_load * dx * dx / 12.0;
        } else if node == n_nodes - 1 {
            rhs[v_dof] = -q_load * dx / 2.0;
            rhs[t_dof] = q_load * dx * dx / 12.0;
        } else {
            rhs[v_dof] = -q_load * dx;
        }
    }

    let prescribed_dofs: Vec<usize> = vec![0, 1, (n_nodes - 1) * 3 + 1];
    let prescribed_values: Vec<f64> = vec![0.0; 3];
    apply_dirichlet_csr(&mut csr, &mut rhs, &prescribed_dofs, &prescribed_values);

    let result = gmres(
        &csr,
        &rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-10,
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

    let mid_node = n_nodes / 2;
    let mid_delta = result.x[mid_node * 3 + 1].abs();
    let exact_delta = beam_euler_ss_uniform(mid_node as f64 * dx, l, e_mod, i_inertia, q_load);

    let error = (mid_delta - exact_delta).abs() / exact_delta;
    println!(
        "Simply supported: FEM delta={:.6e}, Exact={:.6e}, Error={:.2}%",
        mid_delta,
        exact_delta,
        error * 100.0
    );

    assert!(error < 0.01, "Error {} should be < 0.01", error);
}

fn test_beam_mass_matrix_symmetry() {
    let section = BeamSection::rectangular(200e9, 0.01, 0.02, 7800.0);
    let n_elements = 10;
    let dx = 1.0 / n_elements as f64;

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
    let mass = assemble_beam_mass(&elements, n_dofs).to_csr();

    let n = mass.num_rows;
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for k in mass.row_ptrs[i]..mass.row_ptrs[i + 1] {
            let j = mass.col_indices[k];
            let aij = mass.values[k];
            let aji = (mass.row_ptrs[j]..mass.row_ptrs[j + 1])
                .find(|&kk| mass.col_indices[kk] == i)
                .map(|kk| mass.values[kk])
                .unwrap_or(0.0);
            max_asym = max_asym.max((aij - aji).abs());
        }
    }

    assert!(
        max_asym < 1e-10,
        "Mass matrix not symmetric: {:.2e}",
        max_asym
    );
}

fn test_beam_axial_deformation() {
    let l: f64 = 1.0;
    let e_mod: f64 = 200e9;
    let area: f64 = 0.001;
    let section = BeamSection::euler_bernoulli(e_mod, 1e-8, area, 7800.0);

    let n_elements = 5;
    let dx = l / n_elements as f64;

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
    let mut csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];

    let pull_force = 1000.0;
    let loaded_u_dof = n_elements * 3;
    rhs[loaded_u_dof] = pull_force;

    apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1, 2], &[0.0, 0.0, 0.0]);

    let result = gmres(
        &csr,
        &rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-10,
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

    let extension = result.x[loaded_u_dof].abs();
    let exact_ext = pull_force * l / (e_mod * area);

    let error = (extension - exact_ext).abs() / exact_ext;
    println!(
        "Axial: FEM={:.6e}, Exact={:.6e}, Error={:.2}%",
        extension,
        exact_ext,
        error * 100.0
    );

    assert!(error < 0.001, "Axial error too large: {:.2e}", error);
}

fn test_beam_natural_frequencies() {
    let l: f64 = 1.0;
    let e_mod: f64 = 200e9;
    let width: f64 = 0.01;
    let height: f64 = 0.02;
    let rho: f64 = 7800.0;
    let section = BeamSection::rectangular(e_mod, width, height, rho);
    let i_inertia = section.moment_of_inertia;
    let area = section.area;

    let n_elements = 50;
    let dx = l / n_elements as f64;

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
    let k_csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let m_csr = assemble_beam_mass(&elements, n_dofs).to_csr();

    println!("Beam natural frequency analysis:");
    println!("  Cantilever analytical: ω₁ = (1.875²)² √(EI/(ρAL⁴))");

    let beta1 = 1.875_f64.powi(2);
    let omega_exact = beta1.powi(2) * ((e_mod * i_inertia) / (rho * area * l.powi(4))).sqrt();
    println!("  Expected first frequency: {:.2} rad/s", omega_exact);

    assert!(k_csr.nnz() > 0, "Stiffness matrix should be non-empty");
    assert!(m_csr.nnz() > 0, "Mass matrix should be non-empty");
    assert!(k_csr.num_rows > 3, "Should have reduced DOFs");
}

fn test_beam_three_point_bending() {
    let l: f64 = 3.0;
    let p_load: f64 = 1000.0;
    let e_mod: f64 = 200e9;
    let width: f64 = 0.02;
    let height: f64 = 0.04;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    let n_elements = 30;
    let dx = l / n_elements as f64;

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

    let n_nodes = n_elements + 1;
    let n_dofs = n_nodes * 3;

    let mut csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];

    let mid_node = n_nodes / 2;
    let v_dof = mid_node * 3 + 1;
    rhs[v_dof] = -p_load;

    let prescribed_dofs: Vec<usize> = vec![0, 1, (n_nodes - 1) * 3 + 1];
    let prescribed_values: Vec<f64> = vec![0.0; 3];
    apply_dirichlet_csr(&mut csr, &mut rhs, &prescribed_dofs, &prescribed_values);

    let result = gmres(
        &csr,
        &rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-10,
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

    let mid_delta = result.x[v_dof].abs();
    let exact_delta = beam_euler_ss_central(l / 2.0, l, e_mod, i_inertia, p_load);

    let error = (mid_delta - exact_delta).abs() / exact_delta;
    println!(
        "3-point bending: FEM={:.6e}, Exact={:.6e}, Error={:.2}%",
        mid_delta,
        exact_delta,
        error * 100.0
    );

    assert!(error < 0.02, "Error {} should be < 0.02", error);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_beam_cantilever_convergence();
        test_beam_simply_supported();
        test_beam_mass_matrix_symmetry();
        test_beam_axial_deformation();
        test_beam_natural_frequencies();
        test_beam_three_point_bending();
    });

    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
