//! QA: FEM Timoshenko beam — cantilever with end point load.
//!
//! Validates shear-deformable beam theory against exact analytical solution:
//!   w_tip = P*L^3/(3*E*I) + P*L/(kappa*G*A)
//!
//! The first term is bending deflection, the second is shear deflection.
//! For slender beams shear is negligible; for thick beams it dominates.

use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection, assemble_beam_stiffness};
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

/// Exact tip deflection for Timoshenko cantilever with end load P.
fn exact_timoshenko_tip(l: f64, p: f64, e: f64, i: f64, a: f64, nu: f64, kappa: f64) -> f64 {
    let g = e / (2.0 * (1.0 + nu));
    let bending = p * l.powi(3) / (3.0 * e * i);
    let shear = p * l / (kappa * g * a);
    bending + shear
}

/// Exact tip deflection for Euler-Bernoulli cantilever with end load P.
fn exact_euler_bernoulli_tip(l: f64, p: f64, e: f64, i: f64) -> f64 {
    p * l.powi(3) / (3.0 * e * i)
}

fn test_timoshenko_thick_beam() {
    // Thick beam: h/L = 1/5, shear deflection is ~10% of bending
    let l = 1.0;
    let p = 1000.0;
    let e_mod = 200e9;
    let nu = 0.3;
    let width = 0.05;
    let height: f64 = 0.2; // h/L = 0.2
    let kappa = 5.0 / 6.0; // rectangular section
    let a = width * height;
    let i_inertia = width * height.powi(3) / 12.0;

    let section = BeamSection::timoshenko(e_mod, i_inertia, a, 7800.0, nu, kappa);

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

    // Point load P downward at free end (DOF 1 = vertical displacement)
    rhs[(n_nodes - 1) * 3 + 1] = -p;

    // Clamp at left end: u=0, w=0, theta=0
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

    let tip_node = n_nodes - 1;
    let tip_delta = result.x[tip_node * 3 + 1].abs();
    let exact_tip = exact_timoshenko_tip(l, p, e_mod, i_inertia, a, nu, kappa);
    let euler_tip = exact_euler_bernoulli_tip(l, p, e_mod, i_inertia);

    let error = (tip_delta - exact_tip).abs() / exact_tip;
    let shear_ratio = (exact_tip - euler_tip) / exact_tip;

    println!(
        "Timoshenko thick beam: FEM tip={:.6e}, Exact={:.6e}, Error={:.2e} ({:.4}%)\n  \
         Euler-Bernoulli={:.6e}, shear fraction={:.2}%",
        tip_delta,
        exact_tip,
        error,
        error * 100.0,
        euler_tip,
        shear_ratio * 100.0
    );

    assert!(
        error < 1e-9,
        "Timoshenko thick beam tip deflection error too large: {:.2e}",
        error
    );
    // Verify shear deflection is significant (~3% for this geometry)
    assert!(
        shear_ratio > 0.02 && shear_ratio < 0.05,
        "Shear fraction should be ~3% for h/L=0.2, got {:.2}%",
        shear_ratio * 100.0
    );
}

fn test_timoshenko_slender_beam() {
    // Slender beam: h/L = 1/20, shear deflection is negligible
    let l = 1.0;
    let p = 1000.0;
    let e_mod = 200e9;
    let nu = 0.3;
    let width = 0.01;
    let height: f64 = 0.05; // h/L = 0.05
    let kappa = 5.0 / 6.0;
    let a = width * height;
    let i_inertia = width * height.powi(3) / 12.0;

    let section = BeamSection::timoshenko(e_mod, i_inertia, a, 7800.0, nu, kappa);

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

    rhs[(n_nodes - 1) * 3 + 1] = -p;
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

    assert!(result.converged);

    let tip_delta = result.x[(n_nodes - 1) * 3 + 1].abs();
    let exact_tip = exact_timoshenko_tip(l, p, e_mod, i_inertia, a, nu, kappa);
    let euler_tip = exact_euler_bernoulli_tip(l, p, e_mod, i_inertia);

    let error = (tip_delta - exact_tip).abs() / exact_tip;
    let shear_ratio = (exact_tip - euler_tip) / exact_tip;

    println!(
        "Timoshenko slender beam: FEM tip={:.6e}, Exact={:.6e}, Error={:.2e}\n  \
         shear fraction={:.3}%",
        tip_delta,
        exact_tip,
        error,
        shear_ratio * 100.0
    );

    assert!(error < 1e-9, "Slender beam error too large: {:.2e}", error);
    // Shear should be < 1% for h/L = 0.05
    assert!(
        shear_ratio < 0.01,
        "Shear fraction should be <1% for h/L=0.05, got {:.3}%",
        shear_ratio * 100.0
    );
}

fn test_timoshenko_vs_euler_bernoulli() {
    // Direct comparison: same geometry, Timoshenko vs Euler-Bernoulli section
    let l = 1.0;
    let p = 1000.0;
    let e_mod = 200e9;
    let nu = 0.3;
    let width = 0.02;
    let height: f64 = 0.1;
    let kappa = 5.0 / 6.0;
    let a = width * height;
    let i_inertia = width * height.powi(3) / 12.0;

    let section_ts = BeamSection::timoshenko(e_mod, i_inertia, a, 7800.0, nu, kappa);
    let section_eb = BeamSection::euler_bernoulli(e_mod, i_inertia, a, 7800.0);

    let n_elements = 10;
    let dx = l / n_elements as f64;

    for (name, section) in [("Timoshenko", section_ts), ("Euler-Bernoulli", section_eb)] {
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

        rhs[(n_nodes - 1) * 3 + 1] = -p;
        apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1, 2], &[0.0, 0.0, 0.0]);

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
        assert!(result.converged);

        let tip_delta = result.x[(n_nodes - 1) * 3 + 1].abs();
        println!("  {} tip deflection: {:.6e}", name, tip_delta);
    }

    // Timoshenko should be larger (softer) than Euler-Bernoulli
    let elements_ts: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            let x1 = i as f64 * dx;
            let x2 = (i + 1) as f64 * dx;
            BeamElement2D::new([i, i + 1], x1, 0.0, x2, 0.0, section_ts)
        })
        .collect();
    let elements_eb: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            let x1 = i as f64 * dx;
            let x2 = (i + 1) as f64 * dx;
            BeamElement2D::new([i, i + 1], x1, 0.0, x2, 0.0, section_eb)
        })
        .collect();

    let n_nodes = n_elements + 1;
    let n_dofs = n_nodes * 3;

    let mut csr_ts = assemble_beam_stiffness(&elements_ts, n_dofs).to_csr();
    let mut rhs_ts = vec![0.0; n_dofs];
    rhs_ts[(n_nodes - 1) * 3 + 1] = -p;
    apply_dirichlet_csr(&mut csr_ts, &mut rhs_ts, &[0, 1, 2], &[0.0, 0.0, 0.0]);

    let mut csr_eb = assemble_beam_stiffness(&elements_eb, n_dofs).to_csr();
    let mut rhs_eb = vec![0.0; n_dofs];
    rhs_eb[(n_nodes - 1) * 3 + 1] = -p;
    apply_dirichlet_csr(&mut csr_eb, &mut rhs_eb, &[0, 1, 2], &[0.0, 0.0, 0.0]);

    let res_ts = gmres(
        &csr_ts,
        &rhs_ts,
        &GmresConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        },
    );
    let res_eb = gmres(
        &csr_eb,
        &rhs_eb,
        &GmresConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        },
    );

    let tip_ts = res_ts.x[(n_nodes - 1) * 3 + 1].abs();
    let tip_eb = res_eb.x[(n_nodes - 1) * 3 + 1].abs();

    println!(
        "  Ratio Timoshenko/Euler-Bernoulli = {:.4}",
        tip_ts / tip_eb
    );
    assert!(
        tip_ts > tip_eb,
        "Timoshenko should be softer than Euler-Bernoulli"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_timoshenko_thick_beam();
        test_timoshenko_slender_beam();
        test_timoshenko_vs_euler_bernoulli();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
