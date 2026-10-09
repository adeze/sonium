//! QA: FEM thermal — 1D transient conduction in a semi-infinite solid.
//!
//! Analytical: T(x,t) = Ti + (Ts - Ti) * erfc(x / (2*sqrt(alpha*t)))
//! We model a finite bar long enough that the thermal wave does not reach
//! the far end during the simulation horizon.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::thermal::heat_conduction::{
    ConductionForm, HeatCapacityForm, ThermalMaterial, ThermalTimeStepper,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::analytical::thermal_1d_transient;
use sonium_solvers::{CsrMatrix, LinearSolveConfig, LinearSolverType, solve_real};

fn apply_dirichlet_csr(csr: &mut CsrMatrix<f64>, rhs: &mut [f64], dof: usize, value: f64) {
    let range = csr.row_range(dof);
    for i in range {
        csr.values[i] = 0.0;
    }
    let diag_pos = csr.row_entries(dof).position(|(j, _)| j == dof);
    if let Some(idx) = diag_pos {
        let global_idx = csr.row_ptrs[dof] + idx;
        csr.values[global_idx] = 1.0;
    }
    rhs[dof] = value;
}

fn test_thermal_1d_transient() {
    let l = 0.2; // Still >12 diffusion lengths at t=5 s, with useful resolution
    let w = 0.05;
    let k = 50.0;
    let rho = 7800.0;
    let cp = 500.0;
    let alpha = k / (rho * cp);
    let ti = 20.0; // Initial temperature
    let ts = 100.0; // Surface temperature at x=0

    let nx = 400;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, ny);

    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = ThermalMaterial::new(k, rho, cp);
    let conduction = ConductionForm::from_conductivity(k);
    let capacity = HeatCapacityForm::new(&material);

    let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1);
    let mass = assemble_bilinear(&capacity, &space, &space, &mesh, 1);

    let dt = 0.025; // time step, refined with the P1 spatial scale
    let n_steps = 200; // total simulation time = 5.0 s

    // At t=5s, thermal diffusion length ~ 2*sqrt(alpha*t) ≈ 2*sqrt(1.28e-5*5) ≈ 0.016 m
    // A 0.2 m bar is still effectively semi-infinite while avoiding a mesh
    // that spends 99% of its cells far outside the thermal boundary layer.

    let mut temperature = vec![ti; n_dofs];

    let solve_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 1000,
        tolerance: 1e-12,
        restart: 50,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };

    let stepper = ThermalTimeStepper::backward_euler(dt);
    let effective = stepper.effective_matrix(&mass.to_csr(), &stiffness.to_csr());

    // Identify left-boundary DOFs (x = 0, marker 1)
    let left_dofs: Vec<usize> = space.boundary_dofs(&mesh, 1);

    for _step in 0..n_steps {
        let mut rhs = stepper.form_rhs(
            &temperature,
            &vec![0.0; n_dofs],
            None,
            &mass.to_csr(),
            &stiffness.to_csr(),
        );
        let mut csr = effective.clone();

        for &dof in &left_dofs {
            apply_dirichlet_csr(&mut csr, &mut rhs, dof, ts);
        }

        let result = solve_real(&csr, &rhs, &solve_config).expect("Solver should converge");
        assert!(result.converged, "Thermal solver did not converge");

        temperature = result.x;
    }

    let t_final = dt * n_steps as f64;

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let x = node.x;
        let t_exact = thermal_1d_transient(x, t_final, alpha, ti, ts);
        let t_fem = temperature[i];
        max_err = max_err.max((t_fem - t_exact).abs());
    }

    println!(
        "Thermal 1D transient: max_err = {:.4e} at t = {:.1} s",
        max_err, t_final
    );
    let normalized_error = max_err / (ts - ti).abs();
    assert!(
        normalized_error < 0.004,
        "Normalized transient temperature error {:.3}% exceeds 0.4%",
        100.0 * normalized_error
    );
}

fn test_thermal_1d_transient_convergence() {
    let l = 0.2;
    let w = 0.05;
    let k = 50.0;
    let rho = 7800.0;
    let cp = 500.0;
    let alpha = k / (rho * cp);
    let ti = 20.0;
    let ts = 100.0;
    let t_final = 5.0;

    let errors: Vec<f64> = [100, 200, 400]
        .iter()
        .map(|&nx| {
            // Backward Euler is first order in time. Refine dt together with
            // h so a fixed temporal floor cannot masquerade as failed spatial
            // convergence.
            let dt = 10.0 / nx as f64;
            let n_steps = (t_final / dt).round() as usize;
            let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, 2);
            let space = H1Space::new(PolynomialDegree::P1);
            let n_dofs = space.num_global_dofs(&mesh);

            let material = ThermalMaterial::new(k, rho, cp);
            let conduction = ConductionForm::from_conductivity(k);
            let capacity = HeatCapacityForm::new(&material);

            let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1);
            let mass = assemble_bilinear(&capacity, &space, &space, &mesh, 1);

            let mut temperature = vec![ti; n_dofs];

            let solve_config = LinearSolveConfig {
                solver: LinearSolverType::GmresIlu,
                max_iterations: 1000,
                tolerance: 1e-12,
                restart: 50,
                exact_residual_check_interval: 0,
                verbosity: 0,
            };

            let stepper = ThermalTimeStepper::backward_euler(dt);
            let effective = stepper.effective_matrix(&mass.to_csr(), &stiffness.to_csr());
            let left_dofs = space.boundary_dofs(&mesh, 1);

            for _ in 0..n_steps {
                let mut rhs = stepper.form_rhs(
                    &temperature,
                    &vec![0.0; n_dofs],
                    None,
                    &mass.to_csr(),
                    &stiffness.to_csr(),
                );
                let mut csr = effective.clone();
                for &dof in &left_dofs {
                    apply_dirichlet_csr(&mut csr, &mut rhs, dof, ts);
                }
                let result = solve_real(&csr, &rhs, &solve_config).expect("Solver should converge");
                assert!(result.converged);
                temperature = result.x;
            }

            let mut max_err: f64 = 0.0;
            for (i, node) in mesh.nodes.iter().enumerate() {
                let t_exact = thermal_1d_transient(node.x, t_final, alpha, ti, ts);
                max_err = max_err.max((temperature[i] - t_exact).abs());
            }
            max_err
        })
        .collect();

    let rate_1 = (errors[0] / errors[1]).log10() / 2.0f64.log10();
    let rate_2 = (errors[1] / errors[2]).log10() / 2.0f64.log10();

    println!(
        "Transient convergence: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );
    assert!(
        rate_1 > 1.0 && rate_2 > 1.0,
        "Coupled space-time convergence rates are too low: {rate_1:.2}, {rate_2:.2}"
    );
    let normalized_error = errors[2] / (ts - ti).abs();
    assert!(
        normalized_error < 0.004,
        "Finest normalized transient error {:.3}% exceeds 0.4%",
        100.0 * normalized_error
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_thermal_1d_transient();
        test_thermal_1d_transient_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
