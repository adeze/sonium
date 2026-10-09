//! Wolfram transient 1D heat-transfer verification.
//!
//! This reproduces
//! `HeatTransfer-FEM-Transient-1D-Single-HeatTransfer-0001`: a 0.1 m bar,
//! initially at 0, driven at its left end by `100 sin(pi t / 40)` and held at
//! 0 at its right end. Wolfram publishes `T(32 s, 0.02 m) = 36.6`.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::thermal::heat_conduction::{
    ConductionForm, HeatCapacityForm, ThermalMaterial, ThermalTimeStepper,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{CsrMatrix, LinearSolveConfig, LinearSolverType, solve_real};
use std::f64::consts::PI;

const LENGTH_M: f64 = 0.1;
const WIDTH_M: f64 = 0.01;
const CONDUCTIVITY_W_M_K: f64 = 35.0;
const DENSITY_KG_M3: f64 = 7200.0;
const SPECIFIC_HEAT_J_KG_K: f64 = 440.5;
const END_TIME_S: f64 = 32.0;
const WOLFRAM_PROBE_X_M: f64 = 0.02;
const WOLFRAM_PROBE_TEMPERATURE: f64 = 36.6;

fn apply_dirichlet_row(matrix: &mut CsrMatrix<f64>, rhs: &mut [f64], dof: usize, value: f64) {
    for entry in matrix.row_range(dof) {
        matrix.values[entry] = 0.0;
    }
    let diagonal_offset = matrix
        .row_entries(dof)
        .position(|(column, _)| column == dof)
        .expect("Dirichlet row has a diagonal entry");
    matrix.values[matrix.row_ptrs[dof] + diagonal_offset] = 1.0;
    rhs[dof] = value;
}

#[test]
fn sonium_reproduces_wolfram_transient_heated_bar() {
    // A thin insulated rectangle is the 2D weak-form embedding of Wolfram's
    // Line domain. The exact solution is constant through its width.
    let mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, WIDTH_M, 100, 2);
    let space = H1Space::new(PolynomialDegree::P1);
    let dof_count = space.num_global_dofs(&mesh);
    let material = ThermalMaterial::new(CONDUCTIVITY_W_M_K, DENSITY_KG_M3, SPECIFIC_HEAT_J_KG_K);
    let conduction = ConductionForm::from_conductivity(CONDUCTIVITY_W_M_K);
    let capacity = HeatCapacityForm::new(&material);
    let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1).to_csr();
    let mass = assemble_bilinear(&capacity, &space, &space, &mesh, 1).to_csr();

    let time_step_s = 0.1;
    let step_count = (END_TIME_S / time_step_s) as usize;
    let stepper = ThermalTimeStepper::crank_nicolson(time_step_s);
    let effective = stepper.effective_matrix(&mass, &stiffness);
    let zero_source = vec![0.0; dof_count];
    let left_dofs: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| (node.x.abs() < 1.0e-12).then_some(index))
        .collect();
    let right_dofs: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - LENGTH_M).abs() < 1.0e-12).then_some(index))
        .collect();
    assert_eq!(left_dofs.len(), 3);
    assert_eq!(right_dofs.len(), 3);

    let solve_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 500,
        tolerance: 1.0e-11,
        restart: 50,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };
    let mut temperature = vec![0.0; dof_count];

    for step in 1..=step_count {
        let time_s = step as f64 * time_step_s;
        let left_temperature = 100.0 * (PI * time_s / 40.0).sin();
        let mut rhs = stepper.form_rhs(
            &temperature,
            &zero_source,
            Some(&zero_source),
            &mass,
            &stiffness,
        );
        let mut matrix = effective.clone();
        for &dof in &left_dofs {
            apply_dirichlet_row(&mut matrix, &mut rhs, dof, left_temperature);
        }
        for &dof in &right_dofs {
            apply_dirichlet_row(&mut matrix, &mut rhs, dof, 0.0);
        }
        let result = solve_real(&matrix, &rhs, &solve_config).expect("thermal solve");
        assert!(result.converged, "thermal step {step} did not converge");
        temperature = result.x;
    }

    let probe_values: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            ((node.x - WOLFRAM_PROBE_X_M).abs() < 1.0e-12).then_some(temperature[index])
        })
        .collect();
    assert_eq!(probe_values.len(), 3);
    let probe_temperature = probe_values.iter().sum::<f64>() / probe_values.len() as f64;
    let through_width_spread = probe_values
        .iter()
        .map(|value| (value - probe_temperature).abs())
        .fold(0.0_f64, f64::max);
    let relative_error =
        (probe_temperature - WOLFRAM_PROBE_TEMPERATURE).abs() / WOLFRAM_PROBE_TEMPERATURE;

    eprintln!(
        "Wolfram transient heat: T(32 s, 0.02 m) = {probe_temperature:.8}, rel_err = {relative_error:.3e}"
    );
    assert!(
        relative_error < 1.0e-3,
        "probe relative error {relative_error:.3e} exceeds Wolfram's 0.1% gate"
    );
    assert!(
        through_width_spread < 0.05,
        "thin 2D embedding varies across the insulated width by {through_width_spread:.3e} K"
    );
}

fn wolfram_flux_case_reference(time_s: f64, x_m: f64) -> f64 {
    (24.0 + x_m)
        + 8.0 / PI.powi(2) * (PI * x_m / 2.0).cos() * (-PI.powi(2) * time_s / 4.0).exp()
        + 8.0 / (9.0 * PI.powi(2))
            * (3.0 * PI * x_m / 2.0).cos()
            * (-9.0 * PI.powi(2) * time_s / 4.0).exp()
}

#[test]
fn sonium_reproduces_wolfram_transient_flux_profile() {
    // Official case HeatTransfer-FEM-Transient-1D-Single-HeatTransfer-0002:
    // unit material coefficients, T(0,x)=25, q=-1 at x=0, and T=25 at x=1.
    let length_m = 1.0;
    let width_m = 0.01;
    let end_time_s = 0.2;
    let time_step_s = 0.001;
    let mesh = rectangular_mesh_triangles(0.0, length_m, 0.0, width_m, 100, 1);
    let space = H1Space::new(PolynomialDegree::P1);
    let dof_count = space.num_global_dofs(&mesh);
    let material = ThermalMaterial::new(1.0, 1.0, 1.0);
    let stiffness = assemble_bilinear(
        &ConductionForm::from_conductivity(1.0),
        &space,
        &space,
        &mesh,
        1,
    )
    .to_csr();
    let mass =
        assemble_bilinear(&HeatCapacityForm::new(&material), &space, &space, &mesh, 1).to_csr();

    let mut left_dofs: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| (node.x.abs() < 1.0e-12).then_some(index))
        .collect();
    left_dofs.sort_by(|&left, &right| mesh.nodes[left].y.total_cmp(&mesh.nodes[right].y));
    let right_dofs: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - length_m).abs() < 1.0e-12).then_some(index))
        .collect();
    assert_eq!(left_dofs.len(), 2);
    assert_eq!(right_dofs.len(), 2);

    // Consistent P1 boundary load: integral_edge N_i q ds, with Wolfram's
    // HeatFluxValue q=-1. Dividing through by the strip width recovers its 1D
    // Line problem.
    let mut flux_source = vec![0.0; dof_count];
    for edge in left_dofs.windows(2) {
        let edge_length = mesh.nodes[edge[1]].y - mesh.nodes[edge[0]].y;
        flux_source[edge[0]] -= edge_length / 2.0;
        flux_source[edge[1]] -= edge_length / 2.0;
    }

    let stepper = ThermalTimeStepper::crank_nicolson(time_step_s);
    let effective = stepper.effective_matrix(&mass, &stiffness);
    let solve_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 500,
        tolerance: 1.0e-12,
        restart: 50,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };
    let mut temperature = vec![25.0; dof_count];
    let step_count = (end_time_s / time_step_s) as usize;
    for step in 1..=step_count {
        let mut rhs = stepper.form_rhs(
            &temperature,
            &flux_source,
            Some(&flux_source),
            &mass,
            &stiffness,
        );
        let mut matrix = effective.clone();
        for &dof in &right_dofs {
            apply_dirichlet_row(&mut matrix, &mut rhs, dof, 25.0);
        }
        let result = solve_real(&matrix, &rhs, &solve_config).expect("thermal flux solve");
        assert!(
            result.converged,
            "thermal flux step {step} did not converge"
        );
        temperature = result.x;
    }

    let endpoint_average = |x_m: f64| {
        let values: Vec<_> = mesh
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                ((node.x - x_m).abs() < 1.0e-12).then_some(temperature[index])
            })
            .collect();
        values.iter().sum::<f64>() / values.len() as f64
    };
    let computed = [endpoint_average(0.0), endpoint_average(1.0)];
    let reference = [
        wolfram_flux_case_reference(end_time_s, 0.0),
        wolfram_flux_case_reference(end_time_s, 1.0),
    ];
    let error_norm = computed
        .iter()
        .zip(reference)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();

    eprintln!(
        "Wolfram transient flux: endpoints = {computed:?}, reference = {reference:?}, norm = {error_norm:.3e}"
    );
    assert!(
        error_norm < 1.0e-3,
        "endpoint error norm {error_norm:.3e} exceeds Wolfram's 1e-3 gate"
    );
}
