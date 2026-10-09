//! Wolfram transient axisymmetric heat-transfer verification.
//!
//! Reproduces `HeatTransfer-FEM-Transient-2DAxisym-Single-HeatTransfer-0001`:
//! a solid steel cylinder represented on `(r,z) in [0,0.3] x [0,0.4]`, with
//! the outer radius and both end faces held at 1000 and an initial temperature
//! of zero. Wolfram publishes `T(190, 0.1, 0.3) = 186.5` with a 1% gate.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::{
    solid::axisymmetric::{AxisymmetricScalarGradientForm, AxisymmetricScalarMassForm},
    thermal::heat_conduction::ThermalTimeStepper,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{CsrMatrix, LinearSolveConfig, LinearSolverType, solve_real};

const RADIUS_M: f64 = 0.3;
const HEIGHT_M: f64 = 0.4;
const CONDUCTIVITY_W_M_K: f64 = 52.0;
const DENSITY_KG_M3: f64 = 7850.0;
const SPECIFIC_HEAT_J_KG_K: f64 = 460.0;
const BOUNDARY_TEMPERATURE: f64 = 1000.0;
const END_TIME_S: f64 = 190.0;
const WOLFRAM_PROBE_TEMPERATURE: f64 = 186.5;

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
fn sonium_reproduces_wolfram_transient_axisymmetric_cylinder() {
    let mesh = rectangular_mesh_triangles(0.0, RADIUS_M, 0.0, HEIGHT_M, 45, 60);
    let space = H1Space::new(PolynomialDegree::P1);
    let dof_count = space.num_global_dofs(&mesh);
    let stiffness = assemble_bilinear(
        &AxisymmetricScalarGradientForm::new(CONDUCTIVITY_W_M_K),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let mass = assemble_bilinear(
        &AxisymmetricScalarMassForm::new(DENSITY_KG_M3 * SPECIFIC_HEAT_J_KG_K),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();

    let boundary_dofs: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            ((node.x - RADIUS_M).abs() < 1.0e-12
                || node.y.abs() < 1.0e-12
                || (node.y - HEIGHT_M).abs() < 1.0e-12)
                .then_some(index)
        })
        .collect();
    assert_eq!(boundary_dofs.len(), 151);

    let time_step_s = 1.0;
    let stepper = ThermalTimeStepper::crank_nicolson(time_step_s);
    let effective = stepper.effective_matrix(&mass, &stiffness);
    let zero_source = vec![0.0; dof_count];
    let solve_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 500,
        tolerance: 1.0e-11,
        restart: 50,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };
    let mut temperature = vec![0.0; dof_count];

    for step in 1..=END_TIME_S as usize {
        let mut rhs = stepper.form_rhs(
            &temperature,
            &zero_source,
            Some(&zero_source),
            &mass,
            &stiffness,
        );
        let mut matrix = effective.clone();
        for &dof in &boundary_dofs {
            apply_dirichlet_row(&mut matrix, &mut rhs, dof, BOUNDARY_TEMPERATURE);
        }
        let result = solve_real(&matrix, &rhs, &solve_config).expect("axisymmetric thermal solve");
        assert!(
            result.converged,
            "axisymmetric thermal step {step} did not converge"
        );
        temperature = result.x;
    }

    let probe_index = mesh
        .nodes
        .iter()
        .position(|node| (node.x - 0.1).abs() < 1.0e-12 && (node.y - 0.3).abs() < 1.0e-12)
        .expect("probe is a mesh node");
    let probe_temperature = temperature[probe_index];
    let relative_error =
        (probe_temperature - WOLFRAM_PROBE_TEMPERATURE).abs() / WOLFRAM_PROBE_TEMPERATURE;

    eprintln!(
        "Wolfram transient axisymmetric heat: T(190,0.1,0.3) = {probe_temperature:.8}, rel_err = {relative_error:.3e}"
    );
    assert!(
        relative_error < 1.0e-2,
        "probe relative error {relative_error:.3e} exceeds Wolfram's 1% gate"
    );
}
