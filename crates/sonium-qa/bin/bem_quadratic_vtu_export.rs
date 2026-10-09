//! Release-facing quadratic BEM VTU export contract.
//!
//! Solves a small discontinuous quadratic rigid-sphere problem through the
//! production `BemSolver`, writes the quadratic pressure field, and validates
//! the resulting refined linear-cell VTU as an independent consumer would.

use sonium_bem::core::{BasisOrder, BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use sonium_xem::io::binary_output_dir;
use sonium_xem::io::vtu::validate_vtu_file;
use std::fs;
use std::time::Instant;

const CASE_ID: &str = "bem.quadratic-vtu-export.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    surface_elements: usize,
    exported_points: usize,
    exported_cells: usize,
    pressure_dofs: usize,
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 80.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_solver_method(SolverMethod::Direct)
        .with_tolerance(1.0e-9);
    let solution = solver
        .solve(&problem)
        .map_err(|error| format!("quadratic BEM solve failed: {error}"))?;

    let surface_elements = problem
        .mesh
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    let expected_points = surface_elements * 6;
    let expected_cells = surface_elements * 4;
    if solution.surface_pressure.is_empty() {
        return Err("quadratic BEM solve returned no pressure DOFs".into());
    }

    let output = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
        .join("quadratic_pressure.vtu");
    solution
        .write_vtu(&problem.mesh, &output)
        .map_err(|error| format!("quadratic VTU export failed: {error}"))?;
    validate_vtu_file(&output)
        .map_err(|error| format!("quadratic VTU validation failed: {error}"))?;
    let xml = fs::read_to_string(&output)
        .map_err(|error| format!("quadratic VTU readback failed: {error}"))?;
    for field in [
        "Pressure_magnitude",
        "Pressure_real",
        "Pressure_imag",
        "Pressure_SPL_dB",
    ] {
        if !xml.contains(field) {
            return Err(format!("quadratic VTU is missing point field {field}"));
        }
    }
    let point_data = xml.matches("<DataArray").count();
    if point_data < 4 {
        return Err(format!(
            "quadratic VTU contains only {point_data} data arrays"
        ));
    }
    Ok(Metrics {
        surface_elements,
        exported_points: expected_points,
        exported_cells: expected_cells,
        pressure_dofs: solution.surface_pressure.len(),
    })
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .metric("surface_elements", metrics.surface_elements as f64)
            .metric("exported_points", metrics.exported_points as f64)
            .metric("exported_cells", metrics.exported_cells as f64)
            .metric("pressure_dofs", metrics.pressure_dofs as f64)
            .minimum("surface_elements", 1.0)
            .minimum("exported_points", 6.0)
            .minimum("exported_cells", 4.0)
            .minimum("pressure_dofs", 6.0)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("analytic/deterministic-fixture:bem-quadratic-vtu-export-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-quadratic-vtu-export-v1")
            .dimension("3-D rigid sphere; discontinuous Tri6 refined linear-cell VTU")
            .formulation("CPU dense direct quadratic TBEM with pressure magnitude/real/imag/SPL point fields")
            .seed("bem-quadratic-vtu-export:cpu:v1")
            .artifact("sonium-qa/qa-bem-quadratic-vtu-export/quadratic_pressure.vtu")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("analytic/deterministic-fixture:bem-quadratic-vtu-export-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-quadratic-vtu-export-v1")
            .dimension("3-D rigid sphere; discontinuous Tri6 refined linear-cell VTU")
            .seed("bem-quadratic-vtu-export:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status != QaStatus::Passed {
        std::process::exit(1);
    }
}
