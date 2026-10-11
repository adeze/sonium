//! Bounded CPU evidence for genuine constant-triangle Galerkin assembly.

use num_complex::Complex64;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{PhysicsParams, assemble_galerkin_single_layer};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-single-layer.v1";
/// Manifest acceptance bound for `bem.galerkin-single-layer.v1`.
///
/// Both the order-5 matrix and the independent order-7 reference must satisfy
/// this symmetry bound; loosening it weakens the physics-gate claim.
const ASYMMETRY_TOLERANCE: f64 = 2.0e-8;

fn matrix_asymmetry(matrix: &DenseMatrix<Complex64>) -> f64 {
    (0..matrix.nrows())
        .flat_map(|row| (0..matrix.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| (matrix[[row, column]] - matrix[[column, row]]).norm())
        .fold(0.0_f64, f64::max)
}

fn maximum_entry_delta(first: &DenseMatrix<Complex64>, second: &DenseMatrix<Complex64>) -> f64 {
    first
        .as_slice()
        .iter()
        .zip(second.as_slice())
        .map(|(coarse, fine)| (*coarse - *fine).norm())
        .fold(0.0_f64, f64::max)
}

fn reject_nonfinite(matrix: &DenseMatrix<Complex64>, order: usize) -> Result<(), String> {
    if matrix
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(format!(
            "Galerkin single-layer order-{order} matrix has non-finite entries"
        ));
    }
    Ok(())
}

fn run() -> Result<(usize, f64, f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let coarse = assemble_galerkin_single_layer(&elements, &mesh.nodes, &physics, 3)
        .map_err(|error| error.to_string())?;
    let fine = assemble_galerkin_single_layer(&elements, &mesh.nodes, &physics, 5)
        .map_err(|error| error.to_string())?;
    let reference = assemble_galerkin_single_layer(&elements, &mesh.nodes, &physics, 7)
        .map_err(|error| error.to_string())?;
    reject_nonfinite(&coarse, 3)?;
    reject_nonfinite(&fine, 5)?;
    reject_nonfinite(&reference, 7)?;
    let maximum_asymmetry = matrix_asymmetry(&fine);
    if !maximum_asymmetry.is_finite() || maximum_asymmetry > ASYMMETRY_TOLERANCE {
        return Err(format!(
            "Galerkin single-layer asymmetry {maximum_asymmetry:.3e} exceeds {ASYMMETRY_TOLERANCE:.1e}"
        ));
    }
    let reference_asymmetry = matrix_asymmetry(&reference);
    if !reference_asymmetry.is_finite() || reference_asymmetry > ASYMMETRY_TOLERANCE {
        return Err(format!(
            "Galerkin single-layer order-7 reference asymmetry {reference_asymmetry:.3e} exceeds {ASYMMETRY_TOLERANCE:.1e}"
        ));
    }
    let maximum_order_delta = maximum_entry_delta(&coarse, &fine);
    if !maximum_order_delta.is_finite() {
        return Err("Galerkin single-layer order delta is non-finite".to_string());
    }
    let reference_order_delta = maximum_entry_delta(&fine, &reference);
    if !reference_order_delta.is_finite() {
        return Err("Galerkin single-layer reference delta is non-finite".to_string());
    }
    Ok((
        fine.nrows(),
        maximum_asymmetry,
        maximum_order_delta,
        reference_asymmetry,
        reference_order_delta,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((
            dimension,
            maximum_asymmetry,
            maximum_order_delta,
            reference_asymmetry,
            reference_order_delta,
        )) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("maximum_matrix_asymmetry", maximum_asymmetry)
            .metric("order_3_to_5_maximum_delta", maximum_order_delta)
            .metric("reference_matrix_asymmetry", reference_asymmetry)
            .metric("order_5_to_7_maximum_delta", reference_order_delta)
            .maximum("maximum_matrix_asymmetry", ASYMMETRY_TOLERANCE)
            .maximum("reference_matrix_asymmetry", ASYMMETRY_TOLERANCE)
            .input_checksum("bem:icosphere:subdivision-0:constant-tri3")
            .reference_checksum("analytic/deterministic-fixture:bem-galerkin-single-layer-v1")
            .dimension("3-D CPU 20-panel constant-triangle Galerkin single layer")
            .formulation("test and trial surface integration with Duffy/Sauter-Schwab pair routes")
            .tolerance("finite entries; matrix asymmetry <= 2e-8 at orders 5 and 7")
            .seed("bem-galerkin-single-layer:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
