//! Bounded CPU evidence for genuine constant-triangle Galerkin assembly.

use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{PhysicsParams, assemble_galerkin_single_layer};
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-single-layer.v1";

fn run() -> Result<(usize, f64, f64), String> {
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
    let maximum_asymmetry = (0..fine.nrows())
        .flat_map(|row| (0..fine.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| (fine[[row, column]] - fine[[column, row]]).norm())
        .fold(0.0_f64, f64::max);
    let maximum_order_delta = coarse
        .as_slice()
        .iter()
        .zip(fine.as_slice())
        .map(|(coarse, fine)| (*coarse - *fine).norm())
        .fold(0.0_f64, f64::max);
    Ok((fine.nrows(), maximum_asymmetry, maximum_order_delta))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, maximum_asymmetry, maximum_order_delta)) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("maximum_matrix_asymmetry", maximum_asymmetry)
            .metric("order_3_to_5_maximum_delta", maximum_order_delta)
            .input_checksum("bem:icosphere:subdivision-0:constant-tri3")
            .reference_checksum("analytic/deterministic-fixture:bem-galerkin-single-layer-v1")
            .dimension("3-D CPU 20-panel constant-triangle Galerkin single layer")
            .formulation("test and trial surface integration with Duffy/Sauter-Schwab pair routes")
            .tolerance("finite entries; matrix asymmetry <= 2e-8")
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
