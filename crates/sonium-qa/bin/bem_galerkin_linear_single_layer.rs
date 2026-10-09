//! Bounded CPU evidence for discontinuous linear-Tri3 Galerkin coupling.

use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{BasisOrder, PhysicsParams, assemble_galerkin_single_layer_with_basis};
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-linear-single-layer.v1";

fn run() -> Result<(usize, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements.into_iter().take(2).collect::<Vec<_>>();
    for (panel, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * panel, 3 * panel + 1, 3 * panel + 2];
    }
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let matrix = assemble_galerkin_single_layer_with_basis(
        &elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Linear,
        4,
    )
    .map_err(|error| error.to_string())?;
    if matrix
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("linear Galerkin matrix contains a non-finite entry".to_string());
    }
    let mut asymmetry = 0.0_f64;
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            asymmetry = asymmetry.max((matrix[[row, column]] - matrix[[column, row]]).norm());
        }
    }
    Ok((matrix.nrows(), asymmetry))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, asymmetry)) => QaResult::passed(CASE_ID)
            .input_checksum("bem:icosphere:subdivision-0:linear-tri3:two-panels")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-linear-single-layer-v1",
            )
            .dimension(format!(
                "3-D CPU {dimension}-DOF discontinuous linear Tri3; asymmetry={asymmetry:.3e}"
            ))
            .seed("bem-galerkin-linear-single-layer:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
