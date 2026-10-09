//! CPU-only MeshLang-to-FEM order-2 H(curl) transfer reference.
//!
//! This fixture exercises the affine-triangle Nédélec order-2 numerical
//! consumer without starting a solver, mesher, Gmsh process, or simulator.

use sonium_fem::mesh::{transfer_hcurl_order2_triangle_state_checked, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-hcurl-order2-transfer-reference.v1";

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let source_mesh = unit_square_triangles(1);
    let target_mesh = source_mesh.clone();
    let source_space = HcurlSpace::order2();
    let target_space = HcurlSpace::order2();
    let source_dofs = source_space.num_global_dofs(&source_mesh);
    let source_values: Vec<f64> = (0..source_dofs)
        .map(|index| 0.25 + 0.17 * index as f64 - 0.03 * (index * index) as f64)
        .collect();
    let transferred = transfer_hcurl_order2_triangle_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .map_err(|error| error.to_string())?;
    let max_error = transferred
        .iter()
        .zip(source_values.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    if max_error > 1.0e-11 {
        return Err(format!(
            "order-2 H(curl) transfer error {max_error:e} exceeds tolerance"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("max_hcurl_order2_error", max_error)
        .metric("source_dofs", source_dofs as f64)
        .metric("target_dofs", transferred.len() as f64)
        .dimension("2-D affine triangular H(curl) order-2")
        .input_checksum("unit-square-triangles:meshlang-hcurl-order2-v1")
        .reference_checksum("same-mesh-order2-state:meshlang-hcurl-order2-v1")
        .seed("meshlang-fem-hcurl-order2-transfer:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
}
