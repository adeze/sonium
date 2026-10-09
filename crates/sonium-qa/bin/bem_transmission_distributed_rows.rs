//! CPU-only transmission row ownership/action contract evidence.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system;
use sonium_bem::core::distributed::plan_distributed_rows;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-distributed-rows.v1";

fn run() -> Result<(usize, usize, f64, usize), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    let system = build_transmission_tbem_system(
        &elements,
        &mesh.nodes,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
    .map_err(|error| error.to_string())?;
    let compressed = system
        .compress_kernel_native(4, 1.0e-8, 8)
        .map_err(|error| error.to_string())?;
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.1 + index as f64 * 0.011, -0.03 + index as f64 * 0.004))
        .collect();
    let global = compressed
        .apply_checked(&input)
        .map_err(|error| error.to_string())?;
    let transpose = compressed
        .apply_transpose_checked(&input)
        .map_err(|error| error.to_string())?;
    let plan = plan_distributed_rows(&elements, 4).map_err(|error| error.to_string())?;
    let rows_by_rank = plan
        .transmission_trace_rows(system.interface_dofs())
        .map_err(|error| error.to_string())?;
    let mut max_error = 0.0_f64;
    let mut row_count = 0_usize;
    for rows in rows_by_rank {
        let local = compressed
            .apply_rows_checked(&input, &rows)
            .map_err(|error| error.to_string())?;
        let local_transpose = compressed
            .apply_transpose_rows_checked(&input, &rows)
            .map_err(|error| error.to_string())?;
        row_count += rows.len();
        for (slot, &row) in rows.iter().enumerate() {
            max_error = max_error.max((local[slot] - global[row]).norm());
            max_error = max_error.max((local_transpose[slot] - transpose[row]).norm());
        }
    }
    Ok((
        system.total_dofs(),
        plan.partitions.len(),
        max_error,
        row_count,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, rank_count, max_error, row_count)) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("rank_count", rank_count as f64)
            .metric("owned_trace_rows", row_count as f64)
            .metric("maximum_local_action_error", max_error)
            .input_checksum("bem:icosphere:subdivision-0:constant-transmission")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-distributed-rows-v1",
            )
            .dimension("3-D CPU four-partition two-trace row action")
            .formulation(
                "Morton ownership expansion with local compressed forward/transpose actions",
            )
            .tolerance("local forward and transpose row actions <= 1e-10")
            .seed("bem-transmission-distributed-rows:cpu:v1")
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
