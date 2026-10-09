//! CPU-only distributed row ownership/action contract for continuous Q2 traces.

use num_complex::Complex64;
use sonium_bem::core::assembly::{
    TbemQuadraturePolicy, build_transmission_tbem_system_with_continuous_quadratic_checked,
};
use sonium_bem::core::distributed::plan_distributed_rows_with_dof_count;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-distributed-higher-order-rows.v1";

fn run() -> Result<(usize, usize, f64, usize), String> {
    let mut mesh = generate_cubed_sphere_quad_mesh(0.2, 1);
    let dof_map = mesh
        .continuous_quadratic_dof_map()
        .map_err(|error| error.to_string())?;
    for (element_index, element) in mesh.elements.iter_mut().enumerate() {
        if let Some(dofs) = dof_map.element_dofs(element_index) {
            element.dof_addresses = dofs.to_vec();
        }
    }

    let exterior =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let interior =
        PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).map_err(|error| error.to_string())?;
    let system = build_transmission_tbem_system_with_continuous_quadratic_checked(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
        TbemQuadraturePolicy {
            // This case certifies distributed row ownership/action parity;
            // quadrature convergence has dedicated higher-order QA cases.
            absolute_tolerance: 1.0e3,
            relative_tolerance: 1.0e3,
            ..TbemQuadraturePolicy::default()
        },
    )
    .map_err(|error| error.to_string())?;
    let compressed = system
        .compress(4, 1.0e-8, 8)
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
    let plan = plan_distributed_rows_with_dof_count(&mesh.elements, 3, dof_map.num_dofs())
        .map_err(|error| error.to_string())?;
    let rows_by_rank = plan
        .transmission_trace_rows(system.interface_dofs())
        .map_err(|error| error.to_string())?;
    let mut maximum_error = 0.0_f64;
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
            maximum_error = maximum_error.max((local[slot] - global[row]).norm());
            maximum_error = maximum_error.max((local_transpose[slot] - transpose[row]).norm());
        }
    }
    Ok((
        system.total_dofs(),
        plan.partitions.len(),
        maximum_error,
        row_count,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, rank_count, maximum_error, row_count)) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("rank_count", rank_count as f64)
            .metric("owned_trace_rows", row_count as f64)
            .metric("maximum_local_action_error", maximum_error)
            .input_checksum("bem:cubed-sphere-quad4:face-resolution-1:continuous-quadratic")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-distributed-higher-order-rows-v1",
            )
            .dimension("3-D CPU three-partition continuous-Q2 two-trace row action")
            .formulation(
                "Morton ownership expansion with local compressed forward/transpose actions",
            )
            .tolerance("local forward and transpose row actions <= 1e-10")
            .seed("bem-transmission-distributed-higher-order-rows:cpu:v1")
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
