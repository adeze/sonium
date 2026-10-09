//! Deterministic shared-DOF distributed BEM ownership contract.

use sonium_bem::core::distributed::plan_distributed_rows;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.distributed-ownership.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    active_elements: usize,
    unique_dofs: usize,
    ranks: usize,
    duplicate_rows: usize,
    scatter_gather_round_trip: usize,
    owner_queries: usize,
    transmission_owner_queries: usize,
}

fn run() -> Result<Metrics, String> {
    let mut mesh = generate_icosphere_mesh(1.0, 1);
    for (index, element) in mesh.elements.iter_mut().enumerate() {
        // Deliberately share the edge-row DOF with the next panel to model a
        // continuous/shared-node trace, while keeping the fixture tiny.
        element.dof_addresses = vec![index, index + 1];
    }
    let plan = plan_distributed_rows(&mesh.elements, 3).map_err(|error| error.to_string())?;
    let second = plan_distributed_rows(&mesh.elements, 3).map_err(|error| error.to_string())?;
    if plan != second {
        return Err("distributed ownership plan is not deterministic".to_string());
    }
    let mut owned = plan
        .partitions
        .iter()
        .flat_map(|partition| partition.dof_indices.iter().copied())
        .collect::<Vec<_>>();
    let total_rows = owned.len();
    owned.sort_unstable();
    let duplicate_rows =
        total_rows.saturating_sub(owned.windows(2).filter(|pair| pair[0] != pair[1]).count() + 1);
    let expected = (0..mesh.elements.len() + 1).collect::<Vec<_>>();
    if owned != expected || duplicate_rows != 0 || plan.num_dofs != expected.len() {
        return Err(format!(
            "invalid shared-DOF ownership: owned={owned:?}, expected={expected:?}, duplicate_rows={duplicate_rows}, num_dofs={}",
            plan.num_dofs
        ));
    }
    let global_values = (0..plan.num_dofs).collect::<Vec<_>>();
    let local_values = plan
        .scatter_dof_values(&global_values)
        .map_err(|error| error.to_string())?;
    let gathered = plan
        .gather_dof_values(&local_values)
        .map_err(|error| error.to_string())?;
    if gathered != global_values {
        return Err("distributed scatter/gather changed global vector order".to_string());
    }
    let owner_queries = (0..plan.num_dofs)
        .map(|dof| plan.owner_of_dof(dof).map(|_| ()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?
        .len();
    let transmission_owner_queries = (0..plan.num_dofs * 2)
        .map(|row| {
            plan.owner_of_transmission_row(row, plan.num_dofs)
                .map(|_| ())
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?
        .len();
    if plan
        .scatter_dof_values(&global_values[..plan.num_dofs - 1])
        .is_ok()
        || plan.gather_dof_values::<usize>(&[]).is_ok()
    {
        return Err("distributed vector shape admission accepted malformed buffers".to_string());
    }

    if plan_distributed_rows(&mesh.elements, 0).is_ok() {
        return Err("zero-rank distributed plan was accepted".to_string());
    }
    Ok(Metrics {
        active_elements: plan.num_active_elements,
        unique_dofs: plan.num_dofs,
        ranks: plan.partitions.len(),
        duplicate_rows,
        scatter_gather_round_trip: usize::from(gathered == global_values),
        owner_queries,
        transmission_owner_queries,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("active_elements", metrics.active_elements as f64)
            .metric("unique_dofs", metrics.unique_dofs as f64)
            .metric("ranks", metrics.ranks as f64)
            .metric("duplicate_rows", metrics.duplicate_rows as f64)
            .metric(
                "scatter_gather_round_trip",
                metrics.scatter_gather_round_trip as f64,
            )
            .metric("owner_queries", metrics.owner_queries as f64)
            .metric(
                "transmission_owner_queries",
                metrics.transmission_owner_queries as f64,
            )
            .maximum("duplicate_rows", 0.0)
            .minimum("scatter_gather_round_trip", 1.0)
            .minimum("owner_queries", 1.0)
            .minimum("transmission_owner_queries", 2.0)
            .minimum("active_elements", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-distributed-ownership-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-distributed-ownership-v1")
            .dimension("3-rank Morton plan; shared continuous-style DOFs")
            .reference("independent sorted global-row coverage and deterministic replay")
            .seed("bem-distributed-ownership:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic during distributed ownership validation")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    let passed = matches!(&result.status, QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !passed {
        std::process::exit(1);
    }
}
