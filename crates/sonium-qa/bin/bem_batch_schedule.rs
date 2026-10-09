//! Release-backed deterministic bounded-memory BEM batch planner contract.

use sonium_bem::core::{BemBatchWorkload, BemMemoryBudget, BemSchedulingError, plan_bem_batch};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.batch-schedule.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    task_count: usize,
    wave_count: usize,
    peak_reserved_ram_bytes: usize,
    covered_items: usize,
}

fn run() -> Result<Metrics, String> {
    let workload = BemBatchWorkload {
        candidate_count: 2,
        frequency_count: 3,
        rhs_count: 5,
        operator_bytes_per_point: 100,
        rhs_bytes_per_rhs: 10,
    };
    let budget = BemMemoryBudget {
        max_ram_bytes: 130,
        max_concurrent_tasks: 2,
    };
    let schedule = plan_bem_batch(workload, budget).map_err(|error| error.to_string())?;
    schedule
        .validate(workload, budget)
        .map_err(|error| error.to_string())?;
    let repeat = plan_bem_batch(workload, budget).map_err(|error| error.to_string())?;
    if schedule != repeat {
        return Err("bounded BEM schedule is not deterministic".to_string());
    }
    let mut malformed = schedule.clone();
    malformed.waves[0][0].rhs_count = 0;
    if !matches!(
        malformed.validate(workload, budget),
        Err(BemSchedulingError::InvalidSchedule { .. })
    ) {
        return Err("mutated BEM schedule was not rejected".to_string());
    }
    if !matches!(
        plan_bem_batch(
            workload,
            BemMemoryBudget {
                max_ram_bytes: 109,
                max_concurrent_tasks: 1,
            }
        ),
        Err(BemSchedulingError::TaskDoesNotFit { .. })
    ) {
        return Err("non-fitting BEM task was not rejected".to_string());
    }
    let task_count = schedule.waves.iter().map(Vec::len).sum();
    let covered_items = schedule
        .waves
        .iter()
        .flat_map(|wave| wave.iter())
        .map(|task| task.rhs_count)
        .sum::<usize>();
    Ok(Metrics {
        task_count,
        wave_count: schedule.waves.len(),
        peak_reserved_ram_bytes: schedule.peak_reserved_ram_bytes,
        covered_items,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("task_count", metrics.task_count as f64)
            .metric("wave_count", metrics.wave_count as f64)
            .metric(
                "peak_reserved_ram_bytes",
                metrics.peak_reserved_ram_bytes as f64,
            )
            .metric("covered_items", metrics.covered_items as f64)
            .minimum("covered_items", 30.0)
            .maximum("peak_reserved_ram_bytes", 130.0)
            .input_checksum("analytic/deterministic-fixture:bem-batch-schedule-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-batch-schedule-v1")
            .dimension("2 candidates x 3 frequencies x 5 RHS with bounded RAM and task waves")
            .seed("bem-batch-schedule:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(payload) => QaResult::failed(CASE_ID, format!("panic: {payload:?}"))
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA_RESULT serialization")
    );
    if result.status != QaStatus::Passed {
        std::process::exit(1);
    }
}
