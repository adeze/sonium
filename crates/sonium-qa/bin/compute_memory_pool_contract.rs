//! CPU-only contract for memory-pool quota admission and refund semantics.
//!
//! This exercises allocation accounting without constructing or running a
//! simulator workload.

use sonium_compute::{CpuBackend, MemoryPool, MemoryPoolConfig, OomError};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.memory-pool-contract.v1";

struct Metrics {
    quota_rejected: bool,
    drop_refunded: bool,
    reusable: bool,
    zero_allocation: bool,
    high_water_preserved: bool,
}

fn run() -> Result<Metrics, String> {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(16))
        .map_err(|error| format!("pool construction failed: {error}"))?;

    let buffer = pool
        .try_alloc::<u32>(4)
        .map_err(|error| format!("within-cap allocation failed: {error}"))?;
    if pool.bytes_in_use() != 16 || pool.high_water_bytes() != 16 {
        return Err(format!(
            "initial accounting mismatch: bytes={}, high_water={}",
            pool.bytes_in_use(),
            pool.high_water_bytes()
        ));
    }

    let quota_rejected = matches!(
        pool.try_alloc::<u32>(1),
        Err(OomError::QuotaExceeded { .. })
    ) && pool.bytes_in_use() == 16;

    drop(buffer);
    let drop_refunded = pool.bytes_in_use() == 0 && pool.high_water_bytes() == 16;

    let reusable = pool
        .try_alloc::<u32>(4)
        .map(|retry| {
            let valid = pool.bytes_in_use() == 16;
            drop(retry);
            valid && pool.bytes_in_use() == 0
        })
        .unwrap_or(false);

    let zero = pool
        .try_alloc::<u32>(0)
        .map_err(|error| format!("zero-length allocation failed: {error}"))?;
    let zero_allocation = zero.is_empty() && pool.bytes_in_use() == 0;
    drop(zero);

    Ok(Metrics {
        quota_rejected,
        drop_refunded,
        reusable,
        zero_allocation,
        high_water_preserved: pool.high_water_bytes() == 16,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("quota_rejected", metrics.quota_rejected as u8 as f64)
                .metric("drop_refunded", metrics.drop_refunded as u8 as f64)
                .metric("reusable", metrics.reusable as u8 as f64)
                .metric("zero_allocation", metrics.zero_allocation as u8 as f64)
                .metric(
                    "high_water_preserved",
                    metrics.high_water_preserved as u8 as f64,
                )
                .minimum("quota_rejected", 1.0)
                .minimum("drop_refunded", 1.0)
                .minimum("reusable", 1.0)
                .minimum("zero_allocation", 1.0)
                .minimum("high_water_preserved", 1.0)
                .reference("independent memory-pool quota and RAII refund invariants")
                .reference_checksum(
                    "analytic/deterministic-fixture:compute-memory-pool-contract-v1",
                )
                .dimension("CPU MemoryPool; 16-byte cap; u32 allocations")
                .seed("compute-memory-pool-contract:quota-refund:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("memory-pool QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("memory-pool QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
