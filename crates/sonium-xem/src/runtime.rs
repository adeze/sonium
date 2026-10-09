//! Shared execution-contract reporting for user-facing simulation binaries.

use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Machine-readable evidence that a binary resolved one bounded execution budget.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ExecutionContract {
    pub schema: &'static str,
    pub binary: String,
    pub requested_backend: String,
    pub resolved_backend: String,
    pub physical_cpu_count: usize,
    pub logical_cpu_count: usize,
    pub rayon_pool_width: usize,
    pub parallel_splits: BTreeMap<String, usize>,
    pub maximum_observed_concurrent_tasks: usize,
    pub fallback_used: bool,
    pub device_identity: String,
}

/// Install the process-wide Rayon pool with one shared automatic-thread policy.
///
/// `threads == 0` preserves Rayon's native `RAYON_NUM_THREADS` and available-
/// parallelism resolution. A positive value is an explicit user override.
/// Fails when the global pool is already installed; use
/// [`ensure_global_rayon_pool`] for idempotent startup paths.
pub fn install_global_rayon_pool(
    threads: usize,
    worker_stack_size: Option<usize>,
) -> Result<usize, rayon::ThreadPoolBuildError> {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if threads > 0 {
        builder = builder.num_threads(threads);
    }
    if let Some(stack_size) = worker_stack_size {
        builder = builder.stack_size(stack_size);
    }
    builder.build_global()?;
    Ok(rayon::current_num_threads())
}

/// Idempotent [`install_global_rayon_pool`]: a second initialization (e.g.
/// two binaries sharing a test process, or a retry after fork) reports the
/// already-installed pool width instead of failing.
pub fn ensure_global_rayon_pool(
    threads: usize,
    worker_stack_size: Option<usize>,
) -> Result<usize, rayon::ThreadPoolBuildError> {
    match install_global_rayon_pool(threads, worker_stack_size) {
        Ok(width) => Ok(width),
        Err(_) => Ok(rayon::current_num_threads().max(1)),
    }
}

/// Observe a short, bounded workload in the ambient Rayon pool and record its budget.
///
/// Accelerator callers must pass identity obtained from real device work; this helper
/// does not probe or relabel a backend.
pub fn observe_execution_contract(
    binary: impl Into<String>,
    requested_backend: impl Into<String>,
    resolved_backend: impl Into<String>,
    parallel_splits: BTreeMap<String, usize>,
    fallback_used: bool,
    device_identity: impl Into<String>,
) -> ExecutionContract {
    let pool_width = rayon::current_num_threads().max(1);
    let active = AtomicUsize::new(0);
    let maximum = AtomicUsize::new(0);
    // The overlap window parks (rather than busy-spins) its slice so the
    // probe records scheduler-observed concurrency without burning a full
    // timeslice per thread. Counter ordering is Acquire/Release: only the
    // fetch-add/max/sub sequence needs to be atomic, not globally ordered.
    let probe_window = Duration::from_millis(10);
    let probe_start = Instant::now();
    (0..pool_width).into_par_iter().for_each(|_| {
        let concurrent = active.fetch_add(1, Ordering::AcqRel) + 1;
        maximum.fetch_max(concurrent, Ordering::AcqRel);
        let elapsed = probe_start.elapsed();
        if elapsed < probe_window {
            std::thread::sleep(probe_window - elapsed);
        }
        active.fetch_sub(1, Ordering::AcqRel);
    });
    let (physical_cpu_count, logical_cpu_count) = cpu_counts();
    ExecutionContract {
        schema: "sonium.execution-contract.v1",
        binary: binary.into(),
        requested_backend: requested_backend.into(),
        resolved_backend: resolved_backend.into(),
        physical_cpu_count,
        logical_cpu_count,
        rayon_pool_width: pool_width,
        parallel_splits,
        maximum_observed_concurrent_tasks: maximum.load(Ordering::Acquire),
        fallback_used,
        device_identity: device_identity.into(),
    }
}

fn cpu_counts() -> (usize, usize) {
    let logical = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let topology_root = std::path::Path::new("/sys/devices/system/cpu");
    let mut physical = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(topology_root) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.strip_prefix("cpu").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
            }) {
                continue;
            }
            let topology = entry.path().join("topology");
            let package = std::fs::read_to_string(topology.join("physical_package_id"));
            let core = std::fs::read_to_string(topology.join("core_id"));
            if let (Ok(package), Ok(core)) = (package, core) {
                physical.insert((package.trim().to_owned(), core.trim().to_owned()));
            }
        }
    }
    (
        if physical.is_empty() {
            logical
        } else {
            physical.len()
        },
        logical,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_records_machine_pool_and_observed_work() {
        let contract = observe_execution_contract(
            "fixture",
            "cpu",
            "cpu",
            BTreeMap::from([("frequency_lanes".to_string(), 1)]),
            false,
            "host-cpu",
        );
        assert!(contract.physical_cpu_count > 0);
        assert!(contract.logical_cpu_count > 0);
        assert!(contract.rayon_pool_width > 0);
        assert!(contract.maximum_observed_concurrent_tasks > 0);
        assert!(contract.maximum_observed_concurrent_tasks <= contract.rayon_pool_width);
        assert!(!contract.fallback_used);
    }
}
