use std::ops::Range;

/// B-field sample evaluations per thread above which displacement-sweep
/// field integration parallelizes. Each sample is an FEM field
/// interpolation (microseconds), so the grain is large; short sweeps stay
/// serial.
pub(crate) const FIELD_EVALS_PER_THREAD: usize = 4096;

/// Elements per thread above which magnetostatic assembly parallelizes.
/// Each element runs quadrature tabulation plus a small dense integration.
pub(crate) const ASSEMBLY_ELEMENTS_PER_THREAD: usize = 256;

/// Machine-adaptive parallel gate for fine-grained loops (field sweeps,
/// element assembly): `work_items >= items_per_thread * threads`, where
/// `threads` is the active Rayon pool size. Single-threaded
/// processes always run serially. `SONIUM_SPEAKER_MODELING_PAR_THRESHOLD`
/// overrides with an absolute work-item count for tuning and forced-path
/// tests; `SONIUM_SERIAL=1` forces the serial path everywhere.
pub(crate) fn use_modeling_parallel(work_items: usize, items_per_thread: usize) -> bool {
    if serial_override() {
        return false;
    }
    if let Some(threshold) = env_threshold("SONIUM_SPEAKER_MODELING_PAR_THRESHOLD") {
        return work_items >= threshold;
    }
    let threads = rayon::current_num_threads();
    if threads < 2 {
        return false;
    }
    work_items >= items_per_thread.saturating_mul(threads).max(1)
}

/// Parallel gate for heavy-batch loops (displacement batches that rebuild
/// the mesh, per-current Le extractions, per-frequency eddy solves).
/// Each batch runs a full mesh build or FEM solve (milliseconds to
/// seconds), so any split across two or more threads wins; the `>= 2`
/// minimum is structural (parallelism needs two pieces of work), not a
/// machine threshold — the machine dependence is the thread count.
pub(crate) fn use_modeling_parallel_batch(n_batches: usize) -> bool {
    if serial_override() {
        return false;
    }
    if let Some(threshold) = env_threshold("SONIUM_SPEAKER_MODELING_PAR_THRESHOLD") {
        return n_batches >= threshold;
    }
    let threads = rayon::current_num_threads();
    n_batches >= 2 && threads >= 2
}

fn serial_override() -> bool {
    std::env::var("SONIUM_SERIAL")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn env_threshold(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

pub(crate) fn resolved_lane_count(item_count: usize, estimated_bytes_per_lane: usize) -> usize {
    if item_count == 0 {
        return 0;
    }
    if serial_override() {
        return 1;
    }
    let thread_limit = rayon::current_num_threads().min(item_count).max(1);
    let memory_limit = available_memory_bytes()
        .map(|bytes| bytes.saturating_mul(3) / 4 / estimated_bytes_per_lane.max(1))
        .unwrap_or(thread_limit)
        .max(1);
    thread_limit.min(memory_limit)
}

pub(crate) fn contiguous_lane_ranges(item_count: usize, lane_count: usize) -> Vec<Range<usize>> {
    if item_count == 0 {
        return Vec::new();
    }
    let lanes = lane_count.clamp(1, item_count);
    let base = item_count / lanes;
    let remainder = item_count % lanes;
    let mut start = 0;
    (0..lanes)
        .map(|lane| {
            let len = base + usize::from(lane < remainder);
            let range = start..start + len;
            start += len;
            range
        })
        .collect()
}

fn available_memory_bytes() -> Option<usize> {
    if let Ok(value) = std::env::var("SONIUM_SPEAKER_MEMORY_BUDGET_BYTES")
        && let Ok(bytes) = value.parse::<usize>()
        && bytes > 0
    {
        return Some(bytes);
    }
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = meminfo.lines().find_map(|line| {
        line.strip_prefix("MemAvailable:")?
            .split_whitespace()
            .next()?
            .parse::<usize>()
            .ok()
    })?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_ranges_cover_items_once_in_order() {
        assert_eq!(contiguous_lane_ranges(8, 3), vec![0..3, 3..6, 6..8]);
        assert_eq!(contiguous_lane_ranges(2, 8), vec![0..1, 1..2]);
        assert!(contiguous_lane_ranges(0, 4).is_empty());
    }

    #[test]
    fn gates_respect_serial_hatch_and_threshold() {
        // Hermetic single test: process env is shared by all test threads.
        // SAFETY: only path selection reads these keys, and both paths run
        // the same evaluations in the same order, so a concurrent test can
        // at most take the other (correct) path.
        unsafe {
            let prev_threshold = std::env::var_os("SONIUM_SPEAKER_MODELING_PAR_THRESHOLD");
            let prev_serial = std::env::var_os("SONIUM_SERIAL");
            std::env::remove_var("SONIUM_SPEAKER_MODELING_PAR_THRESHOLD");
            std::env::remove_var("SONIUM_SERIAL");

            let threads = rayon::current_num_threads();
            if threads >= 2 {
                assert!(!use_modeling_parallel(1, FIELD_EVALS_PER_THREAD));
                assert!(use_modeling_parallel(
                    threads * FIELD_EVALS_PER_THREAD,
                    FIELD_EVALS_PER_THREAD
                ));
                assert!(!use_modeling_parallel_batch(1));
                assert!(use_modeling_parallel_batch(2));
                assert!(resolved_lane_count(4, 1) >= 1);
            }
            std::env::set_var("SONIUM_SERIAL", "1");
            assert!(!use_modeling_parallel(usize::MAX, 1));
            assert!(!use_modeling_parallel_batch(usize::MAX));
            assert_eq!(resolved_lane_count(4, 1), 1);
            std::env::remove_var("SONIUM_SERIAL");
            std::env::set_var("SONIUM_SPEAKER_MODELING_PAR_THRESHOLD", "1");
            assert!(use_modeling_parallel(1, usize::MAX));
            assert!(use_modeling_parallel_batch(1));

            match prev_threshold {
                Some(value) => std::env::set_var("SONIUM_SPEAKER_MODELING_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_SPEAKER_MODELING_PAR_THRESHOLD"),
            }
            match prev_serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    }
}
