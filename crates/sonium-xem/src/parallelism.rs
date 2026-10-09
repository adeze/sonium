//! Machine-adaptive rayon gates for field-slice extraction.
//!
//! A fixed work-item threshold cannot be right on every machine: the point
//! where pool dispatch pays off moves with thread count, clock speed, and
//! memory bandwidth. Every parallel gate in this crate therefore resolves
//! through [`ParallelPolicy`]:
//!
//! - default: `work_items >= items_per_thread * threads`, where `threads`
//!   is [`std::thread::available_parallelism`]. Single-threaded processes
//!   always run serially;
//! - `SONIUM_XEM_PAR_THRESHOLD=<n>`: absolute work-item override for tuning,
//!   benches, and forced-path tests (units are the call site's work items —
//!   slice cells for [`SLICE_CELLS_PER_THREAD`]);
//! - `SONIUM_SERIAL=1`: process-wide serial escape hatch for deterministic
//!   QA runs.
//!
//! The per-thread grain below is a starting point, not a calibrated value:
//! tune it with benches on the target machine.

/// Slice cells per thread above which
/// [`try_generate_spatial_slices_par`](crate::output::try_generate_spatial_slices_par)
/// parallelizes over solutions. Per-cell pressure evaluation dominates, so
/// the grain is modest.
pub const SLICE_CELLS_PER_THREAD: usize = 65_536;

/// Machine-adaptive parallelization policy.
#[derive(Debug, Clone, Copy)]
pub struct ParallelPolicy {
    threads: usize,
    force_serial: bool,
    threshold_override: Option<usize>,
}

impl ParallelPolicy {
    /// Resolve from the process environment (see module docs).
    pub fn from_env() -> Self {
        Self {
            threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            force_serial: serial_override(),
            threshold_override: env_threshold("SONIUM_XEM_PAR_THRESHOLD"),
        }
    }

    /// Explicit policy without environment reads (tests, benches, callers
    /// that already resolved the machine topology).
    pub fn with_threads(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
            force_serial: false,
            threshold_override: None,
        }
    }

    /// Deterministic serial policy for QA runs.
    pub fn serial() -> Self {
        Self {
            threads: 1,
            force_serial: true,
            threshold_override: None,
        }
    }

    /// True when `work_items` of `items_per_thread`-grained work merit rayon.
    pub fn decide(&self, work_items: usize, items_per_thread: usize) -> bool {
        if self.force_serial || self.threads < 2 {
            return false;
        }
        if let Some(threshold) = self.threshold_override {
            return work_items >= threshold;
        }
        work_items >= items_per_thread.saturating_mul(self.threads).max(1)
    }

    /// Worker thread count this policy assumes.
    pub fn threads(&self) -> usize {
        self.threads
    }
}

/// Convenience gate: [`ParallelPolicy::from_env`] + `decide`.
pub fn use_parallel(work_items: usize, items_per_thread: usize) -> bool {
    ParallelPolicy::from_env().decide(work_items, items_per_thread)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_policy_never_parallelizes() {
        assert!(!ParallelPolicy::serial().decide(usize::MAX, 1));
    }

    #[test]
    fn single_thread_never_parallelizes() {
        assert!(!ParallelPolicy::with_threads(1).decide(usize::MAX, 1));
    }

    #[test]
    fn decision_scales_with_threads() {
        let eight = ParallelPolicy::with_threads(8);
        assert!(eight.decide(8 * 65_536, SLICE_CELLS_PER_THREAD));
        assert!(!eight.decide(8 * 65_536 - 1, SLICE_CELLS_PER_THREAD));
    }
}
