//! Machine-adaptive rayon gates for mesh extraction.
//!
//! A fixed work-item threshold cannot be right on every machine: the point
//! where pool dispatch pays off moves with thread count, clock speed, and
//! memory bandwidth. Every parallel gate in this crate therefore resolves
//! through [`ParallelPolicy`]:
//!
//! - default: `work_items >= items_per_thread * threads`, where `threads`
//!   is [`std::thread::available_parallelism`]. Single-threaded processes
//!   always run serially;
//! - `SONIUM_MESHER_PAR_THRESHOLD=<n>`: absolute work-item override for
//!   tuning, benches, and forced-path tests (units are the call site's work
//!   items — elements for topology/boundaries, points or faces for
//!   quickhull);
//! - `SONIUM_SERIAL=1`: process-wide serial escape hatch for deterministic
//!   QA runs.
//!
//! The per-thread grains below are starting points, not calibrated values:
//! tune them with `benches/tetrahedralize_bench.rs` on the target machine.

/// Elements per thread above which topology/boundary extraction parallelizes.
pub const TOPOLOGY_ITEMS_PER_THREAD: usize = 1024;

/// Points (assignment) or faces (visibility) per thread above which the
/// quickhull parallel passes run.
pub const QUICKHULL_ITEMS_PER_THREAD: usize = 64;

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
            threshold_override: env_threshold("SONIUM_MESHER_PAR_THRESHOLD"),
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

    /// Number of worker shards for a parallel merge: at most one per thread
    /// and at most one per `items_per_thread` of work, so shards stay
    /// load-balanced without degenerating into per-element overhead.
    pub fn shard_count(&self, work_items: usize, items_per_thread: usize) -> usize {
        if !self.decide(work_items, items_per_thread) {
            return 1;
        }
        let per_shard = items_per_thread.max(1);
        let by_work = work_items.div_ceil(per_shard).max(1);
        by_work.min(self.threads.max(1))
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
        let policy = ParallelPolicy::serial();
        assert!(!policy.decide(usize::MAX, 1));
        assert_eq!(policy.shard_count(usize::MAX, 1), 1);
    }

    #[test]
    fn single_thread_never_parallelizes() {
        let policy = ParallelPolicy::with_threads(1);
        assert!(!policy.decide(usize::MAX, 1));
    }

    #[test]
    fn decision_scales_with_threads() {
        let four = ParallelPolicy::with_threads(4);
        assert!(four.decide(4096, 1024));
        assert!(!four.decide(4095, 1024));
        let sixteen = ParallelPolicy::with_threads(16);
        assert!(!sixteen.decide(4096, 1024));
        assert!(sixteen.decide(16_384, 1024));
    }

    #[test]
    fn shard_count_balances_work_and_threads() {
        let policy = ParallelPolicy::with_threads(8);
        // Below the gate there is a single shard (serial path).
        assert_eq!(policy.shard_count(100, 1024), 1);
        assert_eq!(policy.shard_count(3000, 1024), 1);
        // At/above the gate: one shard per grain of work, capped by threads.
        assert_eq!(policy.shard_count(8192, 1024), 8);
        assert_eq!(policy.shard_count(1_000_000, 1024), 8);
        assert_eq!(policy.shard_count(8192 + 1024, 1024), 8);
        let two = ParallelPolicy::with_threads(2);
        assert_eq!(two.shard_count(100_000, 1024), 2);
    }
}
