//! Portable parallel iteration abstractions
//!
//! This module provides parallel iteration using rayon.
//!
//! ## Usage
//!
//! ```ignore
//! use crate::core::parallel::*;
//!
//! // Parallel map over a slice
//! let results: Vec<i32> = parallel_map(&data, |item| item * 2);
//!
//! // Parallel map over indices
//! let results: Vec<i32> = parallel_map_indexed(100, |i| i * 2);
//! ```

/// Check if parallel processing is available
#[inline]
pub fn is_parallel_available() -> bool {
    true
}

/// Kernel evaluations (BEM pair interactions) per thread above which
/// assembly and matvec row loops parallelize. One pair evaluation is tens
/// to hundreds of nanoseconds (singular quadrature more), so the grain
/// keeps pool dispatch amortized; small systems stay serial.
pub const BEM_ENTRIES_PER_THREAD: usize = 65_536;

/// Coarse blocks (clusters, H-matrix partitions, FMM interaction blocks)
/// per thread above which block loops parallelize. Each block is itself a
/// large kernel batch, so the grain is small.
pub const BEM_BLOCKS_PER_THREAD: usize = 4;

/// Machine-adaptive parallelization policy.
///
/// A fixed work-item threshold cannot be right on every machine: the point
/// where pool dispatch pays off moves with thread count, clock speed, and
/// memory bandwidth. Every parallel gate in this crate therefore resolves
/// through this policy:
///
/// - default: `work_items >= items_per_thread * threads`, where `threads`
///   is [`std::thread::available_parallelism`]. Single-threaded processes
///   always run serially;
/// - `SONIUM_BEM_PAR_THRESHOLD=<n>`: absolute work-item override for tuning,
///   benches, and forced-path tests (units are the call site's work items);
/// - `SONIUM_SERIAL=1`: process-wide serial escape hatch for deterministic
///   QA runs.
///
/// The per-thread grains are starting points, not calibrated values: tune
/// them with benches on the target machine.
#[derive(Debug, Clone, Copy)]
pub struct ParallelPolicy {
    threads: usize,
    force_serial: bool,
    threshold_override: Option<usize>,
}

impl ParallelPolicy {
    /// Resolve from the process environment (see type docs).
    pub fn from_env() -> Self {
        Self {
            threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            force_serial: serial_override(),
            threshold_override: env_threshold("SONIUM_BEM_PAR_THRESHOLD"),
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

/// Map over indices with serial auto-select.
///
/// Collects `f(i)` for `i in 0..count` in order, using rayon when the
/// policy selects parallel for `work_items` of `items_per_thread` grain.
/// Both paths visit indices in order, so results are identical.
pub fn gated_map_indexed<U, F>(
    count: usize,
    work_items: usize,
    items_per_thread: usize,
    f: F,
) -> Vec<U>
where
    U: Send,
    F: Fn(usize) -> U + Sync + Send,
{
    use rayon::prelude::*;
    if use_parallel(work_items, items_per_thread) {
        (0..count).into_par_iter().map(f).collect()
    } else {
        (0..count).map(f).collect()
    }
}

/// Map over a slice with serial auto-select. Work is measured by the
/// caller-supplied `work_items` (e.g. rows × columns for assembly loops);
/// both paths preserve order.
pub fn gated_map<T, U, F>(data: &[T], work_items: usize, items_per_thread: usize, f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    use rayon::prelude::*;
    if use_parallel(work_items, items_per_thread) {
        data.par_iter().map(f).collect()
    } else {
        data.iter().map(f).collect()
    }
}

/// Enumerate + map over a slice with serial auto-select. Both paths visit
/// indices in order, so results are identical.
pub fn gated_enumerate_map<T, U, F>(
    data: &[T],
    work_items: usize,
    items_per_thread: usize,
    f: F,
) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> U + Sync + Send,
{
    use rayon::prelude::*;
    if use_parallel(work_items, items_per_thread) {
        data.par_iter().enumerate().map(|(i, x)| f(i, x)).collect()
    } else {
        data.iter().enumerate().map(|(i, x)| f(i, x)).collect()
    }
}

/// Enumerate + filter_map over a slice with serial auto-select. Both paths
/// preserve order.
pub fn gated_enumerate_filter_map<T, U, F>(
    data: &[T],
    work_items: usize,
    items_per_thread: usize,
    f: F,
) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> Option<U> + Sync + Send,
{
    use rayon::prelude::*;
    if use_parallel(work_items, items_per_thread) {
        data.par_iter()
            .enumerate()
            .filter_map(|(i, x)| f(i, x))
            .collect()
    } else {
        data.iter()
            .enumerate()
            .filter_map(|(i, x)| f(i, x))
            .collect()
    }
}

/// Flat-map over a slice with serial auto-select. The per-item function
/// returns a `Vec` (rather than a lazy iterator) so both the rayon and the
/// serial path share one closure shape; inner loops are tiny (local DOFs),
/// so the small per-item allocation matches existing behavior.
pub fn gated_flat_map<T, U, F>(
    data: &[T],
    work_items: usize,
    items_per_thread: usize,
    f: F,
) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Vec<U> + Sync + Send,
{
    use rayon::prelude::*;
    if use_parallel(work_items, items_per_thread) {
        data.par_iter().flat_map(f).collect()
    } else {
        data.iter().flat_map(f).collect()
    }
}

/// Filter-map over indices with serial auto-select. Both paths visit
/// indices in order, so results are identical.
pub fn gated_filter_map_indexed<U, F>(
    count: usize,
    work_items: usize,
    items_per_thread: usize,
    f: F,
) -> Vec<U>
where
    U: Send,
    F: Fn(usize) -> Option<U> + Sync + Send,
{
    use rayon::prelude::*;
    if use_parallel(work_items, items_per_thread) {
        (0..count).into_par_iter().filter_map(f).collect()
    } else {
        (0..count).filter_map(f).collect()
    }
}

/// Mutate disjoint chunks of a slice with serial auto-select.
///
/// Applies `f(i, chunk)` to each `chunk_len` chunk in order, using rayon
/// when the policy selects parallel for the chunk count. Both paths visit
/// chunks in order; chunks are disjoint, so the mutation is race-free
/// either way. Like the underlying chunk iterators, a zero `chunk_len`
/// panics.
pub fn gated_chunks_mut<T, F>(slice: &mut [T], chunk_len: usize, items_per_thread: usize, f: F)
where
    T: Send + Sync,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    use rayon::prelude::*;
    if use_parallel(slice.len().div_ceil(chunk_len), items_per_thread) {
        slice
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    } else {
        for (i, chunk) in slice.chunks_mut(chunk_len).enumerate() {
            f(i, chunk);
        }
    }
}

/// Mutate slice elements with serial auto-select. Both paths visit indices
/// in order; each element is touched by exactly one invocation.
pub fn gated_for_each_mut<T, F>(slice: &mut [T], items_per_thread: usize, f: F)
where
    T: Send + Sync,
    F: Fn(usize, &mut T) + Sync + Send,
{
    use rayon::prelude::*;
    if use_parallel(slice.len(), items_per_thread) {
        slice.par_iter_mut().enumerate().for_each(|(i, x)| f(i, x));
    } else {
        for (i, x) in slice.iter_mut().enumerate() {
            f(i, x);
        }
    }
}

/// Parallel map over a slice
pub fn parallel_map<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    use rayon::prelude::*;
    data.par_iter().map(f).collect()
}

/// Parallel map over a range of indices
pub fn parallel_map_indexed<U, F>(count: usize, f: F) -> Vec<U>
where
    U: Send,
    F: Fn(usize) -> U + Sync + Send,
{
    use rayon::prelude::*;
    (0..count).into_par_iter().map(f).collect()
}

/// Parallel flat_map over a slice
pub fn parallel_flat_map<T, U, I, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    I: rayon::iter::IntoParallelIterator<Item = U>,
    F: Fn(&T) -> I + Sync + Send,
{
    use rayon::prelude::*;
    data.par_iter().flat_map(f).collect()
}

/// Parallel filter_map over a slice
pub fn parallel_filter_map<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Option<U> + Sync + Send,
{
    use rayon::prelude::*;
    data.par_iter().filter_map(f).collect()
}

/// Parallel for_each over a slice
pub fn parallel_for_each<T, F>(data: &[T], f: F)
where
    T: Sync,
    F: Fn(&T) + Sync + Send,
{
    use rayon::prelude::*;
    data.par_iter().for_each(f);
}

/// Parallel enumerate and for_each over a slice
pub fn parallel_enumerate_for_each<T, F>(data: &[T], f: F)
where
    T: Sync,
    F: Fn(usize, &T) + Sync + Send,
{
    use rayon::prelude::*;
    data.par_iter().enumerate().for_each(|(i, x)| f(i, x));
}

/// Parallel enumerate and map over a slice
pub fn parallel_enumerate_map<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> U + Sync + Send,
{
    use rayon::prelude::*;
    data.par_iter().enumerate().map(|(i, x)| f(i, x)).collect()
}

/// Parallel enumerate and filter_map over a slice
pub fn parallel_enumerate_filter_map<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> Option<U> + Sync + Send,
{
    use rayon::prelude::*;
    data.par_iter()
        .enumerate()
        .filter_map(|(i, x)| f(i, x))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_map() {
        let data = vec![1, 2, 3, 4, 5];
        let result = parallel_map(&data, |x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_map_indexed() {
        let result = parallel_map_indexed(5, |i| i * 2);
        assert_eq!(result, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_parallel_filter_map() {
        let data = vec![1, 2, 3, 4, 5];
        let result = parallel_filter_map(&data, |x| if *x % 2 == 0 { Some(*x) } else { None });
        assert_eq!(result, vec![2, 4]);
    }

    #[test]
    fn test_parallel_enumerate_map() {
        let data = vec![10, 20, 30];
        let result = parallel_enumerate_map(&data, |i, x| i + *x);
        assert_eq!(result, vec![10, 21, 32]);
    }

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
        assert!(eight.decide(8 * BEM_ENTRIES_PER_THREAD, BEM_ENTRIES_PER_THREAD));
        assert!(!eight.decide(8 * BEM_ENTRIES_PER_THREAD - 1, BEM_ENTRIES_PER_THREAD));
    }

    /// Restores a process env var on drop. Only path selection reads these
    /// keys, and both paths are exactly correct, so concurrent tests are
    /// unaffected beyond which (correct) path they take.
    struct EnvVarGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var_os(key);
            // SAFETY: path selection only; both paths are exactly correct.
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            // SAFETY: see `set`.
            unsafe {
                std::env::remove_var(key);
            }
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: restores the pre-test value; see `set`.
            unsafe {
                match &self.prev {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn gated_helpers_match_across_paths() {
        fn run_all(data: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<usize>, Vec<i32>, Vec<i32>, Vec<i32>) {
            (
                gated_map_indexed(data.len(), usize::MAX, 1, |i| data[i] * 2),
                gated_map(data, usize::MAX, 1, |x| x * 2),
                gated_enumerate_filter_map(data, usize::MAX, 1, |i, x| (x % 2 == 0).then_some(i)),
                gated_flat_map(data, usize::MAX, 1, |x| vec![*x, *x]),
                gated_filter_map_indexed(data.len(), usize::MAX, 1, |i| {
                    (data[i] % 2 == 0).then_some(data[i])
                }),
                gated_enumerate_map(&data, usize::MAX, 1, |i, x| i as i32 + x),
            )
        }
        fn run_mut(data: &mut [i32]) {
            gated_chunks_mut(data, 7, 1, |i, chunk| {
                for x in chunk.iter_mut() {
                    *x += i as i32;
                }
            });
            gated_for_each_mut(data, 1, |_, x| *x *= 2);
        }
        let data: Vec<i32> = (0..64).collect();
        let serial = {
            let _force_serial = EnvVarGuard::set("SONIUM_SERIAL", "1");
            let mut serial_mut = data.clone();
            run_mut(&mut serial_mut);
            (run_all(&data), serial_mut)
        };
        let parallel = {
            let _threshold = EnvVarGuard::set("SONIUM_BEM_PAR_THRESHOLD", "1");
            let _no_serial = EnvVarGuard::remove("SONIUM_SERIAL");
            let mut parallel_mut = data.clone();
            run_mut(&mut parallel_mut);
            (run_all(&data), parallel_mut)
        };
        // On single-threaded machines the "parallel" side also runs
        // serially; either way every helper must agree exactly.
        assert_eq!(serial, parallel);
        assert_eq!(serial.0.0, data.iter().map(|x| x * 2).collect::<Vec<_>>());
    }
}
