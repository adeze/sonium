//! Resource-budgeted contiguous-lane executor.
//!
//! Splits a set of independent work items (e.g. frequency points) into a
//! small number of contiguous, ordered lanes and executes each lane on a
//! dedicated worker of a local Rayon pool. Each lane constructs its state
//! exactly once and processes its items in ascending order, so per-lane
//! caches (warm starts, recycled subspaces, reused preconditioners)
//! compound instead of being defeated by work-stealing chunk reassignments.
//!
//! The lane count is bounded by an explicit thread budget, an optional
//! maximum, and an optional memory budget divided by the estimated
//! per-item live bytes. This module is generic: it knows nothing about
//! solver types.

use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Total physical host memory in bytes, or `None` when it cannot be
/// determined (unsupported platform or failed query). Callers should treat
/// `None` as "no memory limit".
pub fn host_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut bytes: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        // SAFETY: `bytes` is a valid u64 buffer of `len` bytes; no new
        // value is written (`null` newp). On success the buffer is filled.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut bytes as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && len == std::mem::size_of::<u64>()).then_some(bytes)
    }
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        meminfo.lines().find_map(|line| {
            line.strip_prefix("MemTotal:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
                .map(|kib| kib * 1024)
        })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Budget for lane planning.
#[derive(Debug, Clone)]
pub struct LaneConfig {
    /// Hard cap on the number of lanes. `None` derives the lane count from
    /// the remaining budgets.
    pub max_lanes: Option<usize>,
    /// Maximum number of worker threads the execution may use.
    pub thread_budget: usize,
    /// Total bytes all lanes' live state may occupy together. `None` means
    /// no memory limit (also the fallback when host memory is unknown).
    pub memory_budget_bytes: Option<u64>,
}

impl Default for LaneConfig {
    fn default() -> Self {
        Self {
            max_lanes: None,
            thread_budget: 1,
            memory_budget_bytes: None,
        }
    }
}

/// A nonempty workload cannot fit within the supplied resource budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanePlanError {
    ZeroThreadBudget,
    ZeroLaneLimit,
    InsufficientMemory {
        required_bytes: u64,
        budget_bytes: u64,
    },
}

impl std::fmt::Display for LanePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroThreadBudget => write!(f, "lane thread budget must be positive"),
            Self::ZeroLaneLimit => write!(f, "lane limit must be positive"),
            Self::InsufficientMemory {
                required_bytes,
                budget_bytes,
            } => write!(
                f,
                "one lane requires {required_bytes} bytes, exceeding the {budget_bytes}-byte budget"
            ),
        }
    }
}

impl std::error::Error for LanePlanError {}

/// A lane assignment: `chunks` contains `num_lanes` contiguous, ascending,
/// non-overlapping index ranges that together cover `0..item_count`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanePlan {
    pub num_lanes: usize,
    pub chunks: Vec<Range<usize>>,
}

/// Plan contiguous, roughly count-balanced lanes over `item_count` items.
///
/// `per_item_live_bytes(index)` estimates the live memory occupied while
/// processing one item; per-item memory is assumed similar across items, so
/// only the maximum drives the memory-derived lane cap
/// (`budget / max_per_item_live_bytes`). Chunks are balanced by item count,
/// not by estimated cost. Panics if a nonempty workload cannot fit;
/// use [`try_plan_lanes`] at fallible application boundaries.
pub fn plan_lanes(
    item_count: usize,
    per_item_live_bytes: impl Fn(usize) -> u64,
    cfg: &LaneConfig,
) -> LanePlan {
    try_plan_lanes(item_count, per_item_live_bytes, cfg).expect("lane resource admission failed")
}

/// Fallible lane planning. Never admits even one lane over the memory budget.
/// Estimates must include all per-lane retained state; this is not an OS quota.
pub fn try_plan_lanes(
    item_count: usize,
    per_item_live_bytes: impl Fn(usize) -> u64,
    cfg: &LaneConfig,
) -> Result<LanePlan, LanePlanError> {
    if item_count == 0 {
        return Ok(LanePlan {
            num_lanes: 0,
            chunks: Vec::new(),
        });
    }
    if cfg.thread_budget == 0 {
        return Err(LanePlanError::ZeroThreadBudget);
    }
    let mut num_lanes = item_count.min(cfg.thread_budget);
    if let Some(max_lanes) = cfg.max_lanes {
        if max_lanes == 0 {
            return Err(LanePlanError::ZeroLaneLimit);
        }
        num_lanes = num_lanes.min(max_lanes);
    }
    if let Some(budget) = cfg.memory_budget_bytes {
        let max_item_bytes = (0..item_count).map(&per_item_live_bytes).max().unwrap_or(0);
        if max_item_bytes > budget {
            return Err(LanePlanError::InsufficientMemory {
                required_bytes: max_item_bytes,
                budget_bytes: budget,
            });
        }
        if let Some(per_lane) = budget.checked_div(max_item_bytes) {
            let memory_cap = usize::try_from(per_lane).unwrap_or(usize::MAX);
            num_lanes = num_lanes.min(memory_cap);
        }
    }
    let num_lanes = num_lanes.max(1);
    let base = item_count / num_lanes;
    let remainder = item_count % num_lanes;
    let mut chunks = Vec::with_capacity(num_lanes);
    let mut start = 0;
    for lane in 0..num_lanes {
        let len = base + usize::from(lane < remainder);
        chunks.push(start..start + len);
        start += len;
    }
    Ok(LanePlan { num_lanes, chunks })
}

/// Result of a lane execution. `results[i]` is `Some` for every processed
/// item and `None` for items skipped after cancellation, so output order
/// always matches input order.
#[derive(Debug)]
pub struct LaneOutcome<R> {
    pub results: Vec<Option<R>>,
    pub completed: usize,
    pub cancelled: bool,
}

/// Thread pool shared by all `run_lanes` calls with the same lane count.
///
/// Building a Rayon pool spawns OS threads; doing it per sweep call pays
/// pool creation on every invocation and oversubscribes when nested with the
/// global + bandwidth pools. Idle entries are weak (dropped as soon as the
/// last user releases them) except for the single most-recently-used pool,
/// and [`LanePoolCache::max_retained`] caps how many idle entries may pile
/// up; live pools held by concurrent users are never destroyed by eviction.
#[derive(Debug)]
struct LanePoolCache {
    active: HashMap<usize, Weak<rayon::ThreadPool>>,
    recent: Option<Arc<rayon::ThreadPool>>,
    max_retained: usize,
}

/// Default cap on distinct retained lane-count entries (see [`LanePoolCache`]).
pub const DEFAULT_MAX_RETAINED_LANE_POOLS: usize = 8;

impl Default for LanePoolCache {
    fn default() -> Self {
        Self {
            active: HashMap::new(),
            recent: None,
            max_retained: DEFAULT_MAX_RETAINED_LANE_POOLS,
        }
    }
}

impl LanePoolCache {
    fn get(&mut self, num_lanes: usize) -> Arc<rayon::ThreadPool> {
        self.evict_idle();
        let pool = self
            .active
            .get(&num_lanes)
            .and_then(Weak::upgrade)
            .unwrap_or_else(|| {
                Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(num_lanes)
                        .build()
                        .expect("failed to build lane thread pool"),
                )
            });
        self.active.insert(num_lanes, Arc::downgrade(&pool));
        self.recent = Some(Arc::clone(&pool));
        // Enforce the bound after inserting: drop fully idle entries (no
        // live users) beyond `max_retained`, keeping the just-used pool.
        self.enforce_bound(Some(num_lanes));
        pool
    }

    /// Number of retained entries (live or idle).
    fn retained(&self) -> usize {
        self.active.len()
    }

    /// Drop every entry whose pool has no live users. Returns the number of
    /// entries removed. The most-recently-used pool is strongly held and
    /// therefore always survives.
    fn evict_idle(&mut self) -> usize {
        let before = self.active.len();
        self.active.retain(|_, pool| pool.strong_count() > 0);
        before - self.active.len()
    }

    /// Change the retention bound (applies on the next [`Self::get`]).
    fn set_max_retained(&mut self, max_retained: usize) {
        self.max_retained = max_retained.max(1);
        self.enforce_bound(None);
    }

    /// Drop all entries, including the most-recently-used pin. Pools with
    /// live users keep working; only idle cache entries disappear.
    fn clear(&mut self) {
        self.active.clear();
        self.recent = None;
    }

    fn enforce_bound(&mut self, keep: Option<usize>) {
        if self.active.len() <= self.max_retained.max(1) {
            return;
        }
        // First pass: dead weak entries cost nothing to drop.
        self.active.retain(|_, pool| pool.strong_count() > 0);
        if self.active.len() <= self.max_retained.max(1) {
            return;
        }
        // Second pass: fully idle entries (no live Arc anywhere) beyond the
        // bound, never the just-used pool.
        let mut idle: Vec<usize> = self
            .active
            .iter()
            .filter(|(lanes, pool)| Some(**lanes) != keep && pool.strong_count() == 0)
            .map(|(lanes, _)| *lanes)
            .collect();
        idle.sort_unstable();
        let over = self.active.len().saturating_sub(self.max_retained.max(1));
        for lanes in idle.into_iter().take(over) {
            self.active.remove(&lanes);
        }
    }
}

/// Explicit, injectable lifecycle for lane thread pools.
///
/// The global [`run_lanes`] cache is process-lifetime and shared; a registry
/// lets an application scope pool retention (bursty sweeps with many lane
/// counts, tests asserting teardown, executors with memory accounting).
/// Dropping the registry releases every idle pool it retained; pools with
/// live users keep working until released.
#[derive(Debug, Default)]
pub struct LanePoolRegistry {
    cache: LanePoolCache,
}

impl LanePoolRegistry {
    /// New registry retaining at most `max_retained` idle lane-count entries
    /// (at least one; the most-recently-used pool always survives).
    pub fn new(max_retained: usize) -> Self {
        let mut cache = LanePoolCache::default();
        cache.set_max_retained(max_retained);
        Self { cache }
    }

    /// Get (or build) the shared pool for `num_lanes`.
    pub fn get(&mut self, num_lanes: usize) -> Arc<rayon::ThreadPool> {
        self.cache.get(num_lanes)
    }

    /// Drop idle entries; returns the number removed.
    pub fn evict_idle(&mut self) -> usize {
        self.cache.evict_idle()
    }

    /// Drop all retained entries, including the most-recent pin.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of currently retained entries.
    pub fn retained(&self) -> usize {
        self.cache.retained()
    }
}

fn lane_pool(num_lanes: usize) -> Arc<rayon::ThreadPool> {
    static POOLS: OnceLock<Mutex<LanePoolCache>> = OnceLock::new();
    POOLS
        .get_or_init(|| Mutex::new(LanePoolCache::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(num_lanes)
}

/// Logical CPUs available to this process (floor 1).
///
/// Backs thread-budget validation: an explicit budget above this value
/// oversubscribes the host.
pub fn host_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Clamp an explicit thread budget to host capacity.
///
/// A zero budget stays zero (still invalid: see
/// [`LanePlanError::ZeroThreadBudget`]); any positive budget is clamped to
/// `[1, host_parallelism()]`. Lane execution deliberately does *not* clamp
/// implicit budgets — an explicit caller budget wins — so apply this at
/// application boundaries that derive budgets from machine size.
pub fn clamp_thread_budget_to_host(thread_budget: usize) -> usize {
    if thread_budget == 0 {
        0
    } else {
        thread_budget.min(host_parallelism()).max(1)
    }
}

/// True when `thread_budget` exceeds host capacity, i.e. the budget would
/// oversubscribe the machine if fully staffed with workers.
pub fn thread_budget_exceeds_host(thread_budget: usize) -> bool {
    thread_budget > host_parallelism()
}

/// Execute `items` according to `plan`.
///
/// `state_init` is called exactly once per lane, on that lane's worker, and
/// the lane's items are then processed in ascending index order via
/// `work(&mut state, index, item)`. When `cancel` is provided and becomes
/// `true`, lanes stop between items; already computed results are kept and
/// unprocessed items remain `None`. Lanes run on a dedicated local Rayon
/// pool with `num_lanes` threads (cached across calls, see `lane_pool`) so
/// they don't fight the global pool. Callers that already own a pool should
/// use [`run_lanes_on_pool`] instead, and callers that scope pool retention
/// should use [`run_lanes_on_registry`]. Note the inner `work` still runs on
/// the global pool; callers with rayon-heavy work should prefer
/// [`run_lanes_with_worker_budgets`] to keep lanes × workers in budget.
///
/// Nested execution (called from inside a Rayon worker) runs serially on the
/// calling thread with identical per-chunk state semantics instead of
/// installing another pool, which risks pool-exhaustion deadlock and
/// unbounded thread growth.
pub fn run_lanes<T, S, R>(
    items: &[T],
    plan: &LanePlan,
    state_init: impl Fn() -> S + Send + Sync,
    work: &(impl Fn(&mut S, usize, &T) -> R + Sync),
    cancel: Option<Arc<AtomicBool>>,
) -> LaneOutcome<R>
where
    T: Sync,
    S: Send,
    R: Send,
{
    if plan.num_lanes == 0 || items.is_empty() {
        return empty_outcome(items.len());
    }
    if rayon::current_thread_index().is_some() {
        return run_lanes_serial(items, plan, state_init, work, cancel);
    }
    let pool = lane_pool(plan.num_lanes);
    run_lanes_on_pool(items, plan, &pool, state_init, work, cancel)
}

/// [`run_lanes`] on a caller-managed [`LanePoolRegistry`] (explicit pool
/// retention lifecycle instead of the process-global cache).
pub fn run_lanes_on_registry<T, S, R>(
    items: &[T],
    plan: &LanePlan,
    registry: &mut LanePoolRegistry,
    state_init: impl Fn() -> S + Send + Sync,
    work: &(impl Fn(&mut S, usize, &T) -> R + Sync),
    cancel: Option<Arc<AtomicBool>>,
) -> LaneOutcome<R>
where
    T: Sync,
    S: Send,
    R: Send,
{
    if plan.num_lanes == 0 || items.is_empty() {
        return empty_outcome(items.len());
    }
    if rayon::current_thread_index().is_some() {
        return run_lanes_serial(items, plan, state_init, work, cancel);
    }
    let pool = registry.get(plan.num_lanes);
    run_lanes_on_pool(items, plan, &pool, state_init, work, cancel)
}

fn empty_outcome<R>(len: usize) -> LaneOutcome<R> {
    LaneOutcome {
        results: (0..len).map(|_| None).collect(),
        completed: 0,
        cancelled: false,
    }
}

/// Serial lane execution with identical per-chunk state semantics.
///
/// Used for nested calls (see [`run_lanes`]): every chunk constructs its own
/// state once and processes its items in ascending order, so results match
/// parallel execution exactly, minus the parallelism.
fn run_lanes_serial<T, S, R>(
    items: &[T],
    plan: &LanePlan,
    state_init: impl Fn() -> S,
    work: &impl Fn(&mut S, usize, &T) -> R,
    cancel: Option<Arc<AtomicBool>>,
) -> LaneOutcome<R> {
    let mut results: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    let mut completed = 0;
    for chunk in &plan.chunks {
        let mut state = state_init();
        for index in chunk.clone() {
            if let Some(flag) = &cancel
                && flag.load(Ordering::Relaxed)
            {
                break;
            }
            results[index] = Some(work(&mut state, index, &items[index]));
            completed += 1;
        }
    }
    LaneOutcome {
        cancelled: completed < items.len(),
        results,
        completed,
    }
}

/// [`run_lanes`] with per-lane worker budgets (nested pools).
///
/// Each lane's state construction and `work` run inside a dedicated Rayon
/// pool of `worker_threads_per_lane` threads, so rayon calls inside the
/// work (assembly, SpMV, orthogonalization) stay within the lane's share:
/// `num_lanes × worker_threads_per_lane` bounds total workers. Plain
/// [`run_lanes`] instead lets every lane fan its inner work out to the
/// global pool, which oversubscribes past the memory-bandwidth knee
/// (measured: 2 lanes slower than serial on a 128-thread box,
/// SIM-PERF-20261013.md section 6, recommendation 4). Callers typically
/// pass `thread_budget / num_lanes` (floored at 1).
///
/// Worker pools are built per call (pool construction is milliseconds
/// against sweep seconds) and dropped on return; the outer lane
/// scheduling reuses the cached [`lane_pool`]. Like [`run_lanes`], nested
/// calls from inside a Rayon worker run serially on the calling thread.
///
/// # Panics
///
/// Panics when a worker pool cannot be built (OS thread exhaustion).
pub fn run_lanes_with_worker_budgets<T, S, R>(
    items: &[T],
    plan: &LanePlan,
    worker_threads_per_lane: usize,
    state_init: impl Fn() -> S + Send + Sync,
    work: &(impl Fn(&mut S, usize, &T) -> R + Sync),
    cancel: Option<Arc<AtomicBool>>,
) -> LaneOutcome<R>
where
    T: Sync,
    S: Send,
    R: Send,
{
    if plan.num_lanes == 0 || items.is_empty() {
        return empty_outcome(items.len());
    }
    if rayon::current_thread_index().is_some() {
        return run_lanes_serial(items, plan, state_init, work, cancel);
    }
    let workers = worker_threads_per_lane.max(1);
    let worker_pools: Vec<rayon::ThreadPool> = (0..plan.num_lanes)
        .map(|lane| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(move |index| format!("lane-{lane}-worker-{index}"))
                .build()
                .expect("lane worker pool construction failed")
        })
        .collect();
    let pool = lane_pool(plan.num_lanes);
    let mut results: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    let lane_results: Vec<Vec<(usize, R)>> = pool.install(|| {
        plan.chunks
            .par_iter()
            .enumerate()
            .map(|(lane_idx, chunk)| {
                worker_pools[lane_idx].install(|| {
                    let mut state = state_init();
                    let mut outputs = Vec::with_capacity(chunk.len());
                    for index in chunk.clone() {
                        if let Some(flag) = &cancel
                            && flag.load(Ordering::Relaxed)
                        {
                            break;
                        }
                        outputs.push((index, work(&mut state, index, &items[index])));
                    }
                    outputs
                })
            })
            .collect()
    });
    let mut completed = 0;
    for lane in lane_results {
        for (index, output) in lane {
            results[index] = Some(output);
            completed += 1;
        }
    }
    LaneOutcome {
        cancelled: completed < items.len(),
        results,
        completed,
    }
}

/// [`run_lanes`] on a caller-supplied pool (no pool construction or cache
/// lookup; useful when the caller already scopes its own executor).
///
/// Like [`run_lanes`], nested calls from inside a Rayon worker run serially
/// on the calling thread — including when `pool` is the very pool whose
/// worker is calling — instead of re-installing and risking deadlock.
pub fn run_lanes_on_pool<T, S, R>(
    items: &[T],
    plan: &LanePlan,
    pool: &rayon::ThreadPool,
    state_init: impl Fn() -> S + Send + Sync,
    work: &(impl Fn(&mut S, usize, &T) -> R + Sync),
    cancel: Option<Arc<AtomicBool>>,
) -> LaneOutcome<R>
where
    T: Sync,
    S: Send,
    R: Send,
{
    if plan.num_lanes == 0 || items.is_empty() {
        return empty_outcome(items.len());
    }
    if rayon::current_thread_index().is_some() {
        return run_lanes_serial(items, plan, state_init, work, cancel);
    }
    let mut results: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    let lane_results: Vec<Vec<(usize, R)>> = pool.install(|| {
        plan.chunks
            .par_iter()
            .map(|chunk| {
                // No lock: `state_init` is `Fn + Sync`, so every lane
                // constructs its own state concurrently instead of
                // serializing on a shared mutex (which stalled startup when
                // init builds a preconditioner).
                let mut state = state_init();
                let mut outputs = Vec::with_capacity(chunk.len());
                for index in chunk.clone() {
                    if let Some(flag) = &cancel
                        && flag.load(Ordering::Relaxed)
                    {
                        break;
                    }
                    outputs.push((index, work(&mut state, index, &items[index])));
                }
                outputs
            })
            .collect()
    });
    let mut completed = 0;
    for lane in lane_results {
        for (index, output) in lane {
            results[index] = Some(output);
            completed += 1;
        }
    }
    LaneOutcome {
        cancelled: completed < items.len(),
        results,
        completed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn config(
        max_lanes: Option<usize>,
        thread_budget: usize,
        memory_budget_bytes: Option<u64>,
    ) -> LaneConfig {
        LaneConfig {
            max_lanes,
            thread_budget,
            memory_budget_bytes,
        }
    }

    #[test]
    fn plan_respects_thread_budget() {
        let plan = plan_lanes(10, |_| 1, &config(None, 4, None));
        assert_eq!(plan.num_lanes, 4);
    }

    #[test]
    fn worker_budgets_cap_inner_pool_width() {
        // SIM-PERF-20261013.md section 6, recommendation 4: inner rayon
        // work must see the lane's worker pool, not the global pool.
        let plan = plan_lanes(8, |_| 1, &config(Some(2), 8, None));
        assert_eq!(plan.num_lanes, 2);
        let outcome = run_lanes_with_worker_budgets(
            &[0usize, 1, 2, 3, 4, 5, 6, 7],
            &plan,
            3,
            || 0usize,
            &|_state, _index, _item| rayon::current_num_threads(),
            None,
        );
        assert!(!outcome.cancelled);
        for width in outcome.results.into_iter().flatten() {
            assert_eq!(width, 3);
        }
    }

    #[test]
    fn worker_budgets_cover_every_item_once() {
        let plan = plan_lanes(7, |_| 1, &config(Some(3), 9, None));
        let seen = std::sync::Mutex::new(vec![false; 7]);
        let outcome = run_lanes_with_worker_budgets(
            &[10usize, 11, 12, 13, 14, 15, 16],
            &plan,
            2,
            Vec::new,
            &|state: &mut Vec<usize>, index, item| {
                state.push(*item);
                seen.lock().unwrap()[index] = true;
                *item * 2
            },
            None,
        );
        assert!(!outcome.cancelled);
        assert_eq!(outcome.completed, 7);
        assert!(seen.lock().unwrap().iter().all(|seen| *seen));
        let values: Vec<usize> = outcome.results.into_iter().flatten().collect();
        assert_eq!(values, vec![20, 22, 24, 26, 28, 30, 32]);
    }

    #[test]
    fn plan_respects_max_lanes() {
        let plan = plan_lanes(10, |_| 1, &config(Some(2), 8, None));
        assert_eq!(plan.num_lanes, 2);
    }

    #[test]
    fn plan_respects_memory_budget() {
        // 100 bytes per item, 350 byte budget -> at most 3 lanes.
        let plan = plan_lanes(10, |_| 100, &config(None, 8, Some(350)));
        assert_eq!(plan.num_lanes, 3);
    }

    #[test]
    fn plan_rejects_a_single_item_over_budget() {
        assert_eq!(
            try_plan_lanes(10, |_| 100, &config(None, 8, Some(50))),
            Err(LanePlanError::InsufficientMemory {
                required_bytes: 100,
                budget_bytes: 50
            })
        );
        assert!(try_plan_lanes(1, |_| 1, &config(None, 1, Some(0))).is_err());
        assert_eq!(
            try_plan_lanes(1, |_| 0, &config(None, 1, Some(0)))
                .unwrap()
                .num_lanes,
            1
        );
        assert_eq!(
            try_plan_lanes(2, |_| 1, &config(None, 2, Some(u64::MAX)))
                .unwrap()
                .num_lanes,
            2
        );
        assert_eq!(
            try_plan_lanes(1, |_| 0, &config(None, 0, None)),
            Err(LanePlanError::ZeroThreadBudget)
        );
        assert_eq!(
            try_plan_lanes(1, |_| 0, &config(Some(0), 1, None)),
            Err(LanePlanError::ZeroLaneLimit)
        );
    }

    #[test]
    fn lane_cache_retains_one_idle_pool_and_reuses_active_pools() {
        let mut cache = LanePoolCache::default();
        let first = cache.get(1);
        assert!(Arc::ptr_eq(&first, &cache.get(1)));
        let second = cache.get(2);
        assert!(Arc::ptr_eq(&first, &cache.get(1)));
        let old_second = Arc::downgrade(&second);
        drop(second);
        assert!(old_second.upgrade().is_none());
        let old_first = Arc::downgrade(&first);
        drop(first);
        let _third = cache.get(3);
        assert!(old_first.upgrade().is_none());
        cache.get(3);
        assert_eq!(cache.active.len(), 1);
    }

    #[test]
    fn registry_bounds_idle_retention_and_supports_teardown() {
        let mut registry = LanePoolRegistry::new(2);
        {
            let _p1 = registry.get(1);
            let _p2 = registry.get(2);
            let _p3 = registry.get(3);
            // Live pools are never destroyed by eviction, even over budget.
            assert_eq!(registry.retained(), 3);
        }
        // All released: only the most-recent pin survives eviction.
        assert_eq!(registry.evict_idle(), 2);
        assert_eq!(registry.retained(), 1);
        registry.clear();
        assert_eq!(registry.retained(), 0);
        // Registry stays usable after teardown.
        let plan = plan_lanes(4, |_| 1, &config(Some(2), 2, None));
        let items = vec![1u64, 2, 3, 4];
        let outcome = run_lanes_on_registry(
            &items,
            &plan,
            &mut registry,
            || 0u64,
            &|_, _, &x| x * 2,
            None,
        );
        assert_eq!(outcome.completed, 4);
        let results: Vec<u64> = outcome.results.into_iter().map(Option::unwrap).collect();
        assert_eq!(results, vec![2, 4, 6, 8]);
    }

    #[test]
    fn registry_reuses_pools_and_enforces_bound_on_insert() {
        let mut registry = LanePoolRegistry::new(2);
        let first = registry.get(2);
        assert!(Arc::ptr_eq(&first, &registry.get(2)));
        drop(first);
        // Idle pool beyond the… still reusable while retained.
        assert!(registry.retained() <= 2);
    }

    #[test]
    fn nested_run_lanes_completes_serially_with_identical_results() {
        let items: Vec<u64> = (0..16).collect();
        let expected: Vec<u64> = items.iter().map(|x| x * 2).collect();
        let plan = plan_lanes(items.len(), |_| 1, &config(Some(4), 4, None));
        let outer = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let outcome = outer.install(|| {
            assert!(rayon::current_thread_index().is_some());
            run_lanes(&items, &plan, || 0u64, &|_s, _i, &x| x * 2, None)
        });
        assert!(!outcome.cancelled);
        let results: Vec<u64> = outcome.results.into_iter().map(Option::unwrap).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn nested_run_lanes_on_same_pool_does_not_deadlock() {
        let items: Vec<u64> = (0..16).collect();
        let expected: Vec<u64> = items.iter().map(|x| x * 3).collect();
        let plan = plan_lanes(items.len(), |_| 1, &config(Some(4), 4, None));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        // Re-installing the calling pool from its own worker would stall;
        // the nested guard runs serially instead.
        let outcome = pool.install(|| {
            run_lanes_on_pool(&items, &plan, &pool, || 0u64, &|_s, _i, &x| x * 3, None)
        });
        assert!(!outcome.cancelled);
        let results: Vec<u64> = outcome.results.into_iter().map(Option::unwrap).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn thread_budget_host_validation_clamps_and_detects_oversubscription() {
        let host = host_parallelism();
        assert!(host >= 1);
        assert_eq!(clamp_thread_budget_to_host(0), 0);
        assert_eq!(clamp_thread_budget_to_host(1), 1);
        assert_eq!(clamp_thread_budget_to_host(host), host);
        assert_eq!(clamp_thread_budget_to_host(host.saturating_add(1000)), host);
        assert!(!thread_budget_exceeds_host(1));
        assert!(!thread_budget_exceeds_host(host));
        assert!(thread_budget_exceeds_host(host.saturating_add(1)));
    }

    #[test]
    fn plan_never_exceeds_item_count() {
        let plan = plan_lanes(3, |_| 1, &config(None, 16, None));
        assert_eq!(plan.num_lanes, 3);
        assert_eq!(plan.chunks, vec![0..1, 1..2, 2..3]);
    }

    #[test]
    fn plan_handles_empty_input() {
        let plan = plan_lanes(0, |_| 1, &config(None, 8, None));
        assert_eq!(plan.num_lanes, 0);
        assert!(plan.chunks.is_empty());
    }

    #[test]
    fn plan_chunks_are_contiguous_balanced_and_cover_all_items() {
        for &(items, lanes) in &[(10usize, 3usize), (11, 4), (40, 12), (7, 7)] {
            let plan = plan_lanes(items, |_| 1, &config(Some(lanes), lanes, None));
            assert_eq!(plan.num_lanes, lanes);
            let mut expected_start = 0;
            for chunk in &plan.chunks {
                assert_eq!(chunk.start, expected_start, "chunks must be gapless");
                expected_start = chunk.end;
            }
            assert_eq!(expected_start, items, "chunks must cover every item");
            let max_len = plan.chunks.iter().map(|chunk| chunk.len()).max().unwrap();
            let min_len = plan.chunks.iter().map(|chunk| chunk.len()).min().unwrap();
            assert!(max_len - min_len <= 1, "chunks must be roughly balanced");
        }
    }

    #[test]
    fn run_produces_input_order_results_regardless_of_lane_count() {
        let items: Vec<u64> = (0..25).collect();
        let reference: Vec<u64> = items.iter().map(|x| x * 2).collect();
        for lanes in [1, 2, 3, 8] {
            let plan = plan_lanes(items.len(), |_| 1, &config(Some(lanes), lanes, None));
            let outcome = run_lanes(&items, &plan, || 0u64, &|_state, _i, &x| x * 2, None);
            assert!(!outcome.cancelled);
            assert_eq!(outcome.completed, items.len());
            let results: Vec<u64> = outcome.results.into_iter().map(Option::unwrap).collect();
            assert_eq!(
                results, reference,
                "lane count {lanes} must not change order"
            );
        }
    }

    #[test]
    fn run_initializes_one_state_per_lane_and_processes_in_order() {
        let items: Vec<usize> = (0..10).collect();
        let plan = plan_lanes(items.len(), |_| 1, &config(Some(3), 3, None));
        let state_inits = AtomicUsize::new(0);
        let processing_log = Mutex::new(Vec::new());
        let outcome = run_lanes(
            &items,
            &plan,
            || state_inits.fetch_add(1, Ordering::SeqCst),
            &|lane_id: &mut usize, _index, &item| {
                processing_log.lock().unwrap().push((*lane_id, item));
                (item, *lane_id)
            },
            None,
        );
        assert_eq!(state_inits.load(Ordering::SeqCst), plan.num_lanes);
        // Every item of a lane was processed by that lane's single state,
        // in ascending order.
        let processing_log = processing_log.into_inner().unwrap();
        for chunk in &plan.chunks {
            let lane_ids: Vec<usize> = chunk
                .clone()
                .map(|i| outcome.results[i].as_ref().unwrap().1)
                .collect();
            assert!(
                lane_ids.windows(2).all(|w| w[0] == w[1]),
                "all items of a lane must carry that lane's state id"
            );
            let lane_items: Vec<usize> = processing_log
                .iter()
                .filter(|(lane_id, _)| *lane_id == lane_ids[0])
                .map(|(_, item)| *item)
                .collect();
            assert_eq!(
                lane_items,
                chunk.clone().collect::<Vec<_>>(),
                "lane must process its items in ascending order"
            );
        }
    }

    #[test]
    fn run_cancellation_yields_partial_results() {
        let items: Vec<usize> = (0..10).collect();
        let plan = plan_lanes(items.len(), |_| 1, &config(Some(2), 2, None));
        let cancel = Arc::new(AtomicBool::new(false));
        let seen = AtomicUsize::new(0);
        let outcome = run_lanes(
            &items,
            &plan,
            || (),
            &|_state, _index, &item| {
                if seen.fetch_add(1, Ordering::SeqCst) >= 3 {
                    cancel.store(true, Ordering::SeqCst);
                }
                item
            },
            Some(cancel.clone()),
        );
        assert!(outcome.cancelled);
        assert!(outcome.completed < items.len());
        // Completed results carry their input values; nothing past the last
        // completed index of a lane is filled.
        for (index, result) in outcome.results.iter().enumerate() {
            if let Some(value) = result {
                assert_eq!(*value, index);
            }
        }
        let filled = outcome.results.iter().filter(|r| r.is_some()).count();
        assert_eq!(filled, outcome.completed);
    }

    #[test]
    fn host_memory_bytes_returns_something_on_supported_platforms() {
        let bytes = host_memory_bytes();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(bytes.is_some_and(|b| b > 0));
        }
    }
}
