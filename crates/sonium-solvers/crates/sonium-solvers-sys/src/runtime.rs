//! Runtime configuration and machine-aware thread management.
//!
//! Controls thread pool sizing based on machine characteristics.
//! On heterogeneous architectures (Apple Silicon), limits threads
//! to performance cores to avoid E-core straggler effects in
//! barrier-synchronised work like SpMV.
//!
//! # Background
//!
//! SpMV is memory-bandwidth-bound. On Apple M4 Pro (10 P-cores + 4 E-cores):
//! - 4 threads saturate memory bandwidth for typical problem sizes
//! - E-cores run at ~60% of P-core speed, causing straggler effects
//! - Adding threads beyond the bandwidth saturation point *hurts* performance
//!
//! # Usage
//!
//! Call [`configure_rayon`] once at program startup before any solver operations:
//!
//! ```ignore
//! sonium_solvers::runtime::configure_rayon();
//! ```

use std::sync::{Mutex, OnceLock};

/// Runtime configuration for sonium-solvers.
///
/// Controls thread pool sizing based on machine characteristics.
/// On heterogeneous architectures (Apple Silicon), limits threads
/// to performance cores to avoid E-core straggler effects.
#[derive(Debug, Clone)]
pub struct SolverRuntime {
    /// Recommended thread count for compute-bound work (e.g., AMG setup).
    /// Set to the number of performance cores.
    pub compute_threads: usize,
    /// Recommended thread count for bandwidth-bound work (e.g., SpMV).
    /// Capped below `compute_threads` because memory bandwidth saturates
    /// before all P-cores are busy.
    pub bandwidth_threads: usize,
    /// Total available cores (P-cores + E-cores).
    pub total_cores: usize,
    /// Performance cores (if detectable, else same as `total_cores`).
    pub perf_cores: usize,
}

impl SolverRuntime {
    /// Detect the optimal thread configuration for this machine.
    ///
    /// On macOS (Apple Silicon) this reads `sysctl` to distinguish
    /// performance and efficiency cores. On other platforms it falls back
    /// to [`std::thread::available_parallelism`].
    ///
    /// `SONIUM_BW_THREADS` overrides `bandwidth_threads` at the floor when
    /// set to a positive integer. `0` disables the dedicated bandwidth pool
    /// (SpMV always runs on the global compute pool). Useful for x86 servers
    /// with many memory channels where the 4-thread cap is too conservative.
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        let mut rt = detect_apple_silicon().unwrap_or_else(detect_generic);
        #[cfg(not(target_os = "macos"))]
        let mut rt = detect_generic();

        if let Some(override_val) = read_env_bw_threads() {
            rt.bandwidth_threads = override_val.min(rt.compute_threads).max(1);
        }
        rt
    }
}

/// Bandwidth-pool width for a machine with `total` logical CPUs.
///
/// Four threads saturate Apple-class memory systems; many-channel x86
/// servers need more streams (measured 20-40% faster SpMV at 16 vs 4
/// threads on a 128-thread Zen2 box, even below the mid tier). One thread
/// per eight logical CPUs preserves the historical 4-thread behaviour up
/// to 32-thread machines and scales beyond.
fn generic_bandwidth_threads(total: usize) -> usize {
    (total / 8).max(4).min(total.max(1)).max(1)
}

fn detect_generic() -> SolverRuntime {
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    SolverRuntime {
        compute_threads: total,
        bandwidth_threads: generic_bandwidth_threads(total),
        total_cores: total,
        perf_cores: total,
    }
}

fn read_env_bw_threads() -> Option<usize> {
    std::env::var("SONIUM_BW_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
}

/// Recommended thread count for SpMV on a matrix with `nnz` non-zeros.
///
/// Below the mid tier the working set fits in L2/L3 and the narrow
/// bandwidth pool typically saturates the memory subsystem. Above that,
/// more streams from extra prefetchers help — up to roughly
/// `large_frac * perf_cores` before bus contention takes over.
/// Tier boundaries come from [`BandwidthTiers::from_env`].
///
/// Used by [`with_bandwidth_pool_for_nnz`] to route small-problem SpMV onto
/// the narrow bandwidth pool while letting large-problem SpMV use the full
/// compute pool.
pub fn bandwidth_threads_for_nnz(nnz: usize) -> usize {
    bandwidth_threads_for_nnz_with(nnz, runtime(), &BandwidthTiers::cached())
}

/// Nnz tier boundaries for [`bandwidth_threads_for_nnz`].
///
/// The historical cutoffs (500k/2M nnz, 0.75 large-problem fraction) were
/// measured on Apple Silicon memory subsystems; multi-channel x86 servers
/// saturate at different points, so every tier is env-overridable:
/// `SONIUM_BW_TIER_MID`, `SONIUM_BW_TIER_LARGE` (absolute nnz), and
/// `SONIUM_BW_LARGE_FRAC` (fraction of `perf_cores`, default 0.75).
#[derive(Debug, Clone, Copy)]
pub struct BandwidthTiers {
    /// Nnz below which the narrow bandwidth pool suffices.
    pub mid: usize,
    /// Nnz below which bandwidth pool + 2 threads suffice.
    pub large: usize,
    /// Fraction of `perf_cores` usable at/above `large`.
    pub large_frac: f64,
}

impl BandwidthTiers {
    /// Historical Apple-Silicon-measured defaults.
    pub fn defaults() -> Self {
        Self {
            mid: 500_000,
            large: 2_000_000,
            large_frac: 0.75,
        }
    }

    /// Resolve from the process environment, falling back to [`Self::defaults`]
    /// per field on missing/unparseable values.
    ///
    /// This reads the environment on every call; prefer
    /// [`cached`](Self::cached) on hot paths such as per-SpMV dispatch.
    pub fn from_env() -> Self {
        fn env_usize(name: &str) -> Option<usize> {
            std::env::var(name)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        }
        let defaults = Self::defaults();
        let large_frac = std::env::var("SONIUM_BW_LARGE_FRAC")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|frac| frac.is_finite() && *frac > 0.0)
            .unwrap_or(defaults.large_frac);
        Self {
            mid: env_usize("SONIUM_BW_TIER_MID").unwrap_or(defaults.mid),
            large: env_usize("SONIUM_BW_TIER_LARGE").unwrap_or(defaults.large),
            large_frac,
        }
    }

    /// Once-per-process [`from_env`](Self::from_env) for hot paths.
    ///
    /// SpMV consults the tiers on every call; resolving four environment
    /// variables per call is pure overhead. Set overrides before process
    /// start — same caching contract as the other `SONIUM_*` dispatch flags.
    pub fn cached() -> Self {
        static CACHED: std::sync::OnceLock<BandwidthTiers> = std::sync::OnceLock::new();
        *CACHED.get_or_init(Self::from_env)
    }
}

/// Explicit-tier [`bandwidth_threads_for_nnz`] for tests and callers that
/// already resolved tiers (no environment reads).
pub fn bandwidth_threads_for_nnz_with(
    nnz: usize,
    rt: &SolverRuntime,
    tiers: &BandwidthTiers,
) -> usize {
    if rt.bandwidth_threads >= rt.compute_threads {
        return rt.compute_threads;
    }
    if nnz < tiers.mid {
        rt.bandwidth_threads
    } else if nnz < tiers.large {
        rt.bandwidth_threads
            .saturating_add(2)
            .min(rt.compute_threads)
    } else {
        let upper = ((rt.perf_cores as f64 * tiers.large_frac) as usize).max(rt.bandwidth_threads);
        upper.min(rt.compute_threads)
    }
}

/// Detect Apple Silicon core topology via sysctl.
///
/// Returns a fully configured [`SolverRuntime`] if performance/efficiency
/// core counts can be read, or `None` to fall back to the generic path.
#[cfg(target_os = "macos")]
fn detect_apple_silicon() -> Option<SolverRuntime> {
    fn read_sysctl(name: &core::ffi::CStr) -> Option<usize> {
        let mut val: u64 = 0;
        let mut size = core::mem::size_of::<u64>();
        let ret = unsafe {
            libc_sysctlbyname(
                name.as_ptr(),
                &mut val as *mut u64 as *mut core::ffi::c_void,
                &mut size,
                core::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 { Some(val as usize) } else { None }
    }

    // hw.perflevel0 = P-cores, hw.perflevel1 = E-cores on Apple Silicon.
    let p_cores = read_sysctl(c"hw.perflevel0.physicalcpu")?;
    let e_cores = read_sysctl(c"hw.perflevel1.physicalcpu").unwrap_or(0);

    let total = p_cores + e_cores;
    // Avoid returning zero if something went wrong.
    if p_cores == 0 {
        return None;
    }

    // Use all P-cores for compute-bound work (AMG setup, factorisations).
    let compute_threads = p_cores;

    // For bandwidth-bound work (SpMV), 4 threads typically saturate the
    // memory subsystem on Apple Silicon. Allow up to 6 for very large
    // problems where the extra prefetch streams help marginally.
    let bandwidth_threads = p_cores.min(4);

    Some(SolverRuntime {
        compute_threads,
        bandwidth_threads,
        total_cores: total,
        perf_cores: p_cores,
    })
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    #[link_name = "sysctlbyname"]
    fn libc_sysctlbyname(
        name: *const core::ffi::c_char,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut core::ffi::c_void,
        newlen: usize,
    ) -> core::ffi::c_int;
}

static RUNTIME: OnceLock<SolverRuntime> = OnceLock::new();
static BW_POOL: Mutex<Option<std::sync::Arc<rayon::ThreadPool>>> = Mutex::new(None);

/// Get or initialise the global runtime configuration.
///
/// The first call runs hardware detection; subsequent calls return the
/// cached result.
pub fn runtime() -> &'static SolverRuntime {
    RUNTIME.get_or_init(SolverRuntime::detect)
}

/// Run a closure on the bandwidth-limited thread pool.
///
/// SpMV and other memory-bandwidth-bound operations should use this pool
/// instead of the global rayon pool. On Apple Silicon, the bandwidth pool
/// has ~4 threads (memory bandwidth saturation point), while the global
/// pool uses all P-cores for compute-bound work.
///
/// Worker-thread stack size for every solver Rayon pool (bandwidth pool
/// below, global pool in [`configure_rayon`]).
///
/// 8 MB covers nested `par_iter` closure frames plus the large stack frames
/// from FEM element assembly without risking overflow on deep task trees;
/// the default 2 MB Rayon stack has overflowed there in debug builds.
pub const RAYON_WORKER_STACK_SIZE: usize = 8 * 1024 * 1024;

/// When `bandwidth_threads == compute_threads` (homogeneous architecture),
/// this runs `f` directly on the global pool (no overhead).
pub fn with_bandwidth_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let rt = runtime();
    if rt.bandwidth_threads >= rt.compute_threads {
        // Homogeneous architecture — no need for a separate pool.
        return f();
    }
    let pool = bandwidth_pool();
    pool.install(f)
}

fn bandwidth_pool() -> std::sync::Arc<rayon::ThreadPool> {
    let mut guard = BW_POOL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = guard.as_ref() {
        return std::sync::Arc::clone(pool);
    }
    let rt = runtime();
    let pool = std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(rt.bandwidth_threads)
            .stack_size(RAYON_WORKER_STACK_SIZE)
            .thread_name(|idx| format!("sonium-bw-{idx}"))
            .build()
            .expect("failed to build bandwidth thread pool"),
    );
    *guard = Some(std::sync::Arc::clone(&pool));
    pool
}

/// Drop the cached bandwidth pool, if any.
///
/// The pool is process-shared and otherwise lives forever; teardown lets
/// applications and tests bound thread retention (e.g. after a phase that
/// temporarily needed the pool, or to assert rebuild behavior). Threads exit
/// once in-flight work completes. The next [`with_bandwidth_pool`] call
/// transparently rebuilds the pool.
pub fn teardown_bandwidth_pool() {
    *BW_POOL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

/// True when a bandwidth pool is currently retained.
pub fn bandwidth_pool_live() -> bool {
    BW_POOL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_some()
}

/// Run a closure on a thread pool sized appropriately for SpMV on a matrix
/// with `nnz` non-zeros.
///
/// Small problems (`nnz < 500_000`) install on the dedicated bandwidth
/// pool; larger problems run on the caller's pool (typically the global
/// compute pool) so they get all P-cores. See [`bandwidth_threads_for_nnz`].
pub fn with_bandwidth_pool_for_nnz<R: Send>(nnz: usize, f: impl FnOnce() -> R + Send) -> R {
    let target = bandwidth_threads_for_nnz(nnz);
    let rt = runtime();
    // If the caller already has at least `target` workers, just run.
    if target >= rayon::current_num_threads() {
        return f();
    }
    // Below the small-problem threshold use the narrow BW pool. Above it
    // run on the global compute pool (already wider).
    if target <= rt.bandwidth_threads {
        return with_bandwidth_pool(f);
    }
    f()
}

/// Configure rayon's global thread pool for optimal solver performance.
///
/// Call this **once** at program startup before any solver operations.
/// On Apple Silicon this limits threads to performance cores only,
/// avoiding the E-core straggler effect.
///
/// If the global thread pool has already been initialised (e.g. by another
/// library), this is a no-op and returns `false`, so double-init is visible
/// to the caller instead of silent.
///
/// Returns `true` when this call installed the global pool.
///
/// # Example
///
/// ```ignore
/// fn main() {
///     sonium_solvers::runtime::configure_rayon();
///     // ... solver work ...
/// }
/// ```
pub fn configure_rayon() -> bool {
    let rt = runtime();
    rayon::ThreadPoolBuilder::new()
        .num_threads(rt.compute_threads)
        .stack_size(RAYON_WORKER_STACK_SIZE)
        .build_global()
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_detection() {
        let rt = SolverRuntime::detect();
        assert!(rt.total_cores >= 1);
        assert!(rt.perf_cores >= 1);
        assert!(rt.compute_threads >= 1);
        assert!(rt.bandwidth_threads >= 1);
        assert!(rt.bandwidth_threads <= rt.compute_threads);
        assert!(rt.perf_cores <= rt.total_cores);
    }

    #[test]
    fn test_runtime_singleton_is_consistent() {
        let a = runtime();
        let b = runtime();
        assert_eq!(a.total_cores, b.total_cores);
        assert_eq!(a.perf_cores, b.perf_cores);
        assert_eq!(a.compute_threads, b.compute_threads);
        assert_eq!(a.bandwidth_threads, b.bandwidth_threads);
    }

    #[test]
    fn test_bandwidth_threads_capped() {
        let rt = SolverRuntime::detect();
        // bandwidth_threads must stay within the compute pool.
        assert!(rt.bandwidth_threads >= 1);
        assert!(rt.bandwidth_threads <= rt.compute_threads);
    }

    #[test]
    fn test_generic_bandwidth_threads_scales_with_machine_size() {
        // Historical 4-thread behaviour up to 32-thread machines.
        for total in [1, 2, 4, 8, 14, 16, 32] {
            assert_eq!(generic_bandwidth_threads(total), 4.min(total).max(1));
        }
        // One thread per eight logical CPUs beyond that.
        assert_eq!(generic_bandwidth_threads(64), 8);
        assert_eq!(generic_bandwidth_threads(128), 16);
        assert_eq!(generic_bandwidth_threads(256), 32);
        // Never exceeds the machine itself.
        assert!(generic_bandwidth_threads(3) <= 3);
    }

    #[test]
    fn test_configure_rayon_does_not_panic() {
        // configure_rayon should be safe to call even if the pool is
        // already initialised (e.g. by another test or rayon itself).
        let _ = configure_rayon();
    }

    #[test]
    fn test_bandwidth_pool_teardown_and_rebuild() {
        let rt = runtime();
        if rt.bandwidth_threads >= rt.compute_threads {
            // Homogeneous path retains no pool at all.
            assert!(!bandwidth_pool_live());
            return;
        }
        with_bandwidth_pool(|| {});
        assert!(bandwidth_pool_live());
        teardown_bandwidth_pool();
        assert!(!bandwidth_pool_live());
        let value = with_bandwidth_pool(|| 42);
        assert_eq!(value, 42);
        assert!(bandwidth_pool_live());
        teardown_bandwidth_pool();
        assert!(!bandwidth_pool_live());
    }

    #[test]
    fn test_configure_rayon_reports_double_init() {
        // The global pool installs at most once per process, so the second
        // of two back-to-back calls must report `false` regardless of which
        // test installed it first. The shared stack-size constant pins the
        // documented 8 MB justification to a single definition.
        assert_eq!(RAYON_WORKER_STACK_SIZE, 8 * 1024 * 1024);
        let _ = configure_rayon();
        assert!(!configure_rayon());
    }

    fn tier_test_runtime() -> SolverRuntime {
        SolverRuntime {
            compute_threads: 10,
            bandwidth_threads: 4,
            total_cores: 14,
            perf_cores: 10,
        }
    }

    #[test]
    fn test_bandwidth_tiers_default_boundaries() {
        let rt = tier_test_runtime();
        let tiers = BandwidthTiers::defaults();
        assert_eq!(bandwidth_threads_for_nnz_with(100_000, &rt, &tiers), 4);
        assert_eq!(bandwidth_threads_for_nnz_with(1_000_000, &rt, &tiers), 6);
        // 0.75 * 10 = 7 (truncated), capped by compute_threads.
        assert_eq!(bandwidth_threads_for_nnz_with(5_000_000, &rt, &tiers), 7);
    }

    #[test]
    fn test_bandwidth_tiers_custom_boundaries() {
        let rt = tier_test_runtime();
        // A many-channel server retunes all three knobs: higher tiers and
        // full perf-core usage at the top.
        let tiers = BandwidthTiers {
            mid: 2_000_000,
            large: 8_000_000,
            large_frac: 1.0,
        };
        assert_eq!(bandwidth_threads_for_nnz_with(1_000_000, &rt, &tiers), 4);
        assert_eq!(bandwidth_threads_for_nnz_with(5_000_000, &rt, &tiers), 6);
        assert_eq!(bandwidth_threads_for_nnz_with(50_000_000, &rt, &tiers), 10);
    }

    #[test]
    fn test_bandwidth_tiers_homogeneous_passthrough() {
        let rt = SolverRuntime {
            compute_threads: 4,
            bandwidth_threads: 4,
            total_cores: 4,
            perf_cores: 4,
        };
        let tiers = BandwidthTiers::defaults();
        assert_eq!(bandwidth_threads_for_nnz_with(50_000_000, &rt, &tiers), 4);
    }
}
