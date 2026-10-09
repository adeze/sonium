//! Machine-adaptive rayon gates for propeller evaluation loops.
//!
//! A fixed item count cannot be right on every machine: the point where
//! pool dispatch pays off moves with thread count. Every parallel gate in
//! this crate therefore resolves through [`use_propeller_parallel`]:
//!
//! - default: `work_items >= items_per_thread * threads`, where `threads`
//!   is [`std::thread::available_parallelism`]. Single-threaded processes
//!   always run serially;
//! - `SONIUM_PROPELLER_PAR_THRESHOLD=<n>`: absolute work-item override for
//!   tuning, benches, and forced-path tests;
//! - `SONIUM_SERIAL=1`: process-wide serial escape hatch for deterministic
//!   QA runs.
//!
//! All gated loops collect into index order before any reduction, so the
//! floating-point summation order — and therefore checksummed artifacts —
//! is identical on the serial and parallel paths.
//!
//! The per-thread grains are starting points, not calibrated values: tune
//! them with benches on the target machine.

/// Output panels per thread above which blade replication parallelizes.
/// One rotation is nanoseconds of arithmetic, so the grain is large; the
/// usual 2–6 blade counts with modest panels stay serial.
pub(crate) const REPLICATED_PANELS_PER_THREAD: usize = 4096;

/// Harmonics per thread above which spectrum loops parallelize. Each
/// harmonic evaluates a handful of transcendental band functions.
pub(crate) const HARMONICS_PER_THREAD: usize = 64;

/// Radial-by-azimuth sample points per thread above which radial-row
/// integration parallelizes. Each row runs an inner azimuth loop of
/// transcendental relief evaluations.
pub(crate) const INTEGRATION_POINTS_PER_THREAD: usize = 1024;

/// Machine-adaptive parallel gate. See the module docs.
pub(crate) fn use_propeller_parallel(work_items: usize, items_per_thread: usize) -> bool {
    if serial_override() {
        return false;
    }
    if let Some(threshold) = env_threshold("SONIUM_PROPELLER_PAR_THRESHOLD") {
        return work_items >= threshold;
    }
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    if threads < 2 {
        return false;
    }
    work_items >= items_per_thread.saturating_mul(threads).max(1)
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
    fn serial_hatch_forces_serial() {
        // Hermetic single test: process env is shared by all test threads.
        // SAFETY: only path selection reads these keys, and both paths run
        // the same evaluations in the same order, so a concurrent test can
        // at most take the other (correct) path.
        unsafe {
            let prev_threshold = std::env::var_os("SONIUM_PROPELLER_PAR_THRESHOLD");
            let prev_serial = std::env::var_os("SONIUM_SERIAL");
            std::env::remove_var("SONIUM_PROPELLER_PAR_THRESHOLD");
            std::env::remove_var("SONIUM_SERIAL");

            let threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            if threads >= 2 {
                assert!(!use_propeller_parallel(1, HARMONICS_PER_THREAD));
                assert!(use_propeller_parallel(
                    threads * HARMONICS_PER_THREAD,
                    HARMONICS_PER_THREAD
                ));
                assert!(!use_propeller_parallel(
                    threads * HARMONICS_PER_THREAD - 1,
                    HARMONICS_PER_THREAD
                ));
            }
            std::env::set_var("SONIUM_SERIAL", "1");
            assert!(!use_propeller_parallel(usize::MAX, 1));
            std::env::remove_var("SONIUM_SERIAL");
            std::env::set_var("SONIUM_PROPELLER_PAR_THRESHOLD", "1");
            assert!(use_propeller_parallel(1, usize::MAX));

            match prev_threshold {
                Some(value) => std::env::set_var("SONIUM_PROPELLER_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_PROPELLER_PAR_THRESHOLD"),
            }
            match prev_serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    }
}
