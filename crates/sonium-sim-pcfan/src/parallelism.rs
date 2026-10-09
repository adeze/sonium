//! Machine-adaptive rayon gates for optimizer fan-out.
//!
//! A fixed component count cannot be right on every machine: the point
//! where pool dispatch pays off moves with thread count and per-component
//! evaluation cost. Every parallel gate in this crate therefore resolves
//! through [`use_pcfan_parallel`]:
//!
//! - default: `work_items >= items_per_thread * threads`, where `threads`
//!   is [`std::thread::available_parallelism`]. Single-threaded processes
//!   always run serially;
//! - `SONIUM_PCFAN_PAR_THRESHOLD=<n>`: absolute work-item override for
//!   tuning, benches, and forced-path tests (units are the call site's
//!   work items — FD components for [`PCFAN_FD_COMPONENTS_PER_THREAD`],
//!   experiments for [`PCFAN_EXPERIMENTS_PER_THREAD`]);
//! - `SONIUM_SERIAL=1`: process-wide serial escape hatch for deterministic
//!   QA runs.
//!
//! The per-thread grains are starting points, not calibrated values: tune
//! them with benches on the target machine.

/// Finite-difference components per thread above which FD gradient batches
/// parallelize. Each component runs two full objective evaluations
/// (millisecond-scale reduced-order solves), so the grain is small; tiny
/// design vectors stay serial.
pub(crate) const PCFAN_FD_COMPONENTS_PER_THREAD: usize = 4;

/// Experiments per thread above which multi-DE experiment fan-out
/// parallelizes. Each experiment runs a full DE optimisation (seconds), so
/// the grain is 1: two or more experiments parallelize on any
/// multi-threaded machine, while a single experiment runs in-process with
/// no thread or pool spawn.
pub(crate) const PCFAN_EXPERIMENTS_PER_THREAD: usize = 1;

/// Machine-adaptive parallel gate. See the module docs.
pub(crate) fn use_pcfan_parallel(work_items: usize, items_per_thread: usize) -> bool {
    if serial_override() {
        return false;
    }
    if let Some(threshold) = env_threshold("SONIUM_PCFAN_PAR_THRESHOLD") {
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
        // the same evaluations, so a concurrent test can at most take the
        // other (correct) path.
        unsafe {
            let prev_threshold = std::env::var_os("SONIUM_PCFAN_PAR_THRESHOLD");
            let prev_serial = std::env::var_os("SONIUM_SERIAL");
            std::env::remove_var("SONIUM_PCFAN_PAR_THRESHOLD");
            std::env::remove_var("SONIUM_SERIAL");

            let threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            if threads >= 2 {
                assert!(!use_pcfan_parallel(1, PCFAN_FD_COMPONENTS_PER_THREAD));
                assert!(use_pcfan_parallel(
                    threads * PCFAN_FD_COMPONENTS_PER_THREAD,
                    PCFAN_FD_COMPONENTS_PER_THREAD
                ));
                assert!(!use_pcfan_parallel(
                    threads * PCFAN_FD_COMPONENTS_PER_THREAD - 1,
                    PCFAN_FD_COMPONENTS_PER_THREAD
                ));
            }
            std::env::set_var("SONIUM_SERIAL", "1");
            assert!(!use_pcfan_parallel(usize::MAX, 1));
            std::env::remove_var("SONIUM_SERIAL");
            std::env::set_var("SONIUM_PCFAN_PAR_THRESHOLD", "1");
            assert!(use_pcfan_parallel(1, usize::MAX));

            match prev_threshold {
                Some(value) => std::env::set_var("SONIUM_PCFAN_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_PCFAN_PAR_THRESHOLD"),
            }
            match prev_serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    }
}
