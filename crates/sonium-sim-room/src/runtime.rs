/// Worker stack for every Rayon pool that runs FEM work.
///
/// Nested assembly, Krylov, and ILUT paths blow past the default ~2 MiB
/// worker stack, so the global pool ([`install_fem_rayon_pool`]) and every
/// auxiliary lane pool use this size.
pub const FEM_WORKER_STACK_BYTES: usize = 32 * 1024 * 1024;

/// Install the process-wide Rayon pool used by room FEM binaries.
///
/// `threads == 0` preserves Rayon's automatic/RAYON_NUM_THREADS resolution.
/// The enlarged stack is required by nested assembly, Krylov, and ILUT paths.
pub fn install_fem_rayon_pool(threads: usize) -> Result<usize, rayon::ThreadPoolBuildError> {
    sonium_xem::runtime::install_global_rayon_pool(threads, Some(FEM_WORKER_STACK_BYTES))
}
