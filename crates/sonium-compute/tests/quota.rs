//! Integration tests for [`MemoryPool`]'s strict quota.

use sonium_compute::{CpuBackend, MemoryPool, MemoryPoolConfig, OomError};

#[test]
fn allocates_below_cap() {
    // Cap = 1 KiB; allocate 128 f64 = 1024 B (right at the cap).
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1024)).unwrap();
    let _b = pool.try_alloc::<f64>(128).unwrap();
    assert_eq!(pool.bytes_in_use(), 1024);
    assert_eq!(pool.cap_bytes(), 1024);
}

#[test]
fn rejects_over_cap() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1024)).unwrap();
    let err = pool.try_alloc::<f64>(129).unwrap_err();
    assert!(
        matches!(
            err,
            OomError::QuotaExceeded {
                requested: 1032,
                in_use: 0,
                cap: 1024
            }
        ),
        "got {err:?}"
    );
    // No bytes should be reserved on failure.
    assert_eq!(pool.bytes_in_use(), 0);
}

#[test]
fn drop_releases_bytes() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1024)).unwrap();
    {
        let _b = pool.try_alloc::<f64>(128).unwrap();
        assert_eq!(pool.bytes_in_use(), 1024);
    }
    assert_eq!(pool.bytes_in_use(), 0);
    // High-water mark survives the drop.
    assert_eq!(pool.high_water_bytes(), 1024);
    // Now we can allocate again.
    let _b = pool.try_alloc::<f64>(128).unwrap();
}

#[test]
fn cumulative_allocations_respect_cap() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1024)).unwrap();
    let _a = pool.try_alloc::<f64>(64).unwrap(); // 512 B
    let _b = pool.try_alloc::<f64>(64).unwrap(); // 1024 B total
    let err = pool.try_alloc::<f64>(1).unwrap_err();
    assert!(matches!(err, OomError::QuotaExceeded { .. }));
}

#[test]
fn uninitialized_alloc_uses_quota_and_can_be_initialized() {
    use sonium_compute::Backend;

    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1024)).unwrap();
    let mut b = unsafe { pool.try_alloc_uninitialized::<f64>(4).unwrap() };
    assert_eq!(pool.bytes_in_use(), 32);

    pool.backend()
        .copy_h2d(&[1.0_f64, 2.0, 3.0, 4.0], b.raw_mut());
    let mut out = [0.0_f64; 4];
    pool.backend().copy_d2h(b.raw(), &mut out);
    assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
}

// Env-var-driven scenarios are serialised in a single test because cargo
// runs tests in parallel by default and SONIUM_GPU_MAX_BYTES is process-wide.
#[test]
fn cap_resolution_order() {
    let prev = std::env::var("SONIUM_GPU_MAX_BYTES").ok();

    // 1. Explicit cap wins over env var.
    unsafe { std::env::set_var("SONIUM_GPU_MAX_BYTES", "16") };
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(8192)).unwrap();
    assert_eq!(pool.cap_bytes(), 8192);

    // 2. Env var is used when no explicit cap is provided.
    unsafe { std::env::set_var("SONIUM_GPU_MAX_BYTES", "2KiB") };
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::default()).unwrap();
    assert_eq!(pool.cap_bytes(), 2048);

    // 3. Without explicit cap, env, or backend-reported total memory we error
    //    out — CPU's total_device_memory_bytes() is None on purpose.
    unsafe { std::env::remove_var("SONIUM_GPU_MAX_BYTES") };
    let r = MemoryPool::new(CpuBackend, MemoryPoolConfig::default());
    assert!(matches!(r, Err(OomError::Invalid(_))), "got {r:?}");

    if let Some(v) = prev {
        unsafe { std::env::set_var("SONIUM_GPU_MAX_BYTES", v) };
    }
}
