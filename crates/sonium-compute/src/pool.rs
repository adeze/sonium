//! Memory pool with strict device-memory quota.
//!
//! Every device buffer in sonium goes through this pool. The pool reserves
//! bytes from an atomic counter *before* delegating to
//! [`crate::Backend::raw_alloc`]; if the new total would exceed the
//! configured cap, the allocation fails fast with
//! [`OomError::QuotaExceeded`] and never touches the driver.
//!
//! The cap is configured by [`MemoryPoolConfig`]. When unset, the pool reads
//! `SONIUM_GPU_MAX_BYTES` (e.g. `"4GiB"`, `"512MiB"`, raw bytes); if that is
//! also unset, it defaults to **80 % of total device memory** as reported by
//! the backend. The chosen cap is logged once at construction.
//!
//! # Why a separate counter and not just trust the driver
//!
//! - The driver tells us about its own out-of-memory; we want to fail
//!   *before* fragmenting device memory or competing with other processes.
//! - On CPU there is no driver-level quota at all. Without this, a runaway
//!   solver can swap the host into oblivion.
//! - In multi-tenant GPU environments (CI, shared dev boxes) we want a
//!   per-process budget that is independent of total VRAM.

use std::env;
use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::backend::{Backend, DeviceBufferLike};
use crate::error::OomError;
use crate::scalar::BufferElement;

/// Configuration for [`MemoryPool`] construction.
///
/// Resolution order for the byte cap:
/// 1. `Self::max_bytes` if `Some`.
/// 2. The `SONIUM_GPU_MAX_BYTES` environment variable if set.
/// 3. 80 % of `Backend::total_device_memory_bytes()` if reported.
/// 4. Error: `OomError::Invalid` — refuse to construct without a cap.
#[derive(Debug, Default, Clone)]
pub struct MemoryPoolConfig {
    /// Explicit byte cap. Overrides env and auto-detection.
    pub max_bytes: Option<usize>,
}

impl MemoryPoolConfig {
    /// Build a config with an explicit cap.
    pub fn with_cap(max_bytes: usize) -> Self {
        Self {
            max_bytes: Some(max_bytes),
        }
    }
}

/// Resolve the byte cap from explicit > env > auto-detect.
fn resolve_cap<B: Backend>(cfg: &MemoryPoolConfig, backend: &B) -> Result<usize, OomError> {
    if let Some(cap) = cfg.max_bytes {
        return Ok(cap);
    }
    if let Ok(s) = env::var("SONIUM_GPU_MAX_BYTES") {
        return parse_size(&s).ok_or_else(|| {
            OomError::Invalid(format!("could not parse SONIUM_GPU_MAX_BYTES={s:?}"))
        });
    }
    if let Some(total) = backend.total_device_memory_bytes() {
        // 80 % of total device memory, rounded down.
        return Ok((total / 10) * 8);
    }
    Err(OomError::Invalid(
        "no memory cap: set MemoryPoolConfig::max_bytes, \
         SONIUM_GPU_MAX_BYTES, or use a backend that reports total memory"
            .into(),
    ))
}

/// Parse a size like `"4GiB"`, `"512MiB"`, `"1024"` into a byte count.
fn parse_size(raw: &str) -> Option<usize> {
    let s = raw.trim();
    // Try suffixes longest-first.
    let suffixes: [(&str, usize); 8] = [
        ("GiB", 1usize << 30),
        ("MiB", 1usize << 20),
        ("KiB", 1usize << 10),
        ("GB", 1_000_000_000),
        ("MB", 1_000_000),
        ("KB", 1_000),
        ("B", 1),
        ("", 1),
    ];
    for (suf, mul) in suffixes {
        if let Some(num) = s.strip_suffix(suf) {
            let num = num.trim();
            if num.is_empty() {
                continue;
            }
            return num.parse::<usize>().ok().and_then(|n| n.checked_mul(mul));
        }
    }
    None
}

/// Memory pool over a backend. Hand out [`Buffer`] handles up to a hard cap.
///
/// Cloning a `MemoryPool` is cheap — internals are `Arc`-shared so multiple
/// crates can hold their own clone and still share the same quota counter.
pub struct MemoryPool<B: Backend> {
    inner: Arc<PoolInner<B>>,
}

impl<B: Backend> fmt::Debug for MemoryPool<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPool")
            .field("cap_bytes", &self.inner.cap)
            .field("bytes_in_use", &self.inner.in_use.load(Ordering::Relaxed))
            .field(
                "high_water_bytes",
                &self.inner.high_water.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl<T: BufferElement, B: Backend> fmt::Debug for Buffer<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("len", &self.raw().len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl<B: Backend> Clone for MemoryPool<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct PoolInner<B: Backend> {
    backend: B,
    cap: usize,
    in_use: AtomicUsize,
    high_water: AtomicUsize,
}

/// A quota reservation that is refunded during unwinding as well as on an
/// ordinary allocation failure. The reservation is committed only after the
/// backend has returned an owned buffer.
struct QuotaReservation<B: Backend> {
    pool: Arc<PoolInner<B>>,
    bytes: usize,
    committed: bool,
}

impl<B: Backend> QuotaReservation<B> {
    fn new(pool: Arc<PoolInner<B>>, bytes: usize) -> Self {
        Self {
            pool,
            bytes,
            committed: false,
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl<B: Backend> Drop for QuotaReservation<B> {
    fn drop(&mut self) {
        if !self.committed {
            self.pool.in_use.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

impl<B: Backend> MemoryPool<B> {
    /// Construct a pool, resolving the cap per [`MemoryPoolConfig`] rules.
    ///
    /// Logs the chosen cap and source at `info` level.
    pub fn new(backend: B, cfg: MemoryPoolConfig) -> Result<Self, OomError> {
        let cap = resolve_cap(&cfg, &backend)?;
        // Keep backend-owned workspaces and deferred scratch under the same
        // configured device-memory ceiling as caller-visible pool buffers.
        backend.set_allocation_cap_bytes(cap);
        log::info!(
            target: "sonium_compute::pool",
            "MemoryPool initialised: cap = {cap} B (~{:.1} MiB)",
            cap as f64 / (1024.0 * 1024.0),
        );
        Ok(Self {
            inner: Arc::new(PoolInner {
                backend,
                cap,
                in_use: AtomicUsize::new(0),
                high_water: AtomicUsize::new(0),
            }),
        })
    }

    /// Hard byte cap configured for this pool.
    pub fn cap_bytes(&self) -> usize {
        self.inner.cap
    }

    /// Bytes currently held by outstanding buffers.
    pub fn bytes_in_use(&self) -> usize {
        self.inner.in_use.load(Ordering::Acquire)
    }

    /// Largest `bytes_in_use` ever observed.
    pub fn high_water_bytes(&self) -> usize {
        self.inner.high_water.load(Ordering::Acquire)
    }

    /// Borrow the underlying backend (e.g. for BLAS / SpMV calls).
    pub fn backend(&self) -> &B {
        &self.inner.backend
    }

    /// Allocate a buffer of `len` elements, enforcing the quota first.
    pub fn try_alloc<T: BufferElement>(&self, len: usize) -> Result<Buffer<T, B>, OomError> {
        self.try_alloc_with(len, |backend, len| backend.raw_alloc::<T>(len))
    }

    /// Allocate scratch storage that may contain uninitialized values.
    ///
    /// # Safety
    ///
    /// The returned buffer must be fully initialized before any backend
    /// operation or host copy can read from it. Use [`Self::try_alloc`] when a
    /// zeroed buffer is required.
    pub unsafe fn try_alloc_uninitialized<T: BufferElement>(
        &self,
        len: usize,
    ) -> Result<Buffer<T, B>, OomError> {
        self.try_alloc_with(len, |backend, len| {
            // SAFETY: forwarded from this method's caller.
            unsafe { backend.raw_alloc_uninitialized::<T>(len) }
        })
    }

    fn try_alloc_with<T, F>(&self, len: usize, alloc: F) -> Result<Buffer<T, B>, OomError>
    where
        T: BufferElement,
        F: FnOnce(&B, usize) -> Result<B::Buffer<T>, OomError>,
    {
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            OomError::Invalid(format!(
                "allocation size overflow: len={len}, elem_size={}",
                size_of::<T>()
            ))
        })?;

        // Reserve bytes optimistically with a CAS loop so that the high-water
        // mark stays accurate even under concurrent allocations.
        let mut prev = self.inner.in_use.load(Ordering::Relaxed);
        loop {
            let next = prev.checked_add(bytes).ok_or_else(|| {
                OomError::Invalid(format!(
                    "in_use counter overflow: prev={prev}, requested={bytes}"
                ))
            })?;
            if next > self.inner.cap {
                return Err(OomError::QuotaExceeded {
                    requested: bytes,
                    in_use: prev,
                    cap: self.inner.cap,
                });
            }
            match self.inner.in_use.compare_exchange_weak(
                prev,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Update high-water; not load-bearing for correctness so
                    // a relaxed store-max via fetch_max suffices.
                    self.inner.high_water.fetch_max(next, Ordering::Relaxed);
                    break;
                }
                Err(observed) => prev = observed,
            }
        }

        // Quota reserved — call the driver. The guard also refunds if the
        // backend allocation callback unwinds instead of returning an error.
        let reservation = QuotaReservation::new(Arc::clone(&self.inner), bytes);
        let allocation = catch_unwind(AssertUnwindSafe(|| alloc(&self.inner.backend, len)))
            .map_err(|payload| {
                let reason = if let Some(message) = payload.downcast_ref::<&'static str>() {
                    (*message).to_string()
                } else if let Some(message) = payload.downcast_ref::<String>() {
                    message.clone()
                } else {
                    "non-string panic payload".to_string()
                };
                OomError::DriverOom(format!("backend allocation panicked: {reason}"))
            })?;

        match allocation {
            Ok(buf) => {
                // Backend allocation is a lower-level contract: a successful
                // callback must return exactly the requested element count.  Do
                // not commit quota for a short/long buffer, otherwise every
                // subsequent operation sees inconsistent shape metadata while
                // the pool accounts the wrong number of bytes.
                if buf.len() != len {
                    return Err(OomError::Invalid(format!(
                        "backend returned buffer length {}, requested {len}",
                        buf.len()
                    )));
                }
                reservation.commit();
                Ok(Buffer {
                    buf: Some(buf),
                    bytes,
                    pool: Arc::clone(&self.inner),
                    _phantom: PhantomData,
                })
            }
            Err(e) => Err(e),
        }
    }
}

/// RAII wrapper around a backend buffer. Drop refunds the pool quota.
pub struct Buffer<T: BufferElement, B: Backend> {
    // Wrapped in Option only so Drop can take ownership.
    buf: Option<B::Buffer<T>>,
    bytes: usize,
    pool: Arc<PoolInner<B>>,
    _phantom: PhantomData<T>,
}

impl<T: BufferElement, B: Backend> Buffer<T, B> {
    /// Borrow the underlying backend buffer.
    pub fn raw(&self) -> &B::Buffer<T> {
        self.buf.as_ref().expect("buffer accessed after drop")
    }

    /// Mutably borrow the underlying backend buffer.
    pub fn raw_mut(&mut self) -> &mut B::Buffer<T> {
        self.buf.as_mut().expect("buffer accessed after drop")
    }

    /// Length in elements.
    pub fn len(&self) -> usize {
        self.raw().len()
    }

    /// Empty check.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Byte size held by this buffer.
    pub fn byte_size(&self) -> usize {
        self.bytes
    }
}

impl<T: BufferElement, B: Backend> Drop for Buffer<T, B> {
    fn drop(&mut self) {
        // Drop the underlying buffer first so the driver-side free completes
        // before we credit the pool counter; otherwise a peak right after
        // could see negative headroom.
        drop(self.buf.take());
        self.pool.in_use.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    #[test]
    fn parse_size_handles_suffixes() {
        assert_eq!(parse_size("1024"), Some(1024));
        assert_eq!(parse_size("4GiB"), Some(4 * (1 << 30)));
        assert_eq!(parse_size("512MiB"), Some(512 * (1 << 20)));
        assert_eq!(parse_size("2MB"), Some(2_000_000));
        assert_eq!(parse_size(" 1KiB "), Some(1024));
        assert_eq!(parse_size("nope"), None);
        assert_eq!(parse_size(&format!("{}GiB", usize::MAX)), None);
    }

    #[test]
    fn quota_exhaustion_is_rejected_before_allocation_and_drop_refunds() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(16))
            .expect("explicit CPU quota");
        let buffer = pool.try_alloc::<u32>(4).expect("allocation within quota");
        assert_eq!(pool.bytes_in_use(), 16);
        assert_eq!(pool.high_water_bytes(), 16);

        let error = pool
            .try_alloc::<u32>(1)
            .expect_err("quota must reject before backend allocation");
        assert!(matches!(error, OomError::QuotaExceeded { .. }));
        assert_eq!(pool.bytes_in_use(), 16);

        drop(buffer);
        assert_eq!(pool.bytes_in_use(), 0);
        // The high-water mark is historical and must not be refunded.
        assert_eq!(pool.high_water_bytes(), 16);
        pool.try_alloc::<u32>(4)
            .expect("released quota must be reusable");
    }

    #[test]
    fn driver_allocation_failure_refunds_reserved_quota() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(16))
            .expect("explicit CPU quota");

        let error = pool
            .try_alloc_with::<u32, _>(4, |_backend, _len| {
                Err(OomError::DriverOom("injected allocation failure".into()))
            })
            .expect_err("injected driver allocation failure must be returned");

        assert!(matches!(error, OomError::DriverOom(_)));
        assert_eq!(pool.bytes_in_use(), 0);
        pool.try_alloc::<u32>(4)
            .expect("a failed driver allocation must leave quota retryable");
    }

    #[test]
    fn backend_short_allocation_refunds_quota_and_fails_closed() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(16))
            .expect("explicit CPU quota");

        let error = pool
            .try_alloc_with::<u32, _>(4, |backend, _| backend.raw_alloc::<u32>(3))
            .expect_err("a backend shape mismatch must not become a successful buffer");

        assert!(matches!(error, OomError::Invalid(reason) if reason.contains("buffer length 3")));
        assert_eq!(pool.bytes_in_use(), 0);
        pool.try_alloc::<u32>(4)
            .expect("failed shape validation must leave quota retryable");
    }

    #[test]
    fn allocator_panic_refunds_reserved_quota() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(16))
            .expect("explicit CPU quota");

        let error = pool
            .try_alloc_with::<u32, _>(4, |_backend, _len| {
                panic!("injected allocator panic");
            })
            .expect_err("allocator panic must become a typed driver error");
        assert!(
            matches!(error, OomError::DriverOom(reason) if reason.contains("injected allocator panic"))
        );
        assert_eq!(pool.bytes_in_use(), 0);
        let buffer = pool.try_alloc::<u32>(4).expect("pool must remain reusable");
        assert_eq!(pool.bytes_in_use(), 16);
        drop(buffer);
        assert_eq!(pool.bytes_in_use(), 0);
    }
}
