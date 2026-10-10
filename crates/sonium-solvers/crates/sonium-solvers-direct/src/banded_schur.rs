//! Shared Schur-complement update for blocked dense LU factorization.
//!
//! The rank-`k` update (`A22 -= L21 * U12`) reads both blocks strided in
//! place from the factor and lets the GEMM kernel pack internally — the
//! explicit u12/l21 pack copies were deleted as dead weight (see
//! [`schur_update_strided`]). Also home to the `SendConstPtr`/`SendMutPtr`
//! raw-pointer wrappers shared with the Cholesky band code.

use sonium_solvers_core::traits::ComplexField;

/// Pointer wrappers for the parallel Schur-update scope: tasks own disjoint
/// row bands of the trailing matrix, while the packed panel inputs are
/// read-only and shared.
#[derive(Clone, Copy)]
pub(crate) struct SendConstPtr<T>(*const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
impl<T> SendConstPtr<T> {
    /// Wrap a raw pointer for scoped sharing. Callers guarantee disjoint
    /// mutable access and in-bounds offsets.
    pub(crate) fn new(ptr: *const T) -> Self {
        Self(ptr)
    }
    /// Offset pointer. Callers guarantee the result stays in bounds of the
    /// same allocation. A method (not field access) so scoped closures
    /// capture the `Send` wrapper by value instead of the raw field —
    /// precise closure capture would otherwise grab `self.0` (`*const T`,
    /// never `Send`) and fail the `rayon::scope` bound.
    pub(crate) unsafe fn add(self, count: usize) -> *const T {
        unsafe { self.0.add(count) }
    }
}
#[derive(Clone, Copy)]
pub(crate) struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}
impl<T> SendMutPtr<T> {
    /// Wrap a raw pointer for scoped sharing. Callers guarantee disjoint
    /// mutable access and in-bounds offsets.
    pub(crate) fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }
    /// Offset pointer. Same capture rationale as [`SendConstPtr::add`].
    pub(crate) unsafe fn add(self, count: usize) -> *mut T {
        unsafe { self.0.add(count) }
    }
}

// History: row-band parallel Schur with packed L21/U12 plus a
// single-call packed default shipped 2026-09-26, then both were
// deleted when the strided-direct update below won the A/B by
// ~0.5ms on dgetrf-1024 (GEMM packs internally; the u12/l21 pack
// copies were dead weight). `SendConstPtr`/`SendMutPtr` stay:
// cholesky.rs imports them.

/// Strided Schur-complement update reading L21/U12 in place from the
/// factor: `A22 -= L21 * U12` with all three blocks strided (`lda = ldb =
/// ldc = n`), skipping the pack-to-contiguous copies (shipped 2026-09-26:
/// ~0.5ms faster than packed on dgetrf-1024 — the GEMM packs internally
/// per task, so the u12/l21 pack copies were dead weight).
///
/// Slices alias inside `matrix` (disjoint regions; raw-pointer precedent
/// shared with the cholesky band code using `SendMutPtr` below).
pub(crate) fn schur_update_strided<T: ComplexField>(
    matrix: &mut [T],
    n: usize,
    panel_end: usize,
    trailing: usize,
    panel_width: usize,
) {
    // Empty update: the `(len - 1)` footprint math below underflows at
    // zero, so return before forming any slice.
    if trailing == 0 || panel_width == 0 {
        return;
    }
    debug_assert!(panel_width <= panel_end);
    let panel_start = panel_end - panel_width;
    let base = matrix.as_mut_ptr();
    // SAFETY: row ranges [panel_start..panel_end) (B), [panel_end..n)
    // (A and C) partition the factor; slice lengths cover exactly the
    // strided footprints the GEMM reads/writes (see length math below).
    let a = unsafe {
        std::slice::from_raw_parts(
            base.add(panel_end * n + panel_start) as *const T,
            (trailing - 1) * n + panel_width,
        )
    };
    let b = unsafe {
        std::slice::from_raw_parts(
            base.add(panel_start * n + panel_end) as *const T,
            (panel_width - 1) * n + trailing,
        )
    };
    let c = unsafe {
        std::slice::from_raw_parts_mut(
            base.add(panel_end * n + panel_end),
            (trailing - 1) * n + trailing,
        )
    };
    T::schur_sub_strided(trailing, trailing, panel_width, a, n, b, n, c, n);
}

/// Minimum lookahead-block rows for the parallel route (see
/// [`schur_update_lookahead_block`]): 1024 keeps f64 n <= 1024 (and all
/// f32/complex types, via the discriminator) on the byte-identical
/// serial path, while including every panel the GEMM router would
/// parallelize (single-pass 3M-MAC floor: nb32 needs rows >= 2930,
/// nb48 >= 1303, nb64 >= 733 — the router re-gates below its floor
/// anyway, so sliver panels between 1024 and the MAC floor fall back
/// internally at ~1 us/panel gate cost).
pub(crate) const LOOKAHEAD_PAR_MIN_ROWS: usize = 1024;

/// `SONIUM_LU_LAPAR=0` kill-switch for the parallel lookahead-block
/// route (see [`schur_update_lookahead_block`]; diagnostic/A-B and
/// kill-switch). Read per call: only the factorization driver calls
/// this (serial context, uncontended ~50 ns), matching the per-call
/// `SONIUM_GEMM_*` gate precedent.
fn lapar_on() -> bool {
    std::env::var_os("SONIUM_LU_LAPAR").is_none_or(|v| v != "0")
}

/// Next-panel block-column update for LU lookahead: `C -= L21 * U12blk`
/// over rows `[panel_end..n)` x columns `[panel_end..panel_end +
/// next_width)`.
///
/// Route (shipped 2026-10-02): f64 blocks with at least
/// [`LOOKAHEAD_PAR_MIN_ROWS`] rows take the parallel strided kernel
/// (M-split over row bands); everything else uses the serial kernel (no
/// pool install: small blocks sit below any fan-out payoff). Rationale:
/// the serial thin-n sweep runs ~17 GF/s (~6.5 GB/s, latency-bound on
/// 32 KiB-strided A rows) while the M-split hits ~3.7x on the same
/// shape via aggregate MLP — measured same-process interleaved A/B,
/// RAYON=16: dgetrf-4096 serial-prefix 29.6 -> 20.1ms (wall -4.1%, 4/4
/// pairs), dgetrf-2048 -5.3% (6/6 pairs). f32 stays serial (the parallel
/// route loses up to +80% there — sgemm decomposition pathology on
/// 1984x64x64, unprobed mechanism; do not enable without re-measuring).
/// Bitwise identical while k fits one KC block (single k-block per task,
/// same per-element order): production k is the panel width (<= 64),
/// far below the compute f64 KC of 256. The coupling is pinned at
/// runtime by `lookahead_block_matches_serial_bitwise`, which fails
/// loudly if a compute-side KC change ever breaks the assumption.
///
/// Slices alias inside `matrix` (disjoint regions; same raw-pointer
/// precedent as [`schur_update_strided`]).
pub(crate) fn schur_update_lookahead_block<T: ComplexField>(
    matrix: &mut [T],
    n: usize,
    panel_start: usize,
    panel_end: usize,
    panel_width: usize,
    next_width: usize,
) {
    debug_assert!(panel_end <= n);
    let rows = n - panel_end;
    debug_assert!(panel_end - panel_start == panel_width);
    debug_assert!(panel_end + next_width <= n);
    let Some((a, b, c)) =
        lookahead_block_slices(matrix, n, panel_start, panel_end, panel_width, next_width)
    else {
        return;
    };
    if T::lu_prefers_parallel_lookahead_update() && lapar_on() && rows >= LOOKAHEAD_PAR_MIN_ROWS {
        T::schur_sub_strided(rows, next_width, panel_width, a, n, b, n, c, n);
        return;
    }
    T::schur_sub_serial(rows, next_width, panel_width, a, n, b, n, c, n);
}

/// Forced-parallel [`schur_update_lookahead_block`]: same slicing, always
/// the M-split kernel. Test entry point: production routing depends on
/// `SONIUM_LU_LAPAR`, so parity tests must bypass the dispatcher to stay
/// meaningful under any ambient switch state.
#[cfg(test)]
pub(crate) fn schur_update_lookahead_block_parallel<T: ComplexField>(
    matrix: &mut [T],
    n: usize,
    panel_start: usize,
    panel_end: usize,
    panel_width: usize,
    next_width: usize,
) {
    debug_assert!(panel_end <= n);
    let rows = n - panel_end;
    debug_assert!(panel_end - panel_start == panel_width);
    debug_assert!(panel_end + next_width <= n);
    let Some((a, b, c)) =
        lookahead_block_slices(matrix, n, panel_start, panel_end, panel_width, next_width)
    else {
        return;
    };
    T::schur_sub_strided(rows, next_width, panel_width, a, n, b, n, c, n);
}

/// A/B/C slices for the lookahead block update, or `None` for an empty
/// update (the `(len - 1)` footprint math underflows at zero, so empty
/// shapes return before forming any slice).
fn lookahead_block_slices<T>(
    matrix: &mut [T],
    n: usize,
    panel_start: usize,
    panel_end: usize,
    panel_width: usize,
    next_width: usize,
) -> Option<(&[T], &[T], &mut [T])> {
    let rows = n.saturating_sub(panel_end);
    if rows == 0 || next_width == 0 || panel_width == 0 {
        return None;
    }
    let base = matrix.as_mut_ptr();
    // SAFETY: A (rows below x panel cols), B (panel rows x next cols)
    // and C (rows below x next cols) are disjoint cells; slice lengths
    // cover exactly the strided footprints.
    let a = unsafe {
        std::slice::from_raw_parts(
            base.add(panel_end * n + panel_start) as *const T,
            (rows - 1) * n + panel_width,
        )
    };
    let b = unsafe {
        std::slice::from_raw_parts(
            base.add(panel_start * n + panel_end) as *const T,
            (panel_width - 1) * n + next_width,
        )
    };
    let c = unsafe {
        std::slice::from_raw_parts_mut(
            base.add(panel_end * n + panel_end),
            (rows - 1) * n + next_width,
        )
    };
    Some((a, b, c))
}

/// Strided Schur-complement update over a trailing column subset (LU
/// lookahead remainder): `C -= L21 * U12rest` with `C`/`U12rest` starting
/// at column `panel_end + col_offset` (`col_offset = 0` reproduces
/// [`schur_update_strided`]). L21 is unchanged (full rows below, panel
/// columns); only the B/C column windows shift.
///
/// Slices alias inside `matrix` (disjoint regions; same raw-pointer
/// precedent as [`schur_update_strided`]).
pub(crate) fn schur_update_strided_offset<T: ComplexField>(
    matrix: &mut [T],
    n: usize,
    panel_end: usize,
    trailing: usize,
    panel_width: usize,
    col_offset: usize,
) {
    // Empty update: the `(len - 1)` footprint math below underflows at
    // zero, so return before forming any slice.
    if trailing == 0 || panel_width == 0 {
        return;
    }
    debug_assert!(panel_width <= panel_end);
    debug_assert!(col_offset <= trailing);
    let panel_start = panel_end - panel_width;
    let rest = trailing.saturating_sub(col_offset);
    if rest == 0 {
        return;
    }
    let base = matrix.as_mut_ptr();
    // SAFETY: row ranges [panel_start..panel_end) (B), [panel_end..n)
    // (A and C) partition the factor; slice lengths cover exactly the
    // strided footprints the GEMM reads/writes (see length math below).
    let a = unsafe {
        std::slice::from_raw_parts(
            base.add(panel_end * n + panel_start) as *const T,
            (trailing - 1) * n + panel_width,
        )
    };
    let b = unsafe {
        std::slice::from_raw_parts(
            base.add(panel_start * n + panel_end + col_offset) as *const T,
            (panel_width - 1) * n + rest,
        )
    };
    let c = unsafe {
        std::slice::from_raw_parts_mut(
            base.add(panel_end * n + panel_end + col_offset),
            (trailing - 1) * n + rest,
        )
    };
    T::schur_sub_strided(trailing, rest, panel_width, a, n, b, n, c, n);
}
