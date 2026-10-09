//! LU decomposition solver
//!
//! Pure-Rust LU factorization with partial pivoting for solving dense linear systems.

use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{AccuracyStatus, ComplexField, SolverFailure, SolverStatus};
use thiserror::Error;

/// Panel width used by the right-looking blocked factorization.
///
/// A 32-column panel is large enough to amortize packing and BLAS-3 dispatch
/// while reducing scalar panel-factorization and triangular-solve work. The final
/// short panel naturally handles matrices whose order is not a multiple of
/// this value. Trialled 64 (2026-09, current structure): dgetrf-1024 slows
/// ~1.6x (13.8ms → 21.6ms; pivoted panel search/swap scales worse with nb
/// than Cholesky's POTF2), sgetrf-1024 inconclusive (11.2ms vs 12.5ms
/// across noisy rounds) — 32 stands for the shared const.
/// Trialled 48 (2026-09-26, dgetrf-2048): 53.4 -> 55.1ms med — the phase
/// split shows why (Schur 26.5 -> 24ms from fewer/bigger GEMMs, but panel
/// 11-17ms unchanged: the 786KB compact spills L2 and eats the gains).
/// Do not retry intermediate widths without fixing panel L2 residency.
/// Re-probed 2026-09-26 by flipping the const (nb=16/64/96/128 at
/// dgetrf-2048): cross-run comparison was VOID — the box frequency swung
/// ~1.55x mid-session (nb=32 measured 14.0ms then 9.0ms with no code
/// change), so every apparent nb effect was drift. Any future width trial
/// MUST interleave candidates in one process (same-process A/B); the
/// within-run vendor ratio is the only drift-robust metric, and even it
/// shifts with box state when bottlenecks differ, so repeat on a quiet box.
///
/// In-situ phase split at dgetrf-2048, 2026-09-26 (temporary env-gated
/// timers, reverted): panel 11.0 / TRSM 3.0 / Schur 24.5 / validate 1.2ms
/// of ~40ms. Schur is the majority cost center, not panels — our Schur
/// alone (~24.5ms) matches vendor's whole factorization (~25.4ms). The
/// standalone hot-data Schur sequence sums to only ~19.8ms, so ~5ms is
/// per-call fork-join/pack overhead across the 63 dispatches. Attack
/// Schur dispatch efficiency before panel kernels.
///
/// Per-phase vendor attribution, 2026-09-26 (vendorpanel/trsmsum probes,
/// fresh-copy methodology after fill-subtraction failed): vendor GETF2
/// panel sum 5.91ms vs ours 11.0 (1.86x) + dlaswp swaps 0.90ms; vendor
/// TRSM sum 1.21ms vs ours 3.03 (2.5x — standalone ties in-situ, so a
/// pure kernel gap, no interference); vendor Schur residual ~17.4ms vs
/// ours 24.5 (1.4x). Prize order: Schur ~7ms, panels ~5ms, TRSM ~1.8ms.
/// A TRSM kernel fix also helps Cholesky (shared trsm_f64).
///
/// TRSM-thin trials 2026-09-26 (trsumsum + scopecost probes): serial sum
/// 5.20ms vs parallel-16 2.93 (only 1.78x — overhead-dominated), and raw
/// scope+16-empty-spawns ×63 = 2.76ms ≈ the whole parallel solve, so the
/// solve is ~95% dispatch. No-pack bypass falsified (3.06->3.06/3.23 —
/// the 4KB-conflict pack pays for itself); 4-task cap falsified
/// (3.06->3.9/4.05 — stragglers/underfill dominate). Next idea (untried):
/// explicit L11 inverse + GEMM instead of substitution (2x flops at GEMM
/// rates, no packs; needs residual proof — numerics change).
/// FALSIFIED 2026-09-26 (invtrsm probe): 3.01ms vs 3.03 substitution —
/// same 63 GEMM scopes, same floor; residual 1.15e-16 (numerics fine,
/// no gain). Affinity pinning also neutral on dispatch (scopecost
/// 2.86->2.58ms sum, med identical). Mirror swaps verified optimal
/// (one full-row swap per pivot). TRSM-thin closed: the scope/spawn
/// substrate is the floor; vendor's edge is lighter dispatch.
///
/// Panel decomposition (estimates, unmeasured): gather/overlay ~1.7ms +
/// mirror ~1.6ms + compact GETF2 ~7.7ms vs vendor dgetf2 5.91 + dlaswp
/// 0.9. Suspect: per-column ger dispatch (~2000 dispatches) inside the
/// flat recursion leaves. Next: temp-timer the panel internals
/// (gather/GETF2-flat/GEMM-update/mirror/overlay) same discipline.
///
/// Correction 2026-09-27: the "do not retry 48" verdict above was
/// pre-lookahead (serial-panel cost dominated). Post-lookahead panels
/// hide behind Schur, and interleaved A/B now shows 48 beating 32
/// ~5% on f64-2048 (see the driver note): the threaded f64 default is
/// [`LU_BLOCK_SIZE_F64`]. The shared 32 stands for f32/complex (f32
/// mixed: wins 2048, marginally loses 1024).
///
/// Merge 2026-09-28: main shipped 64 here (see the driver flip note);
/// kept 32 + [`LU_BLOCK_SIZE_F64`] — nb must be re-adjudicated on the
/// merged kernels (TRSM/AVX2/complex3m change the panel/Schur trade).
const LU_BLOCK_SIZE: usize = 32;

/// Threaded f64 panel width (see [`LU_BLOCK_SIZE`] for the shared base).
///
/// f64-only: `nb=48` wins `nb=32` ~1.05x at dgetrf-2048 across four
/// interleaved windows (medians + pair counts, RAYON=16, funnel on),
/// ties at 1024 (5/5), wins at 512 (6/8); `nb=64` loses everywhere and
/// `nb in {16,24}` lose to dispatch cost. f32 takes the 64-wide band
/// at 2048 <= n < 2304 (see [`lu_f32_block_size`]), 32 elsewhere.
/// Full adjudication record lives on the driver `block_size` selection.
/// Applies below [`LU_NARROW_PANEL_THRESHOLD`]; larger orders narrow
/// to the shared 32 (see [`lu_f64_block_size`]).
const LU_BLOCK_SIZE_F64: usize = 48;

/// Order at or above which threaded f64 LU narrows from the 48-wide
/// default to the shared 32-wide panel.
///
/// Adjudicated 2026-10-01 (`luphases` interleaved A/B, RAYON=16,
/// order-rotated rounds, one process per size): nb48 wins 8/8 pairs at
/// 2560 (1.07x) and 2816 (1.08x) plus 5/5 at 2048; nb32 leads 7/8 (+1
/// tie, ~1.5%) at 3072, wins 6/6 at 3584 (1.10x) and 11/12 at 4096
/// (~1.1x, quiet and contended batches agree). The win sits in the
/// Schur phase (thinner-k packs stream better once the trailing block
/// exceeds cache; the lookahead-serial update also halves). Do not move
/// without re-running the `luphases <n> <reps> 32 48` boundary sweep.
const LU_NARROW_PANEL_THRESHOLD: usize = 3072;

/// Threaded f64 panel width by order: the 48-wide optimum below
/// [`LU_NARROW_PANEL_THRESHOLD`], the shared 32-wide panel at or above.
fn lu_f64_block_size(n: usize) -> usize {
    if n >= LU_NARROW_PANEL_THRESHOLD {
        LU_BLOCK_SIZE
    } else {
        LU_BLOCK_SIZE_F64
    }
}

/// Threaded f32 wide panel width (see [`LU_BLOCK_SIZE`] for the shared
/// base): 64-wide halves the panel/dispatch count (64 -> 32 panels at
/// 2048) while the f32 panel stays L2-resident (2048 x 64 x 4B =
/// 0.5MB). Adjudicated 2026-10-02 (same-process interleaved A/B,
/// RAYON=16): wins 9/12 pairs over two windows at 2048 (med 26.5/26.8
/// vs 29.2/29.4, ~1.09x); nb96/128 lose to L2 spill; nb64 loses at
/// 1024 (6/8 pairs for 32), ties-then-loses at 2304 (2-2 pairs,
/// median leans 32) and loses badly at 2560/4096 — hence the
/// two-sided [`lu_f32_block_size`] band, not an open floor.
const LU_BLOCK_SIZE_F32_WIDE: usize = 64;

/// Threaded f32 panel width by order: the 64-wide optimum inside
/// 2048 <= n < 2304, the shared 32-wide panel elsewhere (see
/// [`LU_BLOCK_SIZE_F32_WIDE` for the adjudication record).
fn lu_f32_block_size(n: usize) -> usize {
    if (2048..2304).contains(&n) {
        LU_BLOCK_SIZE_F32_WIDE
    } else {
        LU_BLOCK_SIZE
    }
}

/// Trailing rows at or above which the panel uses the gather-once compact
/// path: below this the in-place sweep is already cache-resident and the
/// gather/overlay round-trip would be pure overhead. Re-trialled 2026-09
/// under current kernels (sizer rule, floor, 6-wide all postdate the old
/// 512 verdict) via a temporary env hook (removed): dgetrf-1024 steady
/// 512: 16.25-16.26, 33: 15.1-15.9, 128: 15.4-15.6, 9: 15.7-16.0,
/// 2000 (all in-place): 19.9 (each x2 interleaved, OB 5.2-5.4); n=512:
/// 33 ties 128 (4.0-4.1). The old mixed-512 verdict flipped — compact
/// fixed costs dropped enough that all-but-last-panel compact wins;
/// all-in-place still loses badly. 33 serves the L1-resident tail.
///
/// Size-cap trialled twice 2026-09-26 (compact panel overflows L2 at
/// n=2048): capping compact at 1024 rows with in-place fallback LOST
/// decisively both times (dgetrf-2048 65 -> 81-83, 12/12 pairs), the
/// second with the in-place search upgraded to the SIMD kernel — so the
/// gap is structural (full-row-stride streaming), not kernel parity. The
/// in-place search upgrade was reverted with the cap (unproven there;
/// the stride-n per-column verdict stands). Do not retry the cap.
const COMPACT_PANEL_THRESHOLD: usize = 33;

/// Errors that can occur during LU factorization
#[derive(Error, Debug)]
pub enum LuError {
    #[error("Matrix is singular or nearly singular")]
    SingularMatrix,
    #[error("Matrix dimensions mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Matrix contains a non-finite value at storage index {index}")]
    NonFiniteMatrix { index: usize },
    #[error("Right-hand side contains a non-finite value at index {index}")]
    NonFiniteRhs { index: usize },
    #[error("Stored LU factorization is invalid: {reason}")]
    InvalidFactorization { reason: &'static str },
    #[error("Multi-RHS shape overflows usize (dimension={dimension}, nrhs={nrhs})")]
    SizeOverflow { dimension: usize, nrhs: usize },
    #[error("LU numerical breakdown during {stage} at index {index}")]
    NumericalBreakdown { stage: &'static str, index: usize },
}

impl LuError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::SingularMatrix
            | Self::InvalidFactorization { .. }
            | Self::NumericalBreakdown { .. } => SolverStatus::Breakdown,
            _ => SolverStatus::NumericalError,
        }
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverFailure for LuError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::DimensionMismatch { .. } | Self::SizeOverflow { .. } => {
                SolverErrorKind::DimensionMismatch
            }
            Self::SingularMatrix
            | Self::InvalidFactorization { .. }
            | Self::NumericalBreakdown { .. } => SolverErrorKind::Breakdown,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        LuError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        LuError::accuracy_status(self)
    }
}

/// Reusable scratch storage for blocked dense LU factorization.
///
/// Keeping this workspace across factorizations removes the per-panel
/// temporary allocations. The Schur update runs in place (`C -= L21 * U12`
/// with `beta = 1`), so no `trailing × trailing` scratch is retained.
#[derive(Debug, Clone)]
pub struct LuWorkspace<T: ComplexField> {
    l11_pack: Vec<T>,
    panel_kseg: Vec<T>,
}

impl<T: ComplexField> Default for LuWorkspace<T> {
    fn default() -> Self {
        Self {
            l11_pack: Vec::new(),
            panel_kseg: Vec::new(),
        }
    }
}

impl<T: ComplexField> LuWorkspace<T> {
    /// Create an empty workspace. Storage grows on the first factorization and
    /// is retained for subsequent matrices.
    pub fn new() -> Self {
        Self::default()
    }
}

/// LU factorization result
///
/// Stores L and U factors along with pivot information
#[derive(Debug, Clone)]
pub struct LuFactorization<T: ComplexField> {
    /// Combined L and U matrices (L is unit lower triangular, stored below diagonal)
    pub lu: DenseMatrix<T>,
    /// Pivot indices
    pub pivots: Vec<usize>,
    /// Matrix dimension
    pub n: usize,
}

/// Recycled factor buffers keyed by element type: dropping a
/// factorization returns its `n²` storage (when large enough to matter),
/// and the next same-size `lu_factorize` checks a warm buffer out instead
/// of mmap+faulting fresh pages (~0.8ms saved on an 8MB factor; the
/// `memcpy` itself is unavoidable). Only the repeated-same-size pattern
/// (benchmarks, frequency sweeps) hits; one-shots malloc exactly as
/// before. Bounded (two buffers per type) so idle memory stays small, and
/// pooling computes nothing — results are identical either way. Fully
/// generic over `T: ComplexField` (the `Any` map keeps each type's
/// buffers separate), so no public signature changes.
static FACTOR_POOL: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<std::any::TypeId, Box<dyn std::any::Any + Send>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

#[cfg(test)]
static FACTOR_POOL_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Check out a buffer with exactly `len` capacity (`None` mallocs as before).
fn factor_pool_pop<T: ComplexField>(len: usize) -> Option<Vec<T>> {
    let mut pool = FACTOR_POOL.lock().ok()?;
    let slot = pool
        .get_mut(&std::any::TypeId::of::<T>())?
        .downcast_mut::<Vec<Vec<T>>>()?;
    let pos = slot.iter().position(|v| v.capacity() == len)?;
    #[cfg(test)]
    FACTOR_POOL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(slot.remove(pos))
}

/// Return a buffer; drops it when small (the page cache recycles those
/// cheaply) or the pool is full.
fn factor_pool_push<T: ComplexField>(mut v: Vec<T>) {
    // Below 1MB the page cache recycles cheaply; pooling only pays above
    // the mmap/fault regime.
    if v.capacity() * std::mem::size_of::<T>() < (1 << 20) {
        return;
    }
    v.clear();
    if let Ok(mut pool) = FACTOR_POOL.lock() {
        let slot = pool
            .entry(std::any::TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<Vec<T>>::new()));
        if let Some(slot) = slot.downcast_mut::<Vec<Vec<T>>>()
            && slot.len() < 2
        {
            slot.push(v);
        }
    }
}

/// Clone a matrix into pooled (warm-page) storage when available.
fn clone_into_factor_buffer<T: ComplexField>(a: &DenseMatrix<T>) -> DenseMatrix<T> {
    let src = a.as_slice();
    let mut buf = factor_pool_pop(src.len()).unwrap_or_else(|| Vec::with_capacity(src.len()));
    buf.extend_from_slice(src);
    DenseMatrix::from_row_major_vec(a.nrows(), a.ncols(), buf)
}

impl<T: ComplexField> Drop for LuFactorization<T> {
    fn drop(&mut self) {
        factor_pool_push(self.lu.take_data());
    }
}

impl<T: ComplexField> LuFactorization<T> {
    /// Solve Ax = b using the pre-computed LU factorization
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LuError> {
        self.check_rhs_lengths(b.len(), self.n)?;
        self.validate_factor_shape()?;
        let mut x = vec![T::zero(); self.n];
        self.solve_into(b, &mut x)?;
        Ok(x)
    }

    /// Solve `Ax = b` into caller-owned storage without allocating.
    pub fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), LuError> {
        self.check_rhs_lengths(b.len(), x.len())?;
        self.validate_factorization()?;
        self.validate_solve_diagonals()?;
        validate_rhs(b)?;

        // Apply row permutations Pb
        for i in 0..self.n {
            x[i] = b[self.pivots[i]];
        }

        // Forward/backward substitution in dot form over contiguous
        // row-major segments so the inner products ride the SIMD
        // `vec_dot_plain` backend instead of scalar element loops.
        // Summation order within a row may differ from AXPY order in the
        // last ulp; residuals are the oracle. Shared with the `nrhs == 1`
        // multi-RHS path.
        Self::forward_dot_flat(&self.lu, self.n, x);
        Self::backward_dot_flat(&self.lu, self.n, x)?;
        Ok(())
    }

    /// Solve several right-hand sides stored as an `n × nrhs` row-major
    /// matrix and return the solutions in the same layout.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, LuError> {
        self.validate_factor_shape()?;
        let expected = self.n.checked_mul(nrhs).ok_or(LuError::SizeOverflow {
            dimension: self.n,
            nrhs,
        })?;
        if b.len() != expected {
            return Err(LuError::DimensionMismatch {
                expected,
                got: b.len(),
            });
        }
        let mut x = vec![T::zero(); expected];
        self.solve_many_into(b, &mut x, nrhs)?;
        Ok(x)
    }

    /// Allocation-free multi-right-hand-side solve.
    ///
    /// Both `b` and `x` use row-major `n × nrhs` storage, so all right-hand
    /// sides for a matrix row are contiguous.
    pub fn solve_many_into(&self, b: &[T], x: &mut [T], nrhs: usize) -> Result<(), LuError> {
        let expected = self.n.checked_mul(nrhs).ok_or(LuError::SizeOverflow {
            dimension: self.n,
            nrhs,
        })?;
        if b.len() != expected || x.len() != expected {
            return Err(LuError::DimensionMismatch {
                expected,
                got: if b.len() != expected {
                    b.len()
                } else {
                    x.len()
                },
            });
        }
        self.validate_factorization()?;
        self.validate_solve_diagonals()?;
        validate_rhs(b)?;
        for i in 0..self.n {
            let source = self.pivots[i] * nrhs;
            let destination = i * nrhs;
            x[destination..destination + nrhs].copy_from_slice(&b[source..source + nrhs]);
        }
        if nrhs == 1 {
            // Flat contiguous rows: same dot-form kernels as `solve_into`.
            Self::forward_dot_flat(&self.lu, self.n, x);
            Self::backward_dot_flat(&self.lu, self.n, x)?;
            return Ok(());
        }
        self.forward_blocked(x, nrhs);
        self.backward_blocked(x, nrhs)?;
        Ok(())
    }

    /// Block height for blocked multi-RHS substitution: diagonal
    /// micro-blocks solve with the serial AXPY loops while panel updates
    /// ride `schur_sub` (BLAS-3), amortizing one GEMM over `nrhs` columns
    /// instead of paying per-element AXPY dispatch per right-hand side.
    /// Matches the factorization panel width so L/U panels stay L2-sized.
    const SOLVE_TRSM_BLOCK: usize = 64;

    /// Forward substitution over flat contiguous rows (single-RHS layout),
    /// shared by `solve_into` and the `nrhs == 1` multi-RHS path.
    fn forward_dot_flat(factors: &DenseMatrix<T>, n: usize, x: &mut [T]) {
        let factors = factors.as_slice();
        for i in 0..n {
            let row_base = i * n;
            x[i] -= T::vec_dot_plain(&factors[row_base..row_base + i], &x[..i]);
        }
    }

    /// Backward substitution over flat contiguous rows (single-RHS layout),
    /// shared by `solve_into` and the `nrhs == 1` multi-RHS path.
    fn backward_dot_flat(factors: &DenseMatrix<T>, n: usize, x: &mut [T]) -> Result<(), LuError> {
        let tolerance = factor_tolerance(factors);
        let raw = factors.as_slice();
        for i in (0..n).rev() {
            let row_base = i * n;
            x[i] -= T::vec_dot_plain(&raw[row_base + i + 1..row_base + n], &x[i + 1..]);
            let u_ii = factors[[i, i]];
            if component_max_norm(u_ii) <= tolerance {
                return Err(LuError::SingularMatrix);
            }
            x[i] *= u_ii.inv();
            ensure_finite(x[i], "back substitution", i)?;
        }
        Ok(())
    }

    /// Blocked forward substitution for `nrhs >= 2`: serial diagonal
    /// micro-blocks plus packed-L `schur_sub` panel updates.
    fn forward_blocked(&self, x: &mut [T], nrhs: usize) {
        let bw = Self::SOLVE_TRSM_BLOCK;
        // Packed L column panel, reused across blocks (grown on demand).
        let mut panel: Vec<T> = Vec::new();
        let mut b0 = 0;
        while b0 < self.n {
            let b1 = (b0 + bw).min(self.n);
            let nb = b1 - b0;
            for i in b0..b1 {
                for j in b0..i {
                    let multiplier = -self.lu[[i, j]];
                    let (prefix, suffix) = x.split_at_mut(i * nrhs);
                    let source = &prefix[j * nrhs..(j + 1) * nrhs];
                    T::vec_axpy(multiplier, source, &mut suffix[..nrhs]);
                }
            }
            let rest = self.n - b1;
            // Zero-RHS solves still run the checks below; they just skip
            // the (empty) panel updates.
            if rest > 0 && nrhs > 0 {
                // Pack L[b1.., b0..b1] contiguously (lda = nb).
                panel.clear();
                panel.reserve(rest * nb);
                for row in b1..self.n {
                    let base = row * self.n + b0;
                    panel.extend_from_slice(&self.lu.as_slice()[base..base + nb]);
                }
                let (done, todo) = x.split_at_mut(b1 * nrhs);
                T::schur_sub(rest, nrhs, nb, &panel, &done[b0 * nrhs..], todo, nrhs);
            }
            b0 = b1;
        }
    }

    /// Blocked backward substitution for `nrhs >= 2`: packed-U `schur_sub`
    /// panel updates plus serial diagonal micro-blocks with the same
    /// singularity/finiteness checks as the unblocked path.
    fn backward_blocked(&self, x: &mut [T], nrhs: usize) -> Result<(), LuError> {
        let bw = Self::SOLVE_TRSM_BLOCK;
        let tolerance = factor_tolerance(&self.lu);
        // Packed U row panel, reused across blocks (grown on demand).
        let mut panel: Vec<T> = Vec::new();
        let mut b1 = self.n;
        while b1 > 0 {
            let b0 = b1.saturating_sub(bw);
            let nb = b1 - b0;
            let rest = self.n - b1;
            if rest > 0 && nrhs > 0 {
                // Pack U[b0..b1, b1..] contiguously (lda = rest).
                panel.clear();
                panel.reserve(nb * rest);
                for row in b0..b1 {
                    let base = row * self.n + b1;
                    panel.extend_from_slice(&self.lu.as_slice()[base..base + rest]);
                }
                let (block, rest_x) = x.split_at_mut(b1 * nrhs);
                T::schur_sub(
                    nb,
                    nrhs,
                    rest,
                    &panel,
                    rest_x,
                    &mut block[b0 * nrhs..],
                    nrhs,
                );
            }
            for i in (b0..b1).rev() {
                for j in (i + 1)..b1 {
                    let multiplier = -self.lu[[i, j]];
                    let (prefix, suffix) = x.split_at_mut(j * nrhs);
                    let destination = &mut prefix[i * nrhs..(i + 1) * nrhs];
                    T::vec_axpy(multiplier, &suffix[..nrhs], destination);
                }
                let diagonal = self.lu[[i, i]];
                if component_max_norm(diagonal) <= tolerance {
                    return Err(LuError::SingularMatrix);
                }
                T::vec_scale(&mut x[i * nrhs..(i + 1) * nrhs], diagonal.inv());
                validate_generated_slice(
                    &x[i * nrhs..(i + 1) * nrhs],
                    "multi-RHS back substitution",
                    i * nrhs,
                )?;
            }
            b1 = b0;
        }
        Ok(())
    }

    /// Solve `Aᵀx = b` using the retained factors.
    pub fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, LuError> {
        self.check_rhs_lengths(b.len(), self.n)?;
        self.validate_factor_shape()?;
        let mut x = vec![T::zero(); self.n];
        self.solve_transpose_into(b, &mut x)?;
        Ok(x)
    }

    /// Solve `Aᵀx = b` into caller-owned storage.
    ///
    /// The destination is unchanged when validation or numerical breakdown
    /// rejects the generated solution.
    pub fn solve_transpose_into(&self, b: &[T], x: &mut [T]) -> Result<(), LuError> {
        self.solve_transposed_into(b, x, false)
    }

    /// Solve Aᴴx = b (conjugate transpose) using the pre-computed LU factorization
    ///
    /// With PA = LU we have Aᴴ = UᴴLᴴP (P is a permutation, so P⁻ᴴ = P),
    /// hence the solve is a forward substitution against Uᴴ, a backward
    /// substitution against the unit-diagonal Lᴴ, and the inverse row
    /// permutation. For real matrices this reduces to the plain transpose
    /// solve. Discrete-adjoint codes reuse the forward factors this way
    /// instead of factorizing Aᴴ from scratch.
    pub fn solve_conj_transpose(&self, b: &[T]) -> Result<Vec<T>, LuError> {
        self.check_rhs_lengths(b.len(), self.n)?;
        self.validate_factor_shape()?;
        let mut x = vec![T::zero(); self.n];
        self.solve_conj_transpose_into(b, &mut x)?;
        Ok(x)
    }

    /// Solve `Aᴴx = b` into caller-owned storage.
    ///
    /// The destination is unchanged when validation or numerical breakdown
    /// rejects the generated solution.
    pub fn solve_conj_transpose_into(&self, b: &[T], x: &mut [T]) -> Result<(), LuError> {
        self.solve_transposed_into(b, x, true)
    }

    fn solve_transposed_into(&self, b: &[T], x: &mut [T], conjugate: bool) -> Result<(), LuError> {
        self.check_rhs_lengths(b.len(), x.len())?;
        self.validate_factorization()?;
        validate_rhs(b)?;
        // Keep the low-level `*_into` contract transactional as well as the
        // dense facade: transpose/adjoint arithmetic can overflow after a
        // finite RHS has been admitted, and caller storage must not expose a
        // partially solved vector on that failure.
        let mut staged = b.to_vec();
        let op = |value: T| if conjugate { value.conj() } else { value };
        // Conjugated-row scratch for the adjoint path: AXPY needs a
        // contiguous (possibly conjugated) factor-row segment, and the
        // adjoint must not mutate the stored factor. The plain-transpose
        // path reads factor rows directly and never touches this.
        let mut urow: Vec<T> = if conjugate {
            vec![T::zero(); self.n]
        } else {
            Vec::new()
        };
        let factors = self.lu.as_slice();

        // Forward substitution: Uᴴy = b (Uᴴ is lower triangular), in AXPY
        // form over contiguous factor-row segments so updates ride the
        // SIMD `vec_axpy` backend. Association order differs from the
        // former dot order in the last ulp; residuals are the oracle.
        // Loop-invariant tolerance, hoisted (see `solve_into`).
        let tolerance = factor_tolerance(&self.lu);
        for i in 0..self.n {
            let u_ii = op(self.lu[[i, i]]);
            if component_max_norm(u_ii) <= tolerance {
                return Err(LuError::SingularMatrix);
            }
            staged[i] *= u_ii.inv();
            ensure_finite(staged[i], "transpose forward substitution", i)?;
            if i + 1 < self.n {
                // NB: `vec_axpy` adds; substitution subtracts.
                let coef = -staged[i];
                if conjugate {
                    for (dst, src) in urow
                        .iter_mut()
                        .zip(&factors[i * self.n + i + 1..(i + 1) * self.n])
                    {
                        *dst = src.conj();
                    }
                    T::vec_axpy(coef, &urow[..self.n - i - 1], &mut staged[i + 1..]);
                } else {
                    T::vec_axpy(
                        coef,
                        &factors[i * self.n + i + 1..(i + 1) * self.n],
                        &mut staged[i + 1..],
                    );
                }
            }
        }

        // Backward substitution: Lᴴz = y (Lᴴ is unit upper triangular),
        // same AXPY form over the strict lower row segments.
        for i in (0..self.n).rev() {
            if i > 0 {
                // NB: `vec_axpy` adds; substitution subtracts.
                let coef = -staged[i];
                if conjugate {
                    for (dst, src) in urow.iter_mut().zip(&factors[i * self.n..i * self.n + i]) {
                        *dst = src.conj();
                    }
                    T::vec_axpy(coef, &urow[..i], &mut staged[..i]);
                } else {
                    T::vec_axpy(coef, &factors[i * self.n..i * self.n + i], &mut staged[..i]);
                }
            }
        }

        // Undo the permutation: w = Px means w[i] = x[pivots[i]]
        apply_position_to_original_permutation(&mut staged, &self.pivots);
        validate_generated_slice(&staged, "transpose permutation", 0)?;
        x.copy_from_slice(&staged);
        Ok(())
    }

    fn check_rhs_lengths(&self, input: usize, output: usize) -> Result<(), LuError> {
        if input != self.n || output != self.n {
            return Err(LuError::DimensionMismatch {
                expected: self.n,
                got: if input != self.n { input } else { output },
            });
        }
        Ok(())
    }

    fn validate_factorization(&self) -> Result<(), LuError> {
        self.validate_factor_shape()?;
        // O(n) permutation check with one visited bitmap. (A cycle-walk over
        // all n starts is O(n^2) per solve and dominated substitution for
        // large dense solves.) Fields stay public for compatibility, so the
        // check must run per solve — a factorization-time cached flag would
        // go stale under caller mutation.
        if self.pivots.len() != self.n {
            return Err(LuError::InvalidFactorization {
                reason: "pivots are not a permutation",
            });
        }
        let mut seen = vec![false; self.n];
        for &pivot in &self.pivots {
            if pivot >= self.n || seen[pivot] {
                return Err(LuError::InvalidFactorization {
                    reason: "pivots are not a permutation",
                });
            }
            seen[pivot] = true;
        }
        if let Some(index) = first_non_finite(self.lu.as_slice()) {
            return Err(LuError::NumericalBreakdown {
                stage: "stored factor validation",
                index,
            });
        }
        Ok(())
    }

    /// Reject pivots that the solve kernels cannot safely invert before they
    /// write into caller-owned solution storage. Factorization normally
    /// catches these values, but the factor fields are public for compatibility
    /// and callers can mutate a valid factor after construction.
    fn validate_solve_diagonals(&self) -> Result<(), LuError> {
        let tolerance = factor_tolerance(&self.lu);
        for index in 0..self.n {
            if component_max_norm(self.lu[[index, index]]) <= tolerance {
                return Err(LuError::SingularMatrix);
            }
        }
        Ok(())
    }

    fn validate_factor_shape(&self) -> Result<(), LuError> {
        if self.lu.nrows() != self.n || self.lu.ncols() != self.n {
            return Err(LuError::InvalidFactorization {
                reason: "factor dimensions do not match its declared order",
            });
        }
        if self.pivots.len() != self.n {
            return Err(LuError::InvalidFactorization {
                reason: "pivot count does not match factor order",
            });
        }
        Ok(())
    }
}

fn apply_position_to_original_permutation<T: Copy>(values: &mut [T], permutation: &[usize]) {
    for start in 0..values.len() {
        let mut cursor = permutation[start];
        let mut is_cycle_leader = true;
        while cursor != start {
            if cursor < start {
                is_cycle_leader = false;
                break;
            }
            cursor = permutation[cursor];
        }
        if !is_cycle_leader {
            continue;
        }
        let mut held = values[start];
        let mut current = start;
        loop {
            let destination = permutation[current];
            std::mem::swap(&mut held, &mut values[destination]);
            current = destination;
            if current == start {
                break;
            }
        }
    }
}

fn factor_tolerance<T: ComplexField>(factor: &DenseMatrix<T>) -> T::Real {
    // The strict lower triangle stores dimensionless L multipliers and must
    // not set the scale for tiny-but-well-conditioned matrices.
    let scale = (0..factor.nrows())
        .map(|index| component_max_norm(factor[[index, index]]))
        .fold(T::Real::zero(), T::Real::max);
    T::Real::epsilon()
        * T::Real::from_usize(factor.nrows().max(1)).unwrap_or_else(T::Real::one)
        * scale.max(T::Real::min_positive_value())
}

fn component_max_norm<T: ComplexField>(value: T) -> T::Real {
    // True modulus, not max(|re|, |im|): the latter understates the pivot
    // scale by up to sqrt(2) and accepts dangerously small pivots on
    // badly-scaled complex factors.
    value.norm()
}

pub(crate) fn first_non_finite<T: ComplexField>(values: &[T]) -> Option<usize> {
    // Vectorized on f32/f64/Complex32/Complex64 via `ComplexField::find_non_finite`.
    T::find_non_finite(values)
}

fn validate_rhs<T: ComplexField>(rhs: &[T]) -> Result<(), LuError> {
    if let Some(index) = first_non_finite(rhs) {
        return Err(LuError::NonFiniteRhs { index });
    }
    Ok(())
}

fn ensure_finite<T: ComplexField>(
    value: T,
    stage: &'static str,
    index: usize,
) -> Result<(), LuError> {
    if !value.re().is_finite() || !value.im().is_finite() {
        return Err(LuError::NumericalBreakdown { stage, index });
    }
    Ok(())
}

fn validate_generated_slice<T: ComplexField>(
    values: &[T],
    stage: &'static str,
    offset: usize,
) -> Result<(), LuError> {
    if let Some(index) = first_non_finite(values) {
        return Err(LuError::NumericalBreakdown {
            stage,
            index: offset + index,
        });
    }
    Ok(())
}

/// Compute LU factorization with partial pivoting.
///
/// Compact-panel GETF2 over `lu` rows `panel_start..`, columns
/// `panel_start..panel_end`.
///
/// Gathers the panel columns into `cbuf` (caller-owned scratch, resized as
/// needed), runs the standard pivot-search / swap / scale / rank-1 sequence
/// on the contiguous copy, mirrors each pivot swap onto the full rows and
/// `pivots` exactly as the in-place sweep would, then overlays the factored
/// block back. Same op order, so values are bit-identical to the in-place
/// sweep; the sweep stays L2-resident instead of re-touching full matrix
/// rows per rank-1 step (L3-latency-bound at large trailing sizes).
/// Recursive unblocked panel LU (LAPACK `dgetrf2` shape) over compact rows
/// `row0..rows` and panel columns `col0..col0+ncols` (full compact width
/// `w`). Narrow blocks (`<= 16` columns) take the flat search-swap-scale
/// loop; wider ones factor the left half, solve the top-right block with
/// serial AXPYs, update the bottom-right block with one BLAS-3 `schur_sub`,
/// then factor the bottom-right block. The flat loop's rank-1 updates run
/// at scalar pace; the GEMM update runs at kernel pace, which is where the
/// panel time goes. Pivot searches see fully-updated values at every step,
/// so the pivot sequence matches the flat algorithm (updates differ in the
/// last ulp only); swaps move full compact rows plus the mirrored `lu`
/// full-row swap and pivot bookkeeping, keeping multipliers aligned.
/// Shared context for one compact-panel recursive factorization (see
/// [`PanelRecursion::factor`]); groups the invariant arguments so the
/// recursive worker stays under the argument-count lint.
struct PanelRecursion<'a, T: ComplexField> {
    cbuf: &'a mut [T],
    lu: &'a mut DenseMatrix<T>,
    w: usize,
    rows: usize,
    panel_start: usize,
    pivots: &'a mut [usize],
    pivot_tolerance: T::Real,
    k_seg: &'a mut Vec<T>,
    /// Lookahead mode: record `(k, max_row)` pivot swaps instead of
    /// mirroring them onto the full `lu` rows (the trailing region is
    /// concurrently updated by the overlapped Schur remainder; the
    /// recorded swaps apply after the join — same permutation, same
    /// values, so bitwise identical). `None` is the standard inline
    /// mirror. The compact-row swap and `pivots` bookkeeping always
    /// stay inline.
    swap_log: Option<&'a mut Vec<(usize, usize)>>,
}

impl<T: ComplexField> PanelRecursion<'_, T> {
    /// Factor compact rows `row0..rows` × panel columns `col0..col0+ncols`.
    fn factor(&mut self, row0: usize, col0: usize, ncols: usize) -> Result<(), LuError> {
        // Measured optimum at n=2048 (1T gap): 16 -> 1.38, 8 -> 1.34,
        // 4 -> 1.35. Below 8 the GEMM updates get too thin for the kernel
        // and dispatch/pack overhead wins back the flops. Flat (no
        // recursion) re-trialled 2026-09-26 under threshold-33 via a
        // temporary env hook (removed): interleaved R/F x3 at n=1024 8T,
        // flat panel +1.4ms, total +1.2..1.3ms — the recursion GEMM updates
        // earn their keep even at 31 compact panels; do not retry flat.
        // Base 16 re-trialled 2026-09-26 post-SIMD-search: still loses
        // (+0.10ms, 7/8 interleaved pairs). Stands at 8.
        const RECURSION_BASE: usize = 8;
        let n = self.lu.nrows();
        let w = self.w;
        let rows = self.rows;
        let panel_start = self.panel_start;
        if ncols <= RECURSION_BASE {
            for ckk in col0..col0 + ncols {
                let k = panel_start + ckk;
                // Pivot search via `vec_pivot_scan` (AVX2 gather-free
                // setr kernel, bitwise-identical contract) — shipped
                // 2026-09-26 (~0.2ms on dgetrf-1024, 12/14 interleaved
                // pairs). A pre-compact per-column wiring lost ~1.2ms;
                // the compact layout changed the geometry (stride-256B
                // L2-resident vs stride-n), which is why the re-trial
                // won — do not generalize that old loss to this layout.
                // (SW prefetch trialled: no win, HW stride prefetcher
                // already covers the walk; a 4-way unrolled max-reduce
                // with identical tie-breaking was neutral end to end;
                // both reverted.) NaN edge: if the scan's first row is
                // NaN the kernel early-returns it and a later finite row
                // beating the diagonal is missed — unreachable past
                // entry validation, and the endscan still errors.
                let (max_val, max_crow) = {
                    let mut max_val = component_max_norm(self.cbuf[ckk * w + ckk]);
                    let mut max_crow = ckk;
                    let first = row0.max(ckk + 1);
                    if first < rows {
                        let (scan_row, scan_val) =
                            T::vec_pivot_scan(&self.cbuf[ckk..], w, first, rows - first);
                        if scan_val > max_val {
                            max_val = scan_val;
                            max_crow = scan_row;
                        }
                    }
                    (max_val, max_crow)
                };
                if max_val <= self.pivot_tolerance {
                    return Err(LuError::SingularMatrix);
                }
                if max_crow != ckk {
                    let (lo_c, hi_c) = if ckk < max_crow {
                        (ckk, max_crow)
                    } else {
                        (max_crow, ckk)
                    };
                    let (left_c, right_c) = self.cbuf.split_at_mut(hi_c * w);
                    left_c[lo_c * w..(lo_c + 1) * w].swap_with_slice(&mut right_c[..w]);
                    let max_row = panel_start + max_crow;
                    match self.swap_log.as_deref_mut() {
                        Some(log) => log.push((k, max_row)),
                        None => {
                            let data = self.lu.as_mut_slice();
                            let (lo, hi) = if k < max_row {
                                (k, max_row)
                            } else {
                                (max_row, k)
                            };
                            let (left, right) = data.split_at_mut(hi * n);
                            left[lo * n..(lo + 1) * n].swap_with_slice(&mut right[..n]);
                        }
                    }
                    self.pivots.swap(k, max_row);
                }
                let pivot_inv = self.cbuf[ckk * w + ckk].inv();
                self.k_seg.clear();
                self.k_seg
                    .extend_from_slice(&self.cbuf[ckk * w + ckk + 1..ckk * w + col0 + ncols]);
                // Rank-1 step via `vec_ger_update` (split scale pass +
                // dispatched ger call). A fused single-pass kernel
                // (`vec_panel_ger`, scale+ger with identical mul+fmadd
                // sequence) was trialled 2026-09-26 and lost on f64
                // (~40% slower per leaf in micro: the fused scale->use
                // chain serializes per row-group while the split hides
                // scale latency in its own streaming pass; f32 won only
                // ~8%). Split stands; do not retry fusion without
                // breaking that chain. Per-column dispatch is affordable
                // here (unlike the old pre-compact `vec_pivot_scan`
                // trial — see the search note above for why that loss
                // doesn't apply to this layout): the rank-1 phase is ~7x
                // larger than the search phase was.
                //
                // SAFETY: `mults` (column `ckk` cells) and `block` rows
                // (columns `ckk + 1 ..`) are disjoint cells of `cbuf`;
                // slice lengths cover exactly the strided footprints.
                let ulen = col0 + ncols - (ckk + 1);
                let rows_below = rows - (ckk + 1);
                let (_, rest) = self.cbuf.split_at_mut((ckk + 1) * w);
                for crow_row in rest.chunks_exact_mut(w) {
                    // Bounds-check-free: every chunk has length `w > ckk`.
                    unsafe {
                        let cell = crow_row.as_mut_ptr().add(ckk);
                        *cell *= pivot_inv;
                    }
                }
                if rows_below > 0 && ulen > 0 {
                    let base = self.cbuf.as_mut_ptr();
                    unsafe {
                        let mults = std::slice::from_raw_parts(
                            base.add((ckk + 1) * w + ckk) as *const T,
                            (rows_below - 1) * w + 1,
                        );
                        let block = std::slice::from_raw_parts_mut(
                            base.add((ckk + 1) * w + ckk + 1),
                            (rows_below - 1) * w + ulen,
                        );
                        T::vec_ger_update(
                            T::from_real(-T::Real::one()),
                            mults,
                            w,
                            self.k_seg.as_slice(),
                            block,
                            w,
                            rows_below,
                            ulen,
                        );
                    }
                }
            }
            return Ok(());
        }
        let s = ncols / 2;
        // Factor the left block (rows row0.., cols col0..col0+s).
        self.factor(row0, col0, s)?;
        // Middle TRSM: top-s rows, right columns (serial AXPYs; s <= 32
        // rows of length <= 64, far below any threading payoff).
        for i in 0..s {
            let r = row0 + i;
            let (done, todo) = self.cbuf.split_at_mut(r * w + col0 + s);
            let current = &mut todo[..ncols - s];
            for q in 0..i {
                let multiplier = -done[r * w + col0 + q];
                let prev = &done[(row0 + q) * w + col0 + s..(row0 + q) * w + col0 + ncols];
                T::vec_axpy(multiplier, prev, current);
            }
        }
        // Bottom-right GEMM update, strided-direct (shipped 2026-09-26):
        // the scratch A/B packs are deleted — the GEMM reads A (rows
        // below, `lda = w`), B (top rows, `ldb = w`) and C strided from
        // `cbuf` and packs internally per task, mirroring the driver
        // `schur_update_strided` win (~0.25ms on dgetrf-1024: panel GEMM
        // 0.83 -> 0.55 per timer account, sgetrf -0.2 isolated). Slices
        // alias inside `cbuf` (disjoint cells; same raw-pointer
        // precedent).
        let m = rows - (row0 + s);
        let nn = ncols - s;
        // `m = 0` (no rows below) skips the update, as the packed path
        // did via the empty GEMM; `s, nn >= 1` holds for `ncols > 8`.
        if m == 0 {
            return self.factor(row0 + s, col0 + s, nn);
        }
        let base = self.cbuf.as_mut_ptr();
        unsafe {
            let a_blk = std::slice::from_raw_parts(
                base.add((row0 + s) * w + col0) as *const T,
                (m - 1) * w + s,
            );
            let b_blk = std::slice::from_raw_parts(
                base.add(row0 * w + col0 + s) as *const T,
                (s - 1) * w + nn,
            );
            let c_blk = std::slice::from_raw_parts_mut(
                base.add((row0 + s) * w + col0 + s),
                (m - 1) * w + nn,
            );
            T::schur_sub_strided(m, nn, s, a_blk, w, b_blk, w, c_blk, w);
        }
        // Factor the bottom-right block.
        self.factor(row0 + s, col0 + s, nn)
    }
}

fn compact_panel_factorize<T: ComplexField>(
    lu: &mut DenseMatrix<T>,
    panel_start: usize,
    panel_end: usize,
    pivots: &mut [usize],
    pivot_tolerance: T::Real,
    cbuf: &mut Vec<T>,
    k_seg: &mut Vec<T>,
    swap_log: Option<&mut Vec<(usize, usize)>>,
) -> Result<(), LuError> {
    let n = lu.nrows();
    let panel_width = panel_end - panel_start;
    let rows = n - panel_start;
    // Sub-phase account re-measured 2026-09-26 via temporary static
    // timers (removed), dgetrf-1024 per factorize: gather 0.20, leaf
    // 1.6-1.7 (search 0.45 / scale 0.39 / ger 0.7 + swaps), rec-GEMMs
    // 0.35, mid-TRSMs 0.05, overlay 0.16 — panel ~2.5 of ~8.4 in-place.
    // The leaf walks sit at their traffic floors (SW prefetch, unrolled
    // max-reduce, and per-column SIMD search all trialled: prefetch and
    // unroll neutral, SIMD search shipped for ~0.2). At n=2048 the
    // 512KB panel overflows L2 and panel-side roughly triples; capping
    // compact at 1024 rows lost decisively twice (65 -> 81-83), so
    // compact-everywhere stands — big-n panel work is the next project.
    // 2048 account, 2026-09-26 (temp timers, reverted), per factorize:
    // gather 1.77 / search 1.31 / swap 0.00 / scale 1.74 / ger 3.07 /
    // midtrsm 0.00 / recGEMM 1.48 / overlay 1.90 = 11.25 (in-situ 11.0).
    // Scalar core (search+scale+ger+swap = 6.12) TIES vendor dgetf2 5.91;
    // the gap is gather+overlay (3.67) + recGEMM (1.48, vendor has none).
    // Mirror swaps are free (contiguous). All-in-place re-trialled
    // 2026-09-26 under current kernels (threshold const flip, reverted):
    // dgetrf-2048 body 44 -> 82ms (2nd run 66-132, drift but far above
    // 44 throughout) — full-row-stride ger is catastrophic, compact
    // earns its keep decisively. Do not retry without changing the
    // in-place sweep's traffic geometry.
    //
    // Gather/overlay CLOSED 2026-09-26 (gatherloc probe): SW prefetch
    // neutral (1.69->1.65, noise); parallel-16 gather only 1.16x
    // (1.69->1.46 — box DRAM wall ~20GB/s, single core saturates it);
    // single NUMA node (no placement loss); in-situ rate == local rate.
    // Gather+overlay (3.67ms) is at the box STREAM floor — irreducible
    // traffic, do not re-attack the copy rate. Compact-pool routing
    // re-verified (SONIUM_COMPACT=0 loses ~1.5ms); L3 is 16MB so the
    // 32MB factor never engages the compact pool at 2048 anyway.
    // Open: Schur in-situ excess (24.5 vs standalone-hot 19.8, same
    // pool) — suspect serial-phase frequency ramp on schedutil box.
    //
    // RESOLVED 2026-09-26 (schurgap/stdgemm probes): the 19.8 was
    // overwarm (med-of-7 per size); honest single-pass costs 23-24
    // in-probe AND in-situ — no excess, and gaps add only ~1ms (ramp
    // is minor). The real gap is single-touch COLD-stream efficiency:
    // 1500x1500x32 ours 65GF vs vendor 153GF (2.35x); 1000-shape ties
    // (L3-resident 8MB); 500-shape 1.6x. Thread sweep saturates both at
    // ~8T (ours 2.88->2.16->2.22, vendor 1.12->0.90->0.88) — box STREAM
    // is the wall and vendor uses 2.5x more of it (~61 vs ~24GB/s on
    // identical 54MB traffic). Next: cold-stream mechanisms (NT stores
    // for C? prefetch/MLP in sweep? task-grid L3 blocking?).
    //
    // LAYOUT SENSITIVITY 2026-09-26 (stdgemm arena + coldbisect):
    // same gemm_f64 1500x1500x32 measures 0.66-1.83ms depending only on
    // allocation layout (arena-contiguous vs split Vecs) and box moment;
    // vendor holds 0.80-0.94 across ALL runs. WE are the brittle ones
    // (4KB-page TLB storm? pack TLB pressure?), vendor is robust —
    // consistent with aggressive SW prefetch in their micro-kernel
    // hiding walk latency. with_buf exonerated (thread-local, no
    // cross-thread contention). Box STREAM ceiling 74-88GB/s exists.
    // Next: prefetch-robustness in OUR sweep, gated by stdgemm-arena.
    //
    // C-ROW PREFETCH FALSIFIED 2026-09-26 (temp SONIUM_GEMM_PF_C in
    // sweep_f64, reverted): next-tile C-row T0 prefetch neutral-to-worse
    // (arena 1.49/1.35 -> 2.32/1.48). Joins the reverted B-walk
    // prefetch; do not retry streamer-covering prefetches. NT stores
    // also dead for beta=1 (no RFO to save — the C read is required).
    //
    // PER-CALL PARITY 2026-09-26 (sizecurve probe, arena, ramped):
    // single-touch dgemm T x T x 32: small sizes we lose mildly
    // (200-600: 0.7-0.9x), large sizes parity-or-win (1200-2000:
    // 0.97-1.16x). T=400 22x one-off NOT reproduced (noise). So the
    // end-to-end Schur gap is dispatch count x scope cost (~2ms) +
    // hetero interference, not kernel rate. Remaining levers:
    // panel-Schur overlap (lookahead, architectural) or accept.
    //
    // LOOKAHEAD SHIPPED 2026-09-26 (default-on, `SONIUM_LU_LOOKAHEAD=0`
    // disables): serial next-panel block update + panel factorize on a
    // pool worker overlapped with the Schur remainder on the caller;
    // deferred wide swaps after the join. Bitwise identical f64+f32
    // (maxdiff 0e0, same pivots, same residuals; oblookahead/oslookahead
    // interleaved, order-rotated). dgetrf-2048 0.90x pooled / 0.94x hot
    // (1.88x -> 1.69x vs OB), plus cold-robustness (first-position
    // 0.87x: the serial panel chain hides slow-core ramp). sgetrf-2048
    // 0.96x bitwise-identical. Complex uses the naive serial nest
    // (correct, last-ulp possible; n256 pivot-exact test green).
    //
    // POST-LOOKAHEAD PHASES 2026-09-26 (temp timers, reverted;
    // dgetrf-2048 warmed in-loop): exposed panel 0.6 / TRSM 2.6 /
    // Schur-scopes 24.2 / lookahead-serial 6.8 (streaming floor, ~20GF:
    // unavoidable) / deferred 0.0 / valid 1.1. Schur runs ~227GF on
    // strided (lda=n) packs vs vendor-implied ~300GF — strided-pack
    // prefetch is the next lever (C-row and B-walk streamer prefetches
    // already falsified; the PACK loops were never tried).
    //
    // QR FRONT 2026-09-26 (obqr + temp QR timers, reverted): dgeqrf-2048
    // 397 vs 247 (1.61x, 150ms gap — biggest absolute). Per factorize:
    // gather 12.2 / reflect 12.0 / tform 39.5 / overlay 10.9 / wy 81.1 /
    // qgather 4.0 / qwy 147.6 = 307. m-thin NoTrans 0.77-0.86x and
    // trans-A W-formation 0.73-0.94x (mthin probe) — GEMM rate explains
    // only ~35ms. Bulk: serial tform (39.5) + panels (35) + Q-pass size
    // (2x traffic). NOTE: vendor 247-249 tight across runs, ours
    // 320-400 (drift) — robustness gap (persistent pool vs per-call
    // scopes) may exceed the kernel gap. Next: tform serial rate,
    // then scope-amortization/persistent-team.
    //
    // WY-T HOIST SHIPPED 2026-09-26: zk dots were O(nb^3/6)/panel
    // (recomputed per i) — hoisted to once per j, O(nb^2/2),
    // bitwise-identical (same inputs/order). dgeqrf: 512 1.59->1.38x,
    // 1024 1.83->1.66x, 2048 best 367 (1.48x) but drift band 367-404
    // vs vendor 246-248 tight — fix proven, variance is the box.
    //
    // QR SCATTER-RFO ANALYSIS 2026-09-26: gather/refresh/overlay are
    // j-outer/r-inner transposes with 256B/16KB-strided WRITES — an RFO
    // per 8B element (64B moved per 8B). Loop-swap doesn't help
    // (symmetric strided side); blocking doesn't reduce lines touched.
    // Only NT stores cut it (skip RFO, WC-combine): needs 32B-align
    // runtime fallback at 3 sites (gather panel/pt, refresh, overlay).
    // Prize ~10ms of the 23ms round-trip. Next trial.
    //
    // PINNED PARITY 2026-09-26 (obqr taskset matrix, RAYON matched):
    // pin0-7/8: ours 355 vs ob 375 (0.95x, WE WIN); pin0-15/16: 338
    // vs 311 (1.09x); unpinned/16: ~400 vs ~250 (1.6x); unpinned/8:
    // 1.75-1.88x (worse!). Count ruled out (8-unpinned loses) —
    // PLACEMENT is the lever, kernels are fine on equal footing.
    // Vendor threads were already matched (obqr sets 16). Open: does
    // OUR sched_setaffinity actually take effect in-probe (sandbox
    // refusal is silently ignored)? Verify via /proc task masks live,
    // then extend one-CCX packing beyond L3-fit problems.
    //
    // CENSUS 2026-09-26 (threadcensus probe, self-inspection): OUR
    // pinning WORKS — 16 sonium-mirror workers, single-CPU masks 0-15.
    // Plus 64 vendor threads at startup (openblas init, roaming 0-127)
    // and +16 ours (rayon global, idle) on first use. No oversubscrip-
    // tion pathology on our side. SONIUM_COMPACT_CPUS=0-7 neutral at
    // LU-2048 (48.9/45.7 vs 44.6/46.0 — guard blocks compact above L3
    // fit). Next: wide-compact driver pool (one CCX, 8 workers) for
    // big factorizations — placement win doesn't need L3 fit (QR
    // pinned-8 won at 0.95x with a 32MB factor).
    //
    // MIRROR-CAP FALSIFIED 2026-09-26 (temp SONIUM_MIRROR_MAX=8,
    // reverted): QR-2048 370/384 -> 490/457. Taskset parity came from
    // constraining VENDOR, not from our topology. Affinity-off
    // (roam like vendor) also neutral-to-worse (411/406). Default
    // pinned-0-15 stands; hetero barrier cost is structural (per-call
    // scopes x slow-core stragglers), addressable only by overlap or
    // fewer, larger dispatches.
    // Gather into spare capacity (every element is written below before any
    // read), skipping the zeroing `resize` would do.
    cbuf.clear();
    cbuf.reserve(rows * panel_width);
    {
        // Row-granular gather at memcpy speed (mirrors the overlay below):
        // per-element writes cost ~2x here. SAFETY: `reserve` guarantees
        // the spare capacity; each row copy writes exactly `panel_width`
        // elements within bounds; source and spare are disjoint (`cbuf`
        // is caller-owned scratch, never aliasing `lu`); `set_len`
        // follows writes to all `rows * panel_width` elements.
        let src = lu.as_slice();
        let dst = cbuf.spare_capacity_mut().as_mut_ptr() as *mut T;
        unsafe {
            for crow in 0..rows {
                let r = panel_start + crow;
                core::ptr::copy_nonoverlapping(
                    src.as_ptr().add(r * n + panel_start),
                    dst.add(crow * panel_width),
                    panel_width,
                );
            }
            cbuf.set_len(rows * panel_width);
        }
    }
    // Reused across panels via the workspace: `with_capacity` here would
    // malloc ~32 tiny pivot-row segments per factorization.
    k_seg.clear();
    k_seg.reserve(panel_width);
    let mut rec = PanelRecursion {
        cbuf,
        lu,
        w: panel_width,
        rows,
        panel_start,
        pivots,
        pivot_tolerance,
        k_seg,
        swap_log,
    };
    rec.factor(0, 0, panel_width)?;
    // Overlay the factored block (diagonal block plus multipliers below).
    // No scan here: each panel's rows are validated once, right after
    // its Schur update in the driver loop (hot), plus the last panel
    // after the loop — no cold end-of-factorize pass. (Per-update tile
    // scans trialled 2026-09-26, reverted — O(n^3/k) traffic, ~1.5ms
    // loss on dgetrf-1024; see banded_schur.)
    let data = rec.lu.as_mut_slice();
    for crow in 0..rows {
        let r = panel_start + crow;
        let src = &rec.cbuf[crow * panel_width..(crow + 1) * panel_width];
        data[r * n + panel_start..r * n + panel_end].copy_from_slice(src);
    }
    Ok(())
}

/// Apply lookahead-deferred pivot swaps recorded by a
/// [`PanelRecursion`] with `swap_log` set: the same full-row swaps the
/// inline mirror would have performed, restricted to the columns the
/// U12 panel solve: pack the `pe - ps` lower block and solve the full
/// trailing (`pe..n`) of the panel rows in place with one
/// `trsm_u12_unit_lower` call.
fn trsm_panel<T: ComplexField>(
    lu: &mut DenseMatrix<T>,
    l11_pack: &mut Vec<T>,
    n: usize,
    panel_start: usize,
    panel_end: usize,
) {
    let panel_width = panel_end - panel_start;
    // Capacity only: the row copies below write every element
    // before the TRSM reads, so a zeroing `resize` is dead stores.
    l11_pack.clear();
    l11_pack.reserve_exact(panel_width * panel_width);
    // SAFETY: fully overwritten by the row copies below before any read.
    unsafe {
        l11_pack.set_len(panel_width * panel_width);
    }
    for i in 0..panel_width {
        l11_pack[i * panel_width..(i + 1) * panel_width]
            .copy_from_slice(&lu.row(panel_start + i)[panel_start..panel_end]);
    }
    // Single thin TRSM call (a 2-way blocked split — solve top
    // half, strided GEMM-update bottom, solve bottom half — was
    // trialled 2026-09-26 and lost ~0.25ms interleaved dgetrf-1024
    // 6/6: two trsm_core dispatches plus the middle GEMM dispatch
    // per panel outweigh halving the 32-step serial chain. The
    // halves keep the same nrhs, so band work doesn't shrink while
    // dispatch doubles. Do not retry).
    let trailing = n - panel_end;
    let lu_slice = lu.as_mut_slice();
    let block_start = panel_start * n + panel_end;
    let (_, block_span) = lu_slice.split_at_mut(block_start);
    T::trsm_u12_unit_lower(l11_pack, panel_width, panel_width, block_span, n, trailing);
}

/// Apply lookahead-deferred pivot swaps recorded by a panel factorization:
/// the same full-row swaps the inline mirror would have performed,
/// restricted to the columns the overlay did not already rewrite
/// (`[0..skip_lo)` and `[skip_hi..n)`). Runs after the overlapped Schur
/// remainder joins — same permutation applied to the same post-update
/// values, so bitwise identical to the inline order (swap and overlay
/// touch disjoint columns, so their relative order is irrelevant).
fn apply_deferred_lu_swaps<T: ComplexField>(
    lu: &mut DenseMatrix<T>,
    swaps: &[(usize, usize)],
    skip_lo: usize,
    skip_hi: usize,
) {
    if swaps.is_empty() {
        return;
    }
    let n = lu.nrows();
    let data = lu.as_mut_slice();
    for &(k, m) in swaps {
        if k == m {
            // Identity pivot (no-op swap): the split below would index
            // out of bounds, so skip before splitting.
            continue;
        }
        let (lo, hi) = if k < m { (k, m) } else { (m, k) };
        let (left, right) = data.split_at_mut(hi * n);
        let lo_row = &mut left[lo * n..(lo + 1) * n];
        let hi_row = &mut right[..n];
        if skip_lo > 0 {
            lo_row[..skip_lo].swap_with_slice(&mut hi_row[..skip_lo]);
        }
        if skip_hi < n {
            lo_row[skip_hi..].swap_with_slice(&mut hi_row[skip_hi..]);
        }
    }
}

/// This follows the right-looking LAPACK `GETRF` structure. Each panel is
/// factorized with scalar partial pivoting, the block row is obtained by a
/// unit-lower triangular solve, and the trailing matrix update is deferred to
/// one `mat_mul_nn` call. For the built-in f32/f64/complex scalar types that
/// final operation routes through `sonium-compute`'s accelerated BLAS-3 path.
pub fn lu_factorize<T: ComplexField>(a: &DenseMatrix<T>) -> Result<LuFactorization<T>, LuError> {
    // Funnel onto a hot pool worker (see `run_on_worker`): the serial
    // panel phases would otherwise run on the caller's possibly cold core
    // at half rate. Single clone, as before — it just moves inside. Tiny
    // matrices skip the funnel: their whole solve is µs, below the spawn
    // round-trip, and frequency barely matters at that scale. Shipped
    // 2026-09-26: dgetrf-1024 8T mask-128 steady 15.7 -> 11.8
    // (0.34x -> 0.46x vs OB-8T 5.4); interleaved direct/funnel A/B ties
    // when the caller is hot and wins ~6ms when cold — strictly dominates.
    if a.nrows() < 128 {
        let mut workspace = LuWorkspace::new();
        return lu_factorize_with_workspace(a, &mut workspace);
    }
    // Tiny-pool factorization (shipped default-on 2026-09-27,
    // `SONIUM_LU_TINY=0` restores the compact/mirror funnel): the whole
    // factorization runs on the tiny pool (16 pinned workers, one
    // install) so every nested scope routes inline with 16-wide sizings
    // and zero per-call pool hops (~200 nested installs, ~10-30µs each).
    // dgetrf-2048 OB-free: 40.0 -> 34.6ms (1.16x, reproduced); same
    // pivots at 1024/2048, maxdiff 5.6e-17 at 2048 (pool-geometry TRSM
    // band splits re-chunk — same class as the corrected TRSM note),
    // residuals equal. Envelope below; huge factors stay on the
    // mirror funnel, small/global-narrow keep existing routing.
    // Tiny-pool envelope (measured OB-free, order-rotated): consistent
    // wins at n=2048 (1.16-1.20x, four runs); n=1024 conflicted across
    // runs (1.08x win vs 0.88x loss — ±30% box drift swamps the ~10%
    // effect, and compact's best 8.3ms beats tiny's there) — floor
    // n>=2048. Cap 64MB (2x the tiny pool's ~32MB L3 over ~2 CCDs,
    // headroom for packs): 4096 (268MB) loses on 16 workers. f64 only
    // (sgetrf-1024 regresses 2x on tiny); inert at global <= 16
    // (router already inline).
    const TINY_MIN_N: usize = 2048;
    const TINY_MAX_EST: u64 = 64 << 20;
    let tiny_off = std::env::var("SONIUM_LU_TINY")
        .map(|v| v == "0")
        .unwrap_or(false);
    let est = a.nrows() as u64 * a.nrows() as u64 * std::mem::size_of::<T>() as u64;
    if !tiny_off
        && T::lu_prefers_tiny_pool()
        && a.nrows() >= TINY_MIN_N
        && rayon::current_num_threads() > 16
        && est <= TINY_MAX_EST
    {
        return sonium_compute::cpu::blas3::run_tiny(|| {
            let mut workspace = LuWorkspace::new();
            lu_factorize_in_place(clone_into_factor_buffer(a), &mut workspace)
        });
    }
    // Compact pool when the factor fits in L3: the clone runs inside so
    // first-touch binds pages to the compact NUMA node; otherwise the
    // mirror-pool funnel as before (clone outside for `'static`). Both
    // clones check the factor-buffer pool out first (warm pages on
    // repeated same-size calls; see `FACTOR_POOL`).
    // Narrow pool (one L3 domain): rank-32 updates are memory-bound and
    // fastest L3-resident on a single CCX.
    if sonium_compute::cpu::blas3::compact_applies(est, false) {
        return sonium_compute::cpu::blas3::run_compact(est, false, || {
            let mut workspace = LuWorkspace::new();
            lu_factorize_in_place(clone_into_factor_buffer(a), &mut workspace)
        });
    }
    let owned = clone_into_factor_buffer(a);
    sonium_compute::cpu::blas3::run_on_worker(move || {
        let mut workspace = LuWorkspace::new();
        lu_factorize_in_place(owned, &mut workspace)
    })
}

/// Factorize a matrix while reusing caller-owned panel storage.
pub fn lu_factorize_with_workspace<T: ComplexField>(
    a: &DenseMatrix<T>,
    workspace: &mut LuWorkspace<T>,
) -> Result<LuFactorization<T>, LuError> {
    lu_factorize_in_place(clone_into_factor_buffer(a), workspace)
}

/// Factorize an owned matrix in place, avoiding the compatibility wrapper's
/// input clone and reusing caller-owned panel storage.
///
/// Like [`lu_factorize`], the solve funnels onto a hot pool worker (see
/// `run_on_worker`): the serial panel phases would otherwise run on the
/// caller's possibly cold core at half rate. When the factor fits in L3
/// the compact pool takes the solve instead (see `run_compact`), exactly
/// as in [`lu_factorize`]. The caller-owned workspace cannot cross the
/// mirror funnel's `'static` bound, so it is swapped out with a fresh
/// one for the flight and restored on return (on panic the slot keeps
/// the fresh workspace); the compact funnel borrows, so it uses the
/// caller workspace directly. Nested calls run inline, tiny matrices
/// skip the funnel — same policy as [`lu_factorize`]. Same kernels,
/// same order; pool choice can re-chunk TRSM bands at last-ulp level
/// (residuals govern, as with the tiny pool).
pub fn lu_factorize_in_place<T: ComplexField>(
    lu: DenseMatrix<T>,
    workspace: &mut LuWorkspace<T>,
) -> Result<LuFactorization<T>, LuError> {
    let n = lu.nrows();
    if n != lu.ncols() {
        return Err(LuError::DimensionMismatch {
            expected: n,
            got: lu.ncols(),
        });
    }
    // `SONIUM_LU_NOFUNNEL=1` skips the funnel (diagnostic/A-B fallback).
    let no_funnel = std::env::var("SONIUM_LU_NOFUNNEL")
        .map(|v| v == "1")
        .unwrap_or(false);
    // Tiny-pool funnel (shipped default-on 2026-09-27, `SONIUM_LU_TINY=0`
    // restores the compact/mirror routing): same one-install geometry as
    // [`lu_factorize`]; `run_tiny` borrows (no `'static` bound), so the
    // caller workspace is used directly with no swap dance. Nested calls
    // run inline, exactly as in the other funnels.
    let tiny_off = std::env::var("SONIUM_LU_TINY")
        .map(|v| v == "0")
        .unwrap_or(false);
    let est_in = n as u64 * n as u64 * std::mem::size_of::<T>() as u64;
    // Same envelope as [`lu_factorize`] (see TINY_MIN_N/TINY_MAX_EST).
    if !tiny_off
        && !no_funnel
        && T::lu_prefers_tiny_pool()
        && n >= 2048
        && rayon::current_num_threads() > 16
        && est_in <= (64 << 20)
    {
        return sonium_compute::cpu::blas3::run_tiny(|| {
            lu_factorize_in_place_direct(lu, workspace)
        });
    }
    if n >= 128 && !no_funnel {
        // Compact pool when the factor fits in L3 (shipped 2026-10-02,
        // mirrors `cholesky_factorize_in_place`): the whole factorization
        // runs on the 4-worker single-CCX pool with the factor L3-resident
        // instead of spread over the 16-worker mirror pool (4 L3 domains
        // + DRAM streaming + cross-domain barriers). `run_compact`
        // borrows (no `'static` bound), so the caller workspace is used
        // directly with no swap dance; nested calls run inline. Measured
        // interleaved A/B, RAYON=16 (compact-simulated vs direct):
        // dgetrf-512 3.26 -> 1.91, dgetrf-1024 13.21 -> 8.67,
        // sgetrf-1024 8.64 -> 4.92 (narrow beats wide everywhere;
        // wide kept out, matching `lu_factorize`). Larger sizes are
        // unaffected by construction (`compact_applies` fails closed
        // above 3/4 L3: f64 n > 1365, f32 n > 1932); n < 128 keeps the
        // direct path; `SONIUM_COMPACT=0` restores the mirror funnel.
        // Pool choice can re-chunk TRSM bands at last-ulp level (same
        // class as the tiny-pool note); pivots and residuals govern.
        if sonium_compute::cpu::blas3::compact_applies(est_in, false) {
            return sonium_compute::cpu::blas3::run_compact(est_in, false, || {
                lu_factorize_in_place_direct(lu, workspace)
            });
        }
        let owned_ws = std::mem::replace(workspace, LuWorkspace::new());
        let (result, owned_ws) = sonium_compute::cpu::blas3::run_on_worker(move || {
            let mut owned_ws = owned_ws;
            let result = lu_factorize_in_place_direct(lu, &mut owned_ws);
            (result, owned_ws)
        });
        *workspace = owned_ws;
        return result;
    }
    lu_factorize_in_place_direct(lu, workspace)
}

/// Direct driver behind [`lu_factorize_in_place`]: identical body, no
/// funnel (the caller already funneled, is nested on a worker, or is
/// below the funnel size threshold).
fn lu_factorize_in_place_direct<T: ComplexField>(
    mut lu: DenseMatrix<T>,
    workspace: &mut LuWorkspace<T>,
) -> Result<LuFactorization<T>, LuError> {
    // Square shape checked by the caller; re-bind the order locally.
    let n = lu.nrows();
    // Entry validation + scale are one fused pass (shipped 2026-09-26):
    // `vec_max_norm_checked` folds the max-abs scale and an
    // OR-accumulated non-finite flag in a single streaming read (~0.15ms
    // off dgetrf-1024: one 8MB pass instead of two). Same scale value as
    // the old `vec_max_norm` on finite input (`abs` exact, finite max
    // unique; complex keeps the hypot loop in the default impl). The
    // first-bad index comes from the existing `find_non_finite` rescan,
    // which only runs when the flag trips (never on valid input), so the
    // error contract is unchanged. (Two earlier single-pass trials lost —
    // a branchy first-seen store blocked SIMD, a branchless NaN
    // accumulator defeated the vectorizer. The dual-accumulator fold
    // sidesteps both: no NaN-propagating max needed.)
    let (matrix_scale, entry_finite) = T::vec_max_norm_checked(lu.as_slice());
    if !entry_finite {
        // Rare (never on valid input): locate the first offender for the
        // error. Exact by construction — same data, exact scan.
        if let Some(index) = first_non_finite(lu.as_slice()) {
            return Err(LuError::NonFiniteMatrix { index });
        }
        debug_assert!(
            false,
            "checked fold signalled non-finite but rescan found none"
        );
    }
    let mut pivots: Vec<usize> = (0..n).collect();
    let pivot_tolerance = T::Real::epsilon()
        * T::Real::from_usize(n.max(1)).unwrap_or_else(T::Real::one)
        * matrix_scale.max(T::Real::min_positive_value());

    // Measurements against the former scalar kernel show that packing does
    // not amortize below roughly 192×192 on Apple silicon. A single full-width
    // panel is exactly the old unblocked algorithm for those small matrices.
    //
    // Single-threaded runs use a double-width panel: the thin-k kernel climbs
    // from ~35 to ~41 GF from k=32 to k=64 (measured), and with no parallel
    // Schur there is no serial-panel penalty to offset it (gap 1.46 -> 1.39
    // at n=2048). Threaded f64 runs take 48, other types 32 (see the
    // 2026-09-27 adjudication below): lookahead hides the panels, so the
    // trade is dispatch count + GEMM fatness against caller-serial
    // lookahead-update exposure, and 48 is the sweet spot. Block
    // boundaries never change the pivot sequence (partial pivoting
    // decides per column); values move in the last ulp with nb (exact
    // comparison 2026-09-27: identical pivots, maxdiff 7.3e-12 at
    // 2048) — residuals govern, not bitwise equality.
    // Merge 2026-09-28: main's thread here shipped nb64-default; kept
    // the nb48/32 adjudication below pending re-measurement on the
    // merged kernels.
    // Re-A/B'd 2026-09-26
    // under the funnel via a temporary env hook (removed): nb=64 ties nb=32
    // (12.6-12.7 interleaved x4 at n=1024 8T) — kernel gains offset the
    // doubled serial; 32 stands, do not retry. Re-trialled once more post
    // strided-Schur/ger/scan/scale wins by flipping the const directly:
    // nb=64+new-kernels ties nb=32-without-them, i.e. nb=64 alone loses
    // ~1ms once the wins are held constant (f32 ties). Verdict confirmed.
    //
    // NB=64 RE-TRIALLED POST-LOOKAHEAD 2026-09-26 (temp `SONIUM_LU_NB`
    // hook, removed): lookahead hides panels, so dispatch-halving was
    // retested — la64 loses 1.08x vs la32 at 2048 (base64 ties base32,
    // reproducing the old verdict). Bigger panels overflow the
    // per-panel Schur-rest cover late plus 4x serial lookahead update.
    //
    // NB64 FLIP FALSIFIED 2026-09-27 (Stage-1 follow-up FAIL cell):
    // the 6/6 nb64 claim above reproduced NOWHERE — validator 11+/11+
    // pairs nb32 (2 loaded runs) plus worker re-adjudication 24/26 pairs
    // nb32 at 2048 over three windows (quiet 8/8 med 40.4 vs 46.4,
    // loaded 6/8 + 10/10; order-rotated, RAYON=16, funnel on) and a
    // 3/3 tiny-pool split (nb32 median better, nb64 higher variance).
    // nb64 never wins a window median in any config; the claimed 35.0
    // absolute was never approached by either party (best observed
    // nb32 ~39.6). Reverted: the shared default is 32 again.
    //
    // NB48 SHIPPED for threaded f64 2026-09-27 (same campaign): nb=48
    // beats nb=32 ~1.05x at dgetrf-2048 in four interleaved windows
    // (med 38.3/39.0/42.7/41.4 vs 39.9/41.1/44.7/42.8; head-to-head
    // pairs 7/10 then 9/10), ties at 1024 (5/5, med 14.85 vs 15.03),
    // wins at 512 (6/8, 3.28 vs 3.42). nb in {16,24} lose to dispatch cost
    // (48.3/43.5 vs 39.9), nb40 ties 32, nb56 loses. Mechanism: 43
    // panels instead of 64 (fewer TRSM/Schur dispatches, fatter k=48
    // GEMMs) while the caller-serial lookahead update stays covered;
    // at nb=64 the doubled serial exposure outweighs the savings.
    // f32 keeps 32 (mixed: 48 wins 2048 6/8 but marginally loses 1024
    // 3/5); complex stays 32 unmeasured. The 09-26 "do not retry 48"
    // verdict was pre-lookahead (serial panels dominated, L2 spill);
    // lookahead hides panels, so it no longer applies.
    //
    // CORRECTION 2026-09-27: the "bitwise-identical across nb" claim
    // above is FALSE — exact element comparison shows last-ulp diffs
    // (2048: 3.0M elems, maxdiff 7.3e-12; 1024: 0.66M, 2.3e-12) with
    // identical pivots. Block boundaries move update association in
    // the last ulp, exactly like the serial/parallel Schur split;
    // residuals govern, not bitwise equality. (The flip-era check
    // compared checksums at 6-digit print precision.)
    //
    // MERGE 2026-09-28: main's thread shipped the opposite verdict
    // (nb64-default, 6/6 pairs med ~35.0 vs ~37.6, "bitwise-identical
    // factors") on pre-batch kernels. RE-ADJUDICATED on merged kernels
    // (post serial-gate fix, RAYON=16): 1024 nb48 ≈ nb32 (10.15/10.59)
    // beats nb64 (11.81, erratic 9-13); 2048 nb64 ≈ nb48 (37.66/38.45,
    // within noise) beats nb32 (43.56; pairs 8/9 and 7/9 vs 32).
    // nb48-f64/32-other STANDS (wins-or-ties everywhere, stable;
    // nb64 disqualified by 1024 instability). Main's nb64 6/6 does
    // not reproduce here.
    //
    // F32 BAND 2026-10-02 (see [`lu_f32_block_size`]): the "f32 keeps
    // 32" verdict above is superseded at 2048 — nb64 wins ~1.09x over
    // two interleaved windows (9/12 pairs; halves dispatches, panel
    // stays L2-resident) while losing at 1024/2304+/2560/4096, hence
    // the two-sided 2048 <= n < 2304 band. f64 adjudication unchanged;
    // complex stays 32 unmeasured.
    // Runtime nb override (diagnostic/A-B + kill-switch:
    // `SONIUM_LU_NB=32` restores the all-type 32 geometry). Read once
    // per factorization, like `SONIUM_LU_NOFUNNEL`.
    let nb_override = std::env::var("SONIUM_LU_NB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (8..=256).contains(v));
    // `lu_prefers_tiny_pool` is the f64-only discriminator (default
    // false; only the f64 impl enables it): threaded f64 takes the
    // size-dependent 48/32 default. `lu_prefers_f32_wide_panel` is the
    // f32-only discriminator for the size-dependent 64/32 band (see
    // [`lu_f32_block_size`]); every other type keeps the shared 32.
    let threaded_default = if T::lu_prefers_tiny_pool() {
        lu_f64_block_size(n)
    } else if T::lu_prefers_f32_wide_panel() {
        lu_f32_block_size(n)
    } else {
        LU_BLOCK_SIZE
    };
    let block_size = if n < 192 {
        n.max(1)
    } else if rayon::current_num_threads() <= 1 {
        LU_BLOCK_SIZE * 2
    } else {
        nb_override.unwrap_or(threaded_default)
    };
    // Compact-panel scratch, reused across panels (grows to the first
    // panel's size and stays there; later panels are smaller).
    let mut cbuf: Vec<T> = Vec::new();
    // Pivot-row staging for the in-place small-trailing sweep, hoisted out
    // of the panel loop: stays a separate object from `lu` (the vectorizer
    // argument in the comment below), but must not malloc per panel.
    let mut k_seg: Vec<T> = Vec::new();
    // Lookahead overlap (shipped 2026-09-26): each panel factorization
    // runs on a pool worker while the caller runs the previous panel's
    // Schur remainder (see the driver branch). Bitwise identical to the
    // serial path for f32/f64; `SONIUM_LU_LOOKAHEAD=0` restores the
    // fully serial panel sequence (diagnostic/A-B, read once per
    // factorization like `SONIUM_LU_NOFUNNEL`).
    let lookahead_on = std::env::var("SONIUM_LU_LOOKAHEAD")
        .map(|v| v != "0")
        .unwrap_or(true);

    // Set when the lookahead branch prefactored the next panel: that
    // iteration skips factorization (TRSM/Schur/validate still run).
    let mut prefactored_next = false;
    // Phase timers (diagnostic, `SONIUM_LU_PHASES` set prints sums to
    // stderr): panel / TRSM / Schur (lookahead serial + remainder) /
    // validation. Same shape as `SONIUM_CHOL_PHASES`; the `Instant`
    // reads run unconditionally (tens of ns per panel, noise).
    let time_phases = std::env::var("SONIUM_LU_PHASES").is_ok();
    let (mut t_panel, mut t_trsm, mut t_schur, mut t_valid) = (
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
    );
    for panel_start in (0..n).step_by(block_size) {
        let panel_end = (panel_start + block_size).min(n);
        // Lookahead (previous iteration already factored this panel via
        // the overlapped task): skip factorization, still run TRSM,
        // Schur, and validation below.
        let skip_factorize = prefactored_next;
        prefactored_next = false;

        if !skip_factorize {
            let t0 = std::time::Instant::now();
            // Compact-panel GETF2 for large trailing sizes: gather the panel
            // columns once, factor the contiguous copy L2-resident, mirror each
            // pivot swap onto the full rows, and overlay the factored block at
            // panel end. The old in-place sweep re-touches full 16 KiB rows per
            // rank-1 step (L3-latency-bound); the compact sweep touches 32-wide
            // contiguous rows. Op order — search, swap, scale, update — is
            // unchanged, so values are bit-identical. Small trailing sizes skip
            // the gather/overlay round-trip and use the in-place sweep, which is
            // already cache-resident there.
            if n - panel_start >= COMPACT_PANEL_THRESHOLD {
                compact_panel_factorize(
                    &mut lu,
                    panel_start,
                    panel_end,
                    &mut pivots,
                    pivot_tolerance,
                    &mut cbuf,
                    &mut workspace.panel_kseg,
                    None,
                )?;
            } else {
                // Unblocked GETF2 on A[panel_start.., panel_start..panel_end].
                // Updates are deliberately restricted to the panel; the rest of the
                // Schur complement is handled by the BLAS-3 update below.
                //
                // Vectorizer-friendly GETF2: the pivot row segment is staged into
                // `k_seg` (a separate object), and trailing rows iterate via one
                // checked chunking — so the inner update reads shared `k_seg` and
                // writes an exclusive `i_row`, which LLVM may freely vectorize
                // (raw pointers would force a may-alias assumption). The operation
                // sequence — pivot search, full-row swap, scale, update — is
                // unchanged, so values are bit-identical to the accessor form.
                for k in panel_start..panel_end {
                    let data = lu.as_mut_slice();
                    let mut max_val = component_max_norm(data[k * n + k]);
                    let mut max_row = k;
                    for i in (k + 1)..n {
                        let val = component_max_norm(data[i * n + k]);
                        if val > max_val {
                            max_val = val;
                            max_row = i;
                        }
                    }
                    if max_val <= pivot_tolerance {
                        return Err(LuError::SingularMatrix);
                    }

                    if max_row != k {
                        // Full-row swap, exactly `DenseMatrix::swap_rows`.
                        let (lo, hi) = if k < max_row {
                            (k, max_row)
                        } else {
                            (max_row, k)
                        };
                        let (left, right) = data.split_at_mut(hi * n);
                        left[lo * n..(lo + 1) * n].swap_with_slice(&mut right[..n]);
                        pivots.swap(k, max_row);
                    }

                    let pivot_inv = data[k * n + k].inv();
                    k_seg.clear();
                    k_seg.extend_from_slice(&data[k * n + k + 1..k * n + panel_end]);
                    let (_, rest) = data.split_at_mut((k + 1) * n);
                    for i_row in rest.chunks_exact_mut(n) {
                        let mult = i_row[k] * pivot_inv;
                        i_row[k] = mult;
                        let dst = &mut i_row[k + 1..panel_end];
                        for (slot, &kval) in dst.iter_mut().zip(k_seg.iter()) {
                            *slot -= mult * kval;
                        }
                    }
                }
            } // end in-place (small-trailing) panel path
            t_panel += t0.elapsed();
        } // end factorize (skipped when the lookahead prefactored it)
        if panel_end == n {
            continue;
        }

        // U12 = L11^-1 * A12, solved in place in the factor by the BLAS
        // left-TRSM (`trsm_u12_unit_lower`): L11 has an implicit unit
        // diagonal (the panel stores multipliers below, pivots on, the
        // diagonal), so no diagonal division appears — matching the
        // previous row-AXPY arithmetic up to FMA rounding, with residuals
        // as the oracle. One dispatch per panel replaces the micro-block
        // AXPY/`schur_sub` loop plus its `trsm_panel` pack and copy-back;
        // solved rows sit in the factor before the pack below and the
        // next panel's pivot mirroring, matching the original timing.
        // Pack-first (pack A12, solve contiguous, copy U back) re-trialled
        // 2026-09-26 under funnel+pinning via a temporary env gate
        // (removed; bitwise-identical factors verified): the contiguous
        // band solve wins ~0.13ms but the copy-back gives it all back —
        // 5-pair interleaved dgetrf-1024 8T ties with worse pack-first
        // tails. The old pre-funnel -2% verdict stands; do not retry.
        let trailing = n - panel_end;
        let panel_width = panel_end - panel_start;
        let t0 = std::time::Instant::now();
        trsm_panel(&mut lu, &mut workspace.l11_pack, n, panel_start, panel_end);
        t_trsm += t0.elapsed();
        // A22 -= L21 * U12, in place (see `banded_schur::schur_update_strided`):
        // U12 was solved in place by the TRSM above and L21 sits in the
        // factor, so the GEMM reads both blocks strided (`lda = ldb = n`)
        // and packs internally per task — no u12/l21 pack copies (shipped
        // 2026-09-26: ~0.5ms faster than packed on dgetrf-1024, 10/11
        // interleaved pairs; sgetrf ties — the f32 pack cost was smaller;
        // the packed `schur_update` was deleted). Large updates stay on
        // the narrow pool: wide (2-CCX) wins big trailings standalone
        // (992: 149 -> 176GF) but loses small ones (384: 104 -> 83GF),
        // nested `run_compact` runs inline so per-call switching needs
        // driver restructuring, and remote-data + migration eat the
        // <=0.2ms theoretical prize. Do not retry without all three.
        // Lookahead overlap (shipped 2026-09-26, `SONIUM_LU_LOOKAHEAD=0`
        // disables): when at least two panels remain, update the next
        // panel's block column serially,
        // then factorize that panel on a pool worker while the caller
        // runs the Schur remainder itself (same dispatch environment as
        // the serial path — the caller thread keeps participating, so
        // the remainder loses no width). The panel's wide pivot swaps
        // are recorded and applied after the join; the serial block
        // update is bitwise identical to the parallel kernel per
        // element and the deferred swaps hit the same values in the
        // same order, so the factor matches the serial path bit for
        // bit (f32/f64; other types fall back to the naive serial
        // nest — correct, last-ulp possible, residuals govern).
        let next_end = (panel_end + block_size).min(n);
        // TAIL GATE FALSIFIED 2026-09-27 (temp hook, reverted):
        // disabling lookahead below trailing {128, 256} at nb48 shows no
        // gradient (med 40.39/39.96 vs 40.05 baseline, non-monotonic —
        // noise). The late-panel scope overhead never exceeds the hidden
        // panel cost; do not retry.
        let lookahead_here = lookahead_on
            && next_end == panel_end + block_size
            && next_end < n
            && n - panel_end >= COMPACT_PANEL_THRESHOLD;
        let t0 = std::time::Instant::now();
        if lookahead_here {
            let next_width = block_size;
            crate::banded_schur::schur_update_lookahead_block(
                lu.as_mut_slice(),
                n,
                panel_start,
                panel_end,
                panel_width,
                next_width,
            );
            let mut swap_log: Vec<(usize, usize)> = Vec::with_capacity(next_width);
            let mut panel_result: Option<Result<(), LuError>> = None;
            {
                let lu_len = lu.as_slice().len();
                let lu_base = lu.as_mut_slice().as_mut_ptr() as usize;
                rayon::scope(|s| {
                    s.spawn(|_| {
                        panel_result = Some(compact_panel_factorize(
                            &mut lu,
                            panel_end,
                            next_end,
                            &mut pivots,
                            pivot_tolerance,
                            &mut cbuf,
                            &mut workspace.panel_kseg,
                            Some(&mut swap_log),
                        ));
                    });
                    // SAFETY: the Schur remainder writes rows
                    // `[panel_end..n)` x columns
                    // `[panel_end + next_width..n)` and reads L21
                    // (panel columns below) plus U12rest (panel rows,
                    // remainder columns) — all disjoint from the panel
                    // task's region (next-panel columns, same rows) and
                    // its scratch (`cbuf`, `k_seg`, `swap_log`). The
                    // raw slice is reconstructed from the base pointer
                    // because both borrows cannot coexist statically.
                    let schur_slice =
                        unsafe { std::slice::from_raw_parts_mut(lu_base as *mut T, lu_len) };
                    crate::banded_schur::schur_update_strided_offset(
                        schur_slice,
                        n,
                        panel_end,
                        trailing,
                        panel_width,
                        next_width,
                    );
                });
            }
            // Panel errors (e.g. singularity) propagate exactly as in
            // the serial path.
            panel_result.expect("lookahead panel task ran in scope")?;
            apply_deferred_lu_swaps(&mut lu, &swap_log, panel_end, next_end);
            // The next panel is already factored (overlay in place,
            // swaps applied): its iteration skips factorization but
            // still runs TRSM, Schur, and validation.
            prefactored_next = true;
        } else {
            crate::banded_schur::schur_update_strided(
                lu.as_mut_slice(),
                n,
                panel_end,
                trailing,
                panel_width,
            );
        }
        t_schur += t0.elapsed();
        // Validate this panel's rows now, while hot — shipped
        // 2026-09-26 (6/6 interleaved pairs at n=2048, −0.8ms; 4/6 at
        // n=1024 on a hot box, mechanistically neutral-or-better).
        // Rows [panel_start, panel_end) are final at this point (later
        // panels only touch rows >= their own panel_end > panel_end;
        // pivot swaps only permute trailing rows >= panel_start), so a
        // per-panel scan here plus the last panel after the loop covers
        // every row exactly once, in order — same first-bad index as the
        // old cold end-of-factorize pass, which this replaces.
        let t0 = std::time::Instant::now();
        validate_generated_slice(
            &lu.as_slice()[panel_start * n..panel_end * n],
            "factorization",
            panel_start * n,
        )?;
        t_valid += t0.elapsed();
    }

    // Last panel (broke out before its TRSM/schur): validate its rows
    // (empty slice when n == 0, matching the old endscan).
    let last_start = if n == 0 {
        0
    } else {
        ((n - 1) / block_size) * block_size
    };
    validate_generated_slice(
        &lu.as_slice()[last_start * n..],
        "factorization",
        last_start * n,
    )?;

    if time_phases {
        eprintln!(
            "lu phases panel {:.3}ms trsm {:.3}ms schur {:.3}ms valid {:.3}ms",
            t_panel.as_secs_f64() * 1000.0,
            t_trsm.as_secs_f64() * 1000.0,
            t_schur.as_secs_f64() * 1000.0,
            t_valid.as_secs_f64() * 1000.0
        );
    }
    Ok(LuFactorization { lu, pivots, n })
}

/// Solve Ax = b using LU decomposition
///
/// This is a convenience function that combines factorization and solve.
pub fn lu_solve<T: ComplexField>(a: &DenseMatrix<T>, b: &[T]) -> Result<Vec<T>, LuError> {
    let factor = lu_factorize(a)?;
    factor.solve(b)
}

/// Solve Aᵀx = b using LU decomposition.
///
/// Convenience wrapper around [`LuFactorization::solve_transpose`] for
/// callers that do not retain the factorization.
pub fn lu_solve_transpose<T: ComplexField>(a: &DenseMatrix<T>, b: &[T]) -> Result<Vec<T>, LuError> {
    let factor = lu_factorize(a)?;
    factor.solve_transpose(b)
}

/// Solve Aᴴx = b using LU decomposition
///
/// Convenience wrapper around [`LuFactorization::solve_conj_transpose`] for
/// callers that do not retain the factorization.
pub fn lu_solve_conj_transpose<T: ComplexField>(
    a: &DenseMatrix<T>,
    b: &[T],
) -> Result<Vec<T>, LuError> {
    let factor = lu_factorize(a)?;
    factor.solve_conj_transpose(b)
}

/// Outcome of [`lu_solve_mixed`]: the refined solution plus convergence metadata.
#[derive(Clone, Debug)]
pub struct MixedLuSolution {
    /// Solution vector. At f64-class accuracy when `fell_back` is false
    /// (f32 factor + f64 refinement); plain f64 direct solve otherwise.
    pub x: Vec<f64>,
    /// Refinement corrections applied (0 when the initial f32 solve already
    /// meets tolerance, or on the f64-direct fallback path).
    pub iterations: usize,
    /// Final relative residual `||b − Ax|| / (||A||_F · ||x||)` (0 for exact).
    pub residual: f64,
    /// True when the f32 factor was unusable (singular/overflow) or
    /// refinement stalled and the result fell back to plain f64
    /// [`lu_solve`].
    pub fell_back: bool,
}

/// Solve Ax = b via mixed-precision LU (the LAPACK `dsgesv` recipe):
/// factor in `f32`, solve, then refine in `f64` to f64-class accuracy.
///
/// The f32 factor costs roughly half the f64 one while the residual,
/// correction solves, and convergence test all run in f64, so
/// well-conditioned systems solve at near-sgetrf speed with dgesv-class
/// accuracy (two corrections typically reach ~1e-16..1e-18 relative
/// residual on random diagonally dominant matrices).
///
/// Any f32-stage failure (factor error, non-finite intermediate, stalled
/// residual) falls back to plain f64 [`lu_solve`] (`fell_back` set), so
/// user-input errors surface with the same [`LuError`] variants as the
/// direct solve and the answer is never worse than it (modulo the wasted
/// f32 attempt).
pub fn lu_solve_mixed(a: &DenseMatrix<f64>, b: &[f64]) -> Result<MixedLuSolution, LuError> {
    /// Acceptance tolerance: f64-class residual, safely above the f64
    /// noise floor so genuinely converged systems stop after 1-2 steps.
    const TOL: f64 = 1e-13;
    /// Refinement corrections after the initial solve. Two suffice for
    /// well-conditioned systems (1e-8 initial -> ~1e-16 -> ~1e-18); the
    /// third covers slower starts before falling back.
    const MAX_ITERS: usize = 3;
    let n = a.nrows();
    if b.len() != n {
        return Err(LuError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // Fallback helper: plain f64 direct solve with its residual attached.
    // Used for every f32-stage failure, so user-input errors (non-finite
    // entries, dimension faults past the `b` check above) surface exactly
    // as [`lu_solve`] reports them.
    let fallback = |a: &DenseMatrix<f64>, b: &[f64]| -> Result<MixedLuSolution, LuError> {
        let x = lu_solve(a, b)?;
        Ok(MixedLuSolution {
            residual: mixed_precision_residual(a, &x, b),
            x,
            iterations: 0,
            fell_back: true,
        })
    };
    // f32 factor of the downcast matrix. `as` truncates toward zero; the
    // refinement loop corrects the rounding either way.
    let a32: Vec<f32> = a.as_slice().iter().map(|&v| v as f32).collect();
    let m32 = DenseMatrix::from_row_major_vec(n, n, a32);
    let factor32 = match lu_factorize(&m32) {
        Ok(f) => f,
        Err(_) => return fallback(a, b),
    };
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let x0low = match factor32.solve(&b32) {
        Ok(x) => x,
        Err(_) => return fallback(a, b),
    };
    let mut x: Vec<f64> = x0low.iter().map(|&v| v as f64).collect();
    if x.iter().any(|v| !v.is_finite()) {
        return fallback(a, b);
    }
    let mut prev = mixed_precision_residual(a, &x, b);
    if !prev.is_finite() {
        return fallback(a, b);
    }
    if prev <= TOL {
        return Ok(MixedLuSolution {
            residual: prev,
            x,
            iterations: 0,
            fell_back: false,
        });
    }
    for i in 1..=MAX_ITERS {
        let ax = a.matvec(&x);
        let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bb, &a)| bb - a).collect();
        let r32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let dlow = match factor32.solve(&r32) {
            Ok(d) => d,
            Err(_) => return fallback(a, b),
        };
        for (xx, &dd) in x.iter_mut().zip(dlow.iter()) {
            *xx += dd as f64;
        }
        if x.iter().any(|v| !v.is_finite()) {
            return fallback(a, b);
        }
        let cur = mixed_precision_residual(a, &x, b);
        if !cur.is_finite() {
            return fallback(a, b);
        }
        if cur <= TOL {
            return Ok(MixedLuSolution {
                residual: cur,
                x,
                iterations: i,
                fell_back: false,
            });
        }
        if cur >= prev {
            // No progress: the f32 factor is too poor a preconditioner
            // (ill-conditioning past f32 range). Fall back rather than
            // burn more steps.
            return fallback(a, b);
        }
        prev = cur;
    }
    // Budget exhausted above tolerance.
    fallback(a, b)
}

/// Relative residual `||b − Ax|| / (||A||_F · ||x||)` (0 when exact,
/// including the empty system). Named apart from the test helper
/// `relative_residual` below (`||r|| / ||b||` form) to avoid shadowing.
fn mixed_precision_residual(a: &DenseMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let ax = a.matvec(x);
    let mut r2 = 0.0f64;
    let mut x2 = 0.0f64;
    for (xx, (&bb, &aa)) in x.iter().zip(b.iter().zip(ax.iter())) {
        let r = bb - aa;
        r2 += r * r;
        x2 += xx * xx;
    }
    if r2 == 0.0 {
        return 0.0;
    }
    let mut a2 = 0.0f64;
    for &v in a.as_slice() {
        a2 += v * v;
    }
    r2.sqrt() / (a2.sqrt() * x2.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn dm<T: ComplexField>(rows: usize, cols: usize, data: Vec<T>) -> DenseMatrix<T> {
        DenseMatrix::from_row_major_vec(rows, cols, data)
    }

    fn deterministic_matrix(n: usize) -> DenseMatrix<f64> {
        DenseMatrix::from_fn(n, n, |i, j| {
            let off_diag = (((i * 37 + j * 17 + 11) % 101) as f64 - 50.0) / 101.0;
            if i == j {
                off_diag + n as f64
            } else {
                off_diag
            }
        })
    }

    fn relative_residual(a: &DenseMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let ax = a.matvec(x);
        let residual = ax
            .iter()
            .zip(b)
            .map(|(lhs, rhs)| (lhs - rhs) * (lhs - rhs))
            .sum::<f64>()
            .sqrt();
        let rhs_norm = b.iter().map(|value| value * value).sum::<f64>().sqrt();
        residual / rhs_norm.max(f64::MIN_POSITIVE)
    }

    fn assert_factor_reconstruction<T: ComplexField<Real = f64>>(
        a: &DenseMatrix<T>,
        factor: &LuFactorization<T>,
        tolerance: f64,
    ) {
        let n = a.nrows();
        let mut max_error = 0.0_f64;
        let mut max_a = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let mut lu_ij = T::zero();
                for k in 0..n {
                    let l = if i == k {
                        T::one()
                    } else if i > k {
                        factor.lu[[i, k]]
                    } else {
                        T::zero()
                    };
                    let u = if k <= j { factor.lu[[k, j]] } else { T::zero() };
                    lu_ij += l * u;
                }
                let pa_ij = a[[factor.pivots[i], j]];
                max_error = max_error.max((lu_ij - pa_ij).norm());
                max_a = max_a.max(pa_ij.norm());
            }
        }
        assert!(
            max_error <= tolerance * max_a.max(1.0),
            "factor reconstruction error {max_error:e} exceeds tolerance"
        );
    }

    #[test]
    fn large_threaded_reconstruction_n1024() {
        // n=1024 exercises the threaded default-width path (f64: 48-wide
        // since 2026-09-27) with lookahead; reconstruction must hold at
        // LU accuracy. 1024 = 21*48 + 16, so the short last panel runs too.
        let n = 1024;
        let a = deterministic_matrix(n);
        let factor = lu_factorize(&a).unwrap();
        assert_factor_reconstruction(&a, &factor, 1e-12);
    }

    #[test]
    fn large_threaded_solve_n2048() {
        // n=2048 end-to-end through the full dispatcher: 48-wide panels,
        // parallel lookahead (early trailings exceed the 1024-row floor),
        // and the tiny-pool funnel. O(n^2) residual check (the O(n^3)
        // reconstruction above would dominate the suite at this size).
        let n = 2048;
        let a = deterministic_matrix(n);
        let x_true: Vec<f64> = (0..n).map(|i| 0.5 + 0.001 * i as f64).collect();
        let b = a.matvec(&x_true);
        let x = lu_solve(&a, &b).unwrap();
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (got, want) in x.iter().zip(x_true.iter()) {
            num += (got - want) * (got - want);
            den += want * want;
        }
        let forward_error = (num.sqrt()) / den.sqrt();
        assert!(
            forward_error < 1e-9,
            "n=2048 forward error {forward_error:e} exceeds tolerance"
        );
    }

    #[test]
    fn large_threaded_solve_n2048_f32() {
        // f32 twin inside the 2048 <= n < 2304 nb64 band: the wide panel
        // path gets end-to-end solve coverage here, not just the width
        // pinning above.
        let n = 2048;
        assert_eq!(lu_f32_block_size(n), 64);
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            let off = (((i * 37 + j * 17 + 11) % 101) as f32 - 50.0) / 101.0;
            if i == j {
                off + n as f32
            } else {
                off
            }
        });
        let x_true: Vec<f32> = (0..n).map(|i| 0.5 + 0.001 * i as f32).collect();
        let b = a.matvec(&x_true);
        let x = lu_solve(&a, &b).unwrap();
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (got, want) in x.iter().zip(x_true.iter()) {
            num += (f64::from(*got) - f64::from(*want)).powi(2);
            den += f64::from(*want).powi(2);
        }
        let forward_error = num.sqrt() / den.sqrt();
        assert!(
            forward_error < 1e-3,
            "f32 n=2048 forward error {forward_error:e} exceeds tolerance"
        );
    }

    #[test]
    fn compact_pool_matches_mirror_funnel_within_ulp() {
        // d1024-class factors route through the compact pool when the
        // estimate fits; pool-geometry TRSM band splits may re-chunk at
        // last-ulp level (admitted, residuals govern), so agreement is
        // pivots-equal plus a tight max-diff bound, not bitwise.
        let n = 1024;
        let a = deterministic_matrix(n);
        let est = n as u64 * n as u64 * 8;
        let via_compact = sonium_compute::cpu::blas3::run_compact(est, false, || {
            let mut workspace = LuWorkspace::new();
            lu_factorize_in_place_direct(clone_into_factor_buffer(&a), &mut workspace).unwrap()
        });
        let owned = clone_into_factor_buffer(&a);
        let via_mirror = sonium_compute::cpu::blas3::run_on_worker(move || {
            let mut workspace = LuWorkspace::new();
            lu_factorize_in_place_direct(owned, &mut workspace).unwrap()
        });
        assert_eq!(via_compact.pivots, via_mirror.pivots);
        let mut max_diff = 0.0f64;
        let mut max_abs = 0.0f64;
        for (x, y) in via_compact
            .lu
            .as_slice()
            .iter()
            .zip(via_mirror.lu.as_slice())
        {
            max_diff = max_diff.max((x - y).norm());
            max_abs = max_abs.max(x.norm());
        }
        assert!(
            max_diff <= 1e-12 * max_abs.max(1.0),
            "compact-vs-mirror max diff {max_diff:e} (max abs {max_abs:e}) exceeds last-ulp envelope"
        );
    }

    #[test]
    fn threaded_default_panel_widths_pinned() {
        // The threaded defaults are measured optima (driver `block_size`
        // note, 2026-09-27 adjudication: f64 48 beats 32 ~1.05x at 2048,
        // 64 loses everywhere, f32/complex keep 32; 2026-10-02 f32 band:
        // 64 wins ~1.09x at 2048 only, loses at 1024/2304+/2560/4096).
        // Pin them so a future flip is a deliberate test + note update,
        // not a silent const edit — the nb=64 flip that motivated this
        // test shipped on a 6/6 claim that reproduced nowhere.
        assert_eq!(LU_BLOCK_SIZE, 32, "shared threaded default");
        assert_eq!(LU_BLOCK_SIZE_F64, 48, "threaded f64 default");
        assert_eq!(LU_BLOCK_SIZE_F32_WIDE, 64, "threaded f32 wide band");
        // Size split (2026-10-01 boundary sweep): 48 through 2816, 32
        // from 3072 up.
        assert_eq!(lu_f64_block_size(2048), 48);
        assert_eq!(lu_f64_block_size(2816), 48);
        assert_eq!(lu_f64_block_size(3072), 32);
        assert_eq!(lu_f64_block_size(4096), 32);
        // f32 two-sided band (2026-10-02 interleaved A/B): 64 inside
        // 2048 <= n < 2304, 32 elsewhere.
        assert_eq!(lu_f32_block_size(1024), 32);
        assert_eq!(lu_f32_block_size(2048), 64);
        assert_eq!(lu_f32_block_size(2304), 32);
        assert_eq!(lu_f32_block_size(4096), 32);
    }

    #[test]
    fn lookahead_block_route_pinned() {
        // Parallel lookahead-block route (2026-10-02 interleaved A/B:
        // dgetrf-4096 wall -4.1% 4/4 pairs, dgetrf-2048 -5.3% 6/6
        // pairs): f64-only — f32 loses up to +80% on the parallel route
        // (sgemm decomposition pathology, unprobed), complex unmeasured.
        // The 1024-row threshold keeps f64 n <= 1024 on the
        // byte-identical serial path. Pin all three so a future flip is
        // a deliberate test + note update, not a silent edit.
        assert!(
            f64::lu_prefers_parallel_lookahead_update(),
            "f64 takes the parallel lookahead-block route"
        );
        assert!(
            !f32::lu_prefers_parallel_lookahead_update(),
            "f32 stays serial (parallel route regresses)"
        );
        assert!(
            !Complex64::lu_prefers_parallel_lookahead_update(),
            "complex stays serial pending measurement"
        );
        assert_eq!(
            crate::banded_schur::LOOKAHEAD_PAR_MIN_ROWS,
            1024,
            "parallel lookahead-block row threshold"
        );
    }

    #[test]
    fn lookahead_block_matches_serial_bitwise() {
        // Route invariance: the lookahead block update reproduces the
        // serial kernel bit for bit on both routes — f64 above the MAC
        // floor (parallel M-split: 1344x48x48 = 3.1M MACs) and f32
        // (serial by discriminator). The f64 arm forces the parallel
        // kernel directly, so the test stays meaningful under ambient
        // `SONIUM_LU_LAPAR=0` (which would otherwise make it a vacuous
        // serial-vs-serial comparison).
        let n = 1408usize;
        let (ps, pe, pw, nw) = (16usize, 64usize, 48usize, 48usize);
        let rows = n - pe;
        assert!(rows >= crate::banded_schur::LOOKAHEAD_PAR_MIN_ROWS);
        assert!(rows * nw * pw >= 3_000_000, "above the MAC floor");
        let base: Vec<f64> = (0..n * n)
            .map(|i| (((i * 37 + 11) % 101) as f64 - 50.0) / 101.0)
            .collect();
        let mut got = base.clone();
        crate::banded_schur::schur_update_lookahead_block_parallel(&mut got, n, ps, pe, pw, nw);
        // Hand-sliced serial reference (same footprints as
        // `banded_schur`; A/B cloned because the kernel borrows
        // them independently of the C slice).
        let mut expect = base.clone();
        let a_owned = expect[pe * n + ps..pe * n + ps + (rows - 1) * n + pw].to_vec();
        let b_owned = expect[ps * n + pe..ps * n + pe + (pw - 1) * n + nw].to_vec();
        let c = &mut expect[pe * n + pe..pe * n + pe + (rows - 1) * n + nw];
        f64::schur_sub_serial(rows, nw, pw, &a_owned, n, &b_owned, n, c, n);
        assert_eq!(got, expect, "f64 lookahead block must match serial bitwise");
        // f32 twin (serial route by discriminator; locks the slicing for
        // the second precision).
        let base32: Vec<f32> = base.iter().map(|&v| v as f32).collect();
        let mut got32 = base32.clone();
        crate::banded_schur::schur_update_lookahead_block(&mut got32, n, ps, pe, pw, nw);
        let mut expect32 = base32.clone();
        let a32 = expect32[pe * n + ps..pe * n + ps + (rows - 1) * n + pw].to_vec();
        let b32 = expect32[ps * n + pe..ps * n + pe + (pw - 1) * n + nw].to_vec();
        let c32 = &mut expect32[pe * n + pe..pe * n + pe + (rows - 1) * n + nw];
        f32::schur_sub_serial(rows, nw, pw, &a32, n, &b32, n, c32, n);
        assert_eq!(got32, expect32, "f32 lookahead block must match serial bitwise");
    }

    #[test]
    fn factor_buffer_pool_push_pop() {
        // Pool mechanics directly: push a buffer, pop it back (hit), pop
        // again (miss). Non-square length: no factorization in this suite
        // can request it, so parallel tests cannot interfere.
        let len = 200000usize; // non-square, >= 1MB pool floor
        // Drain any same-length buffer first (defensive; none expected).
        while factor_pool_pop::<f64>(len).is_some() {}
        let hits_before = FACTOR_POOL_HITS.load(std::sync::atomic::Ordering::Relaxed);
        factor_pool_push(vec![1.0f64; len]);
        // Below-floor buffers are not pooled.
        factor_pool_push(vec![1.0f64; 16]);
        let back = factor_pool_pop::<f64>(len).expect("pushed buffer pops");
        assert_eq!(back.len(), 0, "pooled buffers come back cleared");
        assert_eq!(back.capacity(), len);
        assert!(
            FACTOR_POOL_HITS.load(std::sync::atomic::Ordering::Relaxed) > hits_before,
            "expected a pool hit"
        );
        assert!(factor_pool_pop::<f64>(len).is_none(), "pool drained");
        // Wrong-type slot never aliases (f32 has its own pool).
        factor_pool_push(vec![1.0f32; len]);
        assert!(factor_pool_pop::<f64>(len).is_none());
    }

    #[test]
    fn factor_buffer_pool_reuses_same_size_storage() {
        // Integration: repeated same-size `lu_factorize` calls recycle
        // the still-warm pages (drop pushes, next checkout pops). Factors
        // reconstruct identically through the pooled path; a different
        // size exercises the miss path.
        let m = deterministic_matrix(400);
        let f1 = lu_factorize(&m).expect("factorize");
        assert_factor_reconstruction(&m, &f1, 1e-9);
        drop(f1);
        let f2 = lu_factorize(&m).expect("refactorize");
        assert_factor_reconstruction(&m, &f2, 1e-9);
        let m2 = deterministic_matrix(401);
        let f3 = lu_factorize(&m2).expect("factorize other size");
        assert_factor_reconstruction(&m2, &f3, 1e-9);
    }

    #[test]
    fn test_lu_solve_real() {
        let a = dm(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
        let b = vec![1.0_f64, 2.0];

        let x = lu_solve(&a, &b).expect("LU solve should succeed");

        // Verify: Ax = b
        let ax = a.matvec(&x);
        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lu_solve_mixed_matches_direct() {
        // Well-conditioned: the f32 factor refines to f64-class accuracy
        // without fallback.
        let n = 128;
        let a = deterministic_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 1.0).collect();

        let mixed = lu_solve_mixed(&a, &b).expect("mixed solve should succeed");
        assert!(
            !mixed.fell_back,
            "well-conditioned system must not fall back"
        );
        assert!(
            mixed.iterations <= 3,
            "must converge within budget, took {}",
            mixed.iterations
        );
        assert!(
            mixed.residual <= 1e-13,
            "residual {} exceeds f64-class tolerance",
            mixed.residual
        );
        // Independent residual check with the test-module norm form.
        assert!(relative_residual(&a, &mixed.x, &b) <= 1e-13);
        // Same ballpark as the direct solve (both approximate A⁻¹b; the
        // bound sits far above either forward error).
        let direct = lu_solve(&a, &b).expect("direct solve should succeed");
        let diff = mixed
            .x
            .iter()
            .zip(direct.iter())
            .map(|(m, d)| (m - d) * (m - d))
            .sum::<f64>()
            .sqrt();
        let scale = direct.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
        assert!(diff <= 1e-9 * scale, "mixed/direct differ by {diff:e}");
    }

    #[test]
    fn test_lu_solve_mixed_falls_back_on_f32_singular() {
        // Diagonal with a 1e-10 last pivot: diagonal LU has no elimination
        // dynamics, so the pivot reaches the tolerance check verbatim. The
        // f32 pivot floor (~eps·n·scale ≈ 1e-6) rejects it while the f64
        // floor (~1e-15) accepts, forcing the fallback path deterministically.
        // (Near-duplicate rows were tried first and cannot separate the
        // precisions: their leftover after elimination is second-order in
        // the perturbation, so both precisions agree.)
        let n = 8;
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        data[(n - 1) * n + (n - 1)] = 1e-10;
        let a = dm(n, n, data);
        let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 + 1.0).collect();

        let mixed = lu_solve_mixed(&a, &b).expect("fallback solve should succeed");
        assert!(
            mixed.fell_back,
            "f32-singular system must fall back to f64-direct"
        );
        assert!(mixed.x.iter().all(|v| v.is_finite()));
        assert!(mixed.residual.is_finite());
        // The fallback is the direct answer: residual at f64-direct level.
        assert!(mixed.residual <= 1e-13);
    }

    #[test]
    fn test_lu_solve_transpose_complex() {
        let a = dm(
            2,
            2,
            vec![
                Complex64::new(3.0, 0.5),
                Complex64::new(1.0, -0.25),
                Complex64::new(-0.5, 0.75),
                Complex64::new(2.0, -1.0),
            ],
        );
        let b = vec![Complex64::new(1.0, -0.5), Complex64::new(-2.0, 0.25)];
        let x = lu_solve_transpose(&a, &b).expect("transpose solve should succeed");
        let at = a.transposed();
        let atx = at.matvec(&x);
        for i in 0..2 {
            assert_relative_eq!((atx[i] - b[i]).norm(), 0.0, epsilon = 1e-10);
        }
        let factor = lu_factorize(&a).expect("factorization should succeed");
        let retained = factor
            .solve_transpose(&b)
            .expect("retained transpose solve should succeed");
        for i in 0..2 {
            assert_relative_eq!((retained[i] - x[i]).norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lu_solve_complex() {
        let a = dm(
            2,
            2,
            vec![
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, -1.0),
            ],
        );
        let b = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];

        let x = lu_solve(&a, &b).expect("LU solve should succeed");

        let ax = a.matvec(&x);
        for i in 0..2 {
            assert_relative_eq!((ax[i] - b[i]).norm(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lu_identity() {
        let n = 5;
        let a = DenseMatrix::<f64>::eye(n);
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        let x = lu_solve(&a, &b).expect("LU solve should succeed");

        for i in 0..n {
            assert_relative_eq!(x[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lu_parallel_schur_update() {
        // Large trailing blocks: exercises the strided-direct Schur update
        // (in-place L21/U12 reads, strided GEMM) for correctness.
        let n = 576;
        let a = deterministic_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let x = lu_solve(&a, &b).expect("large LU solve should succeed");
        let residual = relative_residual(&a, &x, &b);
        assert!(
            residual < 1e-10,
            "parallel Schur residual {residual:e} exceeds tolerance"
        );
    }

    #[test]
    fn lookahead_split_matches_full_schur_bitwise() {
        // Lookahead decomposition invariance (f64): serial next-panel
        // block update + column-offset remainder reproduces the full
        // strided Schur update bit for bit (same packs, same per-element
        // sweep order), so the overlapped path cannot change values.
        let n = 96usize;
        let (ps, pe, pw, nw) = (0usize, 32usize, 32usize, 32usize);
        let trailing = n - pe;
        let base: Vec<f64> = (0..n * n)
            .map(|i| (((i * 37 + 11) % 101) as f64 - 50.0) / 101.0)
            .collect();
        let mut full = base.clone();
        crate::banded_schur::schur_update_strided(&mut full, n, pe, trailing, pw);
        let mut split = base.clone();
        crate::banded_schur::schur_update_lookahead_block(&mut split, n, ps, pe, pw, nw);
        crate::banded_schur::schur_update_strided_offset(&mut split, n, pe, trailing, pw, nw);
        assert_eq!(full, split, "lookahead split must match full Schur bitwise");
        // Offset 0 reproduces the un-offset call exactly.
        let mut zero = base.clone();
        crate::banded_schur::schur_update_strided_offset(&mut zero, n, pe, trailing, pw, 0);
        assert_eq!(full, zero, "zero-offset update must match strided update");
    }

    #[test]
    fn deferred_swaps_match_full_row_swaps() {
        // Deferred-swap invariance: recorded swaps applied to the
        // non-panel columns plus panel-column swaps equal the full-row
        // swaps the inline mirror performs.
        let n = 64usize;
        let (skip_lo, skip_hi) = (16usize, 48usize);
        let base: Vec<f64> = (0..n * n)
            .map(|i| (((i * 53 + 7) % 127) as f64 - 63.0) / 127.0)
            .collect();
        let swaps = vec![(16usize, 40usize), (17, 17 + 31), (20, 63), (47, 48)];
        let mut expect = dm(n, n, base.clone());
        for &(k, m) in &swaps {
            let (lo, hi) = if k < m { (k, m) } else { (m, k) };
            let data = expect.as_mut_slice();
            let (left, right) = data.split_at_mut(hi * n);
            left[lo * n..(lo + 1) * n].swap_with_slice(&mut right[..n]);
        }
        let mut got = dm(n, n, base.clone());
        // Panel columns swapped as the overlay would leave them
        // (post-swap order), then the deferred helper for the rest.
        for &(k, m) in &swaps {
            assert!(k != m, "test swaps must actually swap");
            for c in skip_lo..skip_hi {
                let data = got.as_mut_slice();
                data.swap(k * n + c, m * n + c);
            }
        }
        apply_deferred_lu_swaps(&mut got, &swaps, skip_lo, skip_hi);
        assert_eq!(
            expect.as_slice(),
            got.as_slice(),
            "deferred swaps must match full-row swaps"
        );
    }

    #[test]
    fn deferred_swaps_tolerate_identity_pairs() {
        // Identity pivots (k == m) are no-ops: the row split below would
        // index out of bounds on them, so they must be skipped, mixed
        // with real swaps and alone.
        let n = 16usize;
        let base: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let mut only_identity = dm(n, n, base.clone());
        apply_deferred_lu_swaps(&mut only_identity, &[(3, 3), (0, 0)], 4, 12);
        assert_eq!(only_identity.as_slice(), base.as_slice());
        let mut mixed = dm(n, n, base.clone());
        apply_deferred_lu_swaps(&mut mixed, &[(2, 9), (5, 5), (9, 2)], 4, 12);
        let mut expect = dm(n, n, base.clone());
        for &(k, m) in &[(2usize, 9usize), (9, 2)] {
            let (lo, hi) = if k < m { (k, m) } else { (m, k) };
            let data = expect.as_mut_slice();
            let (left, right) = data.split_at_mut(hi * n);
            let lo_row = &mut left[lo * n..(lo + 1) * n];
            let hi_row = &mut right[..n];
            lo_row[..4].swap_with_slice(&mut hi_row[..4]);
            lo_row[12..].swap_with_slice(&mut hi_row[12..]);
        }
        assert_eq!(mixed.as_slice(), expect.as_slice());
    }

    #[test]
    fn strided_updates_tolerate_empty_dims() {
        // Zero trailing rows/widths make the `(len - 1)` footprint math
        // underflow; all three strided entry points must no-op instead.
        let n = 32usize;
        let base: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let mut m = base.clone();
        // Last panel: no trailing rows.
        crate::banded_schur::schur_update_strided(&mut m, n, n, 0, 8);
        crate::banded_schur::schur_update_lookahead_block(&mut m, n, n - 8, n, 8, 0);
        crate::banded_schur::schur_update_strided_offset(&mut m, n, n, 0, 8, 0);
        // Zero panel width / zero widths.
        crate::banded_schur::schur_update_strided(&mut m, n, 16, 16, 0);
        crate::banded_schur::schur_update_lookahead_block(&mut m, n, 16, 16, 0, 8);
        crate::banded_schur::schur_update_lookahead_block(&mut m, n, 8, 16, 8, 0);
        crate::banded_schur::schur_update_strided_offset(&mut m, n, 16, 16, 0, 0);
        // Fully consumed trailing remainder.
        crate::banded_schur::schur_update_strided_offset(&mut m, n, 16, 16, 8, 16);
        assert_eq!(m, base, "empty updates must leave the matrix untouched");
    }

    #[test]
    fn test_lu_compact_panel() {
        // Large enough that early panels cross the compact-panel threshold:
        // exercises the gather-once path (gather, compact GETF2 with
        // mirrored swaps, overlay) for correctness.
        let n = 600;
        let a = deterministic_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let x = lu_solve(&a, &b).expect("compact-panel LU solve should succeed");
        let residual = relative_residual(&a, &x, &b);
        assert!(
            residual < 1e-10,
            "compact-panel residual {residual:e} exceeds tolerance"
        );
    }

    #[test]
    fn test_lu_singular() {
        let a = dm(2, 2, vec![1.0_f64, 2.0, 2.0, 4.0]);
        let b = vec![1.0_f64, 2.0];

        let result = lu_solve(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_lu_solve_conj_transpose_real() {
        // For real matrices Aᴴ = Aᵀ; pivoting must be undone correctly.
        let a = dm(3, 3, vec![0.0_f64, 1.0, 2.0, 1.0, 0.0, 3.0, 4.0, 2.0, 1.0]);
        let b = vec![1.0_f64, 2.0, 3.0];

        let factor = lu_factorize(&a).expect("factorization should succeed");
        let x = factor
            .solve_conj_transpose(&b)
            .expect("conj-transpose solve should succeed");

        let at = a.transposed();
        let atx = at.matvec(&x);
        for i in 0..3 {
            assert_relative_eq!(atx[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lu_solve_conj_transpose_complex() {
        // Non-symmetric complex matrix with a pivot swap on the first column.
        let a = dm(
            3,
            3,
            vec![
                Complex64::new(0.0, 1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(0.0, 1.0),
                Complex64::new(5.0, -1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 2.0),
            ],
        );
        let b = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.0, -1.0),
        ];

        let factor = lu_factorize(&a).expect("factorization should succeed");
        let x = factor
            .solve_conj_transpose(&b)
            .expect("conj-transpose solve should succeed");

        // Verify Aᴴx = b against an explicit conjugate-transpose matvec.
        let mut ahx = vec![Complex64::new(0.0, 0.0); 3];
        a.matvec_hermitian_into(&x, &mut ahx);
        for i in 0..3 {
            assert_relative_eq!((ahx[i] - b[i]).norm(), 0.0, epsilon = 1e-10);
        }

        // The free function must agree with the retained-factor solve.
        let x_free = lu_solve_conj_transpose(&a, &b).expect("free solve should succeed");
        for i in 0..3 {
            assert_relative_eq!((x_free[i] - x[i]).norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lu_factorize_and_solve() {
        let a = dm(3, 3, vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);

        let factorization = lu_factorize(&a).expect("Factorization should succeed");

        let b1 = vec![1.0_f64, 2.0, 3.0];
        let x1 = factorization.solve(&b1).expect("Solve should succeed");

        let ax1 = a.matvec(&x1);
        for i in 0..3 {
            assert_relative_eq!(ax1[i], b1[i], epsilon = 1e-10);
        }

        let b2 = vec![4.0_f64, 5.0, 6.0];
        let x2 = factorization.solve(&b2).expect("Solve should succeed");

        let ax2 = a.matvec(&x2);
        for i in 0..3 {
            assert_relative_eq!(ax2[i], b2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn allocation_free_normal_transpose_and_many_rhs_solves() {
        let a = dm(3, 3, vec![0.0_f64, 2.0, 1.0, 3.0, 1.0, -1.0, 1.0, 4.0, 2.0]);
        let mut workspace = LuWorkspace::new();
        let factor = lu_factorize_in_place(a.clone(), &mut workspace).unwrap();

        let b = vec![1.0, -2.0, 3.0];
        let mut x = vec![0.0; 3];
        factor.solve_into(&b, &mut x).unwrap();
        for (actual, expected) in a.matvec(&x).iter().zip(&b) {
            assert_relative_eq!(actual, expected, epsilon = 1e-12);
        }

        factor.solve_transpose_into(&b, &mut x).unwrap();
        for (actual, expected) in a.transposed().matvec(&x).iter().zip(&b) {
            assert_relative_eq!(actual, expected, epsilon = 1e-12);
        }

        let right_hand_sides = vec![1.0, 4.0, -2.0, 5.0, 3.0, 6.0];
        let mut solutions = vec![0.0; right_hand_sides.len()];
        factor
            .solve_many_into(&right_hand_sides, &mut solutions, 2)
            .unwrap();
        for rhs in 0..2 {
            let solution: Vec<_> = (0..3).map(|row| solutions[row * 2 + rhs]).collect();
            let actual = a.matvec(&solution);
            for row in 0..3 {
                assert_relative_eq!(
                    actual[row],
                    right_hand_sides[row * 2 + rhs],
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn scale_aware_pivots_accept_tiny_well_conditioned_matrix() {
        let scale = 1e-200_f64;
        let a = dm(2, 2, vec![2.0 * scale, 0.5 * scale, 0.25 * scale, scale]);
        let expected = vec![1.0, -2.0];
        let rhs = a.matvec(&expected);
        let actual = lu_solve(&a, &rhs).unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_relative_eq!(actual, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn blocked_lu_reconstructs_across_panel_boundaries() {
        for n in [1, 2, 17, 63, 64, 65, 79, 129] {
            let a = deterministic_matrix(n);
            let factor = lu_factorize(&a).expect("factorization should succeed");
            assert_factor_reconstruction(&a, &factor, 2e-12 * n as f64);
        }
    }

    #[test]
    fn blocked_lu_solves_across_panel_boundaries() {
        for n in [17, 63, 64, 65, 129] {
            let a = deterministic_matrix(n);
            let b: Vec<f64> = (0..n).map(|i| ((i * 19 + 3) as f64).sin() + 0.25).collect();
            let factor = lu_factorize(&a).expect("factorization should succeed");
            let x = factor.solve(&b).expect("solve should succeed");
            assert!(
                relative_residual(&a, &x, &b) < 5e-13 * n as f64,
                "large residual at n={n}"
            );
        }
    }

    #[test]
    fn blocked_lu_handles_pivots_in_multiple_panels() {
        let n = 131;
        let mut a = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            a[[i, (i + 1) % n]] = 2.0;
            a[[i, i]] = 0.25;
            a[[i, (i + n - 1) % n]] = -0.125;
        }
        let factor = lu_factorize(&a).expect("pivot-heavy matrix should factorize");
        assert!(
            factor
                .pivots
                .iter()
                .enumerate()
                .any(|(i, pivot)| i != *pivot)
        );
        assert_factor_reconstruction(&a, &factor, 5e-12 * n as f64);
        let b: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.5).collect();
        let x = factor.solve(&b).expect("solve should succeed");
        assert!(relative_residual(&a, &x, &b) < 1e-11);
    }

    #[test]
    fn blocked_lu_complex_reconstructs_and_solves() {
        let n = 67;
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            let re = (((i * 29 + j * 13 + 7) % 97) as f64 - 48.0) / 97.0;
            let im = (((i * 11 + j * 31 + 5) % 89) as f64 - 44.0) / 178.0;
            Complex64::new(if i == j { re + n as f64 } else { re }, im)
        });
        let factor = lu_factorize(&a).expect("complex factorization should succeed");
        assert_factor_reconstruction(&a, &factor, 2e-11);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64 * 0.5).cos()))
            .collect();
        let x = factor.solve(&b).expect("complex solve should succeed");
        let ax = a.matvec(&x);
        let max_error = ax
            .iter()
            .zip(&b)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_error < 2e-11, "complex solve error {max_error:e}");
    }

    #[test]
    fn blocked_lu_detects_singularity_after_first_panel() {
        let n = 70;
        let mut a = DenseMatrix::<f64>::eye(n);
        for j in 0..n {
            a[[69, j]] = a[[68, j]];
        }
        assert!(matches!(lu_factorize(&a), Err(LuError::SingularMatrix)));
    }

    /// Reference unblocked LU (scalar GETF2 over the full matrix) for
    /// comparing the blocked path against the kernel it is built from.
    fn reference_unblocked_lu<T: ComplexField>(a: &DenseMatrix<T>) -> (DenseMatrix<T>, Vec<usize>) {
        let n = a.nrows();
        let mut lu = a.clone();
        let mut pivots: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut max_val = component_max_norm(lu[[k, k]]);
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = component_max_norm(lu[[i, k]]);
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }
            assert!(
                max_val > T::Real::zero(),
                "reference matrix must be nonsingular"
            );
            if max_row != k {
                lu.swap_rows(k, max_row);
                pivots.swap(k, max_row);
            }
            let pivot_inv = lu[[k, k]].inv();
            for i in (k + 1)..n {
                let mult = lu[[i, k]] * pivot_inv;
                lu[[i, k]] = mult;
                for j in (k + 1)..n {
                    let update = mult * lu[[k, j]];
                    lu[[i, j]] -= update;
                }
            }
        }
        (lu, pivots)
    }

    #[test]
    fn blocked_lu_matches_unblocked_at_n256() {
        // n >= 192 activates the blocked TRSM+GEMM path; 256 crosses 8 full
        // panels, 257 adds a partial one. This matrix keeps the pivot on
        // the diagonal, so both paths make identical pivoting decisions and
        // their factors must agree to rounding level.
        for n in [256, 257] {
            let a = deterministic_matrix(n);
            let (reference_lu, reference_pivots) = reference_unblocked_lu(&a);
            let factor = lu_factorize(&a).expect("blocked factorization should succeed");
            assert_eq!(factor.pivots, reference_pivots, "pivot mismatch at n={n}");
            let mut max_diff = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    max_diff = max_diff.max((factor.lu[[i, j]] - reference_lu[[i, j]]).norm());
                }
            }
            assert!(
                max_diff < 1e-9,
                "blocked vs unblocked factor difference {max_diff:e} at n={n}"
            );

            // Residual parity: both solves must reach the same accuracy.
            let b: Vec<f64> = (0..n).map(|i| ((i * 19 + 3) as f64).sin() + 0.25).collect();
            let x_blocked = factor.solve(&b).expect("blocked solve should succeed");
            let blocked_residual = relative_residual(&a, &x_blocked, &b);
            let reference_factor = LuFactorization {
                lu: reference_lu,
                pivots: reference_pivots,
                n,
            };
            let x_reference = reference_factor
                .solve(&b)
                .expect("reference solve should succeed");
            let reference_residual = relative_residual(&a, &x_reference, &b);
            assert!(
                blocked_residual < 5e-13 * n as f64 && reference_residual < 5e-13 * n as f64,
                "residuals at n={n}: blocked={blocked_residual:e} reference={reference_residual:e}"
            );
        }
    }

    #[test]
    fn blocked_lu_handles_pivot_heavy_matrix_at_n256_and_n300() {
        // Same cyclic pivot-heavy construction as the n=131 test, now large
        // enough to exercise the blocked path across multiple panels
        // (n=256: 8 full panels; n=300: 9 panels + partial).
        for n in [256, 300] {
            let mut a = DenseMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                a[[i, (i + 1) % n]] = 2.0;
                a[[i, i]] = 0.25;
                a[[i, (i + n - 1) % n]] = -0.125;
            }
            let factor = lu_factorize(&a).expect("pivot-heavy matrix should factorize");
            assert!(
                factor
                    .pivots
                    .iter()
                    .enumerate()
                    .any(|(i, pivot)| i != *pivot),
                "test matrix must exercise row pivoting at n={n}"
            );
            assert_factor_reconstruction(&a, &factor, 5e-12 * n as f64);
            let b: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.5).collect();
            let x = factor.solve(&b).expect("solve should succeed");
            assert!(
                relative_residual(&a, &x, &b) < 1e-11,
                "large residual at n={n}"
            );
        }
    }

    #[test]
    fn blocked_lu_fused_schur_path_at_n576() {
        // The strided Schur update engages its large-trailing path once the
        // trailing matrix is big, which no smaller test reaches (n = 576
        // gives a 544-wide trailing update on the first panel).
        // Reconstruction plus solve residual catch any mis-dispatched tile.
        let n = 576;
        let a = deterministic_matrix(n);
        let factor = lu_factorize(&a).expect("factorization should succeed");
        assert_factor_reconstruction(&a, &factor, 5e-12 * n as f64);
        let b: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.5).collect();
        let x = factor.solve(&b).expect("solve should succeed");
        assert!(
            relative_residual(&a, &x, &b) < 1e-11,
            "large residual at n={n}"
        );
    }

    #[test]
    fn blocked_lu_complex_at_n256() {
        // Complex blocked path across 8 panels, vs the unblocked reference.
        let n = 256;
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            let re = (((i * 29 + j * 13 + 7) % 97) as f64 - 48.0) / 97.0;
            let im = (((i * 11 + j * 31 + 5) % 89) as f64 - 44.0) / 178.0;
            Complex64::new(if i == j { re + n as f64 } else { re }, im)
        });
        let factor = lu_factorize(&a).expect("complex factorization should succeed");
        assert_factor_reconstruction(&a, &factor, 2e-11);
        let (reference_lu, reference_pivots) = reference_unblocked_lu(&a);
        assert_eq!(factor.pivots, reference_pivots);
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                max_diff = max_diff.max((factor.lu[[i, j]] - reference_lu[[i, j]]).norm());
            }
        }
        assert!(
            max_diff < 1e-8,
            "complex blocked vs unblocked factor difference {max_diff:e}"
        );
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64 * 0.5).cos()))
            .collect();
        let x = factor.solve(&b).expect("complex solve should succeed");
        let ax = a.matvec(&x);
        let max_error = ax
            .iter()
            .zip(&b)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_error < 2e-11, "complex solve error {max_error:e}");
    }

    #[test]
    fn rejects_non_finite_matrix_and_rhs() {
        let matrix = dm(2, 2, vec![1.0_f64, f64::NAN, 0.0, 1.0]);
        assert!(matches!(
            lu_factorize(&matrix),
            Err(LuError::NonFiniteMatrix { index: 1 })
        ));

        let matrix = dm(2, 2, vec![2.0_f64, 0.0, 0.0, 3.0]);
        let factor = lu_factorize(&matrix).unwrap();
        assert!(matches!(
            factor.solve(&[1.0, f64::INFINITY]),
            Err(LuError::NonFiniteRhs { index: 1 })
        ));
    }

    #[test]
    fn finite_input_overflow_is_a_typed_numerical_breakdown() {
        let max = f64::MAX;
        let matrix = dm(2, 2, vec![max, max, max, -max]);
        assert!(matches!(
            lu_factorize(&matrix),
            Err(LuError::NumericalBreakdown {
                stage: "factorization",
                ..
            })
        ));

        let tiny = dm(1, 1, vec![f64::MIN_POSITIVE]);
        let factor = lu_factorize(&tiny).unwrap();
        assert!(matches!(
            factor.solve(&[f64::MAX]),
            Err(LuError::NumericalBreakdown {
                stage: "back substitution",
                index: 0,
            })
        ));
    }

    #[test]
    fn last_panel_overflow_is_a_typed_numerical_breakdown() {
        // End-scan coverage: overflow introduced inside the trailing
        // in-place panel. n = 224 gives seven 32-wide panels with an
        // in-place tail (32 rows remaining); uniform MAX scaling keeps
        // every pivot above the tolerance screen. The leading MAX*I
        // block factors cleanly with zero L21 rows, while the tail
        // block's top-left 2x2 [[MAX, MAX], [MAX, -MAX]] overflows
        // (-MAX - MAX = -inf) inside the last panel only.
        let max = f64::MAX;
        let n = 224;
        let mut data = vec![0.0f64; n * n];
        for i in 0..192 {
            data[i * n + i] = max;
        }
        for i in 192..n {
            data[i * n + i] = max;
        }
        data[192 * n + 192] = max;
        data[192 * n + 193] = max;
        data[193 * n + 192] = max;
        data[193 * n + 193] = -max;
        let matrix = dm(n, n, data);
        assert!(matches!(
            lu_factorize(&matrix),
            Err(LuError::NumericalBreakdown {
                stage: "factorization",
                ..
            })
        ));
    }

    #[test]
    fn first_panel_overflow_is_caught_by_panel_scan() {
        // Per-panel validation coverage: overflow introduced in an early
        // panel propagates through the TRSM into the trailing update;
        // the hot per-panel scan reports it (same error the old cold
        // end-of-factorize scan produced). n = 256 gives eight 32-wide
        // panels so a Schur update really follows the first panel
        // (n < 192 would collapse to a single compact panel). Uniform
        // MAX scaling keeps pivots above tolerance; the dense
        // off-diagonal ones-blocks make the first Schur complement
        // overflow (MAX - 32*MAX = -inf, possibly already in the
        // TRSM's U12).
        let max = f64::MAX;
        let n = 256;
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            data[i * n + i] = max;
            for j in 0..32 {
                if i >= 32 {
                    data[i * n + j] = max;
                }
            }
            for j in 32..n {
                if i < 32 {
                    data[i * n + j] = max;
                }
            }
        }
        let matrix = dm(n, n, data);
        assert!(matches!(
            lu_factorize(&matrix),
            Err(LuError::NumericalBreakdown {
                stage: "factorization",
                ..
            })
        ));
    }

    #[test]
    fn singular_public_factor_is_rejected_before_forward_destination_write() {
        let matrix = dm(2, 2, vec![2.0_f64, 0.0, 0.0, 3.0]);
        let mut factor = lu_factorize(&matrix).unwrap();
        factor.lu[[1, 1]] = 0.0;

        let mut solution = vec![7.0, -11.0];
        assert!(matches!(
            factor.solve_into(&[1.0, 1.0], &mut solution),
            Err(LuError::SingularMatrix)
        ));
        assert_eq!(solution, vec![7.0, -11.0]);

        let mut block_solution = vec![13.0, -17.0, 19.0, -23.0];
        assert!(matches!(
            factor.solve_many_into(&[1.0, 2.0, 3.0, 4.0], &mut block_solution, 2),
            Err(LuError::SingularMatrix)
        ));
        assert_eq!(block_solution, vec![13.0, -17.0, 19.0, -23.0]);
    }

    #[test]
    fn malformed_public_factor_state_is_rejected_without_indexing() {
        let matrix = dm(2, 2, vec![2.0_f64, 0.0, 0.0, 3.0]);
        let mut factor = lu_factorize(&matrix).unwrap();
        factor.pivots[0] = usize::MAX;
        assert!(matches!(
            factor.solve(&[1.0, 1.0]),
            Err(LuError::InvalidFactorization {
                reason: "pivots are not a permutation"
            })
        ));

        let factor = lu_factorize(&matrix).unwrap();
        assert!(matches!(
            factor.solve_many(&[], usize::MAX),
            Err(LuError::SizeOverflow {
                dimension: 2,
                nrhs: usize::MAX,
            })
        ));
    }
}
