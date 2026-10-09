//! Dense Cholesky factorization for declared SPD/HPD matrices.
//!
//! Factorization uses a right-looking blocked algorithm (LAPACK POTRF
//! structure) above [`CHOLESKY_BLOCKED_THRESHOLD`]: each diagonal panel is
//! factored with the unblocked kernel (POTF2), the panel's row block is
//! obtained by a TRSM-style triangular solve against the panel factor, and
//! the trailing update is deferred to one triangle-only SYRK dispatch.
//! Below the threshold the unblocked kernel alone is used — packing does
//! not amortize at small sizes (same crossover as the blocked dense LU on
//! Apple silicon).

use num_traits::{Float, FromPrimitive, One, Zero};
use rayon::prelude::*;
use sonium_compute::Transpose;
use sonium_compute::cpu::blas3::scope_pinned;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{AccuracyStatus, ComplexField, SolverFailure, SolverStatus};
use thiserror::Error;

/// Panel width used by the right-looking blocked factorization.
/// 64 stands (re-trialled 2026-09 against the row-block TRSM +
/// triangle-only SYRK structure plus the 3M single-pass GEMM floor,
/// rank-k M-split widening, and 32-wide f32 TRSM tier: nb=64 beats
/// nb=32 on spotrf and dpotrf at 512, 1024, and 2048 — 6/6, +8..+50% —
/// and nb=128 collapses everywhere). Two older trials kept 32, but both
/// predate the restructure: with scalar TRSM the O(n²k) row solves cost
/// more than the halved C-traffic saved (+4.5ms at n=2048), and with a
/// band-parallel blocked TRSM the per-panel cross-block GEMM dispatch +
/// `a_pack` gather cost more than the scalar flops they replaced (+5ms
/// at k=64, +12ms at k=32).
///
/// NOTE: `SONIUM_CHOL_NB` (consumed by [`cholesky_block_size`]) still
/// overrides the default for future re-tuning against new structures.
const CHOLESKY_BLOCK_SIZE: usize = 64;

/// Compact-pool width for Cholesky: all types run on the wide
/// (four-domain, 16-worker) pool.
///
/// History: `f64` narrowed to the single-CCX pool in 2026-09
/// (dpotrf-1024 4.48-4.55 narrow vs 4.53-5.25 wide, dpotrf-512 1.33 vs
/// 2.00) while `f32`/complex stayed wide. RE-ADJUDICATED 2026-10-01 on
/// merged kernels (`cholphases` process-interleaved A/B, RAYON=16):
/// wide beats narrow 6/6 pairs at 1024 (4.51-4.84 vs 5.60-5.83, ~1.23x)
/// and 6/6 at 512 (1.29-1.40 vs 1.42-1.58, ~1.1x), ties at 256. The
/// same round read wide-8 beating the 16-worker mirror pool 6/6
/// (4.66 vs 5.24 med) as "residency AND width" — REFINED 2026-10-02:
/// the mirror loss is the spawn funnel, not width (same 16 CPUs via
/// install beat wide-8 14/14 rounds, ~-6% at 1024; install beats spawn
/// 8/8, ~-15%, with the serial panel 2x slower on the spawned worker).
/// Four domains extend the same L3-domain mechanism with width. The
/// merged TRSM/AVX2 kernels raised f64's rate enough that the extra
/// domains' FMAs now pay. `SONIUM_CHOL_WIDE=0/1` forces narrow/wide
/// (diagnostic/A-B and kill-switch; unset takes the all-wide default).
fn cholesky_compact_wide() -> bool {
    if let Ok(v) = std::env::var("SONIUM_CHOL_WIDE") {
        if v == "0" {
            return false;
        }
        if v == "1" {
            return true;
        }
    }
    true
}

/// Clone the lower triangle + diagonal and zero the strict upper in one
/// pass, for all out-of-place entries: the blocked factorization only
/// reads/writes the referenced (lower) triangle, so copying the upper
/// just to zero it afterwards costs a full extra read+write pass (16 MB
/// each way at n=2048) plus the separate [`zero_upper_triangle`] walk.
/// Row bands copy in parallel when the buffer is large (same uninit-`Vec`
/// plus scope-pinned structure, same first-touch placement, same ~2 MB
/// chunk coarsening as the old full clone); each row copies its lower
/// part including the diagonal and fills the strict upper part with zero.
/// Below 1 MB of elements the scope dispatch costs more than the copy.
/// Bitwise identical to `matrix.clone()` on the lower triangle plus
/// diagonal with zeros above (every element receives the same value
/// exactly once). The clone's reads also warm the caller's lower triangle
/// for the validation that follows on the checked entries; their output
/// screen runs zero-free on the pre-zeroed upper (see
/// [`screen_lower_parallel`]).
#[allow(clippy::uninit_vec)] // band/serial row copies below write every element (see SAFETY).
fn parallel_clone_lower_zero_upper<T: ComplexField>(matrix: &DenseMatrix<T>) -> DenseMatrix<T> {
    let (nrows, ncols) = (matrix.nrows(), matrix.ncols());
    let src = matrix.as_slice();
    let mut dst = Vec::with_capacity(src.len());
    // SAFETY: every element is written by the row copies below before the
    // vector is read.
    unsafe {
        dst.set_len(src.len());
    }
    // One row's copy-plus-fill.
    // NOTE (reverted trial): non-temporal streaming stores for the row
    // segments (`nt_copy`/`nt_zero`, no read-for-ownership; exact bytes)
    // wash in-situ (interleaved dpotrf-2048: 3/6 pairs won, median
    // +0.08ms). The traffic saved on `dst` is offset by colder L3 for the
    // factorization that immediately follows — the clone is fabric-bound
    // at memcpy speed either way. Do not re-add without addressing the
    // consume-side residency.
    let copy_row = |row: usize, d: &mut [T], s: &[T]| {
        d[..row + 1].copy_from_slice(&s[..row + 1]);
        d[row + 1..].fill(T::zero());
    };
    if src.len() < PAR_CLONE_MIN_ELS {
        for (row, (d, s)) in dst
            .chunks_exact_mut(ncols)
            .zip(src.chunks_exact(ncols))
            .enumerate()
        {
            copy_row(row, d, s);
        }
    } else {
        // Row bands with the same ~2 MB coarsening as the full clone.
        let bytes = std::mem::size_of_val(src);
        let chunks = bytes.div_ceil(2_097_152).clamp(4, 16).min(nrows).max(1);
        let rows_per = nrows.div_ceil(chunks);
        scope_pinned(|scope| {
            for (band, (d, part)) in dst
                .chunks_mut(rows_per * ncols)
                .zip(src.chunks(rows_per * ncols))
                .enumerate()
            {
                let r0 = band * rows_per;
                scope.spawn(move |_| {
                    for (row, (d, s)) in d
                        .chunks_exact_mut(ncols)
                        .zip(part.chunks_exact(ncols))
                        .enumerate()
                    {
                        copy_row(r0 + row, d, s);
                    }
                });
            }
        });
    }
    DenseMatrix::from_row_major_vec(nrows, ncols, dst)
}

/// Fused clone-validate for the checked out-of-place entries: copies the
/// lower triangle + diagonal and zeroes the strict upper (exactly as
/// [`parallel_clone_lower_zero_upper`]) while folding the entry
/// finiteness + max-norm scale over the source rows in the same band pass
/// (one vectorized [`ComplexField::vec_max_norm_checked`] fold per band
/// after its rows are copied, while the source lines are hot). One
/// parallel pass replaces the clone scope plus the validation wave-1
/// scope and its full re-read (at n=2048: ~16 MB of reads plus a scope).
/// Returns the factor plus the whole-matrix `(max_norm, all_finite)` fold
/// for [`finish_validation`]; the tiled symmetry check still runs
/// separately (it needs the tolerance first). Same values and errors as
/// the split form.
#[allow(clippy::uninit_vec)] // band/serial row copies below write every element (see SAFETY).
fn clone_validated_lower_zero_upper<T: ComplexField>(
    matrix: &DenseMatrix<T>,
) -> (DenseMatrix<T>, T::Real, bool) {
    let (nrows, ncols) = (matrix.nrows(), matrix.ncols());
    let src = matrix.as_slice();
    let mut dst = Vec::with_capacity(src.len());
    // SAFETY: every element is written by the row copies below before the
    // vector is read.
    unsafe {
        dst.set_len(src.len());
    }
    // (max_norm, all_finite) over the source, shared across bands.
    let acc = &std::sync::Mutex::new((T::Real::zero(), true));
    if src.len() < PAR_CLONE_MIN_ELS {
        for (row, (d, s)) in dst
            .chunks_exact_mut(ncols)
            .zip(src.chunks_exact(ncols))
            .enumerate()
        {
            d[..row + 1].copy_from_slice(&s[..row + 1]);
            d[row + 1..].fill(T::zero());
        }
        let (m, ok) = T::vec_max_norm_checked(src);
        return (DenseMatrix::from_row_major_vec(nrows, ncols, dst), m, ok);
    }
    // Row bands with the same ~2 MB coarsening as the plain fused clone.
    let bytes = std::mem::size_of_val(src);
    let chunks = bytes.div_ceil(2_097_152).clamp(4, 16).min(nrows).max(1);
    let rows_per = nrows.div_ceil(chunks);
    scope_pinned(|scope| {
        for (band, (d, part)) in dst
            .chunks_mut(rows_per * ncols)
            .zip(src.chunks(rows_per * ncols))
            .enumerate()
        {
            let r0 = band * rows_per;
            scope.spawn(move |_| {
                for (row, (d, s)) in d
                    .chunks_exact_mut(ncols)
                    .zip(part.chunks_exact(ncols))
                    .enumerate()
                {
                    let r = r0 + row;
                    d[..r + 1].copy_from_slice(&s[..r + 1]);
                    d[r + 1..].fill(T::zero());
                }
                // Fold after the copy, while this band's source lines are
                // hot; the reduction below is order-free (AND + max).
                let (m, ok) = T::vec_max_norm_checked(part);
                let mut a = acc.lock().unwrap();
                if m > a.0 {
                    a.0 = m;
                }
                a.1 &= ok;
            });
        }
    });
    let (max, ok) = {
        let guard = acc.lock().unwrap();
        (guard.0, guard.1)
    };
    (DenseMatrix::from_row_major_vec(nrows, ncols, dst), max, ok)
}

/// Effective panel width: `SONIUM_CHOL_NB` override (accepted when it is
/// a sane width) else a size-dependent default — 48 at `n <= 1024`, 64
/// ([`CHOLESKY_BLOCK_SIZE`]) above. Rationale: 48 wins at 256/512/1024 in
/// both precisions (dpotrf-1024 4.07 vs 4.28, dpotrf-512 1.84 vs 1.92,
/// spotrf-1024 3.37 vs 3.71, dpotrf-256 0.43 vs 0.49) while 2048 prefers
/// 64 (44.6 vs 49.2: the out-of-L3 regime on the wide pool wants coarser
/// panels). Parsed once per call.
fn cholesky_block_size(n: usize) -> usize {
    if let Some(nb) = std::env::var("SONIUM_CHOL_NB")
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|nb| (8..=512).contains(nb))
    {
        return nb;
    }
    if n <= 1024 { 48 } else { CHOLESKY_BLOCK_SIZE }
}

/// Below this order the unblocked kernel is used for the whole matrix.
const CHOLESKY_BLOCKED_THRESHOLD: usize = 192;

/// Below this many elements the clone scope dispatch costs more than the
/// copy (1 MB of f64). Shared by the clone variants below.
const PAR_CLONE_MIN_ELS: usize = 131_072;

/// Errors from dense Cholesky factorization and solves.
#[derive(Debug, Error)]
pub enum CholeskyError {
    #[error("Matrix must be square ({rows} x {cols})")]
    NotSquare { rows: usize, cols: usize },
    #[error("Matrix is not {kind} within the scale-aware tolerance")]
    StructureMismatch { kind: &'static str },
    #[error("Matrix is not positive definite (pivot {pivot})")]
    NotPositiveDefinite { pivot: usize },
    #[error("Right-hand side length mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Multi-RHS shape overflows usize (dimension={dimension}, nrhs={nrhs})")]
    SizeOverflow { dimension: usize, nrhs: usize },
    #[error("Matrix contains a non-finite value at storage index {index}")]
    NonFiniteMatrix { index: usize },
    #[error("Right-hand side contains a non-finite value at index {index}")]
    NonFiniteRhs { index: usize },
    #[error("Cholesky numerical breakdown during {stage} at index {index}")]
    NumericalBreakdown { stage: &'static str, index: usize },
}

impl CholeskyError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NotPositiveDefinite { .. } | Self::NumericalBreakdown { .. } => {
                SolverStatus::Breakdown
            }
            _ => SolverStatus::NumericalError,
        }
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        self.attainable_accuracy()
    }
}

impl SolverFailure for CholeskyError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NotSquare { .. } | Self::DimensionMismatch { .. } | Self::SizeOverflow { .. } => {
                SolverErrorKind::DimensionMismatch
            }
            Self::NotPositiveDefinite { .. } | Self::NumericalBreakdown { .. } => {
                SolverErrorKind::Breakdown
            }
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        CholeskyError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        CholeskyError::accuracy_status(self)
    }
}

/// Lower-triangular Cholesky factor, `A = LLᵀ` or `A = LLᴴ`.
#[derive(Debug, Clone)]
pub struct CholeskyFactorization<T: ComplexField> {
    factor: DenseMatrix<T>,
    hermitian: bool,
}

impl<T: ComplexField> CholeskyFactorization<T> {
    /// Matrix dimension.
    pub fn dimension(&self) -> usize {
        self.factor.nrows()
    }

    /// Whether the factor represents the Hermitian form `LLᴴ`.
    pub fn is_hermitian(&self) -> bool {
        self.hermitian
    }

    /// Borrow the packed lower-triangular factor matrix.
    pub fn factor(&self) -> &DenseMatrix<T> {
        &self.factor
    }

    /// Number of mathematically stored factor entries.
    pub fn factor_nonzeros(&self) -> usize {
        let n = self.dimension();
        n.saturating_mul(n.saturating_add(1)) / 2
    }

    /// Solve `Ax = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, CholeskyError> {
        let mut x = vec![T::zero(); self.dimension()];
        self.solve_into(b, &mut x)?;
        Ok(x)
    }

    /// Allocation-free solve into caller-owned storage.
    /// The destination is unchanged when validation or numerical breakdown
    /// rejects the generated solution.
    pub fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), CholeskyError> {
        let n = self.dimension();
        if b.len() != n || x.len() != n {
            return Err(CholeskyError::DimensionMismatch {
                expected: n,
                got: if b.len() != n { b.len() } else { x.len() },
            });
        }
        validate_rhs(b)?;
        // Stage the result so a finite RHS that overflows during either
        // triangular solve cannot publish a partial low-level factor result.
        let mut staged = b.to_vec();
        let factors = self.factor.as_slice();
        // Forward substitution Ly = b in dot form over contiguous L row
        // segments (SIMD `vec_dot_plain` backend). Association order may
        // differ in the last ulp; residuals are the oracle.
        for i in 0..n {
            let row_base = i * n;
            let dot = T::vec_dot_plain(&factors[row_base..row_base + i], &staged[..i]);
            staged[i] -= dot;
            staged[i] *= self.factor[[i, i]].inv();
            ensure_finite(staged[i], "forward substitution", i)?;
        }
        // Backward substitution Lᴴz = y in AXPY form: processed top-down
        // in reverse, each solved entry updates the earlier rows through
        // the contiguous L row segment. The Hermitian path conjugates the
        // row into scratch (the stored factor must not be mutated).
        let mut lrow: Vec<T> = if self.hermitian {
            vec![T::zero(); n]
        } else {
            Vec::new()
        };
        for i in (0..n).rev() {
            let diagonal = if self.hermitian {
                self.factor[[i, i]].conj()
            } else {
                self.factor[[i, i]]
            };
            staged[i] *= diagonal.inv();
            ensure_finite(staged[i], "back substitution", i)?;
            if i > 0 {
                // NB: `vec_axpy` adds; substitution subtracts. The entry
                // must be diagonally scaled *before* propagating: later
                // (lower-indexed) rows consume the final value.
                let coef = -staged[i];
                if self.hermitian {
                    for (dst, src) in lrow.iter_mut().zip(&factors[i * n..i * n + i]) {
                        *dst = src.conj();
                    }
                    T::vec_axpy(coef, &lrow[..i], &mut staged[..i]);
                } else {
                    T::vec_axpy(coef, &factors[i * n..i * n + i], &mut staged[..i]);
                }
            }
        }
        x.copy_from_slice(&staged);
        Ok(())
    }

    /// Solve an `n × nrhs` row-major block of right-hand sides.
    pub fn solve_many_into(&self, b: &[T], x: &mut [T], nrhs: usize) -> Result<(), CholeskyError> {
        let n = self.dimension();
        let expected = n
            .checked_mul(nrhs)
            .ok_or(CholeskyError::SizeOverflow { dimension: n, nrhs })?;
        if b.len() != expected || x.len() != expected {
            return Err(CholeskyError::DimensionMismatch {
                expected,
                got: if b.len() != expected {
                    b.len()
                } else {
                    x.len()
                },
            });
        }
        validate_rhs(b)?;
        x.copy_from_slice(b);
        let hermitian = self.hermitian;
        // Blocked multi-RHS solves (see `lu.rs` `*_blocked`): one packed
        // `(panel × nrhs) × (nrhs)`-free rank update per panel pair replaces
        // O(panel²) scalar-dispatched `vec_axpy` calls with BLAS-3-shaped
        // `schur_sub` traffic. The Hermitian backward path conjugates the
        // packed Lᴴ panel block at pack time. Association order may differ
        // in the last ulp; residuals are the oracle.
        const CHOL_TRSM_BLOCK: usize = 64;
        if nrhs > 1 {
            let bw = CHOL_TRSM_BLOCK;
            let mut panel: Vec<T> = Vec::new();
            // Forward: Ly = b. Serial diagonal micro-blocks (with the
            // diagonal scale + validation the unblocked path performs)
            // plus packed-L `schur_sub` panel updates.
            let mut b0 = 0;
            while b0 < n {
                let b1 = (b0 + bw).min(n);
                for i in b0..b1 {
                    for j in b0..i {
                        let multiplier = -self.factor[[i, j]];
                        let (prefix, suffix) = x.split_at_mut(i * nrhs);
                        let source = &prefix[j * nrhs..(j + 1) * nrhs];
                        T::vec_axpy(multiplier, source, &mut suffix[..nrhs]);
                    }
                    T::vec_scale(&mut x[i * nrhs..(i + 1) * nrhs], self.factor[[i, i]].inv());
                    validate_generated_slice(
                        &x[i * nrhs..(i + 1) * nrhs],
                        "multi-RHS forward substitution",
                        i * nrhs,
                    )?;
                }
                let rest = n - b1;
                if rest > 0 && nrhs > 0 {
                    panel.clear();
                    panel.reserve(rest * (b1 - b0));
                    for row in b1..n {
                        let base = row * n + b0;
                        panel.extend_from_slice(&self.factor.as_slice()[base..base + (b1 - b0)]);
                    }
                    let (done, todo) = x.split_at_mut(b1 * nrhs);
                    T::schur_sub(rest, nrhs, b1 - b0, &panel, &done[b0 * nrhs..], todo, nrhs);
                }
                b0 = b1;
            }
            // Backward: Lᴴz = y. Packed-Lᴴ `schur_sub` panel updates plus
            // serial diagonal micro-blocks with the Hermitian conjugation
            // the unblocked path applies.
            let mut b1 = n;
            while b1 > 0 {
                let b0 = b1.saturating_sub(bw);
                let rest = n - b1;
                if rest > 0 && nrhs > 0 {
                    // Pack Lᴴ[b0..b1, b1..]: entry (r, c) is the stored
                    // L[c, r], conjugated on the Hermitian path.
                    panel.clear();
                    panel.reserve((b1 - b0) * rest);
                    for r in b0..b1 {
                        for c in b1..n {
                            let v = self.factor[[c, r]];
                            panel.push(if hermitian { v.conj() } else { v });
                        }
                    }
                    let (block, rest_x) = x.split_at_mut(b1 * nrhs);
                    T::schur_sub(
                        b1 - b0,
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
                        let coefficient = if hermitian {
                            self.factor[[j, i]].conj()
                        } else {
                            self.factor[[j, i]]
                        };
                        let (prefix, suffix) = x.split_at_mut(j * nrhs);
                        let destination = &mut prefix[i * nrhs..(i + 1) * nrhs];
                        T::vec_axpy(-coefficient, &suffix[..nrhs], destination);
                    }
                    let diagonal = if hermitian {
                        self.factor[[i, i]].conj()
                    } else {
                        self.factor[[i, i]]
                    };
                    T::vec_scale(&mut x[i * nrhs..(i + 1) * nrhs], diagonal.inv());
                    validate_generated_slice(
                        &x[i * nrhs..(i + 1) * nrhs],
                        "multi-RHS back substitution",
                        i * nrhs,
                    )?;
                }
                b1 = b0;
            }
            return Ok(());
        }
        for i in 0..n {
            for j in 0..i {
                let alpha = -self.factor[[i, j]];
                let (prefix, suffix) = x.split_at_mut(i * nrhs);
                T::vec_axpy(
                    alpha,
                    &prefix[j * nrhs..(j + 1) * nrhs],
                    &mut suffix[..nrhs],
                );
            }
            T::vec_scale(&mut x[i * nrhs..(i + 1) * nrhs], self.factor[[i, i]].inv());
            validate_generated_slice(
                &x[i * nrhs..(i + 1) * nrhs],
                "multi-RHS forward substitution",
                i * nrhs,
            )?;
        }
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let coefficient = if hermitian {
                    self.factor[[j, i]].conj()
                } else {
                    self.factor[[j, i]]
                };
                let (prefix, suffix) = x.split_at_mut(j * nrhs);
                T::vec_axpy(
                    -coefficient,
                    &suffix[..nrhs],
                    &mut prefix[i * nrhs..(i + 1) * nrhs],
                );
            }
            let diagonal = if hermitian {
                self.factor[[i, i]].conj()
            } else {
                self.factor[[i, i]]
            };
            T::vec_scale(&mut x[i * nrhs..(i + 1) * nrhs], diagonal.inv());
            validate_generated_slice(
                &x[i * nrhs..(i + 1) * nrhs],
                "multi-RHS back substitution",
                i * nrhs,
            )?;
        }
        Ok(())
    }
}

/// Factorize a declared symmetric positive-definite matrix.
pub fn cholesky_factorize<T: ComplexField>(
    matrix: &DenseMatrix<T>,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    cholesky_factorize_impl(matrix, false)
}

/// Factorize a declared Hermitian positive-definite matrix.
pub fn cholesky_factorize_hermitian<T: ComplexField>(
    matrix: &DenseMatrix<T>,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    cholesky_factorize_impl(matrix, true)
}

/// Factorize an owned declared symmetric positive-definite matrix in
/// place, avoiding the compatibility entry's input clone (see
/// [`lu_factorize_in_place`](super::lu::lu_factorize_in_place) for the LU
/// twin). Validation is unchanged — finite screen, max norm, symmetry
/// screen with the same tolerance and error precedence, then the same
/// blocked factorization and factor screen — only the working copy is the
/// caller's buffer instead of a clone. Bitwise identical factor to
/// [`cholesky_factorize`] on the same values.
pub fn cholesky_factorize_in_place<T: ComplexField>(
    matrix: DenseMatrix<T>,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    let tolerance = validate_cholesky_input(&matrix, false)?;
    factorize_in_place_funneled(matrix, false, tolerance, true)
}

/// Owned-buffer twin of [`cholesky_factorize_hermitian`]: see
/// [`cholesky_factorize_in_place`].
pub fn cholesky_factorize_in_place_hermitian<T: ComplexField>(
    matrix: DenseMatrix<T>,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    let tolerance = validate_cholesky_input(&matrix, true)?;
    factorize_in_place_funneled(matrix, true, tolerance, true)
}

/// Tiny-pool funnel envelope (shipped default-on 2026-09-27,
/// `SONIUM_CHOL_TINY=0` opt-out): one install, all nested SYRK/TRSM
/// scopes inline at 16-wide sizings. Measured OB-free order-rotated:
/// wins at n=1024 (1.20x) and n=2048 (1.51-1.54x); floor n>=1024
/// (smaller unmeasured — keep existing routing). Cap 64MB (same
/// rationale as the LU envelope: 4096-class factors need wider pools).
/// f64 only via [`ComplexField::chol_prefers_tiny_pool`]; inert at
/// global <= 16 (router already inline). dpotrf-2048 now ~1.2x vs OB
/// (14.4 vs 12.2). No pivots in Cholesky; maxdiff 1.7e-18 at 2048 with
/// equal residuals (256/1024).
fn chol_tiny_here<T: ComplexField>(n: usize) -> bool {
    let est = n as u64 * n as u64 * std::mem::size_of::<T>() as u64;
    !std::env::var("SONIUM_CHOL_TINY")
        .map(|v| v == "0")
        .unwrap_or(false)
        && T::chol_prefers_tiny_pool()
        && n >= 1024
        && rayon::current_num_threads() > 16
        && est <= (64 << 20)
}

/// Owned-buffer factorization through the compact-pool funnel (same
/// placement as the out-of-place [`cholesky_factorize_unchecked`], minus
/// the working clone): the caller's buffer is factored on the compact
/// (or mirror) pool instead of the calling thread, so in-place entries
/// get the same pool residency as out-of-place ones. Nested calls run
/// inline (see `run_compact`); bitwise identical to direct
/// [`factorize_owned`] (same kernels, same order).
fn factorize_in_place_funneled<T: ComplexField>(
    matrix: DenseMatrix<T>,
    hermitian: bool,
    tolerance: T::Real,
    screen_output: bool,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    if matrix.nrows() < 128 {
        return factorize_owned(matrix, hermitian, tolerance, screen_output, false);
    }
    let n = matrix.nrows();
    let est = n as u64 * n as u64 * std::mem::size_of::<T>() as u64;
    let wide = cholesky_compact_wide();
    // Tiny-pool funnel (shipped default-on 2026-09-27,
    // `SONIUM_CHOL_TINY=0` opt-out): in-place twin (`run_tiny` borrows,
    // so the caller buffer is used directly).
    if chol_tiny_here::<T>(n) {
        return sonium_compute::cpu::blas3::run_tiny(|| {
            factorize_owned(matrix, hermitian, tolerance, screen_output, false)
        });
    }
    if sonium_compute::cpu::blas3::compact_applies(est, wide) {
        return sonium_compute::cpu::blas3::run_compact(est, wide, || {
            factorize_owned(matrix, hermitian, tolerance, screen_output, false)
        });
    }
    // Mirror-pool funnel: `install` (borrow) instead of `spawn` (see
    // `run_mirror`; `SONIUM_CHOL_MIRROR_INSTALL=0` restores the spawn
    // for diagnostic/A-B). Same pool, same width, same task splits —
    // only the launching thread differs, so factors match the spawn
    // path bitwise.
    if chol_mirror_install_on() {
        return sonium_compute::cpu::blas3::run_mirror(|| {
            factorize_owned(matrix, hermitian, tolerance, screen_output, false)
        });
    }
    sonium_compute::cpu::blas3::run_on_worker(move || {
        factorize_owned(matrix, hermitian, tolerance, screen_output, false)
    })
}

/// Mirror-funnel launch for [`factorize_in_place_funneled`]: `install`
/// by default, `spawn` with `SONIUM_CHOL_MIRROR_INSTALL=0`
/// (diagnostic/A-B and kill-switch). Shipped default-on 2026-10-02:
/// install beats spawn 8/8 process-interleaved rounds (~-15% at
/// dpotrf-1024 on identical 16 CPUs, RAYON=16) — the serial panel runs
/// 2x slower on the spawned worker and TRSM phases cost +40%,
/// consistent with a cold/parked spawn target turning straggler.
fn chol_mirror_install_on() -> bool {
    !std::env::var("SONIUM_CHOL_MIRROR_INSTALL")
        .map(|v| v == "0")
        .unwrap_or(false)
}

/// Unchecked out-of-place Cholesky: skips [`validate_cholesky_input`]
/// (finiteness + symmetry O(n²) walks) and the output factor screen.
/// The caller guarantees finite symmetric positive-definite input.
/// The pivot tolerance uses diagonal-max scale, which is exact for SPD
/// input (off-diagonals are bounded by the diagonal), so the factor is
/// bitwise identical to [`cholesky_factorize`] on valid input. Otherwise
/// the same blocked algorithm runs; on invalid input it may return a
/// bogus factor or a [`CholeskyError::NotPositiveDefinite`] instead of a
/// structural error.
pub fn cholesky_factorize_unchecked<T: ComplexField>(
    matrix: &DenseMatrix<T>,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    // Hot-worker funnel; see `cholesky_factorize_impl`.
    if matrix.nrows() < 128 {
        let tolerance = unchecked_tolerance(matrix)?;
        return factorize_owned(
            parallel_clone_lower_zero_upper(matrix),
            false,
            tolerance,
            false,
            true,
        );
    }
    let tolerance = unchecked_tolerance(matrix)?;
    // Tiny-pool funnel (shipped default-on 2026-09-27,
    // `SONIUM_CHOL_TINY=0` opt-out): whole factorization on the tiny
    // pool — same hop-elimination recipe as LU lever 29. (A first
    // attempt gated inside `factorize_owned` never fired: it runs
    // nested on a pool worker, where `run_tiny` is an inline no-op.
    // The gate must live at the funnel.)
    if chol_tiny_here::<T>(matrix.nrows()) {
        return sonium_compute::cpu::blas3::run_tiny(|| {
            factorize_owned(
                parallel_clone_lower_zero_upper(matrix),
                false,
                tolerance,
                false,
                true,
            )
        });
    }
    // Compact pool when the factor fits in L3: clone inside so
    // first-touch binds pages to the compact NUMA node; otherwise the
    // mirror-pool funnel as before (clone outside for `'static`). Width
    // is all-wide (see `cholesky_compact_wide`).
    let est = matrix.nrows() as u64 * matrix.nrows() as u64 * std::mem::size_of::<T>() as u64;
    let wide = cholesky_compact_wide();
    if sonium_compute::cpu::blas3::compact_applies(est, wide) {
        let time_entry = std::env::var("SONIUM_CHOL_PHASES").is_ok();
        let t_clone = std::time::Instant::now();
        let run = sonium_compute::cpu::blas3::run_compact(est, wide, || {
            factorize_owned(
                parallel_clone_lower_zero_upper(matrix),
                false,
                tolerance,
                false,
                true,
            )
        });
        if time_entry {
            eprintln!(
                "chol entry clone+funnel+factorize {:.3}ms",
                t_clone.elapsed().as_secs_f64() * 1000.0
            );
        }
        return run;
    }
    // Fused clone-zero (see `parallel_clone_lower_zero_upper`): a serial
    // clone of a multi-MB factor faults every page on one thread (~17ms
    // at n=2048); banded copies fault in parallel.
    let owned = parallel_clone_lower_zero_upper(matrix);
    sonium_compute::cpu::blas3::run_on_worker(move || {
        factorize_owned(owned, false, tolerance, false, true)
    })
}

/// Unchecked twin of [`cholesky_factorize_hermitian`]: see
/// [`cholesky_factorize_unchecked`].
pub fn cholesky_factorize_unchecked_hermitian<T: ComplexField>(
    matrix: &DenseMatrix<T>,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    // Hot-worker funnel; see `cholesky_factorize_impl`.
    if matrix.nrows() < 128 {
        let tolerance = unchecked_tolerance(matrix)?;
        return factorize_owned(
            parallel_clone_lower_zero_upper(matrix),
            true,
            tolerance,
            false,
            true,
        );
    }
    let tolerance = unchecked_tolerance(matrix)?;
    // Tiny-pool funnel: Hermitian twin of the unchecked gate above.
    if chol_tiny_here::<T>(matrix.nrows()) {
        return sonium_compute::cpu::blas3::run_tiny(|| {
            factorize_owned(
                parallel_clone_lower_zero_upper(matrix),
                true,
                tolerance,
                false,
                true,
            )
        });
    }
    // Compact pool when the factor fits in L3 (clone inside for local
    // first-touch); otherwise the mirror-pool funnel as before. Width is
    // all-wide (see `cholesky_compact_wide`).
    let est = matrix.nrows() as u64 * matrix.nrows() as u64 * std::mem::size_of::<T>() as u64;
    let wide = cholesky_compact_wide();
    if sonium_compute::cpu::blas3::compact_applies(est, wide) {
        return sonium_compute::cpu::blas3::run_compact(est, wide, || {
            factorize_owned(
                parallel_clone_lower_zero_upper(matrix),
                true,
                tolerance,
                false,
                true,
            )
        });
    }
    // Fused clone-zero (see `parallel_clone_lower_zero_upper`): a serial
    // clone of a multi-MB factor faults every page on one thread (~17ms
    // at n=2048); banded copies fault in parallel.
    let owned = parallel_clone_lower_zero_upper(matrix);
    sonium_compute::cpu::blas3::run_on_worker(move || {
        factorize_owned(owned, true, tolerance, false, true)
    })
}

/// Unchecked owned-buffer twin of [`cholesky_factorize_in_place`]: skips
/// input validation and the output screen; see
/// [`cholesky_factorize_unchecked`]. No working clone (the caller's buffer
/// is factored directly).
pub fn cholesky_factorize_in_place_unchecked<T: ComplexField>(
    matrix: DenseMatrix<T>,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    let tolerance = unchecked_tolerance(&matrix)?;
    factorize_in_place_funneled(matrix, false, tolerance, false)
}

/// Unchecked owned-buffer twin of [`cholesky_factorize_in_place_hermitian`]:
/// see [`cholesky_factorize_in_place_unchecked`].
pub fn cholesky_factorize_in_place_unchecked_hermitian<T: ComplexField>(
    matrix: DenseMatrix<T>,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    let tolerance = unchecked_tolerance(&matrix)?;
    factorize_in_place_funneled(matrix, true, tolerance, false)
}

/// O(1)-metadata-plus-diagonal scale for the unchecked entries: squareness
/// is still screened (free, and indexing assumes it), then the pivot
/// tolerance uses diagonal-max scale in O(n). Exact for SPD input, so
/// unchecked factors match checked ones bitwise on valid input.
fn unchecked_tolerance<T: ComplexField>(matrix: &DenseMatrix<T>) -> Result<T::Real, CholeskyError> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(CholeskyError::NotSquare {
            rows: n,
            cols: matrix.ncols(),
        });
    }
    let values = matrix.as_slice();
    let mut scale = T::Real::zero();
    for i in 0..n {
        scale = scale.max(values[i * n + i].norm());
    }
    Ok(T::Real::epsilon()
        * T::Real::from_usize(n.max(1)).unwrap_or_else(T::Real::one)
        * scale.max(T::Real::min_positive_value()))
}

fn cholesky_factorize_impl<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    hermitian: bool,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    // Hot-worker funnel (see `lu_factorize`): validation plus the serial
    // panel phases run at the caller's possibly cold core frequency
    // otherwise. Tiny matrices skip it (µs solves, below the round-trip).
    // Shipped 2026-09-26: dpotrf-1024 interleaved direct/funnel 8.75-9.24
    // -> 6.85-7.61 (unchecked entries funnel the same way).
    if matrix.nrows() < 128 {
        let tolerance = validate_cholesky_input(matrix, hermitian)?;
        return factorize_owned(matrix.clone(), hermitian, tolerance, true, false);
    }
    // Tiny-pool funnel: checked-entry twin (fused clone-validate runs
    // inside, same as the compact path).
    if chol_tiny_here::<T>(matrix.nrows()) {
        return sonium_compute::cpu::blas3::run_tiny(|| {
            let (owned, scale_max, all_finite) = clone_validated_lower_zero_upper(matrix);
            let tolerance = finish_validation(
                hermitian,
                matrix.nrows(),
                matrix.as_slice(),
                scale_max,
                all_finite,
            )?;
            factorize_owned(owned, hermitian, tolerance, true, true)
        });
    }
    // Compact pool when the factor fits in L3 (validation reads the
    // caller's buffer, the clone runs inside for local first-touch);
    // otherwise the mirror-pool funnel as before. Width is all-wide
    // (see `cholesky_compact_wide`).
    let est = matrix.nrows() as u64 * matrix.nrows() as u64 * std::mem::size_of::<T>() as u64;
    let wide = cholesky_compact_wide();
    if sonium_compute::cpu::blas3::compact_applies(est, wide) {
        return sonium_compute::cpu::blas3::run_compact(est, wide, || {
            // Fused clone-validate (lower copy + upper zero + entry
            // finiteness/scale fold in one band pass): the symmetry check
            // still runs on the caller's buffer via `finish_validation`,
            // now warm from the clone's reads. Validation errors still
            // precede factorization. The end screen runs zero-free on the
            // pre-zeroed upper (see `factorize_owned`).
            let (owned, scale_max, all_finite) = clone_validated_lower_zero_upper(matrix);
            let tolerance = finish_validation(
                hermitian,
                matrix.nrows(),
                matrix.as_slice(),
                scale_max,
                all_finite,
            )?;
            factorize_owned(owned, hermitian, tolerance, true, true)
        });
    }
    // Mirror-pool fallback (out-of-L3 working set): same fused
    // clone-validate as the compact path, then factorize on a mirror
    // worker with the zero-free parallel screen. A serial clone of a
    // multi-MB factor faults every page on one thread (~17ms at n=2048);
    // banded copies fault in parallel.
    let (owned, scale_max, all_finite) = clone_validated_lower_zero_upper(matrix);
    let tolerance = finish_validation(
        hermitian,
        matrix.nrows(),
        matrix.as_slice(),
        scale_max,
        all_finite,
    )?;
    sonium_compute::cpu::blas3::run_on_worker(move || {
        factorize_owned(owned, hermitian, tolerance, true, true)
    })
}

/// Validate squareness, finiteness, scale, and symmetry/structure,
/// returning the scale-aware pivot tolerance.
///
/// The finiteness and scale passes are fused into one slice walk, and the
/// symmetry pass indexes the raw slice instead of going through the
/// bounds-checked [`DenseMatrix`] accessor per element — the legacy
/// per-element accessor cost dominated the API time at large `n`.
///
/// Parallel validation for large inputs: two band waves in one scope —
/// a vectorized finiteness + max-norm fold over linear chunks, then
/// interleaved row bands for the symmetry scan (rows `w, w+8, …` balance
/// the triangular skew). Reductions are order-free (AND of finite flags,
/// max magnitude, min bad row with diagonal precedence). Error order and
/// error values match the serial walk exactly; the scale itself matches
/// up to the documented `vec_max_norm_checked` boundary policy.
/// Runs on the enclosing pool when nested, else the mirror pool; never
/// changes values.
/// Validate/symmetry band count: 8 while the matrix fits in L3,
/// 16 once DRAM streaming dominates. Measured at dpotrf-4096 (f64,
/// 134MB): 16 bands cut validate-in 8.0 -> 5.7ms min (bandwidth scales;
/// 4 bands collapse to 13.5ms), while at 2048 (33MB) 8 still wins by
/// 0.25ms (L3 bandwidth saturates early; extra bands add steal
/// coordination) and 1024 ties. The 64MB cut sits between them (same
/// rationale as the tiny-pool cap); f32-4096 (67MB) rides 16 on the
/// same mechanism (unmeasured — no gate cell there).
/// `SONIUM_CHOL_VBANDS=n` overrides (diagnostic/A-B).
const VALIDATE_BANDS_SMALL: usize = 8;
const VALIDATE_BANDS_BIG: usize = 16;
const VALIDATE_BIG_BYTES: u64 = 64 << 20;

#[inline]
fn validate_bands(n: usize, elt: usize) -> usize {
    if let Ok(text) = std::env::var("SONIUM_CHOL_VBANDS")
        && let Ok(bands) = text.parse::<usize>()
    {
        return bands.min(rayon::current_num_threads().max(1));
    }
    let est = n as u64 * n as u64 * elt as u64;
    let bands = if est > VALIDATE_BIG_BYTES {
        VALIDATE_BANDS_BIG
    } else {
        VALIDATE_BANDS_SMALL
    };
    bands.min(rayon::current_num_threads().max(1))
}

/// Per-band accumulators for the fused single-scope [`validate_parallel`]:
/// finiteness flag + max-norm scale (wave 1) with the max pair asymmetry
/// and max diagonal imaginary part (wave 2). Bands own disjoint
/// tile-rows, so no synchronization is needed.
struct ValidateAcc<R> {
    all_finite: bool,
    scale_max: R,
    max_asym_sqr: R,
    max_diag_imag: R,
}

fn validate_parallel<T: ComplexField>(
    hermitian: bool,
    n: usize,
    values: &[T],
) -> Result<T::Real, CholeskyError> {
    // Band count (see `validate_bands`): 8 while L3-resident (excess
    // bands multiply steal traffic there), 16 once DRAM streaming
    // dominates (bandwidth scales to 16).
    // the interleaved row pattern below keeps triangular skew balanced at
    // any count.
    // At least one band (a `SONIUM_CHOL_VBANDS=0` override must not
    // produce zero accumulators and a vacuous accept).
    let bands = validate_bands(n, std::mem::size_of::<T>()).max(1);
    // Fused single scope (both waves in one pass): each band folds
    // finiteness + max-norm scale over its tile-row slices (the same
    // `vec_max_norm_checked` fold wave 1 ran over linear chunks — same
    // scale: `abs` exact, finite max unique; complex sub-ulp ties per
    // the kernel docs, tolerance-boundary only) and the tiled pair
    // asymmetries plus diagonal imaginary parts (wave 2's exact pair
    // set, recorded as maxima instead of per-row flags). One scope and
    // one sequential sweep instead of two (the strided pair-partner
    // reads are inherent either way), and no per-band flag vectors.
    // Exactness: accept/reject matches the legacy two-wave verdict —
    // `max > tol` iff some pair exceeds it (all values finite here,
    // else the NonFinite rescan below fires first, as in wave order),
    // and the Hermitian diagonal-vs-pair precedence is unobservable
    // (both report `StructureMismatch { kind: "Hermitian" }`).
    const VT: usize = 32;
    let trows = n.div_ceil(VT);
    let mut accs: Vec<ValidateAcc<T::Real>> = (0..bands)
        .map(|_| ValidateAcc {
            all_finite: true,
            scale_max: T::Real::zero(),
            max_asym_sqr: T::Real::zero(),
            max_diag_imag: T::Real::zero(),
        })
        .collect();
    scope_pinned(|s| {
        for (w, acc) in accs.iter_mut().enumerate() {
            s.spawn(move |_| {
                let mut tr = w;
                while tr < trows {
                    let r0 = tr * VT;
                    let r1 = (r0 + VT).min(n);
                    let (m, ok) = T::vec_max_norm_checked(&values[r0 * n..r1 * n]);
                    acc.all_finite &= ok;
                    if m > acc.scale_max {
                        acc.scale_max = m;
                    }
                    if hermitian {
                        for r in r0..r1 {
                            let d = values[r * n + r].im().abs();
                            if d > acc.max_diag_imag {
                                acc.max_diag_imag = d;
                            }
                        }
                    }
                    for tc in 0..=tr {
                        let c0 = tc * VT;
                        let c1 = (c0 + VT).min(n);
                        if tc == tr {
                            for r in r0..r1 {
                                for j in c0..r.min(c1) {
                                    let expected = if hermitian {
                                        values[j * n + r].conj()
                                    } else {
                                        values[j * n + r]
                                    };
                                    let d = (values[r * n + j] - expected).norm_sqr();
                                    if d > acc.max_asym_sqr {
                                        acc.max_asym_sqr = d;
                                    }
                                }
                            }
                        } else {
                            for r in r0..r1 {
                                for j in c0..c1 {
                                    let expected = if hermitian {
                                        values[j * n + r].conj()
                                    } else {
                                        values[j * n + r]
                                    };
                                    let d = (values[r * n + j] - expected).norm_sqr();
                                    if d > acc.max_asym_sqr {
                                        acc.max_asym_sqr = d;
                                    }
                                }
                            }
                        }
                    }
                    tr += bands;
                }
            });
        }
    });
    let mut all_finite = true;
    let mut scale_max = T::Real::zero();
    let mut max_asym_sqr = T::Real::zero();
    let mut max_diag_imag = T::Real::zero();
    for acc in &accs {
        all_finite &= acc.all_finite;
        if acc.scale_max > scale_max {
            scale_max = acc.scale_max;
        }
        if acc.max_asym_sqr > max_asym_sqr {
            max_asym_sqr = acc.max_asym_sqr;
        }
        if acc.max_diag_imag > max_diag_imag {
            max_diag_imag = acc.max_diag_imag;
        }
    }
    let (tolerance, tol_sqr) = validation_tolerance(n, values, scale_max, all_finite)?;
    if hermitian && max_diag_imag > tolerance {
        return Err(CholeskyError::StructureMismatch { kind: "Hermitian" });
    }
    if max_asym_sqr > tol_sqr {
        let kind = if hermitian { "Hermitian" } else { "symmetric" };
        return Err(CholeskyError::StructureMismatch { kind });
    }
    Ok(tolerance)
}

/// Finish entry validation from a whole-matrix `(max_norm, all_finite)`
/// fold: exact first-bad rescan when the flag trips (rare), the
/// scale-aware tolerance (with the legacy squared-overflow hypot
/// fallback), then the tiled symmetry check. Shared by
/// [`validate_parallel`] and the fused clone-validate out-of-place paths;
/// same errors either way.
/// Scale-to-tolerance block shared by [`finish_validation`] (two-wave
/// legacy path) and the fused single-scope [`validate_parallel`]:
/// exact first-bad rescan when the finite flag trips, the
/// scale-aware tolerance with the legacy squared-overflow hypot
/// fallback, and its square. One source of truth so the two paths
/// cannot diverge on accept/reject or error values.
fn validation_tolerance<T: ComplexField>(
    n: usize,
    values: &[T],
    scale_max: T::Real,
    all_finite: bool,
) -> Result<(T::Real, T::Real), CholeskyError> {
    if !all_finite {
        // Rare (never on valid input): exact first index via the exact
        // `find_non_finite` scan over the whole matrix.
        if let Some(index) = T::find_non_finite(values) {
            return Err(CholeskyError::NonFiniteMatrix { index });
        }
        debug_assert!(false, "checked fold tripped but rescan found none");
    }
    let scale = if (scale_max * scale_max).is_finite() {
        scale_max
    } else {
        // Squared-overflow edge (finite entries whose squares overflow,
        // exactly when the old `scale_sqr` went non-finite): the old
        // hypot-accumulation fallback, same values and same downstream
        // behavior.
        let mut sc = T::Real::zero();
        for value in values.iter() {
            sc = sc.max(value.norm());
        }
        sc
    };
    if !scale.is_finite() {
        return Err(CholeskyError::NumericalBreakdown {
            stage: "matrix scaling",
            index: 0,
        });
    }
    let tolerance = T::Real::epsilon()
        * T::Real::from_usize(n.max(1)).unwrap_or_else(T::Real::one)
        * scale.max(T::Real::min_positive_value());
    let tol_sqr = tolerance * tolerance;
    Ok((tolerance, tol_sqr))
}

fn finish_validation<T: ComplexField>(
    hermitian: bool,
    n: usize,
    values: &[T],
    scale_max: T::Real,
    all_finite: bool,
) -> Result<T::Real, CholeskyError> {
    let (tolerance, tol_sqr) = validation_tolerance(n, values, scale_max, all_finite)?;
    let bands = validate_bands(n, std::mem::size_of::<T>());
    symmetry_check(hermitian, n, values, tolerance, tol_sqr, bands)?;
    Ok(tolerance)
}

/// Tiled symmetry/structure check over interleaved TILE-rows (32x32
/// tiles): the wave-2 half of [`validate_parallel`], shared with the
/// fused clone-validate paths. Each tile pair reads both blocks
/// row-sequentially (full cache lines); tile-rows interleave across bands
/// so triangular skew stays balanced at any band count. Full scan, no
/// early exit: the scalar post-pass over the per-band pair-violation row
/// flags reproduces the exact legacy error precedence (earliest row wins,
/// diagonal before pairs).
fn symmetry_check<T: ComplexField>(
    hermitian: bool,
    n: usize,
    values: &[T],
    tolerance: T::Real,
    tol_sqr: T::Real,
    bands: usize,
) -> Result<(), CholeskyError> {
    let sym_kind = if hermitian { "Hermitian" } else { "symmetric" };
    // Hermitian diagonal-imag screen as a scalar row pass up front
    // (O(n), negligible): first failing row, or `n` when clean. The tiled
    // loop below is then pairs-only.
    let mut diag_bad = n;
    if hermitian {
        for r in 0..n {
            if values[r * n + r].im().abs() > tolerance {
                diag_bad = r;
                break;
            }
        }
    }
    // Per-band pair-violation row flags: the inner pair loop only ORs a
    // flag (no closure call, no early exit), which auto-vectorizes; a
    // scalar post-pass over the flags reproduces the exact legacy
    // precedence (earliest row wins, diagonal before pairs). Bands own
    // disjoint rows (tile-row `tr` belongs to band `tr % bands`), so
    // per-band flag vectors need no synchronization.
    const VT: usize = 32;
    let trows = n.div_ceil(VT);
    let mut pair_flags: Vec<Vec<bool>> = (0..bands).map(|_| vec![false; n]).collect();
    scope_pinned(|s| {
        for (w, flags) in pair_flags.iter_mut().enumerate() {
            s.spawn(move |_| {
                let mut tr = w;
                while tr < trows {
                    let r0 = tr * VT;
                    let r1 = (r0 + VT).min(n);
                    for tc in 0..=tr {
                        let c0 = tc * VT;
                        let c1 = (c0 + VT).min(n);
                        if tc == tr {
                            for r in r0..r1 {
                                let mut bad = false;
                                for j in c0..r.min(c1) {
                                    let expected = if hermitian {
                                        values[j * n + r].conj()
                                    } else {
                                        values[j * n + r]
                                    };
                                    bad |= (values[r * n + j] - expected).norm_sqr() > tol_sqr;
                                }
                                flags[r] |= bad;
                            }
                        } else {
                            for r in r0..r1 {
                                let mut bad = false;
                                for j in c0..c1 {
                                    let expected = if hermitian {
                                        values[j * n + r].conj()
                                    } else {
                                        values[j * n + r]
                                    };
                                    bad |= (values[r * n + j] - expected).norm_sqr() > tol_sqr;
                                }
                                flags[r] |= bad;
                            }
                        }
                    }
                    tr += bands;
                }
            });
        }
    });
    for r in 0..n {
        if r == diag_bad {
            return Err(CholeskyError::StructureMismatch { kind: "Hermitian" });
        }
        for flags in &pair_flags {
            if flags[r] {
                return Err(CholeskyError::StructureMismatch { kind: sym_kind });
            }
        }
    }
    Ok(())
}

fn validate_cholesky_input<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    hermitian: bool,
) -> Result<T::Real, CholeskyError> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(CholeskyError::NotSquare {
            rows: n,
            cols: matrix.ncols(),
        });
    }
    let values = matrix.as_slice();
    // Scale via max squared magnitude: one vectorizable pass with no
    // per-element `hypot` (1.5M hyps cost 4.4ms at n=1024 — more than the
    // factorization). A single sqrt at the end recovers the linear scale;
    // sqrt∘max is monotone like max∘hypot, so the tolerance agrees to ~1
    // ulp and accept/reject only differs for inputs exactly at the
    // boundary. Squared overflow (scale near the scalar range top, or f16
    // inputs past ~200) falls back to hypot accumulation: same answers as
    // before, full speed for every sane input.
    // Large inputs take the band-parallel path (bitwise identical results,
    // exact legacy error order); small inputs stay serial, below scope
    // payoff. Straight-line accumulation (no early return): the
    // per-element branch blocks SIMD; a bad-index flag preserves exact
    // error reporting while the max-reduction vectorizes. Same for the
    // symmetry pass below. Measured 2026-09: validation+ at n=1024 f32
    // 4.26 -> 0.93ms (SIMD restructure 0.5, bands the rest); checked
    // spotrf 8.4 -> 4.5, dpotrf-checked 7.2.
    if n >= 256 {
        return validate_parallel(hermitian, n, values);
    }
    let mut bad_index = usize::MAX;
    let mut scale_sqr = T::Real::zero();
    for (index, value) in values.iter().enumerate() {
        let re = value.re();
        let im = value.im();
        if !re.is_finite() || !im.is_finite() {
            bad_index = bad_index.min(index);
        }
        scale_sqr = scale_sqr.max(re * re + im * im);
    }
    if bad_index != usize::MAX {
        return Err(CholeskyError::NonFiniteMatrix { index: bad_index });
    }
    let scale = if scale_sqr.is_finite() {
        scale_sqr.sqrt()
    } else {
        let mut s = T::Real::zero();
        for value in values.iter() {
            s = s.max(value.norm());
        }
        s
    };
    if !scale.is_finite() {
        return Err(CholeskyError::NumericalBreakdown {
            stage: "matrix scaling",
            index: 0,
        });
    }
    let tolerance = T::Real::epsilon()
        * T::Real::from_usize(n.max(1)).unwrap_or_else(T::Real::one)
        * scale.max(T::Real::min_positive_value());
    // Squared symmetry comparison: same accept/reject as `norm() >
    // tolerance` whenever the scale is finite (overflow analysis: a
    // non-finite squared difference exceeds any finite squared tolerance
    // exactly when the linear comparison rejects, and two infinities
    // compare equal) — no sqrt per pair.
    let tol_sqr = tolerance * tolerance;
    // Straight-line pair scan (see scale pass): the j-loop records a
    // row flag instead of returning, so it vectorizes; the scalar report
    // after it preserves the legacy error order exactly (row i diagonal,
    // then row i pairs, row by row).
    let sym_kind = if hermitian { "Hermitian" } else { "symmetric" };
    for i in 0..n {
        // A Hermitian diagonal is real; a complex-symmetric diagonal is
        // genuinely complex, so screen the imaginary part only on the
        // Hermitian path (screening it for symmetric input wrongly rejects
        // every non-real diagonal, e.g. from A^T·A + nI).
        if hermitian && values[i * n + i].im().abs() > tolerance {
            return Err(CholeskyError::StructureMismatch { kind: "Hermitian" });
        }
        let mut row_bad = false;
        for j in 0..i {
            let expected = if hermitian {
                values[j * n + i].conj()
            } else {
                values[j * n + i]
            };
            if (values[i * n + j] - expected).norm_sqr() > tol_sqr {
                row_bad = true;
            }
        }
        if row_bad {
            return Err(CholeskyError::StructureMismatch { kind: sym_kind });
        }
    }
    Ok(tolerance)
}

fn factorize_owned<T: ComplexField>(
    mut factor: DenseMatrix<T>,
    hermitian: bool,
    tolerance: T::Real,
    screen_output: bool,
    upper_prezeroed: bool,
) -> Result<CholeskyFactorization<T>, CholeskyError> {
    let n = factor.nrows();
    if n >= CHOLESKY_BLOCKED_THRESHOLD {
        cholesky_factorize_blocked(&mut factor, hermitian, tolerance)?;
    } else {
        cholesky_panel(&mut factor, 0, n, hermitian, tolerance)?;
    }

    // End-zero, fused with the output screen when requested (see
    // `zero_upper_triangle_screened`): one row-major walk instead of a
    // zero pass plus a cold full-factor scan. When the upper triangle
    // arrives pre-zeroed (fused clone-zero entries) the screen runs
    // without zero writes (see `screen_lower_parallel`): same errors,
    // no second pass over the upper triangle.
    // Parallel screen threshold: below it the scope dispatch costs more
    // than the serial walk (same convention as validation).
    const SCREEN_PAR_MIN: usize = 256;
    if screen_output && upper_prezeroed {
        screen_lower_parallel(&mut factor, false)?;
    } else if screen_output && n >= SCREEN_PAR_MIN {
        screen_lower_parallel(&mut factor, true)?;
    } else if screen_output {
        zero_upper_triangle_screened(&mut factor)?;
    } else if !upper_prezeroed {
        zero_upper_triangle(&mut factor);
    }
    Ok(CholeskyFactorization { factor, hermitian })
}

fn first_non_finite<T: ComplexField>(values: &[T]) -> Option<usize> {
    // Vectorized on f32/f64/Complex32/Complex64 via `ComplexField::find_non_finite`.
    T::find_non_finite(values)
}

fn validate_rhs<T: ComplexField>(rhs: &[T]) -> Result<(), CholeskyError> {
    if let Some(index) = first_non_finite(rhs) {
        return Err(CholeskyError::NonFiniteRhs { index });
    }
    Ok(())
}

fn ensure_finite<T: ComplexField>(
    value: T,
    stage: &'static str,
    index: usize,
) -> Result<(), CholeskyError> {
    if !value.re().is_finite() || !value.im().is_finite() {
        return Err(CholeskyError::NumericalBreakdown { stage, index });
    }
    Ok(())
}

fn validate_generated_slice<T: ComplexField>(
    values: &[T],
    stage: &'static str,
    offset: usize,
) -> Result<(), CholeskyError> {
    if let Some(index) = first_non_finite(values) {
        return Err(CholeskyError::NumericalBreakdown {
            stage,
            index: offset + index,
        });
    }
    Ok(())
}

/// Unblocked left-looking Cholesky kernel over the diagonal block
/// `[start, end) × [start, end)` (the LAPACK POTF2 panel). The strict upper
/// triangle is left untouched.
fn cholesky_panel<T: ComplexField>(
    factor: &mut DenseMatrix<T>,
    start: usize,
    end: usize,
    hermitian: bool,
    tolerance: T::Real,
) -> Result<(), CholeskyError> {
    for j in start..end {
        let mut diagonal = factor[[j, j]];
        for k in start..j {
            let ljk = factor[[j, k]];
            diagonal -= if hermitian {
                ljk * ljk.conj()
            } else {
                ljk * ljk
            };
        }
        // Hermitian pivots are positive reals. A complex-symmetric (L·Lᵀ)
        // Schur-complement diagonal is complex in general, so only a
        // nonzero pivot can be demanded there, with the complex root below.
        // Exactly-real pivots keep the legacy positive-real path bit-for-bit
        // (including the NotPositiveDefinite rejection of real indefinite
        // input).
        let diagonal = if hermitian {
            if diagonal.im().abs() > tolerance || diagonal.re() <= tolerance {
                return Err(CholeskyError::NotPositiveDefinite { pivot: j });
            }
            T::from_real(diagonal.re().sqrt())
        } else if diagonal.im().abs() <= tolerance {
            if diagonal.re() <= tolerance {
                return Err(CholeskyError::NotPositiveDefinite { pivot: j });
            }
            T::from_real(diagonal.re().sqrt())
        } else {
            if diagonal.norm() <= tolerance {
                return Err(CholeskyError::NotPositiveDefinite { pivot: j });
            }
            diagonal.sqrt()
        };
        factor[[j, j]] = diagonal;
        // Hoisted reciprocal: one division per column instead of one per
        // element (identical arithmetic — `value * (1/diagonal)` — so the
        // factor is bitwise identical).
        let dinv = diagonal.inv();
        // Four-row vectorized column update (non-Hermitian only): the dot
        // of each row against row `j` goes through [`ComplexField::trsm_dot4`]
        // (f32/f64 vectorized twin), four substitution chains interleaved
        // like [`trsm_row4`]. Sum-then-subtract replaces the scalar loop's
        // sequential subtract, so results agree to tolerance, not bitwise
        // (same contract as `trsm_row4`). The Hermitian path keeps the
        // scalar loop (conjugation has no twin); complex-symmetric and
        // real input take the fast path.
        let mut i = j + 1;
        if !hermitian {
            let n = factor.nrows();
            let flat = factor.as_mut_slice();
            while i + 4 <= end {
                let dots = T::trsm_dot4(
                    &flat[j * n + start..j * n + j],
                    [
                        &flat[i * n + start..i * n + j],
                        &flat[(i + 1) * n + start..(i + 1) * n + j],
                        &flat[(i + 2) * n + start..(i + 2) * n + j],
                        &flat[(i + 3) * n + start..(i + 3) * n + j],
                    ],
                );
                for r in 0..4 {
                    let v = flat[(i + r) * n + j];
                    flat[(i + r) * n + j] = (v - dots[r]) * dinv;
                }
                i += 4;
            }
        }
        for i in i..end {
            let mut value = factor[[i, j]];
            for k in start..j {
                let right = if hermitian {
                    factor[[j, k]].conj()
                } else {
                    factor[[j, k]]
                };
                value -= factor[[i, k]] * right;
            }
            factor[[i, j]] = value * dinv;
        }
    }
    Ok(())
}

/// Trailing size from which the TRSM row block fans out over rayon. Below
/// this the single-threaded row loop is faster than dispatching.
/// 64 (was 256; interleaved 2026-10-02 over 0/32/64/96/128/256/384/512):
/// 64 wins or ties on every 512/1024 dtype (-11 to -26% vs 256 on
/// s/c/z512 and d/c1024; 2048 within noise), while 0 loses on tiny
/// trailing panels (dispatch exceeds the serial row loop there).
const TRSM_PAR_THRESHOLD: usize = 64;

/// Rows per TRSM band task. One task per row (16 rows on 128 threads at
/// n=2048) is dispatch-overhead-dominated and amplifies stragglers under
/// load; 64-row bands keep tasks ≈30µs+ while still filling ≤32 workers.
/// Bands must also stay below compute's unblocked-parallel threshold (128
/// rows): 128-row bands were trialled 2026-09-26 and each band re-entered
/// the parallel path (nested scopes), regressing spotrf-1024 ~0.1ms.
const TRSM_BAND_ROWS: usize = 64;

/// Upper band size for worker-sized bands (see `trsm_l21`): one band per
/// worker with this cap keeps bands below the 128-row nesting threshold
/// with margin on every pool width.
const TRSM_BAND_MAX_ROWS: usize = 120;

/// Spin-pool TRSM bands (trial, `SONIUM_CHOL_SPIN=1` opt-in): route the
/// per-panel band solves/packs through [`broadcast_bands`](sonium_compute::cpu::blas3::broadcast_bands)
/// instead of `par_iter`, keeping the exact banding while shedding the
/// ~70µs per-panel dispatch. Read once per factorization (see
/// `cholesky_factorize_blocked`).
///
/// FALSIFIED 2026-10-02 under box load (stays default-off):
/// interleaved potrf-4096 RAYON=16/masked-16, load-matched pairs —
/// spin-unchecked 147-155ms vs legacy 86-89ms (2x). Load-matched
/// phases: TRSM 67-88ms vs legacy 11-24ms; SYRK also +10-13ms on
/// unchanged code. Mechanism: static broadcast is load-fragile —
/// every panel joins all workers so every panel pays the slowest
/// core (~1ms straggler tax x 63 panels), compounded by the
/// submitter's completion-scan spin colocating with a compute worker
/// (guaranteed straggler, even quiet). Rayon's stealing routes
/// around slow cores instead (legacy degrades 8ms quiet to ~17ms
/// loaded, not 80ms). Same family as the falsified single-scope team
/// panel (static + barrier loses here). The bitwise agreement test
/// (`trsm_spin_matches_par_iter`) stays green: the executor is
/// correct, just slower under load. Untried: submitter-substitution
/// (submitter runs the collocated worker's bands instead of
/// spinning). Do not default-on without that plus quiet-box proof;
/// bandwidth-bound bands (BLAS-1, end-zero) are unaffected by this
/// (stragglers don't bind bandwidth). The SYRK poisoning (+5-13ms on
/// unchanged code, 8/8 reps) is the park delay: workers PAUSE-spin
/// 0.5ms before parking, colocated with the SYRK compute for 0.5ms
/// of every panel.
fn chol_spin_bands() -> bool {
    std::env::var("SONIUM_CHOL_SPIN")
        .map(|v| v != "0")
        .unwrap_or(false)
}

/// Solve four consecutive trailing rows at once: same gather, substitute,
/// and scatter as four [`trsm_row`] calls, but with the four substitution
/// chains interleaved. A single chain is FMA-latency-bound (~2.3 GF/s on a
/// 2000×32 panel); four independent chains let the out-of-order engine
/// overlap them (~4.4 GF/s measured). Per-row association order matches
/// [`trsm_row`] exactly, so results agree to tolerance (FMA contraction
/// choices may differ at the bit level).
///
/// `f_row0`/`a_row0` address factor row `t0` (`a` may alias `f`, as in
/// [`trsm_row`]); `l21_row0` addresses the corresponding `l21` row start
/// (usually the row start, or the block-1 offset for split phases).
/// Factor rows stride by `n`, `l21` rows by `l21_stride`, the packed `L11`
/// triangle by `l11_width`; `w` columns are solved. The plain path uses
/// `l21_stride = l11_width = w = panel_width`; split phases pass the
/// sub-block width for `l11_width`/`w` with the full `l21_stride`.
/// Callers guarantee four valid rows and disjointness across threads.
/// Row count need not divide trailing: callers solve the remainder with
/// [`trsm_row`]. Four is the width that pays: an eight-row twin (eight
/// interleaved substitution chains) trialled neutral — the scalar loop
/// is paced by load ports, not FMA latency, so extra chains add pressure
/// without throughput.
#[allow(clippy::too_many_arguments)]
// Index loops are intentional: `r`/`j`/`p` address the factor, `l21`, and
// `l11` buffers at once, which iterators cannot express.
#[allow(clippy::needless_range_loop)]
unsafe fn trsm_row4<T: ComplexField>(
    a_row0: *const T,
    f_row0: *mut T,
    l21_row0: *mut T,
    n: usize,
    l21_stride: usize,
    l11: *const T,
    l11_width: usize,
    w: usize,
    diag_inv: *const T,
    hermitian: bool,
) {
    // Phase 1: gather.
    for r in 0..4 {
        for j in 0..w {
            let value = unsafe { *a_row0.add(r * n + j) };
            unsafe {
                *l21_row0.add(r * l21_stride + j) = if hermitian { value.conj() } else { value };
            }
        }
    }
    // Phase 2: interleaved substitution (four independent chains). The
    // per-`(j, p)` scalar loop is paced by load ports, so the four dots
    // go through one vectorized call sharing the `l11`-row loads
    // (`ComplexField::trsm_dot4`; scalar fallback on complex).
    for j in 0..w {
        let mut acc = [
            unsafe { *l21_row0.add(j) },
            unsafe { *l21_row0.add(l21_stride + j) },
            unsafe { *l21_row0.add(2 * l21_stride + j) },
            unsafe { *l21_row0.add(3 * l21_stride + j) },
        ];
        // SAFETY: `l11` row `j` holds `l11_width >= w > j` elements and
        // each `l21` row holds `l21_stride >= w > j` (caller contracts);
        // zero-length slices borrow live allocations. Lengths match, so
        // the callee's equal-length contract holds.
        let dots = T::trsm_dot4(
            unsafe { core::slice::from_raw_parts(l11.add(j * l11_width), j) },
            [
                unsafe { core::slice::from_raw_parts(l21_row0, j) },
                unsafe { core::slice::from_raw_parts(l21_row0.add(l21_stride), j) },
                unsafe { core::slice::from_raw_parts(l21_row0.add(2 * l21_stride), j) },
                unsafe { core::slice::from_raw_parts(l21_row0.add(3 * l21_stride), j) },
            ],
        );
        acc[0] -= dots[0];
        acc[1] -= dots[1];
        acc[2] -= dots[2];
        acc[3] -= dots[3];
        let d = unsafe { *diag_inv.add(j) };
        unsafe {
            *l21_row0.add(j) = acc[0] * d;
            *l21_row0.add(l21_stride + j) = acc[1] * d;
            *l21_row0.add(2 * l21_stride + j) = acc[2] * d;
            *l21_row0.add(3 * l21_stride + j) = acc[3] * d;
        }
    }
    // Phase 3: factor store + pack (conjugated on the Hermitian path).
    for r in 0..4 {
        for j in 0..w {
            let x = unsafe { *l21_row0.add(r * l21_stride + j) };
            let value = if hermitian { x.conj() } else { x };
            unsafe {
                *f_row0.add(r * n + j) = value;
            }
            unsafe {
                *l21_row0.add(r * l21_stride + j) = value;
            }
        }
    }
}

/// Solve one trailing row: gather `op(A21[t, :])`, forward-substitute
/// against the packed `L11`, and scatter `L21[t, :]` into the factor row
/// and the `l21` row.
///
/// The op-transposed `l21h` pack is vestigial (no readers since the
/// `herk_lower` NoTrans trailing update; its scatter was removed
/// 2026-10-02), so no transpose runs here or in the caller. `l21_row`
/// is row-contiguous per thread, so this loop stays line-owned.
///
/// `a_row`/`f_row`/`l21_row` are the row's `panel_width` elements;
/// `l11` is the packed `panel_width × panel_width` lower triangle.
/// Callers guarantee disjoint rows across threads.
#[allow(clippy::too_many_arguments)]
// Index loops are intentional: `j`/`p` address three buffers at once
// (`a_row`, `l11`, `l21_row`), which iterators cannot express.
#[allow(clippy::needless_range_loop)]
unsafe fn trsm_row<T: ComplexField>(
    a_row: *const T,
    f_row: *mut T,
    l21_row: &mut [T],
    l11: *const T,
    diag_inv: *const T,
    panel_width: usize,
    hermitian: bool,
) {
    for j in 0..panel_width {
        let value = unsafe { *a_row.add(j) };
        l21_row[j] = if hermitian { value.conj() } else { value };
    }
    for j in 0..panel_width {
        let mut acc = l21_row[j];
        for p in 0..j {
            acc -= unsafe { *l11.add(j * panel_width + p) } * l21_row[p];
        }
        l21_row[j] = acc * unsafe { *diag_inv.add(j) };
    }
    for j in 0..panel_width {
        let x = l21_row[j];
        let value = if hermitian { x.conj() } else { x };
        unsafe {
            *f_row.add(j) = value;
        }
        l21_row[j] = value;
    }
}

/// Blocked TRSM: split the panel columns at `s`, solve block 0 per row
/// with [`trsm_row`] (identical to the first `s` steps of the full solve),
/// update block 1 with one BLAS-3 `schur_sub` over all trailing rows,
/// solve block 1 per row. Converts half the scalar substitution flops to
/// kernel pace (measured -21%/-30% on the LU U12 analogue).
///
/// Conjugation audit (Hermitian): `l21` stores true L21 (the `trsm_row`
/// double-conjugation yields exactly `A21·L11^{-H}`, verified on a 2×2).
/// The cross-block term on the raw block-1 buffer is `X0·conj(L10)ᵀ`: the
/// A pack stays plain (phase 3's gather-conjugation supplies the other
/// one) while only the B pack conjugates. The symmetric path is the same
/// code with identity conjugation. Verified against the serial loop by
/// `trsm_blocked_serial_matches_loop`, not by inspection.
/// Factor store timing matches the serial loop (block rows written as
/// solved). `l21h` is a vestigial parameter (never written; see the note
/// at the end of the body).
#[allow(clippy::too_many_arguments)]
fn trsm_l21_blocked_serial<T: ComplexField>(
    factor: &mut DenseMatrix<T>,
    panel_start: usize,
    panel_width: usize,
    panel_end: usize,
    trailing: usize,
    hermitian: bool,
    l21: &mut [T],
    _l21h: &mut [T],
    l11: &[T],
    diag_inv: &[T],
) {
    let n = factor.nrows();
    let (s, nn, l00, l10t, l11b) = pack_trsm_blocks(l11, panel_width, hermitian);
    // Phase 1: per-row block-0 solves (factor store fused, as in trsm_row),
    // four rows at a time via [`trsm_row4`].
    let flat = factor.as_mut_slice();
    let mut t = 0usize;
    for group in l21.chunks_mut(4 * panel_width) {
        let row_base = (panel_end + t) * n + panel_start;
        if group.len() == 4 * panel_width {
            unsafe {
                trsm_row4(
                    flat.as_ptr().add(row_base),
                    flat.as_mut_ptr().add(row_base),
                    group.as_mut_ptr(),
                    n,
                    panel_width,
                    l00.as_ptr(),
                    s,
                    s,
                    diag_inv.as_ptr(),
                    hermitian,
                );
            }
            t += 4;
        } else {
            for l21_row in group.chunks_mut(panel_width) {
                let rb = (panel_end + t) * n + panel_start;
                unsafe {
                    trsm_row(
                        flat.as_ptr().add(rb),
                        flat.as_mut_ptr().add(rb),
                        &mut l21_row[..s],
                        l00.as_ptr(),
                        diag_inv.as_ptr(),
                        s,
                        hermitian,
                    );
                }
                t += 1;
            }
        }
    }
    // Cross-block update over all rows: C(block 1, in factor, raw) -=
    // X0 · conj(L10)ᵀ. A packs stored block 0 plainly on both paths; only
    // the B pack conjugates (Hermitian). Failing to keep A plain breaks
    // the Hermitian path at O(1e-3) (caught by trsm_blocked_serial_matches_loop):
    // phase 3's gather-conjugation already accounts one conjugation, so a
    // second one in A double-counts.
    let mut a_pack = vec![T::zero(); trailing * s];
    for (r, l21_row) in l21.chunks(panel_width).enumerate() {
        for p in 0..s {
            a_pack[r * s + p] = l21_row[p];
        }
    }
    {
        let c_base = panel_end * n + panel_start + s;
        let cblk = &mut flat[c_base..];
        T::schur_sub(trailing, nn, s, &a_pack, &l10t, cblk, n);
    }
    // Phase 3: per-row block-1 solves off the updated factor rows, four
    // rows at a time via [`trsm_row4`].
    let mut t = 0usize;
    for group in l21.chunks_mut(4 * panel_width) {
        let row_base = (panel_end + t) * n + panel_start + s;
        if group.len() == 4 * panel_width {
            unsafe {
                trsm_row4(
                    flat.as_ptr().add(row_base),
                    flat.as_mut_ptr().add(row_base),
                    group.as_mut_ptr().add(s),
                    n,
                    panel_width,
                    l11b.as_ptr(),
                    nn,
                    nn,
                    diag_inv[s..].as_ptr(),
                    hermitian,
                );
            }
            t += 4;
        } else {
            for l21_row in group.chunks_mut(panel_width) {
                let rb = (panel_end + t) * n + panel_start + s;
                unsafe {
                    trsm_row(
                        flat.as_ptr().add(rb),
                        flat.as_mut_ptr().add(rb),
                        &mut l21_row[s..],
                        l11b.as_ptr(),
                        diag_inv[s..].as_ptr(),
                        nn,
                        hermitian,
                    );
                }
                t += 1;
            }
        }
    }
    // No `l21h` scatter: the Hermitian trailing update reads `l21`
    // directly (`herk_lower` NoTrans), so the op-transposed pack has no
    // readers and its strided scatter was a dead store pass (removed
    // 2026-10-02; symmetric was already skipped). `l21h` stays a
    // vestigial parameter so call sites and agreement tests are untouched.
}

/// Block-0/1 `L11` sub-packs for the blocked TRSM paths, from the packed
/// lower triangle (tiny): `l00` (block-0 triangle), `l10t` (block-1-by-0
/// coupling, conjugated on the Hermitian path — see the conjugation audit
/// on [`trsm_l21_blocked_serial`]), `l11b` (block-1 triangle). Returns the
/// split `(s, nn)` alongside the packs.
fn pack_trsm_blocks<T: ComplexField>(
    l11: &[T],
    panel_width: usize,
    hermitian: bool,
) -> (usize, usize, Vec<T>, Vec<T>, Vec<T>) {
    let s = panel_width / 2;
    let nn = panel_width - s;
    let mut l00 = vec![T::zero(); s * s];
    let mut l10t = vec![T::zero(); s * nn];
    let mut l11b = vec![T::zero(); nn * nn];
    for i in 0..s {
        for p in 0..=i {
            l00[i * s + p] = l11[i * panel_width + p];
        }
    }
    for j in 0..nn {
        for p in 0..s {
            let v = l11[(s + j) * panel_width + p];
            l10t[p * nn + j] = if hermitian { v.conj() } else { v };
        }
        for q in 0..=j {
            l11b[j * nn + q] = l11[(s + j) * panel_width + s + q];
        }
    }
    (s, nn, l00, l10t, l11b)
}

/// TRSM-style panel solve `L21 = A21 · L11^{-op}` fused with the factor
/// store and the `l21` pack for the trailing update.
///
/// `l21` (`trailing × panel_width`) is resized by the caller. Wide panels
/// take the blocked serial path ([`trsm_l21_blocked_serial`]) on one
/// thread, the plain band-parallel row loop on threaded runs above
/// [`TRSM_PAR_THRESHOLD`]; narrow/partial panels keep the plain per-row
/// loop (parallel above the threshold, serial below). `l21h` is a
/// vestigial parameter (never written): the Hermitian trailing update
/// reads `l21` directly since `herk_lower` NoTrans, so the Hermitian
/// scatter was a dead store pass and removed with the symmetric one
/// (the symmetric skip alone measured ~1.5ms off potrf-1024-unchecked).
///
/// One band of the band-parallel path: solve the band's factor rows in
/// place through the serial compute kernel, then pack `l21` from the
/// solved rows. Bands are independent.
///
/// # Safety
/// `band` must be the `b`-th disjoint `band_stride`-row chunk of the
/// trailing pack; `flat`/`diag` must be the factor/diagonal bases with
/// `diag_len` readable; all computed offsets in bounds.
#[allow(clippy::too_many_arguments)]
fn trsm_l21_band_work<T: ComplexField>(
    b: usize,
    band: &mut [T],
    band_stride: usize,
    panel_width: usize,
    flat: crate::banded_schur::SendMutPtr<T>,
    diag: crate::banded_schur::SendConstPtr<T>,
    diag_len: usize,
    n: usize,
    panel_start: usize,
    panel_end: usize,
    trans: sonium_compute::Transpose,
) {
    let t_base = b * band_stride;
    let band_rows = band.len() / panel_width;
    // SAFETY: per the caller contract (`b < band_count`,
    // `t_base + band_rows <= trailing`).
    let block = unsafe {
        core::slice::from_raw_parts_mut(
            flat.add((panel_end + t_base) * n + panel_start),
            (band_rows - 1) * n + panel_width,
        )
    };
    let diag_block = unsafe { core::slice::from_raw_parts(diag.add(0), diag_len) };
    T::trsm_l21_right(diag_block, n, panel_width, block, n, band_rows, trans);
    // Pack `l21` from the solved rows as-is. No `l21h` scatter (see the
    // blocked-serial note): the Hermitian trailing update reads `l21`
    // directly.
    for (t_rel, l21_row) in band.chunks_mut(panel_width).enumerate() {
        let frow = unsafe {
            core::slice::from_raw_parts(
                flat.add((panel_end + t_base + t_rel) * n + panel_start),
                panel_width,
            )
        };
        l21_row.copy_from_slice(frow);
    }
}

fn trsm_l21<T: ComplexField>(
    factor: &mut DenseMatrix<T>,
    panel_start: usize,
    panel_width: usize,
    panel_end: usize,
    trailing: usize,
    hermitian: bool,
    l21: &mut [T],
    l21h: &mut [T],
    spin: bool,
) {
    let n = factor.nrows();
    debug_assert_eq!(l21.len(), trailing * panel_width);
    debug_assert_eq!(l21h.len(), panel_width * trailing);
    let flat = factor.as_mut_slice();
    // Band-path gate, hoisted: the compute-kernel band path below solves
    // from the factor rows directly and never reads `l11`/`diag_inv`, so
    // their gather (panel² strided reads) plus `panel_width` divisions
    // plus two allocs are dead work there every panel (~21x at n=1024).
    // Skipped on the band path only (empty vecs, no allocation); both
    // serial paths keep the identical computation. Bitwise-neutral by
    // construction (unread values removed).
    let band_path = trailing >= TRSM_PAR_THRESHOLD && rayon::current_num_threads() > 1;
    let mut l11 = Vec::new();
    let mut diag_inv = Vec::new();
    if !band_path {
        l11 = vec![T::zero(); panel_width * panel_width];
        diag_inv = Vec::with_capacity(panel_width);
        for j in 0..panel_width {
            for p in 0..=j {
                l11[j * panel_width + p] = flat[(panel_start + j) * n + panel_start + p];
            }
            diag_inv.push(flat[(panel_start + j) * n + panel_start + j].inv());
        }
    }
    // Single-threaded wide panels take the blocked serial path (BLAS-3
    // cross-block update); narrow/partial panels keep the plain serial
    // loop, where blocking overhead would dominate. Threaded runs keep the
    // plain banded path: a band-parallel blocked twin was trialled and its
    // per-panel cross-block GEMM dispatch + `a_pack` gather cost more than
    // the scalar flops they replace at every width (+12ms at k=32,
    // +5ms at k=64, n=2048) — removed, not gated.
    if rayon::current_num_threads() <= 1 && panel_width > 16 {
        trsm_l21_blocked_serial(
            factor,
            panel_start,
            panel_width,
            panel_end,
            trailing,
            hermitian,
            l21,
            l21h,
            &l11,
            &diag_inv,
        );
        return;
    }
    // Narrow/partial panels (w <= 16) keep the plain loop below, where
    // blocking overhead would dominate. Wider panels take the blocked
    // serial template when they run serially (see
    // `trsm_l21_blocked_serial`); the parallel path stays plain — a
    // band-parallel blocked twin with per-panel GEMMs lost to doubled
    // dispatch (TRSM 0.83 -> 1.31ms).
    if panel_width > 16 && (trailing < TRSM_PAR_THRESHOLD || rayon::current_num_threads() <= 1) {
        trsm_l21_blocked_serial(
            factor,
            panel_start,
            panel_width,
            panel_end,
            trailing,
            hermitian,
            l21,
            l21h,
            &l11,
            &diag_inv,
        );
        return;
    }
    if band_path {
        use crate::banded_schur::SendMutPtr;
        // Band-parallel compute-TRSM path through `T::trsm_l21_right`
        // (compute's tile-transpose + AVX2 left kernel — several times
        // the scalar row-loop rate). The solve outputs true L21 directly
        // on both paths (`ConjTrans` carries the Hermitian operator,
        // matching the row loop's double-conjugation net effect), so the
        // factor rows and `l21` take the solved values as-is —
        // conjugating here would store conj(L21) (caught by
        // `trsm_band_compute_matches_serial`). Results agree with the row
        // loop to tolerance (same contract as
        // `trsm_blocked_serial_matches_loop`), not bitwise: tile
        // summation order differs.
        //
        // Two structures by worker count. With at most two waves of bands
        // per worker, each band solves its own rows serially inside the
        // single band scope: worker-sized bands stay below compute's
        // parallel threshold, so the serial kernel runs inline — one
        // scope per panel, no nesting, no per-call dispatch. Outnumbered further,
        // bands would serialize behind extra dispatches, so one solve
        // over the whole trailing block runs instead and compute
        // parallelizes internally (its per-call setup is paid once per
        // panel, not once per band). In production (4-worker f64 pool,
        // 8-worker f32 pool) this routes f32's panels to bands (two
        // waves) and f64's big panels to the hoisted solve (spotrf-1024
        // 2.53 vs 2.76, dpotrf-1024 4.13 vs 4.31 — each structure measured
        // best for its pool, not by dtype).
        let trans = if hermitian {
            Transpose::ConjTrans
        } else {
            Transpose::Trans
        };
        let flat_ptr = SendMutPtr::new(flat.as_mut_ptr());
        // Plain factor diagonal block, shared read-only by the solves
        // (compute packs its own operator per call).
        let diag_len = (panel_width - 1) * n + panel_width;
        let diag_base = unsafe { flat.as_ptr().add(panel_start * n + panel_start) };
        // One band per worker: scope dispatch costs ~1µs per task, so
        // filling each worker with a single band minimizes tasks per
        // scope while keeping every worker fed. Floored at the tuned
        // 64-row rate, capped so bands stay serial-kernel-inline (see
        // [`TRSM_BAND_MAX_ROWS`]).
        let workers = rayon::current_num_threads().max(1);
        let band_stride = trailing
            .div_ceil(workers)
            .clamp(TRSM_BAND_ROWS, TRSM_BAND_MAX_ROWS);
        let band_count = trailing.div_ceil(band_stride);
        // NOTE (reverted trial): forcing the whole-trailing solve (one
        // `trsm_l21_right` per panel instead of one per band: 32 setups,
        // not ~512) ties A/B at dpotrf-2048 (bands med 21.0 vs hoisted
        // 21.5, same-process interleaved) — per-band setup is not the
        // cost, so the band path stands. Do not re-hoist without new
        // evidence.
        if band_count <= 2 * rayon::current_num_threads() {
            use crate::banded_schur::SendConstPtr;
            let diag_send = SendConstPtr::new(diag_base);
            // Spin-pool twin (`SONIUM_CHOL_SPIN=1`): identical band
            // partition (`t_base`/`band_rows` math) and identical
            // per-band work, only the executor changes, so results are
            // bitwise identical (see `trsm_spin_matches_par_iter`).
            if spin {
                let l21_ptr = SendMutPtr::new(l21.as_mut_ptr());
                sonium_compute::cpu::blas3::broadcast_bands(band_count, &|b: usize| {
                    let t_base = b * band_stride;
                    // SAFETY: same disjoint partition as the `par_chunks_mut`
                    // twin (`b < band_count`, `t_base + rows <= trailing`,
                    // `l21.len() == trailing * panel_width` by the debug
                    // asserts); `broadcast_bands` joins before returning.
                    let band = unsafe {
                        let rows = (trailing - t_base).min(band_stride);
                        core::slice::from_raw_parts_mut(
                            l21_ptr.add(t_base * panel_width),
                            rows * panel_width,
                        )
                    };
                    let band_rows = band.len() / panel_width;
                    // Solve the band's factor rows in place (serial
                    // kernel: bands stay below the parallel threshold).
                    let block = unsafe {
                        core::slice::from_raw_parts_mut(
                            flat_ptr.add((panel_end + t_base) * n + panel_start),
                            (band_rows - 1) * n + panel_width,
                        )
                    };
                    let diag_block =
                        unsafe { core::slice::from_raw_parts(diag_send.add(0), diag_len) };
                    T::trsm_l21_right(diag_block, n, panel_width, block, n, band_rows, trans);
                    // Pack `l21` from the solved rows as-is (see above).
                    // No `l21h` scatter (see the blocked-serial note): the
                    // Hermitian trailing update reads `l21` directly.
                    for (t_rel, l21_row) in band.chunks_mut(panel_width).enumerate() {
                        let frow = unsafe {
                            core::slice::from_raw_parts(
                                flat_ptr.add((panel_end + t_base + t_rel) * n + panel_start),
                                panel_width,
                            )
                        };
                        l21_row.copy_from_slice(frow);
                    }
                });
            } else {
                // One band per task through the shared band worker (see
                // `trsm_l21_band_work`). Caller-serial bands were trialled
                // (`SONIUM_TRSM_SERIAL_BANDS`, reverted): +10-38% at
                // 512/1024 — the `par_iter` wake is smaller than the lost
                // parallelism. Do not re-serialise without new evidence.
                l21.par_chunks_mut(band_stride * panel_width)
                    .enumerate()
                    .for_each(|(b, band)| {
                        trsm_l21_band_work(
                            b,
                            band,
                            band_stride,
                            panel_width,
                            flat_ptr,
                            diag_send,
                            diag_len,
                            n,
                            panel_start,
                            panel_end,
                            trans,
                        );
                    });
            }
        } else {
            // Outnumbered: one solve over the whole trailing block, then a
            // pack-only band scope.
            let diag_block = unsafe { core::slice::from_raw_parts(diag_base, diag_len) };
            let block_all = unsafe {
                core::slice::from_raw_parts_mut(
                    flat.as_mut_ptr().add(panel_end * n + panel_start),
                    (trailing - 1) * n + panel_width,
                )
            };
            T::trsm_l21_right(diag_block, n, panel_width, block_all, n, trailing, trans);
            // Spin-pool twin (see the banded arm above): same partition,
            // same packs, bitwise identical.
            if spin {
                let l21_ptr = SendMutPtr::new(l21.as_mut_ptr());
                sonium_compute::cpu::blas3::broadcast_bands(band_count, &|b: usize| {
                    let t_base = b * band_stride;
                    // SAFETY: same disjoint partition as the `par_chunks_mut`
                    // twin (see above); `broadcast_bands` joins first.
                    let band = unsafe {
                        let rows = (trailing - t_base).min(band_stride);
                        core::slice::from_raw_parts_mut(
                            l21_ptr.add(t_base * panel_width),
                            rows * panel_width,
                        )
                    };
                    // Pack `l21` from the solved rows as-is: the solve
                    // outputs true L21 on both paths (see above), matching
                    // the row loop's store convention. No `l21h` scatter
                    // (see the blocked-serial note).
                    for (t_rel, l21_row) in band.chunks_mut(panel_width).enumerate() {
                        let frow = unsafe {
                            core::slice::from_raw_parts(
                                flat_ptr.add((panel_end + t_base + t_rel) * n + panel_start),
                                panel_width,
                            )
                        };
                        l21_row.copy_from_slice(frow);
                    }
                });
            } else {
                l21.par_chunks_mut(band_stride * panel_width)
                    .enumerate()
                    .for_each(|(b, band)| {
                        let t_base = b * band_stride;
                        // Pack `l21` from the solved rows as-is: the solve
                        // outputs true L21 on both paths (see above), matching
                        // the row loop's store convention. No `l21h` scatter
                        // (see the blocked-serial note).
                        for (t_rel, l21_row) in band.chunks_mut(panel_width).enumerate() {
                            let frow = unsafe {
                                core::slice::from_raw_parts(
                                    flat_ptr.add((panel_end + t_base + t_rel) * n + panel_start),
                                    panel_width,
                                )
                            };
                            l21_row.copy_from_slice(frow);
                        }
                    });
            }
        }
    } else {
        let (l11_slice, diag_slice) = (&l11[..], &diag_inv[..]);
        // Solve pass: groups of four rows via the interleaved
        // [`trsm_row4`] kernel, remainder with [`trsm_row`].
        let mut t = 0usize;
        for group in l21.chunks_mut(4 * panel_width) {
            let row_base = (panel_end + t) * n + panel_start;
            if group.len() == 4 * panel_width {
                unsafe {
                    trsm_row4(
                        flat.as_ptr().add(row_base),
                        flat.as_mut_ptr().add(row_base),
                        group.as_mut_ptr(),
                        n,
                        panel_width,
                        l11_slice.as_ptr(),
                        panel_width,
                        panel_width,
                        diag_slice.as_ptr(),
                        hermitian,
                    );
                }
                t += 4;
            } else {
                for l21_row in group.chunks_mut(panel_width) {
                    let rb = (panel_end + t) * n + panel_start;
                    unsafe {
                        trsm_row(
                            flat.as_ptr().add(rb),
                            flat.as_mut_ptr().add(rb),
                            l21_row,
                            l11_slice.as_ptr(),
                            diag_slice.as_ptr(),
                            panel_width,
                            hermitian,
                        );
                    }
                    t += 1;
                }
            }
        }
        // No transpose pass: `l21h` has no readers on either path
        // (see the blocked-serial note).
    }
}

/// Right-looking blocked Cholesky over the full matrix (LAPACK POTRF
/// structure). For each panel: factor the diagonal block with the unblocked
/// kernel, solve the panel's row block against the panel factor
/// (TRSM-style), and defer the trailing update to one triangle-only SYRK
/// via [`ComplexField::syrk_lower`]. The strict upper triangle is
/// zeroed at the end.
fn cholesky_factorize_blocked<T: ComplexField>(
    factor: &mut DenseMatrix<T>,
    hermitian: bool,
    tolerance: T::Real,
) -> Result<(), CholeskyError> {
    let n = factor.nrows();
    let mut l21: Vec<T> = Vec::new(); // trailing × panel_width
    // Vestigial: sized for `trsm_l21`'s never-written `l21h` parameter (see
    // its doc); kept so the debug length assert there still holds.
    let mut l21h: Vec<T> = Vec::new();
    // Phase account 2026-09-26 (temp timers, removed), n=1024 nb=64 8T per
    // factorize: panel(POTF2) 0.9, trsm 1.5-2.1, syrk 4.0-5.0 — the SYRK
    // trailing update is ~60% of dpotrf; its band dispatch/pack overhead
    // (not the kernel: 1T syrk beats OB 1.6x) is the next target.
    // dpotrf-2048 16T account (quiet box): loop 12.3 = panel 0.5 + TRSM
    // 2.4-2.7 + SYRK 9.2-9.5; clone 1.7-2.0; spawn roundtrip ~15us
    // (exonerated); harness alloc/free ~0.5-1. TRSM unroll A/B
    // (SONIUM_TRSM_NOUNROLL, m=1984): 2-row beats 1-row by <=7%, so
    // X-traffic sharing is not the bound — do not widen rows for
    // traffic reasons; the kernel gap is elsewhere (see TRSM notes).
    let block = cholesky_block_size(n);
    // Spin-pool TRSM bands (trial `SONIUM_CHOL_SPIN=1`): one read per
    // factorization, threaded into every panel's `trsm_l21`.
    let spin = chol_spin_bands();
    // NOTE (dispatch-gap roadmap): at dpotrf-2048 the phase sums trail the
    // wall by ~2.3ms over 64 scopes (~1000-1500 band tasks): pure spawn +
    // join volume (an empty-scope skeleton reproduces it exactly). Band
    // coarsening is falsified (TRSM width x2 ties; SYRK LPT-pairing loses
    // decisively — oversubscription + nest/small overlap is load-bearing).
    // The single-scope team panel (one scope + barrier, static tasks) is
    // also falsified (4 trials, `SONIUM_CHOL_TEAM` removed 2026-09):
    // mirror-pairing +14% (nest/small overlap destroyed), half-tasks
    // 22 -> 34ms (parallelism loss), LPT-greedy +36% (nested GEMM scopes
    // starve behind coarse static bands: team update CPU 1.7x classic),
    // serial-forced donut +26% (static order loses streaming locality vs
    // dynamic stealing on memory-bound bands). Lesson: the classic
    // dynamic-stealing + nested fan-out structure is load-bearing for
    // skewed streaming work; fewer joins cannot pay for lost overlap.
    // Remaining untried: cross-panel lookahead.
    //
    // LOOKAHEAD FALSIFIED 2026-09-26 (fully reverted): serial
    // next-panel block update + POTF2/TRSM on a worker overlapped with
    // the SYRK remainder, double-buffered packs, deferred nothing (no
    // pivoting — the partition is exact). Correct (maxdiff 3.5e-18 vs
    // serial, same residuals) but 5.5x SLOWER, fully explained by the
    // serial composition alone (no scope): the naive block update costs
    // ~100ms total (0.27 GFLOP at scalar rate) against a ~2.5ms
    // hidden-TRSM prize. Overlap only pays when hidden work dwarfs the
    // update cost (LU won at ~7:1; here ~1:1). Do not retry without a
    // kernel-rate serial block update first.
    for panel_start in (0..n).step_by(block) {
        let panel_end = (panel_start + block).min(n);
        let panel_width = panel_end - panel_start;

        // 1. Factor the diagonal panel (POTF2).
        cholesky_panel(factor, panel_start, panel_end, hermitian, tolerance)?;

        let trailing = n - panel_end;
        if trailing == 0 {
            continue;
        }

        // 2+3. TRSM-style row-block solve L21 = A21 · L11^{-op} (op is the
        //    conjugate transpose for HPD input, the plain transpose for
        //    symmetric input), fused with storing L21 into the factor and
        //    packing it for the trailing update. Trailing rows are
        //    independent — one k×k forward substitution each — so large
        //    row blocks fan out over rayon.
        // `l21` needs capacity only: `trsm_l21` writes every element
        // before any read, so a zeroing `resize` here would be dead
        // stores (same reserve-then-fill contract as the GEMM pack
        // buffers). `l21h` is vestigial (never written, never read);
        // sized only for the debug length assert in `trsm_l21`.
        l21.clear();
        l21.reserve_exact(trailing * panel_width);
        // SAFETY: see above.
        unsafe {
            l21.set_len(trailing * panel_width);
        }
        l21h.clear();
        l21h.reserve_exact(panel_width * trailing);
        // SAFETY: nothing reads `l21h` (`T` needs no drop, so the Vec
        // teardown touches no element); the length only satisfies the
        // callee's debug assert.
        unsafe {
            l21h.set_len(panel_width * trailing);
        }
        trsm_l21(
            factor,
            panel_start,
            panel_width,
            panel_end,
            trailing,
            hermitian,
            &mut l21,
            &mut l21h,
            spin,
        );
        // 4. SYRK trailing update A22 -= L21 · op(L21) in place, writing
        //    only the lower triangle: half the flops and half the writes
        //    of the old full-rectangle `schur_sub` GEMM (which also
        //    touched the strict upper triangle, zeroed at the end, and
        //    needed a per-band strided `u12` gather). Later panels consume
        //    the lower triangle only. Both paths read the row-major `l21`
        //    pack: symmetric via `syrk_lower(NoTrans)`, Hermitian via
        //    `herk_lower` (`C -= L21·L21ᴴ`, conjugation folded into the
        //    sweep's B-block pack). The old Hermitian route packed a
        //    transposed `l21h` (`k × trailing`) and re-read it strided in
        //    every SYRK band (~8x redundant 32KB-stride traffic); the
        //    NoTrans Hermitian form streams instead (removed 2026-10-02).
        //
        // NOTE (reverted trial): skipping the `l21` pack on the symmetric
        // path (SYRK reading the solved factor rows directly with
        // stride-`n` `lda`) loses A/B 5/5 at dpotrf-2048 (pack 20.1 vs
        // skip 22.9ms med, same process interleaved): standalone strided
        // SYRK matches packed rates, but in-situ the packed buffer's
        // cache warmth plus friendlier alignment wins by ~3ms. Do not
        // re-skip without addressing the in-situ strided rate first.
        let flat = factor.as_mut_slice();
        let c_base = panel_end * n + panel_end;
        if hermitian {
            T::herk_lower(
                trailing,
                panel_width,
                -T::one(),
                &l21[..],
                panel_width,
                T::one(),
                &mut flat[c_base..],
                n,
            );
        } else {
            T::syrk_lower(
                Transpose::NoTrans,
                trailing,
                panel_width,
                -T::one(),
                &l21[..],
                panel_width,
                T::one(),
                &mut flat[c_base..],
                n,
            );
        }
    }
    // End-zero (not pre-zero): pre-zeroing at `factorize_owned` top was
    // trialled 2026-09-26 (validation-hot lines go cold) and lost: for
    // n >= 256 validation itself is band-parallel, so its lines scatter
    // over 8 workers' caches and no same-thread hot window exists — the
    // upper-triangle RFOs are compulsory either way, and 3-pair
    // interleaved dpotrf-1024 8T went 7.09 end-zero vs 7.35 pre-zero.
    // Zeroing now lives in `factorize_owned` (fused with the output
    // screen when requested); this path leaves the upper triangle
    // untouched for its caller to finish.
    Ok(())
}

fn zero_upper_triangle<T: ComplexField>(factor: &mut DenseMatrix<T>) {
    let n = factor.nrows();
    // Row-major fills: the old column-stride element loop touched 2M
    // cache lines with bounds checks (~20ms at n=2048); contiguous row
    // fills run at memset speed with identical values.
    //
    // Band-parallel above `ZERO_UPPER_PAR_MIN`: the serial fill is ~5ms
    // at n=2048 (134MB upper triangle), all of it on the in-place path's
    // critical end (in-place dpotrf-2048 carries no clone to hide
    // behind). Rows are disjoint, so bands zero independently with no
    // error semantics to preserve. Plain `rayon::scope` (not
    // `scope_pinned`): same enclosing-pool reasoning as
    // `screen_lower_parallel`. Below the threshold the scope dispatch
    // costs more than the fill.
    const ZERO_UPPER_PAR_MIN: usize = 512;
    if n >= ZERO_UPPER_PAR_MIN && rayon::current_num_threads() > 1 {
        let data_ptr = crate::banded_schur::SendMutPtr::new(factor.as_mut_slice().as_mut_ptr());
        let workers = rayon::current_num_threads().max(1);
        let bands = workers.min(n).max(1);
        let rows_per = n.div_ceil(bands);
        rayon::scope(|s| {
            for b in 0..bands {
                let r0 = b * rows_per;
                let r1 = (r0 + rows_per).min(n);
                s.spawn(move |_| {
                    for j in r0..r1 {
                        // SAFETY: band `r0..r1` owns disjoint rows; bounds
                        // per the `nrows` contract.
                        let row =
                            unsafe { core::slice::from_raw_parts_mut(data_ptr.add(j * n), n) };
                        row[j + 1..].fill(T::zero());
                    }
                });
            }
        });
        return;
    }
    let data = factor.as_mut_slice();
    for j in 0..n {
        data[j * n + j + 1..(j + 1) * n].fill(T::zero());
    }
}

/// Zero the strict upper triangle while screening the generated lower
/// triangle + diagonal for non-finites: the fused validation-zero pass
/// the end-zero note calls for. One row-major walk replaces the zero
/// pass plus the separate end-screen (`validate_generated_slice`), which
/// ran after zeroing and therefore only ever saw the lower triangle
/// anyway (upper already finite zeros). First-bad index, stage, and
/// variant match the old screen exactly; the upper triangle is never
/// screened (old screen saw zeros there too). On error the factor is
/// unspecified, as with the old screen-after-zero.
fn zero_upper_triangle_screened<T: ComplexField>(
    factor: &mut DenseMatrix<T>,
) -> Result<(), CholeskyError> {
    let n = factor.nrows();
    let data = factor.as_mut_slice();
    for j in 0..n {
        let row = &mut data[j * n..(j + 1) * n];
        if let Some(i) = first_non_finite(&row[..=j]) {
            return Err(CholeskyError::NumericalBreakdown {
                stage: "factorization",
                index: j * n + i,
            });
        }
        row[j + 1..].fill(T::zero());
    }
    Ok(())
}

/// Screen the generated lower triangle + diagonal for non-finites, with
/// optional strict-upper zeroing: the parallel twin of
/// [`zero_upper_triangle_screened`]. With `zero_upper = false` the upper
/// triangle is untouched (fused clone-zero entries); with `true` each
/// band zeroes its rows' upper part right after screening them (in-place
/// entries whose upper triangle arrives unzeroed). Row bands over the
/// enclosing pool each report their first bad linear index; the minimum
/// reproduces the serial first-bad index exactly (rows are disjoint,
/// indices are linear). Same errors and values as the serial walk either
/// way.
fn screen_lower_parallel<T: ComplexField>(
    factor: &mut DenseMatrix<T>,
    zero_upper: bool,
) -> Result<(), CholeskyError> {
    let n = factor.nrows();
    let data = factor.as_mut_slice();
    let workers = rayon::current_num_threads().max(1);
    let bands = workers.min(n).max(1);
    let rows_per = n.div_ceil(bands);
    let first_bad = &std::sync::Mutex::new(usize::MAX);
    // Plain `rayon::scope` (not `scope_pinned`): the factor is hot on the
    // enclosing factorization pool and the screen must not migrate it to
    // the mirror pool. Inside a pool install this stays on that pool;
    // from a bare thread it uses the global pool, as before.
    rayon::scope(|s| {
        for b in 0..bands {
            let r0 = b * rows_per;
            let r1 = (r0 + rows_per).min(n);
            let data_ptr = crate::banded_schur::SendMutPtr::new(data.as_mut_ptr());
            s.spawn(move |_| {
                let mut local = usize::MAX;
                for j in r0..r1 {
                    // SAFETY: band `r0..r1` owns disjoint rows; bounds per
                    // the caller contract.
                    let row = unsafe { core::slice::from_raw_parts_mut(data_ptr.add(j * n), n) };
                    if let Some(i) = first_non_finite(&row[..=j]) {
                        local = local.min(j * n + i);
                    }
                    if zero_upper {
                        row[j + 1..].fill(T::zero());
                    }
                }
                if local != usize::MAX {
                    let mut g = first_bad.lock().unwrap();
                    *g = (*g).min(local);
                }
            });
        }
    });
    let bad = first_bad.lock().unwrap();
    if *bad != usize::MAX {
        return Err(CholeskyError::NumericalBreakdown {
            stage: "factorization",
            index: *bad,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn cholesky_default_block_size_by_order() {
        // Locks the size-dependent default (nb=48 at n <= 1024 beats 64
        // on spotrf/dpotrf at 256/512/1024; 64 stands above).
        // `SONIUM_CHOL_NB` still overrides per call. This is the only
        // test touching the variable, so save/remove/restore here is
        // safe.
        let saved = std::env::var_os("SONIUM_CHOL_NB");
        unsafe {
            std::env::remove_var("SONIUM_CHOL_NB");
        }
        assert_eq!(cholesky_block_size(256), 48);
        assert_eq!(cholesky_block_size(1024), 48);
        assert_eq!(cholesky_block_size(2048), 64);
        if let Some(v) = saved {
            unsafe {
                std::env::set_var("SONIUM_CHOL_NB", v);
            }
        }
    }

    #[test]
    fn mirror_install_matches_spawn_bitwise() {
        // The mirror funnel launches by `install` (borrow) instead of
        // `spawn`: same pool, same width, same task splits, so the
        // factor must match the spawn path bitwise. n=256 exercises the
        // blocked path (threshold 192) on a diagonally-dominant SPD
        // matrix.
        let n = 256usize;
        let matrix = DenseMatrix::from_fn(n, n, |i, j| {
            let x = ((i + j) * 31 + (i * j) % 64) as f64;
            (if i == j { n as f64 } else { 0.0 }) + 0.01 * (x * 0.61803398875).sin()
        });
        let tolerance = validate_cholesky_input(&matrix, false).unwrap();
        let installed = sonium_compute::cpu::blas3::run_mirror(|| {
            factorize_owned(matrix.clone(), false, tolerance, true, false).unwrap()
        });
        let spawned = sonium_compute::cpu::blas3::run_on_worker(move || {
            factorize_owned(matrix, false, tolerance, true, false).unwrap()
        });
        assert_eq!(
            installed.factor().as_slice(),
            spawned.factor().as_slice()
        );
    }

    #[test]
    fn spd_solve_and_many_rhs() {
        let matrix = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
        );
        let factor = cholesky_factorize(&matrix).unwrap();
        let rhs = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mut solutions = vec![0.0; rhs.len()];
        factor.solve_many_into(&rhs, &mut solutions, 2).unwrap();
        for right_hand_side in 0..2 {
            let x: Vec<_> = (0..3)
                .map(|row| solutions[row * 2 + right_hand_side])
                .collect();
            let actual = matrix.matvec(&x);
            for row in 0..3 {
                assert!((actual[row] - rhs[row * 2 + right_hand_side]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn hpd_complex_solve() {
        let matrix = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let rhs = vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.5)];
        let factor = cholesky_factorize_hermitian(&matrix).unwrap();
        let solution = factor.solve(&rhs).unwrap();
        let actual = matrix.matvec(&solution);
        for (actual, expected) in actual.iter().zip(&rhs) {
            assert!((*actual - *expected).norm() < 1e-12);
        }
    }

    #[test]
    fn screened_zero_matches_plain_zero_and_screen() {
        // Fused validation-zero contract: on clean input the result is
        // identical to zero_upper_triangle (lower preserved, upper
        // zeroed); a lower-triangle non-finite reports the exact
        // row-major index with stage "factorization" (the old
        // screen-after-zero contract); an upper-triangle non-finite is
        // zeroed away unscreened, exactly as the old screen saw zeros.
        let clean = DenseMatrix::from_row_major_vec(4, 4, (1..=16).map(|v| v as f64).collect());
        let mut fused = clean.clone();
        zero_upper_triangle_screened(&mut fused).unwrap();
        let mut plain = clean.clone();
        zero_upper_triangle(&mut plain);
        assert_eq!(fused.as_slice(), plain.as_slice());
        for j in 0..4 {
            for i in 0..4 {
                let v = fused.as_slice()[j * 4 + i];
                if i <= j {
                    assert_eq!(v, (j * 4 + i + 1) as f64);
                } else {
                    assert_eq!(v, 0.0);
                }
            }
        }

        let mut lower_bad = DenseMatrix::from_row_major_vec(4, 4, vec![1.0f64; 16]);
        lower_bad.as_mut_slice()[9] = f64::INFINITY;
        lower_bad.as_mut_slice()[15] = f64::NAN;
        assert!(matches!(
            zero_upper_triangle_screened(&mut lower_bad),
            Err(CholeskyError::NumericalBreakdown {
                stage: "factorization",
                index: 9,
            })
        ));

        let mut upper_bad = DenseMatrix::from_row_major_vec(4, 4, vec![1.0f64; 16]);
        upper_bad.as_mut_slice()[3] = f64::INFINITY;
        zero_upper_triangle_screened(&mut upper_bad).unwrap();
        assert_eq!(upper_bad.as_slice()[3], 0.0);
    }

    #[test]
    fn blocked_lower_update_large_spd() {
        // Large enough that the first trailing block crosses the parallel
        // threshold: exercises the triangle-only SYRK trailing update for
        // correctness.
        let n = 600;
        let matrix = DenseMatrix::from_fn(n, n, |i, j| {
            if i == j {
                4.0
            } else if i + 1 == j || j + 1 == i {
                -1.0
            } else {
                0.0
            }
        });
        let factor = cholesky_factorize(&matrix).expect("large SPD should factorize");
        let rhs: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let solution = factor.solve(&rhs).expect("large SPD solve should succeed");
        let actual = matrix.matvec(&solution);
        let mut num = 0.0;
        let mut den = 0.0;
        for (a, b) in actual.iter().zip(&rhs) {
            num += (a - b) * (a - b);
            den += b * b;
        }
        let residual = (num / den).sqrt();
        assert!(
            residual < 1e-10,
            "lower-update SPD residual {residual:e} exceeds tolerance"
        );
    }

    #[test]
    fn factorize_in_place_matches_cloning_api() {
        // Owned-input path must validate identically and produce the same
        // factor as the cloning API.
        let matrix = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
        );
        let owned =
            cholesky_factorize_in_place(matrix.clone()).expect("in-place SPD should factorize");
        let cloned = cholesky_factorize(&matrix).expect("cloning SPD should factorize");
        let rhs = vec![1.0, 4.0, 2.0];
        let x_owned = owned.solve(&rhs).expect("in-place solve");
        let x_cloned = cloned.solve(&rhs).expect("cloning solve");
        for (a, b) in x_owned.iter().zip(&x_cloned) {
            assert!((a - b).abs() < 1e-14);
        }
        // Same rejection contract: asymmetric and indefinite inputs fail.
        let nonsymmetric = DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 1.0, 0.0, 2.0]);
        assert!(matches!(
            cholesky_factorize_in_place(nonsymmetric),
            Err(CholeskyError::StructureMismatch { .. })
        ));
        let indefinite = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        assert!(matches!(
            cholesky_factorize_in_place(indefinite),
            Err(CholeskyError::NotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn in_place_matches_borrowed_bitwise() {
        // Owned entry must produce the identical factor (same validation,
        // same kernels, no clone): blocked + unblocked sizes, both dtypes.
        for n in [3usize, 200] {
            let matrix = spd_matrix(n);
            let borrowed = cholesky_factorize(&matrix).unwrap();
            let owned = cholesky_factorize_in_place(DenseMatrix::from_row_major_vec(
                n,
                n,
                matrix.as_slice().to_vec(),
            ))
            .unwrap();
            assert_eq!(
                borrowed.factor().as_slice(),
                owned.factor().as_slice(),
                "f64 factor differs at n={n}"
            );
        }
        let matrix = hermitian_matrix(200);
        let borrowed = cholesky_factorize_hermitian(&matrix).unwrap();
        let owned = cholesky_factorize_in_place_hermitian(DenseMatrix::from_row_major_vec(
            200,
            200,
            matrix.as_slice().to_vec(),
        ))
        .unwrap();
        assert_eq!(
            borrowed.factor().as_slice(),
            owned.factor().as_slice(),
            "complex factor differs"
        );
    }

    #[test]
    fn in_place_rejects_bad_input_like_borrowed() {
        // Same error precedence and payloads as the borrowed entry.
        let nonsymmetric = DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 1.0, 0.0, 2.0]);
        assert!(matches!(
            cholesky_factorize_in_place(nonsymmetric),
            Err(CholeskyError::StructureMismatch { .. })
        ));
        let nonfinite =
            DenseMatrix::from_row_major_vec(2, 2, vec![2.0_f64, f64::NAN, f64::NAN, 2.0]);
        assert!(matches!(
            cholesky_factorize_in_place(nonfinite),
            Err(CholeskyError::NonFiniteMatrix { index: 1 })
        ));
        let indefinite = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        assert!(matches!(
            cholesky_factorize_in_place(indefinite),
            Err(CholeskyError::NotPositiveDefinite { .. })
        ));
        let rectangular = DenseMatrix::from_row_major_vec(2, 3, vec![1.0; 6]);
        assert!(matches!(
            cholesky_factorize_in_place(rectangular),
            Err(CholeskyError::NotSquare { .. })
        ));
    }

    #[test]
    fn rejects_wrong_structure_and_non_positive_definite() {
        let nonsymmetric = DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 1.0, 0.0, 2.0]);
        assert!(matches!(
            cholesky_factorize(&nonsymmetric),
            Err(CholeskyError::StructureMismatch { .. })
        ));
        let indefinite = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        assert!(matches!(
            cholesky_factorize(&indefinite),
            Err(CholeskyError::NotPositiveDefinite { .. })
        ));
    }

    /// Diagonally dominant SPD matrix with deterministic symmetric entries.
    fn spd_matrix(n: usize) -> DenseMatrix<f64> {
        DenseMatrix::from_fn(n, n, |i, j| {
            let (a, b) = (i.min(j), i.max(j));
            let off = (((a * 37 + b * 17 + 11) % 101) as f64 - 50.0) / 101.0;
            if i == j { off.abs() + n as f64 } else { off }
        })
    }

    /// HPD complex matrix `M·Mᴴ + n·I` with deterministic M — non-trivial
    /// off-diagonal phases, comfortably positive definite.
    fn hermitian_matrix(n: usize) -> DenseMatrix<Complex64> {
        let m = DenseMatrix::from_fn(n, n, |i, j| {
            let re = (((i * 29 + j * 13 + 7) % 97) as f64 - 48.0) / 194.0;
            let im = (((i * 11 + j * 31 + 5) % 89) as f64 - 44.0) / 356.0;
            Complex64::new(re, im)
        });
        DenseMatrix::from_fn(n, n, |i, j| {
            let mut value = Complex64::new(if i == j { n as f64 } else { 0.0 }, 0.0);
            for k in 0..n {
                value += m[[i, k]] * m[[j, k]].conj();
            }
            value
        })
    }

    /// Reference unblocked factor over the full matrix, for comparing the
    /// blocked path against the POTF2 kernel it is built from.
    fn unblocked_factor<T: ComplexField>(
        matrix: &DenseMatrix<T>,
        hermitian: bool,
    ) -> DenseMatrix<T> {
        let n = matrix.nrows();
        let scale = matrix
            .as_slice()
            .iter()
            .map(ComplexField::norm)
            .fold(T::Real::zero(), T::Real::max);
        let tolerance = T::Real::epsilon()
            * T::Real::from_usize(n.max(1)).unwrap()
            * scale.max(T::Real::min_positive_value());
        let mut factor = matrix.clone();
        cholesky_panel(&mut factor, 0, n, hermitian, tolerance).unwrap();
        zero_upper_triangle(&mut factor);
        factor
    }

    fn assert_factors_agree<T: ComplexField<Real = f64>>(
        blocked: &DenseMatrix<T>,
        unblocked: &DenseMatrix<T>,
        tolerance: f64,
    ) {
        let n = blocked.nrows();
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..=i {
                max_diff = max_diff.max((blocked[[i, j]] - unblocked[[i, j]]).norm());
            }
        }
        assert!(
            max_diff <= tolerance,
            "blocked vs unblocked factor difference {max_diff:e} exceeds {tolerance:e}"
        );
    }

    /// Blocked serial TRSM matches the plain serial row loop on identical
    /// panel inputs. n=232 gives a full 32-wide panel with trailing=200:
    /// below the rayon threshold, so `trsm_l21` takes its serial loop even
    /// on many threads (deterministic reference), while the blocked path
    /// is invoked directly. Covers both the symmetric and the Hermitian
    /// (conjugation-at-pack-time) branches.
    #[test]
    fn trsm_blocked_serial_matches_loop() {
        for hermitian in [false, true] {
            let n = 232usize;
            let panel_width = 32usize;
            let trailing = n - panel_width;
            if hermitian {
                let a = hermitian_matrix(n);
                let scale = a
                    .as_slice()
                    .iter()
                    .map(ComplexField::norm)
                    .fold(0.0_f64, f64::max);
                let tolerance = f64::EPSILON * n as f64 * scale.max(f64::MIN_POSITIVE);
                let mut f_loop = a.clone();
                cholesky_panel(&mut f_loop, 0, panel_width, true, tolerance).unwrap();
                let mut f_block = f_loop.clone();
                let mut l21_loop = vec![Complex64::new(0.0, 0.0); trailing * panel_width];
                let mut l21h_loop = vec![Complex64::new(0.0, 0.0); panel_width * trailing];
                trsm_l21(
                    &mut f_loop,
                    0,
                    panel_width,
                    panel_width,
                    trailing,
                    true,
                    &mut l21_loop,
                    &mut l21h_loop,
                    false,
                );
                let mut l21_block = vec![Complex64::new(0.0, 0.0); trailing * panel_width];
                let mut l21h_block = vec![Complex64::new(0.0, 0.0); panel_width * trailing];
                let l11_pack = pack_l11_for_test(&f_block, panel_width);
                let diag_pack = diag_inv_for_test(&f_block, panel_width);
                trsm_l21_blocked_serial(
                    &mut f_block,
                    0,
                    panel_width,
                    panel_width,
                    trailing,
                    true,
                    &mut l21_block,
                    &mut l21h_block,
                    &l11_pack,
                    &diag_pack,
                );
                assert_trsm_agree(
                    &l21_loop,
                    &l21_block,
                    &l21h_loop,
                    &l21h_block,
                    &f_loop,
                    &f_block,
                    n,
                    panel_width,
                    1e-12,
                    true,
                );
            } else {
                let a = spd_matrix(n);
                let scale = a
                    .as_slice()
                    .iter()
                    .map(ComplexField::norm)
                    .fold(0.0_f64, f64::max);
                let tolerance = f64::EPSILON * n as f64 * scale.max(f64::MIN_POSITIVE);
                let mut f_loop = a.clone();
                cholesky_panel(&mut f_loop, 0, panel_width, false, tolerance).unwrap();
                let mut f_block = f_loop.clone();
                let mut l21_loop = vec![0.0; trailing * panel_width];
                let mut l21h_loop = vec![0.0; panel_width * trailing];
                trsm_l21(
                    &mut f_loop,
                    0,
                    panel_width,
                    panel_width,
                    trailing,
                    false,
                    &mut l21_loop,
                    &mut l21h_loop,
                    false,
                );
                let mut l21_block = vec![0.0; trailing * panel_width];
                let mut l21h_block = vec![0.0; panel_width * trailing];
                let l11_pack = pack_l11_for_test(&f_block, panel_width);
                let diag_pack = diag_inv_for_test(&f_block, panel_width);
                trsm_l21_blocked_serial(
                    &mut f_block,
                    0,
                    panel_width,
                    panel_width,
                    trailing,
                    false,
                    &mut l21_block,
                    &mut l21h_block,
                    &l11_pack,
                    &diag_pack,
                );
                assert_trsm_agree(
                    &l21_loop,
                    &l21_block,
                    &l21h_loop,
                    &l21h_block,
                    &f_loop,
                    &f_block,
                    n,
                    panel_width,
                    1e-12,
                    false,
                );
            }
        }
    }

    /// Band compute-TRSM path matches the serial reference on identical
    /// panel inputs. n=384 gives a full 32-wide panel with trailing=352:
    /// above the rayon threshold, so `trsm_l21` on a multi-thread pool
    /// takes the band compute-TRSM path, while a single-thread pool
    /// install forces the blocked serial reference (proven loop-equal by
    /// `trsm_blocked_serial_matches_loop`). Covers both the symmetric
    /// and the Hermitian (conjugation-at-pack-time) branches, including
    /// a partial final band (352 = 5.5 × 64).
    #[test]
    fn trsm_band_compute_matches_serial() {
        let pool_one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        // Both structures: 2 workers are outnumbered twice over by the 6
        // bands (352 = 5.5 × 64) so the hoisted solve runs; 8 workers
        // take the banded-serial arm (two waves). Each is compared
        // against the same serial reference.
        for workers in [2usize, 8usize] {
            let pool_multi = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for hermitian in [false, true] {
                let n = 384usize;
                let panel_width = 32usize;
                let trailing = n - panel_width;
                if hermitian {
                    let a = hermitian_matrix(n);
                    let scale = a
                        .as_slice()
                        .iter()
                        .map(ComplexField::norm)
                        .fold(0.0_f64, f64::max);
                    let tolerance = f64::EPSILON * n as f64 * scale.max(f64::MIN_POSITIVE);
                    let mut f_ref = a.clone();
                    cholesky_panel(&mut f_ref, 0, panel_width, true, tolerance).unwrap();
                    let mut f_band = f_ref.clone();
                    let mut l21_ref = vec![Complex64::new(0.0, 0.0); trailing * panel_width];
                    let mut l21h_ref = vec![Complex64::new(0.0, 0.0); panel_width * trailing];
                    pool_one.install(|| {
                        trsm_l21(
                            &mut f_ref,
                            0,
                            panel_width,
                            panel_width,
                            trailing,
                            true,
                            &mut l21_ref,
                            &mut l21h_ref,
                            false,
                        );
                    });
                    let mut l21_band = vec![Complex64::new(0.0, 0.0); trailing * panel_width];
                    let mut l21h_band = vec![Complex64::new(0.0, 0.0); panel_width * trailing];
                    pool_multi.install(|| {
                        trsm_l21(
                            &mut f_band,
                            0,
                            panel_width,
                            panel_width,
                            trailing,
                            true,
                            &mut l21_band,
                            &mut l21h_band,
                            false,
                        );
                    });
                    assert_trsm_agree(
                        &l21_ref,
                        &l21_band,
                        &l21h_ref,
                        &l21h_band,
                        &f_ref,
                        &f_band,
                        n,
                        panel_width,
                        1e-12,
                        true,
                    );
                } else {
                    let a = spd_matrix(n);
                    let scale = a
                        .as_slice()
                        .iter()
                        .map(ComplexField::norm)
                        .fold(0.0_f64, f64::max);
                    let tolerance = f64::EPSILON * n as f64 * scale.max(f64::MIN_POSITIVE);
                    let mut f_ref = a.clone();
                    cholesky_panel(&mut f_ref, 0, panel_width, false, tolerance).unwrap();
                    let mut f_band = f_ref.clone();
                    let mut l21_ref = vec![0.0; trailing * panel_width];
                    let mut l21h_ref = vec![0.0; panel_width * trailing];
                    pool_one.install(|| {
                        trsm_l21(
                            &mut f_ref,
                            0,
                            panel_width,
                            panel_width,
                            trailing,
                            false,
                            &mut l21_ref,
                            &mut l21h_ref,
                            false,
                        );
                    });
                    let mut l21_band = vec![0.0; trailing * panel_width];
                    let mut l21h_band = vec![0.0; panel_width * trailing];
                    pool_multi.install(|| {
                        trsm_l21(
                            &mut f_band,
                            0,
                            panel_width,
                            panel_width,
                            trailing,
                            false,
                            &mut l21_band,
                            &mut l21h_band,
                            false,
                        );
                    });
                    assert_trsm_agree(
                        &l21_ref,
                        &l21_band,
                        &l21h_ref,
                        &l21h_band,
                        &f_ref,
                        &f_band,
                        n,
                        panel_width,
                        1e-12,
                        false,
                    );
                }
            }
        }
    }

    #[test]
    fn trsm_spin_matches_par_iter() {
        // The spin-pool twin changes only the executor (same band
        // partition, same per-band work), so both structures agree with
        // the `par_iter` path bitwise. Both structures as in
        // `trsm_band_compute_matches_serial`: 2 workers run the hoisted
        // solve, 8 workers the banded arm.
        for workers in [2usize, 8usize] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for hermitian in [false, true] {
                let n = 384usize;
                let panel_width = 32usize;
                let trailing = n - panel_width;
                if hermitian {
                    let a = hermitian_matrix(n);
                    let scale = a
                        .as_slice()
                        .iter()
                        .map(ComplexField::norm)
                        .fold(0.0_f64, f64::max);
                    let tolerance = f64::EPSILON * n as f64 * scale.max(f64::MIN_POSITIVE);
                    let mut f_base = a.clone();
                    cholesky_panel(&mut f_base, 0, panel_width, true, tolerance).unwrap();
                    let mut f_par = f_base.clone();
                    let mut f_spin = f_base.clone();
                    let mut l21_par = vec![Complex64::new(0.0, 0.0); trailing * panel_width];
                    let mut l21h_par = vec![Complex64::new(0.0, 0.0); panel_width * trailing];
                    let mut l21_spin = vec![Complex64::new(0.0, 0.0); trailing * panel_width];
                    let mut l21h_spin = vec![Complex64::new(0.0, 0.0); panel_width * trailing];
                    pool.install(|| {
                        trsm_l21(
                            &mut f_par,
                            0,
                            panel_width,
                            panel_width,
                            trailing,
                            true,
                            &mut l21_par,
                            &mut l21h_par,
                            false,
                        );
                    });
                    pool.install(|| {
                        trsm_l21(
                            &mut f_spin,
                            0,
                            panel_width,
                            panel_width,
                            trailing,
                            true,
                            &mut l21_spin,
                            &mut l21h_spin,
                            true,
                        );
                    });
                    assert_eq!(l21_par, l21_spin, "l21 workers={workers}");
                    assert_eq!(l21h_par, l21h_spin, "l21h workers={workers}");
                    assert_eq!(
                        f_par.as_slice(),
                        f_spin.as_slice(),
                        "factor workers={workers}"
                    );
                } else {
                    let a = spd_matrix(n);
                    let scale = a
                        .as_slice()
                        .iter()
                        .map(ComplexField::norm)
                        .fold(0.0_f64, f64::max);
                    let tolerance = f64::EPSILON * n as f64 * scale.max(f64::MIN_POSITIVE);
                    let mut f_base = a.clone();
                    cholesky_panel(&mut f_base, 0, panel_width, false, tolerance).unwrap();
                    let mut f_par = f_base.clone();
                    let mut f_spin = f_base.clone();
                    let mut l21_par = vec![0.0; trailing * panel_width];
                    let mut l21h_par = vec![0.0; panel_width * trailing];
                    let mut l21_spin = vec![0.0; trailing * panel_width];
                    let mut l21h_spin = vec![0.0; panel_width * trailing];
                    pool.install(|| {
                        trsm_l21(
                            &mut f_par,
                            0,
                            panel_width,
                            panel_width,
                            trailing,
                            false,
                            &mut l21_par,
                            &mut l21h_par,
                            false,
                        );
                    });
                    pool.install(|| {
                        trsm_l21(
                            &mut f_spin,
                            0,
                            panel_width,
                            panel_width,
                            trailing,
                            false,
                            &mut l21_spin,
                            &mut l21h_spin,
                            true,
                        );
                    });
                    assert_eq!(l21_par, l21_spin, "l21 workers={workers}");
                    // Symmetric runs skip the `l21h` scatter (dead store),
                    // so only `l21` and the factor agree.
                    assert_eq!(
                        f_par.as_slice(),
                        f_spin.as_slice(),
                        "factor workers={workers}"
                    );
                }
            }
        }
    }

    /// Pack the factored diagonal block's lower triangle + diagonal
    /// inverses exactly as `trsm_l21` does, for driving
    /// `trsm_l21_blocked_serial` directly in tests.
    fn pack_l11_for_test<T: ComplexField>(factor: &DenseMatrix<T>, panel_width: usize) -> Vec<T> {
        let mut l11 = vec![T::zero(); panel_width * panel_width];
        for j in 0..panel_width {
            for p in 0..=j {
                l11[j * panel_width + p] = factor[[j, p]];
            }
        }
        l11
    }

    /// Diagonal inverses exactly as `trsm_l21` computes them.
    fn diag_inv_for_test<T: ComplexField>(factor: &DenseMatrix<T>, panel_width: usize) -> Vec<T> {
        (0..panel_width).map(|j| factor[[j, j]].inv()).collect()
    }

    /// Agreement of the TRSM outputs (packs + stored factor rows) between
    /// the serial loop and the blocked path. The `l21h` pack is only
    /// compared on the Hermitian path: symmetric runs skip its scatter
    /// (dead store), so there is nothing to agree on.
    fn assert_trsm_agree<T: ComplexField<Real = f64>>(
        l21_a: &[T],
        l21_b: &[T],
        l21h_a: &[T],
        l21h_b: &[T],
        f_a: &DenseMatrix<T>,
        f_b: &DenseMatrix<T>,
        n: usize,
        panel_width: usize,
        tolerance: f64,
        check_h: bool,
    ) {
        let mut max_diff = 0.0_f64;
        for (x, y) in l21_a.iter().zip(l21_b.iter()) {
            max_diff = max_diff.max((*x - *y).norm());
        }
        if check_h {
            for (x, y) in l21h_a.iter().zip(l21h_b.iter()) {
                max_diff = max_diff.max((*x - *y).norm());
            }
        }
        for i in 0..n {
            for j in 0..panel_width {
                max_diff = max_diff.max((f_a[[i, j]] - f_b[[i, j]]).norm());
            }
        }
        assert!(
            max_diff <= tolerance,
            "blocked serial TRSM differs from loop by {max_diff:e}"
        );
    }

    /// The interleaved 4-row kernel solves the same rows as four scalar
    /// [`trsm_row`] calls: same gather/substitute/scatter per row, so the
    /// `l21` pack and the factor store must agree (tolerance, since FMA
    /// contraction choices may differ at the bit level). Covers symmetric
    /// and Hermitian conjugation.
    #[test]
    fn trsm_row4_matches_four_rows() {
        fn run<T: ComplexField<Real = f64>>(hermitian: bool, mk: impl Fn(f64, f64) -> T) {
            let w = 8usize;
            let rows = 8usize;
            let n = 16usize;
            let mut l11 = vec![T::zero(); w * w];
            for j in 0..w {
                for p in 0..=j {
                    let v = if j == p {
                        3.0 + j as f64
                    } else {
                        0.25 * ((j * 7 + p * 13) % 5) as f64 - 0.5
                    };
                    l11[j * w + p] = mk(v, 0.0);
                }
            }
            let diag: Vec<T> = (0..w).map(|j| mk(3.0 + j as f64, 0.0).inv()).collect();
            let mut flat = vec![T::zero(); n * n];
            for t in 0..rows {
                for j in 0..w {
                    let v = ((t * 31 + j * 17) % 11) as f64 - 5.0;
                    flat[t * n + j] = mk(0.3 * v, 0.2 * v);
                }
            }
            // Scalar reference: eight `trsm_row` calls.
            let mut f_loop = flat.clone();
            let mut l21_loop = vec![T::zero(); rows * w];
            for t in 0..rows {
                unsafe {
                    trsm_row(
                        f_loop.as_ptr().add(t * n),
                        f_loop.as_mut_ptr().add(t * n),
                        &mut l21_loop[t * w..(t + 1) * w],
                        l11.as_ptr(),
                        diag.as_ptr(),
                        w,
                        hermitian,
                    );
                }
            }
            // Batched: two `trsm_row4` groups.
            let mut f_batch = flat.clone();
            let mut l21_batch = vec![T::zero(); rows * w];
            for g in 0..rows / 4 {
                unsafe {
                    trsm_row4(
                        f_batch.as_ptr().add(g * 4 * n),
                        f_batch.as_mut_ptr().add(g * 4 * n),
                        l21_batch.as_mut_ptr().add(g * 4 * w),
                        n,
                        w,
                        l11.as_ptr(),
                        w,
                        w,
                        diag.as_ptr(),
                        hermitian,
                    );
                }
            }
            let mut max_diff = 0.0_f64;
            for (x, y) in l21_loop.iter().zip(l21_batch.iter()) {
                max_diff = max_diff.max((*x - *y).norm());
            }
            for t in 0..rows {
                for j in 0..w {
                    max_diff = max_diff.max((f_loop[t * n + j] - f_batch[t * n + j]).norm());
                }
            }
            assert!(
                max_diff <= 1e-12,
                "trsm_row4 differs from four trsm_row calls by {max_diff:e} (hermitian={hermitian})"
            );
        }
        run::<f64>(false, |r, _| r);
        run::<Complex64>(false, Complex64::new);
        run::<Complex64>(true, Complex64::new);
    }

    #[test]
    fn blocked_matches_unblocked_spd_across_panels() {
        // n >= 192 activates the blocked path; 256 crosses 8 full panels,
        // 257 adds a partial one.
        for n in [256, 257] {
            let a = spd_matrix(n);
            let blocked = cholesky_factorize(&a).unwrap();
            let unblocked = unblocked_factor(&a, false);
            assert_factors_agree::<f64>(blocked.factor(), &unblocked, 1e-12);

            // Reconstruction A = LLᵀ from the blocked factor.
            let rhs: Vec<f64> = (0..n).map(|i| ((i * 19 + 3) as f64).sin() + 0.25).collect();
            let x = blocked.solve(&rhs).unwrap();
            let actual = a.matvec(&x);
            let max_error = actual
                .iter()
                .zip(&rhs)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_error < 1e-10,
                "blocked solve error {max_error:e} at n={n}"
            );
        }
    }

    #[test]
    fn blocked_matches_unblocked_spd_parallel_schur() {
        // n = 576 crosses the parallel row-band Schur-update threshold
        // (trailing >= 512): exercises the banded in-place update for
        // correctness against the POTF2 reference.
        let n = 576;
        let a = spd_matrix(n);
        let blocked = cholesky_factorize(&a).unwrap();
        let unblocked = unblocked_factor(&a, false);
        assert_factors_agree::<f64>(blocked.factor(), &unblocked, 1e-12);

        let rhs: Vec<f64> = (0..n).map(|i| ((i * 19 + 3) as f64).sin() + 0.25).collect();
        let x = blocked.solve(&rhs).unwrap();
        let actual = a.matvec(&x);
        let max_error = actual
            .iter()
            .zip(&rhs)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_error < 1e-10,
            "parallel-Schur solve error {max_error:e} at n={n}"
        );
    }

    #[test]
    fn blocked_matches_unblocked_hpd_complex_n256() {
        let n = 256;
        let a = hermitian_matrix(n);
        let blocked = cholesky_factorize_hermitian(&a).unwrap();
        let unblocked = unblocked_factor(&a, true);
        assert_factors_agree::<Complex64>(blocked.factor(), &unblocked, 1e-11);

        let rhs: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64 * 0.5).cos()))
            .collect();
        let x = blocked.solve(&rhs).unwrap();
        let actual = a.matvec(&x);
        let max_error = actual
            .iter()
            .zip(&rhs)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_error < 1e-9, "blocked HPD solve error {max_error:e}");
    }

    #[test]
    fn blocked_not_spd_error_matches_unblocked_pivot() {
        // Robustly indefinite at column 200 (inside panel 192..224): both
        // paths must report NotPositiveDefinite at the same pivot.
        let n = 256;
        let mut a = spd_matrix(n);
        a[[200, 200]] = -1.0;
        let tolerance = {
            let scale = a
                .as_slice()
                .iter()
                .map(ComplexField::norm)
                .fold(0.0_f64, f64::max);
            f64::EPSILON * n as f64 * scale
        };
        let blocked = cholesky_factorize(&a);
        let mut unblocked_factor = a.clone();
        let unblocked = cholesky_panel(&mut unblocked_factor, 0, n, false, tolerance);
        match (blocked, unblocked) {
            (
                Err(CholeskyError::NotPositiveDefinite {
                    pivot: blocked_pivot,
                }),
                Err(CholeskyError::NotPositiveDefinite {
                    pivot: unblocked_pivot,
                }),
            ) => {
                assert_eq!(blocked_pivot, 200);
                assert_eq!(unblocked_pivot, 200);
            }
            (blocked, unblocked) => panic!(
                "both paths must report NotPositiveDefinite at pivot 200: \
                 blocked is_err={}, unblocked is_err={unblocked:?}",
                blocked.is_err(),
            ),
        }
    }

    #[test]
    #[ignore = "timing measurement; run explicitly with --release --ignored --nocapture"]
    fn cholesky_blocked_vs_unblocked_timing_n512() {
        let n = 512;
        let a = spd_matrix(n);

        let start = std::time::Instant::now();
        let unblocked = unblocked_factor(&a, false);
        let unblocked_time = start.elapsed();

        let start = std::time::Instant::now();
        let blocked = cholesky_factorize(&a).unwrap();
        let blocked_time = start.elapsed();

        assert_factors_agree::<f64>(blocked.factor(), &unblocked, 1e-11);
        println!(
            "Cholesky n=512: unblocked={unblocked_time:?} blocked={blocked_time:?} \
            speedup={:.2}x",
            unblocked_time.as_secs_f64() / blocked_time.as_secs_f64()
        );
    }

    #[test]
    fn rejects_non_finite_matrix_and_rhs() {
        let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![2.0_f64, f64::NAN, f64::NAN, 2.0]);
        assert!(matches!(
            cholesky_factorize(&matrix),
            Err(CholeskyError::NonFiniteMatrix { index: 1 })
        ));

        let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![2.0_f64]);
        let factor = cholesky_factorize(&matrix).unwrap();
        assert!(matches!(
            factor.solve(&[f64::INFINITY]),
            Err(CholeskyError::NonFiniteRhs { index: 0 })
        ));

        let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(2.0, 0.0)]);
        let factor = cholesky_factorize_hermitian(&matrix).unwrap();
        assert!(matches!(
            factor.solve(&[Complex64::new(1.0, f64::INFINITY)]),
            Err(CholeskyError::NonFiniteRhs { index: 0 })
        ));
    }

    #[test]
    fn finite_rhs_overflow_is_a_typed_numerical_breakdown() {
        let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![f64::MIN_POSITIVE]);
        let factor = cholesky_factorize(&matrix).unwrap();
        assert!(matches!(
            factor.solve(&[f64::MAX]),
            Err(CholeskyError::NumericalBreakdown {
                stage: "forward substitution",
                index: 0,
            })
        ));
    }

    /// Tridiagonal SPD fixture (4 on the diagonal, -1 off): exercises the
    /// blocked path at n = 200.
    fn tridiag_spd(n: usize) -> DenseMatrix<f64> {
        DenseMatrix::from_fn(n, n, |i, j| {
            if i == j {
                4.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            }
        })
    }

    #[test]
    fn unchecked_matches_checked_bitwise() {
        // Same blocked algorithm, same pivot tolerance (diagonal-max scale
        // is exact for SPD input), so the factors must agree exactly.
        let n = 200;
        let matrix = tridiag_spd(n);
        let checked = cholesky_factorize(&matrix).unwrap();
        let unchecked = cholesky_factorize_unchecked(&matrix).unwrap();
        assert_eq!(checked.factor().as_slice(), unchecked.factor().as_slice());
        let unchecked_ip = cholesky_factorize_in_place_unchecked(matrix.clone()).unwrap();
        assert_eq!(
            checked.factor().as_slice(),
            unchecked_ip.factor().as_slice()
        );
        // Hermitian twins agree too.
        let hc = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let h_checked = cholesky_factorize_hermitian(&hc).unwrap();
        let h_unchecked = cholesky_factorize_unchecked_hermitian(&hc).unwrap();
        assert_eq!(
            h_checked.factor().as_slice(),
            h_unchecked.factor().as_slice()
        );
        let h_unchecked_ip = cholesky_factorize_in_place_unchecked_hermitian(hc.clone()).unwrap();
        assert_eq!(
            h_checked.factor().as_slice(),
            h_unchecked_ip.factor().as_slice()
        );
    }

    #[test]
    fn unchecked_ignores_poisoned_strict_upper() {
        // (0, n-1) sits outside every diagonal block (n-1 >= block width),
        // so the lower factorization never reads it: the checked entry
        // rejects the NaN, the unchecked entry factors normally.
        let n = 200;
        let clean = tridiag_spd(n);
        let mut values = clean.as_slice().to_vec();
        values[n - 1] = f64::NAN;
        let poisoned = DenseMatrix::from_row_major_vec(n, n, values);
        assert!(cholesky_factorize(&poisoned).is_err());
        let unchecked = cholesky_factorize_unchecked(&poisoned).unwrap();
        let want = cholesky_factorize(&clean).unwrap();
        for (g, w) in unchecked
            .factor()
            .as_slice()
            .iter()
            .zip(want.factor().as_slice())
        {
            assert!((g - w).abs() < 1e-12, "factor mismatch: {g} vs {w}");
        }
    }

    #[test]
    fn unchecked_non_spd_is_a_typed_breakdown() {
        // Negative pivot: the panel factorization must fail loudly, not
        // silently return garbage.
        let n = 200;
        let mut values = tridiag_spd(n).as_slice().to_vec();
        values[100 * n + 100] = -1.0;
        let bad = DenseMatrix::from_row_major_vec(n, n, values);
        let err = cholesky_factorize_unchecked(&bad).unwrap_err();
        assert!(
            matches!(err, CholeskyError::NotPositiveDefinite { pivot: 100 }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn parallel_entry_paths_match_serial_contract() {
        // Covers the n >= 256 parallel validation + parallel screen paths
        // (fused clone-zero reorder, vectorized scale fold, zero-free
        // screen): exact error values and checked/unchecked agreement with
        // a zeroed upper triangle.
        let n = 300;
        // 1. NonFinite first index in linear order (parallel rescan): the
        // later infinity must not shadow the earlier NaN.
        let mut values = vec![1.0f32; n * n];
        for i in 0..n {
            values[i * n + i] = 2.0;
        }
        values[123 * n + 45] = f32::NAN;
        values[200 * n + 7] = f32::INFINITY;
        let poisoned = DenseMatrix::from_row_major_vec(n, n, values);
        assert!(
            matches!(
                cholesky_factorize(&poisoned),
                Err(CholeskyError::NonFiniteMatrix { index }) if index == 123 * n + 45
            ),
            "first-bad index must be exact"
        );
        // 2. Asymmetry rejects with a structure error (symmetric entry).
        let mut asym = vec![0.0f32; n * n];
        for i in 0..n {
            asym[i * n + i] = 2.0;
        }
        asym[10 * n + 5] = 1.0;
        let asym = DenseMatrix::from_row_major_vec(n, n, asym);
        assert!(matches!(
            cholesky_factorize(&asym),
            Err(CholeskyError::StructureMismatch { .. })
        ));
        // 3. Squared-overflow edge: finite entries whose squares overflow
        // f32 take the hypot-scale fallback and then fail downstream (the
        // all-ones matrix is rank-1, hence not positive definite) — it
        // must error, never silently succeed.
        let big = DenseMatrix::from_row_major_vec(n, n, vec![1.0e20f32; n * n]);
        let err = cholesky_factorize(&big).unwrap_err();
        assert!(
            matches!(
                err,
                CholeskyError::NotPositiveDefinite { .. }
                    | CholeskyError::NumericalBreakdown { .. }
            ),
            "unexpected error: {err:?}"
        );
        // 4. Checked/unchecked/in-place factors agree bitwise on SPD input
        // with a zeroed strict upper (fused clone-zero + parallel screen).
        let mut spd = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = if i == j {
                    n as f32
                } else {
                    ((i * 7 + j * 13) % 11) as f32 * 0.01
                };
                spd[i * n + j] = v;
                spd[j * n + i] = v;
            }
        }
        let matrix = DenseMatrix::from_row_major_vec(n, n, spd);
        let checked = cholesky_factorize(&matrix).unwrap();
        let unchecked = cholesky_factorize_unchecked(&matrix).unwrap();
        assert_eq!(checked.factor().as_slice(), unchecked.factor().as_slice());
        let in_place = cholesky_factorize_in_place(matrix.clone()).unwrap();
        assert_eq!(checked.factor().as_slice(), in_place.factor().as_slice());
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(checked.factor()[[i, j]], 0.0);
            }
        }
    }

    /// Fused parallel validation (`n >= 256`) reports the exact legacy
    /// errors: exact first-bad index for non-finite input, and the same
    /// `StructureMismatch` kinds for pair/diagonal violations (checked
    /// against the serial path at n = 255 and the out-of-place legacy
    /// path, which keep the two-wave implementation).
    #[test]
    fn fused_validation_errors_match_legacy() {
        for n in [255usize, 256, 300] {
            // Hermitian HPD baseline (2n diagonal dominates the O(1)
            // off-diagonals).
            let mut hpd = vec![Complex64::new(0.0, 0.0); n * n];
            for i in 0..n {
                for j in 0..=i {
                    let v = Complex64::new(
                        ((i * 31 + j * 7) % 97) as f64 / 97.0,
                        ((i * 13 + j * 3) % 89) as f64 / 89.0,
                    );
                    hpd[i * n + j] = v;
                    hpd[j * n + i] = v.conj();
                }
                hpd[i * n + i] = Complex64::new(2.0 * n as f64, 0.0);
            }
            let m = DenseMatrix::from_row_major_vec(n, n, hpd.clone());
            assert!(cholesky_factorize_in_place_hermitian(m).is_ok());

            // Non-finite at a known interior index (upper triangle, so
            // the pair check would also see it — NonFinite wins).
            let mut bad = hpd.clone();
            let idx = 7 * n + (n - 9);
            bad[idx] = Complex64::new(f64::NAN, 0.0);
            let m = DenseMatrix::from_row_major_vec(n, n, bad);
            assert!(matches!(
                cholesky_factorize_in_place_hermitian(m),
                Err(CholeskyError::NonFiniteMatrix { index }) if index == idx
            ));

            // Hermitian pair violation (upper/lower disagree past tolerance).
            let mut bad = hpd.clone();
            bad[3 * n + 5] += Complex64::new(1.0, 0.0);
            let m = DenseMatrix::from_row_major_vec(n, n, bad.clone());
            assert!(matches!(
                cholesky_factorize_in_place_hermitian(m),
                Err(CholeskyError::StructureMismatch { kind: "Hermitian" })
            ));
            // Out-of-place legacy path agrees.
            let m = DenseMatrix::from_row_major_vec(n, n, bad);
            assert!(matches!(
                cholesky_factorize_hermitian(&m),
                Err(CholeskyError::StructureMismatch { kind: "Hermitian" })
            ));

            // Hermitian diagonal imaginary part past tolerance.
            let mut bad = hpd.clone();
            bad[11 * n + 11] = Complex64::new(2.0 * n as f64, 1.0);
            let m = DenseMatrix::from_row_major_vec(n, n, bad);
            assert!(matches!(
                cholesky_factorize_in_place_hermitian(m),
                Err(CholeskyError::StructureMismatch { kind: "Hermitian" })
            ));

            // Real symmetric pair violation.
            let spd: Vec<f64> = hpd.iter().map(|z| z.re).collect();
            let mut bad = spd.clone();
            bad[5 * n + 2] += 1.0;
            let m = DenseMatrix::from_row_major_vec(n, n, bad.clone());
            assert!(matches!(
                cholesky_factorize_in_place(m),
                Err(CholeskyError::StructureMismatch { kind: "symmetric" })
            ));
            let m = DenseMatrix::from_row_major_vec(n, n, bad);
            assert!(matches!(
                cholesky_factorize(&m),
                Err(CholeskyError::StructureMismatch { kind: "symmetric" })
            ));
        }
    }
}
