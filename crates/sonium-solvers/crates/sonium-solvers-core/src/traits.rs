//! Core traits for linear algebra operations
//!
//! - [`ComplexField`] — scalar type abstraction (f32, f64, Complex32, Complex64, f16)
//! - [`LinearOperator`] — matrix-like objects that can perform matrix-vector products
//! - [`Preconditioner`] — preconditioning operations
//!
//! All vector operations take plain slices (`&[T]` / `&mut [T]`) or owned
//! `Vec<T>`. BLAS-1 dispatch routes through `sonium_compute::cpu::blas1`,
//! which provides NEON-optimised kernels on AArch64 and auto-vectorised
//! scalar fallbacks elsewhere.

use half::f16;
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, NumAssign, One, ToPrimitive, Zero};
use sonium_compute::backend::Transpose;
use sonium_compute::cpu::blas1 as cblas1;
use sonium_compute::cpu::blas2 as cblas2;
use sonium_compute::cpu::blas3 as cblas3;
use std::fmt::Debug;
use std::ops::Neg;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn catch_solver_panic<T, F>(operation: &'static str, f: F) -> Result<T, SolverError>
where
    F: FnOnce() -> T,
{
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let reason = if let Some(message) = payload.downcast_ref::<&'static str>() {
            (*message).to_string()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "solver operation panicked".to_string()
        };
        SolverError::OperationFailed { operation, reason }
    })
}

fn reject_non_finite_preconditioner_output<T: ComplexField>(
    operation: &'static str,
    values: &[T],
) -> Result<(), SolverError> {
    if values.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "preconditioner generated NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn reject_non_finite_input<T: ComplexField>(
    operation: &'static str,
    values: &[T],
) -> Result<(), SolverError> {
    if values.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "input contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn reject_non_finite_operator_output<T: ComplexField>(
    operation: &'static str,
    values: &[T],
) -> Result<(), SolverError> {
    if values.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "operator generated NaN or infinity".to_string(),
        });
    }
    Ok(())
}

/// Trait for scalar types that can be used in linear algebra operations.
///
/// This trait abstracts over real and complex number types, providing
/// a unified interface for operations like conjugation, norm computation,
/// and conversion from real values.
pub trait ComplexField:
    NumAssign + Clone + Copy + Send + Sync + Debug + Zero + One + Neg<Output = Self> + 'static
{
    /// The real number type underlying this field
    type Real: Float + NumAssign + FromPrimitive + ToPrimitive + Send + Sync + Debug + 'static;

    /// Complex conjugate
    fn conj(&self) -> Self;

    /// Squared magnitude |z|²
    fn norm_sqr(&self) -> Self::Real;

    /// Magnitude |z|, evaluated without overflowing the squared magnitude.
    ///
    /// Finite values near the top of the scalar range must remain finite at
    /// validation boundaries: computing `sqrt(norm_sqr())` first can turn a
    /// finite `f64::MAX` into `sqrt(infinity)`. `hypot` performs the required
    /// scaling for both real and complex implementations.
    fn norm(&self) -> Self::Real {
        self.re().hypot(self.im())
    }

    /// Compute the normwise backward error for an independently recomputed
    /// residual.
    ///
    /// The denominator is the action/RHS scale, with only the smallest
    /// positive representable value as a guard.  A unit floor makes tiny
    /// physical systems appear artificially accurate and is therefore not a
    /// valid relative-error contract.
    #[inline]
    fn normwise_backward_error(
        residual_norm: Self::Real,
        action_norm: Self::Real,
        rhs_norm: Self::Real,
    ) -> Self::Real {
        residual_norm / (action_norm + rhs_norm).max(Self::Real::min_positive_value())
    }

    /// Create from a real value
    fn from_real(r: Self::Real) -> Self;

    /// Create from real and imaginary parts
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self;

    /// Real part
    fn re(&self) -> Self::Real;

    /// Imaginary part
    fn im(&self) -> Self::Real;

    /// Check if this is approximately zero
    fn is_zero_approx(&self, tol: Self::Real) -> bool {
        self.norm() < tol.abs()
    }

    /// Multiplicative inverse (1/z)
    fn inv(&self) -> Self;

    /// Square root
    fn sqrt(&self) -> Self;

    // ------------------------------------------------------------------
    // BLAS-dispatch methods on plain slices.
    //
    // Default implementations are generic loops. The f32/f64/Complex32/
    // Complex64 impls override to route through `sonium_compute::cpu::blas1`
    // for NEON-accelerated paths on AArch64.
    // ------------------------------------------------------------------

    /// Inner product: Σ conj(x_i) * y_i
    fn vec_dot(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = Self::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            sum += xi.conj() * *yi;
        }
        sum
    }

    /// Plain bilinear dot: Σ x_i * y_i (no conjugation).
    ///
    /// This is NOT the Hermitian inner product above: triangular solves
    /// and other bilinear forms need the unconjugated sum, for which
    /// `vec_dot` gives wrong answers on complex types. Default impl is a
    /// generic loop; f32/f64 override with the SIMD backend.
    fn vec_dot_plain(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = Self::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            sum += *xi * *yi;
        }
        sum
    }

    /// Panel dot for unblocked Cholesky (`POTF2`): Σ `x_i · op(y_i)`,
    /// where `op` conjugates `y` when `conj_y` (the Hermitian panel).
    /// Default: four independent accumulator chains, which break the
    /// loop-carried dependency that pins the scalar reduction near
    /// 2 GF/s. Summation order differs from the serial chain, so callers
    /// are tolerance-equal, not bitwise (the contract the SYRK/TRSM
    /// kernels already carry). The f64 impl routes through the AVX2
    /// vector dot instead (f64 conjugation is the identity, so `conj_y`
    /// is exact there too).
    fn potf2_dot(x: &[Self], y: &[Self], conj_y: bool) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut a0 = Self::zero();
        let mut a1 = Self::zero();
        let mut a2 = Self::zero();
        let mut a3 = Self::zero();
        let len = x.len();
        let main = len / 4 * 4;
        if conj_y {
            for t in (0..main).step_by(4) {
                a0 += x[t] * y[t].conj();
                a1 += x[t + 1] * y[t + 1].conj();
                a2 += x[t + 2] * y[t + 2].conj();
                a3 += x[t + 3] * y[t + 3].conj();
            }
            let mut acc = (a0 + a1) + (a2 + a3);
            for t in main..len {
                acc += x[t] * y[t].conj();
            }
            acc
        } else {
            for t in (0..main).step_by(4) {
                a0 += x[t] * y[t];
                a1 += x[t + 1] * y[t + 1];
                a2 += x[t + 2] * y[t + 2];
                a3 += x[t + 3] * y[t + 3];
            }
            let mut acc = (a0 + a1) + (a2 + a3);
            for t in main..len {
                acc += x[t] * y[t];
            }
            acc
        }
    }

    /// Squared vector norm: Σ |x_i|², accumulated with LAPACK
    /// `dlassq`-style scaling for better rounding on long vectors.
    ///
    /// Range contract: when the true sum of squares exceeds
    /// `Real::MAX`, `inf` is the correct result. In particular a finite
    /// `vec_norm` does NOT imply a finite `vec_norm_sqr` (e.g. two
    /// `1e200` entries have norm `~1.4e200` but squared norm `2e400`).
    /// Callers that only need the magnitude (or `sqrt` the result
    /// immediately) must use [`ComplexField::vec_norm`] instead, which
    /// stays finite across the full scalar range.
    fn vec_norm_sqr(x: &[Self]) -> Self::Real {
        scaled_vec_norm_sqr(x)
    }

    /// Vector norm: sqrt(Σ |x_i|²), accumulated without squaring first.
    fn vec_norm(x: &[Self]) -> Self::Real {
        let mut norm = Self::Real::zero();
        for value in x {
            norm = norm.hypot(value.norm());
        }
        norm
    }

    /// Fused `dlassq`-style row accumulation: fold one row of magnitudes
    /// into per-column `(scale, sums)` state, returning row finiteness.
    /// Default is the exact scalar update equations (one division-heavy
    /// pass, kept for complex types and as the portable reference); the
    /// f32/f64 impls route to the vectorized twins, which reproduce this
    /// bitwise by vectorizing across independent columns (each column's
    /// op sequence stays in row order, including NaN/zero masking and
    /// the two-rounding `1 + sums*ratio*ratio` forms).
    fn dlassq_row_update(row: &[Self], scales: &mut [Self::Real], sums: &mut [Self::Real]) -> bool {
        let mut ok = true;
        for (c, &value) in row.iter().enumerate() {
            let magnitude = value.norm();
            if !magnitude.is_finite() {
                ok = false;
            } else if magnitude != Self::Real::zero() {
                if scales[c] < magnitude {
                    let ratio = scales[c] / magnitude;
                    sums[c] = Self::Real::one() + sums[c] * ratio * ratio;
                    scales[c] = magnitude;
                } else {
                    let ratio = magnitude / scales[c];
                    sums[c] += ratio * ratio;
                }
            }
        }
        ok
    }

    /// AXPY: y += α * x
    fn vec_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            *yi += alpha * *xi;
        }
    }

    /// Unit-lower triangular solve `L11 · X = A12` in place (U12 phase of
    /// blocked LU): `l11` is row-major `panel × panel` at stride `lda`
    /// (only the sub-diagonal is read; the unit diagonal is implied),
    /// `block` holds `panel` rows of `ncols` at stride `block_stride` and
    /// is overwritten with the solution. Default: forward substitution as
    /// row AXPYs — identical arithmetic on every type. The f32/f64 impls
    /// route through the compute TRSM band kernel instead: one dispatch
    /// and L1-resident passes in place of `panel² / 2` full-length
    /// streaming AXPYs (same per-element op order; the AVX2 path fuses one
    /// multiply-add per step, ~1 ulp, strictly more accurate).
    fn trsm_u12_unit_lower(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        ncols: usize,
    ) {
        for i in 0..panel {
            for k in 0..i {
                let multiplier = -l11[i * lda + k];
                let (previous, rest) = block.split_at_mut(i * block_stride);
                let current_row = &mut rest[..ncols];
                let previous_row = &previous[k * block_stride..k * block_stride + ncols];
                Self::vec_axpy(multiplier, previous_row, current_row);
            }
        }
    }

    /// Whether the LU driver should prefer the tiny-pool funnel (one
    /// install, all nested scopes inline at 16-wide sizings). Default
    /// false (compact/mirror routing); the f64 impl enables it (measured
    /// 1.16x on dgetrf-2048, same pivots, equal residuals). f32 keeps
    /// the compact narrow pool: its ~2x-faster kernels oversubscribe 16
    /// workers (sgetrf-1024 regresses 2x on tiny). Complex types stay
    /// default-pending measurement.
    fn lu_prefers_tiny_pool() -> bool {
        false
    }

    /// f32-only LU wide-panel gate: selects the 64-wide panel band at
    /// 2048 <= n < 2304 (see `lu_f32_block_size` in sonium-solvers-direct).
    /// Default false (shared 32-wide panel); the f32 impl enables it
    /// (measured ~1.09x on sgetrf-2048 over two interleaved windows,
    /// 9/12 pairs; loses at 1024/2560/4096, hence the two-sided band).
    /// Independent of [`lu_prefers_tiny_pool`] (f32 must NOT take the
    /// tiny pool); complex types stay default pending measurement.
    fn lu_prefers_f32_wide_panel() -> bool {
        false
    }

    /// f64-only LU lookahead-block gate: routes the lookahead next-panel
    /// block update through the parallel strided kernel above
    /// `LOOKAHEAD_PAR_MIN_ROWS` rows (see
    /// `schur_update_lookahead_block` in sonium-solvers-direct).
    /// Default false (serial kernel); the f64 impl enables it (measured
    /// same-process interleaved A/B, RAYON=16: dgetrf-4096 wall -4.1%,
    /// 4/4 pairs; dgetrf-2048 -5.3%, 6/6 pairs; bitwise identical).
    /// f32 must NOT take it (parallel route loses up to +80% on
    /// 1984x64x64); complex types stay default pending measurement.
    fn lu_prefers_parallel_lookahead_update() -> bool {
        false
    }

    /// Cholesky twin of [`lu_prefers_tiny_pool`]: whole-dpotrf tiny-pool
    /// funnel (one install, nested SYRK/TRSM scopes inline). Default
    /// false; the f64 impl enables it (measured 1.51-1.54x on
    /// dpotrf-2048, maxdiff 1.7e-18, equal residuals). f32/complex stay
    /// default pending measurement.
    fn chol_prefers_tiny_pool() -> bool {
        false
    }

    /// QR twin of [`lu_prefers_tiny_pool`]: blocked-QR tiny-pool funnel
    /// (QR has no funnel today, so this is also a caller-temperature
    /// fix). Default false; the f64 impl enables it (measured
    /// 1.27-1.58x at 512/1024 and 1.10-1.13x at 2048, BITWISE-identical
    /// Q/R at all sizes — QR's GEMM splits align across pools).
    /// f32/complex stay default pending measurement.
    fn qr_prefers_tiny_pool() -> bool {
        false
    }

    /// Four simultaneous dots `out[r] = Σ_{i} l[i]·x_r[i]` for the
    /// Cholesky TRSM row kernel (`trsm_row4`): one dot per interleaved
    /// row, sharing the `l11`-row loads. All slices share one length.
    /// The caller subtracts the dots once, so even the default scalar
    /// loop is sum-then-subtract (one rounding at the end) rather than
    /// sequential-subtract — same tolerance class as the inlined loop,
    /// not bitwise. The f32/f64 impls route through the vectorized twin
    /// (FMA per element plus chunked order: same tolerance, not bitwise).
    fn trsm_dot4(l: &[Self], x: [&[Self]; 4]) -> [Self; 4] {
        let mut out = [Self::zero(); 4];
        for i in 0..l.len() {
            out[0] += l[i] * x[0][i];
            out[1] += l[i] * x[1][i];
            out[2] += l[i] * x[2][i];
            out[3] += l[i] * x[3][i];
        }
        out
    }

    /// Right solve `X · op(L11) = B` in place (L21 phase of blocked
    /// Cholesky): `l11` is row-major `panel × panel` at stride `lda` (only
    /// the lower triangle is read), `block` holds `nrows` rows of `panel`
    /// at stride `block_stride` and is overwritten with the solution.
    /// `trans` selects the operator: `Trans` for real/symmetric input,
    /// `ConjTrans` for Hermitian input (conjugation folded into the reads,
    /// which is exact). Default: row-wise forward substitution against the
    /// transposed factor — identical arithmetic on every type. The
    /// f32/f64/Complex32/Complex64 impls route through the compute TRSM
    /// kernel instead (Right/Lower/`trans`; same per-element op order up
    /// to the documented FMA rounding).
    fn trsm_l21_right(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        nrows: usize,
        trans: Transpose,
    ) {
        let conjugate = matches!(trans, Transpose::ConjTrans);
        let mut dinv = Vec::with_capacity(panel);
        for j in 0..panel {
            dinv.push(l11[j * lda + j].inv());
        }
        for i in 0..nrows {
            let row = &mut block[i * block_stride..i * block_stride + panel];
            for j in 0..panel {
                let mut value = row[j];
                for k in 0..j {
                    let a = l11[j * lda + k];
                    value -= if conjugate { a.conj() } else { a } * row[k];
                }
                row[j] = value * dinv[j];
            }
        }
    }

    /// Pivot scan: `(row, value)` of the first maximum-`norm` element of a
    /// strided column (`base[(first+i)*stride]`, `i` in `0..len`), with
    /// strict `>` comparison so the first maximum wins. Single-dispatch
    /// replacement for the LU panel pivot-search loop (millions of scalar
    /// iterations at n = 2048). The f32/f64 impls override to route through
    /// the vectorized `sonium_compute::cpu::blas1` kernel, which reproduces
    /// the scalar index/value bit-for-bit; the default serves the complex
    /// types with the identical loop.
    fn vec_pivot_scan(
        base: &[Self],
        stride: usize,
        first: usize,
        len: usize,
    ) -> (usize, Self::Real) {
        debug_assert!(len >= 1);
        let mut max_row = first;
        let mut max_val = base[first * stride].norm();
        for i in 1..len {
            let val = base[(first + i) * stride].norm();
            if val > max_val {
                max_val = val;
                max_row = first + i;
            }
        }
        (max_row, max_val)
    }

    /// Fused rank-1 update `block[i*bs+j] += alpha*mults[i*ms] * pivot[j]`
    /// over a `rows × cols` row-major block. Single-dispatch replacement
    /// for a row loop of [`Self::vec_axpy`] calls (LU panel factorization,
    /// which otherwise issues one tiny axpy per row per panel column). The
    /// f32/f64 impls override to route through the vectorized
    /// `sonium_compute::cpu::blas1` kernel; the default serves the complex
    /// types with one call (same per-element operation order as the axpy
    /// loop it replaces).
    fn vec_ger_update(
        alpha: Self,
        mults: &[Self],
        mult_stride: usize,
        pivot: &[Self],
        block: &mut [Self],
        block_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        for i in 0..rows {
            let m = alpha * mults[i * mult_stride];
            let row = &mut block[i * block_stride..i * block_stride + cols];
            Self::vec_axpy(m, &pivot[..cols], row);
        }
    }

    /// First index of a non-finite value (NaN or ±inf in any lane), or
    /// `None` when all values are finite. Entry and end-of-factorize
    /// validation for LU/Cholesky (exact first index feeds the error
    /// reports). The f32/f64/Complex32/Complex64 impls route through the
    /// vectorized `sonium_compute::cpu::blas1` scan (exponent-bit test per
    /// block, scalar fixup for the exact index); the default serves the
    /// remaining types with the scalar loop below.
    fn find_non_finite(values: &[Self]) -> Option<usize> {
        values
            .iter()
            .position(|value| !value.re().is_finite() || !value.im().is_finite())
    }

    /// Max entry norm over a slice: `max value.norm()` (`0` when empty).
    /// LU entry scale fold — runs after the entry scan proves finiteness,
    /// so plain `max` is safe. The f32/f64 impls route through the
    /// vectorized abs-max kernel (same value as the scalar fold: `abs` is
    /// exact and a finite max is unique); the default serves the complex
    /// types via `norm` (hypot, robust at the scalar-range edges).
    fn vec_max_norm(values: &[Self]) -> Self::Real {
        values
            .iter()
            .copied()
            .map(|value| value.norm())
            .fold(Self::Real::zero(), Self::Real::max)
    }

    /// Fused entry-validation fold `(max |v|, all_finite)` in one pass
    /// (LU entry scan+scale fusion). Same max value as [`vec_max_norm`](Self::vec_max_norm)
    /// when the flag holds (finite input); the max is unspecified when it
    /// does not — callers must check the flag first and rescan for the
    /// first bad index then (rare). Default: one scalar pass. The f32/f64
    /// impls route through the vectorized twin.
    fn vec_max_norm_checked(values: &[Self]) -> (Self::Real, bool) {
        let mut best = Self::Real::zero();
        let mut ok = true;
        for &value in values {
            let m = value.norm();
            if m > best {
                best = m;
            }
            ok &= m.is_finite();
        }
        (best, ok)
    }

    /// In-place scale: x *= α
    fn vec_scale(x: &mut [Self], alpha: Self) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    /// Row-major dense matrix-matrix multiply, `C = A * B` (no transpose).
    /// `A` is `m × k` (row-major, leading dim `k`), `B` is `k × n` (leading
    /// dim `n`), `C` is `m × n` (leading dim `n`) and is overwritten.
    ///
    /// Default impl is a scalar triple loop. The f32/f64/Complex32/Complex64
    /// impls override to route through `sonium_compute::cpu::blas3`, which
    /// dispatches to the SME2 micro-kernels on aarch64 with `FEAT_SME2`.
    fn mat_mul_nn(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);
        for ci in c.iter_mut() {
            *ci = Self::zero();
        }
        for i in 0..m {
            for kk in 0..k {
                let aik = a[i * k + kk];
                let b_row = &b[kk * n..(kk + 1) * n];
                let c_row = &mut c[i * n..(i + 1) * n];
                for (j, bv) in b_row.iter().enumerate() {
                    c_row[j] += aik * *bv;
                }
            }
        }
    }

    /// Row-major fused subtract-multiply, `C -= A * B` (no transpose).
    /// `A`/`B` share [`mat_mul_nn`]'s tight layout; `C` is the `m × n`
    /// sub-matrix at the start of `c` with leading dimension `ldc`
    /// (`c.len() >= (m - 1) * ldc + n`), updated in place. This is the
    /// Schur-complement update: fusing it into the GEMM epilogue removes
    /// the scratch round-trip (`trailing²` write + reread per panel).
    ///
    /// Default impl buffers through [`mat_mul_nn`] (bitwise identical to
    /// a fused epilogue: same summation order, minus the exact
    /// store/load). The f32/f64/Complex32/Complex64 impls override with
    /// `alpha = -1, beta = 1` through `sonium_compute::cpu::blas3`.
    fn mat_mul_nn_sub(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        b: &[Self],
        c: &mut [Self],
        ldc: usize,
    ) {
        let mut tmp = vec![Self::zero(); m * n];
        Self::mat_mul_nn(m, n, k, a, b, &mut tmp);
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] -= tmp[i * n + j];
            }
        }
    }

    /// Symmetric/Hermitian rank-`k` update of the lower triangle,
    /// `C := alpha * op(A) * op(A)ᴴ + beta * C`.
    /// `NoTrans`: `A` is `n × k` row-major (`lda >= k`),
    /// `C[i][j] += alpha * Σ_l A[i][l] * A[j][l]` for `i >= j` (no
    /// conjugation — symmetric update). `ConjTrans`: `A` is `k × n`
    /// row-major (`lda >= n`),
    /// `C[i][j] += alpha * Σ_l conj(A[l][i]) * A[l][j]` for `i >= j`
    /// (Hermitian update). `C` is the `n × n` sub-matrix at the start of
    /// `c` with leading dimension `ldc`, updated in place; the strict
    /// upper triangle is never read or written. This is the Cholesky
    /// trailing update: half the flops and half the writes of the
    /// full-square [`mat_mul_nn_sub`].
    ///
    /// Default impl is serial lower-triangle loops. The
    /// f32/f64/Complex32/Complex64 impls override through
    /// `sonium_compute::cpu::blas3::syrk_*` (lower, same `trans`).
    fn syrk_lower(
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        match trans {
            Transpose::NoTrans => {
                for i in 0..n {
                    for j in 0..=i {
                        let mut acc = beta * c[i * ldc + j];
                        for l in 0..k {
                            acc += alpha * a[i * lda + l] * a[j * lda + l];
                        }
                        c[i * ldc + j] = acc;
                    }
                }
            }
            _ => {
                for i in 0..n {
                    for j in 0..=i {
                        let mut acc = beta * c[i * ldc + j];
                        for l in 0..k {
                            acc += alpha * a[l * lda + i].conj() * a[l * lda + j];
                        }
                        c[i * ldc + j] = acc;
                    }
                }
            }
        }
    }

    /// Hermitian rank-`k` update from a NoTrans operand (BLAS `*herk`
    /// with `TRANS = 'N'`): `C := alpha * A * Aᴴ + beta * C`, `A` is
    /// `n × k` row-major (`lda >= k`),
    /// `C[i][j] += alpha * Σ_l A[i][l] * conj(A[j][l])` for `i >= j`.
    /// Same `C` contract as [`syrk_lower`]. For real types this is
    /// identical to the NoTrans symmetric update.
    ///
    /// Default impl is serial lower-triangle loops. The f32/f64 impls
    /// forward to [`syrk_lower`] with `NoTrans` (conjugation is a
    /// no-op); Complex32/Complex64 override through
    /// `sonium_compute::cpu::blas3::herk_*`.
    fn herk_lower(
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        for i in 0..n {
            for j in 0..=i {
                let mut acc = beta * c[i * ldc + j];
                for l in 0..k {
                    acc += alpha * a[i * lda + l] * a[j * lda + l].conj();
                }
                c[i * ldc + j] = acc;
            }
        }
    }

    /// Row-major transposed matrix-matrix multiply, `C = Aᵀ * B`.
    /// `A` is `k × m` (row-major, leading dim `m`), `B` is `k × n` (leading
    /// dim `n`), `C` is `m × n` (leading dim `n`) and is overwritten.
    ///
    /// Default impl explicitly transposes `A` into scratch, then delegates
    /// to [`Self::mat_mul_nn`]. The f32/f64 impls override to route through
    /// `sonium_compute::cpu::blas3` with a transposed left operand, which
    /// keeps the operation inside the tuned kernel (AVX2+FMA on x86_64)
    /// with no `k × m` transpose allocation on the caller side.
    fn mat_mul_tn(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);
        let mut at = vec![Self::zero(); m * k];
        for i in 0..k {
            for j in 0..m {
                at[j * k + i] = a[i * m + j];
            }
        }
        Self::mat_mul_nn(m, n, k, &at, b, c);
    }

    /// Rank-k update `C -= A * B` (no transpose) with a strided `C`.
    /// `A` is `m × k` (row-major, leading dim `k`), `B` is `k × n` (leading
    /// dim `n`), `C` is `m × n` with row stride `ldc >= n`: only columns
    /// `..n` of each of the `m` rows are read and written.
    ///
    /// This is the Schur-complement update behind blocked LU/Cholesky.
    /// Routing it through GEMM with `alpha = -1, beta = 1` keeps the update
    /// in place instead of materializing a separate `m × n` buffer and
    /// streaming it back over the target (twice the traffic).
    ///
    /// Default impl allocates one `m × n` scratch through
    /// [`Self::mat_mul_nn`]. The f32/f64/Complex32/Complex64 impls override
    /// with a single `sonium_compute::cpu::blas3` call.
    fn schur_sub(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self], ldc: usize) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert!(ldc >= n);
        debug_assert!(c.len() >= m.saturating_mul(ldc));
        let mut tmp = vec![Self::zero(); m * n];
        Self::mat_mul_nn(m, n, k, a, b, &mut tmp);
        for i in 0..m {
            let row = &mut c[i * ldc..i * ldc + n];
            let trow = &tmp[i * n..(i + 1) * n];
            for (dst, src) in row.iter_mut().zip(trow.iter()) {
                *dst -= *src;
            }
        }
    }

    /// Strided Schur-complement update, `C -= A * B` with row strides
    /// `lda >= k`, `ldb >= n`, `ldc >= n`.
    ///
    /// Same arithmetic as [`Self::schur_sub`](Self::schur_sub) but A and B
    /// may sit inside a larger row-major factor (e.g. L21/U12 in place),
    /// skipping the explicit pack-to-contiguous copies. The f32/f64/
    /// Complex32/Complex64 impls route to `sonium_compute::cpu::blas3`
    /// with the given strides; the default is a naive triple loop.
    fn schur_sub_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        debug_assert!(lda >= k);
        debug_assert!(ldb >= n);
        debug_assert!(ldc >= n);
        debug_assert!(a.len() >= m.saturating_mul(lda));
        debug_assert!(b.len() >= k.saturating_mul(ldb));
        debug_assert!(c.len() >= m.saturating_mul(ldc));
        for i in 0..m {
            for j in 0..n {
                let mut acc = Self::zero();
                for p in 0..k {
                    acc += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] -= acc;
            }
        }
    }

    /// Serial [`Self::schur_sub_strided`] (no thread-pool use anywhere in
    /// the call tree): for overlap callers that already hold the pool
    /// (LU lookahead runs small next-panel block-column updates on the
    /// caller through this kernel; large ones fan out over the idle
    /// pool before the overlap scope). The f32/f64 impls
    /// route to the serial kernels, bitwise identical to the parallel
    /// path per element (same packs, same sweep order); the default is
    /// the naive nest above (correct; last-ulp vs parallel possible).
    fn schur_sub_serial(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        debug_assert!(lda >= k);
        debug_assert!(ldb >= n);
        debug_assert!(ldc >= n);
        // Footprint form (not full rectangles): strided callers pass the
        // packed footprint `(rows - 1) * ld + cols`, like the parallel
        // kernels accept.
        debug_assert!(a.len() >= (m - 1) * lda + k);
        debug_assert!(b.len() >= (k - 1) * ldb + n);
        debug_assert!(c.len() >= (m - 1) * ldc + n);
        for i in 0..m {
            for j in 0..n {
                let mut acc = Self::zero();
                for p in 0..k {
                    acc += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] -= acc;
            }
        }
    }

    /// Strided Hermitian-transpose product, `C = Aᴴ * B` (`beta = 0`,
    /// overwrite) with row strides `lda >= k`, `ldb >= n`, `ldc >= n`.
    /// `A` is `m × k`, `B` is `k × n`, `C` is `m × n`. The Hermitian
    /// (conjugated for complex types) matches [`Self::vec_dot`].
    ///
    /// This is the first half of a compact-WY Householder trailing update
    /// (`W = Vᴴ·A`); the f32/f64/Complex32/Complex64 impls route to
    /// `sonium_compute::cpu::blas3` with the given strides (ConjTrans for
    /// complex); the default is a naive triple loop.
    fn gemm_hermitian_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        debug_assert!(lda >= k);
        debug_assert!(ldb >= n);
        debug_assert!(ldc >= n);
        debug_assert!(a.len() >= m.saturating_mul(lda));
        debug_assert!(b.len() >= k.saturating_mul(ldb));
        debug_assert!(c.len() >= m.saturating_mul(ldc));
        for i in 0..m {
            for j in 0..n {
                let mut acc = Self::zero();
                for p in 0..k {
                    acc += a[p * lda + i].conj() * b[p * ldb + j];
                }
                c[i * ldc + j] = acc;
            }
        }
    }

    /// Strided plain product, `C = A * B` (`beta = 0`, overwrite) with row
    /// strides `lda >= k`, `ldb >= n`, `ldc >= n`. `A` is `m × k`, `B` is
    /// `k × n`, `C` is `m × n`.
    ///
    /// This is the second half of a compact-WY Householder trailing update
    /// (`A -= V·(T·W)` folds through here); the f32/f64/Complex32/
    /// Complex64 impls route to `sonium_compute::cpu::blas3` with the
    /// given strides; the default is a naive triple loop.
    fn gemm_plain_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        debug_assert!(lda >= k);
        debug_assert!(ldb >= n);
        debug_assert!(ldc >= n);
        debug_assert!(a.len() >= m.saturating_mul(lda));
        debug_assert!(b.len() >= k.saturating_mul(ldb));
        debug_assert!(c.len() >= m.saturating_mul(ldc));
        for i in 0..m {
            for j in 0..n {
                let mut acc = Self::zero();
                for p in 0..k {
                    acc += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] = acc;
            }
        }
    }

    /// Row-major dense matrix-vector product, `y = op(A) * x`. `a` is the
    /// `rows × cols` row-major storage (leading dim `cols`) in every case:
    /// - `NoTrans`: `y` (len `rows`) = A·x, `x` len `cols`
    /// - `Trans`: `y` (len `cols`) = Aᵀ·x, `x` len `rows`
    /// - `ConjTrans`: `y` (len `cols`) = Aᴴ·x, `x` len `rows`
    ///
    /// `y` is overwritten. Default impl is a scalar loop. The f32/f64 impls
    /// override to route large panels through `sonium_compute::cpu::blas2`,
    /// which dispatches to the SME2 kernels on aarch64; small panels stay
    /// scalar because the kernel's fixed streaming-mode setup cost is not
    /// amortized there. Complex32/Complex64 override to route panels of at
    /// least 64×64 elements through `blas2::complex_native`, a NEON
    /// (aarch64) / AVX2+FMA (x86_64) kernel that works directly on the
    /// interleaved storage with no per-call repacking; it beats the scalar
    /// loop at every measured size (see `dense_gemv_bench`). The older
    /// `blas2::complex4m` path is not used here: it re-deinterleaves the
    /// whole matrix into split re/im scratch on every call and measures
    /// slower than the scalar loop.
    fn mat_vec_op(op: Transpose, rows: usize, cols: usize, a: &[Self], x: &[Self], y: &mut [Self]) {
        mat_vec_op_scalar(op, rows, cols, a, x, y);
    }
}

/// Overflow-safe squared norm shared by [`ComplexField::vec_norm_sqr`].
///
/// LAPACK `dlassq`-style scaled accumulation: tracks the running maximum
/// magnitude `scale` and the scaled sum `sumsq`, so intermediate squares
/// never overflow. NaN magnitudes propagate to NaN; empty input yields zero.
/// (When the true sum itself exceeds `Real::MAX`, `inf` is the correct
/// result — see the `vec_norm_sqr` range contract.)
fn scaled_vec_norm_sqr<T: ComplexField>(x: &[T]) -> T::Real {
    let mut scale = T::Real::zero();
    let mut sumsq = T::Real::one();
    for xi in x.iter() {
        let ax = xi.norm();
        if ax.is_nan() {
            return ax;
        }
        if ax > scale {
            let r = scale / ax;
            sumsq = T::Real::one() + sumsq * r * r;
            scale = ax;
        } else if ax > T::Real::zero() {
            let r = ax / scale;
            sumsq += r * r;
        }
    }
    scale * scale * sumsq
}

/// Scalar fallback for [`ComplexField::mat_vec_op`]; also called by the
/// f32/f64 blas2 overrides for panels below the dispatch threshold.
fn mat_vec_op_scalar<T: ComplexField>(
    op: Transpose,
    rows: usize,
    cols: usize,
    a: &[T],
    x: &[T],
    y: &mut [T],
) {
    debug_assert_eq!(a.len(), rows * cols);
    match op {
        Transpose::NoTrans => {
            debug_assert_eq!(x.len(), cols);
            debug_assert_eq!(y.len(), rows);
            for (i, yi) in y.iter_mut().enumerate() {
                let row = &a[i * cols..(i + 1) * cols];
                let mut acc = T::zero();
                for (av, bv) in row.iter().zip(x.iter()) {
                    acc += *av * *bv;
                }
                *yi = acc;
            }
        }
        Transpose::Trans | Transpose::ConjTrans => {
            debug_assert_eq!(x.len(), rows);
            debug_assert_eq!(y.len(), cols);
            let conj = matches!(op, Transpose::ConjTrans);
            for v in y.iter_mut() {
                *v = T::zero();
            }
            for i in 0..rows {
                let xi = x[i];
                let row = &a[i * cols..(i + 1) * cols];
                for (j, av) in row.iter().enumerate() {
                    let av = if conj { av.conj() } else { *av };
                    y[j] += av * xi;
                }
            }
        }
    }
}

/// Minimum panel area (rows × cols) for which the SME2-backed blas2 GEMV
/// beats the scalar loop. Measured on Apple M4 Pro: scalar wins at 64²
/// (1.5×), blas2 wins at 256² (2×) and beyond (see `dense_gemv_bench`).
/// Env-overridable per port (`SONIUM_BLAS2_MIN_ELEMS`); x86 needs its own
/// `dense_gemv_bench` run before changing the default.
const MAT_VEC_BLAS2_MIN_ELEMS: usize = 128 * 128;

/// Minimum panel area for which the native interleaved complex GEMV
/// (`blas2::complex_native`, NEON on aarch64 / AVX2+FMA on x86_64) beats
/// the scalar loop. Measured on Apple M4 Pro: the kernel wins at every
/// measured size from 64² up — Complex64 1.7× (NoTrans) / 2.1× (ConjTrans)
/// at 64², still 1.2-1.4× at 4096²; Complex32 2-3× across the range
/// (see `dense_gemv_bench`). Env-overridable per port
/// (`SONIUM_BLAS2_MIN_ELEMS_CPLX`).
const MAT_VEC_BLAS2_MIN_ELEMS_CPLX: usize = 64 * 64;

fn blas2_min_elems() -> usize {
    std::env::var("SONIUM_BLAS2_MIN_ELEMS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(MAT_VEC_BLAS2_MIN_ELEMS)
}

fn blas2_min_elems_cplx() -> usize {
    std::env::var("SONIUM_BLAS2_MIN_ELEMS_CPLX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(MAT_VEC_BLAS2_MIN_ELEMS_CPLX)
}

impl ComplexField for Complex64 {
    type Real = f64;

    #[inline]
    fn conj(&self) -> Self {
        Complex64::conj(self)
    }

    #[inline]
    fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn from_real(r: f64) -> Self {
        Complex64::new(r, 0.0)
    }

    #[inline]
    fn from_re_im(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }

    #[inline]
    fn re(&self) -> f64 {
        self.re
    }

    #[inline]
    fn im(&self) -> f64 {
        self.im
    }

    #[inline]
    fn inv(&self) -> Self {
        // Scale before forming `re² + im²`: direct norm-square arithmetic
        // underflows for finite tiny pivots (and overflows for large values),
        // even when the reciprocal itself is representable.
        let scale = self.re.abs().max(self.im.abs());
        if scale == 0.0 || !scale.is_finite() {
            let denom = self.norm_sqr();
            return Complex64::new(self.re / denom, -self.im / denom);
        }
        let scaled_re = self.re / scale;
        let scaled_im = self.im / scale;
        let inverse_scale = 1.0 / scale;
        let denominator = scaled_re * scaled_re + scaled_im * scaled_im;
        Complex64::new(
            scaled_re * inverse_scale / denominator,
            -scaled_im * inverse_scale / denominator,
        )
    }

    #[inline]
    fn sqrt(&self) -> Self {
        Complex64::sqrt(*self)
    }

    fn vec_dot(x: &[Self], y: &[Self]) -> Self {
        cblas1::dotc_c64(x, y)
    }

    fn vec_norm_sqr(x: &[Self]) -> f64 {
        let n = cblas1::nrm2_c64(x);
        // `nrm2` is exact while squaring stays in range; delegate to the
        // scaled accumulation when the square would overflow (or `n` is
        // non-finite), instead of returning a spurious `inf`.
        if n > 0.0 && n <= f64::MAX.sqrt() {
            n * n
        } else {
            scaled_vec_norm_sqr(x)
        }
    }

    fn vec_norm(x: &[Self]) -> f64 {
        let norm = cblas1::nrm2_c64(x);
        if norm.is_finite() {
            norm
        } else {
            x.iter()
                .fold(0.0_f64, |norm, value| norm.hypot(value.norm()))
        }
    }

    fn vec_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        cblas1::axpy_c64(alpha, x, y);
    }

    fn vec_pivot_scan(
        base: &[Self],
        stride: usize,
        first: usize,
        len: usize,
    ) -> (usize, Self::Real) {
        cblas1::pivot_scan_c64(base, stride, first, len)
    }

    #[inline]
    fn vec_max_norm_checked(values: &[Self]) -> (Self::Real, bool) {
        cblas1::max_norm_checked_c64(values)
    }

    #[inline]
    fn vec_ger_update(
        alpha: Self,
        mults: &[Self],
        mult_stride: usize,
        pivot: &[Self],
        block: &mut [Self],
        block_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        cblas1::ger_update_c64(
            alpha,
            mults,
            mult_stride,
            pivot,
            block,
            block_stride,
            rows,
            cols,
        );
    }

    #[inline]
    fn trsm_l21_right(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        nrows: usize,
        trans: Transpose,
    ) {
        if panel == 0 || nrows == 0 {
            return;
        }
        cblas3::trsm::trsm_c64(
            cblas3::trsm::TrsmSide::Right,
            cblas3::trsm::TrsmUplo::Lower,
            trans,
            false,
            nrows,
            panel,
            Complex64::new(1.0, 0.0),
            l11,
            lda,
            block,
            block_stride,
        );
    }

    #[inline]
    fn trsm_u12_unit_lower(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        ncols: usize,
    ) {
        // LU twin of `trsm_l21_right` (shipped 2026-10-02): the default
        // naive i/k nest made zgetrf-1024 3.6x (complex TRSM ~22ms
        // serial); routing to the blocked `trsm_c64` like f64/f32.
        if panel == 0 || ncols == 0 {
            return;
        }
        cblas3::trsm::trsm_c64(
            cblas3::trsm::TrsmSide::Left,
            cblas3::trsm::TrsmUplo::Lower,
            Transpose::NoTrans,
            true,
            panel,
            ncols,
            Complex64::new(1.0, 0.0),
            l11,
            lda,
            block,
            block_stride,
        );
    }

    fn vec_scale(x: &mut [Self], alpha: Self) {
        cblas1::scal_c64(alpha, x);
    }

    fn mat_mul_nn(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        cblas3::gemm_c64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex64::new(1.0, 0.0),
            a,
            k,
            b,
            n,
            Complex64::new(0.0, 0.0),
            c,
            n,
        );
    }

    fn mat_mul_nn_sub(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        b: &[Self],
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::gemm_c64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex64::new(-1.0, 0.0),
            a,
            k,
            b,
            n,
            Complex64::new(1.0, 0.0),
            c,
            ldc,
        );
    }

    #[inline]
    fn syrk_lower(
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::syrk::syrk_c64(
            cblas3::trsm::TrsmUplo::Lower,
            trans,
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
        );
    }

    fn herk_lower(
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::syrk::herk_c64(
            cblas3::trsm::TrsmUplo::Lower,
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
        );
    }

    fn schur_sub(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self], ldc: usize) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex64::new(-1.0, 0.0),
            a,
            k,
            b,
            n,
            Complex64::new(1.0, 0.0),
            c,
            ldc,
        );
    }

    fn schur_sub_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex64::new(-1.0, 0.0),
            a,
            lda,
            b,
            ldb,
            Complex64::new(1.0, 0.0),
            c,
            ldc,
        );
    }

    fn schur_sub_serial(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        // Serial native sweep (shipped 2026-10-02): the default naive
        // nest made complex LU lookahead ~2x slower than no-lookahead
        // (zgetrf-1024 46 vs 27ms); mirrors the f64 override.
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c64_serial(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex64::new(-1.0, 0.0),
            a,
            lda,
            b,
            ldb,
            Complex64::new(1.0, 0.0),
            c,
            ldc,
        );
    }

    #[inline]
    fn gemm_hermitian_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c64(
            Transpose::ConjTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex64::new(1.0, 0.0),
            a,
            lda,
            b,
            ldb,
            Complex64::new(0.0, 0.0),
            c,
            ldc,
        );
    }

    #[inline]
    fn gemm_plain_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex64::new(1.0, 0.0),
            a,
            lda,
            b,
            ldb,
            Complex64::new(0.0, 0.0),
            c,
            ldc,
        );
    }

    #[inline]
    fn find_non_finite(values: &[Self]) -> Option<usize> {
        // `Complex64` is `#[repr(C)]`, so the slice reinterprets as 2×f64
        // lanes (`re`, `im` per value, in order); the lane index halves to
        // the value index. Same precedent as the neon complex kernels.
        let lanes =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const f64, values.len() * 2) };
        cblas1::find_non_finite_f64(lanes).map(|lane| lane / 2)
    }

    fn mat_vec_op(op: Transpose, rows: usize, cols: usize, a: &[Self], x: &[Self], y: &mut [Self]) {
        if rows * cols < blas2_min_elems_cplx() {
            mat_vec_op_scalar(op, rows, cols, a, x, y);
            return;
        }
        // blas2's Trans/ConjTrans contract stores A as n × m, so the
        // output length `m` is `cols` and the reduce length `n` is `rows`;
        // the row-major storage with lda = cols satisfies lda >= m.
        let (m, n) = match op {
            Transpose::NoTrans => (rows, cols),
            Transpose::Trans | Transpose::ConjTrans => (cols, rows),
        };
        cblas2::complex_native::gemv_c64(
            op,
            m,
            n,
            Complex64::new(1.0, 0.0),
            a,
            cols,
            x,
            Complex64::new(0.0, 0.0),
            y,
        );
    }
}

impl ComplexField for Complex32 {
    type Real = f32;

    #[inline]
    fn conj(&self) -> Self {
        Complex32::conj(self)
    }

    #[inline]
    fn norm_sqr(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn from_real(r: f32) -> Self {
        Complex32::new(r, 0.0)
    }

    #[inline]
    fn from_re_im(re: f32, im: f32) -> Self {
        Complex32::new(re, im)
    }

    #[inline]
    fn re(&self) -> f32 {
        self.re
    }

    #[inline]
    fn im(&self) -> f32 {
        self.im
    }

    #[inline]
    fn inv(&self) -> Self {
        // Scale before forming `re² + im²`: direct norm-square arithmetic
        // underflows for finite tiny pivots (and overflows for large values),
        // even when the reciprocal itself is representable.
        let scale = self.re.abs().max(self.im.abs());
        if scale == 0.0 || !scale.is_finite() {
            let denom = self.norm_sqr();
            return Complex32::new(self.re / denom, -self.im / denom);
        }
        let scaled_re = self.re / scale;
        let scaled_im = self.im / scale;
        let inverse_scale = 1.0 / scale;
        let denominator = scaled_re * scaled_re + scaled_im * scaled_im;
        Complex32::new(
            scaled_re * inverse_scale / denominator,
            -scaled_im * inverse_scale / denominator,
        )
    }

    #[inline]
    fn sqrt(&self) -> Self {
        Complex32::sqrt(*self)
    }

    fn vec_dot(x: &[Self], y: &[Self]) -> Self {
        cblas1::dotc_c32(x, y)
    }

    fn vec_norm_sqr(x: &[Self]) -> f32 {
        let n = cblas1::nrm2_c32(x);
        // `nrm2` is exact while squaring stays in range; delegate to the
        // scaled accumulation when the square would overflow (or `n` is
        // non-finite), instead of returning a spurious `inf`.
        if n > 0.0 && n <= f32::MAX.sqrt() {
            n * n
        } else {
            scaled_vec_norm_sqr(x)
        }
    }

    fn vec_norm(x: &[Self]) -> f32 {
        let norm = cblas1::nrm2_c32(x);
        if norm.is_finite() {
            norm
        } else {
            x.iter()
                .fold(0.0_f32, |norm, value| norm.hypot(value.norm()))
        }
    }

    fn vec_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        cblas1::axpy_c32(alpha, x, y);
    }

    fn vec_pivot_scan(
        base: &[Self],
        stride: usize,
        first: usize,
        len: usize,
    ) -> (usize, Self::Real) {
        cblas1::pivot_scan_c32(base, stride, first, len)
    }

    #[inline]
    fn vec_max_norm_checked(values: &[Self]) -> (Self::Real, bool) {
        cblas1::max_norm_checked_c32(values)
    }

    #[inline]
    fn vec_ger_update(
        alpha: Self,
        mults: &[Self],
        mult_stride: usize,
        pivot: &[Self],
        block: &mut [Self],
        block_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        cblas1::ger_update_c32(
            alpha,
            mults,
            mult_stride,
            pivot,
            block,
            block_stride,
            rows,
            cols,
        );
    }

    #[inline]
    fn trsm_l21_right(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        nrows: usize,
        trans: Transpose,
    ) {
        if panel == 0 || nrows == 0 {
            return;
        }
        cblas3::trsm::trsm_c32(
            cblas3::trsm::TrsmSide::Right,
            cblas3::trsm::TrsmUplo::Lower,
            trans,
            false,
            nrows,
            panel,
            Complex32::new(1.0, 0.0),
            l11,
            lda,
            block,
            block_stride,
        );
    }

    #[inline]
    fn trsm_u12_unit_lower(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        ncols: usize,
    ) {
        // LU twin of `trsm_l21_right` (shipped 2026-10-02, cgetrf-1024
        // 5.8x -> blocked `trsm_c32`; see the c64 note).
        if panel == 0 || ncols == 0 {
            return;
        }
        cblas3::trsm::trsm_c32(
            cblas3::trsm::TrsmSide::Left,
            cblas3::trsm::TrsmUplo::Lower,
            Transpose::NoTrans,
            true,
            panel,
            ncols,
            Complex32::new(1.0, 0.0),
            l11,
            lda,
            block,
            block_stride,
        );
    }

    fn vec_scale(x: &mut [Self], alpha: Self) {
        cblas1::scal_c32(alpha, x);
    }

    fn mat_mul_nn(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        cblas3::gemm_c32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex32::new(1.0, 0.0),
            a,
            k,
            b,
            n,
            Complex32::new(0.0, 0.0),
            c,
            n,
        );
    }

    fn mat_mul_nn_sub(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        b: &[Self],
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::gemm_c32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex32::new(-1.0, 0.0),
            a,
            k,
            b,
            n,
            Complex32::new(1.0, 0.0),
            c,
            ldc,
        );
    }

    #[inline]
    fn syrk_lower(
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::syrk::syrk_c32(
            cblas3::trsm::TrsmUplo::Lower,
            trans,
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
        );
    }

    fn herk_lower(
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::syrk::herk_c32(
            cblas3::trsm::TrsmUplo::Lower,
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
        );
    }

    fn schur_sub(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self], ldc: usize) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex32::new(-1.0, 0.0),
            a,
            k,
            b,
            n,
            Complex32::new(1.0, 0.0),
            c,
            ldc,
        );
    }

    fn schur_sub_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex32::new(-1.0, 0.0),
            a,
            lda,
            b,
            ldb,
            Complex32::new(1.0, 0.0),
            c,
            ldc,
        );
    }

    fn schur_sub_serial(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        // Serial native sweep (shipped 2026-10-02, c32 twin of the c64
        // override — cgetrf lookahead was ~2.8x slower than no-lookahead).
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c32_serial(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex32::new(-1.0, 0.0),
            a,
            lda,
            b,
            ldb,
            Complex32::new(1.0, 0.0),
            c,
            ldc,
        );
    }

    #[inline]
    fn gemm_hermitian_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c32(
            Transpose::ConjTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex32::new(1.0, 0.0),
            a,
            lda,
            b,
            ldb,
            Complex32::new(0.0, 0.0),
            c,
            ldc,
        );
    }

    #[inline]
    fn gemm_plain_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_c32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            Complex32::new(1.0, 0.0),
            a,
            lda,
            b,
            ldb,
            Complex32::new(0.0, 0.0),
            c,
            ldc,
        );
    }

    #[inline]
    fn find_non_finite(values: &[Self]) -> Option<usize> {
        // `Complex32` is `#[repr(C)]`, so the slice reinterprets as 2×f32
        // lanes (`re`, `im` per value, in order); the lane index halves to
        // the value index. Same precedent as the neon complex kernels.
        let lanes =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const f32, values.len() * 2) };
        cblas1::find_non_finite_f32(lanes).map(|lane| lane / 2)
    }

    fn mat_vec_op(op: Transpose, rows: usize, cols: usize, a: &[Self], x: &[Self], y: &mut [Self]) {
        if rows * cols < blas2_min_elems_cplx() {
            mat_vec_op_scalar(op, rows, cols, a, x, y);
            return;
        }
        let (m, n) = match op {
            Transpose::NoTrans => (rows, cols),
            Transpose::Trans | Transpose::ConjTrans => (cols, rows),
        };
        cblas2::complex_native::gemv_c32(
            op,
            m,
            n,
            Complex32::new(1.0, 0.0),
            a,
            cols,
            x,
            Complex32::new(0.0, 0.0),
            y,
        );
    }
}

impl ComplexField for f64 {
    type Real = f64;

    #[inline]
    fn conj(&self) -> Self {
        *self
    }

    #[inline]
    fn norm_sqr(&self) -> f64 {
        *self * *self
    }

    #[inline]
    fn norm(&self) -> f64 {
        // Real types have no imaginary component: `|re|` is the exact
        // norm (`hypot(re, 0) == |re|`, and unlike a squared magnitude it
        // can never overflow), while skipping `hypot`'s scaling path —
        // the dominant cost of O(n²) validation loops (millions of calls
        // per factorization). Same precedent as the f16 override below.
        self.abs()
    }

    #[inline]
    fn from_real(r: f64) -> Self {
        r
    }

    #[inline]
    fn from_re_im(re: f64, _im: f64) -> Self {
        re
    }

    #[inline]
    fn re(&self) -> f64 {
        *self
    }

    #[inline]
    fn im(&self) -> f64 {
        0.0
    }

    #[inline]
    fn inv(&self) -> Self {
        1.0 / *self
    }

    #[inline]
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }

    #[inline]
    fn vec_dot(x: &[Self], y: &[Self]) -> Self {
        cblas1::dot_f64(x, y)
    }

    #[inline]
    fn vec_dot_plain(x: &[Self], y: &[Self]) -> Self {
        cblas1::dot_f64(x, y)
    }

    #[inline]
    fn potf2_dot(x: &[Self], y: &[Self], _conj_y: bool) -> Self {
        // f64 conjugation is the identity: the plain AVX2 vector dot.
        cblas1::dot_f64(x, y)
    }

    #[inline]
    fn vec_norm_sqr(x: &[Self]) -> Self {
        let n = cblas1::nrm2_f64(x);
        // `nrm2` is exact while squaring stays in range; delegate to the
        // scaled accumulation when the square would overflow (or `n` is
        // non-finite), instead of returning a spurious `inf`.
        if n > 0.0 && n <= f64::MAX.sqrt() {
            n * n
        } else {
            scaled_vec_norm_sqr(x)
        }
    }

    #[inline]
    fn vec_norm(x: &[Self]) -> Self {
        let norm = cblas1::nrm2_f64(x);
        if norm.is_finite() {
            norm
        } else {
            x.iter()
                .fold(0.0_f64, |norm, value| norm.hypot(value.norm()))
        }
    }

    #[inline]
    fn dlassq_row_update(row: &[Self], scales: &mut [Self], sums: &mut [Self]) -> bool {
        cblas1::dlassq_row_update_f64(row, scales, sums)
    }

    #[inline]
    fn vec_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        cblas1::axpy_f64(alpha, x, y);
    }

    #[inline]
    fn lu_prefers_tiny_pool() -> bool {
        true
    }

    #[inline]
    fn lu_prefers_parallel_lookahead_update() -> bool {
        true
    }

    #[inline]
    fn chol_prefers_tiny_pool() -> bool {
        true
    }

    #[inline]
    fn qr_prefers_tiny_pool() -> bool {
        true
    }

    #[inline]
    fn trsm_u12_unit_lower(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        ncols: usize,
    ) {
        if panel == 0 || ncols == 0 {
            return;
        }
        cblas3::trsm::trsm_f64(
            cblas3::trsm::TrsmSide::Left,
            cblas3::trsm::TrsmUplo::Lower,
            Transpose::NoTrans,
            true,
            panel,
            ncols,
            1.0,
            l11,
            lda,
            block,
            block_stride,
        );
    }

    #[inline]
    fn trsm_dot4(l: &[Self], x: [&[Self]; 4]) -> [Self; 4] {
        cblas1::dot4_f64(l, x)
    }

    #[inline]
    fn trsm_l21_right(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        nrows: usize,
        trans: Transpose,
    ) {
        if panel == 0 || nrows == 0 {
            return;
        }
        cblas3::trsm::trsm_f64(
            cblas3::trsm::TrsmSide::Right,
            cblas3::trsm::TrsmUplo::Lower,
            trans,
            false,
            nrows,
            panel,
            1.0,
            l11,
            lda,
            block,
            block_stride,
        );
    }

    fn vec_pivot_scan(
        base: &[Self],
        stride: usize,
        first: usize,
        len: usize,
    ) -> (usize, Self::Real) {
        cblas1::pivot_scan_f64(base, stride, first, len)
    }

    #[inline]
    fn vec_ger_update(
        alpha: Self,
        mults: &[Self],
        mult_stride: usize,
        pivot: &[Self],
        block: &mut [Self],
        block_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        cblas1::ger_update_f64(
            alpha,
            mults,
            mult_stride,
            pivot,
            block,
            block_stride,
            rows,
            cols,
        );
    }

    #[inline]
    fn find_non_finite(values: &[Self]) -> Option<usize> {
        cblas1::find_non_finite_f64(values)
    }

    #[inline]
    fn vec_max_norm(values: &[Self]) -> Self::Real {
        cblas1::max_norm_f64(values)
    }

    #[inline]
    fn vec_max_norm_checked(values: &[Self]) -> (Self::Real, bool) {
        cblas1::max_norm_checked_f64(values)
    }

    #[inline]
    fn vec_scale(x: &mut [Self], alpha: Self) {
        cblas1::scal_f64(alpha, x);
    }

    fn mat_mul_nn(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        cblas3::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            k,
            b,
            n,
            0.0,
            c,
            n,
        );
    }

    fn mat_mul_nn_sub(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        b: &[Self],
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            a,
            k,
            b,
            n,
            1.0,
            c,
            ldc,
        );
    }

    #[inline]
    fn syrk_lower(
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::syrk::syrk_f64(
            cblas3::trsm::TrsmUplo::Lower,
            trans,
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
        );
    }

    fn herk_lower(
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        // Conjugation is a no-op for real types: identical to NoTrans.
        Self::syrk_lower(Transpose::NoTrans, n, k, alpha, a, lda, beta, c, ldc);
    }

    fn mat_mul_tn(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        cblas3::gemm_f64(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            m,
            b,
            n,
            0.0,
            c,
            n,
        );
    }

    fn schur_sub(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self], ldc: usize) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            a,
            k,
            b,
            n,
            1.0,
            c,
            ldc,
        );
    }

    fn schur_sub_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            a,
            lda,
            b,
            ldb,
            1.0,
            c,
            ldc,
        );
    }

    fn schur_sub_serial(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f64_serial(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            a,
            lda,
            b,
            ldb,
            1.0,
            c,
            ldc,
        );
    }

    #[inline]
    fn gemm_hermitian_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f64(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            lda,
            b,
            ldb,
            0.0,
            c,
            ldc,
        );
    }

    #[inline]
    fn gemm_plain_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            lda,
            b,
            ldb,
            0.0,
            c,
            ldc,
        );
    }

    fn mat_vec_op(op: Transpose, rows: usize, cols: usize, a: &[Self], x: &[Self], y: &mut [Self]) {
        if rows * cols < blas2_min_elems() {
            mat_vec_op_scalar(op, rows, cols, a, x, y);
            return;
        }
        // blas2's Trans/ConjTrans contract stores A as n × m, so the
        // output length `m` is `cols` and the reduce length `n` is `rows`;
        // the row-major storage with lda = cols satisfies lda >= m.
        let (m, n) = match op {
            Transpose::NoTrans => (rows, cols),
            Transpose::Trans | Transpose::ConjTrans => (cols, rows),
        };
        cblas2::gemv_f64(op, m, n, 1.0, a, cols, x, 0.0, y);
    }
}

impl ComplexField for f32 {
    type Real = f32;

    #[inline]
    fn lu_prefers_f32_wide_panel() -> bool {
        true
    }

    #[inline]
    fn conj(&self) -> Self {
        *self
    }

    #[inline]
    fn norm_sqr(&self) -> f32 {
        *self * *self
    }

    #[inline]
    fn norm(&self) -> f32 {
        // Real-type exact norm without `hypot`'s scaling path; see the
        // f64 override (same precedent as f16).
        self.abs()
    }

    #[inline]
    fn from_real(r: f32) -> Self {
        r
    }

    #[inline]
    fn from_re_im(re: f32, _im: f32) -> Self {
        re
    }

    #[inline]
    fn re(&self) -> f32 {
        *self
    }

    #[inline]
    fn im(&self) -> f32 {
        0.0
    }

    #[inline]
    fn inv(&self) -> Self {
        1.0 / *self
    }

    #[inline]
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }

    #[inline]
    fn vec_dot(x: &[Self], y: &[Self]) -> Self {
        cblas1::dot_f32(x, y)
    }

    #[inline]
    fn vec_dot_plain(x: &[Self], y: &[Self]) -> Self {
        cblas1::dot_f32(x, y)
    }

    #[inline]
    fn vec_norm_sqr(x: &[Self]) -> Self {
        let n = cblas1::nrm2_f32(x);
        // `nrm2` is exact while squaring stays in range; delegate to the
        // scaled accumulation when the square would overflow (or `n` is
        // non-finite), instead of returning a spurious `inf`.
        if n > 0.0 && n <= f32::MAX.sqrt() {
            n * n
        } else {
            scaled_vec_norm_sqr(x)
        }
    }

    #[inline]
    fn vec_norm(x: &[Self]) -> Self {
        let norm = cblas1::nrm2_f32(x);
        if norm.is_finite() {
            norm
        } else {
            x.iter()
                .fold(0.0_f32, |norm, value| norm.hypot(value.norm()))
        }
    }

    #[inline]
    fn dlassq_row_update(row: &[Self], scales: &mut [Self], sums: &mut [Self]) -> bool {
        cblas1::dlassq_row_update_f32(row, scales, sums)
    }

    #[inline]
    fn vec_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        cblas1::axpy_f32(alpha, x, y);
    }

    #[inline]
    fn trsm_u12_unit_lower(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        ncols: usize,
    ) {
        if panel == 0 || ncols == 0 {
            return;
        }
        cblas3::trsm::trsm_f32(
            cblas3::trsm::TrsmSide::Left,
            cblas3::trsm::TrsmUplo::Lower,
            Transpose::NoTrans,
            true,
            panel,
            ncols,
            1.0,
            l11,
            lda,
            block,
            block_stride,
        );
    }

    #[inline]
    fn trsm_dot4(l: &[Self], x: [&[Self]; 4]) -> [Self; 4] {
        cblas1::dot4_f32(l, x)
    }

    #[inline]
    fn trsm_l21_right(
        l11: &[Self],
        lda: usize,
        panel: usize,
        block: &mut [Self],
        block_stride: usize,
        nrows: usize,
        trans: Transpose,
    ) {
        if panel == 0 || nrows == 0 {
            return;
        }
        cblas3::trsm::trsm_f32(
            cblas3::trsm::TrsmSide::Right,
            cblas3::trsm::TrsmUplo::Lower,
            trans,
            false,
            nrows,
            panel,
            1.0,
            l11,
            lda,
            block,
            block_stride,
        );
    }

    fn vec_pivot_scan(
        base: &[Self],
        stride: usize,
        first: usize,
        len: usize,
    ) -> (usize, Self::Real) {
        cblas1::pivot_scan_f32(base, stride, first, len)
    }

    #[inline]
    fn vec_ger_update(
        alpha: Self,
        mults: &[Self],
        mult_stride: usize,
        pivot: &[Self],
        block: &mut [Self],
        block_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        cblas1::ger_update_f32(
            alpha,
            mults,
            mult_stride,
            pivot,
            block,
            block_stride,
            rows,
            cols,
        );
    }

    #[inline]
    fn find_non_finite(values: &[Self]) -> Option<usize> {
        cblas1::find_non_finite_f32(values)
    }

    #[inline]
    fn vec_max_norm(values: &[Self]) -> Self::Real {
        cblas1::max_norm_f32(values)
    }

    #[inline]
    fn vec_max_norm_checked(values: &[Self]) -> (Self::Real, bool) {
        cblas1::max_norm_checked_f32(values)
    }

    #[inline]
    fn vec_scale(x: &mut [Self], alpha: Self) {
        cblas1::scal_f32(alpha, x);
    }

    fn mat_mul_nn(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        cblas3::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            k,
            b,
            n,
            0.0,
            c,
            n,
        );
    }

    fn mat_mul_nn_sub(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        b: &[Self],
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            a,
            k,
            b,
            n,
            1.0,
            c,
            ldc,
        );
    }

    #[inline]
    fn syrk_lower(
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        cblas3::syrk::syrk_f32(
            cblas3::trsm::TrsmUplo::Lower,
            trans,
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
        );
    }

    fn herk_lower(
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        // Conjugation is a no-op for real types: identical to NoTrans.
        Self::syrk_lower(Transpose::NoTrans, n, k, alpha, a, lda, beta, c, ldc);
    }

    fn mat_mul_tn(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        cblas3::gemm_f32(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            m,
            b,
            n,
            0.0,
            c,
            n,
        );
    }

    fn schur_sub(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self], ldc: usize) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            a,
            k,
            b,
            n,
            1.0,
            c,
            ldc,
        );
    }

    fn schur_sub_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            a,
            lda,
            b,
            ldb,
            1.0,
            c,
            ldc,
        );
    }

    fn schur_sub_serial(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f32_serial(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            a,
            lda,
            b,
            ldb,
            1.0,
            c,
            ldc,
        );
    }

    #[inline]
    fn gemm_hermitian_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f32(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            lda,
            b,
            ldb,
            0.0,
            c,
            ldc,
        );
    }

    #[inline]
    fn gemm_plain_strided(
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        cblas3::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            lda,
            b,
            ldb,
            0.0,
            c,
            ldc,
        );
    }

    fn mat_vec_op(op: Transpose, rows: usize, cols: usize, a: &[Self], x: &[Self], y: &mut [Self]) {
        if rows * cols < blas2_min_elems() {
            mat_vec_op_scalar(op, rows, cols, a, x, y);
            return;
        }
        let (m, n) = match op {
            Transpose::NoTrans => (rows, cols),
            Transpose::Trans | Transpose::ConjTrans => (cols, rows),
        };
        cblas2::gemv_f32(op, m, n, 1.0, a, cols, x, 0.0, y);
    }
}

impl ComplexField for f16 {
    type Real = f16;

    #[inline]
    fn conj(&self) -> Self {
        *self
    }

    #[inline]
    fn norm_sqr(&self) -> f16 {
        *self * *self
    }

    #[inline]
    fn norm(&self) -> f16 {
        // The default norm() uses hypot(re, im), but f16 has no complex
        // components and its scalar square still overflows for |v| > 255
        // (v*v exceeds f16::MAX). |v| itself always fits f16, so round-trip
        // through f32 instead.
        f16::from_f32(f32::from(*self).abs())
    }

    #[inline]
    fn from_real(r: f16) -> Self {
        r
    }

    #[inline]
    fn from_re_im(re: f16, _im: f16) -> Self {
        re
    }

    #[inline]
    fn re(&self) -> f16 {
        *self
    }

    #[inline]
    fn im(&self) -> f16 {
        f16::ZERO
    }

    #[inline]
    fn inv(&self) -> Self {
        f16::ONE / *self
    }

    #[inline]
    fn sqrt(&self) -> Self {
        // f16 has no native sqrt; round-trip through f32
        f16::from_f32(f32::from(*self).sqrt())
    }
}

/// Trait for linear operators (matrices) that can perform matrix-vector products.
///
/// This abstraction allows solvers to work with dense matrices, sparse matrices,
/// and matrix-free operators (e.g., FMM) interchangeably. Vectors are passed as
/// plain slices.
pub trait LinearOperator<T: ComplexField>: Send + Sync {
    /// Number of rows in the operator
    fn num_rows(&self) -> usize;

    /// Number of columns in the operator
    fn num_cols(&self) -> usize;

    /// Apply the operator: y = A * x
    fn apply(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.num_rows()];
        self.apply_into(x, &mut y);
        y
    }

    /// Checked forward application with input/output shape validation.
    fn try_apply(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        if x.len() != self.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols(),
                got: x.len(),
            });
        }
        reject_non_finite_input("operator apply", x)?;
        let applied = catch_solver_panic("operator apply", || self.apply(x))?;
        if applied.len() != self.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows(),
                got: applied.len(),
            });
        }
        reject_non_finite_operator_output("operator apply", &applied)?;
        Ok(applied)
    }

    /// Apply the operator in-place: y = A * x
    ///
    /// Writes the result into `y`, avoiding allocation. Implementors should
    /// override this for zero-allocation performance.
    fn apply_into(&self, x: &[T], y: &mut [T]);

    /// Apply the operator to a block of right-hand sides: `Y = A * X`.
    ///
    /// `X` is `num_cols() × k` and `Y` is `num_rows() × k`, both row-major
    /// with one right-hand side per column.
    ///
    /// The default implementation loops over the columns: gather column `j`
    /// into a contiguous buffer, call [`Self::apply_into`], scatter the
    /// result back. That costs one `O(n)` gather/scatter pair per column on
    /// top of the matvec. Implementors with a genuine multi-vector kernel
    /// (dense GEMM via `ComplexField::mat_mul_nn`, sparse SpMM, backend
    /// buffers) should override this to avoid the layout traffic.
    fn apply_block(&self, x: &crate::dense::DenseMatrix<T>, y: &mut crate::dense::DenseMatrix<T>) {
        assert_eq!(x.nrows(), self.num_cols(), "block input row mismatch");
        assert_eq!(y.nrows(), self.num_rows(), "block output row mismatch");
        assert_eq!(x.ncols(), y.ncols(), "block column count mismatch");
        let n = self.num_cols();
        let m = self.num_rows();
        let mut xj = vec![T::zero(); n];
        let mut yj = vec![T::zero(); m];
        for j in 0..x.ncols() {
            for i in 0..n {
                xj[i] = x[[i, j]];
            }
            self.apply_into(&xj, &mut yj);
            y.set_column(j, &yj);
        }
    }

    /// Checked block application with matrix shape validation.
    fn try_apply_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        if x.nrows() != self.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols(),
                got: x.nrows(),
            });
        }
        if y.nrows() != self.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows(),
                got: y.nrows(),
            });
        }
        if x.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: x.ncols(),
                got: y.ncols(),
            });
        }
        reject_non_finite_input("operator block apply", x.as_slice())?;
        catch_solver_panic("operator block apply", || self.apply_block(x, y))?;
        reject_non_finite_operator_output("operator block apply", y.as_slice())
    }

    /// Apply the transpose to a block of right-hand sides: `Y = Aᵀ * X`.
    ///
    /// `X` is `num_rows() × k` and `Y` is `num_cols() × k`. The default
    /// gathers one input column, calls the existing single-vector transpose
    /// callback, and scatters the result. Implementors with a native
    /// multi-vector transpose kernel may override this method.
    fn apply_transpose_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) {
        assert_eq!(
            x.nrows(),
            self.num_rows(),
            "block transpose input row mismatch"
        );
        assert_eq!(
            y.nrows(),
            self.num_cols(),
            "block transpose output row mismatch"
        );
        assert_eq!(
            x.ncols(),
            y.ncols(),
            "block transpose column count mismatch"
        );
        let m = self.num_rows();
        let n = self.num_cols();
        let mut xj = vec![T::zero(); m];
        let mut yj = vec![T::zero(); n];
        for j in 0..x.ncols() {
            for i in 0..m {
                xj[i] = x[[i, j]];
            }
            self.apply_transpose_into(&xj, &mut yj);
            y.set_column(j, &yj);
        }
    }

    /// Checked block transpose application for adjoint-capable solver paths.
    fn try_apply_transpose_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        if self.transpose_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "operator block transpose",
                operator: std::any::type_name::<Self>(),
            });
        }
        if x.nrows() != self.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows(),
                got: x.nrows(),
            });
        }
        if y.nrows() != self.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols(),
                got: y.nrows(),
            });
        }
        if x.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: x.ncols(),
                got: y.ncols(),
            });
        }
        reject_non_finite_input("operator block transpose", x.as_slice())?;
        catch_solver_panic("operator block transpose", || {
            self.apply_transpose_block(x, y)
        })?;
        reject_non_finite_operator_output("operator block transpose", y.as_slice())
    }

    /// Report whether the exact transpose action is available.
    ///
    /// Linear operators historically required `apply_transpose_into`, so the
    /// default remains exact for compatibility. Matrix-free operators that
    /// cannot provide an adjoint should override this to `Unsupported` and
    /// use the checked dispatch methods below.
    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Unknown
    }

    /// Report whether the exact Hermitian action is available.
    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Unknown
    }

    /// Apply the transpose: y = A^T * x
    fn apply_transpose(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.num_cols()];
        self.apply_transpose_into(x, &mut y);
        y
    }

    /// Apply the transpose in-place: y = A^T * x
    fn apply_transpose_into(&self, x: &[T], y: &mut [T]);

    /// Checked transpose application for adjoint-capable solver paths.
    fn try_apply_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        if self.transpose_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "operator transpose",
                operator: std::any::type_name::<Self>(),
            });
        }
        if x.len() != self.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows(),
                got: x.len(),
            });
        }
        reject_non_finite_input("operator transpose", x)?;
        let applied = catch_solver_panic("operator transpose", || self.apply_transpose(x))?;
        if applied.len() != self.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols(),
                got: applied.len(),
            });
        }
        reject_non_finite_operator_output("operator transpose", &applied)?;
        Ok(applied)
    }

    /// Apply the Hermitian (conjugate transpose): y = A^H * x
    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        let x_conj: Vec<T> = x.iter().map(|v| v.conj()).collect();
        let mut y = self.apply_transpose(&x_conj);
        for v in y.iter_mut() {
            *v = v.conj();
        }
        y
    }

    /// Checked Hermitian application for adjoint-capable solver paths.
    fn try_apply_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        if self.hermitian_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "operator Hermitian transpose",
                operator: std::any::type_name::<Self>(),
            });
        }
        if x.len() != self.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows(),
                got: x.len(),
            });
        }
        reject_non_finite_input("operator Hermitian transpose", x)?;
        let applied =
            catch_solver_panic("operator Hermitian transpose", || self.apply_hermitian(x))?;
        if applied.len() != self.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols(),
                got: applied.len(),
            });
        }
        reject_non_finite_operator_output("operator Hermitian transpose", &applied)?;
        Ok(applied)
    }

    /// Apply the Hermitian transpose to a block of right-hand sides:
    /// `Y = Aᴴ * X`.
    ///
    /// `X` is `num_rows() × k` and `Y` is `num_cols() × k`. The default
    /// gathers one input column, calls the existing single-vector Hermitian
    /// callback, and scatters the result.
    fn apply_hermitian_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) {
        assert_eq!(
            x.nrows(),
            self.num_rows(),
            "block Hermitian input row mismatch"
        );
        assert_eq!(
            y.nrows(),
            self.num_cols(),
            "block Hermitian output row mismatch"
        );
        assert_eq!(
            x.ncols(),
            y.ncols(),
            "block Hermitian column count mismatch"
        );
        let m = self.num_rows();
        let n = self.num_cols();
        let mut xj = vec![T::zero(); m];
        for j in 0..x.ncols() {
            for i in 0..m {
                xj[i] = x[[i, j]];
            }
            let yj = self.apply_hermitian(&xj);
            assert_eq!(yj.len(), n, "block Hermitian callback output mismatch");
            y.set_column(j, &yj);
        }
    }

    /// Checked block Hermitian application for adjoint-capable solver paths.
    fn try_apply_hermitian_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        if self.hermitian_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "operator block Hermitian transpose",
                operator: std::any::type_name::<Self>(),
            });
        }
        if x.nrows() != self.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows(),
                got: x.nrows(),
            });
        }
        if y.nrows() != self.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols(),
                got: y.nrows(),
            });
        }
        if x.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: x.ncols(),
                got: y.ncols(),
            });
        }
        reject_non_finite_input("operator block Hermitian transpose", x.as_slice())?;
        catch_solver_panic("operator block Hermitian transpose", || {
            self.apply_hermitian_block(x, y)
        })?;
        reject_non_finite_operator_output("operator block Hermitian transpose", y.as_slice())
    }

    /// Check if the operator is square
    fn is_square(&self) -> bool {
        self.num_rows() == self.num_cols()
    }
}

// Blanket impl so a reference to a `LinearOperator` is itself a
// `LinearOperator`.
impl<T, O> LinearOperator<T> for &O
where
    T: ComplexField,
    O: LinearOperator<T> + ?Sized,
{
    fn num_rows(&self) -> usize {
        (**self).num_rows()
    }
    fn num_cols(&self) -> usize {
        (**self).num_cols()
    }
    fn apply(&self, x: &[T]) -> Vec<T> {
        (**self).apply(x)
    }
    fn try_apply(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        (**self).try_apply(x)
    }
    fn apply_into(&self, x: &[T], y: &mut [T]) {
        (**self).apply_into(x, y);
    }
    fn apply_block(&self, x: &crate::dense::DenseMatrix<T>, y: &mut crate::dense::DenseMatrix<T>) {
        (**self).apply_block(x, y);
    }
    fn try_apply_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        (**self).try_apply_block(x, y)
    }
    fn apply_transpose_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) {
        (**self).apply_transpose_block(x, y);
    }
    fn try_apply_transpose_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        (**self).try_apply_transpose_block(x, y)
    }
    fn apply_transpose(&self, x: &[T]) -> Vec<T> {
        (**self).apply_transpose(x)
    }
    fn transpose_capability(&self) -> AdjointCapability {
        (**self).transpose_capability()
    }
    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        (**self).apply_transpose_into(x, y);
    }
    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        (**self).apply_hermitian(x)
    }
    fn apply_hermitian_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) {
        (**self).apply_hermitian_block(x, y);
    }
    fn try_apply_hermitian_block(
        &self,
        x: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        (**self).try_apply_hermitian_block(x, y)
    }
    fn hermitian_capability(&self) -> AdjointCapability {
        (**self).hermitian_capability()
    }
}

/// Status of an iterative solver
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStatus {
    /// Solver converged to the desired tolerance
    Converged,
    /// Solver reached the maximum number of iterations without converging
    MaxIterationsReached,
    /// Solver encountered a breakdown (e.g., division by zero)
    Breakdown,
    /// Solver stagnated (no progress made)
    Stagnated,
    /// Solver rejected non-finite input or produced a non-finite iterate.
    NumericalError,
    /// Solver diverged (residual is increasing)
    Diverged,
}

/// Version of the shared solver result/status/error contract.
pub const SOLVER_RESULT_CONTRACT_VERSION: u32 = 1;

impl SolverStatus {
    /// Whether the solver established the requested convergence criterion.
    pub const fn is_converged(self) -> bool {
        matches!(self, Self::Converged)
    }

    /// Whether the solver terminated without a trustworthy converged result.
    pub const fn is_failure(self) -> bool {
        !self.is_converged()
    }
}

/// Status of the requested accuracy check on a returned solver iterate.
///
/// `NotAttained` means that the returned iterate did not meet the requested
/// tolerance; it is not a proof that no more accurate iterate exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccuracyStatus {
    /// The solver returned before a finite true-residual check was available.
    NotChecked,
    /// The independently recomputed true residual met the requested tolerance.
    Attained,
    /// The independently recomputed true residual did not meet the tolerance.
    NotAttained,
}

impl AccuracyStatus {
    /// Whether an independently checked true residual met the target.
    pub const fn is_attained(self) -> bool {
        matches!(self, Self::Attained)
    }
}

/// Stable category for a [`SolverError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverErrorKind {
    /// The requested iteration/solve did not converge.
    Convergence,
    /// The numerical operation broke down.
    Breakdown,
    /// Input or operator dimensions were incompatible.
    DimensionMismatch,
    /// A requested solver or preconditioner operation is unavailable.
    UnsupportedOperation,
    /// An operator operation is unavailable.
    UnsupportedOperatorOperation,
    /// The operation failed for a non-taxonomy-specific reason.
    OperationFailed,
}

impl SolverErrorKind {
    /// Return the least-specific stable category implied by a solver status.
    ///
    /// Concrete failure types should override [`SolverFailure::error_kind`]
    /// when they can distinguish dimension and unsupported-operation errors.
    pub const fn from_status(status: SolverStatus) -> Self {
        match status {
            SolverStatus::MaxIterationsReached
            | SolverStatus::Stagnated
            | SolverStatus::Diverged => Self::Convergence,
            SolverStatus::Breakdown => Self::Breakdown,
            SolverStatus::Converged | SolverStatus::NumericalError => Self::OperationFailed,
        }
    }
}

/// Canonical metadata contract for checked solver failures.
///
/// Every solver family may keep its own concrete error enum, but checked
/// callers can use this trait to consume the same status and accuracy
/// vocabulary. The method aliases cover both the legacy `solver_status` /
/// `attainable_accuracy` spelling and the shorter `status` spelling.
pub trait SolverFailure {
    /// Return the stable failure category without parsing display text.
    ///
    /// The default preserves convergence and breakdown semantics for external
    /// implementations. Built-in failures override it for richer categories.
    fn error_kind(&self) -> SolverErrorKind {
        SolverErrorKind::from_status(self.status())
    }

    /// Return the canonical status represented by this failure.
    fn status(&self) -> SolverStatus;

    /// Return whether a requested accuracy was established before failure.
    fn accuracy_status(&self) -> AccuracyStatus;

    /// Compatibility alias for result types that use `solver_status`.
    fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    /// Compatibility alias for result types that use `attainable_accuracy`.
    fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

/// Canonical contract for a solver result returned by a checked entry point.
///
/// Solver families keep their domain-specific result structures, but every
/// checked result must expose the same status/accuracy vocabulary and a
/// validator that rejects contradictory metadata. This is separate from
/// [`SolverFailure`]: a non-converged result is still a valid result object,
/// while a failure is returned as an error.
pub trait SolverResultContract {
    /// Canonical status of the returned iterate or block of iterates.
    fn solver_status(&self) -> SolverStatus;

    /// Whether the requested accuracy was established by the result.
    fn accuracy_status(&self) -> AccuracyStatus;

    /// Validate status/accuracy metadata and finite diagnostics.
    fn validate_contract(&self) -> Result<(), SolverError>;
}

/// Error information from iterative solvers
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    #[error("Solver failed to converge: {status:?}")]
    ConvergenceError {
        status: SolverStatus,
        iterations: usize,
        residual: f64,
    },
    #[error("{operation} broke down: {reason}")]
    Breakdown {
        operation: &'static str,
        reason: String,
    },
    #[error("Linear operator dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("{operation} is not supported by preconditioner {preconditioner}")]
    UnsupportedOperation {
        operation: &'static str,
        preconditioner: &'static str,
    },
    #[error("{operation} is not supported by operator {operator}")]
    UnsupportedOperatorOperation {
        operation: &'static str,
        operator: &'static str,
    },
    #[error("{operation} failed: {reason}")]
    OperationFailed {
        operation: &'static str,
        reason: String,
    },
}

impl SolverError {
    /// Return the stable error category without parsing display text.
    pub const fn kind(&self) -> SolverErrorKind {
        match self {
            Self::ConvergenceError { .. } => SolverErrorKind::Convergence,
            Self::Breakdown { .. } => SolverErrorKind::Breakdown,
            Self::DimensionMismatch { .. } => SolverErrorKind::DimensionMismatch,
            Self::UnsupportedOperation { .. } => SolverErrorKind::UnsupportedOperation,
            Self::UnsupportedOperatorOperation { .. } => {
                SolverErrorKind::UnsupportedOperatorOperation
            }
            Self::OperationFailed { .. } => SolverErrorKind::OperationFailed,
        }
    }

    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::ConvergenceError { status, .. } => *status,
            Self::Breakdown { .. } => SolverStatus::Breakdown,
            Self::DimensionMismatch { .. }
            | Self::UnsupportedOperation { .. }
            | Self::UnsupportedOperatorOperation { .. }
            | Self::OperationFailed { .. } => SolverStatus::NumericalError,
        }
    }

    /// Return whether a requested numerical accuracy was established.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        match self {
            Self::ConvergenceError { status, .. } => match status {
                SolverStatus::MaxIterationsReached
                | SolverStatus::Stagnated
                | SolverStatus::Diverged => AccuracyStatus::NotAttained,
                SolverStatus::Converged
                | SolverStatus::Breakdown
                | SolverStatus::NumericalError => AccuracyStatus::NotChecked,
            },
            Self::Breakdown { .. } => AccuracyStatus::NotChecked,
            Self::DimensionMismatch { .. }
            | Self::UnsupportedOperation { .. }
            | Self::UnsupportedOperatorOperation { .. }
            | Self::OperationFailed { .. } => AccuracyStatus::NotChecked,
        }
    }

    /// Alias matching checked result types that expose `accuracy_status`.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        self.attainable_accuracy()
    }

    /// Alias matching checked result types that expose `solver_status`.
    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    /// Preserve a nested checked failure's breakdown category at a family
    /// boundary. Other categories remain operation failures unless the
    /// caller has richer convergence metadata to construct a
    /// `ConvergenceError` directly.
    pub fn from_failure<E>(operation: &'static str, error: &E) -> Self
    where
        E: SolverFailure + std::fmt::Display + ?Sized,
    {
        match error.status() {
            SolverStatus::Breakdown => Self::Breakdown {
                operation,
                reason: error.to_string(),
            },
            _ => Self::OperationFailed {
                operation,
                reason: error.to_string(),
            },
        }
    }
}

impl SolverFailure for SolverError {
    fn error_kind(&self) -> SolverErrorKind {
        self.kind()
    }

    fn status(&self) -> SolverStatus {
        SolverError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        SolverError::attainable_accuracy(self)
    }
}

/// Whether a preconditioner exposes an exact adjoint operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjointCapability {
    /// No exact adjoint contract has been declared.
    Unknown,
    /// The preconditioner cannot be used for this adjoint operation.
    Unsupported,
    /// The operation is implemented as the exact transpose/Hermitian action
    /// of the stored preconditioner representation.
    Exact,
}

/// Trait for preconditioners used in iterative solvers.
///
/// A preconditioner M approximates A^(-1), so that M*A is better conditioned
/// than A alone. This accelerates convergence of iterative methods.
pub trait Preconditioner<T: ComplexField>: Send + Sync {
    /// Apply the preconditioner: y = M * r
    ///
    /// This should approximate solving A * y = r
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); r.len()];
        self.apply_into(r, &mut y);
        y
    }

    /// Checked forward application. A preconditioner preserves vector
    /// length, so the result shape is verified before it is returned.
    fn try_apply(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        reject_non_finite_input("preconditioner apply", r)?;
        let applied = catch_solver_panic("preconditioner apply", || self.apply(r))?;
        if applied.len() != r.len() {
            return Err(SolverError::DimensionMismatch {
                expected: r.len(),
                got: applied.len(),
            });
        }
        reject_non_finite_preconditioner_output("preconditioner apply", &applied)?;
        Ok(applied)
    }

    /// Apply the preconditioner in-place: y = M * r
    fn apply_into(&self, r: &[T], y: &mut [T]) {
        let applied = self.apply(r);
        assert_eq!(
            applied.len(),
            y.len(),
            "preconditioner output length mismatch"
        );
        y.copy_from_slice(&applied);
    }

    /// Apply the preconditioner to a block of right-hand sides: `Y = M * R`.
    ///
    /// `R` and `Y` are `n × k` row-major with one right-hand side per
    /// column. The default loops over the columns with
    /// [`Self::apply_into`] (one `O(n)` gather/scatter pair per column on
    /// top of the apply); implementors with a multi-vector kernel should
    /// override this to avoid the layout traffic.
    fn apply_block(&self, r: &crate::dense::DenseMatrix<T>, y: &mut crate::dense::DenseMatrix<T>) {
        assert_eq!(r.ncols(), y.ncols(), "block column count mismatch");
        assert_eq!(r.nrows(), y.nrows(), "block row count mismatch");
        let n = r.nrows();
        let mut rj = vec![T::zero(); n];
        let mut yj = vec![T::zero(); n];
        for j in 0..r.ncols() {
            for i in 0..n {
                rj[i] = r[[i, j]];
            }
            self.apply_into(&rj, &mut yj);
            y.set_column(j, &yj);
        }
    }

    /// Checked block preconditioner application.
    fn try_apply_block(
        &self,
        r: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        if r.nrows() != y.nrows() {
            return Err(SolverError::DimensionMismatch {
                expected: r.nrows(),
                got: y.nrows(),
            });
        }
        if r.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: r.ncols(),
                got: y.ncols(),
            });
        }
        reject_non_finite_input("preconditioner block apply", r.as_slice())?;
        catch_solver_panic("preconditioner block apply", || self.apply_block(r, y))?;
        reject_non_finite_preconditioner_output("preconditioner block apply", y.as_slice())
    }

    /// Report whether the exact transpose action is available.
    ///
    /// The default is conservative. A preconditioner that only implements
    /// [`Self::apply`] must not be silently reused for an adjoint solve.
    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Unsupported
    }

    /// Report whether the exact Hermitian action is available.
    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Unsupported
    }

    /// Apply the transpose of the preconditioner: `y = M^T * r`.
    ///
    /// Needed for adjoint solves that reuse the forward factorization. The
    /// default panics: most preconditioners are not adjoint-capable, and
    /// silently substituting the forward apply would be a wrong-result bug
    /// rather than a performance issue. Override this (and preferably
    /// [`Self::apply_hermitian`]) to advertise adjoint capability.
    fn apply_transpose(&self, _r: &[T]) -> Vec<T> {
        panic!(
            "{}: preconditioner is not adjoint-capable (apply_transpose not implemented)",
            std::any::type_name::<Self>()
        )
    }

    /// Checked transpose application for adjoint-capable solver paths.
    ///
    /// Unlike the legacy [`Self::apply_transpose`] method, this returns a
    /// typed error for an unsupported preconditioner and can therefore be
    /// called during dispatch validation before iteration zero.
    fn try_apply_transpose(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        if self.transpose_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "preconditioner transpose",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        reject_non_finite_input("preconditioner transpose", r)?;
        let applied = catch_solver_panic("preconditioner transpose", || self.apply_transpose(r))?;
        if applied.len() != r.len() {
            return Err(SolverError::DimensionMismatch {
                expected: r.len(),
                got: applied.len(),
            });
        }
        reject_non_finite_preconditioner_output("preconditioner transpose", &applied)?;
        Ok(applied)
    }

    /// Apply the transpose of the preconditioner in-place: `y = M^T * r`.
    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        let applied = self.apply_transpose(r);
        assert_eq!(
            applied.len(),
            y.len(),
            "preconditioner transpose output length mismatch"
        );
        y.copy_from_slice(&applied);
    }

    /// Apply the transpose to a block of right-hand sides: `Y = Mᵀ * R`.
    ///
    /// `R` and `Y` are `n × k`. The default gathers one input column, calls
    /// the existing single-vector transpose callback, and scatters the
    /// result. Implementors with a native multi-vector transpose kernel may
    /// override this method.
    fn apply_transpose_block(
        &self,
        r: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) {
        assert_eq!(r.nrows(), y.nrows(), "block transpose row count mismatch");
        assert_eq!(
            r.ncols(),
            y.ncols(),
            "block transpose column count mismatch"
        );
        let n = r.nrows();
        let mut rj = vec![T::zero(); n];
        let mut yj = vec![T::zero(); n];
        for j in 0..r.ncols() {
            for i in 0..n {
                rj[i] = r[[i, j]];
            }
            self.apply_transpose_into(&rj, &mut yj);
            y.set_column(j, &yj);
        }
    }

    /// Checked block transpose application for adjoint-capable solver paths.
    fn try_apply_transpose_block(
        &self,
        r: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        if self.transpose_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "preconditioner block transpose",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        if r.nrows() != y.nrows() {
            return Err(SolverError::DimensionMismatch {
                expected: r.nrows(),
                got: y.nrows(),
            });
        }
        if r.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: r.ncols(),
                got: y.ncols(),
            });
        }
        reject_non_finite_input("preconditioner block transpose", r.as_slice())?;
        catch_solver_panic("preconditioner block transpose", || {
            self.apply_transpose_block(r, y)
        })?;
        reject_non_finite_preconditioner_output("preconditioner block transpose", y.as_slice())
    }

    /// Apply the Hermitian (conjugate transpose) of the preconditioner:
    /// `y = M^H * r`.
    ///
    /// The default routes through [`Self::apply_transpose`] with a
    /// conjugate round-trip: `M^H r = conj(M^T conj(r))`. Implementors
    /// should override with a direct conjugate solve to skip the two
    /// `O(n)` conjugation passes.
    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let r_conj: Vec<T> = r.iter().map(|v| v.conj()).collect();
        let mut y = self.apply_transpose(&r_conj);
        for v in y.iter_mut() {
            *v = v.conj();
        }
        y
    }

    /// Checked Hermitian application for adjoint-capable solver paths.
    fn try_apply_hermitian(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        if self.hermitian_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "preconditioner Hermitian transpose",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        reject_non_finite_input("preconditioner Hermitian transpose", r)?;
        let applied = catch_solver_panic("preconditioner Hermitian transpose", || {
            self.apply_hermitian(r)
        })?;
        if applied.len() != r.len() {
            return Err(SolverError::DimensionMismatch {
                expected: r.len(),
                got: applied.len(),
            });
        }
        reject_non_finite_preconditioner_output("preconditioner Hermitian transpose", &applied)?;
        Ok(applied)
    }

    /// Apply the Hermitian transpose to a block of right-hand sides:
    /// `Y = Mᴴ * R`.
    fn apply_hermitian_block(
        &self,
        r: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) {
        assert_eq!(r.nrows(), y.nrows(), "block Hermitian row count mismatch");
        assert_eq!(
            r.ncols(),
            y.ncols(),
            "block Hermitian column count mismatch"
        );
        let n = r.nrows();
        let mut rj = vec![T::zero(); n];
        let mut yj = vec![T::zero(); n];
        for j in 0..r.ncols() {
            for i in 0..n {
                rj[i] = r[[i, j]];
            }
            self.apply_hermitian_into(&rj, &mut yj);
            y.set_column(j, &yj);
        }
    }

    /// Checked block Hermitian application for adjoint-capable solver paths.
    fn try_apply_hermitian_block(
        &self,
        r: &crate::dense::DenseMatrix<T>,
        y: &mut crate::dense::DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        if self.hermitian_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "preconditioner block Hermitian transpose",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        if r.nrows() != y.nrows() {
            return Err(SolverError::DimensionMismatch {
                expected: r.nrows(),
                got: y.nrows(),
            });
        }
        if r.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: r.ncols(),
                got: y.ncols(),
            });
        }
        reject_non_finite_input("preconditioner block Hermitian transpose", r.as_slice())?;
        catch_solver_panic("preconditioner block Hermitian transpose", || {
            self.apply_hermitian_block(r, y)
        })?;
        reject_non_finite_preconditioner_output(
            "preconditioner block Hermitian transpose",
            y.as_slice(),
        )
    }

    /// Apply the Hermitian of the preconditioner in-place: `y = M^H * r`.
    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        let applied = self.apply_hermitian(r);
        assert_eq!(
            applied.len(),
            y.len(),
            "preconditioner hermitian output length mismatch"
        );
        y.copy_from_slice(&applied);
    }
}

/// Identity preconditioner (no preconditioning)
#[derive(Clone, Debug, Default)]
pub struct IdentityPreconditioner;

impl<T: ComplexField> Preconditioner<T> for IdentityPreconditioner {
    fn apply(&self, r: &[T]) -> Vec<T> {
        r.to_vec()
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        y.copy_from_slice(r);
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        r.to_vec()
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        y.copy_from_slice(r);
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        r.to_vec()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        y.copy_from_slice(r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DenseMatrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_complex64_field() {
        let z = Complex64::new(3.0, 4.0);
        assert_relative_eq!(z.norm_sqr(), 25.0);
        assert_relative_eq!(z.norm(), 5.0);

        let z_conj = z.conj();
        assert_relative_eq!(z_conj.re, 3.0);
        assert_relative_eq!(z_conj.im, -4.0);

        let z_inv = z.inv();
        let product = z * z_inv;
        assert_relative_eq!(product.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(product.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f64_field() {
        let x: f64 = 3.0;
        assert_relative_eq!(x.norm_sqr(), 9.0);
        assert_relative_eq!(x.norm(), 3.0);
        assert_relative_eq!(x.conj(), 3.0);
        assert_relative_eq!(x.inv(), 1.0 / 3.0);
    }

    #[test]
    fn vec_dot_conjugates_left_factor() {
        // (1+i, 2) · (1-i, 1) = conj(1+i)*(1-i) + 2*1 = -2i + 2.
        let x = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];
        let y = vec![Complex64::new(1.0, -1.0), Complex64::new(1.0, 0.0)];
        let d = Complex64::vec_dot(&x, &y);
        assert_relative_eq!(d.re, 2.0);
        assert_relative_eq!(d.im, -2.0);
    }

    #[test]
    fn schur_sub_matches_manual_with_stride() {
        // 2×3 minus (2×2)·(2×3), C row stride 5: padding columns must be
        // untouched and only columns ..3 updated.
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut c = vec![
            100.0_f64, 101.0, 102.0, -1.0, -2.0, 103.0, 104.0, 105.0, -3.0, -4.0,
        ];
        f64::schur_sub(2, 3, 2, &a, &b, &mut c, 5);
        // Row 0: A·B = [1*5+2*8, 1*6+2*9, 1*7+2*10] = [21, 24, 27].
        // Row 1: A·B = [3*5+4*8, 3*6+4*9, 3*7+4*10] = [47, 54, 61].
        let expect = [79.0_f64, 77.0, 75.0, 56.0, 50.0, 44.0];
        for (got, want) in c
            .chunks(5)
            .flat_map(|row| row[..3].iter())
            .zip(expect.iter())
        {
            assert_relative_eq!(*got, *want, epsilon = 1e-12);
        }
        assert_eq!(&c[3..5], &[-1.0, -2.0]);
        assert_eq!(&c[8..10], &[-3.0, -4.0]);
    }

    #[test]
    fn schur_sub_strided_matches_packed() {
        // 2×3 minus (2×2)·(2×3) with row strides (lda 4, ldb 5, ldc 6):
        // strided entry must agree with the packed entry elementwise.
        let a_packed = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b_packed = vec![5.0_f64, 6.0, 7.0, 8.0, 9.0, 10.0];
        // Embed the same blocks with stride padding (pad values nonzero to
        // catch stride mistakes).
        let a = vec![1.0_f64, 2.0, -1.0, -1.0, 3.0, 4.0, -1.0, -1.0];
        let b = vec![5.0_f64, 6.0, 7.0, -1.0, -1.0, 8.0, 9.0, 10.0, -1.0, -1.0];
        let mut c_packed = vec![100.0_f64, 101.0, 102.0, 103.0, 104.0, 105.0];
        let mut c_strided = vec![
            100.0_f64, 101.0, 102.0, -1.0, -1.0, -1.0, 103.0, 104.0, 105.0, -1.0, -1.0, -1.0,
        ];
        f64::schur_sub(2, 3, 2, &a_packed, &b_packed, &mut c_packed, 3);
        f64::schur_sub_strided(2, 3, 2, &a, 4, &b, 5, &mut c_strided, 6);
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(c_strided[i * 6 + j], c_packed[i * 3 + j], epsilon = 1e-12);
            }
        }
        // Padding untouched.
        assert!(c_strided[3..6].iter().all(|&v| v == -1.0));
        assert!(c_strided[9..12].iter().all(|&v| v == -1.0));
    }

    #[test]
    fn find_non_finite_reports_first_bad_index() {
        // Clean input: None. NaN/±inf at block edges, tails, and complex
        // lanes (re vs im map back to the value index).
        let clean = vec![1.0_f64, -2.5, 0.0, 3.75];
        assert_eq!(f64::find_non_finite(&clean), None);
        let mut bad = clean.clone();
        bad[3] = f64::NAN;
        assert_eq!(f64::find_non_finite(&bad), Some(3));
        bad[3] = 1.0;
        bad[0] = f64::NEG_INFINITY;
        assert_eq!(f64::find_non_finite(&bad), Some(0));
        // Tail lane past the 4-wide blocks.
        let mut tail = vec![1.0_f64; 10];
        tail[9] = f64::INFINITY;
        assert_eq!(f64::find_non_finite(&tail), Some(9));
        // f32 tail past the 8-wide blocks.
        let mut tail32 = vec![1.0_f32; 17];
        tail32[16] = f32::NAN;
        assert_eq!(f32::find_non_finite(&tail32), Some(16));
        // Complex: bad `im` of value 1 and bad `re` of value 2.
        let c = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, f64::NAN),
            Complex64::new(f64::INFINITY, 0.0),
        ];
        assert_eq!(Complex64::find_non_finite(&c), Some(1));
        let c32 = vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
            Complex32::new(5.0, f32::NAN),
        ];
        assert_eq!(Complex32::find_non_finite(&c32), Some(2));
        assert_eq!(Complex32::find_non_finite(&c32[..2]), None);
    }

    #[test]
    fn vec_max_norm_matches_scalar_fold() {
        // Negatives, zeros, tails, and f64::MAX (abs is exact: must not
        // overflow to inf). Complex keeps the hypot default.
        let v = vec![-3.0_f64, 1.5, -0.0, 2.25, -7.5, 0.0, 4.0];
        assert_relative_eq!(f64::vec_max_norm(&v), 7.5, epsilon = 0.0);
        assert_relative_eq!(f64::vec_max_norm(&[]), 0.0, epsilon = 0.0);
        assert_relative_eq!(f64::vec_max_norm(&[f64::MAX, 1.0]), f64::MAX, epsilon = 0.0);
        let v32 = vec![-3.0_f32, 8.5, 2.0];
        assert_relative_eq!(f32::vec_max_norm(&v32), 8.5, epsilon = 0.0);
        let c = vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 1.0)];
        assert_relative_eq!(Complex64::vec_max_norm(&c), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn schur_sub_complex64_matches_manual() {
        // 1×2 minus (1×2)·(2×2).
        let a = vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, 2.0)];
        let b = vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];
        let mut c = vec![Complex64::new(10.0, 0.0), Complex64::new(0.0, 10.0)];
        Complex64::schur_sub(1, 2, 2, &a, &b, &mut c, 2);
        // A·B = [(1+i)*3 + 2i*1, (1+i)*i + 2i*2] = [3+5i, -1+5i].
        assert_relative_eq!(c[0].re, 7.0, epsilon = 1e-12);
        assert_relative_eq!(c[0].im, -5.0, epsilon = 1e-12);
        assert_relative_eq!(c[1].re, 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[1].im, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn vec_dot_plain_skips_conjugation() {
        // Same vectors, unconjugated: (1+i)*(1-i) + 2*1 = 2 + 2 = 4.
        let x = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];
        let y = vec![Complex64::new(1.0, -1.0), Complex64::new(1.0, 0.0)];
        let d = Complex64::vec_dot_plain(&x, &y);
        assert_relative_eq!(d.re, 4.0);
        assert_relative_eq!(d.im, 0.0);
        // Real types agree with the Hermitian dot.
        let xr = vec![1.5f64, -2.0, 3.25];
        let yr = vec![-1.0, 4.0, 0.5];
        assert_relative_eq!(
            f64::vec_dot_plain(&xr, &yr),
            f64::vec_dot(&xr, &yr),
            epsilon = 1e-15
        );
        assert_relative_eq!(
            f64::vec_dot_plain(&xr, &yr),
            1.5 * -1.0 + -2.0 * 4.0 + 3.25 * 0.5,
            epsilon = 1e-15
        );
    }

    #[test]
    fn vec_norm_sqr_range_contract() {
        // Two 1e200 entries: true squared norm 2e400 is unrepresentable,
        // so `inf` is correct — but the norm itself stays finite. This is
        // why magnitude-only callers must use `vec_norm`, never
        // `vec_norm_sqr(...).sqrt()`.
        let big = vec![1.0e200_f64, 1.0e200];
        assert_eq!(f64::vec_norm_sqr(&big), f64::INFINITY);
        assert!(f64::vec_norm(&big).is_finite());
        let c = vec![Complex64::new(1.0e200, 1.0e200)];
        assert_eq!(Complex64::vec_norm_sqr(&c), f64::INFINITY);
        assert!(Complex64::vec_norm(&c).is_finite());
        // In-range values keep exact behavior; a single 1e200 entry has
        // true square 1e400 (unrepresentable), so `inf` is correct there.
        assert_relative_eq!(f64::vec_norm_sqr(&[3.0, 4.0]), 25.0);
        assert_eq!(f64::vec_norm_sqr(&[1.0e200]), f64::INFINITY);
        assert_eq!(f64::vec_norm_sqr(&[]), 0.0);
        assert!(f64::vec_norm_sqr(&[f64::NAN]).is_nan());
    }

    #[test]
    fn stable_norm_accepts_finite_values_at_scalar_limit() {
        let real = f64::MAX;
        assert_eq!(real.norm(), f64::MAX);

        let complex = Complex64::new(f64::MAX, 1.0);
        assert_eq!(complex.norm(), f64::MAX);
    }

    #[test]
    fn stable_vec_norm_accepts_finite_values_at_scalar_limit() {
        assert_eq!(f64::vec_norm(&[f64::MAX]), f64::MAX);
        assert_eq!(
            Complex64::vec_norm(&[Complex64::new(f64::MAX, 1.0)]),
            f64::MAX
        );
    }

    #[test]
    fn stable_zero_check_does_not_overflow_tolerance_square() {
        let value = f64::MAX * 0.5;
        let tolerance = f64::MAX * 0.75;
        assert!(value.is_zero_approx(tolerance));
        assert!(Complex64::new(value, 0.0).is_zero_approx(tolerance));
    }

    #[test]
    fn test_identity_preconditioner() {
        let precond = IdentityPreconditioner;
        let r = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = <IdentityPreconditioner as Preconditioner<Complex64>>::apply(&precond, &r);
        assert_eq!(r, y);
        assert_eq!(
            <IdentityPreconditioner as Preconditioner<Complex64>>::transpose_capability(&precond),
            AdjointCapability::Exact
        );
        assert_eq!(precond.try_apply_transpose(&r).unwrap(), r);
        assert_eq!(precond.try_apply_hermitian(&r).unwrap(), r);
    }

    #[test]
    fn checked_preconditioner_application_rejects_non_finite_output() {
        struct NonFinite;
        impl Preconditioner<f64> for NonFinite {
            fn apply(&self, r: &[f64]) -> Vec<f64> {
                vec![f64::NAN; r.len()]
            }
        }

        let error = NonFinite
            .try_apply(&[1.0, 2.0])
            .expect_err("checked preconditioner output must reject NaN");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_operator_application_rejects_non_finite_output() {
        struct NonFinite;
        impl LinearOperator<f64> for NonFinite {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
                y.fill(f64::NAN);
            }

            fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
                y.fill(f64::NAN);
            }
        }

        let error = NonFinite
            .try_apply(&[1.0, 2.0])
            .expect_err("checked operator output must reject NaN");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    fn assert_input_rejected<T>(result: Result<T, SolverError>) {
        assert!(matches!(
            result,
            Err(SolverError::OperationFailed { reason, .. })
                if reason.contains("input contains")
        ));
    }

    #[test]
    fn checked_preconditioner_application_rejects_non_finite_input_before_callbacks() {
        struct PanicAdjoint;

        impl Preconditioner<f64> for PanicAdjoint {
            fn apply(&self, _r: &[f64]) -> Vec<f64> {
                panic!("forward callback must not be called")
            }

            fn apply_block(&self, _r: &DenseMatrix<f64>, _y: &mut DenseMatrix<f64>) {
                panic!("forward block callback must not be called")
            }

            fn transpose_capability(&self) -> AdjointCapability {
                AdjointCapability::Exact
            }

            fn apply_transpose(&self, _r: &[f64]) -> Vec<f64> {
                panic!("transpose callback must not be called")
            }

            fn apply_transpose_block(&self, _r: &DenseMatrix<f64>, _y: &mut DenseMatrix<f64>) {
                panic!("transpose block callback must not be called")
            }

            fn hermitian_capability(&self) -> AdjointCapability {
                AdjointCapability::Exact
            }

            fn apply_hermitian(&self, _r: &[f64]) -> Vec<f64> {
                panic!("Hermitian callback must not be called")
            }

            fn apply_hermitian_block(&self, _r: &DenseMatrix<f64>, _y: &mut DenseMatrix<f64>) {
                panic!("Hermitian block callback must not be called")
            }
        }

        let preconditioner = PanicAdjoint;
        assert_input_rejected(preconditioner.try_apply(&[f64::NAN, 1.0]));
        assert_input_rejected(preconditioner.try_apply_transpose(&[f64::NAN, 1.0]));
        assert_input_rejected(preconditioner.try_apply_hermitian(&[f64::NAN, 1.0]));

        let input = DenseMatrix::from_row_major_vec(2, 1, vec![f64::NAN, 1.0]);
        let mut output = DenseMatrix::from_row_major_vec(2, 1, vec![0.0, 0.0]);
        assert_input_rejected(preconditioner.try_apply_block(&input, &mut output));
        assert_input_rejected(preconditioner.try_apply_transpose_block(&input, &mut output));
        assert_input_rejected(preconditioner.try_apply_hermitian_block(&input, &mut output));
    }

    #[test]
    fn checked_operator_application_rejects_non_finite_input_before_callbacks() {
        struct PanicAdjoint;

        impl LinearOperator<f64> for PanicAdjoint {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply(&self, _x: &[f64]) -> Vec<f64> {
                panic!("forward callback must not be called")
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("forward in-place callback must not be called")
            }

            fn apply_block(&self, _x: &DenseMatrix<f64>, _y: &mut DenseMatrix<f64>) {
                panic!("forward block callback must not be called")
            }

            fn transpose_capability(&self) -> AdjointCapability {
                AdjointCapability::Exact
            }

            fn apply_transpose(&self, _x: &[f64]) -> Vec<f64> {
                panic!("transpose callback must not be called")
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("transpose in-place callback must not be called")
            }

            fn apply_transpose_block(&self, _x: &DenseMatrix<f64>, _y: &mut DenseMatrix<f64>) {
                panic!("transpose block callback must not be called")
            }

            fn hermitian_capability(&self) -> AdjointCapability {
                AdjointCapability::Exact
            }

            fn apply_hermitian(&self, _x: &[f64]) -> Vec<f64> {
                panic!("Hermitian callback must not be called")
            }

            fn apply_hermitian_block(&self, _x: &DenseMatrix<f64>, _y: &mut DenseMatrix<f64>) {
                panic!("Hermitian block callback must not be called")
            }
        }

        let operator = PanicAdjoint;
        assert_input_rejected(operator.try_apply(&[f64::NAN, 1.0]));
        assert_input_rejected(operator.try_apply_transpose(&[f64::NAN, 1.0]));
        assert_input_rejected(operator.try_apply_hermitian(&[f64::NAN, 1.0]));

        let input = DenseMatrix::from_row_major_vec(2, 1, vec![f64::NAN, 1.0]);
        let mut output = DenseMatrix::from_row_major_vec(2, 1, vec![0.0, 0.0]);
        assert_input_rejected(operator.try_apply_block(&input, &mut output));
        assert_input_rejected(operator.try_apply_transpose_block(&input, &mut output));
        assert_input_rejected(operator.try_apply_hermitian_block(&input, &mut output));
    }

    #[test]
    fn unsupported_preconditioner_adjoint_fails_before_application() {
        struct ForwardOnly;

        impl Preconditioner<f64> for ForwardOnly {
            fn apply(&self, r: &[f64]) -> Vec<f64> {
                r.to_vec()
            }
        }

        let precond = ForwardOnly;
        let error = precond
            .try_apply_transpose(&[1.0, 2.0])
            .expect_err("forward-only preconditioner must reject transpose");
        assert!(matches!(
            error,
            SolverError::UnsupportedOperation {
                operation: "preconditioner transpose",
                ..
            }
        ));
    }

    #[test]
    fn unsupported_operator_adjoint_fails_before_application() {
        struct ForwardOnly;

        impl LinearOperator<f64> for ForwardOnly {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply_into(&self, x: &[f64], y: &mut [f64]) {
                y.copy_from_slice(x);
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("forward-only operator transpose must not be called");
            }

            fn transpose_capability(&self) -> AdjointCapability {
                AdjointCapability::Unsupported
            }

            fn hermitian_capability(&self) -> AdjointCapability {
                AdjointCapability::Unsupported
            }
        }

        let operator = ForwardOnly;
        assert!(matches!(
            operator.try_apply_transpose(&[1.0, 2.0]),
            Err(SolverError::UnsupportedOperatorOperation { .. })
        ));
        assert!(matches!(
            operator.try_apply_hermitian(&[1.0, 2.0]),
            Err(SolverError::UnsupportedOperatorOperation { .. })
        ));
        assert!(matches!(
            operator.try_apply_transpose(&[1.0]),
            Err(SolverError::UnsupportedOperatorOperation { .. })
        ));
    }

    #[test]
    fn test_complex64_vec_dot() {
        let x = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(0.5, 0.5),
        ];
        let y = vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 3.0),
            Complex64::new(1.0, -1.0),
        ];
        let mut expected = Complex64::new(0.0, 0.0);
        for (xi, yi) in x.iter().zip(y.iter()) {
            expected += xi.conj() * *yi;
        }
        let result = Complex64::vec_dot(&x, &y);
        assert_relative_eq!(result.re, expected.re, epsilon = 1e-10);
        assert_relative_eq!(result.im, expected.im, epsilon = 1e-10);
    }

    #[test]
    fn test_complex64_vec_norm_sqr() {
        let x = vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, -2.0)];
        let result = Complex64::vec_norm_sqr(&x);
        assert_relative_eq!(result, 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex64_vec_axpy() {
        let alpha = Complex64::new(2.0, 1.0);
        let x = vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, 1.0)];
        let mut y = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let mut y_ref = y.clone();
        for (xi, yi) in x.iter().zip(y_ref.iter_mut()) {
            *yi += alpha * *xi;
        }
        Complex64::vec_axpy(alpha, &x, &mut y);
        for (a, b) in y.iter().zip(y_ref.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_complex64_vec_scale() {
        let alpha = Complex64::new(0.0, 1.0);
        let mut x = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        Complex64::vec_scale(&mut x, alpha);
        assert_relative_eq!(x[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(x[0].im, 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1].re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn normwise_backward_error_preserves_tiny_physical_scale() {
        let residual = 2.0e-30;
        let action = 3.0e-30;
        let rhs = 4.0e-30;
        let error = f64::normwise_backward_error(residual, action, rhs);
        assert!((error - (2.0 / 7.0)).abs() < 1.0e-14);
        assert!(f64::normwise_backward_error(1.0, 0.0, 0.0).is_finite());
    }

    /// U12 unit-lower solve: `||L·X − B||` residual is tiny and the stored
    /// diagonal is never read (filled with NaN — any read poisons the
    /// result). Covers the f64 and Complex64 TRSM-kernel overrides (f16
    /// keeps the generic AXPY default).
    fn check_trsm_u12_residual<L: ComplexField<Real = f64>>(
        ncols: usize,
        solve: impl Fn(&[L], usize, &mut [L]),
    ) {
        let panel = 6;
        let mut l11 = vec![L::zero(); panel * panel];
        for i in 0..panel {
            for k in 0..=i {
                l11[i * panel + k] = if i == k {
                    L::zero() / L::zero() // NaN diagonal: must never be read
                } else {
                    L::from_real(0.1 * ((i * 7 + k * 3) % 5) as f64 - 0.2)
                };
            }
        }
        let b: Vec<L> = (0..panel * ncols)
            .map(|t| L::from_real(((t * 13) % 7) as f64 * 0.3 - 0.5))
            .collect();
        let mut x = b.clone();
        solve(&l11, panel, &mut x);
        for i in 0..panel {
            for j in 0..ncols {
                let mut s = x[i * ncols + j];
                for k in 0..i {
                    s = s + l11[i * panel + k] * x[k * ncols + j];
                }
                let diff = (s - b[i * ncols + j]).norm_sqr().sqrt();
                assert!(diff < 1e-11, "i={i} j={j} residual {diff}");
            }
        }
    }

    #[test]
    fn trsm_u12_unit_lower_f64() {
        // Wide enough to take the parallel band path, like the LU U12 use.
        check_trsm_u12_residual::<f64>(130, |l, lda, x| {
            f64::trsm_u12_unit_lower(l, lda, 6, x, 130, 130)
        });
    }

    #[test]
    fn trsm_u12_unit_lower_complex64() {
        // Wide enough to take the parallel band path, like the LU U12 use.
        check_trsm_u12_residual::<Complex64>(130, |l, lda, x| {
            Complex64::trsm_u12_unit_lower(l, lda, 6, x, 130, 130)
        });
    }

    #[test]
    fn trsm_u12_unit_lower_complex32() {
        // f32 twin of the residual contract (looser tolerance).
        let panel = 6usize;
        let ncols = 130usize;
        let mut l11 = vec![Complex32::zero(); panel * panel];
        for i in 0..panel {
            for k in 0..=i {
                l11[i * panel + k] = if i == k {
                    Complex32::zero() / Complex32::zero() // NaN diagonal
                } else {
                    Complex32::from_real(0.1 * ((i * 7 + k * 3) % 5) as f32 - 0.2)
                };
            }
        }
        let b: Vec<Complex32> = (0..panel * ncols)
            .map(|t| Complex32::from_real(((t * 13) % 7) as f32 * 0.3 - 0.5))
            .collect();
        let mut x = b.clone();
        Complex32::trsm_u12_unit_lower(&l11, panel, panel, &mut x, ncols, ncols);
        for i in 0..panel {
            for j in 0..ncols {
                let mut s = x[i * ncols + j];
                for k in 0..i {
                    s = s + l11[i * panel + k] * x[k * ncols + j];
                }
                let diff = (s - b[i * ncols + j]).norm_sqr().sqrt();
                assert!(diff < 1e-4, "i={i} j={j} residual {diff}");
            }
        }
    }
}
