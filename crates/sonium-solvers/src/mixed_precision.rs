//! Mixed-precision inner/outer iteration support
//!
//! The outer Krylov solver (FGMRES) runs in f64/Complex64 for accuracy.
//! The inner preconditioner runs in f32/Complex32 for ~2x bandwidth savings.
//! This is implemented as a "precision-casting preconditioner" wrapper.
//!
//! # Usage
//!
//! ```ignore
//! use sonium_solvers::{CsrMatrix, DiagonalPreconditioner};
//! use sonium_solvers::mixed_precision::{MixedPrecisionPreconditioner, mixed_precision_solve};
//! use sonium_solvers::FgmresConfig;
//! use num_complex::Complex64;
//!
//! // Build high-precision matrix
//! let matrix_f64: CsrMatrix<f64> = /* ... */;
//!
//! // Cast to low precision and build preconditioner there
//! let matrix_f32: CsrMatrix<f32> = matrix_f64.cast::<f32>();
//! let precond_f32 = DiagonalPreconditioner::from_csr(&matrix_f32);
//!
//! // Solve: outer iteration in f64, inner preconditioner in f32
//! let config = FgmresConfig::default();
//! let solution = mixed_precision_solve(&matrix_f64, &b, Box::new(precond_f32), &config);
//! ```
//!
//! # GMRES-based iterative refinement (Carson–Higham GMRES-IR)
//!
//! [`gmres_ir_f32`] / [`gmres_ir_complex32`] (and the generic [`gmres_ir_lu`]
//! / [`gmres_ir_ilu`]) implement *true* GMRES-IR rather than a plain
//! precision-casting wrapper: A itself is cast down to f32/Complex32 and
//! **factorized** there (sparse LU or ILU), and the low-precision factor is
//! used as a flexible preconditioner inside an f64/Complex64 FGMRES.
//!
//! Convergence theory (Carson & Higham, SIAM J. Sci. Comput. 2017/2018):
//! classical stationary iterative refinement with an LU factorization
//! computed in precision u_low converges only when `cond(A)·u_low < 1`.
//! Replacing the inner solve by GMRES preconditioned with the low-precision
//! factor relaxes this by orders of magnitude: the preconditioned matrix
//! satisfies `M·A ≈ I + O(cond(A)·u_low)`, so GMRES-IR converges whenever
//! `cond(A) ≲ u_low⁻¹` (and in practice well beyond, up to
//! `cond(A) ≲ u_low⁻¹·u_high⁻¹`), while the attainable forward/backward
//! error stays at the working-precision level `u_high`. With f32 factors
//! (u_low ≈ 6e-8) and an f64 outer solve, cond(A) up to ~1e7 is tolerated
//! and the final residual reaches f64 accuracy — with only a handful of
//! outer iterations because the low-precision factor is a near-exact
//! inverse of A.
//!
//! # Matrix-side scaling for narrow-range factors (f16)
//!
//! The vector side of the f16 path already applies Higham–Pranesh max-norm
//! scaling (see [`PrecisionCast::down_cast_max_abs`]). The *matrix* cast
//! needs the analogous treatment: a raw `cast::<f16>()` of a matrix with
//! entries outside f16's usable range (~[6e-5, 65504]) overflows to ±inf
//! or underflows to zero at factorization time, silently producing a
//! garbage factor. [`gmres_ir_lu`] / [`gmres_ir_ilu`] therefore route the
//! cast through an equilibrating wrapper which — only when the target
//! precision opts in via `down_cast_max_abs` — first applies a two-sided
//! max-abs equilibration `Dr·A·Dc` plus a power-of-two scalar `s` in the
//! high precision, so the cast matrix `B = s·Dr·A·Dc` has `max|B| ≈ 2^13`
//! (comfortably inside f16 range with headroom, and with the underflow
//! floor pushed down to ~7e-12 relative to the max).
//!
//! The wrap is preconditioner-safe by exact cancellation: the apply
//! computes `M⁻¹r = s·Dc·B⁻¹·(Dr·r)`, and in exact arithmetic
//! `s·Dc·(s·Dr·A·Dc)⁻¹·Dr = A⁻¹` — the scalings cancel identically, so the
//! preconditioned operator is the same near-identity as without scaling.
//! In low-precision arithmetic the only error is the factorization error
//! of `B`, which is *smaller* than that of a raw cast: equilibration
//! removes artificial ill-conditioning from bad row/column scaling, so the
//! Carson–Higham condition becomes `cond(s·Dr·A·Dc)·u_low ≲ 1` with the
//! equilibrated (typically much smaller) condition number. The f32 and
//! Complex32 paths return `None` from `down_cast_max_abs` and keep the
//! raw, bit-identical cast.

use crate::direct::{SparseLuError, SparseLuFactorization};
#[allow(deprecated)]
use crate::iterative::{
    BlockFgmresConfig, BlockFgmresSolution, DeflationSubspace, FgmresConfig, FgmresSolution,
    GcroDrConfig, GcroDrSubspace, GmresConfig, GmresSolution, KrylovOpCounts, RecycledSubspace,
    try_block_fgmres_preconditioned, try_fgmres_preconditioned_with_abort,
    try_gcro_dr_preconditioned, try_gmres_deflated_preconditioned,
    try_gmres_recycled_preconditioned,
};
use crate::preconditioners::{IluConfig, IluPreconditioner};
use crate::sparse::CsrMatrix;
use crate::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
#[cfg(test)]
use half::f16;
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Trait for casting between precision levels.
///
/// Enables conversion of scalar values between high-precision (e.g. f64)
/// and low-precision (e.g. f32) representations.
pub use sonium_solvers_core::PrecisionCast;
#[cfg(test)]
use sonium_solvers_core::precision::F16_SCALE_TARGET;

/// Two-sided scaling applied to the matrix *before* the low-precision cast
/// in GMRES-IR (see the module docs). The factorized matrix is
/// `B = scalar · Dr · A · Dc`; the preconditioner apply compensates with
/// `M⁻¹r = scalar · Dc · B⁻¹ · (Dr · r)`, which cancels to `A⁻¹r` in exact
/// arithmetic, so the scaled factor is a valid preconditioner for the
/// original (unscaled) operator.
#[derive(Debug, Clone)]
struct FactorizationScaling<R> {
    /// Row scaling `Dr`, applied to the residual before casting down.
    row_scale: Vec<R>,
    /// Column scaling `Dc`, applied to the preconditioner output after
    /// casting back up.
    col_scale: Vec<R>,
    /// Power-of-two scalar (`F16_SCALE_TARGET`) pushing the equilibrated
    /// matrix (max-abs 1) into the f16 sweet spot and pushing the relative
    /// underflow floor down to ~7e-12.
    scalar: R,
}

/// Cast `matrix` down to `TLow` for a low-precision *factorization*,
/// equilibrating first when `TLow` has a narrow exponent range.
///
/// Wide-range targets (f32, Complex32 — `down_cast_max_abs() == None`) get
/// the raw, bit-identical [`CsrMatrix::cast`]. Narrow-range targets (f16)
/// get a two-sided max-abs equilibration computed in `THigh`: each row is
/// scaled by the inverse of its max modulus, then each column of the
/// row-scaled matrix by the inverse of its max modulus (the SuperLU `equil`
/// strategy, without the MC64 matching — the factor is only used as a
/// preconditioner, so pivoting-friendliness of the diagonal is not
/// critical), followed by a power-of-two scalar mapping the resulting
/// max-abs-1 matrix to [`F16_SCALE_TARGET`]. This keeps every significant
/// entry inside the f16 range regardless of how badly scaled `A` is.
///
/// Returns the scaling so the caller can wrap the preconditioner apply
/// (see [`FactorizationScaling`]).
fn cast_for_factorization<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
) -> (CsrMatrix<TLow>, Option<FactorizationScaling<THigh::Real>>)
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    let Some(target) = THigh::down_cast_max_abs() else {
        // Wide-range target: raw cast, bit-identical to `matrix.cast()`.
        return (matrix.cast::<TLow>(), None);
    };

    let one = THigh::Real::one();
    let zero = THigh::Real::zero();

    // Row max-abs scaling.
    let mut row_scale = vec![one; matrix.num_rows];
    for (row, slot) in row_scale.iter_mut().enumerate() {
        let mut row_max = zero;
        for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            row_max = row_max.max(matrix.values[idx].norm());
        }
        if row_max > zero {
            *slot = one / row_max;
        }
    }

    // Column max-abs scaling of Dr · A (accumulate maxima, then invert).
    let mut col_scale = vec![zero; matrix.num_cols];
    for (row, &scale) in row_scale.iter().enumerate() {
        for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let j = matrix.col_indices[idx];
            col_scale[j] = col_scale[j].max(scale * matrix.values[idx].norm());
        }
    }
    for scale in &mut col_scale {
        *scale = if *scale > zero { one / *scale } else { one };
    }

    // Cast B = target · Dr · A · Dc down to TLow; structure is preserved.
    let mut values_low = Vec::with_capacity(matrix.values.len());
    for (row, &scale) in row_scale.iter().enumerate() {
        for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let j = matrix.col_indices[idx];
            let s = scale * col_scale[j] * target;
            values_low.push(THigh::cast_down(matrix.values[idx] * THigh::from_real(s)));
        }
    }

    let scaled = CsrMatrix::from_raw_parts(
        matrix.num_rows,
        matrix.num_cols,
        matrix.row_ptrs.clone(),
        matrix.col_indices.clone(),
        values_low,
    );
    let scaling = FactorizationScaling {
        row_scale,
        col_scale,
        scalar: target,
    };
    (scaled, Some(scaling))
}

/// A preconditioner that casts down to lower precision, applies a wrapped
/// preconditioner, and casts back up. This gives ~2x bandwidth savings
/// on the preconditioner apply (which is often 70%+ of solve time).
///
/// When `TLow` has a narrow exponent range (f16), the input vector is
/// normalized by its inf-norm before casting down and the result is
/// rescaled after casting back up (see
/// [`PrecisionCast::down_cast_max_abs`]). This is exact because the
/// wrapped preconditioner is a linear operator: `P(αr) = αP(r)`.
pub struct MixedPrecisionPreconditioner<THigh, TLow>
where
    THigh: ComplexField,
    TLow: ComplexField,
{
    preconditioner: Box<dyn Preconditioner<TLow> + Send + Sync>,
    buffers: Mutex<Vec<(Vec<TLow>, Vec<TLow>)>>,
    /// Matrix-side equilibration to undo around the apply (GMRES-IR with
    /// narrow-range factors); `None` for plain precision casting.
    matrix_scaling: Option<FactorizationScaling<THigh::Real>>,
    /// Set when an intermediate low-precision cast or factor application
    /// produces a non-finite value. The `Preconditioner` trait is
    /// intentionally infallible for hot-path compatibility, so checked
    /// mixed-precision entry points inspect this sticky flag after the
    /// Krylov call and turn the failure into a typed error.
    low_precision_failure: AtomicBool,
    /// Number of low-precision boundary failures observed by the wrapper.
    ///
    /// The boolean remains the cheap sticky fast-path used by the Krylov
    /// loop; this counter preserves enough provenance for checked callers to
    /// distinguish a first failure from repeated failed applications.
    low_precision_failure_count: AtomicUsize,
    _phantom: PhantomData<THigh>,
}

impl<THigh, TLow> MixedPrecisionPreconditioner<THigh, TLow>
where
    THigh: ComplexField,
    TLow: ComplexField,
{
    /// Create a new mixed-precision preconditioner wrapping a low-precision preconditioner
    pub fn new(preconditioner: Box<dyn Preconditioner<TLow> + Send + Sync>) -> Self {
        Self {
            preconditioner,
            buffers: Mutex::new(Vec::new()),
            matrix_scaling: None,
            low_precision_failure: AtomicBool::new(false),
            low_precision_failure_count: AtomicUsize::new(0),
            _phantom: PhantomData,
        }
    }

    /// Wrap a low-precision preconditioner built from the equilibrated
    /// matrix `B = scalar · Dr · A · Dc` produced by
    /// [`cast_for_factorization`]. The apply becomes
    /// `M⁻¹r = scalar · Dc · B⁻¹ · (Dr · r)`, which equals `A⁻¹r` in exact
    /// arithmetic — the equilibration cancels identically, so this is a
    /// valid (and, for badly scaled A, much better) preconditioner for the
    /// original operator.
    fn with_matrix_scaling(
        preconditioner: Box<dyn Preconditioner<TLow> + Send + Sync>,
        matrix_scaling: Option<FactorizationScaling<THigh::Real>>,
    ) -> Self {
        Self {
            preconditioner,
            buffers: Mutex::new(Vec::new()),
            matrix_scaling,
            low_precision_failure: AtomicBool::new(false),
            low_precision_failure_count: AtomicUsize::new(0),
            _phantom: PhantomData,
        }
    }

    fn has_low_precision_failure(&self) -> bool {
        self.low_precision_failure.load(Ordering::Relaxed)
    }

    /// Return the number of failed low-precision boundary applications.
    pub fn low_precision_failure_count(&self) -> usize {
        self.low_precision_failure_count.load(Ordering::Relaxed)
    }

    fn mark_low_precision_failure(&self) {
        self.low_precision_failure.store(true, Ordering::Relaxed);
        self.low_precision_failure_count
            .fetch_add(1, Ordering::Relaxed);
    }

    fn adjoint_capability(&self, hermitian: bool) -> AdjointCapability {
        if hermitian {
            self.preconditioner.hermitian_capability()
        } else {
            self.preconditioner.transpose_capability()
        }
    }

    fn apply_adjoint_checked(&self, r: &[THigh], hermitian: bool) -> Result<Vec<THigh>, SolverError>
    where
        THigh: PrecisionCast<TLow>,
    {
        let operation = if hermitian {
            "mixed-precision preconditioner Hermitian"
        } else {
            "mixed-precision preconditioner transpose"
        };
        if r.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "input contains NaN or infinity".to_string(),
            });
        }

        let n = r.len();
        let mut r_eff = r.to_vec();
        if let Some(scaling) = &self.matrix_scaling {
            if scaling.row_scale.len() != n || scaling.col_scale.len() != n {
                return Err(SolverError::DimensionMismatch {
                    expected: n,
                    got: scaling.row_scale.len().max(scaling.col_scale.len()),
                });
            }
            // Forward scaling is s·D_c·B⁻¹·D_r. Its adjoint therefore
            // applies (s·D_c) to the input and D_r to the output.
            for (value, &column_scale) in r_eff.iter_mut().zip(&scaling.col_scale) {
                *value *= THigh::from_real(scaling.scalar * column_scale);
            }
        }

        let mut low_input = Vec::with_capacity(n);
        let mut inverse_cast_scale = None;
        if let Some(target) = THigh::down_cast_max_abs() {
            let inf_norm = r_eff
                .iter()
                .map(|value| value.norm())
                .fold(THigh::Real::zero(), |a, b| a.max(b));
            if inf_norm > THigh::Real::zero() {
                let scale = THigh::from_real(target / inf_norm);
                let inverse = THigh::from_real(inf_norm / target);
                low_input.extend(r_eff.iter().map(|value| THigh::cast_down(*value * scale)));
                inverse_cast_scale = Some(inverse);
            } else {
                low_input.resize(n, TLow::zero());
            }
        } else {
            low_input.extend(r_eff.iter().map(|value| THigh::cast_down(*value)));
        }
        if low_input.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "input became non-finite at {} precision boundary",
                    std::any::type_name::<TLow>()
                ),
            });
        }

        let low_output = if hermitian {
            self.preconditioner.try_apply_hermitian(&low_input)?
        } else {
            self.preconditioner.try_apply_transpose(&low_input)?
        };
        if low_output.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: low_output.len(),
            });
        }
        if low_output.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "adjoint action generated a non-finite value in {} precision",
                    std::any::type_name::<TLow>()
                ),
            });
        }

        let mut output: Vec<THigh> = low_output.into_iter().map(THigh::cast_up).collect();
        if let Some(inverse) = inverse_cast_scale {
            for value in &mut output {
                *value *= inverse;
            }
        }
        if let Some(scaling) = &self.matrix_scaling {
            for (value, &row_scale) in output.iter_mut().zip(&scaling.row_scale) {
                *value *= THigh::from_real(row_scale);
            }
        }
        if output.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "adjoint action generated NaN or infinity".to_string(),
            });
        }
        Ok(output)
    }
}

/// Result of casting a matrix for a low-precision factorization, including
/// any scaling that must be restored in high precision.
type FactorizationCastResult<THigh, TLow> = Result<
    (
        CsrMatrix<TLow>,
        Option<FactorizationScaling<<THigh as ComplexField>::Real>>,
    ),
    SolverError,
>;

/// Checked counterpart to [`cast_for_factorization`].
///
/// A finite high-precision matrix can still become non-finite when a wide
/// range value is cast to a narrower factor type (for example `f64::MAX` to
/// `f32`).  Letting that value reach a factorizer hides the actual failure
/// stage behind a generic pivot/factorization error.  Checked GMRES-IR entry
/// points reject the cast at the boundary and retain the low-precision type
/// in the diagnostic.
fn checked_cast_for_factorization<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    operation: &'static str,
) -> FactorizationCastResult<THigh, TLow>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    let (matrix_low, scaling) = cast_for_factorization::<THigh, TLow>(matrix);
    if let Some(index) = matrix_low
        .values
        .iter()
        .position(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "matrix entry {index} became non-finite when cast to {}",
                std::any::type_name::<TLow>()
            ),
        });
    }
    Ok((matrix_low, scaling))
}

/// Adapter used by the checked block-FGMRES entry point.
///
/// Block FGMRES predates the scalar checked abort hook, so its checked
/// boundary contains callback panics and maps them to `SolverError`. A
/// mixed-precision boundary failure must stop before the block kernel applies
/// the high-precision operator to the failed correction; this adapter uses
/// that contained panic path after preserving the sticky failure provenance
/// on the wrapped preconditioner.
struct AbortOnLowPrecisionFailure<'a, THigh, TLow>
where
    THigh: ComplexField,
    TLow: ComplexField,
{
    inner: &'a MixedPrecisionPreconditioner<THigh, TLow>,
}

impl<THigh, TLow> Preconditioner<THigh> for AbortOnLowPrecisionFailure<'_, THigh, TLow>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    fn apply_into(&self, r: &[THigh], y: &mut [THigh]) {
        self.inner.apply_into(r, y);
        if self.inner.has_low_precision_failure() {
            panic!("mixed-precision preconditioner requested early termination");
        }
    }
}

impl<THigh, TLow> Preconditioner<THigh> for MixedPrecisionPreconditioner<THigh, TLow>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    fn apply(&self, r: &[THigh]) -> Vec<THigh> {
        let mut y = vec![THigh::zero(); r.len()];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[THigh], y: &mut [THigh]) {
        let n = r.len();

        if y.len() != n || r.iter().any(|value| !value.norm().is_finite()) {
            self.mark_low_precision_failure();
            y.fill(THigh::zero());
            return;
        }

        // Grab pre-allocated working arrays from pool, or formulate new ones
        let mut bufs = {
            let mut pool = match self.buffers.lock() {
                Ok(pool) => pool,
                Err(poisoned) => poisoned.into_inner(),
            };
            pool.pop()
                .unwrap_or_else(|| (vec![TLow::zero(); n], vec![TLow::zero(); n]))
        };

        if bufs.0.len() != n {
            bufs = (vec![TLow::zero(); n], vec![TLow::zero(); n]);
        }

        // 0. Matrix-side equilibration (GMRES-IR with narrow-range
        // factors): the low-precision factor inverts
        // B = scalar · Dr · A · Dc, so the residual must be premultiplied
        // by Dr before the cast. Done in high precision; skipped entirely
        // on the plain precision-casting path.
        let r_scaled_storage;
        let r_eff: &[THigh] = if let Some(ms) = &self.matrix_scaling {
            debug_assert_eq!(ms.row_scale.len(), n);
            r_scaled_storage = r
                .iter()
                .zip(ms.row_scale.iter())
                .map(|(&ri, &d)| ri * THigh::from_real(d))
                .collect::<Vec<_>>();
            &r_scaled_storage
        } else {
            r
        };

        // 1. Cast input down to low precision, directly into bufs.0.
        //
        // Narrow-range low-precision types (f16) require max-norm
        // scaling (Higham & Pranesh): raw residuals outside [6e-5, 65504]
        // would overflow to ±inf or underflow to zero and silently poison
        // the low-precision apply. Leave square-root-n headroom for the
        // forward/back substitutions: a finite triangular solve can
        // accumulate many scaled terms even when its input fits in f16,
        // while a linear n-fold reduction would discard useful f16
        // subnormal residual components. Undo the scale after casting back
        // up; the operation remains exact in the high-precision
        // preconditioner wrapper. For
        // f32/Complex32 no scaling is needed and the path below is unchanged.
        let scaling = THigh::down_cast_max_abs().and_then(|target| {
            let inf_norm = r_eff
                .iter()
                .map(|v| v.norm())
                .fold(THigh::Real::zero(), |a, b| a.max(b));
            let triangular_headroom = THigh::Real::from_f64(n.max(1) as f64)
                .unwrap_or(THigh::Real::one())
                .sqrt();
            (inf_norm > THigh::Real::zero()).then(|| {
                (
                    THigh::from_real(target / (inf_norm * triangular_headroom)),
                    THigh::from_real((inf_norm * triangular_headroom) / target),
                )
            })
        });

        if let Some((mut scale, mut inv_scale)) = scaling {
            // A triangular factor can have substantially more growth than
            // the dimension-based headroom predicts. Since the wrapped
            // action is linear, retrying with an additional power-of-two
            // scale preserves the high-precision result while keeping f16
            // intermediates finite.
            let retry_factor = THigh::from_real(THigh::Real::from_f64(8.0).unwrap());
            let mut applied = false;
            for _ in 0..8 {
                for (ri, b0) in r_eff.iter().zip(bufs.0.iter_mut()) {
                    *b0 = THigh::cast_down(*ri * scale);
                }
                bufs.1.fill(TLow::zero());
                let attempt = catch_unwind(AssertUnwindSafe(|| {
                    self.preconditioner.apply_into(&bufs.0, &mut bufs.1);
                }));
                if attempt.is_ok() && bufs.1.iter().all(|value| value.norm().is_finite()) {
                    applied = true;
                    break;
                }
                scale /= retry_factor;
                inv_scale *= retry_factor;
            }

            if applied {
                // 3. Cast result back up, undoing the inf-norm scaling.
                for (b1, yi) in bufs.1.iter().zip(y.iter_mut()) {
                    *yi = THigh::cast_up(*b1) * inv_scale;
                }
            } else {
                self.mark_low_precision_failure();
                y.fill(THigh::zero());
            }
        } else {
            for (ri, b0) in r_eff.iter().zip(bufs.0.iter_mut()) {
                *b0 = THigh::cast_down(*ri);
            }

            // 2. Apply inner preconditioner in low precision, into bufs.1
            self.preconditioner.apply_into(&bufs.0, &mut bufs.1);

            // 3. Cast result back up to high precision, directly into y
            for (b1, yi) in bufs.1.iter().zip(y.iter_mut()) {
                *yi = THigh::cast_up(*b1);
            }
        }

        // A finite high-precision residual can overflow when it is cast to
        // the low-precision factor type. Check both working buffers after
        // the wrapped factor application so a preconditioner that sanitizes
        // an infinite input cannot hide the precision-boundary failure.
        if bufs.0.iter().any(|value| !value.norm().is_finite())
            || bufs.1.iter().any(|value| !value.norm().is_finite())
        {
            self.mark_low_precision_failure();
            y.fill(THigh::zero());
        }

        // 4. Undo the matrix-side equilibration:
        // M⁻¹r = scalar · Dc · B⁻¹ · (Dr · r). Multiplied in high
        // precision; the scalar is a power of two so this introduces no
        // rounding beyond the Dc product itself.
        if let Some(ms) = &self.matrix_scaling {
            debug_assert_eq!(ms.col_scale.len(), y.len());
            for (yi, &dc) in y.iter_mut().zip(ms.col_scale.iter()) {
                *yi *= THigh::from_real(ms.scalar * dc);
            }
        }

        if y.iter().any(|value| !value.norm().is_finite()) {
            self.mark_low_precision_failure();
            y.fill(THigh::zero());
        }

        // Return working arrays to the concurrent pool
        match self.buffers.lock() {
            Ok(mut pool) => pool.push(bufs),
            Err(poisoned) => poisoned.into_inner().push(bufs),
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.adjoint_capability(false)
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.adjoint_capability(true)
    }

    fn try_apply_transpose(&self, r: &[THigh]) -> Result<Vec<THigh>, SolverError> {
        if self.transpose_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "mixed-precision preconditioner transpose",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        self.apply_adjoint_checked(r, false)
    }

    fn try_apply_hermitian(&self, r: &[THigh]) -> Result<Vec<THigh>, SolverError> {
        if self.hermitian_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "mixed-precision preconditioner Hermitian",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        self.apply_adjoint_checked(r, true)
    }

    fn apply_transpose(&self, r: &[THigh]) -> Vec<THigh> {
        self.try_apply_transpose(r)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn apply_hermitian(&self, r: &[THigh]) -> Vec<THigh> {
        self.try_apply_hermitian(r)
            .unwrap_or_else(|error| panic!("{error}"))
    }
}

/// Convenience function: solve a linear system using mixed-precision FGMRES.
///
/// The outer iteration (FGMRES) runs in `THigh` precision for accuracy.
/// The preconditioner runs in `TLow` precision for ~2x bandwidth savings.
///
/// # Arguments
///
/// * `operator` - The linear operator (matrix) in high precision
/// * `b` - The right-hand side vector in high precision
/// * `precond` - A preconditioner operating in low precision
/// * `config` - FGMRES configuration (tolerances, restart, etc.)
///
/// # Returns
///
/// The FGMRES solution in high precision
pub fn mixed_precision_solve<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &FgmresConfig<THigh::Real>,
) -> FgmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    mixed_precision_solve_with_guess(operator, b, None, precond, config)
}

/// Infallible compatibility wrapper for mixed-precision FGMRES with an
/// optional high-precision initial iterate.
pub fn mixed_precision_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &FgmresConfig<THigh::Real>,
) -> FgmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    // Keep the historical infallible result shape, but use the checked
    // boundary so a low-precision callback panic or non-finite correction is
    // reported as NumericalError instead of escaping from the wrapper.
    match try_mixed_precision_solve_with_guess(operator, b, initial_guess, precond, config) {
        Ok(result) => result,
        Err(_) => numerical_error_solution(b.len()),
    }
}

/// Solve several right-hand sides with mixed-precision block FGMRES.
///
/// The outer block Krylov iteration remains in `THigh`; only the supplied
/// preconditioner is evaluated in `TLow`.  This is the block analogue of
/// [`mixed_precision_solve`] and keeps the true-residual checks from the
/// checked block FGMRES path.
pub fn mixed_precision_block_solve<THigh, TLow, A>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &BlockFgmresConfig<THigh::Real>,
) -> BlockFgmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    mixed_precision_block_solve_with_guess(operator, b, None, precond, config)
}

/// Infallible mixed-precision block FGMRES with an optional high-precision
/// initial block iterate.
pub fn mixed_precision_block_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<THigh>,
    initial_guess: Option<&sonium_solvers_core::dense::DenseMatrix<THigh>>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &BlockFgmresConfig<THigh::Real>,
) -> BlockFgmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    match try_mixed_precision_block_solve_with_guess(operator, b, initial_guess, precond, config) {
        Ok(result) => result,
        Err(_) => numerical_error_block_solution(b.nrows(), b.ncols()),
    }
}

fn numerical_error_solution<T: ComplexField>(len: usize) -> FgmresSolution<T> {
    FgmresSolution {
        x: vec![T::zero(); len],
        iterations: 0,
        restarts: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts: Default::default(),
    }
}

fn numerical_error_gmres_solution<T: ComplexField>(len: usize) -> GmresSolution<T> {
    GmresSolution {
        x: vec![T::zero(); len],
        iterations: 0,
        restarts: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts: KrylovOpCounts::default(),
    }
}

fn numerical_error_block_solution<T: ComplexField>(
    nrows: usize,
    ncols: usize,
) -> BlockFgmresSolution<T> {
    BlockFgmresSolution {
        x: sonium_solvers_core::dense::DenseMatrix::from_elem(nrows, ncols, T::zero()),
        iterations: 0,
        residuals: vec![T::Real::infinity(); ncols],
        true_residuals: vec![T::Real::infinity(); ncols],
        backward_errors: vec![T::Real::infinity(); ncols],
        attainable_accuracy: vec![AccuracyStatus::NotChecked; ncols],
        converged: false,
        op_counts: Default::default(),
    }
}

/// Mixed-precision factors can make the projected Krylov residual look
/// converged before the high-precision operator agrees. Force at least one
/// original-operator candidate check per Krylov step for GMRES-IR wrappers;
/// this preserves a caller's stricter cadence while making the default
/// final-only policy safe for low precision.
fn mixed_precision_config<R: Clone>(config: &FgmresConfig<R>) -> FgmresConfig<R> {
    let mut checked = config.clone();
    checked.exact_residual_check_interval = checked.exact_residual_check_interval.max(1);
    checked
}

/// Deflated and recycled GMRES use the same projected residual admission
/// mechanism as scalar GMRES.  A low-precision preconditioner can make that
/// estimate optimistic, so mixed-precision wrappers always perform exact
/// high-precision candidate checks while preserving any stricter cadence.
fn mixed_precision_gmres_config<R: Clone>(config: &GmresConfig<R>) -> GmresConfig<R> {
    let mut checked = config.clone();
    checked.exact_residual_check_interval = checked.exact_residual_check_interval.max(1);
    checked
}

/// Enforce the promoted-result contract shared by mixed-precision GMRES
/// variants.  The underlying method owns its recurrence and exact-residual
/// scheduling, but the facade remains the final authority on what it exposes:
/// all diagnostics must be finite and `Converged` must imply the promoted
/// residual is within the caller's tolerance.
fn normalize_mixed_precision_gmres_result<T: ComplexField>(
    mut result: GmresSolution<T>,
    tolerance: T::Real,
    operation: &'static str,
) -> Result<GmresSolution<T>, SolverError> {
    if result.x.iter().any(|value| !value.norm().is_finite())
        || !result.residual.is_finite()
        || !result.estimated_residual.is_finite()
        || !result.backward_error.is_finite()
    {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "mixed-precision result metrics contain NaN or infinity".to_string(),
        });
    }

    let inconsistent_convergence = (result.status == SolverStatus::Converged
        && (!result.converged || result.residual > tolerance))
        || (result.converged && result.status != SolverStatus::Converged);
    if inconsistent_convergence {
        result.converged = false;
        if result.status == SolverStatus::Converged {
            result.status = SolverStatus::MaxIterationsReached;
        }
        result.attainable_accuracy = AccuracyStatus::NotAttained;
    }
    Ok(result)
}

/// Mixed-precision block solves always gate convergence on the original
/// operator's true residual. A low-precision cast can underflow a residual
/// while leaving the high-precision residual non-zero.
fn mixed_precision_block_config<R: Clone>(config: &BlockFgmresConfig<R>) -> BlockFgmresConfig<R> {
    let mut checked = config.clone();
    checked.true_residual_check = true;
    checked.exact_residual_check_interval = checked.exact_residual_check_interval.max(1);
    checked
}

#[derive(Debug, Clone, Copy)]
struct MixedPrecisionCheckpointMetrics<R: Float> {
    iterations: usize,
    restarts: usize,
    residual: R,
    estimated_residual: R,
    backward_error: R,
    attainable_accuracy: AccuracyStatus,
    converged: bool,
    status: SolverStatus,
    op_counts: KrylovOpCounts,
}

/// A checked mixed-precision FGMRES continuation checkpoint.
///
/// The checkpoint stores the latest high-precision iterate and, when captured
/// from a result, its status/accuracy diagnostics and operation counters. It
/// deliberately does not store the low-precision preconditioner: callers
/// provide a fresh low-precision preconditioner on resume so factor lifetime,
/// matrix compatibility, and device ownership remain explicit.
#[derive(Debug, Clone)]
pub struct MixedPrecisionCheckpoint<T: ComplexField> {
    solution: Vec<T>,
    metrics: Option<MixedPrecisionCheckpointMetrics<T::Real>>,
}

/// Version of the portable mixed-precision checkpoint envelope.
pub const MIXED_PRECISION_CHECKPOINT_FORMAT_VERSION: u16 = 1;
/// Magic prefix for [`MixedPrecisionCheckpoint`] byte packets.
pub const MIXED_PRECISION_CHECKPOINT_MAGIC: &[u8; 8] = b"SMPCP001";
/// Version of the portable mixed-precision block checkpoint envelope.
pub const MIXED_PRECISION_BLOCK_CHECKPOINT_FORMAT_VERSION: u16 = 1;
/// Magic prefix for [`MixedPrecisionBlockCheckpoint`] byte packets.
pub const MIXED_PRECISION_BLOCK_CHECKPOINT_MAGIC: &[u8; 8] = b"SMPB001!";

impl<T: ComplexField> MixedPrecisionCheckpoint<T> {
    /// Capture a finite result after validating the shared result contract.
    pub fn from_solution(solution: &FgmresSolution<T>) -> Result<Self, SolverError> {
        if let Some(index) = solution
            .x
            .iter()
            .position(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "mixed-precision checkpoint",
                reason: format!("solution contains NaN or infinity at index {index}"),
            });
        }
        solution.validate_contract()?;
        Ok(Self {
            solution: solution.x.clone(),
            metrics: Some(MixedPrecisionCheckpointMetrics {
                iterations: solution.iterations,
                restarts: solution.restarts,
                residual: solution.residual,
                estimated_residual: solution.estimated_residual,
                backward_error: solution.backward_error,
                attainable_accuracy: solution.attainable_accuracy,
                converged: solution.converged,
                status: solution.status,
                op_counts: solution.op_counts,
            }),
        })
    }

    /// Capture a finite caller-provided initial iterate without result
    /// diagnostics. Resume treats its iteration and operation offsets as zero.
    pub fn from_iterate(iterate: &[T]) -> Result<Self, SolverError> {
        if let Some(index) = iterate.iter().position(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "mixed-precision checkpoint",
                reason: format!("iterate contains NaN or infinity at index {index}"),
            });
        }
        Ok(Self {
            solution: iterate.to_vec(),
            metrics: None,
        })
    }

    /// Number of entries expected in a compatible operator and RHS.
    pub const fn dimension(&self) -> usize {
        self.solution.len()
    }

    /// Borrow the stored high-precision iterate.
    pub fn solution(&self) -> &[T] {
        &self.solution
    }

    /// Number of bytes occupied by the stored iterate.
    pub fn storage_bytes(&self) -> usize {
        self.solution.len().saturating_mul(std::mem::size_of::<T>())
    }

    /// Source iteration count, if captured from a solver result.
    pub fn source_iterations(&self) -> Option<usize> {
        self.metrics.as_ref().map(|metrics| metrics.iterations)
    }

    /// Source restart count, if captured from a solver result.
    pub fn source_restarts(&self) -> Option<usize> {
        self.metrics.map(|metrics| metrics.restarts)
    }

    /// Source residual, if captured from a solver result.
    pub fn source_residual(&self) -> Option<T::Real> {
        self.metrics.map(|metrics| metrics.residual)
    }

    /// Source projected residual, if captured from a solver result.
    pub fn source_estimated_residual(&self) -> Option<T::Real> {
        self.metrics.map(|metrics| metrics.estimated_residual)
    }

    /// Source normwise backward error, if captured from a solver result.
    pub fn source_backward_error(&self) -> Option<T::Real> {
        self.metrics.map(|metrics| metrics.backward_error)
    }

    /// Source attainable-accuracy state, if captured from a solver result.
    pub fn source_accuracy_status(&self) -> Option<AccuracyStatus> {
        self.metrics.map(|metrics| metrics.attainable_accuracy)
    }

    /// Source convergence flag, if captured from a solver result.
    pub fn source_converged(&self) -> Option<bool> {
        self.metrics.map(|metrics| metrics.converged)
    }

    /// Source solver status, if captured from a solver result.
    pub fn source_status(&self) -> Option<SolverStatus> {
        self.metrics.map(|metrics| metrics.status)
    }

    /// Source Krylov operation counters, if captured from a solver result.
    pub fn source_op_counts(&self) -> Option<KrylovOpCounts> {
        self.metrics.as_ref().map(|metrics| metrics.op_counts)
    }
    /// Serialize the iterate and captured result metadata in a strict,
    /// versioned little-endian format.
    ///
    /// Numeric values use canonical f64 real/imaginary pairs. Restoring a
    /// packet requires the same scalar width, real width, and real/complex
    /// family; callers must still provide a fresh low-precision preconditioner
    /// when resuming.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let validated = Self::from_iterate(&self.solution)?;
        let mut output = Vec::new();
        output.extend_from_slice(MIXED_PRECISION_CHECKPOINT_MAGIC);
        output.extend_from_slice(&MIXED_PRECISION_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(if scalar_is_complex::<T>() { 1 } else { 0 });
        output.push(if self.metrics.is_some() { 1 } else { 0 });
        push_mixed_precision_u64(&mut output, validated.solution.len())?;
        push_mixed_precision_u64(&mut output, self.storage_bytes())?;
        for value in &validated.solution {
            let re = value.re().to_f64().ok_or_else(|| {
                mixed_precision_checkpoint_error("real component cannot be represented as f64")
            })?;
            let im = value.im().to_f64().ok_or_else(|| {
                mixed_precision_checkpoint_error("imaginary component cannot be represented as f64")
            })?;
            if !re.is_finite() || !im.is_finite() {
                return Err(mixed_precision_checkpoint_error(
                    "checkpoint contains a non-finite scalar",
                ));
            }
            output.extend_from_slice(&re.to_le_bytes());
            output.extend_from_slice(&im.to_le_bytes());
        }
        if let Some(metrics) = self.metrics {
            push_mixed_precision_u64(&mut output, metrics.iterations)?;
            push_mixed_precision_u64(&mut output, metrics.restarts)?;
            for diagnostic in [
                metrics.residual,
                metrics.estimated_residual,
                metrics.backward_error,
            ] {
                let value = diagnostic.to_f64().ok_or_else(|| {
                    mixed_precision_checkpoint_error("diagnostic cannot be represented as f64")
                })?;
                if !value.is_finite() || value < 0.0 {
                    return Err(mixed_precision_checkpoint_error(
                        "checkpoint diagnostic must be finite and non-negative",
                    ));
                }
                output.extend_from_slice(&value.to_le_bytes());
            }
            output.push(encode_mixed_precision_accuracy(metrics.attainable_accuracy));
            output.push(u8::from(metrics.converged));
            output.push(encode_mixed_precision_status(metrics.status));
            output.push(0);
            let counts = metrics.op_counts;
            for value in [
                counts.spmv_count,
                counts.preconditioner_count,
                counts.dot_reductions,
                counts.norm_count,
                counts.axpy_passes,
                counts.synchronization_points,
                counts.exact_residual_checks,
            ] {
                push_mixed_precision_u64(&mut output, value)?;
            }
            output.extend_from_slice(&counts.basis_bytes_touched.to_le_bytes());
        }
        Ok(output)
    }
    /// Restore a strict portable mixed-precision checkpoint packet.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = MixedPrecisionCheckpointReader { bytes, offset: 0 };
        if reader.take(8)? != MIXED_PRECISION_CHECKPOINT_MAGIC {
            return Err(mixed_precision_checkpoint_error("invalid checkpoint magic"));
        }
        if reader.read_u16()? != MIXED_PRECISION_CHECKPOINT_FORMAT_VERSION {
            return Err(mixed_precision_checkpoint_error(
                "unsupported checkpoint version",
            ));
        }
        let scalar_size = reader.read_u8()?;
        let real_size = reader.read_u8()?;
        let complex = reader.read_u8()?;
        let has_metrics = reader.read_u8()?;
        if scalar_size != std::mem::size_of::<T>() as u8
            || real_size != std::mem::size_of::<T::Real>() as u8
            || complex != u8::from(scalar_is_complex::<T>())
        {
            return Err(mixed_precision_checkpoint_error(
                "checkpoint scalar family does not match target",
            ));
        }
        if has_metrics > 1 {
            return Err(mixed_precision_checkpoint_error(
                "checkpoint metrics flag is invalid",
            ));
        }
        let dimension = reader.read_usize()?;
        let storage_bytes = reader.read_usize()?;
        let expected_bytes = dimension
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| mixed_precision_checkpoint_error("checkpoint storage size overflow"))?;
        if storage_bytes != expected_bytes {
            return Err(mixed_precision_checkpoint_error(
                "checkpoint storage size does not match dimension",
            ));
        }
        let encoded_solution_bytes = dimension
            .checked_mul(std::mem::size_of::<f64>() * 2)
            .ok_or_else(|| mixed_precision_checkpoint_error("checkpoint solution size overflow"))?;
        if encoded_solution_bytes > bytes.len().saturating_sub(reader.offset) {
            return Err(mixed_precision_checkpoint_error("checkpoint is truncated"));
        }
        let mut solution = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let re = T::Real::from_f64(reader.read_f64()?).ok_or_else(|| {
                mixed_precision_checkpoint_error("real component cannot be restored")
            })?;
            let encoded_im = reader.read_f64()?;
            if !scalar_is_complex::<T>() && encoded_im != 0.0 {
                return Err(mixed_precision_checkpoint_error(
                    "real checkpoint contains non-zero imaginary component",
                ));
            }
            let im = T::Real::from_f64(encoded_im).ok_or_else(|| {
                mixed_precision_checkpoint_error("imaginary component cannot be restored")
            })?;
            let value = T::from_re_im(re, im);
            if !value.norm().is_finite() {
                return Err(mixed_precision_checkpoint_error(
                    "checkpoint restores a non-finite scalar",
                ));
            }
            solution.push(value);
        }
        let metrics = if has_metrics == 1 {
            let iterations = reader.read_usize()?;
            let restarts = reader.read_usize()?;
            let residual = T::Real::from_f64(reader.read_f64()?)
                .ok_or_else(|| mixed_precision_checkpoint_error("residual cannot be restored"))?;
            let estimated_residual = T::Real::from_f64(reader.read_f64()?).ok_or_else(|| {
                mixed_precision_checkpoint_error("estimated residual cannot be restored")
            })?;
            let backward_error = T::Real::from_f64(reader.read_f64()?).ok_or_else(|| {
                mixed_precision_checkpoint_error("backward error cannot be restored")
            })?;
            if !residual.is_finite()
                || residual < T::Real::zero()
                || !estimated_residual.is_finite()
                || estimated_residual < T::Real::zero()
                || !backward_error.is_finite()
                || backward_error < T::Real::zero()
            {
                return Err(mixed_precision_checkpoint_error(
                    "checkpoint diagnostics must be finite and non-negative",
                ));
            }
            let attainable_accuracy = decode_mixed_precision_accuracy(reader.read_u8()?)?;
            let converged = match reader.read_u8()? {
                0 => false,
                1 => true,
                _ => {
                    return Err(mixed_precision_checkpoint_error(
                        "checkpoint convergence flag is invalid",
                    ));
                }
            };
            let status = decode_mixed_precision_status(reader.read_u8()?)?;
            if reader.read_u8()? != 0 || converged != status.is_converged() {
                return Err(mixed_precision_checkpoint_error(
                    "checkpoint status metadata is inconsistent",
                ));
            }
            let spmv_count = reader.read_usize()?;
            let preconditioner_count = reader.read_usize()?;
            let dot_reductions = reader.read_usize()?;
            let norm_count = reader.read_usize()?;
            let axpy_passes = reader.read_usize()?;
            let synchronization_points = reader.read_usize()?;
            let exact_residual_checks = reader.read_usize()?;
            let basis_bytes_touched = reader.read_u64()?;
            Some(MixedPrecisionCheckpointMetrics {
                iterations,
                restarts,
                residual,
                estimated_residual,
                backward_error,
                attainable_accuracy,
                converged,
                status,
                op_counts: KrylovOpCounts {
                    spmv_count,
                    preconditioner_count,
                    dot_reductions,
                    norm_count,
                    axpy_passes,
                    basis_bytes_touched,
                    synchronization_points,
                    exact_residual_checks,
                },
            })
        } else {
            None
        };
        if reader.offset != bytes.len() {
            return Err(mixed_precision_checkpoint_error(
                "trailing bytes in checkpoint",
            ));
        }
        Ok(Self { solution, metrics })
    }

    /// Resume checked mixed-precision FGMRES from this iterate.
    ///
    /// The returned iteration/restart counts and operation counters include
    /// the source result offsets when this checkpoint was created with
    /// [`Self::from_solution`].
    pub fn resume<TLow, A>(
        &self,
        operator: &A,
        b: &[T],
        precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
        config: &FgmresConfig<T::Real>,
    ) -> Result<FgmresSolution<T>, SolverError>
    where
        T: PrecisionCast<TLow>,
        TLow: ComplexField,
        A: LinearOperator<T>,
    {
        let mut result = try_mixed_precision_solve_with_guess(
            operator,
            b,
            Some(&self.solution),
            precond,
            config,
        )?;
        if let Some(metrics) = self.metrics {
            result.iterations = metrics.iterations.saturating_add(result.iterations);
            result.restarts = metrics.restarts.saturating_add(result.restarts);
            result.op_counts = add_krylov_op_counts(metrics.op_counts, result.op_counts);
        }
        result.validate_contract()?;
        Ok(result)
    }
}

fn scalar_is_complex<T: ComplexField>() -> bool {
    std::mem::size_of::<T>() == std::mem::size_of::<T::Real>().saturating_mul(2)
}

fn push_mixed_precision_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value = u64::try_from(value)
        .map_err(|_| mixed_precision_checkpoint_error("checkpoint size exceeds u64"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn encode_mixed_precision_accuracy(value: AccuracyStatus) -> u8 {
    match value {
        AccuracyStatus::NotChecked => 0,
        AccuracyStatus::Attained => 1,
        AccuracyStatus::NotAttained => 2,
    }
}

fn decode_mixed_precision_accuracy(value: u8) -> Result<AccuracyStatus, SolverError> {
    match value {
        0 => Ok(AccuracyStatus::NotChecked),
        1 => Ok(AccuracyStatus::Attained),
        2 => Ok(AccuracyStatus::NotAttained),
        _ => Err(mixed_precision_checkpoint_error(
            "checkpoint attainable-accuracy tag is invalid",
        )),
    }
}

fn encode_mixed_precision_status(value: SolverStatus) -> u8 {
    match value {
        SolverStatus::Converged => 0,
        SolverStatus::MaxIterationsReached => 1,
        SolverStatus::Breakdown => 2,
        SolverStatus::Stagnated => 3,
        SolverStatus::NumericalError => 4,
        SolverStatus::Diverged => 5,
    }
}

fn decode_mixed_precision_status(value: u8) -> Result<SolverStatus, SolverError> {
    match value {
        0 => Ok(SolverStatus::Converged),
        1 => Ok(SolverStatus::MaxIterationsReached),
        2 => Ok(SolverStatus::Breakdown),
        3 => Ok(SolverStatus::Stagnated),
        4 => Ok(SolverStatus::NumericalError),
        5 => Ok(SolverStatus::Diverged),
        _ => Err(mixed_precision_checkpoint_error(
            "checkpoint solver-status tag is invalid",
        )),
    }
}

fn mixed_precision_checkpoint_error(reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "mixed-precision checkpoint",
        reason: reason.into(),
    }
}

struct MixedPrecisionCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> MixedPrecisionCheckpointReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| mixed_precision_checkpoint_error("checkpoint offset overflow"))?;
        if end > self.bytes.len() {
            return Err(mixed_precision_checkpoint_error("checkpoint is truncated"));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u64(&mut self) -> Result<u64, SolverError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
            mixed_precision_checkpoint_error("checkpoint integer is truncated")
        })?))
    }

    fn read_usize(&mut self) -> Result<usize, SolverError> {
        usize::try_from(self.read_u64()?)
            .map_err(|_| mixed_precision_checkpoint_error("checkpoint size does not fit usize"))
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let bytes = self.take(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().map_err(|_| {
            mixed_precision_checkpoint_error("checkpoint float is truncated")
        })?))
    }
}

#[derive(Debug, Clone)]
struct MixedPrecisionBlockCheckpointMetrics<R: Float> {
    iterations: usize,
    residuals: Vec<R>,
    true_residuals: Vec<R>,
    backward_errors: Vec<R>,
    attainable_accuracy: Vec<AccuracyStatus>,
    converged: bool,
    op_counts: KrylovOpCounts,
}

/// Checked continuation checkpoint for mixed-precision block FGMRES.
///
/// The checkpoint owns only the high-precision iterate and provenance
/// metadata.  A resumed solve must receive a fresh low-precision
/// preconditioner, so factor lifetime, device ownership, and precision
/// compatibility remain explicit at the call site.  Per-RHS residuals are
/// intentionally not treated as additive; the resumed result reports the
/// current solve's residual vectors while iteration and operation counters
/// retain cumulative source offsets.
#[derive(Debug, Clone)]
pub struct MixedPrecisionBlockCheckpoint<T: ComplexField> {
    solution: sonium_solvers_core::dense::DenseMatrix<T>,
    metrics: Option<MixedPrecisionBlockCheckpointMetrics<T::Real>>,
}

impl<T: ComplexField> MixedPrecisionBlockCheckpoint<T> {
    /// Capture a finite block-FGMRES result and its structural metadata.
    pub fn from_solution(solution: &BlockFgmresSolution<T>) -> Result<Self, SolverError> {
        validate_block_solution_shape(solution)?;
        Ok(Self {
            solution: solution.x.clone(),
            metrics: Some(MixedPrecisionBlockCheckpointMetrics {
                iterations: solution.iterations,
                residuals: solution.residuals.clone(),
                true_residuals: solution.true_residuals.clone(),
                backward_errors: solution.backward_errors.clone(),
                attainable_accuracy: solution.attainable_accuracy.clone(),
                converged: solution.converged,
                op_counts: solution.op_counts,
            }),
        })
    }

    /// Capture a caller-provided finite block iterate without source metrics.
    pub fn from_iterate(
        iterate: &sonium_solvers_core::dense::DenseMatrix<T>,
    ) -> Result<Self, SolverError> {
        if iterate
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "mixed-precision block checkpoint",
                reason: "iterate contains NaN or infinity".to_string(),
            });
        }
        Ok(Self {
            solution: iterate.clone(),
            metrics: None,
        })
    }

    /// Number of rows in the stored block iterate.
    pub fn dimension(&self) -> usize {
        self.solution.nrows()
    }

    /// Number of right-hand sides in the stored block iterate.
    pub fn num_rhs(&self) -> usize {
        self.solution.ncols()
    }

    /// Borrow the stored high-precision block iterate.
    pub fn solution(&self) -> &sonium_solvers_core::dense::DenseMatrix<T> {
        &self.solution
    }

    /// Number of bytes occupied by the stored iterate.
    pub fn storage_bytes(&self) -> usize {
        self.solution
            .as_slice()
            .len()
            .saturating_mul(std::mem::size_of::<T>())
    }

    /// Source block-iteration count, if captured from a solver result.
    pub fn source_iterations(&self) -> Option<usize> {
        self.metrics.as_ref().map(|metrics| metrics.iterations)
    }

    /// Source projected residual estimates, if captured from a block solve.
    /// These remain distinct from the independently recomputed true
    /// residuals when a checkpoint is resumed.
    pub fn source_residuals(&self) -> Option<&[T::Real]> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.residuals.as_slice())
    }

    /// Source true residuals, if captured from a block solve.
    pub fn source_true_residuals(&self) -> Option<&[T::Real]> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.true_residuals.as_slice())
    }

    /// Source normwise backward errors, if captured from a block solve.
    pub fn source_backward_errors(&self) -> Option<&[T::Real]> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.backward_errors.as_slice())
    }

    /// Source attainable-accuracy decisions, if captured from a block solve.
    pub fn source_attainable_accuracy(&self) -> Option<&[AccuracyStatus]> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.attainable_accuracy.as_slice())
    }

    /// Source operation counters, if captured from a solver result.
    pub fn source_op_counts(&self) -> Option<KrylovOpCounts> {
        self.metrics.as_ref().map(|metrics| metrics.op_counts)
    }

    /// Whether source metadata came from a converged result.
    pub fn source_converged(&self) -> Option<bool> {
        self.metrics.as_ref().map(|metrics| metrics.converged)
    }

    /// Resume checked mixed-precision block FGMRES from this iterate.
    pub fn resume<TLow, A>(
        &self,
        operator: &A,
        b: &sonium_solvers_core::dense::DenseMatrix<T>,
        precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
        config: &BlockFgmresConfig<T::Real>,
    ) -> Result<BlockFgmresSolution<T>, SolverError>
    where
        T: PrecisionCast<TLow>,
        TLow: ComplexField,
        A: LinearOperator<T>,
    {
        let mut result = try_mixed_precision_block_solve_with_guess(
            operator,
            b,
            Some(&self.solution),
            precond,
            config,
        )?;
        if let Some(metrics) = self.metrics.as_ref() {
            result.iterations = metrics.iterations.saturating_add(result.iterations);
            result.op_counts = add_krylov_op_counts(metrics.op_counts, result.op_counts);
        }
        validate_block_solution_shape(&result)?;
        Ok(result)
    }

    /// Serialize the block iterate and captured per-RHS metadata in a strict,
    /// versioned little-endian format.
    ///
    /// Matrix values use canonical f64 real/imaginary pairs. The packet keeps
    /// row-major layout and records the scalar family/width so a restore cannot
    /// silently reinterpret a real packet as a complex one (or vice versa).
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let validated = Self::from_iterate(&self.solution)?;
        let rows = validated.dimension();
        let cols = validated.num_rhs();
        let mut output = Vec::new();
        output.extend_from_slice(MIXED_PRECISION_BLOCK_CHECKPOINT_MAGIC);
        output.extend_from_slice(&MIXED_PRECISION_BLOCK_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(if scalar_is_complex::<T>() { 1 } else { 0 });
        output.push(if self.metrics.is_some() { 1 } else { 0 });
        push_mixed_precision_u64(&mut output, rows)?;
        push_mixed_precision_u64(&mut output, cols)?;
        push_mixed_precision_u64(&mut output, validated.storage_bytes())?;

        for value in validated.solution.as_slice() {
            let re = value.re().to_f64().ok_or_else(|| {
                mixed_precision_checkpoint_error(
                    "block real component cannot be represented as f64",
                )
            })?;
            let im = value.im().to_f64().ok_or_else(|| {
                mixed_precision_checkpoint_error(
                    "block imaginary component cannot be represented as f64",
                )
            })?;
            if !re.is_finite() || !im.is_finite() {
                return Err(mixed_precision_checkpoint_error(
                    "block checkpoint contains non-finite scalar",
                ));
            }
            output.extend_from_slice(&re.to_le_bytes());
            output.extend_from_slice(&im.to_le_bytes());
        }

        if let Some(metrics) = self.metrics.as_ref() {
            if metrics.residuals.len() != cols
                || metrics.true_residuals.len() != cols
                || metrics.backward_errors.len() != cols
                || metrics.attainable_accuracy.len() != cols
            {
                return Err(mixed_precision_checkpoint_error(
                    "block metadata width does not match checkpoint columns",
                ));
            }
            push_mixed_precision_u64(&mut output, metrics.iterations)?;
            for diagnostics in [
                &metrics.residuals,
                &metrics.true_residuals,
                &metrics.backward_errors,
            ] {
                for diagnostic in diagnostics.iter() {
                    let value = diagnostic.to_f64().ok_or_else(|| {
                        mixed_precision_checkpoint_error(
                            "block diagnostic cannot be represented as f64",
                        )
                    })?;
                    if !value.is_finite() || value < 0.0 {
                        return Err(mixed_precision_checkpoint_error(
                            "block diagnostic must be finite and non-negative",
                        ));
                    }
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
            for accuracy in &metrics.attainable_accuracy {
                output.push(encode_mixed_precision_accuracy(*accuracy));
            }
            output.push(u8::from(metrics.converged));
            output.push(0);
            let counts = metrics.op_counts;
            for value in [
                counts.spmv_count,
                counts.preconditioner_count,
                counts.dot_reductions,
                counts.norm_count,
                counts.axpy_passes,
                counts.synchronization_points,
                counts.exact_residual_checks,
            ] {
                push_mixed_precision_u64(&mut output, value)?;
            }
            output.extend_from_slice(&counts.basis_bytes_touched.to_le_bytes());
        }
        Ok(output)
    }

    /// Restore a strict portable mixed-precision block checkpoint packet.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = MixedPrecisionCheckpointReader { bytes, offset: 0 };
        if reader.take(8)? != MIXED_PRECISION_BLOCK_CHECKPOINT_MAGIC {
            return Err(mixed_precision_checkpoint_error(
                "block checkpoint magic mismatch",
            ));
        }
        if reader.read_u16()? != MIXED_PRECISION_BLOCK_CHECKPOINT_FORMAT_VERSION {
            return Err(mixed_precision_checkpoint_error(
                "block checkpoint format version mismatch",
            ));
        }
        let scalar_size = reader.read_u8()?;
        let real_size = reader.read_u8()?;
        let complex = reader.read_u8()?;
        let has_metrics = reader.read_u8()?;
        if scalar_size != std::mem::size_of::<T>() as u8
            || real_size != std::mem::size_of::<T::Real>() as u8
            || complex != u8::from(scalar_is_complex::<T>())
        {
            return Err(mixed_precision_checkpoint_error(
                "block checkpoint scalar family does not match target",
            ));
        }
        if has_metrics > 1 {
            return Err(mixed_precision_checkpoint_error(
                "block checkpoint metrics flag is invalid",
            ));
        }
        let rows = reader.read_usize()?;
        let cols = reader.read_usize()?;
        let element_count = rows
            .checked_mul(cols)
            .ok_or_else(|| mixed_precision_checkpoint_error("block checkpoint shape overflow"))?;
        let storage_bytes = reader.read_usize()?;
        let expected_bytes = element_count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| mixed_precision_checkpoint_error("block checkpoint storage overflow"))?;
        if storage_bytes != expected_bytes {
            return Err(mixed_precision_checkpoint_error(
                "block checkpoint storage size does not match shape",
            ));
        }
        let encoded_solution_bytes = element_count
            .checked_mul(std::mem::size_of::<f64>() * 2)
            .ok_or_else(|| mixed_precision_checkpoint_error("block checkpoint payload overflow"))?;
        let encoded_metrics_bytes = if has_metrics == 1 {
            let per_rhs = cols
                .checked_mul(std::mem::size_of::<f64>() * 3 + 1)
                .ok_or_else(|| {
                    mixed_precision_checkpoint_error("block checkpoint metadata overflow")
                })?;
            8usize
                .checked_add(per_rhs)
                .and_then(|bytes| bytes.checked_add(2 + 8 * 8))
                .ok_or_else(|| {
                    mixed_precision_checkpoint_error("block checkpoint metadata overflow")
                })?
        } else {
            0
        };
        let encoded_payload_bytes = encoded_solution_bytes
            .checked_add(encoded_metrics_bytes)
            .ok_or_else(|| mixed_precision_checkpoint_error("block checkpoint payload overflow"))?;
        if encoded_payload_bytes > bytes.len().saturating_sub(reader.offset) {
            return Err(mixed_precision_checkpoint_error(
                "block checkpoint is truncated",
            ));
        }
        let mut data = Vec::with_capacity(element_count);
        for _ in 0..element_count {
            let re = T::Real::from_f64(reader.read_f64()?).ok_or_else(|| {
                mixed_precision_checkpoint_error("block real component cannot be restored")
            })?;
            let encoded_im = reader.read_f64()?;
            if !scalar_is_complex::<T>() && encoded_im != 0.0 {
                return Err(mixed_precision_checkpoint_error(
                    "real block checkpoint contains non-zero imaginary component",
                ));
            }
            let im = T::Real::from_f64(encoded_im).ok_or_else(|| {
                mixed_precision_checkpoint_error("block imaginary component cannot be restored")
            })?;
            let value = T::from_re_im(re, im);
            if !value.norm().is_finite() {
                return Err(mixed_precision_checkpoint_error(
                    "block checkpoint restores non-finite scalar",
                ));
            }
            data.push(value);
        }
        let solution =
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(rows, cols, data);
        let metrics = if has_metrics == 1 {
            let iterations = reader.read_usize()?;
            let mut residuals = Vec::with_capacity(cols);
            let mut true_residuals = Vec::with_capacity(cols);
            let mut backward_errors = Vec::with_capacity(cols);
            for diagnostics in [&mut residuals, &mut true_residuals, &mut backward_errors] {
                for _ in 0..cols {
                    let value = T::Real::from_f64(reader.read_f64()?).ok_or_else(|| {
                        mixed_precision_checkpoint_error("block diagnostic cannot be restored")
                    })?;
                    if !value.is_finite() || value < T::Real::zero() {
                        return Err(mixed_precision_checkpoint_error(
                            "block diagnostic must be finite and non-negative",
                        ));
                    }
                    diagnostics.push(value);
                }
            }
            let mut attainable_accuracy = Vec::with_capacity(cols);
            for _ in 0..cols {
                attainable_accuracy.push(decode_mixed_precision_accuracy(reader.read_u8()?)?);
            }
            let converged = match reader.read_u8()? {
                0 => false,
                1 => true,
                _ => {
                    return Err(mixed_precision_checkpoint_error(
                        "block checkpoint convergence flag is invalid",
                    ));
                }
            };
            if reader.read_u8()? != 0 {
                return Err(mixed_precision_checkpoint_error(
                    "block checkpoint reserved byte is non-zero",
                ));
            }
            let spmv_count = reader.read_usize()?;
            let preconditioner_count = reader.read_usize()?;
            let dot_reductions = reader.read_usize()?;
            let norm_count = reader.read_usize()?;
            let axpy_passes = reader.read_usize()?;
            let synchronization_points = reader.read_usize()?;
            let exact_residual_checks = reader.read_usize()?;
            let basis_bytes_touched = reader.read_u64()?;
            Some(MixedPrecisionBlockCheckpointMetrics {
                iterations,
                residuals,
                true_residuals,
                backward_errors,
                attainable_accuracy,
                converged,
                op_counts: KrylovOpCounts {
                    spmv_count,
                    preconditioner_count,
                    dot_reductions,
                    norm_count,
                    axpy_passes,
                    basis_bytes_touched,
                    synchronization_points,
                    exact_residual_checks,
                },
            })
        } else {
            None
        };
        if reader.offset != bytes.len() {
            return Err(mixed_precision_checkpoint_error(
                "trailing bytes in block checkpoint",
            ));
        }
        let checkpoint = Self { solution, metrics };
        if let Some(metrics) = checkpoint.metrics.as_ref()
            && (metrics.residuals.len() != cols
                || metrics.true_residuals.len() != cols
                || metrics.backward_errors.len() != cols
                || metrics.attainable_accuracy.len() != cols)
        {
            return Err(mixed_precision_checkpoint_error(
                "block checkpoint metadata width does not match shape",
            ));
        }
        Ok(checkpoint)
    }
}

fn validate_block_solution_shape<T: ComplexField>(
    solution: &BlockFgmresSolution<T>,
) -> Result<(), SolverError> {
    let rhs_count = solution.x.ncols();
    let lengths = [
        solution.residuals.len(),
        solution.true_residuals.len(),
        solution.backward_errors.len(),
        solution.attainable_accuracy.len(),
    ];
    if lengths.iter().any(|&length| length != rhs_count) {
        return Err(SolverError::OperationFailed {
            operation: "mixed-precision block checkpoint",
            reason: format!("per-RHS metadata length does not match block width {rhs_count}"),
        });
    }
    if solution
        .x
        .as_slice()
        .iter()
        .any(|value| !value.norm().is_finite())
        || solution
            .residuals
            .iter()
            .chain(&solution.true_residuals)
            .chain(&solution.backward_errors)
            .any(|value| !value.is_finite() || *value < T::Real::zero())
    {
        return Err(SolverError::OperationFailed {
            operation: "mixed-precision block checkpoint",
            reason: "solution contains non-finite or negative diagnostics".to_string(),
        });
    }
    Ok(())
}

/// Checked mixed-precision block FGMRES entry point.
pub fn try_mixed_precision_block_solve<THigh, TLow, A>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &BlockFgmresConfig<THigh::Real>,
) -> Result<BlockFgmresSolution<THigh>, SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    try_mixed_precision_block_solve_with_guess(operator, b, None, precond, config)
}

/// Checked mixed-precision block FGMRES with an optional initial iterate.
pub fn try_mixed_precision_block_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<THigh>,
    initial_guess: Option<&sonium_solvers_core::dense::DenseMatrix<THigh>>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &BlockFgmresConfig<THigh::Real>,
) -> Result<BlockFgmresSolution<THigh>, SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    if b.as_slice().iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "mixed-precision block FGMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(guess) = initial_guess {
        if guess.nrows() != b.nrows() || guess.ncols() != b.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: b.nrows().saturating_mul(b.ncols()),
                got: guess.nrows().saturating_mul(guess.ncols()),
            });
        }
        if guess
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "mixed-precision block FGMRES",
                reason: "initial guess contains NaN or infinity".to_string(),
            });
        }
    }

    let mixed_precond = MixedPrecisionPreconditioner::<THigh, TLow>::new(precond);
    let checked_config = mixed_precision_block_config(config);
    let aborting_precond = AbortOnLowPrecisionFailure {
        inner: &mixed_precond,
    };
    let result = try_block_fgmres_preconditioned(
        operator,
        b,
        &aborting_precond,
        initial_guess,
        &checked_config,
    );
    if mixed_precond.has_low_precision_failure() {
        return Err(SolverError::OperationFailed {
            operation: "mixed-precision block FGMRES preconditioner",
            reason: format!(
                "non-finite intermediate value at {} precision boundary ({} failures)",
                std::any::type_name::<TLow>(),
                mixed_precond.low_precision_failure_count()
            ),
        });
    }
    let result = result?;
    validate_block_solution_shape(&result)?;
    Ok(result)
}

fn add_krylov_op_counts(lhs: KrylovOpCounts, rhs: KrylovOpCounts) -> KrylovOpCounts {
    KrylovOpCounts {
        spmv_count: lhs.spmv_count.saturating_add(rhs.spmv_count),
        preconditioner_count: lhs
            .preconditioner_count
            .saturating_add(rhs.preconditioner_count),
        dot_reductions: lhs.dot_reductions.saturating_add(rhs.dot_reductions),
        norm_count: lhs.norm_count.saturating_add(rhs.norm_count),
        axpy_passes: lhs.axpy_passes.saturating_add(rhs.axpy_passes),
        basis_bytes_touched: lhs
            .basis_bytes_touched
            .saturating_add(rhs.basis_bytes_touched),
        synchronization_points: lhs
            .synchronization_points
            .saturating_add(rhs.synchronization_points),
        exact_residual_checks: lhs
            .exact_residual_checks
            .saturating_add(rhs.exact_residual_checks),
    }
}

/// Checked mixed-precision FGMRES entry point.
///
/// Unlike [`mixed_precision_solve`], this boundary validates configuration,
/// dimensions, RHS values, preconditioner output, and the final high-
/// precision original-operator residual through the checked FGMRES path.
pub fn try_mixed_precision_solve<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &FgmresConfig<THigh::Real>,
) -> Result<FgmresSolution<THigh>, crate::traits::SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    try_mixed_precision_solve_with_guess(operator, b, None, precond, config)
}

fn validate_initial_guess<T: ComplexField>(
    expected: usize,
    initial_guess: Option<&[T]>,
    operation: &'static str,
) -> Result<(), SolverError> {
    let Some(initial_guess) = initial_guess else {
        return Ok(());
    };
    if initial_guess.len() != expected {
        return Err(SolverError::DimensionMismatch {
            expected,
            got: initial_guess.len(),
        });
    }
    if initial_guess.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "initial guess contains NaN or infinity".into(),
        });
    }
    Ok(())
}

/// Checked mixed-precision FGMRES entry point with an optional initial
/// iterate. The iterate is validated by the underlying checked FGMRES path
/// before any operator or preconditioner callback runs.
pub fn try_mixed_precision_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &FgmresConfig<THigh::Real>,
) -> Result<FgmresSolution<THigh>, crate::traits::SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    validate_initial_guess(operator.num_cols(), initial_guess, "mixed-precision FGMRES")?;
    let mixed_precond = MixedPrecisionPreconditioner::<THigh, TLow>::new(precond);
    let checked_config = mixed_precision_config(config);
    let result = try_fgmres_preconditioned_with_abort(
        operator,
        &mixed_precond,
        b,
        initial_guess,
        &checked_config,
        || mixed_precond.has_low_precision_failure(),
    );
    if mixed_precond.has_low_precision_failure() {
        return Err(crate::traits::SolverError::OperationFailed {
            operation: "mixed-precision preconditioner",
            reason: format!(
                "non-finite intermediate value at the {} precision boundary ({} failures)",
                std::any::type_name::<TLow>(),
                mixed_precond.low_precision_failure_count(),
            ),
        });
    }
    result
}

/// Infallible mixed-precision deflated GMRES solve.
pub fn mixed_precision_deflated_solve<THigh, TLow, A>(
    operator: &A,
    deflation: &DeflationSubspace<THigh>,
    b: &[THigh],
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GmresConfig<THigh::Real>,
) -> GmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    mixed_precision_deflated_solve_with_guess(operator, deflation, b, None, precond, config)
}

/// Checked mixed-precision deflated GMRES solve.
pub fn try_mixed_precision_deflated_solve<THigh, TLow, A>(
    operator: &A,
    deflation: &DeflationSubspace<THigh>,
    b: &[THigh],
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GmresConfig<THigh::Real>,
) -> Result<GmresSolution<THigh>, SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    try_mixed_precision_deflated_solve_with_guess(operator, deflation, b, None, precond, config)
}

/// Infallible mixed-precision deflated GMRES continuation solve.
pub fn mixed_precision_deflated_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    deflation: &DeflationSubspace<THigh>,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GmresConfig<THigh::Real>,
) -> GmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    match try_mixed_precision_deflated_solve_with_guess(
        operator,
        deflation,
        b,
        initial_guess,
        precond,
        config,
    ) {
        Ok(result) => result,
        Err(_) => numerical_error_gmres_solution(b.len()),
    }
}

/// Checked mixed-precision deflated GMRES continuation solve.
pub fn try_mixed_precision_deflated_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    deflation: &DeflationSubspace<THigh>,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GmresConfig<THigh::Real>,
) -> Result<GmresSolution<THigh>, SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    validate_initial_guess(
        operator.num_cols(),
        initial_guess,
        "mixed-precision deflated GMRES",
    )?;
    let mixed_precond = MixedPrecisionPreconditioner::<THigh, TLow>::new(precond);
    let aborting_precond = AbortOnLowPrecisionFailure {
        inner: &mixed_precond,
    };
    let checked_config = mixed_precision_gmres_config(config);
    let result = try_gmres_deflated_preconditioned(
        operator,
        &aborting_precond,
        deflation,
        b,
        initial_guess,
        &checked_config,
    );
    if mixed_precond.has_low_precision_failure() {
        return Err(SolverError::OperationFailed {
            operation: "mixed-precision deflated GMRES preconditioner",
            reason: format!(
                "non-finite intermediate value at {} precision boundary ({} failures)",
                std::any::type_name::<TLow>(),
                mixed_precond.low_precision_failure_count(),
            ),
        });
    }
    result.and_then(|result| {
        normalize_mixed_precision_gmres_result(
            result,
            config.tolerance,
            "mixed-precision deflated GMRES",
        )
    })
}

/// Infallible mixed-precision recycled GMRES compatibility solve.
#[allow(deprecated)]
pub fn mixed_precision_recycled_solve<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    subspace: &mut RecycledSubspace<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GmresConfig<THigh::Real>,
) -> GmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    mixed_precision_recycled_solve_with_guess(operator, b, None, subspace, precond, config)
}

/// Checked mixed-precision recycled GMRES compatibility solve.
#[allow(deprecated)]
pub fn try_mixed_precision_recycled_solve<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    subspace: &mut RecycledSubspace<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GmresConfig<THigh::Real>,
) -> Result<GmresSolution<THigh>, SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    try_mixed_precision_recycled_solve_with_guess(operator, b, None, subspace, precond, config)
}

/// Infallible mixed-precision recycled GMRES continuation solve.
#[allow(deprecated)]
pub fn mixed_precision_recycled_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    subspace: &mut RecycledSubspace<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GmresConfig<THigh::Real>,
) -> GmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    match try_mixed_precision_recycled_solve_with_guess(
        operator,
        b,
        initial_guess,
        subspace,
        precond,
        config,
    ) {
        Ok(result) => result,
        Err(_) => numerical_error_gmres_solution(b.len()),
    }
}

/// Checked mixed-precision recycled GMRES continuation solve.
#[allow(deprecated)]
pub fn try_mixed_precision_recycled_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    subspace: &mut RecycledSubspace<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GmresConfig<THigh::Real>,
) -> Result<GmresSolution<THigh>, SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    validate_initial_guess(
        operator.num_cols(),
        initial_guess,
        "mixed-precision recycled GMRES",
    )?;
    let mixed_precond = MixedPrecisionPreconditioner::<THigh, TLow>::new(precond);
    let aborting_precond = AbortOnLowPrecisionFailure {
        inner: &mixed_precond,
    };
    let checked_config = mixed_precision_gmres_config(config);
    let result = try_gmres_recycled_preconditioned(
        operator,
        &aborting_precond,
        b,
        initial_guess,
        subspace,
        &checked_config,
    );
    if mixed_precond.has_low_precision_failure() {
        return Err(SolverError::OperationFailed {
            operation: "mixed-precision recycled GMRES preconditioner",
            reason: format!(
                "non-finite intermediate value at {} precision boundary ({} failures)",
                std::any::type_name::<TLow>(),
                mixed_precond.low_precision_failure_count(),
            ),
        });
    }
    result.and_then(|result| {
        normalize_mixed_precision_gmres_result(
            result,
            config.tolerance,
            "mixed-precision recycled GMRES",
        )
    })
}

/// Infallible mixed-precision GCRO-DR solve.
pub fn mixed_precision_gcro_dr_solve<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    subspace: &mut GcroDrSubspace<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GcroDrConfig<THigh::Real>,
) -> GmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    mixed_precision_gcro_dr_solve_with_guess(operator, b, None, subspace, precond, config)
}

/// Checked mixed-precision GCRO-DR solve with an optional continuation iterate.
pub fn try_mixed_precision_gcro_dr_solve<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    subspace: &mut GcroDrSubspace<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GcroDrConfig<THigh::Real>,
) -> Result<GmresSolution<THigh>, SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    try_mixed_precision_gcro_dr_solve_with_guess(operator, b, None, subspace, precond, config)
}

/// Infallible mixed-precision GCRO-DR continuation solve.
pub fn mixed_precision_gcro_dr_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    subspace: &mut GcroDrSubspace<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GcroDrConfig<THigh::Real>,
) -> GmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    match try_mixed_precision_gcro_dr_solve_with_guess(
        operator,
        b,
        initial_guess,
        subspace,
        precond,
        config,
    ) {
        Ok(result) => result,
        Err(_) => numerical_error_gmres_solution(b.len()),
    }
}

/// Checked mixed-precision GCRO-DR continuation solve.
pub fn try_mixed_precision_gcro_dr_solve_with_guess<THigh, TLow, A>(
    operator: &A,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    subspace: &mut GcroDrSubspace<THigh>,
    precond: Box<dyn Preconditioner<TLow> + Send + Sync>,
    config: &GcroDrConfig<THigh::Real>,
) -> Result<GmresSolution<THigh>, SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
    A: LinearOperator<THigh>,
{
    validate_initial_guess(
        operator.num_cols(),
        initial_guess,
        "mixed-precision GCRO-DR",
    )?;
    let mixed_precond = MixedPrecisionPreconditioner::<THigh, TLow>::new(precond);
    let aborting_precond = AbortOnLowPrecisionFailure {
        inner: &mixed_precond,
    };
    let result = try_gcro_dr_preconditioned(
        operator,
        &aborting_precond,
        b,
        initial_guess,
        subspace,
        config,
    );
    if mixed_precond.has_low_precision_failure() {
        return Err(SolverError::OperationFailed {
            operation: "mixed-precision GCRO-DR preconditioner",
            reason: format!(
                "non-finite intermediate value at {} precision boundary ({} failures)",
                std::any::type_name::<TLow>(),
                mixed_precond.low_precision_failure_count(),
            ),
        });
    }
    result.and_then(|result| {
        normalize_mixed_precision_gmres_result(result, config.tolerance, "mixed-precision GCRO-DR")
    })
}

/// GMRES-based iterative refinement (Carson–Higham GMRES-IR) driven by a
/// low-precision sparse LU factorization of A.
///
/// Pipeline, end-to-end:
/// 1. Cast `matrix` down to `TLow` (structure-preserving CSR cast). When
///    `TLow` has a narrow exponent range (f16), the cast first applies a
///    two-sided max-abs equilibration plus a power-of-two scalar in `THigh`
///    so no entry overflows/underflows the target range — see the module
///    docs for why the wrapped apply remains an exact-algebra preconditioner
///    for the unscaled A.
/// 2. Factorize the low-precision copy with sparse LU (COLAMD + partial
///    pivoting) — this is the step plain mixed-precision FGMRES lacks.
/// 3. Run FGMRES in `THigh` with the low-precision factor as a flexible
///    preconditioner (via [`MixedPrecisionPreconditioner`], which casts the
///    residual down, triangular-solves in `TLow`, and casts the update up).
///
/// Because the f32/Complex32 LU factor is a near-exact inverse of A, the
/// preconditioned operator is `I + O(cond(A)·u_low)` and the outer FGMRES
/// converges in very few iterations to `THigh` accuracy. Per Carson–Higham
/// the scheme tolerates `cond(A)` up to roughly `u_low⁻¹` (≈1e7 for f32,
/// ≈2e3 for f16 measured on the equilibrated matrix), far beyond the
/// `cond(A)·u_low < 1` limit of stationary IR.
///
/// Best suited to small/medium sparse systems where an LU factorization of A
/// is affordable; for large systems use [`gmres_ir_ilu`] instead.
///
/// # Errors
///
/// Returns [`SparseLuError`] if the low-precision factorization fails
/// (e.g. the matrix is exactly singular in `TLow` arithmetic).
pub fn gmres_ir_lu<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    b: &[THigh],
    config: &FgmresConfig<THigh::Real>,
) -> Result<FgmresSolution<THigh>, SparseLuError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    gmres_ir_lu_with_guess(matrix, b, None, config)
}

/// GMRES-IR sparse-LU compatibility wrapper with a caller-provided
/// continuation iterate. The legacy error type is preserved; successful
/// results still carry checked true-residual and backward-error metadata.
pub fn gmres_ir_lu_with_guess<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    config: &FgmresConfig<THigh::Real>,
) -> Result<FgmresSolution<THigh>, SparseLuError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    matrix
        .validate_canonical_structure()
        .map_err(|error| SparseLuError::InvalidCsrStructure(error.to_string()))?;
    if matrix.num_rows != matrix.num_cols {
        return Err(SparseLuError::NotSquare {
            rows: matrix.num_rows,
            cols: matrix.num_cols,
        });
    }
    if b.len() != matrix.num_cols {
        return Err(SparseLuError::DimensionMismatch {
            expected: matrix.num_cols,
            actual: b.len(),
        });
    }
    if let Some(index) = b.iter().position(|value| !value.norm().is_finite()) {
        return Err(SparseLuError::NonFiniteRhs(index));
    }
    config
        .validate()
        .map_err(|error| SparseLuError::InvalidRefinementTolerance(error.to_string()))?;
    let checked_config = mixed_precision_config(config);
    let (matrix_low, scaling) =
        checked_cast_for_factorization::<THigh, TLow>(matrix, "GMRES-IR LU cast")
            .map_err(|error| SparseLuError::IterativeSolve(error.to_string()))?;
    // Preconditioner mode: the sparse LU's strict `eps·n·‖A‖` pivot floor
    // would reject f16 factors already at cond ≈ 1e3, while GMRES-IR
    // tolerates cond up to ~u_low⁻¹ (the outer FGMRES certifies the result
    // against the true residual, so a poor inverse is still usable).
    let factor_low = SparseLuFactorization::factorize_for_preconditioner(&matrix_low)?;
    let precond = MixedPrecisionPreconditioner::<THigh, TLow>::with_matrix_scaling(
        Box::new(factor_low),
        scaling,
    );
    let result = try_fgmres_preconditioned_with_abort(
        matrix,
        &precond,
        b,
        initial_guess,
        &checked_config,
        || precond.has_low_precision_failure(),
    )
    .map_err(|error| SparseLuError::IterativeSolve(error.to_string()));
    if precond.has_low_precision_failure() {
        return Err(SparseLuError::IterativeSolve(format!(
            "GMRES-IR LU preconditioner produced a non-finite intermediate value at the {} precision boundary ({} failures)",
            std::any::type_name::<TLow>(),
            precond.low_precision_failure_count()
        )));
    }
    result
}

/// GMRES-IR driven by a low-precision ILU factorization of A.
///
/// Same downcast–factorize–solve pipeline as [`gmres_ir_lu`], but the
/// low-precision factor is an incomplete LU (see [`IluConfig`] for fill
/// reordering options). This keeps factorization cost at O(nnz) for large
/// systems where exact sparse LU would fill in. The outer FGMRES still
/// reaches `THigh` accuracy; it simply needs more iterations because the
/// preconditioner is no longer a near-exact inverse. The matrix cast is
/// equilibrated exactly as in [`gmres_ir_lu`] when `TLow` is f16.
/// Checked GMRES-IR sparse-LU entry point.
pub fn try_gmres_ir_lu<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    b: &[THigh],
    config: &FgmresConfig<THigh::Real>,
) -> Result<FgmresSolution<THigh>, crate::traits::SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    try_gmres_ir_lu_with_guess(matrix, b, None, config)
}

/// Checked GMRES-IR sparse-LU entry point with a caller-provided continuation
/// iterate. The iterate is used as the restart base, while the returned
/// residual, backward error, projected estimate, and attainable-accuracy state
/// are always recomputed by the checked FGMRES contract.
pub fn try_gmres_ir_lu_with_guess<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    config: &FgmresConfig<THigh::Real>,
) -> Result<FgmresSolution<THigh>, crate::traits::SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    validate_initial_guess(matrix.num_cols, initial_guess, "GMRES-IR LU")?;
    matrix.validate_canonical_structure()?;
    if matrix.num_rows != matrix.num_cols {
        return Err(crate::traits::SolverError::DimensionMismatch {
            expected: matrix.num_cols,
            got: matrix.num_rows,
        });
    }
    if b.len() != matrix.num_cols {
        return Err(crate::traits::SolverError::DimensionMismatch {
            expected: matrix.num_cols,
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(crate::traits::SolverError::OperationFailed {
            operation: "GMRES-IR LU",
            reason: "right-hand side contains NaN or infinity".into(),
        });
    }
    config.validate()?;
    let checked_config = mixed_precision_config(config);

    let (matrix_low, scaling) =
        checked_cast_for_factorization::<THigh, TLow>(matrix, "GMRES-IR LU cast")?;
    let factor_low =
        SparseLuFactorization::factorize_for_preconditioner(&matrix_low).map_err(|error| {
            crate::traits::SolverError::OperationFailed {
                operation: "GMRES-IR LU factorization",
                reason: error.to_string(),
            }
        })?;
    let precond = MixedPrecisionPreconditioner::<THigh, TLow>::with_matrix_scaling(
        Box::new(factor_low),
        scaling,
    );
    let result = try_fgmres_preconditioned_with_abort(
        matrix,
        &precond,
        b,
        initial_guess,
        &checked_config,
        || precond.has_low_precision_failure(),
    );
    if precond.has_low_precision_failure() {
        return Err(crate::traits::SolverError::OperationFailed {
            operation: "GMRES-IR LU preconditioner",
            reason: format!(
                "non-finite intermediate value at the {} precision boundary ({} failures)",
                std::any::type_name::<TLow>(),
                precond.low_precision_failure_count()
            ),
        });
    }
    result
}

pub fn gmres_ir_ilu<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    b: &[THigh],
    ilu_config: &IluConfig,
    config: &FgmresConfig<THigh::Real>,
) -> FgmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    gmres_ir_ilu_with_guess(matrix, b, None, ilu_config, config)
}

/// GMRES-IR ILU compatibility wrapper with a caller-provided continuation
/// iterate. Successful results retain the same checked residual metadata as
/// the fallible entry point; setup or boundary failures remain numerical-error
/// results for compatibility with the historical infallible API.
pub fn gmres_ir_ilu_with_guess<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    ilu_config: &IluConfig,
    config: &FgmresConfig<THigh::Real>,
) -> FgmresSolution<THigh>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    // Keep the historical infallible wrapper best-effort: checked ILU setup
    // deliberately rejects skipped pivots, but the compatibility API has
    // always allowed those diagnostics to flow into FGMRES as a weaker
    // preconditioner. Validate the public boundary first so malformed input
    // still returns a typed numerical result instead of reaching factorization
    // with inconsistent CSR storage.
    let valid_boundary = matrix.validate_canonical_structure().is_ok()
        && matrix.num_rows == matrix.num_cols
        && b.len() == matrix.num_cols
        && matrix.values.iter().all(|value| value.norm().is_finite())
        && b.iter().all(|value| value.norm().is_finite())
        // The infallible compatibility wrapper must not silently replace an
        // invalid pivot tolerance with `IluConfig::default()`.  The checked
        // constructor rejects this configuration; keep the legacy result
        // shape, but fail closed with a numerical-error solution as well.
        && ilu_config.pivot_tolerance.is_finite()
        && (0.0..=1.0).contains(&ilu_config.pivot_tolerance)
        && config.validate().is_ok();
    if !valid_boundary {
        return FgmresSolution {
            x: vec![THigh::zero(); b.len()],
            iterations: 0,
            restarts: 0,
            residual: THigh::Real::infinity(),
            estimated_residual: THigh::Real::infinity(),
            backward_error: THigh::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::NumericalError,
            op_counts: Default::default(),
        };
    }

    let checked_config = mixed_precision_config(config);
    let (matrix_low, scaling) =
        match checked_cast_for_factorization::<THigh, TLow>(matrix, "GMRES-IR ILU cast") {
            Ok(result) => result,
            Err(_) => return numerical_error_solution(b.len()),
        };
    let factor_low = match IluPreconditioner::try_from_csr_with_config(&matrix_low, ilu_config) {
        Ok(factor) => factor,
        Err(_) => return numerical_error_solution(b.len()),
    };
    let precond = MixedPrecisionPreconditioner::<THigh, TLow>::with_matrix_scaling(
        Box::new(factor_low),
        scaling,
    );
    let result = try_fgmres_preconditioned_with_abort(
        matrix,
        &precond,
        b,
        initial_guess,
        &checked_config,
        || precond.has_low_precision_failure(),
    );
    if precond.has_low_precision_failure() {
        numerical_error_solution(b.len())
    } else {
        match result {
            Ok(result) => result,
            Err(_) => numerical_error_solution(b.len()),
        }
    }
}

/// Checked GMRES-IR with a low-precision ILU factor.
///
/// This counterpart validates the high-precision CSR boundary, propagates
/// ILU setup diagnostics, and verifies the final original-operator residual
/// through checked FGMRES. The historical [`gmres_ir_ilu`] API remains
/// available for compatibility and retains its infallible result type.
pub fn try_gmres_ir_ilu<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    b: &[THigh],
    ilu_config: &IluConfig,
    config: &FgmresConfig<THigh::Real>,
) -> Result<FgmresSolution<THigh>, crate::traits::SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    try_gmres_ir_ilu_with_guess(matrix, b, None, ilu_config, config)
}

/// Checked GMRES-IR ILU entry point with a caller-provided continuation
/// iterate. The checked FGMRES path owns exact-residual promotion and reports
/// the same normalized metrics for fresh and resumed solves.
pub fn try_gmres_ir_ilu_with_guess<THigh, TLow>(
    matrix: &CsrMatrix<THigh>,
    b: &[THigh],
    initial_guess: Option<&[THigh]>,
    ilu_config: &IluConfig,
    config: &FgmresConfig<THigh::Real>,
) -> Result<FgmresSolution<THigh>, crate::traits::SolverError>
where
    THigh: PrecisionCast<TLow>,
    TLow: ComplexField,
{
    validate_initial_guess(matrix.num_cols, initial_guess, "GMRES-IR ILU")?;
    matrix.validate_canonical_structure()?;
    if matrix.num_rows != matrix.num_cols {
        return Err(crate::traits::SolverError::DimensionMismatch {
            expected: matrix.num_cols,
            got: matrix.num_rows,
        });
    }
    if b.len() != matrix.num_cols {
        return Err(crate::traits::SolverError::DimensionMismatch {
            expected: matrix.num_cols,
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(crate::traits::SolverError::OperationFailed {
            operation: "GMRES-IR ILU",
            reason: "right-hand side contains NaN or infinity".into(),
        });
    }
    config.validate()?;
    let checked_config = mixed_precision_config(config);
    let (matrix_low, scaling) =
        checked_cast_for_factorization::<THigh, TLow>(matrix, "GMRES-IR ILU cast")?;
    let factor_low = IluPreconditioner::try_from_csr_with_config(&matrix_low, ilu_config)?;
    let precond = MixedPrecisionPreconditioner::<THigh, TLow>::with_matrix_scaling(
        Box::new(factor_low),
        scaling,
    );
    let result = try_fgmres_preconditioned_with_abort(
        matrix,
        &precond,
        b,
        initial_guess,
        &checked_config,
        || precond.has_low_precision_failure(),
    );
    if precond.has_low_precision_failure() {
        return Err(crate::traits::SolverError::OperationFailed {
            operation: "GMRES-IR ILU preconditioner",
            reason: format!(
                "non-finite intermediate value at the {} precision boundary ({} failures)",
                std::any::type_name::<TLow>(),
                precond.low_precision_failure_count()
            ),
        });
    }
    result
}

/// GMRES-IR with an f32 sparse LU factor of the f64 matrix.
///
/// Convenience constructor for the most common pairing; see [`gmres_ir_lu`]
/// for the algorithm and its convergence requirements (`cond(A) ≲ 1e7`).
pub fn gmres_ir_f32(
    matrix: &CsrMatrix<f64>,
    b: &[f64],
    config: &FgmresConfig<f64>,
) -> Result<FgmresSolution<f64>, SparseLuError> {
    gmres_ir_lu::<f64, f32>(matrix, b, config)
}

/// GMRES-IR with a Complex32 sparse LU factor of the Complex64 matrix.
///
/// Convenience constructor for complex systems (Helmholtz, Maxwell); see
/// [`gmres_ir_lu`] for the algorithm and its convergence requirements.
pub fn gmres_ir_complex32(
    matrix: &CsrMatrix<Complex64>,
    b: &[Complex64],
    config: &FgmresConfig<f64>,
) -> Result<FgmresSolution<Complex64>, SparseLuError> {
    gmres_ir_lu::<Complex64, Complex32>(matrix, b, config)
}

/// Checked f32 inner-factor GMRES-IR convenience entry point.
pub fn try_gmres_ir_f32(
    matrix: &CsrMatrix<f64>,
    b: &[f64],
    config: &FgmresConfig<f64>,
) -> Result<FgmresSolution<f64>, crate::traits::SolverError> {
    try_gmres_ir_lu::<f64, f32>(matrix, b, config)
}

/// Checked Complex32 inner-factor GMRES-IR convenience entry point.
pub fn try_gmres_ir_complex32(
    matrix: &CsrMatrix<Complex64>,
    b: &[Complex64],
    config: &FgmresConfig<f64>,
) -> Result<FgmresSolution<Complex64>, crate::traits::SolverError> {
    try_gmres_ir_lu::<Complex64, Complex32>(matrix, b, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;
    use crate::iterative::fgmres_preconditioned;
    use crate::preconditioners::DiagonalPreconditioner;
    use crate::sparse::CsrMatrix;
    use crate::traits::AccuracyStatus;
    use approx::assert_relative_eq;
    use num_complex::{Complex32, Complex64};
    use num_traits::{Float, Zero};

    fn adjoint_property_vector(len: usize, mut seed: u64) -> Vec<Complex64> {
        (0..len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let re = ((seed >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let im = ((seed >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
                Complex64::new(re, im)
            })
            .collect()
    }

    #[test]
    fn mixed_precision_preconditioner_recovers_poisoned_buffer_pool() {
        let matrix = CsrMatrix::<f32>::identity(2);
        let low_preconditioner = DiagonalPreconditioner::from_csr(&matrix);
        let mixed = MixedPrecisionPreconditioner::<f64, f32>::new(Box::new(low_preconditioner));

        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = mixed
                .buffers
                .lock()
                .expect("mixed-precision buffer pool should start healthy");
            panic!("inject buffer-pool poisoning");
        }));
        assert!(poisoned.is_err());

        let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mixed.apply(&[2.0_f64, -3.0])
        }))
        .expect("poisoned mixed-precision buffer pool must not panic");
        assert_eq!(applied, vec![2.0, -3.0]);
    }

    #[test]
    fn mixed_precision_preconditioner_plain_adjoint_matches_low_precision() {
        let low_preconditioner =
            DiagonalPreconditioner::<f32>::from_inverse_diagonal(vec![0.5, -0.25]);
        let mixed = MixedPrecisionPreconditioner::<f64, f32>::new(Box::new(low_preconditioner));
        let input = [4.0_f64, -8.0];

        assert_eq!(mixed.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(mixed.hermitian_capability(), AdjointCapability::Exact);
        assert_eq!(mixed.try_apply_transpose(&input).unwrap(), vec![2.0, 2.0]);
        assert_eq!(mixed.try_apply_hermitian(&input).unwrap(), vec![2.0, 2.0]);
    }

    #[test]
    fn mixed_precision_preconditioner_block_adjoint_matches_column_actions() {
        let low_preconditioner = DiagonalPreconditioner::<Complex32>::from_inverse_diagonal(vec![
            Complex32::new(0.5, 0.25),
            Complex32::new(-0.25, 0.5),
            Complex32::new(0.75, -0.125),
        ]);
        let mixed =
            MixedPrecisionPreconditioner::<Complex64, Complex32>::new(Box::new(low_preconditioner));
        let input = crate::dense::DenseMatrix::from_row_major_vec(
            3,
            2,
            vec![
                Complex64::new(1.0, -0.5),
                Complex64::new(0.25, 2.0),
                Complex64::new(-2.0, 1.0),
                Complex64::new(1.5, -0.75),
                Complex64::new(0.25, 0.75),
                Complex64::new(-0.5, 0.2),
            ],
        );
        let mut transpose = crate::dense::DenseMatrix::zeros(3, 2);
        mixed
            .try_apply_transpose_block(&input, &mut transpose)
            .expect("mixed block transpose action");
        let mut hermitian = crate::dense::DenseMatrix::zeros(3, 2);
        mixed
            .try_apply_hermitian_block(&input, &mut hermitian)
            .expect("mixed block Hermitian action");

        for column in 0..input.ncols() {
            let rhs = input.column_to_vec(column);
            let expected_transpose = mixed
                .try_apply_transpose(&rhs)
                .expect("mixed scalar transpose action");
            let expected_hermitian = mixed
                .try_apply_hermitian(&rhs)
                .expect("mixed scalar Hermitian action");
            for row in 0..input.nrows() {
                assert!((transpose[[row, column]] - expected_transpose[row]).norm() < 1.0e-5);
                assert!((hermitian[[row, column]] - expected_hermitian[row]).norm() < 1.0e-5);
            }
        }
    }

    #[test]
    fn mixed_precision_preconditioner_complex_hermitian_matches_low_precision() {
        let low_preconditioner = DiagonalPreconditioner::<Complex32>::from_inverse_diagonal(vec![
            Complex32::new(0.5, 0.25),
            Complex32::new(-0.25, 0.5),
        ]);
        let mixed =
            MixedPrecisionPreconditioner::<Complex64, Complex32>::new(Box::new(low_preconditioner));
        let input = [Complex64::new(4.0, -2.0), Complex64::new(-8.0, 4.0)];
        let transpose = mixed.try_apply_transpose(&input).unwrap();
        let hermitian = mixed.try_apply_hermitian(&input).unwrap();

        assert_relative_eq!(transpose[0].re, 2.5, epsilon = 1e-6);
        assert_relative_eq!(transpose[0].im, 0.0, epsilon = 1e-6);
        assert_relative_eq!(transpose[1].re, 0.0, epsilon = 1e-6);
        assert_relative_eq!(transpose[1].im, -5.0, epsilon = 1e-6);
        assert_relative_eq!(hermitian[0].re, 1.5, epsilon = 1e-6);
        assert_relative_eq!(hermitian[0].im, -2.0, epsilon = 1e-6);
        assert_relative_eq!(hermitian[1].re, 4.0, epsilon = 1e-6);
        assert_relative_eq!(hermitian[1].im, 3.0, epsilon = 1e-6);

        for seed in 0..16_u64 {
            let r = adjoint_property_vector(2, seed.wrapping_add(1));
            let s = adjoint_property_vector(2, seed.wrapping_add(101));
            let mr = mixed.apply(&r);
            let transpose_s = mixed.try_apply_transpose(&s).unwrap();
            let hermitian_s = mixed.try_apply_hermitian(&s).unwrap();
            let transpose_lhs: Complex64 = mr.iter().zip(&s).map(|(a, b)| *a * *b).sum();
            let transpose_rhs: Complex64 = r.iter().zip(&transpose_s).map(|(a, b)| *a * *b).sum();
            let hermitian_lhs: Complex64 = mr.iter().zip(&s).map(|(a, b)| a.conj() * *b).sum();
            let hermitian_rhs: Complex64 =
                r.iter().zip(&hermitian_s).map(|(a, b)| a.conj() * *b).sum();
            assert!(
                (transpose_lhs - transpose_rhs).norm() < 1e-5,
                "transpose seed {seed}"
            );
            assert!(
                (hermitian_lhs - hermitian_rhs).norm() < 1e-5,
                "Hermitian seed {seed}"
            );
        }
    }

    #[test]
    fn mixed_precision_preconditioner_adjoint_applies_matrix_scaling_in_reverse_order() {
        let low_preconditioner = DiagonalPreconditioner::<Complex32>::from_inverse_diagonal(vec![
            Complex32::new(0.5, 0.25),
            Complex32::new(-0.25, 0.5),
        ]);
        let mixed = MixedPrecisionPreconditioner::<Complex64, Complex32>::with_matrix_scaling(
            Box::new(low_preconditioner),
            Some(FactorizationScaling {
                row_scale: vec![2.0, 3.0],
                col_scale: vec![5.0, 7.0],
                scalar: 11.0,
            }),
        );
        let input = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        let transpose = mixed.try_apply_transpose(&input).unwrap();
        let hermitian = mixed.try_apply_hermitian(&input).unwrap();

        // M^T = Dr · B^-T · (s Dc), with real diagonal scaling factors.
        assert_relative_eq!(transpose[0].re, 0.0, epsilon = 1e-5);
        assert_relative_eq!(transpose[0].im, 137.5, epsilon = 1e-5);
        assert_relative_eq!(transpose[1].re, -288.75, epsilon = 1e-5);
        assert_relative_eq!(transpose[1].im, -577.5, epsilon = 1e-5);
        // Hermitian uses the conjugate low-precision diagonal action.
        assert_relative_eq!(hermitian[0].re, 110.0, epsilon = 1e-5);
        assert_relative_eq!(hermitian[0].im, 82.5, epsilon = 1e-5);
        assert_relative_eq!(hermitian[1].re, 635.25, epsilon = 1e-5);
        assert_relative_eq!(hermitian[1].im, 115.5, epsilon = 1e-5);
    }

    #[test]
    fn mixed_precision_preconditioner_rejects_unsupported_adjoint() {
        struct ForwardOnly;

        impl Preconditioner<f32> for ForwardOnly {
            fn apply(&self, r: &[f32]) -> Vec<f32> {
                r.to_vec()
            }
        }

        let mixed = MixedPrecisionPreconditioner::<f64, f32>::new(Box::new(ForwardOnly));
        assert_eq!(mixed.transpose_capability(), AdjointCapability::Unsupported);
        assert_eq!(mixed.hermitian_capability(), AdjointCapability::Unsupported);

        let transpose_error = mixed.try_apply_transpose(&[1.0]).unwrap_err();
        assert!(matches!(
            transpose_error,
            SolverError::UnsupportedOperation {
                operation: "mixed-precision preconditioner transpose",
                ..
            }
        ));
        let hermitian_error = mixed.try_apply_hermitian(&[1.0]).unwrap_err();
        assert!(matches!(
            hermitian_error,
            SolverError::UnsupportedOperation {
                operation: "mixed-precision preconditioner Hermitian",
                ..
            }
        ));
    }

    // ---------------------------------------------------------------
    // PrecisionCast roundtrip tests
    // ---------------------------------------------------------------

    #[test]
    fn test_cast_roundtrip_f64_f32() {
        let values: Vec<f64> = vec![
            1.0,
            -std::f64::consts::PI,
            1e10,
            1e-10,
            0.0,
            std::f64::consts::PI,
        ];
        for &v in &values {
            let down = <f64 as PrecisionCast<f32>>::cast_down(v);
            let up = <f64 as PrecisionCast<f32>>::cast_up(down);
            // f32 has ~7 decimal digits of precision
            assert_relative_eq!(v, up, epsilon = (v.abs() * 1e-6).max(1e-6));
        }
    }

    #[test]
    fn test_cast_roundtrip_complex64_complex32() {
        let values: Vec<Complex64> = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.5, 2.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(1e5, -1e5),
        ];
        for &v in &values {
            let down = <Complex64 as PrecisionCast<Complex32>>::cast_down(v);
            let up = <Complex64 as PrecisionCast<Complex32>>::cast_up(down);
            assert_relative_eq!(v.re, up.re, epsilon = (v.re.abs() * 1e-6).max(1e-6));
            assert_relative_eq!(v.im, up.im, epsilon = (v.im.abs() * 1e-6).max(1e-6));
        }
    }

    // ---------------------------------------------------------------
    // f16 cast-down scaling (Higham–Pranesh max-norm scaling)
    // ---------------------------------------------------------------

    #[test]
    fn test_cast_f16_scaling_metadata() {
        // Narrow-range types opt in to max-norm scaling; wide-range
        // types keep the raw (unscaled) cast path.
        assert!(<f64 as PrecisionCast<f16>>::down_cast_max_abs().is_some());
        assert!(<f32 as PrecisionCast<f16>>::down_cast_max_abs().is_some());
        assert!(<f64 as PrecisionCast<f32>>::down_cast_max_abs().is_none());
        assert!(<Complex64 as PrecisionCast<Complex32>>::down_cast_max_abs().is_none());

        // The defect being fixed: raw f16 casts overflow/underflow
        // silently well inside the dynamic range of typical residuals.
        assert!(f16::from_f64(1e6).is_infinite());
        assert_eq!(f64::from(f16::from_f64(1e-9)), 0.0);
    }

    #[test]
    fn test_mixed_precision_f16_large_dynamic_range_residual() {
        // Residual entries span 1e-6..1e6; a raw f16 cast of the 1e6
        // entry overflows to +inf. Max-norm scaling must keep every
        // significant component intact and produce no garbage.
        let n = 8;
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 4.0;
        }
        let matrix_f64 = CsrMatrix::from_dense(&dense, 1e-15);
        let matrix_f16: CsrMatrix<f16> = matrix_f64.cast::<f16>();
        let precond_f16 = DiagonalPreconditioner::from_csr(&matrix_f16);

        let r: Vec<f64> = vec![1e-6, -1e-4, 1e-2, 1.0, -1e2, 1e4, 1e6, 0.0];
        let mixed = MixedPrecisionPreconditioner::<f64, f16>::new(Box::new(precond_f16));
        let y = mixed.apply(&r);

        // No overflow garbage: every component must be finite.
        assert!(
            y.iter().all(|v| v.is_finite()),
            "overflow garbage in f16 preconditioner output: {y:?}"
        );

        // Exact result is z = r / 4 (diagonal preconditioner).
        for (i, &ri) in r.iter().enumerate() {
            let expected = ri / 4.0;
            if ri.abs() >= 1e-2 {
                // Components inside the scaled f16 range are recovered to
                // f16 relative accuracy (~1e-3; allow 5e-2 headroom).
                let rel = (y[i] - expected).abs() / expected.abs();
                assert!(
                    rel < 5e-2,
                    "component {i}: y={:e}, expected={expected:e}, rel_err={rel:e}",
                    y[i]
                );
            } else {
                // Components >~8 orders below the inf-norm may underflow
                // to zero in f16; that is acceptable, garbage is not.
                let abs_err = (y[i] - expected).abs();
                assert!(
                    abs_err < 1e-5,
                    "component {i}: y={:e}, expected={expected:e}",
                    y[i]
                );
            }
        }
    }

    #[test]
    fn test_mixed_precision_fgmres_f64_f16_converges() {
        // End-to-end: f64 FGMRES with an f16 diagonal preconditioner on a
        // system whose right-hand side spans ~5 orders of magnitude.
        let n = 20;
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 4.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -1.0;
            }
        }

        let matrix_f64 = CsrMatrix::from_dense(&dense, 1e-15);

        // Solution with wide dynamic range: 1e-2 .. 1e3
        let x_exact: Vec<f64> = (0..n).map(|i| 10f64.powi((i % 6) as i32 - 2)).collect();
        let b: Vec<f64> = matrix_f64.matvec(&x_exact);

        let matrix_f16: CsrMatrix<f16> = matrix_f64.cast::<f16>();
        let precond_f16 = DiagonalPreconditioner::from_csr(&matrix_f16);

        let config = FgmresConfig {
            max_iterations: 200,
            restart: 30,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = mixed_precision_solve(&matrix_f64, &b, Box::new(precond_f16), &config);

        assert!(
            solution.converged,
            "f64/f16 mixed-precision FGMRES should converge, got residual = {:e}",
            solution.residual
        );
        assert_eq!(solution.attainable_accuracy, AccuracyStatus::Attained);
        assert!(solution.estimated_residual.is_finite());
        assert!(solution.backward_error.is_finite());

        let err: f64 = crate::vec_ops::vec_sub(&solution.x, &x_exact)
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        let x_norm: f64 = x_exact.iter().map(|e| e * e).sum::<f64>().sqrt();
        assert!(
            err / x_norm < 1e-8,
            "Solution should be f64-accurate, got relative error = {:e}",
            err / x_norm
        );
    }

    // ---------------------------------------------------------------
    // CsrMatrix::cast test
    // ---------------------------------------------------------------

    #[test]
    fn test_csr_matrix_cast_f64_to_f32() {
        let dense = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let matrix_f64 = CsrMatrix::from_dense(&dense, 1e-15);
        let matrix_f32: CsrMatrix<f32> = matrix_f64.cast::<f32>();

        assert_eq!(matrix_f32.num_rows, 2);
        assert_eq!(matrix_f32.num_cols, 2);
        assert_eq!(matrix_f32.nnz(), matrix_f64.nnz());
        assert_relative_eq!(matrix_f32.get(0, 0), 4.0_f32, epsilon = 1e-6);
        assert_relative_eq!(matrix_f32.get(0, 1), 1.0_f32, epsilon = 1e-6);
        assert_relative_eq!(matrix_f32.get(1, 0), 1.0_f32, epsilon = 1e-6);
        assert_relative_eq!(matrix_f32.get(1, 1), 3.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_csr_matrix_cast_complex64_to_complex32() {
        let dense = array![
            [Complex64::new(4.0, 1.0), Complex64::new(1.0, 0.5)],
            [Complex64::new(1.0, -0.5), Complex64::new(3.0, -1.0)],
        ];
        let matrix_c64 = CsrMatrix::from_dense(&dense, 1e-15);
        let matrix_c32: CsrMatrix<Complex32> = matrix_c64.cast::<Complex32>();

        assert_eq!(matrix_c32.num_rows, 2);
        assert_eq!(matrix_c32.num_cols, 2);
        assert_eq!(matrix_c32.nnz(), matrix_c64.nnz());

        let v = matrix_c32.get(0, 0);
        assert_relative_eq!(v.re, 4.0_f32, epsilon = 1e-6);
        assert_relative_eq!(v.im, 1.0_f32, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------
    // Mixed-precision FGMRES convergence test (f64 outer, f32 inner)
    // ---------------------------------------------------------------

    #[test]
    fn test_mixed_precision_fgmres_f64_f32() {
        // Build a well-conditioned diagonally dominant system
        let n = 20;
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 4.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -1.0;
            }
        }

        let matrix_f64 = CsrMatrix::from_dense(&dense, 1e-15);

        // Known solution: x = [1, 2, 3, ..., n]
        let x_exact: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = matrix_f64.matvec(&x_exact);

        // Build preconditioner in f32
        let matrix_f32: CsrMatrix<f32> = matrix_f64.cast::<f32>();
        let precond_f32 = DiagonalPreconditioner::from_csr(&matrix_f32);

        let config = FgmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = mixed_precision_solve(&matrix_f64, &b, Box::new(precond_f32), &config);

        assert!(
            solution.converged,
            "Mixed-precision FGMRES should converge, got residual = {:e}",
            solution.residual
        );

        // Verify f64-quality accuracy despite f32 preconditioner
        let error: f64 = crate::vec_ops::vec_sub(&solution.x, &x_exact)
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-8,
            "Solution should be f64-accurate, got error = {:e}",
            error
        );
    }

    #[test]
    fn test_mixed_precision_fgmres_complex64_complex32() {
        // Build a complex diagonally dominant system
        let n = 15;
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(5.0, 1.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.2);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, -0.2);
            }
        }

        let matrix_c64 = CsrMatrix::from_dense(&dense, 1e-15);

        // Known solution
        let x_exact: Vec<Complex64> = (1..=n)
            .map(|i| Complex64::new(i as f64, -(i as f64) * 0.5))
            .collect();
        let b: Vec<Complex64> = matrix_c64.matvec(&x_exact);

        // Build preconditioner in Complex32
        let matrix_c32: CsrMatrix<Complex32> = matrix_c64.cast::<Complex32>();
        let precond_c32 = DiagonalPreconditioner::from_csr(&matrix_c32);

        let config = FgmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = mixed_precision_solve(&matrix_c64, &b, Box::new(precond_c32), &config);

        assert!(
            solution.converged,
            "Mixed-precision FGMRES (complex) should converge, got residual = {:e}",
            solution.residual
        );

        // Verify f64-quality accuracy
        let error: f64 = crate::vec_ops::vec_sub(&solution.x, &x_exact)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-8,
            "Solution should be Complex64-accurate, got error = {:e}",
            error
        );
    }

    #[test]
    fn test_mixed_precision_solution_quality_vs_pure_f64() {
        // Verify that mixed-precision gives the same quality as pure f64
        let n = 10;
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 4.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -1.0;
            }
        }

        let matrix_f64 = CsrMatrix::from_dense(&dense, 1e-15);
        let x_exact: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = matrix_f64.matvec(&x_exact);

        let config = FgmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        // Pure f64 solve (with f64 Jacobi preconditioner)
        let precond_f64 = DiagonalPreconditioner::from_csr(&matrix_f64);
        let sol_pure = fgmres_preconditioned(&matrix_f64, &precond_f64, &b, None, &config);

        // Mixed-precision solve (f32 preconditioner)
        let matrix_f32: CsrMatrix<f32> = matrix_f64.cast::<f32>();
        let precond_f32 = DiagonalPreconditioner::from_csr(&matrix_f32);
        let sol_mixed = mixed_precision_solve(&matrix_f64, &b, Box::new(precond_f32), &config);

        assert!(sol_pure.converged, "Pure f64 should converge");
        assert!(sol_mixed.converged, "Mixed precision should converge");

        // Both should achieve f64-quality solutions
        let error_pure: f64 = crate::vec_ops::vec_sub(&sol_pure.x, &x_exact)
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        let error_mixed: f64 = crate::vec_ops::vec_sub(&sol_mixed.x, &x_exact)
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();

        assert!(
            error_pure < 1e-10,
            "Pure f64 error too large: {:e}",
            error_pure
        );
        assert!(
            error_mixed < 1e-10,
            "Mixed precision error too large: {:e}",
            error_mixed
        );
    }

    // ---------------------------------------------------------------
    // GMRES-IR (Carson–Higham) tests
    // ---------------------------------------------------------------

    /// Exact relative residual ||b - A x|| / ||b||, generic over ComplexField.
    fn exact_rel_residual<T: ComplexField>(a: &CsrMatrix<T>, x: &[T], b: &[T]) -> T::Real {
        let ax = a.matvec(x);
        let r = crate::vec_ops::vec_sub(b, &ax);
        let num = r
            .iter()
            .map(|e| e.norm_sqr())
            .fold(T::Real::zero(), |acc, v| acc + v)
            .sqrt();
        let den = b
            .iter()
            .map(|e| e.norm_sqr())
            .fold(T::Real::zero(), |acc, v| acc + v)
            .sqrt();
        num / den
    }

    /// Nonsymmetric 1D convection–diffusion matrix (diag 4, lower -2, upper -1).
    fn convection_diffusion_f64(n: usize) -> CsrMatrix<f64> {
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 4.0;
            if i > 0 {
                dense[[i, i - 1]] = -2.0;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -1.0;
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn fgmres_test_config() -> FgmresConfig<f64> {
        FgmresConfig {
            max_iterations: 200,
            restart: 50,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        }
    }

    #[test]
    fn test_gmres_ir_f32_nonsymmetric_reaches_f64_accuracy() {
        let n = 64;
        let matrix_f64 = convection_diffusion_f64(n);

        let x_exact: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = matrix_f64.matvec(&x_exact);

        let config = fgmres_test_config();
        let solution = gmres_ir_f32(&matrix_f64, &b, &config).expect("f32 LU factorization");

        assert!(solution.converged, "GMRES-IR should converge");

        // Exact residual (not the FGMRES estimate) must be at f64 level
        // even though the factorization was computed in f32.
        let rel_res = exact_rel_residual(&matrix_f64, &solution.x, &b);
        assert!(
            rel_res < 1e-10,
            "GMRES-IR exact relative residual too large: {rel_res:e}"
        );
    }

    /// Nonsymmetric 2D convection–diffusion 5-point stencil on a
    /// `grid`×`grid` mesh (natural row-major ordering). Unlike the 1D case,
    /// exact LU fills in here, so ILU(0) is genuinely incomplete.
    fn convection_diffusion_2d_f64(grid: usize) -> CsrMatrix<f64> {
        let n = grid * grid;
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for row in 0..grid {
            for col in 0..grid {
                let i = row * grid + col;
                dense[[i, i]] = 4.0;
                if col > 0 {
                    dense[[i, i - 1]] = -2.0; // upwind x-neighbor
                }
                if col + 1 < grid {
                    dense[[i, i + 1]] = -1.0; // downwind x-neighbor
                }
                if row > 0 {
                    dense[[i, i - grid]] = -1.0; // y-neighbor
                }
                if row + 1 < grid {
                    dense[[i, i + grid]] = -1.0;
                }
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    #[test]
    fn test_gmres_ir_f32_iteration_count_vs_pure_f64_ilu() {
        // Same 2D convection–diffusion system solved two ways:
        //  - GMRES-IR: f64 FGMRES preconditioned by an *f32 sparse LU* of A
        //  - baseline: pure f64 FGMRES preconditioned by f64 ILU(0)
        //
        // On a 5-point stencil exact LU fills in, so ILU(0) is genuinely
        // incomplete and needs many iterations, while the f32 LU factor is a
        // near-exact inverse (M·A ≈ I + O(cond(A)·u_f32)) and GMRES-IR
        // converges in ~2 matvecs to the same f64-level residual.
        // Measured on the 12×12 case: GMRES-IR 3 matvecs (rel_res 6e-16)
        // vs ILU(0) 40 matvecs (rel_res 2e-13).
        let grid = 12;
        let n = grid * grid;
        let matrix_f64 = convection_diffusion_2d_f64(grid);

        let x_exact: Vec<f64> = (1..=n).map(|i| (i as f64).sin()).collect();
        let b: Vec<f64> = matrix_f64.matvec(&x_exact);

        let config = fgmres_test_config();

        let ir = gmres_ir_f32(&matrix_f64, &b, &config).expect("f32 LU factorization");

        let ilu_f64 = IluPreconditioner::from_csr(&matrix_f64);
        let baseline = fgmres_preconditioned(&matrix_f64, &ilu_f64, &b, None, &config);

        assert!(ir.converged, "GMRES-IR should converge");
        assert!(baseline.converged, "f64 FGMRES + ILU(0) should converge");

        let ir_res = exact_rel_residual(&matrix_f64, &ir.x, &b);
        let base_res = exact_rel_residual(&matrix_f64, &baseline.x, &b);
        println!(
            "GMRES-IR(f32 LU): {} matvecs, rel_res={ir_res:e} | f64 FGMRES+ILU(0): {} matvecs, rel_res={base_res:e}",
            ir.iterations, baseline.iterations
        );

        assert!(ir_res < 1e-10, "GMRES-IR residual too large: {ir_res:e}");
        assert!(
            ir.iterations <= baseline.iterations,
            "GMRES-IR ({} matvecs) should be no worse than f64 ILU(0) baseline ({} matvecs)",
            ir.iterations,
            baseline.iterations
        );
    }

    #[test]
    fn test_gmres_ir_complex32_factor() {
        // Nonsymmetric complex system (Helmholtz-like shifted tridiagonal).
        let n = 48;
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.5);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.5, 0.3);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, -0.3);
            }
        }
        let matrix_c64 = CsrMatrix::from_dense(&dense, 1e-15);

        let x_exact: Vec<Complex64> = (1..=n)
            .map(|i| Complex64::new(i as f64, (i as f64) * 0.25))
            .collect();
        let b: Vec<Complex64> = matrix_c64.matvec(&x_exact);

        let config = fgmres_test_config();
        let solution =
            gmres_ir_complex32(&matrix_c64, &b, &config).expect("Complex32 LU factorization");

        assert!(solution.converged, "Complex GMRES-IR should converge");

        let rel_res = exact_rel_residual(&matrix_c64, &solution.x, &b);
        assert!(
            rel_res < 1e-10,
            "Complex GMRES-IR exact relative residual too large: {rel_res:e}"
        );
    }

    #[test]
    fn test_gmres_ir_ilu_f32() {
        // ILU-based GMRES-IR: O(nnz) f32 ILU(0) factor instead of exact LU.
        // Needs more outer iterations but must still reach f64 accuracy.
        let n = 64;
        let matrix_f64 = convection_diffusion_f64(n);

        let x_exact: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = matrix_f64.matvec(&x_exact);

        let config = fgmres_test_config();
        let solution = gmres_ir_ilu::<f64, f32>(&matrix_f64, &b, &IluConfig::default(), &config);

        assert!(solution.converged, "ILU-based GMRES-IR should converge");

        let rel_res = exact_rel_residual(&matrix_f64, &solution.x, &b);
        assert!(
            rel_res < 1e-10,
            "ILU-based GMRES-IR exact relative residual too large: {rel_res:e}"
        );
    }

    // ---------------------------------------------------------------
    // Matrix-side scaling for the low-precision cast (f16 factors)
    // ---------------------------------------------------------------

    /// Badly scaled convection–diffusion matrix `Dr · K · Dc` whose entries
    /// span ~1e±6: smooth row/column diagonal scalings (amplitude 1e±4 per
    /// side, one sinusoid period over the matrix) of the well-conditioned
    /// diagonally dominant 1D stencil `K` (diag 6, lower −2, upper −1).
    /// The underlying operator is benign — the equilibrated matrix is
    /// diagonally dominant with a moderate condition number — only the
    /// scaling makes a raw narrow-range cast overflow/underflow. (Smooth
    /// scaling keeps adjacent row/column ratios ≤ ~2.5 so the equilibrated
    /// matrix stays diagonally dominant; measured entry range for n = 64
    /// is 1.7e-5 .. 3.3e5.)
    fn badly_scaled_convection_diffusion_f64(n: usize) -> CsrMatrix<f64> {
        let d_row: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(4.0 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()))
            .collect();
        let d_col: Vec<f64> = (0..n)
            .map(|j| {
                10f64.powf(4.0 * (2.0 * std::f64::consts::PI * j as f64 / n as f64 + 0.3).cos())
            })
            .collect();
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 6.0;
            if i > 0 {
                dense[[i, i - 1]] = -2.0;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -1.0;
            }
            for j in 0..n {
                dense[[i, j]] *= d_row[i] * d_col[j];
            }
        }
        CsrMatrix::from_dense(&dense, 1e-300)
    }

    #[test]
    fn test_cast_for_factorization_f32_is_raw_bit_identical_cast() {
        // Wide-range targets must keep the raw cast path: no equilibration,
        // bit-identical values. This pins requirement "f32 path unchanged"
        // at the unit level, below the end-to-end IR tests.
        let matrix = convection_diffusion_f64(16);
        let (low, scaling) = cast_for_factorization::<f64, f32>(&matrix);
        assert!(scaling.is_none(), "f32 target must not be scaled");

        let raw: CsrMatrix<f32> = matrix.cast::<f32>();
        assert_eq!(low.row_ptrs, raw.row_ptrs);
        assert_eq!(low.col_indices, raw.col_indices);
        assert_eq!(low.values, raw.values, "f32 cast must be bit-identical");
    }

    #[test]
    fn test_cast_for_factorization_f16_equilibrates_into_range() {
        // Entries span ~1e±6 (measured 1.7e-5 .. 3.3e5): a raw f16 cast
        // overflows the large end to ±inf and underflows the small end
        // below the f16 normal range. The equilibrated cast must land
        // every significant entry inside the f16 range with the max
        // mapped to F16_SCALE_TARGET.
        let matrix = badly_scaled_convection_diffusion_f64(64);
        let max_abs = matrix.values.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let min_abs = matrix
            .values
            .iter()
            .map(|v| v.abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            max_abs > f64::from(f16::MAX),
            "test matrix should overflow f16: {max_abs:e}"
        );
        assert!(
            min_abs < 6e-5,
            "test matrix should underflow f16: {min_abs:e}"
        );
        assert!(f16::from_f64(max_abs).is_infinite());

        let (low, scaling) = cast_for_factorization::<f64, f16>(&matrix);
        let scaling = scaling.expect("f16 target must be equilibrated");
        assert_eq!(scaling.row_scale.len(), matrix.num_rows);
        assert_eq!(scaling.col_scale.len(), matrix.num_cols);
        assert_eq!(scaling.scalar, F16_SCALE_TARGET);

        let max_low = low
            .values
            .iter()
            .map(|v| f64::from(*v).abs())
            .fold(0.0, f64::max);
        assert!(
            (max_low - F16_SCALE_TARGET).abs() <= 0.5,
            "equilibrated max should map to {F16_SCALE_TARGET}, got {max_low}"
        );
        // No overflow garbage and the sparsity pattern is preserved. (The
        // equilibrated diagonal can legitimately be small — max-abs
        // equilibration without MC64 matching does not enforce diagonal
        // dominance; partial pivoting in the LU handles that.)
        assert!(low.values.iter().all(|v| v.is_finite()));
        assert_eq!(low.nnz(), matrix.nnz());
    }

    #[test]
    fn test_gmres_ir_f16_wide_dynamic_range_matrix() {
        // End-to-end: GMRES-IR with f16 sparse LU and ILU factors of a
        // matrix whose entries span ~1e±6 (measured 1.7e-5 .. 3.3e5). The
        // raw cast overflows/underflows at factorization time (pinned
        // above); with the equilibrating cast the f16 factor is a
        // near-exact inverse of the equilibrated matrix and the outer f64
        // FGMRES must still reach f64-level accuracy, because the scalings
        // cancel exactly in the wrapped preconditioner apply.
        //
        // Measured: f16 LU converges in 4 iterations to rel_res ≈ 1e-13,
        // f16 ILU(0) in 27 iterations to ≈ 9e-13. Note the raw f32 LU
        // path (`gmres_ir_f32`, no equilibration) *fails* on this matrix
        // with `SparseLuError::SingularMatrix` — its `eps·n·||A||` pivot
        // guard trips on the badly scaled entries — so the equilibrated
        // f16 path is not merely as good as the wider raw cast here, it
        // is strictly more robust.
        let n = 64;
        let matrix_f64 = badly_scaled_convection_diffusion_f64(n);

        let x_exact: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = matrix_f64.matvec(&x_exact);

        let config = fgmres_test_config();

        let solution = gmres_ir_lu::<f64, f16>(&matrix_f64, &b, &config)
            .expect("f16 LU factorization of the equilibrated matrix");
        assert!(
            solution.converged,
            "f16 GMRES-IR (LU) should converge on the badly scaled matrix, residual = {:e}",
            solution.residual
        );
        let rel_res = exact_rel_residual(&matrix_f64, &solution.x, &b);
        assert!(
            rel_res < 1e-10,
            "f16 GMRES-IR (LU) exact relative residual too large: {rel_res:e}"
        );

        let solution_ilu =
            gmres_ir_ilu::<f64, f16>(&matrix_f64, &b, &IluConfig::default(), &config);
        assert!(
            solution_ilu.converged,
            "f16 GMRES-IR (ILU) should converge on the badly scaled matrix, residual = {:e}",
            solution_ilu.residual
        );
        let rel_res_ilu = exact_rel_residual(&matrix_f64, &solution_ilu.x, &b);
        assert!(
            rel_res_ilu < 1e-10,
            "f16 GMRES-IR (ILU) exact relative residual too large: {rel_res_ilu:e}"
        );
    }

    /// Dense SPD matrix `Q·Λ·Qᵀ` with orthonormal DCT-II `Q` and geometric
    /// spectrum `[1, kappa]`, so `cond(A) = kappa` exactly and two-sided
    /// equilibration cannot reduce it (all row/column maxima are nearly
    /// equal). Used to probe the Carson–Higham boundary
    /// `cond(A) ≲ u_low⁻¹ ≈ 2048` for f16 factors.
    fn spd_with_spectrum_f64(n: usize, kappa: f64) -> CsrMatrix<f64> {
        let q = |i: usize, k: usize| {
            if k == 0 {
                1.0 / (n as f64).sqrt()
            } else {
                (2.0 / n as f64).sqrt()
                    * (std::f64::consts::PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos()
            }
        };
        let lambda = |k: usize| kappa.powf(k as f64 / (n - 1) as f64);
        let mut dense = crate::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            for j in 0..n {
                let mut aij = 0.0;
                for k in 0..n {
                    aij += lambda(k) * q(i, k) * q(j, k);
                }
                dense[[i, j]] = aij;
            }
        }
        CsrMatrix::from_dense(&dense, 1e-300)
    }

    #[test]
    fn low_precision_failure_count_records_each_checked_boundary_failure() {
        let low = DiagonalPreconditioner::<f32>::from_inverse_diagonal(vec![1.0]);
        let preconditioner = MixedPrecisionPreconditioner::<f64, f32>::new(Box::new(low));

        let mut output = vec![0.0_f64];
        Preconditioner::apply_into(&preconditioner, &[f64::NAN], &mut output);
        assert_eq!(preconditioner.low_precision_failure_count(), 1);

        Preconditioner::apply_into(&preconditioner, &[f64::INFINITY], &mut output);
        assert_eq!(preconditioner.low_precision_failure_count(), 2);
        assert!(preconditioner.has_low_precision_failure());
    }

    #[test]
    fn test_gmres_ir_f16_cond_boundary() {
        // Carson–Higham for f16 factors: u_f16 = 2^-11 ≈ 4.9e-4, so the
        // theory boundary is cond(A) ≲ u_low⁻¹ ≈ 2e3, measured on the
        // equilibrated matrix (here equilibration cannot reduce cond: the
        // DCT-II rows/columns all have nearly equal maxima).
        //
        // Measured behavior (n = 128, restart 30):
        //  - cond = 1e3 (cond·u ≈ 0.5, just inside the boundary): f16
        //    ILU-based GMRES-IR converges in 358 iterations to an exact
        //    relative residual of 9.9e-13 — f64-level accuracy.
        //  - cond = 1e5 (cond·u ≈ 50, past the boundary): convergence
        //    degrades to a slow grind (≈9700 iterations to reach 1e-12);
        //    within a 1000-iteration budget it does not converge. This is
        //    the documented theory limit, not a bug; use f32 factors
        //    (u_low⁻¹ ≈ 1e7) for such systems.
        //  - The f16 *LU* variant uses the sparse LU's preconditioner mode
        //    (relaxed singularity guard + pivot repair + rhs normalization,
        //    see `SparseLuFactorization::factorize_for_preconditioner`): at
        //    cond = 1e3 it factorizes and converges in 10 iterations to
        //    ≈4.5e-13 — much faster than ILU(0), because the exact factor
        //    is a near-exact inverse. The strict direct-solver guard would
        //    reject the factorization with `Err(SingularMatrix)` before
        //    FGMRES even starts.
        let n = 128;
        let config = FgmresConfig {
            max_iterations: 1000,
            restart: 30,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let matrix_f64 = spd_with_spectrum_f64(n, 1e3);
        let x_exact: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b: Vec<f64> = matrix_f64.matvec(&x_exact);

        let solution_lu = gmres_ir_lu::<f64, f16>(&matrix_f64, &b, &config)
            .expect("f16 LU-IR in preconditioner mode must factorize at cond = 1e3");
        assert!(
            solution_lu.converged,
            "f16 GMRES-IR (LU) should converge at cond = 1e3, residual = {:e}",
            solution_lu.residual
        );
        let rel_res_lu = exact_rel_residual(&matrix_f64, &solution_lu.x, &b);
        assert!(
            rel_res_lu < 1e-10,
            "f16 GMRES-IR (LU) residual at cond = 1e3 too large: {rel_res_lu:e}"
        );

        let solution = gmres_ir_ilu::<f64, f16>(&matrix_f64, &b, &IluConfig::default(), &config);
        assert!(
            solution.converged,
            "f16 GMRES-IR (ILU) should converge at cond = 1e3 (inside the u⁻¹ boundary), \
             residual = {:e}",
            solution.residual
        );
        let rel_res = exact_rel_residual(&matrix_f64, &solution.x, &b);
        assert!(
            rel_res < 1e-10,
            "f16 GMRES-IR (ILU) residual at cond = 1e3 too large: {rel_res:e}"
        );

        // Two orders past the boundary: within the same iteration budget
        // the f16 factor is too crude an inverse and FGMRES stagnates.
        let matrix_hard = spd_with_spectrum_f64(n, 1e5);
        let b_hard: Vec<f64> = matrix_hard.matvec(&x_exact);
        let hard = gmres_ir_ilu::<f64, f16>(&matrix_hard, &b_hard, &IluConfig::default(), &config);
        let rel_res_hard = exact_rel_residual(&matrix_hard, &hard.x, &b_hard);
        println!(
            "f16 GMRES-IR (ILU) past the theory boundary (cond=1e5): converged={}, \
             iterations={}, rel_res={rel_res_hard:e}",
            hard.converged, hard.iterations
        );
        assert!(
            !hard.converged || hard.iterations > 4 * solution.iterations,
            "cond = 1e5 should converge markedly slower than cond = 1e3 ({} iters), \
             got converged={}, iters={}, rel_res={rel_res_hard:e}",
            solution.iterations,
            hard.converged,
            hard.iterations
        );
    }
    #[test]
    fn gmres_ir_promoted_residual_gate_rejects_projected_convergence() {
        fn midpoint_tolerance(result: &FgmresSolution<f64>) -> f64 {
            assert!(result.residual.is_finite());
            assert!(result.estimated_residual.is_finite());
            assert!(
                result.estimated_residual < result.residual,
                "fixture must expose projected residual optimism: estimated={:e}, true={:e}",
                result.estimated_residual,
                result.residual
            );
            (result.estimated_residual + result.residual) * 0.5
        }

        let n = 128;
        let matrix_lu = spd_with_spectrum_f64(n, 1e5);
        let x_exact: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b_lu = matrix_lu.matvec(&x_exact);
        let baseline_config = FgmresConfig {
            max_iterations: 80,
            restart: 30,
            tolerance: 1e-30,
            ..fgmres_test_config()
        };
        let baseline_lu = try_gmres_ir_lu::<f64, f16>(&matrix_lu, &b_lu, &baseline_config)
            .expect("f16 LU baseline");
        let gate_config = FgmresConfig {
            tolerance: midpoint_tolerance(&baseline_lu),
            ..baseline_config.clone()
        };
        let gated_lu = try_gmres_ir_lu::<f64, f16>(&matrix_lu, &b_lu, &gate_config)
            .expect("f16 LU promoted gate");
        assert!(!gated_lu.converged);
        assert_eq!(gated_lu.status, SolverStatus::MaxIterationsReached);
        assert!(gated_lu.estimated_residual < gate_config.tolerance);
        assert!(gated_lu.residual > gate_config.tolerance);
        assert_eq!(gated_lu.attainable_accuracy, AccuracyStatus::NotAttained);
        assert!(gated_lu.iterations > 0);

        let matrix_ilu = spd_with_spectrum_f64(n, 1e3);
        let b_ilu = matrix_ilu.matvec(&x_exact);
        let ilu_baseline_config = FgmresConfig {
            max_iterations: 1,
            restart: 30,
            tolerance: 1e-30,
            ..fgmres_test_config()
        };
        let baseline_ilu = try_gmres_ir_ilu::<f64, f16>(
            &matrix_ilu,
            &b_ilu,
            &IluConfig::default(),
            &ilu_baseline_config,
        )
        .expect("f16 ILU baseline");
        let ilu_gate_config = FgmresConfig {
            tolerance: midpoint_tolerance(&baseline_ilu),
            ..ilu_baseline_config.clone()
        };
        let gated_ilu = try_gmres_ir_ilu::<f64, f16>(
            &matrix_ilu,
            &b_ilu,
            &IluConfig::default(),
            &ilu_gate_config,
        )
        .expect("f16 ILU promoted gate");
        assert!(!gated_ilu.converged);
        assert_eq!(gated_ilu.status, SolverStatus::MaxIterationsReached);
        assert!(gated_ilu.estimated_residual < ilu_gate_config.tolerance);
        assert!(gated_ilu.residual > ilu_gate_config.tolerance);
        assert_eq!(gated_ilu.attainable_accuracy, AccuracyStatus::NotAttained);
        assert!(gated_ilu.iterations > 0);
    }

    #[test]
    fn mixed_precision_gmres_result_validator_demotes_false_convergence() {
        let result = GmresSolution {
            x: vec![1.0_f64],
            iterations: 1,
            restarts: 0,
            residual: 1.0e-4,
            estimated_residual: 1.0e-12,
            backward_error: 1.0e-4,
            attainable_accuracy: AccuracyStatus::Attained,
            converged: true,
            status: SolverStatus::Converged,
            op_counts: KrylovOpCounts::default(),
        };
        let normalized = normalize_mixed_precision_gmres_result(result, 1.0e-6, "test")
            .expect("finite result should remain observable");
        assert!(!normalized.converged);
        assert_eq!(normalized.status, SolverStatus::MaxIterationsReached);
        assert_eq!(normalized.attainable_accuracy, AccuracyStatus::NotAttained);

        let inconsistent = GmresSolution {
            x: vec![1.0_f64],
            iterations: 1,
            restarts: 1,
            residual: 1.0e-12,
            estimated_residual: 1.0e-12,
            backward_error: 1.0e-12,
            attainable_accuracy: AccuracyStatus::Attained,
            converged: true,
            status: SolverStatus::MaxIterationsReached,
            op_counts: KrylovOpCounts::default(),
        };
        let normalized = normalize_mixed_precision_gmres_result(inconsistent, 1.0e-6, "test")
            .expect("finite result should remain observable");
        assert!(!normalized.converged);

        let status_only = GmresSolution {
            x: vec![1.0_f64],
            iterations: 1,
            restarts: 1,
            residual: 1.0e-4,
            estimated_residual: 1.0e-12,
            backward_error: 1.0e-4,
            attainable_accuracy: AccuracyStatus::Attained,
            converged: false,
            status: SolverStatus::Converged,
            op_counts: KrylovOpCounts::default(),
        };
        let normalized = normalize_mixed_precision_gmres_result(status_only, 1.0e-6, "test")
            .expect("finite result should remain observable");
        assert!(!normalized.converged);
        assert_eq!(normalized.status, SolverStatus::MaxIterationsReached);

        let non_finite = GmresSolution {
            x: vec![1.0_f64],
            iterations: 0,
            restarts: 0,
            residual: f64::NAN,
            estimated_residual: 0.0,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::NumericalError,
            op_counts: KrylovOpCounts::default(),
        };
        assert!(matches!(
            normalize_mixed_precision_gmres_result(non_finite, 1.0e-6, "test"),
            Err(SolverError::OperationFailed { .. })
        ));
    }
}
