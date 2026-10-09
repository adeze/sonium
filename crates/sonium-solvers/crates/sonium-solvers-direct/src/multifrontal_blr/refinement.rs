//! Iterative refinement on top of a (possibly compressed) multifrontal-BLR
//! factorisation.
//!
//! Given a factor `F ≈ A` and a right-hand side `b`, the loop is:
//!
//! ```text
//!   x_0 = F^{-1} · b
//!   for k = 0, 1, ... :
//!       r_k = b - A · x_k
//!       if ‖r_k‖ ≤ τ · ‖b‖:  stop
//!       δ_k = F^{-1} · r_k
//!       x_{k+1} = x_k + δ_k
//! ```
//!
//! When `F` was built with a loose BLR tolerance `ε ≈ 1e-4 .. 1e-6`, this
//! drives the residual down to machine precision in a handful of cheap
//! iterations — same cost model as MUMPS-BLR's IR loop. Works for both same-
//! precision IR (factor in `f64`, refine in `f64`) and is the building block
//! for mixed-precision IR (factor in `f32`, refine in `f64`).

use num_traits::{Float, FromPrimitive, ToPrimitive};
use sonium_solvers_core::PrecisionCast;
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, SolverError, SolverFailure, SolverResultContract, SolverStatus,
};
use sonium_solvers_sparse::CsrMatrix;
use thiserror::Error;

use super::{BlrError, MultifrontalBlrFactorization};

/// Configuration for [`IterativeRefinement`].
#[derive(Debug, Clone)]
pub struct IrConfig {
    /// Maximum refinement iterations. Default `8`.
    pub max_iterations: usize,
    /// Stop when `‖b − A·x‖ ≤ tolerance · ‖b‖`. Default `1e-12`.
    pub tolerance: f64,
    /// Stop early if the residual stops decreasing — guards against
    /// stagnation when `F` is too inaccurate to drive `‖r‖` below `tol`.
    /// Disabled if `None`. Default: `Some(0.999)` (stop when the new
    /// residual is ≥ this fraction of the previous one).
    pub stagnation_ratio: Option<f64>,
}

impl Default for IrConfig {
    fn default() -> Self {
        Self {
            max_iterations: 8,
            tolerance: 1e-12,
            stagnation_ratio: Some(0.999),
        }
    }
}

/// Checked iterative-refinement construction and solve failures.
#[derive(Debug, Error)]
pub enum IrError {
    #[error("matrix must be square ({rows} x {cols})")]
    MatrixNotSquare { rows: usize, cols: usize },
    #[error("factor dimension {factor} does not match matrix dimension {matrix}")]
    FactorDimensionMismatch { factor: usize, matrix: usize },
    #[error("right-hand side length {actual} does not match matrix dimension {expected}")]
    RhsDimensionMismatch { expected: usize, actual: usize },
    #[error("matrix entry at storage index {0} is non-finite")]
    NonFiniteMatrix(usize),
    #[error("right-hand side at index {0} is non-finite")]
    NonFiniteRhs(usize),
    #[error("invalid CSR structure: {0}")]
    InvalidCsrStructure(String),
    #[error("invalid iterative-refinement configuration field `{field}`")]
    InvalidConfiguration { field: &'static str },
    #[error("non-finite value generated during {stage} at index {index}")]
    NumericalBreakdown { stage: &'static str, index: usize },
    #[error("precision conversion generated a non-finite value during {stage} at index {index}")]
    PrecisionConversion { stage: &'static str, index: usize },
    #[error("direct factor solve failed: {0}")]
    DirectSolve(#[from] BlrError),
}

impl IrError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NumericalBreakdown { .. } => SolverStatus::Breakdown,
            Self::DirectSolve(error) => error.status(),
            Self::MatrixNotSquare { .. }
            | Self::FactorDimensionMismatch { .. }
            | Self::RhsDimensionMismatch { .. }
            | Self::NonFiniteMatrix(_)
            | Self::NonFiniteRhs(_)
            | Self::InvalidCsrStructure(_)
            | Self::InvalidConfiguration { .. }
            | Self::PrecisionConversion { .. } => SolverStatus::NumericalError,
        }
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    /// No requested accuracy is established when refinement fails before a
    /// checked result is returned.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverFailure for IrError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::MatrixNotSquare { .. }
            | Self::FactorDimensionMismatch { .. }
            | Self::RhsDimensionMismatch { .. } => SolverErrorKind::DimensionMismatch,
            Self::NumericalBreakdown { .. } => SolverErrorKind::Breakdown,
            Self::DirectSolve(error) => error.error_kind(),
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        IrError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        IrError::accuracy_status(self)
    }
}

/// Outcome of an [`IterativeRefinement::solve_with_matrix`] call.
#[derive(Debug, Clone)]
pub struct IrResult<T: ComplexField> {
    /// Refined solution vector.
    pub x: Vec<T>,
    /// Number of refinement steps actually executed (excluding the initial
    /// solve).
    pub iterations: usize,
    /// Final relative residual `‖b − A·x‖ / ‖b‖`.
    pub final_residual: f64,
    /// `true` if the residual reached `config.tolerance`.
    pub converged: bool,
}

impl<T: ComplexField> IrResult<T> {
    /// Return the canonical status for this checked refinement result.
    ///
    /// The result schema predates a distinct stagnation field, so a finite
    /// non-converged result is reported as `MaxIterationsReached`. The
    /// residual and `converged` flag remain authoritative for callers that
    /// need to distinguish the returned accuracy from the stop cause.
    pub fn solver_status(&self) -> SolverStatus {
        if !self.final_residual.is_finite()
            || self.final_residual < 0.0
            || self.x.iter().any(|&value| !scalar_is_finite(value))
        {
            SolverStatus::NumericalError
        } else if self.converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        }
    }

    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            SolverStatus::MaxIterationsReached => AccuracyStatus::NotAttained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: ComplexField> SolverResultContract for IrResult<T> {
    fn solver_status(&self) -> SolverStatus {
        IrResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        IrResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if !self.final_residual.is_finite()
            || self.final_residual < 0.0
            || self.x.iter().any(|value| !scalar_is_finite(*value))
        {
            return Err(SolverError::OperationFailed {
                operation: "iterative refinement result contract",
                reason: "solution or residual is invalid".into(),
            });
        }
        if self.converged != (self.solver_status() == SolverStatus::Converged) {
            return Err(SolverError::OperationFailed {
                operation: "iterative refinement result contract",
                reason: "converged flag and solver status disagree".into(),
            });
        }
        Ok(())
    }
}

/// Iterative-refinement driver around a [`MultifrontalBlrFactorization`].
///
/// The factoriser is taken by reference so it can be reused across many
/// right-hand sides; only the residual / correction vectors are allocated
/// per solve.
pub struct IterativeRefinement<'a, T: ComplexField> {
    factor: &'a MultifrontalBlrFactorization<T>,
    matrix: &'a CsrMatrix<T>,
    config: IrConfig,
}

impl<'a, T: ComplexField + FromPrimitive> IterativeRefinement<'a, T> {
    /// Build a new IR driver from a factoriser and the *original* (un-compressed)
    /// matrix that the residual will be evaluated against.
    pub fn new(
        factor: &'a MultifrontalBlrFactorization<T>,
        matrix: &'a CsrMatrix<T>,
        config: IrConfig,
    ) -> Self {
        match Self::try_new(factor, matrix, config) {
            Ok(driver) => driver,
            Err(error) => panic!("invalid iterative-refinement driver: {error}"),
        }
    }

    /// Build a checked IR driver.
    pub fn try_new(
        factor: &'a MultifrontalBlrFactorization<T>,
        matrix: &'a CsrMatrix<T>,
        config: IrConfig,
    ) -> Result<Self, IrError> {
        validate_driver(factor.dimension(), matrix, &config)?;
        Ok(Self {
            factor,
            matrix,
            config,
        })
    }

    /// Run the IR loop for one right-hand side.
    ///
    /// Invariant: on return, `IrResult::final_residual` is `‖b − A·x‖ / ‖b‖`
    /// for the returned `x` — never a stale pre-correction value. We hold
    /// this by computing the residual once before the loop and once after
    /// each correction, and storing the latest value.
    pub fn solve(&self, b: &[T]) -> IrResult<T> {
        match self.try_solve(b) {
            Ok(result) => result,
            Err(error) => panic!("iterative refinement failed: {error}"),
        }
    }

    /// Run the IR loop with checked dimensions, values, and generated state.
    pub fn try_solve(&self, b: &[T]) -> Result<IrResult<T>, IrError> {
        validate_driver(self.factor.dimension(), self.matrix, &self.config)?;
        validate_rhs(b, self.matrix.num_rows)?;
        let n = b.len();

        let b_norm = vec_norm_checked(b, "right-hand-side norm")?;
        let rel = |r_norm: f64| {
            if b_norm > 0.0 {
                r_norm / b_norm
            } else {
                r_norm
            }
        };

        // Initial direct solve.
        let mut x = self.factor.try_solve(b)?;
        let mut r_norm = vec_norm_checked(
            &compute_residual_checked(self.matrix, &x, b)?,
            "initial residual norm",
        )?;
        let mut final_res = checked_relative_residual(rel(r_norm))?;
        let mut converged = final_res <= self.config.tolerance;
        let mut iters = 0usize;

        if converged {
            return Ok(IrResult {
                x,
                iterations: iters,
                final_residual: final_res,
                converged,
            });
        }

        let mut prev_res_norm = r_norm;
        for _ in 0..self.config.max_iterations {
            // Correction step from the current residual.
            let r = compute_residual_checked(self.matrix, &x, b)?;
            let delta = self.factor.try_solve(&r)?;
            for i in 0..n {
                x[i] += delta[i];
                if !scalar_is_finite(x[i]) {
                    return Err(IrError::NumericalBreakdown {
                        stage: "correction accumulation",
                        index: i,
                    });
                }
            }
            iters += 1;

            // Recompute residual on the *updated* x so the returned
            // metadata always describes the returned solution.
            r_norm = vec_norm_checked(
                &compute_residual_checked(self.matrix, &x, b)?,
                "refined residual norm",
            )?;
            final_res = checked_relative_residual(rel(r_norm))?;
            if final_res <= self.config.tolerance {
                converged = true;
                break;
            }
            // Stagnation guard: if the new residual didn't drop enough
            // versus the previous iterate, stop — extra steps won't help.
            if let Some(ratio) = self.config.stagnation_ratio
                && r_norm >= ratio * prev_res_norm
            {
                break;
            }
            prev_res_norm = r_norm;
        }

        Ok(IrResult {
            x,
            iterations: iters,
            final_residual: final_res,
            converged,
        })
    }
}

fn compute_residual_checked<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    x: &[T],
    b: &[T],
) -> Result<Vec<T>, IrError> {
    let n = matrix.num_rows;
    let mut r = vec![T::zero(); n];
    for i in 0..n {
        let mut ax = T::zero();
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            ax += matrix.values[idx] * x[matrix.col_indices[idx]];
            if !scalar_is_finite(ax) {
                return Err(IrError::NumericalBreakdown {
                    stage: "residual matrix-vector product",
                    index: i,
                });
            }
        }
        r[i] = b[i] - ax;
        if !scalar_is_finite(r[i]) {
            return Err(IrError::NumericalBreakdown {
                stage: "residual formation",
                index: i,
            });
        }
    }
    Ok(r)
}

fn vec_norm_checked<T: ComplexField>(v: &[T], stage: &'static str) -> Result<f64, IrError> {
    let mut scale = 0.0_f64;
    let mut sum_squares = 1.0_f64;
    for (index, x) in v.iter().enumerate() {
        let magnitude = x
            .norm()
            .to_f64()
            .filter(|value| value.is_finite())
            .ok_or(IrError::NumericalBreakdown { stage, index })?;
        if magnitude == 0.0 {
            continue;
        }
        if scale < magnitude {
            let ratio = scale / magnitude;
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = magnitude;
        } else {
            let ratio = magnitude / scale;
            sum_squares += ratio * ratio;
        }
    }
    let norm = if scale == 0.0 {
        0.0
    } else {
        scale * sum_squares.sqrt()
    };
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(IrError::NumericalBreakdown {
            stage,
            index: v.len(),
        })
    }
}

fn checked_relative_residual(value: f64) -> Result<f64, IrError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(IrError::NumericalBreakdown {
            stage: "relative residual formation",
            index: 0,
        })
    }
}

fn scalar_is_finite<T: ComplexField>(value: T) -> bool {
    value.re().is_finite() && value.im().is_finite()
}

fn first_non_finite<T: ComplexField>(values: &[T]) -> Option<usize> {
    values.iter().position(|&value| !scalar_is_finite(value))
}

fn validate_driver<T: ComplexField>(
    factor_dimension: usize,
    matrix: &CsrMatrix<T>,
    config: &IrConfig,
) -> Result<(), IrError> {
    if matrix.num_rows != matrix.num_cols {
        return Err(IrError::MatrixNotSquare {
            rows: matrix.num_rows,
            cols: matrix.num_cols,
        });
    }
    if let Some(index) = first_non_finite(&matrix.values) {
        return Err(IrError::NonFiniteMatrix(index));
    }
    matrix
        .validate_structure()
        .map_err(|error| IrError::InvalidCsrStructure(error.to_string()))?;
    if factor_dimension != matrix.num_rows {
        return Err(IrError::FactorDimensionMismatch {
            factor: factor_dimension,
            matrix: matrix.num_rows,
        });
    }
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err(IrError::InvalidConfiguration { field: "tolerance" });
    }
    if let Some(ratio) = config.stagnation_ratio
        && (!ratio.is_finite() || ratio <= 0.0 || ratio > 1.0)
    {
        return Err(IrError::InvalidConfiguration {
            field: "stagnation_ratio",
        });
    }
    Ok(())
}

fn validate_rhs<T: ComplexField>(b: &[T], expected: usize) -> Result<(), IrError> {
    if b.len() != expected {
        return Err(IrError::RhsDimensionMismatch {
            expected,
            actual: b.len(),
        });
    }
    if let Some(index) = first_non_finite(b) {
        return Err(IrError::NonFiniteRhs(index));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Mixed-precision iterative refinement
// ---------------------------------------------------------------------------

/// Iterative refinement with the factor and the residual in *different*
/// precisions. Standard recipe: factor `A_low ≈ A` in low precision (`f32`),
/// compute `r = b − A·x` in high precision (`f64`), then solve
/// `A_low · δ = r` using the low-precision factor, casting `r` down and the
/// correction `δ` back up. The factor cost roughly halves vs. native high
/// precision, while the final solution is at high-precision accuracy.
///
/// Type parameters:
/// - `THigh` — the high-precision scalar that the *original* `A` and `b`
///   live in (the returned solution is also in `THigh`).
/// - `TLow` — the low-precision scalar of the factoriser.
///
/// `THigh: PrecisionCast<TLow>` provides the up / down casts.
pub struct MixedPrecisionRefinement<'a, THigh: ComplexField, TLow: ComplexField> {
    factor_low: &'a MultifrontalBlrFactorization<TLow>,
    matrix_high: &'a CsrMatrix<THigh>,
    config: IrConfig,
}

impl<'a, THigh, TLow> MixedPrecisionRefinement<'a, THigh, TLow>
where
    THigh: ComplexField + PrecisionCast<TLow>,
    TLow: ComplexField + FromPrimitive,
{
    pub fn new(
        factor_low: &'a MultifrontalBlrFactorization<TLow>,
        matrix_high: &'a CsrMatrix<THigh>,
        config: IrConfig,
    ) -> Self {
        match Self::try_new(factor_low, matrix_high, config) {
            Ok(driver) => driver,
            Err(error) => panic!("invalid mixed-precision refinement driver: {error}"),
        }
    }

    /// Build a checked mixed-precision IR driver.
    pub fn try_new(
        factor_low: &'a MultifrontalBlrFactorization<TLow>,
        matrix_high: &'a CsrMatrix<THigh>,
        config: IrConfig,
    ) -> Result<Self, IrError> {
        validate_driver(factor_low.dimension(), matrix_high, &config)?;
        Ok(Self {
            factor_low,
            matrix_high,
            config,
        })
    }

    pub fn solve(&self, b: &[THigh]) -> IrResult<THigh> {
        match self.try_solve(b) {
            Ok(result) => result,
            Err(error) => panic!("mixed-precision iterative refinement failed: {error}"),
        }
    }

    /// Run mixed-precision IR with checked casts and generated values.
    pub fn try_solve(&self, b: &[THigh]) -> Result<IrResult<THigh>, IrError> {
        validate_driver(self.factor_low.dimension(), self.matrix_high, &self.config)?;
        validate_rhs(b, self.matrix_high.num_rows)?;
        let n = b.len();

        let b_norm = vec_norm_checked(b, "right-hand-side norm")?;
        let rel = |r_norm: f64| {
            if b_norm > 0.0 {
                r_norm / b_norm
            } else {
                r_norm
            }
        };

        // Initial direct solve in low precision.
        let b_low: Vec<TLow> = b.iter().map(|&v| THigh::cast_down(v)).collect();
        if let Some(index) = first_non_finite(&b_low) {
            return Err(IrError::PrecisionConversion {
                stage: "right-hand-side downcast",
                index,
            });
        }
        let x_low = self.factor_low.try_solve(&b_low)?;
        let mut x: Vec<THigh> = x_low.into_iter().map(THigh::cast_up).collect();
        if let Some(index) = first_non_finite(&x) {
            return Err(IrError::PrecisionConversion {
                stage: "initial solution upcast",
                index,
            });
        }

        let mut r_norm = vec_norm_checked(
            &compute_residual_checked(self.matrix_high, &x, b)?,
            "initial residual norm",
        )?;
        let mut final_res = checked_relative_residual(rel(r_norm))?;
        let mut converged = final_res <= self.config.tolerance;
        let mut iters = 0usize;
        if converged {
            return Ok(IrResult {
                x,
                iterations: iters,
                final_residual: final_res,
                converged,
            });
        }

        let mut prev = r_norm;
        for _ in 0..self.config.max_iterations {
            // r in high precision, then cast down for the low-precision solve.
            let r_high = compute_residual_checked(self.matrix_high, &x, b)?;
            let r_low: Vec<TLow> = r_high.iter().map(|&v| THigh::cast_down(v)).collect();
            if let Some(index) = first_non_finite(&r_low) {
                return Err(IrError::PrecisionConversion {
                    stage: "residual downcast",
                    index,
                });
            }
            let delta_low = self.factor_low.try_solve(&r_low)?;
            // Accumulate the up-cast correction.
            for i in 0..n {
                let correction = THigh::cast_up(delta_low[i]);
                if !scalar_is_finite(correction) {
                    return Err(IrError::PrecisionConversion {
                        stage: "correction upcast",
                        index: i,
                    });
                }
                x[i] += correction;
                if !scalar_is_finite(x[i]) {
                    return Err(IrError::NumericalBreakdown {
                        stage: "correction accumulation",
                        index: i,
                    });
                }
            }
            iters += 1;

            r_norm = vec_norm_checked(
                &compute_residual_checked(self.matrix_high, &x, b)?,
                "refined residual norm",
            )?;
            final_res = checked_relative_residual(rel(r_norm))?;
            if final_res <= self.config.tolerance {
                converged = true;
                break;
            }
            if let Some(ratio) = self.config.stagnation_ratio
                && r_norm >= ratio * prev
            {
                break;
            }
            prev = r_norm;
        }

        Ok(IrResult {
            x,
            iterations: iters,
            final_residual: final_res,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlrConfig, MultifrontalBlrFactorization, SparseLuFactorization};
    use sonium_solvers_core::dense::DenseMatrix;
    use sonium_solvers_core::traits::{AccuracyStatus, SolverStatus};
    use sonium_solvers_sparse::CsrMatrix;

    fn make_2d_laplacian(nx: usize) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = 4.0;
                if ix > 0 {
                    dense[[i, i - 1]] = -1.0;
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = -1.0;
                }
                if iy > 0 {
                    dense[[i, i - nx]] = -1.0;
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = -1.0;
                }
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn rel_residual(matrix: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let mut res_sq = 0.0;
        let mut b_sq = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                s += matrix.values[idx] * x[matrix.col_indices[idx]];
            }
            let r = b[i] - s;
            res_sq += r * r;
            b_sq += b[i] * b[i];
        }
        if b_sq == 0.0 {
            res_sq.sqrt()
        } else {
            (res_sq / b_sq).sqrt()
        }
    }

    #[test]
    fn refinement_drives_residual_below_compression_tol() {
        // Factor a Helmholtz-like Laplacian (with stronger far-field
        // structure than pure Laplacian) at a relatively loose ε; then
        // verify IR reaches machine precision in a small number of steps.
        let nx = 24;
        let n = nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = 4.0;
                if ix > 0 {
                    dense[[i, i - 1]] = -1.0;
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = -1.0;
                }
                if iy > 0 {
                    dense[[i, i - nx]] = -1.0;
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = -1.0;
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.3).cos()).collect();

        let factor = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 1e-3,
                block_size: 12,
                min_block_size: 12,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("factorise");

        // Plain BLR solve at this ε leaves measurable residual.
        let x_plain = factor.solve(&b);
        let res_plain = rel_residual(&m, &x_plain, &b);

        // Refined solve.
        let ir = IterativeRefinement::new(&factor, &m, IrConfig::default());
        let out = ir.solve(&b);
        assert!(
            out.final_residual <= res_plain + 1e-14,
            "IR final residual {} > plain {} (no improvement)",
            out.final_residual,
            res_plain
        );
        assert!(
            out.final_residual < 1e-10,
            "IR did not converge to 1e-10: {} after {} iters (plain={})",
            out.final_residual,
            out.iterations,
            res_plain
        );
    }

    #[test]
    fn refinement_is_idempotent_when_factor_is_exact() {
        // With ε=0 the BLR factor is exact (modulo round-off). IR should
        // detect convergence on the very first residual check.
        let nx = 10;
        let m = make_2d_laplacian(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

        let factor = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 8,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("factorise");

        let ir = IterativeRefinement::new(&factor, &m, IrConfig::default());
        let out = ir.solve(&b);
        assert!(out.converged);
        assert!(
            out.iterations <= 1,
            "expected ≤ 1 iter, got {}",
            out.iterations
        );

        // And the answer should match exact LU.
        let exact = SparseLuFactorization::factorize(&m).expect("exact LU");
        let x_exact = exact.solve(&b).expect("sparse LU solve");
        for i in 0..n {
            assert!((out.x[i] - x_exact[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn mixed_precision_ir_recovers_high_precision() {
        // Factor in f32, solve / refine in f64. Target accuracy should reach
        // ~1e-12 after a few refinement steps even though the factor is at
        // f32 precision (~1e-7 floor).
        let nx = 12;
        let m_high = make_2d_laplacian(nx);
        let n = m_high.num_rows;

        // Build the f32 copy of the matrix.
        let mut m_low_dense = DenseMatrix::<f32>::zeros(n, n);
        for i in 0..n {
            for idx in m_high.row_ptrs[i]..m_high.row_ptrs[i + 1] {
                let j = m_high.col_indices[idx];
                m_low_dense[[i, j]] = m_high.values[idx] as f32;
            }
        }
        let m_low = CsrMatrix::from_dense(&m_low_dense, 1e-7);

        let factor_low = MultifrontalBlrFactorization::factorize_with_config(
            &m_low,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 12,
                min_block_size: 8,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("f32 factorise");

        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
        let mpr = MixedPrecisionRefinement::<f64, f32>::new(
            &factor_low,
            &m_high,
            IrConfig {
                max_iterations: 12,
                tolerance: 1e-12,
                stagnation_ratio: Some(0.999),
            },
        );
        let out = mpr.solve(&b);
        assert!(
            out.final_residual < 1e-10,
            "mixed-precision IR residual {} too large after {} iters",
            out.final_residual,
            out.iterations
        );
        assert!(out.iterations > 0);
    }

    #[test]
    fn ir_metadata_matches_returned_x() {
        // Invariant: regardless of max_iterations, final_residual is the
        // residual *of the returned x*, never a stale pre-correction value.
        let nx = 16;
        let m = make_2d_laplacian(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.7).sin()).collect();

        let factor = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 1e-3,
                block_size: 12,
                min_block_size: 12,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("factorise");

        // 1. max_iterations = 0 — must still report the residual of the
        //    initial direct solve, not `f64::INFINITY`.
        let ir0 = IterativeRefinement::new(
            &factor,
            &m,
            IrConfig {
                max_iterations: 0,
                tolerance: 1e-30,
                stagnation_ratio: None,
            },
        );
        let out0 = ir0.solve(&b);
        let manual0 = rel_residual(&m, &out0.x, &b);
        assert!(
            (out0.final_residual - manual0).abs() < 1e-13,
            "max_iters=0: reported {} vs manual {}",
            out0.final_residual,
            manual0
        );
        assert!(out0.final_residual.is_finite());
        assert_eq!(out0.status(), SolverStatus::MaxIterationsReached);
        assert_eq!(out0.accuracy_status(), AccuracyStatus::NotAttained);

        // 2. Bounded iterations that may run to the cap without converging
        //    — metadata must still match the returned x.
        let ir_cap = IterativeRefinement::new(
            &factor,
            &m,
            IrConfig {
                max_iterations: 2,
                tolerance: 1e-30,
                stagnation_ratio: None,
            },
        );
        let out_cap = ir_cap.solve(&b);
        let manual_cap = rel_residual(&m, &out_cap.x, &b);
        assert!(
            (out_cap.final_residual - manual_cap).abs() / manual_cap.max(1e-30) < 1e-9,
            "iters={}: reported {} vs manual {}",
            out_cap.iterations,
            out_cap.final_residual,
            manual_cap
        );
    }
}
