//! Chebyshev polynomial preconditioner
//!
//! Applies degree-k Chebyshev iteration as a preconditioner:
//!
//! ```text
//! M^{-1} ≈ p_k(D^{-1}A) D^{-1}
//! ```
//!
//! where `p_k` is the degree-k Chebyshev polynomial on `[lambda_min, lambda_max]`
//! and `D = diag(A)`.
//!
//! **Properties:**
//! - Fully parallel (only needs SpMV + element-wise vector ops)
//! - Zero global communication (unlike CG/GMRES which need dot products)
//! - Matrix-free friendly (only needs `LinearOperator::apply`)
//! - Good for well-conditioned SPD systems as standalone preconditioner
//! - Excellent smoother for multigrid
//!
//! **Eigenvalue bounds:** If not provided, estimated via 20 power iterations
//! on `D^{-1}A` (same method as the AMG Chebyshev smoother).
//!
//! **SPD-only.** The real spectral interval `[lambda_min, lambda_max]` is
//! only meaningful for (symmetric/Hermitian) positive definite input.
//! [`ChebyshevPreconditioner::try_from_csr`] enforces square,
//! symmetric/Hermitian, real-spectrum, and positive-diagonal conditions at
//! setup, and returns a [`ChebyshevError`] otherwise.
//! [`ChebyshevPreconditioner::from_csr`] keeps the crate's infallible
//! constructor convention: it logs a `log::warn!` describing the violation
//! and proceeds, producing a preconditioner that is not mathematically
//! meaningful for that input.

use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use rayon::prelude::*;
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, Preconditioner, SolverError, SolverFailure,
    SolverStatus,
};
use sonium_solvers_sparse::CsrMatrix;
use std::collections::BTreeMap;

const CHEBYSHEV_PAR_THRESHOLD: usize = 8_192;

/// Versioned portable checkpoint format for a Chebyshev preconditioner.
pub const CHEBYSHEV_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const CHEBYSHEV_CHECKPOINT_MAGIC: &[u8; 8] = b"SCHEBCP1";

fn chebyshev_checkpoint_error(message: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "Chebyshev checkpoint",
        reason: message.into(),
    }
}

fn checkpoint_push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value = u64::try_from(value)
        .map_err(|_| chebyshev_checkpoint_error("usize value does not fit checkpoint format"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn checkpoint_push_scalars<T: ComplexField>(
    output: &mut Vec<u8>,
    values: &[T],
) -> Result<(), SolverError> {
    checkpoint_push_u64(output, values.len())?;
    for value in values {
        let re = value
            .re()
            .to_f64()
            .ok_or_else(|| chebyshev_checkpoint_error("real scalar cannot convert to f64"))?;
        let im = value
            .im()
            .to_f64()
            .ok_or_else(|| chebyshev_checkpoint_error("imaginary scalar cannot convert to f64"))?;
        if !re.is_finite() || !im.is_finite() {
            return Err(chebyshev_checkpoint_error(
                "checkpoint contains a non-finite scalar",
            ));
        }
        output.extend_from_slice(&re.to_le_bytes());
        output.extend_from_slice(&im.to_le_bytes());
    }
    Ok(())
}

struct ChebyshevCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ChebyshevCheckpointReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| chebyshev_checkpoint_error("checkpoint offset overflow"))?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| chebyshev_checkpoint_error("truncated Chebyshev checkpoint"))?;
        self.offset = end;
        Ok(result)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, SolverError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let value = f64::from_le_bytes(self.take(8)?.try_into().unwrap());
        if !value.is_finite() {
            return Err(chebyshev_checkpoint_error(
                "checkpoint contains a non-finite scalar parameter",
            ));
        }
        Ok(value)
    }

    fn read_len(&mut self, label: &str, element_bytes: usize) -> Result<usize, SolverError> {
        let length = usize::try_from(self.read_u64()?)
            .map_err(|_| chebyshev_checkpoint_error(format!("{label} does not fit usize")))?;
        if element_bytes == 0
            || length > self.bytes.len().saturating_sub(self.offset) / element_bytes
        {
            return Err(chebyshev_checkpoint_error(format!(
                "{label} exceeds remaining checkpoint storage"
            )));
        }
        Ok(length)
    }

    fn read_usize_vec(&mut self, label: &str) -> Result<Vec<usize>, SolverError> {
        let length = self.read_len(label, 8)?;
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            values.push(usize::try_from(self.read_u64()?).map_err(|_| {
                chebyshev_checkpoint_error(format!("{label} entry does not fit usize"))
            })?);
        }
        Ok(values)
    }

    fn read_scalars<T: ComplexField>(&mut self, label: &str) -> Result<Vec<T>, SolverError> {
        let length = self.read_len(label, 16)?;
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            let re = T::Real::from_f64(self.read_f64()?)
                .ok_or_else(|| chebyshev_checkpoint_error("real scalar cannot be decoded"))?;
            let im = T::Real::from_f64(self.read_f64()?)
                .ok_or_else(|| chebyshev_checkpoint_error("imaginary scalar cannot be decoded"))?;
            let value = T::from_re_im(re, im);
            if !value.norm().is_finite() {
                return Err(chebyshev_checkpoint_error(format!(
                    "{label} contains a non-finite value"
                )));
            }
            values.push(value);
        }
        Ok(values)
    }

    fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Setup-time misuse detection for the Chebyshev preconditioner.
///
/// The Chebyshev iteration is built on a real spectral interval, so it is
/// only valid for SPD (real symmetric positive definite) input. These are
/// cheap *necessary, not sufficient* checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ChebyshevError {
    /// Matrix storage contains a NaN or infinity, so no finite spectral
    /// interval can be established for the Chebyshev iteration.
    #[error(
        "Chebyshev preconditioner requires finite matrix entries: {count} entries are non-finite"
    )]
    NonFiniteEntries { count: usize },

    /// Matrix has entries with nonzero imaginary part; the spectrum is not
    /// real and a real-interval Chebyshev polynomial is meaningless.
    #[error(
        "Chebyshev preconditioner requires a real spectrum (SPD input): \
         matrix has {count} entries with nonzero imaginary part"
    )]
    ComplexEntries {
        /// Number of entries with nonzero imaginary part.
        count: usize,
    },

    /// A diagonal entry is missing or has non-positive real part. SPD
    /// matrices have strictly positive diagonal entries, so this violates a
    /// necessary SPD condition.
    #[error(
        "Chebyshev preconditioner requires SPD input: diagonal entry {index} \
         is missing or non-positive (a necessary SPD condition)"
    )]
    NonPositiveDiagonal {
        /// First row index with a missing or non-positive diagonal entry.
        index: usize,
    },
    /// The matrix is not symmetric/Hermitian, so its spectrum is not
    /// guaranteed to lie on the real interval required by Chebyshev.
    #[error(
        "Chebyshev preconditioner requires a symmetric/Hermitian matrix; entry ({row}, {col}) is not mirrored"
    )]
    NonSymmetric { row: usize, col: usize },
    /// Chebyshev is defined only for square systems.
    #[error("Chebyshev preconditioner requires a square matrix, got {rows}x{cols}")]
    NonSquare { rows: usize, cols: usize },

    /// The spectral interval configuration is not usable.
    #[error("invalid Chebyshev configuration: {reason}")]
    InvalidConfiguration { reason: &'static str },

    /// The CSR storage is malformed and cannot be safely inspected.
    #[error("Chebyshev preconditioner received malformed CSR storage")]
    InvalidCsrStructure,
}

impl SolverFailure for ChebyshevError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        sonium_solvers_core::traits::SolverErrorKind::OperationFailed
    }

    fn status(&self) -> SolverStatus {
        SolverStatus::NumericalError
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }
}

/// Configuration for the Chebyshev preconditioner.
#[derive(Debug, Clone)]
pub struct ChebyshevConfig {
    /// Polynomial degree (number of iterations). Higher = better approximation
    /// but more SpMVs per preconditioner apply. Typical: 3-10.
    pub degree: usize,

    /// Ratio lambda_max / lambda_min. The Chebyshev polynomial is optimised
    /// for the interval `[lambda_max / eigenvalue_ratio, lambda_max]`.
    /// Default: 30.0 (same as PETSc/HYPRE).
    pub eigenvalue_ratio: f64,

    /// Override for lambda_max. If `None`, estimated automatically via
    /// 20 power iterations on `D^{-1}A`.
    pub lambda_max: Option<f64>,
}

impl Default for ChebyshevConfig {
    fn default() -> Self {
        Self {
            degree: 5,
            eigenvalue_ratio: 30.0,
            lambda_max: None,
        }
    }
}

/// Chebyshev polynomial preconditioner for SPD matrices.
///
/// Build from a `CsrMatrix` or any `LinearOperator` + diagonal.
/// Implements `Preconditioner<T>` for use with any Krylov solver.
pub struct ChebyshevPreconditioner<T: ComplexField> {
    /// The system matrix (owned reference for SpMV)
    matrix: CsrMatrix<T>,
    /// Inverse diagonal: D^{-1}
    diag_inv: Vec<T>,
    /// Estimated max eigenvalue of D^{-1}A
    lambda_max: T::Real,
    /// Ratio for computing lambda_min
    eigenvalue_ratio: f64,
    /// Polynomial degree
    degree: usize,
}

impl<T: ComplexField> ChebyshevPreconditioner<T> {
    /// Create a Chebyshev preconditioner from a CSR matrix.
    ///
    /// Eigenvalue bounds are estimated automatically if not provided in config.
    ///
    /// The Chebyshev iteration assumes SPD input. This constructor checks two
    /// cheap necessary conditions (no complex entries, positive diagonal) and
    /// emits a `log::warn!` if they fail, then proceeds anyway — consistent
    /// with the crate's infallible preconditioner constructors. Use
    /// [`ChebyshevPreconditioner::try_from_csr`] to get a [`ChebyshevError`]
    /// instead of a warning.
    pub fn from_csr(matrix: &CsrMatrix<T>, config: ChebyshevConfig) -> Self {
        // Validate-once: the apply path runs `matvec_into_buf_unchecked`
        // against the owned clone, so the memory-safety invariants must be
        // established here. Like the SpMV hot path this accepts unsorted
        // rows, duplicates, and non-finite values; structurally invalid
        // storage now panics here instead of at the first apply.
        super::amg::util::validate_owned_structure(matrix)
            .expect("Chebyshev setup requires valid CSR storage");
        if let Err(err) = check_spd_assumptions(matrix) {
            log::warn!(
                "{err}; building anyway, but the resulting preconditioner is \
                 not mathematically meaningful for this input"
            );
        }
        Self::build(matrix, config)
    }

    /// Fallible constructor: returns a [`ChebyshevError`] when the input
    /// violates the SPD assumptions of the Chebyshev iteration or contains
    /// an invalid spectral interval configuration.
    pub fn try_from_csr(
        matrix: &CsrMatrix<T>,
        config: ChebyshevConfig,
    ) -> Result<Self, ChebyshevError> {
        validate_config(&config)?;
        // `build` converts an explicitly supplied f64 bound to the scalar
        // precision.  Reject values that cannot be represented there before
        // the compatibility constructor's infallible conversion is reached.
        if let Some(lambda_max) = config.lambda_max {
            let converted =
                T::Real::from_f64(lambda_max).ok_or(ChebyshevError::InvalidConfiguration {
                    reason: "lambda_max is not representable in matrix scalar precision",
                })?;
            if !converted.is_finite() {
                return Err(ChebyshevError::InvalidConfiguration {
                    reason: "lambda_max not representable in matrix scalar precision",
                });
            }
            if converted <= T::Real::zero() {
                return Err(ChebyshevError::InvalidConfiguration {
                    reason: "lambda_max must be finite and positive in matrix scalar precision",
                });
            }
        }
        let non_finite_count = matrix
            .values
            .iter()
            .filter(|value| !value.norm().is_finite())
            .count();
        if non_finite_count != 0 {
            return Err(ChebyshevError::NonFiniteEntries {
                count: non_finite_count,
            });
        }
        matrix
            .validate_canonical_structure()
            .map_err(|_| ChebyshevError::InvalidCsrStructure)?;
        check_spd_assumptions(matrix)?;
        let preconditioner = Self::build(matrix, config);
        let lambda_max = preconditioner.lambda_max;
        let ratio = T::Real::from_f64(preconditioner.eigenvalue_ratio).ok_or(
            ChebyshevError::InvalidConfiguration {
                reason: "eigenvalue_ratio is not representable in matrix scalar precision",
            },
        )?;
        let lambda_min = lambda_max / ratio;
        let two = T::Real::from_f64(2.0).ok_or(ChebyshevError::InvalidConfiguration {
            reason: "scalar precision cannot represent Chebyshev constants",
        })?;
        let theta = (lambda_max + lambda_min) / two;
        let delta = (lambda_max - lambda_min) / two;
        let sigma = theta / delta;
        let one = T::Real::from_f64(1.0).ok_or(ChebyshevError::InvalidConfiguration {
            reason: "scalar precision cannot represent Chebyshev constants",
        })?;
        let inv_theta = one / theta;
        if !lambda_max.is_finite()
            || lambda_max <= T::Real::zero()
            || !lambda_min.is_finite()
            || lambda_min <= T::Real::zero()
            || !theta.is_finite()
            || !delta.is_finite()
            || delta <= T::Real::zero()
            || !sigma.is_finite()
            || sigma <= T::Real::zero()
            || !inv_theta.is_finite()
        {
            return Err(ChebyshevError::InvalidConfiguration {
                reason: "estimated spectral interval is not finite or usable",
            });
        }
        Ok(preconditioner)
    }

    fn build(matrix: &CsrMatrix<T>, config: ChebyshevConfig) -> Self {
        // Compute D^{-1}
        let diag_inv = diagonal_inverse(matrix);

        // Estimate lambda_max
        let lambda_max = if let Some(lm) = config.lambda_max {
            T::Real::from_f64(lm).unwrap()
        } else {
            super::amg::util::estimate_max_eigenvalue(matrix, &diag_inv)
        };

        Self {
            matrix: matrix.clone(),
            diag_inv,
            lambda_max,
            eigenvalue_ratio: config.eigenvalue_ratio,
            degree: config.degree,
        }
    }

    /// Return the estimated spectral radius of D^{-1}A.
    pub fn lambda_max(&self) -> T::Real {
        self.lambda_max
    }

    /// Serialize the preconditioner into a strict, portable little-endian
    /// checkpoint. The matrix and inverse diagonal are retained explicitly so
    /// restoring a checkpoint does not depend on the host's spectral-estimate
    /// implementation or floating-point iteration order.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        self.matrix
            .validate_canonical_structure()
            .map_err(|error| chebyshev_checkpoint_error(error.to_string()))?;
        if self.matrix.num_rows != self.matrix.num_cols {
            return Err(chebyshev_checkpoint_error(
                "checkpoint matrix must be square",
            ));
        }
        if self.diag_inv.len() != self.matrix.num_rows {
            return Err(chebyshev_checkpoint_error(
                "inverse diagonal length does not match matrix dimension",
            ));
        }
        if !self.lambda_max.is_finite()
            || self.lambda_max <= T::Real::zero()
            || !self.eigenvalue_ratio.is_finite()
            || self.eigenvalue_ratio <= 1.0
        {
            return Err(chebyshev_checkpoint_error(
                "checkpoint spectral parameters are invalid",
            ));
        }

        let mut output = Vec::new();
        output.extend_from_slice(CHEBYSHEV_CHECKPOINT_MAGIC);
        output.extend_from_slice(&CHEBYSHEV_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(u8::from(
            std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>(),
        ));
        output.push(0); // reserved flags
        checkpoint_push_u64(&mut output, self.matrix.num_rows)?;
        checkpoint_push_u64(&mut output, self.matrix.num_cols)?;
        checkpoint_push_u64(&mut output, self.degree)?;
        output.extend_from_slice(&self.eigenvalue_ratio.to_le_bytes());
        output.extend_from_slice(
            &self
                .lambda_max
                .to_f64()
                .ok_or_else(|| chebyshev_checkpoint_error("lambda_max cannot convert to f64"))?
                .to_le_bytes(),
        );
        checkpoint_push_u64(&mut output, self.matrix.row_ptrs.len())?;
        for &value in &self.matrix.row_ptrs {
            checkpoint_push_u64(&mut output, value)?;
        }
        checkpoint_push_u64(&mut output, self.matrix.col_indices.len())?;
        for &value in &self.matrix.col_indices {
            checkpoint_push_u64(&mut output, value)?;
        }
        checkpoint_push_scalars(&mut output, &self.matrix.values)?;
        checkpoint_push_scalars(&mut output, &self.diag_inv)?;
        Ok(output)
    }

    /// Restore a preconditioner produced by [`Self::to_checkpoint_bytes`].
    /// Every structural, scalar, finite-value, truncation, and trailing-byte
    /// boundary is checked before a usable preconditioner is published.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = ChebyshevCheckpointReader::new(bytes);
        if reader.take(CHEBYSHEV_CHECKPOINT_MAGIC.len())? != CHEBYSHEV_CHECKPOINT_MAGIC {
            return Err(chebyshev_checkpoint_error("invalid checkpoint magic"));
        }
        if reader.read_u16()? != CHEBYSHEV_CHECKPOINT_FORMAT_VERSION {
            return Err(chebyshev_checkpoint_error("unsupported checkpoint version"));
        }
        let scalar_size = reader.read_u8()? as usize;
        let real_size = reader.read_u8()? as usize;
        let expected_complex =
            u8::from(std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>());
        if scalar_size != std::mem::size_of::<T>()
            || real_size != std::mem::size_of::<T::Real>()
            || reader.read_u8()? != expected_complex
        {
            return Err(chebyshev_checkpoint_error(
                "checkpoint scalar metadata does not match requested type",
            ));
        }
        if reader.read_u8()? != 0 {
            return Err(chebyshev_checkpoint_error(
                "checkpoint reserved flags are non-zero",
            ));
        }

        let rows = usize::try_from(reader.read_u64()?)
            .map_err(|_| chebyshev_checkpoint_error("row count does not fit usize"))?;
        let cols = usize::try_from(reader.read_u64()?)
            .map_err(|_| chebyshev_checkpoint_error("column count does not fit usize"))?;
        let degree = usize::try_from(reader.read_u64()?)
            .map_err(|_| chebyshev_checkpoint_error("degree does not fit usize"))?;
        let eigenvalue_ratio = reader.read_f64()?;
        let lambda_max_f64 = reader.read_f64()?;
        if eigenvalue_ratio <= 1.0 || !eigenvalue_ratio.is_finite() || lambda_max_f64 <= 0.0 {
            return Err(chebyshev_checkpoint_error(
                "checkpoint spectral parameters are invalid",
            ));
        }
        let lambda_max = T::Real::from_f64(lambda_max_f64)
            .ok_or_else(|| chebyshev_checkpoint_error("lambda_max cannot be decoded"))?;
        if !lambda_max.is_finite() || lambda_max <= T::Real::zero() {
            return Err(chebyshev_checkpoint_error("decoded lambda_max is invalid"));
        }

        let row_ptrs = reader.read_usize_vec("row-pointer count")?;
        let col_indices = reader.read_usize_vec("column-index count")?;
        let values = reader.read_scalars::<T>("matrix values")?;
        let diag_inv = reader.read_scalars::<T>("inverse diagonal")?;
        if rows != cols
            || row_ptrs.len() != rows.saturating_add(1)
            || col_indices.len() != values.len()
            || diag_inv.len() != rows
        {
            return Err(chebyshev_checkpoint_error(
                "checkpoint CSR dimensions are inconsistent",
            ));
        }
        let matrix = CsrMatrix::try_from_raw_parts(rows, cols, row_ptrs, col_indices, values)
            .map_err(|error| chebyshev_checkpoint_error(error.to_string()))?;
        matrix
            .validate_canonical_structure()
            .map_err(|error| chebyshev_checkpoint_error(error.to_string()))?;
        if !reader.is_at_end() {
            return Err(chebyshev_checkpoint_error(
                "checkpoint contains trailing bytes",
            ));
        }
        Ok(Self {
            matrix,
            diag_inv,
            lambda_max,
            eigenvalue_ratio,
            degree,
        })
    }
}

fn validate_config(config: &ChebyshevConfig) -> Result<(), ChebyshevError> {
    if !config.eigenvalue_ratio.is_finite() || config.eigenvalue_ratio <= 1.0 {
        return Err(ChebyshevError::InvalidConfiguration {
            reason: "eigenvalue_ratio must be finite and greater than one",
        });
    }
    if let Some(lambda_max) = config.lambda_max
        && (!lambda_max.is_finite() || lambda_max <= 0.0)
    {
        return Err(ChebyshevError::InvalidConfiguration {
            reason: "lambda_max must be finite and positive",
        });
    }
    Ok(())
}

/// Checked SPD assumption boundary for Chebyshev iteration. The symmetry
/// check accumulates duplicate CSR entries, so it matches the matrix's
/// matvec semantics while remaining linear in the number of stored entries.
fn check_spd_assumptions<T: ComplexField>(matrix: &CsrMatrix<T>) -> Result<(), ChebyshevError> {
    if matrix.num_rows != matrix.num_cols {
        return Err(ChebyshevError::NonSquare {
            rows: matrix.num_rows,
            cols: matrix.num_cols,
        });
    }

    // Reject non-finite generated values before any norm/ordering checks.  A
    // NaN would otherwise make all comparison-based SPD checks appear to pass
    // and poison the power-iteration eigenvalue estimate.
    let non_finite_count = matrix
        .values
        .iter()
        .filter(|value| !value.norm().is_finite())
        .count();
    if non_finite_count != 0 {
        return Err(ChebyshevError::NonFiniteEntries {
            count: non_finite_count,
        });
    }

    // Complex-entry scan: flag when the largest imaginary part is non-negligible
    // relative to the largest entry magnitude.
    let mut max_im = T::Real::zero();
    let mut max_abs = T::Real::zero();
    for &v in &matrix.values {
        max_im = max_im.max(v.im().abs());
        max_abs = max_abs.max(v.norm());
    }
    // Do not apply an f64-sized absolute threshold to f32 matrices: assembled
    // symmetric values can differ by several single-precision ulps.
    let relative_tol = T::Real::from_f64(1e-12)
        .unwrap()
        .max(T::Real::epsilon() * T::Real::from_f64(32.0).unwrap());
    let tol = relative_tol * max_abs.max(T::Real::from_f64(1e-300).unwrap());
    if max_im > tol {
        let count = matrix.values.iter().filter(|v| v.im().abs() > tol).count();
        return Err(ChebyshevError::ComplexEntries { count });
    }

    // CSR permits duplicate and unsorted entries. Accumulate them first so
    // the check follows the matrix's matvec semantics rather than comparing
    // individual storage records. Use an ordered map to make the reported
    // offending entry deterministic.
    let mut entries = BTreeMap::<(usize, usize), T>::new();
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let column = matrix.col_indices[index];
            *entries.entry((row, column)).or_insert(T::zero()) += matrix.values[index];
        }
    }
    for (&(row, column), &value) in &entries {
        let mirror = entries
            .get(&(column, row))
            .copied()
            .unwrap_or_else(T::zero)
            .conj();
        if (value - mirror).norm() > tol {
            return Err(ChebyshevError::NonSymmetric { row, col: column });
        }
    }

    // Diagonal scan: SPD matrices have a strictly positive diagonal.  Use the
    // coalesced map so duplicate CSR diagonal entries follow additive matvec
    // semantics instead of whichever duplicate happens to be encountered
    // first in storage order.
    for i in 0..matrix.num_rows {
        let Some(diagonal) = entries.get(&(i, i)) else {
            return Err(ChebyshevError::NonPositiveDiagonal { index: i });
        };
        if diagonal.re() <= T::Real::zero() {
            return Err(ChebyshevError::NonPositiveDiagonal { index: i });
        }
    }
    Ok(())
}

impl<T: ComplexField> Preconditioner<T> for ChebyshevPreconditioner<T> {
    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        // The checked constructor admits only symmetric real-valued CSR
        // systems (also when represented as Complex<T>), so the Chebyshev
        // polynomial is exactly its own transpose and Hermitian action.
        self.apply(r)
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        self.apply(r)
    }

    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut z = vec![T::zero(); r.len()];
        self.apply_into(r, &mut z);
        z
    }

    fn apply_into(&self, r: &[T], z: &mut [T]) {
        let n = r.len();
        assert_eq!(z.len(), n, "preconditioner output size mismatch");
        assert_eq!(
            self.diag_inv.len(),
            n,
            "Chebyshev preconditioner size mismatch"
        );

        if self.degree == 0 {
            // Degree 0: just Jacobi (D^{-1} r)
            jacobi_scale_into(&self.diag_inv, r, z);
            return;
        }

        let ratio = T::Real::from_f64(self.eigenvalue_ratio).unwrap();
        let two = T::Real::from_f64(2.0).unwrap();
        let one = T::Real::from_f64(1.0).unwrap();

        let lambda_max = self.lambda_max;
        let lambda_min = lambda_max / ratio;
        let theta = (lambda_max + lambda_min) / two;
        let delta = (lambda_max - lambda_min) / two;
        let sigma = theta / delta;
        let mut rho = one / sigma;
        let inv_theta = one / theta;

        // We solve M z ≈ r by applying Chebyshev iteration to Az = r
        // starting from z = 0:
        //
        // Iteration 0: d = (1/theta) * D^{-1} * r   (since z=0, residual = r)
        //              z = d
        let mut d = vec![T::zero(); n];
        let inv_theta = T::from_real(inv_theta);
        if n >= CHEBYSHEV_PAR_THRESHOLD {
            d.par_iter_mut()
                .zip(self.diag_inv.par_iter())
                .zip(r.par_iter())
                .for_each(|((di, &diag_inv), &ri)| {
                    *di = inv_theta * (diag_inv * ri);
                });
        } else {
            for i in 0..n {
                d[i] = inv_theta * (self.diag_inv[i] * r[i]);
            }
        }
        z.copy_from_slice(&d);

        // Pre-allocate work buffers for sweep loop
        let mut az = vec![T::zero(); n];
        let mut d_new = vec![T::zero(); n];

        // Subsequent sweeps using 3-term recurrence
        for _ in 1..self.degree {
            let rho_new = one / (two * sigma - rho);
            // SAFETY: every constructor establishes the memory-safety
            // invariants up front (`from_csr` via
            // `validate_owned_structure`, `try_from_csr` and
            // `from_checkpoint_bytes` via the stronger canonical check),
            // `matrix` sits behind a private field, and no method exposes
            // `&mut` access to it — the owned clone cannot be corrupted
            // after setup. Dimensions are asserted inside.
            unsafe { self.matrix.matvec_into_buf_unchecked(z, &mut az) };
            let coeff1 = T::from_real(rho_new * rho);
            let coeff2 = T::from_real(two * rho_new / delta);

            if n >= CHEBYSHEV_PAR_THRESHOLD {
                d_new
                    .par_iter_mut()
                    .zip(d.par_iter())
                    .zip(self.diag_inv.par_iter())
                    .zip(r.par_iter())
                    .zip(az.par_iter())
                    .for_each(|((((d_new_i, &d_i), &diag_inv), &ri), &az_i)| {
                        let residual_i = diag_inv * (ri - az_i);
                        *d_new_i = coeff1 * d_i + coeff2 * residual_i;
                    });

                z.par_iter_mut()
                    .zip(d_new.par_iter())
                    .for_each(|(zi, &d_new_i)| {
                        *zi += d_new_i;
                    });
            } else {
                for i in 0..n {
                    let residual_i = self.diag_inv[i] * (r[i] - az[i]);
                    d_new[i] = coeff1 * d[i] + coeff2 * residual_i;
                }
                for i in 0..n {
                    z[i] += d_new[i];
                }
            }
            std::mem::swap(&mut d, &mut d_new);
            rho = rho_new;
        }
    }
}

fn diagonal_inverse<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<T> {
    let n = matrix.num_rows;
    if n >= CHEBYSHEV_PAR_THRESHOLD {
        return (0..n)
            .into_par_iter()
            .map(|i| diagonal_inverse_at(matrix, i))
            .collect();
    }

    (0..n).map(|i| diagonal_inverse_at(matrix, i)).collect()
}

/// Inverse of the diagonal entry of row `i`.
///
/// Matches [`super::diagonal::DiagonalPreconditioner`]: a missing or
/// near-zero diagonal entry is replaced with 1 (identity scaling for that
/// row) instead of silently producing a zero row in the output.
fn diagonal_inverse_at<T: ComplexField>(matrix: &CsrMatrix<T>, i: usize) -> T {
    let start = matrix.row_ptrs[i];
    let end = matrix.row_ptrs[i + 1];
    for idx in start..end {
        if matrix.col_indices[idx] == i {
            let d = matrix.values[idx];
            if d.norm() > T::Real::from_f64(1e-20).unwrap() {
                return d.inv();
            }
            break;
        }
    }
    T::one()
}

fn jacobi_scale_into<T: ComplexField>(diag_inv: &[T], r: &[T], z: &mut [T]) {
    debug_assert_eq!(diag_inv.len(), r.len());
    debug_assert_eq!(z.len(), r.len());

    if r.len() >= CHEBYSHEV_PAR_THRESHOLD {
        z.par_iter_mut()
            .zip(r.par_iter())
            .zip(diag_inv.par_iter())
            .for_each(|((zi, &ri), &di)| {
                *zi = di * ri;
            });
    } else {
        z.iter_mut()
            .zip(r.iter())
            .zip(diag_inv.iter())
            .for_each(|((zi, &ri), &di)| {
                *zi = di * ri;
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_iterative::{GmresConfig, gmres_preconditioned};

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(2.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }

        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    #[test]
    fn from_csr_rejects_structurally_invalid_storage_up_front() {
        // Validate-once: the apply path runs unchecked against the
        // owned clone, so malformed storage must panic here instead
        // of at the first apply.
        let mut malformed =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]);
        malformed.row_ptrs[2] = 3;
        assert!(
            std::panic::catch_unwind(|| {
                ChebyshevPreconditioner::from_csr(&malformed, ChebyshevConfig::default())
            })
            .is_err()
        );
    }

    #[test]
    fn test_chebyshev_reduces_residual() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0; n];

        let precond = ChebyshevPreconditioner::from_csr(&a, ChebyshevConfig::default());
        assert!(precond.lambda_max() > 0.0);

        // Apply preconditioner to b
        let z = precond.apply(&b);
        // The result should be a reasonable approximation to A^{-1} b
        let residual = sonium_solvers_core::vec_ops::vec_sub(&b, &a.matvec(&z));
        let r_norm: f64 = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(
            r_norm / b_norm < 1.0,
            "Chebyshev should reduce residual: rel = {:.4e}",
            r_norm / b_norm
        );
    }

    #[test]
    fn test_chebyshev_with_gmres() {
        let n = 200;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let precond = ChebyshevPreconditioner::from_csr(
            &a,
            ChebyshevConfig {
                degree: 5,
                ..Default::default()
            },
        );

        let config = GmresConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };
        let result = gmres_preconditioned(&a, &precond, &b, &config);
        assert!(
            result.converged,
            "GMRES+Chebyshev should converge (iters: {}, res: {:.2e})",
            result.iterations, result.residual
        );
        assert!(
            result.iterations < 50,
            "Should converge faster than unpreconditioned: {} iters",
            result.iterations
        );
    }

    #[test]
    fn test_chebyshev_degree_zero_is_jacobi() {
        let n = 10;
        let a = laplacian_1d(n);
        let r = vec![1.0_f64; n];

        let cheb = ChebyshevPreconditioner::from_csr(
            &a,
            ChebyshevConfig {
                degree: 0,
                ..Default::default()
            },
        );
        let z = cheb.apply(&r);

        // D^{-1} r = (1/2) * r for 1D Laplacian (diag = 2)
        for i in 0..n {
            assert!((z[i] - 0.5).abs() < 1e-12, "Degree-0 should be Jacobi");
        }
    }

    #[test]
    fn test_chebyshev_large_parallel_vector_ops_reduce_diagonal_residual() {
        let n = 20_000;
        let triplets = (0..n).map(|i| (i, i, 2.0_f64)).collect();
        let a = CsrMatrix::from_triplets(n, n, triplets);
        let r = (0..n)
            .map(|i| (i as f64 * 0.017).sin() + 0.25)
            .collect::<Vec<_>>();

        let cheb = ChebyshevPreconditioner::from_csr(
            &a,
            ChebyshevConfig {
                degree: 4,
                lambda_max: Some(2.0),
                ..Default::default()
            },
        );
        let z = cheb.apply(&r);

        let residual = sonium_solvers_core::vec_ops::vec_sub(&r, &a.matvec(&z));
        let r_norm = r.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();

        assert!(z.iter().all(|value| value.is_finite()));
        assert!(
            residual_norm / r_norm < 0.5,
            "parallel Chebyshev vector ops should reduce diagonal residual: {:.3e}",
            residual_norm / r_norm
        );
    }

    #[test]
    fn test_chebyshev_rejects_indefinite_input() {
        // Negative diagonal: violates a necessary SPD condition.
        let n = 10;
        let mut a = laplacian_1d(n);
        for idx in a.row_ptrs[4]..a.row_ptrs[5] {
            if a.col_indices[idx] == 4 {
                a.values[idx] = -2.0;
            }
        }

        let err = ChebyshevPreconditioner::try_from_csr(&a, ChebyshevConfig::default())
            .err()
            .expect("indefinite input must be rejected");
        assert_eq!(err, ChebyshevError::NonPositiveDiagonal { index: 4 });

        // The infallible constructor still builds (with a logged warning).
        let precond = ChebyshevPreconditioner::from_csr(&a, ChebyshevConfig::default());
        let z = precond.apply(&vec![1.0; n]);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_chebyshev_rejects_asymmetric_input() {
        let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (0, 1, -1.0), (1, 1, 2.0)]);
        let error = ChebyshevPreconditioner::try_from_csr(&matrix, ChebyshevConfig::default())
            .err()
            .expect("non-symmetric input must be rejected");
        assert_eq!(error, ChebyshevError::NonSymmetric { row: 0, col: 1 });
    }

    #[test]
    fn test_chebyshev_rejects_non_square_input() {
        let matrix = CsrMatrix::from_triplets(1, 2, vec![(0, 0, 1.0)]);
        let error = ChebyshevPreconditioner::try_from_csr(&matrix, ChebyshevConfig::default())
            .err()
            .expect("non-square input must be rejected");
        assert_eq!(error, ChebyshevError::NonSquare { rows: 1, cols: 2 });
    }

    #[test]
    fn test_chebyshev_rejects_invalid_configuration() {
        let matrix = CsrMatrix::<f64>::identity(2);
        for config in [
            ChebyshevConfig {
                eigenvalue_ratio: 1.0,
                ..Default::default()
            },
            ChebyshevConfig {
                eigenvalue_ratio: f64::NAN,
                ..Default::default()
            },
            ChebyshevConfig {
                lambda_max: Some(0.0),
                ..Default::default()
            },
            ChebyshevConfig {
                lambda_max: Some(f64::INFINITY),
                ..Default::default()
            },
        ] {
            assert!(matches!(
                ChebyshevPreconditioner::try_from_csr(&matrix, config),
                Err(ChebyshevError::InvalidConfiguration { .. })
            ));
        }
    }

    #[test]
    fn test_chebyshev_rejects_non_finite_matrix_entries() {
        let mut nan_matrix = CsrMatrix::identity(1);
        nan_matrix.values[0] = f64::NAN;
        let nan_error =
            ChebyshevPreconditioner::try_from_csr(&nan_matrix, ChebyshevConfig::default())
                .err()
                .expect("NaN matrix entry must be rejected");
        assert_eq!(nan_error, ChebyshevError::NonFiniteEntries { count: 1 });

        let mut infinity_matrix = CsrMatrix::identity(1);
        infinity_matrix.values[0] = f64::INFINITY;
        let infinity_error =
            ChebyshevPreconditioner::try_from_csr(&infinity_matrix, ChebyshevConfig::default())
                .err()
                .expect("infinite matrix entry must be rejected");
        assert_eq!(
            infinity_error,
            ChebyshevError::NonFiniteEntries { count: 1 }
        );
    }

    #[test]
    fn test_chebyshev_rejects_complex_input() {
        use num_complex::Complex64;
        let n = 8;
        let triplets = (0..n)
            .map(|i| (i, i, Complex64::new(2.0, if i == 3 { 1.0 } else { 0.0 })))
            .collect();
        let a = CsrMatrix::<Complex64>::from_triplets(n, n, triplets);

        let err = ChebyshevPreconditioner::try_from_csr(&a, ChebyshevConfig::default())
            .err()
            .expect("complex input must be rejected");
        assert_eq!(err, ChebyshevError::ComplexEntries { count: 1 });
    }

    #[test]
    fn test_chebyshev_accepts_real_valued_complex_type() {
        // Complex64 storage with zero imaginary parts is legitimate SPD
        // input (common in this crate) and must not trip the guard.
        use num_complex::Complex64;
        let n = 8;
        let triplets = (0..n)
            .flat_map(|i| {
                let mut row = vec![(i, i, Complex64::new(2.0, 0.0))];
                if i > 0 {
                    row.push((i, i - 1, Complex64::new(-1.0, 0.0)));
                }
                if i + 1 < n {
                    row.push((i, i + 1, Complex64::new(-1.0, 0.0)));
                }
                row
            })
            .collect();
        let a = CsrMatrix::<Complex64>::from_triplets(n, n, triplets);
        assert!(ChebyshevPreconditioner::try_from_csr(&a, ChebyshevConfig::default()).is_ok());
    }

    #[test]
    fn test_chebyshev_missing_diagonal_does_not_zero_row() {
        // A row without a diagonal entry must get identity scaling
        // (replace-with-1, matching DiagonalPreconditioner), not a zeroed
        // output row.
        let n = 4;
        // Row 1 has only an off-diagonal entry.
        let triplets = vec![(0, 0, 2.0), (1, 0, 1.0), (2, 2, 2.0), (3, 3, 2.0)];
        let a = CsrMatrix::from_triplets(n, n, triplets);
        let precond = ChebyshevPreconditioner::from_csr(
            &a,
            ChebyshevConfig {
                degree: 0, // pure Jacobi scaling: z = D^{-1} r
                lambda_max: Some(2.0),
                ..Default::default()
            },
        );
        let r = vec![1.0; n];
        let z = precond.apply(&r);
        assert_eq!(z[1], 1.0, "missing diagonal should give identity scaling");
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_higher_degree_converges_faster() {
        let n = 200;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let config = GmresConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let p3 = ChebyshevPreconditioner::from_csr(
            &a,
            ChebyshevConfig {
                degree: 3,
                ..Default::default()
            },
        );
        let p8 = ChebyshevPreconditioner::from_csr(
            &a,
            ChebyshevConfig {
                degree: 8,
                ..Default::default()
            },
        );

        let r3 = gmres_preconditioned(&a, &p3, &b, &config);
        let r8 = gmres_preconditioned(&a, &p8, &b, &config);

        assert!(
            r8.iterations <= r3.iterations,
            "Degree 8 ({} iters) should converge no slower than degree 3 ({} iters)",
            r8.iterations,
            r3.iterations
        );
    }

    #[test]
    fn checked_constructor_rejects_overflowed_spectral_estimate() {
        // All entries are finite and the matrix is symmetric positive
        // definite, but a dominant row action overflows during the power
        // iteration used to estimate lambda_max.
        let scale = f64::MAX;
        let off_diagonal = 0.49 * scale;
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, scale),
                (1, 1, scale),
                (2, 2, scale),
                (0, 1, off_diagonal),
                (1, 0, off_diagonal),
                (0, 2, off_diagonal),
                (2, 0, off_diagonal),
                (1, 2, off_diagonal),
                (2, 1, off_diagonal),
            ],
        );

        let error = match ChebyshevPreconditioner::try_from_csr(&matrix, ChebyshevConfig::default())
        {
            Ok(_) => panic!("overflowed spectral estimate must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            ChebyshevError::InvalidConfiguration {
                reason: "estimated spectral interval is not finite or usable",
            }
        );
    }

    #[test]
    fn checked_constructor_rejects_unrepresentable_explicit_bound() {
        let matrix = CsrMatrix::<f32>::identity(1);
        let error = match ChebyshevPreconditioner::try_from_csr(
            &matrix,
            ChebyshevConfig {
                lambda_max: Some(f64::MAX),
                ..Default::default()
            },
        ) {
            Ok(_) => panic!("unrepresentable f32 spectral bound must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            ChebyshevError::InvalidConfiguration {
                reason: "lambda_max not representable in matrix scalar precision",
            }
        );
    }
}
