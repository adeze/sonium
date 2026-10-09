//! Residual-probed low-rank augmentation of an ILU(0) preconditioner.
//!
//! The augmentation is deliberately advisory: overshot learned terms are
//! damped against setup probes, while invalid or still-worsening terms are
//! disabled and the object continues as plain ILU.

use num_traits::{Float, FromPrimitive, One, ToPrimitive};
use sonium_solvers_core::traits::{
    AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
};
use sonium_solvers_sparse::CsrMatrix;

use crate::{IluPreconditioner, IlutPreconditioner};

/// Hard safety bound for the stack-resident low-rank work vectors.
pub const MAX_HYBRID_ILU_RANK: usize = 64;

/// Validation policy for [`HybridIluPreconditioner`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridIluConfig {
    /// Largest accepted `hybrid probe residual / ILU probe residual` ratio.
    pub max_probe_residual_ratio: f64,
}

impl Default for HybridIluConfig {
    fn default() -> Self {
        Self {
            max_probe_residual_ratio: 1.0,
        }
    }
}

/// Construction-time safety decision for the learned augmentation.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridIluDiagnostics {
    pub requested_rank: usize,
    pub active_rank: usize,
    pub augmentation_active: bool,
    pub probes_evaluated: usize,
    pub baseline_probe_relative_residual: f64,
    pub hybrid_probe_relative_residual: Option<f64>,
    pub worst_probe_residual_ratio: Option<f64>,
    /// Multiplicative damping applied to the learned correction after probe validation.
    pub accepted_correction_scale: Option<f64>,
    pub rejection_reason: Option<String>,
}

/// ILU(0) plus a bounded low-rank action `U D U^H r`.
///
/// `basis` stores the columns of `U`; `correction` stores the dense `D` in
/// row-major order. The augmentation's transpose (`conj(U) D^T U^T`) and
/// Hermitian (`U D^H U^H`) actions are closed-form, so the adjoint capability
/// is exactly the base factor's capability: ILU/ILUT bases give exact adjoint
/// solves, including when the augmentation is inactive and the object is
/// plain ILU.
#[derive(Debug, Clone)]
pub struct HybridIluPreconditioner<
    T: ComplexField,
    B: Preconditioner<T> + Clone = IluPreconditioner<T>,
> {
    base: B,
    dimension: usize,
    basis: Vec<Vec<T>>,
    correction: Vec<T>,
    diagnostics: HybridIluDiagnostics,
}

impl<T: ComplexField> HybridIluPreconditioner<T> {
    /// Build ILU and admit the learned term only if a deterministic probe is
    /// finite and does not worsen the plain-ILU residual.
    pub fn try_from_csr(
        matrix: &CsrMatrix<T>,
        basis: Vec<Vec<T>>,
        correction: Vec<T>,
        config: HybridIluConfig,
    ) -> Result<Self, SolverError> {
        let ilu = IluPreconditioner::try_from_csr(matrix)?;
        Self::try_from_ilu(matrix, ilu, basis, correction, config)
    }

    /// Wrap an already constructed ILU. Invalid learned data falls back to
    /// that ILU and is recorded in diagnostics; invalid matrix/configuration
    /// remains a hard construction error.
    pub fn try_from_ilu(
        matrix: &CsrMatrix<T>,
        ilu: IluPreconditioner<T>,
        basis: Vec<Vec<T>>,
        correction: Vec<T>,
        config: HybridIluConfig,
    ) -> Result<Self, SolverError> {
        let probes = [deterministic_probe::<T>(matrix.num_rows())];
        Self::try_from_ilu_with_probes(matrix, ilu, basis, correction, &probes, config)
    }

    /// Wrap an existing ILU and validate the augmentation on every supplied
    /// setup probe. This is the production entry point: callers should include
    /// the solve RHS (or its first Krylov residual) in addition to any fixed
    /// deterministic probe.
    pub fn try_from_ilu_with_probes(
        matrix: &CsrMatrix<T>,
        ilu: IluPreconditioner<T>,
        basis: Vec<Vec<T>>,
        correction: Vec<T>,
        probes: &[Vec<T>],
        config: HybridIluConfig,
    ) -> Result<Self, SolverError> {
        Self::try_from_base_with_probes(matrix, ilu, basis, correction, probes, config)
    }
}

impl<T: ComplexField> HybridIluPreconditioner<T, IlutPreconditioner<T>> {
    /// Wrap an existing ILUT factor and validate the learned augmentation on
    /// every supplied setup probe.
    pub fn try_from_ilut_with_probes(
        matrix: &CsrMatrix<T>,
        ilut: IlutPreconditioner<T>,
        basis: Vec<Vec<T>>,
        correction: Vec<T>,
        probes: &[Vec<T>],
        config: HybridIluConfig,
    ) -> Result<Self, SolverError> {
        Self::try_from_base_with_probes(matrix, ilut, basis, correction, probes, config)
    }
}

/// Probe metrics at one scale: worst hybrid residual, worst ratio vs the
/// baseline, whether any probe improved, and the first failing probe as
/// `(index, hybrid_residual, permitted)`.
type HybridProbeMetrics = (f64, f64, bool, Option<(usize, f64, f64)>);

impl<T: ComplexField, B: Preconditioner<T> + Clone> HybridIluPreconditioner<T, B> {
    fn try_from_base_with_probes(
        matrix: &CsrMatrix<T>,
        base: B,
        basis: Vec<Vec<T>>,
        correction: Vec<T>,
        probes: &[Vec<T>],
        config: HybridIluConfig,
    ) -> Result<Self, SolverError> {
        matrix.validate_canonical_structure()?;
        if matrix.num_rows() != matrix.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: matrix.num_rows(),
                got: matrix.num_cols(),
            });
        }
        if !config.max_probe_residual_ratio.is_finite() || config.max_probe_residual_ratio < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "hybrid ILU setup",
                reason: "max_probe_residual_ratio must be finite and non-negative".into(),
            });
        }

        let dimension = matrix.num_rows();
        if probes.is_empty()
            || probes.iter().any(|probe| {
                probe.len() != dimension || probe.iter().any(|value| !value.norm().is_finite())
            })
        {
            return Err(SolverError::OperationFailed {
                operation: "hybrid ILU setup",
                reason: "setup probes must be non-empty, finite, and match the matrix dimension"
                    .into(),
            });
        }
        let requested_rank = basis.len();
        let mut candidate = Self {
            base,
            dimension,
            basis,
            correction,
            diagnostics: HybridIluDiagnostics {
                requested_rank,
                active_rank: 0,
                augmentation_active: false,
                probes_evaluated: probes.len(),
                baseline_probe_relative_residual: 0.0,
                hybrid_probe_relative_residual: None,
                worst_probe_residual_ratio: None,
                accepted_correction_scale: None,
                rejection_reason: None,
            },
        };

        if let Err(reason) = candidate.validate_augmentation() {
            candidate.reject(reason);
            return Ok(candidate);
        }

        let mut baseline_actions = Vec::with_capacity(probes.len());
        let mut baseline_residuals = Vec::with_capacity(probes.len());
        for probe in probes {
            let action = candidate.base.try_apply(probe)?;
            baseline_residuals.push(relative_probe_residual(matrix, probe, &action)?);
            baseline_actions.push(action);
        }
        candidate.diagnostics.baseline_probe_relative_residual =
            baseline_residuals.iter().copied().fold(0.0_f64, f64::max);

        if requested_rank == 0 {
            candidate.diagnostics.hybrid_probe_relative_residual =
                Some(candidate.diagnostics.baseline_probe_relative_residual);
            candidate.diagnostics.worst_probe_residual_ratio = Some(1.0);
            return Ok(candidate);
        }

        // A learned correction can have the right direction but overshoot the
        // admissible step. Try deterministic powers-of-two damping before
        // falling back to the classical factor. Every accepted scale is still
        // checked against every caller-supplied physical/Krylov probe.
        const MAX_BACKTRACKS: usize = 12;
        let mut full_scale_reason = None;
        for backtrack in 0..=MAX_BACKTRACKS {
            let scale = 2.0_f64.powi(-(backtrack as i32));
            let metrics = candidate.probe_metrics_at_scale(
                matrix,
                probes,
                &baseline_actions,
                &baseline_residuals,
                scale,
                config.max_probe_residual_ratio,
            )?;
            let Some((worst_hybrid, worst_ratio, any_improvement, first_failure)) = metrics else {
                if backtrack == 0 {
                    full_scale_reason = Some("probe application produced NaN or infinity".into());
                }
                continue;
            };
            if backtrack == 0 {
                candidate.diagnostics.hybrid_probe_relative_residual = Some(worst_hybrid);
                candidate.diagnostics.worst_probe_residual_ratio = Some(worst_ratio);
                if let Some((index, residual, permitted)) = first_failure {
                    full_scale_reason = Some(format!(
                        "probe {index} residual worsened: {residual:.6e} > {permitted:.6e}"
                    ));
                }
            }
            let passes = first_failure.is_none();
            let meaningful = config.max_probe_residual_ratio > 1.0 || any_improvement;
            if passes && meaningful {
                let scale_value = T::from_real(T::Real::from_f64(scale).unwrap());
                for value in &mut candidate.correction {
                    *value *= scale_value;
                }
                candidate.diagnostics.hybrid_probe_relative_residual = Some(worst_hybrid);
                candidate.diagnostics.worst_probe_residual_ratio = Some(worst_ratio);
                candidate.diagnostics.accepted_correction_scale = Some(scale);
                candidate.diagnostics.active_rank = requested_rank;
                candidate.diagnostics.augmentation_active = true;
                return Ok(candidate);
            }
        }

        candidate.reject(full_scale_reason.unwrap_or_else(|| {
            "no non-trivial correction scale improved a setup probe without worsening another"
                .into()
        }));
        Ok(candidate)
    }

    pub fn diagnostics(&self) -> &HybridIluDiagnostics {
        &self.diagnostics
    }

    pub fn augmentation_active(&self) -> bool {
        self.diagnostics.augmentation_active
    }

    fn reject(&mut self, reason: String) {
        self.basis.clear();
        self.correction.clear();
        self.diagnostics.active_rank = 0;
        self.diagnostics.augmentation_active = false;
        self.diagnostics.rejection_reason = Some(reason);
    }

    fn validate_augmentation(&self) -> Result<(), String> {
        let rank = self.basis.len();
        if rank == 0 {
            if self.correction.is_empty() {
                return Ok(());
            }
            return Err("rank-zero basis requires an empty correction matrix".into());
        }
        if rank > MAX_HYBRID_ILU_RANK {
            return Err(format!(
                "augmentation rank {rank} exceeds the supported maximum {MAX_HYBRID_ILU_RANK}"
            ));
        }
        let expected_correction = rank
            .checked_mul(rank)
            .ok_or_else(|| "augmentation rank is too large".to_owned())?;
        if self.correction.len() != expected_correction {
            return Err(format!(
                "correction has {} entries, expected {} for rank {rank}",
                self.correction.len(),
                expected_correction
            ));
        }
        if self
            .basis
            .iter()
            .any(|column| column.len() != self.dimension)
        {
            return Err(format!(
                "basis columns must have system dimension {}",
                self.dimension
            ));
        }
        if self
            .basis
            .iter()
            .flatten()
            .chain(self.correction.iter())
            .any(|value| !value.norm().is_finite())
        {
            return Err("basis or correction contains NaN or infinity".into());
        }
        Ok(())
    }

    fn probe_metrics_at_scale(
        &self,
        matrix: &CsrMatrix<T>,
        probes: &[Vec<T>],
        baseline_actions: &[Vec<T>],
        baseline_residuals: &[f64],
        scale: f64,
        max_ratio: f64,
    ) -> Result<Option<HybridProbeMetrics>, SolverError> {
        let mut worst_hybrid = 0.0_f64;
        let mut worst_ratio = 0.0_f64;
        let mut any_improvement = false;
        let mut first_failure = None;
        for (index, ((probe, baseline_action), &baseline_residual)) in probes
            .iter()
            .zip(baseline_actions)
            .zip(baseline_residuals)
            .enumerate()
        {
            let mut hybrid_action = baseline_action.clone();
            self.add_scaled_augmentation_into(probe, &mut hybrid_action, scale);
            if hybrid_action.iter().any(|value| !value.norm().is_finite()) {
                return Ok(None);
            }
            let hybrid_residual = relative_probe_residual(matrix, probe, &hybrid_action)?;
            let ratio = hybrid_residual / baseline_residual.max(f64::MIN_POSITIVE);
            worst_hybrid = worst_hybrid.max(hybrid_residual);
            worst_ratio = worst_ratio.max(ratio);
            any_improvement |= hybrid_residual
                < baseline_residual - 1.0e-12 * baseline_residual.max(f64::MIN_POSITIVE);
            let permitted = max_ratio * baseline_residual;
            if hybrid_residual > permitted && first_failure.is_none() {
                first_failure = Some((index, hybrid_residual, permitted));
            }
        }
        Ok(Some((
            worst_hybrid,
            worst_ratio,
            any_improvement,
            first_failure,
        )))
    }

    fn apply_augmented_into(&self, r: &[T], y: &mut [T]) {
        self.base.apply_into(r, y);
        self.add_scaled_augmentation_into(r, y, 1.0);
    }

    fn apply_augmented_transpose_into(&self, r: &[T], y: &mut [T]) {
        self.base.apply_transpose_into(r, y);
        self.add_augmentation_transpose_into(r, y);
    }

    fn apply_augmented_hermitian_into(&self, r: &[T], y: &mut [T]) {
        self.base.apply_hermitian_into(r, y);
        self.add_augmentation_hermitian_into(r, y);
    }

    fn add_scaled_augmentation_into(&self, r: &[T], y: &mut [T], scale: f64) {
        let rank = self.basis.len();
        if rank == 0 {
            return;
        }
        let scale = T::from_real(T::Real::from_f64(scale).unwrap());
        let mut projected = [T::zero(); MAX_HYBRID_ILU_RANK];
        let mut mixed = [T::zero(); MAX_HYBRID_ILU_RANK];
        for (index, column) in self.basis.iter().enumerate() {
            projected[index] = T::vec_dot(column, r);
        }
        for (row, slot) in mixed.iter_mut().take(rank).enumerate() {
            for (column, coefficient) in projected[..rank].iter().enumerate() {
                *slot += self.correction[row * rank + column] * *coefficient;
            }
        }
        for (column, coefficient) in self.basis.iter().zip(&mixed[..rank]) {
            for (output, value) in y.iter_mut().zip(column) {
                *output += *value * *coefficient * scale;
            }
        }
    }

    /// Add `(U D U^H)^T r = conj(U) D^T U^T r`. The projection onto `U^T`
    /// is the unconjugated bilinear form, and the expansion uses the
    /// conjugated basis columns.
    fn add_augmentation_transpose_into(&self, r: &[T], y: &mut [T]) {
        let rank = self.basis.len();
        if rank == 0 {
            return;
        }
        let mut projected = [T::zero(); MAX_HYBRID_ILU_RANK];
        let mut mixed = [T::zero(); MAX_HYBRID_ILU_RANK];
        for (index, column) in self.basis.iter().enumerate() {
            projected[index] = T::vec_dot_plain(column, r);
        }
        for (row, slot) in mixed.iter_mut().take(rank).enumerate() {
            for (column, coefficient) in projected[..rank].iter().enumerate() {
                *slot += self.correction[column * rank + row] * *coefficient;
            }
        }
        for (column, coefficient) in self.basis.iter().zip(&mixed[..rank]) {
            for (output, value) in y.iter_mut().zip(column) {
                *output += value.conj() * *coefficient;
            }
        }
    }

    /// Add `(U D U^H)^H r = U D^H U^H r`. The projection onto `U^H` is the
    /// Hermitian inner product, and the dense mix uses the conjugate
    /// transpose of the stored row-major correction.
    fn add_augmentation_hermitian_into(&self, r: &[T], y: &mut [T]) {
        let rank = self.basis.len();
        if rank == 0 {
            return;
        }
        let mut projected = [T::zero(); MAX_HYBRID_ILU_RANK];
        let mut mixed = [T::zero(); MAX_HYBRID_ILU_RANK];
        for (index, column) in self.basis.iter().enumerate() {
            projected[index] = T::vec_dot(column, r);
        }
        for (row, slot) in mixed.iter_mut().take(rank).enumerate() {
            for (column, coefficient) in projected[..rank].iter().enumerate() {
                *slot += self.correction[column * rank + row].conj() * *coefficient;
            }
        }
        for (column, coefficient) in self.basis.iter().zip(&mixed[..rank]) {
            for (output, value) in y.iter_mut().zip(column) {
                *output += *value * *coefficient;
            }
        }
    }
}

impl<T: ComplexField, B: Preconditioner<T> + Clone> Preconditioner<T>
    for HybridIluPreconditioner<T, B>
{
    fn apply_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(r.len(), self.dimension, "hybrid ILU input length mismatch");
        assert_eq!(y.len(), self.dimension, "hybrid ILU output length mismatch");
        if self.augmentation_active() {
            self.apply_augmented_into(r, y);
        } else {
            self.base.apply_into(r, y);
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.base.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.base.hermitian_capability()
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        assert_eq!(
            r.len(),
            self.dimension,
            "hybrid ILU transpose input length mismatch"
        );
        let mut y = vec![T::zero(); self.dimension];
        self.apply_transpose_into(r, &mut y);
        y
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(
            r.len(),
            self.dimension,
            "hybrid ILU transpose input length mismatch"
        );
        assert_eq!(
            y.len(),
            self.dimension,
            "hybrid ILU transpose output length mismatch"
        );
        if self.augmentation_active() {
            self.apply_augmented_transpose_into(r, y);
        } else {
            self.base.apply_transpose_into(r, y);
        }
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        assert_eq!(
            r.len(),
            self.dimension,
            "hybrid ILU Hermitian input length mismatch"
        );
        let mut y = vec![T::zero(); self.dimension];
        self.apply_hermitian_into(r, &mut y);
        y
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(
            r.len(),
            self.dimension,
            "hybrid ILU Hermitian input length mismatch"
        );
        assert_eq!(
            y.len(),
            self.dimension,
            "hybrid ILU Hermitian output length mismatch"
        );
        if self.augmentation_active() {
            self.apply_augmented_hermitian_into(r, y);
        } else {
            self.base.apply_hermitian_into(r, y);
        }
    }
}

fn deterministic_probe<T: ComplexField>(dimension: usize) -> Vec<T> {
    (0..dimension)
        .map(|index| {
            T::from_real(if (index.wrapping_mul(17).wrapping_add(5) & 1) == 0 {
                T::Real::one()
            } else {
                -T::Real::one()
            })
        })
        .collect()
}

fn relative_probe_residual<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    action: &[T],
) -> Result<f64, SolverError> {
    let applied = matrix.try_matvec(action)?;
    let residual: Vec<T> = rhs
        .iter()
        .zip(applied)
        .map(|(right, left)| *right - left)
        .collect();
    let numerator =
        T::vec_norm(&residual)
            .to_f64()
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "hybrid ILU probe",
                reason: "could not convert residual norm to f64".into(),
            })?;
    let denominator = T::vec_norm(rhs)
        .to_f64()
        .ok_or_else(|| SolverError::OperationFailed {
            operation: "hybrid ILU probe",
            reason: "could not convert right-hand-side norm to f64".into(),
        })?;
    Ok(numerator / denominator.max(f64::MIN_POSITIVE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    /// 3x3-grid Laplacian with a complex shift and one asymmetric
    /// coupling. ILU(0) drops fill here, so the setup probe has a genuine
    /// (non-roundoff) baseline residual and a small augmentation is admitted.
    fn nonsymmetric_system() -> CsrMatrix<Complex64> {
        let grid = 3;
        let dimension = grid * grid;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(
            dimension,
            dimension,
            Complex64::new(0.0, 0.0),
        );
        for row in 0..grid {
            for column in 0..grid {
                let index = row * grid + column;
                dense[[index, index]] = Complex64::new(4.0, 0.5);
                if column + 1 < grid {
                    dense[[index, index + 1]] = Complex64::new(-1.0, 0.0);
                    dense[[index + 1, index]] = Complex64::new(-1.0, 0.0);
                }
                if row + 1 < grid {
                    dense[[index, index + grid]] = Complex64::new(-1.0, 0.0);
                    dense[[index + grid, index]] = Complex64::new(-1.0, 0.0);
                }
            }
        }
        dense[[1, 0]] += Complex64::new(0.0, 0.25);
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn rank_two_augmentation() -> (Vec<Vec<Complex64>>, Vec<Complex64>) {
        let dimension = 9;
        let mut basis = Vec::with_capacity(2);
        for mode in 0..2 {
            basis.push(
                (0..dimension)
                    .map(|index| {
                        Complex64::new(
                            0.1 * ((mode + 3 * index) % 7) as f64 - 0.3,
                            0.1 * ((mode + 5 * index) % 5) as f64 - 0.2,
                        )
                    })
                    .collect(),
            );
        }
        let correction = vec![
            Complex64::new(0.01, 0.02),
            Complex64::new(-0.015, 0.005),
            Complex64::new(0.008, -0.01),
            Complex64::new(0.02, 0.0),
        ];
        (basis, correction)
    }

    fn probe_vector() -> Vec<Complex64> {
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-0.5, 1.0),
            Complex64::new(0.75, -0.25),
            Complex64::new(-1.25, 0.5),
            Complex64::new(0.0, -1.5),
            Complex64::new(2.0, 0.75),
            Complex64::new(-0.25, -0.75),
            Complex64::new(1.5, 1.25),
            Complex64::new(0.5, 0.0),
        ]
    }

    /// Dense action of the stored preconditioner: base ILU columns plus the
    /// accepted low-rank term, so the reference tracks probe-time damping.
    fn dense_hybrid_action(
        base: &IluPreconditioner<Complex64>,
        basis: &[Vec<Complex64>],
        correction: &[Complex64],
        scale: f64,
    ) -> Vec<Vec<Complex64>> {
        let dimension = basis
            .first()
            .map(Vec::len)
            .expect("dense reference needs a non-empty basis");
        let rank = basis.len();
        let mut action = vec![vec![Complex64::new(0.0, 0.0); dimension]; dimension];
        for column in 0..dimension {
            let mut unit = vec![Complex64::new(0.0, 0.0); dimension];
            unit[column] = Complex64::new(1.0, 0.0);
            let applied = base.apply(&unit);
            for row in 0..dimension {
                action[row][column] = applied[row];
            }
        }
        for row in 0..dimension {
            for column in 0..dimension {
                for i in 0..rank {
                    for j in 0..rank {
                        action[row][column] += basis[i][row]
                            * correction[i * rank + j]
                            * scale
                            * basis[j][column].conj();
                    }
                }
            }
        }
        action
    }

    fn dense_matvec(matrix: &[Vec<Complex64>], vector: &[Complex64]) -> Vec<Complex64> {
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(vector)
                    .map(|(value, input)| *value * *input)
                    .sum()
            })
            .collect()
    }

    fn transpose(matrix: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        let dimension = matrix.len();
        (0..dimension)
            .map(|row| (0..dimension).map(|column| matrix[column][row]).collect())
            .collect()
    }

    fn hermitian(matrix: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        let dimension = matrix.len();
        (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|column| matrix[column][row].conj())
                    .collect()
            })
            .collect()
    }

    fn assert_complex_close(left: &[Complex64], right: &[Complex64]) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).norm() < 1e-12,
                "entry {index}: {left:?} vs {right:?}"
            );
        }
    }

    #[test]
    fn transpose_and_hermitian_match_dense_reference() {
        let matrix = nonsymmetric_system();
        let (basis, correction) = rank_two_augmentation();
        let base = IluPreconditioner::from_csr(&matrix);
        let hybrid = HybridIluPreconditioner::try_from_ilu(
            &matrix,
            base.clone(),
            basis.clone(),
            correction.clone(),
            HybridIluConfig {
                max_probe_residual_ratio: 10.0,
            },
        )
        .unwrap();
        assert!(hybrid.augmentation_active());
        assert_eq!(hybrid.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(hybrid.hermitian_capability(), AdjointCapability::Exact);

        let scale = hybrid
            .diagnostics()
            .accepted_correction_scale
            .expect("admitted augmentation records its scale");
        let action = dense_hybrid_action(&base, &basis, &correction, scale);
        let r = probe_vector();
        assert_complex_close(&hybrid.apply(&r), &dense_matvec(&action, &r));
        assert_complex_close(
            &hybrid.apply_transpose(&r),
            &dense_matvec(&transpose(&action), &r),
        );
        assert_complex_close(
            &hybrid.apply_hermitian(&r),
            &dense_matvec(&hermitian(&action), &r),
        );

        let mut into = vec![Complex64::new(0.0, 0.0); r.len()];
        hybrid.apply_transpose_into(&r, &mut into);
        assert_complex_close(&into, &hybrid.apply_transpose(&r));
        hybrid.apply_hermitian_into(&r, &mut into);
        assert_complex_close(&into, &hybrid.apply_hermitian(&r));
    }

    #[test]
    fn adjoint_identities_hold_for_active_augmentation() {
        let matrix = nonsymmetric_system();
        let (basis, correction) = rank_two_augmentation();
        let hybrid = HybridIluPreconditioner::try_from_csr(
            &matrix,
            basis,
            correction,
            HybridIluConfig {
                max_probe_residual_ratio: 10.0,
            },
        )
        .unwrap();
        assert!(hybrid.augmentation_active());

        let x = probe_vector();
        let y = probe_vector()
            .into_iter()
            .rev()
            .map(|value| value.conj() * Complex64::new(0.5, -0.25))
            .collect::<Vec<_>>();
        let mx = hybrid.apply(&x);
        let mhy = hybrid.apply_hermitian(&y);
        let mty = hybrid.apply_transpose(&y);
        let hermitian_left: Complex64 = mx
            .iter()
            .zip(&y)
            .map(|(value, other)| value.conj() * *other)
            .sum();
        let hermitian_right: Complex64 = x
            .iter()
            .zip(&mhy)
            .map(|(value, other)| value.conj() * *other)
            .sum();
        assert!((hermitian_left - hermitian_right).norm() < 1e-12);
        let bilinear_left: Complex64 = mx
            .iter()
            .zip(&y)
            .map(|(value, other)| *value * *other)
            .sum();
        let bilinear_right: Complex64 = x
            .iter()
            .zip(&mty)
            .map(|(value, other)| *value * *other)
            .sum();
        assert!((bilinear_left - bilinear_right).norm() < 1e-12);
    }

    #[test]
    fn inactive_augmentation_keeps_exact_base_adjoint() {
        let matrix = nonsymmetric_system();
        let base = IluPreconditioner::from_csr(&matrix);
        let hybrid = HybridIluPreconditioner::try_from_ilu(
            &matrix,
            base.clone(),
            Vec::new(),
            Vec::new(),
            HybridIluConfig::default(),
        )
        .unwrap();
        assert!(!hybrid.augmentation_active());
        assert_eq!(hybrid.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(hybrid.hermitian_capability(), AdjointCapability::Exact);

        let r = probe_vector();
        assert_complex_close(&hybrid.apply_transpose(&r), &base.apply_transpose(&r));
        assert_complex_close(&hybrid.apply_hermitian(&r), &base.apply_hermitian(&r));
    }

    #[test]
    fn checked_adjoint_apply_rejects_bad_inputs() {
        let matrix = nonsymmetric_system();
        let (basis, correction) = rank_two_augmentation();
        let hybrid = HybridIluPreconditioner::try_from_csr(
            &matrix,
            basis,
            correction,
            HybridIluConfig {
                max_probe_residual_ratio: 10.0,
            },
        )
        .unwrap();
        let r = vec![Complex64::new(1.0, 0.0); 9];
        assert!(hybrid.try_apply_transpose(&r).is_ok());
        assert!(hybrid.try_apply_hermitian(&r).is_ok());
        assert!(hybrid.try_apply_transpose(&r[..2]).is_err());
        assert!(hybrid.try_apply_hermitian(&r[..2]).is_err());
        let mut non_finite = r.clone();
        non_finite[0] = Complex64::new(f64::NAN, 0.0);
        assert!(hybrid.try_apply_transpose(&non_finite).is_err());
        assert!(hybrid.try_apply_hermitian(&non_finite).is_err());
    }
}
