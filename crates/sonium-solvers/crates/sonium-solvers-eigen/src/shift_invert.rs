//! Shift-invert Lanczos for interior eigenvalues of real symmetric pencils.

use num_complex::Complex64;
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, SolverError, SolverResultContract, SolverStatus},
};
use sonium_solvers_direct::{
    BlrError, MultifrontalLdltFactorization, SparseCholeskyError, SparseCholeskyFactorization,
    SparseLuError, SparseLuFactorization,
};
use sonium_solvers_sparse::CsrMatrix;

use crate::lobpcg::symmetric_eigen_ql;
use crate::result_contract::real_result_shape_valid;

/// Configuration for interior generalized eigenpairs near one real shift.
#[derive(Debug, Clone)]
pub struct ShiftInvertConfig {
    /// Spectral shift. Returned eigenvalues are ordered by distance to it.
    pub target: f64,
    /// Number of eigenpairs requested.
    pub num_eigenvalues: usize,
    /// Maximum Lanczos basis dimension.
    pub krylov_dimension: usize,
    /// Relative true-residual target.
    pub tolerance: f64,
    /// Relative breakdown threshold: basis expansion stops when the
    /// B-norm of the fully reorthogonalized candidate falls below
    /// `breakdown_tolerance` times the running operator scale (the largest
    /// projected diagonal magnitude seen so far). Relative scaling keeps
    /// over/under-scaled pencils from declaring breakdown too early or too
    /// late; the absolute `1e-28`-style cutoff it replaces was only correct
    /// for unit-scale operators.
    pub breakdown_tolerance: f64,
}

impl Default for ShiftInvertConfig {
    fn default() -> Self {
        Self {
            target: 0.0,
            num_eigenvalues: 6,
            krylov_dimension: 40,
            tolerance: 1.0e-8,
            breakdown_tolerance: 1.0e-14,
        }
    }
}

/// Configuration for inertia-certified FEAST interval eigensolves.
#[derive(Debug, Clone)]
pub struct FeastConfig {
    /// Number of trapezoidal quadrature nodes on the complex ellipse.
    pub quadrature_nodes: usize,
    /// Maximum number of contour-subspace iterations.
    pub max_iterations: usize,
    /// Extra vectors retained beyond the inertia-certified eigenvalue count.
    pub subspace_oversampling: usize,
    /// Relative true-residual target.
    pub tolerance: f64,
}

impl Default for FeastConfig {
    fn default() -> Self {
        Self {
            quadrature_nodes: 16,
            max_iterations: 8,
            subspace_oversampling: 8,
            tolerance: 1.0e-8,
        }
    }
}

/// Interior eigenpairs returned by shift-invert Lanczos.
#[derive(Debug, Clone)]
pub struct TargetedEigenResult {
    /// Eigenvalues ordered by distance to the requested target.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors stored as mass-normalized columns.
    pub eigenvectors: DenseMatrix<f64>,
    /// Independently recomputed relative true residuals.
    pub residual_norms: Vec<f64>,
    /// Number of Lanczos vectors used.
    pub iterations: usize,
    /// Whether every returned pair met the requested tolerance.
    pub converged: bool,
}

impl TargetedEigenResult {
    /// Return the canonical status for the returned Ritz pairs.
    pub fn solver_status(&self) -> SolverStatus {
        let shape_valid = real_result_shape_valid(
            &self.eigenvalues,
            &self.eigenvectors,
            &self.residual_norms,
            true,
        );
        if !shape_valid {
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

impl SolverResultContract for TargetedEigenResult {
    fn solver_status(&self) -> SolverStatus {
        TargetedEigenResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        TargetedEigenResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        let status = TargetedEigenResult::solver_status(self);
        let accuracy = TargetedEigenResult::accuracy_status(self);
        if status == SolverStatus::NumericalError {
            return Err(SolverError::OperationFailed {
                operation: "real eigen result contract",
                reason: "result shape or diagnostic is non-finite".into(),
            });
        }
        if (status == SolverStatus::Converged) != (accuracy == AccuracyStatus::Attained) {
            return Err(SolverError::OperationFailed {
                operation: "real eigen result contract",
                reason: "status and accuracy disagree".into(),
            });
        }
        Ok(())
    }
}

/// Compute eigenpairs of A x = lambda B x nearest a real target.
///
/// The operator (A - target B)^-1 B is self-adjoint in the B inner
/// product. One sparse LU factorization is reused by a fully reorthogonalized
/// B-Lanczos iteration, followed by a dense symmetric Ritz solve.
///
/// On lucky breakdown (the deterministic start vector is nearly B-orthogonal
/// to an invariant subspace) with fewer candidates than requested, a second
/// deterministic start stream — B-orthogonalized against the found Ritz
/// vectors — is tried and both candidate sets are merged, instead of failing
/// on the single unlucky start.
pub fn shift_invert_lanczos_generalized(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    config: &ShiftInvertConfig,
) -> Result<TargetedEigenResult, SolverError> {
    let initial = deterministic_start(a.num_rows, 0);
    shift_invert_lanczos_with_seed(a, b, config, initial, 0)
}

fn shift_invert_lanczos_with_seed(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    config: &ShiftInvertConfig,
    initial: Vec<f64>,
    seed_stream: u64,
) -> Result<TargetedEigenResult, SolverError> {
    validate_inputs(a, b, config)?;
    let n = a.num_rows;
    let max_basis = config.krylov_dimension.min(n);
    let shifted = shifted_matrix(a, b, config.target)?;
    let factor = SparseLuFactorization::factorize(&shifted)
        .map_err(|error| map_sparse_lu_error("shift-invert sparse factorization", error))?;

    let mut initial = initial;
    if initial.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: initial.len(),
        });
    }
    let mut b_initial = apply(b, &initial, "shift-invert initial mass application")?;
    normalize_b(&mut initial, &mut b_initial)?;

    let mut basis = vec![initial];
    let mut b_basis = vec![b_initial];
    let mut diagonal: Vec<f64> = Vec::with_capacity(max_basis);
    let mut off_diagonal: Vec<f64> = Vec::with_capacity(max_basis.saturating_sub(1));
    let mut breakdown = false;
    // Running operator-scale estimate for the relative breakdown test: the
    // projected diagonal entries are Rayleigh quotients of the shift-invert
    // operator, hence share the candidate B-norm scale.
    let mut operator_scale = 0.0f64;

    for step in 0..max_basis {
        let q = &basis[step];
        let rhs = &b_basis[step];
        let mut candidate = factor
            .solve(rhs)
            .map_err(|error| map_sparse_lu_error("shift-invert triangular solve", error))?;
        if candidate.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: candidate.len(),
            });
        }
        ensure_finite_values("shift-invert triangular solve", &candidate)?;
        let mut b_candidate = apply(b, &candidate, "shift-invert mass application")?;
        let alpha = dot(q, &b_candidate);
        ensure_finite_values("shift-invert Lanczos diagonal", &[alpha])?;
        diagonal.push(alpha);
        operator_scale = operator_scale.max(alpha.abs());
        axpy(-alpha, q, &mut candidate);
        axpy(-alpha, rhs, &mut b_candidate);
        if step > 0 {
            let beta_previous = off_diagonal[step - 1];
            axpy(-beta_previous, &basis[step - 1], &mut candidate);
            axpy(-beta_previous, &b_basis[step - 1], &mut b_candidate);
        }

        // Two global B-orthogonalization passes keep clustered interior modes
        // stable and make the projected operator numerically symmetric.
        for _ in 0..2 {
            for (prior, b_prior) in basis.iter().zip(&b_basis) {
                let coefficient = dot(prior, &b_candidate);
                ensure_finite_values("shift-invert reorthogonalization", &[coefficient])?;
                axpy(-coefficient, prior, &mut candidate);
                axpy(-coefficient, b_prior, &mut b_candidate);
            }
            ensure_finite_values("shift-invert reorthogonalization", &candidate)?;
            ensure_finite_values("shift-invert mass reorthogonalization", &b_candidate)?;
        }
        let beta_sq = dot(&candidate, &b_candidate);
        ensure_finite_values("shift-invert Lanczos off-diagonal", &[beta_sq])?;
        // Relative lucky-breakdown test: the candidate B-norm is compared
        // against the operator scale, so over/under-scaled pencils neither
        // miss a genuine exhaustion nor stop on a merely small (but vital)
        // direction. An absolute cutoff here was only valid at unit scale.
        if beta_sq <= (config.breakdown_tolerance * operator_scale).powi(2) {
            breakdown = true;
            break;
        }
        if step + 1 == max_basis {
            break;
        }
        let beta = beta_sq.sqrt();
        off_diagonal.push(beta);
        for value in &mut candidate {
            *value /= beta;
        }
        for value in &mut b_candidate {
            *value /= beta;
        }
        ensure_finite_values("shift-invert Lanczos basis", &candidate)?;
        ensure_finite_values("shift-invert Lanczos mass basis", &b_candidate)?;
        basis.push(candidate);
        b_basis.push(b_candidate);
    }

    let dimension = diagonal.len();
    let projected = DenseMatrix::from_fn(dimension, dimension, |row, column| {
        if row == column {
            diagonal[row]
        } else if row + 1 == column {
            off_diagonal[row]
        } else if column + 1 == row {
            off_diagonal[column]
        } else {
            0.0
        }
    });
    ensure_finite_dense("shift-invert projected matrix", &projected)?;
    let (transformed_values, projected_vectors) = symmetric_eigen_ql(&projected, dimension)?;
    ensure_finite_values("shift-invert projected eigenvalues", &transformed_values)?;
    ensure_finite_dense("shift-invert projected eigenvectors", &projected_vectors)?;
    let mut candidates = Vec::new();
    // Projected spectral radius: transformed Ritz values far below it are
    // eigenvalues far from the target (lambda = target + 1/mu) and are
    // dropped relative to this scale — never by an absolute cutoff, which
    // would discard the most-wanted pair when the shift lands near an
    // eigenvalue (huge mu is never dropped).
    let mu_scale = transformed_values
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    for transformed_index in 0..dimension {
        let mu = transformed_values[transformed_index];
        if mu.abs() <= config.tolerance * mu_scale {
            continue;
        }
        let eigenvalue = config.target + mu.recip();
        ensure_finite_values("shift-invert Ritz value", &[eigenvalue])?;
        let mut vector = vec![0.0; n];
        for basis_index in 0..dimension {
            axpy(
                projected_vectors[[basis_index, transformed_index]],
                &basis[basis_index],
                &mut vector,
            );
        }
        ensure_finite_values("shift-invert Ritz vector reconstruction", &vector)?;
        let mut b_vector = apply(b, &vector, "shift-invert Ritz mass application")?;
        normalize_b(&mut vector, &mut b_vector)?;
        let residual = relative_residual(a, b, eigenvalue, &vector)?;
        let target_distance = (eigenvalue - config.target).abs();
        ensure_finite_values("shift-invert Ritz ordering", &[target_distance])?;
        candidates.push((target_distance, eigenvalue, residual, vector));
    }
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    candidates.truncate(config.num_eigenvalues.min(candidates.len()));
    // Lucky-breakdown restart (first stream only): the start vector was
    // nearly B-orthogonal to an invariant subspace, so its Krylov space is
    // exhausted before covering the requested spectrum. The second
    // deterministic stream is B-orthogonalized against the found Ritz
    // vectors — the B-self-adjoint operator keeps the fresh Krylov space
    // B-orthogonal to those pairs — and both candidate sets merge. The
    // retry re-runs validation and factorization; it fires at most once and
    // only on this rare path. Any retry failure falls through to the
    // shortfall error below (best effort, never masks the original state).
    let (candidates, dimension) = if seed_stream == 0
        && breakdown
        && candidates.len() < config.num_eigenvalues
    {
        match restart_seed(b, &candidates, n)
            .map(|second| shift_invert_lanczos_with_seed(a, b, config, second, 1))
        {
            Ok(Ok(second)) => {
                let dimension = dimension.max(second.iterations);
                let merged =
                    merge_candidate_sets(b, candidates, &second, config, config.num_eigenvalues)?;
                (merged, dimension)
            }
            _ => (candidates, dimension),
        }
    } else {
        (candidates, dimension)
    };
    // Retry streams (seed_stream > 0) return partial candidate sets for the
    // restart merge instead of erroring: their shortfall is the caller's
    // (stream-0) shortfall check to report.
    if seed_stream == 0 && candidates.len() < config.num_eigenvalues {
        return Err(SolverError::Breakdown {
            operation: "shift-invert Lanczos basis expansion",
            reason: format!(
                "Krylov space produced only {} of {} requested eigenpairs after {dimension} vectors{}",
                candidates.len(),
                config.num_eigenvalues,
                if breakdown {
                    " (basis expansion broke down)"
                } else {
                    ""
                }
            ),
        });
    }
    let eigenvalues: Vec<f64> = candidates.iter().map(|entry| entry.1).collect();
    let residual_norms: Vec<_> = candidates.iter().map(|entry| entry.2).collect();
    let eigenvectors =
        DenseMatrix::from_fn(n, candidates.len(), |row, column| candidates[column].3[row]);
    ensure_finite_values("shift-invert final eigenvalues", &eigenvalues)?;
    ensure_finite_values("shift-invert final residuals", &residual_norms)?;
    ensure_finite_dense("shift-invert final eigenvectors", &eigenvectors)?;
    let converged = candidates.len() == config.num_eigenvalues
        && residual_norms
            .iter()
            .all(|&residual| residual <= config.tolerance);
    Ok(TargetedEigenResult {
        eigenvalues,
        eigenvectors,
        residual_norms,
        iterations: dimension,
        converged,
    })
}

/// A Ritz candidate: (distance to target, eigenvalue, relative true residual,
/// B-normalized eigenvector).
type RitzCandidate = (f64, f64, f64, Vec<f64>);

/// Deterministic Lanczos start-vector stream.
///
/// Stream 0 reproduces the historical `sin(1.618x) + 0.5cos(0.755x)` start
/// bit-for-bit so existing results don't shift; higher streams use an
/// independent frequency pair for breakdown restarts.
fn deterministic_start(n: usize, stream: u64) -> Vec<f64> {
    let (first, second) = if stream == 0 {
        (std::f64::consts::GOLDEN_RATIO, 0.754_877_666_246_692_7)
    } else {
        (2.399_963_229_728_653, 1.193_280_050_738_878_7)
    };
    (0..n)
        .map(|index| {
            let x = (index + 1) as f64;
            (x * first).sin() + 0.5 * (x * second).cos()
        })
        .collect()
}

/// Build the breakdown-restart start vector: the next deterministic stream,
/// B-orthogonalized against the already-found Ritz vectors with the same two
/// passes the main loop uses. Because the shift-invert operator is
/// B-self-adjoint, the fresh Krylov space stays B-orthogonal to those pairs
/// and explores the complementary spectrum instead of reconverging.
fn restart_seed(
    b: &CsrMatrix<f64>,
    found: &[RitzCandidate],
    n: usize,
) -> Result<Vec<f64>, SolverError> {
    let mut seed = deterministic_start(n, 1);
    for _ in 0..2 {
        let b_seed = apply(b, &seed, "shift-invert restart seed mass application")?;
        for (_, _, _, vector) in found {
            let coefficient = dot(vector, &b_seed);
            ensure_finite_values("shift-invert restart orthogonalization", &[coefficient])?;
            axpy(-coefficient, vector, &mut seed);
        }
    }
    let mut b_seed = apply(b, &seed, "shift-invert restart seed mass application")?;
    normalize_b(&mut seed, &mut b_seed)?;
    Ok(seed)
}

/// B-cosine between two vectors (1 for collinear pairs, ~0 for distinct
/// Ritz pairs). Sign-insensitive: opposite-sign copies of one eigenvector
/// score near −1 and count as duplicates.
fn b_cosine(b: &CsrMatrix<f64>, left: &[f64], right: &[f64]) -> Result<f64, SolverError> {
    let b_left = apply(b, left, "shift-invert restart merge mass application")?;
    let b_right = apply(b, right, "shift-invert restart merge mass application")?;
    let cross = dot(left, &b_right);
    let norms = dot(left, &b_left).max(0.0).sqrt() * dot(right, &b_right).max(0.0).sqrt();
    if !cross.is_finite() || !norms.is_finite() || norms <= 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "shift-invert restart merge",
            reason: "non-finite or degenerate B inner product".into(),
        });
    }
    Ok(cross / norms)
}

/// Merge a second run's eigenpairs into the first run's candidates, keeping
/// the sharper residual on conservative duplicates (tight eigenvalue
/// agreement plus near-unit B-cosine, so clustered-but-distinct pairs are
/// never merged), then re-sort by target distance and truncate.
fn merge_candidate_sets(
    b: &CsrMatrix<f64>,
    first: Vec<RitzCandidate>,
    second: &TargetedEigenResult,
    config: &ShiftInvertConfig,
    limit: usize,
) -> Result<Vec<RitzCandidate>, SolverError> {
    let mut merged = first;
    for index in 0..second.eigenvalues.len() {
        let eigenvalue = second.eigenvalues[index];
        let residual = second.residual_norms[index];
        let vector: Vec<f64> = (0..second.eigenvectors.nrows())
            .map(|row| second.eigenvectors[[row, index]])
            .collect();
        let mut duplicate = None;
        for (kept_index, kept) in merged.iter().enumerate() {
            let close_value =
                (kept.1 - eigenvalue).abs() <= 1.0e-10 * (1.0 + kept.1.abs() + eigenvalue.abs());
            if !close_value {
                continue;
            }
            if b_cosine(b, &kept.3, &vector)?.abs() > 1.0 - 1.0e-6 {
                duplicate = Some(kept_index);
                break;
            }
        }
        match duplicate {
            Some(kept_index) => {
                if residual < merged[kept_index].2 {
                    merged[kept_index] = (
                        (eigenvalue - config.target).abs(),
                        eigenvalue,
                        residual,
                        vector,
                    );
                }
            }
            None => merged.push((
                (eigenvalue - config.target).abs(),
                eigenvalue,
                residual,
                vector,
            )),
        }
    }
    merged.sort_by(|left, right| left.0.total_cmp(&right.0));
    merged.truncate(limit.min(merged.len()));
    Ok(merged)
}

/// Compute every eigenpair inside an inclusive interval.
///
/// This compatibility entry point maps the residual tolerance to
/// [`feast_generalized`]. The interval result is not truncated by
/// [`ShiftInvertConfig::num_eigenvalues`], because doing so would violate the
/// completeness contract.
pub fn eigenpairs_in_interval(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    lower: f64,
    upper: f64,
    config: &ShiftInvertConfig,
) -> Result<TargetedEigenResult, SolverError> {
    let feast = FeastConfig {
        tolerance: config.tolerance,
        ..Default::default()
    };
    feast_generalized(a, b, lower, upper, &feast)
}

/// Count eigenvalues of a real symmetric pencil in an inclusive interval.
///
/// `B` must be positive definite. The count comes from Sylvester inertia:
/// the number of negative pivots in `A - sigma B` is the number of
/// generalized eigenvalues below `sigma`. A scale-aware endpoint guard makes
/// eigenvalues numerically equal to either endpoint part of the interval.
pub fn eigenvalue_count_in_interval(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    lower: f64,
    upper: f64,
) -> Result<usize, SolverError> {
    validate_interval_inputs(a, b, lower, upper)?;
    SparseCholeskyFactorization::factorize(b).map_err(|error| {
        map_sparse_cholesky_error("interval eigensolver mass validation", error)
    })?;
    let guard = interval_endpoint_guard(lower, upper);
    let below_lower = negative_inertia(a, b, lower - guard)?;
    let through_upper = negative_inertia(a, b, upper + guard)?;
    through_upper
        .checked_sub(below_lower)
        .ok_or_else(|| SolverError::OperationFailed {
            operation: "interval eigenvalue count",
            reason: "inertia count decreased across an ordered interval".into(),
        })
}

/// Compute every eigenpair of `A x = lambda B x` in an inclusive interval.
///
/// This is a FEAST-style contour subspace iteration. Unlike a midpoint
/// shift-invert query, it uses LDLT inertia as an independent completeness
/// certificate, so clustered and repeated eigenvalues cannot silently be
/// omitted. `B` must be symmetric positive definite.
pub fn feast_generalized(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    lower: f64,
    upper: f64,
    config: &FeastConfig,
) -> Result<TargetedEigenResult, SolverError> {
    validate_interval_inputs(a, b, lower, upper)?;
    validate_feast_config(config)?;
    let expected = eigenvalue_count_in_interval(a, b, lower, upper)?;
    if expected == 0 {
        return Ok(TargetedEigenResult {
            eigenvalues: Vec::new(),
            eigenvectors: DenseMatrix::zeros(a.num_rows, 0),
            residual_norms: Vec::new(),
            iterations: 0,
            converged: true,
        });
    }

    let n = a.num_rows;
    let subspace_size = expected.saturating_add(config.subspace_oversampling).min(n);
    let mut subspace = deterministic_subspace(b, subspace_size)?;
    let endpoint_guard = interval_endpoint_guard(lower, upper);
    let mut last_candidates = Vec::new();

    for iteration in 1..=config.max_iterations {
        let projected =
            apply_feast_projector(a, b, lower, upper, config.quadrature_nodes, &subspace)?;
        subspace = b_orthonormalize(b, projected, subspace_size)?;
        if subspace.len() < expected {
            return Err(SolverError::Breakdown {
                operation: "FEAST subspace iteration",
                reason: format!(
                    "spectral projector rank {} is below inertia count {expected}",
                    subspace.len()
                ),
            });
        }

        let projected_a = projected_operator(a, &subspace)?;
        let (ritz_values, ritz_vectors) = symmetric_eigen_ql(&projected_a, subspace.len())?;
        ensure_finite_values("FEAST projected eigenvalues", &ritz_values)?;
        ensure_finite_dense("FEAST projected eigenvectors", &ritz_vectors)?;
        let mut candidates = Vec::new();
        for (ritz_index, &eigenvalue) in ritz_values.iter().enumerate() {
            if eigenvalue < lower - endpoint_guard || eigenvalue > upper + endpoint_guard {
                continue;
            }
            let mut vector = combine_columns(&subspace, &ritz_vectors, ritz_index);
            ensure_finite_values("FEAST Ritz vector reconstruction", &vector)?;
            let mut b_vector = apply(b, &vector, "FEAST Ritz mass application")?;
            normalize_b(&mut vector, &mut b_vector)?;
            let residual = relative_residual(a, b, eigenvalue, &vector)?;
            candidates.push((eigenvalue, residual, vector));
        }
        candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
        if candidates.len() == expected
            && candidates
                .iter()
                .all(|candidate| candidate.1 <= config.tolerance)
        {
            return pack_interval_result(n, candidates, iteration, true);
        }
        last_candidates = candidates;
    }

    let converged = last_candidates.len() == expected
        && last_candidates
            .iter()
            .all(|candidate| candidate.1 <= config.tolerance);
    pack_interval_result(n, last_candidates, config.max_iterations, converged)
}

fn validate_interval_inputs(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    lower: f64,
    upper: f64,
) -> Result<(), SolverError> {
    if a.num_rows == 0
        || a.num_rows != a.num_cols
        || b.num_rows != a.num_rows
        || b.num_cols != a.num_rows
    {
        return Err(SolverError::DimensionMismatch {
            expected: a.num_rows,
            got: b.num_rows.max(b.num_cols),
        });
    }
    ensure_valid_csr("interval eigenvalue operator input", a)?;
    ensure_valid_csr("interval eigenvalue mass input", b)?;
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(SolverError::OperationFailed {
            operation: "interval eigenvalue selection",
            reason: format!("invalid interval [{lower}, {upper}]"),
        });
    }
    Ok(())
}

fn validate_feast_config(config: &FeastConfig) -> Result<(), SolverError> {
    if config.quadrature_nodes < 4
        || !config.quadrature_nodes.is_multiple_of(2)
        || config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
    {
        return Err(SolverError::OperationFailed {
            operation: "FEAST configuration",
            reason: "quadrature nodes must be even and at least four; iterations and tolerance must be positive".into(),
        });
    }
    Ok(())
}

fn interval_endpoint_guard(lower: f64, upper: f64) -> f64 {
    256.0
        * f64::EPSILON
        * lower
            .abs()
            .max(upper.abs())
            .max((upper - lower).abs())
            .max(1.0)
}

fn negative_inertia(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    shift: f64,
) -> Result<usize, SolverError> {
    ensure_finite_values("interval inertia shift", &[shift])?;
    let shifted = shifted_matrix(a, b, shift)?;
    MultifrontalLdltFactorization::factorize(&shifted)
        .map(|factor| factor.inertia().negative)
        .map_err(|error| map_blr_error("interval inertia factorization", error))
}

fn deterministic_subspace(
    b: &CsrMatrix<f64>,
    columns: usize,
) -> Result<Vec<Vec<f64>>, SolverError> {
    let n = b.num_rows;
    let candidates = (0..columns)
        .map(|column| {
            (0..n)
                .map(|row| {
                    let phase = ((row + 1) * (column + 1)) as f64;
                    (phase * std::f64::consts::SQRT_2).sin()
                        + 0.5 * (phase * 0.754_877_666_246_692_7).cos()
                        + f64::from(row == column)
                })
                .collect()
        })
        .collect();
    let basis = b_orthonormalize(b, candidates, columns)?;
    if basis.len() != columns {
        return Err(SolverError::Breakdown {
            operation: "FEAST initial subspace",
            reason: format!(
                "deterministic block has numerical rank {}, expected {columns}",
                basis.len()
            ),
        });
    }
    Ok(basis)
}

fn apply_feast_projector(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    lower: f64,
    upper: f64,
    quadrature_nodes: usize,
    subspace: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, SolverError> {
    let center = 0.5 * (lower + upper);
    let half_width = 0.5 * (upper - lower);
    let real_radius = 1.01 * half_width;
    let imaginary_radius = 0.30 * half_width;
    ensure_finite_values(
        "FEAST contour geometry",
        &[center, half_width, real_radius, imaginary_radius],
    )?;
    let b_subspace: Vec<_> = subspace
        .iter()
        .map(|column| apply(b, column, "FEAST mass application"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut projected = vec![vec![0.0; a.num_rows]; subspace.len()];

    for node in 0..quadrature_nodes {
        let theta = 2.0 * std::f64::consts::PI * (node as f64 + 0.5) / quadrature_nodes as f64;
        let cosine = theta.cos();
        let sine = theta.sin();
        let z = Complex64::new(center + real_radius * cosine, imaginary_radius * sine);
        ensure_finite_complex("FEAST contour node", z)?;
        let matrix = complex_shifted_matrix(a, b, z)?;
        let factor = SparseLuFactorization::factorize(&matrix)
            .map_err(|error| map_sparse_lu_error("FEAST contour factorization", error))?;
        // -dz/(2*pi*i), including the trapezoidal dtheta = 2*pi/N.
        let weight = Complex64::new(
            -imaginary_radius * cosine / quadrature_nodes as f64,
            -real_radius * sine / quadrature_nodes as f64,
        );
        ensure_finite_complex("FEAST contour weight", weight)?;
        for (column, rhs) in b_subspace.iter().enumerate() {
            let complex_rhs: Vec<_> = rhs
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect();
            let solution = factor
                .solve(&complex_rhs)
                .map_err(|error| map_sparse_lu_error("FEAST contour solve", error))?;
            if solution.len() != a.num_rows {
                return Err(SolverError::DimensionMismatch {
                    expected: a.num_rows,
                    got: solution.len(),
                });
            }
            ensure_finite_complex_values("FEAST contour solve", &solution)?;
            for (value, solution_value) in projected[column].iter_mut().zip(solution) {
                *value += (weight * solution_value).re;
            }
            ensure_finite_values("FEAST contour projection", &projected[column])?;
        }
    }
    Ok(projected)
}

fn complex_shifted_matrix(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    shift: Complex64,
) -> Result<CsrMatrix<Complex64>, SolverError> {
    let mut triplets = Vec::with_capacity(a.nnz() + b.nnz());
    for row in 0..a.num_rows {
        triplets.extend(
            a.row_entries(row)
                .map(|(column, value)| (row, column, Complex64::new(value, 0.0))),
        );
        triplets.extend(
            b.row_entries(row)
                .map(|(column, value)| (row, column, -shift * value)),
        );
    }
    CsrMatrix::try_from_triplets(a.num_rows, a.num_cols, triplets).map_err(|error| {
        SolverError::OperationFailed {
            operation: "FEAST shifted-matrix construction",
            reason: error.to_string(),
        }
    })
}

fn b_orthonormalize(
    b: &CsrMatrix<f64>,
    candidates: Vec<Vec<f64>>,
    max_columns: usize,
) -> Result<Vec<Vec<f64>>, SolverError> {
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(max_columns);
    let mut b_basis: Vec<Vec<f64>> = Vec::with_capacity(max_columns);
    for mut candidate in candidates {
        if basis.len() == max_columns {
            break;
        }
        ensure_finite_values("FEAST orthonormalization input", &candidate)?;
        let mut b_candidate = apply(b, &candidate, "FEAST mass application")?;
        for _ in 0..2 {
            for (prior, b_prior) in basis.iter().zip(&b_basis) {
                let coefficient = dot(prior, &b_candidate);
                ensure_finite_values("FEAST B-orthonormalization", &[coefficient])?;
                axpy(-coefficient, prior, &mut candidate);
                axpy(-coefficient, b_prior, &mut b_candidate);
            }
            ensure_finite_values("FEAST B-orthonormalization", &candidate)?;
            ensure_finite_values("FEAST mass orthonormalization", &b_candidate)?;
        }
        let norm_sq = dot(&candidate, &b_candidate);
        if !norm_sq.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "FEAST B-orthonormalization",
                reason: "encountered a non-finite mass norm".into(),
            });
        }
        if norm_sq <= 1.0e-24 {
            continue;
        }
        let norm = norm_sq.sqrt();
        for value in &mut candidate {
            *value /= norm;
        }
        for value in &mut b_candidate {
            *value /= norm;
        }
        ensure_finite_values("FEAST B-orthonormal basis", &candidate)?;
        ensure_finite_values("FEAST mass-orthonormal basis", &b_candidate)?;
        basis.push(candidate);
        b_basis.push(b_candidate);
    }
    Ok(basis)
}

fn projected_operator(
    a: &CsrMatrix<f64>,
    basis: &[Vec<f64>],
) -> Result<DenseMatrix<f64>, SolverError> {
    let a_basis: Vec<_> = basis
        .iter()
        .map(|column| apply(a, column, "FEAST projected operator application"))
        .collect::<Result<Vec<_>, _>>()?;
    let projected = DenseMatrix::from_fn(basis.len(), basis.len(), |row, column| {
        dot(&basis[row], &a_basis[column])
    });
    ensure_finite_dense("FEAST projected operator", &projected)?;
    Ok(projected)
}

fn combine_columns(basis: &[Vec<f64>], coefficients: &DenseMatrix<f64>, column: usize) -> Vec<f64> {
    let mut result = vec![0.0; basis[0].len()];
    for (basis_index, basis_vector) in basis.iter().enumerate() {
        axpy(
            coefficients[[basis_index, column]],
            basis_vector,
            &mut result,
        );
    }
    result
}

fn pack_interval_result(
    rows: usize,
    candidates: Vec<(f64, f64, Vec<f64>)>,
    iterations: usize,
    converged: bool,
) -> Result<TargetedEigenResult, SolverError> {
    if let Some(candidate) = candidates
        .iter()
        .find(|candidate| candidate.2.len() != rows)
    {
        return Err(SolverError::DimensionMismatch {
            expected: rows,
            got: candidate.2.len(),
        });
    }
    let eigenvalues: Vec<f64> = candidates.iter().map(|candidate| candidate.0).collect();
    let residual_norms: Vec<f64> = candidates.iter().map(|candidate| candidate.1).collect();
    let eigenvectors = DenseMatrix::from_fn(rows, candidates.len(), |row, column| {
        candidates[column].2[row]
    });
    ensure_finite_values("FEAST final eigenvalues", &eigenvalues)?;
    ensure_finite_values("FEAST final residuals", &residual_norms)?;
    ensure_finite_dense("FEAST final eigenvectors", &eigenvectors)?;
    Ok(TargetedEigenResult {
        eigenvalues,
        eigenvectors,
        residual_norms,
        iterations,
        converged,
    })
}

fn validate_inputs(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    config: &ShiftInvertConfig,
) -> Result<(), SolverError> {
    if a.num_rows == 0
        || a.num_rows != a.num_cols
        || b.num_rows != a.num_rows
        || b.num_cols != a.num_rows
    {
        return Err(SolverError::DimensionMismatch {
            expected: a.num_rows,
            got: b.num_rows.max(b.num_cols),
        });
    }
    if !config.target.is_finite()
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
        || !config.breakdown_tolerance.is_finite()
        || config.breakdown_tolerance <= 0.0
        || config.num_eigenvalues == 0
        || config.num_eigenvalues > a.num_rows
        || config.krylov_dimension < config.num_eigenvalues
    {
        return Err(SolverError::OperationFailed {
            operation: "shift-invert Lanczos configuration",
            reason: "target, tolerance, breakdown tolerance, eigenvalue count, or Krylov dimension is invalid"
                .into(),
        });
    }
    ensure_valid_csr("shift-invert operator input", a)?;
    ensure_valid_csr("shift-invert mass input", b)?;
    Ok(())
}

fn shifted_matrix(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    shift: f64,
) -> Result<CsrMatrix<f64>, SolverError> {
    ensure_finite_values("shifted-matrix construction", &[shift])?;
    let mut triplets = Vec::with_capacity(a.nnz() + b.nnz());
    for row in 0..a.num_rows {
        triplets.extend(
            a.row_entries(row)
                .map(|(column, value)| (row, column, value)),
        );
        triplets.extend(
            b.row_entries(row)
                .map(|(column, value)| (row, column, -shift * value)),
        );
    }
    let shifted =
        CsrMatrix::try_from_triplets(a.num_rows, a.num_cols, triplets).map_err(|error| {
            SolverError::OperationFailed {
                operation: "shifted-matrix construction",
                reason: error.to_string(),
            }
        })?;
    ensure_finite_values("shifted-matrix construction", &shifted.values)?;
    Ok(shifted)
}

fn apply(
    matrix: &CsrMatrix<f64>,
    vector: &[f64],
    operation: &'static str,
) -> Result<Vec<f64>, SolverError> {
    if vector.len() != matrix.num_cols {
        return Err(SolverError::DimensionMismatch {
            expected: matrix.num_cols,
            got: vector.len(),
        });
    }
    ensure_finite_values(operation, vector)?;
    let result: Vec<_> = (0..matrix.num_rows)
        .map(|row| {
            matrix
                .row_entries(row)
                .map(|(column, value)| value * vector[column])
                .sum()
        })
        .collect();
    ensure_finite_values(operation, &result)?;
    Ok(result)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    // AVX2+FMA dot on x86_64 via the core BLAS helper (scalar fallback
    // elsewhere). Same value up to summation order.
    sonium_solvers_core::blas_helpers::dot_f64(left, right)
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    sonium_solvers_core::blas_helpers::axpy(alpha, x, y);
}

fn normalize_b(vector: &mut [f64], b_vector: &mut [f64]) -> Result<(), SolverError> {
    if vector.len() != b_vector.len() {
        return Err(SolverError::DimensionMismatch {
            expected: vector.len(),
            got: b_vector.len(),
        });
    }
    ensure_finite_values("B-inner-product normalization", vector)?;
    ensure_finite_values("B-inner-product normalization", b_vector)?;
    let norm_sq = dot(vector, b_vector);
    if !norm_sq.is_finite() || norm_sq <= 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "B-inner-product normalization",
            reason: format!("non-positive mass norm {norm_sq}"),
        });
    }
    let norm = norm_sq.sqrt();
    for value in vector.iter_mut() {
        *value /= norm;
    }
    for value in b_vector.iter_mut() {
        *value /= norm;
    }
    ensure_finite_values("B-inner-product normalization", vector)?;
    ensure_finite_values("B-inner-product normalization", b_vector)?;
    Ok(())
}

fn relative_residual(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    eigenvalue: f64,
    vector: &[f64],
) -> Result<f64, SolverError> {
    ensure_finite_values("eigenpair residual eigenvalue", &[eigenvalue])?;
    let ax = apply(a, vector, "eigenpair residual operator application")?;
    let bx = apply(b, vector, "eigenpair residual mass application")?;
    let residual_values: Vec<f64> = ax
        .iter()
        .zip(&bx)
        .map(|(&a_value, &b_value)| a_value - eigenvalue * b_value)
        .collect();
    ensure_finite_values("eigenpair residual", &residual_values)?;
    let residual = stable_norm(&residual_values);
    let scale = stable_norm(&ax) + eigenvalue.abs() * stable_norm(&bx);
    let relative = residual / scale.max(f64::MIN_POSITIVE);
    ensure_finite_values("eigenpair relative residual", &[relative])?;
    Ok(relative)
}

fn numerical_error(operation: &'static str, reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation,
        reason: reason.into(),
    }
}

fn ensure_finite_values(operation: &'static str, values: &[f64]) -> Result<(), SolverError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(numerical_error(operation, "generated NaN or infinity"))
    }
}

fn ensure_finite_complex(operation: &'static str, value: Complex64) -> Result<(), SolverError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(numerical_error(operation, "generated NaN or infinity"))
    }
}

fn ensure_finite_complex_values(
    operation: &'static str,
    values: &[Complex64],
) -> Result<(), SolverError> {
    if values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        Ok(())
    } else {
        Err(numerical_error(operation, "generated NaN or infinity"))
    }
}

fn ensure_finite_dense(
    operation: &'static str,
    matrix: &DenseMatrix<f64>,
) -> Result<(), SolverError> {
    ensure_finite_values(operation, matrix.as_slice())
}

fn ensure_valid_csr(operation: &'static str, matrix: &CsrMatrix<f64>) -> Result<(), SolverError> {
    matrix
        .validate_structure()
        .map_err(|error| numerical_error(operation, error.to_string()))?;
    ensure_finite_values(operation, &matrix.values)
}

fn map_sparse_lu_error(operation: &'static str, error: SparseLuError) -> SolverError {
    match &error {
        SparseLuError::DimensionMismatch { expected, actual } => SolverError::DimensionMismatch {
            expected: *expected,
            got: *actual,
        },
        SparseLuError::MatrixDimensionMismatch {
            expected,
            rows,
            cols,
        } => SolverError::DimensionMismatch {
            expected: *expected,
            got: (*rows).max(*cols),
        },
        SparseLuError::NotSquare { rows, cols } => SolverError::DimensionMismatch {
            expected: *rows,
            got: (*rows).max(*cols),
        },
        _ => SolverError::from_failure(operation, &error),
    }
}

fn map_sparse_cholesky_error(operation: &'static str, error: SparseCholeskyError) -> SolverError {
    match &error {
        SparseCholeskyError::DimensionMismatch { expected, actual } => {
            SolverError::DimensionMismatch {
                expected: *expected,
                got: *actual,
            }
        }
        SparseCholeskyError::NotSquare { rows, cols } => SolverError::DimensionMismatch {
            expected: *rows,
            got: (*rows).max(*cols),
        },
        _ => SolverError::from_failure(operation, &error),
    }
}

fn map_blr_error(operation: &'static str, error: BlrError) -> SolverError {
    match &error {
        BlrError::DimensionMismatch { expected, actual } => SolverError::DimensionMismatch {
            expected: *expected,
            got: *actual,
        },
        BlrError::NotSquare { rows, cols } => SolverError::DimensionMismatch {
            expected: *rows,
            got: (*rows).max(*cols),
        },
        _ => SolverError::from_failure(operation, &error),
    }
}

fn stable_norm(values: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut sum_squares = 1.0_f64;
    for &value in values {
        let magnitude = value.abs();
        if !magnitude.is_finite() {
            return magnitude;
        }
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
    if scale == 0.0 {
        0.0
    } else {
        scale * sum_squares.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal(values: &[f64]) -> CsrMatrix<f64> {
        CsrMatrix::from_triplets(
            values.len(),
            values.len(),
            values
                .iter()
                .enumerate()
                .map(|(index, &value)| (index, index, value))
                .collect(),
        )
    }

    #[test]
    fn targets_interior_generalized_eigenpairs() {
        let mass = diagonal(&[1.0, 2.0, 1.5, 0.5, 3.0, 2.5]);
        let expected = [1.0, 3.0, 5.0, 8.0, 13.0, 21.0];
        let stiffness = diagonal(
            &expected
                .iter()
                .zip(&mass.values)
                .map(|(&lambda, &weight)| lambda * weight)
                .collect::<Vec<_>>(),
        );
        let result = shift_invert_lanczos_generalized(
            &stiffness,
            &mass,
            &ShiftInvertConfig {
                target: 7.0,
                num_eigenvalues: 3,
                krylov_dimension: 6,
                tolerance: 1.0e-10,
                breakdown_tolerance: 1.0e-14,
            },
        )
        .unwrap();
        assert_eq!(result.eigenvalues.len(), 3);
        assert!((result.eigenvalues[0] - 8.0).abs() < 1.0e-10);
        assert!((result.eigenvalues[1] - 5.0).abs() < 1.0e-10);
        assert!((result.eigenvalues[2] - 3.0).abs() < 1.0e-10);
        assert!(result.converged);
    }

    #[test]
    fn interval_returns_inertia_certified_complete_set() {
        let stiffness = diagonal(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
        let mass = diagonal(&[1.0; 6]);
        let result = eigenpairs_in_interval(
            &stiffness,
            &mass,
            8.0,
            26.0,
            &ShiftInvertConfig {
                num_eigenvalues: 5,
                krylov_dimension: 6,
                tolerance: 1.0e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.eigenvalues.len(), 3);
        assert!(result.converged);
        for (&actual, &expected) in result.eigenvalues.iter().zip(&[9.0, 16.0, 25.0]) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }

    #[test]
    fn interval_preserves_repeated_eigenspace_multiplicity() {
        let stiffness = diagonal(&[
            1.0, 4.0, 4.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0,
            169.0, 196.0, 225.0, 256.0, 289.0, 324.0,
        ]);
        let mass = diagonal(&[1.0; 20]);
        assert_eq!(
            eigenvalue_count_in_interval(&stiffness, &mass, 3.0, 5.0).unwrap(),
            3
        );
        let result = feast_generalized(
            &stiffness,
            &mass,
            3.0,
            5.0,
            &FeastConfig {
                tolerance: 1.0e-10,
                subspace_oversampling: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged);
        assert_eq!(result.eigenvalues.len(), 3);
        assert!(
            result
                .eigenvalues
                .iter()
                .all(|eigenvalue| (*eigenvalue - 4.0).abs() < 1.0e-10)
        );
    }

    #[test]
    fn feast_projects_a_strict_subspace_for_coupled_matrix() {
        let mut triplets = vec![(0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)];
        triplets.extend((2..12).map(|index| (index, index, 10.0 + index as f64)));
        let stiffness = CsrMatrix::from_triplets(12, 12, triplets);
        let mass = diagonal(&[1.0; 12]);
        let result = feast_generalized(
            &stiffness,
            &mass,
            0.5,
            1.5,
            &FeastConfig {
                subspace_oversampling: 3,
                tolerance: 1.0e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert_eq!(result.eigenvalues.len(), 1);
        assert!((result.eigenvalues[0] - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn interval_is_inclusive_at_both_endpoints() {
        let stiffness = diagonal(&[1.0, 4.0, 9.0, 16.0]);
        let mass = diagonal(&[1.0; 4]);
        assert_eq!(
            eigenvalue_count_in_interval(&stiffness, &mass, 4.0, 9.0).unwrap(),
            2
        );
        let result = feast_generalized(
            &stiffness,
            &mass,
            4.0,
            9.0,
            &FeastConfig {
                tolerance: 1.0e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged);
        assert_eq!(result.eigenvalues.len(), 2);
        assert!((result.eigenvalues[0] - 4.0).abs() < 1.0e-10);
        assert!((result.eigenvalues[1] - 9.0).abs() < 1.0e-10);
    }

    #[test]
    fn interval_rejects_non_positive_definite_mass() {
        let stiffness = diagonal(&[1.0, 2.0, 3.0]);
        let mass = diagonal(&[1.0, -1.0, 1.0]);
        let error = eigenvalue_count_in_interval(&stiffness, &mass, 0.0, 4.0).unwrap_err();
        assert!(error.to_string().contains("positive definite"));
    }

    #[test]
    fn deterministic_streams_are_stable_and_distinct() {
        let n = 8;
        let legacy: Vec<f64> = (0..n)
            .map(|index| {
                let x = (index + 1) as f64;
                (x * std::f64::consts::GOLDEN_RATIO).sin()
                    + 0.5 * (x * 0.754_877_666_246_692_7).cos()
            })
            .collect();
        // Stream 0 reproduces the historical start bit-for-bit.
        assert_eq!(deterministic_start(n, 0), legacy);
        assert_eq!(deterministic_start(n, 0), deterministic_start(n, 0));
        // The restart stream is a genuinely different direction.
        let second = deterministic_start(n, 1);
        assert!(
            second
                .iter()
                .zip(&legacy)
                .any(|(a, b)| (a - b).abs() > 1.0e-3),
            "restart stream must differ from the primary stream"
        );
    }

    #[test]
    fn restart_seed_is_b_orthogonal_to_found_pairs() {
        let b = diagonal(&[1.0, 1.0, 1.0]);
        let found: Vec<RitzCandidate> = vec![(0.1, 1.0, 1.0e-12, vec![1.0, 0.0, 0.0])];
        let seed = restart_seed(&b, &found, 3).expect("orthogonalizable seed");
        let b_seed = apply(&b, &seed, "test").expect("mass apply");
        // B-orthogonal to the found Ritz vector and B-normalized.
        assert!(dot(&found[0].3, &b_seed).abs() < 1.0e-12);
        assert!((dot(&seed, &b_seed) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn merge_candidate_sets_dedupes_sharpens_and_orders() {
        let b = diagonal(&[1.0, 1.0]);
        let config = ShiftInvertConfig {
            target: 0.0,
            ..ShiftInvertConfig::default()
        };
        let first: Vec<RitzCandidate> = vec![(1.0, 1.0, 1.0e-12, vec![1.0, 0.0])];
        let second = TargetedEigenResult {
            eigenvalues: vec![1.0 + 1.0e-13, 5.0],
            eigenvectors: DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            residual_norms: vec![1.0e-9, 1.0e-11],
            iterations: 2,
            converged: true,
        };
        let merged =
            merge_candidate_sets(&b, first, &second, &config, 2).expect("merge must succeed");
        // Duplicate (1.0 vs 1.0+1e-13, collinear) keeps the sharper residual;
        // the genuinely new pair joins; ordering is by target distance.
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].1, 1.0);
        assert_eq!(merged[0].2, 1.0e-12);
        assert_eq!(merged[1].1, 5.0);
    }
}
