//! Complex nonsymmetric shift-invert Arnoldi for generalized eigenproblems.

use num_complex::Complex64;
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, SolverError, SolverResultContract, SolverStatus},
};
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_sparse::CsrMatrix;

use crate::result_contract::complex_result_shape_valid;

type ComplexDense = Vec<Vec<Complex64>>;

/// Configuration for complex eigenvalues nearest a spectral target.
#[derive(Debug, Clone)]
pub struct ComplexArnoldiConfig {
    /// Complex shift used by the shift-invert transformation.
    pub target: Complex64,
    /// Number of eigenpairs requested.
    pub num_eigenvalues: usize,
    /// Maximum Arnoldi basis dimension.
    pub krylov_dimension: usize,
    /// Relative true-residual target.
    pub tolerance: f64,
    /// Maximum shifted-QR iterations for the projected Hessenberg matrix.
    pub max_qr_iterations: usize,
    /// Relative breakdown threshold: basis expansion stops when the
    /// candidate norm falls below `breakdown_tolerance` times the running
    /// Hessenberg-entry scale, so over/under-scaled pencils neither miss a
    /// genuine exhaustion nor stop on a merely small direction.
    pub breakdown_tolerance: f64,
}

impl Default for ComplexArnoldiConfig {
    fn default() -> Self {
        Self {
            target: Complex64::new(0.0, 0.0),
            num_eigenvalues: 6,
            krylov_dimension: 40,
            tolerance: 1.0e-8,
            max_qr_iterations: 4_000,
            breakdown_tolerance: 1.0e-14,
        }
    }
}

/// Ritz eigenpairs returned by complex shift-invert Arnoldi.
#[derive(Debug, Clone)]
pub struct ComplexTargetedEigenResult {
    /// Eigenvalues ordered by distance to the requested target.
    pub eigenvalues: Vec<Complex64>,
    /// Euclidean-normalized right eigenvectors stored as columns.
    pub eigenvectors: DenseMatrix<Complex64>,
    /// Independently recomputed relative true residuals.
    pub residual_norms: Vec<f64>,
    /// Number of Arnoldi vectors used.
    pub iterations: usize,
    /// Whether every requested Ritz pair met the residual tolerance.
    pub converged: bool,
}

impl ComplexTargetedEigenResult {
    /// Return the canonical status for the returned Ritz pairs.
    pub fn solver_status(&self) -> SolverStatus {
        let shape_valid = complex_result_shape_valid(
            &self.eigenvalues,
            &self.eigenvectors,
            &self.residual_norms,
            false,
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

impl SolverResultContract for ComplexTargetedEigenResult {
    fn solver_status(&self) -> SolverStatus {
        ComplexTargetedEigenResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        ComplexTargetedEigenResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        let status = ComplexTargetedEigenResult::solver_status(self);
        let accuracy = ComplexTargetedEigenResult::accuracy_status(self);
        if status == SolverStatus::NumericalError {
            return Err(SolverError::OperationFailed {
                operation: "complex eigen result contract",
                reason: "result shape or diagnostic is non-finite".into(),
            });
        }
        if (status == SolverStatus::Converged) != (accuracy == AccuracyStatus::Attained) {
            return Err(SolverError::OperationFailed {
                operation: "complex eigen result contract",
                reason: "status and accuracy disagree".into(),
            });
        }
        Ok(())
    }
}

/// Compute eigenpairs of `A x = lambda B x` nearest a complex target.
///
/// A sparse LU factorization of `A - target B` is reused by a fully
/// reorthogonalized Arnoldi iteration. The small projected Hessenberg problem
/// is solved by implicit single-shift complex QR with deflation.
///
/// On lucky breakdown with fewer candidates than requested, a second
/// deterministic start stream — orthogonalized against the found Ritz
/// vectors — is tried and both candidate sets are merged, instead of failing
/// on the single unlucky start.
pub fn complex_shift_invert_arnoldi_generalized(
    a: &CsrMatrix<Complex64>,
    b: &CsrMatrix<Complex64>,
    config: &ComplexArnoldiConfig,
) -> Result<ComplexTargetedEigenResult, SolverError> {
    let initial = complex_deterministic_start(a.num_rows, 0);
    complex_shift_invert_arnoldi_with_seed(a, b, config, initial, 0)
}

fn complex_shift_invert_arnoldi_with_seed(
    a: &CsrMatrix<Complex64>,
    b: &CsrMatrix<Complex64>,
    config: &ComplexArnoldiConfig,
    initial: Vec<Complex64>,
    seed_stream: u64,
) -> Result<ComplexTargetedEigenResult, SolverError> {
    validate_inputs(a, b, config)?;
    let n = a.num_rows;
    let max_basis = config.krylov_dimension.min(n);
    let shifted = shifted_matrix(a, b, config.target)?;
    let factor = SparseLuFactorization::factorize(&shifted)
        .map_err(|error| map_sparse_lu_error("complex shift-invert sparse factorization", error))?;
    let mut initial = initial;
    if initial.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: initial.len(),
        });
    }
    normalize(&mut initial)?;
    let mut basis = vec![initial];
    let mut h = vec![vec![Complex64::new(0.0, 0.0); max_basis]; max_basis + 1];
    let mut dimension = 0;
    let mut breakdown = false;
    // Running Hessenberg-entry scale for the relative breakdown test: the
    // projection coefficients share the candidate-norm scale.
    let mut hessenberg_scale = 0.0f64;
    for column in 0..max_basis {
        let rhs = apply(b, &basis[column], "complex Arnoldi mass application")?;
        let mut candidate = factor
            .solve(&rhs)
            .map_err(|error| map_sparse_lu_error("complex shift-invert triangular solve", error))?;
        if candidate.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: candidate.len(),
            });
        }
        ensure_finite_vector("complex shift-invert triangular solve", &candidate)?;
        for _ in 0..2 {
            for row in 0..=column {
                let coefficient = dotc(&basis[row], &candidate);
                ensure_finite_value("complex Arnoldi orthogonalization", coefficient)?;
                h[row][column] += coefficient;
                ensure_finite_value("complex Arnoldi Hessenberg update", h[row][column])?;
                axpy(-coefficient, &basis[row], &mut candidate);
            }
            ensure_finite_vector("complex Arnoldi orthogonalization", &candidate)?;
        }
        dimension = column + 1;
        if column + 1 == max_basis {
            break;
        }
        let beta = norm(&candidate);
        if !beta.is_finite() {
            return Err(numerical_error(
                "complex Arnoldi basis expansion",
                "candidate norm is NaN or infinity",
            ));
        }
        for entry in h.iter().take(column + 1) {
            hessenberg_scale = hessenberg_scale.max(entry[column].norm());
        }
        h[column + 1][column] = Complex64::new(beta, 0.0);
        // Relative lucky-breakdown test (see the real path): an absolute
        // cutoff was only valid at unit operator scale.
        if beta <= config.breakdown_tolerance * hessenberg_scale {
            breakdown = true;
            break;
        }
        for value in &mut candidate {
            *value /= beta;
        }
        ensure_finite_vector("complex Arnoldi basis expansion", &candidate)?;
        basis.push(candidate);
    }

    let projected: Vec<Vec<_>> = (0..dimension)
        .map(|row| h[row][..dimension].to_vec())
        .collect();
    ensure_finite_dense("complex Arnoldi projected matrix", &projected)?;
    let transformed = complex_qr_eigenvalues(
        projected.clone(),
        config.tolerance.min(1.0e-10),
        config.max_qr_iterations,
    )?;
    ensure_finite_vector("projected complex QR eigensolve", &transformed)?;
    let mut candidates = Vec::new();
    // Projected spectral radius: drop transformed Ritz values far below it
    // (eigenvalues far from the target) relative to this scale, so a shift
    // landing near an eigenvalue (huge mu, the most-wanted pair) is never
    // discarded by an absolute cutoff.
    let mu_scale = transformed
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    for mu in transformed {
        if mu.norm() <= config.tolerance * mu_scale {
            continue;
        }
        let lambda = config.target + Complex64::new(1.0, 0.0) / mu;
        ensure_finite_value("complex Arnoldi Ritz value", lambda)?;
        let projected_vector = inverse_iteration(&projected, mu)?;
        let mut vector = vec![Complex64::new(0.0, 0.0); n];
        for (coefficient, basis_vector) in projected_vector.iter().zip(&basis) {
            axpy(*coefficient, basis_vector, &mut vector);
        }
        normalize(&mut vector)?;
        let residual = relative_residual(a, b, lambda, &vector)?;
        let target_distance = (lambda - config.target).norm();
        if !target_distance.is_finite() {
            return Err(numerical_error(
                "complex Arnoldi Ritz ordering",
                "target distance is NaN or infinity",
            ));
        }
        candidates.push((target_distance, lambda, residual, vector));
    }
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    candidates.truncate(config.num_eigenvalues.min(candidates.len()));
    // Lucky-breakdown restart (first stream only): mirror of the real path —
    // orthogonalize the second deterministic stream against the found Ritz
    // vectors and merge both candidate sets. Any retry failure falls through
    // to the shortfall handling below.
    let (candidates, dimension) =
        if seed_stream == 0 && breakdown && candidates.len() < config.num_eigenvalues {
            match complex_restart_seed(&candidates, n)
                .map(|second| complex_shift_invert_arnoldi_with_seed(a, b, config, second, 1))
            {
                Ok(Ok(second)) => {
                    let dimension = dimension.max(second.iterations);
                    let merged = merge_complex_candidate_sets(
                        &candidates,
                        &second,
                        config,
                        config.num_eigenvalues,
                    )?;
                    (merged, dimension)
                }
                _ => (candidates, dimension),
            }
        } else {
            (candidates, dimension)
        };
    // Retry streams return partial candidate sets for the restart merge
    // (see the real path); only the primary stream reports shortfall here.
    if seed_stream == 0 && breakdown && candidates.len() < config.num_eigenvalues {
        return Err(SolverError::Breakdown {
            operation: "complex Arnoldi basis expansion",
            reason: format!(
                "Krylov space terminated after {dimension} vectors, but {} eigenpairs were requested",
                config.num_eigenvalues
            ),
        });
    }
    let eigenvalues: Vec<Complex64> = candidates.iter().map(|entry| entry.1).collect();
    let residual_norms: Vec<_> = candidates.iter().map(|entry| entry.2).collect();
    let eigenvectors =
        DenseMatrix::from_fn(n, candidates.len(), |row, column| candidates[column].3[row]);
    ensure_finite_vector("complex Arnoldi final eigenvalues", &eigenvalues)?;
    ensure_finite_real_values("complex Arnoldi final residuals", &residual_norms)?;
    ensure_finite_vector(
        "complex Arnoldi final eigenvectors",
        eigenvectors.as_slice(),
    )?;
    let converged = candidates.len() == config.num_eigenvalues
        && residual_norms
            .iter()
            .all(|&value| value <= config.tolerance);
    Ok(ComplexTargetedEigenResult {
        eigenvalues,
        eigenvectors,
        residual_norms,
        iterations: dimension,
        converged,
    })
}

/// A complex Ritz candidate: (distance to target, eigenvalue, relative true
/// residual, Euclidean-normalized eigenvector).
type ComplexRitzCandidate = (f64, Complex64, f64, Vec<Complex64>);

/// Deterministic Arnoldi start-vector stream.
///
/// Stream 0 reproduces the historical start bit-for-bit so existing results
/// don't shift; higher streams use an independent frequency pair for
/// breakdown restarts.
fn complex_deterministic_start(n: usize, stream: u64) -> Vec<Complex64> {
    let (first, second) = if stream == 0 {
        (1.37, 0.73)
    } else {
        (2.11, 1.07)
    };
    (0..n)
        .map(|index| {
            let x = (index + 1) as f64;
            Complex64::new((first * x).sin(), (second * x).cos())
        })
        .collect()
}

/// Build the breakdown-restart start vector: the next deterministic stream,
/// orthogonalized against the already-found Ritz vectors with the same two
/// passes the main loop uses (Euclidean inner product, mirroring it).
fn complex_restart_seed(
    found: &[ComplexRitzCandidate],
    n: usize,
) -> Result<Vec<Complex64>, SolverError> {
    let mut seed = complex_deterministic_start(n, 1);
    for _ in 0..2 {
        for (_, _, _, vector) in found {
            let coefficient = dotc(vector, &seed);
            ensure_finite_value("complex Arnoldi restart orthogonalization", coefficient)?;
            axpy(-coefficient, vector, &mut seed);
        }
    }
    normalize(&mut seed)?;
    Ok(seed)
}

/// Sign-insensitive cosine between two vectors (near ±1 for copies of one
/// eigenvector, ~0 for distinct Ritz pairs). NaN-safe direction: non-finite
/// inputs score NaN, which never exceeds the duplicate threshold, so suspect
/// pairs are kept, never merged.
fn complex_cosine(left: &[Complex64], right: &[Complex64]) -> f64 {
    let norms = norm(left) * norm(right);
    if norms.is_nan() || norms <= 0.0 {
        return 0.0;
    }
    dotc(left, right).norm() / norms
}

/// Merge a second run's eigenpairs into the first run's candidates with the
/// same conservative duplicate rule as the real path, then re-sort by target
/// distance and truncate.
fn merge_complex_candidate_sets(
    first: &[ComplexRitzCandidate],
    second: &ComplexTargetedEigenResult,
    config: &ComplexArnoldiConfig,
    limit: usize,
) -> Result<Vec<ComplexRitzCandidate>, SolverError> {
    let mut merged: Vec<ComplexRitzCandidate> = first.to_vec();
    for index in 0..second.eigenvalues.len() {
        let eigenvalue = second.eigenvalues[index];
        let residual = second.residual_norms[index];
        let vector: Vec<Complex64> = (0..second.eigenvectors.nrows())
            .map(|row| second.eigenvectors[[row, index]])
            .collect();
        ensure_finite_vector("complex Arnoldi restart merge vector", &vector)?;
        ensure_finite_value("complex Arnoldi restart merge eigenvalue", eigenvalue)?;
        let mut duplicate = None;
        for (kept_index, kept) in merged.iter().enumerate() {
            let close_value =
                (kept.1 - eigenvalue).norm() <= 1.0e-10 * (1.0 + kept.1.norm() + eigenvalue.norm());
            if !close_value {
                continue;
            }
            if complex_cosine(&kept.3, &vector).abs() > 1.0 - 1.0e-6 {
                duplicate = Some(kept_index);
                break;
            }
        }
        match duplicate {
            Some(kept_index) => {
                if residual < merged[kept_index].2 {
                    merged[kept_index] = (
                        (eigenvalue - config.target).norm(),
                        eigenvalue,
                        residual,
                        vector,
                    );
                }
            }
            None => merged.push((
                (eigenvalue - config.target).norm(),
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

fn validate_inputs(
    a: &CsrMatrix<Complex64>,
    b: &CsrMatrix<Complex64>,
    config: &ComplexArnoldiConfig,
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
    if !finite(config.target)
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
        || !config.breakdown_tolerance.is_finite()
        || config.breakdown_tolerance <= 0.0
        || config.num_eigenvalues == 0
        || config.num_eigenvalues > a.num_rows
        || config.krylov_dimension < config.num_eigenvalues
        || config.max_qr_iterations == 0
    {
        return Err(SolverError::OperationFailed {
            operation: "complex shift-invert Arnoldi configuration",
            reason: "target, tolerance, breakdown tolerance, eigenvalue count, Krylov dimension, or QR limit is invalid"
                .into(),
        });
    }
    ensure_valid_csr("complex Arnoldi operator input", a)?;
    ensure_valid_csr("complex Arnoldi mass input", b)?;
    Ok(())
}

fn shifted_matrix(
    a: &CsrMatrix<Complex64>,
    b: &CsrMatrix<Complex64>,
    shift: Complex64,
) -> Result<CsrMatrix<Complex64>, SolverError> {
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
                operation: "complex shift-invert matrix construction",
                reason: error.to_string(),
            }
        })?;
    ensure_valid_csr("complex shift-invert matrix construction", &shifted)?;
    Ok(shifted)
}

fn complex_qr_eigenvalues(
    mut matrix: ComplexDense,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<Complex64>, SolverError> {
    ensure_finite_dense("projected complex QR input", &matrix)?;
    let n = matrix.len();
    let mut active = n;
    let mut iterations = 0;
    while active > 1 {
        let scale =
            matrix[active - 1][active - 1].norm() + matrix[active - 2][active - 2].norm() + 1.0;
        if !scale.is_finite() {
            return Err(numerical_error(
                "projected complex QR eigensolve",
                "deflation scale is NaN or infinity",
            ));
        }
        if matrix[active - 1][active - 2].norm() <= tolerance * scale {
            matrix[active - 1][active - 2] = Complex64::new(0.0, 0.0);
            active -= 1;
            continue;
        }
        if iterations >= max_iterations {
            let residual = matrix[active - 1][active - 2].norm();
            return Err(SolverError::ConvergenceError {
                status: SolverStatus::MaxIterationsReached,
                iterations,
                residual,
            });
        }
        let shift = matrix[active - 1][active - 1];
        let mut block = vec![vec![Complex64::new(0.0, 0.0); active]; active];
        for row in 0..active {
            for column in 0..active {
                block[row][column] = matrix[row][column];
            }
            block[row][row] -= shift;
        }
        let (q, r) = dense_qr(&block)?;
        let product = dense_multiply(&r, &q);
        ensure_finite_dense("projected complex QR multiplication", &product)?;
        for row in 0..active {
            for column in 0..active {
                matrix[row][column] = product[row][column];
            }
            matrix[row][row] += shift;
        }
        ensure_finite_dense("projected complex QR update", &matrix)?;
        iterations += 1;
    }
    let eigenvalues: Vec<_> = (0..n).map(|index| matrix[index][index]).collect();
    ensure_finite_vector("projected complex QR eigensolve", &eigenvalues)?;
    Ok(eigenvalues)
}

fn dense_qr(matrix: &[Vec<Complex64>]) -> Result<(ComplexDense, ComplexDense), SolverError> {
    ensure_finite_dense("projected complex QR factorization", matrix)?;
    let n = matrix.len();
    let mut q_columns = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut r = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for column in 0..n {
        let mut vector: Vec<_> = (0..n).map(|row| matrix[row][column]).collect();
        for (prior, q_column) in q_columns.iter().enumerate().take(column) {
            let coefficient = dotc(q_column, &vector);
            r[prior][column] = coefficient;
            axpy(-coefficient, q_column, &mut vector);
        }
        let length = norm(&vector);
        if !length.is_finite() {
            return Err(numerical_error(
                "projected complex QR factorization",
                "column norm is NaN or infinity",
            ));
        }
        if length <= 1.0e-14 {
            vector.fill(Complex64::new(0.0, 0.0));
            vector[column] = Complex64::new(1.0, 0.0);
            for q_column in q_columns.iter().take(column) {
                let coefficient = dotc(q_column, &vector);
                axpy(-coefficient, q_column, &mut vector);
            }
            normalize(&mut vector)?;
        } else {
            for value in &mut vector {
                *value /= length;
            }
            r[column][column] = Complex64::new(length, 0.0);
        }
        q_columns[column] = vector;
    }
    let q: ComplexDense = (0..n)
        .map(|row| (0..n).map(|column| q_columns[column][row]).collect())
        .collect();
    ensure_finite_dense("projected complex QR factorization", &q)?;
    ensure_finite_dense("projected complex QR factorization", &r)?;
    Ok((q, r))
}

fn dense_multiply(left: &[Vec<Complex64>], right: &[Vec<Complex64>]) -> ComplexDense {
    let n = left.len();
    (0..n)
        .map(|row| {
            (0..n)
                .map(|column| {
                    (0..n)
                        .map(|inner| left[row][inner] * right[inner][column])
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn inverse_iteration(
    matrix: &[Vec<Complex64>],
    eigenvalue: Complex64,
) -> Result<Vec<Complex64>, SolverError> {
    ensure_finite_dense("projected inverse iteration input", matrix)?;
    ensure_finite_value("projected inverse iteration eigenvalue", eigenvalue)?;
    let n = matrix.len();
    let perturbation = 1.0e-9 * eigenvalue.norm().max(1.0);
    if !perturbation.is_finite() {
        return Err(numerical_error(
            "projected inverse iteration",
            "eigenvalue perturbation is NaN or infinity",
        ));
    }
    let shift = eigenvalue + Complex64::new(perturbation, perturbation * 0.37);
    let mut vector: Vec<_> = (0..n)
        .map(|i| Complex64::new(1.0, (i + 1) as f64 / n as f64))
        .collect();
    normalize(&mut vector)?;
    for _ in 0..10 {
        let mut shifted = matrix.to_vec();
        for (index, row) in shifted.iter_mut().enumerate() {
            row[index] -= shift;
        }
        vector = dense_solve(shifted, vector)?;
        normalize(&mut vector)?;
    }
    Ok(vector)
}

fn dense_solve(
    mut matrix: Vec<Vec<Complex64>>,
    mut rhs: Vec<Complex64>,
) -> Result<Vec<Complex64>, SolverError> {
    ensure_finite_dense("projected inverse iteration solve", &matrix)?;
    let n = matrix.len();
    if rhs.len() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }
    ensure_finite_vector("projected inverse iteration right-hand side", &rhs)?;
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| {
                matrix[a][column]
                    .norm()
                    .total_cmp(&matrix[b][column].norm())
            })
            .ok_or_else(|| {
                numerical_error("projected inverse iteration", "empty pivot search range")
            })?;
        if matrix[pivot][column].norm() <= 1.0e-15 {
            return Err(SolverError::Breakdown {
                operation: "projected inverse iteration",
                reason: "singular shifted Ritz matrix".into(),
            });
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let pivot_row = matrix[column].clone();
        for row in column + 1..n {
            let factor = matrix[row][column] / pivot_row[column];
            let pivot_rhs = rhs[column];
            rhs[row] -= factor * pivot_rhs;
            for (entry, value) in matrix[row].iter_mut().enumerate().skip(column) {
                *value -= factor * pivot_row[entry];
            }
        }
        ensure_finite_dense("projected inverse iteration elimination", &matrix)?;
        ensure_finite_vector("projected inverse iteration elimination", &rhs)?;
    }
    let mut solution = vec![Complex64::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let tail: Complex64 = (row + 1..n)
            .map(|column| matrix[row][column] * solution[column])
            .sum();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    ensure_finite_vector("projected inverse iteration solution", &solution)?;
    Ok(solution)
}

fn apply(
    matrix: &CsrMatrix<Complex64>,
    vector: &[Complex64],
    operation: &'static str,
) -> Result<Vec<Complex64>, SolverError> {
    if vector.len() != matrix.num_cols {
        return Err(SolverError::DimensionMismatch {
            expected: matrix.num_cols,
            got: vector.len(),
        });
    }
    ensure_finite_vector(operation, vector)?;
    let result: Vec<_> = (0..matrix.num_rows)
        .map(|row| {
            matrix
                .row_entries(row)
                .map(|(column, value)| value * vector[column])
                .sum()
        })
        .collect();
    ensure_finite_vector(operation, &result)?;
    Ok(result)
}

fn dotc(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter().zip(right).map(|(&a, &b)| a.conj() * b).sum()
}

fn axpy(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
    for (&x_value, y_value) in x.iter().zip(y) {
        *y_value += alpha * x_value;
    }
}

fn norm(vector: &[Complex64]) -> f64 {
    // Scaled sum-of-squares (the BLAS nrm2 recurrence) avoids overflowing
    // while the mathematical norm is still representable and never masks a
    // NaN through `f64::max`.
    let mut scale = 0.0_f64;
    let mut sum_squares = 1.0_f64;
    for component in vector.iter().flat_map(|value| [value.re, value.im]) {
        let magnitude = component.abs();
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

fn normalize(vector: &mut [Complex64]) -> Result<(), SolverError> {
    let length = norm(vector);
    if !length.is_finite() || length <= 1.0e-15 {
        return Err(SolverError::OperationFailed {
            operation: "complex Arnoldi normalization",
            reason: format!("invalid vector norm {length}"),
        });
    }
    for value in vector {
        *value /= length;
    }
    Ok(())
}

fn relative_residual(
    a: &CsrMatrix<Complex64>,
    b: &CsrMatrix<Complex64>,
    eigenvalue: Complex64,
    vector: &[Complex64],
) -> Result<f64, SolverError> {
    let ax = apply(a, vector, "complex Arnoldi final operator application")?;
    let bx = apply(b, vector, "complex Arnoldi final mass application")?;
    let residual = ax
        .iter()
        .zip(&bx)
        .map(|(&av, &bv)| (av - eigenvalue * bv).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let relative = residual / (norm(&ax) + eigenvalue.norm() * norm(&bx)).max(f64::MIN_POSITIVE);
    if !relative.is_finite() {
        return Err(numerical_error(
            "complex Arnoldi final residual",
            "relative residual is NaN or infinity",
        ));
    }
    Ok(relative)
}

fn finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn numerical_error(operation: &'static str, reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation,
        reason: reason.into(),
    }
}

fn ensure_finite_value(operation: &'static str, value: Complex64) -> Result<(), SolverError> {
    if finite(value) {
        Ok(())
    } else {
        Err(numerical_error(operation, "generated NaN or infinity"))
    }
}

fn ensure_finite_vector(operation: &'static str, values: &[Complex64]) -> Result<(), SolverError> {
    if values.iter().copied().all(finite) {
        Ok(())
    } else {
        Err(numerical_error(operation, "generated NaN or infinity"))
    }
}

fn ensure_finite_real_values(operation: &'static str, values: &[f64]) -> Result<(), SolverError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(numerical_error(operation, "generated NaN or infinity"))
    }
}

fn ensure_finite_dense(
    operation: &'static str,
    matrix: &[Vec<Complex64>],
) -> Result<(), SolverError> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err(numerical_error(
            operation,
            "matrix must be non-empty and square",
        ));
    }
    if matrix.iter().flatten().copied().all(finite) {
        Ok(())
    } else {
        Err(numerical_error(operation, "generated NaN or infinity"))
    }
}

fn ensure_valid_csr(
    operation: &'static str,
    matrix: &CsrMatrix<Complex64>,
) -> Result<(), SolverError> {
    matrix
        .validate_structure()
        .map_err(|error| numerical_error(operation, error.to_string()))?;
    ensure_finite_vector(operation, &matrix.values)
}

fn map_sparse_lu_error(
    operation: &'static str,
    error: sonium_solvers_direct::SparseLuError,
) -> SolverError {
    match &error {
        sonium_solvers_direct::SparseLuError::DimensionMismatch { expected, actual } => {
            SolverError::DimensionMismatch {
                expected: *expected,
                got: *actual,
            }
        }
        sonium_solvers_direct::SparseLuError::MatrixDimensionMismatch {
            expected,
            rows,
            cols,
        } => SolverError::DimensionMismatch {
            expected: *expected,
            got: (*rows).max(*cols),
        },
        sonium_solvers_direct::SparseLuError::NotSquare { rows, cols } => {
            SolverError::DimensionMismatch {
                expected: *rows,
                got: (*rows).max(*cols),
            }
        }
        _ => SolverError::from_failure(operation, &error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_solve_rejects_empty_and_non_finite_inputs() {
        assert!(matches!(
            dense_solve(Vec::new(), Vec::new()),
            Err(SolverError::OperationFailed { .. })
        ));
        assert!(matches!(
            dense_solve(
                vec![vec![Complex64::new(f64::NAN, 0.0)]],
                vec![Complex64::new(1.0, 0.0)]
            ),
            Err(SolverError::OperationFailed { .. })
        ));
    }

    #[test]
    fn targets_nonsymmetric_complex_generalized_eigenvalues() {
        let eigenvalues = [
            Complex64::new(-2.0, 0.5),
            Complex64::new(0.2, 1.5),
            Complex64::new(1.0, -0.3),
            Complex64::new(4.0, 2.0),
        ];
        let mut triplets = Vec::new();
        for (index, &value) in eigenvalues.iter().enumerate() {
            let weight = Complex64::new((index + 2) as f64, 0.0);
            triplets.push((index, index, weight * value));
            if index + 1 < eigenvalues.len() {
                triplets.push((index, index + 1, weight * Complex64::new(0.4, -0.2)));
            }
        }
        let a = CsrMatrix::from_triplets(4, 4, triplets);
        let b = CsrMatrix::from_triplets(
            4,
            4,
            (0..4)
                .map(|i| (i, i, Complex64::new((i + 2) as f64, 0.0)))
                .collect(),
        );
        let result = complex_shift_invert_arnoldi_generalized(
            &a,
            &b,
            &ComplexArnoldiConfig {
                target: Complex64::new(0.5, 0.5),
                num_eigenvalues: 2,
                krylov_dimension: 4,
                tolerance: 1.0e-7,
                max_qr_iterations: 2_000,
                breakdown_tolerance: 1.0e-14,
            },
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert!(
            (result.eigenvalues[0] - eigenvalues[2]).norm() < 1.0e-7,
            "{result:?}"
        );
        assert!(
            (result.eigenvalues[1] - eigenvalues[1]).norm() < 1.0e-7,
            "{result:?}"
        );
    }

    #[test]
    fn resolves_rotation_pair_near_complex_shift() {
        let a = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 1, Complex64::new(-1.0, 0.0)),
                (1, 0, Complex64::new(1.0, 0.0)),
            ],
        );
        let b = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(1.0, 0.0)),
                (1, 1, Complex64::new(1.0, 0.0)),
            ],
        );
        let result = complex_shift_invert_arnoldi_generalized(
            &a,
            &b,
            &ComplexArnoldiConfig {
                target: Complex64::new(0.1, 0.8),
                num_eigenvalues: 1,
                krylov_dimension: 2,
                tolerance: 1.0e-8,
                max_qr_iterations: 2_000,
                breakdown_tolerance: 1.0e-14,
            },
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert!((result.eigenvalues[0] - Complex64::new(0.0, 1.0)).norm() < 1.0e-8);
    }
}
