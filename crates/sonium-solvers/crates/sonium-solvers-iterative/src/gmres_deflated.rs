//! Deflated GMRES solver for systems with clustered eigenvalues
//!
//! Deflated GMRES projects out a subspace W spanning approximations to
//! problematic eigenmodes before iterating, dramatically reducing iteration
//! counts for problems like Helmholtz where eigenvalues cluster near k².
//!
//! Reference: Erlangga & Nabben (2008), Gaul et al. (2013).

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use sonium_solvers_direct::LuError;
use sonium_solvers_direct::{LuFactorization, lu_factorize};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use super::cgs_orth::{FlatBasis, KrylovOpCounts, fused_basis_update_flat};
use super::gmres::{
    GmresConfig, GmresSolution, finalize_gmres_solution, invalid_gmres_input,
    numerical_error_solution,
};
use super::gmres_common::{
    exact_relative_residual, exact_residual_check_due, givens_rotation, solve_upper_triangular,
};

/// Pre-computed deflation data for deflated GMRES
///
/// Given deflation vectors W = [w₁,...,wᵣ] (n×r, r≪n) and operator A:
/// - AW = A·W columns
/// - E = Wᴴ·A·W (r×r dense), factored via LU
///
/// Provides projectors:
/// - Left: P(v) = v - AW·E⁻¹·Wᴴ·v
/// - Coarse correction: W·E⁻¹·Wᴴ·b
/// - Recovery: Q(v) = v - W·E⁻¹·Wᴴ·A·v
#[derive(Debug, Clone)]
pub struct DeflationSubspace<T: ComplexField> {
    w_columns: Vec<Vec<T>>,
    aw_columns: Vec<Vec<T>>,
    e_lu: LuFactorization<T>,
}

impl<T: ComplexField> DeflationSubspace<T> {
    /// Construct deflation subspace from deflation vectors and operator
    ///
    /// Computes AW = A·W and factors E = Wᴴ·A·W via LU.
    ///
    /// # Arguments
    /// * `w_columns` - Deflation vectors (should be orthonormal or near-orthonormal)
    /// * `operator` - The linear operator A
    pub fn new<A: LinearOperator<T>>(
        w_columns: Vec<Vec<T>>,
        operator: &A,
    ) -> Result<Self, LuError> {
        let r = w_columns.len();
        if r == 0 {
            return Ok(Self {
                w_columns,
                aw_columns: Vec::new(),
                e_lu: LuFactorization {
                    lu: sonium_solvers_core::dense::DenseMatrix::from_elem(0, 0, T::zero()),
                    pivots: Vec::new(),
                    n: 0,
                },
            });
        }

        // Compute AW columns
        let aw_columns: Vec<Vec<T>> = w_columns.iter().map(|w| operator.apply(w)).collect();

        // Build E = W^H * A * W (r x r dense)
        let mut e = sonium_solvers_core::dense::DenseMatrix::from_elem(r, r, T::zero());
        for i in 0..r {
            for j in 0..r {
                e[[i, j]] = inner_product(&w_columns[i], &aw_columns[j]);
            }
        }

        let e_lu = lu_factorize(&e)?;

        Ok(Self {
            w_columns,
            aw_columns,
            e_lu,
        })
    }

    /// Construct a deflation subspace through the checked solver boundary.
    ///
    /// Unlike [`Self::new`], this validates the operator shape, every
    /// deflation vector, and every generated `A * W` column before factoring
    /// the coarse matrix. Callback failures and singular coarse systems are
    /// returned as typed solver errors.
    pub fn try_new<A: LinearOperator<T>>(
        w_columns: Vec<Vec<T>>,
        operator: &A,
    ) -> Result<Self, SolverError> {
        if operator.num_rows() != operator.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: operator.num_cols(),
                got: operator.num_rows(),
            });
        }
        let dimension = operator.num_cols();
        for column in &w_columns {
            if column.len() != dimension {
                return Err(SolverError::DimensionMismatch {
                    expected: dimension,
                    got: column.len(),
                });
            }
            if column.iter().any(|value| !value.norm().is_finite()) {
                return Err(SolverError::OperationFailed {
                    operation: "deflation subspace construction",
                    reason: "deflation vector contains NaN or infinity".to_string(),
                });
            }
        }

        let mut aw_columns = Vec::with_capacity(w_columns.len());
        for column in &w_columns {
            let image = catch_unwind(AssertUnwindSafe(|| operator.try_apply(column))).map_err(
                |payload| SolverError::OperationFailed {
                    operation: "deflation subspace construction",
                    reason: panic_reason(payload),
                },
            )??;
            if image.iter().any(|value| !value.norm().is_finite()) {
                return Err(SolverError::OperationFailed {
                    operation: "deflation subspace construction",
                    reason: "operator generated NaN or infinity".to_string(),
                });
            }
            aw_columns.push(image);
        }

        let rank = w_columns.len();
        let mut coarse = sonium_solvers_core::dense::DenseMatrix::from_elem(rank, rank, T::zero());
        for row in 0..rank {
            for column in 0..rank {
                coarse[[row, column]] = inner_product(&w_columns[row], &aw_columns[column]);
            }
        }
        if coarse
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "deflation subspace construction",
                reason: "coarse deflation matrix contains NaN or infinity".to_string(),
            });
        }
        let e_lu = lu_factorize(&coarse).map_err(|error| SolverError::OperationFailed {
            operation: "deflation subspace construction",
            reason: format!("coarse deflation factorization failed: {error}"),
        })?;

        Ok(Self {
            w_columns,
            aw_columns,
            e_lu,
        })
    }

    /// Number of deflation vectors
    pub fn num_vectors(&self) -> usize {
        self.w_columns.len()
    }

    /// Borrow the original deflation vectors used to build this subspace.
    ///
    /// The operator-dependent images and coarse factor are intentionally not
    /// exposed: a portable checkpoint stores only these vectors and rebuilds
    /// the derived state against the operator used for the resumed solve.
    pub fn vectors(&self) -> &[Vec<T>] {
        &self.w_columns
    }

    /// Dimension of the vectors in this subspace, if it is non-empty.
    pub fn dimension(&self) -> Option<usize> {
        self.w_columns.first().map(Vec::len)
    }

    /// Apply left deflation projector: P(v) = v - AW·E⁻¹·Wᴴ·v
    pub fn apply_left_projector(&self, v: &[T]) -> Vec<T> {
        self.try_apply_left_projector(v)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Checked left deflation projector application.
    pub fn try_apply_left_projector(&self, v: &[T]) -> Result<Vec<T>, SolverError> {
        let mut result = vec![T::zero(); v.len()];
        self.try_apply_left_projector_into(v, &mut result)?;
        Ok(result)
    }

    /// Apply left deflation projector into a pre-allocated output buffer.
    ///
    /// Writes P(v) = v - AW·E⁻¹·Wᴴ·v into `out`. `out` must have the same
    /// length as `v`; its previous contents are overwritten.
    pub fn apply_left_projector_into(&self, v: &[T], out: &mut [T]) {
        self.try_apply_left_projector_into(v, out)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    /// Checked left deflation projector application into an existing buffer.
    pub fn try_apply_left_projector_into(&self, v: &[T], out: &mut [T]) -> Result<(), SolverError> {
        self.validate_vector(v, "left deflation projector input")?;
        if out.len() != v.len() {
            return Err(SolverError::DimensionMismatch {
                expected: v.len(),
                got: out.len(),
            });
        }
        let r = self.w_columns.len();
        if r == 0 {
            out.copy_from_slice(v);
            return Ok(());
        }

        // Compute W^H * v (r-dimensional)
        let mut wh_v = vec![T::zero(); r];
        for (i, slot) in wh_v.iter_mut().enumerate() {
            *slot = inner_product(&self.w_columns[i], v);
        }

        // Solve E * y = W^H * v
        let y = self.solve_coarse(&wh_v, "left deflation projector")?;

        // out = v - AW * y
        out.copy_from_slice(v);
        for (i, &yi) in y.iter().enumerate() {
            axpy(-yi, &self.aw_columns[i], out);
        }
        if out.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "left deflation projector",
                reason: "projected vector contains NaN or infinity".to_string(),
            });
        }
        Ok(())
    }

    /// Compute coarse correction: W·E⁻¹·Wᴴ·b
    pub fn coarse_correction(&self, b: &[T]) -> Vec<T> {
        self.try_coarse_correction(b)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Checked coarse correction `W E^-1 W^H b`.
    pub fn try_coarse_correction(&self, b: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_vector(b, "coarse correction input")?;
        let r = self.w_columns.len();
        let n = b.len();
        if r == 0 {
            return Ok(vec![T::zero(); n]);
        }

        // Compute W^H * b
        let mut wh_b = vec![T::zero(); r];
        for (i, slot) in wh_b.iter_mut().enumerate() {
            *slot = inner_product(&self.w_columns[i], b);
        }

        // Solve E * y = W^H * b
        let y = self.solve_coarse(&wh_b, "coarse correction")?;

        // result = W * y
        let mut result = vec![T::zero(); n];
        for (i, &yi) in y.iter().enumerate() {
            axpy(yi, &self.w_columns[i], &mut result);
        }
        if result.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "coarse correction",
                reason: "correction contains NaN or infinity".to_string(),
            });
        }

        Ok(result)
    }

    /// Apply recovery operator: Q(v) = v - W·E⁻¹·Wᴴ·A·v
    pub fn apply_recovery<A: LinearOperator<T>>(&self, v: &[T], operator: &A) -> Vec<T> {
        self.try_apply_recovery(v, operator)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Checked recovery operator application.
    pub fn try_apply_recovery<A: LinearOperator<T>>(
        &self,
        v: &[T],
        operator: &A,
    ) -> Result<Vec<T>, SolverError> {
        self.validate_vector(v, "deflation recovery input")?;
        if operator.num_rows() != v.len() || operator.num_cols() != v.len() {
            return Err(SolverError::DimensionMismatch {
                expected: v.len(),
                got: operator.num_rows().max(operator.num_cols()),
            });
        }
        let r = self.w_columns.len();
        if r == 0 {
            return Ok(v.to_vec());
        }

        // Compute A * v
        let av =
            catch_unwind(AssertUnwindSafe(|| operator.try_apply(v))).map_err(|payload| {
                SolverError::OperationFailed {
                    operation: "deflation recovery",
                    reason: panic_reason(payload),
                }
            })??;
        if av.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "deflation recovery",
                reason: "operator generated NaN or infinity".to_string(),
            });
        }

        // Compute W^H * A * v
        let mut wh_av = vec![T::zero(); r];
        for (i, slot) in wh_av.iter_mut().enumerate() {
            *slot = inner_product(&self.w_columns[i], &av);
        }

        // Solve E * y = W^H * A * v
        let y = self.solve_coarse(&wh_av, "deflation recovery")?;

        // result = v - W * y
        let mut result = v.to_vec();
        for (i, &yi) in y.iter().enumerate() {
            axpy(-yi, &self.w_columns[i], &mut result);
        }
        if result.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "deflation recovery",
                reason: "recovered vector contains NaN or infinity".to_string(),
            });
        }

        Ok(result)
    }

    fn validate_vector(&self, vector: &[T], operation: &'static str) -> Result<(), SolverError> {
        if let Some(dimension) = self.dimension()
            && vector.len() != dimension
        {
            return Err(SolverError::DimensionMismatch {
                expected: dimension,
                got: vector.len(),
            });
        }
        self.validate_storage(vector.len(), operation)?;
        if vector.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "input contains NaN or infinity".to_string(),
            });
        }
        Ok(())
    }

    fn validate_storage(
        &self,
        dimension: usize,
        operation: &'static str,
    ) -> Result<(), SolverError> {
        let rank = self.w_columns.len();
        let storage_is_valid = self.aw_columns.len() == rank
            && self
                .w_columns
                .iter()
                .all(|column| column.len() == dimension)
            && self
                .aw_columns
                .iter()
                .all(|column| column.len() == dimension)
            && self.e_lu.n == rank
            && self.e_lu.pivots.len() == rank
            && self.e_lu.pivots.iter().all(|&pivot| pivot < rank)
            && self.e_lu.lu.nrows() == rank
            && self.e_lu.lu.ncols() == rank;
        if !storage_is_valid {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "deflation subspace storage is structurally invalid".to_string(),
            });
        }
        if self
            .w_columns
            .iter()
            .chain(self.aw_columns.iter())
            .flatten()
            .chain(self.e_lu.lu.as_slice().iter())
            .any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "deflation subspace storage contains NaN or infinity".to_string(),
            });
        }
        Ok(())
    }

    fn solve_coarse(&self, rhs: &[T], operation: &'static str) -> Result<Vec<T>, SolverError> {
        let solution = self
            .e_lu
            .solve(rhs)
            .map_err(|error| SolverError::OperationFailed {
                operation,
                reason: format!("coarse deflation solve failed: {error}"),
            })?;
        if solution.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "coarse deflation solve generated NaN or infinity".to_string(),
            });
        }
        Ok(solution)
    }
}

struct CheckedDeflatedOperator<'a, A> {
    inner: &'a A,
    invalid_output: &'a AtomicBool,
}

impl<T, A> LinearOperator<T> for CheckedDeflatedOperator<'_, A>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_cols()
    }

    fn apply(&self, input: &[T]) -> Vec<T> {
        let mut output = self.inner.apply(input);
        if output.len() != self.num_rows() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output = vec![T::zero(); self.num_rows()];
        }
        output
    }

    fn apply_into(&self, input: &[T], output: &mut [T]) {
        self.inner.apply_into(input, output);
        if output.len() != self.num_rows() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.inner.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.inner.hermitian_capability()
    }

    fn apply_transpose_into(&self, input: &[T], output: &mut [T]) {
        self.inner.apply_transpose_into(input, output);
        if output.len() != self.num_cols() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }

    fn apply_hermitian(&self, input: &[T]) -> Vec<T> {
        let mut output = self.inner.apply_hermitian(input);
        if output.len() != self.num_cols() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
        output
    }
}

struct CheckedDeflatedPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: &'a AtomicBool,
}

impl<T, P> Preconditioner<T> for CheckedDeflatedPreconditioner<'_, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply(&self, input: &[T]) -> Vec<T> {
        let output = self.inner.apply(input);
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            vec![T::zero(); input.len()]
        } else {
            output
        }
    }

    fn apply_into(&self, input: &[T], output: &mut [T]) {
        self.inner.apply_into(input, output);
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

fn validate_deflation_subspace<T, A>(
    operator: &A,
    deflation: &DeflationSubspace<T>,
) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let expected_dimension = operator.num_cols();
    let expected_rows = operator.num_rows();
    if deflation.w_columns.len() != deflation.aw_columns.len() {
        return Err(SolverError::OperationFailed {
            operation: "deflated GMRES",
            reason: "deflation subspace has inconsistent column storage".to_string(),
        });
    }
    for column in &deflation.w_columns {
        if column.len() != expected_dimension {
            return Err(SolverError::DimensionMismatch {
                expected: expected_dimension,
                got: column.len(),
            });
        }
        if column.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "deflated GMRES",
                reason: "deflation vector contains NaN or infinity".to_string(),
            });
        }
    }
    for column in &deflation.aw_columns {
        if column.len() != expected_rows {
            return Err(SolverError::DimensionMismatch {
                expected: expected_rows,
                got: column.len(),
            });
        }
        if column.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "deflated GMRES",
                reason: "deflated operator image contains NaN or infinity".to_string(),
            });
        }
    }
    if deflation.e_lu.n != deflation.w_columns.len()
        || deflation.e_lu.pivots.len() != deflation.e_lu.n
        || deflation.e_lu.lu.as_slice().len() != deflation.e_lu.n * deflation.e_lu.n
        || deflation
            .e_lu
            .lu
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "deflated GMRES",
            reason: "deflation LU factorization is invalid or non-finite".to_string(),
        });
    }
    Ok(())
}

fn compatibility_error_solution<T: ComplexField>(b: &[T], x0: Option<&[T]>) -> GmresSolution<T> {
    numerical_error_solution(match x0 {
        Some(guess) if guess.len() == b.len() => guess.to_vec(),
        _ => vec![T::zero(); b.len()],
    })
}

/// Solve Ax = b using deflated GMRES (no preconditioner)
///
/// Projects out the deflation subspace before each Arnoldi step.
/// After convergence, applies recovery operator and adds coarse correction.
pub fn gmres_deflated<T, A>(
    operator: &A,
    deflation: &DeflationSubspace<T>,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_gmres_deflated(operator, deflation, b, x0, config) {
        Ok(result) => result,
        Err(_) => compatibility_error_solution(b, x0),
    }
}

/// Checked deflated GMRES entry point using the identity preconditioner.
pub fn try_gmres_deflated<T, A>(
    operator: &A,
    deflation: &DeflationSubspace<T>,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = sonium_solvers_core::traits::IdentityPreconditioner;
    try_gmres_deflated_preconditioned(operator, &identity, deflation, b, x0, config)
}

/// Solve Ax = b using deflated preconditioned GMRES
///
/// Combines left preconditioning M⁻¹ with deflation projection P:
/// At each Arnoldi step: w = M⁻¹ · P(A · vⱼ)
/// After convergence to x̂: x = Q(x̂) + coarse_correction(b)
///
/// Reference: Erlangga & Nabben (2008)
pub fn gmres_deflated_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    deflation: &DeflationSubspace<T>,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_gmres_deflated_preconditioned(operator, precond, deflation, b, x0, config) {
        Ok(result) => result,
        Err(_) => compatibility_error_solution(b, x0),
    }
}

/// Checked deflated GMRES entry point for a left preconditioner.
pub fn try_gmres_deflated_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    deflation: &DeflationSubspace<T>,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if config.max_iterations == 0
        || config.restart == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "deflated GMRES",
            reason: "invalid GMRES configuration".to_string(),
        });
    }
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    if b.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "deflated GMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(guess) = x0
        && guess.len() != operator.num_cols()
    {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: guess.len(),
        });
    }
    if let Some(guess) = x0
        && guess.iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "deflated GMRES",
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }
    validate_deflation_subspace(operator, deflation)?;

    let invalid_operator_output = AtomicBool::new(false);
    let invalid_preconditioner_output = AtomicBool::new(false);
    let checked_operator = CheckedDeflatedOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    let checked_precond = CheckedDeflatedPreconditioner {
        inner: precond,
        invalid_output: &invalid_preconditioner_output,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let result = gmres_deflated_preconditioned_core(
            &checked_operator,
            &checked_precond,
            deflation,
            b,
            x0,
            config,
        )?;
        if invalid_preconditioner_output.load(Ordering::Relaxed) {
            return Err(SolverError::OperationFailed {
                operation: "deflated GMRES preconditioner",
                reason: "preconditioner generated NaN, infinity, or an invalid output".to_string(),
            });
        }
        let result = finalize_gmres_solution(&checked_operator, b, config.tolerance, result);
        if invalid_operator_output.load(Ordering::Relaxed) {
            return Err(SolverError::OperationFailed {
                operation: "deflated GMRES operator",
                reason: "operator generated NaN, infinity, or an invalid output".to_string(),
            });
        }
        if result.status == SolverStatus::NumericalError {
            return Err(SolverError::OperationFailed {
                operation: "deflated GMRES",
                reason: "operator generated NaN or infinity".to_string(),
            });
        }
        SolverResultContract::validate_contract(&result)?;
        Ok(result)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "deflated GMRES",
        reason: panic_reason(payload),
    })??;

    Ok(result)
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "deflated GMRES callback panicked".to_string()
    }
}

fn gmres_deflated_preconditioned_core<T, A, P>(
    operator: &A,
    precond: &P,
    deflation: &DeflationSubspace<T>,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if let Some(invalid) = invalid_gmres_input(
        b,
        x0,
        config.max_iterations,
        config.restart,
        config.tolerance,
    ) {
        return Ok(invalid);
    }

    // If no deflation vectors, fall back to standard preconditioned GMRES
    if deflation.num_vectors() == 0 {
        return Ok(super::gmres::gmres_preconditioned_with_guess(
            operator, precond, b, x0, config,
        ));
    }

    let n = b.len();
    if deflation
        .dimension()
        .is_some_and(|dimension| dimension != n)
    {
        return Ok(numerical_error_solution(match x0 {
            Some(guess) => guess.to_vec(),
            None => vec![T::zero(); n],
        }));
    }
    let m = config.restart;
    let mut op_counts = KrylovOpCounts::default();
    let deflation_vectors = deflation.num_vectors();

    // Compute coarse correction: x_c = W·E⁻¹·Wᴴ·b
    let x_c = deflation.try_coarse_correction(b)?;
    op_counts.dot_reductions += deflation_vectors;
    op_counts.synchronization_points += deflation_vectors;
    op_counts.axpy_passes += deflation_vectors;

    // Initialize solution from initial guess or zero
    let mut x_hat = match x0 {
        Some(guess) => guess.to_vec(),
        None => vec![T::zero(); n],
    };

    let mut av_buf = vec![T::zero(); n];
    let true_b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if true_b_norm < tol_threshold {
        let x = sonium_solvers_core::vec_ops::vec_add(
            &deflation.try_apply_recovery(&x_hat, operator)?,
            &x_c,
        );
        op_counts.spmv_count += 1;
        op_counts.dot_reductions += deflation_vectors;
        op_counts.synchronization_points += deflation_vectors;
        op_counts.axpy_passes += deflation_vectors + 1;

        operator.apply_into(&x, &mut av_buf);
        op_counts.spmv_count += 1;
        for value in &mut av_buf {
            *value = T::zero() - *value;
        }
        let actual_residual = vector_norm(&av_buf);
        op_counts.norm_count += 1;
        let converged = actual_residual < tol_threshold;
        return Ok(GmresSolution {
            x,
            iterations: 0,
            restarts: 0,
            residual: actual_residual,
            estimated_residual: actual_residual,
            backward_error: actual_residual,
            attainable_accuracy: if converged {
                AccuracyStatus::Attained
            } else {
                AccuracyStatus::NotAttained
            },
            converged,
            status: if converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Breakdown
            },
            op_counts,
        });
    }

    // Compute preconditioned deflated RHS norm for relative tolerance
    let pb = precond.apply(&deflation.try_apply_left_projector(b)?);
    op_counts.dot_reductions += deflation_vectors;
    op_counts.synchronization_points += deflation_vectors;
    op_counts.axpy_passes += deflation_vectors + 1;
    op_counts.preconditioner_count += 1;
    let b_norm = vector_norm(&pb);
    op_counts.norm_count += 1;
    if b_norm < tol_threshold {
        // The coarse correction is only a solution if it satisfies the
        // original, unpreconditioned Ax=b system. A singular preconditioner
        // can otherwise annihilate P(b) and manufacture false convergence.
        let x = sonium_solvers_core::vec_ops::vec_add(
            &deflation.try_apply_recovery(&x_hat, operator)?,
            &x_c,
        );
        op_counts.spmv_count += 1;
        op_counts.dot_reductions += deflation_vectors;
        op_counts.synchronization_points += deflation_vectors;
        op_counts.axpy_passes += deflation_vectors + 1;
        let actual_residual =
            exact_relative_residual(operator, &x, b, &mut av_buf, true_b_norm, &mut op_counts);
        let converged = actual_residual < config.tolerance;
        return Ok(GmresSolution {
            x,
            iterations: 0,
            restarts: 0,
            residual: actual_residual,
            estimated_residual: actual_residual,
            backward_error: actual_residual,
            attainable_accuracy: if converged {
                AccuracyStatus::Attained
            } else {
                AccuracyStatus::NotAttained
            },
            converged,
            status: if converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Breakdown
            },
            op_counts,
        });
    }

    let mut total_iterations = 0;
    let mut restarts = 0;
    let mut last_estimated_residual = T::Real::infinity();

    // Pre-allocate Krylov basis and work buffers (reused across restarts)
    let mut v = FlatBasis::new(m + 1, n);
    let mut pav_buf = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];

    // Outer iteration (restarts)
    for _outer in 0..config.max_iterations {
        // Compute deflated preconditioned residual: M⁻¹ · P · (b - A · x̂) into w
        operator.apply_into(&x_hat, &mut av_buf);
        op_counts.spmv_count += 1;
        for k in 0..n {
            av_buf[k] = b[k] - av_buf[k];
        }
        op_counts.axpy_passes += 1;
        deflation.try_apply_left_projector_into(&av_buf, &mut pav_buf)?;
        op_counts.dot_reductions += deflation_vectors;
        op_counts.synchronization_points += deflation_vectors;
        op_counts.axpy_passes += deflation_vectors + 1;
        precond.apply_into(&pav_buf, &mut w);
        op_counts.preconditioner_count += 1;
        let beta = vector_norm(&w);
        op_counts.norm_count += 1;

        let rel_residual = beta / b_norm;
        last_estimated_residual = rel_residual;
        if rel_residual < config.tolerance {
            // Projected/preconditioned convergence is only an estimate. Check
            // the recovered solution against the public Ax=b tolerance.
            let x = sonium_solvers_core::vec_ops::vec_add(
                &deflation.try_apply_recovery(&x_hat, operator)?,
                &x_c,
            );
            op_counts.spmv_count += 1;
            op_counts.dot_reductions += deflation_vectors;
            op_counts.synchronization_points += deflation_vectors;
            op_counts.axpy_passes += deflation_vectors + 1;
            let actual_residual =
                exact_relative_residual(operator, &x, b, &mut av_buf, true_b_norm, &mut op_counts);
            if actual_residual < config.tolerance {
                return Ok(GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: actual_residual,
                    estimated_residual: rel_residual,
                    backward_error: actual_residual,
                    attainable_accuracy: AccuracyStatus::Attained,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                });
            }
            if beta < tol_threshold {
                return Ok(GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: actual_residual,
                    estimated_residual: rel_residual,
                    backward_error: actual_residual,
                    attainable_accuracy: AccuracyStatus::NotAttained,
                    converged: false,
                    status: SolverStatus::Breakdown,
                    op_counts,
                });
            }
        }

        // Initialize v[0] = w / beta
        let inv_beta = T::from_real(T::Real::one() / beta);
        {
            let v0 = v.vector_mut(0);
            for k in 0..n {
                v0[k] = w[k] * inv_beta;
            }
        }
        op_counts.axpy_passes += 1;

        // Upper Hessenberg matrix
        let mut h: sonium_solvers_core::dense::DenseMatrix<T> =
            sonium_solvers_core::dense::DenseMatrix::from_elem(m + 1, m, T::zero());

        // Givens rotation coefficients
        let mut cs: Vec<T> = Vec::with_capacity(m);
        let mut sn: Vec<T> = Vec::with_capacity(m);

        // Right-hand side of least squares problem
        let mut g: Vec<T> = vec![T::zero(); m + 1];
        g[0] = T::from_real(beta);

        let mut arnoldi_breakdown = false;
        let dgks_threshold = T::Real::from_f64(0.707).unwrap();

        // Inner iteration (Arnoldi process with deflation)
        for j in 0..m {
            total_iterations += 1;

            // Deflated preconditioned matvec: w = M⁻¹ · P · A · vⱼ (zero-alloc)
            operator.apply_into(v.vector(j), &mut av_buf);
            op_counts.spmv_count += 1;
            deflation.try_apply_left_projector_into(&av_buf, &mut pav_buf)?;
            op_counts.dot_reductions += deflation_vectors;
            op_counts.synchronization_points += deflation_vectors;
            op_counts.axpy_passes += deflation_vectors + 1;
            precond.apply_into(&pav_buf, &mut w);
            op_counts.preconditioner_count += 1;

            // Classical Gram-Schmidt with DGKS reorthogonalization
            let w_norm_before = vector_norm(&w);
            op_counts.norm_count += 1;
            for i in 0..=j {
                h[[i, j]] = inner_product(v.vector(i), &w);
            }
            op_counts.dot_reductions += j + 1;
            op_counts.synchronization_points += 1;
            for i in 0..=j {
                axpy(-h[[i, j]], v.vector(i), &mut w);
            }
            op_counts.axpy_passes += j + 1;
            op_counts.basis_bytes_touched += ((j + 1) * n * std::mem::size_of::<T>() * 2) as u64;
            let mut w_norm = vector_norm(&w);
            op_counts.norm_count += 1;
            if w_norm < dgks_threshold * w_norm_before {
                for i in 0..=j {
                    let s = inner_product(v.vector(i), &w);
                    h[[i, j]] += s;
                    axpy(-s, v.vector(i), &mut w);
                }
                op_counts.dot_reductions += j + 1;
                op_counts.synchronization_points += 1;
                op_counts.axpy_passes += j + 1;
                op_counts.basis_bytes_touched +=
                    ((j + 1) * n * std::mem::size_of::<T>() * 2) as u64;
                w_norm = vector_norm(&w);
                op_counts.norm_count += 1;
            }
            h[[j + 1, j]] = T::from_real(w_norm);

            // Check for breakdown
            let breakdown_tol = T::Real::from_f64(1e-20).unwrap();
            if w_norm < breakdown_tol {
                arnoldi_breakdown = true;
            } else {
                let inv_norm = T::from_real(T::Real::one() / w_norm);
                let next_v = v.vector_mut(j + 1);
                for k in 0..n {
                    next_v[k] = w[k] * inv_norm;
                }
                op_counts.axpy_passes += 1;
            }

            // Apply previous Givens rotations to new column of H
            for i in 0..j {
                let temp = cs[i].conj() * h[[i, j]] + sn[i].conj() * h[[i + 1, j]];
                h[[i + 1, j]] = T::zero() - sn[i] * h[[i, j]] + cs[i] * h[[i + 1, j]];
                h[[i, j]] = temp;
            }

            // Compute new Givens rotation
            let (c, s) = givens_rotation(h[[j, j]], h[[j + 1, j]]);
            cs.push(c);
            sn.push(s);

            // Apply Givens rotation to H and g
            h[[j, j]] = c.conj() * h[[j, j]] + s.conj() * h[[j + 1, j]];
            h[[j + 1, j]] = T::zero();

            let temp = c.conj() * g[j] + s.conj() * g[j + 1];
            g[j + 1] = T::zero() - s * g[j] + c * g[j + 1];
            g[j] = temp;

            // Check convergence
            let abs_residual = g[j + 1].norm();
            let rel_residual = abs_residual / b_norm;
            last_estimated_residual = rel_residual;
            let abs_tol = T::Real::from_f64(1e-20).unwrap();

            if config.print_interval > 0 && total_iterations % config.print_interval == 0 {
                log::info!(
                    "Deflated GMRES iteration {} (restart {}): relative residual = {:.6e}",
                    total_iterations,
                    restarts,
                    rel_residual.to_f64().unwrap_or(0.0)
                );
            }

            let estimated_converged =
                rel_residual < config.tolerance || (!arnoldi_breakdown && abs_residual < abs_tol);
            if estimated_converged || arnoldi_breakdown {
                // Solve upper triangular system Hy = g
                let y = solve_upper_triangular(&h, &g, j + 1);

                // Validate a temporary candidate. If the preconditioned
                // residual estimate is optimistic, keep expanding this
                // Arnoldi basis instead of committing a one-step update and
                // repeatedly restarting from the same accuracy floor.
                let mut candidate_hat = x_hat.clone();
                fused_basis_update_flat(&v, &y, &mut candidate_hat);
                op_counts.add_basis_update::<T>(y.len(), n);

                // Apply recovery and coarse correction: x = Q(x̂) + x_c
                let x = sonium_solvers_core::vec_ops::vec_add(
                    &deflation.try_apply_recovery(&candidate_hat, operator)?,
                    &x_c,
                );
                op_counts.spmv_count += 1;
                op_counts.dot_reductions += deflation_vectors;
                op_counts.synchronization_points += deflation_vectors;
                op_counts.axpy_passes += deflation_vectors + 1;

                let actual_residual = exact_relative_residual(
                    operator,
                    &x,
                    b,
                    &mut av_buf,
                    true_b_norm,
                    &mut op_counts,
                );
                if actual_residual < config.tolerance {
                    return Ok(GmresSolution {
                        x,
                        iterations: total_iterations,
                        restarts,
                        residual: actual_residual,
                        estimated_residual: rel_residual,
                        backward_error: actual_residual,
                        attainable_accuracy: AccuracyStatus::Attained,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    });
                }
                if arnoldi_breakdown {
                    return Ok(GmresSolution {
                        x,
                        iterations: total_iterations,
                        restarts,
                        residual: actual_residual,
                        estimated_residual: rel_residual,
                        backward_error: actual_residual,
                        attainable_accuracy: AccuracyStatus::NotAttained,
                        converged: false,
                        status: SolverStatus::Breakdown,
                        op_counts,
                    });
                }

                // The estimate was optimistic but Arnoldi can still make
                // progress. Do not truncate the current Krylov cycle.
                continue;
            }

            if exact_residual_check_due(config.exact_residual_check_interval, total_iterations) {
                let y = solve_upper_triangular(&h, &g, j + 1);
                let mut candidate_hat = x_hat.clone();
                fused_basis_update_flat(&v, &y, &mut candidate_hat);
                op_counts.add_basis_update::<T>(y.len(), n);
                let candidate = sonium_solvers_core::vec_ops::vec_add(
                    &deflation.try_apply_recovery(&candidate_hat, operator)?,
                    &x_c,
                );
                op_counts.spmv_count += 1;
                op_counts.dot_reductions += deflation_vectors;
                op_counts.synchronization_points += deflation_vectors;
                op_counts.axpy_passes += deflation_vectors + 1;
                let actual_residual = exact_relative_residual(
                    operator,
                    &candidate,
                    b,
                    &mut av_buf,
                    true_b_norm,
                    &mut op_counts,
                );
                if actual_residual < config.tolerance {
                    return Ok(GmresSolution {
                        x: candidate,
                        iterations: total_iterations,
                        restarts,
                        residual: actual_residual,
                        estimated_residual: rel_residual,
                        backward_error: actual_residual,
                        attainable_accuracy: AccuracyStatus::Attained,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    });
                }
            }
        }

        // Maximum inner iterations reached — update x̂ and restart
        let y = solve_upper_triangular(&h, &g, m);
        fused_basis_update_flat(&v, &y, &mut x_hat);
        op_counts.add_basis_update::<T>(y.len(), n);

        let x = sonium_solvers_core::vec_ops::vec_add(
            &deflation.try_apply_recovery(&x_hat, operator)?,
            &x_c,
        );
        op_counts.spmv_count += 1;
        op_counts.dot_reductions += deflation_vectors;
        op_counts.synchronization_points += deflation_vectors;
        op_counts.axpy_passes += deflation_vectors + 1;
        let actual_residual =
            exact_relative_residual(operator, &x, b, &mut av_buf, true_b_norm, &mut op_counts);
        if actual_residual < config.tolerance {
            return Ok(GmresSolution {
                x,
                iterations: total_iterations,
                restarts,
                residual: actual_residual,
                estimated_residual: last_estimated_residual,
                backward_error: actual_residual,
                attainable_accuracy: AccuracyStatus::Attained,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            });
        }

        restarts += 1;
    }

    // Did not converge — still apply recovery for best-effort solution
    let x = sonium_solvers_core::vec_ops::vec_add(
        &deflation.try_apply_recovery(&x_hat, operator)?,
        &x_c,
    );
    op_counts.spmv_count += 1;
    op_counts.dot_reductions += deflation_vectors;
    op_counts.synchronization_points += deflation_vectors;
    op_counts.axpy_passes += deflation_vectors + 1;

    let rel_residual =
        exact_relative_residual(operator, &x, b, &mut av_buf, true_b_norm, &mut op_counts);

    Ok(GmresSolution {
        x,
        iterations: total_iterations,
        restarts,
        residual: rel_residual,
        estimated_residual: last_estimated_residual,
        backward_error: rel_residual,
        attainable_accuracy: AccuracyStatus::NotAttained,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;
    use sonium_solvers_core::traits::{LinearOperator, Preconditioner, SolverError, SolverStatus};
    use sonium_solvers_sparse::CsrMatrix;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Build a diagonal operator with clustered eigenvalues near `cluster_center`
    /// plus a few well-separated eigenvalues
    fn build_clustered_diagonal(n: usize, cluster_center: f64) -> CsrMatrix<Complex64> {
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            if i < n - 3 {
                // Clustered eigenvalues near cluster_center
                let offset = 0.01 * (i as f64 - (n as f64 - 3.0) / 2.0);
                dense[[i, i]] = Complex64::new(cluster_center + offset, 0.0);
            } else {
                // Well-separated eigenvalues
                dense[[i, i]] = Complex64::new((i + 1) as f64 * 10.0, 0.0);
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    #[test]
    fn test_deflation_subspace_construction() {
        // Simple 3x3 diagonal system with known eigenvectors
        let dense = sonium_solvers_core::array![
            [Complex64::new(1.0, 0.0), Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::new(2.0, 0.0), Complex64::ZERO],
            [Complex64::ZERO, Complex64::ZERO, Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);

        // Deflate first eigenvector
        let w1 =
            sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::ZERO, Complex64::ZERO];
        let deflation = DeflationSubspace::new(vec![w1], &a).unwrap();

        assert_eq!(deflation.num_vectors(), 1);

        // P(e1) should be zero (eigenvector is fully deflated)
        let e1 =
            sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::ZERO, Complex64::ZERO];
        let pe1 = deflation.apply_left_projector(&e1);
        let pe1_norm: f64 = pe1.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(pe1_norm < 1e-12, "Deflated eigenvector should be zero");

        // P(e2) should still be nonzero
        let e2 =
            sonium_solvers_core::array![Complex64::ZERO, Complex64::new(1.0, 0.0), Complex64::ZERO];
        let pe2 = deflation.apply_left_projector(&e2);
        let pe2_norm: f64 = pe2.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(pe2_norm > 0.5, "Non-deflated vector should remain");
    }

    #[test]
    fn checked_deflation_subspace_construction_rejects_invalid_generated_data() {
        struct NonFiniteOperator;

        impl LinearOperator<f64> for NonFiniteOperator {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply(&self, _input: &[f64]) -> Vec<f64> {
                vec![f64::NAN, 0.0]
            }

            fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
                output.copy_from_slice(&[f64::NAN, 0.0]);
            }

            fn apply_transpose_into(&self, _input: &[f64], output: &mut [f64]) {
                output.copy_from_slice(&[f64::NAN, 0.0]);
            }
        }

        let error = DeflationSubspace::try_new(vec![vec![1.0, 0.0]], &NonFiniteOperator)
            .expect_err("checked construction must reject non-finite A * W data");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let identity = CsrMatrix::<f64>::identity(2);
        let error = DeflationSubspace::try_new(vec![vec![1.0]], &identity)
            .expect_err("checked construction must reject wrong-sized deflation vectors");
        assert!(matches!(error, SolverError::DimensionMismatch { .. }));

        let error = DeflationSubspace::try_new(vec![vec![1.0, 0.0], vec![1.0, 0.0]], &identity)
            .expect_err("checked construction must reject a singular coarse system");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn test_deflated_gmres_fewer_iterations() {
        let n = 20;
        let cluster = 5.0;
        let a = build_clustered_diagonal(n, cluster);

        // Build RHS
        let b = (0..n)
            .map(|i| Complex64::new((i + 1) as f64, 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 200,
            restart: 30,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        // Standard GMRES
        let sol_standard = super::super::gmres::gmres(&a, &b, &config);

        // Deflated GMRES with exact eigenvectors of clustered eigenvalues
        let mut w_cols = Vec::new();
        for i in 0..(n - 3) {
            let mut w = vec![Complex64::ZERO; n];
            w[i] = Complex64::new(1.0, 0.0);
            w_cols.push(w);
        }
        let deflation = DeflationSubspace::new(w_cols, &a).unwrap();
        let sol_deflated = gmres_deflated(&a, &deflation, &b, None, &config);

        assert!(sol_standard.converged, "Standard GMRES should converge");
        assert!(sol_deflated.converged, "Deflated GMRES should converge");

        // Deflated version should need fewer or equal iterations
        assert!(
            sol_deflated.iterations <= sol_standard.iterations,
            "Deflated ({}) should use <= iterations than standard ({})",
            sol_deflated.iterations,
            sol_standard.iterations
        );

        // Verify solution correctness
        let ax = a.matvec(&sol_deflated.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Deflated solution should satisfy Ax = b");
    }

    #[test]
    fn test_deflated_gmres_with_preconditioner() {
        let dense = sonium_solvers_core::array![
            [
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::ZERO,
            ],
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            [
                Complex64::ZERO,
                Complex64::new(1.0, 0.0),
                Complex64::new(5.0, 0.0),
            ],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0)
        ];

        // Use Jacobi preconditioner
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        // Single deflation vector (approximate eigenvector)
        let w1 = sonium_solvers_core::array![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.7, 0.0),
            Complex64::new(0.5, 0.0)
        ];
        let w1_norm = vector_norm(&w1);
        let w1_normalized = w1
            .iter()
            .map(|&v| v * Complex64::new(1.0 / w1_norm, 0.0))
            .collect::<Vec<_>>();

        let deflation = DeflationSubspace::new(vec![w1_normalized], &a).unwrap();

        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_deflated_preconditioned(&a, &precond, &deflation, &b, None, &config);
        assert!(
            sol.converged,
            "Deflated preconditioned GMRES should converge"
        );

        // Verify solution
        let ax = a.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution should satisfy Ax = b");
    }

    #[test]
    fn test_deflated_gmres_rejects_false_convergence_on_breakdown() {
        struct BreakdownAfterResidual {
            applications: AtomicUsize,
        }

        impl Preconditioner<Complex64> for BreakdownAfterResidual {
            fn apply_into(&self, r: &[Complex64], y: &mut [Complex64]) {
                if self.applications.fetch_add(1, Ordering::Relaxed) < 2 {
                    y.copy_from_slice(r);
                } else {
                    y.fill(Complex64::ZERO);
                }
            }
        }

        let a: CsrMatrix<Complex64> = CsrMatrix::identity(3);
        let w =
            sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::ZERO, Complex64::ZERO];
        let deflation = DeflationSubspace::new(vec![w], &a).unwrap();
        let b = sonium_solvers_core::array![
            Complex64::ZERO,
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0)
        ];
        let config = GmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_deflated_preconditioned(
            &a,
            &BreakdownAfterResidual {
                applications: AtomicUsize::new(0),
            },
            &deflation,
            &b,
            None,
            &config,
        );

        assert!(!solution.converged);
        assert_eq!(solution.status, SolverStatus::Breakdown);
        assert!(solution.residual > config.tolerance);
    }

    #[test]
    fn test_deflated_gmres_restart_accepts_requested_true_residual() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = r[1];
                y[2] = 100.0 * r[2];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(3);
        let deflation = DeflationSubspace::new(vec![vec![1.0, 0.0, 0.0]], &a).unwrap();
        let b = [0.0, 1.0, 1.0e-6];
        let config = GmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1.0e-3,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
            print_interval: 0,
        };

        let solution = gmres_deflated_preconditioned(
            &a,
            &AnisotropicPreconditioner,
            &deflation,
            &b,
            None,
            &config,
        );

        assert!(
            solution.converged,
            "true residual met tolerance at restart: residual={:.3e}",
            solution.residual
        );
        assert_eq!(solution.status, SolverStatus::Converged);
        assert!(solution.residual < config.tolerance);
    }

    #[test]
    fn test_deflated_gmres_max_iterations_preserves_projected_estimate() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = r[1];
                y[2] = 100.0 * r[2];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(3);
        let deflation = DeflationSubspace::new(vec![vec![1.0, 0.0, 0.0]], &a).unwrap();
        let b = [0.0, 1.0, 1.0e-6];
        let config = GmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1.0e-20,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
            print_interval: 0,
        };

        let solution = gmres_deflated_preconditioned(
            &a,
            &AnisotropicPreconditioner,
            &deflation,
            &b,
            None,
            &config,
        );

        assert_eq!(solution.status, SolverStatus::MaxIterationsReached);
        assert!(!solution.converged);
        assert!(solution.residual.is_finite());
        assert!(solution.estimated_residual.is_finite());
        assert_ne!(
            solution.residual.to_bits(),
            solution.estimated_residual.to_bits(),
            "max-iteration result must preserve the projected estimate"
        );
    }

    #[test]
    fn test_deflated_gmres_optimistic_estimate_keeps_expanding_basis() {
        struct OptimisticPreconditioner;

        impl Preconditioner<f64> for OptimisticPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = r[1];
                y[2] = 0.01 * r[2];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(3);
        let deflation = DeflationSubspace::new(vec![vec![1.0, 0.0, 0.0]], &a).unwrap();
        let b = [0.0, 1.0, 1.0];
        let config = GmresConfig {
            max_iterations: 1,
            restart: 2,
            tolerance: 5.0e-2,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
            print_interval: 0,
        };

        let solution = gmres_deflated_preconditioned(
            &a,
            &OptimisticPreconditioner,
            &deflation,
            &b,
            None,
            &config,
        );

        assert!(
            solution.converged,
            "solver truncated the Krylov cycle at an optimistic estimate: residual={:.3e}",
            solution.residual
        );
        assert_eq!(solution.iterations, 2);
        assert!(solution.residual < config.tolerance);
    }

    #[test]
    fn test_deflated_gmres_honors_exact_residual_check_interval() {
        let a: CsrMatrix<f64> = CsrMatrix::identity(3);
        let deflation = DeflationSubspace::new(vec![vec![1.0, 0.0, 0.0]], &a).unwrap();
        let b = [0.0, 1.0, -2.0];
        let config = GmresConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1e-12,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
            print_interval: 0,
        };

        let solution = gmres_deflated(&a, &deflation, &b, None, &config);

        assert!(solution.converged);
        assert!(
            solution.op_counts.exact_residual_checks > 0,
            "configured exact residual checks were silently ignored"
        );
    }

    #[test]
    fn test_deflated_gmres_reports_krylov_op_counts() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];
        let w1 = sonium_solvers_core::array![
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2
        ];
        let deflation = DeflationSubspace::new(vec![w1.to_vec()], &a).unwrap();
        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_deflated(&a, &deflation, &b, None, &config);

        assert!(sol.converged);
        assert!(sol.op_counts.spmv_count >= sol.iterations);
        assert!(sol.op_counts.preconditioner_count > 0);
        assert!(sol.op_counts.dot_reductions > 0);
        assert!(sol.op_counts.norm_count > 0);
        assert!(sol.op_counts.axpy_passes > 0);
        assert!(sol.residual.is_finite());
        assert!(sol.estimated_residual.is_finite());
        assert!(sol.backward_error.is_finite());
        assert_eq!(sol.attainable_accuracy, AccuracyStatus::Attained);
    }

    #[test]
    fn test_deflated_gmres_zero_vectors_fallback() {
        // r=0 deflation vectors should fall back to standard GMRES
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let deflation = DeflationSubspace::new(Vec::new(), &a).unwrap();
        assert_eq!(deflation.num_vectors(), 0);

        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_deflated(&a, &deflation, &b, None, &config);
        assert!(sol.converged, "Zero-vector deflated GMRES should converge");

        // Should give same result as standard GMRES
        let sol_standard = super::super::gmres::gmres(&a, &b, &config);
        let error: f64 = sonium_solvers_core::vec_ops::vec_sub(&sol.x, &sol_standard.x)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-8,
            "Zero-deflation should match standard GMRES solution"
        );
    }

    #[test]
    fn test_coarse_correction_exact_for_deflation_space() {
        // If b lies entirely in range(A·W), the coarse correction should recover it
        let dense = sonium_solvers_core::array![
            [Complex64::new(2.0, 0.0), Complex64::ZERO],
            [Complex64::ZERO, Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);

        // Deflate both eigenvectors — coarse correction solves exactly
        let w1 = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::ZERO];
        let w2 = sonium_solvers_core::array![Complex64::ZERO, Complex64::new(1.0, 0.0)];
        let deflation = DeflationSubspace::new(vec![w1, w2], &a).unwrap();

        let b = sonium_solvers_core::array![Complex64::new(4.0, 0.0), Complex64::new(9.0, 0.0)];
        let x_c = deflation.coarse_correction(&b);

        // x_c should be the exact solution A⁻¹b = [2, 3]
        let ax_c = a.matvec(&x_c);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax_c, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-10, "Coarse correction should be exact solution");
    }

    #[test]
    fn test_deflated_gmres_f64() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0],];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];

        // Deflate one approximate eigenvector
        let w1 = sonium_solvers_core::array![
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2
        ];
        let w1_norm = vector_norm(&w1);
        let w1_normalized = w1.iter().map(|&v| v / w1_norm).collect::<Vec<_>>();
        let deflation = DeflationSubspace::new(vec![w1_normalized], &a).unwrap();

        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_deflated(&a, &deflation, &b, None, &config);
        assert!(sol.converged);

        let ax = a.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert_relative_eq!(error, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn deflated_gmres_rejects_incompatible_subspace_dimension() {
        let subspace_operator: CsrMatrix<f64> = CsrMatrix::identity(1);
        let deflation = DeflationSubspace::new(vec![vec![1.0]], &subspace_operator).unwrap();
        let operator: CsrMatrix<f64> = CsrMatrix::identity(2);

        let solution = gmres_deflated(
            &operator,
            &deflation,
            &[1.0, 2.0],
            None,
            &GmresConfig::default(),
        );

        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert_eq!(solution.x.len(), 2);
    }

    #[test]
    fn try_deflated_gmres_rejects_non_finite_preconditioner_output() {
        struct NonFinitePreconditioner;

        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply(&self, input: &[f64]) -> Vec<f64> {
                vec![f64::NAN; input.len()]
            }

            fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
                output.fill(f64::NAN);
            }
        }

        let operator = CsrMatrix::<f64>::identity(2);
        let deflation = DeflationSubspace::new(vec![vec![1.0, 0.0]], &operator).unwrap();
        let config = GmresConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-8,
            ..GmresConfig::default()
        };
        let error = try_gmres_deflated_preconditioned(
            &operator,
            &NonFinitePreconditioner,
            &deflation,
            &[1.0, 2.0],
            None,
            &config,
        )
        .expect_err("checked deflated GMRES must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let legacy = gmres_deflated_preconditioned(
            &operator,
            &NonFinitePreconditioner,
            &deflation,
            &[1.0, 2.0],
            None,
            &config,
        );
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn try_deflated_gmres_contains_operator_panics() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply_into(&self, _input: &[f64], _output: &mut [f64]) {
                panic!("operator callback failed");
            }

            fn apply_transpose_into(&self, _input: &[f64], _output: &mut [f64]) {
                panic!("operator transpose callback failed");
            }
        }

        let subspace_operator = CsrMatrix::<f64>::identity(2);
        let deflation = DeflationSubspace::new(vec![vec![1.0, 0.0]], &subspace_operator).unwrap();
        let operator = PanicOperator;
        let config = GmresConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-8,
            ..GmresConfig::default()
        };

        let error = try_gmres_deflated(&operator, &deflation, &[1.0, 2.0], None, &config)
            .expect_err("checked deflated GMRES must contain operator panics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let legacy = gmres_deflated(&operator, &deflation, &[1.0, 2.0], None, &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn checked_deflation_paths_reject_invalid_coarse_factor_without_panicking() {
        let operator = CsrMatrix::<f64>::identity(2);
        let deflation = DeflationSubspace {
            w_columns: vec![vec![1.0, 0.0]],
            aw_columns: vec![vec![1.0, 0.0]],
            e_lu: LuFactorization {
                lu: sonium_solvers_core::dense::DenseMatrix::from_elem(1, 1, 0.0),
                pivots: vec![0],
                n: 1,
            },
        };
        let config = GmresConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-8,
            ..GmresConfig::default()
        };

        let error = deflation
            .try_coarse_correction(&[1.0, 2.0])
            .expect_err("checked coarse correction must return the LU failure");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        let error = deflation
            .try_apply_left_projector(&[1.0, 2.0])
            .expect_err("checked projector must return the LU failure");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        let error = deflation
            .try_apply_recovery(&[1.0, 2.0], &operator)
            .expect_err("checked recovery must return the LU failure");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let error = try_gmres_deflated(&operator, &deflation, &[1.0, 2.0], None, &config)
            .expect_err("checked deflated GMRES must return the coarse solve failure");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let legacy = gmres_deflated(&operator, &deflation, &[1.0, 2.0], None, &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }
}
