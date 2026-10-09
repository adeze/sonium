//! Checked matrix-free coupling for nested and multi-region BEM traces.
//!
//! Each edge connects two independently discretised trace spaces.  The
//! conservative transfer on an edge contributes a pair of reciprocal blocks,
//! so a graph with several interfaces can be applied without assembling one
//! monolithic dense matrix.  Edges are intentionally undirected at the
//! algebraic level: `source_region` and `target_region` only select the two
//! trace spaces and the transfer orientation.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::{AdjointCapability, LinearOperator, Preconditioner, SolverError};

use super::coupling::{ConservativeTraceTransfer, CouplingConvention, CouplingError};

/// A named region and its square trace operator.
pub struct NestedTraceRegion {
    name: String,
    operator: Box<dyn LinearOperator<Complex64>>,
}

impl NestedTraceRegion {
    /// Create a region backed by a matrix-free or sparse trace operator.
    pub fn new(name: impl Into<String>, operator: Box<dyn LinearOperator<Complex64>>) -> Self {
        Self {
            name: name.into(),
            operator,
        }
    }

    /// Stable region name used in diagnostics and scheduling.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// One conservative interface between two trace regions.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedTraceEdge {
    /// Stable edge identifier.
    pub id: String,
    /// Region owning the source side of `transfer`.
    pub source_region: usize,
    /// Region owning the target side of `transfer`.
    pub target_region: usize,
    /// Conservative source-to-target transfer map.
    pub transfer: ConservativeTraceTransfer,
    /// Complex scale applied to the source-row reciprocal block.
    pub coupling_scale: Complex64,
}

impl NestedTraceEdge {
    /// Construct an interface edge.  Endpoint and dimension validation occurs
    /// when the complete graph is built.
    pub fn new(
        id: impl Into<String>,
        source_region: usize,
        target_region: usize,
        transfer: ConservativeTraceTransfer,
        coupling_scale: Complex64,
    ) -> Self {
        Self {
            id: id.into(),
            source_region,
            target_region,
            transfer,
            coupling_scale,
        }
    }
}

/// Matrix-free checked operator for a nested or multi-region trace graph.
pub struct NestedTraceOperator {
    regions: Vec<NestedTraceRegion>,
    edges: Vec<NestedTraceEdge>,
    offsets: Vec<usize>,
    convention: CouplingConvention,
}

/// Sweep strategy for a nested trace preconditioner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NestedTracePreconditionerKind {
    /// Apply each region preconditioner independently.
    BlockDiagonal,
    /// Process regions from the lowest to highest index, retaining lower
    /// reciprocal interface blocks in the forward sweep.
    ForwardSweep,
    /// Process regions from the highest to lowest index, retaining upper
    /// reciprocal interface blocks in the reverse sweep.
    ReverseSweep,
}

/// Checked block preconditioner for a nested or multi-region trace graph.
///
/// Region preconditioners are supplied independently because FEM and BEM
/// blocks commonly use different factorizations.  The explicit dimension
/// vector is required by the solver preconditioner trait, which intentionally
/// does not expose a size method.  Ordered sweeps use the edge endpoint order
/// as a deterministic triangular split; reciprocal transfer orientation and
/// exact adjoint actions are preserved in all three modes.
pub struct NestedTracePreconditioner {
    preconditioners: Vec<Box<dyn Preconditioner<Complex64>>>,
    dimensions: Vec<usize>,
    edges: Vec<NestedTraceEdge>,
    offsets: Vec<usize>,
    kind: NestedTracePreconditionerKind,
    convention: CouplingConvention,
}

impl NestedTraceOperator {
    /// Build a checked graph of independently owned region operators.
    pub fn try_new(
        regions: Vec<NestedTraceRegion>,
        edges: Vec<NestedTraceEdge>,
        convention: CouplingConvention,
    ) -> Result<Self, CouplingError> {
        convention.validate()?;
        if regions.is_empty() {
            return Err(CouplingError::Empty { field: "regions" });
        }

        let mut offsets = Vec::with_capacity(regions.len() + 1);
        offsets.push(0);
        for (region, entry) in regions.iter().enumerate() {
            if entry.name.trim().is_empty() {
                return Err(CouplingError::InvalidRegion {
                    region,
                    reason: "region name must not be empty".to_string(),
                });
            }
            if regions[..region]
                .iter()
                .any(|previous| previous.name == entry.name)
            {
                return Err(CouplingError::InvalidRegion {
                    region,
                    reason: format!("duplicate region name `{}`", entry.name),
                });
            }
            let rows = entry.operator.num_rows();
            let cols = entry.operator.num_cols();
            if rows == 0 || rows != cols {
                return Err(CouplingError::InvalidRegion {
                    region,
                    reason: format!("operator shape {rows}x{cols} is not a non-empty square"),
                });
            }
            if entry.operator.transpose_capability() != AdjointCapability::Exact {
                return Err(CouplingError::InvalidRegion {
                    region,
                    reason: "operator lacks exact transpose action".to_string(),
                });
            }
            if entry.operator.hermitian_capability() != AdjointCapability::Exact {
                return Err(CouplingError::InvalidRegion {
                    region,
                    reason: "operator lacks exact Hermitian action".to_string(),
                });
            }
            offsets.push(offsets[region] + rows);
        }

        let mut edge_ids = Vec::<&str>::with_capacity(edges.len());
        for edge in &edges {
            if edge.id.trim().is_empty() {
                return Err(CouplingError::InvalidEdge {
                    edge: edge.id.clone(),
                    reason: "edge identifier must not be empty".to_string(),
                });
            }
            if edge_ids.contains(&edge.id.as_str()) {
                return Err(CouplingError::InvalidEdge {
                    edge: edge.id.clone(),
                    reason: "duplicate edge identifier".to_string(),
                });
            }
            edge_ids.push(&edge.id);
            if edge.source_region >= regions.len() || edge.target_region >= regions.len() {
                return Err(CouplingError::InvalidEdge {
                    edge: edge.id.clone(),
                    reason: format!(
                        "endpoint out of range ({} -> {}, {} regions)",
                        edge.source_region,
                        edge.target_region,
                        regions.len()
                    ),
                });
            }
            if edge.source_region == edge.target_region {
                return Err(CouplingError::InvalidEdge {
                    edge: edge.id.clone(),
                    reason: "self-coupling is not a nested interface".to_string(),
                });
            }
            edge.transfer.validate()?;
            let source_dim = regions[edge.source_region].operator.num_cols();
            let target_dim = regions[edge.target_region].operator.num_cols();
            if edge.transfer.source_measures.len() != source_dim
                || edge.transfer.target_measures.len() != target_dim
            {
                return Err(CouplingError::InvalidEdge {
                    edge: edge.id.clone(),
                    reason: format!(
                        "transfer dimensions {}x{} do not match region dimensions {}x{}",
                        edge.transfer.source_measures.len(),
                        edge.transfer.target_measures.len(),
                        source_dim,
                        target_dim
                    ),
                });
            }
            if !edge.coupling_scale.re.is_finite() || !edge.coupling_scale.im.is_finite() {
                return Err(CouplingError::InvalidEdge {
                    edge: edge.id.clone(),
                    reason: "coupling scale must be finite".to_string(),
                });
            }
        }

        Ok(Self {
            regions,
            edges,
            offsets,
            convention,
        })
    }

    /// Number of regions in the graph.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Number of interfaces in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Region names in deterministic graph order.
    pub fn region_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.regions.iter().map(|region| region.name.as_str())
    }

    /// Monolithic row offsets, including the final dimension.
    pub fn region_offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Global state dimension.
    pub fn dimension(&self) -> usize {
        *self.offsets.last().expect("non-empty regions")
    }

    /// Validated convention shared by all interfaces.
    pub fn convention(&self) -> CouplingConvention {
        self.convention
    }

    /// Checked forward action over all region blocks and interfaces.
    pub fn try_apply(&self, state: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "nested coupled state")?;
        let mut result =
            self.flatten_region_actions(self.apply_region_actions(state, Action::Forward)?);
        for edge in &self.edges {
            let source = self.region_slice(state, edge.source_region);
            let target = self.region_slice(state, edge.target_region);
            let source_contribution = edge.transfer.apply_adjoint(target)?;
            let target_contribution = edge.transfer.apply(source)?;
            add_scaled(
                &mut result[self.region_range(edge.source_region)],
                edge.coupling_scale,
                &source_contribution,
                "nested source coupling",
            )?;
            add_scaled(
                &mut result[self.region_range(edge.target_region)],
                edge.coupling_scale.conj(),
                &target_contribution,
                "nested target coupling",
            )?;
        }
        super::coupling::validate_complex_values("nested coupled output", &result)?;
        Ok(result)
    }

    /// Checked ordinary algebraic-transpose action.
    pub fn try_apply_transpose(
        &self,
        state: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "nested transpose state")?;
        let mut result =
            self.flatten_region_actions(self.apply_region_actions(state, Action::Transpose)?);
        for edge in &self.edges {
            let source = self.region_slice(state, edge.source_region);
            let target = self.region_slice(state, edge.target_region);
            let source_contribution = edge.transfer.apply_transpose(target)?;
            let target_contribution = edge.transfer.apply_adjoint_transpose(source)?;
            add_scaled(
                &mut result[self.region_range(edge.source_region)],
                edge.coupling_scale.conj(),
                &source_contribution,
                "nested transpose source coupling",
            )?;
            add_scaled(
                &mut result[self.region_range(edge.target_region)],
                edge.coupling_scale,
                &target_contribution,
                "nested transpose target coupling",
            )?;
        }
        super::coupling::validate_complex_values("nested transpose output", &result)?;
        Ok(result)
    }

    /// Checked exact Hermitian-adjoint action.
    pub fn try_apply_hermitian(
        &self,
        state: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "nested Hermitian state")?;
        let mut result =
            self.flatten_region_actions(self.apply_region_actions(state, Action::Hermitian)?);
        for edge in &self.edges {
            let source = self.region_slice(state, edge.source_region);
            let target = self.region_slice(state, edge.target_region);
            let source_contribution = edge.transfer.apply_transpose(target)?;
            let target_contribution = edge.transfer.apply_adjoint_transpose(source)?;
            add_scaled(
                &mut result[self.region_range(edge.source_region)],
                edge.coupling_scale,
                &source_contribution,
                "nested Hermitian source coupling",
            )?;
            add_scaled(
                &mut result[self.region_range(edge.target_region)],
                edge.coupling_scale.conj(),
                &target_contribution,
                "nested Hermitian target coupling",
            )?;
        }
        super::coupling::validate_complex_values("nested Hermitian output", &result)?;
        Ok(result)
    }

    /// Checked forward action for a block of right-hand sides.
    pub fn try_apply_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.apply_block_with(states, Action::Forward)
    }

    /// Checked transpose action for a block of right-hand sides.
    pub fn try_apply_transpose_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.apply_block_with(states, Action::Transpose)
    }

    /// Checked Hermitian action for a block of right-hand sides.
    pub fn try_apply_hermitian_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.apply_block_with(states, Action::Hermitian)
    }

    /// Apply the forward nested operator and return caller-owned output rows.
    ///
    /// Row order, including duplicates, is preserved. The nested region
    /// operators remain matrix-free; this helper only selects rows after the
    /// checked global action and performs no communication.
    pub fn try_apply_rows(
        &self,
        state: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_rows(rows)?;
        let output = self.try_apply(state)?;
        Ok(rows.iter().map(|&row| output[row]).collect())
    }

    /// Apply the algebraic-transpose nested operator on caller-owned rows.
    pub fn try_apply_transpose_rows(
        &self,
        state: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_rows(rows)?;
        let output = self.try_apply_transpose(state)?;
        Ok(rows.iter().map(|&row| output[row]).collect())
    }

    /// Apply the Hermitian-adjoint nested operator on caller-owned rows.
    pub fn try_apply_hermitian_rows(
        &self,
        state: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_rows(rows)?;
        let output = self.try_apply_hermitian(state)?;
        Ok(rows.iter().map(|&row| output[row]).collect())
    }

    /// Apply the forward nested operator to a block and select caller-owned rows.
    pub fn try_apply_rows_block(
        &self,
        states: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.validate_rows(rows)?;
        let output = self.try_apply_block(states)?;
        Ok(select_rows(&output, rows))
    }

    /// Apply the algebraic-transpose nested operator to a block and select rows.
    pub fn try_apply_transpose_rows_block(
        &self,
        states: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.validate_rows(rows)?;
        let output = self.try_apply_transpose_block(states)?;
        Ok(select_rows(&output, rows))
    }

    /// Apply the Hermitian-adjoint nested operator to a block and select rows.
    pub fn try_apply_hermitian_rows_block(
        &self,
        states: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.validate_rows(rows)?;
        let output = self.try_apply_hermitian_block(states)?;
        Ok(select_rows(&output, rows))
    }

    fn validate_rows(&self, rows: &[usize]) -> Result<(), CouplingError> {
        for &row in rows {
            if row >= self.dimension() {
                return Err(CouplingError::InvalidRow {
                    target: row,
                    reason: format!("row outside global range 0..{}", self.dimension()),
                });
            }
        }
        Ok(())
    }

    /// Materialise a dense reference matrix by applying each coordinate basis
    /// vector.  This is intended for diagnostics and tiny contract fixtures.
    pub fn to_dense_matrix(&self) -> Result<DenseMatrix<Complex64>, CouplingError> {
        let dimension = self.dimension();
        let mut matrix = DenseMatrix::from_fn(dimension, dimension, |_, _| Complex64::default());
        for column in 0..dimension {
            let mut basis = vec![Complex64::default(); dimension];
            basis[column] = Complex64::new(1.0, 0.0);
            let action = self.try_apply(&basis)?;
            matrix.set_column(column, &action);
        }
        Ok(matrix)
    }

    /// Solve one nested coupled system with a checked dense LU fallback.
    ///
    /// This is deliberately a bounded orchestration path: it materializes the
    /// matrix and is intended for small partitioned/monolithic FEM--BEM
    /// systems, diagnostics, and reference solves. Large production systems
    /// should pass this operator to an iterative solver instead. The returned
    /// solution is admitted only when the independently applied operator
    /// produces a finite true residual.
    pub fn solve(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(rhs, "nested rhs")?;
        let matrix = self.to_dense_matrix()?;
        let solution = sonium_solvers::direct::lu_solve(&matrix, rhs).map_err(|error| {
            CouplingError::OperatorFailure {
                operation: "nested direct solve",
                reason: error.to_string(),
            }
        })?;
        self.admit_solution(&solution, rhs, Action::Forward, "nested direct solve")?;
        Ok(solution)
    }

    /// Solve the Hermitian-adjoint nested system with checked dense LU.
    ///
    /// As with [`Self::solve`], this is a small-system reference/orchestration
    /// path. The residual is evaluated through the matrix-free Hermitian
    /// action, so an incorrect adjoint implementation cannot silently pass.
    pub fn solve_hermitian(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(rhs, "nested Hermitian rhs")?;
        let matrix = self.to_dense_matrix()?;
        let solution =
            sonium_solvers::direct::lu_solve_conj_transpose(&matrix, rhs).map_err(|error| {
                CouplingError::OperatorFailure {
                    operation: "nested Hermitian direct solve",
                    reason: error.to_string(),
                }
            })?;
        self.admit_solution(
            &solution,
            rhs,
            Action::Hermitian,
            "nested Hermitian direct solve",
        )?;
        Ok(solution)
    }

    /// Solve the algebraic-transpose nested system with checked dense LU.
    pub fn solve_transpose(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(rhs, "nested transpose rhs")?;
        let matrix = self.to_dense_matrix()?;
        let factor = sonium_solvers::direct::lu_factorize(&matrix).map_err(|error| {
            CouplingError::OperatorFailure {
                operation: "nested transpose direct solve",
                reason: error.to_string(),
            }
        })?;
        let solution =
            factor
                .solve_transpose(rhs)
                .map_err(|error| CouplingError::OperatorFailure {
                    operation: "nested transpose direct solve",
                    reason: error.to_string(),
                })?;
        self.admit_solution(
            &solution,
            rhs,
            Action::Transpose,
            "nested transpose direct solve",
        )?;
        Ok(solution)
    }

    /// Solve several nested systems with checked column-wise LU actions.
    pub fn solve_block(
        &self,
        rhs: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.solve_block_with(rhs, Action::Forward)
    }

    /// Solve several Hermitian-adjoint nested systems with checked actions.
    pub fn solve_hermitian_block(
        &self,
        rhs: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.solve_block_with(rhs, Action::Hermitian)
    }

    /// Solve several algebraic-transpose nested systems with checked LU.
    pub fn solve_transpose_block(
        &self,
        rhs: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.solve_block_with(rhs, Action::Transpose)
    }

    /// Solve a nested trace system with checked matrix-free left-preconditioned
    /// GMRES. Unlike [`Self::solve`], this path never materializes the global
    /// dense matrix and is intended for larger CPU-only nested systems.
    pub fn try_gmres_solve<P>(
        &self,
        rhs: &[Complex64],
        preconditioner: &P,
        config: &sonium_solvers::iterative::GmresConfig<f64>,
    ) -> Result<sonium_solvers::iterative::GmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64> + ?Sized,
    {
        self.try_gmres_solve_with_action(rhs, preconditioner, config, Action::Forward)
    }

    /// Solve the algebraic-transpose nested system with checked matrix-free
    /// left-preconditioned GMRES.
    pub fn try_gmres_solve_transpose<P>(
        &self,
        rhs: &[Complex64],
        preconditioner: &P,
        config: &sonium_solvers::iterative::GmresConfig<f64>,
    ) -> Result<sonium_solvers::iterative::GmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64> + ?Sized,
    {
        self.try_gmres_solve_with_action(rhs, preconditioner, config, Action::Transpose)
    }

    /// Solve the Hermitian-adjoint nested system with checked matrix-free
    /// left-preconditioned GMRES.
    pub fn try_gmres_solve_hermitian<P>(
        &self,
        rhs: &[Complex64],
        preconditioner: &P,
        config: &sonium_solvers::iterative::GmresConfig<f64>,
    ) -> Result<sonium_solvers::iterative::GmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64> + ?Sized,
    {
        self.try_gmres_solve_with_action(rhs, preconditioner, config, Action::Hermitian)
    }

    /// Solve several nested trace systems with checked matrix-free block GMRES.
    pub fn try_block_gmres_solve<P>(
        &self,
        rhs: &DenseMatrix<Complex64>,
        preconditioner: &P,
        config: &sonium_solvers::iterative::BlockGmresConfig<f64>,
    ) -> Result<sonium_solvers::iterative::BlockGmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64> + ?Sized,
    {
        self.try_block_gmres_solve_with_action(rhs, preconditioner, config, Action::Forward)
    }

    /// Solve several algebraic-transpose nested systems with checked matrix-free
    /// block GMRES.
    pub fn try_block_gmres_solve_transpose<P>(
        &self,
        rhs: &DenseMatrix<Complex64>,
        preconditioner: &P,
        config: &sonium_solvers::iterative::BlockGmresConfig<f64>,
    ) -> Result<sonium_solvers::iterative::BlockGmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64> + ?Sized,
    {
        self.try_block_gmres_solve_with_action(rhs, preconditioner, config, Action::Transpose)
    }

    /// Solve several Hermitian-adjoint nested systems with checked matrix-free
    /// block GMRES.
    pub fn try_block_gmres_solve_hermitian<P>(
        &self,
        rhs: &DenseMatrix<Complex64>,
        preconditioner: &P,
        config: &sonium_solvers::iterative::BlockGmresConfig<f64>,
    ) -> Result<sonium_solvers::iterative::BlockGmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64> + ?Sized,
    {
        self.try_block_gmres_solve_with_action(rhs, preconditioner, config, Action::Hermitian)
    }

    /// Return a normalized true residual for a nested operator solve.
    pub fn relative_residual(
        &self,
        solution: &[Complex64],
        rhs: &[Complex64],
    ) -> Result<f64, CouplingError> {
        self.relative_residual_with(solution, rhs, Action::Forward)
    }

    /// Return a normalized true residual for a Hermitian-adjoint solve.
    pub fn relative_hermitian_residual(
        &self,
        solution: &[Complex64],
        rhs: &[Complex64],
    ) -> Result<f64, CouplingError> {
        self.relative_residual_with(solution, rhs, Action::Hermitian)
    }

    /// Return a normalized true residual for an algebraic-transpose solve.
    pub fn relative_transpose_residual(
        &self,
        solution: &[Complex64],
        rhs: &[Complex64],
    ) -> Result<f64, CouplingError> {
        self.relative_residual_with(solution, rhs, Action::Transpose)
    }

    fn try_gmres_solve_with_action<P>(
        &self,
        rhs: &[Complex64],
        preconditioner: &P,
        config: &sonium_solvers::iterative::GmresConfig<f64>,
        action: Action,
    ) -> Result<sonium_solvers::iterative::GmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64> + ?Sized,
    {
        self.validate_state(rhs, "nested GMRES rhs")?;
        let operator = NestedTraceActionOperator {
            operator: self,
            action,
        };
        let preconditioner = NestedTraceActionPreconditioner {
            preconditioner,
            action,
        };
        sonium_solvers::iterative::try_gmres_preconditioned(&operator, &preconditioner, rhs, config)
            .map_err(|error| map_iterative_error(action.operation(), error))
    }

    fn try_block_gmres_solve_with_action<P>(
        &self,
        rhs: &DenseMatrix<Complex64>,
        preconditioner: &P,
        config: &sonium_solvers::iterative::BlockGmresConfig<f64>,
        action: Action,
    ) -> Result<sonium_solvers::iterative::BlockGmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64> + ?Sized,
    {
        if rhs.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "nested block GMRES rhs",
                got: rhs.nrows(),
                expected: self.dimension(),
            });
        }
        super::coupling::validate_complex_values("nested block GMRES rhs", rhs.as_slice())?;
        let operator = NestedTraceActionOperator {
            operator: self,
            action,
        };
        let preconditioner = NestedTraceActionPreconditioner {
            preconditioner,
            action,
        };
        sonium_solvers::iterative::try_block_gmres_preconditioned(
            &operator,
            rhs,
            &preconditioner,
            None,
            config,
        )
        .map_err(|error| map_iterative_error(action.operation(), error))
    }

    fn apply_block_with(
        &self,
        states: &DenseMatrix<Complex64>,
        action: Action,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        if states.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "nested coupled state block",
                got: states.nrows(),
                expected: self.dimension(),
            });
        }
        super::coupling::validate_complex_values("nested coupled state block", states.as_slice())?;
        let mut result = DenseMatrix::from_fn(self.dimension(), states.ncols(), |_, _| {
            Complex64::default()
        });
        for column in 0..states.ncols() {
            let applied = match action {
                Action::Forward => self.try_apply(&states.column_to_vec(column))?,
                Action::Transpose => self.try_apply_transpose(&states.column_to_vec(column))?,
                Action::Hermitian => self.try_apply_hermitian(&states.column_to_vec(column))?,
            };
            result.set_column(column, &applied);
        }
        super::coupling::validate_complex_values("nested coupled output block", result.as_slice())?;
        Ok(result)
    }

    fn apply_region_actions(
        &self,
        state: &[Complex64],
        action: Action,
    ) -> Result<Vec<Vec<Complex64>>, CouplingError> {
        self.regions
            .iter()
            .enumerate()
            .map(|(region, entry)| {
                let input = self.region_slice(state, region);
                let result = match action {
                    Action::Forward => entry.operator.try_apply(input),
                    Action::Transpose => entry.operator.try_apply_transpose(input),
                    Action::Hermitian => entry.operator.try_apply_hermitian(input),
                }
                .map_err(|error| CouplingError::OperatorFailure {
                    operation: match action {
                        Action::Forward => "nested region forward",
                        Action::Transpose => "nested region transpose",
                        Action::Hermitian => "nested region Hermitian",
                    },
                    reason: format!("region {region}: {error}"),
                })?;
                if result.len() != input.len() {
                    return Err(CouplingError::InvalidRegion {
                        region,
                        reason: format!(
                            "operator returned {} entries for {}-entry input",
                            result.len(),
                            input.len()
                        ),
                    });
                }
                super::coupling::validate_complex_values("nested region output", &result)?;
                Ok(result)
            })
            .collect()
    }

    fn flatten_region_actions(&self, region_results: Vec<Vec<Complex64>>) -> Vec<Complex64> {
        let mut result = vec![Complex64::default(); self.dimension()];
        for (region, values) in region_results.into_iter().enumerate() {
            result[self.region_range(region)].copy_from_slice(&values);
        }
        result
    }

    fn validate_state(
        &self,
        state: &[Complex64],
        field: &'static str,
    ) -> Result<(), CouplingError> {
        if state.len() != self.dimension() {
            return Err(CouplingError::Dimension {
                field,
                got: state.len(),
                expected: self.dimension(),
            });
        }
        super::coupling::validate_complex_values(field, state)
    }

    fn solve_block_with(
        &self,
        rhs: &DenseMatrix<Complex64>,
        action: Action,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        if rhs.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "nested rhs block",
                got: rhs.nrows(),
                expected: self.dimension(),
            });
        }
        super::coupling::validate_complex_values("nested rhs block", rhs.as_slice())?;
        let matrix = self.to_dense_matrix()?;
        let factor = sonium_solvers::direct::lu_factorize(&matrix).map_err(|error| {
            CouplingError::OperatorFailure {
                operation: "nested block direct solve",
                reason: error.to_string(),
            }
        })?;
        let mut result =
            DenseMatrix::from_fn(self.dimension(), rhs.ncols(), |_, _| Complex64::default());
        for column in 0..rhs.ncols() {
            let column_rhs = rhs.column_to_vec(column);
            let values = match action {
                Action::Forward => factor.solve(&column_rhs),
                Action::Transpose => factor.solve_transpose(&column_rhs),
                Action::Hermitian => factor.solve_conj_transpose(&column_rhs),
            }
            .map_err(|error| CouplingError::OperatorFailure {
                operation: match action {
                    Action::Forward => "nested block direct solve",
                    Action::Transpose => "nested transpose block direct solve",
                    Action::Hermitian => "nested Hermitian block direct solve",
                },
                reason: error.to_string(),
            })?;
            self.admit_solution(
                &values,
                &column_rhs,
                action,
                match action {
                    Action::Forward => "nested block direct solve",
                    Action::Transpose => "nested transpose block direct solve",
                    Action::Hermitian => "nested Hermitian block direct solve",
                },
            )?;
            result.set_column(column, &values);
        }
        Ok(result)
    }

    fn admit_solution(
        &self,
        solution: &[Complex64],
        rhs: &[Complex64],
        action: Action,
        operation: &'static str,
    ) -> Result<(), CouplingError> {
        let residual = self.relative_residual_with(solution, rhs, action)?;
        if !residual.is_finite() {
            return Err(CouplingError::NonFinite {
                field: operation,
                index: 0,
            });
        }
        if residual > 1.0e-8 {
            return Err(CouplingError::OperatorFailure {
                operation,
                reason: format!("true relative residual {residual:e} exceeds 1e-8"),
            });
        }
        Ok(())
    }

    fn relative_residual_with(
        &self,
        solution: &[Complex64],
        rhs: &[Complex64],
        action: Action,
    ) -> Result<f64, CouplingError> {
        self.validate_state(solution, "nested solution")?;
        self.validate_state(rhs, "nested rhs")?;
        let applied = match action {
            Action::Forward => self.try_apply(solution)?,
            Action::Transpose => self.try_apply_transpose(solution)?,
            Action::Hermitian => self.try_apply_hermitian(solution)?,
        };
        let (residual_norm, rhs_norm) =
            applied
                .iter()
                .zip(rhs)
                .fold((0.0, 0.0), |(residual, reference), (value, target)| {
                    (
                        residual + (*value - *target).norm_sqr(),
                        reference + target.norm_sqr(),
                    )
                });
        let denominator = rhs_norm.sqrt().max(f64::MIN_POSITIVE);
        let relative = residual_norm.sqrt() / denominator;
        if !relative.is_finite() {
            return Err(CouplingError::NonFinite {
                field: "nested residual",
                index: 0,
            });
        }
        Ok(relative)
    }

    fn region_range(&self, region: usize) -> std::ops::Range<usize> {
        self.offsets[region]..self.offsets[region + 1]
    }

    fn region_slice<'a>(&self, state: &'a [Complex64], region: usize) -> &'a [Complex64] {
        &state[self.region_range(region)]
    }
}

impl NestedTracePreconditioner {
    /// Construct a checked block-diagonal nested preconditioner.
    pub fn try_new(
        preconditioners: Vec<Box<dyn Preconditioner<Complex64>>>,
        dimensions: Vec<usize>,
        edges: Vec<NestedTraceEdge>,
        convention: CouplingConvention,
    ) -> Result<Self, CouplingError> {
        Self::try_new_with_kind(
            preconditioners,
            dimensions,
            edges,
            NestedTracePreconditionerKind::BlockDiagonal,
            convention,
        )
    }

    /// Construct a checked nested preconditioner with an explicit sweep kind.
    pub fn try_new_with_kind(
        preconditioners: Vec<Box<dyn Preconditioner<Complex64>>>,
        dimensions: Vec<usize>,
        edges: Vec<NestedTraceEdge>,
        kind: NestedTracePreconditionerKind,
        convention: CouplingConvention,
    ) -> Result<Self, CouplingError> {
        convention.validate()?;
        if preconditioners.is_empty() || dimensions.is_empty() {
            return Err(CouplingError::Empty {
                field: "nested preconditioner regions",
            });
        }
        if preconditioners.len() != dimensions.len() {
            return Err(CouplingError::Dimension {
                field: "nested preconditioner regions",
                got: preconditioners.len(),
                expected: dimensions.len(),
            });
        }
        let mut offsets = Vec::with_capacity(dimensions.len() + 1);
        offsets.push(0);
        for (region, (&dimension, preconditioner)) in
            dimensions.iter().zip(&preconditioners).enumerate()
        {
            if dimension == 0 {
                return Err(CouplingError::InvalidRegion {
                    region,
                    reason: "preconditioner dimension must be non-zero".to_string(),
                });
            }
            if preconditioner.transpose_capability() != AdjointCapability::Exact {
                return Err(CouplingError::InvalidRegion {
                    region,
                    reason: "preconditioner lacks exact transpose action".to_string(),
                });
            }
            if preconditioner.hermitian_capability() != AdjointCapability::Exact {
                return Err(CouplingError::InvalidRegion {
                    region,
                    reason: "preconditioner lacks exact Hermitian action".to_string(),
                });
            }
            offsets.push(offsets[region] + dimension);
        }
        validate_preconditioner_edges(&edges, &dimensions)?;
        Ok(Self {
            preconditioners,
            dimensions,
            edges,
            offsets,
            kind,
            convention,
        })
    }

    /// Global state dimension.
    pub fn dimension(&self) -> usize {
        *self.offsets.last().expect("non-empty dimensions")
    }

    /// Number of region blocks.
    pub fn region_count(&self) -> usize {
        self.dimensions.len()
    }

    /// Number of interface edges retained by the sweep.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Selected block factorisation strategy.
    pub fn kind(&self) -> NestedTracePreconditionerKind {
        self.kind
    }

    /// Interface convention used by this preconditioner.
    pub fn convention(&self) -> CouplingConvention {
        self.convention
    }

    /// Checked forward preconditioner application.
    pub fn try_apply_checked(&self, state: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "nested preconditioner state")?;
        match self.kind {
            NestedTracePreconditionerKind::BlockDiagonal => {
                self.apply_block_diagonal(state, SweepAction::Forward)
            }
            NestedTracePreconditionerKind::ForwardSweep => {
                self.apply_lower_sweep(state, SweepAction::Forward)
            }
            NestedTracePreconditionerKind::ReverseSweep => {
                self.apply_upper_sweep(state, SweepAction::Forward)
            }
        }
    }

    /// Checked algebraic-transpose preconditioner application.
    pub fn try_apply_transpose_checked(
        &self,
        state: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "nested preconditioner transpose state")?;
        match self.kind {
            NestedTracePreconditionerKind::BlockDiagonal => {
                self.apply_block_diagonal(state, SweepAction::Transpose)
            }
            // (D + L)^T is solved in reverse order and uses L^T.
            NestedTracePreconditionerKind::ForwardSweep => {
                self.apply_lower_transpose_sweep(state, SweepAction::Transpose)
            }
            // (D + U)^T is solved in forward order and uses U^T.
            NestedTracePreconditionerKind::ReverseSweep => {
                self.apply_upper_transpose_sweep(state, SweepAction::Transpose)
            }
        }
    }

    /// Checked Hermitian-adjoint preconditioner application.
    pub fn try_apply_hermitian_checked(
        &self,
        state: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "nested preconditioner Hermitian state")?;
        match self.kind {
            NestedTracePreconditionerKind::BlockDiagonal => {
                self.apply_block_diagonal(state, SweepAction::Hermitian)
            }
            NestedTracePreconditionerKind::ForwardSweep => {
                self.apply_lower_transpose_sweep(state, SweepAction::Hermitian)
            }
            NestedTracePreconditionerKind::ReverseSweep => {
                self.apply_upper_transpose_sweep(state, SweepAction::Hermitian)
            }
        }
    }

    /// Checked forward block application.
    pub fn try_apply_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.apply_block_with(states, SweepAction::Forward)
    }

    /// Checked transpose block application.
    pub fn try_apply_transpose_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.apply_block_with(states, SweepAction::Transpose)
    }

    /// Checked Hermitian block application.
    pub fn try_apply_hermitian_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.apply_block_with(states, SweepAction::Hermitian)
    }

    fn apply_block_diagonal(
        &self,
        state: &[Complex64],
        action: SweepAction,
    ) -> Result<Vec<Complex64>, CouplingError> {
        let mut result = vec![Complex64::default(); self.dimension()];
        for region in 0..self.region_count() {
            let input = &state[self.region_range(region)];
            let output = self.apply_preconditioner(region, input, action)?;
            result[self.region_range(region)].copy_from_slice(&output);
        }
        super::coupling::validate_complex_values("nested preconditioner output", &result)?;
        Ok(result)
    }

    fn apply_lower_sweep(
        &self,
        state: &[Complex64],
        action: SweepAction,
    ) -> Result<Vec<Complex64>, CouplingError> {
        let mut result = vec![Complex64::default(); self.dimension()];
        for region in 0..self.region_count() {
            let mut rhs = self.region_slice(state, region).to_vec();
            for edge in &self.edges {
                let (low, high) = ordered_endpoints(edge);
                if high != region {
                    continue;
                }
                let contribution = lower_forward_map(edge, self.region_slice(&result, low))?;
                subtract_in_place(&mut rhs, &contribution, "nested lower sweep rhs")?;
            }
            let output = self.apply_preconditioner(region, &rhs, action)?;
            result[self.region_range(region)].copy_from_slice(&output);
        }
        super::coupling::validate_complex_values("nested lower sweep output", &result)?;
        Ok(result)
    }

    fn apply_upper_sweep(
        &self,
        state: &[Complex64],
        action: SweepAction,
    ) -> Result<Vec<Complex64>, CouplingError> {
        let mut result = vec![Complex64::default(); self.dimension()];
        for region in (0..self.region_count()).rev() {
            let mut rhs = self.region_slice(state, region).to_vec();
            for edge in &self.edges {
                let (low, high) = ordered_endpoints(edge);
                if low != region {
                    continue;
                }
                let contribution = upper_forward_map(edge, self.region_slice(&result, high))?;
                subtract_in_place(&mut rhs, &contribution, "nested upper sweep rhs")?;
            }
            let output = self.apply_preconditioner(region, &rhs, action)?;
            result[self.region_range(region)].copy_from_slice(&output);
        }
        super::coupling::validate_complex_values("nested upper sweep output", &result)?;
        Ok(result)
    }

    fn apply_lower_transpose_sweep(
        &self,
        state: &[Complex64],
        action: SweepAction,
    ) -> Result<Vec<Complex64>, CouplingError> {
        let mut result = vec![Complex64::default(); self.dimension()];
        for region in (0..self.region_count()).rev() {
            let mut rhs = self.region_slice(state, region).to_vec();
            for edge in &self.edges {
                let (low, high) = ordered_endpoints(edge);
                if low != region {
                    continue;
                }
                let contribution =
                    lower_transpose_map(edge, self.region_slice(&result, high), action)?;
                subtract_in_place(&mut rhs, &contribution, "nested lower-adjoint sweep rhs")?;
            }
            let output = self.apply_preconditioner(region, &rhs, action)?;
            result[self.region_range(region)].copy_from_slice(&output);
        }
        super::coupling::validate_complex_values("nested lower-adjoint sweep output", &result)?;
        Ok(result)
    }

    fn apply_upper_transpose_sweep(
        &self,
        state: &[Complex64],
        action: SweepAction,
    ) -> Result<Vec<Complex64>, CouplingError> {
        let mut result = vec![Complex64::default(); self.dimension()];
        for region in 0..self.region_count() {
            let mut rhs = self.region_slice(state, region).to_vec();
            for edge in &self.edges {
                let (low, high) = ordered_endpoints(edge);
                if high != region {
                    continue;
                }
                let contribution =
                    upper_transpose_map(edge, self.region_slice(&result, low), action)?;
                subtract_in_place(&mut rhs, &contribution, "nested upper-adjoint sweep rhs")?;
            }
            let output = self.apply_preconditioner(region, &rhs, action)?;
            result[self.region_range(region)].copy_from_slice(&output);
        }
        super::coupling::validate_complex_values("nested upper-adjoint sweep output", &result)?;
        Ok(result)
    }

    fn apply_preconditioner(
        &self,
        region: usize,
        state: &[Complex64],
        action: SweepAction,
    ) -> Result<Vec<Complex64>, CouplingError> {
        let result = match action {
            SweepAction::Forward => self.preconditioners[region].try_apply(state),
            SweepAction::Transpose => self.preconditioners[region].try_apply_transpose(state),
            SweepAction::Hermitian => self.preconditioners[region].try_apply_hermitian(state),
        }
        .map_err(|error| CouplingError::OperatorFailure {
            operation: match action {
                SweepAction::Forward => "nested preconditioner forward",
                SweepAction::Transpose => "nested preconditioner transpose",
                SweepAction::Hermitian => "nested preconditioner Hermitian",
            },
            reason: format!("region {region}: {error}"),
        })?;
        if result.len() != self.dimensions[region] {
            return Err(CouplingError::InvalidRegion {
                region,
                reason: format!(
                    "preconditioner returned {} entries for {}-entry state",
                    result.len(),
                    self.dimensions[region]
                ),
            });
        }
        super::coupling::validate_complex_values("nested preconditioner block output", &result)?;
        Ok(result)
    }

    fn apply_block_with(
        &self,
        states: &DenseMatrix<Complex64>,
        action: SweepAction,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        if states.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "nested preconditioner state block",
                got: states.nrows(),
                expected: self.dimension(),
            });
        }
        super::coupling::validate_complex_values(
            "nested preconditioner state block",
            states.as_slice(),
        )?;
        let mut result = DenseMatrix::from_fn(self.dimension(), states.ncols(), |_, _| {
            Complex64::default()
        });
        for column in 0..states.ncols() {
            let input = states.column_to_vec(column);
            let output = match action {
                SweepAction::Forward => self.try_apply_checked(&input)?,
                SweepAction::Transpose => self.try_apply_transpose_checked(&input)?,
                SweepAction::Hermitian => self.try_apply_hermitian_checked(&input)?,
            };
            result.set_column(column, &output);
        }
        Ok(result)
    }

    fn validate_state(
        &self,
        state: &[Complex64],
        field: &'static str,
    ) -> Result<(), CouplingError> {
        if state.len() != self.dimension() {
            return Err(CouplingError::Dimension {
                field,
                got: state.len(),
                expected: self.dimension(),
            });
        }
        super::coupling::validate_complex_values(field, state)
    }

    fn region_range(&self, region: usize) -> std::ops::Range<usize> {
        self.offsets[region]..self.offsets[region + 1]
    }

    fn region_slice<'a>(&self, state: &'a [Complex64], region: usize) -> &'a [Complex64] {
        &state[self.region_range(region)]
    }
}

#[derive(Clone, Copy)]
enum SweepAction {
    Forward,
    Transpose,
    Hermitian,
}

impl Preconditioner<Complex64> for NestedTracePreconditioner {
    fn apply(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply_checked(state)
            .unwrap_or_else(|error| panic!("nested trace preconditioner failed: {error}"))
    }

    fn try_apply(&self, state: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        Self::try_apply_checked(self, state).map_err(to_solver_error)
    }

    fn apply_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.dimension(),
            "nested preconditioner input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.dimension(),
            "nested preconditioner output length mismatch"
        );
        output.copy_from_slice(&self.apply(state));
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply_transpose_checked(state)
            .unwrap_or_else(|error| panic!("nested trace preconditioner transpose failed: {error}"))
    }

    fn apply_transpose_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.dimension(),
            "nested preconditioner transpose input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.dimension(),
            "nested preconditioner transpose output length mismatch"
        );
        output.copy_from_slice(&self.apply_transpose(state));
    }

    fn apply_hermitian(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply_hermitian_checked(state)
            .unwrap_or_else(|error| panic!("nested trace preconditioner Hermitian failed: {error}"))
    }
}

fn validate_preconditioner_edges(
    edges: &[NestedTraceEdge],
    dimensions: &[usize],
) -> Result<(), CouplingError> {
    let mut edge_ids = Vec::<&str>::with_capacity(edges.len());
    for edge in edges {
        if edge.id.trim().is_empty() {
            return Err(CouplingError::InvalidEdge {
                edge: edge.id.clone(),
                reason: "edge identifier must not be empty".to_string(),
            });
        }
        if edge_ids.contains(&edge.id.as_str()) {
            return Err(CouplingError::InvalidEdge {
                edge: edge.id.clone(),
                reason: "duplicate edge identifier".to_string(),
            });
        }
        edge_ids.push(&edge.id);
        if edge.source_region >= dimensions.len() || edge.target_region >= dimensions.len() {
            return Err(CouplingError::InvalidEdge {
                edge: edge.id.clone(),
                reason: "endpoint out of range".to_string(),
            });
        }
        if edge.source_region == edge.target_region {
            return Err(CouplingError::InvalidEdge {
                edge: edge.id.clone(),
                reason: "self-coupling is not a nested interface".to_string(),
            });
        }
        edge.transfer.validate()?;
        if edge.transfer.source_measures.len() != dimensions[edge.source_region]
            || edge.transfer.target_measures.len() != dimensions[edge.target_region]
        {
            return Err(CouplingError::InvalidEdge {
                edge: edge.id.clone(),
                reason: "transfer dimensions do not match region dimensions".to_string(),
            });
        }
        if !edge.coupling_scale.re.is_finite() || !edge.coupling_scale.im.is_finite() {
            return Err(CouplingError::InvalidEdge {
                edge: edge.id.clone(),
                reason: "coupling scale must be finite".to_string(),
            });
        }
    }
    Ok(())
}

fn ordered_endpoints(edge: &NestedTraceEdge) -> (usize, usize) {
    (
        edge.source_region.min(edge.target_region),
        edge.source_region.max(edge.target_region),
    )
}

fn lower_forward_map(
    edge: &NestedTraceEdge,
    low_state: &[Complex64],
) -> Result<Vec<Complex64>, CouplingError> {
    let (mapped, scale) = if edge.source_region < edge.target_region {
        (edge.transfer.apply(low_state)?, edge.coupling_scale.conj())
    } else {
        (edge.transfer.apply_adjoint(low_state)?, edge.coupling_scale)
    };
    scale_values(mapped, scale, "nested lower coupling")
}

fn upper_forward_map(
    edge: &NestedTraceEdge,
    high_state: &[Complex64],
) -> Result<Vec<Complex64>, CouplingError> {
    let (mapped, scale) = if edge.source_region < edge.target_region {
        (
            edge.transfer.apply_adjoint(high_state)?,
            edge.coupling_scale,
        )
    } else {
        (edge.transfer.apply(high_state)?, edge.coupling_scale.conj())
    };
    scale_values(mapped, scale, "nested upper coupling")
}

fn lower_transpose_map(
    edge: &NestedTraceEdge,
    high_state: &[Complex64],
    action: SweepAction,
) -> Result<Vec<Complex64>, CouplingError> {
    let (mapped, scale) = if edge.source_region < edge.target_region {
        (
            match action {
                SweepAction::Transpose => edge.transfer.apply_transpose(high_state)?,
                SweepAction::Hermitian => edge.transfer.apply_transpose(high_state)?,
                SweepAction::Forward => unreachable!(),
            },
            if matches!(action, SweepAction::Transpose) {
                edge.coupling_scale.conj()
            } else {
                edge.coupling_scale
            },
        )
    } else {
        (
            match action {
                SweepAction::Transpose => edge.transfer.apply_adjoint_transpose(high_state)?,
                SweepAction::Hermitian => edge.transfer.apply_adjoint_transpose(high_state)?,
                SweepAction::Forward => unreachable!(),
            },
            if matches!(action, SweepAction::Transpose) {
                edge.coupling_scale
            } else {
                edge.coupling_scale.conj()
            },
        )
    };
    scale_values(mapped, scale, "nested lower-adjoint coupling")
}

fn upper_transpose_map(
    edge: &NestedTraceEdge,
    low_state: &[Complex64],
    action: SweepAction,
) -> Result<Vec<Complex64>, CouplingError> {
    let (mapped, scale) = if edge.source_region < edge.target_region {
        (
            edge.transfer.apply_adjoint_transpose(low_state)?,
            if matches!(action, SweepAction::Transpose) {
                edge.coupling_scale
            } else {
                edge.coupling_scale.conj()
            },
        )
    } else {
        (
            edge.transfer.apply_transpose(low_state)?,
            if matches!(action, SweepAction::Transpose) {
                edge.coupling_scale.conj()
            } else {
                edge.coupling_scale
            },
        )
    };
    scale_values(mapped, scale, "nested upper-adjoint coupling")
}

fn scale_values(
    values: Vec<Complex64>,
    scale: Complex64,
    field: &'static str,
) -> Result<Vec<Complex64>, CouplingError> {
    let values = values
        .into_iter()
        .map(|value| scale * value)
        .collect::<Vec<_>>();
    super::coupling::validate_complex_values(field, &values)?;
    Ok(values)
}

fn subtract_in_place(
    rhs: &mut [Complex64],
    contribution: &[Complex64],
    field: &'static str,
) -> Result<(), CouplingError> {
    if rhs.len() != contribution.len() {
        return Err(CouplingError::Dimension {
            field,
            got: contribution.len(),
            expected: rhs.len(),
        });
    }
    for (rhs, contribution) in rhs.iter_mut().zip(contribution) {
        *rhs -= *contribution;
    }
    super::coupling::validate_complex_values(field, rhs)
}

fn select_rows(matrix: &DenseMatrix<Complex64>, rows: &[usize]) -> DenseMatrix<Complex64> {
    DenseMatrix::from_fn(rows.len(), matrix.ncols(), |row, column| {
        matrix[[rows[row], column]]
    })
}

#[derive(Clone, Copy)]
enum Action {
    Forward,
    Transpose,
    Hermitian,
}

impl Action {
    fn operation(self) -> &'static str {
        match self {
            Self::Forward => "nested GMRES forward",
            Self::Transpose => "nested GMRES transpose",
            Self::Hermitian => "nested GMRES Hermitian",
        }
    }
}

struct NestedTraceActionOperator<'a> {
    operator: &'a NestedTraceOperator,
    action: Action,
}

impl NestedTraceActionOperator<'_> {
    fn apply_checked(&self, input: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        match self.action {
            Action::Forward => self.operator.try_apply(input),
            Action::Transpose => self.operator.try_apply_transpose(input),
            Action::Hermitian => self.operator.try_apply_hermitian(input),
        }
        .map_err(to_solver_error)
    }

    fn transpose_checked(&self, input: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        match self.action {
            Action::Forward => self.operator.try_apply_transpose(input),
            Action::Transpose => self.operator.try_apply(input),
            // (Aᴴ)ᵀ = conj(A), evaluated without materializing A.
            Action::Hermitian => {
                let conjugated = input.iter().map(|value| value.conj()).collect::<Vec<_>>();
                self.operator
                    .try_apply(&conjugated)
                    .map(|values| values.into_iter().map(|value| value.conj()).collect())
            }
        }
        .map_err(to_solver_error)
    }

    fn hermitian_checked(&self, input: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        match self.action {
            Action::Forward => self.operator.try_apply_hermitian(input),
            // (Aᵀ)ᴴ = conj(A), evaluated without materializing A.
            Action::Transpose => {
                let conjugated = input.iter().map(|value| value.conj()).collect::<Vec<_>>();
                self.operator
                    .try_apply(&conjugated)
                    .map(|values| values.into_iter().map(|value| value.conj()).collect())
            }
            Action::Hermitian => self.operator.try_apply(input),
        }
        .map_err(to_solver_error)
    }
}

impl LinearOperator<Complex64> for NestedTraceActionOperator<'_> {
    fn num_rows(&self) -> usize {
        self.operator.dimension()
    }

    fn num_cols(&self) -> usize {
        self.operator.dimension()
    }

    fn apply(&self, input: &[Complex64]) -> Vec<Complex64> {
        self.apply_checked(input)
            .unwrap_or_else(|error| panic!("{}: {error}", self.action.operation()))
    }

    fn try_apply(&self, input: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        self.apply_checked(input)
    }

    fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            input.len(),
            self.num_cols(),
            "nested GMRES input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.num_rows(),
            "nested GMRES output length mismatch"
        );
        output.copy_from_slice(&self.apply(input));
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose(&self, input: &[Complex64]) -> Vec<Complex64> {
        self.transpose_checked(input).unwrap_or_else(|error| {
            panic!(
                "{} transpose action failed: {error}",
                self.action.operation()
            )
        })
    }

    fn apply_transpose_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            input.len(),
            self.num_rows(),
            "nested GMRES transpose input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.num_cols(),
            "nested GMRES transpose output length mismatch"
        );
        output.copy_from_slice(&self.apply_transpose(input));
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian(&self, input: &[Complex64]) -> Vec<Complex64> {
        self.hermitian_checked(input).unwrap_or_else(|error| {
            panic!(
                "{} Hermitian action failed: {error}",
                self.action.operation()
            )
        })
    }
}

struct NestedTraceActionPreconditioner<'a, P: ?Sized> {
    preconditioner: &'a P,
    action: Action,
}

impl<P> NestedTraceActionPreconditioner<'_, P>
where
    P: Preconditioner<Complex64> + ?Sized,
{
    fn apply_checked(&self, input: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        match self.action {
            Action::Forward => self.preconditioner.try_apply(input),
            Action::Transpose => self.preconditioner.try_apply_transpose(input),
            Action::Hermitian => self.preconditioner.try_apply_hermitian(input),
        }
    }
}

impl<P> Preconditioner<Complex64> for NestedTraceActionPreconditioner<'_, P>
where
    P: Preconditioner<Complex64> + ?Sized,
{
    fn apply(&self, input: &[Complex64]) -> Vec<Complex64> {
        self.apply_checked(input)
            .unwrap_or_else(|error| panic!("nested GMRES preconditioner action failed: {error}"))
    }

    fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            output.len(),
            input.len(),
            "nested GMRES preconditioner output length mismatch"
        );
        output.copy_from_slice(&self.apply(input));
    }
}

impl LinearOperator<Complex64> for NestedTraceOperator {
    fn num_rows(&self) -> usize {
        self.dimension()
    }

    fn num_cols(&self) -> usize {
        self.dimension()
    }

    fn apply(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply(state)
            .unwrap_or_else(|error| panic!("nested trace forward action failed: {error}"))
    }

    fn try_apply(&self, state: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        Self::try_apply(self, state).map_err(to_solver_error)
    }

    fn apply_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.num_cols(),
            "nested trace input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.num_rows(),
            "nested trace output length mismatch"
        );
        output.copy_from_slice(&self.apply(state));
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply_transpose(state)
            .unwrap_or_else(|error| panic!("nested trace transpose action failed: {error}"))
    }

    fn apply_transpose_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.num_rows(),
            "nested trace transpose input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.num_cols(),
            "nested trace transpose output length mismatch"
        );
        output.copy_from_slice(&self.apply_transpose(state));
    }

    fn apply_hermitian(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply_hermitian(state)
            .unwrap_or_else(|error| panic!("nested trace Hermitian action failed: {error}"))
    }
}

fn add_scaled(
    target: &mut [Complex64],
    scale: Complex64,
    contribution: &[Complex64],
    field: &'static str,
) -> Result<(), CouplingError> {
    if target.len() != contribution.len() {
        return Err(CouplingError::Dimension {
            field,
            got: contribution.len(),
            expected: target.len(),
        });
    }
    super::coupling::validate_complex_values(field, contribution)?;
    for (target, contribution) in target.iter_mut().zip(contribution) {
        *target += scale * *contribution;
    }
    super::coupling::validate_complex_values(field, target)
}

fn to_solver_error(error: CouplingError) -> SolverError {
    SolverError::OperationFailed {
        operation: "nested trace operator",
        reason: error.to_string(),
    }
}

fn map_iterative_error(operation: &'static str, error: SolverError) -> CouplingError {
    match error {
        SolverError::DimensionMismatch { expected, got } => CouplingError::Dimension {
            field: operation,
            got,
            expected,
        },
        error => CouplingError::OperatorFailure {
            operation,
            reason: error.to_string(),
        },
    }
}
