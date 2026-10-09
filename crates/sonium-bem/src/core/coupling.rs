//! Conservative interface-trace contracts for exterior BEM coupling.
//!
//! This module intentionally contains no FEM dependency. An interior solver
//! supplies trace values and lumped interface measures; BEM owns the checked
//! transfer, convention, adjoint, and residual contracts at its boundary.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::{AdjointCapability, LinearOperator, Preconditioner, SolverError};
use thiserror::Error;

use crate::core::conventions::BemConvention;

/// Interface convention required before coupling an interior trace to BEM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CouplingConvention {
    /// BEM harmonic and coordinate convention.
    pub bem: BemConvention,
    /// The interface normal points from the interior domain toward the BEM
    /// exterior domain.
    pub normal_points_out_of_interior: bool,
}

impl CouplingConvention {
    /// Fail closed on an incompatible normal orientation or malformed signs.
    pub fn validate(self) -> Result<(), CouplingError> {
        if !self.bem.outward_normals || !self.normal_points_out_of_interior {
            return Err(CouplingError::IncompatibleOrientation);
        }
        if !matches!(self.bem.time_sign, -1 | 1) || !matches!(self.bem.spatial_sign, -1 | 1) {
            return Err(CouplingError::InvalidConvention);
        }
        Ok(())
    }
}

/// Conservative map from interior trace DOFs to exterior BEM trace DOFs.
///
/// `weights[target]` stores `(source, interpolation_weight)` pairs. The map
/// preserves constants row-wise and preserves lumped interface measure
/// column-wise; its weighted Hermitian adjoint is constructed explicitly.
#[derive(Debug, Clone, PartialEq)]
pub struct ConservativeTraceTransfer {
    /// Non-negative source interpolation weights for each target DOF.
    pub weights: Vec<Vec<(usize, f64)>>,
    /// Lumped positive interior trace measures.
    pub source_measures: Vec<f64>,
    /// Lumped positive exterior/BEM trace measures.
    pub target_measures: Vec<f64>,
}

/// Dense partitioned FEM--BEM trace block operator.
///
/// The two diagonal blocks are supplied by the interior (typically FEM) and
/// exterior (BEM) discretizations. The off-diagonal blocks are generated from
/// [`ConservativeTraceTransfer`], so constants, lumped measures, and the
/// mass-weighted adjoint are shared by forward, block, and adjoint actions.
/// This is a small/diagnostic assembly seam; large production systems can use
/// the same transfer with their native sparse or matrix-free diagonal blocks.
#[derive(Debug, Clone)]
pub struct CoupledTraceSystem {
    /// Interior/FEM square trace operator.
    pub interior_operator: DenseMatrix<Complex64>,
    /// Exterior/BEM square trace operator.
    pub exterior_operator: DenseMatrix<Complex64>,
    /// Conservative interior-to-exterior trace map.
    pub transfer: ConservativeTraceTransfer,
    /// Complex coupling scale. The reverse block uses its conjugate.
    pub coupling_scale: Complex64,
    /// Validated harmonic/normal contract used by this block system.
    pub convention: CouplingConvention,
}

/// Matrix-free partitioned FEM--BEM trace operator.
///
/// The diagonal blocks are owned as [`LinearOperator`] trait objects, so a
/// sparse FEM matrix and a compressed BEM action can be coupled without
/// materialising either block (or the off-diagonal transfer). The block
/// layout and signs are identical to [`CoupledTraceSystem`]. Construction is
/// checked up front and requires exact transpose and Hermitian actions from
/// both diagonal blocks; this makes the checked adjoint methods safe for
/// solver and sensitivity workflows.
pub struct CoupledTraceOperator {
    interior_operator: Box<dyn LinearOperator<Complex64>>,
    exterior_operator: Box<dyn LinearOperator<Complex64>>,
    transfer: ConservativeTraceTransfer,
    coupling_scale: Complex64,
    convention: CouplingConvention,
}

/// Coupled block residual evidence, reported in the natural interface scales.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledResidualReport {
    /// Relative interior-block residual norm.
    pub interior_relative_residual: f64,
    /// Relative exterior/BEM-block residual norm.
    pub exterior_relative_residual: f64,
    /// Relative mismatch between transferred interior and exterior traces.
    pub trace_relative_mismatch: f64,
}

/// Trace-transfer and coupling-boundary failures.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CouplingError {
    /// A trace dimension or table is empty.
    #[error("coupling trace transfer requires non-empty {field}")]
    Empty {
        /// Rejected table or vector.
        field: &'static str,
    },
    /// A measure is not finite and strictly positive.
    #[error("coupling {side} measure {index} must be finite and positive")]
    InvalidMeasure {
        /// Interior or exterior side.
        side: &'static str,
        /// Measure index.
        index: usize,
    },
    /// A target row does not define a valid convex interpolation.
    #[error("coupling target row {target} is invalid: {reason}")]
    InvalidRow {
        /// Target BEM trace index.
        target: usize,
        /// Validation diagnostic.
        reason: String,
    },
    /// The map fails measure conservation for one source trace DOF.
    #[error(
        "coupling transfer is not conservative at source {source_index}: expected {expected}, got {actual}"
    )]
    NonConservative {
        /// Source interior trace index.
        source_index: usize,
        /// Required source measure.
        expected: f64,
        /// Weighted mapped measure.
        actual: f64,
    },
    /// Input vector has the wrong length.
    #[error("coupling {field} length {got} does not match expected {expected}")]
    Dimension {
        /// Input field name.
        field: &'static str,
        /// Observed length.
        got: usize,
        /// Required length.
        expected: usize,
    },
    /// A supplied complex trace or residual entry is not finite.
    #[error("coupling {field} entry {index} must be finite")]
    NonFinite {
        /// Trace, residual, or right-hand-side vector name.
        field: &'static str,
        /// Rejected vector entry.
        index: usize,
    },
    /// BEM and interior interface normals do not have the declared relation.
    #[error("coupling requires outward BEM normals and an interior-to-exterior interface normal")]
    IncompatibleOrientation,
    /// BEM convention signs were not valid unit signs.
    #[error("coupling convention signs must be ±1")]
    InvalidConvention,
    /// A supplied diagonal block is not square or does not match its trace.
    #[error(
        "coupling {block} operator has invalid shape {rows}x{cols} (expected {expected}x{expected})"
    )]
    InvalidBlock {
        /// Interior or exterior block name.
        block: &'static str,
        /// Number of matrix rows.
        rows: usize,
        /// Number of matrix columns.
        cols: usize,
        /// Required trace dimension.
        expected: usize,
    },
    /// A supplied block contains a non-finite coefficient.
    #[error("coupling {block} operator entry {index} must be finite")]
    NonFiniteBlock {
        /// Interior or exterior block name.
        block: &'static str,
        /// Flattened coefficient index.
        index: usize,
    },
    /// Coupling scale is non-finite.
    #[error("coupling scale must be finite")]
    NonFiniteScale,
    /// A matrix-free diagonal block does not expose an exact adjoint action.
    #[error("coupling {block} operator lacks exact {operation} action")]
    UnsupportedAdjoint {
        /// Interior or exterior block name.
        block: &'static str,
        /// `transpose` or `Hermitian transpose`.
        operation: &'static str,
    },
    /// A checked diagonal-block action failed after construction.
    #[error("coupling {operation} operator action failed: {reason}")]
    OperatorFailure {
        /// Forward, transpose, or Hermitian action label.
        operation: &'static str,
        /// Underlying solver diagnostic.
        reason: String,
    },
    /// A nested trace region has an invalid identifier or operator shape.
    #[error("coupling region {region} invalid: {reason}")]
    InvalidRegion {
        /// Region index in the nested trace graph.
        region: usize,
        /// Validation diagnostic.
        reason: String,
    },
    /// A nested trace edge has invalid endpoints, dimensions, or scale.
    #[error("coupling edge {edge} invalid: {reason}")]
    InvalidEdge {
        /// Stable edge identifier.
        edge: String,
        /// Validation diagnostic.
        reason: String,
    },
}

#[derive(Clone, Copy)]
enum DenseSolveAction {
    Forward,
    Transpose,
    Hermitian,
}

impl DenseSolveAction {
    fn rhs_field(self) -> &'static str {
        match self {
            Self::Forward => "coupled rhs block",
            Self::Transpose => "coupled transpose rhs block",
            Self::Hermitian => "coupled adjoint rhs block",
        }
    }

    fn solution_field(self) -> &'static str {
        match self {
            Self::Forward => "coupled solution block",
            Self::Transpose => "coupled transpose solution block",
            Self::Hermitian => "coupled Hermitian solution block",
        }
    }

    fn operation(self) -> &'static str {
        match self {
            Self::Forward => "coupled block direct solve",
            Self::Transpose => "coupled transpose block direct solve",
            Self::Hermitian => "coupled Hermitian block direct solve",
        }
    }
}

impl CoupledTraceSystem {
    /// Construct and validate a dense partitioned trace system.
    pub fn try_new(
        interior_operator: DenseMatrix<Complex64>,
        exterior_operator: DenseMatrix<Complex64>,
        transfer: ConservativeTraceTransfer,
        coupling_scale: Complex64,
    ) -> Result<Self, CouplingError> {
        Self::try_new_with_convention(
            interior_operator,
            exterior_operator,
            transfer,
            coupling_scale,
            CouplingConvention {
                bem: crate::core::conventions::ACOUSTIC_CONVENTION,
                normal_points_out_of_interior: true,
            },
        )
    }

    /// Construct a partitioned trace system with an explicit convention.
    pub fn try_new_with_convention(
        interior_operator: DenseMatrix<Complex64>,
        exterior_operator: DenseMatrix<Complex64>,
        transfer: ConservativeTraceTransfer,
        coupling_scale: Complex64,
        convention: CouplingConvention,
    ) -> Result<Self, CouplingError> {
        let system = Self {
            interior_operator,
            exterior_operator,
            transfer,
            coupling_scale,
            convention,
        };
        system.validate()?;
        Ok(system)
    }

    /// Validate block shapes, transfer dimensions, and finite coefficients.
    pub fn validate(&self) -> Result<(), CouplingError> {
        self.convention.validate()?;
        self.transfer.validate()?;
        validate_dense_block(
            "interior",
            &self.interior_operator,
            self.transfer.source_measures.len(),
        )?;
        validate_dense_block(
            "exterior",
            &self.exterior_operator,
            self.transfer.target_measures.len(),
        )?;
        if !self.coupling_scale.re.is_finite() || !self.coupling_scale.im.is_finite() {
            return Err(CouplingError::NonFiniteScale);
        }
        Ok(())
    }

    /// Number of rows/columns in the monolithic state vector.
    pub fn dimension(&self) -> usize {
        self.transfer.source_measures.len() + self.transfer.target_measures.len()
    }

    /// Apply the partitioned forward block operator.
    ///
    /// The state is `[interior_trace, exterior_trace]`. The upper-right block
    /// uses the transfer's mass-weighted Hermitian adjoint; the lower-left
    /// block uses the forward conservative map. This gives callers a
    /// physically scaled FEM--BEM coupling without assembling a dense map by
    /// hand.
    pub fn apply(&self, state: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        let source_count = self.transfer.source_measures.len();
        let target_count = self.transfer.target_measures.len();
        if state.len() != source_count + target_count {
            return Err(CouplingError::Dimension {
                field: "coupled state",
                got: state.len(),
                expected: source_count + target_count,
            });
        }
        validate_complex_values("coupled state", state)?;
        let interior = &state[..source_count];
        let exterior = &state[source_count..];
        let mut interior_result = dense_matvec(&self.interior_operator, interior);
        let mut exterior_result = dense_matvec(&self.exterior_operator, exterior);
        let mapped_to_exterior = self.transfer.apply(interior)?;
        let mapped_to_interior = self.transfer.apply_adjoint(exterior)?;
        for (value, coupling) in interior_result.iter_mut().zip(mapped_to_interior) {
            *value += self.coupling_scale * coupling;
        }
        for (value, coupling) in exterior_result.iter_mut().zip(mapped_to_exterior) {
            *value += self.coupling_scale.conj() * coupling;
        }
        let mut result = interior_result;
        result.extend(exterior_result);
        validate_complex_values("coupled output", &result)?;
        Ok(result)
    }

    /// Apply the coupled operator to several right-hand sides.
    pub fn apply_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        if states.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled state block",
                got: states.nrows(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled state block", states.as_slice())?;
        let mut result = DenseMatrix::from_fn(self.dimension(), states.ncols(), |_, _| {
            Complex64::new(0.0, 0.0)
        });
        for column in 0..states.ncols() {
            let applied = self.apply(&states.column_to_vec(column))?;
            result.set_column(column, &applied);
        }
        validate_complex_values("coupled output block", result.as_slice())?;
        Ok(result)
    }

    /// Apply the exact Hermitian adjoint of the materialized coupled block.
    ///
    /// This dense diagnostic path intentionally derives the adjoint from the
    /// same assembled matrix as [`Self::to_dense_matrix`], avoiding a second
    /// sign or measure convention that could drift from the forward action.
    pub fn apply_adjoint(&self, state: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        if state.len() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled adjoint state",
                got: state.len(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled adjoint state", state)?;
        let matrix = self.to_dense_matrix()?;
        let result = dense_hermitian_matvec(&matrix, state);
        validate_complex_values("coupled adjoint output", &result)?;
        Ok(result)
    }

    /// Apply the exact Hermitian adjoint to multiple right-hand sides.
    /// Apply the algebraic transpose of the assembled coupled block.
    pub fn apply_transpose(&self, state: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        if state.len() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled transpose state",
                got: state.len(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled transpose state", state)?;
        let matrix = self.to_dense_matrix()?;
        let result = dense_transpose_matvec(&matrix, state);
        validate_complex_values("coupled transpose output", &result)?;
        Ok(result)
    }

    /// Apply the exact Hermitian adjoint to multiple right-hand sides.
    pub fn apply_adjoint_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        if states.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled adjoint state block",
                got: states.nrows(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled adjoint state block", states.as_slice())?;
        let mut result = DenseMatrix::from_fn(self.dimension(), states.ncols(), |_, _| {
            Complex64::new(0.0, 0.0)
        });
        for column in 0..states.ncols() {
            let applied = self.apply_adjoint(&states.column_to_vec(column))?;
            result.set_column(column, &applied);
        }
        validate_complex_values("coupled adjoint output block", result.as_slice())?;
        Ok(result)
    }

    /// Materialize the same forward action as a dense monolithic matrix.
    /// Apply the algebraic transpose to several right-hand sides.
    pub fn apply_transpose_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        if states.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled transpose state block",
                got: states.nrows(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled transpose state block", states.as_slice())?;
        let mut result = DenseMatrix::from_fn(self.dimension(), states.ncols(), |_, _| {
            Complex64::new(0.0, 0.0)
        });
        for column in 0..states.ncols() {
            let applied = self.apply_transpose(&states.column_to_vec(column))?;
            result.set_column(column, &applied);
        }
        validate_complex_values("coupled transpose output block", result.as_slice())?;
        Ok(result)
    }

    /// Materialize the validated forward coupled block as a dense matrix.
    pub fn to_dense_matrix(&self) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.validate()?;
        let source_count = self.transfer.source_measures.len();
        let target_count = self.transfer.target_measures.len();
        let mut matrix = DenseMatrix::from_fn(self.dimension(), self.dimension(), |_, _| {
            Complex64::new(0.0, 0.0)
        });
        for row in 0..source_count {
            for column in 0..source_count {
                matrix[[row, column]] = self.interior_operator[[row, column]];
            }
        }
        for row in 0..target_count {
            for column in 0..target_count {
                matrix[[source_count + row, source_count + column]] =
                    self.exterior_operator[[row, column]];
            }
        }
        for (target, row_weights) in self.transfer.weights.iter().enumerate() {
            for &(source, weight) in row_weights {
                let weighted_adjoint = self.transfer.target_measures[target] * weight
                    / self.transfer.source_measures[source];
                matrix[[source, source_count + target]] = self.coupling_scale * weighted_adjoint;
                matrix[[source_count + target, source]] = self.coupling_scale.conj() * weight;
            }
        }
        validate_complex_values("coupled dense matrix", matrix.as_slice())?;
        Ok(matrix)
    }

    /// Return a normalized true residual for a state/right-hand side pair.
    pub fn relative_residual(
        &self,
        state: &[Complex64],
        rhs: &[Complex64],
    ) -> Result<f64, CouplingError> {
        if rhs.len() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled rhs",
                got: rhs.len(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled rhs", rhs)?;
        let applied = self.apply(state)?;
        let numerator = applied
            .iter()
            .zip(rhs)
            .map(|(value, expected)| (*value - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let denominator = rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let residual = numerator / denominator;
        if residual.is_finite() {
            Ok(residual)
        } else {
            Err(CouplingError::NonFinite {
                field: "coupled residual",
                index: 0,
            })
        }
    }

    /// Solve the validated dense coupled system with a bounded LU fallback.
    ///
    /// This path materializes the monolithic matrix and is intended for small
    /// partitioned FEM--BEM systems, diagnostics, and reference solves. Large
    /// production systems should use [`CoupledTraceOperator`] with an
    /// iterative solver instead. The candidate is admitted only when the
    /// independently evaluated forward action has a finite true residual.
    pub fn solve(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate()?;
        self.validate_rhs(rhs, "coupled rhs")?;
        let matrix = self.to_dense_matrix()?;
        let solution = sonium_solvers::direct::lu_solve(&matrix, rhs)
            .map_err(|error| operator_error("coupled direct solve", error))?;
        self.admit_solution(
            &solution,
            rhs,
            DenseSolveAction::Forward,
            "coupled direct solve",
        )?;
        Ok(solution)
    }

    /// Solve the algebraic-transpose coupled system with a checked dense LU.
    pub fn solve_transpose(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate()?;
        self.validate_rhs(rhs, "coupled transpose rhs")?;
        let matrix = self.to_dense_matrix()?;
        let solution = sonium_solvers::direct::lu_solve_transpose(&matrix, rhs)
            .map_err(|error| operator_error("coupled transpose direct solve", error))?;
        self.admit_solution(
            &solution,
            rhs,
            DenseSolveAction::Transpose,
            "coupled transpose direct solve",
        )?;
        Ok(solution)
    }

    /// Solve the exact Hermitian-adjoint coupled system with checked dense LU.
    ///
    /// The residual is checked through [`Self::apply_adjoint`], so an
    /// incorrect adjoint action cannot silently pass the reference path.
    pub fn solve_adjoint(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate()?;
        self.validate_rhs(rhs, "coupled adjoint rhs")?;
        let matrix = self.to_dense_matrix()?;
        let solution = sonium_solvers::direct::lu_solve_conj_transpose(&matrix, rhs)
            .map_err(|error| operator_error("coupled Hermitian direct solve", error))?;
        self.admit_solution(
            &solution,
            rhs,
            DenseSolveAction::Hermitian,
            "coupled Hermitian direct solve",
        )?;
        Ok(solution)
    }

    /// Alias for [`Self::solve_adjoint`] using the operator terminology used
    /// by the generic solver traits.
    pub fn solve_hermitian(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.solve_adjoint(rhs)
    }

    /// Solve several forward coupled systems while reusing one LU factorization.
    pub fn solve_block(
        &self,
        rhs: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.solve_block_with(rhs, DenseSolveAction::Forward)
    }

    /// Solve several Hermitian-adjoint coupled systems while reusing one LU
    /// factorization.
    /// Solve several algebraic-transpose systems while reusing one LU factorization.
    pub fn solve_transpose_block(
        &self,
        rhs: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.solve_block_with(rhs, DenseSolveAction::Transpose)
    }

    /// Solve several Hermitian-adjoint coupled systems while reusing one LU factorization.
    pub fn solve_adjoint_block(
        &self,
        rhs: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.solve_block_with(rhs, DenseSolveAction::Hermitian)
    }

    /// Alias for [`Self::solve_adjoint_block`].
    pub fn solve_hermitian_block(
        &self,
        rhs: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.solve_adjoint_block(rhs)
    }

    /// Return a normalized true residual for a Hermitian-adjoint solve.
    pub fn relative_adjoint_residual(
        &self,
        state: &[Complex64],
        rhs: &[Complex64],
    ) -> Result<f64, CouplingError> {
        self.validate_rhs(rhs, "coupled adjoint rhs")?;
        if state.len() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled adjoint state",
                got: state.len(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled adjoint state", state)?;
        let applied = self.apply_adjoint(state)?;
        let numerator = applied
            .iter()
            .zip(rhs)
            .map(|(value, expected)| (*value - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let denominator = rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let residual = numerator / denominator;
        if residual.is_finite() {
            Ok(residual)
        } else {
            Err(CouplingError::NonFinite {
                field: "coupled adjoint residual",
                index: 0,
            })
        }
    }

    /// Return normalized true residual for an algebraic-transpose solve.
    pub fn relative_transpose_residual(
        &self,
        state: &[Complex64],
        rhs: &[Complex64],
    ) -> Result<f64, CouplingError> {
        self.validate_rhs(rhs, "coupled transpose rhs")?;
        if state.len() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled transpose state",
                got: state.len(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled transpose state", state)?;
        let applied = self.apply_transpose(state)?;
        let numerator = applied
            .iter()
            .zip(rhs)
            .map(|(value, expected)| (*value - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let denominator = rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let residual = numerator / denominator;
        if residual.is_finite() {
            Ok(residual)
        } else {
            Err(CouplingError::NonFinite {
                field: "coupled transpose residual",
                index: 0,
            })
        }
    }

    /// Alias for [`Self::relative_adjoint_residual`].
    pub fn relative_hermitian_residual(
        &self,
        state: &[Complex64],
        rhs: &[Complex64],
    ) -> Result<f64, CouplingError> {
        self.relative_adjoint_residual(state, rhs)
    }

    fn solve_block_with(
        &self,
        rhs: &DenseMatrix<Complex64>,
        action: DenseSolveAction,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.validate()?;
        if rhs.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: action.rhs_field(),
                got: rhs.nrows(),
                expected: self.dimension(),
            });
        }
        validate_complex_values(action.rhs_field(), rhs.as_slice())?;
        let matrix = self.to_dense_matrix()?;
        let factor = sonium_solvers::direct::lu_factorize(&matrix)
            .map_err(|error| operator_error("coupled block factorization", error))?;
        let mut result =
            DenseMatrix::from_fn(self.dimension(), rhs.ncols(), |_, _| Complex64::default());
        for column in 0..rhs.ncols() {
            let column_rhs = rhs.column_to_vec(column);
            let solution = match action {
                DenseSolveAction::Forward => factor.solve(&column_rhs),
                DenseSolveAction::Transpose => factor.solve_transpose(&column_rhs),
                DenseSolveAction::Hermitian => factor.solve_conj_transpose(&column_rhs),
            }
            .map_err(|error| operator_error(action.operation(), error))?;
            self.admit_solution(&solution, &column_rhs, action, action.operation())?;
            result.set_column(column, &solution);
        }
        validate_complex_values(action.solution_field(), result.as_slice())?;
        Ok(result)
    }

    fn validate_rhs(&self, rhs: &[Complex64], field: &'static str) -> Result<(), CouplingError> {
        if rhs.len() != self.dimension() {
            return Err(CouplingError::Dimension {
                field,
                got: rhs.len(),
                expected: self.dimension(),
            });
        }
        validate_complex_values(field, rhs)
    }

    fn admit_solution(
        &self,
        solution: &[Complex64],
        rhs: &[Complex64],
        action: DenseSolveAction,
        operation: &'static str,
    ) -> Result<(), CouplingError> {
        let residual = match action {
            DenseSolveAction::Forward => self.relative_residual(solution, rhs)?,
            DenseSolveAction::Transpose => self.relative_transpose_residual(solution, rhs)?,
            DenseSolveAction::Hermitian => self.relative_adjoint_residual(solution, rhs)?,
        };
        if residual > 1.0e-8 {
            return Err(CouplingError::OperatorFailure {
                operation,
                reason: format!("true relative residual {residual:e} exceeds 1e-8"),
            });
        }
        Ok(())
    }
}

impl CoupledTraceOperator {
    /// Construct a checked matrix-free FEM--BEM trace operator.
    pub fn try_new(
        interior_operator: Box<dyn LinearOperator<Complex64>>,
        exterior_operator: Box<dyn LinearOperator<Complex64>>,
        transfer: ConservativeTraceTransfer,
        coupling_scale: Complex64,
    ) -> Result<Self, CouplingError> {
        Self::try_new_with_convention(
            interior_operator,
            exterior_operator,
            transfer,
            coupling_scale,
            CouplingConvention {
                bem: crate::core::conventions::ACOUSTIC_CONVENTION,
                normal_points_out_of_interior: true,
            },
        )
    }

    /// Construct with an explicit harmonic and normal convention.
    pub fn try_new_with_convention(
        interior_operator: Box<dyn LinearOperator<Complex64>>,
        exterior_operator: Box<dyn LinearOperator<Complex64>>,
        transfer: ConservativeTraceTransfer,
        coupling_scale: Complex64,
        convention: CouplingConvention,
    ) -> Result<Self, CouplingError> {
        convention.validate()?;
        transfer.validate()?;
        validate_operator_block(
            "interior",
            interior_operator.as_ref(),
            transfer.source_measures.len(),
        )?;
        validate_operator_block(
            "exterior",
            exterior_operator.as_ref(),
            transfer.target_measures.len(),
        )?;
        if !coupling_scale.re.is_finite() || !coupling_scale.im.is_finite() {
            return Err(CouplingError::NonFiniteScale);
        }
        for (block, operator) in [
            ("interior", interior_operator.as_ref()),
            ("exterior", exterior_operator.as_ref()),
        ] {
            if operator.transpose_capability() != AdjointCapability::Exact {
                return Err(CouplingError::UnsupportedAdjoint {
                    block,
                    operation: "transpose",
                });
            }
            if operator.hermitian_capability() != AdjointCapability::Exact {
                return Err(CouplingError::UnsupportedAdjoint {
                    block,
                    operation: "Hermitian transpose",
                });
            }
        }
        Ok(Self {
            interior_operator,
            exterior_operator,
            transfer,
            coupling_scale,
            convention,
        })
    }

    /// Number of rows and columns in the coupled state.
    pub fn dimension(&self) -> usize {
        self.transfer.source_measures.len() + self.transfer.target_measures.len()
    }

    /// Access the validated transfer map used by this operator.
    pub fn transfer(&self) -> &ConservativeTraceTransfer {
        &self.transfer
    }

    /// Access the validated harmonic and normal convention.
    pub fn convention(&self) -> CouplingConvention {
        self.convention
    }

    /// Apply the coupled forward action with checked dimensions and values.
    pub fn try_apply(&self, state: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "coupled state")?;
        let source_count = self.transfer.source_measures.len();
        let (interior_state, exterior_state) = state.split_at(source_count);
        let mut interior = self
            .interior_operator
            .try_apply(interior_state)
            .map_err(|error| operator_error("interior forward", error))?;
        let mut exterior = self
            .exterior_operator
            .try_apply(exterior_state)
            .map_err(|error| operator_error("exterior forward", error))?;
        let mapped_to_interior = self.transfer.apply_adjoint(exterior_state)?;
        let mapped_to_exterior = self.transfer.apply(interior_state)?;
        for (value, coupling) in interior.iter_mut().zip(mapped_to_interior) {
            *value += self.coupling_scale * coupling;
        }
        for (value, coupling) in exterior.iter_mut().zip(mapped_to_exterior) {
            *value += self.coupling_scale.conj() * coupling;
        }
        let mut result = interior;
        result.extend(exterior);
        validate_complex_values("coupled output", &result)?;
        Ok(result)
    }

    /// Apply the ordinary algebraic transpose with checked inputs.
    pub fn try_apply_transpose(
        &self,
        state: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "coupled transpose state")?;
        let source_count = self.transfer.source_measures.len();
        let (interior_state, exterior_state) = state.split_at(source_count);
        let mut interior = self
            .interior_operator
            .try_apply_transpose(interior_state)
            .map_err(|error| operator_error("interior transpose", error))?;
        let mut exterior = self
            .exterior_operator
            .try_apply_transpose(exterior_state)
            .map_err(|error| operator_error("exterior transpose", error))?;
        let mapped_to_interior = self.transfer.apply_transpose(exterior_state)?;
        let mapped_to_exterior = self.transfer.apply_adjoint_transpose(interior_state)?;
        for (value, coupling) in interior.iter_mut().zip(mapped_to_interior) {
            *value += self.coupling_scale.conj() * coupling;
        }
        for (value, coupling) in exterior.iter_mut().zip(mapped_to_exterior) {
            *value += self.coupling_scale * coupling;
        }
        let mut result = interior;
        result.extend(exterior);
        validate_complex_values("coupled transpose output", &result)?;
        Ok(result)
    }

    /// Apply the exact Hermitian adjoint with checked inputs.
    pub fn try_apply_hermitian(
        &self,
        state: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "coupled Hermitian state")?;
        let source_count = self.transfer.source_measures.len();
        let (interior_state, exterior_state) = state.split_at(source_count);
        let mut interior = self
            .interior_operator
            .try_apply_hermitian(interior_state)
            .map_err(|error| operator_error("interior Hermitian transpose", error))?;
        let mut exterior = self
            .exterior_operator
            .try_apply_hermitian(exterior_state)
            .map_err(|error| operator_error("exterior Hermitian transpose", error))?;
        let mapped_to_interior = self.transfer.apply_transpose(exterior_state)?;
        let mapped_to_exterior = self.transfer.apply_adjoint_transpose(interior_state)?;
        for (value, coupling) in interior.iter_mut().zip(mapped_to_interior) {
            *value += self.coupling_scale * coupling;
        }
        for (value, coupling) in exterior.iter_mut().zip(mapped_to_exterior) {
            *value += self.coupling_scale.conj() * coupling;
        }
        let mut result = interior;
        result.extend(exterior);
        validate_complex_values("coupled Hermitian output", &result)?;
        Ok(result)
    }

    /// Apply the coupled forward operator to several right-hand sides.
    ///
    /// The diagonal FEM/BEM operators receive one checked block action each,
    /// and the conservative transfer maps all columns in one checked call.
    /// This keeps block Krylov callers from silently falling back to a
    /// scalar action per column while preserving the same coupling signs as
    /// [`Self::try_apply`].
    pub fn try_apply_block(
        &self,
        states: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        self.convention.validate()?;
        self.transfer.validate()?;
        if states.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled state block",
                got: states.nrows(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled state block", states.as_slice())?;
        let source_count = self.transfer.source_measures.len();
        let target_count = self.transfer.target_measures.len();
        let interior_state = DenseMatrix::from_fn(source_count, states.ncols(), |row, column| {
            states[[row, column]]
        });
        let exterior_state = DenseMatrix::from_fn(target_count, states.ncols(), |row, column| {
            states[[source_count + row, column]]
        });
        let mut interior = DenseMatrix::zeros(source_count, states.ncols());
        let mut exterior = DenseMatrix::zeros(target_count, states.ncols());
        self.interior_operator
            .try_apply_block(&interior_state, &mut interior)
            .map_err(|error| operator_error("interior block action", error))?;
        self.exterior_operator
            .try_apply_block(&exterior_state, &mut exterior)
            .map_err(|error| operator_error("exterior block action", error))?;
        let mapped_to_interior = self.transfer.apply_adjoint_block(&exterior_state)?;
        let mapped_to_exterior = self.transfer.apply_block(&interior_state)?;
        for row in 0..source_count {
            for column in 0..states.ncols() {
                interior[[row, column]] += self.coupling_scale * mapped_to_interior[[row, column]];
            }
        }
        for row in 0..target_count {
            for column in 0..states.ncols() {
                exterior[[row, column]] +=
                    self.coupling_scale.conj() * mapped_to_exterior[[row, column]];
            }
        }
        let mut result = DenseMatrix::zeros(self.dimension(), states.ncols());
        for column in 0..states.ncols() {
            let mut values = Vec::with_capacity(self.dimension());
            values.extend((0..source_count).map(|row| interior[[row, column]]));
            values.extend((0..target_count).map(|row| exterior[[row, column]]));
            result.set_column(column, &values);
        }
        validate_complex_values("coupled block output", result.as_slice())?;
        Ok(result)
    }

    /// Solve a coupled FEM--BEM block system with checked matrix-free GMRES.
    /// The solver result includes independently recomputed true residuals for
    /// every RHS column.
    pub fn try_block_gmres(
        &self,
        rhs: &DenseMatrix<Complex64>,
        config: &sonium_solvers::iterative::BlockGmresConfig<f64>,
    ) -> Result<sonium_solvers::iterative::BlockGmresSolution<Complex64>, CouplingError> {
        self.convention.validate()?;
        self.transfer.validate()?;
        if rhs.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled block GMRES rhs",
                got: rhs.nrows(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled block GMRES rhs", rhs.as_slice())?;
        sonium_solvers::iterative::try_block_gmres(self, rhs, config)
            .map_err(|error| operator_error("coupled block GMRES", error))
    }

    /// Solve a coupled FEM--BEM block system with a checked block
    /// preconditioner. The preconditioner may be a coupled triangular or
    /// block-diagonal view and receives the same multi-RHS contract.
    pub fn try_block_gmres_preconditioned<P>(
        &self,
        rhs: &DenseMatrix<Complex64>,
        preconditioner: &P,
        config: &sonium_solvers::iterative::BlockGmresConfig<f64>,
    ) -> Result<sonium_solvers::iterative::BlockGmresSolution<Complex64>, CouplingError>
    where
        P: Preconditioner<Complex64>,
    {
        self.convention.validate()?;
        self.transfer.validate()?;
        if rhs.nrows() != self.dimension() {
            return Err(CouplingError::Dimension {
                field: "coupled block GMRES rhs",
                got: rhs.nrows(),
                expected: self.dimension(),
            });
        }
        validate_complex_values("coupled block GMRES rhs", rhs.as_slice())?;
        sonium_solvers::iterative::try_block_gmres_preconditioned(
            self,
            rhs,
            preconditioner,
            None,
            config,
        )
        .map_err(|error| operator_error("coupled preconditioned block GMRES", error))
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
        validate_complex_values(field, state)
    }
}

impl LinearOperator<Complex64> for CoupledTraceOperator {
    fn num_rows(&self) -> usize {
        self.dimension()
    }

    fn num_cols(&self) -> usize {
        self.dimension()
    }

    fn apply(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply(state)
            .unwrap_or_else(|error| panic!("coupled trace forward action failed: {error}"))
    }

    fn try_apply(&self, state: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        CoupledTraceOperator::try_apply(self, state).map_err(solver_error)
    }

    fn apply_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.num_cols(),
            "coupled trace input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.num_rows(),
            "coupled trace output length mismatch"
        );
        let result = self.apply(state);
        output.copy_from_slice(&result);
    }

    fn apply_block(&self, states: &DenseMatrix<Complex64>, output: &mut DenseMatrix<Complex64>) {
        assert_eq!(
            states.nrows(),
            self.num_cols(),
            "coupled block input row mismatch"
        );
        assert_eq!(
            output.nrows(),
            self.num_rows(),
            "coupled block output row mismatch"
        );
        assert_eq!(
            states.ncols(),
            output.ncols(),
            "coupled block column count mismatch"
        );
        let result = CoupledTraceOperator::try_apply_block(self, states)
            .unwrap_or_else(|error| panic!("coupled trace block action failed: {error}"));
        output.as_mut_slice().copy_from_slice(result.as_slice());
    }

    fn apply_transpose(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply_transpose(state)
            .unwrap_or_else(|error| panic!("coupled trace transpose action failed: {error}"))
    }

    fn apply_transpose_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.num_rows(),
            "coupled trace transpose input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.num_cols(),
            "coupled trace transpose output length mismatch"
        );
        let result = self.apply_transpose(state);
        output.copy_from_slice(&result);
    }

    fn apply_hermitian(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.try_apply_hermitian(state)
            .unwrap_or_else(|error| panic!("coupled trace Hermitian action failed: {error}"))
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

/// Block factorisation used by [`CoupledTracePreconditioner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoupledPreconditionerKind {
    /// Apply the two diagonal preconditioners independently.
    BlockDiagonal,
    /// Lower block-triangular sweep (interior, then exterior).
    LowerTriangular,
    /// Upper block-triangular sweep (exterior, then interior).
    UpperTriangular,
}

/// Matrix-free block preconditioner for a partitioned FEM--BEM trace system.
///
/// The diagonal actions are supplied by the caller, which allows a sparse FEM
/// preconditioner and a compressed BEM preconditioner to be composed without
/// materialising either block.  Triangular variants include one conservative
/// interface transfer in the sweep and expose the mathematically corresponding
/// transpose and Hermitian actions for adjoint Krylov solves.
pub struct CoupledTracePreconditioner {
    interior_preconditioner: Box<dyn Preconditioner<Complex64>>,
    exterior_preconditioner: Box<dyn Preconditioner<Complex64>>,
    transfer: ConservativeTraceTransfer,
    coupling_scale: Complex64,
    kind: CoupledPreconditionerKind,
    convention: CouplingConvention,
}

impl CoupledTracePreconditioner {
    /// Construct a checked block-diagonal preconditioner.
    pub fn try_new(
        interior_preconditioner: Box<dyn Preconditioner<Complex64>>,
        exterior_preconditioner: Box<dyn Preconditioner<Complex64>>,
        transfer: ConservativeTraceTransfer,
        coupling_scale: Complex64,
    ) -> Result<Self, CouplingError> {
        Self::try_new_with_kind(
            interior_preconditioner,
            exterior_preconditioner,
            transfer,
            coupling_scale,
            CoupledPreconditionerKind::BlockDiagonal,
        )
    }

    /// Construct a checked block preconditioner with an explicit sweep kind.
    pub fn try_new_with_kind(
        interior_preconditioner: Box<dyn Preconditioner<Complex64>>,
        exterior_preconditioner: Box<dyn Preconditioner<Complex64>>,
        transfer: ConservativeTraceTransfer,
        coupling_scale: Complex64,
        kind: CoupledPreconditionerKind,
    ) -> Result<Self, CouplingError> {
        Self::try_new_with_kind_and_convention(
            interior_preconditioner,
            exterior_preconditioner,
            transfer,
            coupling_scale,
            kind,
            CouplingConvention {
                bem: crate::core::conventions::ACOUSTIC_CONVENTION,
                normal_points_out_of_interior: true,
            },
        )
    }

    /// Construct with explicit harmonic and interface-normal conventions.
    pub fn try_new_with_kind_and_convention(
        interior_preconditioner: Box<dyn Preconditioner<Complex64>>,
        exterior_preconditioner: Box<dyn Preconditioner<Complex64>>,
        transfer: ConservativeTraceTransfer,
        coupling_scale: Complex64,
        kind: CoupledPreconditionerKind,
        convention: CouplingConvention,
    ) -> Result<Self, CouplingError> {
        convention.validate()?;
        transfer.validate()?;
        if !coupling_scale.re.is_finite() || !coupling_scale.im.is_finite() {
            return Err(CouplingError::NonFiniteScale);
        }
        for (block, preconditioner) in [
            ("interior", interior_preconditioner.as_ref()),
            ("exterior", exterior_preconditioner.as_ref()),
        ] {
            if preconditioner.transpose_capability() != AdjointCapability::Exact {
                return Err(CouplingError::UnsupportedAdjoint {
                    block,
                    operation: "transpose preconditioner",
                });
            }
            if preconditioner.hermitian_capability() != AdjointCapability::Exact {
                return Err(CouplingError::UnsupportedAdjoint {
                    block,
                    operation: "Hermitian preconditioner",
                });
            }
        }
        Ok(Self {
            interior_preconditioner,
            exterior_preconditioner,
            transfer,
            coupling_scale,
            kind,
            convention,
        })
    }

    /// Number of entries in the coupled state vector.
    pub fn dimension(&self) -> usize {
        self.transfer.source_measures.len() + self.transfer.target_measures.len()
    }

    /// Selected block factorisation strategy.
    pub fn kind(&self) -> CoupledPreconditionerKind {
        self.kind
    }

    /// Interface convention used by this preconditioner.
    pub fn convention(&self) -> CouplingConvention {
        self.convention
    }

    /// Checked forward application.
    pub fn apply_checked(&self, state: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "coupled preconditioner state")?;
        let source_count = self.transfer.source_measures.len();
        let (interior_state, exterior_state) = state.split_at(source_count);
        let (interior, exterior) = match self.kind {
            CoupledPreconditionerKind::BlockDiagonal => (
                self.apply_block_forward(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    interior_state,
                )?,
                self.apply_block_forward(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    exterior_state,
                )?,
            ),
            CoupledPreconditionerKind::LowerTriangular => {
                let interior = self.apply_block_forward(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    interior_state,
                )?;
                let mapped = self.transfer.apply(&interior)?;
                let exterior_rhs = subtract_scaled(
                    exterior_state,
                    self.coupling_scale.conj(),
                    &mapped,
                    "exterior preconditioner rhs",
                )?;
                let exterior = self.apply_block_forward(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    &exterior_rhs,
                )?;
                (interior, exterior)
            }
            CoupledPreconditionerKind::UpperTriangular => {
                let exterior = self.apply_block_forward(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    exterior_state,
                )?;
                let mapped = self.transfer.apply_adjoint(&exterior)?;
                let interior_rhs = subtract_scaled(
                    interior_state,
                    self.coupling_scale,
                    &mapped,
                    "interior preconditioner rhs",
                )?;
                let interior = self.apply_block_forward(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    &interior_rhs,
                )?;
                (interior, exterior)
            }
        };
        let mut result = interior;
        result.extend(exterior);
        validate_complex_values("coupled preconditioner output", &result)?;
        Ok(result)
    }

    /// Checked algebraic-transpose application.
    pub fn apply_transpose_checked(
        &self,
        state: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "coupled preconditioner transpose state")?;
        let source_count = self.transfer.source_measures.len();
        let (interior_state, exterior_state) = state.split_at(source_count);
        let (interior, exterior) = match self.kind {
            CoupledPreconditionerKind::BlockDiagonal => (
                self.apply_block_transpose(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    interior_state,
                )?,
                self.apply_block_transpose(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    exterior_state,
                )?,
            ),
            CoupledPreconditionerKind::LowerTriangular => {
                let exterior = self.apply_block_transpose(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    exterior_state,
                )?;
                let mapped = self.transfer.apply_transpose(&exterior)?;
                let interior_rhs = subtract_scaled(
                    interior_state,
                    self.coupling_scale.conj(),
                    &mapped,
                    "interior transpose preconditioner rhs",
                )?;
                let interior = self.apply_block_transpose(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    &interior_rhs,
                )?;
                (interior, exterior)
            }
            CoupledPreconditionerKind::UpperTriangular => {
                let interior = self.apply_block_transpose(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    interior_state,
                )?;
                let mapped = self.transfer.apply_adjoint_transpose(&interior)?;
                let exterior_rhs = subtract_scaled(
                    exterior_state,
                    self.coupling_scale,
                    &mapped,
                    "exterior transpose preconditioner rhs",
                )?;
                let exterior = self.apply_block_transpose(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    &exterior_rhs,
                )?;
                (interior, exterior)
            }
        };
        let mut result = interior;
        result.extend(exterior);
        validate_complex_values("coupled preconditioner transpose output", &result)?;
        Ok(result)
    }

    /// Checked Hermitian-adjoint application.
    pub fn apply_hermitian_checked(
        &self,
        state: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        self.validate_state(state, "coupled preconditioner Hermitian state")?;
        let source_count = self.transfer.source_measures.len();
        let (interior_state, exterior_state) = state.split_at(source_count);
        let (interior, exterior) = match self.kind {
            CoupledPreconditionerKind::BlockDiagonal => (
                self.apply_block_hermitian(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    interior_state,
                )?,
                self.apply_block_hermitian(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    exterior_state,
                )?,
            ),
            CoupledPreconditionerKind::LowerTriangular => {
                let exterior = self.apply_block_hermitian(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    exterior_state,
                )?;
                let mapped = self.transfer.apply_transpose(&exterior)?;
                let interior_rhs = subtract_scaled(
                    interior_state,
                    self.coupling_scale,
                    &mapped,
                    "interior Hermitian preconditioner rhs",
                )?;
                let interior = self.apply_block_hermitian(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    &interior_rhs,
                )?;
                (interior, exterior)
            }
            CoupledPreconditionerKind::UpperTriangular => {
                let interior = self.apply_block_hermitian(
                    "interior",
                    self.interior_preconditioner.as_ref(),
                    interior_state,
                )?;
                let mapped = self.transfer.apply_adjoint_transpose(&interior)?;
                let exterior_rhs = subtract_scaled(
                    exterior_state,
                    self.coupling_scale.conj(),
                    &mapped,
                    "exterior Hermitian preconditioner rhs",
                )?;
                let exterior = self.apply_block_hermitian(
                    "exterior",
                    self.exterior_preconditioner.as_ref(),
                    &exterior_rhs,
                )?;
                (interior, exterior)
            }
        };
        let mut result = interior;
        result.extend(exterior);
        validate_complex_values("coupled preconditioner Hermitian output", &result)?;
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
        validate_complex_values(field, state)
    }

    fn apply_block_forward(
        &self,
        block: &'static str,
        preconditioner: &dyn Preconditioner<Complex64>,
        rhs: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        map_preconditioner_action(block, "forward", preconditioner.try_apply(rhs), rhs.len())
    }

    fn apply_block_transpose(
        &self,
        block: &'static str,
        preconditioner: &dyn Preconditioner<Complex64>,
        rhs: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        map_preconditioner_action(
            block,
            "transpose",
            preconditioner.try_apply_transpose(rhs),
            rhs.len(),
        )
    }

    fn apply_block_hermitian(
        &self,
        block: &'static str,
        preconditioner: &dyn Preconditioner<Complex64>,
        rhs: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        map_preconditioner_action(
            block,
            "Hermitian",
            preconditioner.try_apply_hermitian(rhs),
            rhs.len(),
        )
    }
}

impl Preconditioner<Complex64> for CoupledTracePreconditioner {
    fn apply(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.apply_checked(state)
            .unwrap_or_else(|error| panic!("coupled trace preconditioner failed: {error}"))
    }

    fn try_apply(&self, state: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        self.apply_checked(state)
            .map_err(coupled_preconditioner_error)
    }

    fn apply_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.dimension(),
            "coupled trace preconditioner input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.dimension(),
            "coupled trace preconditioner output length mismatch"
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
        self.apply_transpose_checked(state).unwrap_or_else(|error| {
            panic!("coupled trace preconditioner transpose failed: {error}")
        })
    }

    fn try_apply_transpose(&self, state: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        self.apply_transpose_checked(state)
            .map_err(coupled_preconditioner_error)
    }

    fn apply_transpose_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.dimension(),
            "coupled trace preconditioner transpose input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.dimension(),
            "coupled trace preconditioner transpose output length mismatch"
        );
        output.copy_from_slice(&self.apply_transpose(state));
    }

    fn apply_hermitian(&self, state: &[Complex64]) -> Vec<Complex64> {
        self.apply_hermitian_checked(state).unwrap_or_else(|error| {
            panic!("coupled trace preconditioner Hermitian failed: {error}")
        })
    }

    fn try_apply_hermitian(&self, state: &[Complex64]) -> Result<Vec<Complex64>, SolverError> {
        self.apply_hermitian_checked(state)
            .map_err(coupled_preconditioner_error)
    }

    fn apply_hermitian_into(&self, state: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            state.len(),
            self.dimension(),
            "coupled trace preconditioner Hermitian input length mismatch"
        );
        assert_eq!(
            output.len(),
            self.dimension(),
            "coupled trace preconditioner Hermitian output length mismatch"
        );
        output.copy_from_slice(&self.apply_hermitian(state));
    }
}

fn subtract_scaled(
    rhs: &[Complex64],
    scale: Complex64,
    mapped: &[Complex64],
    field: &'static str,
) -> Result<Vec<Complex64>, CouplingError> {
    if rhs.len() != mapped.len() {
        return Err(CouplingError::Dimension {
            field,
            got: mapped.len(),
            expected: rhs.len(),
        });
    }
    let result = rhs
        .iter()
        .zip(mapped)
        .map(|(&rhs, &mapped)| rhs - scale * mapped)
        .collect::<Vec<_>>();
    validate_complex_values(field, &result)?;
    Ok(result)
}

fn map_preconditioner_action(
    block: &'static str,
    operation: &'static str,
    result: Result<Vec<Complex64>, SolverError>,
    expected: usize,
) -> Result<Vec<Complex64>, CouplingError> {
    let result = result.map_err(|error| operator_error("coupled trace preconditioner", error))?;
    if result.len() != expected {
        return Err(CouplingError::Dimension {
            field: if block == "interior" {
                "interior preconditioner output"
            } else {
                "exterior preconditioner output"
            },
            got: result.len(),
            expected,
        });
    }
    validate_complex_values(
        if block == "interior" {
            if operation == "forward" {
                "interior preconditioner output"
            } else {
                "interior adjoint preconditioner output"
            }
        } else if operation == "forward" {
            "exterior preconditioner output"
        } else {
            "exterior adjoint preconditioner output"
        },
        &result,
    )?;
    Ok(result)
}

fn coupled_preconditioner_error(error: CouplingError) -> SolverError {
    SolverError::OperationFailed {
        operation: "coupled trace preconditioner",
        reason: error.to_string(),
    }
}

fn validate_operator_block(
    block: &'static str,
    operator: &dyn LinearOperator<Complex64>,
    expected: usize,
) -> Result<(), CouplingError> {
    if operator.num_rows() != expected || operator.num_cols() != expected {
        return Err(CouplingError::InvalidBlock {
            block,
            rows: operator.num_rows(),
            cols: operator.num_cols(),
            expected,
        });
    }
    Ok(())
}

fn operator_error(operation: &'static str, error: impl std::fmt::Display) -> CouplingError {
    CouplingError::OperatorFailure {
        operation,
        reason: error.to_string(),
    }
}

fn solver_error(error: CouplingError) -> SolverError {
    SolverError::OperationFailed {
        operation: "coupled trace operator",
        reason: error.to_string(),
    }
}

fn validate_dense_block(
    block: &'static str,
    matrix: &DenseMatrix<Complex64>,
    expected: usize,
) -> Result<(), CouplingError> {
    if matrix.nrows() != expected || matrix.ncols() != expected {
        return Err(CouplingError::InvalidBlock {
            block,
            rows: matrix.nrows(),
            cols: matrix.ncols(),
            expected,
        });
    }
    for (index, value) in matrix.as_slice().iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(CouplingError::NonFiniteBlock { block, index });
        }
    }
    Ok(())
}

fn dense_matvec(matrix: &DenseMatrix<Complex64>, vector: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[[row, column]] * vector[column])
                .sum()
        })
        .collect()
}

fn dense_transpose_matvec(matrix: &DenseMatrix<Complex64>, vector: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]] * vector[row])
                .sum()
        })
        .collect()
}

fn dense_hermitian_matvec(matrix: &DenseMatrix<Complex64>, vector: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]].conj() * vector[row])
                .sum()
        })
        .collect()
}

impl ConservativeTraceTransfer {
    /// Build a conservative transfer for linear triangular face traces.
    ///
    /// Each target face receives the arithmetic mean of its three source
    /// vertices. The returned vertex list maps the transfer's compact source
    /// indices back to caller-owned/global vertex indices. Source measures
    /// are assembled by lumping one third of every target face measure onto
    /// each incident vertex, so the resulting map is conservative by
    /// construction and still passes the normal checked constructor.
    pub fn try_from_tri3_vertex_faces(
        faces: &[[usize; 3]],
        target_measures: &[f64],
    ) -> Result<(Self, Vec<usize>), CouplingError> {
        if faces.is_empty() {
            return Err(CouplingError::Empty {
                field: "target trace",
            });
        }
        if faces.len() != target_measures.len() {
            return Err(CouplingError::Dimension {
                field: "target_measures",
                got: target_measures.len(),
                expected: faces.len(),
            });
        }

        let mut source_nodes = faces.iter().flatten().copied().collect::<Vec<_>>();
        source_nodes.sort_unstable();
        source_nodes.dedup();
        let mut source_measures = vec![0.0; source_nodes.len()];
        let weights = faces
            .iter()
            .enumerate()
            .map(|(target, face)| {
                let share = target_measures[target] / 3.0;
                face.iter()
                    .map(|&node| {
                        let source = source_nodes
                            .binary_search(&node)
                            .expect("source vertex came from the same face list");
                        source_measures[source] += share;
                        (source, 1.0 / 3.0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let transfer = Self::try_new(weights, source_measures, target_measures.to_vec())?;
        Ok((transfer, source_nodes))
    }

    /// Build and validate a conservative trace transfer.
    pub fn try_new(
        weights: Vec<Vec<(usize, f64)>>,
        source_measures: Vec<f64>,
        target_measures: Vec<f64>,
    ) -> Result<Self, CouplingError> {
        let transfer = Self {
            weights,
            source_measures,
            target_measures,
        };
        transfer.validate()?;
        Ok(transfer)
    }

    /// Validate convex interpolation and measure conservation.
    pub fn validate(&self) -> Result<(), CouplingError> {
        if self.source_measures.is_empty() {
            return Err(CouplingError::Empty {
                field: "source_measures",
            });
        }
        if self.target_measures.is_empty() || self.weights.is_empty() {
            return Err(CouplingError::Empty {
                field: "target trace",
            });
        }
        if self.target_measures.len() != self.weights.len() {
            return Err(CouplingError::Dimension {
                field: "target_measures",
                got: self.target_measures.len(),
                expected: self.weights.len(),
            });
        }
        validate_measures("source", &self.source_measures)?;
        validate_measures("target", &self.target_measures)?;
        let mut accumulated = vec![0.0; self.source_measures.len()];
        for (target, row) in self.weights.iter().enumerate() {
            if row.is_empty() {
                return Err(CouplingError::InvalidRow {
                    target,
                    reason: "row is empty".into(),
                });
            }
            let mut row_sum = 0.0;
            let mut seen = vec![false; self.source_measures.len()];
            for &(source, weight) in row {
                if source >= self.source_measures.len() {
                    return Err(CouplingError::InvalidRow {
                        target,
                        reason: "source index is outside source trace".into(),
                    });
                }
                if !weight.is_finite() || weight < 0.0 {
                    return Err(CouplingError::InvalidRow {
                        target,
                        reason: "weight is not finite and non-negative".into(),
                    });
                }
                if seen[source] {
                    return Err(CouplingError::InvalidRow {
                        target,
                        reason: "source index appears more than once".into(),
                    });
                }
                seen[source] = true;
                row_sum += weight;
                accumulated[source] += self.target_measures[target] * weight;
            }
            if (row_sum - 1.0).abs() > 1.0e-12 {
                return Err(CouplingError::InvalidRow {
                    target,
                    reason: "weights do not preserve constants".into(),
                });
            }
        }
        for (source, (&expected, &actual)) in
            self.source_measures.iter().zip(&accumulated).enumerate()
        {
            if (expected - actual).abs() > 1.0e-12 * expected.max(actual).max(1.0) {
                return Err(CouplingError::NonConservative {
                    source_index: source,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Transfer an interior trace onto the exterior BEM trace.
    pub fn apply(&self, source_values: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        if source_values.len() != self.source_measures.len() {
            return Err(CouplingError::Dimension {
                field: "source trace",
                got: source_values.len(),
                expected: self.source_measures.len(),
            });
        }
        validate_complex_values("source trace", source_values)?;
        let transferred = self
            .weights
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(source_index, weight)| source_values[source_index] * weight)
                    .sum()
            })
            .collect::<Vec<_>>();
        validate_complex_values("transferred trace", &transferred)?;
        Ok(transferred)
    }

    /// Apply the ordinary algebraic transpose of the transfer map.
    ///
    /// This is intentionally distinct from [`Self::apply_adjoint`], whose
    /// mass weighting is required by conservative interface coupling.
    pub fn apply_transpose(
        &self,
        target_values: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        if target_values.len() != self.target_measures.len() {
            return Err(CouplingError::Dimension {
                field: "target trace",
                got: target_values.len(),
                expected: self.target_measures.len(),
            });
        }
        validate_complex_values("target trace", target_values)?;
        let mut source = vec![Complex64::default(); self.source_measures.len()];
        for (target, row) in self.weights.iter().enumerate() {
            for &(source_index, weight) in row {
                source[source_index] += target_values[target] * weight;
            }
        }
        validate_complex_values("transposed trace", &source)?;
        Ok(source)
    }

    /// Apply the ordinary transpose of the mass-weighted adjoint map.
    ///
    /// If `B = M_s⁻¹ Tᵀ M_t`, this evaluates `Bᵀ x` without constructing `B`.
    pub fn apply_adjoint_transpose(
        &self,
        source_values: &[Complex64],
    ) -> Result<Vec<Complex64>, CouplingError> {
        if source_values.len() != self.source_measures.len() {
            return Err(CouplingError::Dimension {
                field: "source trace",
                got: source_values.len(),
                expected: self.source_measures.len(),
            });
        }
        validate_complex_values("source trace", source_values)?;
        let scaled = source_values
            .iter()
            .zip(&self.source_measures)
            .map(|(&value, &measure)| value / measure)
            .collect::<Vec<_>>();
        let mut target = self.apply(&scaled)?;
        for (value, &measure) in target.iter_mut().zip(&self.target_measures) {
            *value *= measure;
        }
        validate_complex_values("transposed adjoint trace", &target)?;
        Ok(target)
    }

    /// Apply the conservative map to a block of source traces.
    ///
    /// Columns represent independent right-hand sides. The checked block
    /// path validates the complete buffer before writing output, preserving
    /// the same constants, measure conservation, and finite-value contract
    /// as [`Self::apply`].
    pub fn apply_block(
        &self,
        source_values: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        if source_values.nrows() != self.source_measures.len() {
            return Err(CouplingError::Dimension {
                field: "source trace block",
                got: source_values.nrows(),
                expected: self.source_measures.len(),
            });
        }
        validate_complex_values("source trace block", source_values.as_slice())?;
        let mut target =
            DenseMatrix::from_fn(self.target_measures.len(), source_values.ncols(), |_, _| {
                Complex64::new(0.0, 0.0)
            });
        for column in 0..source_values.ncols() {
            let source = source_values.column_to_vec(column);
            let mapped = self.apply(&source)?;
            target.set_column(column, &mapped);
        }
        validate_complex_values("target trace block", target.as_slice())?;
        Ok(target)
    }

    /// Apply the mass-weighted Hermitian adjoint back onto the interior trace.
    pub fn apply_adjoint(&self, target: &[Complex64]) -> Result<Vec<Complex64>, CouplingError> {
        if target.len() != self.target_measures.len() {
            return Err(CouplingError::Dimension {
                field: "target trace",
                got: target.len(),
                expected: self.target_measures.len(),
            });
        }
        validate_complex_values("target trace", target)?;
        let mut source = vec![Complex64::new(0.0, 0.0); self.source_measures.len()];
        for (target_index, row) in self.weights.iter().enumerate() {
            for &(source_index, weight) in row {
                source[source_index] += target[target_index]
                    * (self.target_measures[target_index] * weight
                        / self.source_measures[source_index]);
            }
        }
        validate_complex_values("adjoint trace", &source)?;
        Ok(source)
    }

    /// Apply the mass-weighted Hermitian adjoint to multiple trace columns.
    pub fn apply_adjoint_block(
        &self,
        target_values: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, CouplingError> {
        if target_values.nrows() != self.target_measures.len() {
            return Err(CouplingError::Dimension {
                field: "target trace block",
                got: target_values.nrows(),
                expected: self.target_measures.len(),
            });
        }
        validate_complex_values("target trace block", target_values.as_slice())?;
        let mut source =
            DenseMatrix::from_fn(self.source_measures.len(), target_values.ncols(), |_, _| {
                Complex64::new(0.0, 0.0)
            });
        for column in 0..target_values.ncols() {
            let target_column = target_values.column_to_vec(column);
            let mapped = self.apply_adjoint(&target_column)?;
            source.set_column(column, &mapped);
        }
        validate_complex_values("adjoint trace block", source.as_slice())?;
        Ok(source)
    }

    /// Check the weighted complex-inner-product adjoint identity on probes.
    pub fn check_adjoint(
        &self,
        source_probe: &[Complex64],
        target_probe: &[Complex64],
    ) -> Result<f64, CouplingError> {
        let transferred = self.apply(source_probe)?;
        let adjoint = self.apply_adjoint(target_probe)?;
        let forward = weighted_inner_product(target_probe, &transferred, &self.target_measures);
        let reverse = weighted_inner_product(&adjoint, source_probe, &self.source_measures);
        let error =
            (forward - reverse).norm() / forward.norm().max(reverse.norm()).max(f64::MIN_POSITIVE);
        if !error.is_finite() {
            return Err(CouplingError::NonFinite {
                field: "adjoint check",
                index: 0,
            });
        }
        Ok(error)
    }

    /// Report normalized coupled-block residuals and interface trace mismatch.
    pub fn residual_report(
        &self,
        interior_residual: &[Complex64],
        interior_rhs: &[Complex64],
        exterior_residual: &[Complex64],
        exterior_rhs: &[Complex64],
        interior_trace: &[Complex64],
        exterior_trace: &[Complex64],
    ) -> Result<CoupledResidualReport, CouplingError> {
        let transferred = self.apply(interior_trace)?;
        validate_complex_values("exterior trace", exterior_trace)?;
        validate_complex_values("interior residual", interior_residual)?;
        validate_complex_values("interior rhs", interior_rhs)?;
        validate_complex_values("exterior residual", exterior_residual)?;
        validate_complex_values("exterior rhs", exterior_rhs)?;
        if exterior_trace.len() != transferred.len() {
            return Err(CouplingError::Dimension {
                field: "exterior trace",
                got: exterior_trace.len(),
                expected: transferred.len(),
            });
        }
        let trace_relative_mismatch = relative_difference(exterior_trace, &transferred);
        if !trace_relative_mismatch.is_finite() {
            return Err(CouplingError::NonFinite {
                field: "trace mismatch",
                index: 0,
            });
        }
        Ok(CoupledResidualReport {
            interior_relative_residual: relative_norm(interior_residual, interior_rhs)?,
            exterior_relative_residual: relative_norm(exterior_residual, exterior_rhs)?,
            trace_relative_mismatch,
        })
    }
}

fn validate_measures(side: &'static str, measures: &[f64]) -> Result<(), CouplingError> {
    for (index, &measure) in measures.iter().enumerate() {
        if !measure.is_finite() || measure <= 0.0 {
            return Err(CouplingError::InvalidMeasure { side, index });
        }
    }
    Ok(())
}

pub(crate) fn validate_complex_values(
    field: &'static str,
    values: &[Complex64],
) -> Result<(), CouplingError> {
    for (index, value) in values.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(CouplingError::NonFinite { field, index });
        }
    }
    Ok(())
}

fn weighted_inner_product(left: &[Complex64], right: &[Complex64], weights: &[f64]) -> Complex64 {
    left.iter()
        .zip(right)
        .zip(weights)
        .map(|((&left, &right), &weight)| left.conj() * right * weight)
        .sum()
}

fn relative_norm(values: &[Complex64], reference: &[Complex64]) -> Result<f64, CouplingError> {
    if values.len() != reference.len() {
        return Err(CouplingError::Dimension {
            field: "block residual",
            got: values.len(),
            expected: reference.len(),
        });
    }
    let numerator = values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let relative = numerator / denominator;
    if !relative.is_finite() {
        return Err(CouplingError::NonFinite {
            field: "block residual",
            index: 0,
        });
    }
    Ok(relative)
}

fn relative_difference(left: &[Complex64], right: &[Complex64]) -> f64 {
    let numerator = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| (left - right).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = left
        .iter()
        .chain(right)
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::conventions::ACOUSTIC_CONVENTION;

    fn transfer() -> ConservativeTraceTransfer {
        ConservativeTraceTransfer::try_new(
            vec![vec![(0, 0.5), (1, 0.5)]],
            vec![1.0, 1.0],
            vec![2.0],
        )
        .expect("area-conservative transfer")
    }

    #[test]
    fn trace_transfer_preserves_constants_measures_and_weighted_adjointness() {
        let transfer = transfer();
        let source = vec![Complex64::new(2.0, -1.0), Complex64::new(2.0, -1.0)];
        assert_eq!(
            transfer.apply(&source).expect("matching source"),
            vec![source[0]]
        );
        let adjoint = transfer
            .apply_adjoint(&[Complex64::new(3.0, 2.0)])
            .expect("matching target");
        assert_eq!(adjoint, vec![Complex64::new(3.0, 2.0); 2]);
        let error = transfer
            .check_adjoint(
                &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 1.0)],
                &[Complex64::new(2.0, -1.0)],
            )
            .expect("matching probes");
        assert!(error < 1.0e-14, "weighted adjoint error {error:e}");
    }

    #[test]
    fn trace_transfer_block_actions_match_columnwise_and_reject_bad_buffers() {
        let transfer = transfer();
        let source = DenseMatrix::from_fn(2, 2, |row, column| {
            Complex64::new(
                (row + 1 + 2 * column) as f64,
                (column as f64) - 0.5 * (row as f64),
            )
        });
        let mapped = transfer.apply_block(&source).expect("block transfer");
        assert_eq!(mapped.nrows(), 1);
        assert_eq!(mapped.ncols(), 2);
        for column in 0..2 {
            assert_eq!(
                mapped.column_to_vec(column),
                transfer
                    .apply(&source.column_to_vec(column))
                    .expect("column transfer")
            );
        }

        let adjoint_input = DenseMatrix::from_fn(1, 2, |_, column| {
            Complex64::new(1.0 + column as f64, -0.25 * column as f64)
        });
        let adjoint = transfer
            .apply_adjoint_block(&adjoint_input)
            .expect("block adjoint transfer");
        for column in 0..2 {
            assert_eq!(
                adjoint.column_to_vec(column),
                transfer
                    .apply_adjoint(&adjoint_input.column_to_vec(column))
                    .expect("column adjoint transfer")
            );
        }

        let wrong_rows = DenseMatrix::from_fn(1, 1, |_, _| Complex64::new(0.0, 0.0));
        assert!(matches!(
            transfer.apply_block(&wrong_rows),
            Err(CouplingError::Dimension {
                field: "source trace block",
                ..
            })
        ));
        let non_finite = DenseMatrix::from_fn(2, 1, |row, _| {
            if row == 1 {
                Complex64::new(f64::NAN, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        assert!(matches!(
            transfer.apply_block(&non_finite),
            Err(CouplingError::NonFinite {
                field: "source trace block",
                index: 1,
            })
        ));
    }

    #[test]
    fn coupled_trace_system_matches_materialized_block_and_reports_true_residual() {
        let system = CoupledTraceSystem::try_new(
            DenseMatrix::from_fn(2, 2, |row, column| {
                Complex64::new(
                    if row == column { 2.0 } else { 0.25 },
                    0.1 * (row + column) as f64,
                )
            }),
            DenseMatrix::from_fn(1, 1, |_, _| Complex64::new(3.0, -0.5)),
            transfer(),
            Complex64::new(0.3, -0.2),
        )
        .expect("valid coupled trace system");
        let state = vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(-2.0, 0.25),
            Complex64::new(0.75, 1.5),
        ];
        let applied = system.apply(&state).expect("coupled action");
        let dense = system.to_dense_matrix().expect("dense coupled matrix");
        let expected = (0..dense.nrows())
            .map(|row| {
                (0..dense.ncols())
                    .map(|column| dense[[row, column]] * state[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        assert_eq!(applied, expected);

        let states = DenseMatrix::from_fn(3, 2, |row, column| {
            state[row] + Complex64::new(column as f64, -0.25 * column as f64)
        });
        let block = system.apply_block(&states).expect("coupled block action");
        for column in 0..states.ncols() {
            assert_eq!(
                block.column_to_vec(column),
                system
                    .apply(&states.column_to_vec(column))
                    .expect("column coupled action")
            );
        }
        let adjoint = system
            .apply_adjoint(&state)
            .expect("coupled adjoint action");
        let expected_adjoint = (0..dense.ncols())
            .map(|column| {
                (0..dense.nrows())
                    .map(|row| dense[[row, column]].conj() * state[row])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        assert_eq!(adjoint, expected_adjoint);
        let adjoint_block = system
            .apply_adjoint_block(&states)
            .expect("coupled adjoint block action");
        for column in 0..states.ncols() {
            assert_eq!(
                adjoint_block.column_to_vec(column),
                system
                    .apply_adjoint(&states.column_to_vec(column))
                    .expect("column coupled adjoint action")
            );
        }
        let residual = system
            .relative_residual(&state, &applied)
            .expect("zero true residual");
        assert!(residual < 1.0e-14, "relative residual {residual:e}");
    }

    #[test]
    fn coupled_trace_system_rejects_invalid_blocks_and_state_buffers() {
        let invalid = CoupledTraceSystem::try_new(
            DenseMatrix::from_fn(1, 2, |_, _| Complex64::new(0.0, 0.0)),
            DenseMatrix::from_fn(1, 1, |_, _| Complex64::new(0.0, 0.0)),
            transfer(),
            Complex64::new(1.0, 0.0),
        )
        .expect_err("non-square interior block must fail");
        assert!(matches!(
            invalid,
            CouplingError::InvalidBlock {
                block: "interior",
                ..
            }
        ));

        let non_finite = CoupledTraceSystem::try_new(
            DenseMatrix::from_fn(2, 2, |row, column| {
                if row == 1 && column == 1 {
                    Complex64::new(f64::NAN, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }),
            DenseMatrix::from_fn(1, 1, |_, _| Complex64::new(0.0, 0.0)),
            transfer(),
            Complex64::new(1.0, 0.0),
        )
        .expect_err("non-finite interior block must fail");
        assert!(matches!(
            non_finite,
            CouplingError::NonFiniteBlock {
                block: "interior",
                index: 3,
            }
        ));

        let system = CoupledTraceSystem::try_new(
            DenseMatrix::from_fn(2, 2, |row, column| {
                if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }),
            DenseMatrix::from_fn(1, 1, |_, _| Complex64::new(1.0, 0.0)),
            transfer(),
            Complex64::new(1.0, 0.0),
        )
        .expect("valid coupled system");
        assert!(matches!(
            system.apply(&[Complex64::new(0.0, 0.0)]),
            Err(CouplingError::Dimension {
                field: "coupled state",
                ..
            })
        ));
        assert!(matches!(
            system.apply_block(&DenseMatrix::from_fn(2, 1, |_, _| Complex64::new(0.0, 0.0))),
            Err(CouplingError::Dimension {
                field: "coupled state block",
                ..
            })
        ));
    }

    #[test]
    fn coupled_trace_system_rejects_invalid_convention() {
        let invalid_convention = CoupledTraceSystem::try_new_with_convention(
            DenseMatrix::from_fn(2, 2, |row, column| {
                if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }),
            DenseMatrix::from_fn(1, 1, |_, _| Complex64::new(1.0, 0.0)),
            transfer(),
            Complex64::new(1.0, 0.0),
            CouplingConvention {
                bem: BemConvention {
                    outward_normals: false,
                    ..ACOUSTIC_CONVENTION
                },
                normal_points_out_of_interior: true,
            },
        )
        .expect_err("inward convention must fail closed");
        assert!(matches!(
            invalid_convention,
            CouplingError::IncompatibleOrientation
        ));
    }

    #[test]
    fn trace_transfer_reports_blocks_and_rejects_nonconservative_or_inward_contracts() {
        let transfer = transfer();
        let report = transfer
            .residual_report(
                &[Complex64::new(0.1, 0.0), Complex64::new(0.1, 0.0)],
                &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
                &[Complex64::new(0.2, 0.0)],
                &[Complex64::new(2.0, 0.0)],
                &[Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
                &[Complex64::new(2.0, 0.0)],
            )
            .expect("matching blocks");
        assert!((report.interior_relative_residual - 0.1).abs() < 1.0e-14);
        assert!((report.exterior_relative_residual - 0.1).abs() < 1.0e-14);
        assert!(report.trace_relative_mismatch.abs() < 1.0e-14);
        assert!(matches!(
            ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0, 1.0], vec![1.0]),
            Err(CouplingError::NonConservative { .. })
        ));
        assert!(matches!(
            CouplingConvention {
                bem: ACOUSTIC_CONVENTION,
                normal_points_out_of_interior: false,
            }
            .validate(),
            Err(CouplingError::IncompatibleOrientation)
        ));
    }

    #[test]
    fn trace_transfer_rejects_non_finite_values_before_mapping() {
        let transfer = transfer();
        assert!(matches!(
            transfer.apply(&[Complex64::new(1.0, 0.0), Complex64::new(f64::NAN, 0.0),]),
            Err(CouplingError::NonFinite {
                field: "source trace",
                index: 1,
            })
        ));
        assert!(matches!(
            transfer.apply_adjoint(&[Complex64::new(f64::INFINITY, 0.0)]),
            Err(CouplingError::NonFinite {
                field: "target trace",
                index: 0,
            })
        ));
    }

    #[test]
    fn tri3_face_constructor_returns_compact_conservative_vertex_map() {
        let (transfer, source_nodes) = ConservativeTraceTransfer::try_from_tri3_vertex_faces(
            &[[10, 20, 30], [20, 30, 40]],
            &[1.0, 2.0],
        )
        .expect("valid triangular face map");
        assert_eq!(source_nodes, vec![10, 20, 30, 40]);
        let mapped = transfer
            .apply(&[Complex64::new(2.0, -1.0); 4])
            .expect("constant trace maps");
        assert_eq!(mapped, vec![Complex64::new(2.0, -1.0); 2]);
        assert!(matches!(
            ConservativeTraceTransfer::try_from_tri3_vertex_faces(&[[1, 1, 2]], &[1.0]),
            Err(CouplingError::InvalidRow { .. })
        ));
    }

    #[test]
    fn residual_report_rejects_non_finite_block_evidence() {
        let transfer = transfer();
        let error = transfer
            .residual_report(
                &[Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, 0.0)],
                &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
                &[Complex64::new(0.0, 0.0)],
                &[Complex64::new(1.0, 0.0)],
                &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
                &[Complex64::new(1.0, 0.0)],
            )
            .expect_err("non-finite residual must fail closed");
        assert!(matches!(
            error,
            CouplingError::NonFinite {
                field: "interior residual",
                index: 0,
            }
        ));
    }
}
