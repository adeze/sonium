//! Constant-panel Müller transmission operator.
//!
//! This module assembles the identity-free two-medium interface equations on
//! a closed surface.  The unknowns are pressure `p` and outward normal
//! velocity `v_n`, with the normal directed from the interior/transmitted
//! medium into the exterior/incident medium.  The two trace equations are
//! subtracted so the `±1/2 I` jumps cancel:
//!
//! ```text
//! (K_i + K_e) p
//!   - i ω (ρ_i V_i + ρ_e V_e) v_n = 0
//!
//! (W_i/ρ_i + W_e/ρ_e) p
//!   - i ω (K'_i + K'_e) v_n = 0.
//! ```
//!
//! The implementation provides dense CPU constant-panel and higher-order
//! collocation building blocks. Discontinuous linear/quadratic and conforming
//! quadratic traces are supported; Galerkin and device paths remain separate
//! roadmap work, while the CPU kernel-native compressor supports panel-local
//! higher-order entries.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{LuFactorization, lu_factorize};
use sonium_solvers::iterative::GmresConfig;
use sonium_solvers::traits::{AdjointCapability, LinearOperator};
use sonium_xem::AcousticMedium;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use crate::core::assembly::common::get_element_coords;
use crate::core::assembly::operators::{
    BoundaryOperatorAssemblyError, BoundaryOperators, active_node_normals,
    assemble_boundary_operators, assemble_boundary_operators_wavenumber_derivative,
    assemble_boundary_operators_wavenumber_derivative_with_basis,
    assemble_boundary_operators_wavenumber_derivative_with_basis_shared_dofs,
    assemble_boundary_operators_with_basis, assemble_boundary_operators_with_basis_shared_dofs,
};
use crate::core::assembly::tbem::{
    TbemQuadraturePairError, TbemQuadraturePolicy, validate_tbem_quadrature_for_basis,
};
use crate::core::incident::IncidentField;
use crate::core::integration::regular::{
    regular_integration_linear, regular_integration_linear_wavenumber_derivative_with_normals,
    regular_integration_linear_with_normals,
};
use crate::core::integration::singular::{
    singular_integration_linear_curved, singular_integration_linear_curved_wavenumber_derivative,
    singular_integration_linear_wavenumber_derivative,
};
use crate::core::integration::{
    regular_integration, regular_integration_wavenumber_derivative, singular_integration,
    singular_integration_wavenumber_derivative,
};
use crate::core::types::{
    BasisOrder, Element, IntegrationResult, Mesh, PhysicsParams, PhysicsParamsError,
};
use crate::materials::{FluidInterface, MaterialError};

/// Failure while admitting or assembling a transmission interface.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TransmissionAssemblyError {
    /// The XEM media do not form a valid interface.
    #[error("invalid XEM transmission interface: {0}")]
    Interface(#[from] MaterialError),
    /// The exterior medium could not be converted to BEM physics parameters.
    #[error("invalid exterior BEM medium: {0}")]
    ExteriorPhysics(PhysicsParamsError),
    /// The interior medium could not be converted to BEM physics parameters.
    #[error("invalid interior BEM medium: {0}")]
    InteriorPhysics(PhysicsParamsError),
    /// Raw BEM media use different frequencies.
    #[error("transmission media must use the same frequency")]
    FrequencyMismatch,
    /// Raw BEM media use different Green-function signs.
    #[error("transmission media must use the same harmonic factor")]
    HarmonicMismatch,
    /// The interface has no active panels.
    #[error("transmission interface contains no active panels")]
    EmptyInterface,
    /// The current bounded implementation only accepts one constant DOF per
    /// active panel.
    #[error(
        "active panel {element} has {dofs} DOFs; transmission assembly requires one constant DOF"
    )]
    NonConstantPanel {
        /// Element index carrying the non-constant layout.
        element: usize,
        /// Number of DOFs found on the element.
        dofs: usize,
    },
    /// Basis-weighted boundary operators could not be assembled.
    #[error("transmission basis operator assembly failed: {0}")]
    OperatorAssembly(#[from] BoundaryOperatorAssemblyError),
    /// Higher-order quadrature admission failed for one medium.
    #[error("{medium} transmission quadrature admission failed: {error}")]
    Quadrature {
        /// Medium whose kernel estimate failed.
        medium: &'static str,
        /// Failing source/field pair and convergence estimate.
        error: TbemQuadraturePairError,
    },
    /// Continuous quadratic topology could not be constructed.
    #[error("invalid continuous quadratic transmission topology: {0}")]
    Topology(String),
    /// Active DOF addresses must be a dense zero-based panel ordering.
    #[error("active panel {element} has DOF {actual}, expected {expected}")]
    DofLayout {
        /// Element index carrying the invalid address.
        element: usize,
        /// Address found on the element.
        actual: usize,
        /// Address required by the dense block layout.
        expected: usize,
    },
}

/// Medium side whose material parameter is being differentiated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransmissionSide {
    /// Exterior/incident medium.
    Exterior,
    /// Interior/transmitted medium.
    Interior,
}

/// Material parameter exposed by the constant-panel transmission derivative
/// API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransmissionMaterialParameter {
    /// Differentiate through the acoustic kernel wave number while holding
    /// frequency, density, and geometry fixed.
    Wavenumber,
    /// Differentiate with respect to sound speed at fixed frequency and
    /// density, using `dk/dc = -k/c`.
    SpeedOfSound,
    /// Differentiate the explicit density factors in the Müller blocks while
    /// holding the acoustic kernel fixed.
    Density,
}

/// Failure while evaluating a transmission matrix material derivative.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TransmissionMaterialDerivativeError {
    /// A medium was modified or deserialized into an invalid state.
    #[error("{side:?} transmission medium is invalid: {source}")]
    InvalidMedium {
        /// Medium side that failed validation.
        side: TransmissionSide,
        /// Validation failure from the BEM physics contract.
        #[source]
        source: PhysicsParamsError,
    },
    /// A basis outside the supported constant/panel-local/conforming paths was
    /// requested. This is reserved for future basis-order extensions.
    #[error("transmission material derivatives do not support basis {basis:?}")]
    UnsupportedBasis {
        /// Basis carried by the assembled transmission system.
        basis: BasisOrder,
    },
    /// Analytic operator derivative assembly failed.
    #[error("transmission material derivative operator assembly failed: {0}")]
    OperatorAssembly(#[from] BoundaryOperatorAssemblyError),
    /// A topology-preserving higher-order reassembly failed while evaluating
    /// the bounded centered material derivative.
    #[error("transmission material derivative reassembly failed: {0}")]
    Reassembly(#[from] TransmissionAssemblyError),
    /// Compression of a material-derivative matrix failed after the
    /// derivative itself was evaluated.
    #[error("transmission material derivative compression failed: {0}")]
    Compression(#[from] TransmissionCompressionError),
    /// A perturbation rebuilt a matrix with a different two-trace dimension.
    #[error(
        "transmission material derivative dimensions differ: expected {expected}x{expected}, plus {plus_rows}x{plus_columns}, minus {minus_rows}x{minus_columns}"
    )]
    Dimension {
        /// Baseline total two-trace dimension.
        expected: usize,
        /// Plus perturbation row count.
        plus_rows: usize,
        /// Plus perturbation column count.
        plus_columns: usize,
        /// Minus perturbation row count.
        minus_rows: usize,
        /// Minus perturbation column count.
        minus_columns: usize,
    },
    /// The assembled derivative contains a non-finite entry.
    #[error("transmission material derivative contains non-finite value at ({row}, {column})")]
    NonFiniteDerivative {
        /// Matrix row containing the invalid derivative.
        row: usize,
        /// Matrix column containing the invalid derivative.
        column: usize,
    },
}

/// Failure while constructing an incident-field RHS for a transmission
/// system.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TransmissionRhsError {
    /// The incident field produced a value that cannot enter a finite solve.
    #[error("incident transmission trace contains non-finite value at index {index}")]
    NonFiniteTrace {
        /// Flat pressure-then-velocity trace index.
        index: usize,
    },
}

/// Dense constant-panel Müller transmission system.
#[derive(Debug, Clone)]
pub struct TransmissionTbemSystem {
    matrix: DenseMatrix<Complex64>,
    interface_dofs: usize,
    basis_order: BasisOrder,
    elements: Vec<Element>,
    nodes: Vec<[f64; 3]>,
    panel_centers: Vec<[f64; 3]>,
    panel_normals: Vec<[f64; 3]>,
    exterior_operators: crate::core::assembly::operators::BoundaryOperators,
    interior_operators: crate::core::assembly::operators::BoundaryOperators,
    /// Exterior/incident-side medium used by the first trace equation.
    pub exterior: PhysicsParams,
    /// Interior/transmitted-side medium used by the second trace equation.
    pub interior: PhysicsParams,
}

/// Failure while constructing a checked compressed transmission operator.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TransmissionCompressionError {
    /// The dense transmission matrix has no rows.
    #[error("transmission compression requires a non-empty system")]
    EmptySystem,
    /// A dense compression source is not a square two-trace matrix.
    #[error(
        "transmission dense compression matrix has shape {rows}x{columns} for {interface_dofs} interface DOFs"
    )]
    InvalidDenseMatrixDimension {
        /// Number of matrix rows.
        rows: usize,
        /// Number of matrix columns.
        columns: usize,
        /// Number of pressure/interface unknowns.
        interface_dofs: usize,
    },
    /// Block size must be positive.
    #[error("transmission compression block size must be positive")]
    InvalidBlockSize,
    /// ACA tolerance must be finite and positive.
    #[error("transmission compression tolerance must be finite and positive, got {0}")]
    InvalidTolerance(f64),
    /// ACA rank cap must be positive.
    #[error("transmission compression maximum rank must be positive")]
    InvalidRank,
    /// The source matrix contained a non-finite value.
    #[error("transmission matrix contains a non-finite value at ({row}, {column})")]
    NonFiniteMatrix {
        /// Row containing the rejected value.
        row: usize,
        /// Column containing the rejected value.
        column: usize,
    },
    /// ACA reduction produced a non-finite relative block error.
    #[error("transmission ACA block error is non-finite at block ({row_start}, {column_start})")]
    NonFiniteBlock {
        /// First global row in the rejected block.
        row_start: usize,
        /// First global column in the rejected block.
        column_start: usize,
    },
    /// An action vector has the wrong length.
    #[error("transmission {operation} input length {actual} does not match expected {expected}")]
    Dimension {
        /// Action being requested.
        operation: &'static str,
        /// Supplied vector length.
        actual: usize,
        /// Required vector length.
        expected: usize,
    },
    /// An action vector contains a non-finite complex value.
    #[error("transmission {operation} input entry {index} must be finite")]
    NonFiniteInput {
        /// Action being requested.
        operation: &'static str,
        /// Rejected vector entry.
        index: usize,
    },
    /// A checked action produced a non-finite complex value.
    #[error("transmission {operation} output entry {index} is non-finite")]
    NonFiniteOutput {
        /// Action being requested.
        operation: &'static str,
        /// Rejected output entry.
        index: usize,
    },
    /// A compressed solve received an invalid GMRES configuration.
    #[error("transmission compressed solve configuration is invalid: {0}")]
    InvalidSolveConfiguration(String),
    /// The near-field matrix supplied for ILU preconditioning has the wrong shape.
    #[error(
        "transmission near-field matrix shape {rows}x{columns} does not match expected {expected}x{expected}"
    )]
    NearFieldDimension {
        /// Number of near-field rows supplied.
        rows: usize,
        /// Number of near-field columns supplied.
        columns: usize,
        /// Required square operator dimension.
        expected: usize,
    },
    /// The compressed iterative solver failed before residual certification.
    #[error("transmission compressed solve failed: {0}")]
    SolverFailure(String),
    /// A certified iterate did not satisfy the requested source/action
    /// residual. Dense-backed constructors use the exact source; on-demand
    /// constructors use their stored compressed action.
    #[error("transmission certified residual {residual:.3e} exceeds tolerance {tolerance:.3e}")]
    SolveResidual {
        /// Certified relative residual.
        residual: f64,
        /// Maximum accepted relative residual.
        tolerance: f64,
    },
    /// A compressed solve produced a non-finite value or residual.
    #[error("transmission compressed solve produced a non-finite {quantity}")]
    NonFiniteSolve {
        /// Quantity that contained the invalid value.
        quantity: &'static str,
    },
    /// An on-demand entry source did not describe a two-trace system.
    #[error(
        "transmission compressed entry dimensions are invalid: interface DOFs {interface_dofs}, total DOFs {total_dofs}"
    )]
    InvalidSystemDimension {
        /// Number of pressure/interface unknowns.
        interface_dofs: usize,
        /// Total pressure-plus-velocity unknown count.
        total_dofs: usize,
    },
    /// Retained panel geometry no longer matches the dense interface layout.
    #[error("transmission kernel geometry layout invalid: {0}")]
    KernelLayout(String),
    /// A requested distributed row is outside the two-trace operator.
    #[error("transmission row index {row} outside 0..{expected}")]
    InvalidRow {
        /// Requested global output row.
        row: usize,
        /// Total number of output rows.
        expected: usize,
    },
}

#[derive(Clone)]
enum CompressedTransmissionBlock {
    Dense {
        row_start: usize,
        column_start: usize,
        matrix: DenseMatrix<Complex64>,
    },
    LowRank {
        row_start: usize,
        column_start: usize,
        u: DenseMatrix<Complex64>,
        v: DenseMatrix<Complex64>,
    },
}

/// Block-ACA compressed representation of a dense Müller transmission matrix.
///
/// The constructor intentionally takes an already assembled checked dense
/// system: it is a bounded CPU compression seam for transmission operators and
/// their exact transpose/Hermitian actions. Farther roadmap work can replace
/// the dense source with on-demand kernel entries without changing this
/// operator contract.
#[derive(Clone)]
pub struct CompressedTransmissionTbemSystem {
    interface_dofs: usize,
    total_dofs: usize,
    blocks: Vec<CompressedTransmissionBlock>,
    compressed_blocks: usize,
    dense_blocks: usize,
    maximum_block_relative_error: f64,
    dense_cache: Arc<OnceLock<DenseMatrix<Complex64>>>,
    factor_cache: Arc<OnceLock<LuFactorization<Complex64>>>,
    exact_matrix: Option<Arc<DenseMatrix<Complex64>>>,
}

/// Certified solution of a compressed transmission operator.
///
/// The values satisfy either `A x = rhs` or `Aᴴ x = rhs`, depending on the
/// solve entry point used. Dense-backed compression certifies the residual
/// against the retained exact source matrix; on-demand kernel compression
/// certifies against its compressed action because no source matrix exists.
/// A projected Krylov estimate is never exposed as the physical certificate.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedTransmissionSolution {
    values: Vec<Complex64>,
    relative_residual: f64,
    iterations: usize,
}

impl CompressedTransmissionSolution {
    /// Return the solved coefficient vector in pressure/velocity ordering.
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Return the certified relative residual. Dense-backed compression uses
    /// the retained exact source matrix; on-demand compression uses its
    /// stored compressed action because no source matrix is available.
    pub const fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Return the number of GMRES iterations used by the solve.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Evaluate a compressed forward transmission solution on exterior points.
    ///
    /// The dense transmission system supplies checked geometry, basis, medium,
    /// and incident-trace context; the coefficient vector itself came from the
    /// compressed solve.
    pub fn evaluate_exterior_field_checked(
        &self,
        system: &TransmissionTbemSystem,
        eval_points: &[[f64; 3]],
        incident: &IncidentField,
    ) -> Result<
        Vec<crate::core::postprocess::pressure::FieldPoint>,
        crate::core::solver::transmission::TransmissionSolveError,
    > {
        crate::core::solver::transmission::evaluate_exterior_field_values_checked(
            &self.values,
            system,
            eval_points,
            incident,
        )
    }

    /// Evaluate a compressed Hermitian-adjoint transmission pressure field.
    pub fn evaluate_exterior_adjoint_pressure_checked(
        &self,
        system: &TransmissionTbemSystem,
        eval_points: &[[f64; 3]],
    ) -> Result<Vec<Complex64>, crate::core::solver::transmission::TransmissionSolveError> {
        crate::core::solver::transmission::evaluate_exterior_adjoint_pressure_values_checked(
            &self.values,
            system,
            eval_points,
        )
    }
}

impl std::fmt::Debug for CompressedTransmissionTbemSystem {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompressedTransmissionTbemSystem")
            .field("interface_dofs", &self.interface_dofs)
            .field("total_dofs", &self.total_dofs)
            .field("compressed_blocks", &self.compressed_blocks)
            .field("dense_blocks", &self.dense_blocks)
            .field(
                "maximum_block_relative_error",
                &self.maximum_block_relative_error,
            )
            .finish()
    }
}

impl CompressedTransmissionTbemSystem {
    /// Attach a checked exact source operator to an on-demand compression.
    ///
    /// Kernel-native builders can use this after an adjacent-order admission
    /// assembled the fine matrix. The returned operator still evaluates
    /// actions through its ACA blocks, while compressed solves certify against
    /// the admitted source rather than only the approximate action.
    pub(crate) fn with_exact_matrix(mut self, matrix: DenseMatrix<Complex64>) -> Self {
        self.exact_matrix = Some(Arc::new(matrix));
        self
    }

    /// Compress a checked dense transmission matrix with deterministic blocks.
    ///
    /// Blocks intersecting the diagonal are retained densely as the near-field
    /// safety path. Off-diagonal blocks use partially pivoted ACA and fall back
    /// to dense storage if the requested rank cap cannot meet the tolerance.
    pub fn from_system(
        system: &TransmissionTbemSystem,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<Self, TransmissionCompressionError> {
        if block_size == 0 {
            return Err(TransmissionCompressionError::InvalidBlockSize);
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(TransmissionCompressionError::InvalidTolerance(tolerance));
        }
        if max_rank == 0 {
            return Err(TransmissionCompressionError::InvalidRank);
        }
        let matrix = system.matrix();
        let total_dofs = matrix.nrows();
        if total_dofs == 0 || matrix.ncols() != total_dofs {
            return Err(TransmissionCompressionError::EmptySystem);
        }
        for row in 0..total_dofs {
            for column in 0..total_dofs {
                let value = matrix[[row, column]];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(TransmissionCompressionError::NonFiniteMatrix { row, column });
                }
            }
        }

        let mut blocks = Vec::new();
        let mut compressed_blocks = 0;
        let mut dense_blocks = 0;
        let mut maximum_block_relative_error = 0.0_f64;
        let row_starts = (0..total_dofs).step_by(block_size);
        for row_start in row_starts {
            let row_len = block_size.min(total_dofs - row_start);
            for column_start in (0..total_dofs).step_by(block_size) {
                let column_len = block_size.min(total_dofs - column_start);
                let block = dense_block(matrix, row_start, row_len, column_start, column_len);
                let block_scale = block
                    .as_slice()
                    .iter()
                    .map(|value| value.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    .max(f64::MIN_POSITIVE);
                let diagonal_block =
                    row_start < column_start + column_len && column_start < row_start + row_len;
                if diagonal_block {
                    blocks.push(CompressedTransmissionBlock::Dense {
                        row_start,
                        column_start,
                        matrix: block,
                    });
                    dense_blocks += 1;
                    continue;
                }

                let (u, v, relative_error) = aca_block(&block, tolerance, max_rank);
                if !relative_error.is_finite() {
                    return Err(TransmissionCompressionError::NonFiniteBlock {
                        row_start,
                        column_start,
                    });
                }
                maximum_block_relative_error = maximum_block_relative_error.max(relative_error);
                if u.ncols() == 0
                    || u.ncols() >= row_len.min(column_len)
                    || relative_error > tolerance
                {
                    blocks.push(CompressedTransmissionBlock::Dense {
                        row_start,
                        column_start,
                        matrix: block,
                    });
                    dense_blocks += 1;
                } else {
                    debug_assert!(relative_error <= block_scale || relative_error.is_finite());
                    blocks.push(CompressedTransmissionBlock::LowRank {
                        row_start,
                        column_start,
                        u,
                        v,
                    });
                    compressed_blocks += 1;
                }
            }
        }

        Ok(Self {
            interface_dofs: system.interface_dofs(),
            total_dofs,
            blocks,
            compressed_blocks,
            dense_blocks,
            maximum_block_relative_error,
            dense_cache: Arc::new(OnceLock::new()),
            factor_cache: Arc::new(OnceLock::new()),
            exact_matrix: Some(Arc::new(matrix.clone())),
        })
    }

    /// Build the checked block-ACA representation from any dense two-trace
    /// transmission matrix.
    ///
    /// This is the bridge used by Galerkin Müller operators: the compression
    /// and action contract is shared with collocation transmission without
    /// pretending that a Galerkin matrix has collocation geometry metadata.
    pub fn from_dense_matrix(
        interface_dofs: usize,
        matrix: &DenseMatrix<Complex64>,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<Self, TransmissionCompressionError> {
        let total_dofs = matrix.nrows();
        if interface_dofs == 0
            || total_dofs == 0
            || interface_dofs.checked_mul(2) != Some(total_dofs)
            || matrix.ncols() != total_dofs
        {
            return Err(TransmissionCompressionError::InvalidDenseMatrixDimension {
                rows: matrix.nrows(),
                columns: matrix.ncols(),
                interface_dofs,
            });
        }
        if block_size == 0 {
            return Err(TransmissionCompressionError::InvalidBlockSize);
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(TransmissionCompressionError::InvalidTolerance(tolerance));
        }
        if max_rank == 0 {
            return Err(TransmissionCompressionError::InvalidRank);
        }

        for row in 0..total_dofs {
            for column in 0..total_dofs {
                let value = matrix[[row, column]];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(TransmissionCompressionError::NonFiniteMatrix { row, column });
                }
            }
        }

        let mut blocks = Vec::new();
        let mut compressed_blocks = 0;
        let mut dense_blocks = 0;
        let mut maximum_block_relative_error = 0.0_f64;
        for row_start in (0..total_dofs).step_by(block_size) {
            let row_len = block_size.min(total_dofs - row_start);
            for column_start in (0..total_dofs).step_by(block_size) {
                let column_len = block_size.min(total_dofs - column_start);
                let block = dense_block(matrix, row_start, row_len, column_start, column_len);
                let diagonal_block =
                    row_start < column_start + column_len && column_start < row_start + row_len;
                if diagonal_block {
                    blocks.push(CompressedTransmissionBlock::Dense {
                        row_start,
                        column_start,
                        matrix: block,
                    });
                    dense_blocks += 1;
                    continue;
                }

                let (u, v, relative_error) = aca_block(&block, tolerance, max_rank);
                if !relative_error.is_finite() {
                    return Err(TransmissionCompressionError::NonFiniteBlock {
                        row_start,
                        column_start,
                    });
                }
                maximum_block_relative_error = maximum_block_relative_error.max(relative_error);
                if u.ncols() == 0
                    || u.ncols() >= row_len.min(column_len)
                    || relative_error > tolerance
                    || !relative_error.is_finite()
                {
                    blocks.push(CompressedTransmissionBlock::Dense {
                        row_start,
                        column_start,
                        matrix: block,
                    });
                    dense_blocks += 1;
                } else {
                    blocks.push(CompressedTransmissionBlock::LowRank {
                        row_start,
                        column_start,
                        u,
                        v,
                    });
                    compressed_blocks += 1;
                }
            }
        }

        Ok(Self {
            interface_dofs,
            total_dofs,
            blocks,
            compressed_blocks,
            dense_blocks,
            maximum_block_relative_error,
            dense_cache: Arc::new(OnceLock::new()),
            factor_cache: Arc::new(OnceLock::new()),
            exact_matrix: Some(Arc::new(matrix.clone())),
        })
    }

    /// Build a compressed transmission operator from on-demand matrix entries.
    ///
    /// Unlike [`Self::from_system`], this constructor never requires the
    /// caller to materialize the full dense transmission matrix. The entry
    /// callback is evaluated once per block entry, diagonal blocks are kept
    /// dense, and off-diagonal blocks are compressed with the same ACA policy.
    /// This is the bounded CPU bridge for future kernel-level assembly paths.
    pub fn from_entry<F>(
        interface_dofs: usize,
        total_dofs: usize,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
        mut entry: F,
    ) -> Result<Self, TransmissionCompressionError>
    where
        F: FnMut(usize, usize) -> Complex64,
    {
        if block_size == 0 {
            return Err(TransmissionCompressionError::InvalidBlockSize);
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(TransmissionCompressionError::InvalidTolerance(tolerance));
        }
        if max_rank == 0 {
            return Err(TransmissionCompressionError::InvalidRank);
        }
        if interface_dofs == 0
            || total_dofs == 0
            || interface_dofs.checked_mul(2) != Some(total_dofs)
        {
            return Err(TransmissionCompressionError::InvalidSystemDimension {
                interface_dofs,
                total_dofs,
            });
        }

        let mut blocks = Vec::new();
        let mut compressed_blocks = 0;
        let mut dense_blocks = 0;
        let mut maximum_block_relative_error = 0.0_f64;
        for row_start in (0..total_dofs).step_by(block_size) {
            let row_len = block_size.min(total_dofs - row_start);
            for column_start in (0..total_dofs).step_by(block_size) {
                let column_len = block_size.min(total_dofs - column_start);
                let mut block = DenseMatrix::zeros(row_len, column_len);
                for row in 0..row_len {
                    for column in 0..column_len {
                        let value = entry(row_start + row, column_start + column);
                        if !value.re.is_finite() || !value.im.is_finite() {
                            return Err(TransmissionCompressionError::NonFiniteMatrix {
                                row: row_start + row,
                                column: column_start + column,
                            });
                        }
                        block[[row, column]] = value;
                    }
                }
                let diagonal_block =
                    row_start < column_start + column_len && column_start < row_start + row_len;
                if diagonal_block {
                    blocks.push(CompressedTransmissionBlock::Dense {
                        row_start,
                        column_start,
                        matrix: block,
                    });
                    dense_blocks += 1;
                    continue;
                }
                let (u, v, relative_error) = aca_block(&block, tolerance, max_rank);
                if !relative_error.is_finite() {
                    return Err(TransmissionCompressionError::NonFiniteBlock {
                        row_start,
                        column_start,
                    });
                }
                maximum_block_relative_error = maximum_block_relative_error.max(relative_error);
                if u.ncols() == 0
                    || u.ncols() >= row_len.min(column_len)
                    || relative_error > tolerance
                {
                    blocks.push(CompressedTransmissionBlock::Dense {
                        row_start,
                        column_start,
                        matrix: block,
                    });
                    dense_blocks += 1;
                } else {
                    blocks.push(CompressedTransmissionBlock::LowRank {
                        row_start,
                        column_start,
                        u,
                        v,
                    });
                    compressed_blocks += 1;
                }
            }
        }
        Ok(Self {
            interface_dofs,
            total_dofs,
            blocks,
            compressed_blocks,
            dense_blocks,
            maximum_block_relative_error,
            dense_cache: Arc::new(OnceLock::new()),
            factor_cache: Arc::new(OnceLock::new()),
            exact_matrix: None,
        })
    }

    /// Number of pressure (and velocity) interface unknowns.
    pub const fn interface_dofs(&self) -> usize {
        self.interface_dofs
    }

    /// Number of coupled pressure/velocity unknowns.
    pub const fn total_dofs(&self) -> usize {
        self.total_dofs
    }

    /// Number of blocks retained as low-rank factors.
    pub const fn compressed_block_count(&self) -> usize {
        self.compressed_blocks
    }

    /// Number of blocks retained densely as near-field or rank-cap fallback.
    pub const fn dense_block_count(&self) -> usize {
        self.dense_blocks
    }

    /// Largest relative Frobenius error among ACA blocks before any dense
    /// fallback. A zero value means no off-diagonal block was compressible.
    pub const fn maximum_block_relative_error(&self) -> f64 {
        self.maximum_block_relative_error
    }

    /// Apply the exact Hermitian action of the stored compressed operator.
    pub fn apply_hermitian(&self, input: &[Complex64]) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); self.total_dofs];
        self.apply_hermitian_into(input, &mut output);
        output
    }

    /// Apply the Hermitian action after validating the caller-owned vector.
    pub fn apply_hermitian_checked(
        &self,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>, TransmissionCompressionError> {
        validate_action_input(input, self.total_dofs, "Hermitian")?;
        let output = self.apply_hermitian(input);
        validate_action_output(&output, "Hermitian")?;
        Ok(output)
    }

    /// Apply the compressed Hermitian action to several right-hand sides.
    ///
    /// The input matrix stores one right-hand side per column. Dense and
    /// low-rank ACA tiles are traversed once per tile rather than routing
    /// through the generic column-by-column `LinearOperator` fallback.
    pub fn apply_hermitian_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, TransmissionCompressionError> {
        validate_block_action_input(input, self.total_dofs, "Hermitian block")?;
        let mut output = DenseMatrix::zeros(self.total_dofs, input.ncols());
        let operator = CompressedTransmissionHermitianOperator { system: self };
        <CompressedTransmissionHermitianOperator<'_> as LinearOperator<Complex64>>::apply_block(
            &operator,
            input,
            &mut output,
        );
        validate_block_action_output(&output, "Hermitian block")?;
        Ok(output)
    }

    /// Apply the exact transpose action of the stored compressed operator.
    pub fn apply_transpose(&self, input: &[Complex64]) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); self.total_dofs];
        self.apply_transpose_into(input, &mut output);
        output
    }

    /// Apply the transpose action after validating the caller-owned vector.
    pub fn apply_transpose_checked(
        &self,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>, TransmissionCompressionError> {
        validate_action_input(input, self.total_dofs, "transpose")?;
        let output = self.apply_transpose(input);
        validate_action_output(&output, "transpose")?;
        Ok(output)
    }

    /// Apply the forward action after validating the caller-owned vector.
    pub fn apply_checked(
        &self,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>, TransmissionCompressionError> {
        validate_action_input(input, self.total_dofs, "forward")?;
        let mut output = vec![Complex64::new(0.0, 0.0); self.total_dofs];
        <Self as LinearOperator<Complex64>>::apply_into(self, input, &mut output);
        validate_action_output(&output, "forward")?;
        Ok(output)
    }

    /// Apply the compressed forward operator only on caller-owned output rows.
    ///
    /// This is the local-action contract used by deterministic BEM row
    /// ownership plans. It performs no communication and does not implement a
    /// distributed solve; callers can exchange the returned row values through
    /// their chosen transport. Row order and duplicates are preserved.
    pub fn apply_rows_checked(
        &self,
        input: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, TransmissionCompressionError> {
        validate_action_input(input, self.total_dofs, "forward rows")?;
        validate_rows(rows, self.total_dofs)?;
        let output = rows
            .iter()
            .map(|&row| self.apply_single_row(row, input, false))
            .collect::<Vec<_>>();
        validate_action_output(&output, "forward rows")?;
        Ok(output)
    }

    /// Apply the exact transpose operator only on caller-owned output rows.
    ///
    /// This mirrors [`Self::apply_rows_checked`] for adjoint/transpose Krylov
    /// ownership. It is a local CPU action contract, not an MPI implementation.
    pub fn apply_transpose_rows_checked(
        &self,
        input: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, TransmissionCompressionError> {
        validate_action_input(input, self.total_dofs, "transpose rows")?;
        validate_rows(rows, self.total_dofs)?;
        let output = rows
            .iter()
            .map(|&row| self.apply_single_row(row, input, true))
            .collect::<Vec<_>>();
        validate_action_output(&output, "transpose rows")?;
        Ok(output)
    }

    /// Apply the compressed forward operator to caller-owned rows for several
    /// right-hand sides. Row order, including duplicates, is preserved.
    ///
    /// This is the local action primitive used by a future distributed
    /// transport layer; it performs no communication or distributed solve.
    pub fn apply_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, TransmissionCompressionError> {
        validate_block_action_input(input, self.total_dofs, "forward rows block")?;
        validate_rows(rows, self.total_dofs)?;
        let mut selected = DenseMatrix::zeros(rows.len(), input.ncols());
        self.apply_rows_block_into(input, rows, &mut selected, false);
        validate_block_action_output(&selected, "forward rows block")?;
        Ok(selected)
    }

    /// Apply the compressed transpose operator to caller-owned rows for
    /// several right-hand sides. Row order, including duplicates, is
    /// preserved. This remains a local CPU action contract, not MPI.
    pub fn apply_transpose_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, TransmissionCompressionError> {
        validate_block_action_input(input, self.total_dofs, "transpose rows block")?;
        validate_rows(rows, self.total_dofs)?;
        let mut selected = DenseMatrix::zeros(rows.len(), input.ncols());
        self.apply_rows_block_into(input, rows, &mut selected, true);
        validate_block_action_output(&selected, "transpose rows block")?;
        Ok(selected)
    }

    /// Apply the compressed Hermitian-adjoint operator on caller-owned rows
    /// for several right-hand sides.
    ///
    /// The returned rows preserve the caller's order, including duplicate
    /// ownership entries. This is a local CPU action primitive; it performs
    /// no MPI communication or distributed solve.
    pub fn apply_hermitian_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, TransmissionCompressionError> {
        validate_block_action_input(input, self.total_dofs, "Hermitian rows block")?;
        validate_rows(rows, self.total_dofs)?;
        let mut selected = DenseMatrix::zeros(rows.len(), input.ncols());
        self.apply_hermitian_rows_block_into(input, rows, &mut selected);
        validate_block_action_output(&selected, "Hermitian rows block")?;
        Ok(selected)
    }

    /// Apply the compressed Hermitian-adjoint operator on caller-owned rows
    /// for one right-hand side.
    ///
    /// Row order, including duplicate ownership entries, is preserved. This
    /// scalar wrapper shares the checked batched implementation so the
    /// distributed row contract has identical admission and tile semantics.
    pub fn apply_hermitian_rows_checked(
        &self,
        input: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, TransmissionCompressionError> {
        validate_action_input(input, self.total_dofs, "Hermitian rows")?;
        validate_rows(rows, self.total_dofs)?;
        let mut block_input = DenseMatrix::zeros(self.total_dofs, 1);
        for (row, value) in input.iter().copied().enumerate() {
            block_input[[row, 0]] = value;
        }
        Ok(self
            .apply_hermitian_rows_block_checked(&block_input, rows)?
            .column_to_vec(0))
    }

    fn apply_single_row(
        &self,
        requested_row: usize,
        input: &[Complex64],
        transpose: bool,
    ) -> Complex64 {
        let mut value = Complex64::new(0.0, 0.0);
        for block in &self.blocks {
            match block {
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } if !transpose
                    && (*row_start..*row_start + matrix.nrows()).contains(&requested_row) =>
                {
                    let local_row = requested_row - row_start;
                    for column in 0..matrix.ncols() {
                        value += matrix[[local_row, column]] * input[column_start + column];
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } if !transpose
                    && (*row_start..*row_start + u.nrows()).contains(&requested_row) =>
                {
                    let local_row = requested_row - row_start;
                    for rank in 0..u.ncols() {
                        let projection = (0..v.nrows())
                            .map(|column| v[[column, rank]] * input[column_start + column])
                            .sum::<Complex64>();
                        value += u[[local_row, rank]] * projection;
                    }
                }
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } if transpose
                    && (*column_start..*column_start + matrix.ncols()).contains(&requested_row) =>
                {
                    let local_column = requested_row - column_start;
                    for row in 0..matrix.nrows() {
                        value += matrix[[row, local_column]] * input[row_start + row];
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } if transpose
                    && (*column_start..*column_start + v.nrows()).contains(&requested_row) =>
                {
                    let local_column = requested_row - column_start;
                    for rank in 0..u.ncols() {
                        let projection = (0..u.nrows())
                            .map(|row| u[[row, rank]] * input[row_start + row])
                            .sum::<Complex64>();
                        value += v[[local_column, rank]] * projection;
                    }
                }
                _ => {}
            }
        }
        value
    }

    /// Solve the compressed forward system with checked ILU-preconditioned
    /// GMRES.
    pub fn solve_with_ilu(
        &self,
        rhs: &[Complex64],
        nearfield_matrix: &DenseMatrix<Complex64>,
        config: &GmresConfig<f64>,
        residual_tolerance: f64,
    ) -> Result<CompressedTransmissionSolution, TransmissionCompressionError> {
        validate_solve_inputs(
            rhs,
            self.total_dofs,
            nearfield_matrix,
            config,
            residual_tolerance,
        )?;
        let iterative = crate::core::solver::try_gmres_solve_with_ilu_operator(
            self,
            nearfield_matrix,
            rhs,
            config,
        )
        .map_err(|error| TransmissionCompressionError::SolverFailure(error.to_string()))
        .and_then(|solution| {
            let certified = certify_compressed_solution(
                self,
                &solution.x,
                rhs,
                residual_tolerance,
                solution.iterations,
            )?;
            match self.exact_matrix.as_deref() {
                Some(matrix) => certify_dense_matrix_solution(
                    matrix,
                    &solution.x,
                    rhs,
                    false,
                    residual_tolerance,
                    solution.iterations,
                ),
                None => Ok(certified),
            }
        });
        match iterative {
            Ok(solution) => Ok(solution),
            Err(error) => self.solve_direct_fallback(rhs, residual_tolerance, false, error),
        }
    }

    /// Solve a Hermitian-adjoint compressed system with checked ILU-GMRES.
    ///
    /// The near-field matrix must approximate `Aᴴ`, not `A`.
    pub fn solve_hermitian_with_ilu(
        &self,
        rhs: &[Complex64],
        nearfield_matrix: &DenseMatrix<Complex64>,
        config: &GmresConfig<f64>,
        residual_tolerance: f64,
    ) -> Result<CompressedTransmissionSolution, TransmissionCompressionError> {
        validate_solve_inputs(
            rhs,
            self.total_dofs,
            nearfield_matrix,
            config,
            residual_tolerance,
        )?;
        let operator = CompressedTransmissionHermitianOperator { system: self };
        let iterative = crate::core::solver::try_gmres_solve_with_ilu_operator(
            &operator,
            nearfield_matrix,
            rhs,
            config,
        )
        .map_err(|error| TransmissionCompressionError::SolverFailure(error.to_string()))
        .and_then(|solution| {
            let certified = certify_compressed_solution(
                &operator,
                &solution.x,
                rhs,
                residual_tolerance,
                solution.iterations,
            )?;
            match self.exact_matrix.as_deref() {
                Some(matrix) => certify_dense_matrix_solution(
                    matrix,
                    &solution.x,
                    rhs,
                    true,
                    residual_tolerance,
                    solution.iterations,
                ),
                None => Ok(certified),
            }
        });
        match iterative {
            Ok(solution) => Ok(solution),
            Err(error) => self.solve_direct_fallback(rhs, residual_tolerance, true, error),
        }
    }

    /// Solve several compressed forward right-hand sides with per-column
    /// residual certificates.
    pub fn solve_many_with_ilu(
        &self,
        rhs_columns: &[Vec<Complex64>],
        nearfield_matrix: &DenseMatrix<Complex64>,
        config: &GmresConfig<f64>,
        residual_tolerance: f64,
    ) -> Result<Vec<CompressedTransmissionSolution>, TransmissionCompressionError> {
        let mut solutions = Vec::with_capacity(rhs_columns.len());
        for rhs in rhs_columns {
            solutions.push(self.solve_with_ilu(
                rhs,
                nearfield_matrix,
                config,
                residual_tolerance,
            )?);
        }
        Ok(solutions)
    }

    /// Solve several compressed Hermitian-adjoint right-hand sides.
    pub fn solve_hermitian_many_with_ilu(
        &self,
        rhs_columns: &[Vec<Complex64>],
        nearfield_matrix: &DenseMatrix<Complex64>,
        config: &GmresConfig<f64>,
        residual_tolerance: f64,
    ) -> Result<Vec<CompressedTransmissionSolution>, TransmissionCompressionError> {
        let mut solutions = Vec::with_capacity(rhs_columns.len());
        for rhs in rhs_columns {
            solutions.push(self.solve_hermitian_with_ilu(
                rhs,
                nearfield_matrix,
                config,
                residual_tolerance,
            )?);
        }
        Ok(solutions)
    }

    fn cached_dense_matrix(&self) -> &DenseMatrix<Complex64> {
        self.dense_cache.get_or_init(|| {
            let mut matrix = DenseMatrix::zeros(self.total_dofs, self.total_dofs);
            for column in 0..self.total_dofs {
                let mut basis = vec![Complex64::new(0.0, 0.0); self.total_dofs];
                basis[column] = Complex64::new(1.0, 0.0);
                let action = self.apply(&basis);
                for row in 0..self.total_dofs {
                    matrix[[row, column]] = action[row];
                }
            }
            matrix
        })
    }

    /// Recover a certified solution for small systems when the caller's
    /// near-field ILU is numerically unsuitable.  This bounded fallback keeps
    /// the compressed API fail-closed for large systems while preserving a
    /// useful deterministic path for compact QA/design problems.
    fn solve_direct_fallback(
        &self,
        rhs: &[Complex64],
        residual_tolerance: f64,
        hermitian: bool,
        iterative_error: TransmissionCompressionError,
    ) -> Result<CompressedTransmissionSolution, TransmissionCompressionError> {
        const MAX_DIRECT_FALLBACK_DOFS: usize = 1_024;
        if self.total_dofs > MAX_DIRECT_FALLBACK_DOFS {
            return Err(iterative_error);
        }
        let source_matrix = self
            .exact_matrix
            .as_deref()
            .unwrap_or_else(|| self.cached_dense_matrix());
        let factors = match self.factor_cache.get() {
            Some(factors) => factors,
            None => {
                let forward_factor =
                    lu_factorize(source_matrix).map_err(|_| iterative_error.clone())?;
                let _ = self.factor_cache.set(forward_factor);
                self.factor_cache.get().expect("factor cache initialized")
            }
        };
        let values = if hermitian {
            factors
                .solve_conj_transpose(rhs)
                .map_err(|_| iterative_error.clone())?
        } else {
            factors.solve(rhs).map_err(|_| iterative_error.clone())?
        };
        if let Some(matrix) = self.exact_matrix.as_deref() {
            certify_dense_matrix_solution(matrix, &values, rhs, hermitian, residual_tolerance, 1)
        } else if hermitian {
            let operator = CompressedTransmissionHermitianOperator { system: self };
            certify_compressed_solution(&operator, &values, rhs, residual_tolerance, 1)
        } else {
            certify_compressed_solution(self, &values, rhs, residual_tolerance, 1)
        }
    }

    fn apply_hermitian_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            input.len(),
            self.total_dofs,
            "transmission Hermitian input mismatch"
        );
        assert_eq!(
            output.len(),
            self.total_dofs,
            "transmission Hermitian output mismatch"
        );
        output.fill(Complex64::new(0.0, 0.0));
        for block in &self.blocks {
            match block {
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } => {
                    for row in 0..matrix.nrows() {
                        for column in 0..matrix.ncols() {
                            output[column_start + column] +=
                                matrix[[row, column]].conj() * input[row_start + row];
                        }
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } => {
                    for rank in 0..u.ncols() {
                        let projection = (0..u.nrows())
                            .map(|row| u[[row, rank]].conj() * input[row_start + row])
                            .sum::<Complex64>();
                        for column in 0..v.nrows() {
                            output[column_start + column] += v[[column, rank]].conj() * projection;
                        }
                    }
                }
            }
        }
    }

    fn apply_hermitian_rows_block_into(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
        output: &mut DenseMatrix<Complex64>,
    ) {
        assert_eq!(
            input.nrows(),
            self.total_dofs,
            "transmission Hermitian rows block input rows"
        );
        assert_eq!(
            output.nrows(),
            rows.len(),
            "transmission Hermitian rows block output rows"
        );
        assert_eq!(
            input.ncols(),
            output.ncols(),
            "transmission Hermitian rows block column count"
        );
        output.fill(Complex64::new(0.0, 0.0));

        for block in &self.blocks {
            match block {
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } => {
                    for (selected_row, &global_row) in rows.iter().enumerate() {
                        let Some(local_column) = global_row
                            .checked_sub(*column_start)
                            .filter(|&local| local < matrix.ncols())
                        else {
                            continue;
                        };
                        for rhs in 0..input.ncols() {
                            for row in 0..matrix.nrows() {
                                output[[selected_row, rhs]] += matrix[[row, local_column]].conj()
                                    * input[[row_start + row, rhs]];
                            }
                        }
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } => {
                    let rank = u.ncols();
                    let rhs_count = input.ncols();
                    let mut projections = vec![Complex64::new(0.0, 0.0); rank * rhs_count];
                    for rhs in 0..rhs_count {
                        for factor in 0..rank {
                            projections[factor * rhs_count + rhs] = (0..u.nrows())
                                .map(|row| u[[row, factor]].conj() * input[[row_start + row, rhs]])
                                .sum();
                        }
                    }
                    for (selected_row, &global_row) in rows.iter().enumerate() {
                        let Some(local_column) = global_row
                            .checked_sub(*column_start)
                            .filter(|&local| local < v.nrows())
                        else {
                            continue;
                        };
                        for rhs in 0..rhs_count {
                            for factor in 0..rank {
                                output[[selected_row, rhs]] += v[[local_column, factor]].conj()
                                    * projections[factor * rhs_count + rhs];
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_rows_block_into(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
        output: &mut DenseMatrix<Complex64>,
        transpose: bool,
    ) {
        assert_eq!(
            input.nrows(),
            self.total_dofs,
            "transmission rows block input rows"
        );
        assert_eq!(
            output.nrows(),
            rows.len(),
            "transmission rows block output rows"
        );
        assert_eq!(
            input.ncols(),
            output.ncols(),
            "transmission rows block column count"
        );
        output.fill(Complex64::new(0.0, 0.0));

        for block in &self.blocks {
            match block {
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } => {
                    if !transpose {
                        for (selected_row, &global_row) in rows.iter().enumerate() {
                            let Some(local_row) = global_row
                                .checked_sub(*row_start)
                                .filter(|&local| local < matrix.nrows())
                            else {
                                continue;
                            };
                            for rhs in 0..input.ncols() {
                                for column in 0..matrix.ncols() {
                                    output[[selected_row, rhs]] += matrix[[local_row, column]]
                                        * input[[column_start + column, rhs]];
                                }
                            }
                        }
                    } else {
                        for (selected_row, &global_row) in rows.iter().enumerate() {
                            let Some(local_column) = global_row
                                .checked_sub(*column_start)
                                .filter(|&local| local < matrix.ncols())
                            else {
                                continue;
                            };
                            for rhs in 0..input.ncols() {
                                for row in 0..matrix.nrows() {
                                    output[[selected_row, rhs]] +=
                                        matrix[[row, local_column]] * input[[row_start + row, rhs]];
                                }
                            }
                        }
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } => {
                    let selected_start = if transpose { *column_start } else { *row_start };
                    let selected_len = if transpose { v.nrows() } else { u.nrows() };
                    if !rows.iter().any(|&global_row| {
                        global_row
                            .checked_sub(selected_start)
                            .is_some_and(|local| local < selected_len)
                    }) {
                        continue;
                    }
                    let rank = u.ncols();
                    let rhs_count = input.ncols();
                    let mut projections = vec![Complex64::new(0.0, 0.0); rank * rhs_count];

                    if !transpose {
                        for rhs in 0..rhs_count {
                            for factor in 0..rank {
                                projections[factor * rhs_count + rhs] = (0..v.nrows())
                                    .map(|column| {
                                        v[[column, factor]] * input[[column_start + column, rhs]]
                                    })
                                    .sum();
                            }
                        }
                        for (selected_row, &global_row) in rows.iter().enumerate() {
                            let Some(local_row) = global_row
                                .checked_sub(*row_start)
                                .filter(|&local| local < u.nrows())
                            else {
                                continue;
                            };
                            for rhs in 0..rhs_count {
                                for factor in 0..rank {
                                    output[[selected_row, rhs]] += u[[local_row, factor]]
                                        * projections[factor * rhs_count + rhs];
                                }
                            }
                        }
                    } else {
                        for rhs in 0..rhs_count {
                            for factor in 0..rank {
                                projections[factor * rhs_count + rhs] = (0..u.nrows())
                                    .map(|row| u[[row, factor]] * input[[row_start + row, rhs]])
                                    .sum();
                            }
                        }
                        for (selected_row, &global_row) in rows.iter().enumerate() {
                            let Some(local_column) = global_row
                                .checked_sub(*column_start)
                                .filter(|&local| local < v.nrows())
                            else {
                                continue;
                            };
                            for rhs in 0..rhs_count {
                                for factor in 0..rank {
                                    output[[selected_row, rhs]] += v[[local_column, factor]]
                                        * projections[factor * rhs_count + rhs];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_hermitian_block_into(
        &self,
        input: &DenseMatrix<Complex64>,
        output: &mut DenseMatrix<Complex64>,
    ) {
        assert_eq!(
            input.nrows(),
            self.total_dofs,
            "transmission Hermitian block input rows"
        );
        assert_eq!(
            output.nrows(),
            self.total_dofs,
            "transmission Hermitian block output rows"
        );
        assert_eq!(
            input.ncols(),
            output.ncols(),
            "transmission Hermitian block column count"
        );
        output.fill(Complex64::new(0.0, 0.0));
        for block in &self.blocks {
            match block {
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } => {
                    for rhs in 0..input.ncols() {
                        for row in 0..matrix.nrows() {
                            let input_value = input[[row_start + row, rhs]];
                            for column in 0..matrix.ncols() {
                                output[[column_start + column, rhs]] +=
                                    matrix[[row, column]].conj() * input_value;
                            }
                        }
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } => {
                    for rhs in 0..input.ncols() {
                        for rank in 0..u.ncols() {
                            let projection = (0..u.nrows())
                                .map(|row| u[[row, rank]].conj() * input[[row_start + row, rhs]])
                                .sum::<Complex64>();
                            for column in 0..v.nrows() {
                                output[[column_start + column, rhs]] +=
                                    v[[column, rank]].conj() * projection;
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_transpose_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(
            input.len(),
            self.total_dofs,
            "transmission transpose input mismatch"
        );
        assert_eq!(
            output.len(),
            self.total_dofs,
            "transmission transpose output mismatch"
        );
        output.fill(Complex64::new(0.0, 0.0));
        for block in &self.blocks {
            match block {
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } => {
                    for row in 0..matrix.nrows() {
                        for column in 0..matrix.ncols() {
                            output[column_start + column] +=
                                matrix[[row, column]] * input[row_start + row];
                        }
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } => {
                    for rank in 0..u.ncols() {
                        let projection = (0..u.nrows())
                            .map(|row| u[[row, rank]] * input[row_start + row])
                            .sum::<Complex64>();
                        for column in 0..v.nrows() {
                            output[column_start + column] += v[[column, rank]] * projection;
                        }
                    }
                }
            }
        }
    }
}

fn validate_rows(rows: &[usize], expected: usize) -> Result<(), TransmissionCompressionError> {
    if let Some(&row) = rows.iter().find(|&&row| row >= expected) {
        return Err(TransmissionCompressionError::InvalidRow { row, expected });
    }
    Ok(())
}

fn validate_action_input(
    input: &[Complex64],
    expected: usize,
    operation: &'static str,
) -> Result<(), TransmissionCompressionError> {
    if input.len() != expected {
        return Err(TransmissionCompressionError::Dimension {
            operation,
            actual: input.len(),
            expected,
        });
    }
    if let Some(index) = input
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TransmissionCompressionError::NonFiniteInput { operation, index });
    }
    Ok(())
}

fn validate_action_output(
    output: &[Complex64],
    operation: &'static str,
) -> Result<(), TransmissionCompressionError> {
    if let Some(index) = output
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TransmissionCompressionError::NonFiniteOutput { operation, index });
    }
    Ok(())
}

fn validate_block_action_output(
    output: &DenseMatrix<Complex64>,
    operation: &'static str,
) -> Result<(), TransmissionCompressionError> {
    if let Some(index) = output
        .as_slice()
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TransmissionCompressionError::NonFiniteOutput { operation, index });
    }
    Ok(())
}

fn validate_block_action_input(
    input: &DenseMatrix<Complex64>,
    expected: usize,
    operation: &'static str,
) -> Result<(), TransmissionCompressionError> {
    if input.nrows() != expected {
        return Err(TransmissionCompressionError::Dimension {
            operation,
            actual: input.nrows(),
            expected,
        });
    }
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            let value = input[[row, column]];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(TransmissionCompressionError::NonFiniteInput {
                    operation,
                    index: column * input.nrows() + row,
                });
            }
        }
    }
    Ok(())
}

struct CompressedTransmissionHermitianOperator<'a> {
    system: &'a CompressedTransmissionTbemSystem,
}

impl LinearOperator<Complex64> for CompressedTransmissionHermitianOperator<'_> {
    fn num_rows(&self) -> usize {
        self.system.total_dofs
    }

    fn num_cols(&self) -> usize {
        self.system.total_dofs
    }

    fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        self.system.apply_hermitian_into(input, output);
    }

    fn apply_block(&self, input: &DenseMatrix<Complex64>, output: &mut DenseMatrix<Complex64>) {
        self.system.apply_hermitian_block_into(input, output);
    }

    fn apply_transpose_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        self.system.apply_into(input, output);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

impl LinearOperator<Complex64> for CompressedTransmissionTbemSystem {
    fn num_rows(&self) -> usize {
        self.total_dofs
    }

    fn num_cols(&self) -> usize {
        self.total_dofs
    }

    fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(input.len(), self.total_dofs, "transmission input mismatch");
        assert_eq!(
            output.len(),
            self.total_dofs,
            "transmission output mismatch"
        );
        output.fill(Complex64::new(0.0, 0.0));
        for block in &self.blocks {
            match block {
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } => {
                    for row in 0..matrix.nrows() {
                        output[row_start + row] += (0..matrix.ncols())
                            .map(|column| matrix[[row, column]] * input[column_start + column])
                            .sum::<Complex64>();
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } => {
                    for rank in 0..u.ncols() {
                        let projection = (0..v.nrows())
                            .map(|column| v[[column, rank]] * input[column_start + column])
                            .sum::<Complex64>();
                        for row in 0..u.nrows() {
                            output[row_start + row] += u[[row, rank]] * projection;
                        }
                    }
                }
            }
        }
    }

    fn apply_block(&self, input: &DenseMatrix<Complex64>, output: &mut DenseMatrix<Complex64>) {
        assert_eq!(
            input.nrows(),
            self.total_dofs,
            "transmission block input rows"
        );
        assert_eq!(
            output.nrows(),
            self.total_dofs,
            "transmission block output rows"
        );
        assert_eq!(
            input.ncols(),
            output.ncols(),
            "transmission block column count"
        );
        output.fill(Complex64::new(0.0, 0.0));

        for block in &self.blocks {
            match block {
                CompressedTransmissionBlock::Dense {
                    row_start,
                    column_start,
                    matrix,
                } => {
                    for rhs in 0..input.ncols() {
                        for row in 0..matrix.nrows() {
                            let action = (0..matrix.ncols())
                                .map(|column| {
                                    matrix[[row, column]] * input[[column_start + column, rhs]]
                                })
                                .sum::<Complex64>();
                            output[[row_start + row, rhs]] += action;
                        }
                    }
                }
                CompressedTransmissionBlock::LowRank {
                    row_start,
                    column_start,
                    u,
                    v,
                } => {
                    for rhs in 0..input.ncols() {
                        for rank in 0..u.ncols() {
                            let projection = (0..v.nrows())
                                .map(|column| {
                                    v[[column, rank]] * input[[column_start + column, rhs]]
                                })
                                .sum::<Complex64>();
                            for row in 0..u.nrows() {
                                output[[row_start + row, rhs]] += u[[row, rank]] * projection;
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_transpose_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        CompressedTransmissionTbemSystem::apply_transpose_into(self, input, output);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn validate_solve_inputs(
    rhs: &[Complex64],
    expected: usize,
    nearfield_matrix: &DenseMatrix<Complex64>,
    config: &GmresConfig<f64>,
    residual_tolerance: f64,
) -> Result<(), TransmissionCompressionError> {
    validate_action_input(rhs, expected, "solve RHS")?;
    if nearfield_matrix.nrows() != expected || nearfield_matrix.ncols() != expected {
        return Err(TransmissionCompressionError::NearFieldDimension {
            rows: nearfield_matrix.nrows(),
            columns: nearfield_matrix.ncols(),
            expected,
        });
    }
    for row in 0..expected {
        for column in 0..expected {
            let value = nearfield_matrix[[row, column]];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(TransmissionCompressionError::NonFiniteMatrix { row, column });
            }
        }
    }
    if config.max_iterations == 0
        || config.restart == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
        || !residual_tolerance.is_finite()
        || residual_tolerance <= 0.0
    {
        return Err(TransmissionCompressionError::InvalidSolveConfiguration(
            "GMRES iterations, restart, and tolerances must be finite and positive".to_string(),
        ));
    }
    Ok(())
}

/// Certify a solution against a materialized source matrix rather than the
/// possibly rank-truncated action retained by the compressor.
fn certify_dense_matrix_solution(
    matrix: &DenseMatrix<Complex64>,
    values: &[Complex64],
    rhs: &[Complex64],
    hermitian: bool,
    residual_tolerance: f64,
    iterations: usize,
) -> Result<CompressedTransmissionSolution, TransmissionCompressionError> {
    if values.len() != matrix.ncols() || rhs.len() != matrix.nrows() {
        return Err(TransmissionCompressionError::SolverFailure(format!(
            "dense residual dimensions are values={} rhs={} matrix={}x{}",
            values.len(),
            rhs.len(),
            matrix.nrows(),
            matrix.ncols(),
        )));
    }
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TransmissionCompressionError::NonFiniteSolve {
            quantity: "solution",
        });
    }

    let mut action = vec![Complex64::default(); matrix.nrows()];
    if hermitian {
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                action[column] += matrix[[row, column]].conj() * values[row];
            }
        }
    } else {
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                action[row] += matrix[[row, column]] * values[column];
            }
        }
    }
    if action
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TransmissionCompressionError::NonFiniteSolve {
            quantity: "exact operator action",
        });
    }

    let (residual_squared, rhs_squared) = action.iter().zip(rhs).fold(
        (0.0, 0.0),
        |(residual_squared, rhs_squared), (action_value, rhs_value)| {
            (
                residual_squared + (*action_value - *rhs_value).norm_sqr(),
                rhs_squared + rhs_value.norm_sqr(),
            )
        },
    );
    let residual = residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE);
    if !residual.is_finite() {
        return Err(TransmissionCompressionError::NonFiniteSolve {
            quantity: "residual",
        });
    }
    if residual > residual_tolerance {
        return Err(TransmissionCompressionError::SolveResidual {
            residual,
            tolerance: residual_tolerance,
        });
    }
    Ok(CompressedTransmissionSolution {
        values: values.to_vec(),
        relative_residual: residual,
        iterations,
    })
}

fn certify_compressed_solution<O: LinearOperator<Complex64>>(
    operator: &O,
    values: &[Complex64],
    rhs: &[Complex64],
    residual_tolerance: f64,
    iterations: usize,
) -> Result<CompressedTransmissionSolution, TransmissionCompressionError> {
    if values.len() != operator.num_cols() {
        return Err(TransmissionCompressionError::SolverFailure(format!(
            "solver returned {} values, expected {}",
            values.len(),
            operator.num_cols()
        )));
    }
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TransmissionCompressionError::NonFiniteSolve {
            quantity: "solution",
        });
    }
    let mut action = vec![Complex64::default(); operator.num_rows()];
    operator.apply_into(values, &mut action);
    if action
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TransmissionCompressionError::NonFiniteSolve {
            quantity: "operator action",
        });
    }
    let mut residual_squared = 0.0;
    let mut rhs_squared = 0.0;
    for (action_value, rhs_value) in action.iter().zip(rhs) {
        residual_squared += (*action_value - *rhs_value).norm_sqr();
        rhs_squared += rhs_value.norm_sqr();
    }
    let residual = residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE);
    if !residual.is_finite() {
        return Err(TransmissionCompressionError::NonFiniteSolve {
            quantity: "residual",
        });
    }
    if residual > residual_tolerance {
        return Err(TransmissionCompressionError::SolveResidual {
            residual,
            tolerance: residual_tolerance,
        });
    }
    Ok(CompressedTransmissionSolution {
        values: values.to_vec(),
        relative_residual: residual,
        iterations,
    })
}

fn dense_block(
    matrix: &DenseMatrix<Complex64>,
    row_start: usize,
    row_len: usize,
    column_start: usize,
    column_len: usize,
) -> DenseMatrix<Complex64> {
    let mut block = DenseMatrix::zeros(row_len, column_len);
    for row in 0..row_len {
        for column in 0..column_len {
            block[[row, column]] = matrix[[row_start + row, column_start + column]];
        }
    }
    block
}

fn aca_block(
    block: &DenseMatrix<Complex64>,
    tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, f64) {
    let rows = block.nrows();
    let columns = block.ncols();
    let initial_norm = block
        .as_slice()
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    if initial_norm <= f64::MIN_POSITIVE {
        return (
            DenseMatrix::zeros(rows, 0),
            DenseMatrix::zeros(columns, 0),
            0.0,
        );
    }
    let mut residual = block.clone();
    let mut u_columns: Vec<Vec<Complex64>> = Vec::new();
    let mut v_columns: Vec<Vec<Complex64>> = Vec::new();
    for _ in 0..max_rank.min(rows).min(columns) {
        let mut pivot_row = 0;
        let mut pivot_column = 0;
        let mut pivot = Complex64::new(0.0, 0.0);
        for row in 0..rows {
            for column in 0..columns {
                let candidate = residual[[row, column]];
                if candidate.norm() > pivot.norm() {
                    pivot_row = row;
                    pivot_column = column;
                    pivot = candidate;
                }
            }
        }
        if pivot.norm() <= tolerance * initial_norm {
            break;
        }
        let u: Vec<Complex64> = (0..rows)
            .map(|row| residual[[row, pivot_column]] / pivot)
            .collect();
        let v: Vec<Complex64> = (0..columns)
            .map(|column| residual[[pivot_row, column]])
            .collect();
        for row in 0..rows {
            for column in 0..columns {
                residual[[row, column]] -= u[row] * v[column];
            }
        }
        u_columns.push(u);
        v_columns.push(v);
        let residual_norm = residual
            .as_slice()
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if residual_norm <= tolerance * initial_norm {
            break;
        }
    }
    let rank = u_columns.len();
    let mut u = DenseMatrix::zeros(rows, rank);
    let mut v = DenseMatrix::zeros(columns, rank);
    for (column, values) in u_columns.into_iter().enumerate() {
        for (row, value) in values.into_iter().enumerate() {
            u[[row, column]] = value;
        }
    }
    for (column, values) in v_columns.into_iter().enumerate() {
        for (row, value) in values.into_iter().enumerate() {
            v[[row, column]] = value;
        }
    }
    let residual_norm = residual
        .as_slice()
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    (u, v, residual_norm / initial_norm)
}

impl TransmissionTbemSystem {
    /// Number of pressure unknowns (and also normal-velocity unknowns).
    pub const fn interface_dofs(&self) -> usize {
        self.interface_dofs
    }

    /// Total number of coupled unknowns, `2 * interface_dofs()`.
    pub const fn total_dofs(&self) -> usize {
        self.interface_dofs * 2
    }
    /// Panel basis used for the interface pressure/velocity traces.
    pub const fn basis_order(&self) -> BasisOrder {
        self.basis_order
    }

    /// Return the dense pressure/velocity block matrix.
    ///
    /// The first block of rows/columns is pressure; the second is outward
    /// normal velocity.
    pub fn matrix(&self) -> &DenseMatrix<Complex64> {
        &self.matrix
    }

    /// Differentiate the dense Müller transmission matrix with respect to one
    /// material parameter on one medium side.
    ///
    /// This bounded analytic path supports constant-panel, panel-local, and
    /// conforming shared-DOF higher-order collocation systems.  Rows and
    /// columns retain the assembled topology while only acoustic kernels are
    /// differentiated.
    /// Wavenumber derivatives hold the interface topology and quadrature
    /// schedule fixed.  Speed derivatives compose that result with
    /// `dk/dc = -k/c` at fixed frequency, while density derivatives apply the
    /// explicit `ρ` and `1/ρ` factors in the two Müller block equations.
    pub fn matrix_material_derivative(
        &self,
        side: TransmissionSide,
        parameter: TransmissionMaterialParameter,
    ) -> Result<DenseMatrix<Complex64>, TransmissionMaterialDerivativeError> {
        self.exterior.validate().map_err(|source| {
            TransmissionMaterialDerivativeError::InvalidMedium {
                side: TransmissionSide::Exterior,
                source,
            }
        })?;
        self.interior.validate().map_err(|source| {
            TransmissionMaterialDerivativeError::InvalidMedium {
                side: TransmissionSide::Interior,
                source,
            }
        })?;

        let selected = match side {
            TransmissionSide::Exterior => &self.exterior,
            TransmissionSide::Interior => &self.interior,
        };
        let i_omega = Complex64::new(0.0, self.exterior.harmonic_factor * self.exterior.omega);
        let n = self.interface_dofs;
        let mut derivative = DenseMatrix::zeros(2 * n, 2 * n);

        match parameter {
            TransmissionMaterialParameter::Wavenumber
            | TransmissionMaterialParameter::SpeedOfSound => {
                let scale = if parameter == TransmissionMaterialParameter::SpeedOfSound {
                    -selected.wave_number / selected.speed_of_sound
                } else {
                    1.0
                };
                let operators = if self.basis_order == BasisOrder::Constant {
                    assemble_boundary_operators_wavenumber_derivative(
                        &self.elements,
                        &self.nodes,
                        selected,
                    )?
                } else if self.has_shared_higher_order_dofs() {
                    assemble_boundary_operators_wavenumber_derivative_with_basis_shared_dofs(
                        &self.elements,
                        &self.nodes,
                        selected,
                        self.basis_order,
                    )?
                } else {
                    assemble_boundary_operators_wavenumber_derivative_with_basis(
                        &self.elements,
                        &self.nodes,
                        selected,
                        self.basis_order,
                    )?
                };
                self.fill_operator_material_derivative(&mut derivative, side, &operators, scale);
            }
            TransmissionMaterialParameter::Density => {
                self.fill_density_material_derivative(&mut derivative, side, i_omega);
            }
        }

        for row in 0..2 * n {
            for column in 0..2 * n {
                let value = derivative[[row, column]];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(TransmissionMaterialDerivativeError::NonFiniteDerivative {
                        row,
                        column,
                    });
                }
            }
        }
        Ok(derivative)
    }

    /// Build a checked compressed action for one transmission material
    /// derivative.
    ///
    /// The material derivative is first evaluated by
    /// [`Self::matrix_material_derivative`], preserving the assembled
    /// pressure/normal-velocity topology, and the resulting matrix is then
    /// passed through the same deterministic block-ACA compressor as the
    /// value operator. This closes the compressed transmission action
    /// contract for constant and higher-order systems on CPU. It is
    /// intentionally a bounded dense-to-compressed bridge: it does not claim
    /// a kernel-native or device/distributed derivative assembly path.
    pub fn compress_material_derivative(
        &self,
        side: TransmissionSide,
        parameter: TransmissionMaterialParameter,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedTransmissionTbemSystem, TransmissionMaterialDerivativeError> {
        let derivative = self.matrix_material_derivative(side, parameter)?;
        CompressedTransmissionTbemSystem::from_dense_matrix(
            self.interface_dofs,
            &derivative,
            block_size,
            tolerance,
            max_rank,
        )
        .map_err(TransmissionMaterialDerivativeError::from)
    }

    /// Diagnostic topology-preserving centered derivative retained for
    /// comparison against the production analytic higher-order path.
    #[allow(dead_code)]
    fn matrix_material_derivative_higher_order(
        &self,
        side: TransmissionSide,
        parameter: TransmissionMaterialParameter,
    ) -> Result<DenseMatrix<Complex64>, TransmissionMaterialDerivativeError> {
        let selected = match side {
            TransmissionSide::Exterior => &self.exterior,
            TransmissionSide::Interior => &self.interior,
        };
        let (plus_value, minus_value, step) = match parameter {
            TransmissionMaterialParameter::Wavenumber => {
                let step = (selected.wave_number.abs() * 1.0e-5).max(1.0e-7);
                let step = step.min(selected.wave_number * 0.1);
                (
                    selected.wave_number + step,
                    selected.wave_number - step,
                    step,
                )
            }
            TransmissionMaterialParameter::SpeedOfSound => {
                let step = (selected.speed_of_sound.abs() * 1.0e-5).max(1.0e-5);
                (
                    selected.speed_of_sound + step,
                    selected.speed_of_sound - step,
                    step,
                )
            }
            TransmissionMaterialParameter::Density => {
                let step = (selected.density.abs() * 1.0e-5).max(1.0e-5);
                (selected.density + step, selected.density - step, step)
            }
        };

        let plus = self.rebuild_material_perturbation(side, parameter, plus_value)?;
        let minus = self.rebuild_material_perturbation(side, parameter, minus_value)?;
        let expected = self.total_dofs();
        let plus_shape = plus.matrix.shape();
        let minus_shape = minus.matrix.shape();
        if plus_shape != (expected, expected) || minus_shape != (expected, expected) {
            return Err(TransmissionMaterialDerivativeError::Dimension {
                expected,
                plus_rows: plus_shape.0,
                plus_columns: plus_shape.1,
                minus_rows: minus_shape.0,
                minus_columns: minus_shape.1,
            });
        }

        let mut derivative = DenseMatrix::zeros(expected, expected);
        for row in 0..expected {
            for column in 0..expected {
                derivative[[row, column]] =
                    (plus.matrix[[row, column]] - minus.matrix[[row, column]]) / (2.0 * step);
                let value = derivative[[row, column]];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(TransmissionMaterialDerivativeError::NonFiniteDerivative {
                        row,
                        column,
                    });
                }
            }
        }
        Ok(derivative)
    }

    #[allow(dead_code)]
    fn rebuild_material_perturbation(
        &self,
        side: TransmissionSide,
        parameter: TransmissionMaterialParameter,
        value: f64,
    ) -> Result<Self, TransmissionMaterialDerivativeError> {
        let mut exterior = self.exterior.clone();
        let mut interior = self.interior.clone();
        let selected = match side {
            TransmissionSide::Exterior => &mut exterior,
            TransmissionSide::Interior => &mut interior,
        };
        match parameter {
            TransmissionMaterialParameter::Wavenumber => {
                if !value.is_finite() || value <= 0.0 {
                    return Err(TransmissionMaterialDerivativeError::NonFiniteDerivative {
                        row: 0,
                        column: 0,
                    });
                }
                selected.speed_of_sound = selected.omega / value;
            }
            TransmissionMaterialParameter::SpeedOfSound => {
                selected.speed_of_sound = value;
            }
            TransmissionMaterialParameter::Density => {
                selected.density = value;
            }
        }
        selected.wave_number = selected.omega / selected.speed_of_sound;
        selected.wave_length = selected.speed_of_sound / selected.frequency;
        selected.pressure_factor = selected.density * selected.omega * selected.harmonic_factor;

        if self.basis_order == BasisOrder::Quadratic && self.has_shared_higher_order_dofs() {
            return build_transmission_tbem_system_with_continuous_quadratic(
                &self.elements,
                &self.nodes,
                exterior,
                interior,
            )
            .map_err(TransmissionMaterialDerivativeError::Reassembly);
        }
        build_transmission_tbem_system_with_basis(
            &self.elements,
            &self.nodes,
            exterior,
            interior,
            self.basis_order,
        )
        .map_err(TransmissionMaterialDerivativeError::Reassembly)
    }

    fn has_shared_higher_order_dofs(&self) -> bool {
        let mut seen = HashSet::new();
        self.elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .flat_map(|element| element.dof_addresses.iter().copied())
            .any(|dof| !seen.insert(dof))
    }

    fn fill_operator_material_derivative(
        &self,
        derivative: &mut DenseMatrix<Complex64>,
        side: TransmissionSide,
        operators: &BoundaryOperators,
        scale: f64,
    ) {
        let n = self.interface_dofs;
        let i_omega = Complex64::new(0.0, self.exterior.harmonic_factor * self.exterior.omega);
        let density = match side {
            TransmissionSide::Exterior => self.exterior.density,
            TransmissionSide::Interior => self.interior.density,
        };
        for row in 0..n {
            for column in 0..n {
                derivative[[row, column]] = operators.double_layer[[row, column]] * scale;
                derivative[[row, n + column]] =
                    -i_omega * density * operators.single_layer[[row, column]] * scale;
                derivative[[n + row, column]] =
                    operators.hypersingular[[row, column]] * (scale / density);
                derivative[[n + row, n + column]] =
                    -i_omega * operators.adjoint_double_layer[[row, column]] * scale;
            }
        }
    }

    fn fill_density_material_derivative(
        &self,
        derivative: &mut DenseMatrix<Complex64>,
        side: TransmissionSide,
        i_omega: Complex64,
    ) {
        let n = self.interface_dofs;
        let operators = match side {
            TransmissionSide::Exterior => &self.exterior_operators,
            TransmissionSide::Interior => &self.interior_operators,
        };
        let density = match side {
            TransmissionSide::Exterior => self.exterior.density,
            TransmissionSide::Interior => self.interior.density,
        };
        for row in 0..n {
            for column in 0..n {
                derivative[[row, n + column]] = -i_omega * operators.single_layer[[row, column]];
                derivative[[n + row, column]] =
                    -operators.hypersingular[[row, column]] / (density * density);
            }
        }
    }

    /// Build a checked block-ACA representation of this transmission system.
    pub fn compress(
        &self,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedTransmissionTbemSystem, TransmissionCompressionError> {
        CompressedTransmissionTbemSystem::from_system(self, block_size, tolerance, max_rank)
    }

    /// Build a compressed representation by evaluating transmission entries
    /// directly from panel geometry and the two medium kernels.
    ///
    /// [`Self::compress`] compresses the already assembled dense matrix. This
    /// path keeps the same block-ACA/action contract but obtains each entry on
    /// demand from the production panel quadrature, so callers can validate a
    /// kernel-native source without materialising a second dense matrix. The
    /// Constant and panel-local or conforming shared-DOF linear/quadratic
    /// collocation kernels are supported. On-demand entries accumulate every
    /// source/field panel incidence for a global row/column, matching the
    /// checked dense assembly contract.
    pub fn compress_kernel_native(
        &self,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedTransmissionTbemSystem, TransmissionCompressionError> {
        let n = self.interface_dofs;
        let dof_layout = kernel_panel_dof_layout(&self.elements, n, self.basis_order)
            .map_err(TransmissionCompressionError::KernelLayout)?;

        let panel_coords: Vec<Option<DenseMatrix<f64>>> = self
            .elements
            .iter()
            .map(|element| {
                (!element.property.is_evaluation())
                    .then(|| get_element_coords(element, &self.nodes))
            })
            .collect();
        let basis_order = self.basis_order;
        let shared_dofs = self.has_shared_higher_order_dofs();
        let nodes = &self.nodes;
        let exterior = self.exterior.clone();
        let interior = self.interior.clone();
        let elements = &self.elements;
        let entry = move |row: usize, column: usize| {
            let row_block = usize::from(row >= n);
            let column_block = usize::from(column >= n);
            let source_dof = row % n;
            let field_dof = column % n;
            let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
            let mut value = Complex64::new(0.0, 0.0);
            let &(source_index, source_local) = dof_layout[source_dof]
                .first()
                .expect("validated source row owner");
            let source = &elements[source_index];
            for &(field_index, field_local) in &dof_layout[field_dof] {
                let field = &elements[field_index];
                let field_coords = panel_coords[field_index]
                    .as_ref()
                    .expect("validated active field panel");
                let exterior_pair = kernel_transmission_panel_pair_with_basis(
                    source,
                    field,
                    field_coords,
                    nodes,
                    &exterior,
                    source_index == field_index,
                    basis_order,
                    source_local,
                    field_local,
                    shared_dofs,
                );
                let interior_pair = kernel_transmission_panel_pair_with_basis(
                    source,
                    field,
                    field_coords,
                    nodes,
                    &interior,
                    source_index == field_index,
                    basis_order,
                    source_local,
                    field_local,
                    shared_dofs,
                );
                value += match (row_block, column_block) {
                    (0, 0) => exterior_pair.dg_dn_integral + interior_pair.dg_dn_integral,
                    (0, 1) => {
                        -i_omega
                            * (exterior.density * exterior_pair.g_integral
                                + interior.density * interior_pair.g_integral)
                    }
                    (1, 0) => {
                        exterior_pair.d2g_dnxdny_integral / exterior.density
                            + interior_pair.d2g_dnxdny_integral / interior.density
                    }
                    (1, 1) => {
                        -i_omega * (exterior_pair.dg_dnx_integral + interior_pair.dg_dnx_integral)
                    }
                    _ => unreachable!("two-trace block indices are binary"),
                };
            }
            value
        };

        let mut compressed = CompressedTransmissionTbemSystem::from_entry(
            n,
            2 * n,
            block_size,
            tolerance,
            max_rank,
            entry,
        )?;

        // `self` already owns the checked dense source matrix used to build
        // the production transmission system.  Retain that source for the
        // kernel-native representation as well, so compressed GMRES/ILU
        // solutions are certified against the exact assembled operator (and
        // can fail closed on a rank-truncated ACA action) rather than merely
        // against the approximate action that drove Krylov.
        compressed.exact_matrix = Some(Arc::new(self.matrix.clone()));
        Ok(compressed)
    }

    /// Build a compressed transmission material-derivative action directly
    /// from the constant or panel-local higher-order kernel quadrature.
    ///
    /// Unlike [`Self::compress_material_derivative`], this path does not
    /// materialise the derivative matrix before block ACA compression. The
    /// wavenumber and speed-of-sound variants reuse the production regular
    /// and singular kernel derivative integrators; density differentiates
    /// only the explicit Müller block factors. This is intentionally limited
    /// to the panel-local/shared-DOF kernel-native representation; repeated
    /// source/field incidences are accumulated before ACA sees the entry.
    pub fn compress_kernel_native_material_derivative(
        &self,
        side: TransmissionSide,
        parameter: TransmissionMaterialParameter,
        block_size: usize,
        tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedTransmissionTbemSystem, TransmissionMaterialDerivativeError> {
        self.exterior.validate().map_err(|source| {
            TransmissionMaterialDerivativeError::InvalidMedium {
                side: TransmissionSide::Exterior,
                source,
            }
        })?;
        self.interior.validate().map_err(|source| {
            TransmissionMaterialDerivativeError::InvalidMedium {
                side: TransmissionSide::Interior,
                source,
            }
        })?;
        let n = self.interface_dofs;
        let dof_layout =
            kernel_panel_dof_layout(&self.elements, n, self.basis_order).map_err(|message| {
                TransmissionMaterialDerivativeError::Compression(
                    TransmissionCompressionError::KernelLayout(message),
                )
            })?;

        let panel_coords: Vec<Option<DenseMatrix<f64>>> = self
            .elements
            .iter()
            .map(|element| {
                (!element.property.is_evaluation())
                    .then(|| get_element_coords(element, &self.nodes))
            })
            .collect();
        let basis_order = self.basis_order;
        let shared_dofs = self.has_shared_higher_order_dofs();
        let nodes = &self.nodes;
        let exterior = self.exterior.clone();
        let interior = self.interior.clone();
        let elements = &self.elements;
        let selected = match side {
            TransmissionSide::Exterior => exterior.clone(),
            TransmissionSide::Interior => interior.clone(),
        };
        let scale = match parameter {
            TransmissionMaterialParameter::SpeedOfSound => {
                -selected.wave_number / selected.speed_of_sound
            }
            TransmissionMaterialParameter::Wavenumber | TransmissionMaterialParameter::Density => {
                1.0
            }
        };
        let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);

        let entry = move |row: usize, column: usize| {
            let row_block = row / n;
            let row_panel = row % n;
            let column_block = column / n;
            let column_panel = column % n;
            let mut value = Complex64::new(0.0, 0.0);
            let &(source_index, source_local) = dof_layout[row_panel]
                .first()
                .expect("validated source row owner");
            let source = &elements[source_index];
            for &(field_index, field_local) in &dof_layout[column_panel] {
                let field = &elements[field_index];
                let field_coords = panel_coords[field_index]
                    .as_ref()
                    .expect("validated active field panel");
                let self_pair = source_index == field_index;

                let exterior_pair = if side == TransmissionSide::Exterior
                    && parameter != TransmissionMaterialParameter::Density
                {
                    kernel_transmission_panel_pair_wavenumber_derivative_with_basis(
                        source,
                        field,
                        field_coords,
                        nodes,
                        &exterior,
                        self_pair,
                        basis_order,
                        source_local,
                        field_local,
                        shared_dofs,
                    )
                } else {
                    kernel_transmission_panel_pair_with_basis(
                        source,
                        field,
                        field_coords,
                        nodes,
                        &exterior,
                        self_pair,
                        basis_order,
                        source_local,
                        field_local,
                        shared_dofs,
                    )
                };
                let interior_pair = if side == TransmissionSide::Interior
                    && parameter != TransmissionMaterialParameter::Density
                {
                    kernel_transmission_panel_pair_wavenumber_derivative_with_basis(
                        source,
                        field,
                        field_coords,
                        nodes,
                        &interior,
                        self_pair,
                        basis_order,
                        source_local,
                        field_local,
                        shared_dofs,
                    )
                } else {
                    kernel_transmission_panel_pair_with_basis(
                        source,
                        field,
                        field_coords,
                        nodes,
                        &interior,
                        self_pair,
                        basis_order,
                        source_local,
                        field_local,
                        shared_dofs,
                    )
                };

                let pair = match side {
                    TransmissionSide::Exterior => exterior_pair,
                    TransmissionSide::Interior => interior_pair,
                };
                value += match parameter {
                    TransmissionMaterialParameter::Wavenumber
                    | TransmissionMaterialParameter::SpeedOfSound => {
                        let density = selected.density;
                        match (row_block, column_block) {
                            (0, 0) => pair.dg_dn_integral * scale,
                            (0, 1) => -i_omega * density * pair.g_integral * scale,
                            (1, 0) => pair.d2g_dnxdny_integral * (scale / density),
                            (1, 1) => -i_omega * pair.dg_dnx_integral * scale,
                            _ => unreachable!("two-trace block indices are binary"),
                        }
                    }
                    TransmissionMaterialParameter::Density => {
                        let density = selected.density;
                        match (row_block, column_block) {
                            (0, 0) | (1, 1) => Complex64::new(0.0, 0.0),
                            (0, 1) => -i_omega * pair.g_integral,
                            (1, 0) => -pair.d2g_dnxdny_integral / (density * density),
                            _ => unreachable!("two-trace block indices are binary"),
                        }
                    }
                };
            }
            value
        };

        CompressedTransmissionTbemSystem::from_entry(
            n,
            2 * n,
            block_size,
            tolerance,
            max_rank,
            entry,
        )
        .map_err(TransmissionMaterialDerivativeError::Compression)
    }

    /// Geometry retained for checked exterior field postprocessing.
    pub(crate) fn field_geometry(&self) -> (&[Element], &[[f64; 3]]) {
        (&self.elements, &self.nodes)
    }

    /// Active-panel centers used for incident trace construction.
    pub(crate) fn panel_centers(&self) -> &[[f64; 3]] {
        &self.panel_centers
    }

    /// Active-panel outward normals used for incident trace construction.
    pub(crate) fn panel_normals(&self) -> &[[f64; 3]] {
        &self.panel_normals
    }

    /// Build the RHS induced by an incident field on the exterior side.
    ///
    /// The returned vector follows the two-block Müller convention used by
    /// [`super::super::solver::transmission::solve_transmission_system`]. It
    /// is twice the exterior Calderón action on the incident pressure and
    /// normal-velocity traces. The factor and sign follow from subtracting
    /// the exterior and interior trace equations with the outward normal
    /// directed into the incident medium; the unknown remains the interface
    /// trace. This
    /// is intentionally an explicit constant-panel CPU boundary, rather than
    /// an implicit fallback in the generic solver.
    pub fn incident_rhs(
        &self,
        incident: &IncidentField,
    ) -> Result<Vec<Complex64>, TransmissionRhsError> {
        let pressure = incident.evaluate_pressure(&self.panel_centers, &self.exterior);
        let normal_derivative = incident.evaluate_normal_derivative(
            &self.panel_centers,
            &self.panel_normals,
            &self.exterior,
        );
        let i_omega_rho = Complex64::new(0.0, self.exterior.pressure_factor);
        let normal_velocity: Vec<Complex64> = normal_derivative
            .into_iter()
            .map(|value| {
                if i_omega_rho.norm() > 1.0e-30 {
                    value / i_omega_rho
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        for (index, value) in pressure.iter().chain(normal_velocity.iter()).enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(TransmissionRhsError::NonFiniteTrace { index });
            }
        }

        let n = self.interface_dofs;
        let mut rhs = vec![Complex64::new(0.0, 0.0); 2 * n];
        let i_omega = Complex64::new(0.0, self.exterior.harmonic_factor * self.exterior.omega);
        for row in 0..n {
            let mut pressure_action = Complex64::new(0.0, 0.0);
            let mut velocity_action = Complex64::new(0.0, 0.0);
            let mut adjoint_pressure_action = Complex64::new(0.0, 0.0);
            let mut adjoint_velocity_action = Complex64::new(0.0, 0.0);
            for column in 0..n {
                pressure_action +=
                    self.exterior_operators.double_layer[[row, column]] * pressure[column];
                velocity_action +=
                    self.exterior_operators.single_layer[[row, column]] * normal_velocity[column];
                adjoint_pressure_action +=
                    self.exterior_operators.hypersingular[[row, column]] * pressure[column];
                adjoint_velocity_action += self.exterior_operators.adjoint_double_layer
                    [[row, column]]
                    * normal_velocity[column];
            }
            rhs[row] = 2.0 * (pressure_action - i_omega * self.exterior.density * velocity_action);
            rhs[n + row] = 2.0
                * (adjoint_pressure_action / self.exterior.density
                    - i_omega * adjoint_velocity_action);
        }
        Ok(rhs)
    }
}

fn active_panel_layout(elements: &[Element]) -> Result<Vec<usize>, TransmissionAssemblyError> {
    let mut active = Vec::new();
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        if element.dof_addresses.len() != 1 {
            return Err(TransmissionAssemblyError::NonConstantPanel {
                element: element_index,
                dofs: element.dof_addresses.len(),
            });
        }
        let expected = active.len();
        let actual = element.dof_addresses[0];
        if actual != expected {
            return Err(TransmissionAssemblyError::DofLayout {
                element: element_index,
                actual,
                expected,
            });
        }
        active.push(element_index);
    }
    if active.is_empty() {
        return Err(TransmissionAssemblyError::EmptyInterface);
    }
    Ok(active)
}

/// Build the panel/local incidence map required by on-demand collocation
/// entries. Every global DOF retains all panel incidences so conforming
/// higher-order entries accumulate exactly like dense assembly.
fn kernel_panel_dof_layout(
    elements: &[Element],
    interface_dofs: usize,
    basis: BasisOrder,
) -> Result<Vec<Vec<(usize, usize)>>, String> {
    let mut layout = vec![Vec::new(); interface_dofs];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        let expected = element.num_dofs_for_basis(basis);
        if element.dof_addresses.len() != expected {
            return Err(format!(
                "active element {element_index} has {} DOFs, expected {expected}",
                element.dof_addresses.len()
            ));
        }
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            if dof >= interface_dofs {
                return Err(format!(
                    "active element {element_index} DOF {dof} exceeds {interface_dofs}"
                ));
            }
            layout[dof].push((element_index, local));
        }
    }
    if let Some(dof) = layout.iter().position(Vec::is_empty) {
        return Err(format!("missing active DOF {dof}"));
    }
    Ok(layout)
}

/// Evaluate one constant-panel boundary-operator interaction using the same
/// production quadrature choices as the dense assembler.
fn kernel_transmission_panel_pair(
    source: &Element,
    field: &Element,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    self_pair: bool,
) -> IntegrationResult {
    if self_pair {
        singular_integration(
            &source.center,
            &source.normal,
            field_coords,
            field.element_type,
            physics,
            None,
            0,
            false,
        )
    } else {
        regular_integration(
            &source.center,
            &source.normal,
            field_coords,
            field.element_type,
            field.area,
            physics,
            None,
            0,
            false,
        )
    }
}

/// Evaluate one panel-local basis interaction at the requested collocation
/// points. Constant panels retain the legacy center/area path.
fn kernel_transmission_panel_pair_with_basis(
    source: &Element,
    field: &Element,
    field_coords: &DenseMatrix<f64>,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    self_pair: bool,
    basis: BasisOrder,
    source_local: usize,
    field_local: usize,
    shared_dofs: bool,
) -> IntegrationResult {
    if basis == BasisOrder::Constant {
        return kernel_transmission_panel_pair(source, field, field_coords, physics, self_pair);
    }
    let source_points = source.collocation_points(basis, nodes);
    // Match the dense panel-local assembler: collocation geometry may be
    // curved, but the BEM trace normal is the element normal for every row.
    let source_normal = if shared_dofs {
        source
            .collocation_normals(basis, nodes)
            .get(source_local)
            .copied()
            .unwrap_or(source.normal)
    } else {
        source.normal
    };
    let source_point = source_points[source_local];
    let num_basis = field.num_dofs_for_basis(basis);
    let result = if self_pair {
        if shared_dofs {
            singular_integration_linear_curved(
                &source_point,
                &source_normal,
                field_coords,
                field.element_type,
                physics,
                num_basis,
                active_node_normals(field),
            )
        } else {
            crate::core::integration::singular::singular_integration_linear(
                &source_point,
                &source_normal,
                field_coords,
                field.element_type,
                physics,
                num_basis,
            )
        }
    } else if shared_dofs {
        regular_integration_linear_with_normals(
            &source_point,
            &source_normal,
            field_coords,
            field.element_type,
            field.area,
            physics,
            num_basis,
            active_node_normals(field),
        )
    } else {
        regular_integration_linear(
            &source_point,
            &source_normal,
            field_coords,
            field.element_type,
            field.area,
            physics,
            num_basis,
        )
    };
    result.basis_results[field_local].clone()
}

/// Evaluate one constant-panel transmission interaction differentiated with
/// respect to the medium wavenumber, using the same quadrature schedule as
/// [`kernel_transmission_panel_pair`].
fn kernel_transmission_panel_pair_wavenumber_derivative(
    source: &Element,
    field: &Element,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    self_pair: bool,
) -> IntegrationResult {
    if self_pair {
        singular_integration_wavenumber_derivative(
            &source.center,
            &source.normal,
            field_coords,
            field.element_type,
            physics,
            None,
            0,
            false,
            None,
        )
    } else {
        regular_integration_wavenumber_derivative(
            &source.center,
            &source.normal,
            field_coords,
            field.element_type,
            field.area,
            physics,
            None,
            0,
            false,
            None,
        )
    }
}

/// Wavenumber derivative of one panel-local basis interaction.
fn kernel_transmission_panel_pair_wavenumber_derivative_with_basis(
    source: &Element,
    field: &Element,
    field_coords: &DenseMatrix<f64>,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    self_pair: bool,
    basis: BasisOrder,
    source_local: usize,
    field_local: usize,
    shared_dofs: bool,
) -> IntegrationResult {
    if basis == BasisOrder::Constant {
        return kernel_transmission_panel_pair_wavenumber_derivative(
            source,
            field,
            field_coords,
            physics,
            self_pair,
        );
    }
    let source_points = source.collocation_points(basis, nodes);
    let source_normal = if shared_dofs {
        source
            .collocation_normals(basis, nodes)
            .get(source_local)
            .copied()
            .unwrap_or(source.normal)
    } else {
        source.normal
    };
    let source_point = source_points[source_local];
    let num_basis = field.num_dofs_for_basis(basis);
    let result = if self_pair {
        if shared_dofs {
            singular_integration_linear_curved_wavenumber_derivative(
                &source_point,
                &source_normal,
                field_coords,
                field.element_type,
                physics,
                num_basis,
                active_node_normals(field),
            )
        } else {
            singular_integration_linear_wavenumber_derivative(
                &source_point,
                &source_normal,
                field_coords,
                field.element_type,
                physics,
                num_basis,
            )
        }
    } else {
        regular_integration_linear_wavenumber_derivative_with_normals(
            &source_point,
            &source_normal,
            field_coords,
            field.element_type,
            field.area,
            physics,
            num_basis,
            shared_dofs.then(|| active_node_normals(field)).flatten(),
        )
    };
    result.basis_results[field_local].clone()
}

fn assemble_transmission_from_physics(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: PhysicsParams,
    interior: PhysicsParams,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    let active = active_panel_layout(elements)?;
    exterior
        .validate()
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    interior
        .validate()
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(TransmissionAssemblyError::FrequencyMismatch);
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(TransmissionAssemblyError::HarmonicMismatch);
    }

    let exterior_operators = assemble_boundary_operators(elements, nodes, &exterior);
    let interior_operators = assemble_boundary_operators(elements, nodes, &interior);
    let n = active.len();
    let mut matrix = DenseMatrix::zeros(2 * n, 2 * n);
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let i_omega_rho_exterior = i_omega * exterior.density;
    let i_omega_rho_interior = i_omega * interior.density;

    for row in 0..n {
        for col in 0..n {
            matrix[[row, col]] = interior_operators.double_layer[[row, col]]
                + exterior_operators.double_layer[[row, col]];
            matrix[[row, n + col]] = -(i_omega_rho_interior
                * interior_operators.single_layer[[row, col]]
                + i_omega_rho_exterior * exterior_operators.single_layer[[row, col]]);
            matrix[[n + row, col]] = interior_operators.hypersingular[[row, col]]
                / interior.density
                + exterior_operators.hypersingular[[row, col]] / exterior.density;
            matrix[[n + row, n + col]] = -i_omega
                * (interior_operators.adjoint_double_layer[[row, col]]
                    + exterior_operators.adjoint_double_layer[[row, col]]);
        }
    }

    Ok(TransmissionTbemSystem {
        matrix,
        interface_dofs: n,
        basis_order: BasisOrder::Constant,
        elements: elements.to_vec(),
        nodes: nodes.to_vec(),
        panel_centers: active.iter().map(|&index| elements[index].center).collect(),
        panel_normals: active.iter().map(|&index| elements[index].normal).collect(),
        exterior_operators,
        interior_operators,
        exterior,
        interior,
    })
}

/// Assemble a dense constant-panel Müller transmission operator from BEM
/// physics parameters.
pub fn build_transmission_tbem_system(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: PhysicsParams,
    interior: PhysicsParams,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    assemble_transmission_from_physics(elements, nodes, exterior, interior)
}

/// Admit typed XEM media and assemble their dense constant-panel Müller
/// transmission operator.
///
/// The `incident` XEM medium is treated as the exterior side and the
/// `transmitted` medium as the interior side.  The mesh normals must point
/// from the transmitted/interior region toward the incident/exterior region.
/// Assemble a dense Müller transmission operator for discontinuous linear or
/// quadratic panel traces.
pub fn build_transmission_tbem_system_with_basis(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: PhysicsParams,
    interior: PhysicsParams,
    basis: BasisOrder,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    if basis == BasisOrder::Constant {
        return build_transmission_tbem_system(elements, nodes, exterior, interior);
    }
    let mut owned = elements.to_vec();
    let mut dof_counter = 0usize;
    let mut active = Vec::new();
    for (element_index, element) in owned.iter_mut().enumerate() {
        if element.property.is_evaluation() {
            element.dof_addresses.clear();
            continue;
        }
        let num_dofs = element.num_dofs_for_basis(basis);
        element.dof_addresses = (dof_counter..dof_counter + num_dofs).collect();
        dof_counter += num_dofs;
        active.push(element_index);
    }
    if active.is_empty() {
        return Err(TransmissionAssemblyError::EmptyInterface);
    }
    exterior
        .validate()
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    interior
        .validate()
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(TransmissionAssemblyError::FrequencyMismatch);
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(TransmissionAssemblyError::HarmonicMismatch);
    }
    let exterior_operators =
        assemble_boundary_operators_with_basis(&owned, nodes, &exterior, basis)?;
    let interior_operators =
        assemble_boundary_operators_with_basis(&owned, nodes, &interior, basis)?;
    let n = exterior_operators.single_layer.nrows();
    let mut matrix = DenseMatrix::zeros(2 * n, 2 * n);
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let i_omega_rho_exterior = i_omega * exterior.density;
    let i_omega_rho_interior = i_omega * interior.density;
    for row in 0..n {
        for col in 0..n {
            matrix[[row, col]] = interior_operators.double_layer[[row, col]]
                + exterior_operators.double_layer[[row, col]];
            matrix[[row, n + col]] = -(i_omega_rho_interior
                * interior_operators.single_layer[[row, col]]
                + i_omega_rho_exterior * exterior_operators.single_layer[[row, col]]);
            matrix[[n + row, col]] = interior_operators.hypersingular[[row, col]]
                / interior.density
                + exterior_operators.hypersingular[[row, col]] / exterior.density;
            matrix[[n + row, n + col]] = -i_omega
                * (interior_operators.adjoint_double_layer[[row, col]]
                    + exterior_operators.adjoint_double_layer[[row, col]]);
        }
    }
    let mut panel_centers = Vec::with_capacity(n);
    let mut panel_normals = Vec::with_capacity(n);
    for &element_index in &active {
        let element = &owned[element_index];
        panel_centers.extend(element.collocation_points(basis, nodes));
        panel_normals.extend(std::iter::repeat_n(
            element.normal,
            element.dof_addresses.len(),
        ));
    }
    Ok(TransmissionTbemSystem {
        matrix,
        interface_dofs: n,
        basis_order: basis,
        elements: owned,
        nodes: nodes.to_vec(),
        panel_centers,
        panel_normals,
        exterior_operators,
        interior_operators,
        exterior,
        interior,
    })
}

/// Assemble panel-local linear/quadratic transmission after checked
/// adjacent-order quadrature admission for both media.
pub fn build_transmission_tbem_system_with_basis_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: PhysicsParams,
    interior: PhysicsParams,
    basis: BasisOrder,
    policy: TbemQuadraturePolicy,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    exterior
        .validate()
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    interior
        .validate()
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    validate_tbem_quadrature_for_basis(elements, nodes, &exterior, basis, &policy).map_err(
        |error| TransmissionAssemblyError::Quadrature {
            medium: "exterior",
            error,
        },
    )?;
    validate_tbem_quadrature_for_basis(elements, nodes, &interior, basis, &policy).map_err(
        |error| TransmissionAssemblyError::Quadrature {
            medium: "interior",
            error,
        },
    )?;
    build_transmission_tbem_system_with_basis(elements, nodes, exterior, interior, basis)
}

/// Typed-XEM adapter for checked panel-local higher-order transmission.
pub fn build_transmission_tbem_system_with_basis_checked_from_xem(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident: AcousticMedium,
    transmitted: AcousticMedium,
    basis: BasisOrder,
    policy: TbemQuadraturePolicy,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    FluidInterface::try_from_xem(incident, transmitted)?;
    let exterior = PhysicsParams::try_new_medium(incident, false)
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    let interior = PhysicsParams::try_new_medium(transmitted, true)
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    build_transmission_tbem_system_with_basis_checked(
        elements, nodes, exterior, interior, basis, policy,
    )
}

/// Assemble a conforming quadratic Müller transmission system.
///
/// Vertex and edge-midpoint trace DOFs are shared across adjacent Tri3/Quad4
/// panels using the mesh topology map; Quad4 center DOFs remain panel-local.
/// This is a CPU collocation path (not a Galerkin formulation) and is kept
/// separate from the historical discontinuous panel-local constructor.
pub fn build_transmission_tbem_system_with_continuous_quadratic(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: PhysicsParams,
    interior: PhysicsParams,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    let mesh = Mesh {
        nodes: nodes.to_vec(),
        elements: elements.to_vec(),
        external_node_numbers: Vec::new(),
        external_element_numbers: Vec::new(),
        num_boundary_nodes: nodes.len(),
        num_evaluation_nodes: 0,
        num_boundary_elements: elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .count(),
        num_evaluation_elements: elements
            .iter()
            .filter(|element| element.property.is_evaluation())
            .count(),
        symmetry_planes: [0; 3],
        symmetry_coordinates: [0.0; 3],
        num_reflections: 0,
        node_to_elements: Vec::new(),
    };
    let dof_map = mesh
        .continuous_quadratic_dof_map()
        .map_err(|error| TransmissionAssemblyError::Topology(error.to_string()))?;

    let mut owned = elements.to_vec();
    for (element_index, element) in owned.iter_mut().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        let dofs = dof_map.element_dofs(element_index).ok_or_else(|| {
            TransmissionAssemblyError::Topology(format!(
                "missing element DOF map for active panel {element_index}"
            ))
        })?;
        element.dof_addresses = dofs.to_vec();
    }

    exterior
        .validate()
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    interior
        .validate()
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    let frequency_scale = exterior
        .frequency
        .abs()
        .max(interior.frequency.abs())
        .max(1.0);
    if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
        return Err(TransmissionAssemblyError::FrequencyMismatch);
    }
    if exterior.harmonic_factor != interior.harmonic_factor {
        return Err(TransmissionAssemblyError::HarmonicMismatch);
    }

    let exterior_operators = assemble_boundary_operators_with_basis_shared_dofs(
        &owned,
        nodes,
        &exterior,
        BasisOrder::Quadratic,
    )?;
    let interior_operators = assemble_boundary_operators_with_basis_shared_dofs(
        &owned,
        nodes,
        &interior,
        BasisOrder::Quadratic,
    )?;
    let n = dof_map.num_dofs();
    let mut matrix = DenseMatrix::zeros(2 * n, 2 * n);
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let i_omega_rho_exterior = i_omega * exterior.density;
    let i_omega_rho_interior = i_omega * interior.density;
    for row in 0..n {
        for col in 0..n {
            matrix[[row, col]] = interior_operators.double_layer[[row, col]]
                + exterior_operators.double_layer[[row, col]];
            matrix[[row, n + col]] = -(i_omega_rho_interior
                * interior_operators.single_layer[[row, col]]
                + i_omega_rho_exterior * exterior_operators.single_layer[[row, col]]);
            matrix[[n + row, col]] = interior_operators.hypersingular[[row, col]]
                / interior.density
                + exterior_operators.hypersingular[[row, col]] / exterior.density;
            matrix[[n + row, n + col]] = -i_omega
                * (interior_operators.adjoint_double_layer[[row, col]]
                    + exterior_operators.adjoint_double_layer[[row, col]]);
        }
    }

    let mut panel_centers = Vec::with_capacity(n);
    let mut panel_normals = Vec::with_capacity(n);
    for dof in 0..n {
        let (element_index, local) = dof_map.dof_owner(dof).ok_or_else(|| {
            TransmissionAssemblyError::Topology(format!("missing owner for continuous DOF {dof}"))
        })?;
        let element = &owned[element_index];
        let point = element
            .collocation_points(BasisOrder::Quadratic, nodes)
            .into_iter()
            .nth(local)
            .ok_or_else(|| {
                TransmissionAssemblyError::Topology(format!(
                    "missing collocation point for continuous DOF {dof}"
                ))
            })?;
        panel_centers.push(point);
        panel_normals.push(element.normal);
    }

    Ok(TransmissionTbemSystem {
        matrix,
        interface_dofs: n,
        basis_order: BasisOrder::Quadratic,
        elements: owned,
        nodes: nodes.to_vec(),
        panel_centers,
        panel_normals,
        exterior_operators,
        interior_operators,
        exterior,
        interior,
    })
}

/// Assemble continuous-quadratic transmission after checked adjacent-order
/// quadrature admission for both exterior and interior kernels.
///
/// The topology and matrix layout are identical to
/// [`build_transmission_tbem_system_with_continuous_quadratic`]. Validation is
/// performed before dense allocation, and a failure identifies which medium's
/// pair estimate exceeded the caller's mixed error budget.
pub fn build_transmission_tbem_system_with_continuous_quadratic_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: PhysicsParams,
    interior: PhysicsParams,
    policy: TbemQuadraturePolicy,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    exterior
        .validate()
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    interior
        .validate()
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    validate_tbem_quadrature_for_basis(elements, nodes, &exterior, BasisOrder::Quadratic, &policy)
        .map_err(|error| TransmissionAssemblyError::Quadrature {
            medium: "exterior",
            error,
        })?;
    validate_tbem_quadrature_for_basis(elements, nodes, &interior, BasisOrder::Quadratic, &policy)
        .map_err(|error| TransmissionAssemblyError::Quadrature {
            medium: "interior",
            error,
        })?;
    build_transmission_tbem_system_with_continuous_quadratic(elements, nodes, exterior, interior)
}

/// Typed-XEM adapter for checked continuous-quadratic transmission assembly.
pub fn build_transmission_tbem_system_with_continuous_quadratic_checked_from_xem(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident: AcousticMedium,
    transmitted: AcousticMedium,
    policy: TbemQuadraturePolicy,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    FluidInterface::try_from_xem(incident, transmitted)?;
    let exterior = PhysicsParams::try_new_medium(incident, false)
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    let interior = PhysicsParams::try_new_medium(transmitted, true)
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    build_transmission_tbem_system_with_continuous_quadratic_checked(
        elements, nodes, exterior, interior, policy,
    )
}

/// Builds a continuous-quadratic transmission TBEM system from XEM media.
///
/// The incident and transmitted media must have compatible frequency and
/// harmonic-convention metadata.
pub fn build_transmission_tbem_system_from_xem(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident: AcousticMedium,
    transmitted: AcousticMedium,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    // Reuse the existing typed interface contract so frequency and phasor
    // compatibility are checked before any dense matrix allocation.
    FluidInterface::try_from_xem(incident, transmitted)?;
    let exterior = PhysicsParams::try_new_medium(incident, false)
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    let interior = PhysicsParams::try_new_medium(transmitted, true)
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    assemble_transmission_from_physics(elements, nodes, exterior, interior)
}

/// Assemble a conforming continuous-quadratic Müller transmission system
/// from convention-carrying XEM media.
///
/// This is the higher-order counterpart to
/// [`build_transmission_tbem_system_from_xem`].  It validates the typed
/// medium/interface contract before constructing the shared vertex and edge
/// midpoint DOF topology, preventing higher-order callers from bypassing
/// the canonical BEM harmonic and normal-orientation conventions.
pub fn build_transmission_tbem_system_with_continuous_quadratic_from_xem(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident: AcousticMedium,
    transmitted: AcousticMedium,
) -> Result<TransmissionTbemSystem, TransmissionAssemblyError> {
    FluidInterface::try_from_xem(incident, transmitted)?;
    let exterior = PhysicsParams::try_new_medium(incident, false)
        .map_err(TransmissionAssemblyError::ExteriorPhysics)?;
    let interior = PhysicsParams::try_new_medium(transmitted, true)
        .map_err(TransmissionAssemblyError::InteriorPhysics)?;
    build_transmission_tbem_system_with_continuous_quadratic(elements, nodes, exterior, interior)
}
